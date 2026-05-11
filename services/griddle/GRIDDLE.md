# Griddle

Griddle is the backend service for processing V2 Grid resources. It handles all grid sources (LANDFIRE, 3DEP, Meta2024) and operations (lookup, resample, blend) in a single Cloud Run service.

## Why "Griddle"?

It's a cooking surface—we're "cooking" grids. Also, it literally has "grid" in the name.

## Core Principles

### 1. Pure Functions

Handlers are pure functions that transform data. They don't know about Firestore, GCS, progress updates, or any infrastructure.

```python
def fetch_fbfm40(
    roi: gpd.GeoDataFrame,
    version: str = "2022",
) -> xr.Dataset:
    """Fetch LANDFIRE FBFM40. That's it."""
```

Benefits:
- Easy to test (no mocking infrastructure)
- Easy to understand (input → output)
- Easy to compose (chain pure functions)

### 2. Dataset Output Convention

All handlers **MUST** return `xr.Dataset` with named 2D `(y, x)` variables. Never return a DataArray or a 3D variable with a "band" dimension.

**Rules:**
- Each variable is a separate 2D `(y, x)` DataArray in the Dataset
- Variable names become band descriptions in GeoTIFF exports
- Never use `xr.concat(..., dim="band")` to combine variables — use `xr.Dataset({name: da, ...})`
- Never use `DataArray.name` for band naming — it applies the same name to ALL bands
- `save_zarr` (in `lib.zarr`) enforces Dataset-only at the storage boundary and raises `TypeError` for DataArray input

```python
# Correct — Dataset with named 2D variables
def fetch_topography(roi, version, bands, progress) -> xr.Dataset:
    variables = {}
    for band in bands:
        variables[band] = _fetch_landfire_raster(roi, band, version)
    return _to_dataset(variables)
# Result: Dataset with variables "elevation", "slope", "aspect"
# Each variable has dims (y, x)

# Wrong — DataArray with 3D "band" dimension
data = xr.concat([elevation, slope, aspect], dim="band")
# Result: Single 3D DataArray, breaks multiband exports

# Wrong — DataArray with .name
data.name = "topography"
# Result: ALL bands named "topography" in GeoTIFF exports
```

Use the `_to_dataset()` helper to build a Dataset from named DataArrays. It propagates CRS and transform via rioxarray:

```python
def _to_dataset(variables: dict[str, DataArray]) -> xr.Dataset:
    first = next(iter(variables.values()))
    ds = xr.Dataset(variables)
    ds = ds.rio.write_crs(first.rio.crs)
    ds = ds.rio.write_transform(first.rio.transform())
    return ds
```

This helper currently lives in `handlers/landfire.py` but should be moved to a shared location (e.g., `griddle/utils.py`) as other handler modules are added.

### Testing Protocol

Every handler **MUST** have a zarr round-trip test that calls `Dataset.rio.to_raster()` on the **full Dataset** (not per-variable). This is the exact operation the exporter performs and catches the entire class of bug where variables are dropped or spatial metadata is lost.

```python
def test_round_trip_to_raster_succeeds(self, mock_load_zarr, tmp_path):
    """Dataset.rio.to_raster() works after handler → zarr round-trip."""
    # ... produce result from handler ...
    save_zarr(str(tmp_path / "test.zarr"), result)
    loaded = load_zarr(str(tmp_path / "test.zarr"))
    loaded.rio.to_raster(str(tmp_path / "multiband.tif"))
    assert (tmp_path / "multiband.tif").exists()
```

### 3. rioxarray Everywhere

All raster operations use rioxarray. No separate georeference models internally. rioxarray accessors work on both DataArray and Dataset.

```python
# CRS, transform, spatial dimensions work on Dataset
ds.rio.crs            # CRS
ds.rio.transform()    # Affine transform
ds.rio.resolution()   # Pixel size
ds.rio.height         # Spatial height (y dimension)
ds.rio.width          # Spatial width (x dimension)

# Access individual bands via Dataset variables
ds["elevation"]       # DataArray for elevation band
list(ds.data_vars)    # ["elevation", "slope", "aspect"]
```

rioxarray handles:
- Reading COGs: `rioxarray.open_rasterio()`
- Reprojection: `data.rio.reproject()`
- Resampling: `data.rio.reproject(resolution=...)`
- Clipping: `data.rio.clip()`
- Writing: `data.rio.to_raster()` or `data.to_zarr()`

### 4. Single reprojection per fetch

Every external-source handler reprojects data exactly once per band. The
fold happens inside `RasterConnection.extract_window`: pass an alignment
destination (`destination_crs` + `destination_transform` + `destination_shape`)
and the single `rio.reproject` call lands the source pixels at the right
lattice. There is no second alignment pass on top.

The handler is responsible for resolving the alignment dict (from the
persisted grid source document) into destination kwargs via
`lib.alignment.resolve_alignment_destination`:

```python
from lib.alignment import RESAMPLING_METHOD_MAP, resolve_alignment_destination

dest = resolve_alignment_destination(
    alignment,
    roi,
    target_grid_doc,
    raster.raster_x_resolution,
    extent_buffer_cells=extent_buffer_cells,
)
data = raster.extract_window(
    roi=roi,
    interpolation_padding_cells=extent_buffer_cells,
    resampling=RESAMPLING_METHOD_MAP[method_name],
    destination_resolution=alignment.get("resolution")
        if alignment["target"] == "native" else None,
    **dest,
)
```

`method_name` is one of the public API names (`"nearest"`, `"bilinear"`,
…, `"median"`, `"root_mean_square"`, etc.) — `RESAMPLING_METHOD_MAP`
translates them to `rasterio.enums.Resampling` members and excludes
`gauss`, which `rasterio.warp.reproject` rejects.

`extent_buffer_cells` must be passed to *both* `resolve_alignment_destination`
(so the destination lattice for `target="domain"` and `target="grid"`
includes the buffer — the trailing clip is skipped on those paths) and
`extract_window` (so the source clip and the CRS-only-override clip are
sized correctly).

`resolve_alignment_destination` returns:
- `{}` for `target="native"` with no resolution change → `extract_window`
  takes its default branch (reproject to ROI CRS, clip).
- `{destination_crs}` for `target="native"` with a custom resolution →
  CRS-only branch (reproject preserving anchor, then clip — clip uses
  `destination_resolution` for the buffer).
- `{destination_crs, destination_transform, destination_shape}` for
  `target="domain"` and `target="grid"` → reproject directly to the
  exact lattice; the buffer is baked into the lattice.

The 3DEP topography handler is the one exception. It fetches at native
source resolution so `numpy.gradient` produces correct slope/aspect
values, then performs a single end-of-pipeline `rio.reproject` to the
alignment destination — two reprojections by design, justified by the
gradient computation.

### 5. Infrastructure in the Orchestrator

All infrastructure concerns (Firestore, GCS, progress, status) live in `main.py`. Handlers never touch infrastructure.

```
┌─────────────────────────────────────────────────────────┐
│ main.py (orchestrator)                                  │
│  - Loads documents from Firestore                       │
│  - Updates progress/status                              │
│  - Saves results to Zarr                                │
│  - Handles errors and cancellation                      │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ handlers/ (pure functions)                        │  │
│  │  - Take data, return data                         │  │
│  │  - No side effects                                │  │
│  │  - No infrastructure imports                      │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Architecture Decisions

### Cloud Run Service (Not Cloud Run Jobs)

| Concern | Cloud Run Jobs | Cloud Run Service |
|---------|----------------|-------------------|
| Cold start | 1+ minute | ~2-3 seconds |
| Max memory | 32GB | 32GB |
| Max timeout | Unlimited | 60 minutes |
| Instance reuse | None | Yes (stays warm) |
| Container support | Yes | Yes |

Cloud Run Service with functions-framework provides fast cold starts and instance reuse between requests. Performance testing shows ~10x improvement over Cloud Run Jobs due to warm instances.

### HTTP Trigger (Cloud Tasks)

```
POST /grids/fbfm40/landfire
    ↓
API creates grid doc (status: pending)
    ↓
API enqueues task to "griddle-queue": {"id": "abc123"}
    ↓
Cloud Tasks invokes Cloud Run service (HTTP POST)
    ↓
Griddle processes, updates status to complete/failed
```

Benefits:
- API stays fast (always returns 201 immediately)
- Built-in retry with exponential backoff (max attempts: 2)
- Task deduplication via task name (grid_id)
- Explicit invocation pattern (command, not broadcast)
- Warm instances handle subsequent requests in ~2-3 seconds

### Zarr Storage (Not GeoTIFF)

Grid data is stored as Zarr arrays in Cloud Storage:

```
gs://<GRIDS_BUCKET>/{grid_id}/
├── .zarray
├── .zattrs        # CRS, transform, band metadata
├── 0.0.0          # Chunked data
├── 0.0.1
└── ...
```

Why Zarr:
- Chunked reads (don't load entire grid for partial access)
- Cloud-native (works directly with GCS)
- Multi-band support
- xarray integration (`xr.open_zarr()`)

## Directory Structure

```
services/griddle/
├── griddle/
│   ├── __init__.py           # Package config (env vars, bucket names)
│   ├── main.py               # Orchestrator (all infrastructure)
│   ├── dispatch.py           # Routes to handlers, loads domains
│   ├── errors.py             # CancelledException, ProcessingError
│   ├── storage.py            # Zarr read/write/delete
│   └── handlers/
│       ├── __init__.py
│       ├── landfire.py       # fetch_fbfm40, fetch_topography + shared helpers
│       ├── lookup.py         # fbfm40_lookup
│       └── resample.py       # resample_grid
│       # Future: dep3.py, meta2024.py, blend.py
├── tests/
│   ├── data/
│   │   └── domains/          # Test domain GeoJSON files
│   ├── handlers/
│   │   └── test_landfire.py  # Integration tests (FBFM40 + topography)
│   ├── test_dispatch.py      # Unit tests for dispatcher
│   └── test_main.py          # Unit tests for orchestrator
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Handler Categories

### Source Handlers

Fetch data from external sources (COGs, APIs) for a domain extent. All source handlers return `xr.Dataset` with named variables (see Dataset Output Convention above).

**Signature:**
```python
def fetch_*(
    roi: gpd.GeoDataFrame,  # Region of interest with CRS
    **source_config,        # version, product, bands, etc.
) -> xr.Dataset:
```

**Current Implementations:**

```python
# handlers/landfire.py

def fetch_fbfm40(roi, version="2022") -> xr.Dataset:
    """Single-band: Dataset with variable 'fbfm' (int16 categorical codes)."""
    data = _fetch_landfire_raster(roi, "FBFM40", version)
    return _to_dataset({"fbfm": data})

def fetch_topography(roi, version, bands, progress) -> xr.Dataset:
    """Multi-band: Dataset with one variable per requested band.
    Bands: elevation (m), slope (degrees), aspect (degrees)."""
    variables = {}
    for band in bands:
        variables[band] = _fetch_landfire_raster(roi, band, version)
    return _to_dataset(variables)
```

**Future Handlers (Not Yet Implemented):**

```python
# handlers/dep3.py

def fetch_elevation(roi: gpd.GeoDataFrame) -> xr.Dataset:
    """Fetch 3DEP elevation data."""
    ...

# handlers/meta2024.py

def fetch_chm(roi: gpd.GeoDataFrame) -> xr.Dataset:
    """Fetch Meta 2024 canopy height model."""
    ...

### Transform Handlers (Not Yet Implemented)

Future handlers that will operate on existing grid data. The following are design specifications for planned implementations.

**Lookup:**
```python
# handlers/lookup.py

def lookup_fbfm40(
    source: xr.DataArray,
    quantities: list[str],
) -> xr.DataArray:
    """Convert FBFM codes to fuel parameters using SB40 tables.

    Args:
        source: DataArray with fbfm band (int16 codes)
        quantities: e.g., ["fuel_load.1hr", "fuel_load.10hr", "savr.1hr"]

    Returns:
        DataArray with dims (band, y, x), one band per quantity
    """
    table = load_sb40_lookup_table()
    bands = []
    for qty in quantities:
        if qty not in table.columns:
            raise ProcessingError(
                code="UNKNOWN_QUANTITY",
                message=f"Unknown quantity: {qty}",
                suggestion=f"Available quantities: {list(table.columns)}",
            )
        # Vectorized lookup
        values = table[qty].values[source.values]
        band = source.copy(data=values).assign_coords(band=qty)
        bands.append(band)
    return xr.concat(bands, dim="band")
```

**Resample:**
```python
# handlers/resample.py

from lib.alignment import RESAMPLING_METHOD_MAP

def resample(
    source: xr.DataArray,
    resolution: float,
    method: str = "bilinear",
    method_overrides: dict[str, str] | None = None,
) -> xr.DataArray:
    """Resample grid to new resolution.

    Args:
        source: Input DataArray
        resolution: Target resolution in CRS units (usually meters)
        method: Default resampling method (public API name, e.g. "bilinear")
        method_overrides: Per-band overrides, e.g., {"fbfm": "nearest"}
    """
    overrides = method_overrides or {}

    if "band" not in source.dims:
        resampling = RESAMPLING_METHOD_MAP[method]
        return source.rio.reproject(source.rio.crs, resolution=resolution, resampling=resampling)

    bands = []
    for band_name in source.coords["band"].values:
        band_data = source.sel(band=band_name)
        method_name = overrides.get(band_name, method)
        resampling = RESAMPLING_METHOD_MAP[method_name]
        resampled = band_data.rio.reproject(
            band_data.rio.crs,
            resolution=resolution,
            resampling=resampling,
        )
        bands.append(resampled)
    return xr.concat(bands, dim="band")
```

**Blend:**
```python
# handlers/blend.py

import operator

OPERATORS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "max": lambda a, b: xr.where(a > b, a, b),
    "min": lambda a, b: xr.where(a < b, a, b),
    "average": lambda a, b: (a + b) / 2,
}

def blend(
    inputs: dict[str, xr.DataArray],  # alias -> data
    select: list[dict],
    compute: list[dict],
) -> xr.DataArray:
    """Blend multiple grids via select/compute operations.

    Args:
        inputs: Mapping of alias to DataArray, e.g., {"a": grid_a, "b": grid_b}
        select: Band selections, e.g., [{"output": "fbfm", "from": "a.fbfm"}]
        compute: Band computations, e.g., [{"output": "fuel_load.1hr", "operator": "add", "operands": ["a.fuel_load.1hr", "b.fuel_load.1hr"]}]
    """
    bands = []

    for sel in select:
        band = select_band(inputs, sel)
        bands.append(band)

    for comp in compute:
        band = compute_band(inputs, comp)
        bands.append(band)

    return xr.concat(bands, dim="band")


def select_band(inputs: dict[str, xr.DataArray], spec: dict) -> xr.DataArray:
    """Select a band from inputs, optionally with conditions."""
    alias, band_name = spec["from"].split(".", 1)
    source = inputs[alias].sel(band=band_name)

    if "conditions" in spec:
        mask = evaluate_conditions(inputs, spec["conditions"])
        else_value = resolve_else(inputs, spec["else"])
        source = xr.where(mask, source, else_value)

    return source.assign_coords(band=spec["output"])


def compute_band(inputs: dict[str, xr.DataArray], spec: dict) -> xr.DataArray:
    """Compute a band from operands."""
    operands = [resolve_operand(inputs, op) for op in spec["operands"]]
    op_func = OPERATORS[spec["operator"]]

    result = operands[0]
    for operand in operands[1:]:
        result = op_func(result, operand)

    if "conditions" in spec:
        mask = evaluate_conditions(inputs, spec["conditions"])
        else_value = resolve_else(inputs, spec["else"])
        result = xr.where(mask, result, else_value)

    return result.assign_coords(band=spec["output"])


def resolve_operand(inputs: dict[str, xr.DataArray], ref: str) -> xr.DataArray:
    """Resolve 'alias.band' reference to actual data."""
    alias, band_name = ref.split(".", 1)
    return inputs[alias].sel(band=band_name)
```

## Shared Utilities (Planned)

The following utilities are planned but not yet implemented. Currently, LANDFIRE data fetching uses `lib.raster.RasterConnection` directly.

### COG Reading (Planned)

```python
# utils/cog.py

import rioxarray

def read_cog_window(
    url: str,
    bounds: tuple[float, float, float, float],
    target_crs: str,
) -> xr.DataArray:
    """Read a window from a Cloud-Optimized GeoTIFF.

    Args:
        url: COG URL (gs://, https://, or local path)
        bounds: (west, south, east, north) in target_crs
        target_crs: Target CRS for output

    Returns:
        DataArray clipped to bounds and reprojected to target_crs
    """
    # Open with rioxarray (handles GCS URLs)
    data = rioxarray.open_rasterio(url, chunks="auto")

    # Clip to bounds (in source CRS first, then reproject)
    # Note: bounds are in target_crs, need to transform
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    source_crs = data.rio.crs
    source_bounds = transform_bounds(
        CRS.from_string(target_crs),
        source_crs,
        *bounds,
    )

    # Clip and reproject
    clipped = data.rio.clip_box(*source_bounds)
    reprojected = clipped.rio.reproject(target_crs)

    # Final clip to exact bounds (reproject can expand slightly)
    return reprojected.rio.clip_box(*bounds)
```

### Modifications

```python
# utils/modifications.py

def apply_modifications(
    data: xr.DataArray,
    modifications: list[dict],
) -> xr.DataArray:
    """Apply modification rules to data.

    Each modification has:
        - conditions: list of {band, operator, value} or spatial conditions
        - actions: list of {band, modifier, value}

    All conditions are ANDed. When all match, actions are applied.
    """
    result = data.copy()

    for mod in modifications:
        mask = evaluate_conditions_for_data(result, mod["conditions"])

        for action in mod["actions"]:
            band_name = action["band"]
            modifier = action["modifier"]
            value = action["value"]

            band_data = result.sel(band=band_name)
            modified = apply_modifier(band_data, modifier, value)

            # Apply only where mask is True
            new_values = xr.where(mask, modified, band_data)

            # Update the band in result
            result.loc[{"band": band_name}] = new_values

    return result


def apply_modifier(data: xr.DataArray, modifier: str, value: float) -> xr.DataArray:
    """Apply a single modifier operation."""
    match modifier:
        case "multiply":
            return data * value
        case "divide":
            return data / value
        case "add":
            return data + value
        case "subtract":
            return data - value
        case "replace":
            return xr.full_like(data, value)
        case _:
            raise ValueError(f"Unknown modifier: {modifier}")
```

### Validation

```python
# utils/validation.py

from griddle.errors import ProcessingError

CONUS_BOUNDS = (-125.0, 24.0, -66.0, 50.0)  # west, south, east, north

def validate_conus_bounds(bounds: tuple[float, float, float, float]) -> None:
    """Raise ProcessingError if bounds are outside CONUS."""
    west, south, east, north = bounds
    cw, cs, ce, cn = CONUS_BOUNDS

    if west < cw or east > ce or south < cs or north > cn:
        raise ProcessingError(
            code="COVERAGE_ERROR",
            message="Domain is outside continental US coverage.",
            suggestion=f"Ensure domain is within ({cw}°W, {cs}°N) to ({ce}°W, {cn}°N).",
        )
```

## The Orchestrator

The orchestrator in `main.py` handles all infrastructure concerns:

```python
# main.py (current implementation)

import functions_framework
from flask import Request
from datetime import datetime, timezone

from griddle.dispatch import dispatch_handler
from griddle.errors import CancelledException, ProcessingError
from griddle.storage import save_zarr, delete_zarr
from lib.firestore import get_document, update_document, DocumentNotFoundError

GRIDS_COLLECTION = "grids-v2"


@functions_framework.http
def process_grid_request(request: Request):
    """Main entry point for grid processing.

    Triggered by Cloud Tasks HTTP request containing {"id": "..."}.

    Cloud Tasks retry behavior:
    - max_attempts=2: First attempt processes normally, second attempt marks as failed
    - X-CloudTasks-TaskRetryCount header indicates retry count (0 on first attempt)
    """
    data = request.get_json(silent=True)
    grid_id = data.get("id") if data else None

    if not grid_id:
        return "Missing id", 400

    # Check if this is a retry (previous attempt crashed)
    retry_count = int(request.headers.get("X-CloudTasks-TaskRetryCount", 0))
    if retry_count > 0:
        try:
            update_status(grid_id, "failed", error={
                "code": "UNEXPECTED_FAILURE",
                "message": "Job failed unexpectedly. Please try again.",
            })
        except (CancelledException, DocumentNotFoundError):
            pass
        return "OK", 200  # Return 200 to prevent further retries

    # Load grid document
    try:
        grid = load_grid(grid_id)
    except DocumentNotFoundError:
        return "OK", 200  # Grid was deleted, nothing to do

    # Update status to running
    try:
        update_status(grid_id, "running")
    except CancelledException:
        return "OK", 200

    try:
        # Dispatch to handler
        progress_callback = make_progress_callback(grid_id)
        result = dispatch_handler(grid, progress_callback)

        # Save to Zarr
        update_progress(grid_id, "Saving...", 90)
        save_zarr(grid_id, result)

        # Update status to completed with georeference
        transform = result.rio.transform()
        update_status(
            grid_id,
            "completed",
            georeference={
                "crs": str(result.rio.crs),
                "transform": list(transform)[:6],  # Affine 6 elements
                "shape": [result.rio.height, result.rio.width],
            },
        )
        return "OK", 200

    except CancelledException:
        delete_zarr(grid_id)
        return "OK", 200

    except ProcessingError as e:
        try:
            update_status(grid_id, "failed", error=e.to_dict())
        except CancelledException:
            delete_zarr(grid_id)
        return "OK", 200  # Return 200 - error is recorded

    except Exception:
        return "Internal error", 500  # Return 500 to trigger retry
```

Key behaviors:
- **Cancellation detection**: If a document update fails with `DocumentNotFoundError`, the grid was deleted (cancelled)
- **Retry handling**: On retry (crashed first attempt), marks as failed immediately
- **Georeference**: Stores CRS, affine transform (6 elements), and shape after processing
- **Progress tracking**: Updates Firestore with message and percent during processing

## Dispatcher

The dispatcher routes grid requests to the appropriate handler based on source type. It loads the domain as a GeoDataFrame (handling Firestore serialization quirks) and passes it to the handler.

```python
# dispatch.py (current implementation)

from typing import Callable
import geopandas as gpd
import xarray as xr

from griddle.errors import ProcessingError
from griddle.handlers import landfire, lookup, resample


def dispatch_handler(
    grid: dict,
    progress_callback: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Route to appropriate handler based on source type."""
    source = grid["source"]
    source_name = source["name"]

    match source_name:
        case "landfire":
            return handle_landfire(grid, source, progress_callback)
        case "lookup":
            return handle_lookup(grid, source, progress_callback)
        case "resample":
            return handle_resample(grid, source, progress_callback)
        case _:
            raise ProcessingError(...)


def handle_landfire(grid, source, progress) -> xr.Dataset:
    """Handle LANDFIRE source grids."""
    domain_gdf = load_domain_gdf(grid["domain_id"])
    product = source["product"]
    version = source.get("version", "2022")

    match product:
        case "fbfm40":
            return landfire.fetch_fbfm40(domain_gdf, version)
        case "topography":
            return landfire.fetch_topography(
                domain_gdf, version, source["bands"], progress
            )
        case _:
            raise ProcessingError(...)
```

## Storage

Grid data is stored as Zarr in Cloud Storage. Storage is type-agnostic — it accepts both `DataArray` and `Dataset` and calls `.to_zarr()` directly. Band names are preserved structurally: as variable names in a Dataset, or as the band coordinate in a DataArray. No extra `attrs` metadata is needed.

```python
# storage.py (current implementation)

def save_zarr(grid_id: str, data: xr.DataArray | xr.Dataset) -> str:
    """Save grid data to Zarr in Cloud Storage."""
    path = f"{GRIDS_BUCKET}/{grid_id}"
    data.to_zarr(path, mode="w", consolidated=True)
    return path

def load_zarr(grid_id: str) -> xr.Dataset:
    """Load grid data from Zarr. xr.open_zarr always returns Dataset."""
    path = f"{GRIDS_BUCKET}/{grid_id}"
    return xr.open_zarr(path)
```

## Error Handling

```python
# errors.py

from dataclasses import dataclass
from typing import Optional


class CancelledException(Exception):
    """Grid was deleted during processing."""
    pass


@dataclass
class ProcessingError(Exception):
    """Structured error with user-friendly message."""
    code: str
    message: str
    suggestion: Optional[str] = None
    traceback: Optional[str] = None

    def to_dict(self) -> dict:
        result = {"code": self.code, "message": self.message}
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.traceback:
            result["traceback"] = self.traceback
        return result
```

Common error codes:

| Code | When |
|------|------|
| `COVERAGE_ERROR` | Domain outside data source coverage |
| `EMPTY_DOMAIN` | Domain has no geometry |
| `UNKNOWN_SOURCE` | Unrecognized source type |
| `UNKNOWN_PRODUCT` | Unrecognized product for source |
| `UNKNOWN_QUANTITY` | Requested quantity not available |
| `INVALID_FBFM_CODES` | Source grid contains codes that are not valid FBFM40 fuel models |
| `SOURCE_GRID_NOT_FOUND` | Referenced grid doesn't exist |
| `SOURCE_GRID_NOT_COMPLETE` | Referenced grid still processing |
| `INTERNAL_ERROR` | Unexpected error (catch-all) |

## Testing

Tests use real domain data stored in `tests/data/domains/` to perform integration tests against LANDFIRE COGs.

### Test Structure

```
services/griddle/tests/
├── data/
│   └── domains/
│       └── blue_mtn.json      # ~1 sq km test domain in Montana
├── handlers/
│   └── test_landfire.py       # Integration tests for LANDFIRE handler
├── test_dispatch.py           # Unit tests for dispatcher
└── test_main.py               # Unit tests for orchestrator
```

### Handler Tests (Integration)

```python
# tests/handlers/test_landfire.py

class TestFetchFbfm40:
    def test_returns_dataset(self, roi):
        result = fetch_fbfm40(roi=roi)
        assert isinstance(result, xr.Dataset)

    def test_has_fbfm_variable(self, roi):
        result = fetch_fbfm40(roi=roi)
        assert "fbfm" in result.data_vars

class TestFetchTopography:
    def test_fetch_all_bands(self, test_domain, roi):
        result = fetch_topography(
            roi=roi, version="2020",
            bands=["elevation", "slope", "aspect"], progress=MagicMock(),
        )
        assert list(result.data_vars) == ["elevation", "slope", "aspect"]
        for var in result.data_vars:
            assert result[var].shape == test_domain.expected_shape
        assert result.rio.crs == roi.crs
```

### Dispatcher Tests (Unit)

The dispatcher tests use mocks to test routing logic without hitting external services.

### Running Tests

```bash
cd services/griddle
uv run pytest

# Run with coverage
uv run pytest --cov=griddle

# Run specific test file
uv run pytest tests/handlers/test_landfire.py
```

## Deployment

### Dockerfile (Current)

```dockerfile
FROM python:3.13-slim

# Install minimal dependencies (libexpat for GDAL)
RUN apt-get update && apt-get install -y libexpat1

# Install uv
RUN pip install uv

# Copy project files into the container
COPY griddle/ /griddle/
COPY lib/lib /griddle/lib

# Set the working directory
WORKDIR /griddle

# Install project dependencies with uv
RUN uv sync --frozen

# Environment variables for functions-framework
ENV PYTHONPATH=/griddle
ENV FUNCTION_TARGET=process_grid_request
ENV FUNCTION_SOURCE=griddle/main.py
ENV FUNCTION_SIGNATURE_TYPE=http

# Tell GDAL to use GCE metadata server for GCS credentials
ENV CPL_MACHINE_IS_GCE=YES

# Run the service
ENTRYPOINT ["uv", "run", "functions-framework"]
```

### Cloud Run Deployment

Griddle is deployed as a Cloud Run Service (not Cloud Functions). The `functions-framework` provides the HTTP handler interface while Cloud Run manages scaling.

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/<GCP_PROJECT>/griddle
gcloud run deploy griddle \
  --image gcr.io/<GCP_PROJECT>/griddle \
  --platform managed \
  --region us-west1 \
  --memory 8Gi \
  --timeout 540s \
  --no-allow-unauthenticated
```

Cloud Tasks uses the Cloud Run service URL for invocation. The task queue is configured with:
- Max attempts: 2
- Task name: grid_id (for deduplication)

### Local Development

```bash
cd services/griddle

# Install dependencies
uv sync

# Run with functions-framework (HTTP on port 8080)
uv run functions-framework --target=process_grid_request --signature-type=http

# Or test a specific grid directly
GRID_ID=abc123 uv run python -m griddle.main
```

### Performance

Performance testing with ~1 km² domains shows:
- **Cold start**: ~22s (first request after deployment)
- **Warm requests**: ~2-3s (subsequent requests while instance is warm)
- **V1 comparison**: ~10x faster than V1 due to Cloud Run instance reuse

## Implementation Checklist

### Phase 1: Foundation ✅
- [x] Set up project structure
- [x] Implement `errors.py` (CancelledException, ProcessingError)
- [x] Implement `storage.py` (Zarr read/write/delete)
- [x] Implement shared library `lib.firestore` (document CRUD)
- [x] Implement shared library `lib.gcs` (blob operations)
- [x] Implement shared library `lib.tasks` (Cloud Tasks HTTP)

### Phase 2: First Handler ✅
- [x] Implement `handlers/landfire.py` (fetch_fbfm40)
- [x] Implement `dispatch.py` (minimal, just LANDFIRE)
- [x] Implement `main.py` (full orchestrator with progress/cancellation)
- [x] End-to-end test: create LANDFIRE FBFM40 grid
- [x] Performance validation: ~10x faster than V1 (2-3s warm vs 25s)

### Phase 3: Transform Handlers
- [x] Implement `handlers/lookup.py` (FBFM40 SB40 lookup with pint unit conversion)
- [x] Implement `handlers/resample.py` (rioxarray `rio.reproject()` with 14 methods and per-band overrides)
- [ ] Implement `utils/modifications.py`

### Phase 4: Remaining Source Handlers
- [x] Implement LANDFIRE topography (elevation, slope, aspect) — returns Dataset with named variables
- [ ] Implement remaining LANDFIRE products (cbd, cbh, cc)
- [ ] Implement `handlers/dep3.py`
- [ ] Implement `handlers/meta2024.py`

### Phase 5: Blend
- [ ] Implement `handlers/blend.py`
- [ ] Test complex blend operations

## References

- [V2 Grid Resource Design](../api/api/resources/grids/grids.md)
- [V2 API Design Philosophy](../api/API_DESIGN.md)
- [rioxarray Documentation](https://corteva.github.io/rioxarray/)
- [Cloud Run Service](https://cloud.google.com/run/docs)
- [Cloud Tasks](https://cloud.google.com/tasks/docs)
