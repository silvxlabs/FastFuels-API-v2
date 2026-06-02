# etcher

Feature-polygon generation service for FastFuels API v2. Given a domain and a
feature request (road or water), etcher reads the relevant OpenStreetMap data,
buffers linear features into polygons based on classification, clips to the
domain, and writes the result as GeoParquet to the features bucket.

## OSM data source

etcher reads OSM from a **static per-state FlatGeobuf snapshot on GCS**, not from
the live Overpass API (Overpass blocks Cloud Run egress IPs). The snapshot is
shared, national, and built/maintained by the v1 feature service.

```
$OSM_BUCKET/road/<state-slug>.fgb     LineStrings;       cols: osm_id, highway, name
$OSM_BUCKET/water/<state-slug>.fgb    polygons + lines;  cols: osm_id, waterway, name
$OSM_BUCKET/index/states.fgb          state polygons;    cols: slug, name, geometry
```

Reads are lazy: a `gs://` path + a bbox makes GDAL fetch only the ROI's bytes via
HTTP range requests. `etcher/osm_source.py` routes a domain's bbox through the
states index to the state file(s) it touches, reads each by bbox, and dedupes
cross-border features on `osm_id`. See `osm_source.read_osm_features`.

## Configuration

All config comes from environment variables (a repo-root `.env` is loaded by
`lib.config`). etcher needs at least:

- `OSM_BUCKET` — the OSM FlatGeobuf source bucket (bare name; `gs://` is added in code)
- `FEATURES_BUCKET` — where generated feature GeoParquet is written
- `GCP_PROJECT`, `GCP_REGION`, `GOOGLE_APPLICATION_CREDENTIALS`

The OSM reads use GDAL's `/vsigs` handler; on Cloud Run, `CPL_MACHINE_IS_GCE=YES`
(set in the Dockerfile) authenticates via the metadata server.

## Tests

```bash
# Unit tests (offline; no GCS/network)
uv run pytest tests/ --ignore=tests/integration -v

# Integration tests (real Firestore + GCS; needs credentials and OSM_BUCKET)
uv run pytest tests/integration/ -v
```

### Live-Overpass parity gate (manual)

`tests/integration/handlers/test_osm_parity.py` cross-checks the FlatGeobuf reader
against live Overpass. It is **skipped by default** and **cannot run in CI**
(GitHub runners are cloud IPs that Overpass blocks — the same reason for this
migration). Run it locally on a non-cloud network, with the `parity` extra:

```bash
RUN_OSM_PARITY=1 uv run --extra parity \
  pytest tests/integration/handlers/test_osm_parity.py -v
```

It asserts identical `osm_id` sets and matching geometries (Hausdorff ~0) for a
single-state ROI, a border-straddling ROI, and a reservoir/basin ROI. Point the
ROI constants at domains you trust before relying on the result.
