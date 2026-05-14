"""Spawned-worker entry for treevox chunk voxelization.

This is the ONLY treevox module that workers may import. Every import here
counts against per-worker spawn startup cost — do not add more without
weighing the cost (see TREEVOX.md on worker isolation).

Specifically do NOT import: xarray, rioxarray, gcsfs, zarr, treevox.storage,
treevox.main. Workers receive numpy arrays + plain dicts and return the same.
"""

from __future__ import annotations

import os
import time
import traceback

import numpy as np

from treevox.voxelize import build_chunk_cache, voxelize_chunk


def run(payload: dict) -> dict:
    """Voxelize one chunk. Top-level, picklable, side-effect-free.

    payload keys:
      chunk_location: (row, col)
      buffers: {band_key: np.ndarray (nz, halo_y, halo_x)} pre-filled with
               each band's fill value
      trees: pd.DataFrame slice for this chunk (must include `_cache_key`
             and `tree_id` columns)
      hr: float horizontal resolution
      vr: float vertical resolution
      x_origin: float NW corner x (west edge of grid)
      y_origin: float NW corner y (north edge of grid)
      source_config: dict (from grid["source"])
      chunk_y_start: int absolute y index of buffer[..., 0, :] in the full grid
      chunk_x_start: int absolute x index of buffer[..., :, 0] in the full grid
      y_slice: slice absolute y range covered by the buffer (for merge placement)
      x_slice: slice absolute x range covered by the buffer
      rng_seed: int seed for deterministic stochastic sampling

    Returns on success:
      {chunk_location, buffers, y_slice, x_slice, pid, num_trees,
       process_time_s}
    Returns on exception (never raises — orchestrator surfaces via
    ProcessingError(VOXELIZATION_FAILED)):
      {chunk_location, error}
    """
    start = time.monotonic()
    num_trees = len(payload["trees"])
    try:
        rng = np.random.default_rng(payload["rng_seed"])
        cache = build_chunk_cache(
            payload["trees"],
            payload["hr"],
            payload["vr"],
            payload["source_config"],
            rng,
        )
        voxelize_chunk(
            payload["trees"],
            payload["buffers"],
            cache,
            payload["chunk_y_start"],
            payload["chunk_x_start"],
            payload["hr"],
            payload["vr"],
            payload["x_origin"],
            payload["y_origin"],
            payload["source_config"],
            rng,
        )
        return {
            "chunk_location": payload["chunk_location"],
            "buffers": payload["buffers"],
            "y_slice": payload["y_slice"],
            "x_slice": payload["x_slice"],
            "pid": os.getpid(),
            "num_trees": num_trees,
            "process_time_s": time.monotonic() - start,
        }
    except NotImplementedError as e:
        return {
            "chunk_location": payload["chunk_location"],
            "error_code": "BIOMASS_COMPONENT_NOT_IMPLEMENTED",
            "error_message": str(e),
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }
    except Exception as e:
        return {
            "chunk_location": payload["chunk_location"],
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }
