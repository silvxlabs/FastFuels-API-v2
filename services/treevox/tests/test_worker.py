"""Unit tests for treevox._worker.

Covers worker isolation (no treevox.storage import), deterministic output
via rng_seed, pickle safety for spawn-Pool use, and exception capture.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle

import numpy as np
import pandas as pd
from treevox import _worker, voxelize

# Payload fixture


def _make_payload(rng_seed=42, trees_n=2):
    """Build a minimal worker payload with mock-friendly data.

    The buffer covers the full grid so placement math doesn't clip trees
    in these unit tests. bulk_density.foliage.live is included so tests can
    observe per-biomass-array variation (volume_fraction only sees the mask).
    """
    dims = voxelize.compute_grid_dimensions(
        domain_gdf=_FakeDomain(),
        df=pd.DataFrame({"height": [10.0]}),
        resolution=(1.0, 1.0, 1.0),
    )
    buffers = {
        "volume_fraction": np.zeros(
            (dims["nz"], dims["ny"], dims["nx"]), dtype="float32"
        ),
        "bulk_density.foliage.live": np.zeros(
            (dims["nz"], dims["ny"], dims["nx"]), dtype="float32"
        ),
        "tree_id": np.full((dims["nz"], dims["ny"], dims["nx"]), -1, dtype="int32"),
    }
    trees = pd.DataFrame(
        {
            "x": [10.0] * trees_n,
            "y": [10.0] * trees_n,
            "fia_species_code": [131] * trees_n,
            "fia_status_code": [1] * trees_n,
            "dbh": [20.0] * trees_n,
            "height": [8.0] * trees_n,
            "crown_ratio": [0.5] * trees_n,
            "tree_id": list(range(trees_n)),
            "_cache_key": [0] * trees_n,
        }
    )
    return {
        "chunk_location": (0, 0),
        "buffers": buffers,
        "trees": trees,
        "hr": dims["hr"],
        "vr": dims["vr"],
        "x_origin": dims["x_origin"],
        "y_origin": dims["y_origin"],
        "source_config": {
            "resolution": (1.0, 1.0, 1.0),
            "crown_profile_model": "purves",
            "biomass_source": {
                "type": "allometry",
                "equations": "nsvb",
                "components": ["foliage"],
                "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
            },
            "moisture_model": {"live": {"method": "uniform", "value": 100.0}},
        },
        "chunk_y_start": 0,
        "chunk_x_start": 0,
        "y_slice": slice(0, dims["ny"]),
        "x_slice": slice(0, dims["nx"]),
        "rng_seed": rng_seed,
    }


class _FakeDomain:
    @property
    def total_bounds(self):
        return np.array([0.0, 0.0, 20.0, 20.0])

    @property
    def crs(self):
        return "EPSG:32610"


def _patch_fastfuels(monkeypatch, canopy_shape=(2, 3, 3), biomass_value=1.0):
    """Mock out fastfuels-core calls so tests are deterministic and fast."""
    canopy = np.ones(canopy_shape, dtype="float32")
    monkeypatch.setattr(
        voxelize, "discretize_crown_profile", lambda *a, **kw: canopy.copy()
    )
    monkeypatch.setattr(
        voxelize,
        "compute_crown_probability_field",
        lambda m, **kw: (m, int(np.count_nonzero(m))),
    )
    monkeypatch.setattr(
        voxelize, "sample_occupancy", lambda m, field, n, **kw: m.copy()
    )

    class FakeVT:
        def __init__(self, tree, mask, hr, vr):
            self.mask = mask

        def distribute_biomass(self):
            return (self.mask * biomass_value).astype("float32")

    monkeypatch.setattr(voxelize, "VoxelizedTree", FakeVT)


# Pure-function behavior


class TestRun:
    def test_returns_expected_keys(self, monkeypatch):
        _patch_fastfuels(monkeypatch)
        result = _worker.run(_make_payload())
        assert set(result.keys()) >= {"chunk_location", "buffers", "y_slice", "x_slice"}
        assert "error" not in result

    def test_buffers_populated(self, monkeypatch):
        _patch_fastfuels(monkeypatch)
        result = _worker.run(_make_payload())
        assert result["buffers"]["volume_fraction"].sum() > 0

    def test_exception_captured_as_error_dict(self, monkeypatch):
        _patch_fastfuels(monkeypatch)
        payload = _make_payload()
        payload["trees"] = payload["trees"].drop(columns=["tree_id"])
        result = _worker.run(payload)
        assert "error" in result
        assert result["chunk_location"] == (0, 0)
        assert "buffers" not in result

    def test_not_implemented_component_sets_error_code(self, monkeypatch):
        _patch_fastfuels(monkeypatch)
        payload = _make_payload()
        payload["buffers"] = {
            "bulk_density.fine.live": np.zeros_like(
                payload["buffers"]["bulk_density.foliage.live"]
            )
        }
        payload["source_config"]["bands"] = ["bulk_density.fine.live"]
        payload["source_config"]["biomass_source"]["components"] = ["fine"]
        payload["source_config"]["biomass_source"]["fine"] = {
            "recipe": "foliage_plus_branchwood_fraction",
            "branchwood_fraction": 0.1,
        }

        result = _worker.run(payload)

        assert result["error_code"] == "BIOMASS_COMPONENT_NOT_IMPLEMENTED"
        assert "fine" in result["error_message"]
        assert "buffers" not in result


# Determinism via rng_seed


class TestDeterminism:
    def test_same_seed_same_output(self, monkeypatch):
        _patch_fastfuels(monkeypatch)
        r1 = _worker.run(_make_payload(rng_seed=123))
        r2 = _worker.run(_make_payload(rng_seed=123))
        np.testing.assert_array_equal(
            r1["buffers"]["volume_fraction"], r2["buffers"]["volume_fraction"]
        )
        np.testing.assert_array_equal(
            r1["buffers"]["tree_id"], r2["buffers"]["tree_id"]
        )

    def test_different_seeds_may_differ(self, monkeypatch):
        """With a cache of >1 realizations, different seeds pick different
        biomass arrays. Mock scales the whole biomass array by the incoming
        sample seed so `bulk_density.foliage.live` observably diverges between runs.
        """
        canopy = np.ones((3, 3, 3), dtype="float32")

        def fake_sample(m, field, n, seed=None, **kw):
            return m.copy() * (1.0 + (float(seed or 1) % 100) * 0.1)

        monkeypatch.setattr(
            voxelize, "discretize_crown_profile", lambda *a, **kw: canopy.copy()
        )
        monkeypatch.setattr(
            voxelize,
            "compute_crown_probability_field",
            lambda m, **kw: (m, int(np.count_nonzero(m))),
        )
        monkeypatch.setattr(voxelize, "sample_occupancy", fake_sample)

        class FakeVT:
            def __init__(self, tree, mask, hr, vr):
                self.mask = mask

            def distribute_biomass(self):
                return self.mask.astype("float32")

        monkeypatch.setattr(voxelize, "VoxelizedTree", FakeVT)

        p1 = _make_payload(rng_seed=1, trees_n=10)
        p2 = _make_payload(rng_seed=999, trees_n=10)
        p1["trees"]["x"] = np.arange(10.0, 20.0)
        p2["trees"]["x"] = np.arange(10.0, 20.0)
        p1["trees"]["y"] = np.full(10, 10.0)
        p2["trees"]["y"] = np.full(10, 10.0)
        r1 = _worker.run(p1)
        r2 = _worker.run(p2)
        assert "error" not in r1, r1.get("error")
        assert "error" not in r2, r2.get("error")
        assert not np.array_equal(
            r1["buffers"]["bulk_density.foliage.live"],
            r2["buffers"]["bulk_density.foliage.live"],
        )


# Pickle roundtrip


class TestPickleRoundtrip:
    def test_payload_pickles(self, monkeypatch):
        _patch_fastfuels(monkeypatch)
        payload = _make_payload()
        blob = pickle.dumps(payload)
        restored = pickle.loads(blob)
        result = _worker.run(restored)
        assert "error" not in result

    def test_result_pickles(self, monkeypatch):
        _patch_fastfuels(monkeypatch)
        result = _worker.run(_make_payload())
        blob = pickle.dumps(result)
        restored = pickle.loads(blob)
        np.testing.assert_array_equal(
            result["buffers"]["volume_fraction"],
            restored["buffers"]["volume_fraction"],
        )


# Import isolation


def _check_isolation(result_queue):
    """Run in a spawned subprocess: import _worker, then check sys.modules."""
    import sys

    import treevox._worker  # noqa: F401

    forbidden = {"treevox.storage", "treevox.main"}
    loaded = forbidden & set(sys.modules.keys())
    result_queue.put(sorted(loaded))


class TestImportIsolation:
    def test_worker_does_not_pull_in_treevox_orchestrator_modules(self):
        """The worker must not import
        treevox.storage or treevox.main. Transitive pulls from fastfuels_core
        (xarray etc.) are out of our control.
        """
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_check_isolation, args=(q,))
        p.start()
        p.join(timeout=60)
        assert p.exitcode == 0
        loaded = q.get_nowait()
        assert loaded == [], (
            f"Worker must not import treevox.storage or treevox.main. Leaked: {loaded}."
        )


# Spawn-Pool end-to-end


def _trivial_payload():
    """Even simpler payload with no fastfuels-core dependency — used only
    to exercise pickle + spawn context. Returns error dict since no mocks
    are applied inside the worker subprocess, but that's fine — we're
    validating the transport, not the compute.
    """
    return _make_payload(trees_n=0)


class TestPoolRoundtrip:
    def test_spawn_pool_runs_worker(self):
        """The worker runs correctly inside a spawn-context
        Pool. Exercises pickle safety + worker isolation + import rules.

        Uses an empty trees DataFrame so no fastfuels-core work happens —
        the test validates the mp transport, not the voxelization compute
        (which is covered in test_voxelize.py).
        """
        ctx = mp.get_context("spawn")
        payloads = [_trivial_payload() for _ in range(2)]
        with ctx.Pool(processes=2) as pool:
            results = pool.map(_worker.run, payloads)
        assert len(results) == 2
        for r in results:
            assert "error" not in r, r.get("error")
            assert "buffers" in r
