"""Unit tests for the domain lattice cell counts (#772). No server, no GCP."""

import asyncio

import pytest
from api.resources.domains.router import get_domain_lattice
from api.resources.grids.exports.alignment import build_domain_lattice

# #772: a 3DEP target grid's non-round cell size on the Blue Mountain domain.
RES = 7.7301175133338464
MINX = 720227.9398802927
MAXY = 5190652.2875135


@pytest.fixture(autouse=True, scope="session")
def _warmup_api():
    """Override the conftest warmup, which needs a live server and Firestore."""
    yield


def make_domain(minx, miny, maxx, maxy):
    ring = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
    return {
        "crs": {"type": "name", "properties": {"name": "EPSG:32611"}},
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "domain"},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }


def lattice(domain, resolution, num_buffer_cells=0):
    return asyncio.run(
        get_domain_lattice(
            domain=domain, resolution=resolution, num_buffer_cells=num_buffer_cells
        )
    )


def test_noisy_bounds_snap_to_integer_counts():
    maxx = MINX + 169 * RES
    miny = MAXY - 115 * RES
    # The raw quotients sit just above the integers, so a plain ceil adds a cell.
    assert (maxx - MINX) / RES > 169
    assert (MAXY - miny) / RES > 115

    result = lattice(make_domain(MINX, miny, maxx, MAXY), RES)

    assert result.shape == (115, 169)
    assert result.transform == (RES, 0.0, MINX, 0.0, -RES, miny + 115 * RES)


def test_exact_multiple_is_unchanged():
    result = lattice(make_domain(500000.0, 5200000.0, 501000.0, 5201000.0), 50.0)

    assert result.shape == (20, 20)
    assert result.transform == (50.0, 0.0, 500000.0, 0.0, -50.0, 5201000.0)


def test_partial_cell_still_rounds_up():
    maxx = MINX + 169.4 * RES
    miny = MAXY - 115 * RES

    result = lattice(make_domain(MINX, miny, maxx, MAXY), RES)

    assert result.shape == (115, 170)


def test_export_domain_lattice_matches_endpoint():
    maxx = MINX + 169 * RES
    miny = MAXY - 115 * RES
    domain = make_domain(MINX, miny, maxx, MAXY)
    domain["bbox"] = [MINX, miny, maxx, MAXY]

    export = build_domain_lattice(domain, RES)
    endpoint = lattice(domain, RES)

    assert (export["ny"], export["nx"]) == endpoint.shape == (115, 169)
    assert tuple(export["transform"]) == endpoint.transform
