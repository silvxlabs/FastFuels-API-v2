"""Unit tests for lib.crs.crs_equal."""

from lib.crs import crs_equal


def test_identical_strings():
    assert crs_equal("EPSG:5070", "EPSG:5070")


def test_epsg_vs_ogc_urn():
    # The exact spellings from the QUIC-Fire export bug report.
    assert crs_equal("EPSG:5070", "urn:ogc:def:crs:EPSG::5070")
    assert crs_equal("urn:ogc:def:crs:EPSG::5070", "EPSG:5070")


def test_different_crs():
    assert not crs_equal("EPSG:5070", "EPSG:4326")


def test_both_none():
    assert crs_equal(None, None)


def test_one_none():
    assert not crs_equal("EPSG:5070", None)
    assert not crs_equal(None, "EPSG:5070")
