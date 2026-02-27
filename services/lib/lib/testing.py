"""Path constants for shared test data used across all services."""

from pathlib import Path

SHARED_TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "shared_data"
SHARED_TEST_DOMAINS_DIR = SHARED_TEST_DATA_DIR / "domains"
SHARED_TEST_GRIDS_DIR = SHARED_TEST_DATA_DIR / "grids"
SHARED_TEST_INVENTORIES_DIR = SHARED_TEST_DATA_DIR / "inventories"
SHARED_TEST_EXPORTS_DIR = SHARED_TEST_DATA_DIR / "exports"
