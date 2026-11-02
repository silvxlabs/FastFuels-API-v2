import pytest
from api.resources.band_metadata import BandEnum, publish_band_defs


class _Band(BandEnum):
    a = "fuel_load.a"
    b = "fuel_load.b"


def test_publish_band_defs_rejects_a_value_without_a_definition():
    with pytest.raises(ValueError, match="fuel_load.b"):
        publish_band_defs(_Band, {_Band.a: {"name": "A", "type": "continuous"}})
