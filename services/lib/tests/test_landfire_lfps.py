"""
Unit tests for lib.landfire_lfps LFPS API client.

All tests run against mocked LFPS responses -- no network access. The product
cache is patched directly so `list_products` never reaches out.

Run with: uv run --extra lfps pytest tests/test_landfire_lfps.py -v
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from lib.landfire_lfps import (
    LFPS_PRODUCTS_TTL_SECONDS,
    LfpsJob,
    LfpsJobFailedError,
    LfpsProduct,
    download,
    list_products,
    poll_status,
    submit_job,
)


def make_products_response(names: list[str]) -> dict:
    """Build a synthetic /api/products response body."""
    return {
        "products": [
            {
                "productName": f"{name} product",
                "theme": "Fuels",
                "layerName": name,
                "acronym": name.split("_")[-1],
                "version": name.split("_")[0].removeprefix("LF"),
                "conus": True,
                "ak": True,
                "hi": False,
                "prvi": False,
                "geoAreas": "SW, NW",
            }
            for name in names
        ]
    }


def _make_product(layer_name: str) -> LfpsProduct:
    """Build an `LfpsProduct` directly, for pre-seeding the cache in tests."""
    return LfpsProduct(
        layer_name=layer_name,
        product_name=f"{layer_name} product",
        theme="Fuels",
        acronym=layer_name.split("_")[-1],
        version=layer_name.split("_")[0].removeprefix("LF"),
        conus=True,
        geo_areas="SW, NW",
    )


def mock_response(json_body: dict, content: bytes = b"") -> MagicMock:
    response = MagicMock()
    response.json.return_value = json_body
    response.content = content
    response.raise_for_status.return_value = None
    return response


class TestListProducts:
    """Tests for product-catalog loading and its TTL cache."""

    def test_fetches_and_parses_products(self):
        body = make_products_response(["LF2024_FBFM40"])
        with patch("lib.landfire_lfps._products", None):
            with patch(
                "lib.landfire_lfps.requests.get",
                return_value=mock_response(body),
            ) as mock_get:
                products = list_products()

        assert len(products) == 1
        assert products[0].layer_name == "LF2024_FBFM40"
        assert products[0].conus is True
        mock_get.assert_called_once()

    def test_cached_products_are_reused_within_the_ttl(self):
        cached = [_make_product("LF2024_FBFM40")]
        with patch("lib.landfire_lfps._products", cached):
            with patch("lib.landfire_lfps._products_fetched_on", datetime.now(UTC)):
                with patch("lib.landfire_lfps.requests.get") as mock_get:
                    result = list_products()

        assert result == cached
        mock_get.assert_not_called()

    def test_refresh_forces_a_re_fetch(self):
        cached = [_make_product("OLD")]
        fresh_body = make_products_response(["NEW"])
        with patch("lib.landfire_lfps._products", cached):
            with patch("lib.landfire_lfps._products_fetched_on", datetime.now(UTC)):
                with patch(
                    "lib.landfire_lfps.requests.get",
                    return_value=mock_response(fresh_body),
                ) as mock_get:
                    result = list_products(refresh=True)

        assert [p.layer_name for p in result] == ["NEW"]
        mock_get.assert_called_once()

    def test_expired_cache_is_reloaded(self):
        cached = [_make_product("OLD")]
        fresh_body = make_products_response(["OLD", "NEW"])
        expired = datetime.now(UTC) - timedelta(seconds=LFPS_PRODUCTS_TTL_SECONDS + 1)
        with patch("lib.landfire_lfps._products", cached):
            with patch("lib.landfire_lfps._products_fetched_on", expired):
                with patch(
                    "lib.landfire_lfps.requests.get",
                    return_value=mock_response(fresh_body),
                ):
                    result = list_products()

        assert [p.layer_name for p in result] == ["OLD", "NEW"]


class TestSubmitJob:
    """Tests for job submission and its query params."""

    def test_builds_expected_query_params(self):
        response_body = {
            "jobId": "job-123",
            "status": "Pending",
            "messages": [],
        }
        with patch("lib.landfire_lfps.LANDFIRE_USER_EMAIL", "test@example.com"):
            with patch(
                "lib.landfire_lfps.requests.get",
                return_value=mock_response(response_body),
            ) as mock_get:
                submit_job(
                    ["LF2024_FBFM40", "LF2024_FBFM13"], "-114.2 46.8 -114.1 46.9"
                )

        args, kwargs = mock_get.call_args
        assert args[0].endswith("/job/submit")
        assert kwargs["params"] == {
            "Email": "test@example.com",
            "Layer_List": "LF2024_FBFM40;LF2024_FBFM13",
            "Area_of_Interest": "-114.2 46.8 -114.1 46.9",
        }

    def test_parses_the_submitted_job(self):
        response_body = {
            "jobId": "job-123",
            "status": "Pending",
            "messages": [],
        }
        with patch(
            "lib.landfire_lfps.requests.get",
            return_value=mock_response(response_body),
        ):
            job = submit_job(["LF2024_FBFM40"], "-114.2 46.8 -114.1 46.9")

        assert job.job_id == "job-123"
        assert job.status == "Pending"
        assert job.output_file is None


class TestPollStatus:
    """Tests for single-shot status checks and Failed-status error surfacing."""

    def test_pending_status(self):
        body = {"jobId": "job-123", "status": "Pending", "messages": []}
        with patch("lib.landfire_lfps.requests.get", return_value=mock_response(body)):
            job = poll_status("job-123")

        assert job.status == "Pending"
        assert job.output_file is None

    def test_executing_status(self):
        body = {"jobId": "job-123", "status": "Executing", "messages": []}
        with patch("lib.landfire_lfps.requests.get", return_value=mock_response(body)):
            job = poll_status("job-123")

        assert job.status == "Executing"

    def test_succeeded_status_carries_output_file_and_geo_area(self):
        body = {
            "jobId": "job-123",
            "status": "Succeeded",
            "messages": [],
            "outputFile": "https://lfps.usgs.gov/.../job-123.zip",
            "geoArea": "SW",
        }
        with patch("lib.landfire_lfps.requests.get", return_value=mock_response(body)):
            job = poll_status("job-123")

        assert job.status == "Succeeded"
        assert job.output_file == "https://lfps.usgs.gov/.../job-123.zip"
        assert job.geo_area == "SW"

    def test_failed_status_raises_with_the_error_line(self):
        body = {
            "jobId": "job-123",
            "status": "Failed",
            "messages": [
                {
                    "type": "esriJobMessageTypeInformative",
                    "description": "Start Time: ...",
                },
                {
                    "type": "esriJobMessageTypeInformative",
                    "description": (
                        "ERROR:  Output geotif with no raster statistics. "
                        "Possible Reason: Area of Interest falls outside the "
                        "LF layer list input data."
                    ),
                },
                {"type": "esriJobMessageTypeInformative", "description": "Failed."},
            ],
        }
        with patch("lib.landfire_lfps.requests.get", return_value=mock_response(body)):
            with pytest.raises(LfpsJobFailedError) as exc_info:
                poll_status("job-123")

        message = str(exc_info.value)
        assert "no raster statistics" in message
        assert "Start Time" not in message
        assert message.strip() != "Failed."

    def test_failed_status_falls_back_to_full_trail_without_an_error_line(self):
        body = {
            "jobId": "job-123",
            "status": "Failed",
            "messages": [
                {
                    "type": "esriJobMessageTypeInformative",
                    "description": "Start Time: ...",
                },
                {"type": "esriJobMessageTypeInformative", "description": "Failed."},
            ],
        }
        with patch("lib.landfire_lfps.requests.get", return_value=mock_response(body)):
            with pytest.raises(LfpsJobFailedError) as exc_info:
                poll_status("job-123")

        message = str(exc_info.value)
        assert "Start Time" in message
        assert "Failed." in message


class TestDownload:
    """Tests for fetching a succeeded job's output."""

    def test_fetches_from_output_file(self):
        job = LfpsJob(
            job_id="job-123",
            status="Succeeded",
            output_file="https://lfps.usgs.gov/.../job-123.zip",
        )
        with patch(
            "lib.landfire_lfps.requests.get",
            return_value=mock_response({}, content=b"PK\x03\x04zip-bytes"),
        ) as mock_get:
            content = download(job)

        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == "https://lfps.usgs.gov/.../job-123.zip"
        assert content == b"PK\x03\x04zip-bytes"


def _succeeded_job(output_file: str):
    from lib.landfire_lfps import LfpsJob

    return LfpsJob(job_id="job-123", status="Succeeded", output_file=output_file)
