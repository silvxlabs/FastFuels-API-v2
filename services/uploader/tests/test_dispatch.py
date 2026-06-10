"""Tests for uploader dispatch."""

from unittest.mock import patch

import pytest
from uploader.dispatch import dispatch_handler

from lib.errors import ProcessingError


def test_unknown_resource_type_raises():
    with pytest.raises(ProcessingError) as exc_info:
        dispatch_handler("widgets", "some-id", "bucket", "widgets/some-id/file.csv", {})
    assert exc_info.value.code == "UNKNOWN_RESOURCE_TYPE"


def test_pointclouds_routes_to_point_cloud_handler():
    doc = {"source": {"name": "upload", "format": "laz"}}
    with patch("uploader.dispatch.handle_point_cloud") as mock_handler:
        dispatch_handler(
            "pointclouds", "pc-1", "uploads", "pointclouds/pc-1/upload.laz", doc
        )
    mock_handler.assert_called_once_with(
        "pc-1", "uploads", "pointclouds/pc-1/upload.laz", doc
    )
