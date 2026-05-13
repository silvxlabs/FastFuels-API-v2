"""Tests for uploader dispatch."""

import pytest
from uploader.dispatch import dispatch_handler

from lib.errors import ProcessingError


def test_unknown_resource_type_raises():
    with pytest.raises(ProcessingError) as exc_info:
        dispatch_handler("widgets", "some-id", "bucket", "widgets/some-id/file.csv", {})
    assert exc_info.value.code == "UNKNOWN_RESOURCE_TYPE"
