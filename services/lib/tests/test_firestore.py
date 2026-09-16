"""Tests for the shared Firestore cancellation-guard primitive."""

from unittest.mock import patch

import pytest

from lib.errors import CancelledException
from lib.firestore import update_document_or_cancel
from lib.firestore.documents import DocumentNotFoundError


def test_update_document_or_cancel_passes_through_on_success():
    """The success path delegates to update_document with the same args."""
    with patch("lib.firestore.documents.update_document") as mock_update:
        result = update_document_or_cancel("grids", "g1", {"status": "running"})

    assert result is None
    mock_update.assert_called_once_with("grids", "g1", {"status": "running"})


def test_update_document_or_cancel_maps_missing_doc_to_cancelled():
    """A missing target document surfaces as CancelledException."""
    with patch(
        "lib.firestore.documents.update_document",
        side_effect=DocumentNotFoundError("Document not found: grids/g1"),
    ):
        with pytest.raises(CancelledException):
            update_document_or_cancel("grids", "g1", {"status": "running"})
