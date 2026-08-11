"""Tests for structured worker errors."""

import pickle

from lib.errors import ProcessingError


def test_processing_error_survives_a_process_boundary():
    error = ProcessingError(
        code="EPT_FETCH_FAILED",
        message="Could not decode a node.",
        suggestion="Try again shortly.",
        traceback="https://example.test/node.laz: corrupt",
    )

    restored = pickle.loads(pickle.dumps(error))

    assert restored == error
