"""Shared error types for v2 worker services (griddle, standgen, uploader)."""

from dataclasses import dataclass


class CancelledException(Exception):
    """Raised when a resource document is deleted during processing (user cancelled)."""

    pass


class ProcessingDeferred(Exception):
    """Raised when a handler has already persisted its own continuation.

    Used by multi-step handlers (e.g. an LFPS job that outlives a single
    Cloud Task invocation): the handler writes its state to Firestore and
    enqueues a follow-up task itself, then raises this so the caller stops
    without marking the resource complete or failed.
    """

    pass


@dataclass
class ProcessingError(Exception):
    """Structured error with user-friendly message.

    Attributes:
        code: Machine-readable error code (e.g., "COVERAGE_ERROR")
        message: User-friendly explanation of what went wrong
        suggestion: Actionable advice for the user
        traceback: Full Python stack trace for debugging (not exposed in API)
    """

    code: str
    message: str
    suggestion: str | None = None
    traceback: str | None = None

    def __reduce__(self):
        """Rebuild all fields when this error crosses a process boundary."""
        return (
            type(self),
            (self.code, self.message, self.suggestion, self.traceback),
        )

    def to_dict(self) -> dict:
        """Convert to dict for Firestore storage."""
        result = {"code": self.code, "message": self.message}
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.traceback:
            result["traceback"] = self.traceback
        return result
