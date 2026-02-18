"""
Error types for Griddle processing.
"""

from dataclasses import dataclass


class CancelledException(Exception):
    """Raised when a grid document is deleted during processing (user cancelled)."""

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

    def to_dict(self) -> dict:
        """Convert to dict for Firestore storage."""
        result = {"code": self.code, "message": self.message}
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.traceback:
            result["traceback"] = self.traceback
        return result
