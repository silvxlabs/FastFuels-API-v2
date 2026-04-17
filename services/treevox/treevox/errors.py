"""Error types for treevox.

Kept separate so every module can import them without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass


class CancelledException(Exception):
    """Raised when a grid document is deleted during processing (user cancelled)."""


@dataclass
class ProcessingError(Exception):
    """Structured error with a user-friendly message.

    Codes emitted by treevox:
      INVENTORY_NOT_FOUND, EMPTY_INVENTORY, INVALID_RESOLUTION,
      UNKNOWN_SOURCE, VOXELIZATION_FAILED, DOMAIN_NOT_FOUND,
      EMPTY_DOMAIN, INVALID_GEOMETRY.
    """

    code: str
    message: str
    suggestion: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict:
        result: dict = {"code": self.code, "message": self.message}
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.traceback:
            result["traceback"] = self.traceback
        return result
