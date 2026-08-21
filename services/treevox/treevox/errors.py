"""Error types for treevox.

Re-exports the shared worker error types from `lib.errors` so treevox,
griddle, and the other worker services all raise and catch the *same*
`ProcessingError` / `CancelledException` classes. This matters now that
`lib.inventory_io` (used by treevox voxelization) raises `lib.errors`
errors: treevox's top-level `except ProcessingError` must resolve to the
same class the shared reader raises.

Codes emitted by treevox:
  INVENTORY_NOT_FOUND, INVENTORY_MISSING_MORPHOLOGY, EMPTY_INVENTORY,
  INVALID_RESOLUTION, BIOMASS_COMPONENT_NOT_IMPLEMENTED, UNKNOWN_SOURCE,
  VOXELIZATION_FAILED, DOMAIN_NOT_FOUND, EMPTY_DOMAIN, INVALID_GEOMETRY.
"""

from __future__ import annotations

from lib.errors import CancelledException, ProcessingError

__all__ = ["CancelledException", "ProcessingError"]
