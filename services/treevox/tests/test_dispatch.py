"""Unit tests for treevox.dispatch — source routing only.

The voxelization job itself (and its stages) is covered by
tests/handlers/test_voxelize.py; here we only assert that the
(operation, input, entity) triple routes to the right handler.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from treevox.dispatch import dispatch_handler
from treevox.errors import ProcessingError


class TestDispatchHandler:
    @patch("treevox.handlers.voxelize.voxelize_inventory")
    def test_inventory_routes_to_voxelize(self, mock_voxelize):
        mock_voxelize.return_value = "result"
        grid = {
            "source": {
                "operation": "voxelize",
                "input": "inventory",
                "entity": "tree",
            }
        }
        result = dispatch_handler(grid, MagicMock(), lambda *a, **k: None)
        assert result == "result"
        mock_voxelize.assert_called_once()

    @patch("treevox.handlers.duet.duet_grid")
    def test_duet_grid_routes_to_duet(self, mock_duet):
        mock_duet.return_value = "result"
        grid = {
            "source": {
                "operation": "duet",
                "input": "grid",
                "entity": "tree",
            }
        }
        result = dispatch_handler(grid, MagicMock(), lambda *a, **k: None)
        assert result == "result"
        mock_duet.assert_called_once()

    def test_unknown_source_raises_processing_error(self):
        grid = {
            "source": {
                "operation": "voxelize",
                "input": "inventory",
                "entity": "lidar",
            }
        }
        with pytest.raises(ProcessingError) as exc:
            dispatch_handler(grid, MagicMock(), lambda *a, **k: None)
        assert exc.value.code == "UNKNOWN_SOURCE"

    def test_duet_from_an_inventory_is_not_a_route(self):
        # DUET consumes a grid, not an inventory — the input leg of the triple
        # is what distinguishes it from voxelize.
        grid = {
            "source": {
                "operation": "duet",
                "input": "inventory",
                "entity": "tree",
            }
        }
        with pytest.raises(ProcessingError) as exc:
            dispatch_handler(grid, MagicMock(), lambda *a, **k: None)
        assert exc.value.code == "UNKNOWN_SOURCE"
