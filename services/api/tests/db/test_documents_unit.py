"""
Unit tests for api.db.documents module.

Tests get_document_async and set_document_async functions with mocked Firestore client.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION

MOCK_OWNER_ID = "TEST_OWNER"

# Apply to all tests in this module
pytestmark = pytest.mark.anyio


# Patch the environment variable before importing the module
@pytest.fixture(scope="module", autouse=True)
def mock_env():
    """Mock the V2_FIRESTORE_DATABASE_ID environment variable."""
    with patch.dict("os.environ", {"V2_FIRESTORE_DATABASE_ID": "test-database"}):
        yield


# We need to import after patching the environment variable
# Use a fixture to handle the import
@pytest.fixture
def documents_module():
    """Import documents module with mocked environment."""
    with patch.dict("os.environ", {"V2_FIRESTORE_DATABASE_ID": "test-database"}):
        with patch("google.cloud.firestore.AsyncClient"):
            from api.db import documents

            yield documents


@pytest.fixture
def mock_document_snapshot():
    """Factory fixture for creating mock document snapshots."""

    def _create_snapshot(exists: bool = True, data: dict | None = None):
        snapshot = MagicMock()
        snapshot.exists = exists
        snapshot.to_dict.return_value = data if data else {}
        return snapshot

    return _create_snapshot


@pytest.fixture
def mock_document_ref():
    """Create a mock async document reference."""
    ref = AsyncMock()
    return ref


class TestGetDocumentAsync:
    """Tests for get_document_async function."""

    async def test_document_found_returns_ref_and_snapshot(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test successful document retrieval returns (ref, snapshot) tuple."""
        snapshot = mock_document_snapshot(exists=True, data={"field": "value"})
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            ref, result_snapshot = await documents_module.get_document_async(
                collection=DOMAINS_COLLECTION,
                document_id="test-id",
            )

        assert ref == mock_document_ref
        assert result_snapshot == snapshot

    async def test_document_not_found_raises_404(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test that non-existent document raises 404 HTTPException."""
        snapshot = mock_document_snapshot(exists=False)
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await documents_module.get_document_async(
                    collection=DOMAINS_COLLECTION,
                    document_id="nonexistent-id",
                )

        assert exc_info.value.status_code == 404
        assert "Document not found: domains/nonexistent-id" in exc_info.value.detail

    async def test_owner_validation_success(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test document retrieval succeeds when owner_id matches."""
        snapshot = mock_document_snapshot(
            exists=True, data={"owner_id": "user-123", "name": "test"}
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            ref, result_snapshot = await documents_module.get_document_async(
                collection=DOMAINS_COLLECTION,
                document_id="test-id",
                owner_id="user-123",
            )

        assert ref == mock_document_ref
        assert result_snapshot == snapshot

    async def test_owner_validation_mismatch_raises_404(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test that owner_id mismatch raises 404 (not 403) for security."""
        snapshot = mock_document_snapshot(
            exists=True, data={"owner_id": "user-123", "name": "test"}
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await documents_module.get_document_async(
                    collection=DOMAINS_COLLECTION,
                    document_id="test-id",
                    owner_id="different-user",
                )

        # Should return 404 to avoid leaking document existence
        assert exc_info.value.status_code == 404
        assert "Document not found" in exc_info.value.detail

    async def test_owner_validation_missing_owner_field_raises_404(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test that missing owner_id field in document raises 404."""
        snapshot = mock_document_snapshot(
            exists=True,
            data={"name": "test"},  # No ownerId field
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await documents_module.get_document_async(
                    collection=DOMAINS_COLLECTION,
                    document_id="test-id",
                    owner_id="user-123",
                )

        assert exc_info.value.status_code == 404

    async def test_status_validation_success(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test document retrieval succeeds when status matches."""
        snapshot = mock_document_snapshot(
            exists=True, data={"status": "completed", "name": "test"}
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            ref, result_snapshot = await documents_module.get_document_async(
                collection="grids",
                document_id="grid-123",
                document_status="completed",
            )

        assert ref == mock_document_ref
        assert result_snapshot == snapshot

    async def test_status_validation_mismatch_raises_422(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test that status mismatch raises 422 with descriptive message."""
        snapshot = mock_document_snapshot(
            exists=True, data={"status": "pending", "name": "test"}
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await documents_module.get_document_async(
                    collection="grids",
                    document_id="grid-123",
                    document_status="completed",
                )

        assert exc_info.value.status_code == 422
        assert "status is 'pending'" in exc_info.value.detail
        assert "expected 'completed'" in exc_info.value.detail

    async def test_combined_owner_and_status_validation_success(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test document retrieval succeeds when both owner_id and status match."""
        snapshot = mock_document_snapshot(
            exists=True,
            data={"owner_id": "user-123", "status": "completed", "name": "test"},
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            ref, result_snapshot = await documents_module.get_document_async(
                collection="grids",
                document_id="grid-123",
                owner_id="user-123",
                document_status="completed",
            )

        assert ref == mock_document_ref
        assert result_snapshot == snapshot

    async def test_combined_validation_owner_fails_first(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test that owner validation is checked before status validation."""
        snapshot = mock_document_snapshot(
            exists=True,
            data={"owner_id": "user-123", "status": "pending", "name": "test"},
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await documents_module.get_document_async(
                    collection="grids",
                    document_id="grid-123",
                    owner_id="different-user",  # Wrong owner
                    document_status="completed",  # Also wrong status
                )

        # Owner check should fail first with 404
        assert exc_info.value.status_code == 404

    async def test_combined_validation_status_fails_when_owner_passes(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test that status validation fails with 422 when owner passes."""
        snapshot = mock_document_snapshot(
            exists=True,
            data={"owner_id": "user-123", "status": "pending", "name": "test"},
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await documents_module.get_document_async(
                    collection="grids",
                    document_id="grid-123",
                    owner_id="user-123",  # Correct owner
                    document_status="completed",  # Wrong status
                )

        # Status check should fail with 422
        assert exc_info.value.status_code == 422

    async def test_domain_id_validation_success(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test document retrieval succeeds when domain_id matches."""
        snapshot = mock_document_snapshot(
            exists=True,
            data={"owner_id": "user-123", "domain_id": "domain-abc", "name": "test"},
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            ref, result_snapshot = await documents_module.get_document_async(
                collection=GRIDS_COLLECTION,
                document_id="grid-123",
                owner_id="user-123",
                domain_id="domain-abc",
            )

        assert ref == mock_document_ref
        assert result_snapshot == snapshot

    async def test_domain_id_validation_mismatch_raises_404(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test that domain_id mismatch raises 404."""
        snapshot = mock_document_snapshot(
            exists=True,
            data={"owner_id": "user-123", "domain_id": "domain-abc", "name": "test"},
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await documents_module.get_document_async(
                    collection=GRIDS_COLLECTION,
                    document_id="grid-123",
                    owner_id="user-123",
                    domain_id="domain-xyz",
                )

        assert exc_info.value.status_code == 404
        assert "Document not found" in exc_info.value.detail

    async def test_domain_id_validation_missing_field_raises_404(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test that missing domain_id field in document raises 404."""
        snapshot = mock_document_snapshot(
            exists=True,
            data={"owner_id": "user-123", "name": "test"},
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await documents_module.get_document_async(
                    collection=GRIDS_COLLECTION,
                    document_id="grid-123",
                    owner_id="user-123",
                    domain_id="domain-abc",
                )

        assert exc_info.value.status_code == 404

    async def test_owner_fails_before_domain_id(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test that owner validation is checked before domain_id validation."""
        snapshot = mock_document_snapshot(
            exists=True,
            data={"owner_id": "user-123", "domain_id": "domain-abc"},
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await documents_module.get_document_async(
                    collection=GRIDS_COLLECTION,
                    document_id="grid-123",
                    owner_id="wrong-user",
                    domain_id="domain-xyz",
                )

        # Owner check should fail first with 404
        assert exc_info.value.status_code == 404

    async def test_domain_id_fails_before_status(
        self, documents_module, mock_document_snapshot, mock_document_ref
    ):
        """Test that domain_id validation is checked before status validation."""
        snapshot = mock_document_snapshot(
            exists=True,
            data={
                "owner_id": "user-123",
                "domain_id": "domain-abc",
                "status": "pending",
            },
        )
        mock_document_ref.get = AsyncMock(return_value=snapshot)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await documents_module.get_document_async(
                    collection=GRIDS_COLLECTION,
                    document_id="grid-123",
                    owner_id="user-123",
                    domain_id="domain-xyz",
                    document_status="completed",
                )

        # Domain check should fail with 404 before status check
        assert exc_info.value.status_code == 404


class TestSetDocumentAsync:
    """Tests for set_document_async function."""

    async def test_sets_document_and_returns_ref(
        self, documents_module, mock_document_ref
    ):
        """Test that document is set and reference is returned."""
        mock_document_ref.set = AsyncMock()

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            result_ref = await documents_module.set_document_async(
                collection=DOMAINS_COLLECTION,
                document_id="test-id",
                data={"name": "Test Domain"},
            )

        assert result_ref == mock_document_ref
        mock_document_ref.set.assert_called_once_with({"name": "Test Domain"})

    async def test_calls_firestore_with_correct_collection_and_id(
        self, documents_module, mock_document_ref
    ):
        """Test that correct collection and document_id are used."""
        mock_document_ref.set = AsyncMock()
        mock_document_func = MagicMock(return_value=mock_document_ref)
        mock_collection = MagicMock(document=mock_document_func)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=mock_collection,
        ) as mock_collection_call:
            await documents_module.set_document_async(
                collection="grids",
                document_id="grid-123",
                data={"field": "value"},
            )

        mock_collection_call.assert_called_once_with("grids")
        mock_document_func.assert_called_once_with("grid-123")

    async def test_passes_data_to_firestore_set(
        self, documents_module, mock_document_ref
    ):
        """Test that data dict is passed correctly to Firestore set."""
        mock_document_ref.set = AsyncMock()
        test_data = {
            "name": "Test",
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            await documents_module.set_document_async(
                collection=DOMAINS_COLLECTION,
                document_id="test-id",
                data=test_data,
            )

        mock_document_ref.set.assert_called_once_with(test_data)


class TestListDocumentsAsync:
    """Tests for list_documents_async function."""

    @pytest.fixture
    def mock_query(self):
        """Create a mock query with chainable methods."""
        query = MagicMock()
        query.where = MagicMock(return_value=query)
        query.order_by = MagicMock(return_value=query)
        query.offset = MagicMock(return_value=query)
        query.limit = MagicMock(return_value=query)
        return query

    @pytest.fixture
    def mock_aggregation_result(self):
        """Create a mock aggregation result."""

        def _create_result(count: int):
            mock_value = MagicMock()
            mock_value.value = count
            return [[mock_value]]

        return _create_result

    async def test_returns_documents_and_count(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that function returns tuple of (documents, total_count)."""
        mock_docs = [MagicMock(), MagicMock()]
        mock_query.get = AsyncMock(return_value=mock_docs)
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(5))
            )
        )

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            docs, total = await documents_module.list_documents_async(
                collection=DOMAINS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
            )

        assert len(docs) == 2
        assert total == 5

    async def test_filters_by_owner_id(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that query filters by owner_id."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )
        mock_where = MagicMock(return_value=mock_query)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=mock_where),
        ):
            await documents_module.list_documents_async(
                collection=DOMAINS_COLLECTION,
                owner_id="user-123",
            )

        # Verify filter was called with owner_id
        mock_where.assert_called_once()
        call_args = mock_where.call_args
        filter_arg = call_args.kwargs.get("filter") or call_args.args[0]
        assert filter_arg.field_path == "owner_id"
        assert filter_arg.value == "user-123"

    async def test_applies_pagination(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that pagination parameters are applied."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            await documents_module.list_documents_async(
                collection=DOMAINS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                page=2,
                size=25,
            )

        # Page 2 with size 25 should offset by 50
        mock_query.offset.assert_called_once_with(50)
        mock_query.limit.assert_called_once_with(25)

    async def test_applies_sorting_descending(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that sorting is applied with descending order."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            await documents_module.list_documents_async(
                collection=DOMAINS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                sort_by="created_on",
                sort_order="descending",
            )

        mock_query.order_by.assert_called_once()
        call_args = mock_query.order_by.call_args
        assert call_args.args[0] == "created_on"

    async def test_applies_sorting_ascending(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that sorting is applied with ascending order."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            await documents_module.list_documents_async(
                collection=DOMAINS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                sort_by="name",
                sort_order="ascending",
            )

        mock_query.order_by.assert_called_once()

    async def test_no_sorting_when_sort_by_not_specified(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that no sorting is applied when sort_by is None."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            await documents_module.list_documents_async(
                collection=DOMAINS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                sort_by=None,
            )

        mock_query.order_by.assert_not_called()

    async def test_default_pagination_values(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that default page=0 and size=100 are used."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            await documents_module.list_documents_async(
                collection=DOMAINS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
            )

        # Default page=0, size=100 means offset=0, limit=100
        mock_query.offset.assert_called_once_with(0)
        mock_query.limit.assert_called_once_with(100)

    async def test_applies_single_filter(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that a single filter is applied to the query."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )
        mock_where = MagicMock(return_value=mock_query)
        mock_query.where = mock_where

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            await documents_module.list_documents_async(
                collection=GRIDS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                filters={"domain_id": "domain-123"},
            )

        # Verify filter was called for domain_id
        mock_where.assert_called_once()
        call_args = mock_where.call_args
        filter_arg = call_args.kwargs.get("filter") or call_args.args[0]
        assert filter_arg.field_path == "domain_id"
        assert filter_arg.value == "domain-123"

    async def test_applies_multiple_filters(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that multiple filters are all applied to the query."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )
        mock_where = MagicMock(return_value=mock_query)
        mock_query.where = mock_where

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            await documents_module.list_documents_async(
                collection=GRIDS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                filters={"domain_id": "domain-123", "status": "completed"},
            )

        # Verify where was called twice (once per filter)
        assert mock_where.call_count == 2

        # Collect all filter field paths
        filter_fields = set()
        for call in mock_where.call_args_list:
            filter_arg = call.kwargs.get("filter") or call.args[0]
            filter_fields.add(filter_arg.field_path)

        assert "domain_id" in filter_fields
        assert "status" in filter_fields

    async def test_empty_filters_dict_same_as_none(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that empty filters dict behaves the same as None."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )
        mock_where = MagicMock(return_value=mock_query)
        mock_query.where = mock_where

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            await documents_module.list_documents_async(
                collection=DOMAINS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                filters={},
            )

        # Verify where was not called on the query (empty filters)
        mock_where.assert_not_called()

    async def test_filters_with_owner_id_both_applied(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that filters are applied in addition to owner_id filter."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )
        mock_where_initial = MagicMock(return_value=mock_query)
        mock_where_chain = MagicMock(return_value=mock_query)
        mock_query.where = mock_where_chain

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=mock_where_initial),
        ):
            await documents_module.list_documents_async(
                collection=GRIDS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                filters={"domain_id": "domain-456"},
            )

        # Verify initial where was called with owner_id
        mock_where_initial.assert_called_once()
        initial_filter = (
            mock_where_initial.call_args.kwargs.get("filter")
            or mock_where_initial.call_args.args[0]
        )
        assert initial_filter.field_path == "owner_id"
        assert initial_filter.value == MOCK_OWNER_ID

        # Verify chained where was called with domain_id
        mock_where_chain.assert_called_once()
        chain_filter = (
            mock_where_chain.call_args.kwargs.get("filter")
            or mock_where_chain.call_args.args[0]
        )
        assert chain_filter.field_path == "domain_id"
        assert chain_filter.value == "domain-456"

    async def test_filters_applied_before_count(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that filters are applied before counting (accurate total_count)."""
        mock_query.get = AsyncMock(return_value=[])
        mock_count_query = MagicMock(
            get=AsyncMock(return_value=mock_aggregation_result(3))
        )
        mock_query.count = MagicMock(return_value=mock_count_query)
        mock_where = MagicMock(return_value=mock_query)
        mock_query.where = mock_where

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            docs, total_count = await documents_module.list_documents_async(
                collection=GRIDS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                filters={"domain_id": "domain-123"},
            )

        # Verify count was called after filters were applied
        mock_query.count.assert_called_once()
        assert total_count == 3

    async def test_applies_array_contains_filter(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that array_contains filter is applied to the query."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )
        mock_where = MagicMock(return_value=mock_query)
        mock_query.where = mock_where

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            await documents_module.list_documents_async(
                collection=GRIDS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                array_contains_filters={"tags": "surface-fuel"},
            )

        # Verify where was called with array_contains operator
        mock_where.assert_called_once()
        call_args = mock_where.call_args
        filter_arg = call_args.kwargs.get("filter") or call_args.args[0]
        assert filter_arg.field_path == "tags"
        assert filter_arg.op_string == "array_contains"
        assert filter_arg.value == "surface-fuel"

    async def test_combines_filters_and_array_contains_filters(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that both equality and array_contains filters are applied."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )
        mock_where = MagicMock(return_value=mock_query)
        mock_query.where = mock_where

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            await documents_module.list_documents_async(
                collection=GRIDS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                filters={"source.name": "landfire"},
                array_contains_filters={"tags": "baseline"},
            )

        # Verify where was called twice (once for equality, once for array_contains)
        assert mock_where.call_count == 2

        # Collect filter info
        filter_ops = {}
        for call in mock_where.call_args_list:
            filter_arg = call.kwargs.get("filter") or call.args[0]
            filter_ops[filter_arg.field_path] = filter_arg.op_string

        assert filter_ops.get("source.name") == "=="
        assert filter_ops.get("tags") == "array_contains"

    async def test_empty_array_contains_filters_dict_same_as_none(
        self, documents_module, mock_query, mock_aggregation_result
    ):
        """Test that empty array_contains_filters dict behaves the same as None."""
        mock_query.get = AsyncMock(return_value=[])
        mock_query.count = MagicMock(
            return_value=MagicMock(
                get=AsyncMock(return_value=mock_aggregation_result(0))
            )
        )
        mock_where = MagicMock(return_value=mock_query)
        mock_query.where = mock_where

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(where=MagicMock(return_value=mock_query)),
        ):
            await documents_module.list_documents_async(
                collection=DOMAINS_COLLECTION,
                owner_id=MOCK_OWNER_ID,
                array_contains_filters={},
            )

        # Verify where was not called on the query (empty filters)
        mock_where.assert_not_called()


class TestUpdateDocumentAsync:
    """Tests for update_document_async function."""

    async def test_updates_document_and_returns_ref(
        self, documents_module, mock_document_ref
    ):
        """Test that document is updated and reference is returned."""
        mock_document_ref.update = AsyncMock()

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            result_ref = await documents_module.update_document_async(
                collection=DOMAINS_COLLECTION,
                document_id="test-id",
                data={"name": "Updated Name"},
            )

        assert result_ref == mock_document_ref
        mock_document_ref.update.assert_called_once_with({"name": "Updated Name"})

    async def test_calls_firestore_with_correct_collection_and_id(
        self, documents_module, mock_document_ref
    ):
        """Test that correct collection and document_id are used."""
        mock_document_ref.update = AsyncMock()
        mock_document_func = MagicMock(return_value=mock_document_ref)
        mock_collection = MagicMock(document=mock_document_func)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=mock_collection,
        ) as mock_collection_call:
            await documents_module.update_document_async(
                collection=GRIDS_COLLECTION,
                document_id="grid-123",
                data={"status": "completed"},
            )

        mock_collection_call.assert_called_once_with(GRIDS_COLLECTION)
        mock_document_func.assert_called_once_with("grid-123")

    async def test_passes_partial_data_to_firestore_update(
        self, documents_module, mock_document_ref
    ):
        """Test that data dict is passed correctly to Firestore update."""
        mock_document_ref.update = AsyncMock()
        test_data = {
            "name": "New Name",
            "tags": ["updated", "test"],
        }

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            await documents_module.update_document_async(
                collection=DOMAINS_COLLECTION,
                document_id="test-id",
                data=test_data,
            )

        mock_document_ref.update.assert_called_once_with(test_data)

    async def test_handles_single_field_update(
        self, documents_module, mock_document_ref
    ):
        """Test that single field updates work correctly."""
        mock_document_ref.update = AsyncMock()

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            await documents_module.update_document_async(
                collection=DOMAINS_COLLECTION,
                document_id="test-id",
                data={"description": "New description"},
            )

        mock_document_ref.update.assert_called_once_with(
            {"description": "New description"}
        )


class TestDeleteDocumentAsync:
    """Tests for delete_document_async function."""

    async def test_deletes_document(self, documents_module, mock_document_ref):
        """Test that document is deleted."""
        mock_document_ref.delete = AsyncMock()

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            await documents_module.delete_document_async(
                collection=DOMAINS_COLLECTION,
                document_id="test-id",
            )

        mock_document_ref.delete.assert_called_once()

    async def test_calls_firestore_with_correct_collection_and_id(
        self, documents_module, mock_document_ref
    ):
        """Test that correct collection and document_id are used."""
        mock_document_ref.delete = AsyncMock()
        mock_document_func = MagicMock(return_value=mock_document_ref)
        mock_collection = MagicMock(document=mock_document_func)

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=mock_collection,
        ) as mock_collection_call:
            await documents_module.delete_document_async(
                collection=GRIDS_COLLECTION,
                document_id="grid-123",
            )

        mock_collection_call.assert_called_once_with(GRIDS_COLLECTION)
        mock_document_func.assert_called_once_with("grid-123")

    async def test_returns_none(self, documents_module, mock_document_ref):
        """Test that delete returns None."""
        mock_document_ref.delete = AsyncMock()

        with patch.object(
            documents_module.firestore_client,
            "collection",
            return_value=MagicMock(document=MagicMock(return_value=mock_document_ref)),
        ):
            result = await documents_module.delete_document_async(
                collection=DOMAINS_COLLECTION,
                document_id="test-id",
            )

        assert result is None
