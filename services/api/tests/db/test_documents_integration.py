"""
Integration tests for api.db.documents module.

Tests get_document_async against real Firestore.
"""

import uuid

import pytest
import pytest_asyncio
from api.db.documents import (
    delete_document_async,
    get_document_async,
    list_documents_async,
    set_document_async,
    update_document_async,
)
from fastapi import HTTPException
from google.cloud import firestore

MOCK_OWNER_ID = "TEST_OWNER"

# Apply to all tests in this module
pytestmark = pytest.mark.asyncio(loop_scope="session")

TEST_COLLECTION = "_test_documents"


@pytest.fixture(scope="session")
def firestore_client():
    """Session-scoped Firestore client."""
    return firestore.AsyncClient()


@pytest_asyncio.fixture(loop_scope="session")
async def test_document(firestore_client):
    """Create a test document, yield its ID, then delete it."""
    doc_id = f"test-{uuid.uuid4()}"
    doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
    await doc_ref.set(
        {
            "name": "Test Document",
            "owner_id": "test-owner-123",
            "status": "completed",
        }
    )
    yield doc_id
    await doc_ref.delete()


@pytest_asyncio.fixture(loop_scope="session")
async def test_document_no_owner(firestore_client):
    """Create a test document without owner_id."""
    doc_id = f"test-{uuid.uuid4()}"
    doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
    await doc_ref.set({"name": "No Owner", "status": "pending"})
    yield doc_id
    await doc_ref.delete()


class TestGetDocumentAsync:
    """Integration tests for get_document_async."""

    async def test_not_found_raises_404(self):
        """Returns 404 for non-existent document."""
        with pytest.raises(HTTPException) as exc:
            await get_document_async(TEST_COLLECTION, "does-not-exist")

        assert exc.value.status_code == 404
        assert "Document not found" in exc.value.detail

    async def test_retrieves_existing_document(self, test_document):
        """Successfully retrieves a document that exists."""
        ref, snapshot = await get_document_async(TEST_COLLECTION, test_document)

        data = snapshot.to_dict()
        assert data["name"] == "Test Document"
        assert data["owner_id"] == "test-owner-123"

    async def test_owner_validation_passes(self, test_document):
        """Succeeds when owner_id matches."""
        ref, snapshot = await get_document_async(
            TEST_COLLECTION, test_document, owner_id="test-owner-123"
        )
        assert snapshot.to_dict()["owner_id"] == "test-owner-123"

    async def test_owner_validation_fails_with_404(self, test_document):
        """Returns 404 (not 403) when owner_id doesn't match."""
        with pytest.raises(HTTPException) as exc:
            await get_document_async(
                TEST_COLLECTION, test_document, owner_id="wrong-owner"
            )
        assert exc.value.status_code == 404

    async def test_missing_owner_field_fails_with_404(self, test_document_no_owner):
        """Returns 404 when document has no owner_id field."""
        with pytest.raises(HTTPException) as exc:
            await get_document_async(
                TEST_COLLECTION, test_document_no_owner, owner_id="any-owner"
            )
        assert exc.value.status_code == 404

    async def test_status_validation_passes(self, test_document):
        """Succeeds when status matches."""
        ref, snapshot = await get_document_async(
            TEST_COLLECTION, test_document, document_status="completed"
        )
        assert snapshot.to_dict()["status"] == "completed"

    async def test_status_validation_fails_with_422(self, test_document):
        """Returns 422 with descriptive message when status doesn't match."""
        with pytest.raises(HTTPException) as exc:
            await get_document_async(
                TEST_COLLECTION, test_document, document_status="pending"
            )

        assert exc.value.status_code == 422
        assert "status is 'completed'" in exc.value.detail
        assert "expected 'pending'" in exc.value.detail

    async def test_combined_validation_passes(self, test_document):
        """Succeeds when both owner_id and status match."""
        ref, snapshot = await get_document_async(
            TEST_COLLECTION,
            test_document,
            owner_id="test-owner-123",
            document_status="completed",
        )

        data = snapshot.to_dict()
        assert data["owner_id"] == "test-owner-123"
        assert data["status"] == "completed"

    async def test_owner_checked_before_status(self, test_document):
        """Owner validation fails first (404) even if status also wrong."""
        with pytest.raises(HTTPException) as exc:
            await get_document_async(
                TEST_COLLECTION,
                test_document,
                owner_id="wrong-owner",
                document_status="wrong-status",
            )
        # Should get 404 from owner check, not 422 from status check
        assert exc.value.status_code == 404


class TestSetDocumentAsync:
    """Integration tests for set_document_async."""

    async def test_creates_new_document(self, firestore_client):
        """Successfully creates a new document."""
        doc_id = f"test-set-{uuid.uuid4()}"
        test_data = {"name": "Created Document", "value": 42}

        try:
            ref = await set_document_async(TEST_COLLECTION, doc_id, test_data)

            # Verify document was created
            snapshot = await ref.get()
            assert snapshot.exists
            assert snapshot.to_dict() == test_data
        finally:
            # Cleanup
            await firestore_client.collection(TEST_COLLECTION).document(doc_id).delete()

    async def test_overwrites_existing_document(self, firestore_client):
        """Overwrites an existing document completely."""
        doc_id = f"test-overwrite-{uuid.uuid4()}"
        original_data = {"name": "Original", "extra_field": "should disappear"}
        new_data = {"name": "Overwritten"}

        try:
            # Create original document
            doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
            await doc_ref.set(original_data)

            # Overwrite with set_document_async
            await set_document_async(TEST_COLLECTION, doc_id, new_data)

            # Verify document was overwritten (extra_field should be gone)
            snapshot = await doc_ref.get()
            assert snapshot.to_dict() == new_data
            assert "extra_field" not in snapshot.to_dict()
        finally:
            await firestore_client.collection(TEST_COLLECTION).document(doc_id).delete()

    async def test_returns_document_reference(self, firestore_client):
        """Returns a valid document reference."""
        doc_id = f"test-ref-{uuid.uuid4()}"

        try:
            ref = await set_document_async(TEST_COLLECTION, doc_id, {"name": "Test"})

            assert ref.id == doc_id
            assert ref.parent.id == TEST_COLLECTION
        finally:
            await firestore_client.collection(TEST_COLLECTION).document(doc_id).delete()

    async def test_handles_nested_data(self, firestore_client):
        """Correctly stores nested data structures."""
        doc_id = f"test-nested-{uuid.uuid4()}"
        nested_data = {
            "name": "Nested Test",
            "metadata": {
                "created_by": "test",
                "tags": ["a", "b", "c"],
            },
            "coordinates": "[[[0, 0], [1, 0], [1, 1], [0, 0]]]",  # Stringified coords
        }

        try:
            ref = await set_document_async(TEST_COLLECTION, doc_id, nested_data)

            snapshot = await ref.get()
            assert snapshot.to_dict() == nested_data
        finally:
            await firestore_client.collection(TEST_COLLECTION).document(doc_id).delete()

    async def test_round_trip_with_get_document_async(self, firestore_client):
        """Document set with set_document_async can be retrieved with get_document_async."""
        doc_id = f"test-roundtrip-{uuid.uuid4()}"
        test_data = {"name": "Round Trip", "owner_id": "test-user", "status": "active"}

        try:
            await set_document_async(TEST_COLLECTION, doc_id, test_data)

            ref, snapshot = await get_document_async(
                TEST_COLLECTION, doc_id, owner_id="test-user"
            )

            assert snapshot.to_dict() == test_data
        finally:
            await firestore_client.collection(TEST_COLLECTION).document(doc_id).delete()


class TestListDocumentsAsync:
    """Integration tests for list_documents_async."""

    @pytest_asyncio.fixture(loop_scope="session")
    async def list_test_documents(self, firestore_client):
        """Create multiple test documents for list testing."""
        owner_id = f"list-test-owner-{uuid.uuid4()}"
        doc_ids = []

        # Create 5 documents with different names and timestamps
        for i in range(5):
            doc_id = f"list-test-{uuid.uuid4()}"
            doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
            await doc_ref.set(
                {
                    "name": f"Document {chr(65 + i)}",  # A, B, C, D, E
                    "owner_id": owner_id,
                    "created_on": f"2024-01-{15 + i:02d}T10:00:00",
                    "index": i,
                }
            )
            doc_ids.append(doc_id)

        yield {"owner_id": owner_id, "doc_ids": doc_ids}

        # Cleanup
        for doc_id in doc_ids:
            await firestore_client.collection(TEST_COLLECTION).document(doc_id).delete()

    @pytest_asyncio.fixture(loop_scope="session")
    async def different_owner_document(self, firestore_client):
        """Create a document owned by a different user."""
        doc_id = f"different-owner-{uuid.uuid4()}"
        doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
        await doc_ref.set(
            {
                "name": "Different Owner Doc",
                "owner_id": "other-owner-xyz",
            }
        )
        yield doc_id
        await doc_ref.delete()

    async def test_returns_documents_for_owner(self, list_test_documents):
        """Returns only documents belonging to the specified owner."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=list_test_documents["owner_id"],
        )

        assert total == 5
        assert len(docs) == 5
        for doc in docs:
            assert doc.to_dict()["owner_id"] == list_test_documents["owner_id"]

    async def test_excludes_other_owners_documents(
        self, list_test_documents, different_owner_document
    ):
        """Does not return documents from other owners."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=list_test_documents["owner_id"],
        )

        doc_ids = [doc.id for doc in docs]
        assert different_owner_document not in doc_ids

    async def test_pagination_first_page(self, list_test_documents):
        """Returns correct items for first page."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=list_test_documents["owner_id"],
            page=0,
            size=2,
        )

        assert total == 5
        assert len(docs) == 2

    async def test_pagination_second_page(self, list_test_documents):
        """Returns correct items for second page."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=list_test_documents["owner_id"],
            page=1,
            size=2,
        )

        assert total == 5
        assert len(docs) == 2

    async def test_pagination_last_page_partial(self, list_test_documents):
        """Returns partial results on last page."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=list_test_documents["owner_id"],
            page=2,
            size=2,
        )

        assert total == 5
        assert len(docs) == 1  # Only 1 item left on page 2

    async def test_pagination_beyond_results(self, list_test_documents):
        """Returns empty list when page is beyond available data."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=list_test_documents["owner_id"],
            page=10,
            size=2,
        )

        assert total == 5
        assert len(docs) == 0

    async def test_sorting_by_name_ascending(self, list_test_documents):
        """Sorts documents by name in ascending order."""
        docs, _ = await list_documents_async(
            TEST_COLLECTION,
            owner_id=list_test_documents["owner_id"],
            sort_by="name",
            sort_order="ascending",
        )

        names = [doc.to_dict()["name"] for doc in docs]
        assert names == sorted(names)

    async def test_sorting_by_name_descending(self, list_test_documents):
        """Sorts documents by name in descending order."""
        docs, _ = await list_documents_async(
            TEST_COLLECTION,
            owner_id=list_test_documents["owner_id"],
            sort_by="name",
            sort_order="descending",
        )

        names = [doc.to_dict()["name"] for doc in docs]
        assert names == sorted(names, reverse=True)

    async def test_empty_result_for_nonexistent_owner(self):
        """Returns empty list and zero count for owner with no documents."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id="nonexistent-owner-12345",
        )

        assert total == 0
        assert len(docs) == 0

    async def test_total_count_independent_of_page_size(self, list_test_documents):
        """Total count reflects all documents regardless of page size."""
        docs1, total1 = await list_documents_async(
            TEST_COLLECTION,
            owner_id=list_test_documents["owner_id"],
            page=0,
            size=1,
        )
        docs2, total2 = await list_documents_async(
            TEST_COLLECTION,
            owner_id=list_test_documents["owner_id"],
            page=0,
            size=100,
        )

        assert total1 == total2 == 5
        assert len(docs1) == 1
        assert len(docs2) == 5

    @pytest_asyncio.fixture(loop_scope="session")
    async def filter_test_documents(self, firestore_client):
        """Create documents with different domain_id and status for filter tests."""
        owner_id = f"filter-test-owner-{uuid.uuid4()}"
        doc_ids = []

        # Create documents with varying domain_id and status
        test_data = [
            {"domain_id": "domain-a", "status": "completed", "name": "Doc 1"},
            {"domain_id": "domain-a", "status": "pending", "name": "Doc 2"},
            {"domain_id": "domain-b", "status": "completed", "name": "Doc 3"},
            {"domain_id": "domain-b", "status": "pending", "name": "Doc 4"},
            {"domain_id": "domain-a", "status": "completed", "name": "Doc 5"},
        ]

        for i, data in enumerate(test_data):
            doc_id = f"filter-test-{uuid.uuid4()}"
            doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
            await doc_ref.set(
                {
                    **data,
                    "owner_id": owner_id,
                    "index": i,
                }
            )
            doc_ids.append(doc_id)

        yield {"owner_id": owner_id, "doc_ids": doc_ids}

        # Cleanup
        for doc_id in doc_ids:
            await firestore_client.collection(TEST_COLLECTION).document(doc_id).delete()

    async def test_filter_by_single_field(self, filter_test_documents):
        """Filters by a single field return only matching documents."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=filter_test_documents["owner_id"],
            filters={"domain_id": "domain-a"},
        )

        assert total == 3
        assert len(docs) == 3
        for doc in docs:
            assert doc.to_dict()["domain_id"] == "domain-a"

    async def test_filter_by_multiple_fields(self, filter_test_documents):
        """Filters by multiple fields return only documents matching all filters."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=filter_test_documents["owner_id"],
            filters={"domain_id": "domain-a", "status": "completed"},
        )

        assert total == 2
        assert len(docs) == 2
        for doc in docs:
            data = doc.to_dict()
            assert data["domain_id"] == "domain-a"
            assert data["status"] == "completed"

    async def test_filter_returns_empty_when_no_match(self, filter_test_documents):
        """Filters return empty results when no documents match."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=filter_test_documents["owner_id"],
            filters={"domain_id": "nonexistent-domain"},
        )

        assert total == 0
        assert len(docs) == 0

    async def test_filter_with_pagination(self, filter_test_documents):
        """Filters work correctly with pagination."""
        # Get first page of domain-a documents (size 2)
        docs_page_0, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=filter_test_documents["owner_id"],
            filters={"domain_id": "domain-a"},
            page=0,
            size=2,
        )

        # Get second page
        docs_page_1, total_again = await list_documents_async(
            TEST_COLLECTION,
            owner_id=filter_test_documents["owner_id"],
            filters={"domain_id": "domain-a"},
            page=1,
            size=2,
        )

        assert total == 3
        assert total_again == 3
        assert len(docs_page_0) == 2
        assert len(docs_page_1) == 1

    async def test_filter_with_sorting(self, filter_test_documents):
        """Filters work correctly with sorting."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=filter_test_documents["owner_id"],
            filters={"domain_id": "domain-a"},
            sort_by="name",
            sort_order="ascending",
        )

        assert total == 3
        names = [doc.to_dict()["name"] for doc in docs]
        assert names == sorted(names)

    async def test_filter_total_count_accurate(self, filter_test_documents):
        """Total count reflects filtered count, not total documents."""
        # Total documents for owner is 5
        all_docs, total_all = await list_documents_async(
            TEST_COLLECTION,
            owner_id=filter_test_documents["owner_id"],
        )
        assert total_all == 5

        # Filtered documents should have accurate count
        filtered_docs, total_filtered = await list_documents_async(
            TEST_COLLECTION,
            owner_id=filter_test_documents["owner_id"],
            filters={"status": "completed"},
        )
        assert total_filtered == 3
        assert len(filtered_docs) == 3

    async def test_empty_filters_returns_all(self, filter_test_documents):
        """Empty filters dict returns all documents for owner."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=filter_test_documents["owner_id"],
            filters={},
        )

        assert total == 5
        assert len(docs) == 5

    async def test_filter_still_respects_owner(
        self, filter_test_documents, firestore_client
    ):
        """Filters still respect owner_id (don't leak other owner's documents)."""
        # Create a document with same domain_id but different owner
        other_doc_id = f"other-owner-{uuid.uuid4()}"
        doc_ref = firestore_client.collection(TEST_COLLECTION).document(other_doc_id)
        await doc_ref.set(
            {
                "domain_id": "domain-a",
                "status": "completed",
                "owner_id": "different-owner",
                "name": "Other Owner Doc",
            }
        )

        try:
            docs, total = await list_documents_async(
                TEST_COLLECTION,
                owner_id=filter_test_documents["owner_id"],
                filters={"domain_id": "domain-a"},
            )

            # Should only see the original owner's documents
            assert total == 3
            doc_ids = [doc.id for doc in docs]
            assert other_doc_id not in doc_ids
        finally:
            await doc_ref.delete()

    # -------------------------------------------------------------------------
    # Array Contains Filter Tests
    # -------------------------------------------------------------------------

    @pytest_asyncio.fixture(loop_scope="session")
    async def array_filter_test_documents(self, firestore_client):
        """Create documents with tags arrays for array-contains filter tests."""
        owner_id = f"array-filter-test-owner-{uuid.uuid4()}"
        doc_ids = []

        # Create documents with varying tags
        test_data = [
            {"name": "Doc 1", "tags": ["surface-fuel", "baseline"]},
            {"name": "Doc 2", "tags": ["surface-fuel", "treatment"]},
            {"name": "Doc 3", "tags": ["topography"]},
            {"name": "Doc 4", "tags": ["canopy-fuel", "baseline"]},
            {"name": "Doc 5", "tags": []},  # Empty tags
        ]

        for data in test_data:
            doc_id = f"array-filter-test-{uuid.uuid4()}"
            doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
            await doc_ref.set({**data, "owner_id": owner_id})
            doc_ids.append(doc_id)

        yield {"owner_id": owner_id, "doc_ids": doc_ids}

        # Cleanup
        for doc_id in doc_ids:
            await firestore_client.collection(TEST_COLLECTION).document(doc_id).delete()

    async def test_array_contains_filter_single_tag(self, array_filter_test_documents):
        """Array-contains filter returns documents containing the specified tag."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=array_filter_test_documents["owner_id"],
            array_contains_filters={"tags": "surface-fuel"},
        )

        assert total == 2
        assert len(docs) == 2
        for doc in docs:
            assert "surface-fuel" in doc.to_dict()["tags"]

    async def test_array_contains_filter_no_matches(self, array_filter_test_documents):
        """Array-contains filter returns empty when no documents match."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=array_filter_test_documents["owner_id"],
            array_contains_filters={"tags": "nonexistent-tag"},
        )

        assert total == 0
        assert len(docs) == 0

    async def test_array_contains_with_equality_filter(
        self, array_filter_test_documents
    ):
        """Can combine array-contains with equality filters."""
        # First, let's update one of the surface-fuel docs to have a specific domain
        # Actually, let's just test combining with another filter that exists

        # Test that array_contains works with the owner filter (which is always applied)
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=array_filter_test_documents["owner_id"],
            array_contains_filters={"tags": "baseline"},
        )

        assert total == 2
        for doc in docs:
            data = doc.to_dict()
            assert "baseline" in data["tags"]
            assert data["owner_id"] == array_filter_test_documents["owner_id"]

    async def test_array_contains_with_pagination(self, array_filter_test_documents):
        """Array-contains filter works with pagination."""
        docs_page_0, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=array_filter_test_documents["owner_id"],
            array_contains_filters={"tags": "surface-fuel"},
            page=0,
            size=1,
        )

        docs_page_1, _ = await list_documents_async(
            TEST_COLLECTION,
            owner_id=array_filter_test_documents["owner_id"],
            array_contains_filters={"tags": "surface-fuel"},
            page=1,
            size=1,
        )

        assert total == 2
        assert len(docs_page_0) == 1
        assert len(docs_page_1) == 1
        # Different documents on each page
        assert docs_page_0[0].id != docs_page_1[0].id

    async def test_empty_array_contains_filters_returns_all(
        self, array_filter_test_documents
    ):
        """Empty array_contains_filters dict returns all documents for owner."""
        docs, total = await list_documents_async(
            TEST_COLLECTION,
            owner_id=array_filter_test_documents["owner_id"],
            array_contains_filters={},
        )

        assert total == 5
        assert len(docs) == 5


class TestUpdateDocumentAsync:
    """Integration tests for update_document_async."""

    @pytest_asyncio.fixture(loop_scope="session")
    async def document_for_update(self, firestore_client):
        """Create a test document for update tests."""
        doc_id = f"test-update-{uuid.uuid4()}"
        doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
        original_data = {
            "name": "Original Name",
            "description": "Original Description",
            "status": "active",
            "owner_id": "update-test-owner",
            "tags": ["original"],
        }
        await doc_ref.set(original_data)
        yield {"doc_id": doc_id, "doc_ref": doc_ref, "original_data": original_data}
        await doc_ref.delete()

    async def test_updates_single_field(self, document_for_update):
        """Updates only the specified field, leaving others unchanged."""
        doc_id = document_for_update["doc_id"]
        doc_ref = document_for_update["doc_ref"]

        await update_document_async(TEST_COLLECTION, doc_id, {"name": "Updated Name"})

        snapshot = await doc_ref.get()
        data = snapshot.to_dict()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Original Description"
        assert data["status"] == "active"

    async def test_updates_multiple_fields(self, document_for_update):
        """Updates multiple fields in a single call."""
        doc_id = document_for_update["doc_id"]
        doc_ref = document_for_update["doc_ref"]

        await update_document_async(
            TEST_COLLECTION,
            doc_id,
            {"description": "New Description", "tags": ["new", "updated"]},
        )

        snapshot = await doc_ref.get()
        data = snapshot.to_dict()
        assert data["description"] == "New Description"
        assert data["tags"] == ["new", "updated"]

    async def test_returns_document_reference(self, document_for_update):
        """Returns a valid document reference."""
        doc_id = document_for_update["doc_id"]

        ref = await update_document_async(
            TEST_COLLECTION, doc_id, {"status": "updated"}
        )

        assert ref.id == doc_id
        assert ref.parent.id == TEST_COLLECTION

    async def test_nonexistent_document_raises_not_found(self):
        """Raises NotFound error when document doesn't exist."""
        from google.api_core.exceptions import NotFound

        with pytest.raises(NotFound):
            await update_document_async(
                TEST_COLLECTION,
                "nonexistent-doc-12345",
                {"name": "Should Fail"},
            )

    async def test_can_add_new_fields(self, document_for_update):
        """Can add fields that didn't exist before."""
        doc_id = document_for_update["doc_id"]
        doc_ref = document_for_update["doc_ref"]

        await update_document_async(TEST_COLLECTION, doc_id, {"new_field": "new_value"})

        snapshot = await doc_ref.get()
        data = snapshot.to_dict()
        assert data["new_field"] == "new_value"

    async def test_round_trip_with_get_document_async(self, document_for_update):
        """Document updated with update_document_async can be retrieved with get_document_async."""
        doc_id = document_for_update["doc_id"]

        await update_document_async(TEST_COLLECTION, doc_id, {"name": "Roundtrip Test"})

        _, snapshot = await get_document_async(
            TEST_COLLECTION, doc_id, owner_id="update-test-owner"
        )

        data = snapshot.to_dict()
        assert data["name"] == "Roundtrip Test"
        assert data["owner_id"] == "update-test-owner"


class TestDeleteDocumentAsync:
    """Integration tests for delete_document_async."""

    @pytest_asyncio.fixture(loop_scope="session")
    async def document_for_delete(self, firestore_client):
        """Create a test document for delete tests."""
        doc_id = f"test-delete-{uuid.uuid4()}"
        doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
        test_data = {
            "name": "Document to Delete",
            "owner_id": "delete-test-owner",
        }
        await doc_ref.set(test_data)
        yield {"doc_id": doc_id, "doc_ref": doc_ref}
        # Cleanup if not already deleted
        doc = await doc_ref.get()
        if doc.exists:
            await doc_ref.delete()

    async def test_deletes_existing_document(self, firestore_client):
        """Successfully deletes an existing document."""
        doc_id = f"test-delete-existing-{uuid.uuid4()}"
        doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
        await doc_ref.set({"name": "To Be Deleted"})

        # Verify document exists
        snapshot = await doc_ref.get()
        assert snapshot.exists

        # Delete it
        await delete_document_async(TEST_COLLECTION, doc_id)

        # Verify document no longer exists
        snapshot = await doc_ref.get()
        assert not snapshot.exists

    async def test_deletes_nonexistent_document_silently(self):
        """Deleting a non-existent document does not raise an error."""
        # This should not raise any exception
        await delete_document_async(TEST_COLLECTION, "nonexistent-doc-xyz")

    async def test_returns_none(self, document_for_delete):
        """Delete returns None."""
        doc_id = document_for_delete["doc_id"]

        result = await delete_document_async(TEST_COLLECTION, doc_id)

        assert result is None

    async def test_document_not_retrievable_after_delete(self, firestore_client):
        """Document cannot be retrieved with get_document_async after deletion."""
        doc_id = f"test-delete-retrieve-{uuid.uuid4()}"
        doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
        await doc_ref.set({"name": "Will Be Deleted", "owner_id": MOCK_OWNER_ID})

        # Verify we can retrieve it
        _, snapshot = await get_document_async(
            TEST_COLLECTION, doc_id, owner_id=MOCK_OWNER_ID
        )
        assert snapshot.exists

        # Delete it
        await delete_document_async(TEST_COLLECTION, doc_id)

        # Verify retrieval raises 404
        with pytest.raises(HTTPException) as exc:
            await get_document_async(TEST_COLLECTION, doc_id)
        assert exc.value.status_code == 404

    async def test_delete_is_idempotent(self, firestore_client):
        """Deleting the same document twice does not raise an error."""
        doc_id = f"test-delete-idempotent-{uuid.uuid4()}"
        doc_ref = firestore_client.collection(TEST_COLLECTION).document(doc_id)
        await doc_ref.set({"name": "Delete Twice"})

        # Delete twice
        await delete_document_async(TEST_COLLECTION, doc_id)
        await delete_document_async(TEST_COLLECTION, doc_id)

        # Should still not exist
        snapshot = await doc_ref.get()
        assert not snapshot.exists
