"""
Unit tests for api/v2/auth.py

Uses unittest.mock.patch to mock Firestore — no real database needed.
"""

import asyncio
import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from api.auth import (
    _api_key_auth,
    _lookup_by_doc_id,
    authenticate_user,
    hash_api_key,
    invalidate_key_cache,
    resolve_api_key,
)
from api.resources.keys.schema import Access, Key, Scope
from ring import lru


class TestHashApiKey:
    def test_sha256_correctness(self):
        raw = "my-secret-key"
        expected = hashlib.sha256(raw.encode()).hexdigest()
        assert hash_api_key(raw) == expected

    def test_determinism(self):
        raw = "deterministic"
        assert hash_api_key(raw) == hash_api_key(raw)

    def test_output_length(self):
        result = hash_api_key("any-key")
        assert len(result) == 64  # SHA-256 hex digest


class TestResolveApiKey:
    def _make_key(self, **overrides) -> Key:
        defaults = dict(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="Test Key",
            expires_on=datetime.now(UTC) + timedelta(days=30),
        )
        defaults.update(overrides)
        return Key(**defaults)

    @pytest.mark.asyncio
    async def test_hash_lookup_success(self):
        key = self._make_key()
        with patch("api.auth._lookup_by_doc_id", new_callable=AsyncMock) as mock_lookup:
            mock_lookup.return_value = key
            result = await resolve_api_key("raw-secret")
            assert result == key
            # Should be called with the hash of the raw secret
            mock_lookup.assert_called_once_with(hash_api_key("raw-secret"))

    @pytest.mark.asyncio
    async def test_miss_raises_401(self):
        with patch("api.auth._lookup_by_doc_id", new_callable=AsyncMock) as mock_lookup:
            mock_lookup.return_value = None
            with pytest.raises(Exception) as exc_info:
                await resolve_api_key("unknown-secret")
            assert exc_info.value.status_code == 401
            mock_lookup.assert_called_once_with(hash_api_key("unknown-secret"))


class TestApiKeyAuth:
    def _make_key(self, **overrides) -> Key:
        defaults = dict(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="Test Key",
            scopes=[Scope.READ, Scope.WRITE],
            expires_on=datetime.now(UTC) + timedelta(days=30),
        )
        defaults.update(overrides)
        return Key(**defaults)

    def _make_request(self, method="GET") -> MagicMock:
        request = MagicMock()
        request.method = method
        request.state = MagicMock()
        return request

    @pytest.mark.asyncio
    async def test_sets_request_state(self):
        key = self._make_key()
        request = self._make_request()
        with patch(
            "api.auth.resolve_api_key", new_callable=AsyncMock, return_value=key
        ):
            result = await _api_key_auth(request, "secret")
            assert result.state.id == "owner"
            assert result.state.access == Access.PERSONAL

    @pytest.mark.asyncio
    async def test_expired_key_raises_401(self):
        key = self._make_key(expires_on=datetime.now(UTC) - timedelta(seconds=1))
        request = self._make_request()
        with patch(
            "api.auth.resolve_api_key", new_callable=AsyncMock, return_value=key
        ):
            with pytest.raises(Exception) as exc_info:
                await _api_key_auth(request, "secret")
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_insufficient_permissions_raises_403(self):
        key = self._make_key(scopes=[Scope.READ])
        request = self._make_request(method="POST")
        with patch(
            "api.auth.resolve_api_key", new_callable=AsyncMock, return_value=key
        ):
            with pytest.raises(Exception) as exc_info:
                await _api_key_auth(request, "secret")
            assert exc_info.value.status_code == 403


class TestTokenAuth:
    def test_valid_token_sets_state(self):
        from api.auth import _token_auth

        request = MagicMock()
        request.state = MagicMock()
        with patch("api.auth.verify_id_token", return_value={"uid": "firebase-user"}):
            result = _token_auth(request, "valid-token")
            assert result.state.id == "firebase-user"
            assert result.state.access == Access.PERSONAL

    def test_invalid_token_raises_401(self):
        from api.auth import _token_auth

        request = MagicMock()
        request.state = MagicMock()
        with patch("api.auth.verify_id_token", side_effect=Exception("bad token")):
            with pytest.raises(Exception) as exc_info:
                _token_auth(request, "bad-token")
            assert exc_info.value.status_code == 401


class TestAuthenticateUser:
    def _make_request(self) -> MagicMock:
        request = MagicMock()
        request.state = MagicMock()
        return request

    @pytest.mark.asyncio
    async def test_api_key_priority_over_bearer(self):
        key = Key(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="Test Key",
            scopes=[Scope.READ, Scope.WRITE],
            expires_on=datetime.now(UTC) + timedelta(days=30),
        )
        request = self._make_request()
        request.method = "GET"
        bearer = MagicMock()
        bearer.credentials = "token"

        with patch(
            "api.auth.resolve_api_key", new_callable=AsyncMock, return_value=key
        ):
            result = await authenticate_user(request, api_key="some-key", bearer=bearer)
            assert result.state.id == "owner"

    @pytest.mark.asyncio
    async def test_bearer_fallback(self):
        request = self._make_request()
        bearer = MagicMock()
        bearer.credentials = "valid-token"

        with patch("api.auth.verify_id_token", return_value={"uid": "fb-user"}):
            result = await authenticate_user(request, api_key=None, bearer=bearer)
            assert result.state.id == "fb-user"

    @pytest.mark.asyncio
    async def test_neither_raises_401(self):
        request = self._make_request()
        with pytest.raises(Exception) as exc_info:
            await authenticate_user(request, api_key=None, bearer=None)
        assert exc_info.value.status_code == 401


# Short-TTL function for testing ring's expire behavior. Same configuration
# as _lookup_by_doc_id (force_asyncio, lru) but with a 1-second TTL.
_call_count = 0


@lru(force_asyncio=True, expire=1)
async def _cached_short_ttl(key: str) -> str:
    global _call_count
    _call_count += 1
    return f"{key}-{_call_count}"


class TestLookupCache:
    """Tests for the ring LRU cache on _lookup_by_doc_id."""

    def _make_key_dict(self, doc_id: str) -> dict:
        now = datetime.now(UTC)
        return dict(
            id=doc_id,
            owner_id="owner",
            creator_id="owner",
            name="Cache Test Key",
            scopes=["read"],
            access="personal",
            valid_days=30,
            created_on=now,
            expires_on=now + timedelta(days=30),
        )

    def _mock_firestore(self, mock_get):
        """Build a mock firestore_client chain: collection -> document -> get."""
        mock_document = MagicMock()
        mock_document.get = mock_get
        mock_collection = MagicMock()
        mock_collection.document.return_value = mock_document
        mock_client = MagicMock()
        mock_client.collection.return_value = mock_collection
        return mock_client

    def _make_doc(self, doc_id: str, exists: bool = True):
        doc = MagicMock()
        doc.exists = exists
        if exists:
            doc.to_dict.return_value = self._make_key_dict(doc_id)
        return doc

    @pytest.fixture(autouse=True)
    def doc_id(self):
        """Unique doc ID per test to avoid cache collisions."""
        self._doc_id = f"cache-{uuid.uuid4().hex[:12]}"
        return self._doc_id

    @pytest_asyncio.fixture(autouse=True)
    async def cleanup_cache(self, doc_id):
        """Evict cache entry after each test."""
        yield
        try:
            await _lookup_by_doc_id.delete(doc_id)
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_repeated_lookup_uses_cache(self, doc_id):
        """Same doc_id looked up twice — Firestore called only once."""
        doc = self._make_doc(doc_id)
        mock_get = AsyncMock(return_value=doc)

        with patch("api.auth.firestore_client", self._mock_firestore(mock_get)):
            r1 = await _lookup_by_doc_id(doc_id)
            r2 = await _lookup_by_doc_id(doc_id)

        assert r1.id == doc_id
        assert r2.id == doc_id
        mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_forces_refetch(self, doc_id):
        """After invalidate_key_cache, the next lookup hits Firestore again."""
        doc = self._make_doc(doc_id)
        mock_get = AsyncMock(return_value=doc)

        with patch("api.auth.firestore_client", self._mock_firestore(mock_get)):
            await _lookup_by_doc_id(doc_id)
            assert mock_get.call_count == 1

            await invalidate_key_cache(doc_id)

            await _lookup_by_doc_id(doc_id)
            assert mock_get.call_count == 2

    @pytest.mark.asyncio
    async def test_deleted_key_gone_after_invalidation(self, doc_id):
        """Key deleted from Firestore + cache invalidated -> returns None."""
        exists_doc = self._make_doc(doc_id, exists=True)
        gone_doc = self._make_doc(doc_id, exists=False)
        mock_get = AsyncMock(side_effect=[exists_doc, gone_doc])

        with patch("api.auth.firestore_client", self._mock_firestore(mock_get)):
            key = await _lookup_by_doc_id(doc_id)
            assert key is not None

            await invalidate_key_cache(doc_id)

            key = await _lookup_by_doc_id(doc_id)
            assert key is None

    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        """Ring LRU expire evicts entries after TTL elapses.

        Uses a test-only function with expire=1 (same config as
        _lookup_by_doc_id but 1s instead of 300s).
        """
        global _call_count
        _call_count = 0

        key = f"ttl-{uuid.uuid4().hex[:8]}"

        try:
            r1 = await _cached_short_ttl(key)
            r2 = await _cached_short_ttl(key)
            assert r1 == r2  # Cached — same return value
            assert _call_count == 1  # Underlying function called once

            await asyncio.sleep(1.1)

            r3 = await _cached_short_ttl(key)
            assert _call_count == 2  # TTL expired — function called again
            assert r3 != r1  # New return value (call count incremented)
        finally:
            try:
                await _cached_short_ttl.delete(key)
            except Exception:
                pass
