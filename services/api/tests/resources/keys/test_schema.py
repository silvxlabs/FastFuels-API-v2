"""
Unit tests for api/v2/resources/keys/schema.py
"""

from datetime import UTC, datetime, timedelta

import pytest
from api.resources.keys.schema import (
    Access,
    CreateKeyRequest,
    CreateKeyResponse,
    Key,
    ListKeysResponse,
    Scope,
)


class TestScope:
    def test_read_value(self):
        assert Scope.READ == "read"

    def test_write_value(self):
        assert Scope.WRITE == "write"

    def test_count(self):
        assert len(Scope) == 2

    def test_string_construction(self):
        assert Scope("read") == Scope.READ
        assert Scope("write") == Scope.WRITE


class TestAccess:
    def test_personal_value(self):
        assert Access.PERSONAL == "personal"

    def test_application_value(self):
        assert Access.APPLICATION == "application"

    def test_count(self):
        assert len(Access) == 2

    def test_string_construction(self):
        assert Access("personal") == Access.PERSONAL
        assert Access("application") == Access.APPLICATION


class TestCreateKeyRequest:
    def test_defaults(self):
        req = CreateKeyRequest(name="Test Key")
        assert req.scopes == [Scope.READ]
        assert req.access == Access.PERSONAL
        assert req.valid_days == 30
        assert req.application_id is None
        assert req.description is None

    def test_name_required(self):
        with pytest.raises(Exception):
            CreateKeyRequest()

    def test_application_id_required_for_application_access(self):
        with pytest.raises(Exception):
            CreateKeyRequest(name="App Key", access=Access.APPLICATION)

    def test_application_access_with_id(self):
        req = CreateKeyRequest(
            name="App Key",
            access=Access.APPLICATION,
            application_id="abc123",
        )
        assert req.application_id == "abc123"

    def test_custom_scopes(self):
        req = CreateKeyRequest(
            name="RW Key",
            scopes=[Scope.READ, Scope.WRITE],
        )
        assert req.scopes == [Scope.READ, Scope.WRITE]

    def test_custom_valid_days(self):
        req = CreateKeyRequest(name="Short Key", valid_days=7)
        assert req.valid_days == 7

    def test_valid_days_min(self):
        with pytest.raises(Exception):
            CreateKeyRequest(name="Bad Key", valid_days=0)


class TestKey:
    def test_required_fields(self):
        key = Key(id="abc", owner_id="owner", creator_id="creator", name="My Key")
        assert key.id == "abc"
        assert key.owner_id == "owner"
        assert key.creator_id == "creator"
        assert key.name == "My Key"

    def test_creator_id_required(self):
        with pytest.raises(Exception):
            Key(id="abc", owner_id="owner", name="My Key")

    def test_creator_id_differs_from_owner_id(self):
        key = Key(id="abc", owner_id="app-123", creator_id="user-456", name="App Key")
        assert key.owner_id == "app-123"
        assert key.creator_id == "user-456"

    def test_defaults(self):
        key = Key(id="abc", owner_id="owner", creator_id="owner", name="My Key")
        assert key.scopes == [Scope.READ]
        assert key.access == Access.PERSONAL
        assert key.valid_days == 30
        assert key.application_id is None
        assert key.description is None

    def test_created_on_auto_factory(self):
        before = datetime.now(UTC)
        key = Key(id="abc", owner_id="owner", creator_id="owner", name="My Key")
        after = datetime.now(UTC)
        assert before <= key.created_on <= after


class TestKeyIsExpired:
    def test_future_expiration_not_expired(self):
        key = Key(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="My Key",
            expires_on=datetime.now(UTC) + timedelta(days=30),
        )
        assert key.is_expired() is False

    def test_past_expiration_is_expired(self):
        key = Key(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="My Key",
            expires_on=datetime.now(UTC) - timedelta(seconds=1),
        )
        assert key.is_expired() is True


class TestKeyHasPermission:
    def test_read_scope_allows_get(self):
        key = Key(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="My Key",
            scopes=[Scope.READ],
        )
        assert key.has_permission("GET") is True

    def test_read_scope_denies_post(self):
        key = Key(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="My Key",
            scopes=[Scope.READ],
        )
        assert key.has_permission("POST") is False

    def test_read_scope_denies_delete(self):
        key = Key(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="My Key",
            scopes=[Scope.READ],
        )
        assert key.has_permission("DELETE") is False

    def test_write_scope_allows_get(self):
        key = Key(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="My Key",
            scopes=[Scope.WRITE],
        )
        assert key.has_permission("GET") is True

    def test_write_scope_allows_post(self):
        key = Key(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="My Key",
            scopes=[Scope.WRITE],
        )
        assert key.has_permission("POST") is True

    def test_write_scope_allows_patch(self):
        key = Key(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="My Key",
            scopes=[Scope.WRITE],
        )
        assert key.has_permission("PATCH") is True

    def test_write_scope_allows_delete(self):
        key = Key(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="My Key",
            scopes=[Scope.WRITE],
        )
        assert key.has_permission("DELETE") is True


class TestCreateKeyResponse:
    def test_inherits_key_fields(self):
        resp = CreateKeyResponse(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="My Key",
            secret="secret123",
        )
        assert resp.id == "abc"
        assert resp.owner_id == "owner"
        assert resp.name == "My Key"

    def test_secret_required(self):
        with pytest.raises(Exception):
            CreateKeyResponse(
                id="abc", owner_id="owner", creator_id="owner", name="My Key"
            )

    def test_secret_value(self):
        resp = CreateKeyResponse(
            id="abc",
            owner_id="owner",
            creator_id="owner",
            name="My Key",
            secret="mysecret",
        )
        assert resp.secret == "mysecret"


class TestListKeysResponse:
    def test_pagination_fields_required(self):
        resp = ListKeysResponse(
            keys=[],
            current_page=0,
            page_size=100,
            total_items=0,
        )
        assert resp.current_page == 0
        assert resp.page_size == 100
        assert resp.total_items == 0

    def test_empty_list_valid(self):
        resp = ListKeysResponse(
            keys=[],
            current_page=0,
            page_size=100,
            total_items=0,
        )
        assert resp.keys == []

    def test_with_keys(self):
        key = Key(id="abc", owner_id="owner", creator_id="owner", name="My Key")
        resp = ListKeysResponse(
            keys=[key],
            current_page=0,
            page_size=100,
            total_items=1,
        )
        assert len(resp.keys) == 1
        assert resp.keys[0].id == "abc"
