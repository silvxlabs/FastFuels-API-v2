"""
Unit tests for api/v2/resources/applications/schema.py
"""

from datetime import UTC, datetime

import pytest
from api.resources.applications.schema import (
    Application,
    CreateApplicationRequest,
    ListApplicationsResponse,
    UpdateApplicationRequest,
)


class TestCreateApplicationRequest:
    def test_name_required(self):
        with pytest.raises(Exception):
            CreateApplicationRequest()

    def test_name_only(self):
        req = CreateApplicationRequest(name="My App")
        assert req.name == "My App"
        assert req.description is None

    def test_with_description(self):
        req = CreateApplicationRequest(name="My App", description="A description")
        assert req.description == "A description"

    def test_rejects_quota_config(self):
        """Admin-only fields cannot be set via the create request."""
        with pytest.raises(Exception):
            CreateApplicationRequest(name="My App", tier="partner")


class TestUpdateApplicationRequest:
    def test_all_fields_optional(self):
        req = UpdateApplicationRequest()
        assert req.name is None
        assert req.description is None

    def test_name_only(self):
        req = UpdateApplicationRequest(name="New Name")
        assert req.name == "New Name"
        assert req.description is None

    def test_description_only(self):
        req = UpdateApplicationRequest(description="New Desc")
        assert req.name is None
        assert req.description == "New Desc"

    def test_model_dump_exclude_none(self):
        req = UpdateApplicationRequest(name="New Name")
        dumped = req.model_dump(exclude_none=True)
        assert dumped == {"name": "New Name"}
        assert "description" not in dumped

    def test_model_dump_exclude_none_all_fields(self):
        req = UpdateApplicationRequest(name="Name", description="Desc")
        dumped = req.model_dump(exclude_none=True)
        assert dumped == {"name": "Name", "description": "Desc"}

    def test_model_dump_exclude_none_empty(self):
        req = UpdateApplicationRequest()
        dumped = req.model_dump(exclude_none=True)
        assert dumped == {}


class TestApplication:
    def test_required_fields(self):
        app = Application(id="abc", owner_id="owner", name="My App")
        assert app.id == "abc"
        assert app.owner_id == "owner"
        assert app.name == "My App"

    def test_description_default(self):
        app = Application(id="abc", owner_id="owner", name="My App")
        assert app.description is None

    def test_created_on_auto_factory(self):
        before = datetime.now(UTC)
        app = Application(id="abc", owner_id="owner", name="My App")
        after = datetime.now(UTC)
        assert before <= app.created_on <= after

    def test_modified_on_auto_factory(self):
        before = datetime.now(UTC)
        app = Application(id="abc", owner_id="owner", name="My App")
        after = datetime.now(UTC)
        assert before <= app.modified_on <= after

    def test_with_all_fields(self):
        now = datetime.now(UTC)
        app = Application(
            id="abc",
            owner_id="owner",
            name="My App",
            description="Full app",
            created_on=now,
            modified_on=now,
        )
        assert app.description == "Full app"
        assert app.created_on == now
        assert app.modified_on == now


class TestListApplicationsResponse:
    def test_pagination_fields_required(self):
        resp = ListApplicationsResponse(
            applications=[],
            current_page=0,
            page_size=100,
            total_items=0,
        )
        assert resp.current_page == 0
        assert resp.page_size == 100
        assert resp.total_items == 0

    def test_empty_list_valid(self):
        resp = ListApplicationsResponse(
            applications=[],
            current_page=0,
            page_size=100,
            total_items=0,
        )
        assert resp.applications == []

    def test_with_applications(self):
        app = Application(id="abc", owner_id="owner", name="My App")
        resp = ListApplicationsResponse(
            applications=[app],
            current_page=0,
            page_size=100,
            total_items=1,
        )
        assert len(resp.applications) == 1
        assert resp.applications[0].id == "abc"
