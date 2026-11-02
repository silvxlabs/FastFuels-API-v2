"""Every route is served under /v2, including the docs (#337). No server needed.

During the migration window the old root paths also answer, outside OpenAPI.
"""

import pytest
from api.app import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


def test_swagger_loads_v2_openapi(api_client):
    response = api_client.get("/v2/docs")
    assert response.status_code == 200
    assert "'/v2/openapi.json'" in response.text


def test_redoc_loads_v2_openapi(api_client):
    response = api_client.get("/v2/redoc")
    assert response.status_code == 200
    assert '"/v2/openapi.json"' in response.text


def test_openapi_paths_are_prefixed(api_client):
    schema = api_client.get("/v2/openapi.json").json()
    assert "servers" not in schema
    assert "/v2/domains" in schema["paths"]
    assert all(path.startswith("/v2/") for path in schema["paths"])


def test_v2_routes_require_auth(api_client):
    assert api_client.get("/v2/domains").status_code == 401


def test_root_paths_still_served_during_migration(api_client):
    assert api_client.get("/domains").status_code == 401


def test_root_docs_and_index_are_gone(api_client):
    for path in ("/", "/docs", "/redoc", "/openapi.json"):
        assert api_client.get(path).status_code == 404


def test_named_routes_resolve_under_v2():
    path = app.url_path_for("create_landfire_fbfm40", domain_id="abc")
    assert path == "/v2/domains/abc/grids/fbfm40/landfire"
