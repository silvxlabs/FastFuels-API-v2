"""Every route is served under /v2, including the docs (#337). No server needed.

During the migration window the old root paths also answer, outside OpenAPI.
"""

import pytest
from api.app import app
from fastapi.routing import APIRoute
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
    assert all(p == "/v2" or p.startswith("/v2/") for p in schema["paths"])


def test_v2_routes_require_auth(api_client):
    assert api_client.get("/v2/domains").status_code == 401


def test_root_paths_still_served_during_migration(api_client):
    assert api_client.get("/domains").status_code == 401


def test_index_is_served_at_v2_without_auth(api_client):
    response = api_client.get("/v2")
    assert response.status_code == 200
    assert response.json()["documentation"]["swagger"]["url"] == "/v2/docs"


def test_swagger_oauth2_redirect_is_under_v2(api_client):
    assert api_client.get("/v2/docs/oauth2-redirect").status_code == 200


def test_root_docs_and_index_are_gone(api_client):
    for path in ("/", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"):
        assert api_client.get(path).status_code == 404


def test_root_routes_mirror_v2_routes():
    routes = [r for r in app.routes if isinstance(r, APIRoute)]
    v2 = {r.path.removeprefix("/v2"): r for r in routes if r.path.startswith("/v2/")}
    root = {r.path: r for r in routes if not r.path.startswith("/v2")}
    assert v2.keys() == root.keys()
    for path, route in v2.items():
        twin = root[path]
        assert twin.endpoint is route.endpoint
        assert twin.methods == route.methods
        assert twin.status_code == route.status_code
        assert twin.response_model == route.response_model
        assert [d.call for d in twin.dependant.dependencies] == [
            d.call for d in route.dependant.dependencies
        ]
        assert not twin.include_in_schema


def test_operation_ids_are_unique(api_client):
    schema = api_client.get("/v2/openapi.json").json()
    ids = [
        op["operationId"] for item in schema["paths"].values() for op in item.values()
    ]
    assert len(ids) == len(set(ids))


def test_named_routes_resolve_under_v2():
    path = app.url_path_for("create_landfire_fbfm40", domain_id="abc")
    assert path == "/v2/domains/abc/grids/fbfm40/landfire"
