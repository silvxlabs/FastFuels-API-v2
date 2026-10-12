"""Every route is served under /v2, including the docs (#337). No server needed."""

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


def test_routes_require_v2_prefix(api_client):
    assert api_client.get("/v2/domains").status_code == 401
    for path in ("/", "/domains", "/docs", "/openapi.json"):
        assert api_client.get(path).status_code == 404
