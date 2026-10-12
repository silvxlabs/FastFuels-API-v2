"""
api/dependencies.py

Shared FastAPI dependencies.
"""

from typing import Annotated

from fastapi import Depends, Request
from ring import lru

from api.db.documents import get_document_async
from lib.config import DOMAINS_COLLECTION

# Every route is served under this prefix; clients use it as part of the base URL.
API_PREFIX = "/v2"


def url_path_from_base(request: Request, name: str, **path_params: str) -> str:
    """Path of a named route relative to the API base URL (without API_PREFIX)."""
    return str(request.app.url_path_for(name, **path_params)).removeprefix(API_PREFIX)


@lru(force_asyncio=True, expire=300)
async def _get_domain(domain_id: str, owner_id: str) -> dict:
    """Cached domain lookup. Returns domain data dict."""
    _, snapshot = await get_document_async(DOMAINS_COLLECTION, domain_id, owner_id)
    return snapshot.to_dict()


async def get_verified_domain(request: Request, domain_id: str) -> dict:
    """FastAPI dependency that validates domain ownership.

    Resolves domain_id from the URL path, verifies the domain exists and
    is owned by the authenticated user, and returns the domain data dict.
    Results are cached with a 5-minute TTL.
    """
    return await _get_domain(domain_id, request.state.id)


async def invalidate_domain_cache(domain_id: str, owner_id: str):
    """Remove a domain from the cache. Call on update or delete."""
    await _get_domain.delete(domain_id, owner_id)


VerifiedDomain = Annotated[dict, Depends(get_verified_domain)]
