"""
api/resources/users/router.py

Read API for the quota system. "me" is the authenticated owner (user or
application); both endpoints work identically for either credential.
"""

from fastapi import APIRouter, Request, status

from api.quota import get_usage, resolve_owner_config
from api.resources.keys.schema import Access
from api.resources.users.schema import Usage, UserMeResponse

router = APIRouter()


# Declared before any future /{user_id} route so "me" is never captured as an id.
@router.get(
    "/me",
    response_model=UserMeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get the authenticated owner",
)
async def get_me(request: Request) -> UserMeResponse:
    """Return the authenticated owner's identity, tier, and resolved quotas."""
    cfg = await resolve_owner_config(request.state.id, request.state.access)
    kind = "user" if request.state.access == Access.PERSONAL else "application"
    return UserMeResponse(
        id=request.state.id, kind=kind, tier=cfg.tier, quotas=cfg.quotas
    )


@router.get(
    "/me/usage",
    response_model=Usage,
    status_code=status.HTTP_200_OK,
    summary="Get the authenticated owner's usage",
)
async def get_me_usage(request: Request) -> Usage:
    """Return current usage against the owner's resolved limits, per resource type."""
    return Usage(**await get_usage(request.state.id, request.state.access))
