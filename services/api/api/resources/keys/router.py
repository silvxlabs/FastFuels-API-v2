"""
api/v2/resources/keys/router.py

CRUD endpoints for API key resources.
"""

import secrets
from datetime import timedelta

from fastapi import APIRouter, HTTPException, Query, Request, status
from google.cloud.firestore import FieldFilter

from api.auth import hash_api_key, invalidate_key_cache
from api.db.documents import (
    delete_document_async,
    firestore_client,
    get_document_async,
    set_document_async,
)
from api.resources.keys.schema import (
    Access,
    CreateKeyRequest,
    CreateKeyResponse,
    Key,
    ListKeysResponse,
)
from lib.config import APPLICATIONS_COLLECTION, KEYS_COLLECTION

router = APIRouter()


async def _get_application_for_ownership(app_id: str, owner_id: str) -> dict:
    """Fetch an application and validate ownership. Raises 404 if not found or not owned."""
    doc = (
        await firestore_client.collection(APPLICATIONS_COLLECTION)
        .document(app_id)
        .get()
    )
    if not doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        )
    data = doc.to_dict()
    if data.get("owner_id") != owner_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        )
    return data


async def _list_keys(
    field: str,
    value: str,
    page: int = 0,
    size: int = 100,
) -> tuple[list[dict], int]:
    """List keys matching a single equality filter with pagination."""
    collection_ref = firestore_client.collection(KEYS_COLLECTION)
    query = collection_ref.where(filter=FieldFilter(field, "==", value))
    count_result = await query.count().get()
    total_items = count_result[0][0].value
    paginated = query.offset(page * size).limit(size)
    docs = await paginated.get()
    keys = [doc.to_dict() for doc in docs]
    return keys, total_items


@router.post(
    "",
    response_model=CreateKeyResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
)
async def create_key(request: Request, body: CreateKeyRequest) -> CreateKeyResponse:
    """Create a new API key for programmatic access.

    Returns the key secret exactly once. The secret cannot be retrieved again —
    only its SHA-256 hash (the key ID) is stored.
    """
    if body.access == Access.APPLICATION:
        # Validate that the user owns the application
        await _get_application_for_ownership(
            body.application_id, owner_id=request.state.id
        )
        owner_id = body.application_id
    else:
        owner_id = request.state.id

    raw_secret = secrets.token_hex(32)
    key_id = hash_api_key(raw_secret)

    key = Key(
        id=key_id,
        owner_id=owner_id,
        creator_id=request.state.id,
        name=body.name,
        description=body.description,
        valid_days=body.valid_days,
        scopes=body.scopes,
        access=body.access,
        application_id=body.application_id,
    )

    # Calculate expiration
    key.expires_on = key.created_on + timedelta(days=body.valid_days)

    await set_document_async(KEYS_COLLECTION, key_id, key.model_dump())

    return CreateKeyResponse(**key.model_dump(), secret=raw_secret)


@router.get(
    "",
    response_model=ListKeysResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="List API keys",
)
async def list_keys(
    request: Request,
    page: int = Query(0, ge=0),
    size: int = Query(100, ge=1, le=100),
) -> ListKeysResponse:
    """List API keys accessible to the authenticated user or application."""
    requester_id = request.state.id
    access_type = request.state.access

    if access_type == Access.PERSONAL:
        # Personal access: list all keys created by this user
        field, value = "creator_id", requester_id
    elif access_type == Access.APPLICATION:
        # Application access: only keys owned by this application
        field, value = "owner_id", requester_id
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid access type",
        )

    key_dicts, total_items = await _list_keys(
        field=field,
        value=value,
        page=page,
        size=size,
    )

    keys = [Key(**d) for d in key_dicts]

    return ListKeysResponse(
        keys=keys,
        current_page=page,
        page_size=size,
        total_items=total_items,
    )


@router.get(
    "/{key_id}",
    response_model=Key,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="Get API key",
)
async def get_key_by_id(request: Request, key_id: str) -> Key:
    """Get an API key by ID with two-tier ownership check."""
    _, snapshot = await get_document_async(KEYS_COLLECTION, key_id)
    key_data = snapshot.to_dict()

    # Two-tier ownership: personal keys check direct ownership,
    # application keys check that user owns the application
    if key_data["access"] == Access.PERSONAL:
        if key_data["owner_id"] != request.state.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Key not found",
            )
    elif key_data["access"] == Access.APPLICATION:
        await _get_application_for_ownership(
            key_data["owner_id"], owner_id=request.state.id
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Key not found",
        )

    return Key(**key_data)


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete API key",
)
async def delete_key(request: Request, key_id: str) -> None:
    """Delete an API key with ownership check. Clears the auth cache."""
    _, snapshot = await get_document_async(KEYS_COLLECTION, key_id)
    key_data = snapshot.to_dict()

    if key_data["access"] == Access.PERSONAL:
        if key_data["owner_id"] != request.state.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Key not found",
            )
    elif key_data["access"] == Access.APPLICATION:
        await _get_application_for_ownership(
            key_data["owner_id"], owner_id=request.state.id
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Key not found",
        )

    await delete_document_async(KEYS_COLLECTION, key_id)
    await invalidate_key_cache(key_id)
