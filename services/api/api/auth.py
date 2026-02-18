"""
api/auth.py

Authentication module. Handles API key and Bearer token auth.

API key secrets are hashed (SHA-256) before storage. The hash serves as the
Firestore document ID. On auth, the incoming secret is hashed and looked up
by document ID.
"""

import hashlib

import firebase_admin
from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader
from firebase_admin.auth import verify_id_token
from ring import lru

from api.db.documents import firestore_client
from api.resources.keys.schema import Access, Key
from lib.config import DEV_API_KEY, DEV_OWNER_ID, FASTFUELS_DEV_MODE, KEYS_COLLECTION

API_KEY_HEADER = APIKeyHeader(name="api-key", auto_error=False)
AUTH_BEARER = HTTPBearer(auto_error=False)


def hash_api_key(raw_secret: str) -> str:
    """Compute the SHA-256 hash of a raw API key secret."""
    return hashlib.sha256(raw_secret.encode()).hexdigest()


@lru(force_asyncio=True, expire=300)
async def _lookup_by_doc_id(doc_id: str) -> Key | None:
    """Cached Firestore lookup by document ID. Returns None on miss."""
    doc = await firestore_client.collection(KEYS_COLLECTION).document(doc_id).get()
    if not doc.exists:
        return None
    return Key(**doc.to_dict())


async def resolve_api_key(raw_secret: str) -> Key:
    """Resolve a raw API key secret to a Key object.

    Hashes the secret (SHA-256) and looks up by document ID.
    """
    key_hash = hash_api_key(raw_secret)
    key = await _lookup_by_doc_id(key_hash)
    if key is not None:
        return key

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


async def invalidate_key_cache(doc_id: str) -> None:
    """Invalidate the cached key entry for a Firestore document ID."""
    await _lookup_by_doc_id.delete(doc_id)


async def _api_key_auth(request: Request, key_id: str) -> Request:
    """Authenticate via API key header."""
    key = await resolve_api_key(key_id)

    if key.is_expired():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )

    if not key.has_permission(request.method):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )

    request.state.id = key.owner_id
    request.state.access = key.access

    return request


def _ensure_firebase_initialized():
    """Initialize Firebase app on first use."""
    if not firebase_admin._apps:
        firebase_admin.initialize_app()


def _token_auth(request: Request, token: str | None) -> Request:
    """Authenticate via Firebase Bearer token."""
    _ensure_firebase_initialized()
    try:
        request.state.id = verify_id_token(token)["uid"]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    request.state.access = Access.PERSONAL
    return request


async def authenticate_user(
    request: Request,
    api_key: str = Security(API_KEY_HEADER),
    bearer: HTTPAuthorizationCredentials = Security(AUTH_BEARER),
) -> Request:
    """Main auth dependency. Checks API key first, then Bearer token."""
    if FASTFUELS_DEV_MODE and DEV_API_KEY and api_key == DEV_API_KEY:
        request.state.id = DEV_OWNER_ID
        request.state.access = Access.PERSONAL
        return request

    if api_key:
        return await _api_key_auth(request, api_key)
    elif bearer:
        return _token_auth(request, bearer.credentials)
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
