"""
api/v2/resources/applications/router.py

CRUD endpoints for application resources.
"""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, Request, status
from google.cloud.firestore import FieldFilter

from api.auth import invalidate_key_cache
from api.db.documents import (
    delete_document_async,
    firestore_client,
    get_document_async,
    list_documents_async,
    set_document_async,
    update_document_async,
)
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas
from api.resources.applications.schema import (
    Application,
    CreateApplicationRequest,
    ListApplicationsResponse,
    UpdateApplicationRequest,
)
from api.resources.keys.schema import Access
from lib.config import APPLICATIONS_COLLECTION, KEYS_COLLECTION

router = APIRouter()


@router.post(
    "",
    response_model=Application,
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
    summary="Create an application",
    responses=QUOTA_429_RESPONSE,
)
async def create_application(
    request: Request,
    body: CreateApplicationRequest,
) -> Application:
    """Create a new application. Only personal-access users can create applications."""
    # Prevent applications from creating applications (privilege escalation)
    if request.state.access == Access.APPLICATION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Applications cannot create other applications",
        )

    await enforce_create_quotas(APPLICATIONS_COLLECTION, request)

    app_id = uuid.uuid4().hex
    request_time = datetime.now(UTC)

    app = Application(
        id=app_id,
        owner_id=request.state.id,
        name=body.name,
        description=body.description,
        created_on=request_time,
        modified_on=request_time,
    )

    await set_document_async(APPLICATIONS_COLLECTION, app_id, app.model_dump())

    return app


@router.get(
    "",
    response_model=ListApplicationsResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="List applications",
)
async def list_applications(
    request: Request,
    page: int = Query(0, ge=0),
    size: int = Query(100, ge=1, le=100),
) -> ListApplicationsResponse:
    """List applications owned by the authenticated user."""
    docs, total_items = await list_documents_async(
        collection=APPLICATIONS_COLLECTION,
        owner_id=request.state.id,
        page=page,
        size=size,
    )

    applications = [Application(**doc.to_dict()) for doc in docs]

    return ListApplicationsResponse(
        applications=applications,
        current_page=page,
        page_size=size,
        total_items=total_items,
    )


@router.get(
    "/{application_id}",
    response_model=Application,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="Get an application by ID",
)
async def get_application(
    request: Request,
    application_id: str,
) -> Application:
    """Get an application by ID with ownership check."""
    _, snapshot = await get_document_async(
        APPLICATIONS_COLLECTION, application_id, owner_id=request.state.id
    )
    return Application(**snapshot.to_dict())


@router.patch(
    "/{application_id}",
    response_model=Application,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="Update an application",
)
async def update_application(
    request: Request,
    application_id: str,
    body: UpdateApplicationRequest,
) -> Application:
    """Update an application's name or description."""
    _, snapshot = await get_document_async(
        APPLICATIONS_COLLECTION, application_id, owner_id=request.state.id
    )
    app_data = snapshot.to_dict()

    update_data = {}
    if body.name is not None:
        update_data["name"] = body.name
    if body.description is not None:
        update_data["description"] = body.description

    if not update_data:
        return Application(**app_data)

    update_data["modified_on"] = datetime.now(UTC)

    await update_document_async(APPLICATIONS_COLLECTION, application_id, update_data)

    # Merge and return
    app_data.update(update_data)
    return Application(**app_data)


@router.delete(
    "/{application_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an application",
)
async def delete_application(
    request: Request,
    application_id: str,
) -> None:
    """Delete an application and all its API keys. Validates ownership first."""
    await get_document_async(
        APPLICATIONS_COLLECTION, application_id, owner_id=request.state.id
    )

    # Cascade delete: remove all keys owned by this application
    keys_query = firestore_client.collection(KEYS_COLLECTION).where(
        filter=FieldFilter("owner_id", "==", application_id)
    )
    key_docs = await keys_query.get()
    if key_docs:
        batch = firestore_client.batch()
        for doc in key_docs:
            batch.delete(doc.reference)
        await batch.commit()
        for doc in key_docs:
            await invalidate_key_cache(doc.id)

    await delete_document_async(APPLICATIONS_COLLECTION, application_id)
