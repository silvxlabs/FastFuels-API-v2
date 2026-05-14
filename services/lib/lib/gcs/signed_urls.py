"""
GCS signed URL generation.

Provides functions to generate signed URLs for client-side uploads and downloads.
These URLs allow clients to upload/download directly to/from GCS without
proxying through the API server.
"""

from datetime import timedelta

from google import auth
from google.auth import impersonated_credentials
from google.auth.credentials import Credentials
from google.auth.transport import requests
from google.cloud import storage

gcs_client: storage.Client = storage.Client()


def generate_upload_signed_url(
    bucket_name: str,
    blob_path: str,
    content_type: str,
    max_size_bytes: int = 500_000_000,
    expiration_minutes: int = 60,
) -> str:
    """
    Generate a signed URL for uploading a file to GCS.

    Args:
        bucket_name: Name of the GCS bucket.
        blob_path: Path where the file will be stored in the bucket.
        content_type: MIME type the client must use when uploading.
        max_size_bytes: Maximum allowed upload size in bytes. Default 500MB.
        expiration_minutes: URL validity period in minutes. Default 60.

    Returns:
        Signed URL string for PUT request. Clients must include both
        Content-Type and x-goog-content-length-range headers matching
        the values used during signing.
    """
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method="PUT",
        content_type=content_type,
        credentials=get_signing_credentials(),
        headers={"x-goog-content-length-range": f"0,{max_size_bytes}"},
    )


def generate_download_signed_url(
    bucket_name: str,
    blob_path: str,
    expiration_days: int = 7,
) -> str:
    """
    Generate a signed URL for downloading a file from GCS.

    Args:
        bucket_name: Name of the GCS bucket.
        blob_path: Path to the file in the bucket.
        expiration_days: URL validity period in days. Default 7.

    Returns:
        Signed URL string for GET request.
    """
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(days=expiration_days),
        method="GET",
        credentials=get_signing_credentials(),
    )


def get_signing_credentials() -> Credentials:
    """
    Get impersonated credentials for signing URLs.

    When running on GCP with Application Default Credentials, we need to
    create impersonated credentials to sign URLs (ADC alone can't sign).

    Returns:
        Credentials object suitable for URL signing.
    """
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials, _ = auth.default(scopes=scopes)

    if credentials.token is None:
        credentials.refresh(requests.Request())

    return impersonated_credentials.Credentials(
        source_credentials=credentials,
        target_principal=credentials.service_account_email,
        target_scopes=scopes,
        lifetime=3600,
        delegates=[credentials.service_account_email],
    )
