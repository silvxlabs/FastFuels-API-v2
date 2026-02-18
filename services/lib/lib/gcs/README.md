# lib/v2/gcs - Synchronous Google Cloud Storage Operations

This module provides synchronous GCS operations for background jobs.

## Modules

- **blobs.py** - GCSFS client and blob upload, download, delete, existence checks
- **signed_urls.py** - GCS client and signed URL generation for uploads/downloads

## Usage

### Blob Operations

```python
from lib.gcs import upload_file, download_file, delete_file, delete_directory

# Upload a local file
upload_file("/local/path/file.csv", "gs://bucket/path/file.csv")

# Download to local
download_file("gs://bucket/path/file.csv", "/local/path/file.csv")

# Delete operations
delete_file("bucket-name", "path/to/file.csv")
delete_directory("bucket-name", "path/to/directory")
```

### Signed URLs

```python
from lib.gcs import generate_upload_signed_url, generate_download_signed_url

# Generate upload URL (PUT)
upload_url = generate_upload_signed_url(
    bucket_name="my-bucket",
    blob_path="uploads/file.csv",
    max_size_bytes=500_000_000,  # 500MB
)

# Generate download URL (GET)
download_url = generate_download_signed_url(
    bucket_name="my-bucket",
    blob_path="exports/file.zip",
)
```

## Note on Clients

This module uses two GCS client libraries:
- **google.cloud.storage.Client** - For signed URL generation (requires credentials)
- **gcsfs.GCSFileSystem** - For file operations (simpler API, better for streaming)
