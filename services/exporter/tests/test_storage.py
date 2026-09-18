"""Unit tests for exporter.storage signed-download URL generation."""

from unittest.mock import patch

from exporter.storage import generate_signed_download


class TestGenerateSignedDownload:
    """generate_signed_download must force an attachment download (#603).

    Browser-renderable formats (CSV, GeoJSON) otherwise open inline in the tab
    instead of downloading. The signer is mocked so no GCS/credentials are
    touched; the assertions are on the parameters passed to it.
    """

    @patch("exporter.storage.generate_download_signed_url")
    def test_inventory_csv_sets_attachment_disposition(self, mock_sign):
        gcs_path = "gs://exports-v2/exp123/My_Export.csv"

        generate_signed_download(gcs_path, expiration_days=7)

        _, kwargs = mock_sign.call_args
        assert kwargs["response_disposition"] == 'attachment; filename="My_Export.csv"'

    @patch("exporter.storage.generate_download_signed_url")
    def test_grid_geotiff_sets_attachment_disposition(self, mock_sign):
        gcs_path = "gs://exports-v2/exp123/My_Export.tif"

        generate_signed_download(gcs_path, expiration_days=7)

        _, kwargs = mock_sign.call_args
        assert kwargs["response_disposition"] == 'attachment; filename="My_Export.tif"'

    @patch("exporter.storage.generate_download_signed_url")
    def test_bucket_blob_and_expiration_still_forwarded(self, mock_sign):
        gcs_path = "gs://exports-v2/exp123/data/My_Export.zip"

        generate_signed_download(gcs_path, expiration_days=3)

        args, kwargs = mock_sign.call_args
        # bucket, blob path, and expiration are forwarded positionally.
        assert args[0] == "exports-v2"
        assert args[1] == "exp123/data/My_Export.zip"
        assert args[2] == 3
        # Filename is the object basename, not the full blob path.
        assert kwargs["response_disposition"] == (
            'attachment; filename="My_Export.zip"'
        )
