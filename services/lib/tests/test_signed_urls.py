"""Unit tests for lib.gcs.signed_urls.

The signer talks to GCS and needs impersonated credentials to sign, so both the
module-level client and the credential helper are mocked. The assertions are on
what gets passed to ``blob.generate_signed_url`` — no network or credentials are
touched.
"""

from unittest.mock import MagicMock, patch

from lib.gcs.signed_urls import generate_download_signed_url


@patch("lib.gcs.signed_urls.get_signing_credentials")
@patch("lib.gcs.signed_urls.gcs_client")
def test_download_url_forwards_response_disposition(mock_client, mock_creds):
    """A response_disposition is signed into the URL (#603)."""
    blob = MagicMock()
    blob.generate_signed_url.return_value = "https://signed.example/download"
    mock_client.bucket.return_value.blob.return_value = blob

    url = generate_download_signed_url(
        "exports-v2",
        "exp123/My_Export.csv",
        expiration_days=7,
        response_disposition='attachment; filename="My_Export.csv"',
    )

    assert url == "https://signed.example/download"
    _, kwargs = blob.generate_signed_url.call_args
    assert kwargs["method"] == "GET"
    assert kwargs["response_disposition"] == ('attachment; filename="My_Export.csv"')


@patch("lib.gcs.signed_urls.get_signing_credentials")
@patch("lib.gcs.signed_urls.gcs_client")
def test_download_url_disposition_defaults_to_none(mock_client, mock_creds):
    """Omitting response_disposition preserves the prior signing behavior."""
    blob = MagicMock()
    mock_client.bucket.return_value.blob.return_value = blob

    generate_download_signed_url("exports-v2", "exp123/data.tif")

    _, kwargs = blob.generate_signed_url.call_args
    assert kwargs["response_disposition"] is None
