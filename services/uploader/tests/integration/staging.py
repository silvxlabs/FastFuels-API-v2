"""Where the integration tests stage their upload objects in GCS.

UPLOADS_BUCKET carries a live Eventarc object-finalize trigger into the deployed
uploader. Staging a test object under the real ``{resource_type}/{id}/{file}``
dispatch convention makes that deployed service pick it up and run the same
handler, on the same object and the same Firestore doc, concurrently with the
in-process handler call the test makes itself. The two race: whichever finishes
first deletes the staged upload out from under the other, and either may update
or observe a doc the other has already torn down (#349).

These tests call handlers directly, and a handler treats ``object_name`` as an
opaque path, so the dispatch convention buys them nothing. Staging under a
resource type the dispatcher does not own makes the trigger inert: main.py
resolves no collection for it and returns before touching Firestore or GCS. The
convention itself is covered where it belongs, in tests/test_main.py.

Keep the name three segments long. main.py rejects any other shape as a
malformed path and logs it at ERROR; a three-segment name with an unowned
resource type takes the quieter "ignore objects we don't own" branch.
"""

STAGING_PREFIX = "ci"


def staged_object_name(resource_id: str, filename: str) -> str:
    """GCS object name for a test upload staged in UPLOADS_BUCKET."""
    return f"{STAGING_PREFIX}/{resource_id}/{filename}"
