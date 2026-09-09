"""
Grandfather existing password-provider Firebase users as email-verified (#570).

Email-verification enforcement (ENFORCE_EMAIL_VERIFICATION) applies only to new
accounts: existing password users are trusted and marked ``email_verified: true``
so they are never gated. Run this once, review the dry-run, then re-run with
--apply, and only then enable the flag. New signups keep ``email_verified: false``
until they verify, so enforcement affects them alone.

A user is in scope when "password" is among its linked sign-in providers and its
email is currently unverified. Google and anonymous users are never touched.

Needs Firebase Admin credentials (GOOGLE_APPLICATION_CREDENTIALS or ADC) with
Auth read/write access. Not imported by app code and not wired into CI.

Usage:
    uv run --active python scripts/grandfather_password_users.py           # dry run
    uv run --active python scripts/grandfather_password_users.py --apply   # writes
"""

import argparse

import firebase_admin
from firebase_admin import auth


def _unverified_password_uids() -> list[str]:
    """Return the uids of unverified password-provider users."""
    if not firebase_admin._apps:
        firebase_admin.initialize_app()

    uids = []
    for user in auth.list_users().iterate_all():
        providers = {p.provider_id for p in user.provider_data}
        if "password" in providers and not user.email_verified:
            uids.append(user.uid)
    return uids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Set email_verified=True on the matched users. Omit for a dry run.",
    )
    args = parser.parse_args()

    uids = _unverified_password_uids()
    print(f"unverified password-provider users: {len(uids)}")

    if not args.apply:
        print("dry run — re-run with --apply to mark them verified")
        return

    for uid in uids:
        auth.update_user(uid, email_verified=True)
    print(f"marked {len(uids)} users email_verified=True")


if __name__ == "__main__":
    main()
