"""
Count password-provider Firebase users whose email address is unverified (#570).

The number gates enabling ENFORCE_EMAIL_VERIFICATION: it decides whether the
rollout needs a grace window or a one-time verification-email campaign. Post the
output on issue #570 before the flag is enabled in any environment.

Needs Firebase Admin credentials (GOOGLE_APPLICATION_CREDENTIALS or ADC) with
Auth read access. Not imported by app code and not wired into CI.

Usage:
    uv run --active python scripts/count_unverified_password_users.py
"""

import firebase_admin
from firebase_admin import auth


def count_unverified_password_users() -> tuple[int, int]:
    """Return (unverified_password_users, total_password_users).

    A user counts as password-provider when "password" is among its linked
    sign-in providers; unverified means ``email_verified`` is false.
    """
    if not firebase_admin._apps:
        firebase_admin.initialize_app()

    total = 0
    unverified = 0
    for user in auth.list_users().iterate_all():
        providers = {p.provider_id for p in user.provider_data}
        if "password" not in providers:
            continue
        total += 1
        if not user.email_verified:
            unverified += 1
    return unverified, total


def main() -> None:
    unverified, total = count_unverified_password_users()
    print(f"password-provider users:            {total}")
    print(f"  of which email_verified == false: {unverified}")


if __name__ == "__main__":
    main()
