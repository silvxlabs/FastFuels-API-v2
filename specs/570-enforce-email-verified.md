# Spec: Enforce `email_verified` for password-provider bearer tokens (#570)

Part of guest mode (tracking: FastFuels-Web#315). Pairs with FastFuels-Web#312
(guest → account conversion + verify-email screen). Ships behind a flag,
default off.

## Problem

Today any email/password signup receives standard-tier quotas and 180-day
retention with no email verification. Once guests can convert to real accounts
(#312), an unverified email address becomes a trivially-minted full account.
Close the gap for every password-provider account.

## Behaviour

In `_token_auth`, after the token is decoded, when **all** of:

- `ENFORCE_EMAIL_VERIFICATION` is on, and
- `firebase.sign_in_provider == "password"`, and
- `email_verified` is falsy (missing counts as false),

reject with **`403`** and a structured detail the webapp keys on:

```json
{ "reason": "EMAIL_NOT_VERIFIED", "message": "Verify your email address to continue." }
```

Google (`google.com`) and anonymous (guest) tokens never match `password`, so
they are unaffected. API-key auth is unaffected. When the flag is off, every
token behaves exactly as it does today.

`email-link` (passwordless) sign-in also reports `sign_in_provider == "password"`,
but those tokens always carry `email_verified: true`, so they pass the check
naturally — no special-casing.

## Contract with the webapp (#312)

- The reason string is exactly `EMAIL_NOT_VERIFIED` (verbatim; the webapp branches
  on it to show the verify-email screen). Do not rename without updating #312.
- A guest who just linked email/password holds a token with `password` +
  `email_verified: false`, so with the flag on they will 403 until they click the
  verification link **and the client force-refreshes the ID token**. That token
  refresh is #312's responsibility — note it there.

## Rollout: grandfather existing users, gate new ones

Enforcement applies to **new accounts only**. Existing password users are trusted
and grandfathered rather than forced to re-verify. There is no account-creation
timestamp in a Firebase ID token, so "old vs new" is not decidable at auth time —
instead a one-time **backfill** marks every existing unverified password user
`email_verified: true` in Firebase Auth. After it runs, only new signups start
`false`, so the unchanged token check gates them alone. No grace window, no
verification-email campaign.

Order: run `scripts/grandfather_password_users.py` (dry run), review the count,
re-run with `--apply`, **then** enable `ENFORCE_EMAIL_VERIFICATION`. The code ships
with the flag off regardless; the backfill and enabling are separate, credentialed,
per-environment steps. Google and anonymous users are never touched by the backfill.

## Acceptance

- Flag on + unverified password token → `403` with `reason: "EMAIL_NOT_VERIFIED"`.
- Flag on + verified password / Google / anonymous token → unchanged (200; guest
  stays guest).
- Flag off + unverified password token → unchanged (200).
- The `403` is documented in the OpenAPI surface.

## Out of scope

- Sending verification emails and the verify-your-email screen (FastFuels-Web #312).
- Running the backfill and enabling the flag in any environment (credentialed, run-once steps).
