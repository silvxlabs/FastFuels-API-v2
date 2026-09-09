# Plan: Enforce `email_verified` for password-provider tokens (#570)

Spec: `specs/570-enforce-email-verified.md`. One feature branch
(`570-enforce-email-verified`), one PR. Flag ships **off** — no behaviour change
on merge.

Build/test commands (run from `services/api`):
- Tests: `uv run pytest tests/auth/test_auth.py`
- Lint/format: `uvx ruff format` && `uvx ruff check`

## Checklist

### [ ] 1. Config flag
`services/lib/lib/config.py`: add
`ENFORCE_EMAIL_VERIFICATION = os.getenv("ENFORCE_EMAIL_VERIFICATION", "false").lower() == "true"`,
placed with the other env-derived settings. Default off.

**Acceptance:** importable from `lib.config`; false unless the env var is set truthy.

### [ ] 2. Structured detail model + OpenAPI 403
Mirror the `QuotaExceededDetail` / `QUOTA_429_RESPONSE` convention already in
`api/quota.py`:
- Add an `EmailNotVerifiedDetail` pydantic model (`reason: str = "EMAIL_NOT_VERIFIED"`,
  `message: str`) — put it where the auth error shape naturally lives (a new
  `api/auth.py` model, or alongside the other error models; keep it simple).
- Document the `403` on the OpenAPI surface. Auth is a **global dependency**, not a
  per-route concern, so add `responses={403: {"model": EmailNotVerifiedDetail, ...}}`
  to the single `include_router(api_router, dependencies=[Depends(authenticate_user)])`
  call in `api/app.py`, so it lands on every route at once. Verify the app still
  builds its OpenAPI schema (`app.openapi()` doesn't raise).

**Acceptance:** `app.openapi()` includes the 403 with the `EmailNotVerifiedDetail`
schema; model default `reason` is `"EMAIL_NOT_VERIFIED"`.

### [ ] 3. Enforcement in `_token_auth` (depends on 1, 2)
`api/auth.py` `_token_auth`, immediately after `decoded = verify_id_token(token)`
and before setting `request.state`:

```python
provider = decoded.get("firebase", {}).get("sign_in_provider")
if (
    ENFORCE_EMAIL_VERIFICATION
    and provider == "password"
    and not decoded.get("email_verified", False)
):
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=EmailNotVerifiedDetail(
            message="Verify your email address to continue."
        ).model_dump(),
    )
```

Keep the existing `is_guest` computation (it already reads `sign_in_provider`; reuse
the local `provider` variable rather than decoding twice). No comment restating the
code — the check is self-evident.

**Acceptance:** guest/Google/verified-password/API-key paths unchanged; only an
unverified password token with the flag on is rejected.

### [ ] 4. Tests
`tests/auth/test_auth.py`, matching the existing style (patch `verify_id_token` and
the flag):
- flag on, unverified password → 403, `detail["reason"] == "EMAIL_NOT_VERIFIED"`.
- flag on, verified password → 200/authorised.
- flag on, Google (`google.com`) → unchanged.
- flag on, anonymous → unchanged, `is_guest is True`.
- flag off, unverified password → unchanged (200).

**Acceptance:** `uv run pytest tests/auth/test_auth.py` green; ruff clean.

### [ ] 5. Counting script (deliverable; run by a credentialed operator)
`scripts/count_unverified_password_users.py`: paginate `firebase_admin.auth.list_users()`,
count users whose provider set includes `password` and whose `email_verified` is
false; print the total (and the grand total of password users for context). Header
comment: needs Firebase Admin credentials; output goes on issue #570 before the flag
is enabled. Not wired into CI or the app.

**Acceptance:** script imports and is runnable (`python scripts/... --help` or a dry
structure check); it is not imported by app code.

## Close-out

`/cross-check` the whole diff against the `main` merge-base, fix findings, then delete
this spec and plan in the final commit. Mark the PR ready with `Closes #570`. The
PR description must note: flag defaults off, enabling is gated on the unverified-user
count, and the `EMAIL_NOT_VERIFIED` contract pairs with FastFuels-Web#312. **Never
merge** — the human reviews and merges.
