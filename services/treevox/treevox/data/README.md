# DUET artifacts

## The DUET binary is NOT stored here — it is fetched at runtime

The DUET executable is a **restricted third-party (LANL) artifact and is not
publicly distributable**, so it must **never** be committed to this repo (which
is public). `.gitignore` blocks `*.exe` under this directory as a backstop.

Instead, `handlers/duet.py` downloads it at runtime from a private bucket
(`DUET_BINARY_GCS`, set via env on the treevox service — the Cloud Run service
account has read access) and caches it once per container. This is what v1 did;
vendoring it into the image, however small, put a restricted binary in a public
repo and was reverted.

The binary is an x86-64 PIE ELF linked against `libgfortran.so.5`, which the
Dockerfile installs (`libgfortran5`). The runtime fetch is exercised in CI by
the deployed-service integration test (`tests/integration/test_duet_grid.py`,
which routes through Cloud Run when `DEPLOYMENT_ENV != "local"`).

### Running DUET locally

The ELF cannot run on an arm64 dev machine. Point `DUET_BINARY_PATH` at a native
build to skip the download entirely (it takes precedence over the fetch):

```bash
DUET_BINARY_PATH=/path/to/native/duet.exe uv run --active \
  pytest tests/integration/test_duet_grid.py::test_duet_on_pim_tree_grid
```

On an arm64 Mac the native build is the macOS arm64 DUET executable (Mach-O
arm64), linked against Homebrew's `libgfortran.5.dylib` / `libquadmath.0.dylib`
(`brew install gcc`). Ask a maintainer for access to the private executables
bucket; do not copy the binary anywhere public.

## The species table (vendored here)

| file | sha256 |
|---|---|
| `FIA_FastFuels_fin_fulllist_populated.txt` | `ee445a2fe878c57693418ca30ece950ebf3fc7dcb14ed0d9f5ce8c8e05caaf3d` |

This FIA species → litter-parameter table is parsed by DUET's Fortran (not only
by us), so it **must stay byte-identical** to the copy the binary reads —
`.pre-commit-config.yaml` excludes this directory from `end-of-file-fixer` and
`trailing-whitespace` so nothing reformats it. `duet_species.py` derives its
remap from this same file, so the two can never disagree.
