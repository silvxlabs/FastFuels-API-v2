# Vendored DUET artifacts

Copied verbatim from `gs://silvx-fastfuels-executables/` (uploaded 2025-08-08).
Baked into the image rather than downloaded at runtime as v1 did: together they
are under 200 KB, and fetching them per job buys nothing while adding cold-start
latency and a failure mode.

| file | source object | sha256 |
|---|---|---|
| `duet_v2.1_FF_linux.exe` | `duet_v2.1_FF_linux.exe` | `42fbf89ba9be0cc78f035a8fdfd2754427097f966645801724b898134de1ff21` |
| `FIA_FastFuels_fin_fulllist_populated.txt` | same name | `ee445a2fe878c57693418ca30ece950ebf3fc7dcb14ed0d9f5ce8c8e05caaf3d` |

**Both must stay byte-identical to those objects.** The species table is parsed
by DUET's Fortran, not only by us, so reformatting it risks changing what the
model reads — `.pre-commit-config.yaml` excludes this directory from
`end-of-file-fixer` and `trailing-whitespace` for that reason. `duet_species.py`
derives its remap from the same copy the binary reads, so the two can never
disagree.

The binary is an x86-64 PIE ELF linked against `libgfortran.so.5`, which the
Dockerfile installs. It cannot run on an arm64 dev machine, so
`tests/handlers/test_duet_end_to_end.py` skips itself there and runs in CI.

Bucket hygiene, in case you go looking: `duet_v2.1_FF_mac.exe` in that bucket is
**misnamed** — it is a byte-identical copy of this Linux ELF. The real macOS
build is `duet_mac_v2.1.exe` (Mach-O arm64), which is not vendored here.
