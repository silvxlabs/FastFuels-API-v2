"""
End-to-end timing benchmark for in-place inventory modifications.

Drives the *deployed* API + standgen pipeline (not a unit/integration test) to
measure how long an in-place ``POST .../{inventory_id}/modifications`` takes to
complete, end to end. Used to A/B the storage rewrite in #355: run once against
the currently-deployed full-rewrite standgen, then again after deploying the
partial-rewrite update, and compare.

What it does:
  1. Creates a sized domain, a TreeMap PIM grid, and a PIM tree inventory
     (deterministic by ``--seed`` so before/after runs operate on identical data).
  2. Reads the inventory's partition count / row count for context.
  3. Times two modification scenarios, each repeated ``--reps`` times
     (the first rep per scenario is discarded as a cold-start warmup):
       - ``scoped``: multiply tree height inside a sub-region of the domain.
         Touches only the partitions that region overlaps -> the partial-rewrite
         win shows up here.
       - ``global``: multiply every tree's height. Touches all partitions ->
         a no-regression check (partial == full when everything changes).
     A multiply (not a remove) is used so every rep does the same work on a
     stable partition layout (no rows are dropped between reps).
  4. Writes results to ``bench_inplace_<label>.json`` and prints a summary.
  5. Deletes the domain/grid/inventory (unless ``--keep``).

Usage (from services/api, so the API venv + repo-root .env are picked up):

    uv run python benchmarks/bench_inplace_inventory.py --label before
    # ... merge #355, deploy standgen-v2-prod ...
    uv run python benchmarks/bench_inplace_inventory.py --label after

Auth/config come from the repo-root ``.env`` (``TEST_API_KEY``). The base URL
defaults to api-v2-prod; override with ``--base-url``.
"""

import argparse
import json
import math
import statistics
import time
import uuid
from pathlib import Path

from dotenv import dotenv_values
from httpx import Client

PROD_BASE_URL = "https://api-v2-prod-nyvjyh5ywa-uw.a.run.app"

# Blue Mountain Recreation Area center (near Missoula, MT) — same locale as the
# canonical test domain, just sized up to yield a multi-partition inventory.
CENTER_LON = -114.1038
CENTER_LAT = 46.8287

# services/api/benchmarks/bench_inplace_inventory.py -> repo root is 3 levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def km_to_deg(center_lat: float, dx_km: float, dy_km: float) -> tuple[float, float]:
    """Convert east/north kilometer extents to (dlon, dlat) degrees at a latitude."""
    dlat = dy_km / 111.32
    dlon = dx_km / (111.32 * math.cos(math.radians(center_lat)))
    return dlon, dlat


def square_bbox(half_km: float) -> tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) of a square domain centered on the site."""
    dlon, dlat = km_to_deg(CENTER_LAT, half_km, half_km)
    return (
        CENTER_LON - dlon,
        CENTER_LAT - dlat,
        CENTER_LON + dlon,
        CENTER_LAT + dlat,
    )


def polygon_feature_collection(
    bbox: tuple[float, float, float, float], name: str
) -> dict:
    min_lon, min_lat, max_lon, max_lat = bbox
    ring = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
        "name": name,
        "description": "Sized domain for in-place modification benchmarking.",
    }


def scoped_strip_geometry(bbox: tuple[float, float, float, float], frac: float) -> dict:
    """A full-width strip across the southern ``frac`` of the domain.

    A strip (vs. a tiny box) is guaranteed to contain trees while still covering
    only a fraction of the inventory's partitions, so the scoped scenario does
    real, repeatable work without depending on lucky tree placement.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    strip_top = min_lat + (max_lat - min_lat) * frac
    ring = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, strip_top],
        [min_lon, strip_top],
        [min_lon, min_lat],
    ]
    return {
        "type": "Polygon",
        "coordinates": [ring],
        # Carried alongside so the caller can attach the CRS to the condition.
    }


def scenarios(
    bbox: tuple[float, float, float, float], frac: float, factor: float
) -> dict[str, dict]:
    """Return {name: modifications-request-body} for each timed scenario."""
    wgs84 = {"type": "name", "properties": {"name": "EPSG:4326"}}
    strip = scoped_strip_geometry(bbox, frac)
    return {
        "scoped": {
            "modifications": [
                {
                    "conditions": {
                        "source": "geometry",
                        "operator": "within",
                        "geometry": strip,
                        "crs": wgs84,
                    },
                    "actions": {
                        "attribute": "height",
                        "modifier": "multiply",
                        "value": factor,
                    },
                }
            ]
        },
        "global": {
            "modifications": [
                {
                    "conditions": {"attribute": "height", "operator": "gt", "value": 0},
                    "actions": {
                        "attribute": "height",
                        "modifier": "multiply",
                        "value": factor,
                    },
                }
            ]
        },
    }


class Bench:
    def __init__(self, client: Client, poll_interval: float):
        self.client = client
        self.poll_interval = poll_interval

    def poll(self, domain_id: str, kind: str, rid: str, timeout: float) -> dict:
        """Poll a resource until it reaches a terminal status; return its doc."""
        url = f"/domains/{domain_id}/{kind}/{rid}"
        start = time.perf_counter()
        while True:
            r = self.client.get(url)
            r.raise_for_status()
            doc = r.json()
            status = doc.get("status")
            if status == "completed":
                return doc
            if status == "failed":
                err = doc.get("error") or {}
                raise RuntimeError(
                    f"{kind}/{rid} failed: {err.get('code')} - {err.get('message')}"
                )
            if time.perf_counter() - start > timeout:
                raise TimeoutError(
                    f"{kind}/{rid} did not complete within {timeout}s "
                    f"(last status={status}, progress={doc.get('progress')})"
                )
            time.sleep(self.poll_interval)

    def create_domain(self, body: dict) -> str:
        r = self.client.post("/domains", json=body, timeout=120.0)
        r.raise_for_status()
        return r.json()["id"]

    def create_and_wait(
        self, domain_id: str, kind: str, endpoint: str, body: dict, timeout: float
    ) -> dict:
        url = f"/domains/{domain_id}{endpoint}"
        r = self.client.post(url, json=body, timeout=120.0)
        if r.status_code != 201:
            raise RuntimeError(f"POST {url} -> {r.status_code}: {r.text}")
        rid = r.json()["id"]
        log(f"  created {kind} {rid}, polling for completion...")
        return self.poll(domain_id, kind, rid, timeout)

    def time_modification(
        self, domain_id: str, inventory_id: str, body: dict, timeout: float
    ) -> float:
        """POST a modification and block until the inventory is completed again.

        Returns wall-clock seconds from POST to observed completion. The POST
        sets status=pending synchronously (in a Firestore transaction) before
        returning, so polling to completed can't observe a stale prior state.
        """
        url = f"/domains/{domain_id}/inventories/{inventory_id}/modifications"
        t0 = time.perf_counter()
        r = self.client.post(url, json=body, timeout=120.0)
        if r.status_code != 200:
            raise RuntimeError(f"POST {url} -> {r.status_code}: {r.text}")
        self.poll(domain_id, "inventories", inventory_id, timeout)
        return time.perf_counter() - t0


def summarize(durations: list[float]) -> dict:
    return {
        "n": len(durations),
        "min": min(durations),
        "median": statistics.median(durations),
        "mean": statistics.fmean(durations),
        "max": max(durations),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="run label, e.g. 'before'/'after'")
    p.add_argument("--base-url", default=PROD_BASE_URL)
    p.add_argument(
        "--reps",
        type=int,
        default=5,
        help="reps per scenario (rep 1 is a discarded warmup)",
    )
    p.add_argument(
        "--half-km",
        type=float,
        default=1.75,
        help="half-edge of the square domain in km (1.75 -> ~12 km^2)",
    )
    p.add_argument(
        "--scoped-frac",
        type=float,
        default=0.25,
        help="southern fraction of the domain the scoped mod covers",
    )
    p.add_argument(
        "--factor", type=float, default=1.001, help="height multiply factor per rep"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gen-timeout",
        type=float,
        default=290.0,
        help="per-resource completion timeout (s)",
    )
    p.add_argument("--mod-timeout", type=float, default=290.0)
    p.add_argument("--poll-interval", type=float, default=0.3)
    p.add_argument(
        "--reuse-inventory", help="skip setup; benchmark this existing inventory"
    )
    p.add_argument("--reuse-domain", help="domain id for --reuse-inventory")
    p.add_argument(
        "--keep", action="store_true", help="do not delete created resources"
    )
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = p.parse_args()

    env = dotenv_values(REPO_ROOT / ".env")
    api_key = (env.get("TEST_API_KEY") or "").strip().strip('"')
    if not api_key:
        raise SystemExit("TEST_API_KEY not found in repo-root .env")

    bbox = square_bbox(args.half_km)
    suffix = uuid.uuid4().hex[:8]
    log(f"Base URL: {args.base_url}")
    log(
        f"Domain: {2 * args.half_km:.2f} x {2 * args.half_km:.2f} km square, bbox={bbox}"
    )

    created = {"domain": None, "grid": None, "inventory": None}
    results = {
        "label": args.label,
        "base_url": args.base_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": {
            "reps": args.reps,
            "half_km": args.half_km,
            "scoped_frac": args.scoped_frac,
            "factor": args.factor,
            "seed": args.seed,
        },
    }

    with Client(
        base_url=args.base_url, headers={"API-KEY": api_key}, timeout=60.0
    ) as client:
        bench = Bench(client, args.poll_interval)
        try:
            if args.reuse_inventory:
                domain_id = args.reuse_domain
                inventory_id = args.reuse_inventory
                if not domain_id:
                    raise SystemExit("--reuse-inventory requires --reuse-domain")
                log(f"Reusing inventory {inventory_id} in domain {domain_id}")
            else:
                log("Creating domain...")
                domain_id = bench.create_domain(
                    polygon_feature_collection(bbox, f"bench-inplace-{suffix}")
                )
                created["domain"] = domain_id
                log(f"  domain {domain_id}")

                log("Creating TreeMap PIM grid...")
                grid = bench.create_and_wait(
                    domain_id, "grids", "/grids/pim/treemap", {}, args.gen_timeout
                )
                created["grid"] = grid["id"]
                log(f"  grid {grid['id']} completed")

                log("Creating PIM tree inventory (point-process expansion)...")
                inv = bench.create_and_wait(
                    domain_id,
                    "inventories",
                    "/inventories/tree/pim",
                    {"source_pim_grid_id": grid["id"], "seed": args.seed},
                    args.gen_timeout,
                )
                created["inventory"] = inv["id"]
                inventory_id = inv["id"]
                log(f"  inventory {inventory_id} completed")

            # Characterize the inventory.
            meta = client.get(
                f"/domains/{domain_id}/inventories/{inventory_id}/data/metadata"
            )
            meta.raise_for_status()
            md = meta.json()
            results["inventory_id"] = inventory_id
            results["domain_id"] = domain_id
            results["num_partitions"] = md.get("num_partitions")
            results["total_rows"] = md.get("total_rows")
            log(
                f"Inventory: {md.get('total_rows')} trees across "
                f"{md.get('num_partitions')} partitions"
            )

            scen = scenarios(bbox, args.scoped_frac, args.factor)
            results["scenarios"] = {}
            for name, body in scen.items():
                log(f"Scenario '{name}': {args.reps} reps (rep 1 discarded as warmup)")
                reps = []
                for i in range(args.reps):
                    dur = bench.time_modification(
                        domain_id, inventory_id, body, args.mod_timeout
                    )
                    tag = "warmup" if i == 0 else "warm"
                    log(f"  rep {i + 1}/{args.reps} ({tag}): {dur:.2f}s")
                    reps.append(dur)
                warm = reps[1:]
                results["scenarios"][name] = {
                    "cold_rep": reps[0],
                    "warm_reps": warm,
                    "warm_stats": summarize(warm) if warm else None,
                }
        finally:
            if not args.keep and not args.reuse_inventory:
                log("Cleaning up...")
                for kind, rid in (
                    ("inventories", created["inventory"]),
                    ("grids", created["grid"]),
                ):
                    if rid and created["domain"]:
                        try:
                            client.delete(
                                f"/domains/{created['domain']}/{kind}/{rid}",
                                timeout=30.0,
                            )
                        except Exception as e:  # noqa: BLE001
                            log(f"  warning: failed to delete {kind}/{rid}: {e}")
                if created["domain"]:
                    try:
                        client.delete(
                            f"/domains/{created['domain']}",
                            params={"force": True},
                            timeout=30.0,
                        )
                    except Exception as e:  # noqa: BLE001
                        log(f"  warning: failed to delete domain: {e}")
            elif created["inventory"]:
                log(
                    f"--keep: leaving inventory {created['inventory']} in domain {created['domain']}"
                )

    out_path = Path(args.out_dir) / f"bench_inplace_{args.label}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    log(f"Wrote {out_path}")

    print("\n=== Summary ===")
    print(
        f"label={results['label']}  partitions={results.get('num_partitions')}  rows={results.get('total_rows')}"
    )
    for name, s in results.get("scenarios", {}).items():
        st = s.get("warm_stats")
        if st:
            print(
                f"  {name:8} warm median={st['median']:.2f}s  "
                f"mean={st['mean']:.2f}s  min={st['min']:.2f}s  "
                f"max={st['max']:.2f}s  (cold={s['cold_rep']:.2f}s, n={st['n']})"
            )


if __name__ == "__main__":
    main()
