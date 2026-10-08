# /// script
# requires-python = ">=3.11"
# dependencies = ["google-cloud-logging>=3.10"]
# ///
"""Pull and cluster Cloud Run warnings/errors for the v2 prod services.

Produces a Markdown digest (and optionally JSON) of log entries at or above a
severity over a trailing window, grouped by normalized error signature. It does
no triage: deciding what is a bug and what is expected noise is left to the
reader (human or agent).

Auth modes (--auth):

- ``client`` (default): the google-cloud-logging client with ADC, or with a
  service-account key from GCP_SA_KEY_B64 / GCP_SA_KEY_JSON.
- ``rest``: plain HTTPS calls to logging.googleapis.com with no credentials of
  their own, for sandboxes whose egress proxy injects a bearer token (the Claude
  cloud "GCP access token" network secret). Set GCP_ACCESS_TOKEN to add the
  header yourself, e.g. ``GCP_ACCESS_TOKEN=$(gcloud auth print-access-token)``.

    uv run scripts/audit_logs.py --hours 24
    uv run scripts/audit_logs.py --hours 24 --auth rest --out digest.md
"""

import argparse
import base64
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

from google.cloud import logging as gcl
from google.oauth2 import service_account

DEFAULT_PROJECT = "silvx-fastfuels"
DEFAULT_SERVICES = [
    "api-v2-prod",
    "griddle-v2-prod",
    "exporter-v2-prod",
    "lakitu-v2-prod",
    "standgen-v2-prod",
    "treevox-v2-prod",
    "uploader-v2-prod",
    "etcher-v2-prod",
    "gdam-api-v2",
]
ID_KEYS = (
    "domain_id",
    "grid_id",
    "inventory_id",
    "feature_id",
    "export_id",
    "point_cloud_id",
    "pointcloud_id",
    "owner_id",
    "owner",
    "job_id",
    "task_id",
)

HEX32 = re.compile(r"\b[0-9a-f]{32}\b")
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
TEST_ID = re.compile(r"\btest-[A-Za-z0-9-]*[0-9a-f]{6,}\b")
NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?")
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\s]*")
FILE_LINE = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
EXC_LINE = re.compile(
    r"^(?:[A-Za-z_][\w.]*\.)?([A-Z][A-Za-z0-9_]*(?:Error|Exception|Exit|Interrupt|Warning))\b(.*)$"
)


@dataclass
class Entry:
    service: str
    severity: str
    timestamp: str
    message: str
    ids: dict
    http: dict | None
    traceback: dict | None
    is_test: bool


@dataclass
class Cluster:
    signature: str
    service: str
    severity: str
    count: int = 0
    test_count: int = 0
    first: str = ""
    last: str = ""
    example: str = ""
    traceback: dict | None = None
    sample_ids: list = field(default_factory=list)
    request_log: bool = False
    status: int | None = None


def credentials_from_env():
    raw = os.environ.get("GCP_SA_KEY_JSON")
    b64 = os.environ.get("GCP_SA_KEY_B64")
    if b64 and not raw:
        raw = base64.b64decode(b64).decode()
    if not raw:
        return None
    info = json.loads(raw)
    return service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/logging.read"]
    )


def build_filter(services, min_severity, start, end):
    svc = " OR ".join(f'resource.labels.service_name="{s}"' for s in services)
    return (
        f'resource.type="cloud_run_revision" AND ({svc}) '
        f"AND severity>={min_severity} "
        f'AND timestamp>="{start.isoformat()}" AND timestamp<"{end.isoformat()}"'
    )


def parse_traceback(msg):
    if "Traceback (most recent call last)" not in msg:
        return None
    frames = FILE_LINE.findall(msg)
    own = [
        f
        for f in frames
        if "site-packages" not in f[0] and "/usr/lib/python" not in f[0]
    ]
    top = (own or frames)[-1] if frames else None
    exc = None
    for line in reversed(msg.strip().splitlines()):
        m = EXC_LINE.match(line.strip())
        if m:
            exc = line.strip()[:300]
            break
    return {
        "exception": exc,
        "file": top[0] if top else None,
        "line": int(top[1]) if top else None,
        "function": top[2] if top else None,
    }


def normalize_path(path):
    path = HEX32.sub("{id}", path)
    path = UUID.sub("{id}", path)
    path = TEST_ID.sub("{id}", path)
    return re.sub(r"/\d+(?=/|$)", "/{n}", path)


def normalize_text(sig):
    sig = TIMESTAMP.sub("{ts}", sig)
    sig = HEX32.sub("{id}", sig)
    sig = UUID.sub("{id}", sig)
    sig = TEST_ID.sub("{id}", sig)
    sig = re.sub(r"gs://[^\s'\"\]]+", "gs://{path}", sig)
    sig = NUMBER.sub("{n}", sig)
    return re.sub(r"\s+", " ", sig)[:240]


def signature_for(msg, http, tb):
    if tb and tb["exception"]:
        sig = normalize_text(tb["exception"])
        if tb["file"]:
            sig += f"  @ {os.path.basename(tb['file'])}:{tb['line']}"
        return sig
    if http and not msg.strip():
        path = normalize_path(urlparse(http.get("requestUrl", "")).path or "/")
        return (
            f"HTTP {http.get('requestMethod', '?')} {path} -> {http.get('status', '?')}"
        )
    first = msg.strip().splitlines()[0] if msg.strip() else "(empty)"
    sig = normalize_text(first)
    if http:
        sig = f"HTTP {http.get('status', '?')}: {sig}"
    return sig


def is_test_traffic(msg, ids, http):
    if any(v == "test-owner" or v.startswith("test-") for v in ids.values()):
        return True
    if TEST_ID.search(msg):
        return True
    if http and TEST_ID.search(http.get("requestUrl", "")):
        return True
    return False


def iter_rest(project, flt, page_size=1000):
    url = "https://logging.googleapis.com/v2/entries:list"
    token = os.environ.get("GCP_ACCESS_TOKEN")
    page_token = None
    while True:
        body = {
            "resourceNames": [f"projects/{project}"],
            "filter": flt,
            "orderBy": "timestamp desc",
            "pageSize": page_size,
        }
        if page_token:
            body["pageToken"] = page_token
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        yield from data.get("entries", [])
        page_token = data.get("nextPageToken")
        if not page_token:
            return


def iter_client(client, flt, page_size=1000):
    for e in client.list_entries(
        filter_=flt, order_by=gcl.DESCENDING, page_size=page_size
    ):
        raw = {
            "severity": str(e.severity),
            "timestamp": e.timestamp.isoformat() if e.timestamp else "",
            "resource": {"labels": dict(e.resource.labels)},
        }
        if e.http_request:
            raw["httpRequest"] = dict(e.http_request)
        if isinstance(e.payload, str):
            raw["textPayload"] = e.payload
        elif isinstance(e.payload, dict):
            raw["jsonPayload"] = e.payload
        elif e.payload is not None:
            raw["protoPayload"] = e.payload
        yield raw


def extract_message(raw):
    if "textPayload" in raw:
        return raw["textPayload"]
    payload = raw.get("jsonPayload")
    if isinstance(payload, dict):
        for k in ("message", "msg", "error", "exception"):
            v = payload.get(k)
            if isinstance(v, str) and v:
                return v
        return json.dumps(payload, sort_keys=True)[:2000]
    proto = raw.get("protoPayload")
    if proto is not None:
        return str(proto)[:2000]
    return ""


def extract_ids(raw):
    ids = {}
    payload = raw.get("jsonPayload")
    if isinstance(payload, dict):
        for k in ID_KEYS:
            v = payload.get(k)
            if isinstance(v, (str, int)):
                ids[k] = str(v)
    return ids


def to_entry(raw):
    http = None
    hr = raw.get("httpRequest")
    if hr:
        http = {
            k: hr.get(k)
            for k in (
                "requestMethod",
                "requestUrl",
                "status",
                "userAgent",
                "latency",
                "remoteIp",
            )
            if hr.get(k) is not None
        }
    msg = extract_message(raw)
    ids = extract_ids(raw)
    tb = parse_traceback(msg)
    return Entry(
        service=raw.get("resource", {}).get("labels", {}).get("service_name", "?"),
        severity=str(raw.get("severity", "DEFAULT")),
        timestamp=raw.get("timestamp", ""),
        message=msg[:4000],
        ids=ids,
        http=http,
        traceback=tb,
        is_test=is_test_traffic(msg, ids, http),
    )


def fetch(raw_iter, limit):
    out = []
    for raw in raw_iter:
        out.append(to_entry(raw))
        if len(out) >= limit:
            break
    return out


def cluster(entries):
    clusters = {}
    for en in entries:
        sig = signature_for(en.message, en.http, en.traceback)
        key = (en.service, en.severity, sig)
        c = clusters.get(key)
        if c is None:
            c = clusters[key] = Cluster(
                signature=sig,
                service=en.service,
                severity=en.severity,
                first=en.timestamp,
                last=en.timestamp,
                example=(en.message or json.dumps(en.http))[:600],
                traceback=en.traceback,
                request_log=bool(en.http) and not en.message.strip(),
                status=en.http.get("status") if en.http else None,
            )
        c.count += 1
        c.test_count += en.is_test
        c.first = min(c.first, en.timestamp)
        c.last = max(c.last, en.timestamp)
        if len(c.sample_ids) < 3 and en.ids:
            c.sample_ids.append(en.ids)
    return sorted(
        clusters.values(), key=lambda c: (severity_rank(c.severity), -c.count)
    )


def severity_rank(s):
    order = [
        "EMERGENCY",
        "ALERT",
        "CRITICAL",
        "ERROR",
        "WARNING",
        "NOTICE",
        "INFO",
        "DEBUG",
        "DEFAULT",
    ]
    return order.index(s) if s in order else len(order)


def render_markdown(entries, clusters, services, start, end, truncated):
    lines = []
    lines.append(
        f"# Cloud Run log digest {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC"
    )
    lines.append("")
    lines.append(
        f"Entries: {len(entries)}"
        + ("  (TRUNCATED at --limit; counts are lower bounds)" if truncated else "")
    )
    lines.append("")
    lines.append("## Per-service counts")
    lines.append("")
    lines.append("| service | ERROR+ | WARNING | test-tagged | clusters |")
    lines.append("|---|---|---|---|---|")
    by_svc = defaultdict(list)
    for en in entries:
        by_svc[en.service].append(en)
    for s in services:
        ens = by_svc.get(s, [])
        err = sum(severity_rank(e.severity) <= severity_rank("ERROR") for e in ens)
        warn = sum(e.severity == "WARNING" for e in ens)
        test = sum(e.is_test for e in ens)
        ncl = sum(c.service == s for c in clusters)
        lines.append(f"| {s} | {err} | {warn} | {test} | {ncl} |")
    lines.append("")
    status_counts = Counter(
        (en.service, en.http.get("status"))
        for en in entries
        if en.http and en.http.get("status")
    )
    if status_counts:
        lines.append("## HTTP status counts (request logs)")
        lines.append("")
        lines.append("| service | status | count |")
        lines.append("|---|---|---|")
        for (s, st), n in sorted(
            status_counts.items(), key=lambda kv: (kv[0][0], -kv[1])
        ):
            lines.append(f"| {s} | {st} | {n} |")
        lines.append("")
    for s in services:
        cl = [c for c in clusters if c.service == s]
        if not cl:
            continue
        lines.append(f"## {s}")
        lines.append("")
        req = [c for c in cl if c.request_log]
        if req:
            lines.append("### Request log (no app message; status is the only signal)")
            lines.append("")
            lines.append("| count | test-tagged | signature | first | last |")
            lines.append("|---|---|---|---|---|")
            for c in sorted(req, key=lambda c: (-(c.status or 0) // 100, -c.count)):
                lines.append(
                    f"| {c.count} | {c.test_count} | `{c.signature}` | {c.first[:19]} | {c.last[:19]} |"
                )
            lines.append("")
        for c in cl:
            if c.request_log:
                continue
            test_note = f", {c.test_count} test-tagged" if c.test_count else ""
            lines.append(f"### [{c.severity}] x{c.count}{test_note}: `{c.signature}`")
            lines.append("")
            lines.append(f"- first: {c.first}  last: {c.last}")
            if c.traceback and c.traceback.get("file"):
                tb = c.traceback
                lines.append(
                    f"- traceback top frame: `{tb['file']}:{tb['line']}` in `{tb['function']}`"
                )
            if c.sample_ids:
                lines.append(f"- sample ids: `{json.dumps(c.sample_ids)}`")
            lines.append("- example:")
            lines.append("")
            lines.append("```")
            lines.append(c.example.rstrip())
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--services", nargs="+", default=DEFAULT_SERVICES)
    p.add_argument("--hours", type=float, default=24.0, help="trailing window length")
    p.add_argument("--end", help="window end, ISO-8601 UTC (default: now)")
    p.add_argument("--min-severity", default="WARNING")
    p.add_argument("--limit", type=int, default=20000, help="max entries to fetch")
    p.add_argument(
        "--auth",
        choices=["client", "rest"],
        default="client",
        help="client: google-cloud-logging with ADC or GCP_SA_KEY_*; rest: bare HTTPS, token injected by a proxy or GCP_ACCESS_TOKEN",
    )
    p.add_argument("--out", help="write Markdown digest here (default: stdout)")
    p.add_argument(
        "--json", dest="json_out", help="also write clusters + entries as JSON"
    )
    args = p.parse_args(argv)

    end = (
        datetime.fromisoformat(args.end).astimezone(UTC)
        if args.end
        else datetime.now(UTC)
    )
    start = end - timedelta(hours=args.hours)
    flt = build_filter(args.services, args.min_severity, start, end)
    if args.auth == "rest":
        raw_iter = iter_rest(args.project, flt)
    else:
        creds = credentials_from_env()
        client = (
            gcl.Client(project=args.project, credentials=creds)
            if creds
            else gcl.Client(project=args.project)
        )
        raw_iter = iter_client(client, flt)
    entries = fetch(raw_iter, args.limit)
    truncated = len(entries) >= args.limit
    clusters = cluster(entries)
    md = render_markdown(entries, clusters, args.services, start, end, truncated)

    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
    else:
        sys.stdout.write(md)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "window": {"start": start.isoformat(), "end": end.isoformat()},
                    "filter": flt,
                    "truncated": truncated,
                    "clusters": [asdict(c) for c in clusters],
                    "entries": [asdict(e) for e in entries],
                },
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
