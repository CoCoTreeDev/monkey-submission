#!/usr/bin/env python3
"""Poll every GET endpoint; Discord-alert when one changes state.

Runs separately from the API it watches, for example under launchd — a
canary that lives inside the thing it monitors is silent exactly when it
matters. Route list comes from /openapi.json so it cannot go stale.

    uv run python infra/canary.py           # one pass, exit 0 if all healthy
    uv run python infra/canary.py --dry-run # no Discord, print the table
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import httpx
from urllib.parse import quote

ENV = pathlib.Path(__file__).resolve().parent.parent / "backend" / ".env"
STATE = pathlib.Path(os.path.expanduser("~/logs/canary-state.json"))
TIMEOUT = 30.0
SLOW_SECONDS = 10.0

# Endpoints that do real work rather than report it: download-images builds a
# zip of every snapshot. Polling that every ten minutes would make the canary
# the outage.
# Endpoints that do real work rather than report it, and would make the canary
# the outage: download-images zips every snapshot, and
# /api/detections/{id}/frames builds a raw-frame archive — measured at 38 s for
# one detection, because those ZIPs average 41 MB. Their storage path is
# covered cheaply by /api/snapshots/{filename} instead.
SKIP = {"/api/admin/download-images", "/api/detections/{detection_id}/frames"}

# Routes whose "failure" is a legitimate empty state. rebuild-review 404s with
# "目前沒有可審查的回填 run" whenever no backfill is pending, which is most of
# the time — alerting on it would teach you to ignore the channel.
ALSO_OK = {"/api/admin/rebuild-review": {404}}


def env(key: str) -> str:
    for line in ENV.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _sampled_paths(client: httpx.Client, base: str, hdr: dict) -> list[tuple[str, str]]:
    """[(label, url_path)] for routes that need a real id substituted in.

    Without these the canary never touches the storage layer, which is exactly
    what a storage migration can break: a missing snapshot path
    would look perfectly healthy from the plain routes alone. MJPEG is left
    out on purpose — those responses never end.
    """
    out: list[tuple[str, str]] = []
    try:
        page = client.get(f"{base}/api/detections?limit=1", headers=hdr, timeout=TIMEOUT).json()
        items = page.get("items") or page.get("detections") or []
        if items:
            name = items[0].get("image_path") or items[0].get("annotated_path")
            if name:
                out.append(("/api/snapshots/{filename}",
                            f"/api/snapshots/{quote(str(name))}"))
    except Exception:
        # A failure here is already reported by the plain /api/detections probe.
        pass
    return out


def probe(client: httpx.Client, base: str, token: str) -> dict[str, str]:
    """Return {route: "ok" | "<failure reason>"} for every checkable GET route."""
    spec = client.get(f"{base}/openapi.json", timeout=TIMEOUT).json()
    hdr = {"Authorization": f"Bearer {token}"}
    results: dict[str, str] = {}

    def check(label: str, path: str) -> None:
        started = time.monotonic()
        try:
            r = client.get(f"{base}{path}", headers=hdr, timeout=TIMEOUT)
            elapsed = time.monotonic() - started
            if r.status_code >= 400 and r.status_code not in ALSO_OK.get(label, ()):
                results[label] = f"HTTP {r.status_code}"
            elif elapsed > SLOW_SECONDS:
                # Slow is the symptom that precedes "Failed to fetch": on
                # 2026-09-20 the API was CPU-starved for days and answered
                # eventually, so a status-only check would have stayed green.
                results[label] = f"slow {elapsed:.0f}s"
            else:
                results[label] = "ok"
        except Exception as exc:
            results[label] = type(exc).__name__

    for path in sorted(
        p for p, ops in spec["paths"].items()
        if "get" in ops and "{" not in p and p not in SKIP
    ):
        check(path, path)

    # Only once the listing itself is healthy, so a dead /api/detections is one
    # failure rather than three.
    if results.get("/api/detections") == "ok":
        for label, path in _sampled_paths(client, base, hdr):
            check(label, path)
    return results


def notify(webhook: str, lines: list[str]) -> None:
    # python-urllib gets a 403 from Discord; httpx does not.
    httpx.post(webhook, json={"content": "\n".join(lines)[:1900]}, timeout=20.0)


def main() -> int:
    dry = "--dry-run" in sys.argv
    base = env("CLOUD_API_URL").rstrip("/")
    code = env("CLOUD_ACCESS_CODE")
    webhook = env("CANARY_DISCORD_WEBHOOK")
    if not base or not code:
        print("CLOUD_API_URL / CLOUD_ACCESS_CODE missing from backend/.env", file=sys.stderr)
        return 2

    with httpx.Client(follow_redirects=True) as client:
        try:
            token = client.post(f"{base}/api/auth/login", json={"code": code},
                                timeout=TIMEOUT).json()["token"]
        except Exception as exc:
            results = {"/api/auth/login": type(exc).__name__}
        else:
            results = probe(client, base, token)

        broken = {p: why for p, why in results.items() if why != "ok"}
        for path, why in sorted(results.items()):
            print(f"  {'ok  ' if why == 'ok' else 'FAIL'}  {path:44s} {why}")
        print(f"{len(results) - len(broken)}/{len(results)} healthy")

        # Alert on transitions only. A ten-minute poll through a six-hour
        # outage is 36 identical pings, which trains you to ignore it.
        was = set()
        if STATE.exists():
            was = set(json.loads(STATE.read_text()).get("broken", []))
        now = set(broken)
        if not dry:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps({"broken": sorted(now), "at": time.time()}))

        lines = [f"🔴 `{p}` — {broken[p]}" for p in sorted(now - was)]
        lines += [f"🟢 `{p}` recovered" for p in sorted(was - now)]
        if lines and webhook and not dry:
            notify(webhook, [f"**API canary** ({base})", *lines])
        elif lines:
            print("would notify:", *lines, sep="\n  ")

    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
