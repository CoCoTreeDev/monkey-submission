"""Apply already-validated Reference media without rereading raw archives.

The expensive archive render is deliberately separate from this commit step.
The apply phase only uploads the small crop artefacts, then points each matched
production row at the validated raw/annotated/crop triplet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "backend"))
from services.storage import storage  # noqa: E402


def request_json(api: str, token: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api.rstrip('/')}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def fetch_detections(api: str, token: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = request_json(
            api, token,
            f"/api/detections?limit=200&offset={offset}&sort_by=time&sort_order=asc",
        )
        items = page.get("items") or []
        rows.extend(items)
        offset += len(items)
        if not items or offset >= int(page.get("total") or 0):
            return rows


def post_media(api: str, token: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api.rstrip('/')}/api/admin/detections/media-bulk",
        data=json.dumps({"items": items}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api", default=os.environ.get("API", ""))
    parser.add_argument("--token", default=os.environ.get("TOKEN", ""))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.api or not args.token:
        raise SystemExit("API and TOKEN are required")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reference_keys = {
        (Path(clip.get("output_path") or "").name, row.get("animal_id"))
        for clip in manifest.get("clips", [])
        for row in clip.get("rows", [])
    }
    rows = fetch_detections(args.api, args.token)
    matched = [
        row for row in rows
        if (Path(row.get("trail_path") or "").name, row.get("animal_id"))
        in reference_keys
    ]
    if len(matched) != len(reference_keys):
        raise SystemExit(
            f"Reference/production match incomplete: {len(matched)}/{len(reference_keys)}"
        )

    media_items: list[dict[str, Any]] = []
    pending_items: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in matched:
        prefix = f"rebuild-media-{int(row['id'])}"
        names = {
            "image_path": f"{prefix}_raw.jpg",
            "annotated_path": f"{prefix}_annotated.jpg",
            "crop_path": f"{prefix}_crop.jpg",
        }
        for name in names.values():
            if not (args.output / name).is_file():
                missing.append(name)
        item = {"detection_id": int(row["id"]), **names}
        media_items.append(item)
        if any(row.get(field) != value for field, value in names.items()):
            pending_items.append(item)
    if missing:
        raise SystemExit(f"validated output missing {len(missing)} files: {missing[:8]}")

    print(json.dumps({
        "production_rows": len(rows),
        "reference_rows": len(reference_keys),
        "matched_rows": len(matched),
        "pending_rows": len(pending_items),
        "apply": args.apply,
        "output": str(args.output),
    }, ensure_ascii=False), flush=True)
    if not args.apply:
        return 0

    for index, item in enumerate(pending_items, 1):
        for field in ("image_path", "annotated_path", "crop_path"):
            name = item[field]
            storage.upload(name, (args.output / name).read_bytes())
        if index % 100 == 0 or index == len(pending_items):
            print(f"uploaded media {index}/{len(pending_items)}", flush=True)

    updated = 0
    for start in range(0, len(pending_items), 200):
        batch = pending_items[start:start + 200]
        result = post_media(args.api, args.token, batch)
        if result.get("updated") != len(batch):
            raise SystemExit(f"media update count mismatch: {result}")
        updated += len(batch)
        print(f"updated rows {updated}/{len(media_items)}", flush=True)
    print(json.dumps({"updated": updated, "skipped": len(media_items) - len(pending_items)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
