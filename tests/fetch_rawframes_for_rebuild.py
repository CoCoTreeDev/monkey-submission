"""Fetch the raw-frame archives missing from an isolated rebuild input.

The renderer must not silently fall back to an annotated trail clip: that would
make the model run cover a different set of frames from the one the review
claims to cover. This helper fills ``<base>/zips`` through the authenticated
raw-frame endpoint and keeps a checkpoint, so an interrupted transfer can be
resumed safely.

Usage on the mini (the token is intentionally supplied through the
environment, never written into the manifest):

    API=https://... TOKEN=... uv run --project ~/Monkey-Detection/backend \
      python tests/fetch_rawframes_for_rebuild.py \
      --manifest ~/rawframes-backfill/rebuild-.../input_manifest.json \
      --base ~/rawframes-backfill
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


def archive_name(trail_path: str) -> str | None:
    stem, marker, _ = trail_path.rpartition("_trail.")
    return f"{stem}_rawframes.zip" if marker else None


def fetch(api: str, token: str, detection_id: int, destination: Path) -> None:
    url = f"{api.rstrip('/')}/api/detections/{detection_id}/frames"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        content = response.read()
    if not content.startswith(b"PK"):
        raise RuntimeError(f"endpoint returned non-ZIP content ({len(content)} bytes)")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if not any(name.lower().endswith((".jpg", ".jpeg")) for name in archive.namelist()):
            raise RuntimeError("ZIP contains no JPEG frames")
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"ZIP contains a corrupt member: {corrupt}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part",
                                     dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--api", default=os.environ.get("API", ""))
    parser.add_argument("--token", default=os.environ.get("TOKEN", ""))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-failed", action="store_true")
    args = parser.parse_args()
    if not args.api or not args.token:
        raise SystemExit("API and TOKEN are required (use environment variables)")

    manifest = json.loads(args.manifest.read_text())
    checkpoint_path = args.manifest.parent / "rawframes_fetch.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {
        "downloaded": [], "failures": []
    }
    downloaded = set(checkpoint.get("downloaded", []))
    failed = {item["trail_path"]: item for item in checkpoint.get("failures", [])}
    clips = [item for item in manifest["clips"] if item.get("archive")]
    missing = [item for item in clips if not (args.base / "zips" / item["archive"]).is_file()]
    if args.limit:
        missing = missing[:args.limit]

    def save_checkpoint() -> None:
        checkpoint_path.write_text(json.dumps({
            "downloaded": sorted(downloaded),
            "failures": sorted(failed.values(), key=lambda item: item["trail_path"]),
        }, ensure_ascii=False, indent=2))

    print(f"需要抓取 {len(missing)} 支 raw-frame archive", flush=True)
    for number, item in enumerate(missing, 1):
        trail = item["trail_path"]
        archive = item["archive"] or archive_name(trail)
        try:
            # The older full-rebuild manifest called these detection_ids;
            # Reference calls them row_ids. Both are provenance only: the
            # downloaded archive is the actual model input.
            ids = item.get("detection_ids") or item.get("row_ids") or []
            if not ids:
                raise RuntimeError("source has no reference row id")
            fetch(args.api, args.token, int(ids[0]), args.base / "zips" / archive)
            downloaded.add(trail)
            failed.pop(trail, None)
            print(f"[{number}/{len(missing)}] {trail}", flush=True)
        except (OSError, RuntimeError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            failure = {"trail_path": trail, "archive": archive, "error": str(exc)}
            failed[trail] = failure
            print(f"[failed] {trail}: {exc}", flush=True)
            if not args.skip_failed:
                save_checkpoint()
                raise
        save_checkpoint()

    print(json.dumps({"downloaded": len(downloaded), "failed": len(failed),
                      "checkpoint": str(checkpoint_path)}, ensure_ascii=False))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
