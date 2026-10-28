"""Prepare an isolated historical rebuild run.

This is deliberately a preparation step, not a migration.  It snapshots the
inputs and refuses to produce an apply-ready manifest when an input archive is
missing.  The renderer can then consume the manifest without silently falling
back to old retrack/offset artifacts.

Usage on the mini:

    uv run --project ~/Monkey-Detection/backend python \
      tests/prepare_full_rebuild.py --base ~/rawframes-backfill
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_name(trail_path: str) -> str | None:
    stem, marker, _ = trail_path.rpartition("_trail.")
    return f"{stem}_rawframes.zip" if marker else None


def git_revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path.home() / "rawframes-backfill")
    parser.add_argument("--repo", type=Path, default=Path.home() / "Monkey-Detection")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = args.model or args.repo / "backend" / "best.pt"
    rows_path = args.base / "detections.json"
    archives = args.base / "zips"
    if not model.is_file():
        raise SystemExit(f"model missing: {model}")
    if not rows_path.is_file():
        raise SystemExit(f"snapshot missing: {rows_path}")
    if not archives.is_dir():
        raise SystemExit(f"archive directory missing: {archives}")

    rows = json.loads(rows_path.read_text())
    by_clip: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        trail = row.get("trail_path") or ""
        if trail:
            by_clip[trail].append(row["id"])

    missing: list[dict[str, object]] = []
    clips: list[dict[str, object]] = []
    for trail, detection_ids in sorted(by_clip.items()):
        archive = archive_name(trail)
        archive_present = bool(archive and (archives / archive).is_file())
        original_present = (args.base / "originals" / trail).is_file()
        present = archive_present or original_present
        if not present:
            missing.append({"trail_path": trail, "archive": archive, "detection_ids": detection_ids})
        clips.append({
            "trail_path": trail,
            "archive": archive,
            "detection_ids": detection_ids,
            "archive_present": archive_present,
            "original_present": original_present,
            "source": "archive" if archive_present else "original" if original_present else None,
        })

    run_id = datetime.now(timezone.utc).strftime("rebuild-%Y%m%dT%H%M%SZ")
    output = args.output or args.base / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "rows_file": str(rows_path),
            "row_count": len(rows),
            "clip_count": len(clips),
            "model": str(model),
            "model_sha256": sha256(model),
            "repo_revision": git_revision(args.repo),
            "imgsz": args.imgsz,
            "device": "mps",
            "tracker": str(args.repo / "backend" / "trackers" / "bytetrack-monkey.yaml"),
        },
        "clips": clips,
        "validation": {
            "missing_sources": len(missing),
            "all_inputs_present": not missing,
            "old_artifacts_are_not_inputs": True,
            "apply_allowed": False,
        },
        "missing": missing,
    }
    (output / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (output / "rows.snapshot.json").write_text(json.dumps(rows, ensure_ascii=False))
    print(json.dumps({
        "run_id": run_id,
        "output": str(output),
        "rows": len(rows),
        "clips": len(clips),
        "missing_sources": len(missing),
        "model_sha256": manifest["source"]["model_sha256"],
        "apply_allowed": False,
    }, ensure_ascii=False))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
