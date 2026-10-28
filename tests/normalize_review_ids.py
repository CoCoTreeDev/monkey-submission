"""Make review-facing track ids match the labels burned into rebuilt MP4s."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize(manifest: dict) -> dict:
    for clip in manifest.get("clips", []):
        for display_id, row in enumerate(clip.get("rows", []), 1):
            row.setdefault("source_track_id", row.get("track_id"))
            row["track_id"] = display_id
    manifest["review_id_scheme"] = "canonical_clip_sequence_v1"
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = normalize(json.loads(args.manifest.read_text()))
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({
        "clips": len(manifest.get("clips", [])),
        "rows": sum(len(clip.get("rows", [])) for clip in manifest.get("clips", [])),
        "review_id_scheme": manifest["review_id_scheme"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
