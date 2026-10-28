"""Validate the human-reviewed acceptance cases for a rebuild manifest.

This is intentionally separate from the renderer: it is a small, repeatable
gate for the clips the operator called out in the review. It prevents a new
run from being published merely because it has a larger row count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "場域丙_visitbf20260818090735-011320_trail.mp4": 1,
    "場域乙_tracks433_20260814_131416_866516b4_trail.mp4": 1,
    "場域乙_tracks435_20260814_131428_7dd43221_trail.mp4": 1,
    "場域乙_tracks5388_20260825_101237_06c32b16_trail.mp4": 2,
    "場域乙_tracks5396_20260825_101919_255236d5_trail.mp4": 1,
    "場域乙_tracks5402_20260825_102024_90fd6ff0_trail.mp4": 1,
    "場域乙_tracks5406_20260825_102215_a754010c_trail.mp4": 1,
    "場域乙_visitbf20260825101124-014791_trail.mp4": 3,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    by_source = {clip.get("source_trail_path"): clip for clip in data.get("clips", [])}
    failures = []
    for source, expected in EXPECTED.items():
        clip = by_source.get(source)
        if clip is None:
            failures.append(f"missing clip: {source}")
            continue
        actual = clip.get("track_count")
        if actual is None:
            actual = len(clip.get("rows") or [])
        if actual != expected:
            failures.append(f"{source}: expected {expected} animals, got {actual}")
        if clip.get("source_kind") != "rawframes":
            failures.append(f"{source}: source is {clip.get('source_kind')!r}, not rawframes")
    print(json.dumps({"checked": len(EXPECTED), "failures": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
