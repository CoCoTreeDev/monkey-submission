"""Structural and acceptance gate for a video-first rebuild manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import TRACK_MIN_HITS, TRACK_MIN_SECONDS  # noqa: E402
from track_policy import is_substantial_track  # noqa: E402


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

ISSUE_CASES = {
    "場域甲_visitbf20260827184032-015878_trail.mp4": 2,
    "場域甲_visitbf20260828070616-016069_trail.mp4": 1,
    "場域甲_tracks635+636_20260829_055759_d3f9af51_trail.mp4": 1,
}

ISSUE_1835 = "場域甲_visitbf20260827183447-015858_trail.mp4"


def validate(
    manifest: dict,
    reference: dict | None = None,
    expected: dict[str, int] | None = EXPECTED,
    check_issue_cases: bool = True,
) -> dict:
    failures: list[str] = []
    clips = manifest.get("clips", [])
    completed = {source for clip in clips for source in clip.get("reference_source_trails", [])}
    source_paths = [
        source
        for clip in clips
        for source in clip.get("reference_source_trails", [])
    ]
    if len(source_paths) != len(set(source_paths)):
        failures.append("reference source assigned to multiple canonical clips")
    output_paths = [clip.get("output_path") for clip in clips]
    if len(output_paths) != len(set(output_paths)):
        failures.append("duplicate output_path")
    poster_paths = [clip.get("poster_path") for clip in clips]
    if any(not path for path in poster_paths):
        failures.append("missing poster_path")
    if len([path for path in poster_paths if path]) != len({path for path in poster_paths if path}):
        failures.append("duplicate poster_path")
    if any(not clip.get("rawframes_path") for clip in clips):
        failures.append("missing rawframes_path")
    if any(clip.get("source_kind") != "rawframes" for clip in clips):
        failures.append("non-raw source in output")
    for clip in clips:
        if not clip.get("camera_id"):
            failures.append(f"{clip.get('clip_id')}: missing camera_id")
        if not clip.get("reference_source_trails"):
            failures.append(f"{clip.get('clip_id')}: missing Reference source")
        if clip.get("status") != "ready":
            failures.append(f"{clip.get('clip_id')}: not ready")
        if (clip.get("track_count") or 0) != len(clip.get("rows", [])):
            failures.append(f"{clip.get('clip_id')}: track_count/row_count mismatch")
        track_ids = [row.get("track_id") for row in clip.get("rows", [])]
        if any(track_id is None or track_id < 1 for track_id in track_ids):
            failures.append(f"{clip.get('clip_id')}: invalid review track id")
        if len(track_ids) != len(set(track_ids)):
            failures.append(f"{clip.get('clip_id')}: duplicate review track id")
        if (clip.get("source_frames") or 0) < 2:
            failures.append(f"{clip.get('clip_id')}: fewer than two source frames")
        duration = float(clip.get("duration_seconds") or 0)
        for row in clip.get("rows", []):
            start = row.get("clip_offset_seconds")
            end = row.get("clip_end_offset_seconds")
            if start is None or end is None or not (0 <= start <= end <= duration):
                failures.append(f"{clip.get('clip_id')}/{row.get('animal_id')}: invalid position")
            if not row.get("clip_segments"):
                failures.append(f"{clip.get('clip_id')}/{row.get('animal_id')}: missing segments")
            if not row.get("boxes"):
                failures.append(f"{clip.get('clip_id')}/{row.get('animal_id')}: missing per-frame boxes")
            else:
                boxes = row["boxes"]
                fps = float(row.get("track_box_fps") or clip.get("fps") or 0)
                span = (
                    (max(float(box[0]) for box in boxes) - min(float(box[0]) for box in boxes)) / fps
                    if fps > 0 else 0.0
                )
                if not is_substantial_track(
                    len(boxes), span,
                    min_hits=TRACK_MIN_HITS,
                    min_seconds=TRACK_MIN_SECONDS,
                ):
                    failures.append(
                        f"{clip.get('clip_id')}/{row.get('animal_id')}: short track "
                        f"({len(boxes)} hits over {span:.3f}s)"
                    )
            if not row.get("image_path") or not row.get("annotated_path"):
                failures.append(
                    f"{clip.get('clip_id')}/{row.get('animal_id')}: missing raw/annotated preview"
                )
            for box in row.get("boxes", []):
                if len(box) < 6 or int(box[0]) < 0 or int(box[0]) >= int(clip.get("source_frames") or 0):
                    failures.append(f"{clip.get('clip_id')}/{row.get('animal_id')}: box outside source")
                    break
                if not (box[1] >= 0 and box[2] >= 0 and box[3] > box[1] and box[4] > box[2]):
                    failures.append(f"{clip.get('clip_id')}/{row.get('animal_id')}: invalid box")
                    break

    for source, expected_count in (expected or {}).items():
        matches = [clip for clip in clips if source in clip.get("reference_source_trails", [])]
        if len(matches) != 1:
            failures.append(f"{source}: expected one canonical clip, got {len(matches)}")
        elif matches[0].get("track_count") != expected_count:
            failures.append(f"{source}: expected {expected_count} tracks, got {matches[0].get('track_count')}")

    # Named the rebuild validation design cases are separate from the user's eight spot checks so
    # the regression gate makes the original bug report executable as well.
    for source, expected_count in (ISSUE_CASES.items() if check_issue_cases else []):
        matches = [clip for clip in clips if source in clip.get("reference_source_trails", [])]
        if len(matches) != 1:
            failures.append(f"issue case {source}: expected one canonical clip, got {len(matches)}")
        elif matches[0].get("track_count") != expected_count:
            failures.append(f"issue case {source}: expected {expected_count} tracks, got {matches[0].get('track_count')}")

    if check_issue_cases:
        matches = [clip for clip in clips if ISSUE_1835 in clip.get("reference_source_trails", [])]
        if len(matches) != 1:
            failures.append(f"issue case {ISSUE_1835}: expected one canonical clip, got {len(matches)}")
        else:
            rows = matches[0].get("rows", [])
            if len(rows) < 2:
                failures.append(f"issue case {ISSUE_1835}: expected multiple animals")
            if not any(
                any(start <= 0.1 <= end for start, end in (row.get("clip_segments") or []))
                for row in rows
            ):
                failures.append(f"issue case {ISSUE_1835}: no animal at clip start")
            if not any(
                any(start <= 5.0 <= end for start, end in (row.get("clip_segments") or []))
                for row in rows
            ):
                failures.append(f"issue case {ISSUE_1835}: no animal around 5 seconds")

    if reference is not None:
        expected_sources = {
            source
            for clip in reference.get("canonical", [])
            for source in clip.get("source_trails", [])
        }
        if completed | {item.get("source_trail_path") for item in manifest.get("failures", [])} != expected_sources:
            failures.append("completed + failed sources do not cover Reference")
        if manifest.get("failed_count", len(manifest.get("failures", []))) != len(manifest.get("failures", [])):
            failures.append("failed_count mismatch")
        # Wall-clock collisions are audit warnings only.  The raw source is the
        # identity boundary and the producer contract says source clips are
        # independent/non-overlapping; only image evidence may merge them.
        if reference.get("review_required"):
            failures.append(f"{len(reference['review_required'])} unexpected unresolved review candidates")

    return {
        "clips": len(clips),
        "rows": sum(len(clip.get("rows", [])) for clip in clips),
        "zero_track_clips": sum(not clip.get("rows") for clip in clips),
        "failures": len(manifest.get("failures", [])),
        "structural_failures": failures,
        "apply_allowed": not failures and not manifest.get("failures"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    result = validate(
        json.loads(args.manifest.read_text()),
        json.loads(args.reference.read_text()) if args.reference else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result["structural_failures"] and result["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
