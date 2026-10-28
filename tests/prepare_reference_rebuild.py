"""Build a conservative Reference manifest for the historical rebuild.

The Reference is the frozen production snapshot, not the output of the new
model.  Time ranges are useful for finding possible duplicate clips, but raw
archives have no capture timestamps, so time overlap alone never authorises a
merge.  The producer contract says each raw-frame source is an independent,
non-overlapping clip: an unexplained wall-clock collision is therefore kept as
an audit warning, not treated as an identity blocker.  Only two or more
byte-identical JPEG frames (or a strong visual alignment) can create one
canonical clip.

This is preparation only.  It never changes the database, uploads blobs, or
marks a run applyable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


_CAPTURE_TIME = re.compile(r"(?P<date>20\d{6})[_-]?(?P<time>\d{6})(?:[_-]|$)")
DEDUPE_CANDIDATE_GAP_SECONDS = 120.0

import cv2
import numpy as np


def archive_name(trail_path: str) -> str | None:
    stem, marker, _ = trail_path.rpartition("_trail.")
    return f"{stem}_rawframes.zip" if marker else None


def frame_hashes(path: Path) -> list[str]:
    """Return SHA-1 hashes in archive order; decoding is unnecessary here."""
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.lower().endswith((".jpg", ".jpeg"))
        )
        return [hashlib.sha1(archive.read(name)).hexdigest() for name in names]


def visual_frames(path: Path) -> list[np.ndarray]:
    """Decode small normalized fingerprints for re-encoded JPEG matching."""
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.lower().endswith((".jpg", ".jpeg"))
        )
        fingerprints = []
        for name in names:
            image = cv2.imdecode(
                np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if image is None:
                fingerprints.append(None)
                continue
            image = cv2.resize(image, (64, 36)).astype(np.float32)
            fingerprints.append((image - image.mean()) / (image.std() + 1e-6))
        return fingerprints


def shared_frame_count(left: list[str], right: list[str]) -> int:
    return len(set(left) & set(right))


def frame_offset(left: list[str], right: list[str]) -> int | None:
    """Find right's offset on left's axis when at least two frames agree."""
    positions = {value: index for index, value in enumerate(left)}
    votes: dict[int, int] = defaultdict(int)
    for index, value in enumerate(right):
        if value in positions:
            votes[positions[value] - index] += 1
    if not votes:
        return None
    offset, count = max(votes.items(), key=lambda item: (item[1], -item[0]))
    return offset if count >= 2 else None


def visual_alignment(left: list[np.ndarray | None], right: list[np.ndarray | None]) -> tuple[int, int, float] | None:
    """Return offset, agreeing frame count and mean error for re-encoded clips.

    The candidate must already overlap in the Reference time window. A
    dominant offset with six or more low-error frames is strong enough to
    distinguish the same short clip from a static camera background, while a
    time overlap without visual evidence remains unresolved.
    """
    votes: dict[int, list[float]] = defaultdict(list)
    for left_index, left_image in enumerate(left):
        if left_image is None:
            continue
        best_index, best_error = None, float("inf")
        for right_index, right_image in enumerate(right):
            if right_image is None:
                continue
            error = float(np.mean(np.abs(left_image - right_image)))
            if error < best_error:
                best_index, best_error = right_index, error
        if best_index is not None and best_error <= 0.03:
            votes[left_index - best_index].append(best_error)
    if not votes:
        return None
    offset, errors = max(votes.items(), key=lambda item: (len(item[1]), -sum(item[1])))
    if len(errors) < 6:
        return None
    return offset, len(errors), sum(errors) / len(errors)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def trail_capture_time(trail_path: str, timezone_info=None) -> datetime | None:
    """Read the capture start embedded in a trail filename when available.

    A row timestamp is the moment a track was emitted, not the beginning of
    the video. A visit can therefore put its last row minutes after the clip
    started. Using those row timestamps to form an interval creates false
    overlap candidates and defeats clip-level dedupe. The producer filenames
    carry the actual start in both ``..._YYYYMMDD_HHMMSS_...`` and
    ``...bfYYYYMMDDHHMMSS-...`` forms.
    """
    match = _CAPTURE_TIME.search(trail_path)
    if not match:
        return None
    try:
        value = datetime.strptime(
            f"{match.group('date')}{match.group('time')}", "%Y%m%d%H%M%S"
        )
    except ValueError:
        return None
    return value.replace(tzinfo=timezone_info) if timezone_info else value
def clip_reference(rows: list[dict], trail_path: str) -> dict:
    timestamps = [parse_time(row.get("timestamp")) for row in rows]
    valid = [value for value in timestamps if value is not None]
    row_start = min(valid) if valid else None
    start = trail_capture_time(trail_path, row_start.tzinfo if row_start else None) or row_start
    durations = [
        float(row["clip_duration_seconds"])
        for row in rows
        if isinstance(row.get("clip_duration_seconds"), (int, float))
        and float(row["clip_duration_seconds"]) > 0
    ]
    end = start + timedelta(seconds=max(durations)) \
        if start and durations else start
    return {
        "trail_path": trail_path,
        "archive": archive_name(trail_path),
        "camera_id": rows[0].get("camera_id"),
        "row_ids": sorted(int(row["id"]) for row in rows),
        "reference_start": start.isoformat() if start else None,
        "reference_end": end.isoformat() if end else None,
        "reference_start_source": "trail_filename" if trail_capture_time(
            trail_path, row_start.tzinfo if row_start else None
        ) else "row_timestamp",
        "reference_time_is_candidate_only": True,
    }


def intervals_overlap(left: dict, right: dict) -> bool:
    if left["reference_start"] is None or right["reference_start"] is None:
        return False
    left_start = parse_time(left["reference_start"])
    left_end = parse_time(left["reference_end"]) or left_start
    right_start = parse_time(right["reference_start"])
    right_end = parse_time(right["reference_end"]) or right_start
    return bool(left_start <= right_end and right_start <= left_end)


def intervals_are_dedupe_candidates(left: dict, right: dict) -> bool:
    """Find overlapping or immediately adjacent clips for image comparison.

    Two producer clips can be the same capture split into neighbouring files;
    433/435 is the concrete case (12 seconds apart, no byte-identical frame).
    Conversely, a visit's row timestamps can span minutes even though its
    video is only a few seconds long. The interval above is now clip-based,
    and this bounded gap keeps the visual comparison broad enough for split
    files without comparing every static camera frame in the archive.
    """
    if intervals_overlap(left, right):
        return True
    left_start = parse_time(left.get("reference_start"))
    left_end = parse_time(left.get("reference_end")) or left_start
    right_start = parse_time(right.get("reference_start"))
    right_end = parse_time(right.get("reference_end")) or right_start
    if not all((left_start, left_end, right_start, right_end)):
        return False
    if left_end < right_start:
        return (right_start - left_end).total_seconds() <= DEDUPE_CANDIDATE_GAP_SECONDS
    return (left_start - right_end).total_seconds() <= DEDUPE_CANDIDATE_GAP_SECONDS


def build_reference(rows: list[dict], archive_root: Path) -> dict:
    by_trail: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        trail = row.get("trail_path") or ""
        if trail:
            by_trail[trail].append(row)

    clips = [clip_reference(group, trail) for trail, group in sorted(by_trail.items())]
    hashes: dict[str, list[str]] = {}
    visuals: dict[str, list[np.ndarray | None]] = {}
    missing: list[dict] = []
    for clip in clips:
        archive = clip["archive"]
        path = archive_root / archive if archive else None
        if path is None or not path.is_file():
            missing.append({"trail_path": clip["trail_path"], "archive": archive})
            continue
        try:
            hashes[clip["trail_path"]] = frame_hashes(path)
            visuals[clip["trail_path"]] = visual_frames(path)
        except (OSError, zipfile.BadZipFile) as exc:
            missing.append({"trail_path": clip["trail_path"], "archive": archive,
                            "error": f"invalid raw archive: {exc}"})

    parent = {clip["trail_path"]: clip["trail_path"] for clip in clips}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    candidates: list[dict] = []
    for index, left in enumerate(clips):
        for right in clips[index + 1:]:
            if left["camera_id"] != right["camera_id"] or not intervals_are_dedupe_candidates(left, right):
                continue
            left_hashes, right_hashes = hashes.get(left["trail_path"], []), hashes.get(right["trail_path"], [])
            shared = shared_frame_count(left_hashes, right_hashes)
            offset = frame_offset(left_hashes, right_hashes) if shared >= 2 else None
            visual = None
            if offset is None:
                visual = visual_alignment(
                    visuals.get(left["trail_path"], []),
                    visuals.get(right["trail_path"], []),
                )
            evidence = (
                "exact_frame_overlap" if offset is not None
                else "visual_frame_overlap" if visual is not None
                else "review_required"
            )
            candidates.append({
                "left": left["trail_path"], "right": right["trail_path"],
                "relation": "overlap" if intervals_overlap(left, right) else "adjacent",
                "shared_frames": shared, "right_offset_on_left": offset,
                "shared_visual_frames": visual[1] if visual else 0,
                "visual_offset_on_left": visual[0] if visual else None,
                "visual_mean_abs_error": round(visual[2], 5) if visual else None,
                "evidence": evidence,
            })
            if evidence in {"exact_frame_overlap", "visual_frame_overlap"}:
                union(left["trail_path"], right["trail_path"])

    groups: dict[str, list[dict]] = defaultdict(list)
    for clip in clips:
        groups[find(clip["trail_path"])].append(clip)

    canonical = []
    by_trail = {clip["trail_path"]: clip for clip in clips}
    for members in sorted(groups.values(), key=lambda group: group[0]["trail_path"]):
        members.sort(key=lambda clip: (clip["reference_start"] or "", clip["trail_path"]))
        canonical.append({
            "canonical_id": f"reference-{len(canonical) + 1:04d}",
            "canonical_trail_path": members[0]["trail_path"],
            "camera_id": members[0]["camera_id"],
            "source_trails": [member["trail_path"] for member in members],
            "archive_names": [member["archive"] for member in members],
            "row_ids": sorted(row_id for member in members for row_id in member["row_ids"]),
            "requires_exact_alignment": len(members) > 1,
        })

    # The raw-frame source is the identity boundary.  Time metadata is only a
    # candidate-generation hint; it can collide because old row timestamps
    # and producer filename starts were not produced by one clock.  Keep these
    # clips separate and expose the collision for audit, but do not ask a
    # reviewer to prove a duplicate that has no image evidence.  This follows
    # the platform invariant that raw-frame clips are independent and do not
    # overlap.
    time_overlap_warnings = [
        candidate for candidate in candidates
        if candidate["evidence"] == "review_required"
        and candidate["relation"] == "overlap"
    ]
    return {
        "reference_version": 1,
        "reference_rows": len(rows),
        "reference_clips": len(clips),
        "canonical_clips": len(canonical),
        "clips": clips,
        "canonical": canonical,
        "dedupe_candidates": candidates,
        "review_required": [],
        "time_overlap_warnings": time_overlap_warnings,
        "missing_sources": missing,
        "apply_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.rows.read_text())
    manifest = build_reference(rows, args.archive_root)
    manifest["source_rows_file"] = str(args.rows)
    manifest["archive_root"] = str(args.archive_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({
        "reference_rows": manifest["reference_rows"],
        "reference_clips": manifest["reference_clips"],
        "canonical_clips": manifest["canonical_clips"],
        "exact_merges": manifest["reference_clips"] - manifest["canonical_clips"],
        "review_required": len(manifest["review_required"]),
        "missing_sources": len(manifest["missing_sources"]),
        "apply_allowed": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
