"""Repair raw/annotated previews for an already-applied rebuild.

This is deliberately model-free. It reuses the frozen track boxes in a review
manifest and the exact raw-frame archives that produced them, then updates
only image_path/annotated_path on the existing rows. The detection data and
the delete lock are never changed.

Usage on the mini:

    API=https://... TOKEN=... uv run --project ~/Monkey-Detection/backend \
      python tests/repair_rebuild_media.py --apply \
      --manifest ~/rawframes-backfill/reference-rebuild-.../reference_rebuild_manifest.json \
      --reference ~/rawframes-backfill/reference-20260830/reference_manifest_v5.json \
      --archive-root ~/rawframes-backfill/zips \
      --output ~/rawframes-backfill/media-repair
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TESTS.parent / "backend"))

from rebuild_clips import colour_index, draw_review_box  # noqa: E402
from rebuild_reference import aligned_frames, encode_crop  # noqa: E402
from services.storage import storage  # noqa: E402


def source_archive_name(trail_path: str) -> str | None:
    stem, marker, _ = trail_path.rpartition("_trail.")
    return f"{stem}_rawframes.zip" if marker else None


def basename(value: str) -> str:
    return Path(value).name


def archive_map(reference: dict) -> dict[str, list[str]]:
    """Map every source trail to its Reference-proven raw archive names."""
    result: dict[str, list[str]] = {}
    entries = list(reference.get("canonical", []))
    entries += list(reference.get("clips", []))
    for entry in entries:
        sources = entry.get("source_trails") or []
        if entry.get("trail_path") and not sources:
            sources = [entry["trail_path"]]
        names = entry.get("archive_names") or []
        if entry.get("archive") and not names:
            names = [entry["archive"]]
        for source in sources:
            if names:
                result[source] = list(dict.fromkeys(names))
    return result


def request_json(api: str, token: str, path: str) -> dict:
    request = urllib.request.Request(
        f"{api.rstrip('/')}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def fetch_detections(
    api: str, token: str, trail_path: str | None = None
) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        query = f"limit=200&offset={offset}"
        if trail_path:
            from urllib.parse import quote
            query += f"&trail_path={quote(trail_path)}"
        page = request_json(api, token, f"/api/detections?{query}")
        items = page.get("items") or []
        rows.extend(items)
        offset += len(items)
        if not items or offset >= int(page.get("total") or 0):
            return rows


def post_media(api: str, token: str, items: list[dict]) -> dict:
    request = urllib.request.Request(
        f"{api.rstrip('/')}/api/admin/detections/media-bulk",
        data=json.dumps({"items": items}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def encoded(frame: np.ndarray) -> bytes:
    ok, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError("could not encode preview JPEG")
    return payload.tobytes()


def rawframes_archive(frames: list[np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, frame in enumerate(frames):
            archive.writestr(f"frame_{index:06d}.jpg", encoded(frame))
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api", default=os.environ.get("API", ""))
    parser.add_argument("--token", default=os.environ.get("TOKEN", ""))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.api or not args.token:
        raise SystemExit("API and TOKEN are required")
    manifest = json.loads(args.manifest.read_text())
    reference = json.loads(args.reference.read_text())
    source_archives = archive_map(reference)
    existing = fetch_detections(args.api, args.token)
    by_key = {
        (basename(row.get("trail_path") or ""), int(row["track_id"])): row
        for row in existing
        if row.get("trail_path") and row.get("track_id") is not None
    }
    args.output.mkdir(parents=True, exist_ok=True)
    media_items: list[dict] = []
    processed_clips = 0
    processed_rows = 0

    for clip in manifest.get("clips", []):
        output_path = clip.get("output_path") or ""
        output_key = basename(output_path)
        # The live edge may insert a row between offset pages. Fill each clip
        # from the exact trail filter when the initial all-rows snapshot missed
        # anything; this prevents a shifted page from aborting a valid repair.
        expected_track_ids = {
            int(row["track_id"]) for row in (clip.get("rows") or [])
        }
        missing_track_ids = expected_track_ids - {
            track_id for (trail, track_id) in by_key if trail == output_key
        }
        if missing_track_ids:
            for current in fetch_detections(args.api, args.token, output_key):
                if current.get("trail_path") and current.get("track_id") is not None:
                    by_key[(basename(current["trail_path"]), int(current["track_id"]))] = current
        archive_names: list[str] = []
        for source in clip.get("reference_source_trails") or []:
            names = source_archives.get(source) or []
            if not names:
                fallback = source_archive_name(source)
                if fallback:
                    names = [fallback]
            archive_names.extend(names)
        archive_names = list(dict.fromkeys(archive_names))
        archive_paths = [args.archive_root / name for name in archive_names]
        missing = [str(path) for path in archive_paths if not path.is_file()]
        if missing:
            raise SystemExit(f"{output_key}: raw archive missing: {missing}")
        frames = aligned_frames(archive_paths)
        raw_archive_name = (
            output_key.rpartition("_trail.")[0] + "_rawframes.zip"
            if "_trail." in output_key else ""
        )
        canvas = [frame.copy() for frame in frames]
        rows = clip.get("rows") or []
        for row in rows:
            animal_id = row.get("animal_id") or row.get("track_id")
            colour = (
                (249, 232, 103), (36, 191, 251), (133, 113, 251), (172, 239, 134),
                (253, 181, 196), (252, 211, 125), (252, 171, 240), (116, 186, 253),
            )[colour_index(animal_id)]
            for box in row.get("boxes") or []:
                frame_number = int(box[0])
                if frame_number < 0 or frame_number >= len(canvas):
                    raise SystemExit(f"{output_key}: box outside raw frame axis")
                draw_review_box(canvas[frame_number], box[1:5], colour,
                                f"#{int(row['track_id'])}")
        if args.apply and raw_archive_name:
            storage.upload(raw_archive_name, rawframes_archive(frames))

        for row in rows:
            track_id = int(row["track_id"])
            current = by_key.get((output_key, track_id))
            if current is None:
                raise SystemExit(f"production row missing: {output_key} track {track_id}")
            peak = max(row.get("boxes") or [], key=lambda box: box[-1])
            frame_number = int(peak[0])
            prefix = f"rebuild-media-{current['id']}"
            raw_name = f"{prefix}_raw.jpg"
            annotated_name = f"{prefix}_annotated.jpg"
            crop_name = f"{prefix}_crop.jpg"
            raw_bytes = encoded(frames[frame_number])
            annotated_bytes = encoded(canvas[frame_number])
            crop_bytes = encode_crop(frames[frame_number], row["bbox"])
            if crop_bytes is None:
                raise SystemExit(f"{output_key}: invalid Reference crop for track {track_id}")
            (args.output / raw_name).write_bytes(raw_bytes)
            (args.output / annotated_name).write_bytes(annotated_bytes)
            (args.output / crop_name).write_bytes(crop_bytes)
            if args.apply:
                storage.upload(raw_name, raw_bytes)
                storage.upload(annotated_name, annotated_bytes)
                storage.upload(crop_name, crop_bytes)
                media_items.append({
                    "detection_id": int(current["id"]),
                    "image_path": raw_name,
                    "annotated_path": annotated_name,
                    "crop_path": crop_name,
                })
            processed_rows += 1
        processed_clips += 1

    if args.apply:
        for start in range(0, len(media_items), 200):
            result = post_media(args.api, args.token, media_items[start:start + 200])
            if result.get("updated") != len(media_items[start:start + 200]):
                raise SystemExit(f"media update count mismatch: {result}")
    print(json.dumps({
        "clips": processed_clips,
        "rows": processed_rows,
        "updated": len(media_items) if args.apply else 0,
        "apply": args.apply,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile,
            urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise SystemExit(f"media repair failed: {exc}") from exc
