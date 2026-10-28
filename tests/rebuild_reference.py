"""Rebuild one review clip per Reference source from raw frames.

The production detection rows are provenance only.  This runner never copies
their boxes or creates one output clip per row: it loads each canonical raw
archive group, tracks the complete frame axis with the current model, and
writes one review clip containing every fresh track found in that source.

It is intentionally review-only.  The output manifest always has
``apply_allowed: false``; a separate, human-gated migration can consume it
only after the review validators pass.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import TRACK_MIN_HITS, TRACK_MIN_SECONDS  # noqa: E402
from track_policy import is_substantial_track  # noqa: E402

from rebuild_clips import (
    colour_index,
    draw_review_box,
    encode,
    merge_track_fragments,
    segments,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_frames(path: Path) -> list[tuple[str, np.ndarray]]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.lower().endswith((".jpg", ".jpeg"))
        )
        frames = []
        for name in names:
            raw = archive.read(name)
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append((hashlib.sha1(raw).hexdigest(), frame))
        return frames


def write_rawframes_archive(frames: list[np.ndarray], destination: Path) -> None:
    """Write the exact canonical frame axis used by tracking."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, frame in enumerate(frames):
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92]
            )
            if not ok:
                raise OSError(f"could not encode raw frame {index}")
            archive.writestr(f"frame_{index:06d}.jpg", encoded.tobytes())


def visual_fingerprint(frame: np.ndarray) -> np.ndarray:
    image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
    return (image - image.mean()) / (image.std() + 1e-6)


def visual_offset(left: list[np.ndarray], right: list[np.ndarray]) -> tuple[int, int] | None:
    votes: dict[int, list[float]] = collections.defaultdict(list)
    for left_index, left_frame in enumerate(left):
        best_index, best_error = None, float("inf")
        for right_index, right_frame in enumerate(right):
            error = float(np.mean(np.abs(left_frame - right_frame)))
            if error < best_error:
                best_index, best_error = right_index, error
        if best_index is not None and best_error <= 0.03:
            votes[left_index - best_index].append(best_error)
    if not votes:
        return None
    offset, errors = max(votes.items(), key=lambda item: (len(item[1]), -sum(item[1])))
    return (offset, len(errors)) if len(errors) >= 6 else None


def aligned_frames(archives: list[Path]) -> list[np.ndarray]:
    """Place exact-overlap archives on one frame axis.

    A multi-archive canonical group is allowed only when Reference found at
    least two byte-identical JPEGs.  This repeats that proof at execution time
    and refuses a guessed concatenation if the files changed.
    """
    if not archives:
        raise ValueError("canonical clip has no raw archives")
    first = archive_frames(archives[0])
    if not first:
        raise ValueError(f"raw archive has no decodable JPEGs: {archives[0].name}")
    axis: dict[int, tuple[str, np.ndarray]] = {
        index: item for index, item in enumerate(first)
    }
    for path in archives[1:]:
        current = archive_frames(path)
        positions = {value: index for index, (value, _frame) in axis.items()}
        votes: collections.Counter[int] = collections.Counter()
        for index, (value, _frame) in enumerate(current):
            if value in positions:
                votes[positions[value] - index] += 1
        visual_proof = False
        if votes and votes.most_common(1)[0][1] >= 2:
            offset, _shared = votes.most_common(1)[0]
        else:
            axis_positions = sorted(axis)
            match = visual_offset(
                [visual_fingerprint(axis[index][1]) for index in axis_positions],
                [visual_fingerprint(frame) for _hash, frame in current],
            )
            if match is None:
                raise ValueError(f"exact or visual alignment proof missing for {path.name}")
            offset = axis_positions[0] + match[0]
            visual_proof = True
        for index, item in enumerate(current):
            slot = offset + index
            existing = axis.get(slot)
            if existing is not None and existing[0] != item[0]:
                if not visual_proof:
                    raise ValueError(f"conflicting frame at axis slot {slot} for {path.name}")
                # A visual match is made from a dominant run of frames, not
                # every boundary sample: one archive can sample just before
                # or after the other while still referring to the same video.
                # Reject a genuinely different scene, but tolerate that
                # boundary jitter and keep the already-established axis frame.
                if float(np.mean(np.abs(
                    visual_fingerprint(existing[1]) - visual_fingerprint(item[1])
                ))) > 0.12:
                    raise ValueError(f"visual conflict at axis slot {slot} for {path.name}")
                continue
            axis[slot] = item
    low, high = min(axis), max(axis)
    if set(axis) != set(range(low, high + 1)):
        raise ValueError("exactly aligned archives leave a hole in the frame axis")
    return [axis[index][1] for index in range(low, high + 1)]


def video_fps(trail_root: Path, trail_paths: list[str]) -> float:
    rates: list[float] = []
    for trail in trail_paths:
        for directory in (trail_root / "originals", trail_root / "fetched"):
            path = directory / trail
            if not path.is_file():
                continue
            capture = cv2.VideoCapture(str(path))
            fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()
            if fps and fps > 0:
                rates.append(float(fps))
            break
    return float(np.median(rates)) if rates else 10.0


def crop_histogram(frame: np.ndarray, box: list[float]) -> np.ndarray | None:
    x1, y1, x2, y2 = (max(0, int(value)) for value in box)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = cv2.resize(frame[y1:y2, x1:x2], (64, 64))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram


def crop_frame(frame: np.ndarray, box: list[float]) -> np.ndarray | None:
    """Return the old detector-style crop for one Reference animal.

    ``box`` is Reference's peak-frame ``xyxy`` box.  Keep the padding and
    integer conversion identical to the live detector so historical crops and
    new crops have the same framing.  The source frame must be that animal's
    confidence peak frame; applying this box to a clip poster is not valid.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    padding = max(20, max(width, height) // 30)
    left = max(0, x1 - padding)
    top = max(0, y1 - padding)
    right = min(width, x2 + padding)
    bottom = min(height, y2 + padding)
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def encode_crop(frame: np.ndarray, box: list[float]) -> bytes | None:
    crop = crop_frame(frame, box)
    if crop is None or crop.size == 0:
        return None
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return encoded.tobytes() if ok else None


def track_source(
    model: YOLO, frames: list[np.ndarray], imgsz: int, conf: float, tracker: Path,
) -> tuple[dict[int, list[list[float]]], dict[int, list[np.ndarray]]]:
    model.predictor = None
    tracks: dict[int, list[list[float]]] = collections.defaultdict(list)
    appearances: dict[int, list[np.ndarray]] = collections.defaultdict(list)
    for frame_number, frame in enumerate(frames):
        result = model.track(
            frame, imgsz=imgsz, conf=conf, classes=[0], device="mps", persist=True,
            tracker=str(tracker), verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or not len(boxes) or boxes.id is None:
            continue
        for xyxy, track_id, confidence in zip(
            boxes.xyxy.tolist(), boxes.id.tolist(), boxes.conf.tolist()
        ):
            track_number = int(track_id)
            tracks[track_number].append([
                frame_number,
                *[round(float(value), 2) for value in xyxy],
                round(float(confidence), 4),
            ])
            histogram = crop_histogram(frame, xyxy)
            if histogram is not None:
                appearances[track_number].append(histogram)
    return tracks, appearances


def rebuild_clip(model: YOLO, clip: dict, archive_root: Path, trail_root: Path,
                 output_dir: Path, imgsz: int, conf: float, tracker: Path,
                 ordinal: int, run_id: str) -> dict:
    archive_paths = [archive_root / name for name in clip["archive_names"]]
    if any(not path.is_file() for path in archive_paths):
        missing = [path.name for path in archive_paths if not path.is_file()]
        raise FileNotFoundError(f"raw archive missing: {', '.join(missing)}")
    frames = aligned_frames(archive_paths)
    if len(frames) < 2:
        raise ValueError("source has fewer than two decodable frames")

    raw_tracks, appearances = track_source(model, frames, imgsz, conf, tracker)
    tracks = merge_track_fragments(raw_tracks, appearance=appearances)
    fps = video_fps(trail_root, clip.get("source_trails", []))
    kept = {
        track_id: boxes
        for track_id, boxes in tracks.items()
        if is_substantial_track(
            len(boxes),
            (max(box[0] for box in boxes) - min(box[0] for box in boxes)) / fps,
            min_hits=TRACK_MIN_HITS,
            min_seconds=TRACK_MIN_SECONDS,
        ) and max(box[-1] for box in boxes) >= conf
    }
    canvas = [frame.copy() for frame in frames]
    asset_stem = f"rebuild-reference-{run_id}-{ordinal:04d}"
    rows = []
    for animal_number, (track_id, boxes) in enumerate(sorted(kept.items()), 1):
        animal_id = f"{clip['canonical_id']}-animal-{animal_number:02d}"
        colour = (
            (249, 232, 103), (36, 191, 251), (133, 113, 251), (172, 239, 134),
            (253, 181, 196), (252, 211, 125), (252, 171, 240), (116, 186, 253),
        )[colour_index(animal_id)]
        for frame_number, x1, y1, x2, y2, _confidence in boxes:
            draw_review_box(canvas[int(frame_number)], [x1, y1, x2, y2], colour,
                            f"#{animal_number}")
        positions = [int(box[0]) for box in boxes]
        peak = max(boxes, key=lambda box: box[-1])
        rows.append({
            "id": animal_number,
            "animal_id": animal_id,
            "animal_source": "model_rebuild",
            # The review-facing id must match the label burned into the MP4.
            # Keep ByteTrack's per-run id separately for provenance; it is not
            # stable enough to expose as the identity users compare in the UI.
            "track_id": animal_number,
            "source_track_id": track_id,
            "confidence": max(box[-1] for box in boxes),
            "bbox": [round(float(value), 2) for value in peak[1:5]],
            "clip_offset_seconds": round(positions[0] / fps, 3),
            "clip_end_offset_seconds": round(positions[-1] / fps, 3),
            "clip_duration_seconds": round(len(frames) / fps, 3),
            "clip_segments": segments(positions, fps),
            "track_box_fps": round(fps, 4),
            "boxes": boxes,
        })

    # A detection row's image_path is the raw, full-resolution peak frame for
    # that animal.  The annotated counterpart is deliberately a separate file:
    # image management/downloads must remain clean training input, while review
    # needs to show the boxes.  Generate these only after every track has been
    # drawn so an annotated preview contains all animals visible in the frame.
    for row in rows:
        peak_frame_number = int(max(row["boxes"], key=lambda box: box[-1])[0])
        if peak_frame_number < 0 or peak_frame_number >= len(frames):
            raise ValueError(f"track peak frame outside source axis: {peak_frame_number}")
        animal_suffix = f"animal-{int(row['track_id']):02d}"
        raw_name = f"{asset_stem}_{animal_suffix}_raw.jpg"
        annotated_name = f"{asset_stem}_{animal_suffix}_annotated.jpg"
        if not cv2.imwrite(str(output_dir / raw_name), frames[peak_frame_number]):
            raise OSError(f"could not write raw preview: {raw_name}")
        if not cv2.imwrite(str(output_dir / annotated_name), canvas[peak_frame_number]):
            raise OSError(f"could not write annotated preview: {annotated_name}")
        crop_name = f"{asset_stem}_{animal_suffix}_crop.jpg"
        crop_bytes = encode_crop(frames[peak_frame_number], row["bbox"])
        if crop_bytes is None:
            raise ValueError(f"could not create crop preview: {crop_name}")
        (output_dir / crop_name).write_bytes(crop_bytes)
        row["image_path"] = f"rebuild-review/{raw_name}"
        row["annotated_path"] = f"rebuild-review/{annotated_name}"
        row["crop_path"] = f"rebuild-review/{crop_name}"

    output_name = f"{asset_stem}_trail.mp4"
    output_path = output_dir / output_name
    encode(canvas, fps, output_path)
    rawframes_name = f"{asset_stem}_rawframes.zip"
    write_rawframes_archive(frames, output_dir / rawframes_name)
    poster_name = output_name.removesuffix("_trail.mp4") + "_poster.jpg"
    poster_path = output_dir / poster_name
    if not cv2.imwrite(str(poster_path), canvas[0]):
        raise OSError(f"could not write poster: {poster_path}")
    return {
        "clip_id": clip["canonical_id"],
        "camera_id": clip.get("camera_id"),
        "timestamp": clip.get("timestamp"),
        "source_trail_path": clip.get("canonical_trail_path"),
        "reference_source_trails": clip.get("source_trails", []),
        "output_path": f"rebuild-review/{output_name}",
        "rawframes_path": f"rebuild-review/{rawframes_name}",
        "poster_path": f"rebuild-review/{poster_name}",
        "source_kind": "rawframes",
        "source_frames": len(frames),
        "fps": round(fps, 4),
        "duration_seconds": round(len(frames) / fps, 3),
        "track_count": len(rows),
        "row_count": len(rows),
        "rows": rows,
        "status": "ready",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--trail-root", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    reference_times = {
        clip["trail_path"]: clip.get("reference_start")
        for clip in reference.get("clips", [])
    }
    # Raw archives have no clock of their own. The old row timestamp is kept
    # only as the clip's display anchor; all in-video positions below still
    # come from the raw frame axis, never from wall-clock interpolation.
    for canonical in reference.get("canonical", []):
        starts = [reference_times.get(source) for source in canonical.get("source_trails", [])]
        canonical["timestamp"] = min((value for value in starts if value), default=None)
    base = args.reference.parent.parent
    archive_root = args.archive_root or base / "zips"
    trail_root = args.trail_root or base
    args.output.mkdir(parents=True, exist_ok=True)
    partial_path = args.output / "reference_rebuild.partial.json"
    partial = json.loads(partial_path.read_text()) if args.resume and partial_path.exists() else {
        "clips": [], "failures": []
    }
    results = partial.get("clips", [])
    failures = partial.get("failures", [])
    processed = {item.get("clip_id") for item in results + failures}
    clips = reference.get("canonical", [])
    if args.limit:
        clips = clips[:args.limit]

    if not args.model.is_file():
        raise SystemExit(f"model missing: {args.model}")
    source = {
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "device": "mps",
        "imgsz": args.imgsz,
        "conf": args.conf,
        "tracker": str(args.tracker),
        "reference": str(args.reference),
    }
    model = YOLO(str(args.model))
    model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=args.imgsz,
                  device="mps", verbose=False)
    backend = getattr(model.predictor, "model", None)
    if backend is not None and hasattr(backend, "model"):
        backend.model.half()
        backend.fp16 = True

    def checkpoint() -> None:
        partial_path.write_text(json.dumps({
            "status": "running", "source": source, "clips": results,
            "failures": failures, "apply_allowed": False,
        }, ensure_ascii=False, indent=2))

    started = time.monotonic()
    for ordinal, clip in enumerate(clips, 1):
        if clip["canonical_id"] in processed:
            continue
        try:
            result = rebuild_clip(
                model, clip, archive_root, trail_root, args.output, args.imgsz,
                args.conf, args.tracker, ordinal, args.output.name,
            )
            results.append(result)
            print(f"[{ordinal}/{len(clips)}] {clip['canonical_id']} "
                  f"{result['source_frames']} frames {result['track_count']} tracks",
                  flush=True)
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
            failure = {"clip_id": clip["canonical_id"],
                       "source_trail_path": clip.get("canonical_trail_path"),
                       "status": "failed", "error": str(exc)}
            if not args.skip_failed:
                checkpoint()
                raise
            failures.append(failure)
            print(f"[failed] {clip['canonical_id']}: {exc}", flush=True)
        processed.add(clip["canonical_id"])
        checkpoint()

    manifest = {
        "run_id": args.output.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "review",
        "source": source,
        "clips": results,
        "clip_count": len(results),
        "row_count": sum(len(item.get("rows", [])) for item in results),
        "failures": failures,
        "failed_count": len(failures),
        "source_counts": dict(collections.Counter(item.get("source_kind", "unknown")
                                                   for item in results)),
        "reference": {
            "reference_clips": reference.get("reference_clips"),
            "canonical_clips": reference.get("canonical_clips"),
            "missing_sources": reference.get("missing_sources", []),
            "review_required": reference.get("review_required", []),
            "time_overlap_warnings": reference.get("time_overlap_warnings", []),
        },
        "rawframes_required": True,
        "apply_allowed": False,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    (args.output / "reference_rebuild_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(json.dumps({
        "clips": len(results), "rows": manifest["row_count"],
        "failures": len(failures), "manifest": str(args.output / "reference_rebuild_manifest.json"),
        "apply_allowed": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
