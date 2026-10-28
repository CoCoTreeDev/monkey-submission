"""Rebuild historical trail clips from one frozen model run.

The input manifest is produced by ``prepare_full_rebuild.py``.  This script
never calls the API and never touches the old artifacts: it writes a new run
directory containing one rendered clip and one track manifest per source clip.
Rows are created from the new tracker output, not copied from old detections.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import TRACK_MIN_HITS, TRACK_MIN_SECONDS, _resolve_ffmpeg  # noqa: E402
from track_policy import is_substantial_track  # noqa: E402


def encode(frames: list[np.ndarray], fps: float, dest: Path) -> None:
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [_resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{width}x{height}",
         "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
         "-g", str(max(1, round(fps * 2))),
         "-movflags", "+faststart", str(dest)], stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait(timeout=300) != 0:
            raise RuntimeError(f"ffmpeg failed for {dest.name}")
    except Exception:
        process.kill()
        raise


def draw_review_box(
    frame: np.ndarray,
    box: list[float],
    colour: tuple[int, int, int],
    label: str,
) -> None:
    """Draw a readable, stable review box over mixed camera backgrounds."""
    x1, y1, x2, y2 = (int(value) for value in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (8, 8, 8), 7, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3, cv2.LINE_AA)
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    top = max(th + baseline + 4, y1)
    left = max(0, min(x1, frame.shape[1] - tw - 10))
    cv2.rectangle(frame, (left, top - th - baseline - 6),
                  (left + tw + 10, top), colour, -1)
    cv2.putText(frame, label, (left + 5, top - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (8, 8, 8), 2, cv2.LINE_AA)


def colour_index(value: int | str) -> int:
    """Match the frontend's stable FNV palette selection."""
    hash_value = 2166136261
    for char in str(value):
        hash_value ^= ord(char)
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return hash_value % 8


def images_from_zip(path: Path, fps: float = 10.0) -> tuple[list[np.ndarray], float]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist()
                       if n.lower().endswith((".jpg", ".jpeg")))
        frames = []
        for name in names:
            frame = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
    return frames, fps


def images_from_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """IoU for xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centre_distance_in_box_widths(a: list[float], b: list[float]) -> float:
    """Distance between boxes, normalised by their average diagonal."""
    ac = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    scale_a = max(1.0, ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5)
    scale_b = max(1.0, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5)
    scale = (scale_a + scale_b) / 2
    return (((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5) / scale


def _predicted_centre(boxes: list[list[float]], gap: int) -> tuple[float, float]:
    """Extrapolate a fragment's centre through a missing-frame gap."""
    tail = boxes[-min(5, len(boxes)):]
    if len(tail) < 2:
        box = tail[-1]
        return ((box[1] + box[3]) / 2, (box[2] + box[4]) / 2)
    first, last = tail[0], tail[-1]
    span = max(1, int(last[0]) - int(first[0]))
    vx = (((last[1] + last[3]) - (first[1] + first[3])) / 2) / span
    vy = (((last[2] + last[4]) - (first[2] + first[4])) / 2) / span
    return (
        (last[1] + last[3]) / 2 + vx * (gap + 1),
        (last[2] + last[4]) / 2 + vy * (gap + 1),
    )


def _predicted_distance(boxes: list[list[float]], next_box: list[float], gap: int) -> float:
    predicted = _predicted_centre(boxes, gap)
    actual = ((next_box[0] + next_box[2]) / 2, (next_box[1] + next_box[3]) / 2)
    scale = max(1.0, ((boxes[-1][3] - boxes[-1][1]) +
                      (next_box[2] - next_box[0])) / 2)
    return ((predicted[0] - actual[0]) ** 2 +
            (predicted[1] - actual[1]) ** 2) ** 0.5 / scale


def _shared_frame_similarity(a: list[list[float]], b: list[list[float]]) -> tuple[float, int, float]:
    """Mean IoU, shared frame count, and shared/shorter-track ratio."""
    by_frame_a = {int(box[0]): box[1:5] for box in a}
    by_frame_b = {int(box[0]): box[1:5] for box in b}
    shared = by_frame_a.keys() & by_frame_b.keys()
    if not shared:
        return 0.0, 0, 0.0
    mean_iou = sum(_bbox_iou(by_frame_a[i], by_frame_b[i]) for i in shared) / len(shared)
    shorter = max(1, min(len(by_frame_a), len(by_frame_b)))
    return mean_iou, len(shared), len(shared) / shorter


def _appearance_similarity(
    left: list[np.ndarray], right: list[np.ndarray],
) -> float:
    """Compare the best crop histograms from two fragmented tracks.

    This is deliberately a within-clip continuation signal, not a cross-camera
    identity claim.  It prevents geometry from joining two nearby animals
    after a long gap while still allowing the same animal to change pose.
    """
    if not left or not right:
        return 0.0
    scores = [
        float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
        for a in left for b in right
    ]
    return float(np.mean(sorted(scores, reverse=True)[:3]))


def _spans_overlap(boxes: list[list[float]], start: int, end: int) -> bool:
    """Whether a track is alive anywhere in an interval of source frames."""
    return int(boxes[0][0]) <= end and int(boxes[-1][0]) >= start


def _merge_boxes(keep: list[list[float]], other: list[list[float]]) -> list[list[float]]:
    """Merge two fragments, retaining one box per frame and best confidence."""
    merged: dict[int, list[float]] = {int(box[0]): box for box in other}
    for box in keep:
        frame = int(box[0])
        if frame not in merged or box[-1] >= merged[frame][-1]:
            merged[frame] = box
    return [merged[frame] for frame in sorted(merged)]


def merge_track_fragments(
    tracks: dict[int, list[list[float]]],
    *,
    appearance: dict[int, list[np.ndarray]] | None = None,
    max_gap_frames: int = 30,
    duplicate_iou: float = 0.35,
    duplicate_min_frames: int = 2,
    duplicate_min_overlap: float = 0.5,
    continuation_distance: float = 1.8,
) -> dict[int, list[list[float]]]:
    """Collapse ByteTrack ids that are fragments of one visible animal.

    Same-frame, consistently-overlapping ids are duplicate tracker claims.
    A strict 0.5 IoU floor left near-identical boxes split when the detector
    jittered one edge by a few pixels, which is how a one-animal clip became
    two rows. Non-overlapping ids
    are joined only across a short temporal gap when the next box is still
    spatially continuous. Concurrent animals are never merged by this rule.
    The result is review-only evidence; it does not invent detections.
    """
    active = [
        (track_id, sorted(boxes, key=lambda box: int(box[0])))
        for track_id, boxes in tracks.items()
        if boxes
    ]
    active.sort(key=lambda item: (int(item[1][0][0]), item[0]))

    changed = True
    while changed:
        changed = False
        for index, (keep_id, keep) in enumerate(active):
            for other_index in range(index + 1, len(active)):
                other_id, other = active[other_index]
                mean_iou, shared, ratio = _shared_frame_similarity(keep, other)
                duplicate = (
                    shared >= duplicate_min_frames
                    and ratio >= duplicate_min_overlap
                    and mean_iou >= duplicate_iou
                )
                keep_end = int(keep[-1][0])
                other_start = int(other[0][0])
                gap = other_start - keep_end - 1
                continuation = False
                if gap >= 0 and gap <= max_gap_frames:
                    # A long-gap continuation is ambiguous when another track
                    # is alive through that gap. Keep the fragment separate in
                    # that case; otherwise a third animal can be swallowed by
                    # the nearest trajectory (as happened in 場域乙 014791).
                    long_gap = gap > 12
                    competing_track = any(
                        candidate_id not in {keep_id, other_id}
                        and _spans_overlap(candidate, keep_end + 1, other_start)
                        for candidate_id, candidate in active
                    )
                    appearance_continuation = (
                        long_gap
                        and appearance is not None
                        and _appearance_similarity(
                            appearance.get(keep_id, []),
                            appearance.get(other_id, []),
                        ) >= 0.88
                    )
                    if not (long_gap and competing_track) or appearance_continuation:
                        continuation = (
                            _bbox_iou(keep[-1][1:5], other[0][1:5]) >= 0.3
                            or _centre_distance_in_box_widths(keep[-1][1:5], other[0][1:5])
                            <= continuation_distance
                            or _predicted_distance(keep, other[0][1:5], gap)
                            <= 3.5
                            or (
                                long_gap
                                and len(active) == 2
                                and _centre_distance_in_box_widths(
                                    keep[-1][1:5], other[0][1:5]
                                ) <= 4.0
                            )
                            or appearance_continuation
                        )
                if not duplicate and not continuation:
                    continue
                # Keep the earlier id for deterministic manifests. A duplicate
                # may be merged even when ByteTrack emitted both in the same
                # frames; a continuation may only be forward in time.
                keep[:] = _merge_boxes(keep, other)
                active.pop(other_index)
                changed = True
                break
            if changed:
                break

    return {track_id: boxes for track_id, boxes in active}


def source_for(
    item: dict,
    run_dir: Path,
    *,
    allow_trail_fallback: bool = True,
) -> tuple[Path | None, str | None]:
    """Choose the complete archived raw-frame source before rendered clips.

    A fetched trail MP4 is only a fallback: using it first silently skipped
    raw-frame inference for clips whose archive was available, which made the
    review run answer a different question from the requested backfill.
    """
    base = run_dir.parent
    archive = item.get("archive")
    if archive:
        for directory in (base / "zips", base / "merged"):
            path = directory / archive
            if path.exists():
                return path, "rawframes"
    if not allow_trail_fallback:
        return None, None
    trail = item["trail_path"]
    for directory in (run_dir / "fetched", base / "originals"):
        path = directory / trail
        if path.exists():
            return path, "trail"
    return None, None


def clip_duration_from_trail(path: Path) -> float | None:
    """Read a rendered trail duration for assigning raw frames to its axis."""
    if path.suffix.lower() not in {".mp4", ".gif"}:
        return None
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    capture.release()
    return count / fps if fps > 0 and count > 0 else None


def segments(frames: list[int], fps: float, gap: float = 1.0) -> list[list[float]]:
    if not frames:
        return []
    out: list[list[float]] = []
    start = previous = frames[0] / fps
    for frame in frames[1:]:
        current = frame / fps
        if current - previous > gap:
            out.append([round(start, 3), round(previous, 3)])
            start = current
        previous = current
    out.append([round(start, 3), round(previous, 3)])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.home() / "Monkey-Detection")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-failed", action="store_true",
                        help="record unreadable clips and continue")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the per-clip checkpoint")
    parser.add_argument("--allow-trail-fallback", action="store_true",
                        help="diagnostic only: allow rendered trails when raw archives are absent")
    args = parser.parse_args()
    rawframes_required = not args.allow_trail_fallback

    manifest = json.loads(args.manifest.read_text())
    run_dir = args.manifest.parent
    source = manifest["source"]
    model_path = Path(source["model"])
    if source["model_sha256"] != __import__("hashlib").sha256(model_path.read_bytes()).hexdigest():
        raise SystemExit("model changed after preparation; refusing to run")
    clips = manifest["clips"][:args.limit] if args.limit else manifest["clips"]
    output = run_dir / "rebuilt"
    output.mkdir(exist_ok=True)
    snapshot_path = run_dir / "rows.snapshot.json"
    snapshot_rows = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else []
    duration_by_trail = {
        trail: max(
            (float(row["clip_duration_seconds"])
             for row in snapshot_rows
             if row.get("trail_path") == trail
             and row.get("clip_duration_seconds") is not None),
            default=0.0,
        )
        for trail in {item["trail_path"] for item in clips}
    }
    model = YOLO(str(model_path))
    model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=source["imgsz"], device="mps", verbose=False)
    backend = getattr(model.predictor, "model", None)
    if backend is not None and hasattr(backend, "model"):
        backend.model.half()
        backend.fp16 = True

    checkpoint_path = run_dir / "rebuilt_manifest.partial.json"
    results = []
    failures = []
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("rawframes_required", rawframes_required) != rawframes_required:
            raise SystemExit(
                "resume mode changed raw-frame policy; start a new run or delete the checkpoint"
            )
        results = checkpoint.get("clips", [])
        failures = checkpoint.get("failures", [])
        print(f"resuming: {len(results) + len(failures)} already processed", flush=True)
    processed = {item["source_trail_path"] for item in results + failures}

    def checkpoint() -> None:
        checkpoint_path.write_text(json.dumps({
            "run_id": manifest["run_id"], "source": source,
            "clips": results, "clip_count": len(results),
            "row_count": sum(len(item["rows"]) for item in results),
            "failures": failures, "failed_count": len(failures),
            "status": "running", "apply_allowed": False,
            "rawframes_required": rawframes_required,
        }, ensure_ascii=False, indent=2))
    started = time.monotonic()
    for number, item in enumerate(clips, 1):
        trail = item["trail_path"]
        if trail in processed:
            continue
        source_path, source_kind = source_for(
            item, run_dir, allow_trail_fallback=not rawframes_required)
        if source_path is None:
            if args.skip_failed:
                failures.append({"source_trail_path": trail, "status": "skipped",
                                 "error": "raw-frame archive missing" if rawframes_required
                                 else "source missing after preparation"})
                checkpoint()
                print(f"[skip] {trail}: source missing after preparation", flush=True)
                continue
            reason = "raw-frame archive missing" if rawframes_required else "source missing after preparation"
            raise SystemExit(f"{reason}: {trail}")
        if source_path.suffix == ".zip":
            source_duration = duration_by_trail.get(trail, 0.0)
            if not source_duration:
                for directory in (run_dir / "fetched", run_dir.parent / "originals"):
                    rendered = directory / trail
                    source_duration = clip_duration_from_trail(rendered) if rendered.exists() else None
                    if source_duration:
                        break
            frames, archive_fps = images_from_zip(source_path)
            # Raw archives contain the selected full-resolution frames, not a
            # new clock. Preserve the source clip's duration when it is known;
            # otherwise retain the conservative archive default.
            fps = len(frames) / source_duration if source_duration and frames else archive_fps
        else:
            frames, fps = images_from_video(source_path)
        if len(frames) < 2:
            if args.skip_failed:
                failures.append({"source_trail_path": trail, "status": "skipped",
                                 "error": "source has fewer than two frames"})
                checkpoint()
                print(f"[skip] {trail}: source has fewer than two frames", flush=True)
                continue
            raise SystemExit(f"source has fewer than two frames: {trail}")

        model.predictor = None
        tracks: dict[int, list[list[float]]] = {}
        for frame_number, frame in enumerate(frames):
            result = model.track(frame, imgsz=source["imgsz"], conf=0.35, classes=[0],
                                 device="mps", persist=True,
                                 tracker=str(args.repo / "backend/trackers/bytetrack-monkey.yaml"),
                                 verbose=False)[0]
            boxes = result.boxes
            if boxes is None or not len(boxes):
                continue
            ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
            confs = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(boxes)
            for xyxy, track_id, confidence in zip(boxes.xyxy.tolist(), ids, confs):
                if track_id is None:
                    continue
                tracks.setdefault(int(track_id), []).append(
                    [frame_number, *[round(float(value), 2) for value in xyxy], round(float(confidence), 4)])

        tracks = merge_track_fragments(tracks)
        kept = {
            track_id: boxes
            for track_id, boxes in tracks.items()
            if is_substantial_track(
                len(boxes),
                (max(box[0] for box in boxes) - min(box[0] for box in boxes)) / fps,
                min_hits=TRACK_MIN_HITS,
                min_seconds=TRACK_MIN_SECONDS,
            ) and max(box[-1] for box in boxes) >= 0.5
        }
        canvas = [frame.copy() for frame in frames]
        palette = [(249, 232, 103), (36, 191, 251), (133, 113, 251), (172, 239, 134),
                   (253, 181, 196), (252, 211, 125), (252, 171, 240), (116, 186, 253)]
        rows = []
        for index, (track_id, boxes) in enumerate(sorted(kept.items())):
            colour = palette[colour_index(track_id)]
            peak = max(boxes, key=lambda box: box[-1])
            for frame_number, x1, y1, x2, y2, _confidence in boxes:
                draw_review_box(canvas[int(frame_number)], [x1, y1, x2, y2],
                                colour, f"#{track_id}")
            positions = [int(box[0]) for box in boxes]
            rows.append({
                "source_trail_path": trail,
                "track_id": track_id,
                "confidence": max(box[-1] for box in boxes),
                "bbox": [round(float(value), 2) for value in peak[1:5]],
                "clip_offset_seconds": round(positions[0] / fps, 3),
                "clip_end_offset_seconds": round(positions[-1] / fps, 3),
                "clip_duration_seconds": round(len(frames) / fps, 3),
                "clip_segments": segments(positions, fps),
                "boxes": boxes,
            })
        rebuilt_name = f"rebuild_{manifest['run_id']}_{number:04d}_trail.mp4"
        rebuilt_path = output / rebuilt_name
        encode(canvas, fps, rebuilt_path)
        results.append({"source_trail_path": trail, "trail_path": rebuilt_name,
                        "source_kind": source_kind, "source_frames": len(frames),
                        "fps": round(fps, 4), "rows": rows})
        checkpoint()
        if number % 5 == 0 or number == len(clips):
            elapsed = time.monotonic() - started
            print(f"[{number}/{len(clips)}] {trail} {len(frames)} frames "
                  f"{len(rows)} tracks {elapsed:.1f}s", flush=True)

    output_manifest = {"run_id": manifest["run_id"], "source": source,
                       "clips": results, "clip_count": len(results),
                       "row_count": sum(len(item["rows"]) for item in results),
                       "failures": failures, "failed_count": len(failures),
                       "source_counts": dict(collections.Counter(
                           item.get("source_kind", "unknown") for item in results)),
                       "apply_allowed": False,
                       "rawframes_required": rawframes_required}
    (run_dir / "rebuilt_manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"clips": len(results), "rows": output_manifest["row_count"],
                      "manifest": str(run_dir / "rebuilt_manifest.json"),
                      "apply_allowed": False,
                      "rawframes_required": rawframes_required}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
