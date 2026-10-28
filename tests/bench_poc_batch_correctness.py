"""Compare serial tracking with the shape-preserving POC batch path.

This is a live-source benchmark, not a deployment command. It captures a
small set of frames from a YouTube HLS rendition, then compares two lanes of
the external-tracker batch path with the existing per-camera ``model.track``
path. The input shape is kept identical so a passing result actually tests
the production rectangular letterbox rather than a different square-padding
configuration.

Run on the M4 mini from the repository root:

    uv run --project backend python tests/bench_poc_batch_correctness.py
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "backend"
load_dotenv(BACKEND / ".env")
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import services.detector as detector  # noqa: E402
from services.detector import _own_model  # noqa: E402
from services.stream import _resolve_capture_url  # noqa: E402
from services.tiling import StandardTracker  # noqa: E402


def _capture(url: str, count: int) -> list[np.ndarray]:
    capture_url = _resolve_capture_url(url)
    if not capture_url:
        raise RuntimeError("source URL could not be resolved")
    cap = cv2.VideoCapture(capture_url)
    frames: list[np.ndarray] = []
    deadline = time.time() + 60
    while len(frames) < count and time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    if len(frames) < count:
        raise RuntimeError(f"captured {len(frames)}/{count} frames")
    return frames


def _tracked_rows(result) -> list[list[float]]:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    return _tracker_rows(boxes.data)


def _tracker_rows(tracks) -> list[list[float]]:
    """Convert tracked [xyxy, id, score, cls, ...] to [xyxy, conf, cls, id]."""
    return [[*row[:4], row[5], row[6], row[4]] for row in tracks.tolist()]


def _normalise(sequence: list[list[list[float]]]) -> list[list[tuple[float, ...]]]:
    """Normalise IDs once per sequence, preserving identity across frames."""
    id_map: dict[int, int] = {}
    normalised = []
    for rows in sequence:
        frame = []
        for row in sorted(rows, key=lambda item: tuple(item[:4])):
            source_id = int(row[6])
            if source_id not in id_map:
                id_map[source_id] = len(id_map) + 1
            frame.append((*[round(value, 5) for value in row[:6]], id_map[source_id]))
        normalised.append(frame)
    return normalised


def _serial(frames: list[np.ndarray], device: str, half: bool, imgsz: int):
    model = _own_model(device, half)
    rows = []
    for frame in frames:
        result = model.track(
            frame,
            imgsz=imgsz,
            conf=config.TRACKER_CONFIDENCE_THRESHOLD,
            persist=True,
            tracker=config.TRACKER_YAML,
            device=device or None,
            verbose=False,
        )[0]
        rows.append(_tracked_rows(result))
    return rows


def _batch(frames: list[np.ndarray], device: str, half: bool, imgsz: int):
    model = _own_model(device, half)
    lane_a, lane_b = StandardTracker(), StandardTracker()
    rows_a, rows_b = [], []
    for frame in frames:
        results = model.predict(
            [frame, frame],
            imgsz=imgsz,
            rect=True,
            batch=2,
            conf=config.TRACKER_CONFIDENCE_THRESHOLD,
            device=device or None,
            verbose=False,
        )
        rows_a.append(_tracker_rows(lane_a.update(results[0])))
        rows_b.append(_tracker_rows(lane_b.update(results[1])))
    return rows_a, rows_b


def _sequential_batch(frames: list[np.ndarray], device: str, half: bool, imgsz: int):
    """Run consecutive frames in one batch, then update one tracker in order."""
    model = _own_model(device, half)
    tracker = StandardTracker()
    rows = []
    for start in range(0, len(frames), 2):
        results = model.predict(
            frames[start : start + 2],
            imgsz=imgsz,
            rect=True,
            batch=2,
            conf=config.TRACKER_CONFIDENCE_THRESHOLD,
            device=device or None,
            verbose=False,
        )
        for result in results:
            rows.append(_tracker_rows(tracker.update(result)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        required=True,
        help="authorized camera or video URL to use for this benchmark",
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--device", default=config.INFERENCE_DEVICE or "cpu")
    parser.add_argument(
        "--sequential-batch",
        action="store_true",
        help="batch consecutive frames from one source instead of duplicating lanes",
    )
    args = parser.parse_args()
    if args.frames < 4 or args.frames % 2:
        parser.error("--frames must be an even number >= 4")

    device = args.device
    half = device == "mps"
    detector.MODEL_PATH = config.MODEL_PATH
    frames = _capture(args.url, args.frames)
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise RuntimeError(f"source changed shape during capture: {shapes}")
    height, width = frames[0].shape[:2]
    imgsz = config.imgsz_for_frame(height, width)
    print(f"captured={len(frames)} shape={width}x{height} imgsz={imgsz} device={device}")

    serial = _serial(frames, device, half, imgsz)
    serial = _normalise(serial)
    if not any(serial):
        print("RESULT: INCONCLUSIVE — no tracked detections; capture a sequence containing animals")
        return 2
    if args.sequential_batch:
        batched = _sequential_batch(frames, device, half, imgsz)
        mismatches = [
            i for i, (a, b) in enumerate(zip_longest(serial, _normalise(batched)))
            if a != b
        ]
        print(f"serial_vs_sequential_batch mismatches={mismatches}")
        if mismatches:
            return 1
        print("RESULT: PASS — consecutive-frame batch preserved boxes, confidence, class, and tracker output")
        return 0

    lane_a, lane_b = _batch(frames, device, half, imgsz)
    mismatches_a = [i for i, (a, b) in enumerate(zip_longest(serial, _normalise(lane_a))) if a != b]
    mismatches_b = [i for i, (a, b) in enumerate(zip_longest(serial, _normalise(lane_b))) if a != b]
    print(f"serial_vs_batch_lane_a mismatches={mismatches_a}")
    print(f"serial_vs_batch_lane_b mismatches={mismatches_b}")
    if mismatches_a or mismatches_b:
        return 1
    print("RESULT: PASS — same-shape batch preserved boxes, confidence, class, and tracker output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
