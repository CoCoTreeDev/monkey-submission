"""Profile one raw clip at several confidence thresholds on the mini."""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def load_frames(path: Path) -> list[np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        result = []
        for name in sorted(archive.namelist()):
            if name.lower().endswith((".jpg", ".jpeg")):
                image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    result.append(image)
        return result


def run(model: YOLO, frames: list[np.ndarray], threshold: float, tracker: Path) -> dict:
    model.predictor = None
    tracks: dict[int, list[int]] = defaultdict(list)
    per_frame = []
    for number, frame in enumerate(frames):
        result = model.track(frame, imgsz=2048, conf=threshold, device="mps",
                             persist=True, tracker=str(tracker), verbose=False)[0]
        boxes = result.boxes
        count = 0 if boxes is None or not len(boxes) or boxes.id is None else len(boxes)
        per_frame.append(count)
        if boxes is not None and len(boxes) and boxes.id is not None:
            for track_id in boxes.id.tolist():
                tracks[int(track_id)].append(number)
    return {
        "conf": threshold,
        "track_count": len(tracks),
        "track_lengths": sorted((len(v) for v in tracks.values()), reverse=True),
        "max_detections_in_frame": max(per_frame, default=0),
        "frames_with_detections": sum(value > 0 for value in per_frame),
        "per_frame": per_frame,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = load_frames(args.archive)
    model = YOLO(str(args.model))
    model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=2048, device="mps", verbose=False)
    backend = getattr(model.predictor, "model", None)
    if backend is not None and hasattr(backend, "model"):
        backend.model.half()
        backend.fp16 = True
    result = {"frames": len(frames), "runs": [
        run(model, frames, threshold, args.tracker)
        for threshold in (0.5, 0.35, 0.25, 0.1, 0.01)
    ]}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
