"""Make a compact raw/rebuilt contact sheet for one canonical clip."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np


def raw_frames(path: Path) -> list[np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        result = []
        for name in sorted(archive.namelist()):
            if not name.lower().endswith((".jpg", ".jpeg")):
                continue
            image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                result.append(image)
        return result


def video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    result = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        result.append(image)
    capture.release()
    return result


def make_sheet(groups: list[tuple[str, list[np.ndarray]]], output: Path) -> None:
    thumb_w, thumb_h = 640, 360
    canvas = np.full((len(groups) * (thumb_h + 24), thumb_w * 3, 3), 24, np.uint8)
    for row, (label, images) in enumerate(groups):
        cv2.putText(canvas, label, (4, row * (thumb_h + 24) + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        if not images:
            continue
        indices = sorted({0, len(images) // 2, len(images) - 1})
        for col, index in enumerate(indices):
            image = cv2.resize(images[index], (thumb_w, thumb_h - 24))
            y = row * (thumb_h + 24) + 24
            canvas[y:y + thumb_h - 24, col * thumb_w:(col + 1) * thumb_w] = image
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_sheet([("raw frames: first / middle / last", raw_frames(args.archive)),
                ("rebuilt video: first / middle / last", video_frames(args.video))], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
