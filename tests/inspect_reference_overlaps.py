"""Render raw-frame contact sheets for Reference overlap candidates."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


def frames(path: Path) -> list[np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.lower().endswith((".jpg", ".jpeg"))
        )
        result = []
        for name in names:
            image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                result.append(image)
        return result


def samples(images: list[np.ndarray]) -> list[np.ndarray]:
    if not images:
        return []
    indices = sorted({0, len(images) // 2, len(images) - 1})
    return [images[index] for index in indices]


def sheet(left: Path, right: Path, output: Path, number: int) -> None:
    columns = [samples(frames(left)), samples(frames(right))]
    height, width = 180, 320
    canvas = np.full((height * 2 + 32, width * 3, 3), 24, np.uint8)
    labels = [f"{number}: {left.stem}", f"{number}: {right.stem}"]
    for row, (label, images) in enumerate(zip(labels, columns)):
        cv2.putText(canvas, label[:105], (4, row * height + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
        for column, image in enumerate(images):
            thumbnail = cv2.resize(image, (width, height - 16))
            y = row * height + 18
            canvas[y:y + height - 16, column * width:(column + 1) * width] = thumbnail
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidates = reference.get("review_required", []) or reference.get("time_overlap_warnings", [])
    for number, candidate in enumerate(candidates, 1):
        left = args.archive_root / candidate["left"].replace("_trail.mp4", "_rawframes.zip")
        right = args.archive_root / candidate["right"].replace("_trail.mp4", "_rawframes.zip")
        sheet(left, right, args.output / f"overlap-{number:02d}.jpg", number)
        print(number, left.name, right.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
