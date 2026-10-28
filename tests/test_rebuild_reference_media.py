"""Regression test: a rebuilt row gets peak-frame raw, annotated, and crop assets."""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rebuild_reference  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        archives = root / "zips"
        output = root / "output"
        archives.mkdir()
        output.mkdir()

        archive_path = archives / "source_rawframes.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for index in range(11):
                frame = np.full((80, 100, 3), 40 + index * 20, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", frame)
                assert ok
                archive.writestr(f"frame_{index:03d}.jpg", encoded.tobytes())

        original_track = rebuild_reference.track_source
        original_merge = rebuild_reference.merge_track_fragments
        original_encode = rebuild_reference.encode
        rebuild_reference.track_source = lambda *_args: ({
            7: [
                [0, 10, 10, 40, 40, 0.9],
                [5, 11, 11, 41, 41, 0.8],
                [10, 12, 12, 42, 42, 0.85],
            ],
        }, {7: []})
        rebuild_reference.merge_track_fragments = lambda tracks, **_kwargs: tracks
        rebuild_reference.encode = lambda frames, _fps, destination: destination.write_bytes(
            b"test-video"
        )
        try:
            result = rebuild_reference.rebuild_clip(
                object(),
                {
                    "canonical_id": "reference-0001",
                    "camera_id": "cam-a",
                    "timestamp": "2026-08-30T10:00:00+08:00",
                    "archive_names": [archive_path.name],
                    "source_trails": [],
                    "canonical_trail_path": "source.mp4",
                },
                archives,
                root,
                output,
                2048,
                0.35,
                root / "bytetrack.yaml",
                1,
                "test-run",
            )
        finally:
            rebuild_reference.track_source = original_track
            rebuild_reference.merge_track_fragments = original_merge
            rebuild_reference.encode = original_encode

        row = result["rows"][0]
        raw_name = Path(row["image_path"]).name
        annotated_name = Path(row["annotated_path"]).name
        crop_name = Path(row["crop_path"]).name
        assert (output / raw_name).is_file()
        assert (output / annotated_name).is_file()
        assert (output / crop_name).is_file()
        assert raw_name != annotated_name
        assert crop_name not in {raw_name, annotated_name}
        assert cv2.imread(str(output / raw_name)).shape == (80, 100, 3)
        assert cv2.imread(str(output / annotated_name)).shape == (80, 100, 3)
        crop = cv2.imread(str(output / crop_name))
        assert crop is not None
        assert crop.shape[0] == 60
        assert crop.shape[1] == 60
        rawframes_name = Path(result["rawframes_path"]).name
        with zipfile.ZipFile(output / rawframes_name) as archive:
            assert len([name for name in archive.namelist() if name.endswith(".jpg")]) == 11
        assert result["poster_path"].endswith("_poster.jpg")
    print("ALL PASS — rebuild raw/annotated/crop media")


def test_short_track_is_not_emitted() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        archives = root / "zips"
        output = root / "output"
        archives.mkdir()
        output.mkdir()

        archive_path = archives / "short_rawframes.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for index in range(3):
                frame = np.full((80, 100, 3), 40 + index * 20, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", frame)
                assert ok
                archive.writestr(f"frame_{index:03d}.jpg", encoded.tobytes())

        original_track = rebuild_reference.track_source
        original_merge = rebuild_reference.merge_track_fragments
        original_encode = rebuild_reference.encode
        rebuild_reference.track_source = lambda *_args: ({
            7: [[0, 10, 10, 40, 40, 0.8], [1, 11, 11, 41, 41, 0.9]],
        }, {7: []})
        rebuild_reference.merge_track_fragments = lambda tracks, **_kwargs: tracks
        rebuild_reference.encode = lambda frames, _fps, destination: destination.write_bytes(
            b"test-video"
        )
        try:
            result = rebuild_reference.rebuild_clip(
                object(),
                {
                    "canonical_id": "short-track",
                    "camera_id": "cam-a",
                    "timestamp": "2026-08-30T10:00:00+08:00",
                    "archive_names": [archive_path.name],
                    "source_trails": [],
                    "canonical_trail_path": "source.mp4",
                },
                archives,
                root,
                output,
                2048,
                0.35,
                root / "bytetrack.yaml",
                1,
                "test-run",
            )
        finally:
            rebuild_reference.track_source = original_track
            rebuild_reference.merge_track_fragments = original_merge
            rebuild_reference.encode = original_encode

        assert result["rows"] == []


def test_rebuild_tracks_monkeys_only():
    import torch
    from ultralytics.engine.results import Results

    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    class Model:
        def track(self, image, **kwargs):
            rows = torch.tensor([[1, 1, 20, 20, 7, .9, 0], [25, 1, 50, 60, 8, .99, 1]])
            if kwargs.get("classes") is not None:
                rows = rows[torch.isin(rows[:, 6], torch.tensor(kwargs["classes"]))]
            return [Results(image, "", {0: "monkey", 1: "human"}, boxes=rows)]

    tracks, appearances = rebuild_reference.track_source(Model(), [frame], 640, .35, Path("unused.yaml"))
    assert set(tracks) == {7}, "human track 8 must never become a rebuilt detection"
    assert set(appearances) == {7}


if __name__ == "__main__":
    main()
    test_rebuild_tracks_monkeys_only()
