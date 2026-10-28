"""Unit tests for the conservative Reference manifest builder."""

import zipfile
from pathlib import Path

import numpy as np

from prepare_reference_rebuild import build_reference, visual_alignment


def _write_archive(root: Path, name: str, frames: list[bytes]) -> None:
    with zipfile.ZipFile(root / name, "w") as archive:
        for index, frame in enumerate(frames):
            archive.writestr(f"frame_{index:04d}.jpg", frame)


def _row(row_id: int, trail: str, start: str, duration: float = 4.0) -> dict:
    return {
        "id": row_id,
        "camera_id": "cam-a",
        "timestamp": start,
        "clip_duration_seconds": duration,
        "trail_path": trail,
    }


def test_exact_shared_frames_make_one_canonical_clip(tmp_path: Path) -> None:
    archive_root = tmp_path / "zips"
    archive_root.mkdir()
    _write_archive(archive_root, "a_rawframes.zip", [b"a", b"b", b"c"])
    _write_archive(archive_root, "b_rawframes.zip", [b"b", b"c", b"d"])
    rows = [
        _row(1, "a_trail.mp4", "2026-08-30T10:00:00+08:00"),
        _row(2, "b_trail.mp4", "2026-08-30T10:00:01+08:00"),
    ]

    manifest = build_reference(rows, archive_root)

    assert manifest["canonical_clips"] == 1
    assert manifest["canonical"][0]["source_trails"] == [
        "a_trail.mp4",
        "b_trail.mp4",
    ]
    assert manifest["canonical"][0]["camera_id"] == "cam-a"
    assert manifest["dedupe_candidates"][0]["evidence"] == "exact_frame_overlap"
    assert manifest["review_required"] == []
    assert manifest["apply_allowed"] is False


def test_time_overlap_alone_never_merges(tmp_path: Path) -> None:
    archive_root = tmp_path / "zips"
    archive_root.mkdir()
    _write_archive(archive_root, "a_rawframes.zip", [b"a", b"b", b"c"])
    _write_archive(archive_root, "b_rawframes.zip", [b"x", b"y", b"z"])
    rows = [
        _row(1, "a_trail.mp4", "2026-08-30T10:00:00+08:00"),
        _row(2, "b_trail.mp4", "2026-08-30T10:00:01+08:00"),
    ]

    manifest = build_reference(rows, archive_root)

    assert manifest["canonical_clips"] == 2
    assert manifest["review_required"] == []
    assert manifest["time_overlap_warnings"][0]["evidence"] == "review_required"
    assert manifest["dedupe_candidates"][0]["shared_frames"] == 0


def test_adjacent_clips_with_image_evidence_make_one_canonical_clip(tmp_path: Path) -> None:
    archive_root = tmp_path / "zips"
    archive_root.mkdir()
    _write_archive(archive_root, "cam_tracks1_20260830_100000_a_rawframes.zip", [b"a", b"b", b"c"])
    _write_archive(archive_root, "cam_tracks2_20260830_100010_b_rawframes.zip", [b"b", b"c", b"d"])
    rows = [
        _row(1, "cam_tracks1_20260830_100000_a_trail.mp4", "2026-08-30T10:00:00+08:00"),
        _row(2, "cam_tracks2_20260830_100010_b_trail.mp4", "2026-08-30T10:00:10+08:00"),
    ]

    manifest = build_reference(rows, archive_root)

    assert manifest["canonical_clips"] == 1
    assert manifest["dedupe_candidates"][0]["evidence"] == "exact_frame_overlap"


def test_adjacent_clips_without_image_evidence_stay_separate_without_blocking(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "zips"
    archive_root.mkdir()
    _write_archive(archive_root, "cam_tracks1_20260830_100000_a_rawframes.zip", [b"a", b"b"])
    _write_archive(archive_root, "cam_tracks2_20260830_100010_b_rawframes.zip", [b"x", b"y"])
    rows = [
        _row(1, "cam_tracks1_20260830_100000_a_trail.mp4", "2026-08-30T10:00:00+08:00"),
        _row(2, "cam_tracks2_20260830_100010_b_trail.mp4", "2026-08-30T10:00:10+08:00"),
    ]

    manifest = build_reference(rows, archive_root)

    assert manifest["canonical_clips"] == 2
    assert manifest["dedupe_candidates"][0]["relation"] == "adjacent"
    assert manifest["review_required"] == []


def test_reference_interval_uses_capture_time_in_trail_name(tmp_path: Path) -> None:
    archive_root = tmp_path / "zips"
    archive_root.mkdir()
    _write_archive(archive_root, "cam_tracks12_20260830_100000_abcd_rawframes.zip", [b"a", b"b", b"c"])
    rows = [
        _row(1, "cam_tracks12_20260830_100000_abcd_trail.mp4", "2026-08-30T10:00:02+08:00", 4.0),
        _row(2, "cam_tracks12_20260830_100000_abcd_trail.mp4", "2026-08-30T10:01:00+08:00", 4.0),
    ]

    manifest = build_reference(rows, archive_root)
    clip = manifest["clips"][0]

    assert clip["reference_start"] == "2026-08-30T10:00:00+08:00"
    assert clip["reference_end"] == "2026-08-30T10:00:04+08:00"
    assert clip["reference_start_source"] == "trail_filename"
    assert manifest["review_required"] == []


def test_missing_archive_blocks_apply_and_is_explicit(tmp_path: Path) -> None:
    archive_root = tmp_path / "zips"
    archive_root.mkdir()
    rows = [_row(7, "missing_trail.mp4", "2026-08-30T10:00:00+08:00")]

    manifest = build_reference(rows, archive_root)

    assert manifest["missing_sources"] == [
        {"trail_path": "missing_trail.mp4", "archive": "missing_rawframes.zip"}
    ]
    assert manifest["canonical_clips"] == 1
    assert manifest["apply_allowed"] is False


def test_visual_alignment_accepts_reencoded_frames_at_one_offset() -> None:
    rng = np.random.default_rng(4)
    base = [rng.normal(size=(36, 64)).astype("float32") + index * 0.01 for index in range(12)]
    left = [((frame - frame.mean()) / (frame.std() + 1e-6)) for frame in base]
    right = [((frame - frame.mean()) / (frame.std() + 1e-6)) for frame in base[2:]]

    result = visual_alignment(left, right)

    assert result is not None
    assert result[0] == 2
    assert result[1] >= 6


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_exact_shared_frames_make_one_canonical_clip(Path(directory))
    print("ok prepare_reference_rebuild")
