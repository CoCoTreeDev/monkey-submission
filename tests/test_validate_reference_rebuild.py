"""Regression tests for the video-first rebuild acceptance gate."""

from validate_reference_rebuild import validate


def _clip(source: str, output: str, animal: str = "a1") -> dict:
    return {
        "clip_id": output,
        "camera_id": "cam-a",
        "reference_source_trails": [source],
        "output_path": output,
        "rawframes_path": output.replace("_trail.mp4", "_rawframes.zip"),
        "poster_path": output.replace(".mp4", "_poster.jpg"),
        "source_kind": "rawframes",
        "status": "ready",
        "source_frames": 10,
        "duration_seconds": 1.0,
        "track_count": 1,
        "rows": [{
            "animal_id": animal,
            "track_id": 1,
            "clip_offset_seconds": 0.0,
            "clip_end_offset_seconds": 0.9,
            "clip_segments": [[0.0, 0.9]],
            "boxes": [
                [0, 1, 1, 4, 4, 0.9],
                [5, 1, 1, 4, 4, 0.9],
                [9, 1, 1, 4, 4, 0.9],
            ],
            "track_box_fps": 10.0,
            "image_path": f"{output}_raw.jpg",
            "annotated_path": f"{output}_annotated.jpg",
        }],
    }


def test_reference_source_cannot_belong_to_two_outputs() -> None:
    result = validate({"clips": [_clip("same.mp4", "one.mp4"), _clip("same.mp4", "two.mp4")]}, expected={}, check_issue_cases=False)

    assert "reference source assigned to multiple canonical clips" in result["structural_failures"]
    assert result["apply_allowed"] is False


def test_valid_clip_has_no_structural_failures() -> None:
    result = validate({"clips": [_clip("source.mp4", "one.mp4")]}, expected={}, check_issue_cases=False)

    assert result["structural_failures"] == []
    assert result["apply_allowed"] is True


def test_rebuild_row_without_raw_preview_is_rejected() -> None:
    clip = _clip("source.mp4", "one.mp4")
    clip["rows"][0].pop("image_path")
    result = validate({"clips": [clip]}, expected={}, check_issue_cases=False)

    assert any("missing raw/annotated preview" in error for error in result["structural_failures"])
    assert result["apply_allowed"] is False


def test_short_track_is_rejected() -> None:
    clip = _clip("source.mp4", "one.mp4")
    clip["rows"][0]["boxes"] = [
        [0, 1, 1, 4, 4, 0.9],
        [1, 1, 1, 4, 4, 0.9],
    ]
    clip["rows"][0]["clip_end_offset_seconds"] = 0.1
    clip["rows"][0]["clip_segments"] = [[0.0, 0.1]]
    result = validate({"clips": [clip]}, expected={}, check_issue_cases=False)

    assert any("short track" in error for error in result["structural_failures"])
    assert result["apply_allowed"] is False
