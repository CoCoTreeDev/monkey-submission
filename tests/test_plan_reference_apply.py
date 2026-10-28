from plan_reference_apply import build_apply_plan


def _reference(*, missing=False, unresolved=False):
    clips = [
        {"trail_path": "a_trail.mp4", "row_ids": [1, 2]},
        {"trail_path": "b_trail.mp4", "row_ids": [3]},
    ]
    return {
        "reference_clips": 2,
        "clips": clips,
        "missing_sources": ([{"trail_path": "b_trail.mp4"}] if missing else []),
        "review_required": ([{"left": "a_trail.mp4", "right": "b_trail.mp4"}] if unresolved else []),
    }


def _manifest(*, include_b=True):
    clips = [{
        "clip_id": "reference-0001",
        "status": "ready",
        "source_kind": "rawframes",
        "reference_source_trails": ["a_trail.mp4"],
        "output_path": "a_rebuilt.mp4",
        "poster_path": "a_rebuilt_poster.jpg",
        "track_count": 1,
        "rows": [{
            "track_id": 1,
            "animal_id": "reference-0001-animal-01",
            "confidence": 0.9,
            "bbox": [1, 2, 10, 20],
            "clip_offset_seconds": 0.0,
            "clip_end_offset_seconds": 1.0,
            "clip_duration_seconds": 2.0,
            "clip_segments": [[0.0, 1.0]],
            "boxes": [[0, 1, 2, 10, 20, 0.9]],
            "image_path": "a_raw.jpg",
            "annotated_path": "a_annotated.jpg",
            "track_box_fps": 10.0,
        }],
    }]
    if include_b:
        clips.append({
            "clip_id": "reference-0002",
            "status": "ready",
            "source_kind": "rawframes",
            "reference_source_trails": ["b_trail.mp4"],
            "output_path": "b_rebuilt.mp4",
            "poster_path": "b_rebuilt_poster.jpg",
            "track_count": 0,
            "rows": [],
        })
    return {"run_id": "run-1", "clips": clips, "failures": []}


def test_plan_archives_each_source_row_and_carries_track_geometry():
    plan = build_apply_plan(_manifest(), _reference())

    assert plan["apply_allowed"] is True
    assert plan["old_detection_ids_to_archive"] == [1, 2, 3]
    assert plan["replacement_row_count"] == 1
    assert plan["replacement_rows"][0]["track_boxes"] == [[0, 1, 2, 10, 20, 0.9]]
    assert plan["replacement_rows"][0]["image_path"] == "a_raw.jpg"
    assert plan["replacement_rows"][0]["annotated_path"] == "a_annotated.jpg"
    assert plan["rollback_required"] is True


def test_failed_source_requires_explicit_skip_policy():
    plan = build_apply_plan(_manifest(include_b=False), _reference(missing=True))
    assert plan["apply_allowed"] is False
    assert any("skip_failed" in error for error in plan["errors"])

    plan = build_apply_plan(_manifest(include_b=False), _reference(missing=True), skip_failed=True)
    assert plan["apply_allowed"] is True
    assert plan["skipped_sources"] == ["b_trail.mp4"]
    assert plan["old_detection_ids_to_archive"] == [1, 2]


def test_unresolved_overlap_blocks_apply_even_when_outputs_are_valid():
    plan = build_apply_plan(_manifest(), _reference(unresolved=True))
    assert plan["apply_allowed"] is False
    assert plan["errors"] == ["1 unresolved overlap candidates"]
