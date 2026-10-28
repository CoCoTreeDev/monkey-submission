"""Human-in-the-loop ReID labels stay separate from production detections.

This exercises the real candidate and write handlers against a scratch SQLite
database. A reviewer may label two tracklets, but the action must never mutate
the detection rows or allow a cross-clip identity claim.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

_TMP = tempfile.TemporaryDirectory()
_DB_PATH = Path(_TMP.name) / "reid_labels.db"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_DB_PATH}"

import main  # noqa: E402
from database import Base  # noqa: E402
from models import Detection, ReIDPairLabel  # noqa: E402
from schemas import ReIDLabelCreate  # noqa: E402
from services.reid import reid_geometry_features, reid_review_priority, reid_system_decision  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

FAILS: list[str] = []
T0 = datetime(2026, 8, 31, 10, 0, 0)


def check(condition: bool, label: str) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + label)
    if not condition:
        FAILS.append(label)


def row(
    det_id: int,
    *,
    trail: str = "shared_trail.mp4",
    camera: str = "cam-a",
    start: float | None = 0.0,
    end: float | None = 1.0,
    boxes: list | None = None,
) -> Detection:
    return Detection(
        id=det_id,
        camera_id=camera,
        timestamp=T0 + timedelta(seconds=det_id),
        confidence=0.8 + det_id / 1000,
        bbox_x=10.0,
        bbox_y=20.0,
        bbox_w=50.0,
        bbox_h=60.0,
        image_path=f"{det_id}.jpg",
        annotated_path=f"{det_id}_annotated.jpg",
        crop_path=f"{det_id}_crop.jpg",
        track_id=100 + det_id,
        trail_path=trail,
        clip_duration_seconds=10.0,
        clip_offset_seconds=start,
        clip_end_offset_seconds=end,
        clip_segments=[[start, end]] if start is not None and end is not None else None,
        track_boxes=boxes if boxes is not None else [[0, 10, 20, 60, 80, 0.8]],
        track_box_fps=10.0,
    )


async def run() -> None:
    sample_candidates = [
        {"pair_id": f"{index}:{index + 1}", "trail_path": f"clip-{index % 3}"}
        for index in range(405)
    ]
    sample = main._select_reid_batch(sample_candidates, 400)
    repeat = main._select_reid_batch(sample_candidates, 400)
    check(len(sample) == 400, "the review batch is capped at the requested size")
    check(sample == repeat, "the review batch selection is deterministic")
    check({candidate["trail_path"] for candidate in sample} == {"clip-0", "clip-1", "clip-2"},
          "the review batch covers multiple clips")
    uneven = [
        {"pair_id": pair_id, "trail_path": trail_path}
        for trail_path, pair_id in [
            ("clip-c", "9:10"), ("clip-a", "2:3"), ("clip-b", "5:6"),
            ("clip-a", "3:4"), ("clip-b", "4:5"), ("clip-a", "1:2"),
            ("clip-a", "6:7"),
        ]
    ]
    check(
        [candidate["pair_id"] for candidate in main._select_reid_batch(uneven, 6)]
        == ["1:2", "4:5", "9:10", "2:3", "5:6", "3:4"],
        "uneven clips keep round-robin order after a short clip is exhausted",
    )

    obvious_different = {
        "temporal_relation": "overlap",
        "a": {"track_boxes": [[frame, 0, 0, 10, 10, 0.9] for frame in range(3)]},
        "b": {"track_boxes": [[frame, 100, 0, 110, 10, 0.9] for frame in range(3)]},
    }
    geometry = reid_geometry_features(obvious_different["a"], obvious_different["b"])
    check(geometry["common_frames"] == 3 and geometry["max_iou"] == 0,
          "geometry measures shared frames and separated boxes")
    system_label, reason = reid_system_decision(obvious_different)
    check(system_label == "different" and reason is not None,
          "obvious simultaneous negatives get a system-only different decision")

    ambiguous = {
        **obvious_different,
        "b": {"track_boxes": [[frame, 5, 0, 15, 10, 0.9] for frame in range(3)]},
    }
    check(reid_system_decision(ambiguous) == (None, None),
          "close overlapping boxes remain human-review candidates")
    priority, priority_reason = reid_review_priority(ambiguous)
    check(priority >= 45 and priority_reason, "ambiguous candidates receive a review priority")
    separate = {
        "temporal_relation": "separate",
        "gap_seconds": 0.2,
        "a": {"track_boxes": [[0, 0, 0, 10, 10, 0.9]]},
        "b": {"track_boxes": [[1, 12, 0, 22, 10, 0.9]]},
    }
    check(reid_system_decision(separate) == (None, None),
          "the system never auto-labels a separated pair as same")

    engine = create_async_engine(f"sqlite+aiosqlite:///{_DB_PATH}", echo=False)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)

    # Three rows in one clip make three candidates. The other clip and the
    # malformed cross-camera row must never enter the queue.
    detections = [
        row(1, start=0.0, end=2.0, boxes=[[0, 1, 2, 3, 4, 0.8]] * 3),
        row(2, start=1.0, end=3.0, boxes=[[1, 1, 2, 3, 4, 0.8]] * 2),
        row(3, start=6.0, end=7.0),
        row(4, trail="another_trail.mp4"),
        row(5, camera="cam-b", start=4.0, end=5.0),
        row(6, trail="", start=8.0, end=9.0),
        row(
            7,
            trail="obvious_different.mp4",
            start=0.0,
            end=2.0,
            boxes=[[frame, 0, 0, 10, 10, 0.9] for frame in range(3)],
        ),
        row(
            8,
            trail="obvious_different.mp4",
            start=1.0,
            end=3.0,
            boxes=[[frame, 100, 0, 110, 10, 0.9] for frame in range(1, 3)],
        ),
    ]
    async with maker() as session:
        session.add_all(detections)
        await session.commit()

    async with maker() as session:
        pending = await main.get_reid_pairs(
            status="pending", limit=50, offset=0, review_size=400, session=session
        )
        check(pending.total == 3, f"three unresolved pairs stay in the human queue (got {pending.total})")
        check(pending.candidate_total == 4 and pending.batch_total == 4,
              "small candidate pools stay entirely in the review batch")
        check(pending.batch_pending == 3 and pending.batch_labeled == 0 and pending.batch_auto_resolved == 1,
              "the review batch separates human work from system-only decisions")
        check(pending.pending == 3 and pending.labeled == 0 and pending.auto_resolved == 1,
              "new pairs expose pending, human, and automatic counts separately")
        pair_by_id = {pair.pair_id: pair for pair in pending.pairs}
        check(pair_by_id["1:2"].temporal_relation == "overlap", "overlapping tracklets are marked as overlap")
        check(pair_by_id["1:3"].gap_seconds == 4.0, "separate tracklets expose the measured gap")
        check(pair_by_id["1:2"].a.frame_count == 3, "candidate includes persisted frame count")
        check(pair_by_id["1:2"].a.track_boxes == [[0, 1, 2, 3, 4, 0.8]] * 3,
              "candidate includes per-frame boxes for video review")
        check(pair_by_id["1:2"].a.track_box_fps == 10.0,
              "candidate includes the video box time base")
        check("4" not in " ".join(pair_by_id), "a different clip is not paired")

        auto = await main.get_reid_pairs(
            status="auto", limit=50, offset=0, review_size=400, session=session
        )
        check(auto.total == 1 and auto.pairs[0].pair_id == "7:8",
              "system-only filter returns the obvious negative")
        check(auto.pairs[0].system_label == "different" and auto.pairs[0].label is None,
              "automatic decisions stay separate from human labels")

        before = {
            detection.id: (
                detection.animal_id,
                detection.track_id,
                detection.trail_path,
                detection.clip_segments,
            )
            for detection in detections
        }
        saved = await main.save_reid_label(
            ReIDLabelCreate(
                detection_a_id=2,
                detection_b_id=1,
                label="same",
                note="畫面姿態連續",
            ),
            session=session,
        )
        check(saved.detection_a_id == 1 and saved.detection_b_id == 2, "pair ids are normalized on write")
        check(saved.label == "same" and saved.note == "畫面姿態連續", "the human verdict and note are persisted")

        unchanged = {
            detection.id: (
                detection.animal_id,
                detection.track_id,
                detection.trail_path,
                detection.clip_segments,
            )
            for detection in (
                await session.execute(select(Detection).where(Detection.id.in_([1, 2])))
            ).scalars().all()
        }
        check(unchanged == {key: before[key] for key in (1, 2)}, "labeling does not mutate detection rows")

        after_save = await main.get_reid_pairs(
            status="pending", limit=50, offset=0, review_size=400, session=session
        )
        check(after_save.total == 2 and after_save.pending == 2 and after_save.labeled == 1,
              "saving one verdict removes only that pair from the pending queue")
        check(after_save.batch_total == 4 and after_save.batch_pending == 2 and after_save.batch_labeled == 1 and after_save.batch_auto_resolved == 1,
              "saving a verdict shrinks the fixed batch without refilling it")
        labeled = await main.get_reid_pairs(
            status="labeled", limit=50, offset=0, review_size=400, session=session
        )
        check(labeled.total == 1 and labeled.pairs[0].pair_id == "1:2", "labeled filter returns the saved pair")

        updated = await main.save_reid_label(
            ReIDLabelCreate(
                detection_a_id=1,
                detection_b_id=2,
                label="different",
                note="重新檢視後判定不同隻",
            ),
            session=session,
        )
        check(updated.label == "different", "the same pair can be corrected without a duplicate row")
        count = (await session.execute(select(ReIDPairLabel))).scalars().all()
        check(len(count) == 1, "a corrected pair still occupies one label record")

        try:
            await main.save_reid_label(
                ReIDLabelCreate(detection_a_id=1, detection_b_id=4, label="same"),
                session=session,
            )
        except main.HTTPException as exc:
            check(exc.status_code == 400, "cross-clip labels are rejected")
        else:
            check(False, "cross-clip labels are rejected")

        try:
            await main.save_reid_label(
                ReIDLabelCreate(detection_a_id=1, detection_b_id=5, label="same"),
                session=session,
            )
        except main.HTTPException as exc:
            check(exc.status_code == 400, "cross-camera labels are rejected")
        else:
            check(False, "cross-camera labels are rejected")

        try:
            await main.save_reid_label(
                ReIDLabelCreate(detection_a_id=1, detection_b_id=1, label="uncertain"),
                session=session,
            )
        except main.HTTPException as exc:
            check(exc.status_code == 400, "self-pairs are rejected")
        else:
            check(False, "self-pairs are rejected")

        try:
            await main.save_reid_label(
                ReIDLabelCreate(detection_a_id=1, detection_b_id=999, label="uncertain"),
                session=session,
            )
        except main.HTTPException as exc:
            check(exc.status_code == 404, "missing detections are rejected")
        else:
            check(False, "missing detections are rejected")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run())
    if FAILS:
        print("\nFAILED:")
        for failure in FAILS:
            print(f"- {failure}")
        raise SystemExit(1)
    print("\nAll ReID labeling checks passed.")
