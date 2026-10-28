"""The human-gated Reference rebuild replacement and rollback contract."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

_TMP = tempfile.TemporaryDirectory()
_DB_PATH = Path(_TMP.name) / "rebuild_apply.db"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_DB_PATH}"

import main  # noqa: E402
from database import Base  # noqa: E402
from models import Detection, HistoricalRebuildRun  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class FakeStorage:
    def __init__(self, manifest: dict, assets: set[str]) -> None:
        self.manifest = manifest
        self.assets = set(assets)

    def download(self, name: str) -> bytes | None:
        if name == "rebuild-review/latest.json":
            return json.dumps(self.manifest).encode()
        return None

    def exists(self, name: str) -> bool:
        return name in self.assets

    def upload(self, name: str, payload: bytes) -> None:
        if name == "rebuild-review/latest.json":
            self.manifest = json.loads(payload)


def old_detection() -> Detection:
    return Detection(
        id=1,
        camera_id="cam-a",
        timestamp=datetime(2026, 8, 30, 10, 0),
        confidence=0.8,
        bbox_x=1,
        bbox_y=2,
        bbox_w=8,
        bbox_h=18,
        image_path="old.jpg",
        annotated_path="old_a.jpg",
        crop_path="old_c.jpg",
        track_id=99,
        trail_path="source_trail.mp4",
        locked=True,
    )


MANIFEST = {
    "run_id": "reference-rebuild-test",
    "status": "review",
    "reference": {
        "reference_clips": 2,
        "review_required": [],
        "missing_sources": [{"trail_path": "missing_trail.mp4"}],
    },
    "failures": [{"source_trail_path": "missing_trail.mp4", "status": "failed"}],
    "clips": [{
        "clip_id": "reference-0001",
        "camera_id": "cam-a",
        "timestamp": "2026-08-30T10:00:00+08:00",
        "status": "ready",
        "source_kind": "rawframes",
        "reference_source_trails": ["source_trail.mp4"],
        "output_path": "rebuilt_trail.mp4",
        "poster_path": "rebuilt_poster.jpg",
        "duration_seconds": 2.0,
        "fps": 10.0,
        "rows": [{
            "track_id": 1,
            "animal_id": "reference-0001-animal-01",
            "confidence": 0.95,
            "bbox": [10, 20, 30, 50],
            "clip_offset_seconds": 0.0,
            "clip_end_offset_seconds": 1.0,
            "clip_duration_seconds": 2.0,
            "clip_segments": [[0.0, 1.0]],
            "track_box_fps": 10.0,
            "boxes": [[0, 10, 20, 30, 50, 0.95], [10, 11, 21, 31, 51, 0.9]],
            "image_path": "rebuilt_animal_01_raw.jpg",
            "annotated_path": "rebuilt_animal_01_annotated.jpg",
            "crop_path": "",
        }],
    }],
}


async def main_async() -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{_DB_PATH}")
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    async with maker() as session:
        session.add(old_detection())
        await session.commit()

    main.storage = FakeStorage(
        MANIFEST,
        {
            "rebuilt_trail.mp4",
            "rebuilt_poster.jpg",
            "rebuilt_animal_01_raw.jpg",
            "rebuilt_animal_01_annotated.jpg",
            "repaired_raw.jpg",
            "repaired_annotated.jpg",
        },
    )
    async with maker() as session:
        dry = await main.apply_rebuild_review(
            main.RebuildApplyRequest(
                run_id="reference-rebuild-test", skip_failed=True,
            ),
            session=session,
        )
        assert dry["status"] == "blocked"
        assert "allow_locked=true" in dry["errors"][0]

        dry = await main.apply_rebuild_review(
            main.RebuildApplyRequest(
                run_id="reference-rebuild-test", skip_failed=True,
                allow_locked=True,
            ),
            session=session,
        )
        assert dry["dry_run"] is True
        assert dry["old_detection_count"] == 1
        assert dry["created_new_rows"] == 1

    async with maker() as session:
        before = list((await session.execute(select(Detection))).scalars())
        assert [row.id for row in before] == [1]

        applied = await main.apply_rebuild_review(
            main.RebuildApplyRequest(
                run_id="reference-rebuild-test", dry_run=False,
                confirm=True, skip_failed=True, allow_locked=True,
            ),
            session=session,
        )
        assert applied["status"] == "accepted"
        assert applied["created_new_rows"] == 1

    async with maker() as session:
        rows = list((await session.execute(select(Detection))).scalars())
        assert len(rows) == 1
        assert rows[0].trail_path == "rebuilt_trail.mp4"
        assert rows[0].image_path == "rebuilt_animal_01_raw.jpg"
        assert rows[0].annotated_path == "rebuilt_animal_01_annotated.jpg"
        assert rows[0].track_boxes[0][0] == 0
        assert rows[0].locked is False

        rows[0].locked = True
        await session.commit()
        media = await main.update_detection_media_bulk(
            main.DetectionMediaBulkRequest(items=[
                main.DetectionMediaItem(
                    detection_id=rows[0].id,
                    image_path="repaired_raw.jpg",
                    annotated_path="repaired_annotated.jpg",
                ),
            ]),
            session=session,
        )
        assert media["updated"] == 1
        await session.refresh(rows[0])
        assert rows[0].image_path == "repaired_raw.jpg"
        assert rows[0].annotated_path == "repaired_annotated.jpg"
        assert rows[0].locked is True

        dry_rollback = await main.rollback_rebuild_review(
            main.RebuildRollbackRequest(run_id="reference-rebuild-test"),
            session=session,
        )
        assert dry_rollback["restore_count"] == 1

        rolled_back = await main.rollback_rebuild_review(
            main.RebuildRollbackRequest(
                run_id="reference-rebuild-test", dry_run=False, confirm=True,
            ),
            session=session,
        )
        assert rolled_back["status"] == "rolled_back"

    async with maker() as session:
        rows = list((await session.execute(select(Detection))).scalars())
        assert len(rows) == 1
        assert rows[0].id == 1
        assert rows[0].trail_path == "source_trail.mp4"
        run = (await session.execute(select(HistoricalRebuildRun))).scalar_one()
        assert run.status == "rolled_back"

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main_async())
