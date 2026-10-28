"""The animal-group backfill: reversible, and a human verdict outranks the rule.

Real endpoint, real SQLAlchemy, real SQLite — the same harness as the lock
tests. What these hold:

  * grouping writes exactly two columns and nothing else, so clear restores
    the ungrouped truth (the reversibility the animal-grouping design requires);
  * a 'rule' request can neither overwrite nor clear a 'human' verdict;
  * a group is a claim about one visit — a stray row is reported, never
    stamped with a cross-visit identity;
  * re-applying the same grouping is a no-op.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

# Point the app at a scratch DB before importing it, so nothing here can touch
# the developer's detections.db.
_TMP = tempfile.TemporaryDirectory()
_DB_PATH = Path(_TMP.name) / "animal_test.db"
import os  # noqa: E402

os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_DB_PATH}"

import main  # noqa: E402
from database import Base  # noqa: E402
from models import Detection  # noqa: E402
from schemas import DetectionOut  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession, async_sessionmaker, create_async_engine,
)

FAILS: list[str] = []
T0 = datetime(2026, 8, 28, 7, 6, 28)


def check(cond: bool, label: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILS.append(label)


def row(det_id: int, *, visit: str | None = "bf-v1", secs: float = 0.0,
        animal: str | None = None, source: str | None = None) -> Detection:
    return Detection(
        id=det_id, camera_id="cam", timestamp=T0 + timedelta(seconds=secs),
        confidence=0.8, bbox_x=10.0, bbox_y=10.0, bbox_w=50.0, bbox_h=50.0,
        image_path=f"{det_id}.jpg", annotated_path=f"{det_id}_a.jpg",
        crop_path=f"{det_id}_c.jpg", track_id=det_id * 100,
        trail_path=f"{det_id}_trail.mp4", visit_id=visit,
        animal_id=animal, animal_source=source,
    )


class Fixture:
    def __init__(self, rows: list[Detection]) -> None:
        self.rows = rows
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{_DB_PATH}", echo=False
        )
        self.maker = async_sessionmaker(self.engine, class_=AsyncSession,
                                        expire_on_commit=False)

    async def __aenter__(self) -> "Fixture":
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        async with self.maker() as session:
            for det in self.rows:
                session.add(det)
            await session.commit()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.engine.dispose()

    def session(self):
        return self.maker()

    async def animal_of(self, det_id: int) -> tuple[str | None, str | None]:
        async with self.session() as s:
            det = (await s.execute(
                select(Detection).where(Detection.id == det_id)
            )).scalar_one()
            return det.animal_id, det.animal_source


def grp(visit: str, ids: list[int]) -> main.AnimalGroup:
    return main.AnimalGroup(visit_id=visit, detection_ids=ids,
                            links=[{"a": "1", "b": "2", "gap": 0.2,
                                    "err_widths": 0.3}])


async def test_stamp_and_reverse() -> None:
    print("\n寫入、可逆性")
    async with Fixture([row(1), row(2), row(3)]) as fx:
        body = main.BackfillAnimalGroupsRequest(groups=[grp("bf-v1", [1, 2])])
        async with fx.session() as s:
            out = await main.backfill_animal_groups(body, session=s)
        check(out["dry_run"] is True and out["rows_stamped"] == 2,
              "dry run 預設開啟，回報會寫的列數")
        check(await fx.animal_of(1) == (None, None), "dry run 沒有寫任何東西")

        body = main.BackfillAnimalGroupsRequest(
            dry_run=False, evidence="test", groups=[grp("bf-v1", [1, 2])])
        async with fx.session() as s:
            out = await main.backfill_animal_groups(body, session=s)
        check(out["rows_stamped"] == 2, "apply 寫入兩列")
        check(await fx.animal_of(1) == ("bf-v1:a1", "rule")
              and await fx.animal_of(2) == ("bf-v1:a1", "rule"),
              "同組兩列拿到同一個 animal_id，來源是 rule")
        check(await fx.animal_of(3) == (None, None), "組外的列不受影響")

        # Only the two columns move; the row's identity stays as emitted.
        async with fx.session() as s:
            det = (await s.execute(
                select(Detection).where(Detection.id == 1)
            )).scalar_one()
        check(det.track_id == 100 and det.visit_id == "bf-v1"
              and det.timestamp == T0, "track_id/visit_id/timestamp 原封不動")

        # Re-applying is a no-op, not a re-stamp.
        async with fx.session() as s:
            out = await main.backfill_animal_groups(body, session=s)
        check(out["rows_stamped"] == 0, "重跑同一份分組是 no-op")

        # And clear puts it back.
        body = main.BackfillAnimalGroupsRequest(
            dry_run=False, clear=True, visit_ids=["bf-v1"])
        async with fx.session() as s:
            out = await main.backfill_animal_groups(body, session=s)
        check(out["rows_cleared"] == 2, "clear 清掉 rule 寫的兩列")
        check(await fx.animal_of(1) == (None, None)
              and await fx.animal_of(2) == (None, None),
              "清掉之後回到未分組的原狀")


async def test_human_outranks_rule() -> None:
    print("\nhuman 裁決優先")
    async with Fixture([
        row(1, animal="bf-v1:a1", source="human"),
        row(2, animal="bf-v1:a1", source="human"),
        row(3),
    ]) as fx:
        # A rule request may not move a human-confirmed row into its group…
        body = main.BackfillAnimalGroupsRequest(
            dry_run=False, groups=[grp("bf-v1", [2, 3])])
        async with fx.session() as s:
            out = await main.backfill_animal_groups(body, session=s)
        check(out["rows_stamped"] == 0 and 2 in out["kept_human"],
              "rule 不改 human 標過的列，且該組只剩一列時整組不寫")
        check(await fx.animal_of(2) == ("bf-v1:a1", "human"), "human 裁決原樣")

        # …nor clear it.
        body = main.BackfillAnimalGroupsRequest(
            dry_run=False, clear=True, visit_ids=["bf-v1"])
        async with fx.session() as s:
            out = await main.backfill_animal_groups(body, session=s)
        check(out["rows_cleared"] == 0 and out["kept_human"] == 2,
              "rule 的 clear 留下 human 列")

        # A human request can regroup them. Rows already carrying the exact
        # verdict are left alone — only the newcomer is stamped — and the end
        # state is what matters: all three share the id.
        body = main.BackfillAnimalGroupsRequest(
            dry_run=False, source="human", groups=[grp("bf-v1", [1, 2, 3])])
        async with fx.session() as s:
            out = await main.backfill_animal_groups(body, session=s)
        check(out["rows_stamped"] == 1
              and await fx.animal_of(3) == ("bf-v1:a1", "human"),
              "human 請求可以把新列併進 human 組（已標對的列不重寫）")


async def test_refusals() -> None:
    print("\n拒絕的部分")
    async with Fixture([row(1), row(2), row(3, visit="bf-v2"), row(4)]) as fx:
        # A row from another visit is reported, never stamped.
        body = main.BackfillAnimalGroupsRequest(
            dry_run=False, groups=[grp("bf-v1", [1, 2, 3])])
        async with fx.session() as s:
            out = await main.backfill_animal_groups(body, session=s)
        check([w["id"] for w in out["rows_in_wrong_visit"]] == [3],
              "跨造訪的列被點名")
        check(await fx.animal_of(3) == (None, None), "而且沒有被蓋章")
        check(out["rows_stamped"] == 2, "同造訪的其他列照常分組")

        # One row in two groups is a caller bug, not a tiebreak.
        body = main.BackfillAnimalGroupsRequest(
            groups=[grp("bf-v1", [1, 2]), grp("bf-v1", [2, 4])])
        async with fx.session() as s:
            try:
                await main.backfill_animal_groups(body, session=s)
                check(False, "同一列進兩組被拒絕")
            except main.HTTPException as exc:
                check(exc.status_code == 400, "同一列進兩組被拒絕 (400)")

        # A missing row is reported by id.
        body = main.BackfillAnimalGroupsRequest(groups=[grp("bf-v1", [1, 99])])
        async with fx.session() as s:
            out = await main.backfill_animal_groups(body, session=s)
        check(out["missing_rows"] == [99], "不存在的列被點名")

        # A group of one is refused at the schema, clear needs visit ids.
        try:
            main.AnimalGroup(visit_id="bf-v1", detection_ids=[1])
            check(False, "單列的分組被 schema 拒絕")
        except Exception:
            check(True, "單列的分組被 schema 拒絕")
        body = main.BackfillAnimalGroupsRequest(dry_run=False, clear=True)
        async with fx.session() as s:
            try:
                await main.backfill_animal_groups(body, session=s)
                check(False, "clear 沒給 visit_ids 被拒絕")
            except main.HTTPException as exc:
                check(exc.status_code == 400, "clear 沒給 visit_ids 被拒絕 (400)")


async def test_wire_format() -> None:
    print("\nAPI 表面")
    det = row(1, animal="bf-v1:a1", source="rule")
    det.track_boxes = [[0, 10.0, 11.0, 60.0, 61.0, 0.91]]
    det.track_box_fps = 10.0
    out = DetectionOut.model_validate(det)
    check(out.animal_id == "bf-v1:a1" and out.animal_source == "rule",
          "animal_id/animal_source 出現在 DetectionOut")
    check(out.track_boxes == det.track_boxes and out.track_box_fps == 10.0,
          "逐幀 boxes/fps 出現在 DetectionOut")
    bare = DetectionOut.model_validate(row(2))
    check(bare.animal_id is None and bare.animal_source is None,
          "沒分組的列回 NULL")


async def _main() -> None:
    await test_stamp_and_reverse()
    await test_human_outranks_rule()
    await test_refusals()
    await test_wire_format()
    print()
    if FAILS:
        print(f"{len(FAILS)} 個失敗"); sys.exit(1)
    print("全部通過")


if __name__ == "__main__":
    asyncio.run(_main())
