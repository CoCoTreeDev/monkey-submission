"""The delete guard: a reviewed detection that no delete path may take.

Real endpoints, real SQLAlchemy, real SQLite — only the blob store is a stub, so
what runs here is exactly the code the dashboard calls. No model, no camera, no
network.

Why this exists: 影像管理 hard-deletes with no undo, and the only thing between a
confirmed true positive and permanent loss was a `window.confirm`. A record a
human has reviewed is the most expensive row in the table. These tests hold the
line that both delete paths refuse it and its blobs survive.
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
_DB_PATH = Path(_TMP.name) / "lock_test.db"
import os  # noqa: E402

os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_DB_PATH}"

import main  # noqa: E402
from database import Base  # noqa: E402
from models import Detection  # noqa: E402
from schemas import DetectionOut  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession, async_sessionmaker, create_async_engine,
)

FAILS: list[str] = []
T0 = datetime(2026, 8, 17, 9, 0, 0)


def check(cond: bool, label: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILS.append(label)


class FakeStorage:
    """The calls the delete paths make, over a set of names.

    `download` is here so clear_snapshots exercises the real storage read
    path rather than a stub — the clips are the blobs the review UI
    prefers, and they are the ones a naive keep-set drops.
    """

    def __init__(self, names: set[str]) -> None:
        self.names = set(names)

    def download(self, name: str) -> bytes | None:
        return None

    def delete(self, name: str) -> bool:
        if name in self.names:
            self.names.discard(name)
            return True
        return False

    def delete_all(self) -> int:
        n = len(self.names)
        self.names.clear()
        return n

    def list_files(self) -> list[tuple[str, int]]:
        return [(n, 1) for n in sorted(self.names)]


def row(det_id: int, *, secs: float = 0.0, trail: str | None = None,
        locked: bool = False, conf: float = 0.8, camera: str = "cam",
        when: datetime | None = None) -> Detection:
    """A detection with the full blob set the real ones carry."""
    return Detection(
        id=det_id, camera_id=camera,
        timestamp=(when or T0) + timedelta(seconds=secs),
        confidence=conf, bbox_x=10.0, bbox_y=10.0, bbox_w=50.0, bbox_h=50.0,
        image_path=f"{det_id}.jpg", annotated_path=f"{det_id}_annot.jpg",
        crop_path=f"{det_id}_crop.jpg", track_id=det_id,
        trail_path=trail if trail is not None else f"{det_id}_trail.mp4",
        locked=locked,
    )


def files_of(det: Detection) -> set[str]:
    return main._detection_files(det)


class Fixture:
    """A fresh DB + blob store seeded with the given rows."""

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
        blobs: set[str] = set()
        for det in self.rows:
            blobs |= files_of(det)
        self.storage = FakeStorage(blobs)
        self._real_storage = main.storage
        main.storage = self.storage
        # (`from services.storage import storage`), so patching main.storage
        # alone leaves it reading the real container.
        import services.storage as _storage_mod
        self._storage_mod = _storage_mod
        self._real_mod_storage = _storage_mod.storage
        _storage_mod.storage = self.storage
        return self

    async def __aexit__(self, *exc) -> None:
        main.storage = self._real_storage
        self._storage_mod.storage = self._real_mod_storage
        await self.engine.dispose()

    def session(self):
        return self.maker()

    async def ids(self) -> list[int]:
        async with self.session() as s:
            from sqlalchemy import select
            return sorted((await s.execute(select(Detection.id))).scalars().all())

    async def locked_ids(self) -> list[int]:
        async with self.session() as s:
            from sqlalchemy import select
            return sorted(
                (await s.execute(
                    select(Detection.id).where(Detection.locked.is_(True))
                )).scalars().all()
            )


# ── delete-selected ─────────────────────────────────────────────────────────

async def test_delete_selected_refuses_locked_and_keeps_its_blobs() -> None:
    print("\ndelete-selected — a locked row survives a mixed selection")
    kept, gone = row(1, locked=True), row(2)
    async with Fixture([kept, gone]) as fx:
        async with fx.session() as s:
            result = await main.delete_selected(
                main.DeleteSelectRequest(ids=[1, 2]), session=s
            )
        check(result["deleted_detections"] == 1,
              f"only the unlocked row is deleted (got {result['deleted_detections']})")
        check(result["refused_locked"] == [1],
              f"the locked id is named in refused_locked (got {result['refused_locked']})")
        check(await fx.ids() == [1], "the locked row is still in the table")
        surviving = files_of(kept)
        check(surviving <= fx.storage.names,
              f"every blob of the locked row survives (missing {sorted(surviving - fx.storage.names)})")
        check(not (files_of(gone) & fx.storage.names),
              "the false positive's blobs are gone — the lock costs nothing else")


async def test_delete_selected_all_locked_is_a_no_op() -> None:
    print("\ndelete-selected — an all-locked selection deletes nothing at all")
    a, b = row(1, locked=True), row(2, locked=True)
    async with Fixture([a, b]) as fx:
        before = set(fx.storage.names)
        async with fx.session() as s:
            result = await main.delete_selected(
                main.DeleteSelectRequest(ids=[1, 2]), session=s
            )
        check(result["deleted_detections"] == 0 and result["deleted_files"] == 0,
              "nothing is reported deleted")
        check(sorted(result["refused_locked"]) == [1, 2],
              f"both ids are refused (got {result['refused_locked']})")
        check(fx.storage.names == before, "not one blob was touched")
        check(await fx.ids() == [1, 2], "both rows are still there")


async def test_locked_row_protects_a_shared_trail_clip() -> None:
    """Overlapping tracks share one trail MP4 + rawframes ZIP."""
    print("\ndelete-selected — a clip shared with a locked row is not reclaimed")
    shared = "group_trail.mp4"
    kept, gone = row(1, trail=shared, locked=True), row(2, trail=shared)
    async with Fixture([kept, gone]) as fx:
        check("group_rawframes.zip" in fx.storage.names,
              "the companion rawframes ZIP is part of the fixture")
        async with fx.session() as s:
            await main.delete_selected(main.DeleteSelectRequest(ids=[2]), session=s)
        check(shared in fx.storage.names,
              "the shared trail MP4 survives on the locked row's account")
        check("group_rawframes.zip" in fx.storage.names,
              "so does its raw-frame ZIP")
        check("2.jpg" not in fx.storage.names,
              "the deleted row's own snapshot is still reclaimed")


async def test_unlocked_delete_is_unchanged() -> None:
    print("\ndelete-selected — with nothing locked, behaviour is exactly as before")
    async with Fixture([row(1), row(2)]) as fx:
        async with fx.session() as s:
            result = await main.delete_selected(
                main.DeleteSelectRequest(ids=[1, 2]), session=s
            )
        check(result["deleted_detections"] == 2, "both rows deleted")
        check(result["refused_locked"] == [], "nothing refused")
        check(fx.storage.names == set(), "every blob reclaimed")
        check(await fx.ids() == [], "table empty")


# ── clear-snapshots ─────────────────────────────────────────────────────────

async def test_clear_snapshots_spares_locked_rows() -> None:
    print("\nclear-snapshots — the one-click nuke steps around locked rows")
    kept, gone = row(1, locked=True), row(2)
    async with Fixture([kept, gone]) as fx:
        async with fx.session() as s:
            result = await main.clear_snapshots(session=s)
        check(result["deleted_detections"] == 1,
              f"only the unlocked row is counted (got {result['deleted_detections']})")
        check(result["kept_locked"] == 1, "the response says one row was kept")
        check(await fx.ids() == [1], "the locked row survived the nuke")
        check(files_of(kept) <= fx.storage.names,
              "and so did every blob it points at")
        check(not (files_of(gone) & fx.storage.names),
              "everything else is gone")


async def test_clear_snapshots_with_no_locks_still_clears_everything() -> None:
    print("\nclear-snapshots — no locks means the fast path, same as before")
    async with Fixture([row(1), row(2)]) as fx:
        async with fx.session() as s:
            result = await main.clear_snapshots(session=s)
        check(result["deleted_detections"] == 2, "both rows deleted")
        check(result["kept_locked"] == 0, "nothing kept")
        check(fx.storage.names == set(), "the container is empty")


# ── the lock endpoints ──────────────────────────────────────────────────────

async def test_lock_bulk_reaches_past_the_200_row_window() -> None:
    """The backfill case: lock everything before the cutoff, ids unknown."""
    print("\nlock-bulk — a date filter locks records the tab cannot display")
    old_a = row(1, when=datetime(2026, 8, 10, 12, 0, 0))
    old_b = row(2, when=datetime(2026, 8, 17, 23, 59, 0))   # cutoff day, late
    new_c = row(3, when=datetime(2026, 8, 18, 0, 1, 0))     # after the cutoff
    async with Fixture([old_a, old_b, new_c]) as fx:
        async with fx.session() as s:
            result = await main.set_detection_lock_bulk(
                main.LockBulkRequest(date_to="2026-08-17", locked=True), session=s
            )
        check(result["updated"] == 2, f"two rows locked (got {result['updated']})")
        check(await fx.locked_ids() == [1, 2],
              f"date_to is inclusive of the whole cutoff day (got {await fx.locked_ids()})")
        check(sorted(result["detection_ids"]) == [1, 2],
              "the response names which rows it changed")
        # And the lock it just set is honoured by delete.
        async with fx.session() as s:
            deleted = await main.delete_selected(
                main.DeleteSelectRequest(ids=[1, 2, 3]), session=s
            )
        check(deleted["deleted_detections"] == 1 and deleted["refused_locked"] == [1, 2],
              "only the post-cutoff row can still be deleted")


async def test_lock_bulk_unlocks_and_filters_narrow() -> None:
    print("\nlock-bulk — unlocking, and the narrower filters")
    a = row(1, camera="cam-a", conf=0.5, locked=True)
    b = row(2, camera="cam-b", conf=0.95, locked=True)
    async with Fixture([a, b]) as fx:
        async with fx.session() as s:
            result = await main.set_detection_lock_bulk(
                main.LockBulkRequest(camera_id="cam-a", locked=False), session=s
            )
        check(result["updated"] == 1 and await fx.locked_ids() == [2],
              "camera_id scopes the unlock to one camera")
        async with fx.session() as s:
            result = await main.set_detection_lock_bulk(
                main.LockBulkRequest(max_confidence=0.9, locked=True), session=s
            )
        check(result["updated"] == 1 and await fx.locked_ids() == [1, 2],
              "max_confidence selects only the marginal row")


async def test_single_lock_patch_round_trip() -> None:
    print("\nPATCH /{id}/lock — one row, both directions")
    async with Fixture([row(1)]) as fx:
        async with fx.session() as s:
            out = await main.set_detection_lock(1, main.LockRequest(locked=True), session=s)
        check(out == {"id": 1, "locked": True}, f"locks and echoes back (got {out})")
        check(await fx.locked_ids() == [1], "the row is locked in the DB")
        async with fx.session() as s:
            out = await main.set_detection_lock(1, main.LockRequest(locked=False), session=s)
        check(out == {"id": 1, "locked": False}, f"unlocks again (got {out})")
        check(await fx.locked_ids() == [], "and the DB agrees")
        async with fx.session() as s:
            try:
                await main.set_detection_lock(999, main.LockRequest(locked=True), session=s)
                check(False, "a missing id raises 404")
            except main.HTTPException as exc:
                check(exc.status_code == 404, f"a missing id raises 404 (got {exc.status_code})")


async def test_lock_and_visibility_are_independent() -> None:
    print("\nlock and user_visible move independently")
    async with Fixture([row(1)]) as fx:
        async with fx.session() as s:
            await main.set_detection_lock(1, main.LockRequest(locked=True), session=s)
            await main.set_detection_visibility(
                1, main.VisibilityRequest(visible=False), session=s
            )
            from sqlalchemy import select
            det = (await s.execute(select(Detection).where(Detection.id == 1))).scalar_one()
            check(det.locked is True and det.user_visible is False,
                  "hiding a record from users does not unlock it")
            await main.set_detection_visibility(
                1, main.VisibilityRequest(visible=None), session=s
            )
            det = (await s.execute(select(Detection).where(Detection.id == 1))).scalar_one()
            check(det.locked is True, "nor does resetting the visibility override")
            await main.set_detection_lock(1, main.LockRequest(locked=False), session=s)
            det = (await s.execute(select(Detection).where(Detection.id == 1))).scalar_one()
            check(det.user_visible is None and det.locked is False,
                  "and unlocking leaves visibility exactly where it was")


async def test_default_is_unlocked_and_it_reaches_the_api_surface() -> None:
    print("\ndefaults and the wire format")
    fresh = Detection(
        camera_id="cam", timestamp=T0, confidence=0.8,
        bbox_x=1.0, bbox_y=1.0, bbox_w=2.0, bbox_h=2.0,
    )
    async with Fixture([]) as fx:
        async with fx.session() as s:
            s.add(fresh)
            await s.commit()
            from sqlalchemy import select
            det = (await s.execute(select(Detection))).scalar_one()
        check(det.locked is False,
              "a row inserted without mentioning `locked` lands unlocked")
        out = DetectionOut.model_validate(det)
        check(out.locked is False,
              "and `locked` is on DetectionOut, so the grid can mark cards "
              "without a second request")

    # A Detection built in Python has locked=None until the INSERT default
    # applies — /api/detections/events validates exactly such hand-built rows,
    # so a strict bool field here is a 500 on that endpoint.
    hand_built = Detection(
        id=7, camera_id="cam", timestamp=T0, confidence=0.8,
        bbox_x=1.0, bbox_y=1.0, bbox_w=2.0, bbox_h=2.0,
        image_path="7.jpg", annotated_path="7_annot.jpg", crop_path="7_crop.jpg",
        track_id=7, trail_path="7_trail.mp4",   # `locked` deliberately unset
    )
    check(hand_built.locked is None, "a pre-INSERT row really does report None")
    check(DetectionOut.model_validate(hand_built).locked is False,
          "NULL validates as unlocked instead of raising")


async def test_unfiltered_unlock_is_refused() -> None:
    print("\nlock-bulk — a filterless unlock cannot disarm the whole table")
    async with Fixture([row(1, locked=True), row(2, locked=True)]) as fx:
        for body, label in (
            (main.LockBulkRequest(locked=False), "no filters at all"),
            (main.LockBulkRequest(locked=False, detection_ids=[]), "an empty id list"),
            (main.LockBulkRequest(locked=False, date_to=""), "a blank date box"),
        ):
            async with fx.session() as s:
                try:
                    await main.set_detection_lock_bulk(body, session=s)
                    check(False, f"{label} is refused")
                except main.HTTPException as exc:
                    check(exc.status_code == 400, f"{label} is refused (400)")
        check(await fx.locked_ids() == [1, 2], "both rows are still locked")
        # Locking everything stays allowed — it only ever adds protection.
        async with fx.session() as s:
            out = await main.set_detection_lock_bulk(
                main.LockBulkRequest(locked=True), session=s
            )
        check(out["updated"] == 2, "a filterless LOCK is still allowed")
        # And a filtered unlock still works.
        async with fx.session() as s:
            out = await main.set_detection_lock_bulk(
                main.LockBulkRequest(locked=False, detection_ids=[1]), session=s
            )
        check(out["updated"] == 1 and await fx.locked_ids() == [2],
              "a filtered unlock works normally")


async def test_bad_dates_are_422_not_500() -> None:
    print("\nlock-bulk — a typo'd date is a 422 naming the field")
    async with Fixture([row(1)]) as fx:
        for bad in ("2026/08/17", "17-08-2026", "not-a-date"):
            async with fx.session() as s:
                try:
                    await main.set_detection_lock_bulk(
                        main.LockBulkRequest(locked=True, date_to=bad), session=s
                    )
                    check(False, f"{bad!r} is rejected")
                except main.HTTPException as exc:
                    check(exc.status_code == 422 and "date_to" in str(exc.detail),
                          f"{bad!r} → 422 naming date_to")
        check(await fx.locked_ids() == [], "nothing was locked by a rejected call")


async def test_locked_validator_does_not_coerce_strings() -> None:
    print("\nDetectionOut.locked — no truthiness coercion")
    base = dict(
        id=1, camera_id="cam", timestamp=T0, confidence=0.8,
        bbox_x=1.0, bbox_y=1.0, bbox_w=2.0, bbox_h=2.0,
        image_path="1.jpg", annotated_path="1a.jpg", crop_path="1c.jpg",
        track_id=1, trail_path="1t.mp4",
    )
    check(DetectionOut.model_validate({**base, "locked": None}).locked is False,
          "None validates to False")
    check(DetectionOut.model_validate({**base, "locked": "false"}).locked is False,
          '"false" is NOT a lock badge on a deletable record')
    check(DetectionOut.model_validate({**base, "locked": True}).locked is True,
          "a real True still locks")


async def main_async() -> None:
    for test in (
        test_delete_selected_refuses_locked_and_keeps_its_blobs,
        test_delete_selected_all_locked_is_a_no_op,
        test_locked_row_protects_a_shared_trail_clip,
        test_unlocked_delete_is_unchanged,
        test_clear_snapshots_spares_locked_rows,
        test_clear_snapshots_with_no_locks_still_clears_everything,
        test_lock_bulk_reaches_past_the_200_row_window,
        test_lock_bulk_unlocks_and_filters_narrow,
        test_single_lock_patch_round_trip,
        test_lock_and_visibility_are_independent,
        test_default_is_unlocked_and_it_reaches_the_api_surface,
        test_unfiltered_unlock_is_refused,
        test_bad_dates_are_422_not_500,
        test_locked_validator_does_not_coerce_strings,
    ):
        await test()


if __name__ == "__main__":
    asyncio.run(main_async())
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL PASS")
