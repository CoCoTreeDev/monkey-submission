"""SQLAlchemy ORM models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class Detection(Base):
    """A single monkey detection event."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), index=True
    )
    confidence: Mapped[float] = mapped_column(Float)
    bbox_x: Mapped[float] = mapped_column(Float)
    bbox_y: Mapped[float] = mapped_column(Float)
    bbox_w: Mapped[float] = mapped_column(Float)
    bbox_h: Mapped[float] = mapped_column(Float)
    image_path: Mapped[str] = mapped_column(String(256), default="")
    annotated_path: Mapped[str] = mapped_column(String(256), default="")
    crop_path: Mapped[str] = mapped_column(String(256), default="")
    track_id: Mapped[int] = mapped_column(Integer, default=-1, index=True)
    trail_path: Mapped[str] = mapped_column(String(256), default="")
    # Manual override of the non-admin confidence gate. NULL = follow the rule
    # (pre-cutoff records visible, newer ones need USER_CONFIDENCE_THRESHOLD);
    # True = push to users regardless of confidence; False = hide from users
    # even though the rule would show it. Admins always see everything.
    user_visible: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    # Delete guard, orthogonal to `user_visible`: that one answers "may a user
    # see this", this one answers "may anyone delete this". Set by a human who
    # reviewed the record and confirmed it — the most expensive thing in the
    # table, because someone looked at it. Both delete paths (delete-selected,
    # clear-snapshots) refuse a locked row and keep the blobs it references.
    # Scope, stated honestly: that covers the files _detection_files names, and
    # the daily orphan sweep builds its referenced-set from the same helper — so
    # the two agree by construction.
    locked: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    # Which VISIT this row belongs to. A row is one animal; a visit is the
    # stretch of time animals were in front of the camera, and it is what people
    # actually mean by "the monkeys came" — 874 rows are 341 visits.
    #
    # Until 2026-08-28 a visit only existed as a read-time guess: /detections/
    # events regrouped rows by a 20 s gap on every request, which is why that
    # view is admin-only (a row is a fact, a regrouping is an interpretation).
    # Now the producer emits a whole visit at once (_VisitQueue), so the visit
    # is a thing that was RECORDED and can be stored rather than re-derived.
    # NULL means "written before this existed, or backfilled has not run" — the
    # 20 s regrouping stays the fallback for those, never the source of truth
    # for rows that carry a real id.
    visit_id: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, index=True
    )
    # The most animals visible AT ONCE anywhere in this visit — counted from the
    # tracks' own frame ranges at emit time, not inferred afterwards.
    #
    # It exists because the alternatives are both wrong. `distinct_tracks`
    # over-counts (one animal lost and re-acquired is two ids), and the
    # `estimated_animals` guess it replaces de-duplicated those ids by IoU and
    # time — reading a row's box, which is from its PEAK frame, against another
    # row's timestamp, which is when its animal ARRIVED. Two different moments
    # compared as if they were one. Frames do not have that problem: a frame is
    # an instant, so counting tracks present in the same frame is a measurement.
    #
    # Same value on every row of a visit. NULL where visit_id is NULL.
    visit_peak: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Which ANIMAL of its visit this row is. One level below visit_id: a visit
    # is the stretch of time animals were in front of the camera, and inside it
    # the tracker still hands one animal several ids when it fragments (the
    # 2026-08-28 07:06:28 ground truth: one macaque crossing the frame, eight
    # rows). Rows of one visit that carry the same animal_id are the same
    # animal chained back together; NULL means nobody has grouped this row, and
    # a row alone in its group keeps NULL rather than a group of one.
    #
    # Grouping is an INTERPRETATION layered on the rows, never a rewrite of
    # them: track_id, timestamp and every other column stay exactly as emitted,
    # so clearing animal_id restores the ungrouped truth — this is what makes
    # the merge reversible where the old row-DELETE dedup was not.
    animal_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )
    # Who says so: 'rule' = the motion-continuation rule scored the chain
    # (a calibrated grouping threshold), 'human' = a person confirmed it.
    # The UI reads this the way the timeline separates 記錄/重建, and the
    # backfill refuses to overwrite a 'human' verdict with a 'rule' one.
    animal_source: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    # WHERE IN THE CLIP this animal is, in seconds from the clip's first frame.
    #
    # `timestamp` says when the animal arrived in wall clock; a video player
    # cannot use that. It needs "second 12.4", and until these existed every row
    # opened its clip at the beginning and left you to find the animal yourself
    # — with seven macaques in one visit, seven times over.
    #
    # They also close the gap that has now bitten three times: a row recorded
    # when its track STARTED and never when it ended, which is why
    # `estimated_animals` could not be fixed and why a timeline could only ever
    # draw a dot. Everything needed is in `_TrackState` at emit time and was
    # being thrown away.
    #
    # NULL on rows written before this existed.
    clip_offset_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clip_end_offset_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # The clip's own length — the timeline's axis. Same on every row of a visit,
    # like visit_peak, because it belongs to the clip they share.
    clip_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # The stretches this animal was actually visible, [[from, to], ...] in the
    # same seconds. A single start→end bar claims continuous presence, which is
    # wrong exactly where it matters: a macaque that walks out of frame and
    # comes back is the case the tracking work spent weeks on, and painting it
    # as one unbroken bar hides the very thing worth seeing.
    #
    # JSON because it is a list whose length varies per row and nothing queries
    # inside it — same reason `User.allowed_cameras` is one.
    clip_segments: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Per-frame [frame_index, x1, y1, x2, y2, confidence] boxes produced by a
    # video-first rebuild.  Nullable so legacy producer rows keep their
    # existing semantics and can continue using clip_segments only.
    track_boxes: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    track_box_fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class ReIDPairLabel(Base):
    """A human verdict for whether two track rows are the same animal.

    Pair labels are research ground truth, not an identity assignment. Keeping
    them separate from Detection means a reviewer can correct a label without
    changing the detection row or the current timeline.
    """

    __tablename__ = "reid_pair_labels"
    __table_args__ = (
        UniqueConstraint(
            "detection_a_id", "detection_b_id", name="uq_reid_pair_labels_detections"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detection_a_id: Mapped[int] = mapped_column(
        ForeignKey("detections.id", ondelete="CASCADE"), index=True
    )
    detection_b_id: Mapped[int] = mapped_column(
        ForeignKey("detections.id", ondelete="CASCADE"), index=True
    )
    label: Mapped[str] = mapped_column(String(16))
    note: Mapped[str] = mapped_column(String(1000), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )


class HistoricalRebuildRun(Base):
    """Audit/rollback record for a human-approved video-first replacement."""

    __tablename__ = "historical_rebuild_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(24), default="applied")
    # Complete pre-apply Detection rows. This is intentionally JSON: it is an
    # immutable rollback payload, not a query surface.
    backup_rows: Mapped[list] = mapped_column(JSON, default=list)
    replacement_ids: Mapped[list] = mapped_column(JSON, default=list)
    skipped_sources: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    rolled_back_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Camera(Base):
    """Registered RTSP camera."""

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    rtsp_url: Mapped[str] = mapped_column(String(512))
    name: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class User(Base):
    """A platform account. The access code (hashed) is both credential and identity.

    Source of truth for role and — from Phase 2 — per-user permissions. Everything
    that will evolve lives in JSON columns (`allowed_cameras`, `feature_flags`) so
    the table never needs an ALTER (init_db only ever CREATEs, never migrates).
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), default="")  # display label
    code_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(16), default="user")  # 'admin' | 'user'
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # Camera IDs this user may access; ["*"] = all. Enforced from Phase 2.
    allowed_cameras: Mapped[list] = mapped_column(JSON, default=list)
    # Per-user feature switches, e.g. {"webhooks": true}. Enforced from Phase 2.
    feature_flags: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class EvalResult(Base):
    """本站評測 — one eval_ratchet.py run for one (model, split) on the edge runner.

    Posted by .github/workflows/model-evaluate.yml (our accuracy standard:
    imgsz 1600, fp16, MPS — only the Mac mini can produce these numbers).
    Rows are append-only history; readers take the latest per (model_id, split).
    """

    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    split: Mapped[str] = mapped_column(String(16))  # 'test' | 'val'
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    f1: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)  # ratchet exit 0
    baseline_f1: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class DerivedDataset(Base):
    """A deduplicated TRAIN-split copy of the master dataset, on the Platform.

    The master (`monkey cocotree`) keeps every image and the pinned test/val;
    training runs on one of these instead, because near-duplicate frames dilute
    the training signal (measured: -25% images, +5 pts recall). Append-only
    history — readers take the newest row, which is what 用訓練集開始訓練 trains on.
    """

    __tablename__ = "derived_datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    dataset_id: Mapped[str] = mapped_column(String(64), default="")  # platform id
    kept: Mapped[int] = mapped_column(Integer, default=0)  # train images kept
    dropped: Mapped[int] = mapped_column(Integer, default=0)  # near-dupes removed
    negatives_before: Mapped[int] = mapped_column(Integer, default=0)
    negatives_after: Mapped[int] = mapped_column(Integer, default=0)
    val_images: Mapped[int] = mapped_column(Integer, default=0)  # copied untouched
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class Webhook(Base):
    """A configured webhook endpoint."""

    __tablename__ = "webhooks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    url: Mapped[str] = mapped_column(String(512))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_error: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
