"""Pydantic schemas for request / response validation."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Detection ────────────────────────────────────────────────────────────────

class DetectionOut(BaseModel):
    id: int
    camera_id: str
    timestamp: datetime
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    image_path: str
    annotated_path: str
    crop_path: str
    track_id: int
    trail_path: str
    # Manual gate override (None = rule-based) and the resulting user-facing
    # visibility — the dashboard badges/actions read these instead of
    # re-implementing the rule client-side.
    user_visible: Optional[bool] = None
    visible_to_users: bool = True
    # Delete guard — a human confirmed this record, so both delete paths refuse
    # it. Independent of the visibility fields above. Shipped in the list
    # response so 影像管理 can mark locked cards without a second request.
    locked: bool = False
    # Which visit this row belongs to — every row the producer emitted for one
    # stretch of animals in front of the camera shares this. NULL on anything
    # written before 2026-08-28; the 20 s regrouping is the fallback for those.
    visit_id: Optional[str] = None
    # The most animals visible at once in that visit — counted, not estimated.
    visit_peak: Optional[int] = None
    # Which animal of its visit this row is — rows sharing the value are one
    # animal chained across track fragments. NULL = ungrouped.
    # `animal_source` says who chained it: 'rule' (motion-continuation rule,
    # calibrated threshold) or 'human'. The UI must keep the two apart the way
    # the timeline keeps 記錄/重建 apart.
    animal_id: Optional[str] = None
    animal_source: Optional[str] = None
    # Where in the clip this animal is, in seconds from the clip's first frame,
    # and how long the clip runs. What lets a player open at the animal instead
    # of at the beginning.
    clip_offset_seconds: Optional[float] = None
    clip_end_offset_seconds: Optional[float] = None
    clip_duration_seconds: Optional[float] = None
    clip_segments: Optional[list] = None
    # Optional逐幀 geometry for video-first clips. Legacy rows do not have it.
    track_boxes: Optional[list] = None
    track_box_fps: Optional[float] = None

    @field_validator("locked", mode="before")
    @classmethod
    def _null_is_unlocked(cls, value: object) -> object:
        """NULL reads as unlocked; everything else keeps normal bool parsing.

        A Detection built in Python reports None until the INSERT default
        applies, and a row predating the column could too. Unlocked is the
        honest reading — it matches what the delete paths do with a falsy
        `locked`, so the badge can never promise protection the delete path
        would not honour. Only None is remapped: coercing with `bool()` would
        turn the string "false" into a lock badge on a deletable record.
        """
        return False if value is None else value

    model_config = {"from_attributes": True}


class DetectionListOut(BaseModel):
    total: int
    items: list[DetectionOut]


class DetectionStatsOut(BaseModel):
    total_today: int
    total_all: int
    last_detection: Optional[datetime]
    cameras_active: int


class DetectionEventOut(BaseModel):
    """One visit by a troop, not one animal.

    A detection row is emitted per track when that track is LOST, so six
    macaques sitting on a wall arrive as a stream of separate rows spread over
    minutes — measured 2026-08-13, a scene with 6 animals visible produced ~12
    rows. Read one row at a time it looks like the system saw one monkey and
    missed five; they were all detected, just filed separately. This groups
    rows back into the visit they belong to.
    """

    event_id: str                  # camera + first timestamp, stable for the UI
    camera_id: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    detection_count: int           # rows filed under this visit
    distinct_tracks: int           # distinct track ids — the UPPER bound on animals
    # The most animals followed AT ONCE, counted at emit time from the tracks'
    # own frames. Replaces the old `estimated_animals`, which de-duplicated
    # track ids by comparing one row's PEAK-frame box against another row's
    # ARRIVAL timestamp — two different moments read as one. NULL on visits
    # recorded before this existed; the UI shows the row count alone rather
    # than substituting a guess.
    peak_animals: Optional[int] = None
    # "recorded" — the producer emitted this visit and stamped its rows.
    # "gap"      — reconstructed by cutting on a 20 s silence. Kept visible so
    #              a measurement and a reconstruction never look identical.
    basis: str = "gap"
    max_confidence: float
    trail_paths: list[str]
    detection_ids: list[int]
    cover: DetectionOut            # highest-confidence row, for the thumbnail


class DetectionEventListOut(BaseModel):
    total: int
    items: list[DetectionEventOut]
    gap_seconds: float
    scanned: int                   # rows examined (see the scan cap in main.py)
    truncated: bool


class HeatmapCellOut(BaseModel):
    date: str  # YYYY-MM-DD, TW-local
    hour: int  # 0-23
    camera_id: str
    count: int


class DetectionHeatmapOut(BaseModel):
    start: str  # first date of the range (inclusive, TW-local)
    end: str  # last date of the range (inclusive, TW-local)
    total: int
    cells: list[HeatmapCellOut]


# ── Camera ───────────────────────────────────────────────────────────────────

class CameraCreate(BaseModel):
    camera_id: str
    rtsp_url: str
    name: str = ""


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None


class CameraOut(BaseModel):
    id: int
    camera_id: str
    rtsp_url: str
    name: str
    created_at: datetime
    poc: bool = False
    # Live status, populated from stream_manager at response time. `connected`
    # = RTSP transport open; `is_alive` = also delivering non-black frames.
    connected: bool = False
    is_alive: bool = False

    model_config = {"from_attributes": True}


# ── Webhook ──────────────────────────────────────────────────────────────

class WebhookCreate(BaseModel):
    name: str = ""
    url: str
    enabled: bool = True


class WebhookUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    enabled: Optional[bool] = None


class WebhookOut(BaseModel):
    id: int
    name: str
    url: str
    enabled: bool
    last_error: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Admin ───────────────────────────────────────────────────────────────────

class StorageStatsOut(BaseModel):
    total_files: int
    total_size_bytes: int
    total_size_human: str
    total_detections: int


class InferenceCameraStatsOut(BaseModel):
    camera_id: str
    stream_connected: bool
    frame_skip: int = 1
    coverage_ceiling: float = 1.0
    inference_ms_last: float
    inference_ms_avg: float
    throughput_fps: float
    sample_fps: float
    total_inferences: int
    # Wasted-work counters. total_inferences alone hid a loop that re-ran 74% of
    # its inferences on frames it had already processed, so distinct frames,
    # duplicates and unused GPU turns are reported separately:
    # duplicate_inferences must stay 0. Defaulted so a node running an older
    # build can still report through the heartbeat aggregation.
    frames_processed: int = 0
    frames_unchanged: int = 0
    duplicate_inferences: int = 0
    idle_ratio: float = 0.0
    # Idle gate: `idle_gated` is the live verdict (this feed carries no image,
    # so it is being sampled at a heartbeat), `gated_ratio` the recent share of
    # delivered frames it dropped, `frame_stddev` the pixel variance the verdict
    # was made on — surfaced so the threshold can be judged from the dashboard.
    idle_gated: bool = False
    gated_ratio: float = 0.0
    frames_gated: int = 0
    frame_stddev: float = 0.0
    capture_fps: float = 0.0
    capture_gaps_ms: dict[str, float] = Field(default_factory=dict)
    stream_reconnects: int = 0
    stage_sec: dict[str, float] = Field(default_factory=dict)
    seconds_since_last: Optional[float]
    # Inference coverage: sample_fps / capture_fps — the share of the frames this
    # camera DELIVERED that were actually looked at. Both rates were already
    # reported and nobody divided them, so "11.4 FPS" read as healthy while two
    # thirds of every second went by unexamined.
    coverage_ratio: float = 0.0
    # Detector-thread liveness. `stream_connected` answers a different thread; the
    # two disagreed for 62 hours on 2026-08-08 and the dashboard believed the
    # wrong one. Restart counters expose flapping that a live flag would hide.
    worker_alive: bool = True
    worker_crashes: int = 0
    worker_revivals: int = 0
    # The size THIS camera is actually inferred at — the INFERENCE_IMGSZ floor
    # or its own frame size, whichever is larger. None until its first frame.
    # Reporting only the floor is how a 1920x1080 feed being squeezed into 1600
    # stayed invisible on the dashboard.
    imgsz: Optional[int] = None
    # POC feeds may use CPU while core cameras keep the MPS/GPU path. These
    # counters make the shape-preserving batch path visible in the dashboard.
    inference_device: str = "cpu"
    batch_count: int = 0
    batch_frames: int = 0
    batch_size_avg: float = 0.0
    batch_size_last: int = 0
    # White-balance rescues. A camera whose WB has failed is invisible in every
    # other metric — the 2026-08-15 dawn failure produced 1 detection across 59
    # frames and nothing said why.
    frames_wb_corrected: int = 0
    last_cast: float = 0.0


class InferenceStatsOut(BaseModel):
    # Floor, not the size used — see InferenceCameraStatsOut.imgsz per camera.
    inference_imgsz: int
    frame_skip: int
    model_path: str
    # Highest coverage the loop can reach: it consumes one frame in FRAME_SKIP.
    coverage_ceiling: float = 1.0
    # Stage 2: enabled / queued / dropped / processed / extra_boxes /
    # errors / queue_depth. `dropped` is the one that matters — it is the
    # difference between 'the GPU is keeping up' and 'it is not'.
    second_stage: dict = Field(default_factory=dict)
    cameras: list[InferenceCameraStatsOut]


class CameraDeleteOut(BaseModel):
    camera_id: str
    detections_preserved: int


# ── Users (Phase 2: per-user permission control) ─────────────────────────────

class UserCreate(BaseModel):
    code: str  # the access code (login credential); stored hashed
    name: str = ""
    role: str = "user"  # 'admin' | 'user'
    enabled: bool = True
    allowed_cameras: list[str] = []  # ["*"] = all cameras
    feature_flags: dict[str, bool] = {}


class UserUpdate(BaseModel):
    code: Optional[str] = None  # provide to reset the access code
    name: Optional[str] = None
    role: Optional[str] = None
    enabled: Optional[bool] = None
    allowed_cameras: Optional[list[str]] = None
    feature_flags: Optional[dict[str, bool]] = None


class UserOut(BaseModel):
    id: int
    name: str
    role: str
    enabled: bool
    allowed_cameras: list[str]
    feature_flags: dict[str, bool]
    created_at: datetime

    model_config = {"from_attributes": True}


class MeOut(BaseModel):
    """The current user's own identity + effective permissions (for the client)."""

    id: int
    name: str
    role: str
    allowed_cameras: list[str]
    feature_flags: dict[str, bool]

    model_config = {"from_attributes": True}


class FeatureOut(BaseModel):
    key: str
    label: str


# ── ReID human-in-the-loop research labels ──────────────────────────────────

class ReIDLabelCreate(BaseModel):
    detection_a_id: int = Field(gt=0)
    detection_b_id: int = Field(gt=0)
    label: Literal["same", "different", "uncertain"]
    note: str = Field(default="", max_length=1000)


class ReIDTrackOut(BaseModel):
    id: int
    track_id: int
    camera_id: str
    timestamp: datetime
    confidence: float
    crop_path: str
    image_path: str
    annotated_path: str
    trail_path: str
    clip_duration_seconds: Optional[float] = None
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    frame_count: int = 0
    # Per-frame [frame_index, x1, y1, x2, y2, confidence] geometry.  The
    # review player uses this to draw an explicit A/B overlay on the video.
    track_boxes: Optional[list] = None
    track_box_fps: Optional[float] = None


class ReIDPairOut(BaseModel):
    pair_id: str
    camera_id: str
    trail_path: str
    clip_duration_seconds: Optional[float] = None
    temporal_relation: Literal["separate", "overlap", "unknown"]
    gap_seconds: Optional[float] = None
    a: ReIDTrackOut
    b: ReIDTrackOut
    label: Optional[Literal["same", "different", "uncertain"]] = None
    note: str = ""
    labeled_at: Optional[datetime] = None
    # A deterministic queue shortcut is deliberately separate from `label`:
    # only a human label becomes ReID ground truth.
    system_label: Optional[Literal["different"]] = None
    decision_source: Optional[Literal["human", "geometry_auto"]] = None
    review_required: bool = True
    review_priority: int = 0
    review_reason: str = ""


class ReIDPairListOut(BaseModel):
    total: int
    candidate_total: int
    pending: int
    labeled: int
    batch_total: int
    batch_pending: int
    batch_labeled: int
    auto_resolved: int = 0
    batch_auto_resolved: int = 0
    batch_review_pending: int = 0
    pairs: list[ReIDPairOut]


class ReIDLabelOut(BaseModel):
    detection_a_id: int
    detection_b_id: int
    label: Literal["same", "different", "uncertain"]
    note: str
    updated_at: datetime
