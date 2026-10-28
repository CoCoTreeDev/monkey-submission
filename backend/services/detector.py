"""YOLO-based monkey detection worker with object tracking.

Uses model.track() (ByteTrack) so the same object keeps a consistent ID
across frames. A detection event is only emitted once per track — when the
tracker loses the object (or it leaves the frame). This eliminates
overcounting.

When multiple tracks overlap temporally (e.g. two people in the same scene)
they are batched and a single combined H.264 trail clip is generated showing ALL
movement paths on the shared frames.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
import math
import queue
import statistics
import subprocess
import threading
import time
import tempfile
import uuid
import collections
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone

# Taiwan Standard Time (UTC+8)
_TW_TZ = timezone(timedelta(hours=8))
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from config import (
    ANE_FUSION_ADD_CONF,
    ANE_FUSION_CAMERAS,
    ANE_FUSION_GATE_CONF,
    ANE_TILE_CAMERAS,
    ANE_TILE_MODEL_PATH,
    CANDIDATE_CONFIDENCE_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    DEV_RAW_WEBHOOK,
    DEV_RAW_WEBHOOK_COOLDOWN,
    DEV_RAW_WEBHOOK_MIN_CONF,
    FRAME_SKIP,
    FFMPEG_BINARY,
    frame_skip_for_camera,
    IDLE_GATE_HEARTBEAT_SECONDS,
    INFERENCE_DEVICE,
    INFERENCE_HALF,
    INFERENCE_IMGSZ,
    INFERENCE_SERIALISE,
    inference_device_for_camera,
    inference_half_for_camera,
    poc_batch_for_camera,
    POC_BATCH_MAX,
    POC_BATCH_WAIT_MS,
    WHITE_BALANCE_CAST_THRESHOLD,
    colour_cast,
    correct_white_balance,
    imgsz_for_frame,
    MODEL_PATH,
    RAW_FRAMES_MAX_BYTES,
    TRACK_CONTINUATION_IOU,
    TRACK_CONTINUATION_SECONDS,
    TRACK_MOTION_ERR_WIDTHS,
    TRACK_MOTION_GAP_SECONDS,
    TRACK_DUPLICATE_IOU,
    TRACK_DUPLICATE_MIN_FRAMES,
    TRACK_DUPLICATE_MIN_OVERLAP,
    TRACK_MIN_HITS,
    TRACK_MIN_SECONDS,
    VISIT_HOLD_MAX_SECONDS,
    VISIT_HOLD_SECONDS,
    SEGMENT_GAP_SECONDS,
    VISIT_QUEUE_MAX,
    TRACK_LOST_SECONDS,
    TRACKER_CONFIDENCE_THRESHOLD,
    TRACKER_YAML,
    TRAIL_GIF_MAX_FRAMES,
    TRAIL_GIF_SCALE,
    TRAIL_MP4_CRF,
    TRAIL_VIDEO_FPS,
    TRAIL_VIDEO_MAX_FRAMES,
    TRAIL_POST_BUFFER_SECONDS,
    TRAIL_PRE_BUFFER_SECONDS,
    SECOND_STAGE_CONF,
    SECOND_STAGE_DEDUP_IOU,
    SECOND_STAGE_GATE_CONF,
    SECOND_STAGE_MODEL_PATH,
    SECOND_STAGE_QUEUE_MAX,
    SECOND_STAGE_RESULT_MAX,
    VERIFIER_IMGSZ,
    VERIFIER_MODEL_PATH,
)
from track_policy import is_substantial_track
from services.tiling import StandardTracker, TiledTracker, fused_predict, tiled_predict

logger = logging.getLogger(__name__)

# Silence ultralytics verbose logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Lazy-loaded YOLO model (loaded once on first use)
_model = None
_model_lock = threading.Lock()

class _FairLock:
    """Priority-fair lock — core turns precede queued POC turns.

    A plain threading.Lock lets the releasing thread barge straight back in,
    because it is already running while the others still have to be woken.
    Once the workers stopped sleeping between frames that turned into
    starvation: one camera ran at 2.65 FPS while its peers held 4.0-4.65, and
    aggregate throughput fell from 26.8 to 15.6 FPS. The old 10 ms sleep had
    been providing fairness by accident; this provides it on purpose, without
    paying 20 ms per cycle for it.
    """

    _LOW_PRIORITY_MAX_WAIT = 0.1

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        # The first item is the current owner. Core cameras use priority 0;
        # optional POC work uses priority 1. POC is allowed one turn after a
        # short maximum wait so a continuously busy core queue cannot starve
        # the measured spare GPU capacity.
        self._waiters: deque[tuple[int, int, threading.Event, float]] = deque()
        self._sequence = 0

    def acquire(self, *, priority: int = 0) -> None:
        ev = threading.Event()
        with self._mutex:
            free = not self._waiters
            self._sequence += 1
            item = (priority, self._sequence, ev, time.monotonic())
            self._waiters.append(item)
        if not free:
            ev.wait()

    def release(self) -> None:
        with self._mutex:
            self._waiters.popleft()
            if self._waiters:
                now = time.monotonic()

                def key(item: tuple[int, int, threading.Event, float]):
                    priority, sequence, _event, queued_at = item
                    aged = (
                        priority > 0
                        and now - queued_at >= self._LOW_PRIORITY_MAX_WAIT
                    )
                    return (0 if aged else priority, sequence)

                waiters = list(self._waiters)
                next_index = min(range(len(waiters)), key=lambda i: key(waiters[i]))
                next_item = waiters.pop(next_index)
                self._waiters = deque([next_item, *waiters])
                next_item[2].set()

    def __enter__(self) -> "_FairLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# Serialises GPU work across camera workers. Four independent models issuing
# concurrent MPS command streams measured SLOWER in aggregate than taking
# turns (58-63 ms/call and 21.9 FPS concurrent, vs 37-41 ms and 25.3 FPS when
# calls are serialised), so throughput is better with one camera on the GPU at
# a time. POC batch turns enter at lower priority. INFERENCE_SERIALISE=0
# restores unrestricted concurrency for the standard camera workers.
_gpu_lock = _FairLock()

# Lazy-loaded cascade verifier (larger model, event-driven; None = disabled)
_verifier = None
_verifier_lock = threading.Lock()

# Distinct colors for multi-track trails (BGR)
_TRACK_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 165, 255),  # orange
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (255, 255, 0),  # cyan
    (128, 0, 255),  # purple
    (0, 128, 255),  # gold
]

# How far (in frames) an observation may be from a rendered frame and still
# be outlined in an annotated still or trail video. Small on purpose: the point
# is "these animals were in THIS frame", not "were around at some point".
_COMPANION_MAX_FRAME_GAP = 3


def _own_model(device: str = INFERENCE_DEVICE, half: bool = INFERENCE_HALF):
    """Build a private YOLO instance for one camera worker.

    Deliberately NOT the shared singleton: ultralytics stores tracker state on
    the predictor, so sharing a model means sharing a ByteTracker across
    cameras — which silently blinded every camera but one. A private instance
    also keeps four worker threads off one predictor's mutable state.

    Loading is serialised on the model lock so concurrent workers don't race
    inside ultralytics' own lazy init, and the pre-warm builds the predictor
    (and applies fp16) before the first real frame.
    """
    from ultralytics import YOLO

    model_path = Path(MODEL_PATH)
    if not model_path.is_file():
        raise RuntimeError(f"YOLO model not found at {MODEL_PATH}")
    with _model_lock:
        model = YOLO(MODEL_PATH)
        model.predict(
            np.zeros((640, 640, 3), dtype=np.uint8),
            verbose=False,
            device=device or None,
            half=half,
        )
    return model


def _force_half(model) -> None:
    """Convert a built predictor's backend to fp16.

    For callers whose predictor is already built; new predictors can use
    `half=True` on their first predict call. Preprocess reads the backend's
    fp16 flag to choose the input dtype. Failure is non-fatal: fp32 still works.
    """
    try:
        backend = model.predictor.model
        backend.model.half()
        backend.fp16 = True
        logger.info("Inference running in fp16")
    except Exception:
        logger.exception("Could not enable fp16 — continuing in fp32")


def _get_verifier():
    """Load the cascade verifier model (thread-safe, singleton).

    Returns None when VERIFIER_MODEL_PATH is unset or the file is missing —
    the cascade is optional and must never take the realtime pipeline down.
    """
    global _verifier
    if not VERIFIER_MODEL_PATH:
        return None
    if _verifier is None:
        with _verifier_lock:
            if _verifier is None:
                if not Path(VERIFIER_MODEL_PATH).exists():
                    logger.warning(
                        "Verifier model %s not found — cascade disabled", VERIFIER_MODEL_PATH,
                    )
                    return None
                from ultralytics import YOLO
                logger.info("Loading verifier model from %s …", VERIFIER_MODEL_PATH)
                candidate = YOLO(VERIFIER_MODEL_PATH)
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                candidate.predict(dummy, verbose=False, device=INFERENCE_DEVICE or None)
                _verifier = candidate
                logger.info("Verifier model loaded")
    return _verifier


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two (x, y, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class _SecondStage:
    """26x behind 26n, at FRAME level, off the detector thread.

    Three properties matter, in this order:

    1. **It can never block a camera.** `offer()` is a non-blocking put on a
       bounded queue; a full queue drops the frame and counts it. The detector
       thread has died twice on inline I/O (2026-08-06, 2026-08-08) and the
       cost was 62 hours of one camera not being watched. A dropped second
       stage costs one box.
    2. **Frame level, not box level.** The existing `_verify_track` only
       re-scores boxes 26n already nominated, and 40.1% of hand-labelled
       macaques produce no 26n box at all — there is nothing to adjudicate.
       Re-running the whole frame finds animals 26n never proposed.
    3. **Union, not replacement.** Measured on the crowd holdout, the union
       beats both models alone (recall 0.454 / 0.490 alone, 0.599 together):
       they are blind to different animals.

    Results are stored per (camera, frame index) and read at finalisation.
    Whatever has arrived is merged; whatever has not is simply absent, so
    finalisation never waits.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple[str, int, Any, int]]" = queue.Queue(
            maxsize=max(1, SECOND_STAGE_QUEUE_MAX)
        )
        self._results: "OrderedDict[tuple[str, int], list[tuple[float, float, float, float]]]" = OrderedDict()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._model = None
        self._load_failed = False
        self.queued = self.dropped = self.processed = self.extra_boxes = self.errors = 0

    @property
    def enabled(self) -> bool:
        return bool(SECOND_STAGE_MODEL_PATH)

    def _ensure_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="second-stage", daemon=True
            )
            self._thread.start()

    def offer(self, camera_id: str, frame_idx: int, frame, imgsz: int) -> None:
        """Non-blocking. Never raises, never waits."""
        if not self.enabled:
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait((camera_id, frame_idx, frame.copy(), imgsz))
            self.queued += 1
        except queue.Full:
            self.dropped += 1
        except Exception:
            self.dropped += 1

    def boxes_for(self, camera_id: str, frame_idx: int):
        with self._lock:
            return list(self._results.get((camera_id, frame_idx), ()))

    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None          # complain once, not once per frame
        if not Path(SECOND_STAGE_MODEL_PATH).exists():
            self._load_failed = True
            logger.warning(
                "Second-stage model %s not found — stage 2 disabled", SECOND_STAGE_MODEL_PATH
            )
            return None
        from ultralytics import YOLO

        logger.info("Loading second-stage model from %s …", SECOND_STAGE_MODEL_PATH)
        model = YOLO(SECOND_STAGE_MODEL_PATH)
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False,
                      device=INFERENCE_DEVICE or None)
        if INFERENCE_HALF and INFERENCE_DEVICE == "mps":
            _force_half(model)
        self._model = model
        logger.info("Second-stage model loaded")
        return self._model

    def _run(self) -> None:
        while True:
            try:
                camera_id, frame_idx, frame, imgsz = self._queue.get()
            except Exception:                       # pragma: no cover
                return
            try:
                model = self._load()
                if model is None:
                    continue
                # Takes the same GPU turn as the camera workers: concurrent MPS
                # streams measurably slow each other down, and this pass is long
                # (~477 ms measured) so overlapping it would cost every camera.
                def _run_one():
                    return model.predict(frame, imgsz=imgsz, conf=SECOND_STAGE_CONF, classes=[0],
                                         verbose=False, device=INFERENCE_DEVICE or None)

                results = _run_one() if not INFERENCE_SERIALISE else None
                if results is None:
                    with _gpu_lock:
                        results = _run_one()
                boxes = []
                for result in results:
                    b = result.boxes
                    if b is None or not len(b):
                        continue
                    boxes.extend(tuple(row) for row in b.xyxy.tolist())
                with self._lock:
                    self._results[(camera_id, frame_idx)] = boxes
                    while len(self._results) > max(1, SECOND_STAGE_RESULT_MAX):
                        self._results.popitem(last=False)
                self.processed += 1
            except Exception:
                self.errors += 1
                logger.exception("Second stage failed on %s frame %d", camera_id, frame_idx)
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "queued": self.queued,
            "dropped": self.dropped,
            "processed": self.processed,
            "extra_boxes": self.extra_boxes,
            "errors": self.errors,
            "queue_depth": self._queue.qsize(),
        }


second_stage = _SecondStage()


def _second_stage_extras(camera_id: str, frame_idx: int, tracks) -> list:
    """26x boxes on this frame that no track already covers — the UNION half.

    A box overlapping a tracked one by SECOND_STAGE_DEDUP_IOU is the same
    animal seen twice, not a new one. What survives is what 26n never proposed,
    which is the whole point: on the crowd holdout the two models miss
    different animals, so together they find more than either alone.
    """
    if not second_stage.enabled:
        return []
    boxes = second_stage.boxes_for(camera_id, frame_idx)
    if not boxes:
        return []
    known = []
    for t in tracks:
        b = t.bbox_near(frame_idx, _COMPANION_MAX_FRAME_GAP)
        if b:
            known.append((b[0], b[1], b[0] + b[2], b[1] + b[3]))
    extras = []
    for x1, y1, x2, y2 in boxes:
        cand = (x1, y1, x2 - x1, y2 - y1)
        if any(_bbox_iou(cand, (k[0], k[1], k[2] - k[0], k[3] - k[1])) >= SECOND_STAGE_DEDUP_IOU
               for k in known):
            continue
        extras.append((x1, y1, x2, y2))
    second_stage.extra_boxes += len(extras)
    return extras


def _verify_track(track: "_TrackState", frame_buf: "_FrameBuffer") -> Optional[float]:
    """Re-score a track's best frame with the cascade verifier.

    Returns the highest verifier confidence among boxes overlapping the
    track's best bbox (IoU >= 0.3) — 0.0 means the verifier looked and saw
    no monkey there. Returns None when the verifier is disabled or fails,
    so the caller falls back to the realtime model's confidence (fail-open:
    a broken verifier must not lose real detections).
    """
    try:
        model = _get_verifier()
        if model is None:
            return None
        # Prefer the track's raw best frame — the buffer's JPEG round-trip
        # measurably deflates confidence and would cause false rejections.
        frame = track.best_frame
        if frame is None:
            frame = frame_buf.get(track.best_frame_idx)
        if frame is None:
            return None
        # Takes the same GPU turn as the camera workers: concurrent MPS streams
        # measurably slow each other down, and the big model's pass is long
        # enough (~240 ms at 1280) to matter if it overlaps them.
        def _run():
            return model.predict(
                frame,
                imgsz=VERIFIER_IMGSZ,
                conf=0.05,
                classes=[0],
                verbose=False,
                device=INFERENCE_DEVICE or None,
            )

        if INFERENCE_SERIALISE:
            with _gpu_lock:
                results = _run()
        else:
            results = _run()
        best = 0.0
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cand = (x1, y1, x2 - x1, y2 - y1)
                if _bbox_iou(track.best_bbox, cand) >= 0.3:
                    best = max(best, float(box.conf[0]))
        return best
    except Exception:
        logger.exception("Verifier failed for track #%d — falling back", track.track_id)
        return None


# ── Detection result ─────────────────────────────────────────────────────────


class DetectionResult:
    """Holds a finalised track result before it's persisted."""

    def __init__(
        self,
        camera_id: str,
        confidence: float,
        bbox: tuple[float, float, float, float],
        image_path: str,
        annotated_path: str,
        crop_path: str,
        track_id: int,
        trail_path: str,
        timestamp: datetime,
        visit_id: Optional[str] = None,
        visit_peak: Optional[int] = None,
        clip_offset_seconds: Optional[float] = None,
        clip_end_offset_seconds: Optional[float] = None,
        clip_duration_seconds: Optional[float] = None,
        clip_segments: Optional[list] = None,
    ) -> None:
        self.camera_id = camera_id
        self.confidence = confidence
        self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h = bbox
        self.image_path = image_path
        self.annotated_path = annotated_path
        self.crop_path = crop_path
        self.track_id = track_id
        self.trail_path = trail_path
        self.timestamp = timestamp
        self.visit_id = visit_id
        self.visit_peak = visit_peak
        self.clip_offset_seconds = clip_offset_seconds
        self.clip_end_offset_seconds = clip_end_offset_seconds
        self.clip_duration_seconds = clip_duration_seconds
        self.clip_segments = clip_segments


# ── Track state ──────────────────────────────────────────────────────────────


class _TrackState:
    """In-memory state for a single tracked object.

    Stores center points, per-frame bounding boxes, and frame indices.
    Actual frames live in the shared FrameBuffer owned by the DetectorWorker.
    """

    __slots__ = (
        "track_id", "camera_id", "first_seen", "last_seen_frame",
        "min_frame_idx", "max_frame_idx",
        "centers", "frame_indices", "frame_bboxes",
        "best_conf", "best_bbox", "best_frame_idx", "best_frame",
        "first_seen_ts", "last_seen_ts", "first_seen_capture_ts",
        "last_seen_capture_ts",
    )

    def __init__(self, track_id: int, camera_id: str, frame_idx: int) -> None:
        self.track_id = track_id
        self.camera_id = camera_id
        self.first_seen = datetime.now(_TW_TZ)
        self.last_seen_frame = frame_idx
        # Wall clock, alongside the datetime, because continuation matching
        # compares this to another track's last_seen_ts and the two must be the
        # same clock. first_seen is what the DB row carries.
        self.first_seen_ts = time.time()
        self.last_seen_ts = self.first_seen_ts
        # Capture time is separate from processing time. GPU latency can make
        # two adjacent frames look half a second apart even though the animal
        # was only observed for a few milliseconds.
        self.first_seen_capture_ts: Optional[float] = None
        self.last_seen_capture_ts: Optional[float] = None
        self.min_frame_idx = frame_idx
        self.max_frame_idx = frame_idx
        self.centers: list[tuple[int, int]] = []
        self.frame_indices: list[int] = []
        self.frame_bboxes: list[tuple[float, float, float, float]] = []
        self.best_conf: float = 0.0
        self.best_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
        self.best_frame_idx: int = frame_idx
        # Raw (never JPEG'd) copy of the best frame. The shared _FrameBuffer
        # stores compressed frames, and a single q90 round-trip measurably
        # drops model confidence (0.83 → 0.22 on a real sample), which would
        # sabotage the cascade verifier and degrade the emitted snapshots.
        # One raw frame per active track is bounded and small.
        self.best_frame: Optional[np.ndarray] = None

    def update(
        self,
        center: tuple[int, int],
        conf: float,
        bbox: tuple[float, float, float, float],
        frame_idx: int,
        frame: Optional[np.ndarray] = None,
        captured_ts: Optional[float] = None,
    ) -> None:
        self.last_seen_frame = frame_idx
        self.last_seen_ts = time.time()
        self.max_frame_idx = frame_idx
        self.centers.append(center)
        self.frame_indices.append(frame_idx)
        self.frame_bboxes.append(bbox)
        if captured_ts is not None:
            if self.first_seen_capture_ts is None:
                self.first_seen_capture_ts = captured_ts
            self.last_seen_capture_ts = captured_ts

        if conf > self.best_conf:
            self.best_conf = conf
            self.best_bbox = bbox
            self.best_frame_idx = frame_idx
            if frame is not None:
                self.best_frame = frame

    def bbox_at(self, fi: int) -> Optional[tuple[float, float, float, float]]:
        """Return the closest stored bbox for a given frame index."""
        if not self.frame_indices:
            return None
        best_i, best_dist = 0, abs(self.frame_indices[0] - fi)
        for i, idx in enumerate(self.frame_indices):
            d = abs(idx - fi)
            if d < best_dist:
                best_i, best_dist = i, d
        return self.frame_bboxes[best_i]

    def bbox_near(
        self, fi: int, max_gap: int
    ) -> Optional[tuple[float, float, float, float]]:
        """bbox at frame fi, but only if this track was actually seen near fi.

        bbox_at() returns the NEAREST stored box however far away it is, which
        is wrong for drawing current presence in a still or video: a track that
        already left would be drawn at a stale position, inventing a monkey
        that is not in that frame.
        """
        if not self.frame_indices:
            return None
        best_i, best_dist = 0, abs(self.frame_indices[0] - fi)
        for i, idx in enumerate(self.frame_indices):
            d = abs(idx - fi)
            if d < best_dist:
                best_i, best_dist = i, d
        if best_dist > max_gap:
            return None
        return self.frame_bboxes[best_i]


def _track_observed_seconds(track: _TrackState) -> float:
    """Return the track's actual capture-time span, not processing latency."""
    if track.first_seen_capture_ts is not None and track.last_seen_capture_ts is not None:
        return max(
            0.0, track.last_seen_capture_ts - track.first_seen_capture_ts
        )
    # Unit callers and legacy test harnesses do not have capture timestamps.
    return max(0.0, track.last_seen_ts - track.first_seen_ts)


def _track_is_substantial(track: _TrackState) -> bool:
    """Require repeated evidence over capture time before persisting a row."""
    return is_substantial_track(
        len(track.centers),
        _track_observed_seconds(track),
        min_hits=TRACK_MIN_HITS,
        min_seconds=TRACK_MIN_SECONDS,
    )


# ── Shared frame buffer ─────────────────────────────────────────────────────


def _tail_velocity(track: "_TrackState", frames: int = 5) -> tuple[float, float]:
    """Pixels per second at the end of a track, from its last few boxes.

    Per-box timestamps are not kept, so the interval is the track's own average
    — its wall-clock span over the boxes it stored. That is accurate enough for
    a gap of a few frames, which is the only thing this is used across.
    """
    boxes = track.frame_bboxes
    if len(boxes) < 2:
        return 0.0, 0.0
    span = track.last_seen_ts - track.first_seen_ts
    if span <= 0:
        return 0.0, 0.0
    per_box = span / (len(boxes) - 1)
    tail = boxes[-frames:]
    dt = per_box * (len(tail) - 1)
    if dt <= 0:
        return 0.0, 0.0
    # Boxes here are (x, y, w, h) — see _bbox_iou — not corner pairs.
    cx = lambda b: b[0] + b[2] / 2
    cy = lambda b: b[1] + b[3] / 2
    return (cx(tail[-1]) - cx(tail[0])) / dt, (cy(tail[-1]) - cy(tail[0])) / dt


def _continues_by_motion(track: "_TrackState", cand: "_TrackState", gap: float) -> float:
    """How well `cand` sits where `track` was heading, in box widths.

    Returns the miss; smaller is better, and `inf` when the pair is not a
    candidate at all. Overlap cannot answer this — a macaque crossing the frame
    moves nearly its own width between fragments, so even a perfect prediction
    scores an IoU of 0.2-0.3 — but the distance from the predicted position,
    measured in box widths, stays small.
    """
    if gap < 0 or gap > TRACK_MOTION_GAP_SECONDS:
        return float("inf")
    if not track.frame_bboxes or not cand.frame_bboxes:
        return float("inf")
    last, first = track.frame_bboxes[-1], cand.frame_bboxes[0]
    vx, vy = _tail_velocity(track)
    # (x, y, w, h), as everywhere else in this module.
    px = last[0] + last[2] / 2 + vx * gap
    py = last[1] + last[3] / 2 + vy * gap
    bx = first[0] + first[2] / 2
    by = first[1] + first[3] / 2
    wa, wb = last[2], first[2]
    width = max((wa + wb) / 2, 1.0)
    # Size is a weak check here: a running macaque's box changes a lot with
    # pose — 184px to 95px inside a fifth of a second on the measured run — so
    # it is only used to refuse the wildly different, never to decide.
    if max(wa, wb) / max(min(wa, wb), 1.0) > 2.5:
        return float("inf")
    return ((px - bx) ** 2 + (py - by) ** 2) ** 0.5 / width


def _clip_fps(timestamps: list[float]) -> float:
    """Return the single playback rate shared by the MP4 and its offsets."""
    return max(1.0, TRAIL_VIDEO_FPS)


def _resample_trail_frames(
    frames: list[Any],
    timestamps: list[float],
    target_fps: float,
) -> tuple[list[Any], list[float], float]:
    """Put selected frames on a bounded, constant-rate playback timeline.

    A detector normally sees fewer frames than the source camera delivers. The
    old encoder wrote those frames at their median inter-arrival rate, which
    made RTSP bursts accelerate a long wall-clock visit and look like missing
    frames. This helper keeps every selected frame, then repeats the most recent
    frame into empty 24-FPS slots. Repeated references share the same image
    object, so the fix does not multiply decoded-frame memory.

    Very long visits still honour the existing output ceiling. In that case all
    selected frames remain in order, but the bounded timeline is compressed;
    this is preferable to allowing finalisation to allocate an unbounded clip.
    """
    fps = max(1.0, float(target_fps))
    if not frames:
        return [], [], fps

    usable_timestamps = list(timestamps[: len(frames)])
    if len(usable_timestamps) < len(frames):
        usable_timestamps.extend([0.0] * (len(frames) - len(usable_timestamps)))

    def valid_timestamp(value: Any) -> float | None:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return None
        return candidate if candidate > 0 and math.isfinite(candidate) else None

    first_ts = next(
        (candidate for value in usable_timestamps
         if (candidate := valid_timestamp(value)) is not None),
        0.0,
    )
    # `output_timestamps` is the wall-clock axis used to map a source event to
    # its encoded slot. It is intentionally not the encoder's regular 24-FPS
    # clock: repeated slots carry the timestamp of the real frame they repeat.
    # Keeping this axis alongside the encoded frames prevents a dense burst from
    # being pushed forward for ordering while binary-search still maps it to an
    # earlier regular-clock slot.
    wall_timestamps: list[float] = []
    elapsed: list[float] = []
    last_elapsed = 0.0
    last_wall = first_ts
    for raw_ts in usable_timestamps:
        ts = valid_timestamp(raw_ts)
        if first_ts > 0:
            if ts is not None and ts >= first_ts:
                last_wall = max(last_wall, ts)
            wall_timestamps.append(last_wall)
            last_elapsed = max(last_elapsed, last_wall - first_ts)
        else:
            wall_timestamps.append(0.0)
        elapsed.append(last_elapsed)

    source_positions: list[int] = []
    previous = -1
    for i, seconds in enumerate(elapsed):
        wanted = int(round(seconds * fps)) if first_ts > 0 else i
        position = max(previous + 1, wanted)
        source_positions.append(position)
        previous = position

    required = source_positions[-1] + 1
    output_count = max(len(frames), required)
    if output_count > TRAIL_VIDEO_MAX_FRAMES:
        # Keep every source frame, but spread them over the bounded output
        # timeline. TRAIL_GIF_MAX_FRAMES already bounds `len(frames)` in the
        # normal path, so this branch only handles a long wall-clock gap.
        output_count = max(TRAIL_VIDEO_MAX_FRAMES, len(frames))
        source_positions = [
            round(i * (output_count - 1) / max(len(frames) - 1, 1))
            for i in range(len(frames))
        ]

    output: list[Any] = [frames[0]] * output_count
    for position, frame in zip(source_positions, frames):
        output[position] = frame

    # Fill gaps with the most recent real frame. Comparing by index, rather
    # than object equality, also works when callers use numpy arrays or PIL
    # images whose equality returns an array.
    real_positions = dict(zip(source_positions, frames))
    last_frame = frames[0]
    for i in range(output_count):
        if i in real_positions:
            last_frame = real_positions[i]
        else:
            output[i] = last_frame

    output_timestamps = [0.0] * output_count
    for position, timestamp in zip(source_positions, wall_timestamps):
        output_timestamps[position] = timestamp
    last_timestamp = 0.0
    for index, timestamp in enumerate(output_timestamps):
        if timestamp > 0:
            last_timestamp = max(last_timestamp, timestamp)
        elif last_timestamp > 0:
            output_timestamps[index] = last_timestamp
    return output, output_timestamps, fps


def _clip_axis(timestamps: list[float]) -> list[float]:
    """The kept frames' wall clock, made non-decreasing so it can be searched.

    A frame whose timestamp went missing is written as 0.0 upstream, which would
    break the ordering; carry the previous frame's instead, and drop leading
    placeholders that have nothing to carry.
    """
    axis: list[float] = []
    last = 0.0
    for ts in timestamps:
        if ts > 0:
            last = max(last, ts)
        if last > 0:
            axis.append(last)
    return axis


def _video_seconds(wall: float, axis: list[float], fps: float) -> float:
    """The second of the CLIP that a wall-clock instant is played at.

    Frame i plays at i/fps, and that is not `wall - t0`: the frames did not
    arrive evenly, so the clip is a resampling of the visit, not a recording of
    it in real time.
    """
    if not axis or fps <= 0:
        return 0.0
    i = min(max(bisect.bisect_left(axis, wall), 0), len(axis) - 1)
    return i / fps


def _clip_offsets(
    track: "_TrackState",
    axis: list[float],
    fps: float,
    frame_ts: Optional[dict[int, float]] = None,
) -> dict[str, object]:
    """Where this track sits inside its clip, in the CLIP's seconds.

    Not in wall-clock seconds: the clip is written at a constant rate over the
    frames it kept, so what a player needs is the frame's position in that
    sequence. Clamped to the clip because a track can be seen slightly before
    the first frame the clip kept — the pre-buffer is bounded, the track is not.

    Everything is None when the clip has no time base — better a row that says
    "unknown" than one that says "second 0" and sends every player to the start,
    which is the behaviour these fields exist to end.
    """
    if not axis or fps <= 0:
        return {"clip_offset_seconds": None, "clip_end_offset_seconds": None,
                "clip_duration_seconds": None, "clip_segments": None}
    duration = len(axis) / fps
    start = min(max(0.0, _video_seconds(track.first_seen_ts, axis, fps)), duration)
    end = min(max(start, _video_seconds(track.last_seen_ts, axis, fps)), duration)
    return {"clip_offset_seconds": round(start, 3),
            "clip_end_offset_seconds": round(end, 3),
            "clip_duration_seconds": round(duration, 3),
            "clip_segments": _presence_segments(track, frame_ts or {}, axis, fps)}


def _presence_segments(
    track: "_TrackState",
    frame_ts: dict[int, float],
    axis: list[float],
    fps: float,
) -> Optional[list[list[float]]]:
    """The stretches this animal was actually visible, in clip seconds.

    A track carries the frames it was DETECTED in. Where consecutive detections
    sit more than SEGMENT_GAP_SECONDS apart, the animal was out of frame or the
    detector lost it — and that gap is the thing worth drawing. One start→end
    bar would paint it as continuous presence, hiding exactly the behaviour the
    tracking work spent weeks on: a macaque leaving and coming back.

    None when the frames carry no timestamps, so the UI falls back to the plain
    start→end bar rather than inventing a shape.
    """
    stamps = sorted(
        frame_ts[i] for i in set(track.frame_indices) if i in frame_ts
    )
    if not stamps:
        return None
    segments: list[list[float]] = []
    seg_start = prev = stamps[0]
    for ts in stamps[1:]:
        if ts - prev > SEGMENT_GAP_SECONDS:
            segments.append([seg_start, prev])
            seg_start = ts
        prev = ts
    segments.append([seg_start, prev])
    duration = len(axis) / fps if axis and fps > 0 else 0.0
    out = []
    for a, b in segments:
        lo = min(max(0.0, _video_seconds(a, axis, fps)), duration)
        hi = min(max(lo, _video_seconds(b, axis, fps)), duration)
        out.append([round(lo, 3), round(hi, 3)])
    return out


def _peak_concurrent(tracks: list["_TrackState"]) -> int:
    """The most tracks present in any ONE frame of this visit.

    A frame is an instant, so this is counted, not estimated. It replaces the
    `estimated_animals` guess, which compared a row's PEAK-frame box against
    another row's ARRIVAL timestamp — two different moments treated as one.

    Note what it is NOT: an upper bound on animals. A macaque nobody's track
    ever covered is not here, and two ids on one animal in the same frames were
    already collapsed by _merge_duplicate_tracks before this runs. It answers
    "how many did we follow at once", which is the honest claim.
    """
    if not tracks:
        return 0
    present: collections.Counter[int] = collections.Counter()
    for track in tracks:
        for idx in set(track.frame_indices):
            present[idx] += 1
    return max(present.values(), default=0)


class _PendingVisit:
    """One finalised group waiting to see whether the visit is really over."""

    __slots__ = ("camera_id", "survivors", "concurrent", "frames",
                 "opened_at", "deadline", "groups")

    def __init__(self, camera_id, survivors, concurrent, frames, now):
        self.camera_id = camera_id
        self.survivors = list(survivors)
        self.concurrent = list(concurrent)
        self.frames = frames
        self.opened_at = now
        self.deadline = now + VISIT_HOLD_SECONDS
        self.groups = 1

    def absorb(self, survivors, concurrent, frames, now) -> None:
        self.survivors.extend(survivors)
        self.concurrent.extend(concurrent)
        self.frames.merge_from(frames)
        self.groups += 1
        # Each new group restarts the window but can never push the visit past
        # its ceiling: a troop that keeps producing groups would otherwise be a
        # clip that never lands.
        self.deadline = min(now + VISIT_HOLD_SECONDS,
                            self.opened_at + VISIT_HOLD_MAX_SECONDS)


class _VisitQueue:
    """Holds finalised groups so one visit renders as one clip.

    A group finalises when its tracks are all lost, but the visit is not over —
    measured 2026-08-27, 341 visits produced 570 clips (median 1.7, worst 7).
    Holding a group for VISIT_HOLD_SECONDS and merging whatever finalises
    behind it on the same camera collapses those into one.

    Why this is a worker and not a timer on the detector thread: rendering a
    clip uploads blobs and writes the DB, and that work has killed the detector
    thread twice (2026-08-06, 08-08 — one of them stopped 場域甲 being watched
    for 62 hours). `offer()` never blocks, never raises, and returns False when
    it will not take the work, so the caller renders inline exactly as before.

    Memory is the real bound, not queue depth: a held visit carries its own
    frames because the ring buffer evicts them long before the hold expires.
    """

    def __init__(self) -> None:
        self._pending: dict[str, _PendingVisit] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._render = None
        self.held = self.merged = self.rendered = self.refused = 0

    @property
    def enabled(self) -> bool:
        return VISIT_HOLD_SECONDS > 0

    @property
    def _frame_ceiling(self) -> int:
        return max(1, VISIT_QUEUE_MAX) * TRAIL_GIF_MAX_FRAMES

    def _ensure_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="visit-queue", daemon=True)
        self._thread.start()

    def offer(self, camera_id, survivors, concurrent, frames, render) -> bool:
        """Take the visit, or return False so the caller renders it now."""
        if not self.enabled or not survivors:
            return False
        now = time.monotonic()
        with self._lock:
            open_visit = self._pending.get(camera_id)
            if open_visit is not None:
                open_visit.absorb(survivors, concurrent, frames, now)
                self.merged += 1
                self._ensure_worker()
                return True
            held_frames = sum(len(v.frames) for v in self._pending.values())
            if held_frames + len(frames) > self._frame_ceiling:
                # Refusing is the safe failure: today's behaviour, not a heap
                # that grows until the box is killed.
                self.refused += 1
                return False
            self._pending[camera_id] = _PendingVisit(
                camera_id, survivors, concurrent, frames, now)
            self._render = render
            self.held += 1
            self._ensure_worker()
            return True

    def _due(self, now: float) -> list[_PendingVisit]:
        with self._lock:
            due = [v for v in self._pending.values() if now >= v.deadline]
            for v in due:
                self._pending.pop(v.camera_id, None)
            return due

    def _run(self) -> None:
        while True:
            time.sleep(0.5)
            try:
                for visit in self._due(time.monotonic()):
                    try:
                        if self._render is not None:
                            self._render(visit)
                        self.rendered += 1
                        logger.info(
                            "Camera %s: visit rendered from %d group(s), %d track(s)"
                            " — one visit, one clip",
                            visit.camera_id, visit.groups, len(visit.survivors),
                        )
                    except Exception:
                        logger.exception(
                            "Rendering held visit for camera %s failed — "
                            "detection lost, worker continues", visit.camera_id,
                        )
            except Exception:
                logger.exception("Visit queue loop error — continuing")


_visit_queue = _VisitQueue()


class _FrameBuffer:
    """Ring buffer that keeps the last N frames JPEG-compressed, keyed by frame_count.

    Shared across all tracks so we don't duplicate frame copies. Each entry
    stores JPEG bytes and the wall-clock timestamp so GIF generation can
    compute real-time playback duration.

    Frames MUST stay compressed here: raw 1080p BGR is ~6 MB/frame, so a full
    600-frame buffer would be ~3.7 GB per camera — with 4 cameras that
    OOM-killed the 8Gi container every ~10 minutes. JPEG q90 keeps the same
    window at ~0.4 MB/frame; decoding only happens during (rare) finalisation.
    """

    _JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 90]

    def __init__(self, max_size: int = TRAIL_GIF_MAX_FRAMES * 4) -> None:
        self._buf: OrderedDict[int, tuple[bytes, float]] = OrderedDict()
        self._max = max_size

    def put(self, idx: int, frame: np.ndarray, ts: float) -> None:
        ok, jpg = cv2.imencode(".jpg", frame, self._JPEG_PARAMS)
        if not ok:
            return
        self._buf[idx] = (jpg.tobytes(), ts)
        while len(self._buf) > self._max:
            self._buf.popitem(last=False)

    def get(self, idx: int) -> Optional[np.ndarray]:
        """Decode and return the frame at idx (a fresh array, safe to mutate)."""
        entry = self._buf.get(idx)
        if entry is None:
            return None
        return cv2.imdecode(np.frombuffer(entry[0], dtype=np.uint8), cv2.IMREAD_COLOR)

    def get_jpeg(self, idx: int) -> Optional[bytes]:
        """Return the stored JPEG bytes without decoding (for ZIP/upload reuse)."""
        entry = self._buf.get(idx)
        return entry[0] if entry is not None else None

    def timestamps(self) -> dict[int, float]:
        """{frame index: wall clock} for everything held."""
        return {idx: ts for idx, (_, ts) in self._buf.items()}

    def get_ts(self, idx: int) -> Optional[float]:
        entry = self._buf.get(idx)
        return entry[1] if entry is not None else None

    def __len__(self) -> int:
        return len(self._buf)

    def snapshot(self, indices: list[int]) -> "_FrameBuffer":
        """A detached buffer holding just these frames.

        A held visit renders minutes after the detector saw it, by which point
        the shared ring has evicted everything it needs — so the visit has to
        carry its own copy. Entries are moved across as the stored (jpeg, ts)
        tuples: re-encoding here would cost a second q90 round trip, and one
        round trip has already been measured dropping model confidence from
        0.83 to 0.22.
        """
        out = _FrameBuffer(max_size=max(1, len(indices)))
        for idx in indices:
            entry = self._buf.get(idx)
            if entry is not None:
                out._buf[idx] = entry
        return out

    def merge_from(self, other: "_FrameBuffer") -> None:
        """Absorb another snapshot's frames, keeping frame order."""
        self._buf.update(other._buf)
        self._buf = OrderedDict(sorted(self._buf.items()))
        self._max = max(self._max, len(self._buf))

    def indices_in_time_range(self, t_start: float, t_end: float) -> list[int]:
        """Indices whose capture timestamp falls in [t_start, t_end], sorted."""
        return sorted(i for i, (_, ts) in self._buf.items() if t_start <= ts <= t_end)

    def indices_in_range(self, start: int, end: int) -> list[int]:
        """Return all stored indices within [start, end], inclusive, sorted."""
        return sorted(i for i in self._buf if start <= i <= end)


# ── Combined trail video ────────────────────────────────────────────────────


def _draw_track_on_canvas(
    canvas: np.ndarray,
    t: "_TrackState",
    color: tuple,
    fi: int,
    n_centers: int,
    line_thickness: int,
    draw_bbox: bool = True,
) -> None:
    """Draw one track's trail and current bbox onto canvas at frame fi.

    No text is burned into the image (per-track colour distinguishes
    monkeys); non-admin users see these images, so no confidence/ID label.

    `draw_bbox=False` is used for post-detection buffer frames so the trail
    stays visible but the bbox disappears (monkey has left the scene).
    """
    pts = t.centers[:n_centers]

    # Trail lines: fade from dim to bright, thicken toward present
    for j in range(1, len(pts)):
        ratio = j / max(len(pts) - 1, 1)
        dim = 0.35 + 0.65 * ratio
        c = tuple(int(ch * dim) for ch in color)
        thick = max(line_thickness, int(line_thickness * 1.5 * ratio))
        cv2.line(canvas, pts[j - 1], pts[j], c, thick, cv2.LINE_AA)

    if not draw_bbox:
        return

    # Keep historical trails, but only draw a box near an actual observation.
    # An unbounded nearest match paints ghost boxes across tracking dropouts.
    bbox = t.bbox_near(fi, _COMPANION_MAX_FRAME_GAP)
    if bbox is not None:
        bx1, by1, bw, bh = bbox
        bx2, by2 = int(bx1 + bw), int(by1 + bh)
        cv2.rectangle(canvas, (int(bx1), int(by1)), (bx2, by2), color, line_thickness, cv2.LINE_AA)


# Pause before a crashed detector loop restarts — long enough that a hard,
# repeating failure (a broken model file, say) cannot spin the CPU or flood the
# log, short enough that a transient one costs a couple of seconds of watching.
_LOOP_RESTART_DELAY = 3.0


def _encode_mp4(frames: list, timestamps: list[float]) -> Optional[bytes]:
    """Encode every rendered trail frame as browser-playable H.264 MP4.

    MP4 needs a seekable output for ``+faststart``, so ffmpeg writes to a
    short-lived local file while RGB frames stream over stdin. The detector
    never relies on this succeeding: callers treat a failed video as a lost
    optional artefact and continue watching the camera.
    """
    if len(frames) < 2:
        return None

    width, height = frames[0].size
    if any(frame.size != (width, height) for frame in frames):
        logger.warning("Trail MP4 skipped: rendered frames have inconsistent dimensions")
        return None

    # The frames have already been placed on a constant-rate timeline by
    # `_resample_trail_frames`; use the same rate here so player offsets and
    # encoded playback cannot drift apart.
    fps = _clip_fps(timestamps)
    output_path = None
    process = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output:
            output_path = output.name

        process = subprocess.Popen(
            [
                FFMPEG_BINARY, "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
                "-r", f"{fps:.3f}", "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", str(TRAIL_MP4_CRF), "-pix_fmt", "yuv420p",
                # Bound the download/decode span when opening at an animal's offset.
                "-g", str(max(1, round(fps * 2))),
                "-movflags", "+faststart", "-f", "mp4", "-y", output_path,
            ],
            stdin=subprocess.PIPE,
            # DEVNULL, not PIPE. Writing every frame to stdin before reading
            # stderr deadlocks if ffmpeg ever fills the ~64 KB stderr buffer
            # while still running: it blocks writing stderr, this thread blocks
            # writing stdin, and `wait(timeout=...)` never runs because we never
            # reach it. This is the detector thread — a hang here is the failure
            # that stopped 場域甲 being watched for 62 hours. The exit status
            # already says whether the encode worked, which is the only thing
            # the caller acts on.
            stderr=subprocess.DEVNULL,
        )
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        status = process.wait(timeout=30)
        if status != 0:
            logger.warning("Trail MP4 encoding failed (exit %d)", status)
            return None
        data = Path(output_path).read_bytes()
        if not data:
            logger.warning("Trail MP4 encoding produced an empty file")
            return None
        return data
    except Exception as exc:
        logger.warning("Trail MP4 encoding failed: %r", exc)
        return None
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if output_path:
            Path(output_path).unlink(missing_ok=True)


def _pack_raw_frames_under_limit(
    items: list[tuple[str, bytes]],
) -> tuple[bytes, int]:
    """ZIP the clean full-resolution JPEGs, subsampled to fit the byte ceiling.

    Stored (not deflated), so the archive is ~the sum of its parts and the fit
    can be decided before building it.
    """
    import io
    import zipfile

    kept = items
    if RAW_FRAMES_MAX_BYTES > 0:
        while len(kept) > 2 and sum(len(b) for _, b in kept) > RAW_FRAMES_MAX_BYTES:
            kept = kept[::2]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for fname, jpg_bytes in kept:
            zf.writestr(fname, jpg_bytes)
    return buf.getvalue(), len(kept)


def _generate_combined_trail_video(
    tracks: list["_TrackState"],
    frame_buf: "_FrameBuffer",
    camera_id: str,
) -> tuple[str, list[float], float]:
    """Draw all active tracks into a single H.264 MP4 and raw-frame ZIP.

    Returns the clip's name together with the time base it was written on — the
    wall clock of every frame it kept, and the rate. The rows that point into
    the clip are placed with these, so they cannot describe a different video
    from the one the encoder produced.

    Each monkey gets its own colour. Bounding boxes are drawn at every frame
    using nearby stored detections; observation gaps retain the historical
    trail without painting an animal at a stale position.

    Also saves a ZIP of clean (un-annotated) raw frames alongside the clip
    for dataset labelling purposes.
    """
    from PIL import Image

    if not tracks:
        return "", [], 0.0

    # Detection window across all tracks
    det_min = min(t.min_frame_idx for t in tracks)
    det_max = max(t.max_frame_idx for t in tracks)

    # Expand with pre/post buffer so the clip shows the monkey walking IN
    # (before first YOLO detection) and OUT (after last detection). The
    # approximation assumes ~25fps camera and FRAME_SKIP inference ratio.
    # Select by wall clock, not tick arithmetic: ticks-per-second depends on
    # the current throughput, so a tick-based window drifts whenever the frame
    # rate changes.
    t_first = frame_buf.get_ts(det_min)
    t_last = frame_buf.get_ts(det_max)
    if t_first is None or t_last is None:
        sorted_indices = frame_buf.indices_in_range(det_min, det_max)
    else:
        sorted_indices = frame_buf.indices_in_time_range(
            t_first - TRAIL_PRE_BUFFER_SECONDS, t_last + TRAIL_POST_BUFFER_SECONDS,
        )

    if len(sorted_indices) < 2:
        return "", [], 0.0

    # Uniform-time subsampling: pick frames evenly spread across the real
    # wall-clock span (not by index step). This ensures consistent playback
    # speed regardless of detector tick rate variations.
    if len(sorted_indices) > TRAIL_GIF_MAX_FRAMES:
        ts_pairs = [(i, frame_buf.get_ts(i)) for i in sorted_indices]
        ts_pairs = [(i, t) for i, t in ts_pairs if t is not None]
        if len(ts_pairs) > TRAIL_GIF_MAX_FRAMES and ts_pairs[-1][1] > ts_pairs[0][1]:
            t_start, t_end = ts_pairs[0][1], ts_pairs[-1][1]
            targets = [t_start + (t_end - t_start) * i / (TRAIL_GIF_MAX_FRAMES - 1) for i in range(TRAIL_GIF_MAX_FRAMES)]
            chosen: list[int] = []
            ptr = 0
            for tt in targets:
                while ptr + 1 < len(ts_pairs) and abs(ts_pairs[ptr + 1][1] - tt) < abs(ts_pairs[ptr][1] - tt):
                    ptr += 1
                chosen.append(ts_pairs[ptr][0])
            sorted_indices = chosen
        else:
            # Fallback to index-based subsample if timestamps unavailable
            step = len(sorted_indices) / TRAIL_GIF_MAX_FRAMES
            sorted_indices = [sorted_indices[int(i * step)] for i in range(TRAIL_GIF_MAX_FRAMES)]

    # Pre-compute cumulative center counts per track per frame
    track_data: list[dict] = []
    for color_idx, t in enumerate(tracks):
        idx_to_n: dict[int, int] = {}
        for i, fi in enumerate(t.frame_indices):
            idx_to_n[fi] = i + 1
        track_data.append({
            "track": t,
            "color": _TRACK_COLORS[color_idx % len(_TRACK_COLORS)],
            "idx_to_n": idx_to_n,
        })

    scale = TRAIL_GIF_SCALE
    pil_frames: list[Image.Image] = []
    frame_timestamps: list[float] = []  # wall-clock ts of each kept frame, for duration calc
    raw_jpeg_data: list[tuple[str, bytes]] = []  # (filename, jpeg_bytes) for raw frames ZIP

    for frame_seq, fi in enumerate(sorted_indices):
        # Buffer already holds JPEG bytes — reuse them verbatim for the
        # raw-frames ZIP (no re-encode) and decode once for annotation.
        raw_jpg = frame_buf.get_jpeg(fi)
        frame = frame_buf.get(fi)
        if raw_jpg is None or frame is None:
            continue

        raw_jpeg_data.append((
            f"{camera_id}_raw_frame{frame_seq:04d}.jpg",
            raw_jpg,
        ))

        canvas = frame  # get() returns a fresh decode; no copy needed
        h, w = canvas.shape[:2]

        # Scale drawing primitives to full-res frame dimensions
        ref_dim = max(w, h)
        line_thickness = max(3, ref_dim // 300)

        for td in track_data:
            t = td["track"]
            color = td["color"]
            idx_to_n = td["idx_to_n"]

            # Count centers seen up to this frame
            n = 0
            for past_fi in t.frame_indices:
                if past_fi <= fi:
                    n = idx_to_n[past_fi]
                else:
                    break
            if n < 1:
                continue

            # Post-buffer (fi > track's last detection): show trail but
            # suppress bbox — the monkey has left the scene.
            draw_bbox = fi <= t.max_frame_idx
            _draw_track_on_canvas(
                canvas, t, color, fi, n,
                line_thickness,
                draw_bbox=draw_bbox,
            )

        # Stage 2: animals the big model found on this frame that no track
        # covers. Drawn dashed-white so a viewer can tell "26n tracked this"
        # from "26x also saw this" — they are different claims. Whatever stage 2
        # has finished by now is used; nothing here waits for it.
        for ex in _second_stage_extras(camera_id, fi, tracks):
            x1, y1, x2, y2 = (int(v) for v in ex)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), max(2, line_thickness - 1))
            cv2.putText(canvas, "x", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.4, ref_dim / 2400), (255, 255, 255), 2, cv2.LINE_AA)

        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))
        ts = frame_buf.get_ts(fi)
        frame_timestamps.append(ts if ts is not None else 0.0)

    if len(pil_frames) < 2:
        return "", [], 0.0

    ts_str = datetime.now(_TW_TZ).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    tids = "+".join(str(t.track_id) for t in tracks)
    base_name = f"{camera_id}_tracks{tids}_{ts_str}_{uid}"
    video_name = f"{base_name}_trail.mp4"

    # H.264 inter-frame compression makes the selected clip comfortably small.
    # Empty playback slots are repeated references, not extra decoded images.
    video_frames, video_timestamps, video_fps = _resample_trail_frames(
        pil_frames, frame_timestamps, TRAIL_VIDEO_FPS,
    )
    video_bytes = _encode_mp4(video_frames, video_timestamps)

    # Neither artefact may cost the camera its detector thread. Both are
    # required artefacts layered on the detection record — the trail animation
    # and the labelling material — and both are pushed over a site uplink to
    # Azure. An uncaught upload timeout here propagated out of _run() twice and
    # left 場域甲 blind for 62 hours (2026-08-06, 2026-08-08). A failed upload is
    # logged and the caller drops the incomplete detection, while the worker
    # itself keeps running.
    from services.storage import storage

    upload_failed = False
    if video_bytes is None:
        video_name = ""
    else:
        try:
            storage.upload(video_name, video_bytes)
        except Exception:
            logger.exception(
                "Trail MP4 upload failed (%s, %.1f MB) — detection will be dropped",
                video_name, len(video_bytes) / 1024 / 1024,
            )
            video_name = ""
            upload_failed = True

    # Save raw (clean, no annotation) frames as a ZIP for dataset labelling.
    # Skipped outright when the clip upload just failed: this thread is not
    # inferring its camera while it uploads, so once the network has shown it is
    # down there is nothing to gain by spending the timeout budget again on the
    # bigger of the two artefacts.
    raw_zip_name = f"{base_name}_rawframes.zip"
    zip_frames = 0
    if upload_failed:
        raw_zip_name = ""
        logger.warning("Skipping raw-frames ZIP upload — storage just failed")
    else:
        zip_bytes, zip_frames = _pack_raw_frames_under_limit(raw_jpeg_data)
        try:
            storage.upload(raw_zip_name, zip_bytes)
        except Exception:
            logger.exception(
                "Raw-frames ZIP upload failed (%s, %.1f MB) — video and detection unaffected",
                raw_zip_name, len(zip_bytes) / 1024 / 1024,
            )
            raw_zip_name = ""

    logger.info(
        "Combined trail MP4 saved: %s (%d tracks, %d selected/%d encoded frames @ %.1f FPS, raw ZIP: %s, %d frames)",
        video_name or "(skipped)", len(tracks), len(pil_frames), len(video_frames), video_fps,
        raw_zip_name or "(failed)", zip_frames,
    )
    return video_name, _clip_axis(video_timestamps), video_fps


# ── Inference metrics (rolling window, thread-safe) ─────────────────────────


class _InferenceMetrics:
    """Tracks recent model.track() latency, sample rate, and wasted work.

    Latency and FPS alone hid the bug that motivated the turn-first loop: the
    polling loop this replaced re-ran inference on frames it had already
    processed (74% of its 42 inferences/sec were duplicate work on an
    unchanged frame — only 11 distinct frames/sec) and reported all of it as
    throughput. So distinct frames, duplicates and idle turns are counted
    separately here: `duplicate_inferences` must stay 0, and `idle_ratio` is
    how often this camera's GPU turn found nothing to do.
    """

    _WINDOW = 30

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=self._WINDOW)
        self._wall_times: deque[float] = deque(maxlen=self._WINDOW)
        # Unchanged-poll count as of each recorded inference, so the idle
        # fraction is reported over the same rolling window as sample_fps
        # instead of averaging over the whole process lifetime.
        self._idle_marks: deque[int] = deque(maxlen=self._WINDOW)
        # Same trick for the idle gate, so `gated_ratio` reports the recent
        # window rather than the whole process lifetime.
        self._gated_marks: deque[int] = deque(maxlen=self._WINDOW)
        self.last_ms: float = 0.0
        self.total_inferences: int = 0
        self.frames_processed: int = 0
        self.stage_sec: dict[str, float] = {}
        self.frames_unchanged: int = 0
        self.frames_gated: int = 0
        # White-balance rescues, and the cast that triggered the last one. A
        # camera whose WB has failed looks identical in every other metric.
        self.frames_wb_corrected: int = 0
        self.last_cast: float = 0.0
        self.duplicate_inferences: int = 0
        self._last_seq: int = -1

    def record(self, elapsed_sec: float, *, frame_seq: int = -1) -> None:
        """Record one inference. `frame_seq` is the camera capture sequence the
        frame came from; a sequence that is not newer than the previous one is
        counted as duplicate work, never as a processed frame.
        """
        now = time.time()
        with self._lock:
            self._latencies.append(elapsed_sec)
            self.last_ms = elapsed_sec * 1000.0
            self.total_inferences += 1
            if 0 <= frame_seq <= self._last_seq:
                # Duplicate work must never inflate sample_fps — the wall-time
                # window is built from DISTINCT frames only, so re-inferring an
                # unchanged frame shows up as the red 重複 counter, not as
                # throughput. (The old polling loop hid a 74%-duplicate rate
                # behind exactly this metric.)
                self.duplicate_inferences += 1
            else:
                self.frames_processed += 1
                self._last_seq = frame_seq
                self._wall_times.append(now)
                self._idle_marks.append(self.frames_unchanged)
                self._gated_marks.append(self.frames_gated)

    def record_unchanged(self) -> None:
        """A turn that found no frame newer than the last one consumed."""
        with self._lock:
            self.frames_unchanged += 1

    def record_wb_corrected(self, cast: float) -> None:
        """A frame whose colour cast was bad enough to rebalance before inference.

        Surfaced because it is the only visible sign the camera's white balance
        has failed — the dawn failure that motivated this produced 1 detection
        across 59 frames and nothing else in the system said why.
        """
        with self._lock:
            self.frames_wb_corrected += 1
            self.last_cast = round(cast, 3)

    def record_gated(self) -> None:
        """A frame the idle gate dropped — the scene carried no image at all.

        Counted separately from `frames_unchanged` (no frame was available) so
        the dashboard can tell "this camera is being deliberately sampled at a
        heartbeat" from "this camera has stopped delivering".
        """
        with self._lock:
            self.frames_gated += 1

    def add_stage(self, stage: str, sec: float) -> None:
        """Accumulate wall time per loop stage — production has ~250 ms/frame
        unaccounted for and per-stage totals are the only way to see where."""
        with self._lock:
            self.stage_sec[stage] = self.stage_sec.get(stage, 0.0) + sec

    def snapshot(self, *, stream_connected: bool) -> dict:
        with self._lock:
            if not self._latencies:
                return {
                    "stream_connected": stream_connected,
                    "inference_ms_last": 0.0,
                    "inference_ms_avg": 0.0,
                    "throughput_fps": 0.0,
                    "sample_fps": 0.0,
                    "total_inferences": self.total_inferences,
                    "frames_processed": self.frames_processed,
                    "stage_sec": {k: round(v, 1) for k, v in self.stage_sec.items()},
                    "frames_unchanged": self.frames_unchanged,
                    "frames_gated": self.frames_gated,
                "frames_wb_corrected": self.frames_wb_corrected,
                "last_cast": self.last_cast,
                    "duplicate_inferences": self.duplicate_inferences,
                    "idle_ratio": 0.0,
                    "gated_ratio": 0.0,
                    "seconds_since_last": None,
                }

            avg_ms = statistics.mean(self._latencies) * 1000.0
            throughput = 1000.0 / avg_ms if avg_ms > 0 else 0.0

            sample_fps = 0.0
            if len(self._wall_times) >= 2:
                span = self._wall_times[-1] - self._wall_times[0]
                if span > 0:
                    sample_fps = (len(self._wall_times) - 1) / span

            # Turns in this window that found nothing, over all turns taken —
            # the share of GPU turns this camera gave back unused.
            window_idle = self.frames_unchanged - self._idle_marks[0]
            window_turns = window_idle + len(self._wall_times) - 1
            idle_ratio = window_idle / window_turns if window_turns > 0 else 0.0

            # Share of delivered frames the idle gate dropped over the same
            # window. High is the feature working (a black feed sampled at a
            # heartbeat), not work being lost.
            window_gated = self.frames_gated - self._gated_marks[0]
            window_frames = window_gated + len(self._wall_times) - 1
            gated_ratio = window_gated / window_frames if window_frames > 0 else 0.0

            since_last = time.time() - self._wall_times[-1]

            return {
                "stream_connected": stream_connected,
                "inference_ms_last": round(self.last_ms, 1),
                "inference_ms_avg": round(avg_ms, 1),
                "throughput_fps": round(throughput, 2),
                "sample_fps": round(sample_fps, 2),
                "total_inferences": self.total_inferences,
                "frames_processed": self.frames_processed,
                "stage_sec": {k: round(v, 1) for k, v in self.stage_sec.items()},
                "frames_unchanged": self.frames_unchanged,
                "frames_gated": self.frames_gated,
                "frames_wb_corrected": self.frames_wb_corrected,
                "last_cast": self.last_cast,
                "duplicate_inferences": self.duplicate_inferences,
                "idle_ratio": round(idle_ratio, 3),
                "gated_ratio": round(gated_ratio, 3),
                "seconds_since_last": round(since_last, 1),
            }


class _POCBatchRequest:
    """One or more frames waiting for the low-priority POC inference turn."""

    def __init__(
        self,
        frame,
        imgsz: int,
        device: str,
        half: bool,
        tracker,
        *,
        frames: Optional[list[np.ndarray]] = None,
    ) -> None:
        self.frames = list(frames) if frames is not None else [frame]
        self.frame = self.frames[0]
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.tracker = tracker
        self.done = threading.Event()
        self.results = None
        self.tracks = None
        self.error: Optional[Exception] = None


class _POCBatchInference:
    """Batch same-shaped POC frames while keeping tracker state per camera.

    The worker owns the only shared predictor and therefore serialises all
    model calls itself. It never mixes source shapes: Ultralytics' production
    ``rect=True`` preprocessing is only identical to the single-frame path
    when every image in the batch has the same shape. A source with a different
    shape is processed as a one-item batch, which keeps the exact same model
    input and still uses the same tracker adapter.
    """

    def __init__(self, device: str, half: bool) -> None:
        self.device = device
        self.half = half
        self._queue: queue.Queue[_POCBatchRequest] = queue.Queue(maxsize=32)
        self._model = None
        self._stats_lock = threading.Lock()
        self._batch_count = 0
        self._frame_count = 0
        self._batch_size_sum = 0
        self._last_batch_size = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"poc-batch-{device or 'cpu'}",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame, imgsz: int, tracker):
        outputs = self.submit_many([frame], imgsz, tracker)
        return outputs[0]

    def submit_many(self, frames: list[np.ndarray], imgsz: int, tracker):
        if not frames:
            return []
        request = _POCBatchRequest(
            frames[0], imgsz, self.device, self.half, tracker, frames=frames,
        )
        self._queue.put(request)
        if not request.done.wait(timeout=60):
            raise TimeoutError("POC batch inference did not complete within 60 seconds")
        if request.error is not None:
            raise request.error
        return list(zip(request.results, request.tracks))

    @staticmethod
    def _compatible(first: _POCBatchRequest, other: _POCBatchRequest) -> bool:
        return (
            first.imgsz == other.imgsz
            and first.device == other.device
            and first.half == other.half
            and all(frame.shape == first.frame.shape for frame in other.frames)
        )

    def _model_for_request(self, request: _POCBatchRequest):
        if self._model is None:
            self._model = _own_model(request.device, request.half)
        return self._model

    def _run_batch(self, batch: list[_POCBatchRequest]) -> None:
        first = batch[0]
        model = self._model_for_request(first)
        frames = [frame for request in batch for frame in request.frames]
        with self._stats_lock:
            self._batch_count += 1
            self._frame_count += len(frames)
            self._batch_size_sum += len(frames)
            self._last_batch_size = len(frames)
        # This is the same preprocessing/NMS route as model.track(), minus its
        # callback-owned tracker. StandardTracker applies that callback's
        # ByteTrack update independently for each camera below.
        results = model.predict(
            frames,
            imgsz=first.imgsz,
            rect=True,
            batch=len(frames),
            verbose=False,
            conf=TRACKER_CONFIDENCE_THRESHOLD,
            classes=[0],
            device=first.device or None,
        )
        if len(results) != len(frames):
            raise RuntimeError(
                f"POC batch returned {len(results)} results for {len(frames)} frames"
            )
        offset = 0
        for request in batch:
            count = len(request.frames)
            request.results = list(results[offset : offset + count])
            request.tracks = [
                request.tracker.update(result) for result in request.results
            ]
            offset += count

    def stats(self) -> dict[str, float | int]:
        with self._stats_lock:
            return {
                "batch_count": self._batch_count,
                "batch_frames": self._frame_count,
                "batch_size_avg": (
                    self._batch_size_sum / self._batch_count
                    if self._batch_count else 0.0
                ),
                "batch_size_last": self._last_batch_size,
            }

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            batch = [first]
            batch_frames = len(first.frames)
            deferred: list[_POCBatchRequest] = []
            deadline = time.monotonic() + POC_BATCH_WAIT_MS / 1000.0
            while batch_frames < POC_BATCH_MAX:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    request = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if (
                    self._compatible(first, request)
                    and batch_frames + len(request.frames) <= POC_BATCH_MAX
                ):
                    batch.append(request)
                    batch_frames += len(request.frames)
                else:
                    deferred.append(request)
            for request in deferred:
                self._queue.put(request)

            try:
                if self.device not in ("", "cpu") and INFERENCE_SERIALISE:
                    _gpu_lock.acquire(priority=1)
                    try:
                        self._run_batch(batch)
                    finally:
                        _gpu_lock.release()
                else:
                    self._run_batch(batch)
            except Exception as exc:
                for request in batch:
                    request.error = exc
            finally:
                for request in batch:
                    request.done.set()


_poc_batchers: dict[tuple[str, bool], _POCBatchInference] = {}
_poc_batchers_lock = threading.Lock()


def _poc_batcher(device: str, half: bool) -> _POCBatchInference:
    key = (device, half)
    with _poc_batchers_lock:
        batcher = _poc_batchers.get(key)
        if batcher is None:
            batcher = _POCBatchInference(device, half)
            _poc_batchers[key] = batcher
        return batcher


# ── Detector worker ──────────────────────────────────────────────────────────


class DetectorWorker:
    """Background worker that runs YOLO tracking on camera frames."""

    def __init__(
        self,
        camera_id: str,
        on_detection: Optional[callable] = None,
        on_raw_detection: Optional[callable] = None,
    ) -> None:
        self.camera_id = camera_id
        self._on_detection = on_detection
        self._on_raw_detection = on_raw_detection
        self.frame_skip = frame_skip_for_camera(camera_id)
        self.inference_device = inference_device_for_camera(camera_id)
        self.inference_half = inference_half_for_camera(camera_id)
        self._poc_batch = poc_batch_for_camera(camera_id)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_raw_notify: float = 0.0
        self._metrics = _InferenceMetrics()
        # Resolved from the first frame this camera delivers and then fixed —
        # a stream does not change resolution mid-run, and feeding ultralytics a
        # varying imgsz would rebuild the predictor (and with it the tracker).
        self._imgsz: Optional[int] = None
        # Liveness accounting, surfaced to the dashboard: a camera that stops
        # being inferred used to be indistinguishable from a slow one.
        self.crashes = 0     # loop restarts after an uncaught exception
        self.revivals = 0    # thread restarts by the manager's watchdog

    def get_metrics(self, *, stream_connected: bool) -> dict:
        m = self._metrics.snapshot(stream_connected=stream_connected)
        # The size actually used, not the floor — they differ on any camera that
        # delivers more than INFERENCE_IMGSZ, and a dashboard showing the floor
        # is how the 1600-vs-1920x1080 mismatch stayed invisible.
        m["imgsz"] = self._imgsz
        m["inference_device"] = self.inference_device or "cpu"
        if self._poc_batch:
            batcher = _poc_batchers.get((self.inference_device, self.inference_half))
            if batcher is not None:
                m.update(batcher.stats())
        return m

    @property
    def alive(self) -> bool:
        """Is this camera's detector THREAD actually running?

        Not the same question as `stream_connected`, which asks the capture
        thread. On 2026-08-08 the two disagreed for 62 hours: the stream kept
        reconnecting happily while the detector thread was dead, so the UI
        showed 場域甲 as connected at its last-known FPS.
        """
        return self._thread is not None and self._thread.is_alive()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._running:
            return
        self._running = True
        self._loop = loop
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Detector (tracking) for camera %s started", self.camera_id)

    def revive(self) -> bool:
        """Restart the thread if it should be running but isn't. True if it did.

        Last resort behind the in-thread supervisor in `_run` — for whatever kills
        a thread outright rather than raising inside the loop.
        """
        if not self._running or self.alive:
            return False
        self.revivals += 1
        logger.error(
            "Detector thread for camera %s is DEAD — reviving (revival #%d)",
            self.camera_id, self.revivals,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
        logger.info("Detector for camera %s stopped", self.camera_id)

    # ── main loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Thread entry — supervises `_detect_loop` so no exception can end a camera.

        The loop below used to BE the thread target, which made every line in it
        a single point of failure for the camera: on 2026-08-06 and again on
        2026-08-08 a blob-upload timeout raised out of `_finalise_batch`, out of
        the thread, and 場域甲 silently stopped being watched (62 hours the second
        time) while its stream stayed connected and the dashboard kept showing
        its last rolling-window FPS. Individual failures are contained closer to
        where they happen; this is the guarantee that a miss anywhere costs a
        restart, not the camera. Tracker state and the frame buffer are rebuilt
        on restart — the same price the watchdog would charge, paid faster.
        """
        while self._running:
            try:
                self._detect_loop()
                return  # clean exit: _running went False
            except Exception:
                self.crashes += 1
                logger.exception(
                    "Detector loop for camera %s crashed (crash #%d) — restarting in %.0fs",
                    self.camera_id, self.crashes, _LOOP_RESTART_DELAY,
                )
                time.sleep(_LOOP_RESTART_DELAY)

    def _init_tiled_path(self) -> tuple[Optional[object], Optional[TiledTracker]]:
        """Set up the ANE tiled CoreML path for this camera, if configured.

        Returns (tile_model, tiled_tracker) or (None, None) when the camera
        is not in ANE_TILE_CAMERAS or the CoreML model cannot be loaded — the
        caller then falls back to the standard model.track() path so a broken
        export never produces a dead camera. Each worker loads its OWN model
        instance (coremltools thread-safety across camera threads is
        unproven), so each worker builds its own instance here too.
        """
        fusion = self.camera_id in ANE_FUSION_CAMERAS
        if not fusion and self.camera_id not in ANE_TILE_CAMERAS:
            return None, None
        try:
            if not Path(ANE_TILE_MODEL_PATH).exists():
                raise FileNotFoundError(f"CoreML model not found at {ANE_TILE_MODEL_PATH}")
            from ultralytics import YOLO
            tile_model = YOLO(ANE_TILE_MODEL_PATH, task="detect")
            # Pre-warm so the first real frame doesn't pay compilation cost.
            tile_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)
            tiled_tracker = TiledTracker()
            logger.info(
                "Camera %s: %s active (%s)",
                self.camera_id,
                "ANE fusion (full-frame + gated tile sweep)" if fusion else "ANE tiled inference",
                ANE_TILE_MODEL_PATH,
            )
            if DEV_RAW_WEBHOOK:
                logger.info(
                    "Camera %s: DEV raw webhook disabled on the tiled path "
                    "(needs a Results object)", self.camera_id,
                )
            return tile_model, tiled_tracker
        except Exception:
            logger.exception(
                "Camera %s: tiled CoreML path failed to initialise (%s) — "
                "falling back to standard path", self.camera_id, ANE_TILE_MODEL_PATH,
            )
            return None, None

    def _update_track(
        self,
        active_tracks: dict[int, _TrackState],
        tid: int,
        conf: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        frame_count: int,
        frame: np.ndarray,
        captured_ts: Optional[float] = None,
    ) -> None:
        """Per-box track bookkeeping shared by both inference paths."""
        bw, bh = x2 - x1, y2 - y1
        cx, cy = int(x1 + bw / 2), int(y1 + bh / 2)

        if tid not in active_tracks:
            active_tracks[tid] = _TrackState(tid, self.camera_id, frame_count)

        active_tracks[tid].update(
            center=(cx, cy),
            conf=conf,
            bbox=(x1, y1, bw, bh),
            frame_idx=frame_count,
            frame=frame,
            captured_ts=captured_ts,
        )

    def _postprocess_poc_frame(
        self,
        *,
        frame: np.ndarray,
        seq: int,
        captured_ts: float,
        results,
        tracks: np.ndarray,
        frame_count: int,
        active_tracks: dict[int, _TrackState],
        frame_buf: _FrameBuffer,
    ) -> int:
        """Apply one POC frame's tracker output after a possibly shared batch."""
        frame_count += 1
        t_post = time.perf_counter()
        frame_buf.put(frame_count, frame, captured_ts)

        if second_stage.enabled and results is not None:
            peak = 0.0
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes):
                    peak = max(peak, float(boxes.conf.max()))
            if peak >= SECOND_STAGE_GATE_CONF:
                second_stage.offer(self.camera_id, frame_count, frame, self._imgsz)

        for row in tracks:
            self._update_track(
                active_tracks,
                tid=int(row[4]),
                conf=float(row[5]),
                x1=float(row[0]),
                y1=float(row[1]),
                x2=float(row[2]),
                y2=float(row[3]),
                frame_count=frame_count,
                frame=frame,
                captured_ts=captured_ts,
            )


        self._metrics.add_stage("bookkeeping", time.perf_counter() - t_post)
        now_ts = time.time()
        lost_ids = [
            tid for tid, ts in active_tracks.items()
            if (now_ts - ts.last_seen_ts) > TRACK_LOST_SECONDS
        ]
        if lost_ids:
            lost_tracks = [active_tracks.pop(tid) for tid in lost_ids]
            t_fin = time.perf_counter()
            self._finalise_batch(lost_tracks, active_tracks, frame_buf)
            self._metrics.add_stage("finalise", time.perf_counter() - t_fin)
        return frame_count

    def _detect_loop(self) -> None:
        from services.stream import stream_manager

        tile_model, tiled_tracker = self._init_tiled_path()
        fusion = tile_model is not None and self.camera_id in ANE_FUSION_CAMERAS
        # Each worker gets its OWN model instance, which is what makes tracking
        # correct: ultralytics keeps ByteTrack state on the predictor, so a
        # shared model means a shared tracker — and a shared tracker means only
        # the first camera to establish a track ever gets IDs (a new track needs
        # two consecutive updates to activate, which interleaved cameras never
        # provide). Every other camera stayed permanently blind. Separate
        # instances also stop four threads racing on one predictor's state.
        batch_tracker = StandardTracker() if self._poc_batch and tile_model is None else None
        if (tile_model is None or fusion) and not self._poc_batch:
            # Keep the legacy zero-argument seam for tests and callers that use
            # the global device. Only a real per-camera override needs the
            # parameterised constructor.
            if (
                self.inference_device == INFERENCE_DEVICE
                and self.inference_half == INFERENCE_HALF
            ):
                model = _own_model()
            else:
                model = _own_model(self.inference_device, self.inference_half)
        else:
            model = None
        frame_count = 0        # processed frames (drives track bookkeeping)
        captured_seq = 0       # last camera frame this worker consumed
        last_infer_ts = 0.0    # wall clock of the last inference (idle gate)
        active_tracks: dict[int, _TrackState] = {}
        frame_buf = _FrameBuffer()
        # Only the model.track() path queues for the GPU; the ANE tiled/fusion
        # paths run on the Neural Engine and never took a turn.
        serialised = (
            INFERENCE_SERIALISE
            and tile_model is None
            and not self._poc_batch
            and self.inference_device not in ("", "cpu")
        )

        while self._running:
            stream = stream_manager.get(self.camera_id)
            if stream is None or not stream.connected:
                time.sleep(1)
                continue

            frame = None
            results = None   # model.track() path
            tracks = None    # ANE tiled path

            # Claim the GPU turn BEFORE picking a frame. Waiting on the camera
            # first meant a worker held nothing while it waited, so whenever all
            # four waited at once the GPU sat idle: 18 distinct frames/sec
            # against a ~40/sec ceiling (25-28 ms per inference at imgsz 1600,
            # fp16) at 45-74% utilisation. Queued first, the GPU always has a
            # caller, and a turn comes round every ~100 ms while frames arrive
            # every ~33 ms, so the frame taken is essentially always unseen.
            # Pull the oldest unseen frame from the camera's burst queue.
            # RTSP delivers in bursts (inter-frame gaps p50 ~1 ms, silences of
            # 200-3000 ms between bursts, measured in production), so a
            # latest-frame-only consumer could act just once per burst — 3-4
            # fps against 33 fps of delivery. Draining the queue processes the
            # whole burst during the silence that follows it. Waiting is timed
            # with measured wall clock so the accounting cannot lie.
            t_wait = time.perf_counter()
            frame, seq, cap_ts = stream.next_frame(
                captured_seq + self.frame_skip - 1, timeout=0.5,
            )
            self._metrics.add_stage("wait_frame", time.perf_counter() - t_wait)
            if frame is None:
                self._metrics.record_unchanged()
                continue
            captured_seq = seq
            captured_ts = cap_ts

            # Idle gate — the GPU turn this camera does NOT take is the whole
            # point: a lens-black night feed is sampled once every
            # IDLE_GATE_HEARTBEAT_SECONDS so the turns go to cameras whose
            # frames contain an image. Measured in production: three blacked-out
            # cameras (pixel std 1.0-1.7) were being served the same 11.5 FPS as
            # the one lit camera, because _FairLock splits the GPU evenly and
            # nothing in this loop knew the difference.
            #
            # Deliberately NOT the motion gate reverted in 7098971: that one
            # skipped frames in a scene you could see, where a motionless animal
            # could hide. Here `is_blank` means there is no image to look at, at
            # any confidence. Three guards keep it honest:
            #   - a camera with a live track is never gated, whatever the light;
            #   - the heartbeat keeps the model itself checking the scene, so a
            #     wrong std verdict costs one interval, not the night;
            #   - vitality re-arms full rate within ~1 s of the scene lighting up.
            # Gated frames are not buffered either: a black pre-roll adds nothing
            # to a trail clip, and the JPEG encode is pure CPU cost.
            if (
                IDLE_GATE_HEARTBEAT_SECONDS > 0
                and not active_tracks
                and stream.is_blank
                and (time.time() - last_infer_ts) < IDLE_GATE_HEARTBEAT_SECONDS
            ):
                self._metrics.record_gated()
                continue

            # Never shrink the frame. Resolved once per camera: 1280x720 feeds
            # infer at the INFERENCE_IMGSZ floor exactly as before, and 場域甲's
            # 1920x1080 stops being squeezed into 1600 — which cost it 22% of
            # its recall while nobody could see the setting had stopped matching
            # the camera.
            if frame is not None and self._imgsz is None:
                self._imgsz = imgsz_for_frame(frame.shape[0], frame.shape[1])
                if self._imgsz != INFERENCE_IMGSZ:
                    logger.info(
                        "Camera %s delivers %dx%d — inferring at %d, not the %d floor",
                        self.camera_id, frame.shape[1], frame.shape[0],
                        self._imgsz, INFERENCE_IMGSZ,
                    )
            imgsz = self._imgsz or INFERENCE_IMGSZ

            # A broken white balance blinds the detector far more than it
            # bothers the eye. Only frames whose cast is genuinely off are
            # touched; a healthy frame skips the copy entirely.
            if frame is not None and WHITE_BALANCE_CAST_THRESHOLD > 0:
                cast = colour_cast(frame)
                if cast > WHITE_BALANCE_CAST_THRESHOLD:
                    frame = correct_white_balance(frame)
                    self._metrics.record_wb_corrected(cast)

            # A POC worker may have another sampled frame already waiting in
            # the burst queue. Pull one consecutive frame so a single YouTube
            # source can benefit from batching too; this is still one camera's
            # private tracker, so frame order remains deterministic. If the
            # source changes shape, submit the two frames separately rather
            # than changing rectangular preprocessing.
            poc_inputs = [(frame, seq, captured_ts)]
            if self._poc_batch and POC_BATCH_MAX > 1:
                frame2, seq2, cap_ts2 = stream.next_frame(
                    captured_seq + self.frame_skip - 1,
                    timeout=POC_BATCH_WAIT_MS / 1000.0,
                )
                if frame2 is not None:
                    captured_seq = seq2
                    if (
                        IDLE_GATE_HEARTBEAT_SECONDS > 0
                        and not active_tracks
                        and stream.is_blank
                        and (time.time() - last_infer_ts) < IDLE_GATE_HEARTBEAT_SECONDS
                    ):
                        self._metrics.record_gated()
                    else:
                        if WHITE_BALANCE_CAST_THRESHOLD > 0:
                            cast = colour_cast(frame2)
                            if cast > WHITE_BALANCE_CAST_THRESHOLD:
                                frame2 = correct_white_balance(frame2)
                                self._metrics.record_wb_corrected(cast)
                        poc_inputs.append((frame2, seq2, cap_ts2))

            t_loop = time.perf_counter()
            if serialised:
                _gpu_lock.acquire()
            self._metrics.add_stage("lock_wait", time.perf_counter() - t_loop)
            try:
                # The frame is already in hand (popped from the burst queue
                # above, one consume per capture — the duplicate-inference
                # tripwire in _InferenceMetrics still guards the invariant).
                # The GPU turn covers inference only.
                if frame is not None:
                    if not self._poc_batch:
                        frame_count += 1
                    if self._poc_batch:
                        t0 = time.perf_counter()
                        batcher = _poc_batcher(
                            self.inference_device, self.inference_half,
                        )
                        poc_frames = [item[0] for item in poc_inputs]
                        if len({item.shape for item in poc_frames}) == 1:
                            poc_outputs = batcher.submit_many(
                                poc_frames, imgsz, batch_tracker,
                            )
                        else:
                            poc_outputs = [
                                batcher.submit(item[0], imgsz, batch_tracker)
                                for item in poc_inputs
                            ]
                        elapsed = time.perf_counter() - t0
                        per_frame = elapsed / len(poc_outputs)
                        for item, _output in zip(poc_inputs, poc_outputs):
                            self._metrics.record(per_frame, frame_seq=item[1])
                    elif tile_model is not None:
                        # ANE tiled path: per-tile CoreML predict + cross-tile
                        # NMS, then the same ByteTrack config model.track()
                        # would use. conf is the LOW tracker threshold for the
                        # same reason as below; DEV raw webhook is skipped here
                        # (needs a Results object — disabled once at startup
                        # for this camera).
                        t0 = time.perf_counter()
                        if fusion:
                            dets = fused_predict(
                                model,
                                tile_model,
                                frame,
                                conf=TRACKER_CONFIDENCE_THRESHOLD,
                                imgsz=imgsz,
                                device=INFERENCE_DEVICE or None,
                                gate_conf=ANE_FUSION_GATE_CONF,
                                add_conf=ANE_FUSION_ADD_CONF,
                            )
                        else:
                            dets = tiled_predict(tile_model, frame, TRACKER_CONFIDENCE_THRESHOLD)
                        tracks = tiled_tracker.update(dets)
                        self._metrics.record(time.perf_counter() - t0, frame_seq=seq)
                    else:
                        # Tracking runs against THIS worker's own ByteTracker:
                        # model.track() keeps the tracker on the predictor, so a
                        # shared model means every camera feeding ONE tracker,
                        # and all but the first went permanently blind (a track
                        # needs two consecutive updates to activate, which
                        # interleaved cameras never gave it).
                        #
                        # conf stays at the LOW tracker threshold so ByteTrack's
                        # 2nd-stage association still sees weak boxes; the real
                        # emit gate is the track's PEAK confidence in
                        # _finalise_batch.
                        #
                        # Timed around the call alone so the reported cost is
                        # this camera's own inference, not the wait for its turn
                        # — queueing shows up in the sampled FPS, where
                        # throughput belongs.
                        t0 = time.perf_counter()
                        results = model.track(
                            frame,
                            imgsz=imgsz,
                            verbose=False,
                            conf=TRACKER_CONFIDENCE_THRESHOLD,
                            classes=[0],  # Human is a discriminating class, never a monitored target.
                            persist=True,
                            tracker=TRACKER_YAML,
                            device=self.inference_device or None,
                        )
                        self._metrics.record(time.perf_counter() - t0, frame_seq=seq)
            finally:
                if serialised:
                    _gpu_lock.release()
                self._metrics.add_stage("turn_total", time.perf_counter() - t_loop)
                last_infer_ts = time.time()

            if self._poc_batch:
                for (poc_frame, poc_seq, poc_ts), (poc_results, poc_tracks) in zip(
                    poc_inputs, poc_outputs,
                ):
                    frame_count = self._postprocess_poc_frame(
                        frame=poc_frame,
                        seq=poc_seq,
                        captured_ts=poc_ts,
                        results=poc_results,
                        tracks=poc_tracks,
                        frame_count=frame_count,
                        active_tracks=active_tracks,
                        frame_buf=frame_buf,
                    )
                continue

            # Everything below runs OUTSIDE the turn: per-camera CPU work that
            # must overlap the other cameras' GPU turns. The buffer write
            # therefore lands after this frame's inference, but under the same
            # key (frame_count, what _TrackState records) and the CAPTURE
            # timestamp (what _generate_combined_trail_video selects frames by),
            # so the finalisation window is unchanged. The only ordering
            # requirement is that it precede the lost-track finalisation below,
            # which is what reads the buffer.
            t_post = time.perf_counter()
            frame_buf.put(frame_count, frame, captured_ts)
            self._metrics.add_stage("buffer_put", time.perf_counter() - t_post)

            # Stage 2: once the small model says "something is here", the
            # big model gets the WHOLE frame — it finds animals 26n never
            # proposed, which a box-level verifier structurally cannot. The
            # offer is non-blocking and the queue is bounded, so this line can
            # never be the reason a camera stops being watched.
            if second_stage.enabled and results is not None:
                peak = 0.0
                for result in results:
                    b = result.boxes
                    if b is not None and len(b):
                        peak = max(peak, float(b.conf.max()))
                if peak >= SECOND_STAGE_GATE_CONF:
                    second_stage.offer(self.camera_id, frame_count, frame, self._imgsz)

            if tile_model is not None:
                # Track columns: [x1, y1, x2, y2, track_id, score, cls, idx]
                for row in tracks:
                    self._update_track(
                        active_tracks,
                        tid=int(row[4]),
                        conf=float(row[5]),
                        x1=float(row[0]),
                        y1=float(row[1]),
                        x2=float(row[2]),
                        y2=float(row[3]),
                        frame_count=frame_count,
                        frame=frame,
                        captured_ts=captured_ts,
                    )
            else:
                # DEV back door: fire a webhook on ANY raw detection, regardless
                # of tracking state. Used to diagnose why tracked detections fail
                # to emit. Cooldown-limited per camera.
                if DEV_RAW_WEBHOOK and self._on_raw_detection and self._loop:
                    self._maybe_fire_raw_webhook_from_results(results)

                for result in results:
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0 or boxes.id is None:
                        continue
                    for box, tid_tensor in zip(boxes, boxes.id):
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        self._update_track(
                            active_tracks,
                            tid=int(tid_tensor.item()),
                            conf=float(box.conf[0]),
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            frame_count=frame_count,
                            frame=frame,
                            captured_ts=captured_ts,
                        )


            # Collect lost tracks
            # Seconds, not processed frames: the frame rate changes with
            # resolution, model and camera count, and a frame-counted timeout
            # silently shortens as throughput rises — splitting one visit into
            # several detections.
            self._metrics.add_stage("bookkeeping", time.perf_counter() - t_post)
            now_ts = time.time()
            lost_ids = [
                tid for tid, ts in active_tracks.items()
                if (now_ts - ts.last_seen_ts) > TRACK_LOST_SECONDS
            ]

            if lost_ids:
                lost_tracks = [active_tracks.pop(tid) for tid in lost_ids]
                t_fin = time.perf_counter()
                self._finalise_batch(lost_tracks, active_tracks, frame_buf)
                self._metrics.add_stage("finalise", time.perf_counter() - t_fin)

        # Finalise remaining tracks on shutdown (best effort — a failure here
        # must not be reported as a crash, the loop is already leaving).
        if active_tracks:
            remaining = list(active_tracks.values())
            active_tracks.clear()
            try:
                self._finalise_batch(remaining, {}, frame_buf)
            except Exception:
                logger.exception(
                    "Shutdown finalise failed for camera %s", self.camera_id,
                )

    # ── dev raw-detection webhook (bypasses tracking) ────────────────────

    def _maybe_fire_raw_webhook_from_results(self, results) -> None:
        """Adapter for the model.track() path: pull raw boxes out of Results."""
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            dets = np.concatenate(
                [
                    boxes.xyxy.cpu().numpy().astype(np.float32),
                    boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1),
                    boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1),
                ],
                axis=1,
            )
            n_tracked = int(boxes.id.numel()) if boxes.id is not None else 0
            self._maybe_fire_raw_webhook(result.orig_img, dets, n_tracked)
            return

    def _maybe_fire_raw_webhook(
        self, frame: np.ndarray, dets: np.ndarray, n_tracked: int = 0,
    ) -> None:
        """Fire a debug webhook for any raw detection, with cooldown.

        Takes the raw detections array rather than a Results object (the
        pipeline no longer produces one), so boxes are drawn here. No DB
        write, no snapshot persisted — both JPEGs are encoded in memory, sent
        to Discord, and discarded. Cooldown prevents Discord rate-limiting.
        """
        now = time.time()
        if now - self._last_raw_notify < DEV_RAW_WEBHOOK_COOLDOWN:
            return
        if len(dets) == 0:
            return

        top_conf = float(dets[:, 4].max())
        # Noise floor: with tracker conf dropped to 0.1, YOLO fires on lots
        # of junk. Don't spam Discord (or reset cooldown) below this.
        if top_conf < DEV_RAW_WEBHOOK_MIN_CONF:
            return

        self._last_raw_notify = now

        try:
            annotated = frame.copy()
            thickness = max(2, max(frame.shape[:2]) // 400)
            for d in dets:
                cv2.rectangle(
                    annotated,
                    (int(d[0]), int(d[1])), (int(d[2]), int(d[3])),
                    (0, 255, 0), thickness, cv2.LINE_AA,
                )
            # Annotated for display: JPEG 85 is plenty. Raw for training
            # labels: JPEG 95 to preserve detail for dataset use.
            _, annot_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, raw_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            annot_bytes = annot_buf.tobytes()
            raw_bytes = raw_buf.tobytes()
        except Exception as exc:
            logger.warning("[DEV Webhook] failed to encode frame: %s", exc)
            return

        asyncio.run_coroutine_threadsafe(
            self._on_raw_detection(
                self.camera_id, raw_bytes, annot_bytes, top_conf, len(dets), n_tracked,
            ),
            self._loop,
        )

    # ── batch finalisation ───────────────────────────────────────────────

    def _finalise_batch(
        self,
        lost_tracks: list[_TrackState],
        still_active: dict[int, _TrackState],
        frame_buf: _FrameBuffer,
    ) -> None:
        """Finalise a batch of lost tracks.

        Tracks that overlapped temporally are grouped together so they
        share a single combined trail clip.
        """
        # Filter out very short / spurious tracks. A peak confidence alone is
        # not evidence of an animal: require repeated hits over capture time.
        valid = []
        for track in lost_tracks:
            if _track_is_substantial(track):
                valid.append(track)
            else:
                logger.info(
                    "Track #%d dropped as too short: hits=%d observed=%.3fs "
                    "(required hits=%d observed=%.3fs)",
                    track.track_id,
                    len(track.centers),
                    _track_observed_seconds(track),
                    TRACK_MIN_HITS,
                    TRACK_MIN_SECONDS,
                )
        if not valid:
            return

        # A track that was lost and immediately came back under a new id is not
        # a second animal. Hand it forward to the live track that continues it
        # BEFORE grouping, so it never becomes its own row or its own clip.
        # A live track can only be the continuation of ONE lost track. Two lost
        # tracks ending on the same spot are two animals converging on it, and
        # folding both into one would undercount AND interleave their histories.
        # Contained here rather than left to _run()'s supervisor: a merge is an
        # optimisation, and failing it should cost the merge, not a loop restart
        # that throws away tracker state and the frame buffer for every camera
        # this worker owns. Degrades to the old behaviour — both tracks emit.
        claimed: set[int] = set()
        try:
            valid = [t for t in valid if not self._adopt_continuation(t, still_active, claimed)]
        except Exception:
            logger.exception(
                "Continuation merge failed on camera %s — emitting tracks %s unmerged",
                self.camera_id, [t.track_id for t in valid],
            )
        if not valid:
            return

        # Group tracks by temporal overlap
        groups = self._group_overlapping(valid)

        # Within a group, two tracks that were box-on-box in the SAME frames are
        # one animal wearing two ids. Collapse them before anything is emitted,
        # so the duplicate never becomes its own row.
        #
        # Contained PER GROUP, not around the whole comprehension: the merge
        # mutates the survivor in place, so a comprehension that raises halfway
        # would leave earlier groups already absorbed while the unmerged list is
        # restored — emitting both the duplicate AND a survivor carrying its
        # frames. Per group, a failure costs that group its merge and nothing
        # else. (_merge_duplicate_tracks itself plans before it mutates, so a
        # failure in planning mutates nothing at all.)
        merged_groups: list[list[_TrackState]] = []
        for group in groups:
            try:
                merged_groups.append(self._merge_duplicate_tracks(group))
            except Exception:
                logger.exception(
                    "Duplicate-track merge failed on camera %s for tracks %s — "
                    "emitting that group unmerged",
                    self.camera_id, [t.track_id for t in group],
                )
                merged_groups.append(group)
        groups = merged_groups

        for group in groups:
            # Per group, so one bad group cannot drop the others AND cannot end
            # the worker. Everything downstream of here touches the network —
            # blob uploads, the DB write, the Discord webhook — and this thread
            # is the camera: on 2026-08-08 an upload timeout raised out of here,
            # out of _run(), and 場域甲 stopped being watched for 62 hours while
            # the dashboard still showed its last FPS. A lost detection is bad;
            # a lost camera is worse.
            try:
                self._finalise_group(group, still_active, frame_buf)
            except Exception:
                logger.exception(
                    "Finalising tracks %s on camera %s failed — detection lost, worker continues",
                    [t.track_id for t in group], self.camera_id,
                )

    def _finalise_group(
        self,
        group: list[_TrackState],
        still_active: dict[int, _TrackState],
        frame_buf: _FrameBuffer,
    ) -> None:
        """Gate, verify and emit one temporally-overlapping group of tracks."""
        # Gate + cascade-verify BEFORE any file generation so rejected
        # tracks leave no orphan clip/ZIP behind.
        # 1) Candidate gate on the realtime model's PEAK confidence.
        #    Tracking itself runs at the permissive
        #    TRACKER_CONFIDENCE_THRESHOLD so ByteTrack survives flicker.
        #    With a verifier enabled this gate can sit LOWER than the final
        #    threshold (CANDIDATE_CONFIDENCE_THRESHOLD) — the small model
        #    then only has to nominate, and the big model decides, which
        #    buys recall without paying in false positives.
        # 2) The verifier re-scores the best frame; its verdict REPLACES
        #    the realtime confidence and must clear CONFIDENCE_THRESHOLD.
        survivors: list[tuple[_TrackState, Optional[float]]] = []
        for track in group:
            if track.best_conf < CANDIDATE_CONFIDENCE_THRESHOLD:
                logger.debug(
                    "Track #%d dropped (best_conf=%.2f < %.2f)",
                    track.track_id, track.best_conf, CANDIDATE_CONFIDENCE_THRESHOLD,
                )
                continue
            verified = _verify_track(track, frame_buf)
            # No verifier: the realtime score itself must clear the final
            # gate, so a lowered candidate threshold cannot leak through.
            final_conf = verified if verified is not None else track.best_conf
            if final_conf < CONFIDENCE_THRESHOLD:
                logger.info(
                    "Track #%d rejected (%s %.2f < %.2f, realtime %.2f)",
                    track.track_id,
                    "verifier" if verified is not None else "realtime",
                    final_conf, CONFIDENCE_THRESHOLD, track.best_conf,
                )
                continue
            survivors.append((track, verified))

        if not survivors:
            return

        # Keep overlapping still-active tracks as visit context for peak/count
        # decisions, but do not put them in the persisted clip. They have not
        # crossed the emit gate and therefore must not appear as a coloured
        # track without a corresponding API row.
        group_min = min(t.min_frame_idx for t, _ in survivors)
        group_max = max(t.max_frame_idx for t, _ in survivors)
        concurrent = [
            t for t in still_active.values()
            if t.min_frame_idx <= group_max and t.max_frame_idx >= group_min
            and _track_is_substantial(t)
        ]

        # Everything above decided WHAT to emit and needed the GPU turn this
        # thread already holds. Everything below is blob uploads and a DB write,
        # which is the work that killed this thread twice — so it is handed to
        # the visit queue, which also gets to merge the next group of the same
        # visit into it. A refusal renders inline, exactly as before.
        held = _visit_queue.offer(
            self.camera_id, survivors, concurrent,
            self._snapshot_frames(survivors, [], frame_buf),
            self._render_visit,
        )
        if held:
            return
        self._render_visit(_PendingVisit(
            self.camera_id, survivors, concurrent, frame_buf, time.monotonic()))

    def _snapshot_frames(
        self,
        survivors: list[tuple[_TrackState, Optional[float]]],
        concurrent: list[_TrackState],
        frame_buf: _FrameBuffer,
    ) -> _FrameBuffer:
        """The frames this visit will need, detached from the shared ring.

        Selected by the SAME time window the clip generator will ask for, so a
        held visit renders from exactly the frames an inline one would have —
        the hold changes when it is drawn, never what is in it.
        """
        tracks = [t for t, _ in survivors] + concurrent
        det_min = min(t.min_frame_idx for t in tracks)
        det_max = max(t.max_frame_idx for t in tracks)
        t_first, t_last = frame_buf.get_ts(det_min), frame_buf.get_ts(det_max)
        if t_first is None or t_last is None:
            indices = frame_buf.indices_in_range(det_min, det_max)
        else:
            indices = frame_buf.indices_in_time_range(
                t_first - TRAIL_PRE_BUFFER_SECONDS,
                t_last + TRAIL_POST_BUFFER_SECONDS,
            )
        return frame_buf.snapshot(indices)

    def _render_visit(self, visit: "_PendingVisit") -> None:
        """Draw one clip for the whole visit and emit one row per track.

        Every row emitted here carries the SAME `visit_id`. That id is the whole
        point of holding the group: the visit is now something the producer
        recorded, not something a reader regroups by a 20 s gap afterwards.
        Sortable by time so a glance at the id says when, and short enough to
        read in a log line.
        """
        # Only emitted survivors may be drawn. `concurrent` is observational
        # context; drawing it creates boxes that the API cannot explain.
        all_for_clip = [t for t, _ in visit.survivors]
        started = min((t.first_seen for t, _ in visit.survivors),
                      default=datetime.now(_TW_TZ))
        visit_id = f"{started:%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
        visit_peak = _peak_concurrent([t for t, _ in visit.survivors])
        # The clip's time axis comes back from the generator, because only the
        # generator knows which frames survived encoding and at what rate it
        # wrote them. Deriving it from the buffer's wall-clock span instead is
        # what put offsets past the end of their own video.
        frame_ts = visit.frames.timestamps()
        clip_name, clip_axis, clip_fps = _generate_combined_trail_video(
            all_for_clip, visit.frames, self.camera_id)
        if not clip_name or not clip_axis or clip_fps <= 0:
            # A row without the clip it describes is not a complete detection
            # anymore. This also closes the old path where a one-frame/failed
            # encode produced snapshots and an empty trail_path in the DB.
            logger.warning(
                "Camera %s: skipping %d detection row(s); no valid trail clip",
                self.camera_id,
                len(visit.survivors),
            )
            return
        # all_for_clip is passed through so each still outlines exactly the
        # tracks that the clip draws; every visible box has a matching API row.
        for track, verified in visit.survivors:
            offsets = _clip_offsets(track, clip_axis, clip_fps, frame_ts)
            self._emit_detection(
                track, visit.frames, clip_name,
                verified_conf=verified, group_tracks=all_for_clip,
                visit_id=visit_id, visit_peak=visit_peak, **offsets,
            )

    def _adopt_continuation(
        self,
        track: _TrackState,
        still_active: dict[int, _TrackState],
        claimed: set[int],
    ) -> bool:
        """Stitch a lost track into the live track that continues it.

        Returns True when `track` was absorbed and must not be emitted.

        The case (measured 2026-08-19): a macaque sits still, the detector
        misses it for several seconds, ByteTrack retires the id and mints a new
        one when it comes back. Two rows, two clips, one animal. Four such
        pairs from the 2026-08-13 harvest were inspected frame by frame — same
        individual, same perch, 7-18 s apart. Raising track_buffer (see
        TRACKER_YAML) removes most of them; this catches the remainder, because
        a dropout longer than the buffer will always be possible.

        Deliberately narrow, because this is the re-identification the visit-continuation design
        called unsafe, and it is only safe under these restrictions:

        * **Forward only.** The candidate must START after this track ENDED.
          Tracks that overlap are concurrent animals — _group_overlapping's
          job, and merging them would delete a real animal from the count.
        * **Continuity, not identity.** Either the candidate's first box
          overlaps this track's last box — "the animal never left that spot" —
          or, within a much shorter window, it sits where this track's own
          velocity says it was heading. Never "this is the same animal that
          walked off earlier".
        * **Still active.** Only tracks alive right now are candidates, so
          nothing already written to the DB is ever rewritten.

        The residual risk is two animals using one perch in sequence, which
        this WILL merge. That is a real undercount, and it is the price of not
        filing one monkey as two — chosen deliberately, and bounded by
        TRACK_CONTINUATION_SECONDS.
        """
        if TRACK_CONTINUATION_SECONDS <= 0 or not track.frame_bboxes:
            return False

        last_box = track.frame_bboxes[-1]
        best: Optional[_TrackState] = None
        best_iou = TRACK_CONTINUATION_IOU
        best_miss = TRACK_MOTION_ERR_WIDTHS
        why = ""
        for cand in still_active.values():
            if cand.track_id in claimed:
                continue
            if cand.min_frame_idx <= track.max_frame_idx or not cand.frame_bboxes:
                continue
            gap = cand.first_seen_ts - track.last_seen_ts
            if gap < 0 or gap > TRACK_CONTINUATION_SECONDS:
                continue
            iou = _bbox_iou(last_box, cand.frame_bboxes[0])
            if iou >= best_iou:
                best, best_iou, why = cand, iou, f"IoU {iou:.2f}"
                continue
            # It was not sitting still. A macaque crossing the frame leaves no
            # overlap at all between fragments, which is why the tracker handed
            # out eight ids for one crossing on 2026-08-28 07:06:28 and none of
            # them was stitched. Ask instead whether it carried on at the speed
            # it was going.
            miss = _continues_by_motion(track, cand, gap)
            if miss < best_miss:
                best, best_miss, why = cand, miss, f"外推誤差 {miss:.2f} 框寬"

        if best is None:
            return False

        claimed.add(best.track_id)
        gap = best.first_seen_ts - track.last_seen_ts  # before it is overwritten below

        # Prepend this track's history so the survivor carries the whole visit:
        # the clip draws the full trail and the row's timestamp is when the
        # animal actually arrived, not when it was re-acquired.
        best.first_seen = track.first_seen
        best.first_seen_ts = track.first_seen_ts
        best.min_frame_idx = track.min_frame_idx
        best.centers[:0] = track.centers
        best.frame_indices[:0] = track.frame_indices
        best.frame_bboxes[:0] = track.frame_bboxes
        # Keep the better evidence — the gate and the verifier both read the
        # PEAK, so a strong first segment must not be thrown away by a weak
        # second one (or the merge would cost us a detection).
        if track.best_conf > best.best_conf:
            best.best_conf = track.best_conf
            best.best_bbox = track.best_bbox
            best.best_frame_idx = track.best_frame_idx
            best.best_frame = track.best_frame

        logger.info(
            "Camera %s: track #%d continues as #%d (gap %.1fs, %s) — "
            "one visit, one row",
            self.camera_id, track.track_id, best.track_id, gap, why,
        )
        return True

    @staticmethod
    def _shared_frame_iou(a: _TrackState, b: _TrackState) -> tuple[float, int, float]:
        """Mean IoU over the frames BOTH tracks appear in, how many those are,
        and what fraction of the SHORTER track they represent.

        Frame indices, not timestamps: two boxes in the same frame are two
        claims about the same instant, which is the only comparison that can
        show one body being counted twice. The ratio is what separates that
        from an occlusion — see TRACK_DUPLICATE_MIN_OVERLAP.
        """
        boxes_a = dict(zip(a.frame_indices, a.frame_bboxes))
        boxes_b = dict(zip(b.frame_indices, b.frame_bboxes))
        shared = boxes_a.keys() & boxes_b.keys()
        if not shared:
            return 0.0, 0, 0.0
        total = sum(_bbox_iou(boxes_a[i], boxes_b[i]) for i in shared)
        shortest = min(len(boxes_a), len(boxes_b)) or 1
        return total / len(shared), len(shared), len(shared) / shortest

    @staticmethod
    def _absorb_duplicate(keep: _TrackState, other: _TrackState) -> None:
        """Fold `other` into `keep` — one animal, one history, one row."""
        keep.first_seen = min(keep.first_seen, other.first_seen)
        keep.first_seen_ts = min(keep.first_seen_ts, other.first_seen_ts)
        keep.last_seen_ts = max(keep.last_seen_ts, other.last_seen_ts)
        keep.min_frame_idx = min(keep.min_frame_idx, other.min_frame_idx)
        keep.max_frame_idx = max(keep.max_frame_idx, other.max_frame_idx)

        # Rebuild the three parallel lists in frame order. Where both tracks
        # have a box for the same frame the survivor's own box wins — they are
        # the same animal, so taking one is right and interleaving both would
        # draw the trail as a zigzag between two nearly identical boxes.
        merged: dict[int, tuple[tuple[int, int], tuple[float, float, float, float]]] = {}
        for idx, center, box in zip(other.frame_indices, other.centers, other.frame_bboxes):
            merged[idx] = (center, box)
        for idx, center, box in zip(keep.frame_indices, keep.centers, keep.frame_bboxes):
            merged[idx] = (center, box)
        keep.frame_indices = sorted(merged)
        keep.centers = [merged[i][0] for i in keep.frame_indices]
        keep.frame_bboxes = [merged[i][1] for i in keep.frame_indices]

        # No peak swap here, deliberately. The gate and the verifier both read
        # the PEAK, so the better evidence must survive a merge — and it does by
        # construction: _merge_duplicate_tracks only ever absorbs into a track
        # that already sorted at or above `other` on best_conf. A conditional
        # swap here would be unreachable code dressed as a safety property, so
        # the invariant is asserted at the one place that can break it instead.
        assert keep.best_conf >= other.best_conf, (
            "survivor must hold the peak: #%d (%.3f) absorbed #%d (%.3f)"
            % (keep.track_id, keep.best_conf, other.track_id, other.best_conf)
        )

    def _merge_duplicate_tracks(self, group: list[_TrackState]) -> list[_TrackState]:
        """Collapse tracks in one group that are the same animal seen twice.

        The safe half of the visit-continuation design's "duplicate emissions": simultaneous,
        high-IoU tracks. Unlike cross-time re-identification — which that issue
        calls unsafe and which nothing here attempts — this needs no judgement
        about identity over a gap. The two tracks are in the SAME frames with
        their boxes on top of each other, so they cannot be two bodies.

        Survivor is the higher-confidence track, matching how a group already
        picks its cover elsewhere. Order-independent: candidates are considered
        strongest-first, so which track absorbs which does not depend on the
        order ByteTrack happened to hand them over in.
        """
        if len(group) < 2 or TRACK_DUPLICATE_IOU <= 0:
            return group

        # Strongest first, track_id breaking a tie: plain reverse sorting is
        # stable, so two tracks that peak at the same confidence — common when
        # both ids lock onto one animal in the same frame — would otherwise let
        # ByteTrack's hand-over order decide which id reaches the database.
        ordered = sorted(group, key=lambda t: (-t.best_conf, t.track_id))

        # PLAN first, mutate second. _absorb_duplicate rewrites the survivor in
        # place, so deciding everything before touching anything is what lets
        # the caller treat a failure as "this group was not merged" rather than
        # as a half-merged group it cannot tell apart from a merged one.
        plan: list[tuple[_TrackState, _TrackState, float, int, float]] = []
        survivors: list[_TrackState] = []
        for track in ordered:
            twin = None
            for kept in survivors:
                iou, shared, ratio = self._shared_frame_iou(kept, track)
                if (
                    shared >= TRACK_DUPLICATE_MIN_FRAMES
                    and ratio >= TRACK_DUPLICATE_MIN_OVERLAP
                    and iou >= TRACK_DUPLICATE_IOU
                ):
                    twin = (kept, iou, shared, ratio)
                    break
            if twin is None:
                survivors.append(track)
                continue
            kept, iou, shared, ratio = twin
            plan.append((kept, track, iou, shared, ratio))

        for kept, track, iou, shared, ratio in plan:
            self._absorb_duplicate(kept, track)
            logger.info(
                "Camera %s: track #%d is track #%d seen twice "
                "(IoU %.2f over %d shared frames, %.0f%% of the shorter track) "
                "— one animal, one row",
                self.camera_id, track.track_id, kept.track_id, iou, shared,
                ratio * 100,
            )
        # Back into first-seen order so the clip and the rows read chronologically
        survivors.sort(key=lambda t: t.min_frame_idx)
        return survivors

    @staticmethod
    def _group_overlapping(tracks: list[_TrackState]) -> list[list[_TrackState]]:
        """Group tracks whose time ranges overlap into clusters."""
        if len(tracks) <= 1:
            return [tracks]

        # Sort by start frame
        sorted_tracks = sorted(tracks, key=lambda t: t.min_frame_idx)
        groups: list[list[_TrackState]] = []
        current_group = [sorted_tracks[0]]
        current_max = sorted_tracks[0].max_frame_idx

        for t in sorted_tracks[1:]:
            if t.min_frame_idx <= current_max:
                # Overlaps with current group
                current_group.append(t)
                current_max = max(current_max, t.max_frame_idx)
            else:
                groups.append(current_group)
                current_group = [t]
                current_max = t.max_frame_idx

        groups.append(current_group)
        return groups

    def _emit_detection(
        self,
        track: _TrackState,
        frame_buf: _FrameBuffer,
        trail_clip_name: str,
        visit_id: Optional[str] = None,
        visit_peak: Optional[int] = None,
        clip_offset_seconds: Optional[float] = None,
        clip_end_offset_seconds: Optional[float] = None,
        clip_duration_seconds: Optional[float] = None,
        clip_segments: Optional[list] = None,
        verified_conf: Optional[float] = None,
        group_tracks: Optional[list[_TrackState]] = None,
    ) -> None:
        """Save per-track snapshots and emit the detection event.

        Gating (confidence threshold + cascade verification) happens in
        _finalise_batch before any files are generated; every call here is
        a confirmed emit. verified_conf, when present, is the cascade
        verifier's verdict and REPLACES the realtime model's confidence in
        everything downstream (DB, Discord, the >=0.9 visibility gate).

        group_tracks is every track sharing this detection's moment. One row
        still means ONE animal — that is what the bbox and confidence describe
        — but the annotated still now also outlines the others, dimmed, so a
        troop of six does not render as six pictures each showing one monkey
        and looking like five were missed.
        """
        # Prefer the raw best frame kept on the track (pristine quality for
        # the snapshot outputs); fall back to the JPEG buffer.
        best_frame = track.best_frame
        from_buffer = best_frame is None
        if best_frame is None:
            best_frame = frame_buf.get(track.best_frame_idx)
        if best_frame is None:
            # Fall back to the most recent of the track's frames still
            # buffered — probe newest-first so we decode at most one JPEG.
            for fi in reversed(track.frame_indices):
                best_frame = frame_buf.get(fi)
                if best_frame is not None:
                    break
            if best_frame is None:
                return

        frame = best_frame
        x1, y1, bw, bh = track.best_bbox
        conf = verified_conf if verified_conf is not None else track.best_conf
        h_orig, w_orig = frame.shape[:2]

        ts_str = datetime.now(_TW_TZ).strftime('%Y%m%d_%H%M%S')
        uid = uuid.uuid4().hex[:8]
        base_name = f"{track.camera_id}_t{track.track_id}_{ts_str}_{uid}"

        # Scale drawing primitives to the actual frame resolution
        ref_dim = max(w_orig, h_orig)
        box_thickness = max(3, ref_dim // 250)
        trail_thickness = max(3, ref_dim // 300)
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

        # 1) Raw snapshot — encode the pristine frame at q95 when we have it;
        # otherwise reuse the buffered JPEG rather than re-encoding a decode.
        from services.storage import storage
        snapshot_name = f"{base_name}.jpg"
        raw_jpg = frame_buf.get_jpeg(track.best_frame_idx) if from_buffer else None
        if raw_jpg is None:
            _, raw_buf = cv2.imencode(".jpg", frame, jpeg_params)
            raw_jpg = raw_buf.tobytes()
        storage.upload(snapshot_name, raw_jpg)

        # 2) Annotated snapshot — bbox + full trail
        annotated = frame.copy()
        ix1, iy1 = int(x1), int(y1)
        ix2, iy2 = int(x1 + bw), int(y1 + bh)
        color = _TRACK_COLORS[track.track_id % len(_TRACK_COLORS)]

        # Draw trail
        for j in range(1, len(track.centers)):
            ratio = j / max(len(track.centers) - 1, 1)
            dim = 0.35 + 0.65 * ratio
            c = tuple(int(ch * dim) for ch in color)
            thick = max(trail_thickness, int(trail_thickness * 1.5 * ratio))
            cv2.line(annotated, track.centers[j - 1], track.centers[j], c, thick, cv2.LINE_AA)

        # Companion monkeys present in this same frame, drawn first so this
        # detection's own box stays on top. Dimmed and thinner: they are
        # context, not the subject of this row. bbox_near (not bbox_at) so a
        # track that had already left is not drawn at a stale position.
        for other in (group_tracks or ()):
            if other is track:
                continue
            ob = other.bbox_near(track.best_frame_idx, _COMPANION_MAX_FRAME_GAP)
            if ob is None:
                continue
            ox1, oy1, ow, oh = ob
            oc = _TRACK_COLORS[other.track_id % len(_TRACK_COLORS)]
            cv2.rectangle(
                annotated,
                (int(ox1), int(oy1)), (int(ox1 + ow), int(oy1 + oh)),
                tuple(int(ch * 0.55) for ch in oc),
                max(1, box_thickness // 2), cv2.LINE_AA,
            )

        # Draw bounding box — no text label: non-admin users see this image,
        # so track ID / confidence must not be burned into the pixels.
        cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), color, box_thickness, cv2.LINE_AA)

        annotated_name = f"{base_name}_annot.jpg"
        _, annot_buf = cv2.imencode(".jpg", annotated, jpeg_params)
        storage.upload(annotated_name, annot_buf.tobytes())

        # 3) Crop — generous padding, full quality
        pad = max(20, ref_dim // 30)
        cy1 = max(0, iy1 - pad)
        cy2 = min(h_orig, iy2 + pad)
        cx1 = max(0, ix1 - pad)
        cx2 = min(w_orig, ix2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        crop_name = f"{base_name}_crop.jpg"
        if crop.size > 0:
            _, crop_buf = cv2.imencode(".jpg", crop, jpeg_params)
            storage.upload(crop_name, crop_buf.tobytes())
        else:
            crop_name = ""

        detection = DetectionResult(
            camera_id=track.camera_id,
            confidence=conf,
            bbox=(x1, y1, bw, bh),
            image_path=snapshot_name,
            annotated_path=annotated_name,
            crop_path=crop_name,
            track_id=track.track_id,
            trail_path=trail_clip_name,
            timestamp=track.first_seen,
            visit_id=visit_id,
            visit_peak=visit_peak,
            clip_offset_seconds=clip_offset_seconds,
            clip_end_offset_seconds=clip_end_offset_seconds,
            clip_duration_seconds=clip_duration_seconds,
            clip_segments=clip_segments,
        )

        logger.info(
            "Track #%d finalised: camera=%s conf=%.2f trail_points=%d clip=%s",
            track.track_id, track.camera_id, conf, len(track.centers), trail_clip_name,
        )

        if self._on_detection and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._on_detection(detection), self._loop
            )


# ── Manager ──────────────────────────────────────────────────────────────────


class DetectorManager:
    """Registry of detection workers (one per camera)."""

    def __init__(self) -> None:
        self._workers: dict[str, DetectorWorker] = {}

    def start(
        self,
        camera_id: str,
        loop: asyncio.AbstractEventLoop,
        on_detection: Optional[callable] = None,
        on_raw_detection: Optional[callable] = None,
    ) -> DetectorWorker:
        if camera_id in self._workers:
            return self._workers[camera_id]
        worker = DetectorWorker(
            camera_id,
            on_detection=on_detection,
            on_raw_detection=on_raw_detection,
        )
        worker.start(loop)
        self._workers[camera_id] = worker
        return worker

    def stop(self, camera_id: str) -> None:
        worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()

    def stop_all(self) -> None:
        for w in self._workers.values():
            w.stop()
        self._workers.clear()

    def revive_dead(self) -> list[str]:
        """Restart any worker whose thread died. Returns the cameras revived.

        Nothing supervised these threads before: `start()` set a flag and never
        looked again, so a dead worker stayed dead until someone restarted the
        whole process — 62 hours, the one time it mattered.
        """
        return [cid for cid, w in sorted(self._workers.items()) if w.revive()]

    def inference_stats(self) -> dict:
        from services.stream import stream_manager

        cameras: list[dict] = []
        for camera_id, worker in sorted(self._workers.items()):
            stream = stream_manager.get(camera_id)
            connected = bool(stream and stream.connected)
            metrics = worker.get_metrics(stream_connected=connected)
            capture_fps = stream.capture_fps if stream else 0.0
            cameras.append({
                "camera_id": camera_id,
                # Live gate state, not a rolling average: whether THIS camera is
                # being sampled at the idle heartbeat right now.
                "idle_gated": bool(stream and stream.is_blank),
                "frame_stddev": round(stream.frame_stddev, 2) if stream else 0.0,
                "capture_fps": capture_fps,
                "capture_gaps_ms": stream.capture_gaps_ms if stream else {},
                "stream_reconnects": stream.reconnects if stream else 0,
                # Share of the frames the camera DELIVERED that were actually
                # inferred. The two rates were both already reported and nobody
                # divided them: four cameras at 11.4 FPS each looks healthy until
                # you notice they are delivering 33 and the GPU is only covering
                # a third of what arrives. Capped at 1.0 — the two rolling
                # windows are measured independently and can cross transiently.
                "coverage_ratio": (
                    round(min(1.0, metrics["sample_fps"] / capture_fps), 3)
                    if capture_fps > 0 else 0.0
                ),
                # Is the detector THREAD alive, and how often has it had to be
                # brought back? A dead worker used to read as a slow one.
                "worker_alive": worker.alive,
                "worker_crashes": worker.crashes,
                "worker_revivals": worker.revivals,
                "frame_skip": worker.frame_skip,
                "coverage_ceiling": round(1.0 / worker.frame_skip, 3),
                **metrics,
            })

        return {
            # The FLOOR. Each camera also reports the size it actually uses.
            "inference_imgsz": INFERENCE_IMGSZ,
            "frame_skip": FRAME_SKIP,
            "model_path": Path(MODEL_PATH).name,
            # Ceiling the coverage above can reach: the loop deliberately
            # consumes one frame in FRAME_SKIP, so at 2 the best possible
            # coverage is 50% and painting that as a shortfall would be wrong.
            "coverage_ceiling": round(1.0 / FRAME_SKIP, 3),
            "cameras": cameras,
        }


# Module-level singleton
detector_manager = DetectorManager()
