"""Application configuration."""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
# Where LocalStorage keeps snapshots when no Azure connection string is set.
# Overridable so the mini can point at the 17.4 GB mirror pulled out of Blob
# storage rather than re-accumulating a second copy inside the repo.
SNAPSHOTS_DIR = Path(os.getenv("SNAPSHOTS_DIR") or (BASE_DIR / "snapshots"))
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Azure Blob Storage (empty = use local filesystem)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "snapshots")
# Blob upload durability. The edge node pushes tens of MB per detection (trail
# GIF + raw-frames ZIP) over a site uplink to eastasia. The SDK default sends
# anything under 64 MB as ONE PUT, so a slow uplink hits the socket write
# timeout and loses the whole upload — twice that exception propagated out of a
# detector thread and killed a camera (2026-08-06, 2026-08-08). Chunked, each
# block is a separate request with its own retry, so a stall costs one block.
BLOB_MAX_SINGLE_PUT_BYTES = int(float(os.getenv("BLOB_MAX_SINGLE_PUT_MB", "4")) * 1024 * 1024)
BLOB_BLOCK_BYTES = int(float(os.getenv("BLOB_BLOCK_MB", "4")) * 1024 * 1024)
# The timeouts and retry counts are all deliberately BELOW the SDK defaults
# (connect 20 s, read 60 s, retry_total 10). The detector thread that finalises a
# track is blocked while it uploads, so every second of retrying is a second the
# camera is not being inferred — the defaults could block it for many minutes,
# which is how the fatal upload also became a long stall before it raised. A
# 4 MB block needs ~16 s on a 2 Mbps uplink, so 30 s fails a genuinely stalled
# socket fast while never cutting off a working one.
BLOB_CONNECT_TIMEOUT = float(os.getenv("BLOB_CONNECT_TIMEOUT", "15"))
BLOB_READ_TIMEOUT = float(os.getenv("BLOB_READ_TIMEOUT", "30"))
BLOB_RETRY_TOTAL = int(os.getenv("BLOB_RETRY_TOTAL", "2"))  # SDK-level, per request
BLOB_UPLOAD_ATTEMPTS = int(os.getenv("BLOB_UPLOAD_ATTEMPTS", "2"))  # whole-upload retries

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{BASE_DIR / 'detections.db'}")

# YOLO Model
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "best.pt"))
# Emit / webhook threshold — a track's PEAK confidence must reach this to
# fire a Discord notification or be persisted to DB.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
# Tracker threshold — fed to model.track() so ByteTrack sees low-score
# boxes and can run its 2nd-stage association. Must be well below
# CONFIDENCE_THRESHOLD or unconfirmed tracks die whenever confidence dips,
# which was the root cause of "0/1 no ID" drops.
TRACKER_CONFIDENCE_THRESHOLD = float(os.getenv("TRACKER_CONFIDENCE_THRESHOLD", "0.1"))
# A confidence score is not enough to call an animal. Require repeated tracker
# hits over a real capture-time span so a single-frame flash cannot become a
# persisted detection or a one-frame trail clip. Both the live detector and
# the historical rebuild use this same policy.
TRACK_MIN_HITS = int(os.getenv("TRACK_MIN_HITS", "3"))
TRACK_MIN_SECONDS = float(os.getenv("TRACK_MIN_SECONDS", "0.5"))
# Candidate gate — the realtime model's PEAK confidence needed to NOMINATE a
# track for adjudication. Defaults to CONFIDENCE_THRESHOLD (no change in
# behaviour). With a cascade verifier enabled this can be lowered so the small
# model nominates generously and the big model decides: recall goes up without
# paying in false positives, because the verifier's verdict is what has to
# clear CONFIDENCE_THRESHOLD. Ignored (treated as the final gate) when no
# verifier is configured.
CANDIDATE_CONFIDENCE_THRESHOLD = float(
    os.getenv("CANDIDATE_CONFIDENCE_THRESHOLD", str(CONFIDENCE_THRESHOLD))
)
# Non-admin visibility gate — records timestamped on/after USER_CONFIDENCE_SINCE
# (00:00 Taiwan time) are shown to non-admin users only when confidence >=
# USER_CONFIDENCE_THRESHOLD. Older records are human-verified and stay fully
# visible. Admins always see everything.
#
# Kept at 0.9 deliberately, and it is a gate on FUTURE records only. Dropping it
# to 0 was tried on 2026-08-14 and rolled back: it does make every record
# visible, but it also removes the filter from everything that arrives later,
# and the rare genuine false positive (場域乙 that day produced one at conf 0.55
# on foliage) then reaches users unreviewed.
#
# The right shape is the one now in use: the gate stays armed, and a BACKLOG of
# real-but-low-confidence sightings is pushed in bulk via
# POST /api/admin/detections/visibility-bulk. Measured on 611 hand-labelled
# boxes, precision is 0.93-0.99 across 24 model/resolution combinations — the
# errors that matter here are misses, not invented animals — so most of what
# this gate catches is worth pushing, but that stays a decision someone makes,
# not a default.
USER_CONFIDENCE_THRESHOLD = float(os.getenv("USER_CONFIDENCE_THRESHOLD", "0.9"))
USER_CONFIDENCE_SINCE = os.getenv("USER_CONFIDENCE_SINCE", "2026-07-27")  # YYYY-MM-DD, TW time

# POC-only camera sources. These are configuration-owned so a non-admin cannot
# gain access merely by having allowed_cameras=["*"]. The Edge node opts them
# in only when POC_EDGE_ENABLED is explicitly set.
POC_CAMERA_SOURCES = {
    "poc-camera-1": {
        "name": "POC｜測試串流一",
        "url": os.getenv("POC_CAMERA_1_URL", "").strip(),
    },
    "poc-camera-2": {
        "name": "POC｜測試串流二",
        "url": os.getenv("POC_CAMERA_2_URL", "").strip(),
    },
}
POC_CAMERA_IDS = frozenset(POC_CAMERA_SOURCES)
# POC streams are registered for admins independently of this switch. Enabling
# it on the Edge node opts only these sources into local capture/inference; the
# cloud API never starts their streams. Keep the default off for fresh nodes so
# adding a POC row cannot spend GPU by accident.
POC_EDGE_ENABLED = os.getenv("POC_EDGE_ENABLED", "0").lower() in ("1", "true", "yes")
POC_FRAME_SKIP = max(1, int(os.getenv("POC_FRAME_SKIP", "5")))
# YouTube HLS delivers several seconds of frames in a burst. A larger queue is
# only for POC feeds, so the burst can be drained by the detector instead of
# evicting its history from the 16-frame queue used by RTSP cameras.
POC_FRAME_QUEUE_MAX = max(16, int(os.getenv("POC_FRAME_QUEUE_MAX", "128")))
# POC-only batching. Requests are combined only when their source frames have
# the same shape, because that is the condition under which Ultralytics keeps
# the production rectangular letterbox. A mismatched pair falls back to the
# same single-frame model path, so batching can never silently change scale.
POC_BATCH_ENABLED = os.getenv("POC_BATCH_ENABLED", "1").lower() in ("1", "true", "yes")
POC_BATCH_MAX = max(2, int(os.getenv("POC_BATCH_MAX", "2")))
POC_BATCH_WAIT_MS = max(0.0, float(os.getenv("POC_BATCH_WAIT_MS", "8")))


def frame_skip_for_camera(camera_id: str) -> int:
    """Return the sampling budget for one camera.

    Existing cameras keep the deployment-wide setting. POC cameras use a
    deliberately small share of Edge turns so they can be enabled on the mini
    without taking the current cameras' GPU slots by default.
    """
    if POC_EDGE_ENABLED and camera_id in POC_CAMERA_IDS:
        return POC_FRAME_SKIP
    return FRAME_SKIP


def poc_batch_for_camera(camera_id: str) -> bool:
    """Whether this camera may use the low-priority POC batcher."""
    return POC_EDGE_ENABLED and POC_BATCH_ENABLED and camera_id in POC_CAMERA_IDS

# Torch device for YOLO ("" = ultralytics default/CPU; "mps" on Apple Silicon)
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "")
# Half precision. Ultralytics' predict(half=True) is a no-op on MPS, so the
# backend module is converted directly instead — measured 28.1 -> 21.9 ms
# (+28%) at imgsz 1600 with detections unchanged (confidence delta 0.0001,
# box delta 0.4 px). Default on for MPS only; fp16 buys nothing on CPU.
INFERENCE_HALF = os.getenv("INFERENCE_HALF", "1" if INFERENCE_DEVICE == "mps" else "0").lower() in ("1", "true", "yes")
# POC stays on CPU by default so the four existing cameras retain their MPS
# budget. Set it to mps only after measuring the core-camera FPS again.
POC_INFERENCE_DEVICE = os.getenv("POC_INFERENCE_DEVICE", "cpu").strip().lower()


def inference_device_for_camera(camera_id: str) -> str:
    """Return the inference device for one camera, with a POC-only override."""
    if POC_EDGE_ENABLED and camera_id in POC_CAMERA_IDS and POC_INFERENCE_DEVICE:
        return POC_INFERENCE_DEVICE
    return INFERENCE_DEVICE


def inference_half_for_camera(camera_id: str) -> bool:
    """Resolve fp16 after applying the optional per-camera device override."""
    device = inference_device_for_camera(camera_id)
    if POC_EDGE_ENABLED and camera_id in POC_CAMERA_IDS and POC_INFERENCE_DEVICE:
        return device == "mps"
    return INFERENCE_HALF


# Take turns on the GPU. Each camera worker owns its own model (required for
# correct per-camera tracking), but four concurrent MPS command streams
# measured slower in aggregate than serialised ones — 21.9 FPS concurrent vs
# 25.3 FPS when they queue. Set to 0 to let workers run the GPU concurrently.
INFERENCE_SERIALISE = os.getenv("INFERENCE_SERIALISE", "1").lower() in ("1", "true", "yes")
# ANE tiled inference (Apple Neural Engine via CoreML). Comma-separated camera
# IDs that run tiled 640x640 CoreML inference instead of the model.track()
# path; empty (default) = feature off, behavior unchanged. Cameras listed here
# fall back to the standard path if the CoreML model fails to load.
ANE_TILE_CAMERAS = {c.strip() for c in os.getenv("ANE_TILE_CAMERAS", "").split(",") if c.strip()}
ANE_TILE_MODEL_PATH = os.getenv("ANE_TILE_MODEL_PATH", str(BASE_DIR / "best_coreml_640.mlpackage"))
# ANE fusion (recommended over pure tiling): full-frame GPU inference every
# frame PLUS a gated ANE tile sweep that only fires when the frame already
# looks interesting. Measured on the test split: F1 0.9268 -> 0.9327,
# recall 0.892 -> 0.911 (4 fewer missed monkeys per 213) at unchanged FPS,
# because empty scenes — ~99% of frames — never trigger the ANE pass.
# Pure tiling (ANE_TILE_CAMERAS) trades accuracy for throughput and FAILS the
# ratchet (F1 0.9129); fusion is the path that keeps both.
ANE_FUSION_CAMERAS = {c.strip() for c in os.getenv("ANE_FUSION_CAMERAS", "").split(",") if c.strip()}
# Full-frame confidence that arms the ANE sweep (below this the scene is empty).
ANE_FUSION_GATE_CONF = float(os.getenv("ANE_FUSION_GATE_CONF", "0.3"))
# Tile detections must reach this confidence to be ADDED as new objects.
ANE_FUSION_ADD_CONF = float(os.getenv("ANE_FUSION_ADD_CONF", "0.6"))
# Cascade verifier: a larger model (e.g. monkey-trained YOLO26x) that re-scores
# each track's best frame before emit. Empty = disabled (current behavior).
# Event-driven — runs once per finalised track, not per frame, so the realtime
# 26n pipeline keeps its FPS while the big model owns the final say on
# confidence (and therefore the DB record, Discord alert, and the >=0.9
# user-visibility gate).
VERIFIER_MODEL_PATH = os.getenv("VERIFIER_MODEL_PATH", "")
VERIFIER_IMGSZ = int(os.getenv("VERIFIER_IMGSZ", "1280"))

# ── Second stage: frame-level 26x behind 26n ───────────────────────────
# The verifier above is BOX-level — it only re-scores what 26n already
# nominated, and 40.1% of hand-labelled macaques produce no 26n box at all at
# conf 0.1, so there is nothing for it to adjudicate. Frame-level is a
# different mechanism: once 26n says "something is here", the big model re-runs
# the WHOLE frame and can find animals 26n never proposed. Measured on the
# crowd holdout (96 images, never trained on): recall 0.454 -> 0.599, F1
# 0.612 -> 0.728 — the union beats BOTH models alone, because they are blind to
# different animals.
#
# Empty path = disabled, which is the default: the weights this runs on are
# still training. Nothing about the realtime loop changes when it is off.
SECOND_STAGE_MODEL_PATH = os.getenv("SECOND_STAGE_MODEL_PATH", "")
# 26n's max confidence on a frame, above which that frame is worth a second
# look. 0 sends every frame, which production cannot afford (~20 inferences/s
# per camera at 477 ms each).
SECOND_STAGE_GATE_CONF = float(os.getenv("SECOND_STAGE_GATE_CONF", "0.25"))
# Confidence a 26x box needs to be ADDED as an animal 26n missed. This one
# creates a detection rather than confirming one, so it is not lenient.
SECOND_STAGE_CONF = float(os.getenv("SECOND_STAGE_CONF", "0.5"))
# A 26x box overlapping a 26n box by at least this is the same animal, not a
# new one.
SECOND_STAGE_DEDUP_IOU = float(os.getenv("SECOND_STAGE_DEDUP_IOU", "0.5"))
# The queue is BOUNDED and offers are non-blocking. A full queue drops the
# frame and counts it: a missed second stage costs one box, a blocked detector
# thread costs the whole camera — which is how 場域甲 went unwatched for 62
# hours (2026-08-06, 2026-08-08).
SECOND_STAGE_QUEUE_MAX = int(os.getenv("SECOND_STAGE_QUEUE_MAX", "8"))
# How many finished frame results to keep. Finalisation reads them by frame
# index; anything older than the trail window is dead weight.
SECOND_STAGE_RESULT_MAX = int(os.getenv("SECOND_STAGE_RESULT_MAX", "600"))
# Node heartbeat period, and how stale a beat may be before the 節點 tab shows
# the node offline. Names kept from the edge/cloud split so existing .env files
# still apply.
EDGE_HEARTBEAT_INTERVAL = int(os.getenv("EDGE_HEARTBEAT_INTERVAL", "30"))  # seconds
EDGE_OFFLINE_ALERT_SECONDS = int(os.getenv("EDGE_OFFLINE_ALERT_SECONDS", "120"))  # missed-heartbeat threshold
# A node can be online, its streams connected, and one camera still blind: on
# 2026-08-08 場域甲's detector thread died on an uncaught blob-upload timeout and
# nobody noticed for 62 hours, because the node kept beating and the dashboard
# kept showing that camera's last rolling-window FPS. Alert per CAMERA when a
# connected feed stops being inferred. Must sit well above the idle-gate
# heartbeat (2 s) and above a normal RTSP reconnect (~5 s), so a gated or
# briefly-reconnecting camera never trips it.
CAMERA_STALL_ALERT_SECONDS = int(os.getenv("CAMERA_STALL_ALERT_SECONDS", "300"))
# How often the node that owns the detectors checks that each worker THREAD is
# still alive and restarts the ones that are not. Nothing supervised them
# before: DetectorWorker.start() set a flag and never looked again.
WORKER_WATCHDOG_SECONDS = float(os.getenv("WORKER_WATCHDOG_SECONDS", "20"))

# RTSP / Stream
# Process every Nth CAPTURED frame. 1 = every frame the camera delivers, which
# is the sane default now that the detector consumes frames by capture sequence
# instead of polling on a timer: inference itself is the throttle, so skipping
# only threw away freshness. Raise it to deliberately cap the inference rate.
# Clamped to >=1: the sequence-threshold consumer asks for frame captured_seq
# + FRAME_SKIP - 1, so 0 would request an already-consumed frame and silently
# re-infer it forever (the old modulo loop at least crashed loudly).
FRAME_SKIP = max(1, int(os.getenv("FRAME_SKIP", "1")))
STREAM_FPS = int(os.getenv("STREAM_FPS", "15"))  # MJPEG output FPS cap
# Capture FPS is a delivery-rate metric, not the instantaneous rate at which
# an HLS/RTSP decoder drains a burst. Ten seconds smooths packet/HLS bursts
# while still reacting quickly to a real source stall.
CAPTURE_FPS_WINDOW_SECONDS = max(
    2.0, float(os.getenv("CAPTURE_FPS_WINDOW_SECONDS", "10"))
)
# Idle-camera inference gate. A feed carrying no image at all — a lens-black
# night scene — is sampled once every IDLE_GATE_HEARTBEAT_SECONDS instead of
# on every captured frame, so the GPU turns go to the cameras that can
# actually see something. Measured in production at 22:30: three blacked-out
# cameras sat at pixel std 1.0-1.7 (mean brightness 23/255) while the one lit
# camera was at 55, and all four were served an identical 11.5 FPS by the FIFO
# lock — three quarters of the GPU inferring black.
#
# The threshold is deliberately BELOW CameraStream's vitality threshold (5.0,
# which drives the UI's 運作中 dot): a dim-but-real scene the dashboard already
# shows as dark still gets full-rate inference. This gate is not the motion
# gate reverted in 7098971 — that one skipped frames in a VISIBLE scene where a
# still animal could hide. Here there is no image to look at, and the camera is
# never silenced: the heartbeat keeps the model itself checking, and vitality
# re-arms full rate within ~1 s of the scene lighting up.
#
# 0 disables the gate entirely (every connected camera inferred at full rate).
IDLE_GATE_STDDEV = float(os.getenv("IDLE_GATE_STDDEV", "3.0"))
IDLE_GATE_HEARTBEAT_SECONDS = float(os.getenv("IDLE_GATE_HEARTBEAT_SECONDS", "2.0"))

# YOLO imgsz (letterbox, preserves aspect ratio) — a FLOOR, not a fixed size.
# The worker infers at max(this, the frame's own long side rounded up to 32), so
# a frame is upscaled to the floor but NEVER shrunk.
#
# It used to be a fixed 1600, chosen when every camera delivered 1280x720. 場域甲
# now delivers 1920x1080 and nothing noticed, so production was shrinking the
# only feed that ever sees a macaque by 0.833x — a median 67x71 px animal
# arriving at the network as 56x59. Measured on 195 hand-labelled 1920x1080
# frames (2026-08-13): recall 0.349 at a fixed 1600 vs 0.426 inferred at native
# 1920, +22% relative, precision 0.977 -> 0.956.
#
# The floor stays 1600 because upscaling small sources genuinely helps: the
# 1280x720 cameras still infer at 1600, unchanged and free. What the rule
# removes is the silent mismatch — a constant cannot track a camera that is
# swapped or reconfigured, and this one did not.
#
# Never-downscale beat the fixed 1600 on BOTH gate splits (test F1 0.8429 vs
# 0.8363, val 0.8542 vs 0.8534) while keeping all three must_sight assertions,
# where a fixed 1920 lost one outright: the 2048x1094 fixture shrunk by 0.94x
# drops box[0] from IoU 0.452 to 0.000. Rescaling is what hurts, not any
# particular number, which is the whole reason this is a rule and not a knob.
INFERENCE_IMGSZ = int(os.getenv("INFERENCE_IMGSZ", "1600"))


def imgsz_for_frame(height: int, width: int) -> int:
    """Inference size for a frame: the floor, or its own size — never smaller."""
    long_side = max(int(height), int(width))
    return max(INFERENCE_IMGSZ, -(-long_side // 32) * 32)


# Rebalance a frame's channels when its colour cast is broken. At dawn the
# camera's white balance fails hard — 2026-08-15 06:17 delivered B/R 1.92
# against a normal 1.07 — and the detector goes nearly blind: 6 boxes across 59
# frames, where five macaques were walking down the road in plain view.
# Rebalancing recovers 96.
#
# GATED, not always-on. Correcting every frame also raises recall on normal
# ones (+27% on the crowd set) but costs F1 on the existing benchmark
# (test 0.8429 -> 0.8283, past tolerance), so it does not ship that way. Above
# 1.5 the correction is a no-op on everything healthy: 0 of 285 sampled harvest
# frames trip it, benchmark test F1 is unchanged to 4 decimal places, and the
# broken dawn frames trip it 100% of the time.
#
# 0 disables the correction entirely.
WHITE_BALANCE_CAST_THRESHOLD = float(os.getenv("WHITE_BALANCE_CAST_THRESHOLD", "1.5"))


def colour_cast(frame) -> float:
    """max/min of the per-channel means. 1.0 = neutral; the dawn failure is 1.9.

    Sampled every 8th pixel — a mean does not need every pixel, and this sits on
    the inference thread where a full-frame reduction would cost real FPS.
    """
    small = frame[::8, ::8]
    means = [float(small[:, :, i].mean()) for i in range(small.shape[2])]
    lo = min(means)
    return max(means) / lo if lo > 1e-6 else 1.0


def correct_white_balance(frame):
    """Scale each channel to the mean of the three (gray-world)."""
    import numpy as _np

    out = frame.astype(_np.float32)
    means = [out[:, :, i].mean() for i in range(3)]
    target = sum(means) / 3
    for i in range(3):
        out[:, :, i] *= target / max(1e-6, means[i])
    return _np.clip(out, 0, 255).astype(_np.uint8)

# Tracking
# Seconds a track may go unseen before it is finalised. Deliberately wall-clock
# rather than a frame count: throughput changes with resolution, model, camera
# count and hardware, and a frame-counted timeout silently shrinks as the frame
# rate rises — which would split a single visit into several detections. 9 s
# matches the old 60-frame default at the throughput it was tuned on.
TRACK_LOST_SECONDS = float(os.getenv("TRACK_LOST_SECONDS", "9.0"))
# Our own tracker config, NOT ultralytics' stock bytetrack.yaml. The stock
# track_buffer=30 is 30 processed frames, which is one second at the 30 FPS it
# assumes but only ~4.4 s at our 6.5-7.0 — so a macaque the detector missed for
# six seconds came back under a NEW id and was filed as a second animal.
# Raised to 46 (~7.1 s at the 6.5 FPS low end); the file's header carries the sweep.
#
# Its value in seconds MUST stay below TRACK_LOST_SECONDS above. That timeout
# is when we finalise and write the row, so if the tracker outlives it we file
# the row first and then start a fresh track for the same id when the animal
# returns — the same duplicate, one layer up, and past the reach of the
# continuation merge below. Asserted in tests/test_detection_events.py.
TRACKER_YAML = os.getenv("TRACKER_YAML", str(BASE_DIR / "trackers" / "bytetrack-monkey.yaml"))
# Nominal per-camera throughput, for converting the tracker's FRAME counts into
# seconds. Measured in production, not a target: Deployment measurements record 6.5-7.0 FPS
# per camera at 4 cameras. Deliberately the LOW end — a frame count lasts LONGER
# in wall clock when throughput drops, so the low end is the case that can push
# the tracker past TRACK_LOST_SECONDS. Only used for that conversion and the
# invariant check.
TRACKER_NOMINAL_FPS = float(os.getenv("TRACKER_NOMINAL_FPS", "6.5"))
# A lost track that comes back is still the same visit. When a track is
# finalised, a still-active track on the same camera that STARTED where the
# lost one ENDED, within this window, is treated as its continuation: the two
# are stitched into one row and one clip instead of two. Deliberately narrow —
# it requires spatial continuity, never identity across a gap in space.
TRACK_CONTINUATION_SECONDS = float(os.getenv("TRACK_CONTINUATION_SECONDS", "20.0"))
TRACK_CONTINUATION_IOU = float(os.getenv("TRACK_CONTINUATION_IOU", "0.5"))
# A MOVING animal needs a different test. Overlap works when the macaque was
# lost where it sat, but one running across the frame moves nearly a box width
# between fragments, so even a perfect prediction of where it went overlaps its
# own next box by only 0.2-0.3 (measured on the 2026-08-28 07:06:28 run, where
# the tracker handed out eight ids for one crossing). Distance from the
# extrapolated position, in box widths, does not collapse like that: the same
# eight fragments sit within 0.30-0.82.
#
# The window is far shorter than TRACK_CONTINUATION_SECONDS on purpose. Overlap
# claims "it never left that spot"; this claims "it carried on at that speed",
# which two animals running the same beam in sequence could also satisfy. Half
# a second is too short for the second one to have arrived.
TRACK_MOTION_GAP_SECONDS = float(os.getenv("TRACK_MOTION_GAP_SECONDS", "0.5"))
# 0.9, calibrated 2026-08-30. SAM2 propagation at native
# resolution, one predictor per subprocess, adjudicated 754 real fragment
# pairs plus 915 certain-negative pairs (two animals on screen at once):
# every pixel-confirmed TRUE continuation sat at or below 0.87 of a box width
# (the eight-fragment ground-truth crossing spans 0.30-0.82), while confirmed
# FALSE merges appeared from 0.81 up and the 0.9-1.0 sliver contained only
# false ones. The old 1.0 therefore bought no real merge and admitted the
# worst offenders; 0.9 keeps every observed true link with a 1.10x margin
# over the ground-truth maximum. Geometry cannot separate 0.8-0.9 at all —
# two animals handed off 0.8 widths apart in one frame were photographed
# doing it — so the residual risk there is accepted live and resolved
# offline, where the backfill can afford a pixel judge per pair.
TRACK_MOTION_ERR_WIDTHS = float(os.getenv("TRACK_MOTION_ERR_WIDTHS", "0.9"))
# Two tracks that are on screen AT THE SAME TIME with box-on-box overlap are
# one animal re-acquired under a second id, not two animals — one body cannot
# fill two overlapping boxes in the same frame. Measured 2026-08-13:
# 9 such pairs, ~11% of rows; t432/t433 were 2 px apart in the same second.
# 2026-08-27 over all 669 production rows: 44 pairs, and the pair that prompted
# this (#15157/#15158, t7467/t7466) was 0.2 s apart at IoU 0.79.
#
# _adopt_continuation cannot reach these. It is forward-only and skips overlaps
# ON PURPOSE, because overlapping usually means two animals side by side. The
# claim here is narrower and checkable: overlapping IN THE SAME FRAMES, box on
# box, measured on the frames both tracks actually appear in.
TRACK_DUPLICATE_IOU = float(os.getenv("TRACK_DUPLICATE_IOU", "0.5"))
# One coincidental overlap is not evidence. Require the two to be box-on-box
# across at least this many shared frames before calling them one animal.
TRACK_DUPLICATE_MIN_FRAMES = int(os.getenv("TRACK_DUPLICATE_MIN_FRAMES", "3"))
# A frame count alone cannot separate a duplicate from an OCCLUSION: two
# macaques crossing are mutually contained for the two or three frames one
# passes in front of the other, and each track only holds the frames it was
# actually detected in, so the crossing can be most of what they share. A
# duplicate is different in kind — the two ids co-exist for most of the shorter
# track's life, not for a moment of it. Shared frames must therefore be at least
# this fraction of the SHORTER track before the two are called one animal.
TRACK_DUPLICATE_MIN_OVERLAP = float(os.getenv("TRACK_DUPLICATE_MIN_OVERLAP", "0.5"))

# ── One visit, one clip (the visit-continuation design, deferred-write half) ─────────────────────
# A group of tracks finalises when they are all lost, but a visit is not over
# just because one group ended: measured 2026-08-27, 341 visits produced 570
# clips, a median of 1.7 per visit and a worst case of 7. The fix is to hold a
# finalised group briefly and merge whatever else finalises on the same camera
# behind it.
#
# 20 s deliberately matches EVENT_GAP_SECONDS, the window the events endpoint
# has grouped rows by since 2026-08-13 — one definition of "still the same
# visit", not two that can drift.
VISIT_HOLD_SECONDS = float(os.getenv("VISIT_HOLD_SECONDS", "20.0"))
# A visit that keeps producing groups must still be written eventually; without
# a ceiling a troop that stays for an hour is one clip that never lands.
VISIT_HOLD_MAX_SECONDS = float(os.getenv("VISIT_HOLD_MAX_SECONDS", "120.0"))
# Held visits carry their OWN frames — the ring buffer has evicted them long
# before the hold expires — so the queue is the memory bound, not a nicety:
# ~150 frames x ~0.4 MB is ~60 MB per busy visit. Full means render NOW rather
# than queue, which degrades to today's behaviour instead of growing the heap.
# The 2026-07-24 OOM (+1.6 GB/min, SIGKILL every ~10 min) was this same class
# of spend, so this ceiling is load-bearing.
VISIT_QUEUE_MAX = int(os.getenv("VISIT_QUEUE_MAX", "4"))
# A track records the frames it was DETECTED in. A gap wider than this between
# consecutive detections is drawn as the animal being away rather than as
# continuous presence — which is the distinction the timeline exists to show.
#
# 1.5 s is a little over the frame interval production actually delivers
# (6.5-7.0 FPS per camera, so ~0.15 s between processed frames, and the raw
# archive is subsampled on top of that). Small enough that walking out of frame
# reads as a break; large enough that one missed frame does not.
SEGMENT_GAP_SECONDS = float(os.getenv("SEGMENT_GAP_SECONDS", "1.5"))
TRAIL_GIF_MAX_FRAMES = int(os.getenv("TRAIL_GIF_MAX_FRAMES", "150"))  # Max unique frames to include in a trail
TRAIL_GIF_SCALE = float(os.getenv("TRAIL_GIF_SCALE", "0.5"))  # Downscale trail frames to save space (0.5 = half res)
# Trail MP4s use a real constant-rate timeline. Missing playback slots repeat
# the most recent selected frame, so a bursty RTSP source cannot silently turn
# eight seconds of capture time into four seconds of accelerated video. The
# unique source-frame ceiling above remains the memory/storage guard; repeated
# references do not duplicate the decoded image in memory.
TRAIL_VIDEO_FPS = float(os.getenv("TRAIL_VIDEO_FPS", "24"))
TRAIL_VIDEO_MAX_FRAMES = int(os.getenv("TRAIL_VIDEO_MAX_FRAMES", "600"))
# MP4 is inter-frame compressed, so the trail keeps every selected frame rather
# than throwing half of them away to satisfy the old GIF byte ceiling. ffmpeg
# writes an H.264 MP4 with +faststart, allowing the browser to begin playback
# before it has downloaded the whole clip.


def _resolve_ffmpeg() -> str:
    """Find an encoder without assuming the host has one installed.

    The container installs ffmpeg via apt, but the EDGE NODE — where inference
    actually runs — had neither ffmpeg nor Homebrew. Measured 2026-08-17: the
    mini pulled the MP4 change and every clip silently encoded to nothing (the
    fail-open path did its job, so the cameras kept watching and no error
    surfaced). `imageio-ffmpeg` is a normal dependency that ships a static
    build per platform, so `uv sync` now brings the encoder with it and a fresh
    node cannot repeat this.

    A system ffmpeg still wins when present: it is usually newer and it is what
    the container image already has.
    """
    override = os.getenv("FFMPEG_BINARY")
    if override:
        return override
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"      # let the encoder fail open rather than import-error


FFMPEG_BINARY = _resolve_ffmpeg()
TRAIL_MP4_CRF = int(os.getenv("TRAIL_MP4_CRF", "28"))
# Ceiling on ONE detection's raw-frame archive. Over it, _pack_raw_frames_under_limit
# throws away every other frame (`kept[::2]`) and repeats until the rest fits, so
# this number decides how much of a visit can still be re-examined later.
#
# Raised 32 -> 128 on 2026-08-27, after re-running the model over every archived
# frame (555 clips, 29,665 frames) showed what 32 MB had cost: only **216 of 555
# archives still held every frame** — 202 had lost half and 137 three quarters.
# That is not a detail. A frame is one instant, so per-frame boxes are the only
# way to measure how many animals a clip really held; with the peak frame thrown
# away, "this clip's rows outnumber the animals in it" cannot be claimed. Six out
# of every ten archives could not support that check, which is most of the value
# the archive exists for.
#
# 128 MB is sized from the same measurement: restoring every dropped frame would
# take the corpus from 13.12 GB to 26.81 GB, the quartered archives to ~82 MB
# each and the halved ones to ~52 MB, so 128 MB keeps essentially all of them
# whole. Blob storage goes from 18.5 GB to roughly 32 GB at steady state — about
# $0.25/month more against ~$42/month of credits.
#
# The reason storage feels dangerous here is the 2026-07-15 cleanup, where 202 GB
# of orphans had to be deleted. That was a LEAK — delete_selected never removed
# the rawframes ZIP — and it was fixed on 2026-08-19 (`_detection_files` now names
# it, plus a daily orphan sweep). This ceiling was never what stood between us and
# that, and paying for it in lost frames buys nothing.
#
# Forward-looking only: archives already cut stay cut.
RAW_FRAMES_MAX_BYTES = int(float(os.getenv("RAW_FRAMES_MAX_MB", "128")) * 1024 * 1024)

TRAIL_PRE_BUFFER_SECONDS = float(os.getenv("TRAIL_PRE_BUFFER_SECONDS", "8.0"))  # Seconds of clean frames BEFORE first detection (captures the "walk-in")
TRAIL_POST_BUFFER_SECONDS = float(os.getenv("TRAIL_POST_BUFFER_SECONDS", "1.0"))  # Seconds of clean frames AFTER last detection

# Dev back door: fire webhook on ANY raw YOLO detection (bypassing tracking, DB, snapshots).
# Use for diagnosing tracking drops. Set DEV_RAW_WEBHOOK=0 to disable.
DEV_RAW_WEBHOOK = os.getenv("DEV_RAW_WEBHOOK", "1").lower() in ("1", "true", "yes")
DEV_RAW_WEBHOOK_COOLDOWN = float(os.getenv("DEV_RAW_WEBHOOK_COOLDOWN", "5"))  # seconds between messages per camera
# Back door noise floor: with TRACKER_CONFIDENCE_THRESHOLD lowered to 0.1,
# YOLO fires on low-score noise. Only surface detections above this to keep
# the Discord diagnostic channel readable.
DEV_RAW_WEBHOOK_MIN_CONF = float(os.getenv("DEV_RAW_WEBHOOK_MIN_CONF", "0.3"))

# CORS — comma-separated, strip whitespace; "*" means allow all (no credentials)
_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000").strip()
CORS_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()]

# Auth: HMAC secret used to sign session tokens AND to hash access codes.
# This public fallback is for a local demo only. Set ADMIN_PASSWORD_SALT to a
# persistent secret before deployment; changing it invalidates stored code hashes.
AUTH_SECRET = os.getenv("ADMIN_PASSWORD_SALT", "local-demo-only-change-before-deployment")

# Seed access codes — used ONLY to create the initial rows when the users table
# is empty. Use the demo codes locally; override both before deployment.
SEED_ADMIN_CODE = os.getenv("SEED_ADMIN_CODE", "demo-admin")
SEED_USER_CODE = os.getenv("SEED_USER_CODE", "demo-user")

# Per-user toggleable features (key → 顯示名稱). Extend here; the admin UI renders
# a switch per entry and the backend enforces the wired ones. Admins bypass all.
# NOTE: live_feeds/downloads default ON for users (_feature_enabled); training is
# the exception — it can start paid GPU jobs and trigger deploys, so it is OPT-IN
# (require_training_access treats a missing flag as off).
FEATURES: dict[str, str] = {
    "live_feeds": "即時影像",       # 檢視即時串流（feeds 分頁 + 總覽最新影像）
    "downloads": "下載影像／影格",  # 下載偵測影格 ZIP
    "training": "訓練",             # 模型訓練控制台：上傳資料集、啟動雲端訓練、觸發上板（預設關閉）
}

# ── Ultralytics Platform (model-lifecycle control plane) ────────────────────
# The Platform is the single source of truth for datasets/models/training; the
# dashboard drives it through services/platform.py. Key format: ul_ + 40 hex.
ULTRALYTICS_API_KEY = os.getenv("ULTRALYTICS_API_KEY", "")
ULTRALYTICS_PROJECT = os.getenv("ULTRALYTICS_PROJECT", "monkey_v2")
# The dataset new material accumulates into, and the fallback when a request
# names none. Moved off `monkey cocotree` (the July set) on 2026-08-17: the
# shipping model was trained on `monkey cocotree allin 0816` — 1220 images, the
# train+val of all six labelled datasets merged — while this still named the
# 1095-image July set, so 開始訓練 from the dashboard would have quietly used
# LESS data than production. Both the browser and the training dialog now pick
# explicitly; this is only the default they fall back to.
ULTRALYTICS_DATASET = os.getenv("ULTRALYTICS_DATASET", "monkey cocotree v2")

# ── Training-set deduplication (master + derived) ───────────────────────────
# The platform dataset is the MASTER: everything lands there, nothing is deleted,
# test/val stay pinned. Training runs on a DERIVED dataset whose TRAIN split has
# perceptual near-duplicates collapsed to one representative each. Measured
# 2026-08-03: 27% of the corpus is near-duplicate, and a 25%-smaller train split
# trained a BETTER model (test recall 0.704 -> 0.756, F1 0.7979 -> 0.8320).
# Max dHash Hamming distance (of 64 bits) still counted as the same picture.
DEDUP_THRESHOLD = int(os.getenv("DEDUP_HAMMING_THRESHOLD", "5"))
# Where downloaded master images are cached while a derived set is built. Shares
# the ratchet's gitignored cache (same content-hash filenames) so the two never
# download the same image twice; falls back inside backend/ off a repo checkout.
DEDUP_CACHE_DIR = Path(
    os.getenv(
        "DEDUP_CACHE_DIR",
        str(
            BASE_DIR.parent / "tests" / ".dataset_cache"
            if (BASE_DIR.parent / "tests").is_dir()
            else BASE_DIR / ".dataset_cache"
        ),
    )
)
# How long the master's perceptual hashes are reused for the add-to-dataset
# duplicate warning (they cost one thumbnail fetch per master image).
DEDUP_MASTER_TTL_SECONDS = float(os.getenv("DEDUP_MASTER_TTL_SECONDS", "600"))

# GitHub workflow dispatch — the 上板 (promote) button POSTs a workflow_dispatch
# for model-promote.yml; the self-hosted mac-mini runner runs the ratchet gate.
GITHUB_DISPATCH_TOKEN = os.getenv("GITHUB_DISPATCH_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "").strip()
