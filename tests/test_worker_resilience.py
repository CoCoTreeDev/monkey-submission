"""Detector-worker survival tests — no model, no camera, no network.

The bug these exist for: on 2026-08-06 and again on 2026-08-08, a trail-clip /
raw-frames-ZIP upload to Azure Blob timed out, the exception propagated out of
`_finalise_batch` and out of the thread, and 場域甲 silently stopped being
inferred — 62 hours the second time, while its stream stayed connected and the
dashboard kept showing that camera's final rolling-window FPS (1.9) as if it
were merely slow.

So the invariants asserted here are, in order of how much they matter:

  1. an artefact upload that fails NEVER ends the worker;
  2. a clip/ZIP failure drops the incomplete detection, but never the camera;
  3. an exception anywhere in the loop restarts the loop, not the camera;
  4. a thread that dies anyway is revived by the manager's watchdog;
  5. the video path keeps its selected frames; its dedicated contract test
     asserts H.264, faststart, and exact frame count;
  6. coverage = inferred / delivered, reported per camera.

Runs the real `DetectorWorker._run`, the real `_finalise_batch`, and the real
encoders against stubs.

    uv run --project backend python tests/test_worker_resilience.py
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import config  # noqa: E402

config.VERIFIER_MODEL_PATH = ""      # no cascade verifier in these tests
config.IDLE_GATE_HEARTBEAT_SECONDS = 0.0  # gate off; this is not a gate test

from services import detector as det_mod  # noqa: E402
from services.detector import DetectorWorker, detector_manager  # noqa: E402

# This suite exercises finalisation and failure containment, not codec quality.
# Keep it independent of a host ffmpeg installation; test_trail_video.py runs
# the real encoder and verifies H.264, faststart, and frame count separately.
det_mod._encode_mp4 = lambda _frames, _timestamps: b"test-mp4"

det_mod.IDLE_GATE_HEARTBEAT_SECONDS = 0.0
det_mod.TRACK_LOST_SECONDS = 0.3     # finalise a track in a test-sized window
det_mod._LOOP_RESTART_DELAY = 0.05   # keep the crash-restart tests sub-second
det_mod.FRAME_SKIP = 1               # pinned: backend/.env may set anything
# These tests are about what survives a failing STORAGE, so finalisation has to
# stay synchronous: with the visit hold on, `offer()` takes the work and the
# assertions race a 20 s timer. 0 disables the hold, which is also the real
# kill switch — `_VisitQueue.enabled` is False and every group renders inline,
# exactly as it did before the visit-continuation design's deferred half. The hold's own behaviour
# is covered in tests/test_detection_events.py.
det_mod.VISIT_HOLD_SECONDS = 0.0
# Keep this suite focused on storage resilience. The dedicated validity tests
# cover the production defaults (3 hits over 0.5 s).
det_mod.TRACK_MIN_HITS = 2
det_mod.TRACK_MIN_SECONDS = 0.0

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{f' — {detail}' if detail else ''}")
    if not cond:
        FAILURES.append(name)


# ── stubs ───────────────────────────────────────────────────────────────────


class _StubStream:
    """Delivers a textured 720p frame at ~100 fps."""

    def __init__(self) -> None:
        self.connected = True
        self.is_blank = False
        self.frame_stddev = 40.0
        self.capture_fps = 30.0
        self.capture_gaps_ms: dict[str, float] = {}
        self.reconnects = 0
        self._seq = 0
        rng = np.random.default_rng(1)
        self._frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    def next_frame(self, after_seq: int, timeout: float):
        time.sleep(0.01)
        self._seq += 1
        return self._frame, self._seq, time.time()


class _Tensor:
    """Stands in for the torch tensors ultralytics hands back."""

    def __init__(self, v) -> None:
        self._v = v

    def item(self):
        return self._v

    def tolist(self):
        return self._v


class _StubBoxes:
    def __init__(self) -> None:
        box = type(
            "Box", (), {"xyxy": [_Tensor([100.0, 100.0, 300.0, 400.0])], "conf": [0.9]},
        )()
        self._boxes = [box]
        self.id = [_Tensor(7)]

    def __len__(self):
        return len(self._boxes)

    def __iter__(self):
        return iter(self._boxes)


class _StubModel:
    """Tracks one box for `detect_frames` calls, then sees nothing — so the
    track goes lost and the worker runs the real finalise/emit path."""

    def __init__(self, detect_frames: int) -> None:
        self.calls = 0
        self._detect_frames = detect_frames

    def track(self, *a, **kw):
        assert kw["classes"] == [0], "human must not enter the tracker"
        self.calls += 1
        if self.calls > self._detect_frames:
            return []
        return [type("Result", (), {"boxes": _StubBoxes()})()]


class _FakeStorage:
    """Records uploads; raises for names matching `fail_suffixes`.

    `probe` snapshots the inference count at the first upload attempt, so the
    tests can assert the worker kept going PAST that point instead of asserting
    a total that just measures how fast the machine is.
    """

    def __init__(self, fail_suffixes: tuple[str, ...] = (), probe=lambda: 0) -> None:
        self.uploaded: list[str] = []
        self.failed: list[str] = []
        self.inferences_at_first_upload: int | None = None
        self._fail = fail_suffixes
        self._probe = probe

    def upload(self, name: str, data: bytes) -> None:
        if self.inferences_at_first_upload is None:
            self.inferences_at_first_upload = self._probe()
        if self._fail and name.lower().endswith(self._fail):
            self.failed.append(name)
            # The real thing raises azure.core ServiceResponseError here.
            raise OSError("('Connection aborted.', TimeoutError('write timed out'))")
        self.uploaded.append(name)


def _run_worker(fail_suffixes: tuple[str, ...]):
    """Wait for finalisation and measurable subsequent inference, with a timeout.

    Fixed sleeps test host throughput instead of worker survival; encoding and
    competing builds can consume most of that interval.
    """
    import services.storage as storage_mod
    import services.stream as stream_mod

    stub_stream = _StubStream()
    # Detected briefly, then not: the track has to go LOST well inside the run,
    # so the finalise being tested happens in the loop and not in the shutdown
    # path (where a passing assertion would prove much less).
    stub_model = _StubModel(detect_frames=20)
    fake_storage = _FakeStorage(fail_suffixes, probe=lambda: stub_model.calls)

    saved_get, saved_own = stream_mod.stream_manager.get, det_mod._own_model
    saved_storage = storage_mod.storage
    stream_mod.stream_manager.get = lambda cid: stub_stream
    det_mod._own_model = lambda: stub_model
    storage_mod.storage = fake_storage

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    emitted: list[object] = []

    async def on_detection(detection):
        emitted.append(detection)

    worker = DetectorWorker("cam", on_detection=on_detection)
    worker._loop = loop
    worker._running = True
    thread = threading.Thread(target=worker._run, daemon=True)
    worker._thread = thread
    thread.start()

    deadline = time.monotonic() + 8.0
    while thread.is_alive() and time.monotonic() < deadline:
        first_upload = fake_storage.inferences_at_first_upload
        if first_upload is not None and stub_model.calls - first_upload > 10:
            break
        time.sleep(0.02)

    alive_during = thread.is_alive()
    inferences_at_end = stub_model.calls
    worker._running = False
    thread.join(timeout=5)

    stream_mod.stream_manager.get = saved_get
    det_mod._own_model = saved_own
    storage_mod.storage = saved_storage
    time.sleep(0.1)  # let any scheduled on_detection callback actually run
    loop.call_soon_threadsafe(loop.stop)

    return worker, fake_storage, emitted, alive_during, inferences_at_end


# ── 1 & 2: an upload failure costs an artefact, not the camera ───────────────

print("\nUpload failure containment")

def _progress_after_failure(store, calls: int) -> int:
    """Inferences run after the first upload attempt — i.e. after the failure."""
    return calls - (store.inferences_at_first_upload or 0)


worker, store, emitted, alive, calls = _run_worker(fail_suffixes=(".mp4", ".zip"))
check("the harness reached a real finalise", store.inferences_at_first_upload is not None,
      f"first upload attempted after {store.inferences_at_first_upload} inferences")
check("the MP4 upload failed as set up", any(n.endswith(".mp4") for n in store.failed),
      f"failed: {[n.rsplit('_', 1)[-1] for n in store.failed]}")
check("...and the ZIP was not even attempted (no second timeout budget spent)",
      not any(n.endswith(".zip") for n in store.failed + store.uploaded),
      f"uploaded: {[n.rsplit('_', 1)[-1] for n in store.uploaded]}")
check("the worker thread SURVIVES the failed upload", alive, f"{calls} inferences")
check("...and keeps inferring past it", _progress_after_failure(store, calls) > 10,
      f"{_progress_after_failure(store, calls)} inferences after the failure")
check("...with no crash restart needed (contained at the group)",
      worker.crashes == 0, f"crashes={worker.crashes}")
check("the incomplete detection is not emitted", len(emitted) == 0,
      f"{len(emitted)} emitted")

# The harsher case: EVERY upload fails, so even the snapshots are gone and the
# detection cannot be emitted. The camera must still be watching.
worker, store, emitted, alive, calls = _run_worker(fail_suffixes=(".jpg", ".mp4", ".zip"))
check("a total storage outage still leaves the worker alive", alive,
      f"{calls} inferences, {len(store.failed)} failed uploads")
check("...still inferring", _progress_after_failure(store, calls) > 10,
      f"{_progress_after_failure(store, calls)} inferences after the failure")
check("...and the lost detection is the only casualty", len(emitted) == 0,
      f"{len(emitted)} emitted")

# Baseline: with storage healthy the whole path works and emits.
worker, store, emitted, alive, calls = _run_worker(fail_suffixes=())
check("healthy storage emits the detection with its trail MP4",
      len(emitted) >= 1 and emitted[0].trail_path.endswith(".mp4"),
      f"{len(emitted)} emitted, {len(store.uploaded)} uploads")
check("...uploading an MP4 and a raw-frames ZIP",
      any(n.endswith(".mp4") for n in store.uploaded)
      and any(n.endswith(".zip") for n in store.uploaded),
      ", ".join(n.rsplit("_", 1)[-1] for n in store.uploaded))


# ── 3: any exception restarts the loop, never the camera ─────────────────────

print("\nLoop crash recovery")

worker = DetectorWorker("cam")
entries: list[int] = []


def _crashing_loop() -> None:
    entries.append(1)
    if len(entries) <= 2:
        raise RuntimeError("boom")
    while worker._running:
        time.sleep(0.01)


worker._detect_loop = _crashing_loop
worker._running = True
thread = threading.Thread(target=worker._run, daemon=True)
worker._thread = thread  # what start() does; `alive` reads it
thread.start()
time.sleep(0.5)
check("the loop is restarted after each crash", len(entries) == 3, f"{len(entries)} entries")
check("...counted, so a flapping worker is visible", worker.crashes == 2,
      f"crashes={worker.crashes}")
check("...and the thread is still the same live thread", thread.is_alive() and worker.alive)
worker._running = False
thread.join(timeout=2)


# ── 4: a thread that dies anyway gets revived ───────────────────────────────

print("\nWatchdog revival")

worker = DetectorWorker("cam2")
starts: list[int] = []


def _dying_loop() -> None:
    starts.append(1)
    if len(starts) == 1:
        raise SystemExit  # not an Exception: kills the thread, as the real one did
    while worker._running:
        time.sleep(0.01)


worker._detect_loop = _dying_loop
worker._running = True
worker._thread = threading.Thread(target=worker._run, daemon=True)
worker._thread.start()
time.sleep(0.2)
check("a thread killed outside `except Exception` really dies", not worker.alive)
check("...and is NOT reported as merely slow (alive is False)", worker.alive is False)

detector_manager._workers["cam2"] = worker
revived = detector_manager.revive_dead()
time.sleep(0.1)
check("the watchdog revives it", revived == ["cam2"], f"revived={revived}")
check("...bringing the thread back", worker.alive)
check("...and counting the revival", worker.revivals == 1, f"revivals={worker.revivals}")
check("a healthy worker is left alone", detector_manager.revive_dead() == [])
worker._running = False


# ── 5: raw-frame archives retain their own byte ceiling ─────────────────────

print("\nRaw-frame archive byte ceiling")

frames = [(f"f{i:04d}.jpg", bytes(10_000)) for i in range(100)]  # 1 MB total
saved_raw = det_mod.RAW_FRAMES_MAX_BYTES
det_mod.RAW_FRAMES_MAX_BYTES = 300_000
zipped, n_zipped = det_mod._pack_raw_frames_under_limit(frames)
check("the raw-frames ZIP is subsampled to fit", len(zipped) <= 400_000 and n_zipped < 100,
      f"{n_zipped} frames, {len(zipped) / 1024:.0f} KB")
det_mod.RAW_FRAMES_MAX_BYTES = saved_raw
whole, n_whole = det_mod._pack_raw_frames_under_limit(frames)
check("...and kept whole when it fits", n_whole == 100, f"{n_whole} frames")


# ── 6: coverage = inferred / delivered ─────────────────────────────────────

print("\nInference coverage")

import services.stream as stream_mod  # noqa: E402

stub_stream = _StubStream()
stub_stream.capture_fps = 30.0
saved_get = stream_mod.stream_manager.get
stream_mod.stream_manager.get = lambda cid: stub_stream

cov_worker = DetectorWorker("covcam")
now = time.time()
# Ten distinct frames over one second: 9 gaps / 1 s → sample_fps ~9.
for i in range(10):
    cov_worker._metrics._latencies.append(0.022)
    cov_worker._metrics._wall_times.append(now - 1.0 + i / 9.0)
    cov_worker._metrics._idle_marks.append(0)
    cov_worker._metrics._gated_marks.append(0)
cov_worker._metrics.last_ms = 22.0

detector_manager._workers = {"covcam": cov_worker}
stats = detector_manager.inference_stats()
row = stats["cameras"][0]
expected = round(min(1.0, row["sample_fps"] / 30.0), 3)
check("coverage is inferred FPS over delivered FPS",
      row["coverage_ratio"] == expected,
      f'{row["sample_fps"]}/30.0 → {row["coverage_ratio"]} (expected {expected})')
check("...which is well under 1 when the GPU cannot keep up",
      0.2 < row["coverage_ratio"] < 0.4, f'{row["coverage_ratio"]}')
check("the reachable ceiling is 1/FRAME_SKIP",
      stats["coverage_ceiling"] == round(1.0 / det_mod.FRAME_SKIP, 3),
      f'ceiling={stats["coverage_ceiling"]}, FRAME_SKIP={det_mod.FRAME_SKIP}')

stub_stream.capture_fps = 0.0  # camera delivering nothing → no division by zero
check("a camera delivering nothing reports 0 coverage, not a crash",
      detector_manager.inference_stats()["cameras"][0]["coverage_ratio"] == 0.0)

# Clamp: the two rolling windows are independent and can cross transiently.
stub_stream.capture_fps = 1.0
check("coverage never exceeds 100%",
      detector_manager.inference_stats()["cameras"][0]["coverage_ratio"] == 1.0)

# Liveness travels with the stats, so the dashboard can tell dead from slow.
row = detector_manager.inference_stats()["cameras"][0]
check("worker liveness is reported alongside throughput",
      row["worker_alive"] is False and "worker_revivals" in row,
      f'worker_alive={row["worker_alive"]}')

stream_mod.stream_manager.get = saved_get
detector_manager._workers = {}

print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
sys.exit(1 if FAILURES else 0)
