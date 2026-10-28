"""Idle-gate behaviour tests — no model, no camera, no network.

The gate decides when the detector STOPS inferring, so the tests that matter
are the ones asserting it never closes for the wrong reason: a young stream, a
dim-but-real scene, or a camera with a live track. Runs the real
`CameraStream.is_blank` state machine against synthetic frames, and the real
`DetectorWorker._run` loop against a stub stream with the model stubbed out.

    uv run --project backend python tests/test_idle_gate.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import config  # noqa: E402

config.IDLE_GATE_STDDEV = 3.0
config.IDLE_GATE_HEARTBEAT_SECONDS = 0.2  # keep the loop tests sub-second

from services import stream as stream_mod  # noqa: E402

stream_mod.IDLE_GATE_STDDEV = config.IDLE_GATE_STDDEV
from services.stream import CameraStream  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{f' — {detail}' if detail else ''}")
    if not cond:
        FAILURES.append(name)


def _frame(mean: int, noise: float) -> np.ndarray:
    """720p frame whose std SURVIVES the 160x90 downsample the gate measures.

    Structure has to be coarse: independent per-pixel noise is averaged away
    by the 8x8 INTER_AREA box (std/8), so a fixture built that way would
    measure ~6 where it meant 55 and test nothing.
    """
    import cv2

    rng = np.random.default_rng(0)
    coarse = np.clip(rng.normal(mean, noise, (90, 160, 3)), 0, 255).astype(np.uint8)
    return cv2.resize(coarse, (1280, 720), interpolation=cv2.INTER_NEAREST)


def _stddev_of(frame: np.ndarray) -> float:
    import cv2

    return float(np.std(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)))


def _armed_stream(stale_after: float = 0.3) -> CameraStream:
    """A connected stream with a short stale window, ready to fingerprint."""
    s = CameraStream("cam", "rtsp://example/none")
    s._VITALITY_STALE_AFTER = stale_after
    s._VITALITY_CHECK_INTERVAL = 0.0  # fingerprint every frame we hand it
    s._connected = True
    s._last_lit_frame_ts = time.time()  # what _capture_loop does on connect
    return s


def _feed(s: CameraStream, frame: np.ndarray, seconds: float) -> None:
    for _ in range(int(seconds / 0.02)):
        s._check_vitality(frame)
        time.sleep(0.02)


# ── vitality → gate state machine ───────────────────────────────────────────

print("\nCameraStream.is_blank")

# The production night frames measured 22:30: mean ~23, std 1.0-1.7.
black = _frame(23, 1.2)
dim = _frame(40, 4.5)     # below the vitality threshold (5), above the gate (3)
lit = _frame(90, 55.0)

check("black frame is under the gate", _stddev_of(black) < config.IDLE_GATE_STDDEV,
      f"std={_stddev_of(black):.2f}")
check("dim frame sits between gate and vitality",
      config.IDLE_GATE_STDDEV < _stddev_of(dim) < CameraStream._VITALITY_STDDEV_THRESHOLD,
      f"std={_stddev_of(dim):.2f}")
check("lit frame is well above both", _stddev_of(lit) > 20, f"std={_stddev_of(lit):.2f}")

s = _armed_stream()
s._check_vitality(black)
check("young stream is not gated on its first black frame", not s.is_blank)

_feed(s, black, 0.45)  # past the stale window
check("sustained black closes the gate", s.is_blank)

s._check_vitality(lit)
check("one lit frame re-opens the gate instantly", not s.is_blank)

# A dim-but-real scene: `is_alive` says dark (UI dot grey) but the gate must
# stay OPEN — this is the margin that separates this from the motion gate.
s2 = _armed_stream()
_feed(s2, dim, 0.45)
check("dim-but-real scene is NOT gated", not s2.is_blank)
check("...while still reporting dark to the UI", not s2.is_alive)

s3 = _armed_stream()
_feed(s3, black, 0.45)
s3._connected = False
check("disconnected stream is not reported as gated", not s3.is_blank)

s4 = CameraStream("file", "/tmp/clip.mp4")
s4._VITALITY_STALE_AFTER = 0.3
s4._connected = True
s4._last_lit_frame_ts = time.time() - 10
check("file sources are never gated", not s4.is_blank)

s5 = _armed_stream()
_feed(s5, black, 0.45)
stream_mod.IDLE_GATE_STDDEV, saved = 0.0, stream_mod.IDLE_GATE_STDDEV
check("IDLE_GATE_STDDEV=0 disables the gate", not s5.is_blank)
stream_mod.IDLE_GATE_STDDEV = saved


# ── the worker loop actually skips inference ────────────────────────────────

print("\nDetectorWorker loop")


class _StubStream:
    """Delivers frames at ~100 fps with a settable is_blank verdict."""

    def __init__(self) -> None:
        self.connected = True
        self.is_blank = False
        self.frame_stddev = 0.0
        self._seq = 0
        self._frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def next_frame(self, after_seq: int, timeout: float):
        time.sleep(0.01)
        self._seq += 1
        return self._frame, self._seq, time.time()


class _Scalar:
    """Stands in for the 0-d tensors ultralytics hands back."""

    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v

    def tolist(self):
        return self._v


class _StubBoxes:
    def __init__(self) -> None:
        box = type("Box", (), {"xyxy": [_Scalar([100.0, 100.0, 200.0, 200.0])], "conf": [0.9]})()
        self._boxes = [box]
        self.id = [_Scalar(1)]

    def __len__(self):
        return len(self._boxes)

    def __iter__(self):
        return iter(self._boxes)


class _StubModel:
    """`detecting=True` returns one tracked box every frame, so the worker
    builds a live track — which the gate must refuse to close over."""

    def __init__(self, detecting: bool = False) -> None:
        self.calls = 0
        self._detecting = detecting

    def track(self, *a, **kw):
        self.calls += 1
        if not self._detecting:
            return []
        return [type("Result", (), {"boxes": _StubBoxes()})()]


def _run_worker(blank: bool, seconds: float, detecting: bool = False) -> tuple[int, int]:
    """Run the real loop for `seconds`; returns (inferences, gated frames)."""
    from services import detector as det_mod
    from services.detector import DetectorWorker

    stub_stream, stub_model = _StubStream(), _StubModel(detecting)
    stub_stream.is_blank = blank

    det_mod.IDLE_GATE_HEARTBEAT_SECONDS = config.IDLE_GATE_HEARTBEAT_SECONDS

    import services.stream as sm
    saved_get = sm.stream_manager.get
    sm.stream_manager.get = lambda cid: stub_stream
    saved_own = det_mod._own_model
    det_mod._own_model = lambda: stub_model

    worker = DetectorWorker("cam")
    worker._running = True
    t = threading.Thread(target=worker._run, daemon=True)
    t.start()
    time.sleep(seconds)
    worker._running = False
    t.join(timeout=5)

    sm.stream_manager.get = saved_get
    det_mod._own_model = saved_own
    return stub_model.calls, worker._metrics.frames_gated


DURATION = 1.0
HEARTBEAT = config.IDLE_GATE_HEARTBEAT_SECONDS
expected_beats = DURATION / HEARTBEAT  # ~5 at 0.2 s

lit_calls, lit_gated = _run_worker(blank=False, seconds=DURATION)
check("a lit camera infers every delivered frame", lit_calls > 30 and lit_gated == 0,
      f"{lit_calls} inferences, {lit_gated} gated")

blank_calls, blank_gated = _run_worker(blank=True, seconds=DURATION)
check("a blank camera drops to the heartbeat rate",
      blank_calls <= expected_beats + 2, f"{blank_calls} inferences in {DURATION}s")
check("...and still infers — never silenced", blank_calls >= 2,
      f"{blank_calls} inferences")
check("...with the skipped frames counted as gated", blank_gated > 20,
      f"{blank_gated} gated")
check("the gate is a real saving", blank_calls < lit_calls / 5,
      f"{blank_calls} vs {lit_calls} inferences")

# The guard that matters most: a monkey found in a technically-black frame
# keeps the camera at full rate until the track is finalised, so the gate can
# never cut away mid-visit.
track_calls, track_gated = _run_worker(blank=True, seconds=DURATION, detecting=True)
check("a live track overrides the gate", track_calls > 30 and track_gated <= 1,
      f"{track_calls} inferences, {track_gated} gated")

print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
sys.exit(1 if FAILURES else 0)
