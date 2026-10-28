"""Second stage contract — no model, no camera, no network.

26x runs behind 26n at frame level and the union of their boxes beats either
alone (crowd holdout: recall 0.454 / 0.490 separately, 0.599 together). But the
accuracy is not what this file guards. What it guards is that the mechanism
carrying it cannot cost a camera:

  1. `offer()` NEVER blocks and never raises — the detector thread has died
     twice on inline I/O (2026-08-06, 2026-08-08) and each time one camera went
     unwatched, 62 hours the second time;
  2. a full queue DROPS and COUNTS, rather than waiting;
  3. finalisation reads whatever finished and never waits for the rest;
  4. the union only adds boxes no track already covers;
  5. disabled is the default, and disabled changes nothing.

Run with ``uv run --project backend python tests/test_second_stage.py``.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "backend")

import numpy as np  # noqa: E402

from services import detector as det_mod  # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label + (f": {detail}" if not cond else ""))
    if not cond:
        FAILS.append(label)


frame = np.zeros((64, 64, 3), dtype=np.uint8)

# ── 5. disabled by default, and disabled changes nothing ────────────────────
print("Disabled by default")
stage = det_mod._SecondStage()
check("SECOND_STAGE_MODEL_PATH is empty out of the box", not stage.enabled,
      repr(det_mod.SECOND_STAGE_MODEL_PATH))
stage.offer("cam", 1, frame, 1600)
check("...so an offer is a no-op", stage.queued == 0 and stage.dropped == 0, stage.stats())
check("...and the union contributes nothing",
      det_mod._second_stage_extras("cam", 1, []) == [])

# ── 1 & 2. offer never blocks; a full queue drops and counts ────────────────
print("\nA full queue drops rather than waits")
det_mod.SECOND_STAGE_MODEL_PATH = "/nonexistent-on-purpose.pt"
stage = det_mod._SecondStage()
# Stall the consumer. A worker that drains as fast as offers arrive never
# creates backpressure, so it would not exercise the path this asserts — the
# one where the GPU is busy and the queue is genuinely full.
stage._ensure_worker = lambda: None
CAP = det_mod.SECOND_STAGE_QUEUE_MAX
t0 = time.perf_counter()
for i in range(CAP * 4):
    stage.offer("cam", i, frame, 1600)
elapsed = time.perf_counter() - t0
check("offering 4x the queue capacity is near-instant", elapsed < 2.0, f"{elapsed:.2f}s")
check("the queue never exceeded its bound", stage.queued <= CAP + 2,
      f"queued={stage.queued} cap={CAP}")
check("the excess was DROPPED, and counted", stage.dropped > 0, stage.stats())
check("queued + dropped accounts for every offer",
      stage.queued + stage.dropped == CAP * 4, stage.stats())

# ── offer must not raise even when the frame is unusable ───────────────────
print("\nA bad frame loses one second stage, not the thread")


class _Unfriendly:
    def copy(self):
        raise RuntimeError("frame is gone")


before = stage.dropped
try:
    stage.offer("cam", 999, _Unfriendly(), 1600)
    raised = False
except Exception as exc:                                    # pragma: no cover
    raised = True
    print("   ", repr(exc))
check("offer() swallows a broken frame", not raised)
check("...and counts it as a drop", stage.dropped == before + 1, stage.stats())

# ── 3. finalisation reads what finished, never waits ───────────────────────
print("\nFinalisation never waits for stage 2")
stage = det_mod._SecondStage()
det_mod.second_stage = stage
det_mod.SECOND_STAGE_MODEL_PATH = "x.pt"          # enabled, but nothing processed
t0 = time.perf_counter()
got = det_mod._second_stage_extras("cam", 7, [])
check("a frame stage 2 has not reached yet returns immediately",
      got == [] and time.perf_counter() - t0 < 0.05)

# ── 4. the union adds only what no track covers ────────────────────────────
print("\nUnion adds only uncovered boxes")


class _FakeTrack:
    def __init__(self, box):
        self._box = box

    def bbox_near(self, _idx, _gap):
        return self._box            # (x, y, w, h)


with stage._lock:
    stage._results[("cam", 7)] = [
        (10.0, 10.0, 50.0, 50.0),    # same animal a track already has
        (200.0, 200.0, 240.0, 240.0),  # one 26n never proposed
    ]
extras = det_mod._second_stage_extras("cam", 7, [_FakeTrack((10.0, 10.0, 40.0, 40.0))])
check("the box a track already covers is dropped as a duplicate",
      all(e[0] != 10.0 for e in extras), extras)
check("the box no track covers survives",
      any(abs(e[0] - 200.0) < 1e-6 for e in extras), extras)
check("exactly one extra", len(extras) == 1, extras)
check("extras are counted", stage.extra_boxes == 1, stage.stats())

# ── the result store is bounded ────────────────────────────────────────────
print("\nThe result store is bounded")
stage = det_mod._SecondStage()
with stage._lock:
    for i in range(det_mod.SECOND_STAGE_RESULT_MAX + 50):
        stage._results[("cam", i)] = []
        while len(stage._results) > det_mod.SECOND_STAGE_RESULT_MAX:
            stage._results.popitem(last=False)
check("stored results cannot grow without bound",
      len(stage._results) <= det_mod.SECOND_STAGE_RESULT_MAX, len(stage._results))

print("\n" + ("ALL PASS" if not FAILS else "FAILED: %d\n  - %s" % (len(FAILS), "\n  - ".join(FAILS))))
sys.exit(1 if FAILS else 0)
