"""How many track ids does ByteTrack spend on one animal? Replayed per scene.

Why this exists (measured 2026-08-19): a macaque sitting still on the guardrail
was filed as TWO detections seven seconds apart, under two track ids, with two
GIFs — and the pair was visually identical down to the reflector behind it.
The animal was never lost. ByteTrack's id was.

`bytetrack.yaml` ships `track_buffer: 30`, and ultralytics builds the tracker
with a hardcoded `frame_rate=30` (trackers/track.py), so
`max_time_lost = int(30/30 * 30) = 30` PROCESSED FRAMES. That constant assumes
30 FPS video, where it buys one second. Production runs **6.5-7.0 FPS per
camera** (deployment measurements), so the same 30 frames buy **~4.4 seconds** — while our own
`TRACK_LOST_SECONDS` waits 9. The layer underneath gives up first, so a dropout
between 4.4 s and 9 s mints a new id before our timeout can hold the track
together, and every new id is a new row and a new clip.

This measures the metric the visit-continuation design accepts on: **fragmentation** = distinct track
ids a scene produces / peak concurrent animals in that scene. It replays the
real BYTETracker over the 2026-08-13 harvest's per-frame consensus boxes.

Fragmentation is NOT the id-switch metric the visit-continuation design used to put ByteTrack out of
scope. Zero id switches means the tracker never CONFUSED two animals; it says
nothing about how often it drops one and starts over. A tracker can score
perfectly on the first and terribly on the second, which is what happened here.

Run: python tests/bench_track_fragmentation.py
Needs the harvest at ~/Desktop/monkey-harvest-2026-08-13 (skips cleanly if absent).
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

HARVEST = Path.home() / "Desktop/monkey-harvest-2026-08-13"
CONSENSUS = HARVEST / "consensus.json"

# Production per-camera throughput (deployment measurements: "6.5-7.0 FPS per camera").
# The whole point of the bench is converting frames <-> seconds honestly.
PROD_FPS = 6.5

# track_buffer values to sweep, in FRAMES at the tracker's assumed 30 FPS.
# 30 is the stock default; the rest are candidate settings.
SWEEP = [30, 40, 46, 50, 55, 60, 135]

_FRAME_RE = re.compile(r"frame(\d+)")


def load_scenes() -> dict[str, list[tuple[int, list[list[float]]]]]:
    """{scene: [(frame_index, [[x1,y1,x2,y2,conf], ...]), ...]} sorted by index."""
    raw = json.load(open(CONSENSUS))
    scenes: dict[str, list[tuple[int, list[list[float]]]]] = defaultdict(list)
    for key, val in raw.items():
        parts = Path(key).parts
        scene, fname = parts[-2], parts[-1]
        m = _FRAME_RE.search(fname)
        if not m:
            continue
        boxes = [
            [*c["box"], float(c.get("maxconf", 0.5))]
            for c in (val.get("clusters") or [])
        ]
        scenes[scene].append((int(m.group(1)), boxes))
    for frames in scenes.values():
        frames.sort(key=lambda p: p[0])
    return scenes


def stride_of(frames: list[tuple[int, list]]) -> int:
    """On-disk subsample stride. The harvest saved 1-in-1, 1-in-2 or 1-in-4.

    The replay therefore sees FEWER frames than production did over the same
    wall-clock, so a production track_buffer of N frames corresponds to N/stride
    replay frames. Scaling here rather than fabricating the missing frames keeps
    the comparison unbiased — inventing empty frames would invent dropouts.
    """
    if len(frames) < 2:
        return 1
    diffs = [b[0] - a[0] for a, b in zip(frames, frames[1:])]
    return max(1, int(statistics.median(diffs)))


def replay(frames: list[tuple[int, list]], track_buffer: int) -> tuple[int, int, int]:
    """Run BYTETracker over one scene.

    Returns (distinct_ids, peak_concurrent, teleports) where a *teleport* is an
    update in which an existing id jumps to a box sharing no overlap with where
    that id last was — the shape a WRONG re-association takes. Raising
    track_buffer keeps lost tracks alive longer, so teleports are the cost side
    of the trade and are reported alongside the benefit.
    """
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    cfg = dict(YAML.load(check_yaml("bytetrack.yaml")))
    cfg["track_buffer"] = track_buffer
    tracker = BYTETracker(args=IterableSimpleNamespace(**cfg), frame_rate=30)

    seen: set[int] = set()
    peak = 0
    teleports = 0
    last_box: dict[int, tuple[float, float, float, float]] = {}

    for _, boxes in frames:
        peak = max(peak, len(boxes))
        if boxes:
            arr = np.array(boxes, dtype=np.float32)
            dets = np.concatenate(
                [arr[:, :4], arr[:, 4:5], np.zeros((len(arr), 1), np.float32)], axis=1
            )
        else:
            dets = np.zeros((0, 6), dtype=np.float32)

        out = tracker.update(_Dets(dets))
        for row in out:
            x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
            box = (x1, y1, x2, y2)
            if tid in last_box and _iou(last_box[tid], box) == 0.0:
                teleports += 1
            last_box[tid] = box
            seen.add(tid)

    return len(seen), peak, teleports


def _Dets(dets: np.ndarray):
    """Wrap (N, 6) [x1,y1,x2,y2,conf,cls] in production's own tracker adapter.

    Imported from services.tiling rather than reimplemented, so the bench and
    the tiled production path cannot drift apart in how they feed BYTETracker.
    """
    from services.tiling import _Detections

    if len(dets):
        xywh = np.stack(
            [
                (dets[:, 0] + dets[:, 2]) / 2,
                (dets[:, 1] + dets[:, 3]) / 2,
                dets[:, 2] - dets[:, 0],
                dets[:, 3] - dets[:, 1],
            ],
            axis=1,
        ).astype(np.float32)
    else:
        xywh = np.zeros((0, 4), np.float32)
    return _Detections(xywh, dets[:, 4], dets[:, 5])


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    if not CONSENSUS.exists():
        print(f"  SKIP  harvest not present ({CONSENSUS})")
        return 0

    scenes = load_scenes()
    usable = {s: f for s, f in scenes.items() if any(b for _, b in f)}
    print(f"Replaying {len(usable)} scenes from the 2026-08-13 harvest\n")

    print(f"{'track_buffer':>13} {'≈seconds':>9} {'median frag':>12} "
          f"{'mean frag':>10} {'scenes >1.5':>12} {'teleports':>10}")
    print("-" * 72)

    results = {}
    for tb in SWEEP:
        ratios, teleports = [], 0
        for _scene, frames in usable.items():
            eff = max(1, round(tb / stride_of(frames)))
            ids, peak, tp = replay(frames, eff)
            teleports += tp
            if peak > 0:
                ratios.append(ids / peak)
        results[tb] = (statistics.median(ratios), sum(ratios) / len(ratios),
                       sum(1 for r in ratios if r > 1.5), teleports)
        med, mean, bad, tp = results[tb]
        print(f"{tb:>13} {tb / PROD_FPS:>8.1f}s {med:>12.2f} {mean:>10.2f} "
              f"{bad:>12} {tp:>10}")

    base = results[SWEEP[0]][0]
    best = min(results.items(), key=lambda kv: (kv[1][0], kv[1][3]))
    print(f"\nStock (track_buffer=30, ≈{30 / PROD_FPS:.1f}s) median fragmentation: {base:.2f}")
    print(f"Best swept: track_buffer={best[0]} (≈{best[0] / PROD_FPS:.1f}s) "
          f"median {best[1][0]:.2f}, teleports {best[1][3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
