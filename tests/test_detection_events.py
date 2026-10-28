"""Detection rows grouped back into visits, and companion boxes on the still.

No model, no camera, no database — these drive the real grouping helpers from
main.py and the real _TrackState geometry from the detector against rows built
by hand plus the actual 2026-08-13 harvest.

Why this exists: on 2026-08-13 a troop of six macaques on a wall produced ~12
separate detection rows, each annotated with a single box. Read one row at a
time it looked like the detector found one monkey and missed five. Nothing was
missed — 46 of the 47 scenes that looked short were fully covered once every
row within 20 s was counted together. Two changes came out of that, and this
file guards both.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))


import main  # noqa: E402
from models import Detection  # noqa: E402
from services.detector import (  # noqa: E402
    DetectorWorker, _TrackState, _COMPANION_MAX_FRAME_GAP,
)

FAILS: list[str] = []
T0 = datetime(2026, 8, 13, 18, 0, 0)


def check(cond: bool, label: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILS.append(label)


def row(det_id: int, *, secs: float, track: int, box=(100.0, 100.0, 60.0, 60.0),
        conf: float = 0.8, camera: str = "cam") -> Detection:
    x, y, w, h = box
    return Detection(
        id=det_id, camera_id=camera, timestamp=T0 + timedelta(seconds=secs),
        confidence=conf, bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
        image_path=f"{det_id}.jpg", annotated_path=f"{det_id}_annot.jpg",
        crop_path=f"{det_id}_crop.jpg", track_id=track,
        trail_path=f"trail_{det_id}.gif",
    )


def group(rows: list[Detection], gap: float = 20.0) -> list[list[Detection]]:
    """The same sequential walk the endpoint does, isolated from FastAPI."""
    out: list[list[Detection]] = []
    bucket: list[Detection] = []
    for det in sorted(rows, key=lambda d: d.timestamp):
        if bucket and (det.timestamp - bucket[-1].timestamp).total_seconds() > gap:
            out.append(bucket)
            bucket = []
        bucket.append(det)
    if bucket:
        out.append(bucket)
    return out


print("Grouping rows into visits")
rows = [row(i, secs=i * 3, track=i) for i in range(6)]
check(len(group(rows)) == 1, "rows a few seconds apart are one visit — 6 rows → 1 event")

rows = [row(0, secs=0, track=0), row(1, secs=21, track=1)]
check(len(group(rows)) == 2, "a gap wider than the window splits the visit")

rows = [row(0, secs=0, track=0), row(1, secs=20, track=1)]
check(len(group(rows)) == 1, "a gap exactly at the window still counts as one visit")

# The chain matters: each row is compared to the PREVIOUS row, not to the first,
# so a troop that lingers stays one event however long it stays.
rows = [row(i, secs=i * 19, track=i) for i in range(10)]
check(len(group(rows)) == 1,
      "a long visit chains — 10 rows spanning %ds stay one event" % (9 * 19))

ev = main._build_event("cam", group([row(i, secs=i * 2, track=i) for i in range(4)])[0])
check(ev.detection_count == 4, "event counts its rows — %d" % ev.detection_count)
check(ev.distinct_tracks == 4, "event counts distinct tracks — %d" % ev.distinct_tracks)
check(ev.duration_seconds == 6.0, "event duration spans first to last — %.1fs" % ev.duration_seconds)
check(ev.cover.id == 0 or ev.max_confidence == 0.8, "cover row is the most confident one")

# The `_estimate_animals` block that stood here is gone with the function it
# tested. It de-duplicated track ids by comparing one row's PEAK-frame box
# against another row's ARRIVAL timestamp — two different moments read as one —
# and no threshold could fix that, because the row simply does not record when
# its animal left. `_peak_concurrent` replaces it by counting tracks present in
# the SAME FRAME, which is a measurement; its tests are further down.

print("\nCompanion boxes on the annotated still")
subject = _TrackState(track_id=1, camera_id="cam", frame_idx=100)
subject.update((110, 110), 0.9, (100.0, 100.0, 60.0, 60.0), 100)

present = _TrackState(track_id=2, camera_id="cam", frame_idx=100)
present.update((510, 110), 0.8, (500.0, 100.0, 60.0, 60.0), 100)
check(present.bbox_near(100, _COMPANION_MAX_FRAME_GAP) == (500.0, 100.0, 60.0, 60.0),
      "a companion seen in the same frame is drawn")

near = _TrackState(track_id=3, camera_id="cam", frame_idx=98)
near.update((510, 110), 0.8, (500.0, 100.0, 60.0, 60.0), 98)
check(near.bbox_near(100, _COMPANION_MAX_FRAME_GAP) is not None,
      "a companion seen %d frames earlier is still drawn" % 2)

gone = _TrackState(track_id=4, camera_id="cam", frame_idx=50)
gone.update((510, 110), 0.8, (500.0, 100.0, 60.0, 60.0), 50)
check(gone.bbox_near(100, _COMPANION_MAX_FRAME_GAP) is None,
      "a companion that had already left is NOT drawn at its stale position")
check(gone.bbox_at(100) is not None,
      "...while bbox_at still returns it, because the GIF trail wants the whole clip")
check(_TrackState(track_id=5, camera_id="cam", frame_idx=0).bbox_near(0, 3) is None,
      "a track with no stored boxes yields nothing, not an index error")

print("\nThe tracker must give up before we do")
# Whichever layer retires a track first decides who fragments. ByteTrack's
# track_buffer is a FRAME count and TRACK_LOST_SECONDS is wall clock, so a
# throughput change can silently invert them — and inverted, we write the row
# at 9 s while ByteTrack still holds the id, then build a fresh track when the
# animal returns under that same id. That is a duplicate row the continuation
# merge cannot reach, because the first one is already in the database.
from ultralytics.utils import YAML  # noqa: E402
from ultralytics.utils.checks import check_yaml  # noqa: E402

import config  # noqa: E402

_buffer_frames = YAML.load(check_yaml(config.TRACKER_YAML))["track_buffer"]
_buffer_seconds = _buffer_frames / config.TRACKER_NOMINAL_FPS
print("    track_buffer=%d frames = %.1fs at %.2f FPS; TRACK_LOST_SECONDS=%.1fs"
      % (_buffer_frames, _buffer_seconds, config.TRACKER_NOMINAL_FPS,
         config.TRACK_LOST_SECONDS))
check(_buffer_seconds < config.TRACK_LOST_SECONDS,
      "the tracker retires an id before we finalise its row (%.1fs < %.1fs)"
      % (_buffer_seconds, config.TRACK_LOST_SECONDS))
check(_buffer_frames > 30,
      "and it is longer than ultralytics' stock 30 frames, which was only 4.4s here")

print("\nProducer-side continuation merge")
# The read side can only group rows it was given. On 2026-08-18 場域丙 filed
# #11320 and #11321 eleven seconds apart, tracks 5090 and 5091 — one macaque
# that walked, was missed for a few seconds, and came back climbing. The row
# list is a plain query (main.list_detections), so a duplicate that reaches the
# DB is a duplicate a user sees. It has to not be written in the first place.


def seg(track_id, *, first_frame, last_frame, born, died, box, conf=0.8):
    """A finished track segment with its wall clock stated outright."""
    t = _TrackState(track_id=track_id, camera_id="cam", frame_idx=first_frame)
    for fi in range(first_frame, last_frame + 1):
        t.update((int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)), conf, box, fi)
    t.first_seen_ts, t.last_seen_ts = born, died
    return t


def adopt(lost, actives, claimed=None):
    """Drive the real method; it only needs camera_id off the worker."""
    return DetectorWorker._adopt_continuation(
        SimpleNamespace(camera_id="cam"), lost, {a.track_id: a for a in actives},
        set() if claimed is None else claimed,
    )


PERCH = (620.0, 350.0, 44.0, 48.0)
SHIFTED = (626.0, 347.0, 43.0, 49.0)   # same perch, re-acquired a few px over
FAR = (120.0, 900.0, 44.0, 48.0)       # other end of the frame

lost = seg(5090, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH)
live = seg(5091, first_frame=160, last_frame=170, born=1018.0, died=1019.0, box=SHIFTED)
check(adopt(lost, [live]) is True,
      "the 場域丙 pair: re-acquired on the same perch 11 s later is ONE row, not two")
check(live.first_seen == lost.first_seen,
      "...and the surviving row is stamped when the animal arrived, not when it was re-acquired")
check(live.min_frame_idx == 100 and live.frame_indices[0] == 100,
      "...with the earlier segment's trail prepended, so the clip shows the whole visit")

# The gate and the cascade verifier both read the PEAK, so a strong first
# segment followed by a weak second must not lose the detection outright.
lost = seg(1, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH, conf=0.94)
live = seg(2, first_frame=160, last_frame=170, born=1010.0, died=1011.0, box=SHIFTED, conf=0.31)
adopt(lost, [live])
check(live.best_conf == 0.94, "the merged track keeps the better evidence — %.2f" % live.best_conf)

# Concurrent animals must survive as separate rows; that is what _group_overlapping
# is for, and merging them would delete a monkey from the count.
lost = seg(1, first_frame=100, last_frame=170, born=1000.0, died=1010.0, box=PERCH)
live = seg(2, first_frame=150, last_frame=180, born=1005.0, died=1011.0, box=SHIFTED)
check(adopt(lost, [live]) is False,
      "two tracks alive at the same time stay two rows — a troop is not collapsed")

lost = seg(1, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH)
live = seg(2, first_frame=160, last_frame=170, born=1012.0, died=1013.0, box=FAR)
check(adopt(lost, [live]) is False,
      "a track that reappears somewhere else is a different animal, not a continuation")

lost = seg(1, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH)
live = seg(2, first_frame=160, last_frame=170, born=1040.0, died=1041.0, box=SHIFTED)
check(adopt(lost, [live]) is False,
      "same perch half a minute later is a new visit — the window is bounded")

# Two animals converging on one perch: at most one of them continues into it.
claimed: set[int] = set()
a1 = seg(1, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH)
a2 = seg(2, first_frame=100, last_frame=141, born=1000.0, died=1008.0, box=SHIFTED)
live = seg(3, first_frame=160, last_frame=170, born=1012.0, died=1013.0, box=PERCH)
taken = [adopt(a1, [live], claimed), adopt(a2, [live], claimed)]
check(taken.count(True) == 1,
      "one live track can only continue ONE lost track — the second stays its own row")


# ── same instant, box on box: one animal wearing two ids ────────────────────
# The class _adopt_continuation structurally cannot reach, because it is
# forward-only and skips overlaps. Measured on 2026-08-27 across 669 production
# rows: 44 such pairs. #15157/#15158 (t7467/t7466) were 0.2 s apart at IoU 0.79.

def dedupe(group):
    """Drive the real method; it needs camera_id and the two static helpers."""
    worker = SimpleNamespace(
        camera_id="cam",
        _shared_frame_iou=DetectorWorker._shared_frame_iou,
        _absorb_duplicate=DetectorWorker._absorb_duplicate,
    )
    return DetectorWorker._merge_duplicate_tracks(worker, group)


twin_a = seg(7466, first_frame=100, last_frame=140, born=1000.0, died=1007.0,
             box=PERCH, conf=0.92)
twin_b = seg(7467, first_frame=100, last_frame=140, born=1000.0, died=1007.0,
             box=SHIFTED, conf=0.91)
merged = dedupe([twin_a, twin_b])
check(len(merged) == 1,
      "box on box in the SAME frames is one animal — %d row(s)" % len(merged))
check(merged[0].track_id == 7466,
      "the higher-confidence track survives — kept #%d" % merged[0].track_id)
check(merged[0].best_conf == 0.92, "the survivor keeps the better peak")

# Two animals side by side must NOT be collapsed: that would delete a monkey.
pair = dedupe([seg(1, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH),
               seg(2, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=FAR)])
check(len(pair) == 2,
      "two tracks in the same frames but apart stay two rows — a troop is not collapsed")

# Overlapping in a single frame is a coincidence, not evidence.
brief_a = seg(1, first_frame=100, last_frame=100, born=1000.0, died=1000.5, box=PERCH)
brief_b = seg(2, first_frame=100, last_frame=130, born=1000.0, died=1005.0, box=SHIFTED)
check(len(dedupe([brief_a, brief_b])) == 2,
      "one shared frame is below TRACK_DUPLICATE_MIN_FRAMES — not merged on a coincidence")

# An OCCLUSION is the dangerous look-alike: two macaques cross, and for the few
# frames one passes in front of the other their boxes are mutually contained.
# Frame COUNT cannot tell that from a duplicate; the fraction of the shorter
# track can, because a duplicate co-exists for most of it and a crossing does not.
crossing_a = seg(1, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH)
crossing_b = seg(2, first_frame=136, last_frame=176, born=1006.0, died=1013.0, box=SHIFTED)
iou_c, shared_c, ratio_c = DetectorWorker._shared_frame_iou(crossing_a, crossing_b)
check(shared_c >= config.TRACK_DUPLICATE_MIN_FRAMES and iou_c >= config.TRACK_DUPLICATE_IOU,
      "the crossing DOES clear the frame count and the IoU bar (%d frames, IoU %.2f)"
      % (shared_c, iou_c))
check(len(dedupe([crossing_a, crossing_b])) == 2,
      "...but %.0f%% of the shorter track is below TRACK_DUPLICATE_MIN_OVERLAP — "
      "two crossing animals stay two rows" % (ratio_c * 100))

# Equal peaks must not let ByteTrack's hand-over order pick the surviving id.
tie_fwd = dedupe([seg(9, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH, conf=0.9),
                  seg(4, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=SHIFTED, conf=0.9)])
tie_rev = dedupe([seg(4, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=SHIFTED, conf=0.9),
                  seg(9, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH, conf=0.9)])
check([t.track_id for t in tie_fwd] == [t.track_id for t in tie_rev] == [4],
      "a best_conf TIE is broken by track_id, not by hand-over order")

# Tracks that never share a frame are the continuation case, not this one.
before = seg(1, first_frame=100, last_frame=120, born=1000.0, died=1003.0, box=PERCH)
after = seg(2, first_frame=140, last_frame=160, born=1006.0, died=1009.0, box=SHIFTED)
check(len(dedupe([before, after])) == 2,
      "no shared frames is _adopt_continuation's job — this merge does not guess across gaps")

# The survivor has to carry BOTH histories, in frame order, or the clip draws
# half a visit.
wide_a = seg(1, first_frame=100, last_frame=120, born=1000.0, died=1003.0, box=PERCH, conf=0.9)
wide_b = seg(2, first_frame=110, last_frame=150, born=1001.0, died=1006.0, box=SHIFTED, conf=0.5)
out = dedupe([wide_a, wide_b])[0]
check(out.frame_indices == sorted(out.frame_indices), "the merged trail is in frame order")
check(out.min_frame_idx == 100 and out.max_frame_idx == 150,
      "the merged track spans both segments (%d-%d)" % (out.min_frame_idx, out.max_frame_idx))
check(len(out.frame_indices) == len(set(out.frame_indices)),
      "a shared frame contributes ONE box, not two stacked on each other")
check(len(out.frame_indices) == len(out.centers) == len(out.frame_bboxes),
      "the three parallel lists stay the same length")

# Which track absorbs which must not depend on the order ByteTrack handed them over.
fwd = dedupe([seg(1, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH, conf=0.6),
              seg(2, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=SHIFTED, conf=0.8)])
rev = dedupe([seg(2, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=SHIFTED, conf=0.8),
              seg(1, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH, conf=0.6)])
check([t.track_id for t in fwd] == [t.track_id for t in rev] == [2],
      "the merge is order-independent — both orders keep #2")

# A single track in, a single track out, untouched.
solo = seg(1, first_frame=100, last_frame=140, born=1000.0, died=1007.0, box=PERCH)
check(dedupe([solo]) == [solo], "a group of one is returned as-is")


# ── the retroactive twin: merging rows already in the table ─────────────────

PERCH_ROW = (620.0, 350.0, 44.0, 48.0)
SHIFT_ROW = (626.0, 347.0, 43.0, 49.0)
FAR_ROW = (120.0, 900.0, 44.0, 48.0)

pair = [row(1, secs=0, track=7466, box=PERCH_ROW, conf=0.92),
        row(2, secs=0.2, track=7467, box=SHIFT_ROW, conf=0.91)]
g, = main._duplicate_row_groups(pair)
check(g["keep"] == 1 and g["drop"] == [2],
      "#15157/#15158 shape: 0.2 s apart, box on box — the stronger row survives")
check(g["classes"] == "A",
      "...and it is classed A — the NARROWER inference, not a proof: the row "
      "carries an arrival time and a peak-frame box, so nothing here shows the "
      "two were ever simultaneous")

# A lock is an endorsement of the box, so it outranks raw confidence.
endorsed = [row(1, secs=0, track=1, box=PERCH_ROW, conf=0.60),
            row(2, secs=0.5, track=2, box=SHIFT_ROW, conf=0.99)]
endorsed[0].locked = True
g, = main._duplicate_row_groups(endorsed)
check(g["keep"] == 1 and g["locked"] is True,
      "the endorsed row survives even at lower confidence — the verified box is the one kept")

# The survivor is stamped when the animal arrived.
late = [row(1, secs=0, track=1, box=PERCH_ROW, conf=0.5),
        row(2, secs=8, track=2, box=SHIFT_ROW, conf=0.9)]
g, = main._duplicate_row_groups(late, same_instant_only=False)
check(g["keep"] == 2 and g["retimed"] is True and g["timestamp"] == late[0].timestamp,
      "the survivor takes the earliest timestamp, so the visit does not start late")

check(main._duplicate_row_groups(
          [row(1, secs=0, track=1, box=PERCH_ROW), row(2, secs=1, track=2, box=FAR_ROW)]) == [],
      "two animals at opposite ends of the frame are never merged")

check(main._duplicate_row_groups(
          [row(1, secs=0, track=1, box=PERCH_ROW, camera="a"),
           row(2, secs=0.2, track=2, box=SHIFT_ROW, camera="b")]) == [],
      "rows from different cameras are never merged")

# The 5-20 s class is the riskier judgement call, so it is OPT-IN: a caller who
# omits the flag on a no-undo delete must get the narrower one.
spaced = [row(1, secs=0, track=1, box=PERCH_ROW, conf=0.9),
          row(2, secs=11, track=2, box=SHIFT_ROW, conf=0.8)]
check(main._duplicate_row_groups(spaced) == []
      and len(main._duplicate_row_groups(spaced, same_instant_only=False)) == 1,
      "the 5-20 s class is off by default and only merges when asked for")

# Transitive chaining must not walk a whole afternoon into one row. Every link
# here is inside the window; the SPAN is what stops it.
perch_sitter = [row(i, secs=i * 8, track=i, box=PERCH_ROW, conf=0.8) for i in range(1, 21)]
chained = main._duplicate_row_groups(perch_sitter, same_instant_only=False)
check(all(g["span_seconds"] <= main._MERGE_MAX_SPAN_SECONDS for g in chained),
      "no group spans more than the ceiling — widest is %.0fs"
      % max([g["span_seconds"] for g in chained], default=0))
check(sum(len(g["drop"]) for g in chained) < len(perch_sitter) - 1,
      "a 160 s perch-sitter does not collapse to one row (%d of %d rows dropped)"
      % (sum(len(g["drop"]) for g in chained), len(perch_sitter)))

# Deleting an endorsed row spends a human's review, so it is named, never counted.
both_locked = [row(1, secs=0, track=1, box=PERCH_ROW, conf=0.99),
               row(2, secs=0.4, track=2, box=SHIFT_ROW, conf=0.97)]
both_locked[0].locked = both_locked[1].locked = True
g, = main._duplicate_row_groups(both_locked)
check(g["dropped_locked"] == [2],
      "the endorsed row that does NOT survive is named in dropped_locked")

check(main._duplicate_row_groups(
          [row(1, secs=0, track=1, box=PERCH_ROW), row(2, secs=45, track=2, box=SHIFT_ROW)]) == [],
      "the same perch a minute later is a new visit — the window is bounded")

# Three fragments of one animal chain into ONE row, not two pairs.
chain = [row(1, secs=0, track=1, box=PERCH_ROW, conf=0.5),
         row(2, secs=6, track=2, box=SHIFT_ROW, conf=0.9),
         row(3, secs=12, track=3, box=PERCH_ROW, conf=0.7)]
g, = main._duplicate_row_groups(chain, same_instant_only=False)
check(g["keep"] == 2 and sorted(g["drop"]) == [1, 3],
      "a three-fragment chain collapses to one row, not two overlapping pairs")


# ── pointing rows at the rebuilt clip ──────────────────────────────────────
# 406 of the clips on record are GIFs, which the review pane renders through an
# <img>: no scrubber, no pause. They were rebuilt from their archived raw frames
# as {stem}_trail.mp4. Switching is only safe while two things hold — the
# rebuild is really in storage, and the original is left alone.

def gifrow(det_id, *, trail="cam_tracks1_trail.gif", secs=0):
    r = row(det_id, secs=secs, track=det_id)
    r.trail_path = trail
    return r


present = lambda name: True          # noqa: E731 — every rebuild landed
absent = lambda name: False          # noqa: E731 — none did

plan, skip = main._retarget_plan([gifrow(1)], present)
check(len(plan) == 1 and plan[0]["to"] == "cam_tracks1_trail.mp4",
      "a GIF row is pointed at the rebuilt MP4 beside it")
check(plan[0]["from"] == "cam_tracks1_trail.gif",
      "...and the manifest names what it came from, so it can be switched back")

plan, skip = main._retarget_plan([gifrow(1)], absent)
check(not plan and skip["rebuild_missing"] == 1,
      "no rebuild in storage means no switch — a row is never pointed at a missing file")

plan, skip = main._retarget_plan([gifrow(1, trail="cam_tracks1_trail.mp4")], present)
check(not plan and skip["already_mp4"] == 1,
      "a clip the detector already wrote as MP4 is left alone")

noclip = row(2, secs=0, track=2); noclip.trail_path = ""
plan, skip = main._retarget_plan([noclip], present)
check(not plan and skip["no_clip"] == 1, "a row with no clip is skipped, not crashed on")

# A clip shared by several rows is ONE blob: check it once, switch them all.
calls = []
def counting(name):
    calls.append(name)
    return True

shared = [gifrow(i, trail="cam_tracks1+2+3_trail.gif", secs=i) for i in range(1, 6)]
plan, skip = main._retarget_plan(shared, counting)
check(len(plan) == 5, "every row on a shared clip is switched (%d)" % len(plan))
check(len(calls) == 1,
      "...on ONE existence check, not one per row (%d calls)" % len(calls))

# The stem must survive, or _raw_frames_name stops finding the archive.
plan, _ = main._retarget_plan([gifrow(1, trail="cam_tracks9_20260422_1911_ab12_trail.gif")], present)
check(plan[0]["to"] == "cam_tracks9_20260422_1911_ab12_trail.mp4",
      "the stem is untouched, so the raw-frames ZIP still resolves from it")


# ── one visit, one clip: the hold queue ────────────────────────────────────
# 341 visits produced 570 clips (median 1.7, worst 7). A group finalises when
# its tracks are lost, but the visit is not over — so a finalised group is held
# and whatever finalises behind it on the same camera is merged in.

import time  # noqa: E402

from services.detector import _VisitQueue, _PendingVisit, _FrameBuffer  # noqa: E402
import services.detector as det_mod  # noqa: E402


def buf(n, start=0):
    b = _FrameBuffer(max_size=max(1, n))
    for i in range(start, start + n):
        b._buf[i] = (b"jpeg", float(i))
    return b


def surv(track_id):
    return [(seg(track_id, first_frame=0, last_frame=4, born=0.0, died=1.0,
                 box=PERCH), None)]


q = _VisitQueue()
rendered: list = []
check(q.offer("cam", surv(1), [], buf(10), rendered.append) is True,
      "a finalised group is taken by the queue instead of rendering at once")
check(not rendered, "...and nothing is drawn while the visit may still continue")
check(q.offer("cam", surv(2), [], buf(10, start=10), rendered.append) is True,
      "a second group on the same camera is merged into the open visit")
check(q.merged == 1 and q.held == 1, "one visit held, one merge — not two visits")

visit = q._due(time.monotonic() + config.VISIT_HOLD_SECONDS + 1)[0]
check(len(visit.survivors) == 2 and visit.groups == 2,
      "the rendered visit carries BOTH groups' tracks (%d)" % len(visit.survivors))
check(len(visit.frames) == 20,
      "...and both groups' frames, so the clip shows the whole visit (%d)" % len(visit.frames))
check(not q._due(time.monotonic() + 999), "a rendered visit is not rendered twice")

# A different camera is a different visit; merging them would invent an event.
q2 = _VisitQueue()
q2.offer("cam-a", surv(1), [], buf(5), rendered.append)
q2.offer("cam-b", surv(2), [], buf(5), rendered.append)
check(len(q2._pending) == 2, "two cameras hold two separate visits")

# A troop that keeps producing groups must still land.
q3 = _VisitQueue()
q3.offer("cam", surv(1), [], buf(5), rendered.append)
opened = q3._pending["cam"].opened_at
for _ in range(50):
    q3.offer("cam", surv(2), [], buf(5), rendered.append)
check(q3._pending["cam"].deadline <= opened + config.VISIT_HOLD_MAX_SECONDS + 1e-6,
      "the deadline is capped — a visit that never ends still gets written")

# Full means render NOW, not grow the heap: the 2026-07-24 OOM was this spend.
q4 = _VisitQueue()
q4.offer("cam-a", surv(1), [], buf(q4._frame_ceiling), rendered.append)
check(q4.offer("cam-b", surv(2), [], buf(50), rendered.append) is False,
      "over the frame ceiling the queue refuses, so the caller renders inline")
check(q4.refused == 1, "...and the refusal is counted, not silent")

# The kill switch has to actually kill it.
saved = det_mod.VISIT_HOLD_SECONDS
det_mod.VISIT_HOLD_SECONDS = 0.0
try:
    check(_VisitQueue().offer("cam", surv(1), [], buf(5), rendered.append) is False,
          "VISIT_HOLD_SECONDS=0 disables the hold entirely — every group renders inline")
finally:
    det_mod.VISIT_HOLD_SECONDS = saved


# ── the visit becomes a stored fact ────────────────────────────────────────
# Until now a visit only existed at read time: /detections/events regrouped
# rows by a 20 s gap on every request, which is why that view is admin-only.
# The producer emits a whole visit at once, so the rows can carry the id.

worker = SimpleNamespace(camera_id="cam", _emit_detection=None)
emitted: list = []
rendered_track_ids: list[list[int]] = []
worker._emit_detection = lambda track, frames, clip, **kw: emitted.append(
    (track.track_id, kw.get("visit_id")))
def render_clip(tracks, *args, **kwargs):
    rendered_track_ids.append([track.track_id for track in tracks])
    return ("clip.mp4", [1000.0 + i * 0.04 for i in range(50)], 25.0)

vis = _PendingVisit("cam",
                    [(seg(1, first_frame=0, last_frame=4, born=0.0, died=1.0, box=PERCH), None),
                     (seg(2, first_frame=0, last_frame=4, born=0.0, died=1.0, box=FAR), None)],
                    [seg(99, first_frame=0, last_frame=4, born=0.0, died=1.0, box=FAR)],
                    buf(4), time.monotonic())
# The generator hands back the clip AND the time base it wrote it on.
det_mod._generate_combined_trail_video = render_clip
DetectorWorker._render_visit(worker, vis)
ids = {v for _, v in emitted}
check(len(emitted) == 2, "both animals of the visit are still their own row (%d)" % len(emitted))
check(rendered_track_ids == [[1, 2]],
      "the clip draws only emitted survivors, not concurrent context (%s)" % rendered_track_ids)
check(len(ids) == 1 and next(iter(ids)),
      "...and every row of one visit carries the SAME visit_id (%s)" % ids)
check(next(iter(ids))[:8].isdigit(),
      "the id sorts by time, so a glance at it says when — %s" % next(iter(ids)))

vis2 = _PendingVisit("cam",
                     [(seg(3, first_frame=0, last_frame=4, born=0.0, died=1.0, box=PERCH), None)],
                     [], buf(4), time.monotonic())
emitted.clear()
DetectorWorker._render_visit(worker, vis2)
check({v for _, v in emitted} != ids, "a separate visit gets a separate id")


# ── stamping history with the visit it belonged to ─────────────────────────
# The grouping is proved off-box (clips sharing JPEG bytes were cut from the
# same instant); this endpoint only validates and applies. What it refuses is
# the interesting part.

class _Assign:
    def __init__(self, visit_id, detection_ids):
        self.visit_id, self.detection_ids = visit_id, detection_ids


def claim(visits):
    """The conflict check, isolated from FastAPI and the DB."""
    claimed, conflicts = {}, []
    for v in visits:
        for i in v.detection_ids:
            if i in claimed and claimed[i] != v.visit_id:
                conflicts.append(i)
            claimed[i] = v.visit_id
    return claimed, conflicts


claimed, conflicts = claim([_Assign("v1", [1, 2, 3]), _Assign("v2", [4])])
check(not conflicts and len(claimed) == 4,
      "rows spread over two visits are accepted (%d rows)" % len(claimed))

_, conflicts = claim([_Assign("v1", [1, 2]), _Assign("v2", [2, 3])])
check(conflicts == [2],
      "one row claimed by two visits is a conflict, not a last-writer-wins (%s)" % conflicts)

# A producer id beats a reconstruction, and `overwrite` must not change that:
# the producer WATCHED that visit happen, this endpoint worked it out after.
PREFIX = main.BACKFILL_VISIT_PREFIX


def would_write(existing, overwrite):
    """The endpoint's precedence rule, isolated."""
    if existing and not existing.startswith(PREFIX):
        return False                      # producer id — never
    if existing and not overwrite:
        return False                      # an earlier reconstruction, left alone
    return True


check(would_write("20260828144151-0f744b", overwrite=False) is False,
      "a producer id is not overwritten")
check(would_write("20260828144151-0f744b", overwrite=True) is False,
      "...and `overwrite` does NOT reach it either — that is the whole invariant")
check(would_write(None, overwrite=False) is True, "a blank row is filled")
check(would_write("bf20260813173556-008274", overwrite=False) is False,
      "an earlier reconstruction is kept unless overwrite is asked for")
check(would_write("bf20260813173556-008274", overwrite=True) is True,
      "...and re-run when it is, so regrouping after recovering more archives works")


# ── a reconstruction may not describe a different video ────────────────────
# The producer carried exactly this defect until 2026-08-28: offsets measured
# on one time base stored beside a duration measured on another, so a row
# claimed second 8.11 of a 4.64 s clip. The backfill writes the same columns
# from re-tracked frames, so it is a second way in for the same mistake — and
# the write boundary is where it gets refused, not a player.

Assign = main.TimelineAssignment


def refused(**kw):
    try:
        Assign(**kw)
        return None
    except Exception as e:
        return str(e)


ok = Assign(id=1, clip_offset_seconds=1.0, clip_end_offset_seconds=2.5,
            clip_duration_seconds=4.64, clip_segments=[[1.0, 1.8], [2.2, 2.5]])
check(ok.clip_duration_seconds == 4.64, "a position inside its own clip is accepted")

check(refused(id=2, clip_offset_seconds=8.11, clip_duration_seconds=4.64),
      "second 8.11 of a 4.64 s clip is REFUSED — the exact shape that made every "
      "seek land on the last frame")
check(refused(id=3, clip_offset_seconds=1.0, clip_end_offset_seconds=9.0,
              clip_duration_seconds=4.64),
      "an end past the video is refused too, not just a start")
check(refused(id=4, clip_offset_seconds=2.0, clip_end_offset_seconds=1.0,
              clip_duration_seconds=4.64),
      "a start later than its end is refused")
check(refused(id=5, clip_duration_seconds=4.64, clip_segments=[[1.0, 9.9]]),
      "a stretch that runs off the end is refused")
check(refused(id=6, clip_offset_seconds=1.0),
      "a position with no duration is refused — there is nothing to be inside of")
check(refused(id=7, clip_offset_seconds=1.0, clip_duration_seconds=0.0),
      "a zero-length clip cannot hold a position")
check(refused(id=8, clip_offset_seconds=-0.5, clip_duration_seconds=4.64),
      "and a negative offset, which would seek a player backwards out of the video")
check(Assign(id=9).clip_duration_seconds is None,
      "a row with nothing to say is still allowed through, and stays empty")

# Repointing a row at a merged clip and placing it on that clip's axis is ONE
# write. A visit spread over thirteen files cannot be read as one timeline, and
# the merge is only safe if the two halves cannot come apart.
moved = Assign(id=11, trail_path="場域甲_visit_merged_trail.mp4",
               clip_offset_seconds=2.0, clip_end_offset_seconds=3.0,
               clip_duration_seconds=9.0)
check(moved.trail_path.endswith(".mp4") and moved.clip_offset_seconds == 2.0,
      "a row can be pointed at another clip and placed on it in the same row")
check(refused(id=12, trail_path="x_trail.mp4", clip_offset_seconds=12.0,
              clip_duration_seconds=9.0),
      "...and the position is still checked against the clip it is moving TO")
check(Assign(id=13, clip_offset_seconds=1.0, clip_duration_seconds=9.0).trail_path is None,
      "leaving trail_path out keeps the row on the clip it already has")


# ── the visit becomes something users may see ──────────────────────────────
# It was admin-only because it was an inference. It is not one any more, and
# the two kinds must not be flattened into looking alike.

def peak_of(tracks):
    return det_mod._peak_concurrent(tracks)

a = seg(1, first_frame=0, last_frame=10, born=0.0, died=1.0, box=PERCH)
b = seg(2, first_frame=0, last_frame=10, born=0.0, died=1.0, box=FAR)
check(peak_of([a, b]) == 2, "two tracks in the same frames are two animals at once")
c = seg(3, first_frame=20, last_frame=30, born=2.0, died=3.0, box=PERCH)
check(peak_of([a, c]) == 1,
      "two tracks that never share a frame were never on screen together (%d)" % peak_of([a, c]))
check(peak_of([]) == 0, "no tracks is zero, not a crash")
check(peak_of([a, b, c]) == 2,
      "the peak is the most AT ONCE, not the total seen (%d)" % peak_of([a, b, c]))

stamped = [row(1, secs=0, track=1), row(2, secs=1, track=2)]
for r in stamped:
    r.visit_id, r.visit_peak = "20260828144151-0f744b", 2
ev = main._build_event("cam", stamped, basis="recorded")
check(ev.event_id == "20260828144151-0f744b",
      "a recorded visit is identified by the id the producer gave it")
check(ev.peak_animals == 2 and ev.basis == "recorded",
      "...and reports the counted peak, marked as recorded")

old_rows = [row(3, secs=0, track=3)]
ev = main._build_event("cam", old_rows, basis="gap")
check(ev.peak_animals is None and ev.basis == "gap",
      "a reconstructed visit has no peak to report and says so — never a substituted guess")


# ── where in the clip the animal actually is ───────────────────────────────
# A row's timestamp is wall clock; a player needs "second 12.4". Until these
# existed every row opened its clip at the beginning — with seven macaques in
# one visit, seven times over.

off = det_mod._clip_offsets
axis_of = det_mod._clip_axis

# A clip is a bounded, constant-rate resampling of the visit. Empty 24-FPS
# playback slots repeat the most recent selected frame, so a bursty RTSP source
# no longer turns a long wall-clock visit into an accelerated clip.
burst = ([1000.0 + i * 0.02 for i in range(50)]          # one dense second
         + [1008.0 + i * 0.02 for i in range(50)])       # then 7 s of silence
burst_frames = [object() for _ in burst]
_, axis, fps = det_mod._resample_trail_frames(burst_frames, burst, 24.0)
check(len(axis) == 242 and fps == 24.0,
      "the clumped axis keeps every frame and writes at 24 fps (%s, %s)" % (len(axis), fps))
check(abs(burst[-1] - burst[0] - 8.98) < 0.01 and abs(len(axis) / fps - 10.0833) < 0.01,
      "...and fills the long gap without accelerating it (%s s)" % (len(axis) / fps))

late = seg(1, first_frame=0, last_frame=10, born=1008.0, died=1008.98, box=PERCH)
r = off(late, axis, fps)
check(r["clip_offset_seconds"] == 8.0,
      "a track arriving 8 s into the wall clock is at second 8.0 OF THE VIDEO (%s)"
      % r["clip_offset_seconds"])
check(r["clip_duration_seconds"] == 10.083,
      "the clip's own length includes the filled playback timeline (%s)"
      % r["clip_duration_seconds"])
check(r["clip_end_offset_seconds"] <= r["clip_duration_seconds"],
      "no offset may point past the video it describes (%s > %s)"
      % (r["clip_end_offset_seconds"], r["clip_duration_seconds"]))

even = [1000.0 + i * 0.04 for i in range(100)]
even_frames = [object() for _ in even]
_, even_axis, even_fps = det_mod._resample_trail_frames(even_frames, even, 24.0)
r = off(seg(2, first_frame=0, last_frame=10, born=1002.0, died=1003.0, box=PERCH),
        even_axis, even_fps)
check(r["clip_offset_seconds"] == 2.083,
      "when frames DO arrive evenly the source mapping remains stable (%s)"
      % r["clip_offset_seconds"])

# The pre-buffer is bounded, a track is not: it can start before the first frame
# the clip kept. A negative offset seeks a player backwards out of the video.
early = seg(3, first_frame=0, last_frame=4, born=990.0, died=1000.5, box=PERCH)
check(off(early, axis, fps)["clip_offset_seconds"] == 0.0,
      "a track older than the clip clamps to its start, never negative")

outlives = seg(4, first_frame=0, last_frame=4, born=1008.9, died=1090.0, box=PERCH)
r = off(outlives, axis, fps)
# The LAST FRAME, not the end of the video: frame 241 of 242 plays at 10.04 s,
# while 10.08 s is where the clip stops. Seeking a player to the end shows it
# nothing, so that is not where a track that is still present belongs.
check(r["clip_end_offset_seconds"] == 10.042,
      "and one outliving it clamps to the last frame, which is where a player "
      "can still see it (%s)" % r["clip_end_offset_seconds"])
check(r["clip_offset_seconds"] <= r["clip_end_offset_seconds"],
      "start never overtakes end after clamping")

check(all(v is None for v in off(late, [], 0.0).values()),
      "no time base yields None, not 0 — a zero would send every player to the "
      "start, which is the behaviour these fields exist to end")

# A frame whose timestamp went missing arrives as 0.0 and would break the
# ordering the axis is searched with.
holes = axis_of([0.0, 0.0, 1000.0, 0.0, 1000.5, 1001.0])
check(holes == [1000.0, 1000.0, 1000.5, 1001.0] and holes == sorted(holes),
      "missing stamps carry the previous frame's, leading ones are dropped %s" % holes)

# ── leaving and coming back has to SHOW ────────────────────────────────────
# A single start→end bar claims continuous presence. The case the tracking work
# spent weeks on is a macaque walking out and returning — paint that as one bar
# and the timeline hides the only thing worth looking at.

seg_fn = det_mod._presence_segments
gone = seg(9, first_frame=0, last_frame=6, born=1000.0, died=1010.0, box=PERCH)
gone.frame_indices = [0, 1, 2, 20, 21, 22]
stamps = {0: 1000.0, 1: 1000.2, 2: 1000.4, 20: 1006.0, 21: 1006.2, 22: 1006.4}
plain = [1000.0 + i * 0.04 for i in range(200)]
plain_frames = [object() for _ in plain]
_, p_axis, p_fps = det_mod._resample_trail_frames(plain_frames, plain, 24.0)
segs = seg_fn(gone, stamps, p_axis, p_fps)
check(len(segs) == 2, "a track with a 5.6 s hole is TWO stretches, not one (%s)" % segs)
check(segs[0] == [0.0, 0.417] and segs[1] == [6.25, 6.667],
      "...and each stretch spans only when it was actually seen %s" % segs)

tight = seg(10, first_frame=0, last_frame=3, born=1000.0, died=1001.0, box=PERCH)
tight.frame_indices = [0, 1, 2, 3]
segs = seg_fn(tight, {0: 1000.0, 1: 1000.2, 2: 1000.4, 3: 1000.6}, p_axis, p_fps)
check(len(segs) == 1,
      "one missed frame is not a departure — under SEGMENT_GAP_SECONDS stays one (%s)" % segs)

check(seg_fn(gone, {}, p_axis, p_fps) is None,
      "frames without timestamps yield None, so the UI falls back to the plain "
      "bar rather than inventing a shape")

full = off(gone, p_axis, p_fps, stamps)
check(full["clip_segments"] == [[0.0, 0.417], [6.25, 6.667]],
      "the offsets carry the stretches through with them")
check(off(gone, [], 0.0)["clip_segments"] is None, "no time base, no stretches")


# ── an animal that was MOVING when the tracker lost it ─────────────────────
# Ground truth from the user, 2026-08-28 07:06:28 on 場域甲: one macaque crosses
# the frame left to right and the tracker hands out EIGHT ids. Centre x goes 15
# to 1628 while y barely moves. Overlap cannot stitch that — measured IoU
# between consecutive fragments is 0.04-0.21, nowhere near the 0.5 the
# stationary rule needs, because the animal moves nearly its own box width in
# the gap. Distance from where its own velocity says it went does not collapse
# like that: the same fragments sit within 0.30-0.82 box widths.

motion = det_mod._continues_by_motion
vel = det_mod._tail_velocity


def running(tid, *, x0, x1, y, w, born, died, frames):
    """A track crossing the frame at constant speed."""
    t = seg(tid, first_frame=frames[0], last_frame=frames[-1], born=born, died=died, box=PERCH)
    n = len(frames)
    # (x, y, w, h) — the convention _bbox_iou documents and the tracker uses.
    t.frame_bboxes = [
        (x0 + (x1 - x0) * i / max(n - 1, 1), y, w, w) for i in range(n)
    ]
    t.frame_indices = list(frames)
    return t


# 500 px/s to the right, lost, and re-acquired 0.2 s later exactly where that
# speed would have carried it.
a = running(1, x0=0, x1=500, y=900, w=120, born=1000.0, died=1001.0, frames=range(0, 26))
vx, vy = vel(a)
check(abs(vx - 500) < 60 and abs(vy) < 1,
      "the tail velocity of a track crossing at 500 px/s reads back (%.0f, %.0f)" % (vx, vy))

b = running(2, x0=600, x1=900, y=900, w=120, born=1001.2, died=1002.0, frames=range(30, 51))
check(motion(a, b, 0.2) < 1.0,
      "a fragment landing where the animal was heading is a continuation (%.2f 框寬)"
      % motion(a, b, 0.2))
check(det_mod._bbox_iou(a.frame_bboxes[-1], b.frame_bboxes[0]) < 0.5,
      "...and the overlap rule alone would have missed it, which is the bug")

# A second animal entering from somewhere else is not a continuation.
c = running(3, x0=200, x1=400, y=200, w=120, born=1001.2, died=1002.0, frames=range(30, 51))
check(motion(a, c, 0.2) > 1.0,
      "a fragment somewhere else entirely is refused (%.2f 框寬)" % motion(a, c, 0.2))

# The window is short on purpose: over seconds, another animal could have run
# the same beam, and this rule cannot tell them apart.
check(motion(a, b, 5.0) == float("inf"),
      "and a long gap is refused outright, whatever the arithmetic says")

# Sitting still is the case the old rule already handled, and it still works:
# zero velocity means the prediction is the last box itself.
still = running(4, x0=300, x1=300, y=900, w=120, born=1000.0, died=1001.0, frames=range(0, 26))
after = running(5, x0=305, x1=305, y=900, w=120, born=1001.1, died=1002.0, frames=range(30, 51))
check(motion(still, after, 0.1) < 0.2,
      "a stationary animal re-acquired in place is still a continuation (%.2f)"
      % motion(still, after, 0.1))

print("\nAgainst the real 2026-08-13 harvest")
manifest = Path.home() / "Desktop/monkey-harvest-2026-08-13/manifest.json"
if not manifest.exists():
    print("  SKIP  harvest manifest not present (%s)" % manifest)
else:
    data = json.load(open(manifest))
    real = [
        row(d["id"], secs=(datetime.fromisoformat(d["timestamp"]) - T0).total_seconds(),
            track=d["track_id"], conf=d["confidence"],
            box=(d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"]),
            camera=d["camera_id"])
        for d in data["detections"]
    ]
    per_cam: dict[str, list[Detection]] = {}
    for d in real:
        per_cam.setdefault(d.camera_id, []).append(d)
    events = [main._build_event(c, b, basis="gap") for c, rs in per_cam.items() for b in group(rs)]
    biggest = max(events, key=lambda e: e.detection_count)
    print("    %d rows → %d events; biggest holds %d rows / %d tracks"
          % (len(real), len(events), biggest.detection_count, biggest.distinct_tracks))
    check(len(events) < len(real),
          "grouping actually collapses the log — %d rows → %d events" % (len(real), len(events)))
    check(biggest.detection_count >= 5,
          "the troop-on-a-wall visit is reunited — biggest event holds %d rows"
          % biggest.detection_count)
    check(all(e.peak_animals is None for e in events),
          "harvest rows predate visit_peak, so the count is absent rather than guessed")
    check(all(e.started_at <= e.ended_at for e in events), "every event starts before it ends")
    print("    rows %d across %d events" % (len(real), len(events)))

print("\n" + ("ALL PASS" if not FAILS else "FAILED: %d\n  - %s" % (len(FAILS), "\n  - ".join(FAILS))))
sys.exit(1 if FAILS else 0)
