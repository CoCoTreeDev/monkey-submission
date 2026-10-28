"""Live-stream check on the continuation merge: does it fire, and does it misfire?

Offline replay proved the mechanism on real frames (rows 2 -> 1 for dropouts of
6-8 s, unchanged past 9 s). What replay cannot show is whether the merge picks
the WRONG partner against real traffic, where several animals move at once and
the detector fires on foliage. That is what this measures.

Runs ALONGSIDE production on purpose — it opens its own `-trackverify` streams
and never touches the running detector, exactly like tests/bench_edge_delivery.py.
Nothing is written: no DB row, no blob, no Discord. `_finalise_group` is stubbed
and only records what WOULD have been emitted.

Every merge is printed with the numbers needed to judge it by hand — the gap,
the IoU, both boxes, both peak confidences — and its two crops are saved under
~/track-verify-out/ for eyeballing afterwards.

Dispatch:
  gh workflow run edge-bench.yml --ref <branch> \
    -f script=tests/bench_track_live.py -f args="--minutes 14 --cameras 2"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Shadow the DEPLOYED backend with this checkout's, so the tracker yaml and the
# merge under test are the ones from the branch being verified — while .env,
# the venv and the model still come from the node itself.
WS = Path(os.getenv("GITHUB_WORKSPACE", Path(__file__).resolve().parent.parent))
DEPLOYED = Path.home() / "Monkey-Detection" / "backend"
sys.path.insert(0, str(DEPLOYED))
sys.path.insert(0, str(WS / "backend"))
os.chdir(DEPLOYED)

import cv2  # noqa: E402
import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env")

import config  # noqa: E402
from services.detector import DetectorWorker, _FrameBuffer  # noqa: E402
from services.stream import stream_manager  # noqa: E402

OUT = Path.home() / "track-verify-out"


class Probe:
    """The real tracking methods; emit replaced by a recorder."""

    _update_track = DetectorWorker._update_track
    _adopt_continuation = DetectorWorker._adopt_continuation
    _finalise_batch = DetectorWorker._finalise_batch
    _group_overlapping = staticmethod(DetectorWorker._group_overlapping)

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.rows = 0
        self.merges: list[dict] = []
        # Tracks EVER created, so "0 rows" can be told apart from "0 animals".
        # A run with tracks but no rows means the confidence gate rejected them;
        # a run with neither means nothing walked past the camera, and says
        # nothing at all about the merge.
        self.tracks: set[int] = set()
        self.peak_conf = 0.0
        self._buf: _FrameBuffer | None = None

    def _finalise_group(self, group, still_active, frame_buf):
        self.rows += len(group)
        for t in group:
            print(f"    ROW {self.camera_id} track #{t.track_id} "
                  f"peak={t.best_conf:.2f} frames={len(t.frame_indices)}", flush=True)


def instrument(probe: Probe) -> None:
    """Wrap the merge so every hit is recorded with its evidence."""
    real = DetectorWorker._adopt_continuation

    def wrapped(self, track, still_active, claimed):
        before = {t.track_id: (t.frame_bboxes[0] if t.frame_bboxes else None,
                               t.best_conf) for t in still_active.values()}
        last = track.frame_bboxes[-1] if track.frame_bboxes else None
        got = real(self, track, still_active, claimed)
        if got:
            # Whichever candidate now starts where this track began is the one
            # it was folded into.
            winner = next((t for t in still_active.values()
                           if t.min_frame_idx == track.min_frame_idx), None)
            self.merges.append({
                "lost": track.track_id,
                "into": winner.track_id if winner else None,
                "lost_last_box": last,
                "cand_first_box": before.get(winner.track_id, (None, None))[0] if winner else None,
                "lost_conf": round(track.best_conf, 3),
                "cand_conf": round(before.get(winner.track_id, (None, 0.0))[1], 3) if winner else None,
                "lost_frame": track.best_frame_idx,
            })
        return got

    probe._adopt_continuation = wrapped.__get__(probe, Probe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=14.0)
    ap.add_argument("--cameras", type=int, default=2,
                    help="how many cameras to watch; each one costs GPU that "
                         "production is also using")
    ap.add_argument("--only", default="",
                    help="comma-separated camera ids to watch instead of the "
                         "first --cameras. Worth setting: on 2026-08-13, 160 of "
                         "162 detections came from 場域甲, so a run that omits it "
                         "is very likely to see nothing at all.")
    args = ap.parse_args()

    api = os.getenv("CLOUD_API_URL")
    tok = httpx.post(f"{api}/api/auth/login",
                     json={"code": os.getenv("CLOUD_ACCESS_CODE")},
                     timeout=30).json()["token"]
    cams = httpx.get(f"{api}/api/cameras",
                     headers={"Authorization": f"Bearer {tok}"},
                     timeout=30).json()
    if args.only:
        want = {c.strip() for c in args.only.split(",") if c.strip()}
        cams = [c for c in cams if c["camera_id"] in want]
        missing = want - {c["camera_id"] for c in cams}
        if missing:
            print(f"WARNING: no such camera(s): {sorted(missing)}", flush=True)
    else:
        cams = cams[: args.cameras]
    if not cams:
        print("no cameras selected — nothing to watch")
        return 1

    from ultralytics import YOLO

    yaml_path = config.TRACKER_YAML
    print(f"tracker yaml : {yaml_path}")
    print(f"track_buffer : {open(yaml_path).read().split('track_buffer:')[1].split('#')[0].strip()}"
          f"  (TRACK_LOST_SECONDS={config.TRACK_LOST_SECONDS})")
    print(f"continuation : {config.TRACK_CONTINUATION_SECONDS}s, IoU>={config.TRACK_CONTINUATION_IOU}")
    print(f"cameras      : {[c['camera_id'] for c in cams]}")
    print(f"duration     : {args.minutes} min\n", flush=True)

    OUT.mkdir(exist_ok=True)
    state = []
    for c in cams:
        cid = c["camera_id"] + "-trackverify"
        stream_manager.add(cid, c["rtsp_url"])
        probe = Probe(c["camera_id"])
        instrument(probe)
        state.append({
            "probe": probe, "cid": cid, "model": YOLO(config.MODEL_PATH),
            "active": {}, "buf": _FrameBuffer(), "n": 0, "frames": 0, "seq": 0,
        })

    deadline = time.time() + args.minutes * 60
    last_report = time.time()
    while time.time() < deadline:
        for s in state:
            # frame_after, not frame: only DISTINCT frames, the same contract
            # the production loop uses. Re-running the tracker on a repeated
            # frame would advance ByteTrack's frame_id without new evidence and
            # age lost tracks faster than production does.
            stream = stream_manager.get(s["cid"])
            if stream is None:
                continue
            frame, s["seq"] = stream.frame_after(s["seq"])
            if frame is None:
                continue
            s["frames"] += 1
            s["n"] += 1
            i = s["n"]
            res = s["model"].track(
                frame, imgsz=config.imgsz_for_frame(*frame.shape[:2]),
                verbose=False, conf=config.TRACKER_CONFIDENCE_THRESHOLD,
                persist=True, tracker=yaml_path,
                device=config.INFERENCE_DEVICE or None,
            )
            for r in res:
                b = r.boxes
                if b is None or b.id is None:
                    continue
                for box, tid, cf in zip(b.xyxy.cpu().numpy(),
                                        b.id.cpu().numpy(),
                                        b.conf.cpu().numpy()):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    s["probe"].tracks.add(int(tid))
                    s["probe"].peak_conf = max(s["probe"].peak_conf, float(cf))
                    s["probe"]._update_track(s["active"], int(tid), float(cf),
                                             x1, y1, x2, y2, i, frame)
            s["buf"].put(i, frame, time.time())

            now = time.time()
            lost = [t for t, st in s["active"].items()
                    if (now - st.last_seen_ts) > config.TRACK_LOST_SECONDS]
            if lost:
                s["probe"]._finalise_batch([s["active"].pop(t) for t in lost],
                                           s["active"], s["buf"])
        if time.time() - last_report > 120:
            last_report = time.time()
            for s in state:
                p = s["probe"]
                print(f"  [{int(deadline - time.time()):>4}s left] {p.camera_id}: "
                      f"{s['frames']} frames, {len(p.tracks)} tracks, "
                      f"{p.rows} rows, {len(p.merges)} merges, "
                      f"peak conf {p.peak_conf:.2f}", flush=True)

    print("\n=== result ===")
    total_rows = total_merges = 0
    for s in state:
        p = s["probe"]
        for t in list(s["active"].values()):
            pass  # tracks still live at the deadline are not counted either way
        total_rows += p.rows
        total_merges += len(p.merges)
        print(f"\n{p.camera_id}: {s['frames']} frames, {len(p.tracks)} tracks ever, "
              f"peak conf {p.peak_conf:.2f}, {p.rows} rows emitted, "
              f"{len(p.merges)} merges")
        if not p.tracks:
            print("  NOTHING WALKED PAST — this run says nothing about the merge.")
        for m in p.merges:
            print(f"  MERGE #{m['lost']} -> #{m['into']}  "
                  f"conf {m['lost_conf']}/{m['cand_conf']}\n"
                  f"    lost last box {m['lost_last_box']}\n"
                  f"    cand first box {m['cand_first_box']}")
            frame = s["buf"].get(m["lost_frame"])
            if frame is not None and m["lost_last_box"]:
                x, y, w, h = (int(v) for v in m["lost_last_box"])
                pad = max(w, h)
                crop = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
                if crop.size:
                    cv2.imwrite(str(OUT / f"merge_{p.camera_id}_{m['lost']}_{m['into']}.jpg"), crop)

    print(f"\nTOTAL: {total_rows} rows, {total_merges} merges")
    print(f"crops (if any) under {OUT}")
    print("\nA merge is CORRECT when both boxes sit on the same animal in the "
          "same place. Judge them by the boxes above and the saved crops — "
          "count, on its own, proves nothing.")

    for s in state:
        stream_manager.remove(s["cid"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
