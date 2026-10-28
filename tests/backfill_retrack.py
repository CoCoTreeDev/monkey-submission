"""Recover per-animal presence from the archived frames, for rows that predate
the producer recording it.

The honest limits, stated up front because they shape everything below:

  * The archive has NO timestamps — only JPEGs named by a post-subsample index.
    Seconds are therefore interpolated from the clip's known duration across an
    evenly-spaced archive. Close, not exact.
  * Re-tracking a SUBSAMPLED archive mints extra ids (measured 2026-08-28: a
    94-frame clip produced 19). So a track here is not the track the producer
    had, and rows are matched to it by box overlap, not by identity.

Both are why every row this fills is marked as reconstructed rather than
recorded. It answers "roughly when was something at this spot", not "this is
where animal #7 was".
"""
import json, os, sys, time, zipfile
from pathlib import Path

HOME = Path.home(); BASE = HOME / "rawframes-backfill"
sys.path.insert(0, str(HOME / "Monkey-Detection" / "backend"))
import config, cv2, numpy as np                       # noqa: E402
from ultralytics import YOLO                          # noqa: E402

OUT = BASE / "retrack"; OUT.mkdir(exist_ok=True)
IMGSZ, CONF = 1600, 0.35
rows = json.load(open(BASE / "detections.json"))
by_clip = {}
for r in rows:
    stem, mk, _ = (r["trail_path"] or "").rpartition("_trail.")
    if mk:
        by_clip.setdefault(f"{stem}_rawframes.zip", []).append(r)

model = YOLO(str(HOME / "Monkey-Detection" / "backend" / "best.pt"))
model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=IMGSZ, device="mps", verbose=False)

todo = sorted(by_clip)
# A time budget so this can be driven in foreground batches: each run does what
# it can and exits cleanly, and the next picks up where it left off, because a
# clip whose result file already exists is skipped.
BUDGET = float(os.environ.get("BUDGET", "0")) or None
started = time.monotonic()
print(f"{len(todo)} 支 clip 要重新追蹤"
      + (f"，本批預算 {BUDGET:.0f} 秒" if BUDGET else ""), flush=True)
for n, clip in enumerate(todo, 1):
    dest = OUT / (clip + ".json")
    if dest.exists():
        continue
    if BUDGET and time.monotonic() - started > BUDGET:
        print(f"預算用完，停在 {n}/{len(todo)}", flush=True)
        break
    zp = BASE / "zips" / clip
    if not zp.exists():
        dest.write_text(json.dumps({"clip": clip, "error": "no archive"}))
        continue
    zf = zipfile.ZipFile(zp)
    names = [x for x in sorted(zf.namelist()) if x.lower().endswith((".jpg", ".jpeg"))]
    model.predictor = None            # fresh tracker per clip; ids must not leak
    tracks: dict[int, list] = {}
    for pos, name in enumerate(names):
        a = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
        if a is None:
            continue
        if config.WHITE_BALANCE_CAST_THRESHOLD > 0 and \
                config.colour_cast(a) > config.WHITE_BALANCE_CAST_THRESHOLD:
            a = config.correct_white_balance(a)
        res = model.track(a, imgsz=IMGSZ, conf=CONF, classes=[0], device="mps", persist=True,
                          tracker=config.TRACKER_YAML, verbose=False)[0]
        b = res.boxes
        if b is None or not len(b):
            continue
        ids = b.id.tolist() if b.id is not None else [None] * len(b)
        for xyxy, tid in zip(b.xyxy.tolist(), ids):
            key = int(tid) if tid is not None else -1
            tracks.setdefault(key, []).append(
                [pos, *[round(float(v), 1) for v in xyxy]])
    dest.write_text(json.dumps({"clip": clip, "frames": len(names),
                                "tracks": tracks}))
    if n % 25 == 0:
        print(f"[{n}/{len(todo)}] {clip[-38:]} {len(names)} 幀 {len(tracks)} 條軌跡", flush=True)
print("DONE", flush=True)
