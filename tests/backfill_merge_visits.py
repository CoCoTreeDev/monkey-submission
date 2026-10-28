"""Render one clip per visit from its aligned archives, and place its rows on it.

A reconstructed visit is spread over several clips because its rows each
brought their own — 146 visits over 479 files, one of them over thirteen. You
cannot read "which monkey is which, and when" off thirteen different videos
with thirteen different axes, which is the whole point of the timeline.

`align_visits.py` has already put every archive of a visit on ONE frame
sequence, exactly: frames carry no timestamps, but archives cut from the same
stream share byte-equal JPEGs, so hashing and matching gives the true offset
with nothing interpolated. All 146 aligned with no split and no missing archive.

This renders that sequence, which also means tracking runs over the COMPLETE
footage rather than each subsampled archive on its own — the fragmentation that
made re-tracking mint ids the producer never had is much reduced.

  --limit N   stop after N visits
  --upload    push to blob storage (default: local only)
  --force     rebuild visits that already have a sidecar/clip

順序（都在 mini 上跑，各步可續跑；斷線只是少做幾個，不會做壞）：

  1. backfill_retrack.py         在單支存檔上重跑追蹤 —— 只給「還沒合併」的列用
  2. backfill_align_visits.py    用共用影格把一次造訪的多支存檔對齊成一條序列
  3. backfill_merge_visits.py    渲染合併後的影片、算出每列在它上面的位置，
                                 產出 merge_payload.json
  4. backfill_visit_archives.py  產生對應的合併存檔並上傳
  5. 把 payload POST 到 /api/admin/detections/backfill-timeline
     （backfill_timeline.py 自己會送；merge_payload.json 直接送即可）

第 4 步不是可選的：後端從 trail_path 推導原始影格 zip 的名字，少了合併存檔，
原本的 zip 就不再被任何列引用，孤兒清理會把這整套回填的素材刪掉。

本檔是第 3 步。
"""
from __future__ import annotations

import argparse, collections, hashlib, json, subprocess, sys, zipfile
from pathlib import Path

BASE = Path.home() / "rawframes-backfill"
sys.path.insert(0, str(Path.home() / "Monkey-Detection" / "backend"))
import config                                        # noqa: E402
import cv2, numpy as np                              # noqa: E402
from ultralytics import YOLO                         # noqa: E402

OUT = BASE / "merged"; OUT.mkdir(exist_ok=True)
IMGSZ, CONF = 1600, 0.35
# Same palette idea as the producer: one colour per track so a wall of macaques
# reads as several animals rather than a swarm of identical boxes.
PALETTE = [(249, 232, 103), (36, 191, 251), (133, 113, 251), (172, 239, 134),
           (253, 181, 196), (252, 211, 125), (252, 171, 240), (116, 186, 253)]


def colour_index(value):
    """Match the frontend's stable FNV palette selection."""
    hash_value = 2166136261
    for char in str(value):
        hash_value ^= ord(char)
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return hash_value % len(PALETTE)


def draw_review_box(frame, box, colour, label):
    """Draw a readable, stable review box over mixed camera backgrounds."""
    x1, y1, x2, y2 = (int(value) for value in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (8, 8, 8), 7, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3, cv2.LINE_AA)
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    top = max(th + baseline + 4, y1)
    left = max(0, min(x1, frame.shape[1] - tw - 10))
    cv2.rectangle(frame, (left, top - th - baseline - 6),
                  (left + tw + 10, top), colour, -1)
    cv2.putText(frame, label, (left + 5, top - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (8, 8, 8), 2, cv2.LINE_AA)


def frames_of(visit: dict) -> list[np.ndarray]:
    """The visit's frames, in aligned order, decoded once."""
    want = {e["hash"]: e["slot"] for e in visit["sequence"]}
    got: dict[int, np.ndarray] = {}
    for z in visit["archives"]:
        p = BASE / "zips" / z
        if not p.exists():
            continue
        zf = zipfile.ZipFile(p)
        for n in sorted(zf.namelist()):
            if not n.lower().endswith((".jpg", ".jpeg")):
                continue
            raw = zf.read(n)
            h = hashlib.sha1(raw).hexdigest()
            slot = want.get(h)
            if slot is None or slot in got:
                continue
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                got[slot] = img
    return [got[k] for k in sorted(got)]


def encode(frames: list[np.ndarray], fps: float, dest: Path) -> bool:
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        [config.FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{w}x{h}",
         "-r", f"{fps:.3f}", "-i", "-", "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
        stdin=subprocess.PIPE)
    try:
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        return proc.wait(timeout=180) == 0
    except Exception:
        proc.kill()
        return False


def source_fps(archives) -> float | None:
    """The rate the clips being merged already played at.

    The first version of this divided the frame count by a target duration,
    which is not a frame rate at all — every merged clip came out exactly 8
    seconds regardless of how long the visit was, and a 17-frame visit played
    at the same length as a 547-frame one.

    The honest rate is the one the sources had: each was written at a constant
    rate the producer chose, so N/duration off the file gives it back. Merging
    adds frames, and at the same rate more frames simply means a longer clip —
    which is the point, because the visit really was longer.

    Fitting the rate from the archives' own timestamps was tried first and
    thrown away: `frame_identity`'s `ts` is when the row was EMITTED, not when
    its first frame was captured, so two archives sharing frames can carry
    timestamps minutes apart. It fitted 0.2-1.2 fps, which is nonsense.
    """
    import statistics
    rates = []
    for z in archives:
        trail = z.replace("_rawframes.zip", "_trail.mp4")
        for d in ("rebuilt", "originals", "fetched"):
            p = BASE / d / trail
            if not p.exists():
                continue
            cap = cv2.VideoCapture(str(p))
            f, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if f > 0 and n > 0:
                rates.append(f)
            break
    return statistics.median(rates) if rates else None


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_rows_to_tracks(rows: list[dict], tracks: dict[int, list],
                         threshold: float = 0.1) -> dict[int, int]:
    """Match recorded rows to re-tracked tracks, once, by their best overlap.

    The retracker's ids are not production ids.  They are only an internal
    handle for finding the boxes to draw and must never be shown to users.
    More importantly, a retracked track with no matching row is not a recorded
    detection and must not be burned into the rebuilt clip.
    """
    matches: dict[int, int] = {}
    taken: set[int] = set()
    for row in sorted(rows, key=lambda item: -item["confidence"]):
        want = (row["bbox_x"], row["bbox_y"],
                row["bbox_x"] + row["bbox_w"], row["bbox_y"] + row["bbox_h"])
        best, score = None, 0.0
        for tid, boxes in tracks.items():
            overlap = max((iou(want, box[1:]) for box in boxes), default=0.0)
            if tid in taken:
                overlap *= 0.5
            if overlap > score:
                best, score = tid, overlap
        if best is not None and score >= threshold:
            matches[row["id"]] = best
            taken.add(best)
    return matches


UPLOADED = BASE / "uploaded_merged.json"


def upload_once(dest: Path) -> None:
    """Upload each merged clip exactly once, across runs.

    Rendering and uploading are separate failures: a run cut off after the
    encode leaves a file on disk that nothing in storage knows about, and a
    resume that skips on `dest.exists()` would never send it.
    """
    done = set(json.loads(UPLOADED.read_text())) if UPLOADED.exists() else set()
    if dest.name in done:
        return
    from services.storage import storage
    storage.upload(dest.name, dest.read_bytes())
    done.add(dest.name)
    UPLOADED.write_text(json.dumps(sorted(done)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = json.load(open(BASE / "detections.json"))
    byvisit: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        if r.get("visit_id"):
            byvisit[r["visit_id"]].append(r)

    files = sorted((BASE / "alignment").glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    model = YOLO(str(Path.home() / "Monkey-Detection" / "backend" / "best.pt"))
    model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=IMGSZ, device="mps", verbose=False)

    manifest, skipped = [], collections.Counter()
    for n, f in enumerate(files, 1):
        visit = json.loads(f.read_text())
        vid = visit["visit"]
        vrows = byvisit.get(vid, [])
        dest = OUT / f"{vrows[0]['camera_id']}_visit{vid}_trail.mp4" if vrows else None
        if not vrows or dest is None:
            skipped["沒有列"] += 1
            continue
        side = OUT / f"{vid}.json"
        if side.exists() and not args.force:
            manifest.extend(json.loads(side.read_text()))
            if args.upload:
                upload_once(dest)
            skipped["已完成"] += 1
            continue

        frames = frames_of(visit)
        if len(frames) < 2:
            skipped["影格太少"] += 1
            continue

        # Track over the COMPLETE sequence — this is the part the per-archive
        # re-tracking could not do.
        model.predictor = None
        tracks: dict[int, list] = collections.defaultdict(list)
        for i, img in enumerate(frames):
            a = img
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
                tracks[int(tid) if tid is not None else -1].append([i, *xyxy])

        # First match the re-tracked handles to the rows that were actually
        # recorded.  Drawing every re-tracked handle made the clip disagree
        # with the database: it showed extra boxes and used ids that only
        # existed in this backfill process.
        matches = match_rows_to_tracks(vrows, tracks)
        rows_by_id = {row["id"]: row for row in vrows}
        canvas = [f.copy() for f in frames]
        for ci, (row_id, tid) in enumerate(sorted(matches.items())):
            row = rows_by_id[row_id]
            boxes = tracks[tid]
            col = PALETTE[colour_index(row.get("animal_id") or row["track_id"])]
            for i, x1, y1, x2, y2 in boxes:
                draw_review_box(canvas[i], [x1, y1, x2, y2], col, f"#{row['track_id']}")

        fps = source_fps(visit["archives"]) or 10.0
        if dest.exists() and not args.force:
            skipped["影片已存在，只重算軌跡"] += 1
        elif not encode(canvas, fps, dest):
            skipped["編碼失敗"] += 1
            continue
        duration = len(frames) / fps

        # place each row on the merged axis by box overlap, as before
        placed = []
        for r in sorted(vrows, key=lambda x: -x["confidence"]):
            best = matches.get(r["id"])
            entry = {"id": r["id"], "trail_path": dest.name,
                     "clip_duration_seconds": round(duration, 3)}
            if best is not None:
                pos = sorted({b[0] for b in tracks[best]})
                st = [p / fps for p in pos]
                segs, a, prev = [], st[0], st[0]
                for t in st[1:]:
                    if t - prev > config.SEGMENT_GAP_SECONDS:
                        segs.append([round(a, 3), round(prev, 3)]); a = t
                    prev = t
                segs.append([round(a, 3), round(prev, 3)])
                entry.update(clip_offset_seconds=round(st[0], 3),
                             clip_end_offset_seconds=round(st[-1], 3),
                             clip_segments=segs)
            placed.append(entry)
        manifest.extend(placed)

        # Keep the per-frame boxes. Not keeping them is what stopped the hover
        # outline from following the animal, made the SAM-propagation trial
        # meaningless (it was prompted with a box from the wrong moment), and
        # leaves the motion-continuity merge in #66 with nothing to work from.
        # They cost a few KB per visit and the tracking that produces them is
        # the expensive part of this script.
        boxes = {str(tid): [[b[0], *[round(v, 1) for v in b[1:]]] for b in bs]
                 for tid, bs in tracks.items()}
        (OUT / f"{vid}.boxes.json").write_text(json.dumps({
            "visit": vid, "fps": round(fps, 4), "frames": len(frames), "tracks": boxes}))
        side.write_text(json.dumps(placed))
        if args.upload:
            upload_once(dest)
        if n % 10 == 0:
            print(f"[{n}/{len(files)}] {vid} {len(frames)} 格 {len(tracks)} 軌 {len(placed)} 列", flush=True)

    (BASE / "merge_payload.json").write_text(json.dumps({"dry_run": True, "rows": manifest}))
    print(f"完成：{len(manifest)} 列可重指，跳過 {dict(skipped)}", flush=True)


if __name__ == "__main__":
    main()
