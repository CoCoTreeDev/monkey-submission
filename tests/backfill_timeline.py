"""Turn re-tracked frame positions into the seconds a player can seek to.

Runs on the mini, where the archives, the re-tracking results and most of the
clips already are. Writes a payload for
POST /api/admin/detections/backfill-timeline.

Two things decide whether the numbers are usable at all:

  * **The axis is the MP4's own length**, read off the file. Not the visit's
    wall-clock span — the clip is written at a constant rate over frames that
    arrived in clumps, so the two differ by up to 1.8x (measured on rows
    16318/16319, 2026-08-28) and offsets taken from the wrong one point past
    the end of the video they describe.
  * **A row is matched to a track by box overlap**, because re-tracking a
    subsampled archive mints ids the producer never had. The row's own box is
    the only identity available.

A row nothing matched still gets the clip's LENGTH, with no position: the panel
then draws the axis and marks that lane "—". Saying "somewhere in these 7
seconds" is worth something; guessing a second the re-tracking never found is
not.

These approximations are why every row this fills stays marked as a
reconstruction rather than a recording.

Usage, on the mini (dry run unless --apply):

    API=… TOKEN=… uv run --project ~/Monkey-Detection/backend \
        python tests/backfill_timeline.py [--apply]
"""
import json, os, sys, urllib.error, urllib.parse, urllib.request
from pathlib import Path

BASE = Path.home() / "rawframes-backfill"
sys.path.insert(0, str(Path.home() / "Monkey-Detection" / "backend"))
import cv2                                                    # noqa: E402
from config import SEGMENT_GAP_SECONDS                        # noqa: E402

API = os.environ["API"]
TOKEN = os.environ["TOKEN"]
rows = json.load(open(BASE / "detections.json"))


CACHE = BASE / "fetched"
CACHE.mkdir(exist_ok=True)


def _probe(path: Path) -> float | None:
    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if fps > 0 and n > 0 else None


def _fetch(trail: str) -> Path | None:
    """Pull a clip the mini does not hold. The API answers with a 302 to signed
    blob storage, which rejects the Authorization header if it is carried along
    — so the redirect is followed by hand, without it."""
    dest = CACHE / trail
    if dest.exists():
        return dest
    url = f"{API}/api/snapshots/" + urllib.parse.quote(trail)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    try:
        data = urllib.request.build_opener(_NoRedirect).open(req, timeout=120).read()
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            return None
        data = urllib.request.urlopen(e.headers["Location"], timeout=180).read()
    except Exception:
        return None
    dest.write_bytes(data)
    return dest


def mp4_seconds(trail: str) -> float | None:
    """The clip's real length, from the file itself."""
    for d in ("rebuilt", "originals", "fetched"):
        p = BASE / d / trail
        if p.exists() and (secs := _probe(p)):
            return secs
    got = _fetch(trail)
    return _probe(got) if got else None


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


by_clip: dict[str, list] = {}
for r in rows:
    stem, mk, _ = (r.get("trail_path") or "").rpartition("_trail.")
    if mk:
        by_clip.setdefault(f"{stem}_rawframes.zip", []).append(r)

out = []
skipped = {"沒有重跑結果": 0, "抓不到影片": 0, "位置不明（只填長度）": 0, "影格太少": 0}
for clip, clip_rows in sorted(by_clip.items()):
    # The clip's length comes first, because it is worth having on its own.
    # Some clips have no archive to re-track at all — the zip was never kept,
    # or holds a single frame — and skipping those outright left seven rows
    # with no axis when every other row had one.
    duration = mp4_seconds(clip_rows[0]["trail_path"])
    if not duration or duration <= 0:
        skipped["抓不到影片"] += len(clip_rows); continue

    res = BASE / "retrack" / (clip + ".json")
    data = json.loads(res.read_text()) if res.exists() else {}
    tracks, n = data.get("tracks") or {}, data.get("frames") or 0
    if n < 2 or not tracks:
        # Length only: the panel draws the axis and marks every lane "—".
        reason = "沒有重跑結果" if not res.exists() else "影格太少"
        skipped[reason] += len(clip_rows)
        out.extend({"id": r["id"], "clip_offset_seconds": None,
                    "clip_end_offset_seconds": None,
                    "clip_duration_seconds": round(duration, 3),
                    "clip_segments": None} for r in clip_rows)
        continue

    def secs(pos: int) -> float:
        # The archive is assumed evenly spaced across the clip; that assumption
        # is the reconstruction's main inaccuracy and is stated in the docstring.
        return min(duration, max(0.0, pos / (n - 1) * duration))

    taken: set[str] = set()
    for r in sorted(clip_rows, key=lambda x: -x["confidence"]):
        want = (r["bbox_x"], r["bbox_y"], r["bbox_x"] + r["bbox_w"], r["bbox_y"] + r["bbox_h"])
        best, best_score = None, 0.0
        for tid, boxes in tracks.items():
            score = max((iou(want, b[1:]) for b in boxes), default=0.0)
            if tid in taken:
                score *= 0.5          # one row is one animal; prefer an unclaimed track
            if score > best_score:
                best, best_score = tid, score
        if best is None or best_score < 0.1:
            # Nothing overlapped this row's box. The clip's LENGTH is still
            # known, so hand that over on its own: the panel then draws the
            # axis and marks this lane "—" instead of claiming a second the
            # re-tracking never found. Refusing to guess is the point.
            skipped["位置不明（只填長度）"] += 1
            out.append({"id": r["id"],
                        "clip_offset_seconds": None,
                        "clip_end_offset_seconds": None,
                        "clip_duration_seconds": round(duration, 3),
                        "clip_segments": None})
            continue
        taken.add(best)
        pos = sorted({b[0] for b in tracks[best]})
        stamps = [secs(p) for p in pos]
        segs, start, prev = [], stamps[0], stamps[0]
        for t in stamps[1:]:
            if t - prev > SEGMENT_GAP_SECONDS:
                segs.append([round(start, 3), round(prev, 3)]); start = t
            prev = t
        segs.append([round(start, 3), round(prev, 3)])
        out.append({"id": r["id"],
                    "clip_offset_seconds": round(stamps[0], 3),
                    "clip_end_offset_seconds": round(stamps[-1], 3),
                    "clip_duration_seconds": round(duration, 3),
                    "clip_segments": segs})

apply = "--apply" in sys.argv
print(f"可回填 {len(out)} 列，跳過 {skipped}", flush=True)
payload = {"dry_run": not apply, "rows": out}
(BASE / "timeline_payload.json").write_text(json.dumps(payload))

if not out:
    sys.exit(0)

# Dry run unless asked, mirroring the endpoint's own default. It refuses any
# row the producer recorded, and skips rows already carrying these values, so
# re-running after more archives are recovered is safe.
req = urllib.request.Request(
    f"{API}/api/admin/detections/backfill-timeline",
    data=json.dumps(payload).encode(),
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    method="POST")
try:
    print(json.load(urllib.request.urlopen(req, timeout=300)))
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read()[:600].decode(errors='replace')}")
    sys.exit(1)
if not apply:
    print("（這是試算。加上 --apply 才會真的寫入）")
