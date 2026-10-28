"""Stamp historical rows with which-animal-of-its-visit they are.

Runs on the mini, where the per-frame boxes are. Three steps:

  1. read the chains link_tracks.py wrote (track_links.json — run it at the
     CALIBRATED threshold first);
  2. match each visit's rows to the merged per-frame tracks by box overlap,
     the same way the timeline backfill does — the re-tracked ids share no id
     space with the producer's track_id, so the row's own box is the only
     identity available;
  3. turn each chain into a group of DETECTION ids and hand it to
     POST /api/admin/detections/backfill-animal-groups.

The grouping stays an interpretation: no row is deleted or re-timed, and the
endpoint's clear mode reverses everything this writes.

Usage, on the mini (dry run unless --apply):

    API=… TOKEN=… uv run --project ~/Monkey-Detection/backend \
        python tests/backfill_animal_groups.py --evidence "animal-group calibration θ=…" [--apply]
"""
import argparse, json, os, sys, urllib.error, urllib.request
from pathlib import Path

BASE = Path.home() / "rawframes-backfill"

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--evidence", required=True,
                help="哪一份證據定的門檻，寫進 manifest，例如 animal-group calibration θ=1.0 calib 2026-08-30")
ap.add_argument("--links", default=str(BASE / "track_links.json"))
args = ap.parse_args()

API = os.environ["API"]
TOKEN = os.environ["TOKEN"]

links_data = json.load(open(args.links))
rows = json.load(open(BASE / "detections.json"))

by_visit: dict[str, list] = {}
for r in rows:
    t = r.get("trail_path") or ""
    if "_visit" in t:
        by_visit.setdefault(
            t.split("_visit")[1].replace("_trail.mp4", ""), []).append(r)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


groups, unmatched = [], 0
for visit_id, v in links_data["visits"].items():
    visit_rows = by_visit.get(visit_id)
    boxes_file = BASE / "merged" / f"{visit_id}.boxes.json"
    if not visit_rows or not boxes_file.exists():
        continue
    tracks = json.loads(boxes_file.read_text())["tracks"]

    # Row → merged track, greedily by confidence, preferring unclaimed tracks
    # — to_timeline.py's matching, against the MERGED tracks link_tracks read.
    row_track: dict[int, str] = {}
    taken: set[str] = set()
    for r in sorted(visit_rows, key=lambda x: -x["confidence"]):
        want = (r["bbox_x"], r["bbox_y"],
                r["bbox_x"] + r["bbox_w"], r["bbox_y"] + r["bbox_h"])
        best, best_score = None, 0.0
        for tid, bs in tracks.items():
            score = max((iou(want, b[1:]) for b in bs), default=0.0)
            if tid in taken:
                score *= 0.5
            if score > best_score:
                best, best_score = tid, score
        if best is None or best_score < 0.1:
            unmatched += 1
            continue
        taken.add(best)
        row_track[r["id"]] = best

    links = v.get("links", [])
    for g in v["groups"]:
        gset = set(g)
        det_ids = sorted(i for i, t in row_track.items() if t in gset)
        if len(det_ids) < 2:
            continue          # the chain holds fewer than two ROWS: nothing to group
        groups.append({
            "visit_id": visit_id,
            "detection_ids": det_ids,
            "links": [l for l in links if l["a"] in gset or l["b"] in gset],
        })

print(f"track_links.json: gap={links_data.get('gap')} err={links_data.get('err')}"
      f" → {len(groups)} 個分組（≥2 列），{unmatched} 列對不到軌跡")
if not groups:
    sys.exit(0)

payload = {
    "dry_run": not args.apply,
    "source": "rule",
    "evidence": args.evidence,
    "groups": groups,
}
req = urllib.request.Request(
    f"{API}/api/admin/detections/backfill-animal-groups",
    data=json.dumps(payload).encode(),
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    method="POST")
try:
    out = json.load(urllib.request.urlopen(req, timeout=300))
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read()[:600].decode(errors='replace')}")
    sys.exit(1)
print(json.dumps({k: v for k, v in out.items() if k != "detail"},
                 ensure_ascii=False, indent=1))
for p in out.get("detail", [])[:20]:
    print(" ", p)
if not args.apply:
    print("（這是試算。加上 --apply 才會真的寫入）")
