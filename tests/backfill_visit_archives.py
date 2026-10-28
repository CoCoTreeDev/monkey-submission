"""Build the merged raw-frame archive that a merged clip needs to exist.

`_raw_frames_name()` derives a row's archive from its clip name by swapping
`_trail.mp4` for `_rawframes.zip`. Point a row at a merged clip without also
writing the matching archive and two things break: "下載全部幀" 404s, and — the
dangerous one — the ORIGINAL archives stop being referenced by any row, so the
orphan sweep at main.py's `referenced.add(_raw_frames_name(...))` would be free
to delete the very frames every backfill here is built from.

No decode and no GPU: the frames are copied byte-for-byte in the order
`align_visits.py` worked out, which is also the order the merged clip is drawn
in, so the archive and the video agree frame for frame.

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

本檔是第 4 步。
"""
import collections, hashlib, json, sys, zipfile
from pathlib import Path

BASE = Path.home() / "rawframes-backfill"
sys.path.insert(0, str(Path.home() / "Monkey-Detection" / "backend"))
OUT = BASE / "merged"; OUT.mkdir(exist_ok=True)
UPLOADED = BASE / "uploaded_merged_zips.json"

rows = json.load(open(BASE / "detections.json"))
byvisit = collections.defaultdict(list)
for r in rows:
    if r.get("visit_id"):
        byvisit[r["visit_id"]].append(r)

done = set(json.loads(UPLOADED.read_text())) if UPLOADED.exists() else set()
files = sorted((BASE / "alignment").glob("*.json"))
made = skipped = 0
for f in files:
    vis = json.loads(f.read_text())
    vid = vis["visit"]
    vrows = byvisit.get(vid)
    if not vrows:
        skipped += 1
        continue
    name = f"{vrows[0]['camera_id']}_visit{vid}_rawframes.zip"
    dest = OUT / name
    if name in done:
        skipped += 1
        continue
    if not dest.exists():
        want = {e["hash"]: e["slot"] for e in vis["sequence"]}
        picked: dict[int, tuple[str, bytes]] = {}
        for z in vis["archives"]:
            p = BASE / "zips" / z
            if not p.exists():
                continue
            zf = zipfile.ZipFile(p)
            for n in sorted(zf.namelist()):
                if not n.lower().endswith((".jpg", ".jpeg")):
                    continue
                raw = zf.read(n)
                slot = want.get(hashlib.sha1(raw).hexdigest())
                if slot is None or slot in picked:
                    continue
                picked[slot] = (n, raw)
        if len(picked) < 2:
            skipped += 1
            continue
        cam = vrows[0]["camera_id"]
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as out:
            for i, slot in enumerate(sorted(picked)):
                out.writestr(f"{cam}_raw_frame{i:04d}.jpg", picked[slot][1])
    from services.storage import storage
    storage.upload(name, dest.read_bytes())
    done.add(name)
    UPLOADED.write_text(json.dumps(sorted(done)))
    made += 1
    if made % 20 == 0:
        print(f"{made} 個合併存檔已上傳", flush=True)

print(f"完成：新增 {made}，跳過 {skipped}", flush=True)
