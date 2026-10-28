"""Align the archives of one visit onto a single frame sequence.

A reconstructed visit groups rows that each brought their OWN clip — 146 of 330
visits span more than one, one of them thirteen (2026-08-29). Their archives
overlap: of those 146, every internal clip pair shares frames in 121 and some
pair shares in the other 25; none is unrelated.

The alignment is EXACT, not interpolated. Frames carry no timestamps — only a
sequence number — but two archives cut from the same stream contain byte-equal
JPEGs where they overlap, so hashing every frame and matching gives the precise
offset of one archive against another. Chaining those offsets across a visit's
archives puts every frame on one axis with nothing guessed.

Writes alignment/<visit>.json: the ordered, de-duplicated frame list and, per
archive, its offset into it.

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

本檔是第 2 步。
"""
import collections, hashlib, json, zipfile
from pathlib import Path

BASE = Path.home() / "rawframes-backfill"
OUT = BASE / "alignment"; OUT.mkdir(exist_ok=True)
rows = json.load(open(BASE / "detections.json"))


def zipname(trail: str) -> str:
    stem, _, _ = trail.rpartition("_trail.")
    return f"{stem}_rawframes.zip"


byvisit: dict[str, list] = collections.defaultdict(list)
for r in rows:
    if r.get("visit_id") and r.get("trail_path"):
        byvisit[r["visit_id"]].append(r)
multi = {v: rs for v, rs in byvisit.items() if len({r["trail_path"] for r in rs}) > 1}
print(f"{len(multi)} 個造訪橫跨多支影片", flush=True)

_hash_cache: dict[str, list[str]] = {}


def hashes(z: str) -> list[str]:
    """Content hash of every frame, in archive order."""
    if z in _hash_cache:
        return _hash_cache[z]
    p = BASE / "zips" / z
    if not p.exists():
        _hash_cache[z] = []
        return []
    zf = zipfile.ZipFile(p)
    names = sorted(n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg")))
    out = [hashlib.sha1(zf.read(n)).hexdigest() for n in names]
    _hash_cache[z] = out
    return out


def offset_of(a: list[str], b: list[str]) -> int | None:
    """Where b starts relative to a, from the frames they share. None if none."""
    pos = {h: i for i, h in enumerate(a)}
    votes = collections.Counter(pos[h] - j for j, h in enumerate(b) if h in pos)
    if not votes:
        return None
    off, n = votes.most_common(1)[0]
    return off if n >= 2 or len(votes) == 1 else off


done = {"aligned": 0, "split": 0, "no archive": 0}
for visit, rs in sorted(multi.items()):
    zs = sorted({zipname(r["trail_path"]) for r in rs})
    hs = {z: hashes(z) for z in zs}
    if any(not h for h in hs.values()):
        done["no archive"] += 1
        continue
    # anchor on the archive with the most frames, then chain the rest onto it
    anchor = max(zs, key=lambda z: len(hs[z]))
    placed = {anchor: 0}
    for _ in range(len(zs)):
        for z in zs:
            if z in placed:
                continue
            for base, boff in list(placed.items()):
                o = offset_of(hs[base], hs[z])
                if o is not None:
                    placed[z] = boff + o
                    break
    if len(placed) < len(zs):
        done["split"] += 1
    else:
        done["aligned"] += 1
    # union of frames on the shared axis, de-duplicated by content
    axis: dict[int, str] = {}
    for z, off in placed.items():
        for i, h in enumerate(hs[z]):
            axis.setdefault(off + i, h)
    lo = min(axis)
    seq = [{"slot": k - lo, "hash": axis[k]} for k in sorted(axis)]
    (OUT / f"{visit}.json").write_text(json.dumps({
        "visit": visit,
        "archives": {z: placed[z] - lo for z in placed},
        "unplaced": [z for z in zs if z not in placed],
        "frames": len(seq),
        "sequence": seq,
    }))

print("結果:", done, flush=True)
print(f"寫到 {OUT}")
