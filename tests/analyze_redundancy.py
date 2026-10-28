"""Dataset redundancy audit.

Fixed cameras keep re-shooting the same tree, wall and patch of sky. This tool
quantifies how much of an ndjson export is likely teaching the model the same
scene again.

Usage:
  uv run --project backend python tests/analyze_redundancy.py
  uv run --project backend python tests/analyze_redundancy.py --split train --threshold 3
  uv run --project backend python tests/analyze_redundancy.py --ndjson ml/datasets/foo.ndjson

Images are read from the same content-hash cache as tests/eval_ratchet.py:
tests/.dataset_cache/. Run eval_ratchet or fetch the dataset first if the cache
is sparse.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from services import dedup  # noqa: E402

_HASH_RE = re.compile(r"/i/([0-9a-f]{32})\.")
_TRACK_RE = re.compile(r"(.+?)_t(\d+)_(\d{8})_")


def find_ndjson() -> Path:
    candidates = sorted((REPO / "ml" / "datasets").glob("*.ndjson"))
    if not candidates:
        sys.exit("no *.ndjson dataset export found in ml/datasets/")
    if len(candidates) > 1:
        print(f"warning: multiple ndjson files, using {candidates[-1].name}")
    return candidates[-1]


def cache_path(cache: Path, rec: dict) -> Path:
    """Same cache key as tests/eval_ratchet.py: URL tail without query."""
    tail = (rec.get("url") or "").split("?")[0].rsplit("/", 1)[-1]
    return cache / tail


def load_records(ndjson: Path, split: str) -> list[dict]:
    records = []
    for line in ndjson.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("type") != "image":
            continue
        if split != "all" and rec.get("split") != split:
            continue
        m = _HASH_RE.search(rec.get("url", ""))
        if not m:
            continue
        boxes = (rec.get("annotations") or {}).get("boxes") or []
        records.append({
            "hash": m.group(1),
            "file": rec.get("file", ""),
            "split": rec.get("split"),
            "boxes": boxes,
            "url": rec.get("url", ""),
        })
    return records


def burst_groups(records: list[dict]) -> tuple[int, dict[tuple[str, str, str], list[dict]]]:
    bursts: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    unparsed = 0
    for rec in records:
        m = _TRACK_RE.match(rec["file"])
        if not m:
            unparsed += 1
            continue
        camera, track_id, date = m.groups()
        bursts[(camera, date, track_id)].append(rec)
    return unparsed, {k: v for k, v in bursts.items() if len(v) > 1}


def perceptual_groups(records: list[dict], cache: Path, threshold: int) -> tuple[int, list[list[str]]]:
    pairs = []
    missing = 0
    for rec in records:
        path = cache_path(cache, rec)
        if not path.exists():
            missing += 1
            continue
        h = dedup.dhash(path)
        if h is not None:
            pairs.append((rec["hash"], h))
    groups = [g for g in dedup.group_near_duplicates(pairs, threshold) if len(g) > 1]
    return missing, groups


def print_distribution(records: list[dict]) -> None:
    cams = Counter(
        m.group(1) if (m := _TRACK_RE.match(rec["file"])) else "其他"
        for rec in records
    )
    dates = Counter(
        m.group(1) for rec in records if (m := re.search(r"_(\d{8})_", rec["file"]))
    )
    areas = sorted(box[3] * box[4] for rec in records for box in rec["boxes"])
    negs = sum(1 for rec in records if not rec["boxes"])

    print("── 3. 分佈（瘦身後必須保留的多樣性）")
    print(f"   split: {dict(Counter(rec['split'] for rec in records))}")
    print(f"   攝影機: {dict(cams.most_common(6))}")
    print(f"   日期數 {len(dates)} · 前三多: {dict(dates.most_common(3))}")
    if areas:
        print(
            f"   框面積佔比 p10 {areas[len(areas)//10]*100:.2f}% · "
            f"p50 {areas[len(areas)//2]*100:.2f}% · "
            f"p90 {areas[9*len(areas)//10]*100:.2f}%"
        )
    print(f"   負樣本 {negs} ({negs/max(len(records), 1):.0%})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndjson", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=REPO / "tests" / ".dataset_cache")
    parser.add_argument("--split", default="all", choices=("all", "train", "val", "test"))
    parser.add_argument("--threshold", type=int, default=5)
    args = parser.parse_args()

    ndjson = args.ndjson or find_ndjson()
    records = load_records(ndjson, args.split)
    if not records:
        sys.exit(f"{ndjson} has no {args.split} image records")

    print(f"資料集 {ndjson.name} · {args.split} · {len(records)} 張\n")

    unparsed, multi = burst_groups(records)
    in_bursts = sum(len(v) for v in multi.values())
    print("── 1. 同軌跡連拍（同攝影機+同日+同 track id）")
    print(f"   可解析檔名 {len(records) - unparsed} 張 · 落在多幀軌跡的 {in_bursts} 張")
    print(f"   軌跡數 {len(multi)} · 每軌平均 {in_bursts / max(len(multi), 1):.1f} 幀")
    sizes = Counter(len(v) for v in multi.values())
    print(f"   軌跡幀數分佈: {dict(sorted(sizes.items())[:8])}")
    print(
        f"   → 每軌只留 1 幀可減 {in_bursts - len(multi)} 張 "
        f"({(in_bursts - len(multi)) / len(records):.0%} 的資料集)\n"
    )

    print(f"── 2. 感知近重複（dHash，Hamming <= {args.threshold}）")
    missing, groups = perceptual_groups(records, args.cache, args.threshold)
    dupes = sum(len(g) - 1 for g in groups)
    available = len(records) - missing
    if available == 0:
        print(
            f"   cache missing {missing} 張 · 無法評估感知近重複 "
            "（先跑 eval_ratchet 或 regenerate 管線下載影像）\n"
        )
    else:
        print(
            f"   cache available {available} 張 · missing {missing} 張 · "
            f"近重複群 {len(groups)} 組 · 冗餘 {dupes} 張 "
            f"({dupes / available:.0%})\n"
        )

    print_distribution(records)


if __name__ == "__main__":
    main()
