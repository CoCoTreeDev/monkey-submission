"""Chain the fragments of one animal's continuous motion back together.

The case that named this (2026-08-28 07:06:28, ground truth from the user): a
macaque runs left to right across 場域甲 and the tracker hands out EIGHT ids for
it. Centre x goes 15 → 1628 while y barely moves and the box grows as it nears
the camera. One animal, one run, eight records.

No appearance model is needed for that shape. Where a track ends and another
starts a few frames later, extrapolating the first one's velocity across the
gap lands on the second one's box — on the ground-truth case every link is
within 0.23-0.84 of a box width. That is the whole rule.

What it deliberately cannot do: two animals on screen AT THE SAME TIME are
never candidates, because a track that overlaps another in time is a different
animal by construction. That is also the error direction that matters — merging
two animals into one under-counts the troop, which is the number this system
exists to report — so every threshold here is set to refuse rather than guess.

  --gap N     seconds a break may last                (default 0.5)
  --err N     allowed miss, in box widths             (default 1.0)
  --ratio N   allowed box-width change across a break  (default 2.5)
  --apply     write the links (default: report only)
"""
from __future__ import annotations

import argparse, collections, json
from pathlib import Path

BASE = Path.home() / "rawframes-backfill"
MERGED = BASE / "merged"


def tail_velocity(boxes, fps, n=5):
    """Pixels per second at the end of a track, from its last few frames."""
    tail = boxes[-n:]
    if len(tail) < 2:
        return 0.0, 0.0
    dt = (tail[-1][0] - tail[0][0]) / fps
    if dt <= 0:
        return 0.0, 0.0
    cx = lambda b: (b[1] + b[3]) / 2
    cy = lambda b: (b[2] + b[4]) / 2
    return (cx(tail[-1]) - cx(tail[0])) / dt, (cy(tail[-1]) - cy(tail[0])) / dt


def link_visit(data, gap_max: float, err_max: float, ratio_max: float):
    """Which of a visit's tracks are the same animal. Returns groups of ids."""
    fps = data["fps"]
    tracks = {t: sorted(bs, key=lambda b: b[0]) for t, bs in data["tracks"].items()}
    span = {t: (bs[0][0], bs[-1][0]) for t, bs in tracks.items()}

    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    links = []
    for a in tracks:
        for b in tracks:
            if a == b:
                continue
            a1, b0 = span[a][1], span[b][0]
            if a1 >= b0:
                continue                       # overlapping in time: two animals
            gap = (b0 - a1) / fps
            if gap < 0 or gap > gap_max:
                continue
            A, B = tracks[a], tracks[b]
            vx, vy = tail_velocity(A, fps)
            ax = (A[-1][1] + A[-1][3]) / 2
            ay = (A[-1][2] + A[-1][4]) / 2
            bx = (B[0][1] + B[0][3]) / 2
            by = (B[0][2] + B[0][4]) / 2
            err = ((ax + vx * gap - bx) ** 2 + (ay + vy * gap - by) ** 2) ** 0.5
            wa, wb = A[-1][3] - A[-1][1], B[0][3] - B[0][1]
            width = max((wa + wb) / 2, 1.0)
            if err / width > err_max:
                continue
            # A weak secondary check only. A running macaque's box changes a
            # lot with pose — legs out, legs tucked — so size is not the
            # distance cue it looks like: on the ground-truth run the box went
            # 184px wide to 95px in a fifth of a second while the velocity
            # extrapolation was still within 0.30 of a box width. Set loose
            # enough to let real motion through; the velocity term is what
            # actually decides.
            ratio = max(wa, wb) / max(min(wa, wb), 1.0)
            if ratio > ratio_max:
                continue
            union(a, b)
            links.append({"a": a, "b": b, "gap": round(gap, 3),
                          "err_widths": round(err / width, 2)})

    groups = collections.defaultdict(list)
    for t in tracks:
        groups[find(t)].append(t)
    return [sorted(v) for v in groups.values() if len(v) > 1], links


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--err", type=float, default=1.0)
    ap.add_argument("--ratio", type=float, default=2.5)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    out, stats = {}, collections.Counter()
    sizes = collections.Counter()
    for f in sorted(MERGED.glob("*.boxes.json")):
        data = json.loads(f.read_text())
        groups, links = link_visit(data, args.gap, args.err, args.ratio)
        stats["visits"] += 1
        if groups:
            stats["visits with a chain"] += 1
        stats["tracks"] += len(data["tracks"])
        stats["tracks absorbed"] += sum(len(g) - 1 for g in groups)
        for g in groups:
            sizes[len(g)] += 1
        if groups:
            out[data["visit"]] = {"groups": groups, "links": links}

    print(f"造訪 {stats['visits']}，其中 {stats['visits with a chain']} 個有可串的軌跡")
    print(f"軌跡 {stats['tracks']} 條 → 可吸收掉 {stats['tracks absorbed']} 條 "
          f"（{stats['tracks absorbed'] / max(stats['tracks'], 1) * 100:.1f}%）")
    print(f"鏈長分佈: {dict(sorted(sizes.items()))}")
    dest = BASE / "track_links.json"
    dest.write_text(json.dumps({"gap": args.gap, "err": args.err, "visits": out}))
    print(f"寫到 {dest}")


if __name__ == "__main__":
    main()
