"""Perceptual deduplication — the training set's redundancy filter.

Why this exists (measured 2026-08-03): the cameras are fixed, so the corpus
re-shoots the same tree, wall and patch of sky for months. An audit of the
master dataset (tests/analyze_redundancy.py) found **27% of it is perceptual
near-duplicates** (dHash Hamming <= 5). A controlled experiment — train-only
dedup 688 -> 514 images (-25%), negative ratio held constant, identical
val/hyperparameters/exam — trained a BETTER model: test recall 0.704 -> 0.756,
F1 0.7979 -> 0.8320. Near-duplicates dilute the training signal rather than
augmenting it.

Hence the architecture this module serves: **master + derived**. The platform
dataset `monkey cocotree` is the MASTER — everything is added there, nothing is
ever deleted or deduplicated in it, and test/val stay pinned there so the
ratchet keeps sitting the same exam. Training runs on a DERIVED dataset,
regenerated from the master with the TRAIN split deduplicated.

**Never dedup val or test.** Dropping an image from the exam moves the exam,
which invalidates every historical F1.

Ported verbatim (same threshold, same survivor rule) from the experiment that
won, so the production path and the experiment cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import config

K = TypeVar("K")

# Sentinel "box area" for an image with no boxes. Larger than any real relative
# area (which is <= 1.0), so a negative never wins the smallest-box tiebreak
# against a positive — but negatives still compare equal to each other.
_NO_BOX_AREA = 9.0


def dhash(source: bytes | bytearray | str | Path, size: int = 8) -> int | None:
    """64-bit difference hash — robust to re-compression, sensitive to content.

    Grayscale, resize to (size+1) x size, then one bit per adjacent-column
    comparison. Accepts raw image bytes (what the platform hands us) or a path.
    Returns None when the bytes are not a decodable image.
    """
    import cv2
    import numpy as np

    if isinstance(source, (bytes, bytearray)):
        buf = np.frombuffer(bytes(source), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None
    small = cv2.resize(img, (size + 1, size))
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | int(small[r, c + 1] > small[r, c])
    return bits


def hamming(a: int, b: int) -> int:
    """Number of differing bits — the near-duplicate distance."""
    return bin(a ^ b).count("1")


def group_near_duplicates(
    items: Iterable[tuple[K, int]], threshold: int | None = None
) -> list[list[K]]:
    """Cluster (key, dHash) pairs into near-duplicate groups.

    Greedy single pass in input order: the first unclaimed key seeds a group and
    absorbs every later key within `threshold` bits. Deliberately NOT transitive
    closure — chaining would merge a whole day of gradual light change into one
    group and throw away real variation.

    Every key appears in exactly one group and **singletons are included**, so
    the result is a partition: `len(groups)` is the deduplicated size and
    `len(items) - len(groups)` is what dedup drops. Callers that only want the
    redundancy clusters filter for `len(g) > 1`.
    """
    if threshold is None:
        threshold = config.DEDUP_THRESHOLD
    pairs = list(items)
    hashes = dict(pairs)
    keys = [k for k, _ in pairs]
    seen: set[K] = set()
    groups: list[list[K]] = []
    for i, a in enumerate(keys):
        if a in seen:
            continue
        group = [a]
        seen.add(a)
        for b in keys[i + 1:]:
            if b not in seen and hamming(hashes[a], hashes[b]) <= threshold:
                group.append(b)
                seen.add(b)
        groups.append(group)
    return groups


def _boxes_of(meta: Any) -> list[Sequence[float]]:
    """Boxes out of a metadata entry: a record dict or a bare box list."""
    if meta is None:
        return []
    if isinstance(meta, Mapping):
        return list(meta.get("boxes") or [])
    return list(meta)


def _box_area(box: Sequence[float]) -> float:
    """Relative area of a YOLO box: [cls, cx, cy, w, h] (or [cx, cy, w, h])."""
    if len(box) >= 5:
        return float(box[3]) * float(box[4])
    if len(box) == 4:
        return float(box[2]) * float(box[3])
    return _NO_BOX_AREA


def pick_representative(group: Sequence[K], metadata: Mapping[K, Any]) -> K:
    """The one image of a near-duplicate group worth training on.

    Exactly the rule the winning experiment used:
      1. most boxes  — the busiest frame carries the most supervision
      2. then smallest median box — small/distant monkeys are the hard cases
      3. then the key itself, so the choice is stable across runs

    Ranking within a group also preserves its composition: a positives-only
    group keeps a positive and a negatives-only group keeps a negative, so
    dedup cannot silently swing the negative ratio of the training set.
    """

    def informativeness(key: K) -> tuple[int, float, str]:
        boxes = _boxes_of(metadata.get(key))
        if boxes:
            areas = sorted(_box_area(b) for b in boxes)
            median = areas[len(areas) // 2]
        else:
            median = _NO_BOX_AREA
        return (-len(boxes), median, str(key))

    return min(group, key=informativeness)
