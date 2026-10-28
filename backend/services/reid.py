"""Small, deterministic helpers for the ReID research queue.

These decisions are deliberately separate from ``ReIDPairLabel``.  A geometry
shortcut can remove an obvious negative from the human queue, but it is not a
human label and must not become training ground truth by accident.
"""

from __future__ import annotations

from math import hypot, isfinite
from typing import Any, Literal


ReIDSystemLabel = Literal["different"]


def _box_by_frame(track: dict[str, Any]) -> dict[int, tuple[float, float, float, float]]:
    """Return valid frame boxes, keeping the first observation per frame."""
    boxes: dict[int, tuple[float, float, float, float]] = {}
    for value in track.get("track_boxes") or []:
        if not isinstance(value, (list, tuple)) or len(value) < 5:
            continue
        try:
            frame = int(value[0])
            x1, y1, x2, y2 = (float(value[index]) for index in range(1, 5))
        except (TypeError, ValueError):
            continue
        if not all(isfinite(number) for number in (x1, y1, x2, y2)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.setdefault(frame, (x1, y1, x2, y2))
    return boxes


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _centre_distance(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    first_centre = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_centre = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    first_diagonal = hypot(first[2] - first[0], first[3] - first[1])
    second_diagonal = hypot(second[2] - second[0], second[3] - second[1])
    scale = (first_diagonal + second_diagonal) / 2
    return hypot(first_centre[0] - second_centre[0], first_centre[1] - second_centre[1]) / scale if scale > 0 else float("inf")


def reid_geometry_features(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float | int | None]:
    """Measure how two tracklets relate in their shared video frames."""
    first_boxes = _box_by_frame(first)
    second_boxes = _box_by_frame(second)
    common_frames = sorted(first_boxes.keys() & second_boxes.keys())
    if not common_frames:
        return {
            "common_frames": 0,
            "max_iou": None,
            "min_center_distance": None,
        }

    return {
        "common_frames": len(common_frames),
        "max_iou": max(_iou(first_boxes[frame], second_boxes[frame]) for frame in common_frames),
        "min_center_distance": min(_centre_distance(first_boxes[frame], second_boxes[frame]) for frame in common_frames),
    }


def reid_system_decision(
    candidate: dict[str, Any],
) -> tuple[ReIDSystemLabel | None, str | None]:
    """Return only a high-precision automatic negative, never an automatic same.

    The thresholds are calibrated against the current 51 human seeds.  On the
    current seed they accepted 30 overlap pairs and all 30 were ``different``;
    all four overlap ``uncertain`` examples stayed outside the rule.  The
    decision is intentionally conservative and remains auditable in the API.
    """
    if candidate.get("temporal_relation") != "overlap":
        return None, None
    features = reid_geometry_features(candidate["a"], candidate["b"])
    common_frames = features["common_frames"]
    max_iou = features["max_iou"]
    min_distance = features["min_center_distance"]
    if (
        isinstance(common_frames, int)
        and common_frames >= 2
        and isinstance(max_iou, (float, int))
        and max_iou <= 0.01
        and isinstance(min_distance, (float, int))
        and min_distance >= 0.8
    ):
        return "different", "同時出現，但至少兩幀的框持續分離"
    return None, None


def reid_review_priority(candidate: dict[str, Any]) -> tuple[int, str]:
    """Rank unresolved pairs by expected information gain for a human.

    This is active-review triage, not a learned identity model.  Separate
    tracklets with a short gap and a small transition are the best remaining
    positive candidates; overlapping tracklets near the geometry boundary are
    the most useful hard negatives/uncertain examples.
    """
    features = reid_geometry_features(candidate["a"], candidate["b"])
    relation = candidate.get("temporal_relation")
    common_frames = int(features["common_frames"] or 0)
    max_iou = float(features["max_iou"] or 0.0)
    min_distance = features["min_center_distance"]

    if relation == "separate":
        gap = candidate.get("gap_seconds")
        gap_score = max(0.0, 1.0 - min(float(gap), 10.0) / 10.0) if isinstance(gap, (int, float)) else 0.0
        # The first/last boxes are the closest available evidence when the
        # tracklets do not overlap in time.
        first_boxes = _box_by_frame(candidate["a"])
        second_boxes = _box_by_frame(candidate["b"])
        transition_distance = None
        if first_boxes and second_boxes:
            transition_distance = _centre_distance(first_boxes[max(first_boxes)], second_boxes[min(second_boxes)])
        transition_score = max(0.0, 1.0 - min(transition_distance or 10.0, 10.0) / 10.0)
        priority = round(55 + 30 * gap_score + 15 * transition_score)
        return priority, "時間分開但間隔／位移較小，優先確認是否同一隻"

    if relation == "overlap":
        # Pairs close to the auto-different boundary are deliberately kept for
        # humans.  More shared frames make that judgement more observable.
        if common_frames < 2 or min_distance is None:
            return 30, "時間重疊，但逐幀框證據不足，請人工確認"
        boundary_score = 1.0 - min(abs(float(min_distance or 0.0) - 0.8), 0.8) / 0.8
        overlap_score = min(max_iou / 0.5, 1.0)
        frame_score = min(common_frames / 12, 1.0)
        priority = round(45 + 35 * boundary_score + 15 * overlap_score + 5 * frame_score)
        return priority, "同時出現且接近判定邊界，優先確認外觀與連續動作"

    return 20, "逐幀框或時間資訊不足，保留人工判斷"
