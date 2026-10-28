"""Shared policy for deciding whether a track is real enough to persist."""

from __future__ import annotations


def is_substantial_track(
    hit_count: int,
    observed_seconds: float,
    *,
    min_hits: int,
    min_seconds: float,
) -> bool:
    """Require repeated evidence over time, not just a high-confidence flash."""
    return hit_count >= min_hits and observed_seconds >= min_seconds
