"""Regression checks for the isolated historical rebuild renderer."""

from pathlib import Path
from tempfile import TemporaryDirectory

from rebuild_clips import merge_track_fragments, source_for


def test_raw_archive_wins_over_fetched_trail() -> None:
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        run = base / "run"
        (base / "zips").mkdir(parents=True)
        (run / "fetched").mkdir(parents=True)
        item = {"trail_path": "cam_trail.mp4", "archive": "cam_rawframes.zip"}
        (run / "fetched" / item["trail_path"]).touch()
        (base / "zips" / item["archive"]).touch()
        path, kind = source_for(item, run)
        assert path == base / "zips" / item["archive"]
        assert kind == "rawframes"
        path, kind = source_for(item, run, allow_trail_fallback=False)
        assert path == base / "zips" / item["archive"]
        assert kind == "rawframes"
        missing = {"trail_path": "missing_trail.mp4", "archive": "missing_rawframes.zip"}
        path, kind = source_for(missing, run, allow_trail_fallback=False)
        assert path is None and kind is None


def test_duplicate_and_short_forward_fragments_collapse() -> None:
    tracks = {
        # Same animal, two simultaneous ByteTrack ids.
        1: [[0, 0, 0, 20, 20, 0.90], [1, 1, 0, 21, 20, 0.91]],
        2: [[0, 0.5, 0, 20.5, 20, 0.80], [1, 1.5, 0, 21.5, 20, 0.82]],
        # Same animal after a short gap, but no overlapping frame.
        3: [[4, 3, 0, 23, 20, 0.75], [5, 4, 0, 24, 20, 0.76]],
        # A concurrent animal is far away and must survive.
        4: [[1, 100, 100, 125, 125, 0.88], [2, 101, 100, 126, 125, 0.89]],
    }
    merged = merge_track_fragments(tracks)
    assert len(merged) == 2
    assert sorted(len(boxes) for boxes in merged.values()) == [2, 4]


def test_long_gap_can_join_alone_but_not_swallow_concurrent_track() -> None:
    # The object reappears after a long detector gap and is the only track
    # alive in that interval, so it is safe for the review rebuild to join it.
    alone = {
        2: [[0, 100, 100, 130, 140, 0.9], [4, 100, 90, 130, 130, 0.9]],
        6: [[28, 90, 10, 120, 50, 0.9], [30, 90, 5, 120, 45, 0.9]],
    }
    assert len(merge_track_fragments(alone)) == 1

    # A third track remains alive through the long gap. Do not use proximity
    # alone to collapse it into one of the two real animals.
    concurrent = {
        1: [[0, 100, 100, 130, 140, 0.9], [4, 100, 90, 130, 130, 0.9]],
        2: [[8, 400, 100, 430, 140, 0.9], [30, 400, 90, 430, 130, 0.9]],
        8: [[28, 190, 80, 220, 120, 0.9], [30, 190, 70, 220, 110, 0.9]],
    }
    assert len(merge_track_fragments(concurrent)) == 3


if __name__ == "__main__":
    test_raw_archive_wins_over_fetched_trail()
    test_duplicate_and_short_forward_fragments_collapse()
    test_long_gap_can_join_alone_but_not_swallow_concurrent_track()
    print("ok rebuild_clips: raw source precedence and fragment dedupe")
