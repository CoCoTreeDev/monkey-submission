"""LocalStorage must support everything the orphan sweep calls.

Production used BlobStorage until 2026-09-21, so LocalStorage's listing path
had never run in anger — and list_files_with_mtime referenced an undefined
module name, failing the sweep daily on the mini.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from services.storage import LocalStorage  # noqa: E402


def test_list_files_with_mtime_reads_its_own_directory(tmp_path) -> None:
    store = LocalStorage(tmp_path / "snaps")
    store.upload("a.jpg", b"12345")
    store.upload("b.mp4", b"x")
    (tmp_path / "snaps" / "subdir").mkdir()          # directories are skipped

    rows = {name: (size, mtime) for name, size, mtime in store.list_files_with_mtime()}
    assert set(rows) == {"a.jpg", "b.mp4"}
    assert rows["a.jpg"][0] == 5
    assert abs(rows["a.jpg"][1] - time.time()) < 60


def test_listing_ignores_siblings_of_the_store(tmp_path) -> None:
    """The mini keeps a quarantine dir next to the store; the sweep must not see it."""
    (tmp_path / "quarantine").mkdir()
    (tmp_path / "quarantine" / "keep.jpg").write_bytes(b"k")
    store = LocalStorage(tmp_path / "snapshots")
    store.upload("live.jpg", b"l")
    assert [n for n, *_ in store.list_files_with_mtime()] == ["live.jpg"]
