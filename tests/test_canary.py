"""The canary's only non-trivial logic is the transition filter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infra"))

import canary  # noqa: E402


def transitions(was: set[str], now: set[str]) -> list[str]:
    """Mirror of the alert lines main() builds, which is the part worth testing."""
    return [f"🔴 `{p}`" for p in sorted(now - was)] + [
        f"🟢 `{p}` recovered" for p in sorted(was - now)
    ]


def test_only_new_breakage_alerts() -> None:
    # Still broken on the next pass -> silence, not a second ping.
    assert transitions({"/a"}, {"/a"}) == []
    assert transitions(set(), {"/a"}) == ["🔴 `/a`"]
    assert transitions({"/a"}, set()) == ["🟢 `/a` recovered"]
    # A new failure while an old one persists reports only the new one.
    assert transitions({"/a"}, {"/a", "/b"}) == ["🔴 `/b`"]


def test_expensive_routes_are_not_polled() -> None:
    # download-images zips every snapshot; polling it would cause the outage.
    assert "/api/admin/download-images" in canary.SKIP


def test_env_reads_values_and_tolerates_absence(tmp_path, monkeypatch) -> None:
    envfile = tmp_path / ".env"
    envfile.write_text('# comment\nCLOUD_API_URL="https://example.test/"\nOTHER=x\n')
    monkeypatch.setattr(canary, "ENV", envfile)
    assert canary.env("CLOUD_API_URL") == "https://example.test/"
    assert canary.env("MISSING") == ""


def test_state_file_round_trips(tmp_path) -> None:
    state = tmp_path / "canary-state.json"
    state.write_text(json.dumps({"broken": ["/a", "/b"], "at": 1.0}))
    assert set(json.loads(state.read_text())["broken"]) == {"/a", "/b"}


def test_empty_state_is_not_an_outage() -> None:
    # rebuild-review 404s whenever no backfill is pending — the normal case.
    assert 404 in canary.ALSO_OK["/api/admin/rebuild-review"]
    assert canary.ALSO_OK.get("/api/health", ()) == ()


def test_snapshots_dir_is_overridable(monkeypatch, tmp_path) -> None:
    """The mini points this at the 17.4 GB blob mirror instead of the repo."""
    import importlib, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    monkeypatch.setenv("SNAPSHOTS_DIR", str(tmp_path / "mirror" / "snapshots"))
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    config = importlib.reload(importlib.import_module("config"))
    assert config.SNAPSHOTS_DIR == tmp_path / "mirror" / "snapshots"
    assert config.SNAPSHOTS_DIR.is_dir(), "must be created, including parents"
    monkeypatch.delenv("SNAPSHOTS_DIR")
    importlib.reload(config)


def test_snapshot_route_is_sampled_from_a_real_detection() -> None:
    """The storage path must be covered — it is what the mini migration swaps.

    Plain routes never touch a stored file, so a broken snapshot path after
    cutover would leave the canary fully green.
    """
    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    class _Client:
        def get(self, url, **kw):
            assert "/api/detections?limit=1" in url
            return _Resp({"items": [{"id": 5, "image_path": "場域丙_a b.jpg"}]})

    sampled = canary._sampled_paths(_Client(), "https://x.test", {})
    assert len(sampled) == 1
    label, path = sampled[0]
    assert label == "/api/snapshots/{filename}"
    assert path == "/api/snapshots/%E5%A0%B4%E5%9F%9F%E4%B8%99_a%20b.jpg", "must be URL-encoded"


def test_sampling_is_silent_when_the_listing_is_broken() -> None:
    class _Boom:
        def get(self, *a, **kw):
            raise RuntimeError("listing is down")

    # The dead listing is already reported by its own probe; sampling must not
    # turn one outage into three.
    assert canary._sampled_paths(_Boom(), "https://x.test", {}) == []


def test_archive_builders_are_never_polled() -> None:
    assert "/api/admin/download-images" in canary.SKIP
    # 38 s and a 41 MB zip per call — covered by /api/snapshots instead.
    assert "/api/detections/{detection_id}/frames" in canary.SKIP
