"""Smoke test for backend/services/platform.py using httpx.MockTransport.

Verifies, without any network:
  - every Platform API call carries the Authorization: Bearer header
  - the signed-url PUT does NOT carry the API key (the URL is the credential)
  - the 2026-08 DataSys addressing: owner-scoped listings, {owner}/{url-name}
    sub-resources, and top-level images/ write routes keyed by ObjectId
  - datasets/{owner}/{dataset}/ingest retries on 409, once per upload session
  - list_models normalizes status + metrics
  - dispatch_promote refuses cleanly when GITHUB_DISPATCH_TOKEN is unset,
    and hits the workflow-dispatch endpoint with the token when set
  - dispatch_evaluate targets model-evaluate.yml
  - list_dataset_images sends includeLabels=true, converts center-format
    bboxes to top-left relative coords, reports split counts, and serves
    repeat calls from the 60 s cache
  - upload_files runs the signed-url → PUT → complete → ingest flow per file
  - training_status caches the passthrough for 1 s (N viewers, one call)
  - update_image_labels converts top-left boxes back to YOLO center format
    (exact inverse of list_dataset_images) and invalidates the images cache
  - move_images_split sends {imageIds,split:TARGET} to images/bulk and
    delete_images sends {imageIds}, both invalidating the caches; the hash the
    dashboard uses is resolved to an ObjectId inside the client
  - services/dedup.py: dHash survives re-encoding and downscaling, hamming,
    grouping partitions with singletons, representative picks most boxes then
    smallest median box
  - regenerating the derived dataset: export → image download → TRAIN-only
    dedup → zip with a per-record `split` manifest → new dataset → ingest
  - add_detections_to_dataset reports near-duplicates WITHOUT dropping them
  - start_training(dataset=…) points the ul:// data URI at the derived slug

Run:  uv run --project backend python tests/test_platform_smoke.py
(also collectable by pytest)
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

import httpx  # noqa: E402

import config  # noqa: E402

config.ULTRALYTICS_API_KEY = "ul_" + "a" * 40  # test key — never a real secret

from fastapi import HTTPException  # noqa: E402
from services import platform  # noqa: E402
from services import storage as storage_mod  # noqa: E402

API_KEY = config.ULTRALYTICS_API_KEY
DS_ID = "1ee3ac997a550c086574b565"

# The 2026-08 DataSys refactor addresses everything by {owner}/{url-name}; the
# id-based paths these tests used to assert no longer resolve at all.
OWNER = "submission-owner"
DS_SLUG = "monkey-cocotree-july"
PJ_SLUG = "monkeyv2"
DS_BASE = f"/api/datasets/{OWNER}/{DS_SLUG}"
# Name it from config rather than a literal: the default master moved from
# `monkey cocotree` to `monkey cocotree allin 0816` on 2026-08-17, and a stub
# pinned to the old string fails for a reason that has nothing to do with the
# behaviour under test.
DS_RECORD = {
    "id": DS_ID, "name": config.ULTRALYTICS_DATASET, "owner": OWNER, "dataset": DS_SLUG,
}
PJ_RECORD = {
    "id": "c4da2e3b258218a1c731bf00", "name": "monkey_v2",
    "owner": OWNER, "project": PJ_SLUG,
}


def _reset_platform_caches() -> None:
    """Module-level caches leak across tests otherwise — the owner lookup and
    the hash→id map are both process-wide by design."""
    platform._owner_cache.clear()
    platform._hash_ids.clear()
    platform._images_cache.clear()
    platform._status_cache.clear()


def _common_routes(
    request: httpx.Request,
    datasets: list[dict] | None = None,
    projects: list[dict] | None = None,
) -> httpx.Response | None:
    """The calls EVERY flow now makes: the owner lookup and the owner-scoped
    dataset/project listings. Returns None when the test should handle it.

    Asserts the Bearer header itself, since it answers before each test's own
    auth assertion gets a chance to."""
    url = str(request.url)
    if "/api/" in url and request.method != "PUT":
        assert request.headers.get("authorization") == f"Bearer {API_KEY}"
    if "/api/account/summary" in url:
        return httpx.Response(200, json={"username": OWNER})
    path = url.split("?")[0].rstrip("/")
    if path.endswith(f"/api/datasets/{OWNER}"):
        recs = datasets if datasets is not None else [DS_RECORD]
        return httpx.Response(200, json={"datasets": recs, "total": len(recs)})
    if path.endswith(f"/api/projects/{OWNER}"):
        recs = projects if projects is not None else [PJ_RECORD]
        return httpx.Response(200, json={"projects": recs, "total": len(recs)})
    return None


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    """Quacks like AsyncSession for the single select() the function issues."""

    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _query):
        return _FakeResult(self._rows)


def _run(coro):
    return asyncio.run(coro)


def _skip_duplicate_check() -> None:
    """Seed an EMPTY master-hash cache so the duplicate warning does no network.

    Tests that are about the upload flow itself would otherwise fire the
    duplicate check at their handler, whose AssertionError the warning path
    (correctly) swallows — hiding a real failure. The dedicated duplicate tests
    exercise that path for real.
    """
    import time

    platform._master_cache = (time.monotonic(), {})


# ── add_detections_to_dataset: auth header, PUT w/o auth, 409 retry ──────────


def test_add_to_dataset_upload_flow():
    seen = {"puts": [], "ingests": [], "signed": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        url = str(request.url)
        if request.method == "PUT":
            # Signed URL: MUST NOT leak the API key.
            assert "authorization" not in request.headers, "PUT must not carry the API key"
            # Ingest only processes archives — the flow must ship ONE zip.
            assert request.headers.get("content-type") == "application/zip"
            import io
            import zipfile
            zf = zipfile.ZipFile(io.BytesIO(request.content))
            names = zf.namelist()
            assert "cam1_t5_20260730_120000_ab12cd34.jpg" in names, names
            # Chinese camera name must be sanitized to pure ASCII (ingest
            # drops archives with non-ASCII entries) with a stable cam prefix
            import re as _re
            assert any(_re.fullmatch(r"cam[0-9a-f]{6}_2_t9_20260730_120100_ef56ab78\.jpg", n) for n in names), names
            assert all(n.isascii() for n in names) and len(names) == 2
            assert zf.read("cam1_t5_20260730_120000_ab12cd34.jpg") == b"fake-jpeg-bytes"
            seen["puts"].append(url)
            return httpx.Response(200)
        # Every Platform API call must be authenticated.
        assert request.headers.get("authorization") == f"Bearer {API_KEY}"
        if url.split("?")[0].endswith(f"{DS_BASE}/ingest"):
            body = json.loads(request.content)
            seen["ingests"].append(body["sessionId"])
            assert "datasetId" not in body, "dataset is in the PATH now"
            if len(seen["ingests"]) == 1:
                return httpx.Response(409, json={"error": "ingest already running"})
            return httpx.Response(200, json={"queued": True})
        if "/api/datasets" in url:
            return httpx.Response(200, json={"datasets": [DS_RECORD]})
        if "/api/upload/signed-url" in url:
            body = json.loads(request.content)
            assert body["assetType"] == "datasets" and body["assetId"] == DS_ID
            assert body["contentType"] == "application/zip"
            seen["signed"] += 1
            return httpx.Response(200, json={
                "uploadUrl": f"https://storage.example.com/put/{seen['signed']}",
                "sessionId": f"sess-{seen['signed']}",
            })
        if "/api/upload/complete" in url:
            return httpx.Response(200, json={"success": True})
        raise AssertionError(f"unexpected request: {request.method} {url}")

    platform._transport = httpx.MockTransport(handler)
    platform._INGEST_BACKOFF_SECONDS = 0  # no 15 s waits in tests
    _skip_duplicate_check()
    original_download = storage_mod.storage.download
    storage_mod.storage.download = lambda name: b"fake-jpeg-bytes"
    try:
        rows = [
            SimpleNamespace(id=1, image_path="cam1_t5_20260730_120000_ab12cd34.jpg"),
            SimpleNamespace(id=2, image_path="場域甲2_t9_20260730_120100_ef56ab78.jpg"),
        ]
        result = _run(platform.add_detections_to_dataset([1, 2, 999], _FakeDB(rows)))
    finally:
        storage_mod.storage.download = original_download
        platform._transport = None
        platform._INGEST_BACKOFF_SECONDS = 15.0

    assert result["uploaded"] == 2, result
    assert [f["id"] for f in result["failed"]] == [999], result  # unknown id reported
    assert result["duplicates"] == []
    assert len(seen["puts"]) == 1, "must upload exactly one zip"
    # one zip session; first ingest 409s then the retry lands
    assert seen["ingests"] == ["sess-1", "sess-1"]
    print("ok add_to_dataset: auth header, unauthenticated zip PUT, 409-retry, single ingest")


# ── list_models: status normalization + metrics shape ────────────────────────


def test_list_models():
    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        url = str(request.url)
        assert request.headers.get("authorization") == f"Bearer {API_KEY}"
        if f"/api/models/{OWNER}/{PJ_SLUG}" in url:
            # the project moved into the path; there is no projectId param
            assert "projectId" not in request.url.params
            return httpx.Response(200, json={"models": [
                {"id": "m1", "name": "exp-1", "status": "completed",
                 "metrics": {"precision": 0.95, "recall": 0.93, "mAP50": 0.96},
                 "createdAt": "2026-07-29T00:00:00Z"},
                {"id": "m2", "name": "exp-2", "status": "training"},
                {"id": "m3", "name": "exp-3", "status": "failed"},
            ]})
        raise AssertionError(f"unexpected request: {url}")

    platform._transport = httpx.MockTransport(handler)
    try:
        models = _run(platform.list_models("monkey_v2"))
    finally:
        platform._transport = None

    assert [m["status"] for m in models] == ["completed", "training", "failed"]
    assert models[0]["metrics"] == {"precision": 0.95, "recall": 0.93, "map50": 0.96}
    assert models[1]["metrics"] == {"precision": None, "recall": None, "map50": None}
    print("ok list_models: status normalized, metrics shaped")


# ── dispatch_promote: token guard + dispatch call ─────────────────────────────


def test_dispatch_promote():
    model_id = "6a" + "0" * 22

    config.GITHUB_DISPATCH_TOKEN = ""
    try:
        _run(platform.dispatch_promote(model_id))
        raise AssertionError("expected HTTPException when token unset")
    except HTTPException as exc:
        assert exc.status_code == 400 and "GITHUB_DISPATCH_TOKEN" in exc.detail

    # A token alone must not send a request to an implicit repository.
    def forbidden_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("dispatch must reject an unset repository before HTTP")

    config.GITHUB_DISPATCH_TOKEN = "ghp_test"
    config.GITHUB_REPO = ""
    platform._transport = httpx.MockTransport(forbidden_request)
    try:
        _run(platform.dispatch_promote(model_id))
        raise AssertionError("expected HTTPException when repository unset")
    except HTTPException as exc:
        assert exc.status_code == 400 and "GITHUB_REPO" in exc.detail
    finally:
        platform._transport = None
        config.GITHUB_DISPATCH_TOKEN = ""

    dispatched = {}

    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        assert "api.github.com" in str(request.url)
        assert request.url.path.startswith("/repos/submission-owner/submission/")
        assert request.headers.get("authorization") == "Bearer ghp_test"
        assert request.url.path.endswith("/actions/workflows/model-promote.yml/dispatches")
        dispatched.update(json.loads(request.content))
        return httpx.Response(204)

    config.GITHUB_DISPATCH_TOKEN = "ghp_test"
    config.GITHUB_REPO = "submission-owner/submission"
    platform._transport = httpx.MockTransport(handler)
    try:
        _run(platform.dispatch_promote(model_id))
        default_body = dict(dispatched)
        # 強制上板 rides the same workflow with one extra input. Booleans must go
        # over the wire as the STRINGS "true"/"false" — GitHub rejects a JSON
        # bool for a workflow_dispatch input, so a real `False` here would make
        # every forced promotion fail at dispatch with a 422.
        _run(platform.dispatch_promote(model_id, force=True))
        forced_body = dict(dispatched)
    finally:
        platform._transport = None
        config.GITHUB_DISPATCH_TOKEN = ""
        config.GITHUB_REPO = ""

    assert default_body == {
        "ref": "main",
        "inputs": {"model_id": model_id, "force": "false"},
    }
    assert forced_body == {
        "ref": "main",
        "inputs": {"model_id": model_id, "force": "true"},
    }
    for body in (default_body, forced_body):
        assert isinstance(body["inputs"]["force"], str)
    print("ok dispatch_promote: token guard + dispatch body + force as a string")


# ── list_dataset_images: labels param, box conversion, counts, cache ─────────


def test_list_dataset_images():
    seen = {"pages": 0, "count_probes": []}

    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        url = str(request.url)
        assert request.headers.get("authorization") == f"Bearer {API_KEY}"
        if f"{DS_BASE}/images" in url:
            params = request.url.params
            if params.get("limit") == "1":  # split-count probe
                seen["count_probes"].append(params.get("split", ""))
                totals = {"": 896, "train": 700, "val": 106, "test": 90}
                return httpx.Response(200, json={
                    "images": [], "total": totals[params.get("split", "")],
                    "hasMore": True,
                })
            # the page fetch MUST ask for labels, or no boxes come back
            assert params.get("includeLabels") == "true"
            assert params.get("split") == "test"
            assert params.get("offset") == "0" and params.get("limit") == "24"
            seen["pages"] += 1
            return httpx.Response(200, json={
                "images": [{
                    "id": "img1", "hash": "h1", "ext": "jpg",
                    "thumbnailUrl": "https://cdn.example/t/h1_256.webp?sig=x",
                    "width": 1920, "height": 1080, "split": "test",
                    "labelCount": 1, "name": "cam_x", "bytes": 1234,
                    # observed platform shape: normalised YOLO CENTER format
                    "labels": [{"classId": 0, "bbox": [0.5, 0.5, 0.2, 0.1]}],
                }, {
                    "id": "img2", "hash": "h2", "ext": "jpg",
                    "thumbnailUrl": "https://cdn.example/t/h2_256.webp?sig=y",
                    "width": 1920, "height": 1080, "split": "test",
                    "labelCount": 0, "name": "bg", "bytes": 999,
                }],
                "total": 90, "hasMore": True, "classes": ["monkey"],
                "errorCount": 0, "nextCursor": "img2",
            })
        if "/api/datasets" in url:
            return httpx.Response(200, json={"datasets": [DS_RECORD]})
        raise AssertionError(f"unexpected request: {request.method} {url}")

    platform._transport = httpx.MockTransport(handler)
    platform._images_cache.clear()
    try:
        page = _run(platform.list_dataset_images("test", 0, 24))
        again = _run(platform.list_dataset_images("test", 0, 24))  # must hit cache
    finally:
        platform._transport = None
        platform._images_cache.clear()

    assert page["total"] == 90 and page["has_more"] is True
    assert page["counts"] == {"all": 896, "train": 700, "val": 106, "test": 90}
    assert sorted(seen["count_probes"]) == ["", "test", "train", "val"]
    img = page["images"][0]
    assert img["id"] == "img1" and img["thumbnail_url"].startswith("https://cdn.example/")
    assert img["width"] == 1920 and img["height"] == 1080 and img["split"] == "test"
    box = img["boxes"][0]
    # center (0.5, 0.5, 0.2, 0.1) → top-left (0.4, 0.45), same size, cls 0
    assert abs(box["x"] - 0.4) < 1e-9 and abs(box["y"] - 0.45) < 1e-9
    assert box["w"] == 0.2 and box["h"] == 0.1 and box["cls"] == 0
    assert page["images"][1]["boxes"] == []  # negative image → empty overlay
    assert seen["pages"] == 1 and again == page, "second call must come from the 60 s cache"
    print("ok list_dataset_images: includeLabels, center→top-left boxes, counts, cache")


# ── upload_files: drag-and-drop flow reuses the platform upload pipeline ─────


def test_upload_files():
    seen = {"puts": [], "ingests": [], "signed": []}

    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        url = str(request.url)
        if request.method == "PUT":
            assert "authorization" not in request.headers, "PUT must not carry the API key"
            seen["puts"].append(request.headers.get("content-type"))
            return httpx.Response(200)
        assert request.headers.get("authorization") == f"Bearer {API_KEY}"
        if "/ingest" in url and request.method == "POST":
            body = json.loads(request.content)
            seen["ingests"].append(body["sessionId"])
            return httpx.Response(200, json={"queued": True})
        if "/api/datasets" in url:
            return httpx.Response(200, json={"datasets": [DS_RECORD]})
        if "/api/upload/signed-url" in url:
            body = json.loads(request.content)
            assert body["assetType"] == "datasets" and body["assetId"] == DS_ID
            seen["signed"].append((body["filename"], body["contentType"], body["totalBytes"]))
            n = len(seen["signed"])
            return httpx.Response(200, json={
                "uploadUrl": f"https://storage.example.com/put/{n}",
                "sessionId": f"sess-{n}",
            })
        if "/api/upload/complete" in url:
            return httpx.Response(200, json={"success": True})
        raise AssertionError(f"unexpected request: {request.method} {url}")

    platform._transport = httpx.MockTransport(handler)
    platform._images_cache.clear()
    _skip_duplicate_check()
    try:
        result = _run(platform.upload_files([
            ("field.jpg", b"jpeg-bytes", "image/jpeg"),
            ("bg.png", b"png-bytes", "image/png"),
        ]))
    finally:
        platform._transport = None
        platform._images_cache.clear()

    assert result == {"uploaded": 2, "failed": [], "duplicates": []}, result
    # both files travel in ONE zip archive: one signed-url, one PUT, one ingest
    assert len(seen["signed"]) == 1 and seen["signed"][0][1] == "application/zip"
    assert seen["puts"] == ["application/zip"]
    assert seen["ingests"] == ["sess-1"]
    print("ok upload_files: single zip archive, single ingest job")


# ── training_status: 1 s TTL cache collapses concurrent viewers ──────────────


def test_training_status_cache():
    model_id = "6b" + "1" * 22
    model_slug = "exp-42"
    seen = {"calls": 0, "listings": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        url = str(request.url)
        assert request.headers.get("authorization") == f"Bearer {API_KEY}"
        # the id is no longer addressable — it must be resolved to owner/project/model
        if url.split("?")[0].endswith(f"/api/models/{OWNER}/{PJ_SLUG}"):
            seen["listings"] += 1
            return httpx.Response(200, json={"models": [
                {"id": model_id, "name": "exp 42", "owner": OWNER,
                 "project": PJ_SLUG, "model": model_slug, "status": "training"},
            ]})
        assert f"/api/models/{OWNER}/{PJ_SLUG}/{model_slug}/training" in url, url
        seen["calls"] += 1
        return httpx.Response(200, json={"job": {"status": "training", "progress": {
            "currentEpoch": 42, "totalEpochs": 1000, "percentage": 4.2,
            "startedAt": "2026-07-30T00:00:00Z",
        }}})

    platform._transport = httpx.MockTransport(handler)
    _reset_platform_caches()
    try:
        first = _run(platform.training_status(model_id))
        second = _run(platform.training_status(model_id))
    finally:
        platform._transport = None
        platform._status_cache.clear()

    assert first["job"]["progress"]["currentEpoch"] == 42
    assert second == first
    assert seen["calls"] == 1, "second poll within 1 s must be served from cache"
    assert seen["listings"] == 1, "the id→slug resolution must be cached too"
    print("ok training_status: id→owner/project/model resolution + 1 s TTL cache")


# ── dispatch_evaluate: targets the evaluate workflow ──────────────────────────


def test_dispatch_evaluate():
    model_id = "6c" + "2" * 22
    dispatched = {}

    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        assert request.url.path.endswith("/actions/workflows/model-evaluate.yml/dispatches")
        assert request.url.path.startswith("/repos/submission-owner/submission/")
        assert request.headers.get("authorization") == "Bearer ghp_test"
        dispatched.update(json.loads(request.content))
        return httpx.Response(204)

    config.GITHUB_DISPATCH_TOKEN = "ghp_test"
    config.GITHUB_REPO = "submission-owner/submission"
    platform._transport = httpx.MockTransport(handler)
    try:
        _run(platform.dispatch_evaluate(model_id))
    finally:
        platform._transport = None
        config.GITHUB_DISPATCH_TOKEN = ""
        config.GITHUB_REPO = ""

    assert dispatched == {"ref": "main", "inputs": {"model_id": model_id}}
    print("ok dispatch_evaluate: model-evaluate.yml workflow dispatch")


# ── update_image_labels: top-left → center round-trip + cache invalidation ───


def test_update_image_labels():
    IMG_HASH = "6d12a197e72920155ffa050632e50d24"
    IMG_ID = "46cebd217b0dbc52b5d9c4cc"
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        url = str(request.url)
        assert request.headers.get("authorization") == f"Bearer {API_KEY}"
        # the write is a PATCH on the TOP-LEVEL image route, keyed by ObjectId
        if request.method == "PATCH":
            assert url.endswith(f"/api/images/{IMG_ID}"), url
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={
                "success": True, "labels": seen["labels"], "labelCount": len(seen["labels"]),
            })
        # the client resolves hash → id from the dataset listing
        if f"{DS_BASE}/images" in url:
            return httpx.Response(200, json={
                "images": [{"id": IMG_ID, "hash": IMG_HASH, "split": "train",
                            "width": 1280, "height": 720, "labelCount": 1,
                            "name": "x.jpg", "ext": "jpg"}],
                "hasMore": False, "classes": ["monkey"], "errorCount": 0,
            })
        raise AssertionError(f"unexpected request: {request.method} {url}")

    platform._transport = httpx.MockTransport(handler)
    _reset_platform_caches()
    platform._images_cache["stale"] = (0.0, {"images": []})  # must be flushed
    try:
        # top-left (0.4, 0.45, 0.2, 0.1) — the shape _normalize_image produces
        # from center [0.5, 0.5, 0.2, 0.1]; conversion back must be the inverse.
        result = _run(platform.update_image_labels(IMG_HASH, [
            {"x": 0.4, "y": 0.45, "w": 0.2, "h": 0.1, "cls": 0},
        ]))
    finally:
        platform._transport = None

    assert result == {"saved": True, "label_count": 1}, result
    assert seen["labels"] == [{"classId": 0, "bbox": [0.5, 0.5, 0.2, 0.1]}], seen
    assert "stale" not in platform._images_cache, "images cache must be invalidated"
    # malformed hash refused before any network call
    try:
        _run(platform.update_image_labels("not-a-hash", []))
        raise AssertionError("expected HTTPException for bad hash")
    except HTTPException as exc:
        assert exc.status_code == 400
    print("ok update_image_labels: center-format round-trip, cache flush, hash guard")


# ── move_images_split / delete_images: bulk shapes + cache invalidation ──────


def test_move_and_delete_images():
    IMG_HASH = "6d12a197e72920155ffa050632e50d24"
    IMG_ID = "46cebd217b0dbc52b5d9c4cc"
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        url = str(request.url)
        assert request.headers.get("authorization") == f"Bearer {API_KEY}"
        # both bulk ops are TOP-LEVEL now and take imageIds, not hashes
        if request.method == "PATCH":
            assert url.endswith("/api/images/bulk"), url
            seen["patch"] = json.loads(request.content)
            return httpx.Response(200, json={
                "success": True, "modifiedCount": 1, "skippedCount": 0, "targetSplit": "val",
            })
        if request.method == "DELETE":
            assert url.endswith("/api/images/bulk"), url
            seen["delete"] = json.loads(request.content)
            return httpx.Response(200, json={
                "success": True, "deletedCount": 1, "deletedHashes": [IMG_HASH],
            })
        if f"{DS_BASE}/images" in url:
            return httpx.Response(200, json={
                "images": [{"id": IMG_ID, "hash": IMG_HASH, "split": "train",
                            "width": 1280, "height": 720, "labelCount": 0,
                            "name": "x.jpg", "ext": "jpg"}],
                "hasMore": False, "classes": ["monkey"], "errorCount": 0,
            })
        raise AssertionError(f"unexpected request: {request.method} {url}")

    platform._transport = httpx.MockTransport(handler)
    _reset_platform_caches()
    platform._images_cache["stale"] = (0.0, {"images": []})
    try:
        moved = _run(platform.move_images_split(
            [{"hash": IMG_HASH, "split": "train"}], "val"
        ))
        assert "stale" not in platform._images_cache
        platform._images_cache["stale2"] = (0.0, {"images": []})
        deleted = _run(platform.delete_images([IMG_HASH]))
    finally:
        platform._transport = None
        platform._images_cache.clear()

    assert moved == {"moved": 1, "skipped": 0, "target_split": "val"}, moved
    # the per-image SOURCE split is gone — ids identify the doc on their own
    assert seen["patch"] == {"imageIds": [IMG_ID], "split": "val"}, seen["patch"]
    assert deleted == {"deleted": 1}, deleted
    assert seen["delete"] == {"imageIds": [IMG_ID]}, seen["delete"]
    print("ok move/delete images: top-level bulk routes keyed by imageIds")


# ── services/dedup.py — pure unit tests (no network, no platform) ────────────


def _jpeg(seed: int, quality: int = 90, size: tuple[int, int] = (320, 240)) -> bytes:
    """A deterministic JPEG with large-scale structure: a 12x9 random field
    upscaled bicubically. Big smooth blobs give every dHash cell a clearly
    distinct mean — like a real scene, and unlike flat noise, whose cell means
    are all ~127 and whose hash is therefore pure coin-flips."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    low = rng.integers(0, 256, (9, 12, 3), dtype=np.uint8)
    img = cv2.resize(low, size, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def test_dedup_unit():
    import cv2
    import numpy as np

    from services import dedup

    # ── dhash: stable across re-encoding AND across a thumbnail downscale ────
    original = _jpeg(7, quality=95)
    reencoded = _jpeg(7, quality=35)  # same pixels, much harsher compression
    h_orig = dedup.dhash(original)
    h_reenc = dedup.dhash(reencoded)
    assert h_orig is not None and h_reenc is not None
    # Usually identical; at most one marginal cell may flip at quality 35. What
    # has to hold is that the pair still reads as the SAME picture.
    drift = dedup.hamming(h_orig, h_reenc)
    assert drift <= 1, f"re-encode moved the hash by {drift} bits"
    assert dedup.group_near_duplicates([("a", h_orig), ("b", h_reenc)]) == [["a", "b"]]

    decoded = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_COLOR)
    thumb = cv2.imencode(".jpg", cv2.resize(decoded, (256, 192)))[1].tobytes()
    # The duplicate warning hashes THUMBNAILS against full-res images, so this
    # invariance is load-bearing, not a nicety.
    assert dedup.hamming(h_orig, dedup.dhash(thumb)) <= 2

    other = dedup.dhash(_jpeg(99))
    assert dedup.hamming(h_orig, other) > 5, "different scenes must not look alike"
    assert dedup.dhash(b"not-an-image") is None  # undecodable → None, never raises

    # ── hamming ─────────────────────────────────────────────────────────────
    assert dedup.hamming(0, 0) == 0
    assert dedup.hamming(0b1011, 0b0001) == 2
    assert dedup.hamming(0, (1 << 64) - 1) == 64

    # ── grouping: partition, singletons included, non-transitive ─────────────
    groups = dedup.group_near_duplicates(
        [("a", 0b0), ("b", 0b1), ("c", 0b11111), ("d", (1 << 64) - 1)], threshold=5
    )
    assert groups == [["a", "b", "c"], ["d"]], groups
    assert sum(len(g) for g in groups) == 4  # every key lands in exactly one group
    # threshold is a policy knob (config.DEDUP_HAMMING_THRESHOLD)
    assert dedup.group_near_duplicates([("a", 0b0), ("b", 0b111)], threshold=2) == [
        ["a"], ["b"],
    ]

    # ── representative: most boxes, then smallest median box, then key ───────
    meta = {
        "one_big": [[0, 0.5, 0.5, 0.40, 0.40]],
        "two_boxes": [[0, 0.5, 0.5, 0.40, 0.40], [0, 0.2, 0.2, 0.10, 0.10]],
        "one_tiny": [[0, 0.5, 0.5, 0.05, 0.05]],
        "negative": [],
    }
    assert dedup.pick_representative(
        ["one_big", "two_boxes", "negative"], meta
    ) == "two_boxes", "more boxes wins"
    assert dedup.pick_representative(["one_big", "one_tiny"], meta) == "one_tiny", \
        "on a tie, the smaller (harder) target wins"
    assert dedup.pick_representative(["negative", "one_tiny"], meta) == "one_tiny", \
        "a negative must never displace a positive"
    # all-negative group keeps a negative (composition preserved) + stable key
    assert dedup.pick_representative(["z", "a"], {"z": [], "a": []}) == "a"
    # median, not mean: 3 boxes → the middle area decides
    meta3 = {
        "wide_spread": [[0, 0, 0, 0.01, 0.01], [0, 0, 0, 0.5, 0.5], [0, 0, 0, 0.6, 0.6]],
        "tight": [[0, 0, 0, 0.2, 0.2], [0, 0, 0, 0.2, 0.2], [0, 0, 0, 0.2, 0.2]],
    }
    assert dedup.pick_representative(["wide_spread", "tight"], meta3) == "tight"
    print("ok dedup: dhash stability (re-encode + thumbnail), hamming, grouping, representative")


# ── regenerate_derived_dataset: master → deduplicated derived dataset ────────

NEW_DS_ID = "cb1f4c1906e69f201e51b1aa"


def _master_export(entries: list[tuple[str, str, list]]) -> str:
    """ndjson export text. entries: (content hash, split, boxes)."""
    lines = [json.dumps({"type": "dataset", "name": "monkey cocotree"})]
    for key, split, boxes in entries:
        lines.append(json.dumps({
            "type": "image",
            "file": f"images/{key}.jpg",
            "url": f"https://cdn.example/i/{key}.jpg?sig=abc",
            "width": 320, "height": 240, "split": split,
            "annotations": {"boxes": boxes},
        }))
    return "\n".join(lines) + "\n"


def test_regenerate_derived_dataset():
    from services import platform as plat

    box_big = [0, 0.5, 0.5, 0.4, 0.4]
    box_small = [0, 0.2, 0.2, 0.05, 0.05]
    # a1/a2/a3 are the SAME scene re-encoded → one near-duplicate group; a2 has
    # two boxes so it must be the survivor. b is a different scene, c a negative.
    fixtures = {
        "a" * 32: (_jpeg(1, 95), "train", [box_big]),
        "b" * 32: (_jpeg(1, 40), "train", [box_big, box_small]),   # a2, most boxes
        "c" * 32: (_jpeg(1, 70), "train", [box_small]),
        "d" * 32: (_jpeg(2), "train", [box_big]),
        "e" * 32: (_jpeg(3), "train", []),                          # negative
        "f" * 32: (_jpeg(4), "val", [box_big]),
        "0" * 32: (_jpeg(5), "val", []),
        "9" * 32: (_jpeg(6), "test", [box_big]),                    # must NOT be copied
    }
    export = _master_export([(k, v[1], v[2]) for k, v in fixtures.items()])
    seen: dict = {"created": None, "zip": None, "ingest": [], "landing": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        url = str(request.url)
        if request.method == "PUT":
            assert "authorization" not in request.headers
            assert request.headers.get("content-type") == "application/zip"
            seen["zip"] = request.content
            return httpx.Response(200)
        if url.startswith("https://cdn.example/i/"):
            key = url.split("/i/")[1].split(".")[0]
            return httpx.Response(200, content=fixtures[key][0])
        if url.startswith("https://cdn.example/export.ndjson"):
            return httpx.Response(200, text=export)
        assert request.headers.get("authorization") == f"Bearer {API_KEY}"
        if f"{DS_BASE}/export" in url:
            return httpx.Response(200, json={"cached": True,
                                             "downloadUrl": "https://cdn.example/export.ndjson"})
        derived = (seen["created"] or {}).get("dataset")
        if derived and f"/api/datasets/{OWNER}/{derived}/images" in url:
            seen["landing"] += 1
            # first probe: ingest still running; second: everything landed
            return httpx.Response(200, json={"images": [], "total": 0 if seen["landing"] == 1 else 5})
        if "/ingest" in url and request.method == "POST":
            body = json.loads(request.content)
            # the target dataset is in the PATH now — it must be the DERIVED one
            assert derived and f"/api/datasets/{OWNER}/{derived}/ingest" in url, url
            assert "datasetId" not in body
            seen["ingest"].append(body["sessionId"])
            return httpx.Response(200, json={"queued": True})
        if "/api/upload/signed-url" in url:
            body = json.loads(request.content)
            assert body["assetId"] == NEW_DS_ID and body["contentType"] == "application/zip"
            return httpx.Response(200, json={
                "uploadUrl": "https://storage.example.com/put/derived", "sessionId": "sess-d",
            })
        if "/api/upload/complete" in url:
            return httpx.Response(200, json={"success": True})
        if request.method == "POST" and url.rstrip("/").endswith("/api/datasets"):
            seen["created"] = json.loads(request.content)
            # the create response is itself a ref: {id, owner, dataset}
            return httpx.Response(201, json={"id": NEW_DS_ID, "owner": OWNER,
                                             "dataset": seen["created"]["dataset"]})
        raise AssertionError(f"unexpected request: {request.method} {url}")

    tmp = tempfile.TemporaryDirectory()
    original_cache = config.DEDUP_CACHE_DIR
    config.DEDUP_CACHE_DIR = Path(tmp.name) / "cache"
    plat._transport = httpx.MockTransport(handler)
    plat._LANDING_POLL_SECONDS = 0
    plat._INGEST_BACKOFF_SECONDS = 0

    async def drive():
        stored: list[dict] = []

        async def on_complete(result: dict) -> None:
            stored.append(result)

        started = await plat.start_regenerate(on_complete)
        assert started["state"] == "running"
        await plat._regen_task
        return stored

    try:
        stored = _run(drive())
    finally:
        plat._transport = None
        plat._LANDING_POLL_SECONDS = 10.0
        plat._INGEST_BACKOFF_SECONDS = 15.0
        plat._regen_task = None
        config.DEDUP_CACHE_DIR = original_cache
        tmp.cleanup()

    state = plat.regenerate_state()
    assert state["state"] == "done", state
    result = state["result"]
    assert stored == [result], "the completion callback must receive the result"

    # train 5 → 3 (a1/a2/a3 collapse to one), val copied untouched, test dropped
    assert result["train_before"] == 5 and result["kept"] == 3 and result["dropped"] == 2, result
    assert result["negatives_before"] == 1 and result["negatives_after"] == 1, \
        "dedup must not swing the negative ratio"
    assert result["val_images"] == 2 and result["expected_images"] == 5
    assert result["images_landed"] == 5 and seen["landing"] == 2  # polled until it landed
    assert result["dataset_slug"].startswith("monkey-cocotree-july-dedup-")
    assert seen["ingest"] == ["sess-d"]

    # POST /api/datasets: the URL name field is `dataset` (was `slug`)
    assert set(seen["created"]) == {"dataset", "name", "task", "visibility"}, seen["created"]
    assert seen["created"]["task"] == "detect" and seen["created"]["visibility"] == "private"
    assert seen["created"]["dataset"] == result["dataset_slug"]

    # The archive: images/ + a data.ndjson whose per-record split the ingest
    # worker honours (that is what keeps train/val identical to the master).
    zf = zipfile.ZipFile(io.BytesIO(seen["zip"]))
    names = zf.namelist()
    assert "data.ndjson" in names and all(n.isascii() for n in names)
    manifest = [json.loads(line) for line in zf.read("data.ndjson").decode().splitlines()]
    assert len(manifest) == 5
    by_split: dict[str, list] = {}
    for rec in manifest:
        by_split.setdefault(rec["split"], []).append(rec)
    assert sorted(by_split) == ["train", "val"], "test must never be copied to the derived set"
    assert len(by_split["train"]) == 3 and len(by_split["val"]) == 2
    files = {rec["file"] for rec in manifest}
    assert f"images/{'b' * 32}.jpg" in files, "survivor = the frame with most boxes"
    assert f"images/{'a' * 32}.jpg" not in files and f"images/{'c' * 32}.jpg" not in files
    assert f"images/{'9' * 32}.jpg" not in files  # the test split stayed home
    # only manifest images travel — no orphan bytes in the archive
    assert {n for n in names if n != "data.ndjson"} == files
    assert manifest[0]["annotations"]["boxes"] == [box_big, box_small]
    print("ok regenerate_derived_dataset: TRAIN-only dedup, split manifest, new dataset, ingest")


def test_regenerate_rejects_concurrent_runs():
    """Two runs would fight over the cache and create a second dataset."""
    from services import platform as plat

    async def drive():
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow():
            started.set()
            await release.wait()
            return {"dataset_slug": "x", "kept": 0, "dropped": 0}

        plat._regen_task = asyncio.create_task(slow())
        await started.wait()
        try:
            await plat.start_regenerate(None)
            raise AssertionError("expected 409 while a job is running")
        except HTTPException as exc:
            assert exc.status_code == 409 and "進行中" in exc.detail
        finally:
            release.set()
            await plat._regen_task
            plat._regen_task = None

    _run(drive())
    print("ok regenerate: concurrent run refused with 409")


# ── add_detections_to_dataset: near-duplicates reported, never dropped ───────


def test_add_to_dataset_reports_duplicates():
    from services import platform as plat

    dupe = _jpeg(11)              # byte-identical scene to the master's image
    fresh = _jpeg(12)             # nothing like it
    master_thumb = _jpeg(11, quality=60, size=(256, 192))  # what the CDN serves

    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        url = str(request.url)
        if url.startswith("https://cdn.example/t/"):
            return httpx.Response(200, content=master_thumb)
        if request.method == "PUT":
            return httpx.Response(200)
        assert request.headers.get("authorization") == f"Bearer {API_KEY}"
        if f"{DS_BASE}/images" in url:
            return httpx.Response(200, json={"images": [{
                "id": "i1", "hash": "1" * 32, "name": "master_bg.jpg",
                "thumbnailUrl": "https://cdn.example/t/1_256.webp?sig=x",
                "width": 320, "height": 240, "split": "train",
            }], "total": 1, "hasMore": False})
        if "/ingest" in url and request.method == "POST":
            return httpx.Response(200, json={"queued": True})
        if "/api/upload/signed-url" in url:
            return httpx.Response(200, json={
                "uploadUrl": "https://storage.example.com/put/1", "sessionId": "sess-1",
            })
        if "/api/upload/complete" in url:
            return httpx.Response(200, json={"success": True})
        if "/api/datasets" in url:
            return httpx.Response(200, json={"datasets": [DS_RECORD]})
        raise AssertionError(f"unexpected request: {request.method} {url}")

    blobs = {"dupe.jpg": dupe, "fresh.jpg": fresh}
    plat._transport = httpx.MockTransport(handler)
    plat._INGEST_BACKOFF_SECONDS = 0
    plat._invalidate_master_hashes()
    original_download = storage_mod.storage.download
    storage_mod.storage.download = lambda name: blobs[name]
    try:
        rows = [
            SimpleNamespace(id=1, image_path="dupe.jpg"),
            SimpleNamespace(id=2, image_path="fresh.jpg"),
        ]
        result = _run(plat.add_detections_to_dataset([1, 2], _FakeDB(rows)))
    finally:
        storage_mod.storage.download = original_download
        plat._transport = None
        plat._INGEST_BACKOFF_SECONDS = 15.0
        plat._invalidate_master_hashes()

    # BOTH images still upload — the master archives everything; the warning is
    # only so the user knows what will be collapsed out of the training set.
    assert result["uploaded"] == 2 and result["failed"] == [], result
    assert [d["name"] for d in result["duplicates"]] == ["dupe.jpg"], result["duplicates"]
    note = result["duplicates"][0]
    assert note["similar_to"] == "master_bg.jpg" and note["distance"] <= 5, note
    # a successful upload must drop the cached master hashes (they are now stale)
    assert plat._master_cache is None
    print("ok add_to_dataset: near-duplicate reported, still uploaded, master cache dropped")


def test_duplicate_check_never_blocks_upload():
    """The master listing failing must not fail the upload — it is a warning."""
    from services import platform as plat

    def handler(request: httpx.Request) -> httpx.Response:
        common = _common_routes(request)
        if common is not None:
            return common
        url = str(request.url)
        if request.method == "PUT":
            return httpx.Response(200)
        if f"{DS_BASE}/images" in url:
            return httpx.Response(500, text="master listing exploded")
        if "/ingest" in url and request.method == "POST":
            return httpx.Response(200, json={"queued": True})
        if "/api/upload/signed-url" in url:
            return httpx.Response(200, json={
                "uploadUrl": "https://storage.example.com/put/1", "sessionId": "sess-1",
            })
        if "/api/upload/complete" in url:
            return httpx.Response(200, json={"success": True})
        if "/api/datasets" in url:
            return httpx.Response(200, json={"datasets": [DS_RECORD]})
        raise AssertionError(f"unexpected request: {request.method} {url}")

    plat._transport = httpx.MockTransport(handler)
    plat._invalidate_master_hashes()
    try:
        result = _run(plat.upload_files([("bg.jpg", _jpeg(21), "image/jpeg")]))
    finally:
        plat._transport = None
        plat._invalidate_master_hashes()

    assert result == {"uploaded": 1, "failed": [], "duplicates": []}, result
    print("ok duplicate check: master fetch failure degrades to no warning")


# ── start_training(dataset=…): the derived slug reaches the ul:// data URI ────


def test_start_training_on_derived_dataset():
    derived_slug = "monkey-cocotree-july-dedup-20260803-120000"
    seen: dict = {"train": [], "created": []}
    derived_record = {
        "id": NEW_DS_ID, "name": "monkey cocotree dedup 20260803-120000",
        "owner": OWNER, "dataset": derived_slug,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        # the derived dataset must be visible in the owner-scoped listing
        common = _common_routes(request, datasets=[DS_RECORD, derived_record])
        if common is not None:
            return common
        url = str(request.url)
        assert request.headers.get("authorization") == f"Bearer {API_KEY}"
        if "/api/training/start" in url:
            seen["train"].append(json.loads(request.content))
            return httpx.Response(200, json={"jobId": "job-1"})
        if request.method == "POST" and url.rstrip("/").endswith("/api/models"):
            seen["created"].append(json.loads(request.content))
            return httpx.Response(201, json={"modelId": "6d" + "3" * 22})
        raise AssertionError(f"unexpected request: {request.method} {url}")

    platform._transport = httpx.MockTransport(handler)
    _reset_platform_caches()
    try:
        out = _run(platform.start_training("exp-dedup", dataset=derived_slug))
        default = _run(platform.start_training("exp-master"))
    finally:
        platform._transport = None

    assert out["modelId"] == "6d" + "3" * 22 and default["modelId"]
    assert seen["train"][0]["trainArgs"]["data"] == (
        f"ul://{OWNER}/datasets/{derived_slug}"
    )
    master_uri = seen["train"][1]["trainArgs"]["data"]
    assert master_uri == f"ul://{OWNER}/datasets/{DS_SLUG}", \
        "no dataset argument must still train on the master"
    # POST /models identifies the project by URL NAME now, not projectId
    assert seen["created"][0] == {
        "owner": OWNER, "project": PJ_SLUG, "model": "exp-dedup",
    }, seen["created"][0]
    assert all("projectId" not in b for b in seen["train"]), \
        "training/start no longer takes projectId"
    print("ok start_training: derived slug in ul:// URI, project by URL name")


if __name__ == "__main__":
    test_add_to_dataset_upload_flow()
    test_list_models()
    test_dispatch_promote()
    test_list_dataset_images()
    test_upload_files()
    test_training_status_cache()
    test_dispatch_evaluate()
    test_update_image_labels()
    test_move_and_delete_images()
    test_dedup_unit()
    test_regenerate_derived_dataset()
    test_regenerate_rejects_concurrent_runs()
    test_add_to_dataset_reports_duplicates()
    test_duplicate_check_never_blocks_upload()
    test_start_training_on_derived_dataset()
    print("all platform smoke tests passed")
