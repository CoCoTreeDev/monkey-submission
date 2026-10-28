"""Ultralytics Platform API client — the dashboard's model-lifecycle bridge.

The Platform (https://platform.ultralytics.com) is the single source of truth
for datasets, models and training runs; this module is the async backend for
the 訓練 tab.

Requests go through the official **`ultralytics-platform` SDK** (`AsyncPlatform`),
which owns URL assembly, auth, typed responses and retries. That is why nothing
here builds paths any more: resources are addressed by passing `(owner, url-name)`
to a typed resource method. Two facts about that addressing still matter to us:

- the record field holding the URL name is named after the resource —
  `dataset` / `project` / `model`, never `slug` (see `_slug`/`_ref`);
- models nest under their project, so their calls take three segments.

What deliberately stays hand-rolled:

- `_bare_client` — the signed upload URL is its own credential and must NOT
  receive the API key. The SDK hands back the URL; it does not PUT the bytes.
- the hash→id map (`_hash_to_ids`) — write endpoints key images by ObjectId
  while the dashboard keys them by the 32-hex content hash in ~14 places, so
  the translation lives here rather than rippling into the UI.
- the 409 retry loop on ingest — the SDK only auto-retries idempotent verbs,
  and ingest is a POST that 409s while another job runs on the same dataset.
- `_as_http_error` — every public function raises HTTPException with a readable
  zh-TW detail. The API key must NEVER appear in errors or logs, so details only
  ever carry the platform's response status/body, never the request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ultralytics_platform import APIConnectionError, APIError, AsyncPlatform

import config
from services import dedup

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

# 409 on datasets/ingest means an ingest job is already running for the
# dataset — verified behavior; wait and retry (tests patch these to run fast).
_INGEST_MAX_ATTEMPTS = 20
_INGEST_BACKOFF_SECONDS = 15.0

# Test seam: httpx.MockTransport goes here; None = real network.
_transport: httpx.AsyncBaseTransport | None = None

# In-process TTL caches (single replica / single edge node, so this is enough).
# _status_cache: N dashboard viewers polling training progress at 1 Hz must
# not multiply platform calls — 1 s TTL collapses them to one upstream call.
# _images_cache: the 資料集 browser pages + split counts, 60 s per key.
_STATUS_TTL_SECONDS = 1.0
_status_cache: dict[str, tuple[float, dict]] = {}
_IMAGES_TTL_SECONDS = 60.0
_images_cache: dict[object, tuple[float, dict]] = {}


# ── Client / low-level helpers ───────────────────────────────────────────────


def _require_key() -> str:
    key = config.ULTRALYTICS_API_KEY
    if not key:
        raise HTTPException(
            status_code=400,
            detail="未設定 ULTRALYTICS_API_KEY — 請在後端環境變數提供 Ultralytics Platform API 金鑰",
        )
    return key


def _client(timeout: float = 60) -> AsyncPlatform:
    """Authenticated Platform SDK client.

    The SDK owns URL assembly, auth, and retries (GETs, plus 408/409/429/5xx,
    honouring retry-after) — which is why this module no longer hand-assembles
    `{owner}/{url-name}` paths or tracks each endpoint's `limit` ceiling.
    `http_client` is the test seam: tests inject httpx.MockTransport there.
    """
    return AsyncPlatform(
        api_key=_require_key(),
        timeout=timeout,
        http_client=httpx.AsyncClient(timeout=timeout, transport=_transport)
        if _transport is not None
        else None,
    )


def _bare_client(timeout: float = 120) -> httpx.AsyncClient:
    """Unauthenticated client — signed upload URLs are their own credential and
    must NOT receive the API key (and GCS rejects a stray Authorization header).
    Stays raw httpx: the SDK hands back an upload URL, it does not PUT the bytes."""
    return httpx.AsyncClient(timeout=timeout, transport=_transport)


def _as_http_error(exc: Exception) -> HTTPException:
    """Translate an SDK/httpx failure into a readable zh-TW HTTPException.

    Only the platform's response status/body may appear in the detail — never
    the request (whose headers carry the API key).
    """
    status, body = None, ""
    if isinstance(exc, APIError):
        status, body = exc.status_code, exc.body or ""
    elif isinstance(exc, httpx.HTTPStatusError):  # the bare signed-URL PUT
        status, body = exc.response.status_code, exc.response.text
    if status is not None:
        return HTTPException(
            status_code=502,
            detail=f"Ultralytics Platform 錯誤（HTTP {status}）：{body[:200]}",
        )
    return HTTPException(status_code=502, detail="無法連線 Ultralytics Platform，請稍後再試")


# Every SDK call raises one of these two; httpx errors only escape the bare
# signed-URL PUT, so callers guard on all three.
_PLATFORM_ERRORS = (APIError, APIConnectionError, httpx.HTTPError)


def _id(x: dict) -> str:
    return x.get("id") or x.get("_id") or ""


_owner_cache: list[str] = []


async def _owner(client: AsyncPlatform) -> str:
    """The API key's account username — every list endpoint is owner-scoped."""
    if _owner_cache:
        return _owner_cache[0]
    payload = await client.account.retrieve_summary()
    name = (payload or {}).get("username") or ""
    if not name:
        raise HTTPException(
            status_code=502,
            detail="無法從 Ultralytics Platform 取得帳號 username（account/summary）",
        )
    _owner_cache.append(str(name))
    return _owner_cache[0]


# The URL-addressing field is NOT called "slug": datasets carry `dataset`,
# projects `project`, models `model` — each "name used in Platform URLs".
_URL_KEY = {"datasets": "dataset", "projects": "project", "models": "model"}

# Each list endpoint caps `limit` differently and 400s above its own ceiling.
_LIST_LIMIT = {"datasets": 1000, "projects": 500, "models": 100}


def _slug(rec: dict, kind: str) -> str:
    """The record's URL name — `dataset` / `project` / `model`, never `slug`."""
    return str(rec.get(_URL_KEY.get(kind, "slug")) or rec.get("slug") or "")


def _ref(rec: dict, kind: str) -> tuple[str, str]:
    """(owner, url-name) — the two arguments every SDK sub-resource call takes."""
    owner, slug = rec.get("owner") or "", _slug(rec, kind)
    if not owner or not slug:
        raise HTTPException(
            status_code=502,
            detail=f"Ultralytics Platform 回傳的{kind}記錄缺少 owner/{_URL_KEY.get(kind)} 欄位",
        )
    return owner, slug


async def _resolve(client: httpx.AsyncClient, kind: str, name_or_id: str) -> dict:
    """Find a project/dataset record by display name, URL slug or id.

    Lists and matches rather than trusting a raw id, because callers need the
    FULL record: the new routes address by `{owner}/{slug}`, and the id alone
    cannot be turned into a URL. Listing is owner-scoped.
    """
    owner = await _owner(client)
    lister = {"datasets": client.datasets, "projects": client.projects}[kind]
    payload = await lister.list(owner, limit=_LIST_LIMIT[kind])
    for x in payload.get(kind) or []:
        if name_or_id in (x.get("name"), _slug(x, kind), x.get("slug"), _id(x)):
            return x
    kind_zh = {"projects": "專案", "datasets": "資料集"}.get(kind, kind)
    raise HTTPException(
        status_code=400,
        detail=f"Ultralytics Platform 上找不到{kind_zh}「{name_or_id}」",
    )


# ── Models ───────────────────────────────────────────────────────────────────


def _normalize_status(m: dict) -> str:
    """Collapse the platform's model/training status into the dashboard's four."""
    raw = m.get("status") or ""
    training = m.get("training")
    if not raw and isinstance(training, dict):
        raw = training.get("status") or ""
    raw = str(raw).lower()
    if any(k in raw for k in ("complete", "success", "finish", "done", "trained")):
        return "completed"
    if any(k in raw for k in ("fail", "error", "cancel", "stop")):
        return "failed"
    if any(k in raw for k in ("train", "run", "pend", "queue", "start", "progress")):
        return "training"
    if m.get("metrics"):
        return "completed"  # metrics only exist once a training run produced them
    return "idle"


async def list_datasets() -> list[dict]:
    """Every dataset on the platform, newest first, with its image count.

    `ULTRALYTICS_DATASET` pins ONE master, which was right when there was one
    dataset. There are six benchmarks now plus derived training bundles, and a
    single pin meant the 資料集 browser always showed `monkey cocotree` — the
    July set — while the shipping model had been trained on something else.
    Pinning a different name would only move the staleness, so the client picks.
    """
    try:
        async with _client() as client:
            # Listing is owner-scoped — the SDK requires it positionally.
            owner = await _owner(client)
            payload = await client.datasets.list(owner, limit=_LIST_LIMIT["datasets"])
    except _PLATFORM_ERRORS as exc:
        raise _as_http_error(exc)
    rows = payload.get("datasets") or [] if isinstance(payload, dict) else (payload or [])
    out = []
    for d in rows:
        out.append({
            "id": _id(d),
            "name": d.get("name") or "",
            "slug": _slug(d, "datasets") or "",
            "images": d.get("numImages") or d.get("imageCount") or d.get("images") or 0,
            "updatedAt": d.get("updatedAt") or d.get("updated_at") or d.get("createdAt"),
            "is_default": (d.get("name") or "") == config.ULTRALYTICS_DATASET,
        })
    out.sort(key=lambda r: str(r.get("updatedAt") or ""), reverse=True)
    return out


async def list_models(project: str) -> list[dict]:
    """All models in the project, with normalized status + headline metrics."""
    try:
        async with _client() as client:
            proj = await _resolve(client, "projects", project)
            payload = await client.models.list(*_ref(proj, "projects"), limit=_LIST_LIMIT["models"])
    except _PLATFORM_ERRORS as exc:
        raise _as_http_error(exc)
    out = []
    for m in payload.get("models") or []:
        met = m.get("metrics") or {}
        # The metric key is "mAP50" (and "mAP50-95"), not "map50" — reading the
        # old spelling silently showed a blank column instead of failing.
        map50 = met.get("mAP50")
        if map50 is None:
            map50 = met.get("map50")
        out.append({
            "id": _id(m),
            "name": m.get("name") or "",
            "status": _normalize_status(m),
            "metrics": {
                "precision": met.get("precision"),
                "recall": met.get("recall"),
                "map50": map50,
            },
            "createdAt": m.get("createdAt") or m.get("created_at") or m.get("created"),
        })
    return out


# ── Dataset browser (資料集 — thumbnails + rendered annotations) ──────────────

_SPLITS = ("train", "val", "test")


def _normalize_image(img: dict) -> dict:
    """Platform image record → dashboard shape with RELATIVE top-left boxes.

    Observed record shape (re-probed live 2026-08-13): GET
    datasets/{owner}/{dataset}/images
    returns {images, total, hasMore, classes, errorCount, nextCursor}; each
    image is {id, hash, ext, thumbnailUrl (signed, expiring), width, height,
    split, labelCount, name, bytes} plus — only with includeLabels=true —
    labels: [{classId, bbox: [cx, cy, w, h]}] in normalised YOLO CENTER format
    (verified identical to the ndjson export's boxes for the same image).
    The client overlays SVG rects, so convert to top-left origin here.
    """
    boxes = []
    for lab in img.get("labels") or []:
        bbox = lab.get("bbox") or []
        if len(bbox) != 4:
            continue
        cx, cy, w, h = (float(v) for v in bbox)
        boxes.append({
            "x": max(0.0, min(1.0, cx - w / 2)),
            "y": max(0.0, min(1.0, cy - h / 2)),
            "w": max(0.0, min(1.0, w)),
            "h": max(0.0, min(1.0, h)),
            "cls": int(lab.get("classId") or 0),
        })
    return {
        "id": _id(img),
        # hash is the per-image key for ALL write endpoints (labels/split/delete)
        "hash": img.get("hash") or "",
        "thumbnail_url": img.get("thumbnailUrl") or "",
        # full-resolution signed URL — present with includeImageUrls=true
        "full_url": img.get("imageUrl") or None,
        "width": img.get("width") or 0,
        "height": img.get("height") or 0,
        "split": img.get("split") or "",
        "boxes": boxes,
    }


async def _split_counts(client: httpx.AsyncClient, ds: dict) -> dict:
    """{'all': n, 'train': n, 'val': n, 'test': n} via limit=1 probes (total
    respects the split filter). Cached 60 s alongside the image pages, keyed by
    dataset — there are six of them now, and a shared key would answer for the
    wrong one."""
    hit = _images_cache.get(("counts",) + _ref(ds, "datasets"))
    if hit and time.monotonic() - hit[0] < _IMAGES_TTL_SECONDS:
        return hit[1]
    counts: dict[str, int] = {}
    owner, slug = _ref(ds, "datasets")
    payload = await client.datasets.list_images(owner, slug, limit=1)
    counts["all"] = int(payload.get("total") or 0)
    for s in _SPLITS:
        p = await client.datasets.list_images(owner, slug, limit=1, split=s)
        counts[s] = int(p.get("total") or 0)
    _images_cache[("counts",) + _ref(ds, "datasets")] = (time.monotonic(), counts)
    return counts


async def list_dataset_images(
    split: str | None = None, offset: int = 0, limit: int = 24,
    dataset: str | None = None,
) -> dict:
    """One page of a dataset's images, boxes normalised for the client's SVG
    overlay. 60 s in-process cache per (dataset, split, offset, limit) so grid
    scrolling / multiple viewers don't hammer the platform.

    `dataset` is explicit because ULTRALYTICS_DATASET pins ONE master, and the
    project now has six (july, crowd, sites, night, negatives, 0814-0816). The
    browser and every edit path below take the same argument, so a viewer can
    never be looking at one dataset while its edits land in another."""
    key = (dataset or "", split or "", offset, limit)
    hit = _images_cache.get(key)
    if hit and time.monotonic() - hit[0] < _IMAGES_TTL_SECONDS:
        return hit[1]
    try:
        async with _client() as client:
            ds = await _resolve(client, "datasets", dataset or config.ULTRALYTICS_DATASET)
            # SDK kwargs are snake_case; it maps them to the query names.
            params: dict = {
                "limit": limit,
                "offset": offset,
                "include_labels": "true",
                # full-res signed URLs for the annotation editor (adds
                # `imageUrl` per record, same CDN as thumbnails)
                "include_image_urls": "true",
            }
            if split:
                params["split"] = split
            payload = await client.datasets.list_images(*_ref(ds, "datasets"), **params)
            counts = await _split_counts(client, ds)
    except _PLATFORM_ERRORS as exc:
        raise _as_http_error(exc)
    raw = payload.get("images") or [] if isinstance(payload, dict) else []
    images = [_normalize_image(i) for i in raw]
    out = {
        "total": int(payload.get("total") or len(images)) if isinstance(payload, dict) else len(images),
        "has_more": bool(payload.get("hasMore")) if isinstance(payload, dict) else False,
        "counts": counts,
        "images": images,
    }
    _images_cache[key] = (time.monotonic(), out)
    return out


# ── Dataset editing (annotations / split / delete — Platform stays SoT) ──────
#
# These moved OUT from under datasets/ to TOP-LEVEL image routes, and the key
# changed from the 32-hex content hash to the image's 24-hex ObjectId:
#   PATCH  images/{imageId}   {"labels":[{"classId",bbox:[cx,cy,w,h]}]}
#   PATCH  images/bulk        {"imageIds":[...],"split":TARGET,"conflictPolicy"?}
#          (409 = name/content conflict in the target split)
#   DELETE images/bulk        {"imageIds":[...]}
# The old datasets/{id}/images/... paths no longer resolve at all.
#
# This module's OWN surface stays hash-keyed: `hash` is stable content identity
# and the dashboard uses it as its React key in ~14 places, so the id lookup
# lives here instead of rippling into the UI.

_HASH_RE = re.compile(r"[0-9a-f]{32}")

# hash → imageId map for the configured dataset. One listing covers the whole
# dataset (limit caps at 5000), so this is a single request, cached, and
# dropped whenever a write invalidates the image pages.
_HASH_ID_TTL_SECONDS = 300.0
_hash_ids: dict[str, tuple[float, dict[str, str]]] = {}
_HASH_ID_PAGE = 5000


def _require_hashes(hashes: list[str]) -> None:
    if not hashes:
        raise HTTPException(status_code=400, detail="未指定任何影像")
    for h in hashes:
        if not _HASH_RE.fullmatch(h):
            raise HTTPException(status_code=400, detail="影像 hash 格式錯誤")


async def _hash_to_ids(client: httpx.AsyncClient, ds: dict, hashes: list[str]) -> list[str]:
    """Resolve content hashes to the ObjectIds the write routes now require."""
    hit = _hash_ids.get("map")
    if not (hit and time.monotonic() - hit[0] < _HASH_ID_TTL_SECONDS
            and all(h in hit[1] for h in hashes)):
        mapping: dict[str, str] = {}
        owner, slug = _ref(ds, "datasets")
        cursor = None
        while True:
            params: dict = {"limit": _HASH_ID_PAGE, "include_thumbnails": "false",
                            "include_labels": "false", "include_total": "false"}
            if cursor:
                params["cursor"] = cursor
            page = await client.datasets.list_images(owner, slug, **params)
            for im in (page.get("images") or []):
                if im.get("hash"):
                    mapping[str(im["hash"])] = _id(im)
            cursor = page.get("nextCursor")
            if not page.get("hasMore") or not cursor:
                break
        _hash_ids["map"] = (time.monotonic(), mapping)
        hit = _hash_ids["map"]

    missing = [h for h in hashes if not hit[1].get(h)]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"資料集中找不到 {len(missing)} 張指定影像（可能已被刪除）",
        )
    return [hit[1][h] for h in hashes]


async def update_image_labels(
    image_hash: str, boxes: list[dict], dataset: str | None = None
) -> dict:
    """Replace one image's labels. `boxes` arrive in the dashboard's top-left
    relative shape ({x, y, w, h, cls}) and are converted back to the platform's
    normalised YOLO CENTER format — the exact inverse of _normalize_image."""
    _require_hashes([image_hash])
    labels = []
    for b in boxes:
        w = float(b["w"])
        h = float(b["h"])
        cx = float(b["x"]) + w / 2
        cy = float(b["y"]) + h / 2
        labels.append({
            "classId": int(b.get("cls") or 0),
            "bbox": [round(cx, 5), round(cy, 5), round(w, 5), round(h, 5)],
        })
    try:
        async with _client() as client:
            ds = await _resolve(client, "datasets", dataset or config.ULTRALYTICS_DATASET)
            image_id, = await _hash_to_ids(client, ds, [image_hash])
            resp = await client.images.update(image_id, body={"labels": labels})
    except _PLATFORM_ERRORS as exc:
        raise _as_http_error(exc)
    _images_cache.clear()  # cached pages now carry stale boxes
    return {"saved": True, "label_count": resp.get("labelCount", len(labels))}


async def move_images_split(
    images: list[dict], target_split: str, dataset: str | None = None
) -> dict:
    """Move images between splits. Each item is {hash, split} where split is
    the image's CURRENT split — the platform keys source docs by (hash, split)
    because a hash may exist once per split. `target_split` is where they go."""
    hashes = [str(i.get("hash") or "") for i in images]
    _require_hashes(hashes)
    try:
        async with _client() as client:
            ds = await _resolve(client, "datasets", dataset or config.ULTRALYTICS_DATASET)
            image_ids = await _hash_to_ids(client, ds, hashes)
            resp = await client.images.update_bulk(
                image_ids=image_ids, split=target_split
            )
    except APIError as exc:
        if exc.status_code == 409:
            raise HTTPException(
                status_code=409,
                detail="目標切分已有同名或相同內容的影像，無法移動",
            ) from exc
        raise _as_http_error(exc) from exc
    except _PLATFORM_ERRORS as exc:
        raise _as_http_error(exc)
    _images_cache.clear()  # splits and counts changed
    _hash_ids.clear()
    return {
        "moved": resp.get("modifiedCount", 0),
        "skipped": resp.get("skippedCount", 0),
        "target_split": resp.get("targetSplit", target_split),
    }


async def delete_images(hashes: list[str], dataset: str | None = None) -> dict:
    """Delete images from the dataset (bulk, by hash). Destructive."""
    _require_hashes(hashes)
    try:
        async with _client() as client:
            ds = await _resolve(client, "datasets", dataset or config.ULTRALYTICS_DATASET)
            image_ids = await _hash_to_ids(client, ds, hashes)
            resp = await client.images.delete_bulk(image_ids=image_ids)
    except APIError as exc:
        if exc.status_code == 404:
            raise HTTPException(
                status_code=404, detail="找不到符合的影像（可能已被刪除）"
            ) from exc
        raise _as_http_error(exc) from exc
    except _PLATFORM_ERRORS as exc:
        raise _as_http_error(exc)
    _images_cache.clear()
    _hash_ids.clear()  # deleted ids must not be handed out again
    return {"deleted": resp.get("deletedCount", 0)}


# ── Dataset upload (add detections as training images) ───────────────────────


async def _upload_one(
    api: httpx.AsyncClient,
    bare: httpx.AsyncClient,
    ds_id: str,
    filename: str,
    data: bytes,
    content_type: str,
) -> str:
    """signed-url → PUT bytes → complete. Returns the sessionId.

    Raises httpx errors on failure — callers decide how to report them."""
    meta = await api.upload.retrieve_file_url(
        asset_type="datasets",
        asset_id=ds_id,
        filename=filename,
        content_type=content_type,
        total_bytes=len(data),
    )
    put = await bare.put(
        meta["uploadUrl"], content=data,
        headers={
            "Content-Type": content_type,
            # GCS signs this precondition into the URL; omitting it returns
            # 400 MalformedSecurityHeader before the upload can complete.
            "x-goog-if-generation-match": "0",
        },
    )
    put.raise_for_status()
    await api.upload.complete(session_id=meta["sessionId"])
    return meta["sessionId"]


async def _ingest_session(api: httpx.AsyncClient, ds: dict, session_id: str) -> bool:
    """Queue one uploaded session for ingest; 409 = another ingest is running
    for this dataset — wait and retry.

    The dataset moved into the PATH (`datasets/{owner}/{dataset}/ingest`); the
    old unscoped `datasets/ingest` collides with `datasets/{owner}` and answers
    405 with an empty body."""
    owner, slug = _ref(ds, "datasets")
    for _ in range(_INGEST_MAX_ATTEMPTS):
        try:
            await api.datasets.ingest(owner, slug, session_id=session_id)
            return True
        except APIError as exc:
            if exc.status_code != 409:
                return False
            await asyncio.sleep(_INGEST_BACKOFF_SECONDS)
        except (APIConnectionError, httpx.HTTPError):
            return False
    return False


def _ascii_name(name: str) -> str:
    """ASCII-safe zip entry name — the ingest worker silently drops archives
    whose entries carry non-ASCII names (A/B verified 2026-07-31: identical
    zips, ascii landed in 30 s, 中文 name never processed). Camera IDs are
    Chinese, so every real snapshot needs this. A stable crc32 prefix keeps
    per-camera grouping and avoids collisions across cameras whose names
    strip to the same ASCII remainder."""
    import zlib

    ascii_part = name.encode("ascii", "ignore").decode().lstrip("_")
    if ascii_part == name:
        return name
    tag = f"{zlib.crc32(name.encode('utf-8')) & 0xFFFFFF:06x}"
    return f"cam{tag}_{ascii_part or 'image.jpg'}"


def _zip_images(entries: list[tuple[str, bytes]]) -> bytes:
    """Bundle images into one ZIP (stored, images are already compressed).

    The ingest worker only processes ARCHIVES: loose JPEG sessions queue as
    jobs but never land (verified 2026-07-31 — four loose uploads sat
    "queued" for a day while a ZIP of two frames ingested in minutes). One
    ZIP also means one session and one ingest job, which sidesteps the 409
    one-job-per-dataset serialisation entirely.
    """
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, data in entries:
            zf.writestr(_ascii_name(name), data)
    return buf.getvalue()


async def _upload_zip_and_ingest(api, bare, ds: dict, entries: list[tuple[str, bytes]]) -> bool:
    """ZIP → signed-url → PUT → complete → ingest. True when the job queued.

    upload/signed-url still keys by raw `assetId`, so the id is used there and
    the owner/slug ref only for ingest."""
    blob = _zip_images(entries)
    sid = await _upload_one(api, bare, _id(ds), "dashboard_upload.zip", blob, "application/zip")
    return await _ingest_session(api, ds, sid)


# ── Duplicate warning (informational — the master keeps everything) ──────────
#
# Fixed cameras re-shoot the same scene for months, so a lot of what gets added
# is a near-copy of something already in the dataset. That is NOT a reason to
# refuse it: the master is the archive (nothing is ever deleted there) and the
# DERIVED training set is where redundancy gets collapsed. So we only tell the
# user which uploads are near-duplicates, and upload them regardless.

_MASTER_PAGE = 500
_MASTER_CONCURRENCY = 8
_master_cache: tuple[float, dict[str, int]] | None = None


async def _fetch_bytes(
    bare: httpx.AsyncClient, url: str, sem: asyncio.Semaphore
) -> bytes | None:
    """GET bytes off a signed CDN URL; None on any failure (expired sig, 404…)."""
    if not url:
        return None
    async with sem:
        try:
            r = await bare.get(url, follow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception:
            return None


async def _master_hashes() -> dict[str, int]:
    """{image name: dHash} for every image in the master dataset.

    Thumbnails, not full-resolution files: dHash downsizes to 9x8 anyway, so a
    256 px thumbnail hashes the same as the original at ~1% of the bytes.
    Cached for DEDUP_MASTER_TTL_SECONDS — this drives a warning, not a gate, so
    slightly stale hashes are harmless.
    """
    global _master_cache
    hit = _master_cache
    if hit and time.monotonic() - hit[0] < config.DEDUP_MASTER_TTL_SECONDS:
        return hit[1]

    records: list[dict] = []
    async with _client(timeout=120) as api, _bare_client(timeout=120) as bare:
        ds = await _resolve(api, "datasets", config.ULTRALYTICS_DATASET)
        owner, slug = _ref(ds, "datasets")
        offset = 0
        while True:
            page = await api.datasets.list_images(
                owner, slug, limit=_MASTER_PAGE, offset=offset
            )
            imgs = (page.get("images") or []) if isinstance(page, dict) else []
            records.extend(imgs)
            offset += len(imgs)
            if not imgs or offset >= int(page.get("total") or 0):
                break
        sem = asyncio.Semaphore(_MASTER_CONCURRENCY)
        blobs = await asyncio.gather(
            *(_fetch_bytes(bare, r.get("thumbnailUrl") or "", sem) for r in records)
        )

    def _hash_all() -> dict[str, int]:
        out: dict[str, int] = {}
        for rec, data in zip(records, blobs):
            if not data:
                continue
            h = dedup.dhash(data)
            if h is not None:
                out[rec.get("name") or rec.get("hash") or "?"] = h
        return out

    hashes = await asyncio.to_thread(_hash_all)
    _master_cache = (time.monotonic(), hashes)
    return hashes


async def _duplicate_notes(entries: list[tuple[str, bytes]]) -> list[dict]:
    """Which incoming images the master already has a near-copy of.

    Returns [{name, similar_to, distance}]. Purely informational: any failure
    yields an empty list, because a warning must never be able to block an
    upload.
    """
    try:
        master = await _master_hashes()
    except Exception as exc:  # noqa: BLE001 — never fail an upload over a warning
        logger.warning("Duplicate check skipped: %s", type(exc).__name__)
        return []
    if not master:
        return []

    def _compare() -> list[dict]:
        notes = []
        for name, data in entries:
            h = dedup.dhash(data)
            if h is None:
                continue
            best_label, best_dist = "", 65
            for label, mh in master.items():
                d = dedup.hamming(h, mh)
                if d < best_dist:
                    best_label, best_dist = label, d
            if best_dist <= config.DEDUP_THRESHOLD:
                notes.append(
                    {"name": name, "similar_to": best_label, "distance": best_dist}
                )
        return notes

    return await asyncio.to_thread(_compare)


async def add_detections_to_dataset(detection_ids: list[int], db: AsyncSession) -> dict:
    """Upload the RAW snapshots of the given detections into the configured dataset.

    Uses Detection.image_path — the pristine un-annotated frame written by the
    detector (never the annotated/crop variants: burned-in boxes would poison
    training). Images arrive UNLABELED on the platform, i.e. they act as
    background negatives until someone draws boxes in the Platform UI.

    Returns {"uploaded": n, "failed": [{"id", "reason"}, ...],
             "duplicates": [{"name", "similar_to", "distance"}, ...]} — near-
    duplicates are REPORTED, never dropped: the master archives everything and
    the derived training set is what gets deduplicated.
    """
    from models import Detection
    from services.storage import storage

    rows = (
        await db.execute(select(Detection).where(Detection.id.in_(detection_ids)))
    ).scalars().all()
    by_id = {d.id: d for d in rows}

    uploaded = 0
    failed: list[dict] = []
    entries: list[tuple[int, str, bytes]] = []

    for det_id in detection_ids:
        det = by_id.get(det_id)
        if det is None:
            failed.append({"id": det_id, "reason": "偵測紀錄不存在"})
            continue
        if not det.image_path:
            failed.append({"id": det_id, "reason": "此紀錄沒有原始快照"})
            continue
        data = storage.download(det.image_path)
        if data is None:
            failed.append({"id": det_id, "reason": f"讀不到快照檔 {det.image_path}"})
            continue
        entries.append((det_id, det.image_path, data))

    duplicates: list[dict] = []
    if entries:
        duplicates = await _duplicate_notes([(n, d) for _, n, d in entries])
        try:
            async with _client(timeout=300) as api, _bare_client(timeout=300) as bare:
                ds = await _resolve(api, "datasets", config.ULTRALYTICS_DATASET)
                ok = await _upload_zip_and_ingest(
                    api, bare, ds, [(name, data) for _, name, data in entries]
                )
                if ok:
                    uploaded = len(entries)
                else:
                    failed.extend({"id": i, "reason": "資料集匯入排程失敗"} for i, _, _ in entries)
        except HTTPException:
            raise
        except _PLATFORM_ERRORS as exc:
            raise _as_http_error(exc)

    if uploaded:
        _images_cache.clear()  # the 資料集 browser should see fresh counts soon
        _invalidate_master_hashes()
    return {"uploaded": uploaded, "failed": failed, "duplicates": duplicates}


async def upload_files(files: list[tuple[str, bytes, str]]) -> dict:
    """Upload user-supplied image files (訓練 tab drag & drop) into the dataset.

    files: (filename, data, content_type) triples, pre-validated by the route.
    Same platform flow as add_detections_to_dataset; images arrive UNLABELED
    (background negatives until boxes are drawn in the Platform UI).

    Returns {"uploaded": n, "failed": [{"name", "reason"}, ...],
             "duplicates": [{"name", "similar_to", "distance"}, ...]}.
    """
    uploaded = 0
    failed: list[dict] = []
    duplicates: list[dict] = []

    if files:
        duplicates = await _duplicate_notes([(n, d) for n, d, _ in files])
        try:
            async with _client(timeout=300) as api, _bare_client(timeout=300) as bare:
                ds = await _resolve(api, "datasets", config.ULTRALYTICS_DATASET)
                ok = await _upload_zip_and_ingest(
                    api, bare, ds, [(name, data) for name, data, _ in files]
                )
                if ok:
                    uploaded = len(files)
                else:
                    failed.extend({"name": n, "reason": "資料集匯入排程失敗"} for n, _, _ in files)
        except HTTPException:
            raise
        except _PLATFORM_ERRORS as exc:
            raise _as_http_error(exc)

    if uploaded:
        _images_cache.clear()
        _invalidate_master_hashes()
    return {"uploaded": uploaded, "failed": failed, "duplicates": duplicates}


def _invalidate_master_hashes() -> None:
    """Forget the master's hashes — new images just landed (or are landing)."""
    global _master_cache
    _master_cache = None


# ── Derived training dataset (master + derived) ──────────────────────────────
#
# The master (`monkey cocotree`) is the archive: every image ever added, nothing
# deleted, test/val pinned so the ratchet keeps sitting the same exam. Training
# runs on a DERIVED dataset regenerated from it, with the TRAIN split
# deduplicated — measured 2026-08-03, a 25%-smaller train split trained a BETTER
# model (test recall 0.704 -> 0.756, F1 0.7979 -> 0.8320), because near-duplicate
# frames dilute the training signal instead of augmenting it.
#
# Regeneration is minutes long (export → download ~900 images → hash → zip →
# upload → wait for ingest), so it runs as a background task and the HTTP layer
# only ever starts it or reads its progress.

_CDN_HASH_RE = re.compile(r"/i/([0-9a-f]{32})\.")
_DOWNLOAD_CONCURRENCY = 8
# Ingest is async on the platform; poll the new dataset's image count until the
# upload lands (tests patch these to run instantly).
_LANDING_MAX_ATTEMPTS = 60
_LANDING_POLL_SECONDS = 10.0

_TW_TZ = timezone(timedelta(hours=8))

_regen: dict = {
    "state": "idle",  # idle | running | done | failed
    "step": "",
    "done": 0,
    "total": 0,
    "error": "",
    "result": None,
    "started_at": None,
    "finished_at": None,
}
_regen_task: asyncio.Task | None = None


def _progress(step: str, done: int = 0, total: int = 0) -> None:
    _regen.update(step=step, done=done, total=total)


def regenerate_state() -> dict:
    """Snapshot of the regeneration job (idle/running/done/failed + progress)."""
    return dict(_regen)


async def start_regenerate(
    on_complete: Callable[[dict], Awaitable[None]] | None = None,
) -> dict:
    """Kick off regeneration in the background; returns immediately.

    409 while one is already running — two concurrent runs would fight over the
    image cache and create two derived datasets nobody asked for.
    """
    global _regen_task
    if _regen_task is not None and not _regen_task.done():
        raise HTTPException(status_code=409, detail="已有一個重新生成作業進行中")
    _regen.update(
        state="running",
        step="準備中",
        done=0,
        total=0,
        error="",
        result=None,
        started_at=datetime.now(_TW_TZ).isoformat(timespec="seconds"),
        finished_at=None,
    )
    _regen_task = asyncio.create_task(_run_regenerate(on_complete))
    return regenerate_state()


async def _run_regenerate(
    on_complete: Callable[[dict], Awaitable[None]] | None,
) -> None:
    try:
        result = await _regenerate_derived_dataset()
        _regen.update(state="done", step="完成", result=result, error="")
        if on_complete is not None:
            await on_complete(result)
    except HTTPException as exc:
        _regen.update(state="failed", error=str(exc.detail))
        logger.warning("Derived dataset regeneration failed: %s", exc.detail)
    except Exception as exc:  # noqa: BLE001 — background task, must not vanish
        logger.exception("Derived dataset regeneration crashed")
        _regen.update(state="failed", error=f"重新生成訓練集失敗（{type(exc).__name__}）")
    finally:
        _regen["finished_at"] = datetime.now(_TW_TZ).isoformat(timespec="seconds")


async def _export_records(
    api: httpx.AsyncClient, bare: httpx.AsyncClient, ds: dict
) -> list[dict]:
    """The master's ndjson export, parsed to image records.

    Each record gains `_key` (the CDN content hash, which is also the cache
    filename) and `_boxes` (normalised YOLO center boxes, possibly empty for a
    background negative).
    """
    exp = await api.datasets.retrieve_export(*_ref(ds, "datasets"))
    url = exp.get("downloadUrl") if isinstance(exp, dict) else None
    if not url:
        raise HTTPException(status_code=502, detail="Platform 匯出未提供下載網址")
    r = await bare.get(url, follow_redirects=True)
    r.raise_for_status()
    out = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("type") != "image":
            continue
        m = _CDN_HASH_RE.search(rec.get("url", ""))
        if not m:
            continue
        rec["_key"] = m.group(1)
        rec["_boxes"] = (rec.get("annotations") or {}).get("boxes") or []
        out.append(rec)
    return out


def _cache_name(rec: dict) -> str:
    """Cache filename = the CDN content hash, so re-signed URLs reuse the file
    (same convention as tests/eval_ratchet.py — they share the directory)."""
    tail = (rec.get("url") or "").split("?")[0].rsplit("/", 1)[-1]
    return tail or f"{rec['_key']}.jpg"


async def _download_records(
    bare: httpx.AsyncClient, records: list[dict]
) -> dict[str, Path]:
    """Download (or reuse cached) image files. {content hash: path}."""
    cache = config.DEDUP_CACHE_DIR
    sem = asyncio.Semaphore(_DOWNLOAD_CONCURRENCY)

    async def one(rec: dict) -> Path | None:
        dst = cache / _cache_name(rec)
        if not dst.exists():
            data = await _fetch_bytes(bare, rec.get("url") or "", sem)
            if data is None:
                _regen["done"] += 1
                return None
            await asyncio.to_thread(dst.write_bytes, data)
        _regen["done"] += 1
        return dst

    paths = await asyncio.gather(*(one(r) for r in records))
    return {r["_key"]: p for r, p in zip(records, paths) if p is not None}


def _dedup_train(train: list[dict], paths: dict[str, Path]) -> tuple[list[dict], dict]:
    """Collapse near-duplicate TRAIN images to one representative each (sync)."""
    by_key = {r["_key"]: r for r in train}
    pairs: list[tuple[str, int]] = []
    for rec in train:
        p = paths.get(rec["_key"])
        if p is None:
            continue
        h = dedup.dhash(p)
        if h is not None:
            pairs.append((rec["_key"], h))
        _regen["done"] += 1

    groups = dedup.group_near_duplicates(pairs)
    metadata = {key: by_key[key]["_boxes"] for key, _ in pairs}
    keep = [by_key[dedup.pick_representative(g, metadata)] for g in groups]
    # Undecodable but downloaded images keep their place — never silently lost.
    hashed = {key for key, _ in pairs}
    keep += [
        r for r in train if r["_key"] not in hashed and paths.get(r["_key"]) is not None
    ]

    available = [r for r in train if paths.get(r["_key"]) is not None]
    stats = {
        "train_before": len(train),
        "kept": len(keep),
        "dropped": len(available) - len(keep),
        "missing": len(train) - len(available),
        "negatives_before": sum(1 for r in available if not r["_boxes"]),
        "negatives_after": sum(1 for r in keep if not r["_boxes"]),
    }
    return keep, stats


def _build_zip(
    rows: list[tuple[dict, str]], paths: dict[str, Path], dst: Path
) -> tuple[Path, int]:
    """Write the upload archive: images/ + a data.ndjson manifest.

    The ingest worker honours the per-record `split` field in data.ndjson
    (verified 2026-08-03), which is what keeps the derived set's train/val
    boundary identical to the master's.
    """
    import zipfile

    manifest: list[str] = []
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zf:
        written: set[str] = set()
        for rec, split in rows:
            p = paths.get(rec["_key"])
            if p is None:
                continue
            arc = f"images/{p.name}"
            if arc not in written:
                zf.write(p, arc)
                written.add(arc)
            manifest.append(
                json.dumps(
                    {
                        "type": "image",
                        "file": arc,
                        "width": rec.get("width"),
                        "height": rec.get("height"),
                        "split": split,
                        "annotations": {"boxes": rec["_boxes"]},
                    },
                    ensure_ascii=False,
                )
            )
            _regen["done"] += 1
        zf.writestr("data.ndjson", "\n".join(manifest))
    return dst, len(manifest)


async def _await_landing(api: httpx.AsyncClient, ds: dict, expected: int) -> int:
    """Poll the new dataset's image count until the ingest lands (or we give up)."""
    total = 0
    for _ in range(_LANDING_MAX_ATTEMPTS):
        try:
            page = await api.datasets.list_images(*_ref(ds, "datasets"), limit=1)
            total = int(page.get("total") or 0) if isinstance(page, dict) else 0
        except _PLATFORM_ERRORS:
            pass
        _regen["done"] = total
        if total >= expected:
            return total
        await asyncio.sleep(_LANDING_POLL_SECONDS)
    return total


async def _regenerate_derived_dataset() -> dict:
    """Build a fresh deduplicated derived dataset from the master.

    TRAIN is deduplicated; VAL is copied untouched (it is the tuning set the
    platform trains against). TEST is deliberately NOT copied — the exam lives
    in the master and only the ratchet reads it.
    """
    cache = config.DEDUP_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / f"_derived_{int(time.time())}.zip"

    try:
        async with _client(timeout=300) as api, _bare_client(timeout=600) as bare:
            _progress("讀取主檔匯出")
            ds = await _resolve(api, "datasets", config.ULTRALYTICS_DATASET)
            records = await _export_records(api, bare, ds)
            train = [r for r in records if r.get("split") == "train"]
            val = [r for r in records if r.get("split") == "val"]
            if not train:
                raise HTTPException(
                    status_code=400, detail="主檔沒有 train 切分，無法生成訓練集"
                )

            wanted = train + val
            _progress("下載主檔影像", 0, len(wanted))
            paths = await _download_records(bare, wanted)
            if not paths:
                raise HTTPException(
                    status_code=502,
                    detail="主檔影像全數下載失敗（簽名網址可能已過期，請重新匯出）",
                )

            _progress("計算感知雜湊（只去重 train）", 0, len(train))
            keep, stats = await asyncio.to_thread(_dedup_train, train, paths)

            rows = [(r, "train") for r in keep] + [(r, "val") for r in val]
            _progress("打包訓練集", 0, len(rows))
            _, entry_count = await asyncio.to_thread(_build_zip, rows, paths, zip_path)

            stamp = datetime.now(_TW_TZ).strftime("%Y%m%d-%H%M%S")
            base_slug = (_slug(ds, "datasets") or "monkey-cocotree").split("-dedup-")[0]
            slug = f"{base_slug}-dedup-{stamp}"[:80]
            name = f"{ds.get('name') or 'monkey cocotree'} dedup {stamp}"
            _progress(f"建立衍生資料集 {slug}")
            # The URL name field is `dataset` (was `slug`). The response
            # carries {id, owner, dataset} — i.e. it is already a usable ref.
            created = await api.datasets.create(
                dataset=slug, name=name, task="detect", visibility="private"
            )
            new_ds = {
                "id": created.get("datasetId") or _id(created),
                "owner": created.get("owner") or await _owner(api),
                "dataset": created.get("dataset") or slug,
            }
            if not new_ds["id"]:
                raise HTTPException(
                    status_code=502, detail="Platform 未回傳資料集 ID，無法建立訓練集"
                )

            _progress("上傳訓練集", 0, entry_count)
            blob = await asyncio.to_thread(zip_path.read_bytes)
            session_id = await _upload_one(
                api, bare, new_ds["id"], f"{slug}.zip", blob, "application/zip"
            )
            del blob
            if not await _ingest_session(api, new_ds, session_id):
                raise HTTPException(status_code=502, detail="訓練集匯入排程失敗")

            _progress("等待平台匯入", 0, entry_count)
            landed = await _await_landing(api, new_ds, entry_count)
    except HTTPException:
        raise
    except _PLATFORM_ERRORS as exc:
        raise _as_http_error(exc)
    finally:
        zip_path.unlink(missing_ok=True)

    result = {
        "dataset_slug": slug,
        "dataset_id": new_ds["id"],
        "name": name,
        "kept": stats["kept"],
        "dropped": stats["dropped"],
        "negatives_before": stats["negatives_before"],
        "negatives_after": stats["negatives_after"],
        "train_before": stats["train_before"],
        "missing": stats["missing"],
        "val_images": len(val),
        "expected_images": entry_count,
        "images_landed": landed,
    }
    logger.info(
        "Derived dataset %s: train %d -> %d (-%d), val %d, landed %d/%d",
        slug, stats["train_before"], stats["kept"], stats["dropped"],
        len(val), landed, entry_count,
    )
    return result


# ── Training ─────────────────────────────────────────────────────────────────


async def start_training(
    name: str,
    base: str = "yolo26n.pt",
    epochs: int = 1000,
    imgsz: int = 1920,
    gpu: str = "rtx-6000-ada",
    dataset: str | None = None,
) -> dict:
    """Start a cloud training run on the configured project + dataset.

    `dataset` (name/slug/id) overrides the configured master — that is how
    用訓練集開始訓練 points a run at the deduplicated derived dataset.

    Two platform facts learned the hard way (2026-07-30): training/start
    demands an EXISTING model record (400 "Missing required field: modelId"
    otherwise), so one is created first; and the GPU enum is exact — "rtx-6000"
    is invalid, the Ada card is "rtx-6000-ada".
    """
    try:
        async with _client() as client:
            proj = await _resolve(client, "projects", config.ULTRALYTICS_PROJECT)
            ds = await _resolve(client, "datasets", dataset or config.ULTRALYTICS_DATASET)
            ds_owner, ds_slug = ds.get("owner"), _slug(ds, "datasets")
            if not (ds_owner and ds_slug):
                raise HTTPException(
                    status_code=502,
                    detail="Ultralytics Platform 資料集缺少 owner/dataset，無法組出 ul:// 資料位址",
                )
            # POST /models now identifies the project by its URL NAME, not id.
            proj_owner, proj_slug = _ref(proj, "projects")
            created = await client.models.create(
                body={"owner": proj_owner, "project": proj_slug, "model": name}
            )
            model_id = created.get("modelId") or _id(created)
            if not model_id:
                raise HTTPException(status_code=502, detail="Platform 未回傳 modelId，無法啟動訓練")
            # gpu_type is a typed enum in the SDK — it rejects e.g. "rtx-6000"
            # (the real name is "rtx-6000-ada") before spending anything.
            out = await client.training.start(
                model_id=model_id,
                gpu_type=gpu,
                train_args={
                    "model": base,
                    "data": f"ul://{ds_owner}/datasets/{ds_slug}",
                    "epochs": epochs,
                    "imgsz": imgsz,
                },
            )
            out.setdefault("modelId", model_id)
            return out
    except _PLATFORM_ERRORS as exc:
        raise _as_http_error(exc)


async def training_status(model_id: str) -> dict:
    """Passthrough of GET models/{id}/training (progress polling).

    1 s TTL cache: the dashboard polls running models at 1 Hz per viewer, so
    N viewers must still mean at most one platform call per second."""
    if not re.fullmatch(r"[0-9a-f]{24}", model_id):
        raise HTTPException(status_code=400, detail="模型 ID 格式錯誤")
    hit = _status_cache.get(model_id)
    if hit and time.monotonic() - hit[0] < _STATUS_TTL_SECONDS:
        return hit[1]
    try:
        async with _client() as client:
            # Training status is addressed by owner/project/model now, so the
            # id has to be resolved back to a model record first.
            proj = await _resolve(client, "projects", config.ULTRALYTICS_PROJECT)
            listing = await client.models.list(
                *_ref(proj, "projects"), limit=_LIST_LIMIT["models"]
            )
            rec = next((m for m in (listing.get("models") or []) if _id(m) == model_id), None)
            if rec is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"專案「{config.ULTRALYTICS_PROJECT}」中找不到模型 {model_id}",
                )
            owner, project, model = (
                rec.get("owner") or "", rec.get("project") or "", _slug(rec, "models"),
            )
            if not (owner and project and model):
                raise HTTPException(
                    status_code=502,
                    detail="Ultralytics Platform 回傳的模型記錄缺少 owner/project/model 欄位",
                )
            data = await client.models.retrieve_training(owner, project, model)
    except _PLATFORM_ERRORS as exc:
        raise _as_http_error(exc)
    if len(_status_cache) > 100:  # bound the cache across model churn
        _status_cache.clear()
    _status_cache[model_id] = (time.monotonic(), data)
    return data


# ── Promotion (GitHub workflow dispatch) ─────────────────────────────────────


async def _dispatch_workflow(
    workflow: str, model_id: str, action_zh: str, extra_inputs: dict | None = None
) -> None:
    """POST a workflow_dispatch for `workflow` with inputs {model_id, **extra}."""
    if not re.fullmatch(r"[0-9a-f]{24}", model_id):
        raise HTTPException(status_code=400, detail="模型 ID 格式錯誤")
    if not config.GITHUB_DISPATCH_TOKEN:
        raise HTTPException(
            status_code=400,
            detail=f"未設定 GITHUB_DISPATCH_TOKEN — 無法觸發{action_zh}，"
            "請在後端環境變數提供具 workflow 權限的 GitHub token",
        )
    if not config.GITHUB_REPO:
        raise HTTPException(
            status_code=400,
            detail=f"未設定 GITHUB_REPO — 無法觸發{action_zh}，"
            "請指定要使用的 owner/repository",
        )
    url = (
        f"{GITHUB_API}/repos/{config.GITHUB_REPO}"
        f"/actions/workflows/{workflow}/dispatches"
    )
    # workflow_dispatch inputs go over the wire as strings even when the
    # workflow declares `type: boolean`; a real bool serialises to JSON `true`,
    # which GitHub rejects for some input types. Normalise here.
    inputs = {"model_id": model_id}
    for k, v in (extra_inputs or {}).items():
        inputs[k] = "true" if v is True else "false" if v is False else str(v)
    try:
        async with _bare_client(timeout=30) as client:
            r = await client.post(
                url,
                json={"ref": "main", "inputs": inputs},
                headers={
                    "Authorization": f"Bearer {config.GITHUB_DISPATCH_TOKEN}",
                    "Accept": "application/vnd.github+json",
                },
            )
            r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"GitHub 觸發失敗（HTTP {exc.response.status_code}）：{exc.response.text[:200]}",
        )
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="無法連線 GitHub API，請稍後再試")


async def dispatch_promote(model_id: str, force: bool = False) -> None:
    """Trigger .github/workflows/model-promote.yml on the self-hosted runner.

    The workflow downloads the model's weights, runs the ratchet accuracy gate
    on BOTH the test and val splits, and only on a double pass swaps
    backend/best.pt and pushes (which deploys cloud + edge).

    `force=True` still runs both gates and still reports the verdict — it only
    skips the abort, and only for a genuine RATCHET FAIL; an eval that never
    completed still blocks. A forced promotion never refreshes the baseline.
    """
    await _dispatch_workflow(
        "model-promote.yml",
        model_id,
        "強制上板流程" if force else "上板流程",
        {"force": force},
    )


async def dispatch_evaluate(model_id: str) -> None:
    """Trigger .github/workflows/model-evaluate.yml on the self-hosted runner.

    Evaluation only — the workflow scores the model on BOTH splits at
    deployment settings (imgsz 1600, fp16, MPS) and POSTs the numbers back to
    /api/admin/training/eval-results; best.pt is never touched.
    """
    await _dispatch_workflow("model-evaluate.yml", model_id, "本站評測")
