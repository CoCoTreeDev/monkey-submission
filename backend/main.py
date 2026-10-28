"""FastAPI application — Monkey Detection Platform."""

from __future__ import annotations

import os

# Suppress noisy OpenCV/FFmpeg native warnings (e.g. RTSP timeout spam)
# Must be set before cv2 is imported by any module.
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import asyncio
import hashlib
import json
import logging
import re
import socket
import subprocess
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

# Taiwan Standard Time (UTC+8)
_TW_TZ = timezone(timedelta(hours=8))

from fastapi import (
    Depends, FastAPI, File, Header, HTTPException, Query, UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from urllib.parse import quote

import config
from database import async_session, get_session, init_db
from models import (
    Camera,
    DerivedDataset,
    Detection,
    EvalResult,
    HistoricalRebuildRun,
    ReIDPairLabel,
    User,
    Webhook,
)
from schemas import (
    CameraCreate,
    CameraDeleteOut,
    CameraOut,
    CameraUpdate,
    DetectionEventListOut,
    DetectionEventOut,
    DetectionHeatmapOut,
    DetectionListOut,
    DetectionOut,
    DetectionStatsOut,
    FeatureOut,
    HeatmapCellOut,
    InferenceStatsOut,
    MeOut,
    ReIDLabelCreate,
    ReIDLabelOut,
    ReIDPairListOut,
    StorageStatsOut,
    UserCreate,
    UserOut,
    UserUpdate,
    WebhookCreate,
    WebhookOut,
    WebhookUpdate,
)
from services.detector import DetectionResult, detector_manager
from services.reid import reid_review_priority, reid_system_decision
from services.storage import LocalStorage, storage
from services.stream import stream_manager
from services.webhook import send_notification, send_raw_debug_detection, send_system_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# The Azure SDK logs every blob request AND its full headers at INFO, so a
# global INFO level makes it 87% of the file: on the edge node the log reached
# 110 MB, of which 1.64 M of 1.88 M lines were header-dump continuations
# (measured 2026-08-17). That buried the lines that matter — the per-camera
# track finalisations — and made the log useless for diagnosis. httpx stays at
# INFO on purpose: its one line per request is the only record that a detection
# actually reached the API.
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ── Auth (stateless HMAC tokens + DB-backed access codes) ────────────────────
#
# Login is by a single access CODE. The code is hashed (HMAC-SHA256, keyed by
# AUTH_SECRET) and looked up in the `users` table — the source of truth for
# identity, role and (from Phase 2) per-user permissions. The issued token is a
# stateless signed payload {sub, role, exp}; there is no server-side session
# store, so it survives container restarts. A token arrives either as a Bearer
# header or as a ?token= query param (used by <img>/MJPEG tags, which cannot set
# headers).

import hmac
import base64
import time

_TOKEN_MAX_AGE = 86400  # 24 hours


def _hash_code(code: str) -> str:
    """Deterministic keyed hash of an access code (enables an indexed lookup)."""
    return hmac.new(
        config.AUTH_SECRET.encode(), code.encode(), hashlib.sha256
    ).hexdigest()


def _sign_token(user_id: int, role: str) -> str:
    """Create a stateless signed token carrying the user id, role and expiry."""
    payload = json.dumps(
        {"sub": user_id, "role": role, "exp": int(time.time()) + _TOKEN_MAX_AGE}
    )
    payload_b64 = base64.urlsafe_b64encode(payload.encode()).decode()
    sig = hmac.new(
        config.AUTH_SECRET.encode(), payload_b64.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload_b64}.{sig}"


def _verify_token(token: str) -> Optional[dict]:
    """Return the decoded payload for a valid, unexpired token, else None."""
    parts = token.split(".", 1)
    if len(parts) != 2:
        return None
    payload_b64, sig = parts
    expected_sig = hmac.new(
        config.AUTH_SECRET.encode(), payload_b64.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None
    if payload.get("exp", 0) <= time.time():
        return None
    return payload


def _extract_token(authorization: Optional[str], token: Optional[str]) -> str:
    """Pull a token from a Bearer header or a ?token= query param."""
    if authorization:
        scheme, _, val = authorization.partition(" ")
        if scheme.lower() == "bearer" and val:
            return val
    return token or ""


async def require_user(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
) -> dict:
    """Dependency — any valid logged-in user (admin or user). Returns the payload."""
    payload = _verify_token(_extract_token(authorization, token))
    if not payload:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return payload


async def require_admin(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
) -> dict:
    payload = await require_user(authorization, token)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    return payload


async def get_current_user(
    payload: dict = Depends(require_user),
    session: AsyncSession = Depends(get_session),
) -> User:
    """Load the live User row for the token's subject.

    Reads from the DB per request (not the token) so permission changes and
    account disabling take effect immediately, without waiting for the token
    to expire.
    """
    user = (
        await session.execute(select(User).where(User.id == payload.get("sub")))
    ).scalar_one_or_none()
    if user is None or not user.enabled:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _camera_scope_criteria(user: User, column) -> list:
    """SQL criteria for every camera-scoped read available to a user.

    POC cameras are excluded independently of ``allowed_cameras``. The
    separate deny is important because ``["*"]`` is the normal default for a
    user account and must not become an accidental POC grant.
    """
    if user.role == "admin":
        return []
    criteria = []
    if config.POC_CAMERA_IDS:
        criteria.append(column.not_in(config.POC_CAMERA_IDS))
    allowed = user.allowed_cameras or []
    if "*" not in allowed:
        criteria.append(column.in_(allowed))
    return criteria


def _camera_allowed(user: User, camera_id: str) -> bool:
    if user.role == "admin":
        return True
    if camera_id in config.POC_CAMERA_IDS:
        return False
    allowed = user.allowed_cameras or []
    return "*" in allowed or camera_id in allowed


# Naive TW-local midnight cutoff for the non-admin confidence gate.
# Detection.timestamp is stored naive in TW local time (see _on_detection),
# so a naive comparison is correct on both SQLite and PostgreSQL.
_USER_CONF_CUTOFF = datetime.fromisoformat(config.USER_CONFIDENCE_SINCE)


def _visibility_criteria(user: User) -> list:
    """SQL criteria restricting rows a non-admin may see; [] = unrestricted.

    Records from before the cutoff are human-verified and stay fully visible;
    newer ones must clear USER_CONFIDENCE_THRESHOLD. Admins see everything.
    """
    if user.role == "admin":
        return []
    rule = or_(
        Detection.timestamp < _USER_CONF_CUTOFF,
        Detection.confidence >= config.USER_CONFIDENCE_THRESHOLD,
    )
    # A manual override always wins over the confidence rule: admins push
    # genuine low-confidence sightings through, or hide a high-confidence
    # false positive the rule would otherwise show.
    return [
        or_(
            Detection.user_visible.is_(True),
            and_(rule, Detection.user_visible.is_(None)),
        )
    ]


def _visible_to_users(det: Detection) -> bool:
    """What a NON-admin would experience for this record (override first).

    Also surfaced to admins as `visible_to_users` so the dashboard can badge
    rows without duplicating the rule in the client.
    """
    if det.user_visible is not None:
        return det.user_visible
    return (
        det.timestamp < _USER_CONF_CUTOFF
        or det.confidence >= config.USER_CONFIDENCE_THRESHOLD
    )


def _feature_enabled(user: User, key: str) -> bool:
    """Per-user feature switch. Admin bypasses; users default to enabled."""
    if user.role == "admin":
        return True
    return bool((user.feature_flags or {}).get(key, True))


async def _seed_default_users() -> None:
    """Create the default admin + user accounts if the users table is empty.

    Seeds only when EMPTY, so admin-managed deletions (Phase 2) stick. Wiping all
    users restores the defaults, which is a deliberate anti-lockout fallback.
    """
    async with async_session() as session:
        count = (await session.execute(select(func.count(User.id)))).scalar() or 0
        if count:
            return
        session.add(
            User(
                name="管理員",
                code_hash=_hash_code(config.SEED_ADMIN_CODE),
                role="admin",
                allowed_cameras=["*"],
                feature_flags={},
            )
        )
        session.add(
            User(
                name="一般使用者",
                code_hash=_hash_code(config.SEED_USER_CODE),
                role="user",
                allowed_cameras=["*"],  # default user sees the full app minus admin
                feature_flags={},
            )
        )
        await session.commit()
        logger.info("Seeded default admin + user accounts")


class LoginRequest(BaseModel):
    code: str


# ── Detection callback (persists to DB + fires webhook) ─────────────────────


async def _on_detection(result: DetectionResult) -> None:
    """Called from the detector worker when a monkey is detected."""
    # Naive datetime: the column is TIMESTAMP WITHOUT TIME ZONE.
    ts = result.timestamp
    ts_naive = ts.replace(tzinfo=None) if hasattr(ts, "tzinfo") and ts.tzinfo else ts

    try:
        async with async_session() as session:
            det = Detection(
                camera_id=result.camera_id,
                timestamp=ts_naive,
                confidence=result.confidence,
                bbox_x=result.bbox_x,
                bbox_y=result.bbox_y,
                bbox_w=result.bbox_w,
                bbox_h=result.bbox_h,
                image_path=result.image_path,
                annotated_path=result.annotated_path,
                crop_path=result.crop_path,
                track_id=result.track_id,
                trail_path=result.trail_path,
                visit_id=getattr(result, "visit_id", None),
                visit_peak=getattr(result, "visit_peak", None),
                clip_offset_seconds=getattr(result, "clip_offset_seconds", None),
                clip_end_offset_seconds=getattr(result, "clip_end_offset_seconds", None),
                clip_duration_seconds=getattr(result, "clip_duration_seconds", None),
                clip_segments=getattr(result, "clip_segments", None),
                track_boxes=getattr(result, "track_boxes", None),
                track_box_fps=getattr(result, "track_box_fps", None),
            )
            session.add(det)
            await session.commit()
    except Exception as e:
        logger.exception("Failed to persist detection: %s", e)
        return

    # Fire webhook (non-blocking; errors are logged inside)
    await send_notification(result)


async def _on_raw_detection(
    camera_id: str,
    raw_jpeg: bytes,
    annotated_jpeg: bytes,
    top_conf: float,
    n_boxes: int,
    n_tracked: int,
) -> None:
    """DEV back door: forward raw YOLO detections straight to Discord."""
    await send_raw_debug_detection(
        camera_id, raw_jpeg, annotated_jpeg, top_conf, n_boxes, n_tracked,
    )


# ── App lifecycle ────────────────────────────────────────────────────────────


# Latest heartbeat per node, keyed by short hostname. There is one node — the
# process itself — and it records its own beat every EDGE_HEARTBEAT_INTERVAL s.
# In-memory: a restart shows the node again within one interval.
_edge_nodes: dict[str, dict] = {}


def _node_key(hostname: str) -> str:
    """Stable node identity: short hostname, case-folded.

    macOS reports a different suffix depending on how the network resolved it
    (`mini.local` vs `mini.hitronhub.home`), which would otherwise register the
    same machine twice — once live, once permanently 失聯.
    """
    return (hostname or "edge").split(".")[0].lower()


# Nodes silent this long are treated as gone (renamed/retired) and dropped
# from the dashboard rather than lingering as ghosts.
_NODE_FORGET_SECONDS = 3600


# Per-node utilisation history backing the 節點 tab line charts. In-memory
# ONLY — like the node registry itself, it resets on every deploy/restart
# (accepted for now: single replica, and 24 h refills within a day).
_NODE_HISTORY_MAXLEN = 2880  # 24 h at one 30 s heartbeat
_node_history: dict[str, deque[dict]] = {}


def _record_heartbeat(node_key: str, payload: dict) -> None:
    """Store a node's latest beat AND append a history point for the charts.

    """
    st = _edge_nodes.setdefault(node_key, {"alerted": False})
    st["last_seen_ts"] = time.time()
    st["last_seen_iso"] = datetime.now(_TW_TZ).replace(tzinfo=None).isoformat(timespec="seconds")
    st["payload"] = payload
    _node_history.setdefault(node_key, deque(maxlen=_NODE_HISTORY_MAXLEN)).append({
        "ts": st["last_seen_ts"],       # epoch, for range filtering
        "t": st["last_seen_iso"],       # naive TW-local ISO, same as last_seen
        "cpu": payload.get("cpu_percent"),
        "gpu": payload.get("gpu_percent"),
        "rss_gb": payload.get("process_rss_gb"),
    })


def _fresh_nodes() -> list[dict]:
    """Nodes with a recent heartbeat, newest first. Prunes long-dead entries."""
    now = time.time()
    for host in [h for h, st in _edge_nodes.items()
                 if now - st["last_seen_ts"] > _NODE_FORGET_SECONDS]:
        _edge_nodes.pop(host, None)
        _node_history.pop(host, None)
        logger.info("Forgot stale node %s", host)
    out = []
    for host, st in _edge_nodes.items():
        age = now - st["last_seen_ts"]
        out.append({
            "hostname": host,
            "online": age <= config.EDGE_OFFLINE_ALERT_SECONDS,
            "last_seen": st["last_seen_iso"],
            "seconds_since": round(age, 1),
            "heartbeat": st["payload"],
        })
    return sorted(out, key=lambda n: n["seconds_since"])


_LOCAL_NODE = _node_key(socket.gethostname())


def _gpu_percent() -> float | None:
    """Apple Silicon GPU busy % via ioreg (no sudo needed). None off-macOS."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "2"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


async def _node_heartbeat_loop() -> None:
    """Record this node's utilisation and per-camera inference stats.

    Until 2026-09-24 this was two samplers: the edge node's (which read the
    GPU) posted to the cloud, and the cloud container's (cgroup CPU, no GPU)
    recorded itself as "azure-api". Once one process did both jobs it ran the
    container sampler, so the 節點 tab named the Mac mini "azure-api" and lost
    the GPU figure — the number that actually shows the mini's headroom.
    """
    import psutil

    psutil.cpu_percent(interval=None)  # prime: the first reading is always 0.0
    while True:
        await asyncio.sleep(config.EDGE_HEARTBEAT_INTERVAL)
        try:
            mem = psutil.virtual_memory()
            _record_heartbeat(_LOCAL_NODE, {
                "hostname": _LOCAL_NODE,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "gpu_percent": await asyncio.to_thread(_gpu_percent),
                "memory_percent": mem.percent,
                "memory_used_gb": round(mem.used / 1073741824, 2),
                "memory_total_gb": round(mem.total / 1073741824, 2),
                "process_rss_gb": round(psutil.Process().memory_info().rss / 1073741824, 2),
                "cameras": detector_manager.inference_stats()["cameras"],
            })
        except Exception:
            logger.exception("Node heartbeat iteration failed")


async def _orphan_sweep_loop() -> None:
    """Daily storage garbage collection.

    History: deleting detections once left their SHARED trail GIF + rawframes
    ZIP behind whenever the group's last row went away — 3,019 orphans / 70 GB
    accumulated 07-15..07-27 before the shared-file check landed. The check
    now handles the normal path; this sweep is the backstop for everything
    else (crash mid-finalisation, edge push failures, future regressions).
    Only files older than 48 h are eligible, so in-flight detections are never
    touched. Runs once a day; alerts Discord when it reclaims real space.
    """
    import time as _time

    await asyncio.sleep(600)  # let the node settle after deploy
    while True:
        try:
            referenced: set[str] = set()
            async with async_session() as session:
                rows = (await session.execute(select(Detection))).scalars().all()
                for det in rows:
                    referenced |= _detection_files(det)
            cutoff = _time.time() - 48 * 3600

            def _sweep_blocking() -> tuple[int, int]:
                # Sync Azure SDK calls must stay OFF the event loop: listing
                # 4k+ blobs plus per-file deletes blocked it for minutes on
                # 2026-08-03, freezing every in-flight request — the DB pool
                # (5+10) drained and the whole API 500'd until a restart.
                removed = freed = 0
                for name, size, mtime in storage.list_files_with_mtime():
                    if name in referenced or mtime > cutoff:
                        continue
                    if storage.delete(name):
                        removed += 1
                        freed += size
                return removed, freed

            removed, freed = await asyncio.to_thread(_sweep_blocking)
            if removed:
                logger.info("Orphan sweep: removed %d files, %.2f GB", removed, freed / 1e9)
                if freed > 1e9:
                    await send_system_alert(
                        "🧹 儲存空間自動清理",
                        f"回收 {removed} 個孤兒檔案（{freed / 1e9:.1f} GB）— 無偵測紀錄引用且超過 48 小時。",
                        color=0x3498DB,
                    )
        except Exception:
            logger.exception("Orphan sweep failed — retrying tomorrow")
        await asyncio.sleep(24 * 3600)

async def _worker_watchdog_loop() -> None:
    """Detector node: restart any camera whose worker THREAD has died.

    The failure this exists for: on 2026-08-08 an uncaught blob-upload timeout
    ended 場域甲's detector thread. The process stayed up, the stream thread kept
    reconnecting, the node kept beating — and that camera was simply not being
    watched for 62 hours. `DetectorWorker._run` now contains its own exceptions,
    so this is the backstop for whatever kills a thread without raising in it.
    """
    while True:
        try:
            revived = detector_manager.revive_dead()
            if revived:
                logger.error("Watchdog revived dead detector threads: %s", ", ".join(revived))
        except Exception:
            logger.exception("Worker watchdog iteration failed")
        await asyncio.sleep(config.WORKER_WATCHDOG_SECONDS)


def _stalled_cameras(payload: dict | None) -> dict[str, tuple[float, bool]]:
    """{camera: (seconds since last inference, thread is dead)} for blind feeds.

    Reads a node's heartbeat, so the cloud can alarm on a camera going blind even
    though the node itself is perfectly healthy — the exact gap that let 場域甲 sit
    dead for 62 hours. Only connected cameras count: a stream mid-reconnect is a
    different (and self-healing) problem. A gated camera still infers every
    IDLE_GATE_HEARTBEAT_SECONDS, so heartbeat sampling never trips this.
    """
    out: dict[str, tuple[float, bool]] = {}
    for cam in (payload or {}).get("cameras", []) or []:
        if not cam.get("stream_connected"):
            continue
        since = cam.get("seconds_since_last")
        dead = cam.get("worker_alive") is False
        if dead or (since is not None and since > config.CAMERA_STALL_ALERT_SECONDS):
            out[str(cam.get("camera_id", "?"))] = (float(since or 0.0), dead)
    return out


def _reported_cameras(payload: dict | None) -> set[str]:
    """Camera ids the node is currently reporting at all (removed ones vanish)."""
    return {
        str(c.get("camera_id", "?")) for c in (payload or {}).get("cameras", []) or []
    }


async def _check_stalled(host: str, st: dict) -> None:
    """Alert per camera that is connected but no longer inferring.

    A live node is not the same as a watched site — alert once per stall and
    once again on recovery. Shared by the remote-node path and the local node,
    which in a single-node deployment owns every camera there is.
    """
    stalled = _stalled_cameras(st.get("payload"))
    known = st.setdefault("stalled_cams", set())
    for cam, (since, dead) in sorted(stalled.items()):
        if cam in known:
            continue
        known.add(cam)
        logger.error(
            "Camera %s on %s connected but not inferred for %.0fs (thread_dead=%s)",
            cam, host, since, dead,
        )
        await send_system_alert(
            "🔴 攝影機停止偵測",
            f"`{host}` 的 **{cam}** 串流仍連線，但已 {int(since)} 秒沒有推論 — "
            f"該攝影機目前沒有在監測。"
            + ("偵測執行緒已停止，看門狗正在重啟。"
               if dead else "偵測執行緒還活著，請查看節點日誌。"),
        )
    # Recovery only for cameras the node still reports — one that was deleted
    # while stalled has not recovered, it has left.
    still_reported = _reported_cameras(st.get("payload"))
    for cam in sorted(known - set(stalled)):
        known.discard(cam)
        if cam not in still_reported:
            logger.info("Stalled camera %s no longer reported by %s", cam, host)
            continue
        logger.info("Camera %s on %s resumed inference", cam, host)
        await send_system_alert(
            "🟢 攝影機恢復偵測",
            f"`{host}` 的 **{cam}** 已恢復推論。",
            color=0x2ECC71,
        )


async def _camera_watchdog_loop() -> None:
    """Alert Discord per camera that is connected but no longer inferring.

    There is no node-offline alert: the only node is this process, and a dead
    process cannot report its own death. The external canary covers that —
    it polls the public API from outside and alerts when it stops answering.
    """
    while True:
        try:
            st = _edge_nodes.get(_LOCAL_NODE)
            if st is not None:
                await _check_stalled(_LOCAL_NODE, st)
        except Exception:
            logger.exception("Camera watchdog iteration failed")
        await asyncio.sleep(15)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    logger.info("Database initialised")
    await _seed_default_users()

    async with async_session() as session:
        cameras = (await session.execute(select(Camera))).scalars().all()
    for cam in cameras:
        _start_camera_services(cam.camera_id, cam.rtsp_url)
        logger.info("Auto-started camera %s (%s)", cam.camera_id, cam.rtsp_url)

    tasks = [
        asyncio.create_task(_node_heartbeat_loop()),
        asyncio.create_task(_camera_watchdog_loop()),
        asyncio.create_task(_worker_watchdog_loop()),
        asyncio.create_task(_orphan_sweep_loop()),
    ]
    logger.info("Detector thread watchdog started (every %.0fs)", config.WORKER_WATCHDOG_SECONDS)

    yield

    # Shutdown
    for task in tasks:
        task.cancel()
    detector_manager.stop_all()
    stream_manager.stop_all()
    logger.info("All streams and detectors stopped")


app = FastAPI(
    title="Monkey Detection Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: "*" => allow all (no credentials); else explicit origins (with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth ─────────────────────────────────────────────────────────────────────


@app.post("/api/auth/login")
async def login(body: LoginRequest, session: AsyncSession = Depends(get_session)):
    user = (
        await session.execute(
            select(User).where(User.code_hash == _hash_code(body.code.strip()))
        )
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="存取碼錯誤")
    if not user.enabled:
        raise HTTPException(status_code=403, detail="帳號已停用")
    return {"token": _sign_token(user.id, user.role), "role": user.role}


@app.get("/api/auth/me", response_model=MeOut)
async def me(user: User = Depends(get_current_user)):
    """The current user's identity + effective permissions (for UI gating)."""
    return user


# ── Health ───────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── Cameras ──────────────────────────────────────────────────────────────────


def _build_camera_out(cam: Camera) -> CameraOut:
    s = stream_manager.get(cam.camera_id)
    return CameraOut(
        id=cam.id,
        camera_id=cam.camera_id,
        rtsp_url=cam.rtsp_url,
        name=cam.name,
        created_at=cam.created_at,
        poc=cam.camera_id in config.POC_CAMERA_IDS,
        connected=bool(s and s.connected),
        is_alive=bool(s and s.is_alive),
    )





def _start_camera_services(camera_id: str, rtsp_url: str) -> None:
    # POC sources run only where POC_EDGE_ENABLED opts them in. The old test
    # also required NODE_ROLE == "edge", so after the mini became the whole
    # system (NODE_ROLE=all) poc-camera-2 was silently never started, even
    # though the mini's .env sets POC_EDGE_ENABLED=true (2026-09-21..24).
    if camera_id in config.POC_CAMERA_IDS and not config.POC_EDGE_ENABLED:
        logger.info("POC camera %s registered; POC_EDGE_ENABLED is off", camera_id)
        return
    stream_manager.add(camera_id, rtsp_url)
    detector_manager.start(
        camera_id,
        asyncio.get_running_loop(),
        on_detection=_on_detection,
        on_raw_detection=_on_raw_detection,
    )


def _stop_camera_services(camera_id: str) -> None:
    detector_manager.stop(camera_id)
    stream_manager.remove(camera_id)


@app.post("/api/cameras", response_model=CameraOut, status_code=201, dependencies=[Depends(require_admin)])
async def add_camera(body: CameraCreate, session: AsyncSession = Depends(get_session)):
    # Check uniqueness
    existing = (
        await session.execute(select(Camera).where(Camera.camera_id == body.camera_id))
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Camera ID already exists")

    cam = Camera(camera_id=body.camera_id, rtsp_url=body.rtsp_url, name=body.name)
    session.add(cam)
    await session.commit()
    await session.refresh(cam)

    _start_camera_services(cam.camera_id, cam.rtsp_url)
    return _build_camera_out(cam)


@app.get("/api/cameras", response_model=list[CameraOut])
async def list_cameras(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Camera).order_by(Camera.created_at.desc()))
    cameras = result.scalars().all()
    cameras = [c for c in cameras if _camera_allowed(user, c.camera_id)]
    return [_build_camera_out(cam) for cam in cameras]


@app.patch("/api/admin/cameras/{camera_id}", response_model=CameraOut, dependencies=[Depends(require_admin)])
async def admin_update_camera(
    camera_id: str,
    body: CameraUpdate,
    session: AsyncSession = Depends(get_session),
):
    cam = (
        await session.execute(select(Camera).where(Camera.camera_id == camera_id))
    ).scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    rtsp_changed = False
    if body.name is not None:
        cam.name = body.name.strip()
    if body.rtsp_url is not None:
        new_url = body.rtsp_url.strip()
        if not new_url:
            raise HTTPException(status_code=400, detail="RTSP URL cannot be empty")
        rtsp_changed = new_url != cam.rtsp_url
        cam.rtsp_url = new_url

    if body.name is None and body.rtsp_url is None:
        raise HTTPException(status_code=400, detail="No fields to update")

    await session.commit()
    await session.refresh(cam)

    if rtsp_changed:
        _stop_camera_services(camera_id)
        _start_camera_services(camera_id, cam.rtsp_url)
        logger.info("Camera %s RTSP updated and stream restarted", camera_id)

    return _build_camera_out(cam)


@app.delete("/api/admin/cameras/{camera_id}", response_model=CameraDeleteOut, dependencies=[Depends(require_admin)])
async def admin_delete_camera(camera_id: str, session: AsyncSession = Depends(get_session)):
    cam = (
        await session.execute(select(Camera).where(Camera.camera_id == camera_id))
    ).scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    detections_preserved = (
        await session.execute(
            select(func.count(Detection.id)).where(Detection.camera_id == camera_id)
        )
    ).scalar() or 0

    _stop_camera_services(camera_id)
    await session.delete(cam)
    await session.commit()
    logger.info(
        "Camera %s removed (%d detection records preserved)",
        camera_id,
        detections_preserved,
    )
    return CameraDeleteOut(
        camera_id=camera_id,
        detections_preserved=detections_preserved,
    )


# ── Stream ───────────────────────────────────────────────────────────────────


@app.get("/api/stream/{camera_id}/mjpeg")
async def stream_mjpeg(camera_id: str, user: User = Depends(get_current_user)):
    if not _feature_enabled(user, "live_feeds") or not _camera_allowed(user, camera_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    stream = stream_manager.get(camera_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Camera stream not found")
    return StreamingResponse(
        stream_manager.mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Detections ───────────────────────────────────────────────────────────────


@app.get("/api/detections", response_model=DetectionListOut)
async def list_detections(
    camera_id: Optional[str] = Query(None),
    trail_path: Optional[str] = Query(
        None,
        max_length=255,
        description="只回傳共用這支影片的列。時間軸需要整支影片的動物，"
        "而它們不一定落在同一頁。",
    ),
    sort_by: Literal["time", "count"] = Query(
        "time",
        description="排序基準：time 依時間；count 依同次造訪動物數。",
    ),
    sort_order: Literal["asc", "desc"] = Query(
        "desc",
        description="排序方向：asc 升冪；desc 降冪。",
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    descending = sort_order == "desc"
    order_columns = [Detection.timestamp, Detection.id]
    if sort_by == "count":
        order_columns.insert(0, func.coalesce(Detection.visit_peak, 0))
    query = select(Detection).order_by(
        *(column.desc() if descending else column.asc() for column in order_columns)
    )
    count_query = select(func.count(Detection.id))

    # One clip now carries every animal of a visit — sixteen rows on the
    # longest — and the page they land on is an accident of when they were
    # recorded. Filtering the current page for them, which is what the review
    # panel did, silently dropped every lane that fell on the other side of a
    # page boundary: five rows split 2/3 across pages 1 and 2 showed a timeline
    # with two animals on it (measured 2026-08-30).
    if trail_path:
        query = query.where(Detection.trail_path == trail_path)
        count_query = count_query.where(Detection.trail_path == trail_path)

    if camera_id:
        if not _camera_allowed(user, camera_id):
            raise HTTPException(status_code=403, detail="Forbidden")
        query = query.where(Detection.camera_id == camera_id)
        count_query = count_query.where(Detection.camera_id == camera_id)
    else:
        scope = _camera_scope_criteria(user, Detection.camera_id)
        query = query.where(*scope)
        count_query = count_query.where(*scope)

    visibility = _visibility_criteria(user)
    query = query.where(*visibility)
    count_query = count_query.where(*visibility)

    total = (await session.execute(count_query)).scalar() or 0
    items = (await session.execute(query.offset(offset).limit(limit))).scalars().all()

    # visible_to_users is computed, not stored — fill it per row so the admin
    # dashboard can badge/act without re-implementing the gate client-side.
    out = []
    for det in items:
        row = DetectionOut.model_validate(det)
        row.visible_to_users = _visible_to_users(det)
        out.append(row)
    return DetectionListOut(total=total, items=out)


# A detection row is one LOST TRACK, not one visit. Rows separated by less than
# this are the same troop still in front of the camera. 20 s measured against
# 2026-08-13: it reunited 46 of the 47 scenes that looked like under-reporting.
EVENT_GAP_SECONDS = float(os.getenv("EVENT_GAP_SECONDS", "20"))
# Grouping is sequential, so it needs the rows, not a COUNT. Bounded so a year
# of data cannot turn one request into a full-table read; `truncated` tells the
# client when the window was cut rather than silently showing partial events.
EVENT_SCAN_CAP = int(os.getenv("EVENT_SCAN_CAP", "5000"))
# Two rows this close in space and time are one animal that lost and regained
# its id, not two animals. Verified on t432/t433: same second, 2 px apart.
_DUP_IOU = 0.5
_DUP_SECONDS = 5.0


def _bbox_iou_rows(a: Detection, b: Detection) -> float:
    ax2, ay2 = a.bbox_x + a.bbox_w, a.bbox_y + a.bbox_h
    bx2, by2 = b.bbox_x + b.bbox_w, b.bbox_y + b.bbox_h
    ix1, iy1 = max(a.bbox_x, b.bbox_x), max(a.bbox_y, b.bbox_y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.bbox_w * a.bbox_h + b.bbox_w * b.bbox_h - inter
    return inter / union if union > 0 else 0.0


# Retroactive twin of the two producer-side merges, for rows already written.
#
# 🛑 READ THIS BEFORE TRUSTING A CLASS LABEL. A detection row does NOT record a
# moment. `timestamp` is `track.first_seen` — when the animal ARRIVED — while
# `bbox` is `track.best_bbox`, the box from the PEAK-confidence frame, which can
# be a minute later (detector.py `_emit_detection`). So a small time gap and a
# high IoU describe two DIFFERENT instants, and no comparison available here can
# establish that two rows were ever simultaneous. The producer-side merge can:
# it compares boxes on shared FRAME INDICES. This one cannot.
#
# Both classes below are therefore INFERENCES, and the honest difference between
# them is only how much room the inference has:
#   A  arrivals within _MERGE_SAME_INSTANT_SECONDS, boxes overlapping.
#   B  arrivals further apart, up to _MERGE_SEQUENTIAL_SECONDS.
# Class B additionally inherits _adopt_continuation's documented residual risk:
# two animals using one perch in sequence WILL be merged. A is the narrower bet,
# not a proof — which is why it is the default and B is opt-in.
_MERGE_SAME_INSTANT_SECONDS = _DUP_SECONDS
# Deliberately its OWN constant, not TRACK_CONTINUATION_SECONDS. Coupling them
# meant that disabling the live continuation merge (setting it to 0) silently
# disabled this backfill too, and that lowering it could make the "narrow"
# same-instant window the WIDER of the two.
_MERGE_SEQUENTIAL_SECONDS = 20.0
# Pairwise windows chain transitively, and without a ceiling that chaining has
# no bound at all: a macaque that sits on one perch for an hour, filed every
# ~15 s, links 1-2, 2-3, ... into ONE group of 150 rows and 149 deletions. A
# group may not span more than this no matter how tight its individual links.
_MERGE_MAX_SPAN_SECONDS = 60.0


def _duplicate_row_groups(
    rows: list[Detection], same_instant_only: bool = True
) -> list[dict]:
    """Chain rows that are probably one animal recorded more than once.

    Returns one dict per group: which row survives, which are folded into it,
    and what the survivor has to become. Reads only — nothing is written here,
    so the dry run and the apply cannot drift apart.
    """
    window = _MERGE_SAME_INSTANT_SECONDS if same_instant_only else _MERGE_SEQUENTIAL_SECONDS
    per_camera: dict[str, list[Detection]] = {}
    for det in rows:
        per_camera.setdefault(det.camera_id, []).append(det)

    groups: list[dict] = []
    for dets in per_camera.values():
        dets.sort(key=lambda d: d.timestamp)
        # Grow groups in time order rather than union-find, so the span ceiling
        # is enforced as the chain is built instead of being discovered after a
        # whole day has already been linked into one blob.
        open_groups: list[dict] = []
        for det in dets:
            joined = None
            for grp in open_groups:
                span = (det.timestamp - grp["members"][0].timestamp).total_seconds()
                if span > _MERGE_MAX_SPAN_SECONDS:
                    continue
                link = next(
                    (
                        (m, gap, iou)
                        for m in grp["members"]
                        for gap in [(det.timestamp - m.timestamp).total_seconds()]
                        if gap <= window
                        for iou in [_bbox_iou_rows(m, det)]
                        if iou >= _DUP_IOU
                    ),
                    None,
                )
                if link:
                    joined = (grp, link)
                    break
            if joined is None:
                open_groups.append({"members": [det], "classes": set(), "links": []})
                continue
            grp, (mate, gap, iou) = joined
            grp["members"].append(det)
            grp["classes"].add("A" if gap <= _MERGE_SAME_INSTANT_SECONDS else "B")
            grp["links"].append({
                "ids": [mate.id, det.id],
                "gap_seconds": round(gap, 2),
                "iou": round(iou, 3),
            })
        groups.extend(g for g in open_groups if len(g["members"]) > 1)

    out: list[dict] = []
    for grp in groups:
        members = grp["members"]
        by_id = {d.id: d for d in members}
        # A lock is the user's endorsement that there IS an animal there and the
        # box is right (2026-08-27), so an endorsed row is the preferred
        # survivor — its box is the one worth keeping — and the endorsement
        # carries onto whichever row survives. Every endorsed row that does NOT
        # survive is named in `dropped_locked`, because deleting one spends a
        # human's review and that must never be summarised as a count.
        locked = [d for d in members if d.locked]
        keep = max(locked or members, key=lambda d: (d.confidence, d.id))
        drop = [d for d in members if d.id != keep.id]
        out.append({
            "camera_id": keep.camera_id,
            "classes": "".join(sorted(grp["classes"])),
            "keep": keep.id,
            "drop": [d.id for d in drop],
            "keep_confidence": keep.confidence,
            # The row should be stamped when the animal arrived, not when the
            # surviving fragment happened to be acquired — same rule
            # _adopt_continuation applies by prepending first_seen.
            "timestamp": min(d.timestamp for d in members),
            "retimed": min(d.timestamp for d in members) != keep.timestamp,
            "locked": bool(locked),
            "dropped_locked": [d.id for d in drop if d.locked],
            "span_seconds": round(
                (max(d.timestamp for d in members)
                 - min(d.timestamp for d in members)).total_seconds(), 2
            ),
            "track_ids": [by_id[d.id].track_id for d in members],
            "links": grp["links"],
        })
    out.sort(key=lambda g: g["timestamp"])
    return out



@app.get("/api/detections/events", response_model=DetectionEventListOut,
         dependencies=[Depends(require_admin)])
async def list_detection_events(
    camera_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    gap_seconds: float = Query(EVENT_GAP_SECONDS, ge=1, le=3600),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Detections grouped into the visit they belong to.

    This used to be admin-only, and the reason was sound: a visit was an
    INFERENCE laid over the rows — a 20 s gap plus `estimated_animals`, a guess
    at how many animals were behind the track ids — and users should only see
    things that can be taken as true (2026-08-14).

    That premise is now half gone. The producer emits a whole visit at once and
    stamps every row of it with the same `visit_id`, so a NEW visit is something
    that was RECORDED, and `visit_peak` is counted from the tracks' own frames
    rather than estimated.

    It stays admin-only anyway, and deliberately. 223 of 818 rows still have no
    id and fall back to cutting on a 20 s silence, and the 587 that were
    backfilled were grouped from frame evidence AFTER the fact — neither is the
    producer watching a visit happen. `basis` distinguishes them for us; putting
    a mixed view in front of users would ask them to read that distinction, and
    the rule (2026-08-14) is that users get what can be taken as true, not what
    needs a caveat. Revisit when `basis` is "recorded" for everything a user can
    reach.
    """
    query = select(Detection).order_by(Detection.timestamp.desc())

    if camera_id:
        if not _camera_allowed(user, camera_id):
            raise HTTPException(status_code=403, detail="Forbidden")
        query = query.where(Detection.camera_id == camera_id)
    else:
        query = query.where(*_camera_scope_criteria(user, Detection.camera_id))

    query = query.where(*_visibility_criteria(user))

    rows = (await session.execute(query.limit(EVENT_SCAN_CAP + 1))).scalars().all()
    truncated = len(rows) > EVENT_SCAN_CAP
    rows = rows[:EVENT_SCAN_CAP]

    # A row that carries a visit_id was emitted as part of that visit; nothing
    # here has to guess where it belongs. Only the rows without one fall back to
    # cutting on a gap, and they are marked as such.
    events: list[DetectionEventOut] = []
    recorded: dict[str, list[Detection]] = {}
    unstamped: dict[str, list[Detection]] = {}
    for det in rows:
        if det.visit_id:
            recorded.setdefault(det.visit_id, []).append(det)
        else:
            unstamped.setdefault(det.camera_id, []).append(det)

    for dets in recorded.values():
        dets.sort(key=lambda d: d.timestamp)
        events.append(_build_event(dets[0].camera_id, dets, basis="recorded"))

    for cam, dets in unstamped.items():
        dets.sort(key=lambda d: d.timestamp)
        bucket: list[Detection] = []
        for det in dets:
            if bucket and (det.timestamp - bucket[-1].timestamp).total_seconds() > gap_seconds:
                events.append(_build_event(cam, bucket, basis="gap"))
                bucket = []
            bucket.append(det)
        if bucket:
            events.append(_build_event(cam, bucket, basis="gap"))

    events.sort(key=lambda e: e.started_at, reverse=True)
    return DetectionEventListOut(
        total=len(events),
        items=events[offset:offset + limit],
        gap_seconds=gap_seconds,
        scanned=len(rows),
        truncated=truncated,
    )


def _build_event(
    camera_id: str, bucket: list[Detection], basis: str = "gap"
) -> DetectionEventOut:
    cover = max(bucket, key=lambda d: d.confidence)
    cover_out = DetectionOut.model_validate(cover)
    cover_out.visible_to_users = _visible_to_users(cover)
    started, ended = bucket[0].timestamp, bucket[-1].timestamp
    return DetectionEventOut(
        event_id=bucket[0].visit_id or f"{camera_id}:{started.isoformat()}",
        camera_id=camera_id,
        started_at=started,
        ended_at=ended,
        duration_seconds=(ended - started).total_seconds(),
        detection_count=len(bucket),
        distinct_tracks=len({d.track_id for d in bucket}),
        # Counted at emit time from the tracks' own frames. None on rows old
        # enough to predate it — the UI shows the row count alone there rather
        # than substituting a guess.
        peak_animals=next((d.visit_peak for d in bucket if d.visit_peak), None),
        basis=basis,
        max_confidence=cover.confidence,
        trail_paths=sorted({d.trail_path for d in bucket if d.trail_path}),
        detection_ids=[d.id for d in bucket],
        cover=cover_out,
    )


@app.get("/api/detections/stats", response_model=DetectionStatsOut)
async def detection_stats(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    now = datetime.now(_TW_TZ)
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Use naive datetime for PostgreSQL TIMESTAMP WITHOUT TIME ZONE comparison
    start_of_today_naive = start_of_today.replace(tzinfo=None)

    # Scope every figure to the cameras this user may see (admin/'*' = all)
    # AND to the rows the non-admin confidence gate lets them see.
    vis = _visibility_criteria(user)

    def _scope(query):
        return query.where(*_camera_scope_criteria(user, Detection.camera_id), *vis)

    total_all = (
        await session.execute(_scope(select(func.count(Detection.id))))
    ).scalar() or 0

    total_today = (
        await session.execute(
            _scope(
                select(func.count(Detection.id)).where(
                    Detection.timestamp >= start_of_today_naive
                )
            )
        )
    ).scalar() or 0

    last_det = (
        await session.execute(
            _scope(select(Detection.timestamp))
            .order_by(Detection.timestamp.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    # Count only cameras delivering meaningful (non-black, non-frozen) frames.
    # `is_alive` verifies actual frame variance, so a dead feed falls out of the
    # count within ~15s. Restricted users only count their allowed cameras.
    cameras_active = sum(
        1
        for cid in stream_manager.list_ids()
        if _camera_allowed(user, cid)
        and (s := stream_manager.get(cid)) is not None
        and s.is_alive
    )

    return DetectionStatsOut(
        total_today=total_today,
        total_all=total_all,
        last_detection=last_det,
        cameras_active=cameras_active,
    )


@app.get("/api/detections/heatmap", response_model=DetectionHeatmapOut)
async def detection_heatmap(
    days: int = Query(366, ge=1, le=731),
    camera_id: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Detection counts bucketed by (date, hour, camera) for the analytics tab.

    Timestamps are stored naive in TW local time, so the date/hour fall
    straight out of the stored value. Bucketing happens in Python to stay
    dialect-neutral (SQLite locally, PostgreSQL in production).
    """
    now = datetime.now(_TW_TZ).replace(tzinfo=None)
    since = (now - timedelta(days=days - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    query = select(Detection.camera_id, Detection.timestamp).where(
        Detection.timestamp >= since
    )
    if camera_id:
        if not _camera_allowed(user, camera_id):
            raise HTTPException(status_code=403, detail="Forbidden")
        query = query.where(Detection.camera_id == camera_id)
    else:
        query = query.where(*_camera_scope_criteria(user, Detection.camera_id))

    query = query.where(*_visibility_criteria(user))

    rows = (await session.execute(query)).all()

    buckets: dict[tuple[str, int, str], int] = {}
    for cam, ts in rows:
        key = (ts.strftime("%Y-%m-%d"), ts.hour, cam)
        buckets[key] = buckets.get(key, 0) + 1

    return DetectionHeatmapOut(
        start=since.strftime("%Y-%m-%d"),
        end=now.strftime("%Y-%m-%d"),
        total=len(rows),
        cells=[
            HeatmapCellOut(date=d, hour=h, camera_id=c, count=n)
            for (d, h, c), n in sorted(buckets.items())
        ],
    )


# ── Webhooks ─────────────────────────────────────────────────────────────


@app.post("/api/webhooks", response_model=WebhookOut, status_code=201, dependencies=[Depends(require_admin)])
async def create_webhook(body: WebhookCreate, session: AsyncSession = Depends(get_session)):
    wh = Webhook(name=body.name, url=body.url, enabled=body.enabled)
    session.add(wh)
    await session.commit()
    await session.refresh(wh)
    return wh


@app.get("/api/webhooks", response_model=list[WebhookOut], dependencies=[Depends(require_admin)])
async def list_webhooks(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Webhook).order_by(Webhook.created_at.desc()))
    return result.scalars().all()


@app.patch("/api/webhooks/{webhook_id}", response_model=WebhookOut, dependencies=[Depends(require_admin)])
async def update_webhook(
    webhook_id: int, body: WebhookUpdate, session: AsyncSession = Depends(get_session)
):
    wh = (
        await session.execute(select(Webhook).where(Webhook.id == webhook_id))
    ).scalar_one_or_none()
    if not wh:
        raise HTTPException(status_code=404, detail="Webhook not found")
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(wh, field, value)
    await session.commit()
    await session.refresh(wh)
    return wh


@app.delete("/api/webhooks/{webhook_id}", status_code=204, dependencies=[Depends(require_admin)])
async def delete_webhook(webhook_id: int, session: AsyncSession = Depends(get_session)):
    wh = (
        await session.execute(select(Webhook).where(Webhook.id == webhook_id))
    ).scalar_one_or_none()
    if not wh:
        raise HTTPException(status_code=404, detail="Webhook not found")
    await session.delete(wh)
    await session.commit()


@app.post("/api/webhooks/{webhook_id}/test", dependencies=[Depends(require_admin)])
async def test_single_webhook(webhook_id: int, session: AsyncSession = Depends(get_session)):
    wh = (
        await session.execute(select(Webhook).where(Webhook.id == webhook_id))
    ).scalar_one_or_none()
    if not wh:
        raise HTTPException(status_code=404, detail="Webhook not found")

    import httpx
    now = datetime.now(_TW_TZ).isoformat()
    payload = {
        "content": f"\U0001f514 **Primate Watch** \u2014 Test notification\nTimestamp: {now}",
        "embeds": [{
            "title": "Webhook Test",
            "description": f"Test notification from webhook: {wh.name or wh.url}",
            "color": 0x22C55E,
            "timestamp": now,
        }],
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(wh.url, json=payload)
            resp.raise_for_status()
        wh.last_error = ""
        await session.commit()
        return {"status": "sent", "http_status": resp.status_code}
    except Exception as exc:
        wh.last_error = str(exc)[:500]
        await session.commit()
        raise HTTPException(status_code=502, detail=f"Webhook failed: {exc}")


# ── Snapshots ────────────────────────────────────────────────────────────────


@app.api_route("/api/snapshots/{filename:path}", methods=["GET", "HEAD"], dependencies=[Depends(require_user)])
async def get_snapshot(filename: str):
    # Detection assets are flat names. The isolated rebuild review is the one
    # deliberate exception: its publisher keeps clips under the controlled
    # `rebuild-review/` blob prefix. Reject every other path before it reaches
    # either LocalStorage or BlobStorage.
    if "\\" in filename or ".." in filename or (
        "/" in filename and not filename.startswith("rebuild-review/")
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not storage.exists(filename):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    if filename.endswith(".mp4"):
        media = "video/mp4"
    elif filename.endswith(".gif"):
        media = "image/gif"
    else:
        media = "image/jpeg"
    # Blob-backed deployments: 302 to a short-lived SAS URL so the bytes flow
    # straight from Azure Storage to the browser — the 1 vCPU container was
    # the pipe for every 12-15 MB trail GIF and dominated load times. The
    # redirect itself is cacheable for most of the SAS lifetime.
    signed = getattr(storage, "signed_url", lambda *_: None)(filename)
    if signed:
        return RedirectResponse(signed, status_code=302,
                                headers={"Cache-Control": "private, max-age=2700"})
    if isinstance(storage, LocalStorage):
        # Browsers probe MP4 headers and seek with byte ranges. A generic
        # stream ignores those ranges, breaking WebKit playback and sending
        # the whole clip for each seek. FileResponse handles Range and HEAD.
        return FileResponse(storage.local_path(filename), media_type=media,
                            headers={"Cache-Control": "public, max-age=86400"})
    stream = storage.download_stream(filename)
    if stream is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return StreamingResponse(stream, media_type=media, headers={"Cache-Control": "public, max-age=86400"})


def _rebuild_manifest_name() -> str:
    return os.getenv("REBUILD_REVIEW_MANIFEST", "rebuild-review/latest.json")


def _load_rebuild_manifest() -> dict:
    """Read and validate the immutable review manifest from shared storage."""
    manifest_name = _rebuild_manifest_name()
    raw = storage.download(manifest_name)
    if raw is None:
        raise HTTPException(status_code=404, detail="目前沒有可審查的回填 run")
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("Invalid rebuild review manifest %s: %s", manifest_name, exc)
        raise HTTPException(status_code=502, detail="回填審查 manifest 無法解析") from exc
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=502, detail="回填審查 manifest 格式錯誤")
    return manifest


@app.get("/api/admin/rebuild-review", dependencies=[Depends(require_admin)])
async def get_rebuild_review():
    """Return the isolated historical-rebuild manifest for admin review."""
    return _load_rebuild_manifest()


def _reid_window(det: Detection) -> tuple[float, float] | None:
    """Return the measured visibility window for a track row, if available."""
    windows: list[tuple[float, float]] = []
    if isinstance(det.clip_segments, list):
        for segment in det.clip_segments:
            if not isinstance(segment, (list, tuple)) or len(segment) < 2:
                continue
            try:
                start, end = float(segment[0]), float(segment[1])
            except (TypeError, ValueError):
                continue
            if end >= start:
                windows.append((start, end))
    if windows:
        return min(start for start, _ in windows), max(end for _, end in windows)

    if det.clip_offset_seconds is None:
        return None
    start = float(det.clip_offset_seconds)
    end = (
        float(det.clip_end_offset_seconds)
        if det.clip_end_offset_seconds is not None
        else start
    )
    return min(start, end), max(start, end)


def _reid_temporal_relation(
    first: Detection, second: Detection
) -> tuple[Literal["separate", "overlap", "unknown"], float | None]:
    """Describe whether two tracklets overlap in their shared clip."""
    first_window = _reid_window(first)
    second_window = _reid_window(second)
    if first_window is None or second_window is None:
        return "unknown", None
    if first_window[1] <= second_window[0]:
        return "separate", max(0.0, second_window[0] - first_window[1])
    if second_window[1] <= first_window[0]:
        return "separate", max(0.0, first_window[0] - second_window[1])
    return "overlap", 0.0


def _reid_track_payload(det: Detection) -> dict:
    window = _reid_window(det)
    return {
        "id": det.id,
        "track_id": det.track_id,
        "camera_id": det.camera_id,
        "timestamp": det.timestamp,
        "confidence": det.confidence,
        "crop_path": det.crop_path,
        "image_path": det.image_path,
        "annotated_path": det.annotated_path,
        "trail_path": det.trail_path,
        "clip_duration_seconds": det.clip_duration_seconds,
        "start_seconds": window[0] if window else None,
        "end_seconds": window[1] if window else None,
        "frame_count": len(det.track_boxes) if isinstance(det.track_boxes, list) else 0,
        "track_boxes": det.track_boxes if isinstance(det.track_boxes, list) else None,
        "track_box_fps": det.track_box_fps,
    }


def _select_reid_batch(candidates: list[dict], size: int) -> list[dict]:
    """Choose one deterministic, clip-balanced review batch.

    The batch is selected before labels are filtered. That means labeling a
    pair shrinks this round instead of pulling a new pair in from the tail of
    the candidate pool. Round-robin selection gives small clips a chance to be
    represented instead of spending the whole batch on one crowded clip.
    """
    if len(candidates) <= size:
        return candidates

    by_clip: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_clip[candidate["trail_path"]].append(candidate)
    for clip_candidates in by_clip.values():
        clip_candidates.sort(key=lambda candidate: candidate["pair_id"])

    selected: list[dict] = []
    clips = deque(iter(by_clip[trail_path]) for trail_path in sorted(by_clip))
    while clips and len(selected) < size:
        clip = clips.popleft()
        candidate = next(clip, None)
        if candidate is not None:
            selected.append(candidate)
            clips.append(clip)
    return selected


def _annotate_reid_candidate(candidate: dict) -> dict:
    """Add ephemeral queue decisions without changing the database row."""
    system_label, system_reason = reid_system_decision(candidate)
    priority, review_reason = reid_review_priority(candidate)
    candidate["system_label"] = system_label if candidate.get("label") is None else None
    candidate["decision_source"] = (
        "human" if candidate.get("label") is not None
        else "geometry_auto" if system_label is not None
        else None
    )
    candidate["review_required"] = candidate.get("label") is None and system_label is None
    candidate["review_priority"] = priority
    candidate["review_reason"] = system_reason or review_reason
    return candidate


def _sort_reid_review(candidates: list[dict], status: str) -> list[dict]:
    """Put the highest-information unresolved pair first, stably."""
    if status == "pending":
        return sorted(candidates, key=lambda candidate: (-candidate["review_priority"], candidate["pair_id"]))
    return sorted(candidates, key=lambda candidate: candidate["pair_id"])


@app.get(
    "/api/admin/reid/pairs",
    response_model=ReIDPairListOut,
    dependencies=[Depends(require_admin)],
)
async def get_reid_pairs(
    status: Literal["pending", "labeled", "auto", "all"] = Query("pending"),
    limit: int = Query(1, ge=1, le=50),
    offset: int = Query(0, ge=0),
    review_size: int = Query(400, ge=1, le=2000),
    session: AsyncSession = Depends(get_session),
):
    """List deterministic same-clip tracklet pairs for ReID annotation.

    This is deliberately a research queue, not an identity resolver. Candidate
    pairs are only made from rows sharing one trail video and camera, so a
    reviewer never has to make a cross-clip identity claim by accident.
    """
    detections = (
        await session.execute(
            select(Detection)
            .where(Detection.trail_path.is_not(None), Detection.trail_path != "")
            .order_by(Detection.trail_path.asc(), Detection.id.asc())
        )
    ).scalars().all()
    labels = {
        (min(label.detection_a_id, label.detection_b_id),
         max(label.detection_a_id, label.detection_b_id)): label
        for label in (
            await session.execute(select(ReIDPairLabel))
        ).scalars().all()
    }

    by_clip: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_clip[detection.trail_path].append(detection)

    candidates: list[dict] = []
    for trail_path in sorted(by_clip):
        rows = sorted(
            by_clip[trail_path],
            key=lambda row: (
                _reid_window(row)[0] if _reid_window(row) is not None else float("inf"),
                row.id,
            ),
        )
        for index, first in enumerate(rows):
            for second in rows[index + 1 :]:
                # A shared filename should identify one clip, but this guard
                # keeps a malformed historical path from becoming a cross-camera
                # identity label.
                if first.camera_id != second.camera_id:
                    continue
                first_id, second_id = sorted((first.id, second.id))
                saved = labels.get((first_id, second_id))
                relation, gap = _reid_temporal_relation(first, second)
                candidates.append(
                    {
                        "pair_id": f"{first_id}:{second_id}",
                        "camera_id": first.camera_id,
                        "trail_path": trail_path,
                        "clip_duration_seconds": (
                            first.clip_duration_seconds
                            if first.clip_duration_seconds is not None
                            else second.clip_duration_seconds
                        ),
                        "temporal_relation": relation,
                        "gap_seconds": gap,
                        "a": _reid_track_payload(first),
                        "b": _reid_track_payload(second),
                        "label": saved.label if saved else None,
                        "note": saved.note if saved else "",
                        "labeled_at": saved.updated_at if saved else None,
                    }
                )

    for candidate in candidates:
        _annotate_reid_candidate(candidate)

    pending = sum(candidate["review_required"] for candidate in candidates)
    labeled = sum(candidate["label"] is not None for candidate in candidates)
    auto_resolved = sum(candidate["system_label"] is not None for candidate in candidates)
    review_batch = _select_reid_batch(candidates, review_size)
    batch_pending = sum(candidate["review_required"] for candidate in review_batch)
    batch_labeled = sum(candidate["label"] is not None for candidate in review_batch)
    batch_auto_resolved = sum(candidate["system_label"] is not None for candidate in review_batch)
    if status == "pending":
        filtered = [candidate for candidate in review_batch if candidate["review_required"]]
    elif status == "labeled":
        filtered = [candidate for candidate in review_batch if candidate["label"] is not None]
    elif status == "auto":
        filtered = [candidate for candidate in review_batch if candidate["system_label"] is not None]
    else:
        filtered = review_batch
    filtered = _sort_reid_review(filtered, status)

    return ReIDPairListOut(
        total=len(filtered),
        candidate_total=len(candidates),
        pending=pending,
        labeled=labeled,
        batch_total=len(review_batch),
        batch_pending=batch_pending,
        batch_labeled=batch_labeled,
        auto_resolved=auto_resolved,
        batch_auto_resolved=batch_auto_resolved,
        batch_review_pending=batch_pending,
        pairs=filtered[offset : offset + limit],
    )


@app.post(
    "/api/admin/reid/labels",
    response_model=ReIDLabelOut,
    dependencies=[Depends(require_admin)],
)
async def save_reid_label(
    body: ReIDLabelCreate,
    session: AsyncSession = Depends(get_session),
):
    """Save a human ReID verdict without mutating either Detection row."""
    if body.detection_a_id == body.detection_b_id:
        raise HTTPException(status_code=400, detail="不能把同一筆偵測和自己配對")

    first_id, second_id = sorted((body.detection_a_id, body.detection_b_id))
    detections = (
        await session.execute(
            select(Detection).where(Detection.id.in_([first_id, second_id]))
        )
    ).scalars().all()
    by_id = {detection.id: detection for detection in detections}
    if len(by_id) != 2:
        raise HTTPException(status_code=404, detail="找不到要標記的偵測紀錄")

    first, second = by_id[first_id], by_id[second_id]
    if (
        not first.trail_path
        or first.trail_path != second.trail_path
        or first.camera_id != second.camera_id
    ):
        raise HTTPException(
            status_code=400,
            detail="ReID 標記只允許同一支影片、同一台攝影機的兩條軌跡",
        )

    saved = (
        await session.execute(
            select(ReIDPairLabel).where(
                ReIDPairLabel.detection_a_id == first_id,
                ReIDPairLabel.detection_b_id == second_id,
            )
        )
    ).scalar_one_or_none()
    if saved is None:
        saved = ReIDPairLabel(
            detection_a_id=first_id,
            detection_b_id=second_id,
        )
        session.add(saved)
    saved.label = body.label
    saved.note = body.note.strip()
    saved.updated_at = datetime.now(_TW_TZ).replace(tzinfo=None)

    await session.commit()
    await session.refresh(saved)
    return ReIDLabelOut(
        detection_a_id=saved.detection_a_id,
        detection_b_id=saved.detection_b_id,
        label=saved.label,
        note=saved.note,
        updated_at=saved.updated_at,
    )
def _backup_detection_row(det: Detection) -> dict:
    """Serialize every Detection column so rollback does not lose new fields."""
    result = {}
    for column in Detection.__table__.columns:
        value = getattr(det, column.name)
        result[column.name] = value.isoformat() if isinstance(value, datetime) else value
    return result


def _naive_rebuild_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("clip 缺少 timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def _rebuild_visit_peak(rows: list[dict]) -> int | None:
    counts: dict[int, int] = {}
    for row in rows:
        for box in row.get("boxes") or []:
            if box:
                frame = int(box[0])
                counts[frame] = counts.get(frame, 0) + 1
    return max(counts.values(), default=None)


# A previous review publish is flattened to this prefix when it is applied.
# Those rows no longer carry the original source trail path, so matching only
# Reference sources would append a second historical dataset instead of
# replacing the first one. Keep this deliberately narrower than a generic
# ``rebuild-`` match: live detector assets and unrelated rebuild artefacts must
# remain outside the replacement set.
_HISTORICAL_REBUILD_TRAIL_PREFIX = "rebuild-reference-%"


async def _build_rebuild_apply_plan(
    manifest: dict,
    session: AsyncSession,
    *,
    skip_failed: bool,
    allow_locked: bool,
) -> dict:
    """Validate a manifest and return all replacement work without writing."""
    errors: list[str] = []
    reference = manifest.get("reference") or {}
    if manifest.get("status") not in {"review", "accepted"}:
        errors.append(f"manifest status 不能套用：{manifest.get('status')!r}")
    if reference.get("review_required"):
        errors.append(f"有 {len(reference['review_required'])} 組未裁決 review candidate")

    failed_sources = {
        item.get("source_trail_path") or item.get("trail_path")
        for item in manifest.get("failures", [])
        if item.get("source_trail_path") or item.get("trail_path")
    }
    failed_sources.update(
        item.get("trail_path")
        for item in reference.get("missing_sources", [])
        if item.get("trail_path")
    )
    if failed_sources and not skip_failed:
        errors.append("存在失敗 raw source；必須明確指定 skip_failed")

    source_to_clip: dict[str, dict] = {}
    output_paths: set[str] = set()
    poster_paths: set[str] = set()
    ready_clips: list[dict] = []
    required_assets: set[str] = set()
    for clip in manifest.get("clips", []):
        sources = [source for source in clip.get("reference_source_trails") or [] if source]
        if not sources:
            errors.append(f"{clip.get('clip_id')}: 缺少 Reference source")
            continue
        if any(source in failed_sources for source in sources):
            errors.append(f"{clip.get('clip_id')}: ready output claims a failed source")
            continue
        if clip.get("status") != "ready" or clip.get("source_kind") != "rawframes":
            errors.append(f"{clip.get('clip_id')}: 不是 ready raw-frame output")
            continue
        if not clip.get("camera_id"):
            errors.append(f"{clip.get('clip_id')}: 缺少 camera_id")
        output = clip.get("output_path")
        poster = clip.get("poster_path")
        if not output or output in output_paths:
            errors.append(f"{clip.get('clip_id')}: output_path 缺失或重複")
        else:
            output_paths.add(output)
            required_assets.add(output)
        if not poster or poster in poster_paths:
            errors.append(f"{clip.get('clip_id')}: poster_path 缺失或重複")
        else:
            poster_paths.add(poster)
            required_assets.add(poster)
        for row in clip.get("rows") or []:
            required_assets.update(
                asset for asset in (
                    row.get("image_path"),
                    row.get("annotated_path"),
                    row.get("crop_path"),
                ) if asset
            )
        if clip.get("rawframes_path"):
            required_assets.add(clip["rawframes_path"])
        for source in sources:
            if source in source_to_clip:
                errors.append(f"Reference source 被重複指派：{source}")
            source_to_clip[source] = clip
        ready_clips.append(clip)

    # Blob existence is checked once per unique asset, off the event loop.
    asset_results = await asyncio.gather(*(
        asyncio.to_thread(storage.exists, asset) for asset in sorted(required_assets)
    ))
    missing_assets = [asset for asset, exists in zip(sorted(required_assets), asset_results) if not exists]
    if missing_assets:
        errors.append(f"review asset 不存在：{missing_assets[:8]}")

    ready_sources = set(source_to_clip)
    reference_source_count = reference.get("reference_clips")
    if reference_source_count is not None:
        covered_sources = len(ready_sources | failed_sources)
        if covered_sources != int(reference_source_count):
            errors.append(
                f"Reference source coverage 不完整：{covered_sources}/{reference_source_count}"
            )
    old_rows: list[Detection] = []
    if ready_sources:
        # Include rows produced by earlier Reference runs. Once a review
        # manifest has been published, the API stores the flattened output
        # path, not the source trail used to build it. Without this branch a
        # second rebuild would pass dry-run with old_detection_count=0 and
        # silently duplicate every historical row.
        old_rows = (await session.execute(
            select(Detection).where(or_(
                Detection.trail_path.in_(sorted(ready_sources)),
                Detection.trail_path.like(_HISTORICAL_REBUILD_TRAIL_PREFIX),
            ))
        )).scalars().all()
    locked_old_rows = [det for det in old_rows if det.locked]
    if locked_old_rows and not allow_locked:
        errors.append(
            f"有 {len(locked_old_rows)} 筆舊 detection 已鎖定；"
            "正式替換必須明確指定 allow_locked=true"
        )

    replacement_payloads: list[dict] = []
    for clip in ready_clips:
        try:
            timestamp = _naive_rebuild_timestamp(clip.get("timestamp"))
        except ValueError as exc:
            errors.append(f"{clip.get('clip_id')}: {exc}")
            continue
        visit_id = f"rb-{clip.get('clip_id') or 'unknown'}"
        if len(visit_id) > 32:
            errors.append(f"{clip.get('clip_id')}: clip id 超過 visit_id 長度")
            continue
        visit_peak = _rebuild_visit_peak(clip.get("rows") or [])
        for row in clip.get("rows") or []:
            bbox = row.get("bbox") or []
            if len(bbox) != 4:
                errors.append(f"{clip.get('clip_id')}: row 缺少 bbox")
                continue
            if row.get("track_id") is None or not row.get("boxes"):
                errors.append(f"{clip.get('clip_id')}: row 缺少 track_id 或逐幀 boxes")
                continue
            replacement_payloads.append({
                "camera_id": clip["camera_id"],
                "timestamp": timestamp,
                "confidence": float(row.get("confidence") or 0),
                "bbox_x": float(bbox[0]),
                "bbox_y": float(bbox[1]),
                "bbox_w": max(0.0, float(bbox[2]) - float(bbox[0])),
                "bbox_h": max(0.0, float(bbox[3]) - float(bbox[1])),
                # New manifests carry one clean and one annotated peak frame
                # per animal. Keep the poster fallback for old review manifests
                # so the apply contract remains backward compatible.
                "image_path": row.get("image_path") or clip["poster_path"],
                "annotated_path": row.get("annotated_path") or clip["poster_path"],
                "crop_path": row.get("crop_path") or "",
                "track_id": int(row["track_id"]),
                "trail_path": clip["output_path"],
                "user_visible": None,
                "locked": False,
                "visit_id": visit_id,
                "visit_peak": visit_peak,
                "animal_id": row.get("animal_id"),
                "animal_source": "model",
                "clip_offset_seconds": row.get("clip_offset_seconds"),
                "clip_end_offset_seconds": row.get("clip_end_offset_seconds"),
                "clip_duration_seconds": row.get("clip_duration_seconds") or clip.get("duration_seconds"),
                "clip_segments": row.get("clip_segments"),
                "track_boxes": row.get("boxes"),
                "track_box_fps": row.get("track_box_fps") or clip.get("fps"),
            })

    skipped_sources = sorted(failed_sources) if skip_failed else []
    return {
        "run_id": manifest.get("run_id"),
        "errors": errors,
        "old_rows": old_rows,
        "old_detection_ids": sorted(det.id for det in old_rows),
        "locked_old_detection_ids": sorted(det.id for det in locked_old_rows),
        "replacement_payloads": replacement_payloads,
        "ready_sources": sorted(ready_sources),
        "skipped_sources": skipped_sources,
        "replacement_row_count": len(replacement_payloads),
    }


class RebuildApplyRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=128)
    dry_run: bool = True
    confirm: bool = False
    skip_failed: bool = False
    # Existing lock means a human endorsed the row. Rebuild replacement is a
    # separate, explicit decision and must name that it is consuming locks.
    allow_locked: bool = False


@app.post("/api/admin/rebuild-review/apply", dependencies=[Depends(require_admin)])
async def apply_rebuild_review(
    body: RebuildApplyRequest,
    session: AsyncSession = Depends(get_session),
):
    """Dry-run or atomically replace old source rows with an accepted rebuild."""
    manifest = _load_rebuild_manifest()
    if manifest.get("run_id") != body.run_id:
        raise HTTPException(status_code=409, detail="run_id 與目前 review manifest 不一致")
    existing = (await session.execute(
        select(HistoricalRebuildRun).where(HistoricalRebuildRun.run_id == body.run_id)
    )).scalar_one_or_none()
    if existing and existing.status == "applied":
        return {"status": "accepted", "run_id": body.run_id,
                "created_new_rows": len(existing.replacement_ids or []),
                "message": "這個 run 已套用"}
    if not body.dry_run and not body.confirm:
        raise HTTPException(status_code=400, detail="正式套用必須明確帶 confirm=true")

    plan = await _build_rebuild_apply_plan(
        manifest, session, skip_failed=body.skip_failed,
        allow_locked=body.allow_locked,
    )
    if plan["errors"]:
        if not body.dry_run:
            raise HTTPException(status_code=409, detail={"errors": plan["errors"]})
        return {"status": "blocked", "run_id": body.run_id,
                "dry_run": True, "errors": plan["errors"],
                "old_detection_ids": plan["old_detection_ids"],
                "locked_old_detection_ids": plan["locked_old_detection_ids"],
                "replacement_row_count": plan["replacement_row_count"],
                "skipped_sources": plan["skipped_sources"]}

    summary = {
        "status": "ready" if body.dry_run else "accepted",
        "run_id": body.run_id,
        "dry_run": body.dry_run,
        "old_detection_count": len(plan["old_rows"]),
        "old_detection_ids": plan["old_detection_ids"],
        "locked_old_detection_ids": plan["locked_old_detection_ids"],
        "created_new_rows": plan["replacement_row_count"],
        "skipped_sources": plan["skipped_sources"],
        "rollback_required": True,
    }
    if body.dry_run:
        return summary

    backup = [_backup_detection_row(det) for det in plan["old_rows"]]
    new_rows = [Detection(**payload) for payload in plan["replacement_payloads"]]
    session.add_all(new_rows)
    await session.flush()
    new_ids = [det.id for det in new_rows]
    for det in plan["old_rows"]:
        await session.delete(det)
    if existing:
        existing.status = "applied"
        existing.backup_rows = backup
        existing.replacement_ids = new_ids
        existing.skipped_sources = plan["skipped_sources"]
        existing.applied_at = datetime.now(_TW_TZ).replace(tzinfo=None)
        existing.rolled_back_at = None
    else:
        session.add(HistoricalRebuildRun(
            run_id=body.run_id,
            status="applied",
            backup_rows=backup,
            replacement_ids=new_ids,
            skipped_sources=plan["skipped_sources"],
            applied_at=datetime.now(_TW_TZ).replace(tzinfo=None),
        ))
    await session.commit()
    # DB is authoritative. Updating the review pointer is best-effort so a
    # transient blob outage cannot turn a successful transaction into a 500.
    try:
        manifest["status"] = "accepted"
        manifest["applied_at"] = datetime.now(_TW_TZ).isoformat()
        storage.upload(_rebuild_manifest_name(), json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    except Exception:
        logger.exception("Rebuild %s applied but review manifest status update failed", body.run_id)
    return {**summary, "created_new_rows": len(new_ids)}


class RebuildRollbackRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=128)
    dry_run: bool = True
    confirm: bool = False


@app.post("/api/admin/rebuild-review/rollback", dependencies=[Depends(require_admin)])
async def rollback_rebuild_review(
    body: RebuildRollbackRequest,
    session: AsyncSession = Depends(get_session),
):
    """Dry-run or restore the exact rows archived by a rebuild apply."""
    record = (await session.execute(
        select(HistoricalRebuildRun).where(HistoricalRebuildRun.run_id == body.run_id)
    )).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="找不到可回滾的 rebuild run")
    if record.status != "applied":
        raise HTTPException(status_code=409, detail=f"run 目前狀態為 {record.status}，不可回滾")
    if not body.dry_run and not body.confirm:
        raise HTTPException(status_code=400, detail="正式回滾必須明確帶 confirm=true")

    replacement_ids = [int(value) for value in (record.replacement_ids or [])]
    current = (await session.execute(
        select(Detection).where(Detection.id.in_(replacement_ids))
    )).scalars().all() if replacement_ids else []
    backup_rows = record.backup_rows or []
    summary = {
        "status": "ready" if body.dry_run else "rolled_back",
        "run_id": body.run_id,
        "dry_run": body.dry_run,
        "replacement_count": len(current),
        "restore_count": len(backup_rows),
    }
    if body.dry_run:
        return summary
    for det in current:
        await session.delete(det)
    await session.flush()
    existing_ids = set((await session.execute(
        select(Detection.id).where(Detection.id.in_([
            int(row["id"]) for row in backup_rows if row.get("id") is not None
        ]))
    )).scalars().all()) if backup_rows else set()
    if existing_ids:
        raise HTTPException(status_code=409, detail="rollback row id 已被其他 detection 佔用")
    for payload in backup_rows:
        restored = dict(payload)
        if isinstance(restored.get("timestamp"), str):
            restored["timestamp"] = _naive_rebuild_timestamp(restored["timestamp"])
        session.add(Detection(**restored))
    record.status = "rolled_back"
    record.rolled_back_at = datetime.now(_TW_TZ).replace(tzinfo=None)
    await session.commit()
    try:
        manifest = _load_rebuild_manifest()
        manifest["status"] = "rolled_back"
        manifest["rolled_back_at"] = datetime.now(_TW_TZ).isoformat()
        storage.upload(_rebuild_manifest_name(), json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    except Exception:
        logger.exception("Rebuild %s rolled back but review manifest status update failed", body.run_id)
    return summary


@app.get("/api/detections/{detection_id}/frames")
async def extract_detection_frames(
    detection_id: int,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Download raw (un-annotated) frames for a detection as a ZIP of JPGs.

    Prefers the pre-built raw-frames ZIP saved alongside the trail GIF.
    Falls back to extracting from the GIF (annotated) for older detections.
    """
    import io
    import zipfile
    from PIL import Image

    if not _feature_enabled(user, "downloads"):
        raise HTTPException(status_code=403, detail="Forbidden")

    det = (
        await session.execute(select(Detection).where(Detection.id == detection_id))
    ).scalar_one_or_none()
    if not det:
        raise HTTPException(status_code=404, detail="Detection not found")
    # 404 (not 403) for confidence-hidden records so IDs can't be probed.
    if user.role != "admin" and not _visible_to_users(det):
        raise HTTPException(status_code=404, detail="Detection not found")
    if not _camera_allowed(user, det.camera_id):
        raise HTTPException(status_code=403, detail="Forbidden")

    if not det.trail_path:
        raise HTTPException(status_code=404, detail="No trail data for this detection")

    # Try the pre-built raw-frames ZIP first (clean, no annotations). New MP4
    # and historical GIF clips share this basename convention.
    raw_zip_name = _raw_frames_name(det.trail_path)
    raw_stream = storage.download_stream(raw_zip_name)
    if raw_stream is not None:
        zip_filename = f"{det.camera_id}_t{det.track_id}_rawframes.zip"
        logger.info("Serving pre-built raw frames ZIP: %s", raw_zip_name)
        return StreamingResponse(
            raw_stream,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(zip_filename)}"},
        )

    # Only historic GIFs can be extracted locally. MP4 detections always write
    # their raw ZIP alongside the clip, so a missing ZIP is an artefact loss,
    # not a reason to decode and re-encode a video in the API process.
    if not det.trail_path.endswith(".gif"):
        raise HTTPException(status_code=404, detail="Raw frames ZIP not found")

    # Fallback for older detections: extract from GIF (these will have annotations)
    gif_bytes = storage.download(det.trail_path)
    if gif_bytes is None:
        raise HTTPException(status_code=404, detail="Trail GIF not found")

    try:
        gif = Image.open(io.BytesIO(gif_bytes))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read trail GIF")
    buf = io.BytesIO()
    frame_count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        while True:
            try:
                gif.seek(frame_count)
            except EOFError:
                break
            rgb = gif.convert("RGB")
            frame_buf = io.BytesIO()
            rgb.save(frame_buf, format="JPEG", quality=95)
            zf.writestr(f"{det.camera_id}_t{det.track_id}_frame{frame_count:04d}.jpg", frame_buf.getvalue())
            frame_count += 1

    if frame_count == 0:
        raise HTTPException(status_code=404, detail="No frames found in trail GIF")

    zip_bytes = buf.getvalue()
    zip_filename = f"{det.camera_id}_t{det.track_id}_frames.zip"
    logger.info("Extracted %d frames from GIF (annotated fallback): %s", frame_count, det.trail_path)
    return _zip_response(zip_bytes, zip_filename)


# ── Admin ────────────────────────────────────────────────────────────────────


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


@app.get("/api/admin/storage-stats", response_model=StorageStatsOut, dependencies=[Depends(require_admin)])
async def storage_stats(session: AsyncSession = Depends(get_session)):
    files = storage.list_files()
    total_bytes = sum(size for _, size in files)
    total_detections = (
        await session.execute(select(func.count(Detection.id)))
    ).scalar() or 0
    return StorageStatsOut(
        total_files=len(files),
        total_size_bytes=total_bytes,
        total_size_human=_human_size(total_bytes),
        total_detections=total_detections,
    )


@app.get("/api/admin/inference-stats", response_model=InferenceStatsOut, dependencies=[Depends(require_admin)])
async def inference_stats():
    """Live YOLO inference throughput per camera (rolling window)."""
    stats = detector_manager.inference_stats()
    # Cloud (api role) runs no detectors — surface per-camera stats from all
    # online nodes' heartbeats so the dashboard panel stays live.
    if not stats["cameras"]:
        stats["cameras"] = [
            cam
            for node in _fresh_nodes() if node["online"]
            for cam in (node["heartbeat"] or {}).get("cameras", [])
        ]
    # Stage 2 counters travel with the inference stats: a queue that is
    # silently dropping every frame looks identical to one that is off.
    from services.detector import second_stage as _stage2
    return InferenceStatsOut(
        second_stage=_stage2.stats(),**stats)


def _build_images_zip(detections) -> tuple[bytes, int]:
    """Zip the unique original snapshots (image_path only) for a set of detections."""
    import io
    import zipfile

    filenames = [
        fname for fname in dict.fromkeys(det.image_path for det in detections)
        if fname and storage.exists(fname)
    ]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for fname in filenames:
            data = storage.download(fname)
            if data:
                zf.writestr(fname, data)
    return buf.getvalue(), len(filenames)


def _zip_response(zip_bytes: bytes, filename: str) -> Response:
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


@app.get("/api/admin/download-images", dependencies=[Depends(require_admin)])
async def download_images(
    camera_id: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    query = select(Detection)
    if camera_id:
        query = query.where(Detection.camera_id == camera_id)
    if date_from:
        query = query.where(Detection.timestamp >= datetime.fromisoformat(date_from))
    if date_to:
        # Include the entire end date
        end = datetime.fromisoformat(date_to) + timedelta(days=1)
        query = query.where(Detection.timestamp < end)
    query = query.order_by(Detection.timestamp.desc())

    detections = (await session.execute(query)).scalars().all()
    zip_bytes, count = _build_images_zip(detections)
    if count == 0:
        raise HTTPException(status_code=404, detail="No images found for the given filters")

    zip_filename = f"primate-watch-images-{datetime.now(_TW_TZ):%Y%m%d}.zip"
    logger.info("ZIP created: %s (%d files, %d bytes)", zip_filename, count, len(zip_bytes))
    return _zip_response(zip_bytes, zip_filename)


@app.delete("/api/admin/clear-snapshots", dependencies=[Depends(require_admin)])
async def clear_snapshots(session: AsyncSession = Depends(get_session)):
    """Delete ALL snapshot files and their detection records — except locked ones.

    A lock that survived 刪除所選 but not this button would be worth nothing:
    this is one click and it used to take the whole table. Locked rows stay, and
    so do the blobs they reference, which is also what keeps the orphan sweep
    off those files.
    """
    # Collect referenced files from DB
    detections = (await session.execute(select(Detection))).scalars().all()
    kept = [det for det in detections if det.locked]
    doomed = [det for det in detections if not det.locked]
    for det in doomed:
        await session.delete(det)
    await session.commit()

    if kept:
        # delete_all() is off the table — walk the container instead and spare
        # every file a surviving locked record still points at.
        keep_files: set[str] = set()
        for det in kept:
            keep_files |= _detection_files(det)

        def _clear_blocking() -> int:
            # Sync Azure calls must stay OFF the event loop: listing 4k+ blobs
            # plus a delete per file blocked it for minutes on 2026-08-03,
            # drained the DB pool and 500'd every in-flight request until a
            # restart (see _orphan_sweep_loop). delete_all() has the same shape,
            # so it moves too.
            gone = 0
            for fname, _size in storage.list_files():
                if fname not in keep_files and storage.delete(fname):
                    gone += 1
            return gone

        removed = await asyncio.to_thread(_clear_blocking)
    else:
        removed = await asyncio.to_thread(storage.delete_all)

    logger.info(
        "Cleared %d snapshot files and %d detection records (%d locked kept)",
        removed, len(doomed), len(kept),
    )
    return {
        "deleted_files": removed,
        "deleted_detections": len(doomed),
        "kept_locked": len(kept),
    }


class CreateDetectionRequest(BaseModel):
    camera_id: str
    timestamp: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    image_path: str = ""
    annotated_path: str = ""
    crop_path: str = ""
    track_id: int = -1
    trail_path: str = ""
    # Set by the edge node, where the visit is actually observed, so the cloud
    # never has to re-derive which rows belonged together.
    visit_id: Optional[str] = None
    visit_peak: Optional[int] = None
    clip_offset_seconds: Optional[float] = None
    clip_end_offset_seconds: Optional[float] = None
    clip_duration_seconds: Optional[float] = None
    clip_segments: Optional[list] = None
    track_boxes: Optional[list] = None
    track_box_fps: Optional[float] = None


@app.post("/api/admin/create-detection", dependencies=[Depends(require_admin)])
async def create_detection(body: CreateDetectionRequest, session: AsyncSession = Depends(get_session)):
    """Manually insert a detection record (admin only)."""
    det = Detection(
        **body.model_dump(exclude={"timestamp"}),
        timestamp=datetime.fromisoformat(body.timestamp),
    )
    session.add(det)
    await session.commit()
    await session.refresh(det)
    logger.info("Admin created detection id=%d camera=%s", det.id, det.camera_id)
    return {"id": det.id}


@app.get("/api/admin/nodes", dependencies=[Depends(require_admin)])
async def list_nodes():
    """All known inference nodes with liveness + utilization (節點 tab)."""
    return {"nodes": _fresh_nodes()}


@app.get("/api/admin/nodes/history", dependencies=[Depends(require_admin)])
async def nodes_history(minutes: int = 180):
    """Per-node utilisation time series for the 節點 tab charts.

    Backed by the in-memory ring buffers fed by _record_heartbeat — history
    RESETS on deploy/restart (accepted for now; it refills at one point per
    30 s beat, 24 h max). Stride-downsampled to <=360 points per node so a
    24 h query stays light; the newest point is always kept.
    """
    minutes = max(5, min(1440, minutes))
    cutoff = time.time() - minutes * 60
    nodes = []
    for node_id in sorted(_node_history):
        pts = [p for p in _node_history[node_id] if p["ts"] >= cutoff]
        if len(pts) > 360:
            stride = -(-len(pts) // 360)  # ceil division
            pts = pts[::-1][::stride][::-1]  # sample from the newest backwards
        nodes.append({
            "node_id": node_id,
            "points": [
                {"t": p["t"], "cpu": p["cpu"], "gpu": p["gpu"], "rss_gb": p["rss_gb"]}
                for p in pts
            ],
        })
    return {"nodes": nodes}


@app.delete("/api/admin/nodes/{hostname}", dependencies=[Depends(require_admin)])
async def delete_node(hostname: str):
    """Forget a node (e.g. a renamed or retired machine left showing 失聯).

    A live node re-registers on its next heartbeat, so this only removes
    ghosts — it cannot take a running node out of service.
    """
    if _edge_nodes.pop(_node_key(hostname), None) is None:
        raise HTTPException(status_code=404, detail="Node not found")
    _node_history.pop(_node_key(hostname), None)
    logger.info("Admin removed edge node %s", hostname)
    return {"removed": _node_key(hostname)}


@app.post("/api/admin/upload-snapshot", dependencies=[Depends(require_admin)])
async def upload_snapshot(file: UploadFile = File(...)):
    """Upload a snapshot file to the snapshots directory."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    # Security: strip path components
    safe_name = Path(file.filename).name
    if not safe_name or "/" in safe_name or "\\" in safe_name or ".." in safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    content = await file.read()
    storage.upload(safe_name, content)
    logger.info("Uploaded snapshot: %s (%d bytes)", safe_name, len(content))
    return {"filename": safe_name, "size": len(content)}


class DeleteSelectRequest(BaseModel):
    ids: list[int]


@app.post("/api/admin/download-selected", dependencies=[Depends(require_admin)])
async def download_selected(body: DeleteSelectRequest, session: AsyncSession = Depends(get_session)):
    """Zip the original snapshots for the selected detections (POST — id lists
    are arbitrary/large, so this can't ride the ?token= GET link like the
    filter download does)."""
    if not body.ids:
        raise HTTPException(status_code=400, detail="No IDs provided")
    detections = (
        await session.execute(select(Detection).where(Detection.id.in_(body.ids)))
    ).scalars().all()
    zip_bytes, count = _build_images_zip(detections)
    if count == 0:
        raise HTTPException(status_code=404, detail="No images found for the selected detections")
    zip_filename = f"primate-watch-selected-{datetime.now(_TW_TZ):%Y%m%d}.zip"
    logger.info("Selected ZIP created: %s (%d files)", zip_filename, count)
    return _zip_response(zip_bytes, zip_filename)


def _raw_frames_name(trail_path: str) -> str:
    """Companion raw-frame archive for either a new MP4 or legacy GIF clip."""
    stem, marker, _ = trail_path.rpartition("_trail.")
    return f"{stem}_rawframes.zip" if marker else ""


def _detection_files(det: Detection) -> set[str]:
    """Every storage file belonging to a detection, including its raw-frame ZIP."""
    files = {
        fname
        for fname in (det.image_path, det.annotated_path, det.crop_path, det.trail_path)
        if fname
    }
    if det.trail_path:
        files.add(_raw_frames_name(det.trail_path))
    return files


class VisibilityRequest(BaseModel):
    """null = follow the confidence rule; true = force show; false = force hide."""

    visible: Optional[bool] = None


def _bulk_date(value: str, field: str) -> datetime:
    """Parse a YYYY-MM-DD bulk filter, rejecting junk with a 422 not a 500.

    These endpoints are documented as a hand-written JSON body, so a typo'd
    date is a normal input, not an internal error — it has to come back naming
    the field.
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"{field} must be an ISO date (YYYY-MM-DD), got {value!r}",
        )


def _bulk_filter_criteria(
    detection_ids: Optional[list[int]] = None,
    camera_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    max_confidence: Optional[float] = None,
) -> list:
    """The row selection both bulk admin actions share, as WHERE criteria.

    `date_to` is inclusive of that whole day (`timestamp < date_to + 1 day`), so
    a "before the 18th" cutoff is written `2026-08-17`. One implementation on
    purpose: visibility and lock must never drift apart on what a date means.

    Returned as criteria rather than a Select so the lock path can spend them on
    an UPDATE instead of hydrating every matching row.
    """
    crits = []
    if detection_ids:
        crits.append(Detection.id.in_(detection_ids))
    if camera_id:
        crits.append(Detection.camera_id == camera_id)
    if date_from:
        crits.append(Detection.timestamp >= _bulk_date(date_from, "date_from"))
    if date_to:
        crits.append(
            Detection.timestamp < _bulk_date(date_to, "date_to") + timedelta(days=1)
        )
    if max_confidence is not None:
        crits.append(Detection.confidence <= max_confidence)
    return crits


class VisibilityBulkRequest(BaseModel):
    """Apply one visibility decision to a whole backlog.

    Same three states as the per-record override. Either name the ids, or leave
    them out and let the filters select — `only_hidden` picks exactly the
    records the confidence gate is currently withholding, which is the case this
    exists for: a busy day arrives, most of it sits just under the threshold,
    and someone has to decide about all of it at once rather than clicking
    through hundreds of rows.
    """

    visible: Optional[bool] = None
    detection_ids: Optional[list[int]] = None
    camera_id: Optional[str] = None
    date_from: Optional[str] = None          # YYYY-MM-DD, TW-local, inclusive
    date_to: Optional[str] = None            # YYYY-MM-DD, TW-local, inclusive
    only_hidden: bool = True                 # skip records already visible
    max_confidence: Optional[float] = None   # e.g. push only the marginal ones


@app.post("/api/admin/detections/visibility-bulk",
          dependencies=[Depends(require_admin)])
async def set_detection_visibility_bulk(
    body: VisibilityBulkRequest,
    session: AsyncSession = Depends(get_session),
):
    """Set the visibility override on many records in one request.

    The confidence gate is a rule about FUTURE records and should stay armed —
    dropping it to 0 was tried on 2026-08-14 and rolled back, because it also
    stops filtering everything that arrives later. What was actually needed was
    this: clear the backlog in one call, leave the gate up.
    """
    query = select(Detection).where(*_bulk_filter_criteria(
        body.detection_ids, body.camera_id, body.date_from, body.date_to,
        body.max_confidence,
    ))
    rows = (await session.execute(query)).scalars().all()
    # `only_hidden` is evaluated in Python against the same helper the API
    # reports with, so "hidden" here means exactly what the dashboard shows.
    if body.only_hidden:
        rows = [d for d in rows if not _visible_to_users(d)]

    for det in rows:
        det.user_visible = body.visible
    await session.commit()
    logger.info(
        "Bulk visibility override → %s on %d detections (camera=%s %s..%s)",
        body.visible, len(rows), body.camera_id, body.date_from, body.date_to,
    )
    return {"updated": len(rows), "visible": body.visible,
            "detection_ids": [d.id for d in rows]}


@app.patch("/api/admin/detections/{detection_id}/visibility",
           dependencies=[Depends(require_admin)])
async def set_detection_visibility(
    detection_id: int,
    body: VisibilityRequest,
    session: AsyncSession = Depends(get_session),
):
    """Override (or reset) whether non-admin users can see one detection.

    The confidence gate is a blunt rule; this is the human in the loop for the
    two cases it gets wrong — a real sighting below the threshold, and a
    false positive above it.
    """
    det = (
        await session.execute(select(Detection).where(Detection.id == detection_id))
    ).scalar_one_or_none()
    if det is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    det.user_visible = body.visible
    await session.commit()
    await session.refresh(det)
    logger.info("Detection %s visibility override → %s", detection_id, body.visible)
    return {
        "id": det.id,
        "user_visible": det.user_visible,
        "visible_to_users": _visible_to_users(det),
    }


class LockRequest(BaseModel):
    """true = refuse every delete path; false = release."""

    locked: bool


class LockBulkRequest(BaseModel):
    """Lock (or unlock) a whole backlog in one request.

    Mirrors VisibilityBulkRequest field for field, and for the same reason: the
    records most worth protecting are the reviewed ones, and 影像管理 only ever
    loads the 200 most recent, so anything id-based cannot reach them. Either
    name the ids, or leave them out and let the filters select.
    """

    locked: bool
    detection_ids: Optional[list[int]] = None
    camera_id: Optional[str] = None
    date_from: Optional[str] = None          # YYYY-MM-DD, TW-local, inclusive
    date_to: Optional[str] = None            # YYYY-MM-DD, TW-local, inclusive
    max_confidence: Optional[float] = None   # e.g. lock only the marginal ones


@app.post("/api/admin/detections/lock-bulk",
          dependencies=[Depends(require_admin)])
async def set_detection_lock_bulk(
    body: LockBulkRequest,
    session: AsyncSession = Depends(get_session),
):
    """Set the delete guard on many records in one request.

    This is the shape the backfill needs — everything reviewed before the
    cutoff, including the records 影像管理 cannot currently display:

        {"date_to": "2026-08-17", "locked": true}
    """
    crits = _bulk_filter_criteria(
        body.detection_ids, body.camera_id, body.date_from, body.date_to,
        body.max_confidence,
    )
    # Locking everything is harmless; UNLOCKING everything disarms every guard in
    # the database in one request, and an empty `detection_ids` or a blank date
    # box both arrive here as "no filter at all". Refuse that shape outright —
    # the caller has to say which records lose their protection.
    if not crits and not body.locked:
        raise HTTPException(
            status_code=400,
            detail="Unlocking requires at least one filter "
                   "(detection_ids, camera_id, date_from, date_to or max_confidence)",
        )

    # A set-based UPDATE ... RETURNING, not a read-then-mutate loop: the
    # documented backfill matches essentially the whole table, and hydrating that
    # many ORM objects inside one request is how this container runs out of
    # memory. RETURNING still gives the exact ids that changed.
    result = await session.execute(
        update(Detection).where(*crits).values(locked=body.locked)
        .returning(Detection.id)
    )
    changed = sorted(row[0] for row in result)
    await session.commit()
    logger.info(
        "Bulk lock → %s on %d detections (camera=%s %s..%s)",
        body.locked, len(changed), body.camera_id, body.date_from, body.date_to,
    )
    return {"updated": len(changed), "locked": body.locked,
            "detection_ids": changed}


@app.patch("/api/admin/detections/{detection_id}/lock",
           dependencies=[Depends(require_admin)])
async def set_detection_lock(
    detection_id: int,
    body: LockRequest,
    session: AsyncSession = Depends(get_session),
):
    """Protect one detection from deletion, or release it.

    Deliberately a human decision and never derived from confidence: the whole
    value of the flag is that somebody looked at the record. Orthogonal to the
    visibility override — locking does not change who can see a row.
    """
    det = (
        await session.execute(select(Detection).where(Detection.id == detection_id))
    ).scalar_one_or_none()
    if det is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    det.locked = body.locked
    await session.commit()
    await session.refresh(det)
    logger.info("Detection %s lock → %s", detection_id, body.locked)
    return {"id": det.id, "locked": det.locked}


class DetectionMediaItem(BaseModel):
    detection_id: int = Field(gt=0)
    image_path: str = Field(min_length=1, max_length=256)
    annotated_path: str = Field(min_length=1, max_length=256)
    crop_path: str = Field(default="", max_length=256)


class DetectionMediaBulkRequest(BaseModel):
    """Replace preview asset paths without changing detection semantics."""

    items: list[DetectionMediaItem] = Field(min_length=1, max_length=1000)


@app.post("/api/admin/detections/media-bulk",
          dependencies=[Depends(require_admin)])
async def update_detection_media_bulk(
    body: DetectionMediaBulkRequest,
    session: AsyncSession = Depends(get_session),
):
    """Point existing rows at uploaded raw and annotated preview assets.

    This repair is deliberately independent from the delete lock. A locked row
    can still have its media corrected; the lock only protects deletion.
    """
    ids = [item.detection_id for item in body.items]
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=400, detail="detection_id 不可重複")

    paths = {
        path
        for item in body.items
        for path in (item.image_path, item.annotated_path, item.crop_path)
        if path
    }
    invalid = sorted(
        path for path in paths
        if "\\" in path or ".." in path or path.startswith("/")
        or ("/" in path and not path.startswith("rebuild-review/"))
    )
    if invalid:
        raise HTTPException(status_code=400, detail={"invalid_paths": invalid[:8]})

    asset_results = await asyncio.gather(*(
        asyncio.to_thread(storage.exists, path) for path in sorted(paths)
    ))
    missing = [path for path, exists in zip(sorted(paths), asset_results) if not exists]
    if missing:
        raise HTTPException(status_code=409, detail={"missing_assets": missing[:8]})

    detections = (await session.execute(
        select(Detection).where(Detection.id.in_(ids))
    )).scalars().all()
    by_id = {det.id: det for det in detections}
    missing_ids = sorted(set(ids) - set(by_id))
    if missing_ids:
        raise HTTPException(status_code=404, detail={"missing_detection_ids": missing_ids})

    for item in body.items:
        det = by_id[item.detection_id]
        det.image_path = item.image_path
        det.annotated_path = item.annotated_path
        det.crop_path = item.crop_path
    await session.commit()
    logger.info("Bulk detection media update on %d rows", len(ids))
    return {"updated": len(ids), "detection_ids": sorted(ids)}


@app.post("/api/admin/delete-selected", dependencies=[Depends(require_admin)])
async def delete_selected(body: DeleteSelectRequest, session: AsyncSession = Depends(get_session)):
    """Delete selected detections and their snapshot files, minus locked ones.

    A locked row is REFUSED, not silently skipped: the rest of a mixed
    selection still goes, so the tab stays useful, and `refused_locked` names
    what survived — otherwise the only signal would be a count that does not
    match the selection, which reads as a bug rather than as the lock working.
    """
    if not body.ids:
        raise HTTPException(status_code=400, detail="No IDs provided")

    detections = (
        await session.execute(select(Detection).where(Detection.id.in_(body.ids)))
    ).scalars().all()

    refused_locked = sorted(det.id for det in detections if det.locked)
    doomed = [det for det in detections if not det.locked]
    if not doomed:
        logger.info("Delete refused: all %d selected detections are locked", len(refused_locked))
        return {
            "deleted_detections": 0,
            "deleted_files": 0,
            "refused_locked": refused_locked,
        }

    # Collect unique file paths to delete
    files_to_delete: set[str] = set()
    for det in doomed:
        files_to_delete |= _detection_files(det)

    # Check if any other detections still reference these files (shared trail
    # GIFs). The refused locked rows are part of "other" now, so a clip shared
    # with a locked record survives on their account too.
    doomed_ids = [det.id for det in doomed]
    other_dets = (
        await session.execute(select(Detection).where(Detection.id.not_in(doomed_ids)))
    ).scalars().all()
    still_used: set[str] = set()
    for det in other_dets:
        still_used |= _detection_files(det)

    # Only delete files not used by other detections
    safe_to_delete = files_to_delete - still_used
    removed = 0
    for fname in safe_to_delete:
        if storage.delete(fname):
            removed += 1

    for det in doomed:
        await session.delete(det)
    await session.commit()

    logger.info(
        "Deleted %d detections, %d files (%d refused: locked)",
        len(doomed), removed, len(refused_locked),
    )
    return {
        "deleted_detections": len(doomed),
        "deleted_files": removed,
        "refused_locked": refused_locked,
    }


class MergeDuplicatesRequest(BaseModel):
    """Backfill filter. `dry_run` defaults TRUE — this deletes rows."""

    dry_run: bool = True
    detection_ids: Optional[list[int]] = None
    camera_id: Optional[str] = None
    date_from: Optional[str] = None          # YYYY-MM-DD, TW-local, inclusive
    date_to: Optional[str] = None
    max_confidence: Optional[float] = None
    # Defaults to the NARROWER class. The 5-20 s class merges two animals that
    # used one perch in sequence — documented, accepted for the live merge, and
    # not something a caller should get by omitting a field from a request that
    # deletes rows with no undo.
    same_instant_only: bool = True


@app.post("/api/admin/detections/merge-duplicates", dependencies=[Depends(require_admin)])
async def merge_duplicate_detections(
    body: MergeDuplicatesRequest,
    session: AsyncSession = Depends(get_session),
):
    """Fold rows that are probably one animal recorded twice into a single row.

    The retroactive half of the visit-continuation design. The producer now collapses same-instant
    duplicates before they are ever written (`_merge_duplicate_tracks`), but
    everything recorded before that shipped is still in the table.

    **This is an inference, not a proof** — see the note above
    `_duplicate_row_groups`: a row carries its ARRIVAL time and its PEAK-frame
    box, so nothing here can establish two rows were ever simultaneous. Read the
    dry run before applying it.

    Unlike /api/admin/delete-selected this does NOT refuse locked rows: a lock
    is an endorsement of the animal and its box, not a do-not-touch flag, so an
    endorsed row is the preferred survivor and its lock carries forward. Every
    endorsed row that is nonetheless deleted is named per group in
    `dropped_locked` and logged by id.

    Returns the same manifest shape whether or not it applied, so a dry run can
    be saved, read, and then replayed with `dry_run: false`.
    """
    crits = _bulk_filter_criteria(
        body.detection_ids, body.camera_id, body.date_from, body.date_to,
        body.max_confidence,
    )
    # An empty `detection_ids` and a blank date box both arrive as "no filter at
    # all", and this is the one bulk endpoint that destroys rows. The sibling
    # lock endpoint refuses the same shape for merely flipping a boolean.
    if not crits and not body.dry_run:
        raise HTTPException(
            status_code=400,
            detail="合併需要至少一個篩選條件"
                   "（detection_ids、camera_id、date_from、date_to 或 max_confidence）；"
                   "全表合併請先用 dry_run 確認",
        )

    query = select(Detection).where(*crits)
    # Bounded like the events walk it mirrors: grouping needs the rows, not a
    # COUNT, and an unbounded scan is how this container runs out of memory.
    query = query.order_by(Detection.timestamp.desc()).limit(EVENT_SCAN_CAP + 1)
    rows = (await session.execute(query)).scalars().all()
    truncated = len(rows) > EVENT_SCAN_CAP
    rows = rows[:EVENT_SCAN_CAP]

    groups = _duplicate_row_groups(rows, body.same_instant_only)
    doomed_ids = {i for g in groups for i in g["drop"]}
    manifest = {
        "dry_run": body.dry_run,
        "same_instant_only": body.same_instant_only,
        "scanned": len(rows),
        "truncated": truncated,
        "groups": len(groups),
        "rows_removed": len(doomed_ids),
        "retimed": sum(1 for g in groups if g["retimed"]),
        "locked_removed": sorted({i for g in groups for i in g["dropped_locked"]}),
        "detail": [{**g, "timestamp": g["timestamp"].isoformat()} for g in groups],
    }
    if not groups:
        return {**manifest, "files_removed": 0, "files_freeable": []}

    # Which blobs actually become unreferenced. A clip is shared by every row in
    # its group, so this is computed against every row that SURVIVES — including
    # rows outside the filter, which is why the reference set is queried across
    # the whole table. Columns only: hydrating the table as ORM objects twice is
    # the pattern that OOMs this container.
    by_id = {d.id: d for d in rows}
    doomed = [by_id[i] for i in sorted(doomed_ids)]
    files_to_delete: set[str] = set()
    for det in doomed:
        files_to_delete |= _detection_files(det)
    referenced: set[str] = set()
    paths = await session.execute(
        select(Detection.id, Detection.image_path, Detection.annotated_path,
               Detection.crop_path, Detection.trail_path)
    )
    for det_id, image_path, annotated_path, crop_path, trail_path in paths:
        if det_id in doomed_ids:
            continue
        referenced |= {p for p in (image_path, annotated_path, crop_path, trail_path) if p}
        if trail_path:
            referenced.add(_raw_frames_name(trail_path))
    freeable = sorted(files_to_delete - referenced)

    if body.dry_run:
        logger.info(
            "Merge dry run: %d groups, %d rows would go (%d endorsed), "
            "%d files would be freed",
            len(groups), len(doomed_ids), len(manifest["locked_removed"]), len(freeable),
        )
        return {**manifest, "files_removed": 0, "files_freeable": freeable}

    for group in groups:
        survivor = by_id[group["keep"]]
        survivor.timestamp = group["timestamp"]
        if group["locked"]:
            survivor.locked = True
    for det in doomed:
        await session.delete(det)
    # Commit BEFORE touching storage. The other order leaves surviving rows
    # pointing at blobs that are already gone if the commit fails, and there is
    # no undo for either half.
    await session.commit()

    for det_id in manifest["locked_removed"]:
        logger.info("Merge deleted ENDORSED detection #%d into its group survivor", det_id)

    # Blocking network calls, one per blob, off the event loop: a whole-table
    # backfill frees thousands, and stalling the loop is what blacked out the
    # dashboard before.
    def _purge() -> int:
        return sum(1 for fname in freeable if storage.delete(fname))

    removed = await asyncio.to_thread(_purge)

    logger.info(
        "Merged %d duplicate groups: %d rows removed (%d endorsed), %d files removed",
        len(groups), len(doomed_ids), len(manifest["locked_removed"]), removed,
    )
    return {**manifest, "files_removed": removed, "files_freeable": freeable}


def _retarget_plan(
    rows: list[Detection], blob_exists
) -> tuple[list[dict], dict[str, int]]:
    """Which rows can be switched to their rebuilt MP4, and why the rest cannot.

    `blob_exists(name) -> bool` is passed in rather than reached for, so the rule
    can be exercised without storage. Pure: decides, writes nothing.
    """
    planned: list[dict] = []
    skipped = {"already_mp4": 0, "no_clip": 0, "rebuild_missing": 0}
    # One existence check per distinct clip, not per row: a clip carrying ten
    # rows is one blob, and a check per row would be ten round trips to learn
    # the same fact.
    checked: dict[str, bool] = {}
    for det in rows:
        stem, marker, ext = (det.trail_path or "").rpartition("_trail.")
        if not marker:
            skipped["no_clip"] += 1
            continue
        if ext.lower() == "mp4":
            # Already an MP4 the detector wrote. A rebuild exists for these too,
            # but switching is only worth doing where the player is broken, and
            # the player is broken exactly on the GIFs.
            skipped["already_mp4"] += 1
            continue
        target = f"{stem}_trail.mp4"
        if target not in checked:
            checked[target] = bool(blob_exists(target))
        if not checked[target]:
            skipped["rebuild_missing"] += 1
            continue
        planned.append({"id": det.id, "camera_id": det.camera_id,
                        "timestamp": det.timestamp.isoformat(),
                        "from": det.trail_path, "to": target})
    return planned, skipped


class RetargetClipsRequest(BaseModel):
    """Point rows at the rebuilt MP4 beside their original clip. `dry_run` TRUE."""

    dry_run: bool = True
    detection_ids: Optional[list[int]] = None
    camera_id: Optional[str] = None
    date_from: Optional[str] = None          # YYYY-MM-DD, TW-local, inclusive
    date_to: Optional[str] = None
    max_confidence: Optional[float] = None


@app.post("/api/admin/detections/retarget-clips", dependencies=[Depends(require_admin)])
async def retarget_detection_clips(
    body: RetargetClipsRequest,
    session: AsyncSession = Depends(get_session),
):
    """Switch `trail_path` from a legacy clip to the rebuilt MP4 beside it.

    406 of the clips on record are GIFs, which the review pane renders through
    an `<img>` — so they have no scrubber, no pause, no timeline. They were
    rebuilt from their archived raw frames as `{stem}_trail.mp4` (same stem, so
    `_raw_frames_name` still resolves), and this points the rows at them.

    Two properties make it safe to run before anyone has looked at the results:

    * **The rebuilt blob must already exist.** A row is never pointed at a name
      that is not in storage — a missing file would render as a dead player with
      no way to tell which rows lost their clip.
    * **Nothing is deleted.** The original GIF stays exactly where it is, so
      reverting is switching `trail_path` back, not restoring from a backup.
      Reclaiming those bytes is a separate, later decision.
    """
    crits = _bulk_filter_criteria(
        body.detection_ids, body.camera_id, body.date_from, body.date_to,
        body.max_confidence,
    )
    query = select(Detection).where(*crits)
    rows = (await session.execute(
        query.order_by(Detection.timestamp.desc()).limit(EVENT_SCAN_CAP + 1)
    )).scalars().all()
    truncated = len(rows) > EVENT_SCAN_CAP
    rows = rows[:EVENT_SCAN_CAP]

    planned, skipped = await asyncio.to_thread(
        _retarget_plan, rows, storage.exists
    )

    manifest = {
        "dry_run": body.dry_run,
        "scanned": len(rows),
        "truncated": truncated,
        "retargeted": len(planned),
        "distinct_clips": len({p["to"] for p in planned}),
        "skipped": skipped,
        "detail": planned,
    }
    if body.dry_run or not planned:
        logger.info("Retarget dry run: %d rows over %d clips would switch",
                    len(planned), manifest["distinct_clips"])
        return manifest

    by_id = {d.id: d for d in rows}
    for p in planned:
        by_id[p["id"]].trail_path = p["to"]
    await session.commit()
    logger.info("Retargeted %d rows across %d rebuilt clips; originals kept",
                len(planned), manifest["distinct_clips"])
    return manifest


# Reconstructed visits are prefixed so a row's id says where it came from: a
# bare timestamp id was written by the producer that WATCHED the visit happen,
# anything starting with this was worked out afterwards from the frames.
BACKFILL_VISIT_PREFIX = "bf"


class VisitAssignment(BaseModel):
    """One visit: the id, every row proven to belong to it, and its peak."""

    visit_id: str
    detection_ids: list[int]
    # Most animals visible at once, counted off the raw frames. Optional because
    # a clip whose archive is gone can still be grouped by the clips around it,
    # and a missing count is honest where a zero would read as "no animals".
    peak: Optional[int] = None


class BackfillVisitsRequest(BaseModel):
    """Assign visit ids to rows written before the producer emitted them.

    `dry_run` TRUE by default. `overwrite` is separate and also off: a row that
    already carries an id got it from the producer, which OBSERVED the visit —
    that must never be replaced by a reconstruction.
    """

    dry_run: bool = True
    # Re-runs the reconstruction over rows a PREVIOUS reconstruction stamped.
    # Never touches a producer id — see the endpoint.
    overwrite: bool = False
    visits: list[VisitAssignment]


@app.post("/api/admin/detections/backfill-visits", dependencies=[Depends(require_admin)])
async def backfill_detection_visits(
    body: BackfillVisitsRequest,
    session: AsyncSession = Depends(get_session),
):
    """Stamp historical rows with the visit they belong to.

    The grouping is computed OUTSIDE this endpoint, on the box that holds the
    raw-frame archives, because the evidence is in the frames: the archive
    stores the ring buffer's JPEG bytes verbatim, so two clips containing the
    same instant contain the same bytes. Rows on clips that share bytes are one
    visit — a proof, not the 20 s-gap regrouping `/detections/events` does at
    read time. 484 clip pairs share frames; 640 rows collapse to 314 visits
    where the read-time guess says 386.

    This endpoint therefore validates and applies; it does not decide. What it
    refuses is what matters:

    * **a producer id is never overwritten, `overwrite` or not.** The producer
      watched that visit happen; this endpoint worked it out afterwards, and a
      reconstruction must not outrank an observation. `overwrite` only re-runs
      over rows an earlier reconstruction stamped, which is what makes it safe
      to re-group as more archives are recovered;
    * a row id that does not exist is reported, not silently skipped;
    * one row may not be claimed by two visits in the same request.
    """
    if not body.visits:
        raise HTTPException(status_code=400, detail="沒有提供任何造訪")

    claimed: dict[int, tuple[str, Optional[int]]] = {}
    conflicts: list[dict] = []
    for visit in body.visits:
        for det_id in visit.detection_ids:
            if det_id in claimed and claimed[det_id][0] != visit.visit_id:
                conflicts.append({"id": det_id, "visits": [claimed[det_id][0], visit.visit_id]})
            claimed[det_id] = (visit.visit_id, visit.peak)
    if conflicts:
        raise HTTPException(
            status_code=400,
            detail=f"同一列被指派到兩個造訪：{conflicts[:5]}",
        )

    rows = (await session.execute(
        select(Detection).where(Detection.id.in_(sorted(claimed)))
    )).scalars().all()
    found = {d.id: d for d in rows}

    planned, skipped_existing, missing = [], [], sorted(set(claimed) - set(found))
    observed = 0
    for det_id, (visit_id, peak) in claimed.items():
        det = found.get(det_id)
        if det is None:
            continue
        if det.visit_id and not det.visit_id.startswith(BACKFILL_VISIT_PREFIX):
            # A producer id. Nothing here may replace it.
            observed += 1
            continue
        if det.visit_id and not body.overwrite:
            skipped_existing.append({"id": det_id, "keeps": det.visit_id})
            continue
        if det.visit_id == visit_id and det.visit_peak == peak:
            continue
        planned.append({"id": det_id, "camera_id": det.camera_id,
                        "timestamp": det.timestamp.isoformat(),
                        "from": det.visit_id, "to": visit_id, "peak": peak})

    manifest = {
        "dry_run": body.dry_run,
        "visits": len(body.visits),
        "rows_assigned": len(planned),
        "skipped_producer_ids": observed,
        "skipped_already_assigned": len(skipped_existing),
        "missing_rows": missing,
        "detail": planned,
    }
    if body.dry_run or not planned:
        logger.info("Visit backfill dry run: %d rows over %d visits",
                    len(planned), len(body.visits))
        return manifest

    for p in planned:
        det = found[p["id"]]
        det.visit_id = p["to"]
        det.visit_peak = p["peak"]
    await session.commit()
    logger.info("Visit backfill: %d rows stamped across %d visits (%d kept a producer id)",
                len(planned), len(body.visits), len(skipped_existing))
    return manifest


class TimelineAssignment(BaseModel):
    """One row's reconstructed position inside its clip, and which clip.

    `trail_path` travels with the offsets on purpose. A reconstructed visit is
    spread over several clips — 146 of 330 visits, one of them over thirteen
    (measured 2026-08-29) — and merging them means pointing a row at a
    different video AND placing it on that video's axis. Doing either alone
    leaves the row describing a position in a film it is no longer showing,
    which is the defect this whole area has already produced once.
    """

    id: int
    trail_path: Optional[str] = None
    clip_offset_seconds: Optional[float] = None
    clip_end_offset_seconds: Optional[float] = None
    clip_duration_seconds: Optional[float] = None
    clip_segments: Optional[list] = None

    @model_validator(mode="after")
    def _inside_its_own_clip(self) -> "TimelineAssignment":
        """An offset may not point past the video it describes.

        This is the exact defect the producer was carrying until 2026-08-28:
        offsets measured against one time base, written beside a duration
        measured against another, so a row claimed second 8.11 of a 4.64 s clip
        and every seek landed on the last frame. A reconstruction is a second
        way in for the same mistake, so it is refused here rather than stored
        and discovered later in a player.
        """
        d = self.clip_duration_seconds
        if d is None:
            if self.clip_offset_seconds is not None or self.clip_segments:
                raise ValueError(f"列 {self.id}：沒有影片長度就不能給位置")
            return self
        if d <= 0:
            raise ValueError(f"列 {self.id}：影片長度必須為正，收到 {d}")
        bounds = [("clip_offset_seconds", self.clip_offset_seconds),
                  ("clip_end_offset_seconds", self.clip_end_offset_seconds)]
        for name, v in bounds:
            if v is not None and not (0 <= v <= d):
                raise ValueError(f"列 {self.id}：{name}={v} 落在 0–{d} 秒的影片之外")
        a, b = self.clip_offset_seconds, self.clip_end_offset_seconds
        if a is not None and b is not None and a > b:
            raise ValueError(f"列 {self.id}：起點 {a} 晚於終點 {b}")
        for seg in self.clip_segments or []:
            if len(seg) != 2 or not (0 <= seg[0] <= seg[1] <= d):
                raise ValueError(f"列 {self.id}：片段 {seg} 落在 0–{d} 秒之外")
        return self


class BackfillTimelineRequest(BaseModel):
    """`dry_run` TRUE by default; producer-written rows are never touched."""

    dry_run: bool = True
    rows: list[TimelineAssignment]


@app.post("/api/admin/detections/backfill-timeline", dependencies=[Depends(require_admin)])
async def backfill_detection_timeline(
    body: BackfillTimelineRequest,
    session: AsyncSession = Depends(get_session),
):
    """Fill in where-in-the-clip for rows written before the producer recorded it.

    Reconstructed off the archived frames, and worth less than what the producer
    writes — say so plainly, because the numbers look identical once they are in
    the same columns:

    * the archive carries **no timestamps**, only frame indices, so seconds are
      interpolated across an evenly-spaced archive from the clip's known length;
    * re-tracking a **subsampled** archive mints ids the producer never had (a
      94-frame clip produced 19), so rows are matched to a track by box overlap,
      not by identity.

    What keeps the two apart afterwards is the visit id: a producer visit is a
    bare timestamp, a reconstructed one carries `BACKFILL_VISIT_PREFIX`. This
    endpoint **refuses any row whose visit was recorded by the producer**, so a
    reconstruction can never overwrite an observation — the same invariant as
    the visit backfill, applied to the same rows for the same reason.
    """
    if not body.rows:
        raise HTTPException(status_code=400, detail="沒有提供任何列")

    wanted = {r.id: r for r in body.rows}
    found = {
        d.id: d
        for d in (await session.execute(
            select(Detection).where(Detection.id.in_(sorted(wanted)))
        )).scalars().all()
    }

    missing_clips = sorted({
        w.trail_path for w in body.rows
        if w.trail_path and not storage.exists(w.trail_path)
    })
    if missing_clips:
        raise HTTPException(
            status_code=400,
            detail=f"這些影片不在儲存空間裡，不能指過去: {missing_clips[:5]}",
        )

    planned, observed, unchanged = [], [], 0
    for det_id, want in wanted.items():
        det = found.get(det_id)
        if det is None:
            continue
        if det.visit_id and not det.visit_id.startswith(BACKFILL_VISIT_PREFIX):
            observed.append(det_id)
            continue
        if det.clip_duration_seconds == want.clip_duration_seconds and \
                det.clip_offset_seconds == want.clip_offset_seconds and \
                (not want.trail_path or det.trail_path == want.trail_path):
            unchanged += 1
            continue
        planned.append(want)

    manifest = {
        "dry_run": body.dry_run,
        "submitted": len(body.rows),
        "rows_filled": len(planned),
        "rows_retargeted": sum(1 for w in planned if w.trail_path),
        "skipped_producer_rows": len(observed),
        "already_matching": unchanged,
        "missing_rows": sorted(set(wanted) - set(found)),
    }
    if body.dry_run or not planned:
        logger.info("Timeline backfill dry run: %d rows would be filled", len(planned))
        return manifest

    for want in planned:
        det = found[want.id]
        if want.trail_path:
            det.trail_path = want.trail_path
        det.clip_offset_seconds = want.clip_offset_seconds
        det.clip_end_offset_seconds = want.clip_end_offset_seconds
        det.clip_duration_seconds = want.clip_duration_seconds
        det.clip_segments = want.clip_segments
    await session.commit()
    logger.info("Timeline backfill: %d rows filled, %d producer rows left alone",
                len(planned), len(observed))
    return manifest


class AnimalGroup(BaseModel):
    """One animal inside one visit: the rows whose track fragments chain.

    Groups name DETECTION ids, not track ids, for the same reason the
    timeline backfill matches by box overlap: a reconstructed visit's
    per-frame tracks were re-minted off the archive and share no id space
    with the producer's track_id. The row↔track matching is evidence work
    and happens where the frames are; this endpoint only records the verdict.
    """

    visit_id: str
    # Two or more — a group of one says nothing and stays NULL by doctrine.
    detection_ids: list[int] = Field(min_length=2)
    # The links that formed the chain ({a, b, gap, err_widths}, …) — carried
    # into the manifest so an applied merge can always show its working.
    links: Optional[list] = None


class BackfillAnimalGroupsRequest(BaseModel):
    """Stamp rows with which-animal-of-its-visit, reversibly.

    `dry_run` TRUE by default. `source` records who decided — 'rule' for the
    calibrated motion rule, 'human' for a person — and a 'human' verdict is
    never overwritten or cleared by a 'rule' request. `clear` empties
    rule-assigned groups on the named visits instead of writing new ones,
    which is the whole reversibility story: no row is deleted or altered
    beyond these two columns, so clearing them restores the ungrouped truth.
    """

    dry_run: bool = True
    source: str = "rule"
    # Where the threshold came from, e.g. "animal-group calibration θ=1.0, calib 2026-08-30".
    evidence: str = ""
    clear: bool = False
    groups: list[AnimalGroup] = []
    # clear mode takes visit ids, not groups.
    visit_ids: list[str] = []

    @field_validator("source")
    @classmethod
    def _known_source(cls, v: str) -> str:
        if v not in ("rule", "human"):
            raise ValueError(f"source 必須是 rule 或 human，收到 {v!r}")
        return v


@app.post("/api/admin/detections/backfill-animal-groups", dependencies=[Depends(require_admin)])
async def backfill_animal_groups(
    body: BackfillAnimalGroupsRequest,
    session: AsyncSession = Depends(get_session),
):
    """Group a visit's rows by which animal they are — one level below visit_id.

    The grouping is computed OUTSIDE, on the box that holds the per-frame
    boxes, because the evidence is in the frames (the same division of labour
    as the visit backfill). This endpoint validates and applies:

    * a group names TRACK ids; rows are matched on (visit_id, track_id). A
      track that never produced a row is legal — the producer only files
      tracks that pass the gate — and a group left with fewer than two rows
      is reported and skipped, never written as a group of one;
    * **a 'human' verdict outranks a 'rule' one**: a rule request neither
      overwrites nor clears a row whose animal_source is 'human';
    * nothing else on the row is touched. track_id keeps the fragment ids
      exactly as the tracker minted them, which is what makes the merge
      reversible — `clear` puts every rule-stamped row back to NULL.
    """
    if body.clear:
        if not body.visit_ids:
            raise HTTPException(status_code=400, detail="clear 需要 visit_ids")
        rows = (await session.execute(
            select(Detection).where(
                Detection.visit_id.in_(body.visit_ids),
                Detection.animal_id.is_not(None),
            )
        )).scalars().all()
        clearable = [d for d in rows if d.animal_source != "human" or body.source == "human"]
        kept_human = len(rows) - len(clearable)
        if not body.dry_run:
            for det in clearable:
                det.animal_id = None
                det.animal_source = None
            await session.commit()
        logger.info("Animal-group clear (%s): %d rows cleared, %d human rows kept",
                    "dry run" if body.dry_run else "applied", len(clearable), kept_human)
        return {"dry_run": body.dry_run, "rows_cleared": len(clearable),
                "kept_human": kept_human}

    if not body.groups:
        raise HTTPException(status_code=400, detail="沒有提供任何分組")

    # One row may not be claimed by two groups.
    claimed: dict[int, int] = {}
    for i, grp in enumerate(body.groups):
        for det_id in grp.detection_ids:
            if det_id in claimed and claimed[det_id] != i:
                raise HTTPException(
                    status_code=400,
                    detail=f"列 {det_id} 被指派到兩個分組",
                )
            claimed[det_id] = i

    rows = (await session.execute(
        select(Detection).where(Detection.id.in_(sorted(claimed)))
    )).scalars().all()
    found = {d.id: d for d in rows}

    planned: list[dict] = []
    kept_human: list[int] = []
    wrong_visit: list[dict] = []
    missing = sorted(set(claimed) - set(found))
    single_row_groups = 0
    for grp in body.groups:
        members = [found[i] for i in grp.detection_ids if i in found]
        # A group is a claim about ONE visit; a row from another visit in it
        # means the caller's mapping drifted, and stamping it would fabricate
        # a cross-visit identity — which requires separate identity validation.
        stray = [d for d in members if d.visit_id != grp.visit_id]
        if stray:
            wrong_visit += [{"id": d.id, "visit_id": d.visit_id,
                             "claimed": grp.visit_id} for d in stray]
            members = [d for d in members if d.visit_id == grp.visit_id]
        writable = [d for d in members
                    if body.source == "human" or d.animal_source != "human"]
        kept_human += [d.id for d in members if d not in writable]
        if len(writable) < 2:
            single_row_groups += 1
            continue
        # Deterministic and stable under re-runs: the visit plus the smallest
        # row id in the group. Re-applying the same grouping is a no-op.
        animal_id = f"{grp.visit_id}:a{min(d.id for d in writable)}"
        for det in writable:
            if det.animal_id == animal_id and det.animal_source == body.source:
                continue
            planned.append({"id": det.id, "track_id": det.track_id,
                            "from": det.animal_id, "to": animal_id,
                            "links": grp.links})

    manifest = {
        "dry_run": body.dry_run,
        "source": body.source,
        "evidence": body.evidence,
        "groups": len(body.groups),
        "rows_stamped": len(planned),
        "groups_without_two_rows": single_row_groups,
        "kept_human": kept_human,
        "rows_in_wrong_visit": wrong_visit,
        "missing_rows": missing,
        "detail": planned[:200],
    }
    if body.dry_run or not planned:
        logger.info("Animal-group backfill dry run: %d rows over %d groups",
                    len(planned), len(body.groups))
        return manifest

    for p in planned:
        det = found[p["id"]]
        det.animal_id = p["to"]
        det.animal_source = body.source
    await session.commit()
    logger.info("Animal-group backfill (%s): %d rows stamped over %d groups — %s",
                body.source, len(planned), len(body.groups), body.evidence)
    return manifest


# ── Admin: user management (Phase 2) ─────────────────────────────────────────


async def _enabled_admin_count(
    session: AsyncSession, exclude_id: Optional[int] = None
) -> int:
    """Number of enabled admins, optionally excluding one row (lockout guard)."""
    q = select(func.count(User.id)).where(
        User.role == "admin", User.enabled.is_(True)
    )
    if exclude_id is not None:
        q = q.where(User.id != exclude_id)
    return (await session.execute(q)).scalar() or 0


@app.get("/api/admin/features", response_model=list[FeatureOut], dependencies=[Depends(require_admin)])
async def list_features():
    """The catalog of per-user toggleable features (for the admin UI)."""
    return [FeatureOut(key=k, label=v) for k, v in config.FEATURES.items()]


@app.get("/api/admin/users", response_model=list[UserOut], dependencies=[Depends(require_admin)])
async def list_users(session: AsyncSession = Depends(get_session)):
    users = (
        await session.execute(select(User).order_by(User.created_at))
    ).scalars().all()
    return users


@app.post("/api/admin/users", response_model=UserOut, status_code=201, dependencies=[Depends(require_admin)])
async def create_user(body: UserCreate, session: AsyncSession = Depends(get_session)):
    code = body.code.strip()
    if not code:
        raise HTTPException(status_code=400, detail="存取碼不可為空")
    if body.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="角色無效")
    code_hash = _hash_code(code)
    existing = (
        await session.execute(select(User).where(User.code_hash == code_hash))
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="存取碼已被使用")
    user = User(
        name=body.name.strip(),
        code_hash=code_hash,
        role=body.role,
        enabled=body.enabled,
        allowed_cameras=body.allowed_cameras,
        feature_flags=body.feature_flags,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    logger.info("Admin created user id=%d role=%s", user.id, user.role)
    return user


@app.patch("/api/admin/users/{user_id}", response_model=UserOut, dependencies=[Depends(require_admin)])
async def update_user(
    user_id: int, body: UserUpdate, session: AsyncSession = Depends(get_session)
):
    user = (
        await session.execute(select(User).where(User.id == user_id))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="使用者不存在")

    new_role = body.role if body.role is not None else user.role
    if new_role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="角色無效")
    new_enabled = body.enabled if body.enabled is not None else user.enabled

    # Lockout guard: never leave zero enabled admins.
    if (user.role == "admin" and user.enabled) and not (
        new_role == "admin" and new_enabled
    ):
        if await _enabled_admin_count(session, exclude_id=user.id) == 0:
            raise HTTPException(status_code=400, detail="至少需保留一位啟用中的管理員")

    if body.code is not None:
        code = body.code.strip()
        if not code:
            raise HTTPException(status_code=400, detail="存取碼不可為空")
        code_hash = _hash_code(code)
        clash = (
            await session.execute(
                select(User).where(User.code_hash == code_hash, User.id != user.id)
            )
        ).scalar_one_or_none()
        if clash:
            raise HTTPException(status_code=409, detail="存取碼已被使用")
        user.code_hash = code_hash
    if body.name is not None:
        user.name = body.name.strip()
    if body.role is not None:
        user.role = body.role
    if body.enabled is not None:
        user.enabled = body.enabled
    if body.allowed_cameras is not None:
        user.allowed_cameras = body.allowed_cameras
    if body.feature_flags is not None:
        user.feature_flags = body.feature_flags

    await session.commit()
    await session.refresh(user)
    logger.info("Admin updated user id=%d", user.id)
    return user


@app.delete("/api/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def delete_user(user_id: int, session: AsyncSession = Depends(get_session)):
    user = (
        await session.execute(select(User).where(User.id == user_id))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="使用者不存在")
    if (user.role == "admin" and user.enabled) and await _enabled_admin_count(
        session, exclude_id=user.id
    ) == 0:
        raise HTTPException(status_code=400, detail="至少需保留一位啟用中的管理員")
    await session.delete(user)
    await session.commit()
    logger.info("Admin deleted user id=%d", user_id)
    return {"deleted": user_id}


# ── Admin: model training (Ultralytics Platform control plane) ───────────────
#
# The Platform is the single source of truth for datasets/models/training; the
# dashboard drives it via services/platform.py. Promotion (上板) dispatches the
# model-promote.yml workflow on the self-hosted mac-mini runner, which runs the
# ratchet accuracy gate on BOTH splits before swapping backend/best.pt.


async def require_training_access(user: User = Depends(get_current_user)) -> User:
    """Admin OR the per-user 'training' feature flag.

    Unlike live_feeds/downloads (default ON via _feature_enabled), training is
    OPT-IN: it can start paid GPU jobs and trigger deploys, so a missing flag
    means no access.
    """
    if user.role == "admin" or bool((user.feature_flags or {}).get("training", False)):
        return user
    raise HTTPException(status_code=403, detail="Forbidden")


class AddToDatasetRequest(BaseModel):
    detection_ids: list[int]


class TrainingStartRequest(BaseModel):
    name: str
    base: str = "yolo26n.pt"
    epochs: int = 1000
    imgsz: int = 1920
    gpu: str = "rtx-6000-ada"
    # Which dataset to train on. None = the ULTRALYTICS_DATASET master, which
    # was the only option while there was one dataset. The shipping model was
    # trained on `monkey-cocotree-allin-0816` (1220 images, the train/val of all
    # six merged) while this button still sent the 1095-image July master — so a
    # retrain from the UI would have quietly used less data than production.
    dataset: Optional[str] = None


class PromoteRequest(BaseModel):
    model_id: str
    # 強制上板 — ship even though the ratchet rejected the model. The gate still
    # runs and still reports; only the abort is skipped, and the baseline is
    # left alone so the floor never moves to fit the model.
    force: bool = False


class EvalResultIn(BaseModel):
    """Gate scores POSTed by model-evaluate.yml from the edge runner."""

    model_id: str
    split: str  # 'test' | 'val'
    precision: float
    recall: float
    f1: float
    passed: bool
    baseline_f1: float = 0.0


@app.get("/api/admin/training/models", dependencies=[Depends(require_training_access)])
async def training_models(session: AsyncSession = Depends(get_session)):
    """Project model list with normalized status + P/R/mAP50 (訓練 tab).

    Each model also carries `eval`: the latest 本站評測 scores per split
    (from EvalResult — our imgsz-1600/fp16/MPS standard, not the platform's).
    """
    from services import platform as ultralytics_platform

    models = await ultralytics_platform.list_models(config.ULTRALYTICS_PROJECT)
    if models:
        rows = (
            await session.execute(
                select(EvalResult)
                .where(EvalResult.model_id.in_([m["id"] for m in models]))
                .order_by(EvalResult.id.asc())
            )
        ).scalars().all()
        latest: dict[str, dict] = {}
        for r in rows:  # ascending id → later rows win = latest per (model, split)
            latest.setdefault(r.model_id, {})[r.split] = {
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "passed": r.passed,
                "baseline_f1": r.baseline_f1,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
        for m in models:
            m["eval"] = latest.get(m["id"])
    return models


@app.post("/api/admin/training/add-to-dataset", dependencies=[Depends(require_training_access)])
async def training_add_to_dataset(
    body: AddToDatasetRequest,
    session: AsyncSession = Depends(get_session),
):
    """Upload the RAW snapshots of the selected detections into the dataset.

    Images arrive unlabeled on the Platform — background negatives until boxes
    are drawn in the Platform UI.
    """
    from services import platform as ultralytics_platform

    if not body.detection_ids:
        raise HTTPException(status_code=400, detail="未選取任何偵測紀錄")
    result = await ultralytics_platform.add_detections_to_dataset(
        body.detection_ids, session
    )
    logger.info(
        "Training dataset upload: %d uploaded, %d failed, %d near-duplicate",
        result["uploaded"], len(result["failed"]), len(result.get("duplicates", [])),
    )
    return result


@app.post("/api/admin/training/start", dependencies=[Depends(require_training_access)])
async def training_start(body: TrainingStartRequest):
    """Start a cloud training run — platform response passthrough (modelId)."""
    from services import platform as ultralytics_platform

    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="請輸入模型名稱")
    result = await ultralytics_platform.start_training(
        name=name, base=body.base, epochs=body.epochs, imgsz=body.imgsz, gpu=body.gpu,
        dataset=body.dataset,
    )
    logger.info(
        "Training started: name=%s base=%s epochs=%d dataset=%s",
        name, body.base, body.epochs, body.dataset or config.ULTRALYTICS_DATASET,
    )
    return result


@app.get("/api/admin/training/status/{model_id}", dependencies=[Depends(require_training_access)])
async def training_status(model_id: str):
    """Passthrough of the platform's training-status payload (progress polling)."""
    from services import platform as ultralytics_platform

    return await ultralytics_platform.training_status(model_id)


@app.post("/api/admin/training/promote", dependencies=[Depends(require_training_access)])
async def training_promote(body: PromoteRequest):
    """Dispatch the accuracy-gated promotion workflow for a trained model."""
    from services import platform as ultralytics_platform

    await ultralytics_platform.dispatch_promote(body.model_id, force=body.force)
    logger.info(
        "Promotion workflow dispatched for model %s (force=%s)",
        body.model_id,
        body.force,
    )
    return {
        "message": (
            "已觸發強制上板 — 棘輪照跑照記錄，但不通過也會部署；baseline 不會更動"
            if body.force
            else "已觸發上板流程（棘輪雙檢通過才會部署）"
        ),
        "model_id": body.model_id,
        "force": body.force,
    }


@app.get("/api/admin/training/datasets", dependencies=[Depends(require_training_access)])
async def training_datasets():
    """資料集清單 — every Platform dataset, newest first.

    `ULTRALYTICS_DATASET` pins one master, which was right when there was one.
    There are six benchmarks now; the browser picks instead of showing whichever
    one the env var happens to name (it named the July set, 7th newest)."""
    from services import platform as ultralytics_platform

    return await ultralytics_platform.list_datasets()


@app.get("/api/admin/training/dataset-images", dependencies=[Depends(require_training_access)])
async def training_dataset_images(
    split: Optional[str] = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=24, ge=1, le=100),
    dataset: Optional[str] = Query(default=None),
):
    """資料集瀏覽 — one page of platform images with relative-coord boxes.

    Proxies GET datasets/{id}/images (60 s cached in services/platform.py so
    grid paging and multiple viewers don't hammer the platform)."""
    from services import platform as ultralytics_platform

    if split in ("", "all"):
        split = None
    if split not in (None, "train", "val", "test"):
        raise HTTPException(status_code=400, detail="split 只能是 train / val / test")
    return await ultralytics_platform.list_dataset_images(split, offset, limit, dataset)


_TRAINING_UPLOAD_MAX_FILES = 50


@app.post("/api/admin/training/upload-files", dependencies=[Depends(require_training_access)])
async def training_upload_files(files: list[UploadFile] = File(...)):
    """拖曳上傳 — push local JPG/PNG files straight into the training dataset.

    Images arrive UNLABELED on the Platform (= background negatives until
    boxes are drawn in the Platform UI) — same semantics as add-to-dataset."""
    from services import platform as ultralytics_platform

    if len(files) > _TRAINING_UPLOAD_MAX_FILES:
        raise HTTPException(
            status_code=400, detail=f"一次最多上傳 {_TRAINING_UPLOAD_MAX_FILES} 張圖片"
        )
    payload: list[tuple[str, bytes, str]] = []
    rejected: list[dict] = []
    for f in files:
        name = f.filename or "unnamed"
        lower = name.lower()
        is_image = (f.content_type or "").lower() in ("image/jpeg", "image/png") or \
            lower.endswith((".jpg", ".jpeg", ".png"))
        if not is_image:
            rejected.append({"name": name, "reason": "僅支援 JPG / PNG 圖片"})
            continue
        data = await f.read()
        if not data:
            rejected.append({"name": name, "reason": "空檔案"})
            continue
        content_type = "image/png" if lower.endswith(".png") else "image/jpeg"
        payload.append((name, data, content_type))

    result = {"uploaded": 0, "failed": [], "duplicates": []}
    if payload:
        result = await ultralytics_platform.upload_files(payload)
    result["failed"] = rejected + result["failed"]
    logger.info(
        "Training file upload: %d uploaded, %d failed, %d near-duplicate",
        result["uploaded"], len(result["failed"]), len(result.get("duplicates", [])),
    )
    return result


@app.post("/api/admin/training/eval-results", dependencies=[Depends(require_training_access)])
async def training_eval_results(
    body: EvalResultIn, session: AsyncSession = Depends(get_session)
):
    """本站評測回報 — model-evaluate.yml POSTs per-split gate scores here.

    Append-only history; GET /api/admin/training/models surfaces the latest
    row per (model_id, split)."""
    if body.split not in ("test", "val"):
        raise HTTPException(status_code=400, detail="split 只能是 test / val")
    session.add(EvalResult(
        model_id=body.model_id,
        split=body.split,
        precision=body.precision,
        recall=body.recall,
        f1=body.f1,
        passed=body.passed,
        baseline_f1=body.baseline_f1,
    ))
    await session.commit()
    logger.info(
        "Eval result stored: model=%s split=%s F1=%.4f passed=%s",
        body.model_id, body.split, body.f1, body.passed,
    )
    return {"stored": True}


@app.post("/api/admin/training/evaluate", dependencies=[Depends(require_training_access)])
async def training_evaluate(body: PromoteRequest):
    """Dispatch the edge-runner evaluation workflow（本站評測，不動線上模型）."""
    from services import platform as ultralytics_platform

    await ultralytics_platform.dispatch_evaluate(body.model_id)
    logger.info("Evaluation workflow dispatched for model %s", body.model_id)
    return {
        "message": "已觸發本站評測 — 結果約數分鐘後回寫到模型列表",
        "model_id": body.model_id,
    }


# ── 去重訓練集 (master + derived) ─────────────────────────────────────────────
#
# The platform dataset is the MASTER — everything is added there, nothing is
# ever deleted, test/val stay pinned. Training runs on a DERIVED dataset
# regenerated from it with the TRAIN split deduplicated (measured 2026-08-03:
# 25% fewer images trained a better model, test recall 0.704 -> 0.756).
#
# Regeneration takes minutes, so POST only STARTS it and GET reports progress;
# the newest DerivedDataset row is what 用訓練集開始訓練 trains on.


def _derived_row(row: DerivedDataset | None) -> Optional[dict]:
    if row is None:
        return None
    return {
        "slug": row.slug,
        "name": row.name,
        "dataset_id": row.dataset_id,
        "kept": row.kept,
        "dropped": row.dropped,
        "negatives_before": row.negatives_before,
        "negatives_after": row.negatives_after,
        "val_images": row.val_images,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


async def _latest_derived(session: AsyncSession) -> DerivedDataset | None:
    return (
        await session.execute(
            select(DerivedDataset).order_by(DerivedDataset.id.desc()).limit(1)
        )
    ).scalars().first()


async def _store_derived(result: dict) -> None:
    """Persist a finished regeneration. Runs in the background task, after the
    request's session is long gone — hence its own session."""
    async with async_session() as session:
        session.add(DerivedDataset(
            slug=result["dataset_slug"],
            name=result.get("name", ""),
            dataset_id=result.get("dataset_id", ""),
            kept=result.get("kept", 0),
            dropped=result.get("dropped", 0),
            negatives_before=result.get("negatives_before", 0),
            negatives_after=result.get("negatives_after", 0),
            val_images=result.get("val_images", 0),
        ))
        await session.commit()
    logger.info("Derived dataset recorded: %s", result["dataset_slug"])


@app.post(
    "/api/admin/training/regenerate-dataset",
    dependencies=[Depends(require_training_access)],
)
async def training_regenerate_dataset():
    """Start rebuilding the deduplicated derived dataset (background job).

    Returns immediately with the job state — the work is minutes long (export →
    download every train/val image → perceptual hash → zip → upload → wait for
    the platform ingest). Poll GET on the same path for progress."""
    from services import platform as ultralytics_platform

    state = await ultralytics_platform.start_regenerate(_store_derived)
    logger.info("Derived dataset regeneration started")
    return state


@app.get(
    "/api/admin/training/regenerate-dataset",
    dependencies=[Depends(require_training_access)],
)
async def training_regenerate_status(session: AsyncSession = Depends(get_session)):
    """Regeneration progress + the newest derived dataset on record."""
    from services import platform as ultralytics_platform

    state = ultralytics_platform.regenerate_state()
    state["latest"] = _derived_row(await _latest_derived(session))
    return state


@app.post(
    "/api/admin/training/train-on-derived",
    dependencies=[Depends(require_training_access)],
)
async def training_train_on_derived(
    body: TrainingStartRequest, session: AsyncSession = Depends(get_session)
):
    """Start a cloud training run against the newest deduplicated 訓練集."""
    from services import platform as ultralytics_platform

    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="請輸入模型名稱")
    row = await _latest_derived(session)
    if row is None:
        raise HTTPException(
            status_code=400, detail="尚未生成去重訓練集 — 請先按「重新生成訓練集」"
        )
    result = await ultralytics_platform.start_training(
        name=name, base=body.base, epochs=body.epochs, imgsz=body.imgsz,
        gpu=body.gpu, dataset=row.slug,
    )
    result["dataset_slug"] = row.slug
    logger.info("Training started on derived dataset %s: name=%s", row.slug, name)
    return result


# ── Dataset editing (annotation editor → Ultralytics Platform) ────────────────


class BoxIn(BaseModel):
    """One box in the dashboard's shape: RELATIVE top-left origin, 0-1."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)
    cls: int = Field(default=0, ge=0)


class LabelsUpdateRequest(BaseModel):
    boxes: list[BoxIn] = Field(max_length=200)
    # Which dataset the edit was made against. None = the ULTRALYTICS_DATASET
    # master, so older clients keep working.
    dataset: Optional[str] = None


class SplitMoveItem(BaseModel):
    hash: str
    split: str  # the image's CURRENT split (platform keys sources by hash+split)


class SplitMoveRequest(BaseModel):
    images: list[SplitMoveItem] = Field(min_length=1, max_length=100)
    split: str  # target
    dataset: Optional[str] = None


class DeleteImagesRequest(BaseModel):
    hashes: list[str] = Field(min_length=1, max_length=100)
    dataset: Optional[str] = None


@app.put(
    "/api/admin/training/dataset-images/{image_hash}/labels",
    dependencies=[Depends(require_training_access)],
)
async def training_update_image_labels(image_hash: str, body: LabelsUpdateRequest):
    """標註編輯 — replace one image's boxes on the Platform (source of truth).

    Boxes arrive top-left relative; platform.py converts back to YOLO center
    format and invalidates the browser cache."""
    from services import platform as ultralytics_platform

    result = await ultralytics_platform.update_image_labels(
        image_hash, [b.model_dump() for b in body.boxes], body.dataset
    )
    logger.info(
        "Dataset labels updated: hash=%s boxes=%d", image_hash, len(body.boxes)
    )
    return result


@app.post(
    "/api/admin/training/dataset-images/split",
    dependencies=[Depends(require_training_access)],
)
async def training_move_images_split(body: SplitMoveRequest):
    """切分搬移 — move images between train/val/test on the Platform.

    Moving INTO test/val changes the ratchet benchmark — the client warns that
    基準線需要重定錨 before calling this."""
    from services import platform as ultralytics_platform

    if body.split not in ("train", "val", "test"):
        raise HTTPException(status_code=400, detail="split 只能是 train / val / test")
    for item in body.images:
        if item.split not in ("train", "val", "test"):
            raise HTTPException(status_code=400, detail="來源 split 只能是 train / val / test")
    result = await ultralytics_platform.move_images_split(
        [i.model_dump() for i in body.images], body.split, body.dataset
    )
    logger.info(
        "Dataset split move: %d image(s) → %s (moved=%s)",
        len(body.images), body.split, result.get("moved"),
    )
    return result


@app.post(
    "/api/admin/training/dataset-images/delete",
    dependencies=[Depends(require_training_access)],
)
async def training_delete_images(body: DeleteImagesRequest):
    """刪除影像 — bulk delete from the Platform dataset. Destructive."""
    from services import platform as ultralytics_platform

    result = await ultralytics_platform.delete_images(body.hashes, body.dataset)
    logger.info(
        "Dataset images deleted: requested=%d deleted=%s",
        len(body.hashes), result.get("deleted"),
    )
    return result
