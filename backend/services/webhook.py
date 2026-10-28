"""Webhook notification service — fans out to all enabled webhooks."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import httpx
from sqlalchemy import select

from database import async_session
from models import Webhook
from services.storage import storage

if TYPE_CHECKING:
    from services.detector import DetectionResult

logger = logging.getLogger(__name__)

_TW_TZ = timezone(timedelta(hours=8))


def _read_snapshot(filename: str) -> bytes | None:
    """Read a snapshot file from storage, return bytes or None."""
    if not filename:
        return None
    return storage.download(filename)


async def send_system_alert(title: str, description: str, color: int = 0xE74C3C) -> None:
    """Send a plain embed (no images) to all enabled webhooks — system events
    like the edge node going offline/recovering. Errors are logged, never raised."""
    async with async_session() as session:
        webhooks = (
            await session.execute(
                select(Webhook).where(Webhook.enabled == True)  # noqa: E712
            )
        ).scalars().all()
    if not webhooks:
        logger.warning("[Webhook] System alert '%s' but no enabled webhooks", title)
        return

    payload = {
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "footer": {"text": "系統監控"},
            "timestamp": datetime.now(_TW_TZ).isoformat(),
        }],
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        for wh in webhooks:
            try:
                resp = await client.post(wh.url, json=payload)
                resp.raise_for_status()
            except Exception as e:
                logger.error("[Webhook] System alert to '%s' failed: %s", wh.name, repr(e))


async def send_notification(detection: "DetectionResult") -> None:
    """Send webhook notifications to ALL enabled webhooks."""
    logger.info(
        "[Webhook] Monkey detected! camera=%s confidence=%.2f timestamp=%s",
        detection.camera_id,
        detection.confidence,
        detection.timestamp.isoformat(),
    )

    async with async_session() as session:
        webhooks = (
            await session.execute(
                select(Webhook).where(Webhook.enabled == True)  # noqa: E712
            )
        ).scalars().all()

    if not webhooks:
        logger.debug("[Webhook] No enabled webhooks — skipping")
        return

    # Load both raw and annotated images
    raw_bytes = _read_snapshot(detection.image_path)
    annot_bytes = _read_snapshot(detection.annotated_path)

    ts = detection.timestamp.isoformat()
    payload = {
        "content": (
            f"\U0001f412 **偵測到猴子！**\n"
            f"攝影機：`{detection.camera_id}` | 信心度：`{detection.confidence:.0%}`"
        ),
        "embeds": [{
            "title": "猴子偵測警報",
            "color": 0xEF4444,
            "fields": [
                {"name": "攝影機", "value": detection.camera_id, "inline": True},
                {"name": "信心度", "value": f"{detection.confidence:.1%}", "inline": True},
                {"name": "追蹤 ID", "value": f"#{detection.track_id}", "inline": True},
            ],
            "timestamp": ts,
        }],
    }

    # Build file attachments: raw + annotated
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    if raw_bytes:
        files.append(("files[0]", ("raw.jpg", raw_bytes, "image/jpeg")))
        payload["embeds"][0]["image"] = {"url": "attachment://raw.jpg"}
    if annot_bytes:
        idx = len(files)
        files.append((f"files[{idx}]", ("annotated.jpg", annot_bytes, "image/jpeg")))

    async with httpx.AsyncClient(timeout=15) as client:
        for wh in webhooks:
            try:
                if files:
                    resp = await client.post(
                        wh.url,
                        data={"payload_json": json.dumps(payload)},
                        files=files,
                    )
                else:
                    resp = await client.post(wh.url, json=payload)
                resp.raise_for_status()
                logger.info("[Webhook] Sent to '%s' (status=%d)", wh.name or wh.url, resp.status_code)
                error = ""
            except Exception as exc:
                logger.error("[Webhook] Failed to send to '%s': %s", wh.name or wh.url, exc)
                error = str(exc)[:500]

            # Persist delivery status
            async with async_session() as session:
                db_wh = (
                    await session.execute(select(Webhook).where(Webhook.id == wh.id))
                ).scalar_one_or_none()
                if db_wh:
                    db_wh.last_error = error
                    await session.commit()


async def send_raw_debug_detection(
    camera_id: str,
    raw_jpeg: bytes,
    annotated_jpeg: bytes,
    top_conf: float,
    n_boxes: int,
    n_tracked: int,
) -> None:
    """DEV back door: fire webhook on any raw YOLO detection.

    Bypasses tracking, DB persistence, and snapshot storage. Sends TWO
    images to Discord:
      - annotated.jpg: with YOLO bboxes drawn (for diagnosis)
      - raw_<ts>.jpg:  clean original frame at JPEG 95 (download and feed
                       back into training as additional labels/hard negatives)

    Use this to diagnose tracking drops: the message shows `n_boxes` (raw
    YOLO positives) vs `n_tracked` (how many got a ByteTrack ID). If
    n_boxes > 0 but n_tracked == 0, that's the tracking gap.
    """
    async with async_session() as session:
        webhooks = (
            await session.execute(
                select(Webhook).where(Webhook.enabled == True)  # noqa: E712
            )
        ).scalars().all()

    if not webhooks:
        return

    now_tw = datetime.now(_TW_TZ)
    ts_iso = now_tw.isoformat()
    fname_ts = now_tw.strftime("%Y%m%d_%H%M%S")
    raw_fname = f"{camera_id}_raw_{fname_ts}.jpg"
    annot_fname = f"{camera_id}_annot_{fname_ts}.jpg"

    track_note = (
        f"{n_tracked}/{n_boxes} 有 track ID" if n_tracked else f"0/{n_boxes} ⚠️ 全部沒 ID"
    )
    payload = {
        "content": (
            f"\U0001f527 **[DEV] Raw 偵測（繞過 tracking）**\n"
            f"攝影機：`{camera_id}` | 最高信心度：`{top_conf:.0%}` | {track_note}\n"
            f"📎 `{raw_fname}` 為乾淨原圖，下載可作訓練樣本。"
        ),
        "embeds": [{
            "title": "[DEV] Raw YOLO detection",
            "color": 0x3B82F6,
            "fields": [
                {"name": "攝影機", "value": camera_id, "inline": True},
                {"name": "最高信心度", "value": f"{top_conf:.1%}", "inline": True},
                {"name": "Box 數", "value": str(n_boxes), "inline": True},
                {"name": "Track ID 數", "value": str(n_tracked), "inline": True},
            ],
            "timestamp": ts_iso,
            "image": {"url": f"attachment://{annot_fname}"},
        }],
    }
    files = [
        ("files[0]", (annot_fname, annotated_jpeg, "image/jpeg")),
        ("files[1]", (raw_fname, raw_jpeg, "image/jpeg")),
    ]

    async with httpx.AsyncClient(timeout=10) as client:
        for wh in webhooks:
            try:
                resp = await client.post(
                    wh.url,
                    data={"payload_json": json.dumps(payload)},
                    files=files,
                )
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("[DEV Webhook] send to '%s' failed: %s", wh.name or wh.url, exc)
