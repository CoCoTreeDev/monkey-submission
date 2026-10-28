"""Storage abstraction — local filesystem or Azure Blob Storage."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import config

logger = logging.getLogger(__name__)


class LocalStorage:
    """Store snapshots on local filesystem (development / fallback)."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(exist_ok=True)

    def upload(self, name: str, data: bytes) -> None:
        (self._dir / name).write_bytes(data)

    def local_path(self, name: str) -> Path:
        """Path for file responses after the caller validates the asset name."""
        return self._dir / name

    def download(self, name: str) -> bytes | None:
        p = self._dir / name
        if not p.is_file():
            return None
        return p.read_bytes()

    def download_stream(self, name: str) -> Iterator[bytes] | None:
        p = self._dir / name
        if not p.is_file():
            return None

        def _chunks() -> Iterator[bytes]:
            with open(p, "rb") as f:
                while chunk := f.read(64 * 1024):
                    yield chunk

        return _chunks()

    def delete(self, name: str) -> bool:
        p = self._dir / name
        if p.is_file():
            p.unlink()
            return True
        return False

    def delete_all(self) -> int:
        removed = 0
        for f in self._dir.iterdir():
            if f.is_file():
                f.unlink()
                removed += 1
        return removed

    def list_files_with_mtime(self) -> list[tuple[str, int, float]]:
        """(name, size, mtime-epoch) — for the orphan sweep."""
        out = []
        # Was SNAPSHOTS_DIR, a name this module never defines. Harmless while
        # production ran BlobStorage; after the 2026-09-21 move to the mini it
        # made the orphan sweep raise NameError every day, so nothing ever
        # cleaned the disk.
        for f in self._dir.iterdir():
            if f.is_file():
                st = f.stat()
                out.append((f.name, st.st_size, st.st_mtime))
        return out

    def list_files(self) -> list[tuple[str, int]]:
        return [
            (f.name, f.stat().st_size)
            for f in self._dir.iterdir()
            if f.is_file()
        ]

    def exists(self, name: str) -> bool:
        return (self._dir / name).is_file()


def _content_type(name: str) -> str:
    """MIME type from the filename — snapshots, videos, legacy GIFs and ZIPs."""
    lower = name.lower()
    if lower.endswith(".mp4"):
        return "video/mp4"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".zip"):
        return "application/zip"
    return "image/jpeg"


class BlobStorage:
    """Store snapshots in Azure Blob Storage."""

    def __init__(self, connection_string: str, container_name: str) -> None:
        from azure.storage.blob import BlobServiceClient

        # Chunk anything over BLOB_MAX_SINGLE_PUT_BYTES instead of the SDK's
        # 64 MB default. A detection's trail GIF or raw-frames ZIP is tens of MB
        # and the edge node pushes it over a site uplink: as one PUT, a slow
        # uplink hits the socket write timeout and the whole upload dies with it.
        # Chunked, each block is its own retried request, so a stall costs one
        # block. (Two detector threads died on exactly that timeout —
        # 2026-08-06 and 2026-08-08 — taking 場域甲 blind for 62 hours.)
        self._client = BlobServiceClient.from_connection_string(
            connection_string,
            max_single_put_size=config.BLOB_MAX_SINGLE_PUT_BYTES,
            max_block_size=config.BLOB_BLOCK_BYTES,
            connection_timeout=config.BLOB_CONNECT_TIMEOUT,
            read_timeout=config.BLOB_READ_TIMEOUT,
            retry_total=config.BLOB_RETRY_TOTAL,
        )
        self._container = self._client.get_container_client(container_name)
        # Ensure container exists
        try:
            self._container.get_container_properties()
        except Exception:
            self._container.create_container()
            logger.info("Created blob container: %s", container_name)

    def upload(self, name: str, data: bytes) -> None:
        # Content type must be set at upload time. Azure defaults to
        # application/octet-stream, and a browser navigating to such a URL
        # DOWNLOADS the file instead of rendering it — so the dashboard's 檢視
        # button appeared broken even when the bytes arrived fine.
        from azure.storage.blob import ContentSettings

        # Whole-upload retry on top of the SDK's per-request one: a transport
        # error mid-way through a chunked upload fails the operation, and the
        # caller (a detector worker finalising a track) has the bytes in hand and
        # nothing better to do with them. Raises after the last attempt — callers
        # that can live without the artefact must catch it.
        settings = ContentSettings(content_type=_content_type(name))
        attempts = max(1, config.BLOB_UPLOAD_ATTEMPTS)
        for attempt in range(1, attempts + 1):
            try:
                self._container.upload_blob(
                    name, data, overwrite=True, content_settings=settings,
                )
                return
            except Exception as exc:
                if attempt == attempts:
                    raise
                logger.warning(
                    "Blob upload failed for %s (%.1f MB, attempt %d/%d): %s — retrying",
                    name, len(data) / 1024 / 1024, attempt, attempts, exc,
                )
                # Short: the caller is a detector thread, and it is not watching
                # its camera while it sleeps here.
                time.sleep(2)

    def download(self, name: str) -> bytes | None:
        try:
            return self._container.download_blob(name).readall()
        except Exception:
            return None

    def download_stream(self, name: str) -> Iterator[bytes] | None:
        try:
            stream = self._container.download_blob(name)
        except Exception:
            return None
        return stream.chunks()

    def signed_url(self, name: str, ttl_seconds: int = 3600) -> str | None:
        """Short-lived read-only SAS URL so browsers download straight from
        Blob instead of pushing 12-15 MB trail GIFs through the 1 vCPU
        container. Returns None when the account key is unavailable."""
        try:
            from datetime import datetime, timedelta, timezone

            from azure.storage.blob import BlobSasPermissions, generate_blob_sas

            bc = self._container.get_blob_client(name)
            sas = generate_blob_sas(
                account_name=self._client.account_name,
                container_name=self._container.container_name,
                blob_name=name,
                account_key=self._client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            )
            return f"{bc.url}?{sas}"
        except Exception:
            logger.exception("SAS URL generation failed for %s", name)
            return None

    def delete(self, name: str) -> bool:
        try:
            self._container.delete_blob(name)
            return True
        except Exception:
            return False

    def delete_all(self) -> int:
        removed = 0
        for blob in self._container.list_blobs():
            self._container.delete_blob(blob.name)
            removed += 1
        return removed

    def list_files_with_mtime(self) -> list[tuple[str, int, float]]:
        """(name, size, last_modified-epoch) — for the orphan sweep."""
        return [
            (b.name, b.size, b.last_modified.timestamp() if b.last_modified else 0.0)
            for b in self._container.list_blobs()
        ]

    def list_files(self) -> list[tuple[str, int]]:
        return [
            (blob.name, blob.size)
            for blob in self._container.list_blobs()
        ]

    def exists(self, name: str) -> bool:
        try:
            self._container.get_blob_client(name).get_blob_properties()
            return True
        except Exception:
            return False


# ── Module-level singleton ──────────────────────────────────────────────────

def _init_storage() -> LocalStorage | BlobStorage:
    if config.AZURE_STORAGE_CONNECTION_STRING:
        logger.info(
            "Using Azure Blob Storage (container=%s)",
            config.AZURE_STORAGE_CONTAINER,
        )
        return BlobStorage(
            config.AZURE_STORAGE_CONNECTION_STRING,
            config.AZURE_STORAGE_CONTAINER,
        )
    logger.info("Using local filesystem storage (%s)", config.SNAPSHOTS_DIR)
    return LocalStorage(config.SNAPSHOTS_DIR)


storage = _init_storage()
