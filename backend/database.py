"""Database setup with async SQLAlchemy."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config import DATABASE_URL

logger = logging.getLogger(__name__)

# asyncpg uses ssl=, not sslmode=; strip sslmode to avoid TypeError, use SSL context
_url = DATABASE_URL
if "sslmode=" in _url:
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    parsed = urlparse(_url)
    qs = parse_qs(parsed.query)
    qs.pop("sslmode", None)
    parsed = parsed._replace(query=urlencode(qs, doseq=True) if qs else "")
    _url = urlunparse(parsed)

_connect_args = {}
if "postgresql" in _url:
    import ssl

    _connect_args["ssl"] = ssl.create_default_context()
    _connect_args["timeout"] = 10  # Fail fast if DB unreachable

engine = create_async_engine(_url, echo=False, connect_args=_connect_args)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Create all tables, then apply additive column migrations.

    create_all never ALTERs an existing table, so columns added after a table
    first shipped need an explicit (idempotent) migration here.
    """
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    for stmt in _COLUMN_MIGRATIONS:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(stmt))
        except Exception as exc:  # column already there → SQLite raises, PG no-ops
            if "duplicate column" not in str(exc).lower():
                logger.debug("Migration skipped (%s): %s", stmt, exc)


# Additive-only, safe to re-run. PostgreSQL supports IF NOT EXISTS; SQLite
# raises "duplicate column name" on a repeat, which init_db swallows.
_SQLITE = DATABASE_URL.startswith("sqlite")
_COLUMN_MIGRATIONS = tuple(
    f"ALTER TABLE detections ADD COLUMN {'IF NOT EXISTS ' if not _SQLITE else ''}{column}"
    for column in (
        "user_visible BOOLEAN",
        # Existing rows land unlocked, rather than NULL.
        f"locked BOOLEAN NOT NULL DEFAULT {'0' if _SQLITE else 'FALSE'}",
        # NULL means the row predates visit recording; do not invent a visit.
        "visit_id VARCHAR(32)",
        "visit_peak INTEGER",
        "clip_offset_seconds FLOAT",
        "clip_end_offset_seconds FLOAT",
        "clip_duration_seconds FLOAT",
        "clip_segments JSON",
        # NULL means nobody has grouped the row into an animal identity.
        "animal_id VARCHAR(64)",
        "animal_source VARCHAR(8)",
        # Preserve the boxes burned into accepted review videos.
        "track_boxes JSON",
        "track_box_fps FLOAT",
    )
)


async def get_session() -> AsyncSession:  # type: ignore[misc]
    """Dependency that yields an async DB session."""
    async with async_session() as session:
        yield session
