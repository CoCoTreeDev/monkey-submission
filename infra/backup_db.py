"""Create a consistent local SQLite backup and keep the newest 30 copies.

Uses sqlite3's backup API so writes in progress cannot produce a torn snapshot.
Configure the database and destination paths below for the intended host.
These local copies do not protect against losing that host.
"""
import datetime as dt
import pathlib
import sqlite3

DB = pathlib.Path.home() / "monkey.db"
OUT = pathlib.Path.home() / "backups" / "monkey-db"
KEEP = 30

OUT.mkdir(parents=True, exist_ok=True)
stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
dest_path = OUT / f"monkey-{stamp}.db"

src = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
dest = sqlite3.connect(dest_path)
src.backup(dest)
rows = dest.execute("select count(*) from detections").fetchone()[0]
dest.close()
src.close()

old = sorted(OUT.glob("monkey-*.db"))[:-KEEP]
for p in old:
    p.unlink()
print(f"{dest_path.name}  {dest_path.stat().st_size/1e6:.2f} MB  detections={rows}  "
      f"kept={len(list(OUT.glob('monkey-*.db')))} pruned={len(old)}")
