"""Pull datasets and model weights from the Ultralytics Platform API.

Auth: ULTRALYTICS_API_KEY in the environment or backend/.env (ul_ + 40 hex).

  # list what's available
  uv run --project backend python tests/fetch_platform.py --list

  # download a dataset's NDJSON export (to repo root by default)
  uv run --project backend python tests/fetch_platform.py --dataset "monkey cocotree"

  # list a project's models with metrics / download one's weights
  uv run --project backend python tests/fetch_platform.py --models monkey_v2
  uv run --project backend python tests/fetch_platform.py --weights <modelId> -o candidate.pt

Built for the ratchet workflow: pull the fresh export + candidate weights,
then gate with tests/eval_ratchet.py before any model swap.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import httpx
from ultralytics_platform import APIError, Platform

REPO = Path(__file__).resolve().parent.parent


def _key() -> str:
    key = os.getenv("ULTRALYTICS_API_KEY", "")
    if not key:
        env = REPO / "backend" / ".env"
        if env.exists():
            m = re.search(r"^ULTRALYTICS_API_KEY=(\S+)", env.read_text(), re.M)
            key = m.group(1) if m else ""
    if not re.fullmatch(r"ul_[0-9a-f]{40}", key):
        sys.exit("ULTRALYTICS_API_KEY missing or malformed (expect ul_ + 40 hex)")
    return key


_sdk: list = []


def api() -> "Platform":
    """The shared SDK client. Owns URL assembly, auth, typed responses, retries."""
    if not _sdk:
        _sdk.append(Platform(api_key=_key(), timeout=120))
    return _sdk[0]


def _id(x: dict) -> str:
    return x.get("id") or x.get("_id") or ""


# The 2026-08 DataSys refactor: everything is addressed by {owner}/{url-name}.
# The unscoped list endpoints are POST-only now and answer 405 with an EMPTY
# body; id-based sub-resource paths 404. The URL name lives in a field named
# after the resource (`dataset`/`project`/`model`), never `slug`.
_URL_KEY = {"datasets": "dataset", "projects": "project", "models": "model"}
_LIST_LIMIT = {"datasets": 1000, "projects": 500, "models": 100}
_owner_cache: list[str] = []


def _owner() -> str:
    if not _owner_cache:
        me = api().account.retrieve_summary()
        name = (me or {}).get("username") or ""
        if not name:
            sys.exit("could not read account username from account/summary")
        _owner_cache.append(str(name))
    return _owner_cache[0]


def _slug(rec: dict, kind: str) -> str:
    return str(rec.get(_URL_KEY.get(kind, "slug")) or rec.get("slug") or "")


def _ref(rec: dict, kind: str) -> tuple[str, ...]:
    """The positional args every SDK sub-resource call takes: (owner, url-name),
    and (owner, project, model) for models, which nest under their project."""
    owner, slug = rec.get("owner") or "", _slug(rec, kind)
    mid = str(rec.get("project") or "") if kind == "models" else ""
    if not owner or not slug or (kind == "models" and not mid):
        sys.exit(f"{kind} record is missing owner/{_URL_KEY.get(kind)}: {str(rec)[:200]}")
    return (owner, mid, slug) if mid else (owner, slug)


def _list(kind: str, **params) -> list[dict]:
    params.setdefault("limit", _LIST_LIMIT.get(kind, 100))
    res = {"datasets": api().datasets, "projects": api().projects}[kind].list(
        _owner(), **params
    )
    return res.get(kind) or []


def _resolve_model(name_or_id: str, project: str | None = None) -> dict:
    """Find a model record (owner + project + model) from an id/name/slug.

    A bare model id cannot be turned into a URL any more, so the owner's
    projects are searched until it turns up."""
    projects = [_resolve("projects", project)] if project else _list("projects")
    for proj in projects:
        listing = api().models.list(*_ref(proj, "projects"), limit=_LIST_LIMIT["models"])
        for m in (listing.get("models") or []):
            if name_or_id in (_id(m), m.get("name"), _slug(m, "models")):
                m.setdefault("owner", proj.get("owner"))
                m.setdefault("project", _slug(proj, "projects"))
                return m
    sys.exit(f"model '{name_or_id}' not found in any project")


def _resolve(kind: str, name_or_id: str, **params) -> dict:
    """Match on display name, URL name or id — the FULL record is needed now,
    because an id alone cannot be turned into a URL."""
    for x in _list(kind, **params):
        if name_or_id in (x.get("name"), _slug(x, kind), x.get("slug"), _id(x)):
            return x
    sys.exit(f"{kind[:-1]} '{name_or_id}' not found")


def _download(url: str, dest: Path) -> None:
    # ml/models/ holds only gitignored artifacts, so a fresh checkout has no
    # directory — create it instead of failing on the first CI download.
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    print(f"→ {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list datasets and projects")
    ap.add_argument("--dataset", help="dataset name or id — download NDJSON export")
    ap.add_argument("--models", help="project name or id — list its models")
    ap.add_argument("--weights", help="model id — download its .pt weights")
    ap.add_argument("--train", nargs=2, metavar=("PROJECT", "DATASET"),
                    help="start a cloud training run: project name/id + dataset name/id")
    ap.add_argument("--base", default="yolo26n.pt", help="base model for --train")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--gpu", default="rtx-6000-ada",
                    help="gpuType for --train (employee credits cover RTX 6000)")
    ap.add_argument("--name", help="model name for --train")
    ap.add_argument("--model-id", help="reuse an existing model record for --train")
    ap.add_argument("--dry-run", action="store_true", help="print the training request without sending")
    ap.add_argument("--train-status", metavar="MODEL_ID", help="poll a training job")
    ap.add_argument("--upload", metavar="DIR", help="upload every image in DIR to --to dataset")
    ap.add_argument("--upload-labeled", metavar="DIR",
                    help="upload a YOLO layout (images/ + labels/ + data.yaml) WITH boxes")
    ap.add_argument("--split", default="train", choices=("train", "val", "test"),
                    help="target split for --upload-labeled")
    ap.add_argument("--limit", type=int, default=0,
                    help="upload only the first N images (probe runs)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N images — upload the rest after a probe")
    ap.add_argument("--create-dataset", metavar="SLUG", help="create a new empty dataset")
    ap.add_argument("--dataset-name", help="display name for --create-dataset")
    ap.add_argument("--dataset-description", help="description for --create-dataset")
    ap.add_argument("--to", help="target dataset name/id for --upload / --upload-labeled")
    ap.add_argument("-o", "--out", help="output path")
    args = ap.parse_args()

    if args.list:
        print("== datasets ==")
        for x in _list("datasets"):
            print(f"  {_id(x)}  {x.get('name')!r}  [{_slug(x, 'datasets')}]  "
                  f"({x.get('imageCount', '?')} imgs)")
        print("== projects ==")
        for x in _list("projects"):
            print(f"  {_id(x)}  {x.get('name')!r}  [{_slug(x, 'projects')}]")

    if args.dataset:
        ds = _resolve("datasets", args.dataset)
        exp = api().datasets.retrieve_export(*_ref(ds, "datasets"))
        name = (ds.get("name") or ds["id"]).replace(" ", "-")
        dest = Path(args.out) if args.out else REPO / f"{name}.ndjson"
        print(f"export cached={exp.get('cached')}")
        _download(exp["downloadUrl"], dest)

    if args.models:
        proj = _resolve("projects", args.models)
        listing = api().models.list(*_ref(proj, "projects"), limit=_LIST_LIMIT["models"])
        for m in (listing.get("models") or []):
            met = m.get("metrics") or {}
            # the metric key is "mAP50", not "map50"
            print(f"  {_id(m)}  {m.get('name')!r}  [{_slug(m, 'models')}]  "
                  f"P={met.get('precision', '?')} R={met.get('recall', '?')} "
                  f"mAP50={met.get('mAP50', met.get('map50', '?'))}")

    if args.weights:
        mrec = _resolve_model(args.weights)
        files = api().models.retrieve_files(*_ref(mrec, "models"))
        cand = files if isinstance(files, list) else (files.get("files") or [files])
        pt = next((f for f in cand if str(f.get("name", f.get("format", ""))).endswith(".pt")
                   or f.get("format") in ("pt", "pytorch")), cand[0] if cand else None)
        if not pt or not pt.get("downloadUrl", pt.get("url")):
            sys.exit(f"no downloadable weights in response: {str(files)[:300]}")
        dest = Path(args.out) if args.out else REPO / "ml" / "models" / "candidate.pt"
        _download(pt.get("downloadUrl") or pt.get("url"), dest)

    if args.upload:
        if not args.to:
            sys.exit("--upload needs --to <dataset>")
        upload_dir(Path(args.upload), _resolve("datasets", args.to))

    if args.create_dataset:
        rec = api().datasets.create(
            dataset=args.create_dataset,
            name=args.dataset_name or args.create_dataset,
            description=args.dataset_description or "",
        )
        rec = rec if isinstance(rec, dict) else rec.model_dump()
        inner = rec.get("dataset") if isinstance(rec.get("dataset"), dict) else rec
        print(f"created dataset {_id(inner)}  {inner.get('name')!r}  [{inner.get('slug')}]")

    if args.upload_labeled:
        if not args.to:
            sys.exit("--upload-labeled needs --to <dataset>")
        upload_yolo_dir(Path(args.upload_labeled), _resolve("datasets", args.to),
                        args.split, limit=args.limit, skip=args.skip)

    if args.train or args.train_status:
        handle_training(args)


def upload_yolo_dir(src: Path, ds: dict, split: str, limit: int = 0, skip: int = 0) -> None:
    """Upload a YOLO-layout directory (images/ + labels/ + data.yaml) WITH boxes.

    upload_dir() sends bare images, which arrive unannotated — fine for
    background negatives, useless for pre-labels. Ingest does read a YOLO
    archive: verified 2026-08-13 against the `probe yolo format` dataset, where
    both probe images came back out of the export carrying their boxes and the
    requested split.

    Same two footguns as upload_dir: ingest only processes ARCHIVES (loose
    sessions queue forever, 2026-07-31), and a non-ASCII entry name silently
    kills the job — so callers must have written ASCII filenames already; this
    refuses rather than mangling the pairing between an image and its label.
    """
    import io
    import time
    import zipfile

    imgs = sorted(p for p in (src / "images").iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not imgs:
        sys.exit(f"no images in {src / 'images'}")
    if skip:
        imgs = imgs[skip:]
    if limit:
        imgs = imgs[:limit]

    bad = [p.name for p in imgs if not p.name.isascii()]
    if bad:
        sys.exit(f"non-ASCII image names would kill the ingest job: {bad[:3]}")

    buf = io.BytesIO()
    labelled = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for p in imgs:
            zf.writestr(f"images/{p.name}", p.read_bytes())
            lbl = src / "labels" / (p.stem + ".txt")
            if lbl.exists():
                zf.writestr(f"labels/{lbl.name}", lbl.read_bytes())
                if lbl.read_text().strip():
                    labelled += 1
        yaml = src / "data.yaml"
        if yaml.exists():
            zf.writestr("data.yaml", yaml.read_bytes())
    blob = buf.getvalue()

    print(f"uploading {len(imgs)} images ({labelled} with boxes) as one zip "
          f"({len(blob)/1e6:.1f} MB) -> {ds.get('name') or _id(ds)} [{split}]")
    meta = api().upload.retrieve_file_url(
        asset_type="datasets", asset_id=_id(ds), filename="cli_upload_yolo.zip",
        content_type="application/zip", total_bytes=len(blob),
    )
    r = httpx.put(meta["uploadUrl"], content=blob,
                  headers={
                      "Content-Type": "application/zip",
                      # GCS signs this precondition into the URL; omitting the
                      # matching header returns 400 MalformedSecurityHeader.
                      "x-goog-if-generation-match": "0",
                  }, timeout=600)
    r.raise_for_status()
    api().upload.complete(session_id=meta["sessionId"])
    for attempt in range(20):
        try:
            api().datasets.ingest(*_ref(ds, "datasets"),
                                  session_id=meta["sessionId"], target_split=split)
            break
        except APIError as exc:
            if exc.status_code != 409:      # 409 = another ingest still running
                raise
            time.sleep(15)
    else:
        sys.exit("ingest kept returning 409 — another job is still running")
    print("ingest queued (async — poll imageCount to confirm it landed)")


def upload_dir(src: Path, ds: dict) -> None:
    """Upload every jpg/png in src via signed-url -> PUT -> complete -> ingest.

    Verified 2026-07-30 end to end (PUT 200, complete success, ingest queued).
    Images arrive UNANNOTATED — a no-label image is a background/negative once
    accepted; positives get boxes in the platform UI. Ingest is async and the
    job-status endpoint is not exposed, so confirm via imageCount later.
    """
    imgs = sorted(q for q in src.iterdir()
                  if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not imgs:
        sys.exit(f"no images in {src}")
    ds_id = _id(ds)
    # Ingest only processes ARCHIVES — loose image sessions queue forever
    # (verified 2026-07-31). Bundle everything into one ZIP: one session,
    # one ingest job, no 409 serialisation.
    import io
    import zipfile
    import zlib
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for q in imgs:
            # non-ASCII entry names silently kill the ingest job (verified
            # 2026-07-31) — sanitize with a stable per-name crc prefix
            name = q.name.encode("ascii", "ignore").decode().lstrip("_")
            if name != q.name:
                name = f"cam{zlib.crc32(q.name.encode()) & 0xFFFFFF:06x}_{name or 'image.jpg'}"
            zf.writestr(name, q.read_bytes())
    blob = buf.getvalue()
    print(f"uploading {len(imgs)} images as one zip ({len(blob)/1e6:.1f} MB) -> {ds.get('name') or ds_id}")
    meta = api().upload.retrieve_file_url(
        asset_type="datasets", asset_id=ds_id, filename="cli_upload.zip",
        content_type="application/zip", total_bytes=len(blob),
    )
    # raw httpx on purpose: the signed URL is its own credential and must NOT
    # carry the API key, and the SDK does not PUT the bytes for us
    r = httpx.put(meta["uploadUrl"], content=blob,
                  headers={
                      "Content-Type": "application/zip",
                      # GCS signs this precondition into the URL.
                      "x-goog-if-generation-match": "0",
                  }, timeout=300)
    r.raise_for_status()
    api().upload.complete(session_id=meta["sessionId"])
    sessions = [meta["sessionId"]]
    import time
    for sid in sessions:
        # 409 = an ingest job is already running for this dataset; wait and retry
        for attempt in range(20):
            try:
                api().datasets.ingest(*_ref(ds, "datasets"), session_id=sid)
                break
            except APIError as exc:
                # 409 = another ingest is running on this dataset. The SDK only
                # auto-retries idempotent verbs and ingest is a POST.
                if exc.status_code != 409:
                    raise
                time.sleep(15)
        else:
            print(f"  ⚠ session {sid[:8]}… 排入失敗（多次 409）")
    print(f"ingest queued for {len(sessions)} files (async — check imageCount later)")


def handle_training(args) -> None:
    if args.train_status:
        st = api().models.retrieve_training(*_ref(_resolve_model(args.train_status), "models"))
        print(__import__("json").dumps(st, ensure_ascii=False, indent=1)[:1500])
    if args.train:
        proj = _resolve("projects", args.train[0])
        ds = _resolve("datasets", args.train[1])
        ds_owner, ds_slug = ds.get("owner"), _slug(ds, "datasets")
        if not (ds_owner and ds_slug):
            sys.exit(f"dataset record lacks owner/dataset: {str(ds)[:200]}")
        data_uri = f"ul://{ds_owner}/datasets/{ds_slug}"
        body = {
            "gpuType": args.gpu,
            "trainArgs": {
                "model": args.base,
                "data": data_uri,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
            },
        }
        if args.dry_run:
            print("DRY RUN — POST /api/models + /api/training/start")
            print(__import__("json").dumps(body, ensure_ascii=False, indent=1))
            return
        # training/start requires a model record to exist first
        if args.model_id:
            model_id = args.model_id
        else:
            # POST /models identifies the project by its URL NAME, not id
            created = api().models.create(body={
                "owner": proj.get("owner"),
                "project": _slug(proj, "projects"),
                "model": args.name or "untitled",
            })
            model_id = created.get("modelId") or _id(created)
            print(f"model record: {model_id} ({created.get('model') or args.name})")
        # gpu_type is a typed enum in the SDK — it rejects e.g. "rtx-6000"
        # (the real name is "rtx-6000-ada") before any credits are spent.
        out = api().training.start(
            model_id=model_id, gpu_type=args.gpu, train_args=body["trainArgs"]
        )
        print(__import__("json").dumps(out, ensure_ascii=False, indent=1)[:800])
        print(f"追蹤進度: --train-status {model_id}")


if __name__ == "__main__":
    _args = sys.argv
    main()
