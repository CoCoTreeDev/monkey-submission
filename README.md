# CoCoTree Monkey Monitoring

**English** · [繁體中文](README.zh-TW.md)

A working Taiwanese macaque monitoring prototype built with FastAPI, SQLite,
YOLO and ByteTrack, with a Traditional Chinese Next.js/React dashboard.
Camera workers detect and track macaques, then save event records with images,
video clips and track timelines for review.

## Included capabilities

- Camera management, live viewing and per-camera tracking.
- Event browsing, saved clip playback, detection review and visibility controls.
- Local media storage, with optional Azure Blob Storage configuration.
- Model training/evaluation integration and regression-check scripts.
- Local validation scripts and optional edge-service examples.

The included detector has two classes: `monkey=0` and `human=1`. Monitoring
requests class 0; the second class does not enable human tracking or alerts.
Track IDs belong to a camera's tracking session and do not establish animal
identity across cameras. Saved visits are not a population estimate.

5G/UGen300 deployment, warning-zone rules and LINE group routing are planned
competition work. The current webhook integration is separate from that plan.

## Run locally

Use Python 3.12 or newer, `uv`, Node.js 20.9 or newer, and `npm`. Run commands
from the repository root.

```bash
make install
```

Create a private configuration file before the first regular run. This command
writes a persistent signing/hash secret and random initial access codes. It
refuses to overwrite an existing `.env` file.

```bash
python3 - <<'PYCONFIG'
from pathlib import Path
import os
import secrets

path = Path("backend/.env")
text = Path("backend/.env.example").read_text()
values = {
    "ADMIN_PASSWORD_SALT": secrets.token_hex(32),
    "SEED_ADMIN_CODE": secrets.token_urlsafe(18),
    "SEED_USER_CODE": secrets.token_urlsafe(18),
}
lines = [line for line in text.splitlines()
         if not any(line.startswith(key + "=") for key in values)]
lines += [f"{key}={value}" for key, value in values.items()]
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as stream:
    stream.write("\n".join(lines) + "\n")
print("Created backend/.env. Read its SEED_* values locally to sign in.")
PYCONFIG
```

Keep `ADMIN_PASSWORD_SALT` stable after creating users: it signs sessions and
hashes stored access codes. Seed codes are used only while the user table is
empty. The fallback `demo-admin` / `demo-user` codes and demo salt are for local
demonstrations only; set your own configuration before regular use.

```bash
make dev
```

Open the dashboard at `http://localhost:3000`; the API and interactive API docs
are at `http://localhost:8000` and `http://localhost:8000/docs`. Sign in with the
admin code in your local `.env`, then add cameras you are authorized to access.
No camera credentials or private field database are included.

`make backend` and `make frontend` start the services separately. The backend
uses a local SQLite database and `backend/snapshots/` by default. Set
`NEXT_PUBLIC_API_URL` for a dashboard served separately from the API.
[backend/config.py](backend/config.py) defines the environment options, with
common settings in [backend/.env.example](backend/.env.example).

## Model and evaluation

`backend/best.pt` contains the macaque detector. A missing checkpoint is a
startup error. Training and dataset commands require your own Platform account,
API key and authorized datasets; source field images are not distributed here.
See [dataset notes](ml/datasets/README.md) for the included metadata format.

The latest model was accepted with a documented test-gate exception: test F1
was **0.823166**, below the **0.8254** floor; validation F1 was **0.888889**.
The stronger stored baselines remain unchanged. These are recorded benchmark
results, not a new evaluation performed by this repository or a guarantee of
field accuracy. See [model and evaluation notes](ml/PROMOTIONS.md).

Human review and model evaluation are development steps. The running detector
does not automatically learn from new sightings. A candidate should be checked
on both frozen test and validation splits, and difficult regression cases,
before replacing the included checkpoint.

## Validation and optional deployment

```bash
make ci
```

The [local test runner](infra/ci/README.md) runs backend checks, TypeScript,
lint and a frontend production build. It does not deploy services.

GitHub Actions deployment and model-promotion automation are inactive in this
repository. [Workflow examples](infra/workflow-examples/README.md) are stored
outside the Actions directory and require explicit runner, credential and
service configuration. [Edge-service examples](infra/edge/README.md) describe
local startup; they do not synchronize or reset the checkout automatically.

## Layout

| Directory | Contents |
|---|---|
| `backend/` | API, camera workers, detector, tracking and storage |
| `frontend/` | Dashboard and review interface |
| `tests/` | Unit checks, regression cases and offline analysis tools |
| `ml/` | Dataset metadata and evaluation notes |
| `infra/` | Local validation and optional service examples |
| `docs/` | Video timing contract and exploratory tracking research |
