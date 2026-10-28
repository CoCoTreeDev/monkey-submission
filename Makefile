ROOT := $(shell pwd)
# Local development database.
LOCAL_DB := sqlite+aiosqlite:///detections.db

.PHONY: help dev backend frontend install ci

help:
	@echo "Monkey Detection — local dev"
	@echo ""
	@echo "  make dev        Run backend (:8000) + frontend (:3000) together (Ctrl+C stops both)"
	@echo "  make backend    Run only the backend (FastAPI/uvicorn, local SQLite)"
	@echo "  make frontend   Run only the frontend (Next.js dev)"
	@echo "  make install    Install backend (uv) + frontend (npm) dependencies"
	@echo "  make ci         Run the local validation suite"

dev:
	@echo "==> Starting backend on http://localhost:8000 + frontend on http://localhost:3000"
	@cd $(ROOT)/backend && DATABASE_URL="$(LOCAL_DB)" uv run uvicorn main:app --host 0.0.0.0 --port 8000 & \
	BACKEND_PID=$$!; \
	trap 'echo; echo "==> Stopping backend ($$BACKEND_PID)"; kill $$BACKEND_PID 2>/dev/null || true' EXIT INT TERM; \
	cd $(ROOT)/frontend && npm run dev

backend:
	cd $(ROOT)/backend && DATABASE_URL="$(LOCAL_DB)" uv run uvicorn main:app --host 0.0.0.0 --port 8000

frontend:
	cd $(ROOT)/frontend && npm run dev

install:
	cd $(ROOT)/backend && uv sync
	cd $(ROOT)/frontend && npm install

ci:
	bash $(ROOT)/infra/ci/run.sh
