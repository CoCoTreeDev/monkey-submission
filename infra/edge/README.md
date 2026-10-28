# Optional edge services

Run the backend from this checkout with:

```bash
bash infra/edge/start-edge.sh
```

The launcher reads `backend/.env`, installs the locked backend dependencies and
starts the API on `127.0.0.1:8000`. It does not fetch code, reset a Git branch,
replace model files, or install a service. `MONKEY_REPO_DIR` and `UV_BIN` can
override the checkout and executable locations.

`fetch-models.sh` is a separate, explicit Azure model-download utility. It
requires your own connection string. `export-coreml.sh` is an optional local
export for configured ANE paths. Run either only when that integration is needed.

The `.plist` files are launchd templates. Before installing a copy, replace
`__REPO_DIR__`, `__LOG_DIR__` and `__UV_BIN__` with absolute paths on your own
host, create the log directory, and review the service command and schedule.
The backup example expects its database path to match `infra/backup_db.py`.
The canary requires its own API/access-code configuration in `backend/.env`.
No template is installed automatically.
