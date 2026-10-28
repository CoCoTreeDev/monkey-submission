# Local validation

From the repository root, run:

```bash
make ci
```

The runner uses `uv` for Python and `npm` for the frontend. It runs backend
checks, verifies route registration, installs frontend dependencies from the
lockfile, and runs TypeScript, lint and a production build. Unit suites use
local fixtures or stubs; model accuracy evaluation and live-camera benchmarks
are separate commands requiring their own data and configuration.

Set `UV_BIN` if `uv` is not installed at `$HOME/.local/bin/uv`. Logs default
to `$HOME/logs/monkey-ci`; set `CI_LOG_DIR` to choose another writable directory.
The runner stops at the first failing phase and returns a nonzero status.

This runner does not commit, push, deploy services or alter a field database.
Optional deployment examples are documented in
[workflow-examples](../workflow-examples/README.md).
