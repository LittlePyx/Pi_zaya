# Pi-zaya Production Deployment Runbook

This runbook is for the FastAPI + React product path only. Do not use `app.py`, `streamlit run`, or port `8501` for production checks.

## 1. Preflight

Run these checks before a release:

```powershell
git status --short
python --version
node --version
```

Recommended version anchors are `.python-version` and `.nvmrc`.

Install dependencies:

```powershell
pip install -r requirements.txt
cd web
npm ci
npm run build
cd ..
```

`web/dist/index.html` must exist before using the single-service production entry.

## 2. Configure `.env`

Copy the template and edit secrets/paths:

```powershell
Copy-Item .env.production.example .env
notepad .env
```

Required production settings:

- `KB_ENV=production`
- `KB_PRIVATE_INSTANCE_AUTH=0`, `KB_ENABLE_AUTH_GATE=0`, `KB_REQUIRE_AUTH=0`, and `KB_ALLOW_LOCAL_AUTH_GATE=0` for public user-facing deployments; use `KB_PRIVATE_INSTANCE_AUTH=1`, `KB_ENABLE_AUTH_GATE=1`, and `KB_REQUIRE_AUTH=1` plus `KB_ACCESS_TOKEN` or `KB_ACCESS_TOKEN_SHA256` only for private/internal production instances
- `KB_ENABLE_INTERNAL_API=0` for public user-facing deployments. Maintenance, diagnostics, backup/restore, and quality-collector inspection APIs stay server-side hidden unless the instance is private/authenticated or a local developer explicitly enables them.
- `KB_API_ALLOW_ORIGINS` with explicit origins, not `*`
- `KB_API_JSON_MAX_BODY_BYTES=1048576` for ordinary JSON API requests; upload endpoints keep their route-specific file limits
- `KB_USER_ISSUES_LOCAL_RATE_LIMIT_PER_MIN=180` and `KB_USER_ISSUES_INGEST_RATE_LIMIT_PER_MIN=600` to keep quality-data collection from being spammed if an instance is exposed
- `KB_STARTUP_PREFLIGHT=1`
- `KB_DB_DIR`, `KB_CHAT_DB`, `KB_LIBRARY_DB`
- `KB_BACKUP_DIR`, `KB_DIAGNOSTICS_DIR`
- `KB_AUTO_BACKUP=1`
- `KB_UPDATE_CHECK_ENABLED=1` and `KB_UPDATE_REPO=LittlePyx/Pi_zaya` when users should see GitHub Release update reminders
- `KB_UPDATE_GITHUB_TOKEN` with read-only repository access when the deployment should avoid GitHub's anonymous API limit
- At least one text model key: `DEEPSEEK_API_KEY`, `QWEN_API_KEY`, or `OPENAI_API_KEY`
- A dedicated vision key, usually `QWEN_API_KEY`, for image/table/figure-heavy workflows

Ordinary users should not need an access token to open the app. API access protection is a developer-only opt-in: leave `KB_PRIVATE_INSTANCE_AUTH=0`, `KB_ENABLE_AUTH_GATE=0`, `KB_REQUIRE_AUTH=0`, and `KB_ALLOW_LOCAL_AUTH_GATE=0` for public deployments. Local `development` mode ignores the auth gate unless the private-instance marker and `KB_ALLOW_LOCAL_AUTH_GATE=1` are also set, so a cloned project is not locked by stray auth variables. Set `KB_PRIVATE_INSTANCE_AUTH=1`, `KB_ENABLE_AUTH_GATE=1`, and `KB_REQUIRE_AUTH=1` only when the whole production instance is meant to be private. For local HTTP testing of the private gate, additionally set `KB_ALLOW_LOCAL_AUTH_GATE=1` and keep `KB_AUTH_COOKIE_SECURE=0`. Behind HTTPS, set `KB_AUTH_COOKIE_SECURE=1`.

Internal maintenance APIs are separate from the public app path. In production, use them through a private/authenticated instance (`KB_PRIVATE_INSTANCE_AUTH=1`, `KB_ENABLE_AUTH_GATE=1`, and `KB_REQUIRE_AUTH=1`) and pass the access token. For local development only, you can set `KB_ENABLE_INTERNAL_API=1` while `KB_ENV` is not production.

Set `KB_STARTUP_STRICT=1` in deployment scripts when startup should stop immediately on blocking readiness errors.

Use GitHub Releases as the stable update source. Set `KB_APP_VERSION` to the shipped release tag, for example `v1.2.0`; the app checks `/api/app/update-check` quietly on a browser-side cooldown, the backend caches GitHub's latest release for `KB_UPDATE_CHECK_TTL_S`, and the Settings UI only shows the cached result unless the user explicitly checks again. For offline or private-network deployments, set `KB_UPDATE_CHECK_ENABLED=0`.

Optional remote quality telemetry:

- On your collector deployment, set `KB_USER_ISSUES_INGEST_TOKEN=<secret>` and expose `POST /api/user-issues/ingest`.
- On user deployments that should send quality data back, set `KB_USER_ISSUES_REMOTE_ENABLED=1`, `KB_USER_ISSUES_REMOTE_URL=https://your-host/api/user-issues/ingest`, and `KB_USER_ISSUES_REMOTE_TOKEN=<secret>`. The collector URL must be a full HTTPS URL without embedded `user:pass@host` credentials for real users; localhost HTTP is only accepted for local testing when `KB_USER_ISSUES_ALLOW_LOCAL_REMOTE=1`. Sender authentication is required by default; only private test collectors should set `KB_USER_ISSUES_ALLOW_UNAUTHENTICATED_REMOTE=1`.
- Set `KB_USER_ISSUES_CLIENT_ID=<stable-id>` only when you need per-install aggregation; it is hashed before upload.
- Keep `KB_USER_ISSUES_INGEST_RATE_LIMIT_PER_MIN` enabled on the collector. It limits bad token attempts and noisy clients before anything is written to `user_issues.sqlite3`.
- The reporter keeps the local `user_issues.sqlite3` behavior, sends in the background, caps each issue payload with `KB_USER_ISSUES_MAX_BODY_BYTES`, and redacts local paths, email addresses, tokens, and sensitive path/key fields before upload.

## 3. Start

Single-service local production mode:

```powershell
python server.py
```

In production mode, `server.py` prints a startup preflight summary. It does not stop the server unless `KB_STARTUP_STRICT=1`, so first-run operators can still open the UI and fix API keys or paths.

Or run the ASGI app directly:

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/`, then open Settings -> Connection -> Release readiness. Public user-facing deployments should not show an access-token gate.

## 4. Readiness Checks

Public liveness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Production readiness:

```powershell
python tools\check_production_readiness.py --base-url http://127.0.0.1:8000
```

Use JSON output when wiring this into deployment scripts:

```powershell
python tools\check_production_readiness.py --json --base-url http://127.0.0.1:8000
```

When a private production instance has `KB_PRIVATE_INSTANCE_AUTH=1`, `KB_ENABLE_AUTH_GATE=1`, and `KB_REQUIRE_AUTH=1`, add `--token $env:KB_ACCESS_TOKEN` to the readiness command.

Exit codes:

- `0`: ready
- `1`: warning, unless `--allow-warning` is set
- `2`: blocking error or readiness request failed

## 5. Smoke Test

Before announcing a public user-facing release:

1. Open the app and confirm no access-token gate appears.
2. Open Settings -> Version & updates and confirm update checks behave as expected.
3. Open Settings -> Connection and confirm release readiness is `OK`.
4. Upload or select a small known PDF.
5. Run conversion and update the knowledge base.
6. Ask one short question and verify `src` citation chips persist after refresh.
7. Open Reader from the answer and verify references, figures, tables, and basket actions still work.
8. For private/internal production deployments only, set `KB_PRIVATE_INSTANCE_AUTH=1`, `KB_ENABLE_AUTH_GATE=1`, and `KB_REQUIRE_AUTH=1`, build the frontend with `VITE_PRIVATE_INSTANCE_AUTH=1` and `VITE_ENABLE_AUTH_GATE=1`, restart, and confirm the login gate accepts `KB_ACCESS_TOKEN`.

## 6. Backup and Restore

Backup, restore, diagnostics, and quality-collector inspection are internal maintenance operations. Public user-facing deployments keep these APIs hidden. Use one of these modes before running the commands below:

- Private/internal production: set `KB_PRIVATE_INSTANCE_AUTH=1`, `KB_ENABLE_AUTH_GATE=1`, and `KB_REQUIRE_AUTH=1`, configure `KB_ACCESS_TOKEN`, restart, and pass `X-KB-Access-Token`.
- Local developer diagnostics: set `KB_ENABLE_INTERNAL_API=1` while `KB_ENV` is not production, then restart.

Manual backup API:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/maintenance/backups `
  -Headers @{ "X-KB-Access-Token" = $env:KB_ACCESS_TOKEN } `
  -Body '{"label":"before-upgrade"}' `
  -ContentType 'application/json'
```

Diagnostic package:

```powershell
Invoke-WebRequest `
  -Uri http://127.0.0.1:8000/api/maintenance/diagnostics/export `
  -Headers @{ "X-KB-Access-Token" = $env:KB_ACCESS_TOKEN } `
  -OutFile diagnostics.zip
```

Backups are written to `KB_BACKUP_DIR`. Diagnostics are written to `KB_DIAGNOSTICS_DIR`.
The diagnostic package is designed for support: it includes readiness, counts, environment summary and redacted log tails, but not chat rows, chunks, PDFs, or API keys.

Automatic snapshots:

- In production, high-risk operations create a backup before running by default.
- Covered operations include library file deletion, replace conversion, quality repair, figure-asset refresh, manual or repair-flow reindex, batch metadata updates, and chat/project/message/citation-shelf deletion.
- If `KB_AUTO_BACKUP` is not explicitly set, the local React Settings -> Advanced switch can control automatic snapshots.
- If `KB_AUTO_BACKUP` is set, deployment configuration is authoritative and the UI switch is locked.
- `KB_AUTO_BACKUP=0` disables automatic snapshots.
- `KB_AUTO_BACKUP_MIN_INTERVAL_S=30` rate-limits repeated snapshots for the same operation type.
- `KB_AUTO_BACKUP_STRICT=1` blocks the operation when snapshot creation fails.
- `KB_BACKUP_KEEP_N=30` is the default retention count used by backup cleanup.

Use the maintenance API to verify a backup archive before relying on it, and to clean up old backup archives after confirming the newest backups verify successfully.
Before restoring data, run the restore dry-run through the maintenance API. It extracts the backup to a temporary directory, checks SQLite integrity, lists the configured restore targets, and reports blocking errors without overwriting current files.
The actual restore action requires typing `RESTORE <backup-file-name>`. The server creates a pre-restore backup first, writes an audit record to `KB_RESTORE_AUDIT_PATH` or `.runtime/restore_audit.jsonl`, restores the selected data files, and then requires a server restart before continuing.

Back up these paths before upgrades:

- `.env`
- `chat.sqlite3` or the configured `KB_CHAT_DB`
- `library.sqlite3` or the configured `KB_LIBRARY_DB`
- `db/` or the configured `KB_DB_DIR`
- `KB_BACKUP_DIR`
- `user_prefs.json` when users configured model/path settings in the UI

Restore by stopping the server, copying the saved files back, then running the readiness check again.

## 7. Upgrade Flow

```powershell
git pull --ff-only
pip install -r requirements.txt
cd web
npm ci
npm run build
cd ..
python -m pytest tests\unit\test_check_production_readiness.py tests\sanity\test_api_auth_guard.py -q
python server.py
```

After restart, run:

```powershell
python tools\check_production_readiness.py --base-url http://127.0.0.1:8000
```

If API access protection is enabled in a private production instance with all auth-gate switches, append `--token $env:KB_ACCESS_TOKEN`.

## 8. Troubleshooting

`401 Unauthorized` on `/api/readiness`: the production instance has `KB_PRIVATE_INSTANCE_AUTH=1`, `KB_ENABLE_AUTH_GATE=1`, and `KB_REQUIRE_AUTH=1`; pass `--token`, set `KB_ACCESS_TOKEN`, or log in through the private UI build.

Unexpected login gate for users: set `KB_PRIVATE_INSTANCE_AUTH=0`, `KB_ENABLE_AUTH_GATE=0`, `KB_REQUIRE_AUTH=0`, `KB_ALLOW_LOCAL_AUTH_GATE=0`, `VITE_PRIVATE_INSTANCE_AUTH=0`, and `VITE_ENABLE_AUTH_GATE=0`, rebuild/restart, and keep `KB_ACCESS_TOKEN` only for private/internal instances. In local development, restart with `.\run_new.ps1 -StopExisting`; it clears the auth gate for the process by default.

`404 Not Found` on `/api/maintenance/*` or `/api/user-issues/remote/status`: those internal maintenance APIs are hidden. Use a private/authenticated instance, or set `KB_ENABLE_INTERNAL_API=1` for local non-production diagnostics.

`frontend_build` blocking readiness: run `cd web; npm run build; cd ..`.

`cors` blocking readiness: replace `KB_API_ALLOW_ORIGINS=*` with explicit app origins.

Login works on localhost but not HTTPS: set `KB_AUTH_COOKIE_SECURE=1` behind HTTPS and make sure the proxy preserves cookies.

Login fails on local HTTP: set `KB_AUTH_COOKIE_SECURE=0` while testing without HTTPS.

Model readiness is missing or failed: configure the key/base URL/model in `.env` or in Settings, then use the Settings connection test before rerunning readiness.
