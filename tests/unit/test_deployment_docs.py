from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_production_env_template_keeps_required_release_keys() -> None:
    template = (ROOT / ".env.production.example").read_text(encoding="utf-8")

    required = [
        "KB_ENV=production",
        "KB_ACCESS_TOKEN",
        "KB_PRIVATE_INSTANCE_AUTH=0",
        "KB_ENABLE_AUTH_GATE=0",
        "KB_REQUIRE_AUTH=0",
        "KB_ENABLE_INTERNAL_API=0",
        "KB_AUTH_COOKIE_SECURE",
        "KB_API_ALLOW_ORIGINS",
        "KB_API_JSON_MAX_BODY_BYTES=1048576",
        "KB_USER_ISSUES_MAX_BODY_BYTES=65536",
        "KB_USER_ISSUES_LOCAL_RATE_LIMIT_PER_MIN=180",
        "KB_USER_ISSUES_INGEST_RATE_LIMIT_PER_MIN=600",
        "KB_STARTUP_PREFLIGHT=1",
        "KB_STARTUP_STRICT=0",
        "KB_DB_DIR",
        "KB_CHAT_DB",
        "KB_LIBRARY_DB",
        "KB_BACKUP_DIR",
        "KB_DIAGNOSTICS_DIR",
        "KB_AUTO_BACKUP=1",
        "KB_AUTO_BACKUP_MIN_INTERVAL_S",
        "KB_AUTO_BACKUP_STRICT",
        "KB_BACKUP_KEEP_N",
        "KB_RESTORE_AUDIT_PATH",
        "KB_UPDATE_CHECK_ENABLED=1",
        "KB_UPDATE_REPO=LittlePyx/Pi_zaya",
        "KB_UPDATE_GITHUB_TOKEN",
        "DEEPSEEK_API_KEY",
        "QWEN_API_KEY",
        "KB_PDF_FIGURE_DPI=320",
    ]
    for needle in required:
        assert needle in template

    assert "streamlit run" not in template.lower()


def test_deployment_runbook_documents_readiness_and_fastapi_react_path() -> None:
    runbook = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")
    lower = runbook.lower()

    required = [
        "fastapi + react",
        "python server.py",
        "npm run build",
        "/api/health",
        "/api/readiness",
        "tools\\check_production_readiness.py",
        "/api/maintenance/backups",
        "/api/maintenance/diagnostics/export",
        "restore dry-run",
        "KB_ACCESS_TOKEN",
        "KB_ACCESS_TOKEN_SHA256",
        "ordinary users should not need an access token",
        "KB_PRIVATE_INSTANCE_AUTH=0",
        "KB_ENABLE_AUTH_GATE=0",
        "KB_REQUIRE_AUTH=0",
        "KB_ENABLE_INTERNAL_API",
        "KB_API_ALLOW_ORIGINS",
        "KB_API_JSON_MAX_BODY_BYTES",
        "KB_USER_ISSUES_MAX_BODY_BYTES",
        "KB_USER_ISSUES_LOCAL_RATE_LIMIT_PER_MIN",
        "KB_USER_ISSUES_INGEST_RATE_LIMIT_PER_MIN",
        "KB_STARTUP_PREFLIGHT",
        "KB_STARTUP_STRICT",
        "KB_BACKUP_DIR",
        "KB_DIAGNOSTICS_DIR",
        "KB_AUTO_BACKUP",
        "KB_AUTO_BACKUP_STRICT",
        "KB_BACKUP_KEEP_N",
        "KB_RESTORE_AUDIT_PATH",
        "KB_UPDATE_CHECK_ENABLED",
        "KB_UPDATE_REPO",
        "KB_UPDATE_GITHUB_TOKEN",
        "/api/app/update-check",
        "src",
    ]
    for needle in required:
        assert needle.lower() in lower

    assert "do not use `app.py`, `streamlit run`, or port `8501`" in lower


def test_readme_keeps_public_deployment_token_free_by_default() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lower = readme.lower()

    assert "`kb_private_instance_auth=0`" in lower
    assert "面向普通用户的公开部署保持 `kb_private_instance_auth=0`、`kb_enable_auth_gate=0` 和 `kb_require_auth=0`" in lower
    assert "用户打开应用不需要访问令牌" in readme
    assert "python tools\\check_production_readiness.py --base-url http://127.0.0.1:8000\n" in readme
    assert "--base-url http://127.0.0.1:8000 --token" not in lower
