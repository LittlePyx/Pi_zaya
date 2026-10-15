from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import re
from pathlib import Path
import socket
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from tools.research_qa import run_optimization_acceptance as acceptance
from tools.research_qa.run_research_qa_eval import (
    ResearchQaFixture,
    _get_json,
    _patch_json,
    _post_json,
    load_fixture,
    run_case,
)


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@contextmanager
def _backend(root: Path, product: Path, settings, label: str):
    acceptance.ROOT = product
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = acceptance._env(root, settings)
    env.update(
        {
            "KB_MD_DIR": str(root / "markdown"),
            "KB_APP_DATA_DIR": str(root),
            "KB_CROSSREF_BUDGET_S": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    python = product / "runtime/python/python.exe"
    command = [
        str(python),
        str(PROJECT / "tools/research_qa/run_optimization_acceptance.py"),
        "--serve",
        "--root",
        str(root),
        "--port",
        str(port),
    ]
    with (root / f"backend-{label}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=product,
            env=env,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("acceptance backend exited")
                try:
                    _get_json(url, "/api/health", timeout_s=2)
                    break
                except OSError:
                    time.sleep(0.25)
            else:
                raise TimeoutError("backend startup exceeded 60s")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _corpus_digest(root: Path) -> dict:
    paths = [
        root / "db/docs.json",
        *sorted((root / "db/chunks").glob("*.jsonl")),
        *sorted((root / "markdown").rglob("*.md")),
        *sorted((root / "pdfs").glob("*.pdf")),
    ]
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file()
    }


def _prepare(root: Path, product: Path, fixture, sources: Path, settings) -> None:
    import requests

    root.mkdir(parents=True, exist_ok=False)
    for name in ("db", "markdown", "pdfs"):
        (root / name).mkdir()
    (root / "user_prefs.json").write_text("{}", encoding="utf-8")
    events = []
    with _backend(root, product, settings, "prepare") as url:
        for doc in fixture.docs:
            pdf = sources / (doc["id"] + ".pdf")
            assert hashlib.sha256(pdf.read_bytes()).hexdigest() == doc["pdfSha256"]
            started = time.monotonic()
            with pdf.open("rb") as handle:
                response = requests.post(
                    url + "/api/library/upload",
                    files={"file": (pdf.name, handle, "application/pdf")},
                    data={"base_name": doc["id"]},
                    timeout=90,
                )
            response.raise_for_status()
            saved = response.json()
            assert saved["name"] == doc["id"] + ".pdf"
            result = _post_json(
                url,
                "/api/library/convert",
                {
                    "pdf_name": saved["name"],
                    "speed_mode": "no_llm",
                    "no_llm": True,
                    "replace": False,
                },
                timeout_s=30,
            )
            events.append(
                {
                    "id": doc["id"],
                    "upload": saved,
                    "convert": result,
                    "upload_enqueue_seconds": round(time.monotonic() - started, 3),
                }
            )
            _write(root / "prepare-events.json", events)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            files = _get_json(url, "/api/library/files?scope=all", timeout_s=20)
            ready = [
                item
                for item in files.get("items", [])
                if item.get("index_ready") and item.get("index_chunk_exists")
            ]
            _write(root / "library-files.json", files)
            print(
                f"PDF conversion/index: {len(ready)}/{len(fixture.docs)} ready",
                flush=True,
            )
            if len(ready) == len(fixture.docs):
                break
            time.sleep(2)
        else:
            raise TimeoutError(
                "PDF conversion/index did not become ready within 600s; inspect diagnostics"
            )
        _write(
            root / "onboarding-prepared.json",
            _get_json(url, "/api/app/onboarding-status", timeout_s=10),
        )
    _write(root / "corpus-fingerprint.json", _corpus_digest(root))


def _run(
    root: Path, product: Path, fixture, settings, freeze: dict, result_root: Path
) -> None:
    docs = []
    for doc in fixture.docs:
        assert (root / "markdown" / doc["id"] / (doc["id"] + ".en.md")).is_file()
        docs.append(doc)
    bound = ResearchQaFixture(
        db_root=str(root / "markdown"),
        docs=docs,
        cases=fixture.cases,
        forbidden_phrases=fixture.forbidden_phrases,
    )
    before = _corpus_digest(root)
    report = {
        "schema_version": 1,
        "freeze": freeze,
        "model": settings.text_model,
        "product_root": str(product),
        "limits": "Agent-authored independent questions; same sequential phase order BM25 then hybrid; API trace instrumentation; automatic contracts are not human faithfulness. Original PDF page review is separate.",
    }
    with _backend(root, product, settings, "run") as url:
        assert _post_json(
            url, "/api/settings/test-llm", {"target": "text"}, timeout_s=30
        ).get("ok")
        for mode in ("bm25", "hybrid_local"):
            output = result_root / mode
            output.mkdir(exist_ok=False)
            if mode == "hybrid_local":
                # The consent helper resolves sources from this real converted index.
                component = _get_json(
                    url, "/api/retrieval/local-component", timeout_s=10
                )
                if component.get("hybrid_local", {}).get("state") == "preparing":
                    deadline = time.monotonic() + 180
                    while component.get("hybrid_local", {}).get("state") == "preparing" and time.monotonic() < deadline:
                        time.sleep(0.25)
                        component = _get_json(url, "/api/retrieval/local-component", timeout_s=10)
                    assert component.get("hybrid_local", {}).get("state") == "ready", "existing component recovery did not become ready"
                if component.get("hybrid_local", {}).get("state") != "ready":
                    acceptance._enable_hybrid_ui(url, root)
                else:
                    acceptance._capture_hybrid_ready_ui(url, root)
            rows = []
            for case in fixture.cases:
                _patch_json(
                    url,
                    "/api/settings",
                    {
                        "ui_locale": case["expected"]["renderLocale"],
                        "refs_card_locale": case["expected"]["renderLocale"],
                    },
                    timeout_s=10,
                )
                try:
                    row = run_case(
                        base_url=url,
                        fixture=bound,
                        case=case,
                        timeout_s=90,
                        case_timeout_s=120,
                        top_k=6,
                        max_tokens=1200,
                        retrieval_mode=mode,
                    )
                except Exception as error:
                    row = {
                        "id": case["id"],
                        "quality": {"ok": False},
                        "error_type": type(error).__name__,
                    }
                rows.append(row)
                _write(output / "cases.json", rows)
                print(
                    f"{mode}: {case['id']} -> {'pass' if row.get('quality', {}).get('ok') else 'fail'}",
                    flush=True,
                )
            checks = []
            for row in rows:
                locate_output = output / row["id"]
                locate_output.mkdir()
                try:
                    checks.extend(acceptance._reader_checks(url, locate_output, [row]))
                except Exception as error:
                    checks.append(
                        {
                            "id": row["id"],
                            "ok": False,
                            "status": "browser_check_failed",
                            "error_type": type(error).__name__,
                        }
                    )
            _write(output / "reader_checks.json", checks)
            report[mode] = {
                "case_count": len(rows),
                "passed": sum(bool(row.get("quality", {}).get("ok")) for row in rows),
                "hybrid_used_count": sum(
                    row.get("final_payload", {})
                    .get("research_trace", {})
                    .get("retrieval", {})
                    .get("local_retrieval", {})
                    .get("used_mode")
                    == "hybrid_local"
                    for row in rows
                ),
                "reader_checks": checks,
                "cases": rows,
            }
            _write(result_root / "report.json", report)
    after = _corpus_digest(root)
    report["corpus_unchanged"] = before == after
    _write(result_root / "report.json", report)
    assert before == after, "converted corpus changed during paired evaluation"


def _repair(root: Path, product: Path, fixture, settings) -> None:
    with _backend(root, product, settings, "repair") as url:
        _write(
            root / "conversion-resume.json",
            _post_json(url, "/api/library/convert/resume-all", {}, timeout_s=20),
        )
        _wait_conversion_idle(url)
        result = _post_json(
            url,
            "/api/library/quality/repair",
            {
                "pdf_names": ["dncnn.pdf", "noise2self.pdf"],
                "no_llm": True,
                "speed_mode": "no_llm",
                "replace": False,
                "md_autofix": True,
            },
            timeout_s=120,
        )
        _write(root / "source-repair.json", result)
        _wait_conversion_idle(url)
        for doc in fixture.docs:
            response = _post_json(
                url,
                "/api/library/reindex/file",
                {"pdf_name": doc["id"] + ".pdf"},
                timeout_s=120,
            )
            _write(root / (doc["id"] + "-reindex.json"), response)
        files = _get_json(url, "/api/library/files?scope=all", timeout_s=20)
        _write(root / "library-files-repaired.json", files)
        ready = sum(bool(item.get("index_ready")) for item in files.get("items", []))
        assert ready == len(fixture.docs), (
            "source repair still blocks a paper; retain diagnostic gates"
        )
    _write(root / "corpus-fingerprint.json", _corpus_digest(root))


def _wait_conversion_idle(url: str) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        files = _get_json(url, "/api/library/files?scope=all", timeout_s=20)
        active = [
            item
            for item in files.get("items", [])
            if item.get("task_state") in {"queued", "running"}
        ]
        if not active:
            return
        print(
            "Waiting for conversion/repair:",
            [item["name"] for item in active],
            flush=True,
        )
        time.sleep(2)
    raise TimeoutError("conversion repair still active after 600s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "repair", "run"])
    parser.add_argument("--root", type=Path, default=Path("F:/pz-independent-20261013"))
    parser.add_argument("--product-root", type=Path, required=True)
    parser.add_argument(
        "--sources",
        type=Path,
        default=PROJECT / "test_results/independent_acceptance_20261013",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=PROJECT / "docs/research_qa_independent_denoising_v1.json",
    )
    parser.add_argument(
        "--round",
        default="",
        help="Separate follow-up result folder; keeps the original independent result.",
    )
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    from kb.config import load_settings

    settings = load_settings()
    fixture = load_fixture(args.fixture)
    freeze_file = args.sources / "freeze.json"
    current = {
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "case_count": len(fixture.cases),
        "sources": json.loads(
            (args.sources / "sources.json").read_text(encoding="utf-8")
        ),
    }
    if freeze_file.exists():
        assert json.loads(freeze_file.read_text(encoding="utf-8")) == current, (
            "frozen fixture/source mismatch"
        )
    else:
        _write(freeze_file, current)
    if args.action == "prepare":
        _prepare(
            args.root.resolve(),
            args.product_root.resolve(),
            fixture,
            args.sources,
            settings,
        )
    elif args.action == "repair":
        _repair(args.root.resolve(), args.product_root.resolve(), fixture, settings)
    else:
        if args.case_id:
            selected = [case for case in fixture.cases if case["id"] in args.case_id]
            assert len(selected) == len(set(args.case_id)), "unknown frozen case"
            fixture = ResearchQaFixture(
                db_root=fixture.db_root,
                docs=fixture.docs,
                cases=selected,
                forbidden_phrases=fixture.forbidden_phrases,
            )
        assert not args.round or re.fullmatch(r"[a-z0-9_-]+", args.round)
        result_root = (
            args.root.resolve() / args.round if args.round else args.root.resolve()
        )
        if args.round:
            result_root.mkdir(exist_ok=False)
        _run(
            args.root.resolve(),
            args.product_root.resolve(),
            fixture,
            settings,
            current,
            result_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
