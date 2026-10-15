from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("KB_ACCEPTANCE_PRODUCT_ROOT") or PROJECT_ROOT).resolve()
sys.path.insert(0, str(ROOT))
if ROOT != PROJECT_ROOT:
    sys.path.append(str(PROJECT_ROOT))


def _rewrite(value, mapping: list[tuple[Path, Path]]):
    if isinstance(value, dict):
        return {_rewrite(key, mapping): _rewrite(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, mapping) for item in value]
    if isinstance(value, str):
        for old, new in mapping:
            value = value.replace(str(old), str(new)).replace(old.as_posix(), new.as_posix())
    return value


def _snapshot(root: Path, fixture, cases, *, reuse: bool = False) -> list[dict]:
    from tools.research_qa.run_research_qa_eval import source_path_for_doc

    source = Path(fixture.db_root).resolve()
    selected = {doc_id for case in cases for doc_id in case.get("docIds", [])}
    aliases = {fixture.docs_by_id[doc_id]["dir"]: doc_id for doc_id in selected}
    if not (reuse and (root / "db/docs.json").is_file()):
        root.mkdir(parents=True, exist_ok=False)
    else:
        return [{**doc, "dir": aliases.get(doc["dir"], doc["dir"]), "file": doc.get("file") or doc["dir"]} for doc in fixture.docs]
    db = root / "db"
    (db / "chunks").mkdir(parents=True)
    mapping = [(source / name, db / alias) for name, alias in aliases.items()]
    for name, alias in aliases.items():
        shutil.copytree(source / name, db / alias, ignore=shutil.ignore_patterns(".conversion_cache", "*.bak", "__pycache__"))
        for path in (db / alias).rglob("*.json"):
            try:
                data = _rewrite(json.loads(path.read_text(encoding="utf-8")), mapping)
            except ValueError:
                continue
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    docs = json.loads((source / "docs.json").read_text(encoding="utf-8"))
    published = {}
    allowed = {str(Path(source_path_for_doc(fixture, doc_id)).resolve()).casefold() for doc_id in selected}
    for doc_id, record in docs.items():
        if str(Path(record.get("path", "")).resolve()).casefold() not in allowed:
            continue
        published[doc_id] = _rewrite(record, mapping)
        raw = (source / "chunks" / f"{doc_id}.jsonl").read_text(encoding="utf-8")
        rows = [_rewrite(json.loads(line), mapping) for line in raw.splitlines() if line.strip()]
        (db / "chunks" / f"{doc_id}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (db / "docs.json").write_text(json.dumps(published, ensure_ascii=False), encoding="utf-8")
    for filename in ("references_index.json", "crossref_cache.json"):
        path = source / filename
        if path.is_file():
            value = _rewrite(json.loads(path.read_text(encoding="utf-8")), mapping)
            (db / filename).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    for name in ("pdfs", "markdown"):
        (root / name).mkdir()
    (root / "user_prefs.json").write_text("{}", encoding="utf-8")
    return [{**doc, "dir": aliases.get(doc["dir"], doc["dir"]), "file": doc.get("file") or doc["dir"]} for doc in fixture.docs]


def _env(root: Path, settings) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "KB_TEST_ISOLATED": "1", "KB_TEST_DATA_ROOT": str(root),
        "KB_DB_DIR": str(root / "db"), "KB_CHAT_DB": str(root / "chat.sqlite3"),
        "KB_LIBRARY_DB": str(root / "library.sqlite3"), "KB_USER_ISSUES_DB": str(root / "issues.sqlite3"),
        "KB_USER_PREFS_PATH": str(root / "user_prefs.json"),
        "KB_MODEL_CONNECTION_STATE_PATH": str(root / "connection_checks.json"),
        "KB_PDF_DIR": str(root / "pdfs"), "KB_MD_DIR": str(root / "db"),
        "KB_ACCEPTANCE_PRODUCT_ROOT": str(ROOT),
        "KB_USER_ISSUES_REMOTE_ENABLED": "0", "KB_UPDATE_CHECK_ENABLED": "0",
        "KB_ENABLE_AUTH_GATE": "0", "KB_REQUIRE_AUTH": "0", "KB_PRIVATE_INSTANCE_AUTH": "0",
        "KB_ENABLE_INTERNAL_API": "1", "KB_APP_ENV": "development",
        "LOCALAPPDATA": str(root / "profile/Local"),
        "KB_LOCAL_OBSERVABILITY_PATH": str(root / "observations.jsonl"),
        "DEEPSEEK_API_KEY": settings.text_api_key, "DEEPSEEK_BASE_URL": settings.text_base_url,
        "DEEPSEEK_MODEL": settings.text_model,
    })
    return env


def _serve(root: Path, port: int, baseline: bool) -> None:
    from kb.test_isolation import enforce_isolated_test_paths
    enforce_isolated_test_paths()
    from server import app
    if trace_path := os.environ.get("KB_ACCEPTANCE_ANSWER_TRACE"):
        from tools.research_qa.answer_chain_trace import install
        install(Path(trace_path))
    if baseline:
        import kb.claim_evidence_runtime as runtime
        import kb.evidence_binding as binding
        runtime.claim_polarity_conflicts = lambda _claim, _evidence: False
        runtime.has_checkable_action_fact = lambda _claim: False
        binding.claim_polarity_conflicts = lambda _claim, _evidence: False
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _phase(root: Path, fixture, cases, settings, baseline: bool, retrieval_mode: str | None = None, *, resume: bool = False) -> dict:
    from tools.research_qa.run_research_qa_eval import ResearchQaFixture, run_case, _post_json, _get_json, _patch_json

    docs = _snapshot(root, fixture, cases, reuse=resume)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    child_python = str(ROOT / "runtime/python/python.exe") if ROOT != PROJECT_ROOT else sys.executable
    command = [child_python, str(Path(__file__).resolve()), "--serve", "--root", str(root), "--port", str(port)]
    if baseline:
        command.append("--baseline")
    rebound = ResearchQaFixture(db_root=str(root / "db"), docs=docs, cases=fixture.cases, forbidden_phrases=fixture.forbidden_phrases, splits=fixture.splits, suites=fixture.suites)
    rows = []
    if resume and (root / "cases.json").is_file():
        cached = json.loads((root / "cases.json").read_text(encoding="utf-8"))
        if [row.get("id") for row in cached] == [case["id"] for case in cases] and all(row.get("quality", {}).get("ok") for row in cached):
            rows = cached
    reused_answers = bool(rows)
    with (root / "backend.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=_env(root, settings), stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("isolated acceptance backend exited; inspect backend.log")
                try:
                    with urllib.request.urlopen(url + "/api/health", timeout=2):
                        break
                except OSError:
                    time.sleep(0.25)
            else:
                raise TimeoutError("isolated acceptance backend startup exceeded 60 seconds")
            connection = _post_json(url, "/api/settings/test-llm", {"target": "text"}, timeout_s=30)
            if not connection.get("ok"):
                raise RuntimeError("configured text model connection check failed")
            before = _get_json(url, "/api/app/onboarding-status", timeout_s=10)
            component = _get_json(url, "/api/retrieval/local-component", timeout_s=10)
            if retrieval_mode == "hybrid_local" and component.get("hybrid_local", {}).get("state") != "ready":
                _enable_hybrid_ui(url, root)
                component = _get_json(url, "/api/retrieval/local-component", timeout_s=10)
                if component.get("hybrid_local", {}).get("state") != "ready":
                    raise RuntimeError("actual component installation/index did not become ready")
            elif retrieval_mode == "hybrid_local":
                _capture_hybrid_ready_ui(url, root)
            for case in ([] if reused_answers else cases):
                locale = case.get("expected", {}).get("renderLocale")
                if locale in {"zh", "en"}:
                    _patch_json(url, "/api/settings", {"ui_locale": locale}, timeout_s=10)
                try:
                    row = run_case(base_url=url, fixture=rebound, case=case, timeout_s=90, case_timeout_s=120, top_k=6, max_tokens=1800, retrieval_mode=retrieval_mode)
                except Exception as error:
                    row = {"id": case["id"], "quality": {"ok": False}, "error_type": type(error).__name__}
                rows.append(row)
                (root / "cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"{root.name}: {case['id']} -> {'pass' if row.get('quality', {}).get('ok') else 'fail'}", flush=True)
            after_answers = _get_json(url, "/api/app/onboarding-status", timeout_s=10)
            reader_checks = _reader_checks(url, root, rows)
            after_locate = _get_json(url, "/api/app/onboarding-status", timeout_s=10)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    hybrid_used = sum(row.get("final_payload", {}).get("research_trace", {}).get("retrieval", {}).get("local_retrieval", {}).get("used_mode") == "hybrid_local" for row in rows)
    return {"case_count": len(rows), "passed": sum(bool(row.get("quality", {}).get("ok")) for row in rows), "reused_answer_rows": reused_answers, "retrieval_mode": retrieval_mode, "hybrid_used_count": hybrid_used, "component": component, "cases": rows, "reader_checks": reader_checks, "onboarding": {"before": before, "after_answers": after_answers, "after_locate": after_locate}}


def _enable_hybrid_ui(url: str, root: Path) -> None:
    """Exercise consent through the ordinary React UI and the real install API."""
    from tools.research_qa.run_research_qa_eval import _post_json
    indexed = json.loads((root / "db/docs.json").read_text(encoding="utf-8"))
    source = next(iter(indexed.values()))["path"]
    conversation = _post_json(url, "/api/conversations", {"title": "hybrid-install-acceptance", "mode": "paper_guide", "bound_source_path": source, "bound_source_name": Path(source).name, "bound_source_ready": True}, timeout_s=10)
    script = f"""
    const {{chromium}}=require({json.dumps(str(PROJECT_ROOT / 'web/node_modules/playwright'))});
    (async()=>{{const browser=await chromium.launch({{executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true}});
    const page=await browser.newPage({{viewport:{{width:1280,height:900}}}});await page.goto(process.argv[1]+'/?conversation='+process.argv[3]);
    await page.getByTestId('chat-local-retrieval-control').click();
    await page.getByText(/启用本地语义检索|Enable local semantic (?:retrieval|search)/,{{exact:true}}).click();
    const dialog=page.getByTestId('local-semantic-consent');await dialog.waitFor();
    if(!/安装无需联网|installation does not require a network connection/.test(await dialog.innerText()))throw Error('offline installation disclosure missing');
    await page.screenshot({{path:process.argv[2]+'-consent.png'}});
    await page.getByRole('button',{{name:/^安装并启用$|^Install and enable$/}}).click();
    const deadline=Date.now()+600000;let state;while(Date.now()<deadline){{const status=await(await fetch(process.argv[1]+'/api/retrieval/local-component')).json();state=status.hybrid_local.state;
    require('fs').writeFileSync(process.argv[2]+'-status.json',JSON.stringify(status,null,2));
    if(state==='ready')break;if(state==='bm25_available'||state==='needs_space'||state==='paused')throw Error('component install stopped: '+JSON.stringify(status.hybrid_local));await page.waitForTimeout(1000);}}
    if(state!=='ready')throw Error('component preparation timed out');
    await page.reload();await page.getByTestId('chat-local-retrieval-control').click();
    await page.getByRole('menu').getByText(/本地语义检索|Local semantic search/,{{exact:true}}).click();
    await page.getByTestId('chat-local-retrieval-control').filter({{hasText:/本地语义检索|Local semantic search/}}).waitFor();
    await page.screenshot({{path:process.argv[2]+'-ready.png'}});await browser.close();}})().catch(e=>{{console.error(e);process.exit(1)}});
    """
    subprocess.run(["node", "-e", script, url, str(root / "component"), conversation["id"]], cwd=ROOT, check=True, timeout=660)


def _capture_hybrid_ready_ui(url: str, root: Path) -> None:
    """Verify the actual ready mode after resuming an interrupted UI check."""
    from tools.research_qa.run_research_qa_eval import _post_json
    indexed = json.loads((root / "db/docs.json").read_text(encoding="utf-8"))
    source = next(iter(indexed.values()))["path"]
    conversation = _post_json(url, "/api/conversations", {"title": "hybrid-ready-acceptance", "mode": "paper_guide", "bound_source_path": source, "bound_source_name": Path(source).name, "bound_source_ready": True}, timeout_s=10)
    script = f"""
    const {{chromium}}=require({json.dumps(str(PROJECT_ROOT / 'web/node_modules/playwright'))});
    (async()=>{{const browser=await chromium.launch({{executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true}});
    const page=await browser.newPage({{viewport:{{width:1280,height:900}}}});await page.goto(process.argv[1]+'/?conversation='+process.argv[3]);
    await page.getByTestId('chat-local-retrieval-control').click();
    await page.getByRole('menu').getByText(/本地语义检索|Local semantic search/,{{exact:true}}).click();
    await page.getByTestId('chat-local-retrieval-control').filter({{hasText:/本地语义检索|Local semantic search/}}).waitFor();
    await page.screenshot({{path:process.argv[2]}});await browser.close();}})().catch(e=>{{console.error(e);process.exit(1)}});
    """
    subprocess.run(["node", "-e", script, url, str(root / "component-ready.png"), conversation["id"]], cwd=ROOT, check=True, timeout=45)


def _reader_checks(url: str, root: Path, rows: list[dict]) -> list[dict]:
    from tools.research_qa.run_research_qa_eval import _post_json, _get_json

    checks = []
    for row in rows:
        cards = row.get("assistant_message", {}).get("cite_details", [])
        card = next((card for card in cards if not card.get("is_inpaper") and card.get("source_path") and card.get("evidence_quote")), None)
        if not card:
            checks.append({"id": row["id"], "status": "no_local_card"})
            continue
        payload = {"sourcePath": card["source_path"], "sourceName": card.get("source_name", ""), "snippet": card["evidence_quote"], "highlightSnippet": card["evidence_quote"], "headingPath": card.get("heading_path", ""), "blockId": card.get("block_id", ""), "anchorId": card.get("anchor_id", ""), "anchorKind": card.get("anchor_kind", ""), "pageStart": card.get("page_start", 0), "locateRequestId": 1}
        session = _post_json(url, "/api/reader/sessions", {"payload": payload, "conversation_id": row["conv_id"], "message_id": row["assistant_msg_id"]}, timeout_s=10)
        screenshot = root / f"reader-{row['id']}.png"
        script = f"""
        const {{chromium}}=require({json.dumps(str(PROJECT_ROOT / 'web/node_modules/playwright'))});const fs=require('fs');
        (async()=>{{const browser=await chromium.launch({{executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true}});
        const page=await browser.newPage({{viewport:{{width:1280,height:800}},deviceScaleFactor:2}});
        const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('requestfailed',r=>errors.push(r.url()+':'+r.failure()?.errorText));
        await page.goto(process.argv[1]);
        const deadline=Date.now()+20000;while(Date.now()<deadline){{const record=await(await fetch('{url}/api/reader/sessions/{session['id']}')).json();if(record.state?.locateResult)break;await page.waitForTimeout(500);}}
        await page.screenshot({{path:process.argv[2]}});fs.writeFileSync(process.argv[2]+'.browser.json',JSON.stringify({{errors,url:page.url()}}));await browser.close();}})();
        """
        subprocess.run(["node", "-e", script, f"{url}/reader/session/{session['id']}", str(screenshot)], cwd=ROOT, check=True, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        state = _get_json(url, f"/api/reader/sessions/{session['id']}", timeout_s=10).get("state", {})
        result = state.get("locateResult", {})
        checks.append({"id": row["id"], "status": result.get("status", "not_reported"), "ok": result.get("ok", False), "precision": result.get("precision", ""), "screenshot": str(screenshot)})
    (root / "reader_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser(description="Isolated real-model paired answer/citation contract acceptance; never writes the user library.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "test_results/project_optimization_20261013/live")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--product-root", type=Path, help="Run the copied product's backend/static UI in child processes; the harness remains in this checkout.")
    parser.add_argument("--hybrid-comparison", action="store_true", help="Compare BM25 and actual consented local hybrid with identical questions/sources; require at least one actual hybrid query.")
    parser.add_argument("--case-id", action="append", help="Replay only specified frozen live-smoke cases.")
    parser.add_argument("--resume", action="store_true", help="Finish interrupted UI verification without regenerating already-passing answers; requires identical product fingerprints.")
    args = parser.parse_args()
    if args.serve:
        _serve(args.root, args.port, args.baseline)
        return 0
    from kb.config import load_settings
    from tools.research_qa.run_research_qa_eval import load_fixture, select_fixture_cases
    settings = load_settings()
    if not settings.text_api_key:
        raise RuntimeError("real-model acceptance requires a configured text model")
    fixture = load_fixture()
    cases = select_fixture_cases(fixture, suite_names=["live_smoke_v1"])
    if args.case_id:
        cases = [case for case in cases if case["id"] in args.case_id]
        if len(cases) != len(set(args.case_id)):
            raise RuntimeError("case-id must identify a frozen live-smoke case")
    if args.hybrid_comparison:
        cases = [{**case, "queryScope": case.get("queryScope") or "basket"} for case in cases]
    if args.product_root:
        ROOT = args.product_root.resolve()
        if not (ROOT / "server.py").is_file() or not (ROOT / "web/dist/index.html").is_file():
            raise RuntimeError("product root must contain the staged backend and prebuilt frontend")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    files = [ROOT / "VERSION", ROOT / "server.py"]
    for folder in ("api", "kb", "ui"):
        files.extend((ROOT / folder).rglob("*.py"))
    files.extend((ROOT / "web/dist").rglob("*"))
    manifest = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files) if path.is_file()}
    if args.resume and (output / "source_manifest.json").is_file():
        if manifest != json.loads((output / "source_manifest.json").read_text(encoding="utf-8")):
            raise RuntimeError("cannot reuse answers after product code/UI changed")
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {"schema_version": 1, "suite": "live_smoke_v1", "model": settings.text_model, "product_root": str(ROOT), "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "fixture_sha256": hashlib.sha256((PROJECT_ROOT / "web/src/testing/researchQaData.json").read_bytes()).hexdigest(), "source_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(), "interpretation": "Real-API answer/citation contracts with the configured model and copied paper sources; deterministic generation bypasses remain in traces. This is not human correctness, entailment, or a BM25/hybrid answer comparison. Baseline simulates the current candidate with only action-polarity constraints disabled; it is not an earlier checkout. Stochastic variation is retained."}
    phases = (("candidate", False),) if args.candidate_only else (("baseline", True), ("candidate", False))
    if args.hybrid_comparison:
        phases = (("bm25", False), ("hybrid_local", False))
        report["interpretation"] = "Same-question/source real API comparison of BM25 and explicitly consented local BM25+dense retrieval. Both sides retain all claim guards. Specialized exact/section retrieval may preserve its original channel; require actual hybrid use in at least one query. Automatic contracts and source locations are not human faithfulness or an overall quality guarantee."
    for name, baseline in phases:
        mode = name if args.hybrid_comparison else None
        report[name] = _phase(output / name, fixture, cases, settings, baseline, retrieval_mode=mode, resume=args.resume)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate = report["hybrid_local" if args.hybrid_comparison else "candidate"]
    return 0 if (
        candidate["passed"] == len(cases)
        and len(candidate["reader_checks"]) == len(cases)
        and all(check.get("ok") for check in candidate["reader_checks"])
        and candidate["onboarding"]["after_locate"].get("completed")
        and (not args.hybrid_comparison or (report["bm25"]["passed"] == len(cases) and all(check.get("ok") for check in report["bm25"]["reader_checks"]) and candidate["hybrid_used_count"] > 0))
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
