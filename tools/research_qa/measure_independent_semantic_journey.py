from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("KB_ACCEPTANCE_PRODUCT_ROOT") or str(PROJECT))
sys.path.append(str(PROJECT))

from tools.research_qa import run_optimization_acceptance as acceptance
if not os.environ.get("KB_ACCEPTANCE_PRODUCT_ROOT"):
    from tools.research_qa.run_independent_pdf_acceptance import _corpus_digest
from tools.research_qa.run_research_qa_eval import _get_json, _post_json

# Helper imports may prepend the development tree. Child servers must resolve
# all runtime modules from the specified staged product.
if os.environ.get("KB_ACCEPTANCE_PRODUCT_ROOT"):
    sys.path.insert(0, os.environ["KB_ACCEPTANCE_PRODUCT_ROOT"])


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _serve(root: Path, port: int) -> None:
    # Observe the actual product adapter's return value without changing its work.
    from kb.portable_local_semantic_runtime import PortableLocalSemanticRuntimeAdapter

    original = PortableLocalSemanticRuntimeAdapter.prepare_index

    def measured(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        with (root / "performance-builds.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result)) + "\n")
        return result

    PortableLocalSemanticRuntimeAdapter.prepare_index = measured
    acceptance._serve(root, port, False)


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
    command = [
        str(product / "runtime/python/python.exe"),
        str(Path(__file__).resolve()),
        "--serve",
        "--root",
        str(root),
        "--port",
        str(port),
    ]
    started = time.monotonic()
    with (root / (label + ".log")).open("w", encoding="utf-8") as log:
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
            while time.monotonic() - started < 60:
                if process.poll() is not None:
                    raise RuntimeError("backend exited; inspect isolated journey log")
                try:
                    _get_json(url, "/api/health", timeout_s=2)
                    break
                except OSError:
                    time.sleep(0.25)
            else:
                raise TimeoutError("backend startup")
            yield url, time.monotonic() - started
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _status(url: str) -> dict:
    return _get_json(url, "/api/retrieval/local-component", timeout_s=30)[
        "hybrid_local"
    ]


def _wait(url: str, expected: str, seconds: int = 180) -> dict:
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        status = _status(url)
        if status["state"] == expected:
            return status
        if status["state"] in {"degraded_bm25", "needs_space"}:
            raise RuntimeError(status.get("reason_id"))
        time.sleep(0.15)
    raise TimeoutError(expected)


def _kernel_measurement(root: Path) -> dict:
    from kb.local_semantic_component import (
        LocalSemanticComponentStore,
        local_semantic_component_root,
    )
    from kb.portable_local_semantic_runtime import PortableLocalSemanticRuntimeAdapter

    store = LocalSemanticComponentStore(local_semantic_component_root(root / "db"))
    adapter = PortableLocalSemanticRuntimeAdapter(root / "db", component_store=store)
    started = time.monotonic()
    cold = adapter.load_ready()
    cold_ms = (time.monotonic() - started) * 1000
    query = "training a denoiser without a clean target image"
    elapsed = []
    for _ in range(5):
        started = time.monotonic()
        assert cold.retriever.search(query, top_k=6)
        elapsed.append((time.monotonic() - started) * 1000)
    started = time.monotonic()
    warm = adapter.load_ready()
    warm_ms = (time.monotonic() - started) * 1000
    result = adapter.prepare_index(
        store.load_current().root,
        should_cancel=lambda: False,
        on_progress=lambda *_: None,
    )
    adapter.release_runtime_handles()
    return {
        "cold_runtime_load_ms": cold_ms,
        "warm_runtime_load_ms": warm_ms,
        "cold_start": cold.cold_start,
        "warm_cache_hit": warm.index_cache_hit,
        "dense_query_ms": elapsed,
        "unchanged_index_build": asdict(result),
    }


def _run(args) -> None:
    import requests
    from kb.config import load_settings

    root, product = args.root.resolve(), args.product_root.resolve()
    output = root / "performance-report.json"
    assert not output.exists(), "Preserve prior measurement; do not overwrite."
    before = _corpus_digest(root)
    _write(root / "performance-corpus-before.json", before)
    result = {
        "limits": "One Windows host; instrumented real product API and real verified E5 kernel. Extra paper added only after frozen paired Q&A. Cold means process/runtime cold with existing model files, not a cold OS disk cache."
    }
    settings = load_settings()
    pointer = root / "db/local_semantic_index/e5_small_v1/current.json"
    old_pointer = pointer.read_bytes()
    with _backend(root, product, settings, "performance-before") as (url, startup):
        result["backend_health_startup_s"] = startup
        _wait(url, "ready")
        # Measure in the embedded verified runtime, outside the HTTP lifecycle worker.
        env = acceptance._env(root, settings)
        subprocess.run(
            [
                str(product / "runtime/python/python.exe"),
                str(Path(__file__).resolve()),
                "--kernel",
                "--root",
                str(root),
            ],
            cwd=product,
            env=env,
            check=True,
        )
        result["kernel"] = json.loads(
            (root / "performance-kernel.json").read_text(encoding="utf-8")
        )
        started = time.monotonic()
        pdf_name = args.pdf.name
        with args.pdf.open("rb") as handle:
            response = requests.post(
                url + "/api/library/upload",
                files={"file": (pdf_name, handle, "application/pdf")},
                data={"base_name": args.pdf.stem},
                timeout=90,
            )
        response.raise_for_status()
        result["upload"] = response.json()
        _post_json(
            url,
            "/api/library/convert",
            {
                "pdf_name": pdf_name,
                "no_llm": True,
                "speed_mode": "no_llm",
                "replace": False,
            },
            timeout_s=30,
        )
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            files = _get_json(url, "/api/library/files?scope=all", timeout_s=30)
            item = next(
                x for x in files["items"] if x["name"] == pdf_name
            )
            if item.get("index_ready"):
                break
            if item.get("task_state") not in {"queued", "running"} and item.get(
                "md_exists"
            ):
                _write(root / "performance-extra-source-quality.json", item)
                raise RuntimeError("extra source is quality blocked; no override")
            time.sleep(1)
        else:
            raise TimeoutError("extra PDF conversion")
        result["upload_to_normal_index_ready_s"] = time.monotonic() - started
        _write(output, result)
        active = _status(url)
        assert active["state"] != "ready", (
            "New corpus must invalidate old semantic snapshot."
        )
        deadline = time.monotonic() + 60
        while (active.get("reason_id") != "semantic_index_preparing"
               or not active.get("completed_bytes")
               or not 0 < int(active.get("total_bytes") or 0) < 100000):
            assert active["state"] != "ready", "Index completed before a measured mid-build cancellation."
            if time.monotonic() > deadline:
                raise TimeoutError("incremental index progress")
            time.sleep(0.1)
            active = _status(url)
        result["active_index_at_cancel"] = active
        result["cancel_response"] = _post_json(
            url,
            "/api/retrieval/local-component/cancel",
            {"task_id": active["task_id"]},
            timeout_s=30,
        )
        result["paused"] = _wait(url, "paused")
        result["cancel_kept_committed_pointer"] = old_pointer == pointer.read_bytes()
        assert result["cancel_kept_committed_pointer"]
        result["bm25_docs_while_paused"] = len(
            json.loads((root / "db/docs.json").read_text(encoding="utf-8"))
        )
        _write(output, result)
    with _backend(root, product, settings, "performance-restart") as (url, startup):
        result["restart_health_startup_s"] = startup
        result["restart_state"] = _status(url)
        assert result["restart_state"]["state"] == "paused"
        started = time.monotonic()
        _post_json(url, "/api/retrieval/local-component/resume", {}, timeout_s=30)
        result["resumed_ready"] = _wait(url, "ready")
        result["resume_to_ready_s"] = time.monotonic() - started
        builds = [
            json.loads(line)
            for line in (root / "performance-builds.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        result["real_api_incremental_builds"] = builds
        assert builds[-1]["reused_parent_count"] == result["kernel"]["unchanged_index_build"]["parent_count"]
        assert builds[-1]["embedded_parent_count"] > 0
        after = _corpus_digest(root)
        original = {
            key: value
            for key, value in before.items()
            if key.startswith(("pdfs/", "markdown/"))
        }
        result["original_pdf_markdown_unchanged"] = all(
            after.get(key) == value for key, value in original.items()
        )
        assert result["original_pdf_markdown_unchanged"]
        _write(output, result)
    print("Real incremental API index, cancel/restart/resume passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("F:/pz-independent-20261013"))
    parser.add_argument("--product-root", type=Path)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PROJECT
        / "test_results/independent_acceptance_20261013/neighbor2neighbor.pdf",
    )
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--kernel", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.serve:
        _serve(args.root, args.port)
    elif args.kernel:
        _write(args.root / "performance-kernel.json", _kernel_measurement(args.root))
    else:
        assert args.product_root
        _run(args)
