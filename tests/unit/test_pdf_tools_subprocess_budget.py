from __future__ import annotations

import io
from pathlib import Path

import fitz

from kb.pdf_tools import _split_subprocess_llm_budget
from kb import pdf_tools


def test_split_subprocess_llm_budget_keeps_single_doc_budget(monkeypatch):
    monkeypatch.delenv("KB_LLM_MAX_INFLIGHT", raising=False)
    monkeypatch.setattr(pdf_tools.os, "cpu_count", lambda: 4)

    workers, llm_workers, per_doc_inflight, active_docs, global_inflight = _split_subprocess_llm_budget(
        no_llm_mode=False,
        workers=4,
        llm_workers=2,
        max_active_docs=1,
    )

    assert (workers, llm_workers) == (4, 2)
    assert per_doc_inflight == 8
    assert active_docs == 1
    assert global_inflight == 8


def test_split_subprocess_llm_budget_uses_adaptive_single_doc_budget_on_large_hosts(monkeypatch):
    monkeypatch.delenv("KB_LLM_MAX_INFLIGHT", raising=False)
    monkeypatch.setattr(pdf_tools.os, "cpu_count", lambda: 16)

    workers, llm_workers, per_doc_inflight, active_docs, global_inflight = _split_subprocess_llm_budget(
        no_llm_mode=False,
        workers=4,
        llm_workers=3,
        max_active_docs=1,
    )

    assert (workers, llm_workers) == (4, 3)
    assert per_doc_inflight == 12
    assert active_docs == 1
    assert global_inflight == 12


def test_split_subprocess_llm_budget_splits_global_budget_across_docs(monkeypatch):
    monkeypatch.setenv("KB_LLM_MAX_INFLIGHT", "8")

    workers, llm_workers, per_doc_inflight, active_docs, global_inflight = _split_subprocess_llm_budget(
        no_llm_mode=False,
        workers=4,
        llm_workers=3,
        max_active_docs=2,
    )

    assert (workers, llm_workers) == (2, 2)
    assert workers * llm_workers <= 4
    assert per_doc_inflight == 4
    assert active_docs == 2
    assert global_inflight == 8


def test_split_subprocess_llm_budget_uses_adaptive_default_for_multi_doc_runs(monkeypatch):
    monkeypatch.delenv("KB_LLM_MAX_INFLIGHT", raising=False)
    monkeypatch.setattr(pdf_tools.os, "cpu_count", lambda: 16)

    workers, llm_workers, per_doc_inflight, active_docs, global_inflight = _split_subprocess_llm_budget(
        no_llm_mode=False,
        workers=4,
        llm_workers=3,
        max_active_docs=2,
    )

    assert workers * llm_workers <= 8
    assert per_doc_inflight == 8
    assert active_docs == 2
    assert global_inflight == 16


def test_split_subprocess_llm_budget_leaves_no_llm_unchanged(monkeypatch):
    monkeypatch.setenv("KB_LLM_MAX_INFLIGHT", "12")

    workers, llm_workers, per_doc_inflight, active_docs, global_inflight = _split_subprocess_llm_budget(
        no_llm_mode=True,
        workers=6,
        llm_workers=1,
        max_active_docs=3,
    )

    assert (workers, llm_workers) == (6, 1)
    assert per_doc_inflight is None
    assert active_docs == 3
    assert global_inflight == 0


def test_run_pdf_to_md_overrides_child_env_for_split_budget(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "tiny.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    monkeypatch.setenv("KB_PDF_WORKERS", "4")
    monkeypatch.setenv("KB_PDF_LLM_WORKERS", "3")
    monkeypatch.setenv("KB_LLM_MAX_INFLIGHT", "8")

    captured: dict[str, object] = {}

    class _FakeProc:
        def __init__(self, *, env: dict[str, str], args: list[str]):
            captured["env"] = dict(env)
            captured["args"] = list(args)
            self.stdout = io.StringIO("")
            self.pid = 4321

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    def _fake_popen(args, **kwargs):
        return _FakeProc(env=kwargs.get("env") or {}, args=list(args))

    monkeypatch.setattr(pdf_tools.subprocess, "Popen", _fake_popen)

    ok, out = pdf_tools.run_pdf_to_md(
        pdf_path=pdf_path,
        out_root=tmp_path / "out",
        no_llm=False,
        keep_debug=False,
        eq_image_fallback=False,
        speed_mode="normal",
        max_active_conversions=2,
    )

    assert ok is True
    assert str(out).endswith(str(Path("out") / "tiny"))
    child_env = dict(captured["env"])
    assert child_env["KB_LLM_MAX_INFLIGHT"] == "4"


def test_run_pdf_to_md_uses_adaptive_multi_doc_budget_when_env_missing(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "tiny.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    monkeypatch.setenv("KB_PDF_WORKERS", "4")
    monkeypatch.setenv("KB_PDF_LLM_WORKERS", "3")
    monkeypatch.delenv("KB_LLM_MAX_INFLIGHT", raising=False)
    monkeypatch.setattr(pdf_tools.os, "cpu_count", lambda: 16)

    captured: dict[str, object] = {}

    class _FakeProc:
        def __init__(self, *, env: dict[str, str], args: list[str]):
            captured["env"] = dict(env)
            captured["args"] = list(args)
            self.stdout = io.StringIO("")
            self.pid = 4321

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    def _fake_popen(args, **kwargs):
        return _FakeProc(env=kwargs.get("env") or {}, args=list(args))

    monkeypatch.setattr(pdf_tools.subprocess, "Popen", _fake_popen)

    ok, out = pdf_tools.run_pdf_to_md(
        pdf_path=pdf_path,
        out_root=tmp_path / "out",
        no_llm=False,
        keep_debug=False,
        eq_image_fallback=False,
        speed_mode="normal",
        max_active_conversions=2,
    )

    assert ok is True
    assert str(out).endswith(str(Path("out") / "tiny"))
    child_env = dict(captured["env"])
    assert child_env["KB_LLM_MAX_INFLIGHT"] == "8"


def test_run_pdf_to_md_uses_adaptive_single_doc_budget_when_env_missing(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "ten_pages.pdf"
    doc = fitz.open()
    for _ in range(10):
        doc.new_page()
    doc.save(pdf_path)
    doc.close()

    monkeypatch.delenv("KB_PDF_WORKERS", raising=False)
    monkeypatch.delenv("KB_PDF_LLM_WORKERS", raising=False)
    monkeypatch.delenv("KB_LLM_MAX_INFLIGHT", raising=False)
    monkeypatch.setattr(pdf_tools.os, "cpu_count", lambda: 16)

    captured: dict[str, object] = {}

    class _FakeProc:
        def __init__(self, *, env: dict[str, str], args: list[str]):
            captured["env"] = dict(env)
            captured["args"] = list(args)
            self.stdout = io.StringIO("")
            self.pid = 4321

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    def _fake_popen(args, **kwargs):
        return _FakeProc(env=kwargs.get("env") or {}, args=list(args))

    monkeypatch.setattr(pdf_tools.subprocess, "Popen", _fake_popen)

    ok, out = pdf_tools.run_pdf_to_md(
        pdf_path=pdf_path,
        out_root=tmp_path / "out",
        no_llm=False,
        keep_debug=False,
        eq_image_fallback=False,
        speed_mode="normal",
        max_active_conversions=1,
    )

    assert ok is True
    assert str(out).endswith(str(Path("out") / "ten_pages"))
    child_env = dict(captured["env"])
    child_args = list(captured["args"])
    assert child_env["KB_LLM_MAX_INFLIGHT"] == "12"
    assert "--workers" in child_args and child_args[child_args.index("--workers") + 1] == "4"
    assert "--llm-workers" in child_args and child_args[child_args.index("--llm-workers") + 1] == "3"
