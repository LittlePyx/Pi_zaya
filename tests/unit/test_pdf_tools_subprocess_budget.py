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


def test_split_subprocess_llm_budget_keeps_validated_single_doc_budget_on_large_hosts(monkeypatch):
    monkeypatch.delenv("KB_LLM_MAX_INFLIGHT", raising=False)
    monkeypatch.setattr(pdf_tools.os, "cpu_count", lambda: 16)

    workers, llm_workers, per_doc_inflight, active_docs, global_inflight = _split_subprocess_llm_budget(
        no_llm_mode=False,
        workers=4,
        llm_workers=3,
        max_active_docs=1,
    )

    assert workers * llm_workers <= 8
    assert per_doc_inflight == 8
    assert active_docs == 1
    assert global_inflight == 8


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


def test_split_subprocess_llm_budget_keeps_validated_default_for_multi_doc_runs(monkeypatch):
    monkeypatch.delenv("KB_LLM_MAX_INFLIGHT", raising=False)
    monkeypatch.setattr(pdf_tools.os, "cpu_count", lambda: 16)

    workers, llm_workers, per_doc_inflight, active_docs, global_inflight = _split_subprocess_llm_budget(
        no_llm_mode=False,
        workers=4,
        llm_workers=3,
        max_active_docs=2,
    )

    assert workers * llm_workers <= 4
    assert per_doc_inflight == 4
    assert active_docs == 2
    assert global_inflight == 8


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


def test_run_pdf_to_md_dynamic_global_budget_keeps_workers_and_forwards_coordinator(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "tiny.pdf"
    doc = fitz.open()
    for _ in range(10):
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

    monkeypatch.setattr(
        pdf_tools.subprocess,
        "Popen",
        lambda args, **kwargs: _FakeProc(env=kwargs.get("env") or {}, args=list(args)),
    )

    coordinator = tmp_path / "global-inflight"
    ok, _out = pdf_tools.run_pdf_to_md(
        pdf_path=pdf_path,
        out_root=tmp_path / "out",
        no_llm=False,
        keep_debug=False,
        eq_image_fallback=False,
        speed_mode="normal",
        max_active_conversions=2,
        global_inflight_coordinator=coordinator,
        global_inflight_owner="task-123",
        global_inflight_limit=8,
    )

    assert ok is True
    child_env = dict(captured["env"])
    child_args = list(captured["args"])
    assert child_args[child_args.index("--workers") + 1] == "4"
    assert child_args[child_args.index("--llm-workers") + 1] == "3"
    assert child_env["KB_LLM_MAX_INFLIGHT"] == "8"
    assert child_env["KB_LLM_GLOBAL_COORDINATOR"] == str(coordinator.resolve())
    assert child_env["KB_LLM_GLOBAL_OWNER"] == "task-123"
    assert child_env["KB_LLM_GLOBAL_MAX_INFLIGHT"] == "8"
    assert child_env["KB_LLM_GLOBAL_REQUIRED"] == "1"


def test_run_pdf_to_md_forwards_effective_vision_settings_to_child(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "tiny.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    class _VisionSettings:
        vision_model = "saved-vision-model"
        vision_base_url = "https://vision.example/v1"
        vision_api_key = "saved-vision-key"

    import kb.config as config_module

    monkeypatch.setattr(config_module, "load_settings", lambda: _VisionSettings())
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

    monkeypatch.setattr(
        pdf_tools.subprocess,
        "Popen",
        lambda args, **kwargs: _FakeProc(env=kwargs.get("env") or {}, args=list(args)),
    )

    ok, _out = pdf_tools.run_pdf_to_md(
        pdf_path=pdf_path,
        out_root=tmp_path / "out",
        no_llm=False,
        keep_debug=False,
        eq_image_fallback=False,
        speed_mode="normal",
        max_active_conversions=1,
    )

    assert ok is True
    child_args = list(captured["args"])
    child_env = dict(captured["env"])
    assert child_args[child_args.index("--model") + 1] == "saved-vision-model"
    assert child_args[child_args.index("--base-url") + 1] == "https://vision.example/v1"
    assert child_args[child_args.index("--api-key-env") + 1] == "KB_PDF_RUNTIME_VISION_API_KEY"
    assert child_env["KB_PDF_RUNTIME_VISION_API_KEY"] == "saved-vision-key"


def test_run_pdf_to_md_splits_validated_multi_doc_budget_when_env_missing(monkeypatch, tmp_path: Path):
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
    assert child_env["KB_LLM_MAX_INFLIGHT"] == "4"


def test_run_pdf_to_md_uses_validated_single_doc_budget_when_env_missing(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "ten_pages.pdf"
    doc = fitz.open()
    for _ in range(10):
        doc.new_page()
    doc.save(pdf_path)
    doc.close()

    monkeypatch.delenv("KB_PDF_WORKERS", raising=False)
    monkeypatch.delenv("KB_PDF_LLM_WORKERS", raising=False)
    monkeypatch.delenv("KB_LLM_MAX_INFLIGHT", raising=False)
    monkeypatch.delenv("KB_PDF_LLM_TIMEOUT_S", raising=False)
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
    assert child_env["KB_LLM_MAX_INFLIGHT"] == "8"
    assert "--workers" in child_args and child_args[child_args.index("--workers") + 1] == "2"
    assert "--llm-workers" in child_args and child_args[child_args.index("--llm-workers") + 1] == "3"
    assert "--llm-timeout" in child_args and child_args[child_args.index("--llm-timeout") + 1] == "120"


def test_run_pdf_to_md_tracks_true_remaining_pages_after_out_of_order_completion(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "parallel.pdf"
    doc = fitz.open()
    for _ in range(21):
        doc.new_page()
    doc.save(pdf_path)
    doc.close()

    class _FakeProc:
        def __init__(self):
            self.stdout = io.StringIO(
                "Processing page 17/21 (vision-direct) ...\n"
                "Processing page 21/21 (vision-direct) ...\n"
                "Finished page 21/21 (1.0s, 100 chars)\n"
                "[VISION_DIRECT] still running pages: [17] | workers=4 llm_inflight=8\n"
            )
            self.pid = 4321

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    monkeypatch.setattr(pdf_tools.subprocess, "Popen", lambda *_args, **_kwargs: _FakeProc())
    snapshots: list[list[int]] = []

    ok, _out = pdf_tools.run_pdf_to_md(
        pdf_path=pdf_path,
        out_root=tmp_path / "out",
        no_llm=False,
        keep_debug=False,
        eq_image_fallback=False,
        running_pages_cb=lambda pages: snapshots.append(list(pages)),
    )

    assert ok is True
    assert [17, 21] in snapshots
    remaining_index = snapshots.index([17], snapshots.index([17, 21]) + 1)
    assert remaining_index > 0
    assert snapshots[-1] == []


def test_run_pdf_to_md_deduplicates_and_filters_running_page_updates(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "small.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    doc.save(pdf_path)
    doc.close()

    class _FakeProc:
        def __init__(self):
            self.stdout = io.StringIO(
                "Rendering page 2/3 ...\n"
                "[Page 2] layout pass\n"
                "[VISION_DIRECT] still running pages: [2, 2, 99] | workers=1 llm_inflight=1\n"
                "Finished page 2/3\n"
            )
            self.pid = 4321

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    monkeypatch.setattr(pdf_tools.subprocess, "Popen", lambda *_args, **_kwargs: _FakeProc())
    snapshots: list[list[int]] = []

    pdf_tools.run_pdf_to_md(
        pdf_path=pdf_path,
        out_root=tmp_path / "out",
        no_llm=False,
        keep_debug=False,
        eq_image_fallback=False,
        running_pages_cb=lambda pages: snapshots.append(list(pages)),
    )

    assert [2] in snapshots
    assert all(99 not in pages for pages in snapshots)
    assert snapshots[-1] == []
