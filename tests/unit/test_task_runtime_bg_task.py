from pathlib import Path

from kb.task_runtime import _build_bg_task


def test_ultra_fast_does_not_force_no_llm():
    task = _build_bg_task(
        pdf_path=Path("a.pdf"),
        out_root=Path("out"),
        db_dir=Path("db"),
        no_llm=False,
        replace=False,
        speed_mode="ultra_fast",
    )
    assert task["speed_mode"] == "ultra_fast"
    assert task["no_llm"] is False


def test_no_llm_mode_respects_flag():
    task = _build_bg_task(
        pdf_path=Path("a.pdf"),
        out_root=Path("out"),
        db_dir=Path("db"),
        no_llm=True,
        replace=False,
        speed_mode="no_llm",
    )
    assert task["speed_mode"] == "no_llm"
    assert task["no_llm"] is True
