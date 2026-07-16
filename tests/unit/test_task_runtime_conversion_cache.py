from pathlib import Path

from kb import task_runtime


def _build_output(root: Path) -> Path:
    output = root / "paper"
    cache = output / ".conversion_cache" / "pages" / "00001"
    cache.mkdir(parents=True)
    (cache / "entry.json").write_text("{}", encoding="utf-8")
    (output / "assets").mkdir()
    (output / "assets" / "page_1_fig_1.png").write_bytes(b"old")
    (output / "output.md").write_text("old markdown", encoding="utf-8")
    return output


def test_safe_clear_conversion_output_preserves_page_cache_for_repeat_conversion(tmp_path: Path) -> None:
    output = _build_output(tmp_path)

    task_runtime._safe_clear_conversion_output(output, tmp_path, preserve_page_cache=True)

    assert (output / ".conversion_cache" / "pages" / "00001" / "entry.json").exists()
    assert not (output / "assets").exists()
    assert not (output / "output.md").exists()


def test_safe_clear_conversion_output_discards_cache_for_quality_repair(tmp_path: Path) -> None:
    output = _build_output(tmp_path)

    task_runtime._safe_clear_conversion_output(output, tmp_path, preserve_page_cache=False)

    assert not output.exists()
