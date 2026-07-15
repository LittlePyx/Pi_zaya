from __future__ import annotations

import os

import pdf_to_md
from kb.converter.pipeline import PDFConverter


def _base_args(tmp_path, *extra: str) -> list[str]:
    return [
        "--pdf",
        str(tmp_path / "paper.pdf"),
        "--out",
        str(tmp_path / "out"),
        *extra,
    ]


def _use_fake_qwen(monkeypatch) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "test-key")
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


def test_locked_profile_keeps_explicit_ultra_fast_mode(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)

    cfg = pdf_to_md._parse_args(_base_args(tmp_path, "--speed-mode", "ultra_fast"))

    assert cfg.speed_mode == "ultra_fast"
    assert PDFConverter._get_speed_mode_config(
        PDFConverter.__new__(PDFConverter),
        cfg.speed_mode,
        10,
    )["dpi"] == 150


def test_locked_profile_keeps_explicit_balanced_legacy_mode(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)

    cfg = pdf_to_md._parse_args(_base_args(tmp_path, "--speed-mode", "balanced"))

    assert cfg.speed_mode == "balanced"


def test_locked_profile_defaults_to_normal_mode(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)

    cfg = pdf_to_md._parse_args(_base_args(tmp_path))

    assert cfg.speed_mode == "normal"
    assert os.environ["KB_PDF_ULTRA_FAST_VISION_TIMEOUT_S"] == "45"


def test_locked_profile_preserves_explicit_ultra_fast_timeout(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)
    monkeypatch.setenv("KB_PDF_ULTRA_FAST_VISION_TIMEOUT_S", "72")

    cfg = pdf_to_md._parse_args(_base_args(tmp_path, "--speed-mode", "ultra_fast"))

    assert cfg.speed_mode == "ultra_fast"
    assert os.environ["KB_PDF_ULTRA_FAST_VISION_TIMEOUT_S"] == "72"


def test_pdf_cli_uses_versioned_qwen_default_but_keeps_explicit_override(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)
    default_cfg = pdf_to_md._parse_args(_base_args(tmp_path))
    explicit_cfg = pdf_to_md._parse_args(
        _base_args(tmp_path, "--model", "user-selected-vision-model")
    )

    assert default_cfg.llm is not None
    assert default_cfg.llm.model == "qwen3.7-plus-2026-05-26"
    assert explicit_cfg.llm is not None
    assert explicit_cfg.llm.model == "user-selected-vision-model"
