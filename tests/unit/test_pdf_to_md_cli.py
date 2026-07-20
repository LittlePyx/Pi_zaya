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
    monkeypatch.delenv("QWEN_VISION_MODEL", raising=False)
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
    assert PDFConverter._get_speed_mode_config(
        PDFConverter.__new__(PDFConverter),
        cfg.speed_mode,
        10,
    ) == PDFConverter._get_speed_mode_config(
        PDFConverter.__new__(PDFConverter),
        "normal",
        10,
    )


def test_legacy_fast_mode_uses_ultra_fast_execution_profile(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)

    cfg = pdf_to_md._parse_args(_base_args(tmp_path, "--speed-mode", "fast"))

    assert cfg.speed_mode == "fast"
    assert PDFConverter._get_speed_mode_config(
        PDFConverter.__new__(PDFConverter),
        cfg.speed_mode,
        10,
    )["dpi"] == 150


def test_full_llm_keeps_quality_first_execution_profile(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)

    cfg = pdf_to_md._parse_args(_base_args(tmp_path, "--speed-mode", "full_llm"))

    assert cfg.speed_mode == "full_llm"
    profile = PDFConverter._get_speed_mode_config(
        PDFConverter.__new__(PDFConverter),
        cfg.speed_mode,
        10,
    )
    assert profile["dpi"] == 220
    assert profile["max_tokens"] == 4096


def test_locked_profile_defaults_to_normal_mode(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)
    monkeypatch.delenv("KB_PDF_LLM_TIMEOUT_S", raising=False)

    cfg = pdf_to_md._parse_args(_base_args(tmp_path))

    assert cfg.speed_mode == "normal"
    assert cfg.llm is not None and cfg.llm.timeout_s == 120.0
    assert os.environ["KB_PDF_ULTRA_FAST_VISION_TIMEOUT_S"] == "45"


def test_locked_profile_respects_explicit_llm_timeout(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)

    cfg = pdf_to_md._parse_args(_base_args(tmp_path, "--llm-timeout", "25"))

    assert cfg.llm is not None
    assert cfg.llm.timeout_s == 25.0


def test_locked_profile_preserves_explicit_ultra_fast_timeout(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)
    monkeypatch.setenv("KB_PDF_ULTRA_FAST_VISION_TIMEOUT_S", "72")

    cfg = pdf_to_md._parse_args(_base_args(tmp_path, "--speed-mode", "ultra_fast"))

    assert cfg.speed_mode == "ultra_fast"
    assert os.environ["KB_PDF_ULTRA_FAST_VISION_TIMEOUT_S"] == "72"


def test_pdf_cli_uses_vision_qwen_default_but_keeps_explicit_override(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)
    default_cfg = pdf_to_md._parse_args(_base_args(tmp_path))
    explicit_cfg = pdf_to_md._parse_args(
        _base_args(tmp_path, "--model", "user-selected-vision-model")
    )

    assert default_cfg.llm is not None
    assert default_cfg.llm.model == "qwen3-vl-plus"
    assert explicit_cfg.llm is not None
    assert explicit_cfg.llm.model == "user-selected-vision-model"


def test_pdf_cli_prefers_role_specific_vision_model_over_legacy_shared_model(tmp_path, monkeypatch):
    _use_fake_qwen(monkeypatch)
    monkeypatch.setenv("QWEN_MODEL", "legacy-shared-model")
    monkeypatch.setenv("QWEN_VISION_MODEL", "dedicated-vision-model")

    cfg = pdf_to_md._parse_args(_base_args(tmp_path))

    assert cfg.llm is not None
    assert cfg.llm.model == "dedicated-vision-model"
