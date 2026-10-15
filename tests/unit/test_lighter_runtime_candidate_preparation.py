from __future__ import annotations

import copy
import json

import pytest

from tools.retrieval_eval.prepare_lighter_runtime_candidate import (
    DEFAULT_PROTOCOL,
    DEFAULT_REQUIREMENTS,
    _locked_requirements,
    _select_candidate,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_lighter_runtime_requirement_lock_is_exact_and_forbids_heavy_frameworks() -> None:
    locked = _locked_requirements(DEFAULT_REQUIREMENTS)
    assert locked["onnxruntime"] == "1.23.2"
    assert locked["tokenizers"] == "0.21.4"
    assert not {"torch", "transformers", "optimum"} & set(locked)


def test_lighter_runtime_candidate_one_is_selected_without_predecessor_report() -> None:
    selected = _select_candidate(
        _protocol(), "qwen3_embedding_0_6b_official_quint8_avx2_onnx_v1"
    )
    assert selected["evaluation_order"] == 1
    assert selected["primary_artifact"]["bytes"] == 600127788


def test_lighter_runtime_unknown_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown lighter runtime candidate"):
        _select_candidate(_protocol(), "invented")


def test_lighter_runtime_duplicate_requirement_is_rejected(tmp_path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("onnxruntime==1.23.2\nONNXRuntime==1.23.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be unique"):
        _locked_requirements(requirements)


def test_lighter_runtime_candidate_contract_retains_no_product_enablement() -> None:
    changed = copy.deepcopy(_protocol())
    assert changed["execution"]["artifact_downloads"] == 0
    assert changed["execution"]["model_inference_runs"] == 0
    assert changed["product_boundary"]["product_enabled"] is False


def test_lighter_runtime_candidate_two_is_authorized_only_after_qwen_rejection() -> None:
    selected = _select_candidate(_protocol(), "multilingual_e5_small_official_onnx_v1")
    assert selected["evaluation_order"] == 2
    assert selected["tokenizer"]["required_files"] == [
        "config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert selected["stage_1_quality_route"]["silent_long_text_tail_loss_allowed"] is False
