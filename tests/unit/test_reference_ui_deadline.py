from __future__ import annotations

import time

import pytest

from api import reference_ui


def test_refs_card_polish_budget_env_is_bounded(monkeypatch):
    monkeypatch.setenv("KB_REFS_CARD_POLISH_TIMEOUT_S", "999")
    monkeypatch.setenv("KB_REFS_CARD_POLISH_MAX_RETRIES", "99")
    monkeypatch.setenv("KB_REFS_CARD_POLISH_TOP_N", "99")

    assert reference_ui._refs_card_polish_timeout_s() == 12.0
    assert reference_ui._refs_card_polish_max_retries() == 1
    assert reference_ui._refs_card_polish_top_n() == 8

    monkeypatch.setenv("KB_REFS_CARD_POLISH_TIMEOUT_S", "bad")
    monkeypatch.setenv("KB_REFS_CARD_POLISH_MAX_RETRIES", "bad")
    monkeypatch.setenv("KB_REFS_CARD_POLISH_TOP_N", "bad")

    assert reference_ui._refs_card_polish_timeout_s(7.0) == 7.0
    assert reference_ui._refs_card_polish_max_retries() == 0
    assert reference_ui._refs_card_polish_top_n() == 6


def test_refs_card_polish_llm_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("KB_REFS_CARD_POLISH_USE_LLM", raising=False)

    assert reference_ui._refs_card_polish_llm_enabled() is True

    monkeypatch.setenv("KB_REFS_CARD_POLISH_USE_LLM", "0")

    assert reference_ui._refs_card_polish_llm_enabled() is False


def test_enrich_refs_payload_skips_expensive_steps_after_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_expensive(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("deadline-expired refs rendering should not call expensive enrichment")

    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", fail_expensive)
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", fail_expensive)
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", fail_expensive)
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", fail_expensive)

    def fake_build_hit_ui_meta(hit: dict, **kwargs) -> dict:
        assert kwargs.get("allow_exact_locate") is False
        return {
            "display_name": hit["meta"]["source_path"],
            "heading_path": hit["meta"].get("ref_best_heading_path", ""),
            "summary_line": "A deterministic fast card.",
            "why_line": "A deterministic fast relevance note.",
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)

    refs = {
        1: {
            "prompt": "Which papers discuss ADMM?",
            "hits": [
                {
                    "text": "This paper discusses ADMM for reconstruction.",
                    "meta": {
                        "source_path": "paper-a.pdf",
                        "ref_pack_state": "ready",
                        "ref_best_heading_path": "Related Work",
                        "explicit_doc_match_score": 7.0,
                    },
                },
                {
                    "text": "This paper also discusses ADMM variants.",
                    "meta": {
                        "source_path": "paper-b.pdf",
                        "ref_pack_state": "ready",
                        "ref_best_heading_path": "Methods",
                        "explicit_doc_match_score": 7.0,
                    },
                },
            ],
        }
    }

    out = reference_ui.enrich_refs_payload(
        refs,
        pdf_root=None,
        md_root=None,
        lib_store=None,
        allow_expensive_llm_for_ready=True,
        allow_exact_locate=True,
        deadline_at=time.perf_counter() - 1.0,
    )

    pack = out[1]
    assert len(pack["hits"]) == 2
    assert pack["pipeline_debug"]["deadline_exhausted"] is True
