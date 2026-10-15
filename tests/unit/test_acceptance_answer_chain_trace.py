from __future__ import annotations

import json

import pytest

from tools.research_qa.answer_chain_trace import install


def test_raw_trace_requires_isolated_backend(monkeypatch, tmp_path):
    monkeypatch.delenv("KB_TEST_ISOLATED", raising=False)
    path = tmp_path / "trace.jsonl"
    with pytest.raises(RuntimeError, match="isolated"):
        install(path)
    assert not path.exists()


@pytest.mark.parametrize("raises", [False, True])
def test_raw_trace_keeps_pre_mutation_input_and_omits_settings(monkeypatch, tmp_path, raises):
    import kb.generation_answer_finalize_runtime as finalize
    import kb.task_runtime as runtime

    monkeypatch.setenv("KB_TEST_ISOLATED", "1")

    def candidate(answer, **kwargs):
        kwargs["answer_hits"][0]["text"] = "mutated"
        if raises:
            raise ValueError("failed candidate")
        return {"answer": answer, "answer_quality": {"minimum_ok": False}}

    # Restore every wrapped attribute after the test, including the real gates.
    for module, names in (
        (runtime, ("_finalize_generation_answer", "_apply_research_facet_quality",
                   "_build_exact_relation_grounded_fallback", "_build_scoped_not_found_answer")),
        (finalize, ("audit_and_repair_claim_evidence", "_normalize_citation_plan_supported_terms",
                    "_build_answer_quality_probe")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(runtime, "_finalize_generation_answer", candidate)
    path = tmp_path / "trace.jsonl"
    install(path)
    kwargs = {"answer_hits": [{"text": "original"}], "settings_obj": "private-key-must-not-appear"}
    if raises:
        with pytest.raises(ValueError, match="failed candidate"):
            runtime._finalize_generation_answer("raw draft", **kwargs)
    else:
        assert runtime._finalize_generation_answer("raw draft", **kwargs)["answer"] == "raw draft"
    raw = path.read_text(encoding="utf8")
    assert "private-key" not in raw and "settings_obj" not in raw
    row = json.loads(raw)
    assert row["args"] == ["raw draft"]
    assert row["kwargs"]["answer_hits"][0]["text"] == "original"
    assert ("error" in row) is raises
