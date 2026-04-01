import kb.paper_guide_direct_answer_runtime as direct_answer_runtime
from kb.paper_guide_direct_answer_runtime import _build_paper_guide_direct_answer_override


def test_build_paper_guide_direct_answer_override_prefers_abstract_path():
    calls = {}

    def _build_direct_abstract_answer(**kwargs):
        calls["abstract"] = kwargs
        return "ABSTRACT"

    def _build_direct_citation_lookup_answer(**kwargs):
        calls["citation"] = kwargs
        return "CITATION"

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="abstract",
        prompt_for_user="把摘要原文给出并翻译",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="",
        db_dir="db",
        llm="llm-client",
        build_direct_abstract_answer=_build_direct_abstract_answer,
        build_direct_citation_lookup_answer=_build_direct_citation_lookup_answer,
    )

    assert out == "ABSTRACT"
    assert calls["abstract"]["source_path"] == "direct.md"
    assert calls["abstract"]["llm"] == "llm-client"
    assert "citation" not in calls


def test_build_paper_guide_direct_answer_override_uses_citation_lookup_without_explicit_target():
    calls = {}

    def _build_direct_abstract_answer(**kwargs):
        calls["abstract"] = kwargs
        return "ABSTRACT"

    def _build_direct_citation_lookup_answer(**kwargs):
        calls["citation"] = kwargs
        return "CITATION"

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="citation_lookup",
        prompt_for_user="Which references are cited for RVT, and where is that stated exactly?",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[{"meta": {"source_path": "focus.md"}}],
        special_focus_block="focus",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=_build_direct_abstract_answer,
        build_direct_citation_lookup_answer=_build_direct_citation_lookup_answer,
    )

    assert out == "CITATION"
    assert calls["citation"]["source_path"] == "focus.md"
    assert calls["citation"]["special_focus_block"] == "focus"
    assert "abstract" not in calls


def test_build_paper_guide_direct_answer_override_skips_citation_lookup_when_targeted():
    calls = {}

    def _build_direct_abstract_answer(**kwargs):
        calls["abstract"] = kwargs
        return "ABSTRACT"

    def _build_direct_citation_lookup_answer(**kwargs):
        calls["citation"] = kwargs
        return "CITATION"

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="citation_lookup",
        prompt_for_user="From Figure 2 only, which references are cited for RVT?",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="focus",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=_build_direct_abstract_answer,
        build_direct_citation_lookup_answer=_build_direct_citation_lookup_answer,
    )

    assert out == ""
    assert calls == {}


def test_build_paper_guide_direct_answer_override_supports_exact_method_prompt_in_reproduce_family(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_resolve_exact_method_support_from_source",
        lambda source_path, **_kwargs: {
            "heading_path": "Experimental setup",
            "locate_anchor": "Thus, the beat frequency of these two beams is 62,500 Hz. The data acquisition card uses a sampling rate of 1.25 Ms/s.",
            "source_path": source_path,
        },
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="reproduce",
        prompt_for_user="In the Experimental setup section, what beat frequency do they use, and what sampling rate does the data acquisition card use? Point me to the exact supporting sentence.",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    assert "Experimental setup" in out
    assert "62,500 Hz" in out
    assert "1.25 Ms/s" in out


def test_build_paper_guide_direct_answer_override_supports_exact_equation_prompt(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_resolve_exact_equation_support_from_source",
        lambda source_path, **_kwargs: {
            "source_path": source_path,
            "heading_path": "3. Method / 3.1. Background on NeRF",
            "equation_number": 1,
            "equation_markdown": "$$\nC(r)=... \\tag{1}\n$$",
            "explanation_text": "where t_n and t_f are near and far bounds.",
        },
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="equation",
        prompt_for_user=(
            "What does Equation (1) define in this paper, and where do the authors define the variables "
            "like t_n and t_f? Point me to the exact supporting part."
        ),
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    assert out == "Equation support (resolving exact equation + variable definitions)..."
