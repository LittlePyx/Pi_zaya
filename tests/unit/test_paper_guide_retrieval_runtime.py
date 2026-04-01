from pathlib import Path

from kb import paper_guide_retrieval_runtime as retrieval_runtime
from kb.paper_guide_retrieval_runtime import (
    _build_paper_guide_direct_citation_lookup_answer,
    _extract_paper_guide_local_citation_lookup_refs,
    _filter_hits_for_paper_guide,
    _paper_guide_fallback_deepread_hits,
    _paper_guide_citation_lookup_fragments,
    _paper_guide_citation_lookup_query_tokens,
    _paper_guide_citation_lookup_signal_score,
    _paper_guide_deepread_heading,
    _paper_guide_has_requested_target_hits,
    _paper_guide_should_force_rescue,
    _select_paper_guide_raw_target_hits,
    _paper_guide_targeted_source_block_hits,
    _select_paper_guide_deepread_extras,
)
from tests._paper_guide_fixtures import build_paper_guide_runtime_fixture


def test_paper_guide_deepread_heading_prefers_meta_heading_then_markdown_header():
    assert _paper_guide_deepread_heading({"meta": {"heading_path": "Methods > APR"}}) == "Methods > APR"
    assert _paper_guide_deepread_heading({"text": "# Discussion\nFuture work."}) == "Discussion"


def test_select_paper_guide_deepread_extras_skips_reference_like_snippets_for_abstract():
    out = _select_paper_guide_deepread_extras(
        [
            {"text": "## References\n[34] Smith et al. 2020.", "score": 99},
            {"text": "# Abstract\nWe used low illumination power.", "score": 10},
        ],
        prompt="Show the abstract and translate it.",
        prompt_family="abstract",
        limit=1,
    )
    assert out == ["# Abstract\nWe used low illumination power."]


def test_paper_guide_has_requested_target_hits_detects_box_hit():
    assert _paper_guide_has_requested_target_hits(
        [{"meta": {"heading_path": "Box 1"}, "text": "Box 1. A transform-domain condition is stated here."}],
        prompt="From Box 1 only, what condition on M is given?",
    )


def test_filter_hits_for_paper_guide_keeps_only_bound_source_matches():
    hits = [
        {"meta": {"source_path": "paper_a.md"}, "text": "A"},
        {"meta": {"source_path": "paper_b.md"}, "text": "B"},
    ]
    out = _filter_hits_for_paper_guide(
        hits,
        bound_source_path="paper_a.md",
        bound_source_name="paper_a",
    )
    assert [str((item.get("meta") or {}).get("source_path") or "") for item in out] == ["paper_a.md"]


def test_paper_guide_targeted_source_block_hits_extracts_box_block(tmp_path: Path):
    fixture = build_paper_guide_runtime_fixture(tmp_path)
    hits = _paper_guide_targeted_source_block_hits(
        bound_source_path=str(fixture["nat_source_path"]),
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
        db_dir=Path(fixture["db_root"]),
        limit=4,
        citation_lookup_query_tokens=lambda prompt: [tok for tok in prompt.lower().split() if tok],
        citation_lookup_signal_score=lambda **_kwargs: 0.0,
        resolve_support_slot_block=lambda **_kwargs: {"heading_path": "Box 1", "block_id": "blk_box1", "anchor_id": "a1"},
    )
    assert hits
    assert any("Box 1" in str(hit.get("text") or "") for hit in hits)
    assert any("M \\ge O(K \\log(N/K))" in str(hit.get("text") or "") for hit in hits)


def test_paper_guide_fallback_deepread_hits_prefers_targeted_hits_for_box_query(tmp_path: Path):
    fixture = build_paper_guide_runtime_fixture(tmp_path)
    hits = _paper_guide_fallback_deepread_hits(
        bound_source_path=str(fixture["nat_source_path"]),
        bound_source_name="NatPhoton-2019-Principles and prospects for single-pixel imaging",
        query="box 1 transform domain condition",
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
        prompt_family="overview",
        top_k=3,
        db_dir=Path(fixture["db_root"]),
        citation_lookup_query_tokens=lambda prompt: [tok for tok in prompt.lower().split() if tok],
        citation_lookup_signal_score=lambda **_kwargs: 0.0,
        resolve_support_slot_block=lambda **_kwargs: {"heading_path": "Box 1", "block_id": "blk_box1", "anchor_id": "a1"},
    )
    assert hits
    assert all(bool((hit.get("meta") or {}).get("paper_guide_targeted_block")) for hit in hits)


def test_paper_guide_targeted_source_block_hits_uses_family_seed_tokens_for_cjk_prompt(tmp_path: Path):
    source_pdf = tmp_path / "DemoMethod.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    db_root = tmp_path / "db"
    md_dir = db_root / "DemoMethod"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoMethod.en.md"
    md_main.write_text(
        (
            "## Materials and Methods\n\n"
            "APR was performed using image registration based on phase correlation.\n\n"
            "## Results\n\n"
            "The method improves CNR and sectioning.\n"
        ),
        encoding="utf-8",
    )

    hits = _paper_guide_targeted_source_block_hits(
        bound_source_path=str(source_pdf),
        prompt="这个方法具体怎么实现？请给出关键步骤。",
        db_dir=db_root,
        limit=3,
        citation_lookup_query_tokens=lambda prompt: [tok for tok in prompt.lower().split() if tok],
        citation_lookup_signal_score=lambda **_kwargs: 0.0,
        resolve_support_slot_block=lambda **_kwargs: {"heading_path": "Materials and Methods", "block_id": "blk_method", "anchor_id": "a1"},
    )
    assert hits
    assert any("methods" in str((hit.get("meta") or {}).get("heading_path") or "").lower() for hit in hits)


def test_seed_query_tokens_merges_augmented_family_terms_when_prompt_has_cjk_tokens():
    tokens = retrieval_runtime._paper_guide_seed_query_tokens_for_targeted_scan(
        prompt="请解释这个方法的关键步骤",
        family="method",
        bound_source_path=r"X:\demo.pdf",
    )
    assert {"algorithm", "analysis", "apr"}.intersection(tokens)


def test_paper_guide_should_force_rescue_for_exact_method_when_only_generic_hits_exist():
    hits = [
        {
            "score": 8.0,
            "text": "This paper studies neural rendering from compressive imaging.",
            "meta": {
                "source_path": r"db\demo\paper.en.md",
                "heading_path": "Introduction",
                "block_id": "blk_intro",
            },
        }
    ]

    assert _paper_guide_should_force_rescue(
        scoped_hits=hits,
        prompt=(
            "In the Implementation details paragraph, how many iterations do they train for, "
            "and what is the batch size measured in rays? Point me to the exact supporting sentence."
        ),
        prompt_family="method",
    )


def test_paper_guide_should_force_rescue_skips_when_targeted_hit_already_exists():
    hits = [
        {
            "score": 15.0,
            "text": "We train for 100-200K iterations, with 5000 rays as batch size.",
            "meta": {
                "source_path": r"db\demo\paper.en.md",
                "heading_path": "Experiments / Implementation details",
                "block_id": "blk_impl",
                "paper_guide_targeted_block": True,
            },
        }
    ]

    assert not _paper_guide_should_force_rescue(
        scoped_hits=hits,
        prompt=(
            "In the Implementation details paragraph, how many iterations do they train for, "
            "and what is the batch size measured in rays?"
        ),
        prompt_family="method",
    )


def test_paper_guide_should_force_rescue_for_overview_when_only_reference_hits_exist():
    hits = [
        {
            "score": 9.0,
            "text": "[12] Doe et al. A method for imaging.",
            "meta": {
                "source_path": r"db\demo\paper.en.md",
                "heading_path": "References",
                "block_id": "blk_ref_12",
            },
        },
        {
            "score": 8.4,
            "text": "[13] Roe et al. Another related work.",
            "meta": {
                "source_path": r"db\demo\paper.en.md",
                "heading_path": "References",
                "block_id": "blk_ref_13",
            },
        },
    ]

    assert _paper_guide_should_force_rescue(
        scoped_hits=hits,
        prompt="What are the core contributions of this paper?",
        prompt_family="overview",
    )


def test_paper_guide_fallback_deepread_hits_merges_original_and_translated_targeted_scans(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        retrieval_runtime,
        "_resolve_paper_guide_md_path",
        lambda *_args, **_kwargs: Path(r"db\demo\paper.en.md"),
    )

    def _fake_targeted(*, prompt: str, **_kwargs):
        calls.append(prompt)
        if prompt == "原始问题":
            return [
                {
                    "text": "原始问题命中了方法段。",
                    "score": 13.0,
                    "meta": {"block_id": "blk_orig", "heading_path": "Methods"},
                }
            ]
        if prompt == "translated query":
            return [
                {
                    "text": "Translated query matched the implementation details.",
                    "score": 15.0,
                    "meta": {"block_id": "blk_trans", "heading_path": "Experiments / Implementation details"},
                }
            ]
        return []

    monkeypatch.setattr(
        retrieval_runtime,
        "_paper_guide_targeted_source_block_hits",
        _fake_targeted,
    )

    out = _paper_guide_fallback_deepread_hits(
        bound_source_path=r"X:\demo.pdf",
        bound_source_name="Demo",
        query="translated query",
        prompt="原始问题",
        prompt_family="method",
        top_k=4,
        db_dir=Path("db"),
    )

    assert calls == ["原始问题", "translated query"]
    assert {str((hit.get("meta") or {}).get("block_id") or "") for hit in out} == {"blk_orig", "blk_trans"}


def test_paper_guide_fallback_deepread_hits_rebinds_block_anchor_for_locate(monkeypatch, tmp_path: Path):
    source_pdf = tmp_path / "DemoMethod.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    db_root = tmp_path / "db"
    md_dir = db_root / "DemoMethod"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoMethod.en.md"
    md_main.write_text(
        (
            "## Materials and Methods\n\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images.\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        retrieval_runtime,
        "_paper_guide_targeted_source_block_hits",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        retrieval_runtime,
        "_deep_read_md_for_context",
        lambda *_args, **_kwargs: [
            {
                "text": "APR was performed using image registration based on phase correlation of the off-axis raw images.",
                "score": 1.0,
                "meta": {},
            }
        ],
    )

    out = _paper_guide_fallback_deepread_hits(
        bound_source_path=str(source_pdf),
        bound_source_name="DemoMethod",
        query="这个方法具体怎么实现？",
        prompt="这个方法具体怎么实现？",
        prompt_family="method",
        top_k=2,
        db_dir=db_root,
        citation_lookup_query_tokens=lambda prompt: [tok for tok in prompt.lower().split() if tok],
        citation_lookup_signal_score=lambda **_kwargs: 0.0,
        resolve_support_slot_block=lambda **_kwargs: {"heading_path": "Materials and Methods", "block_id": "blk_method", "anchor_id": "a1"},
    )
    assert out
    meta = out[0].get("meta", {}) or {}
    assert str(meta.get("block_id") or "").strip()
    assert str(meta.get("anchor_id") or "").strip()


def test_paper_guide_citation_lookup_query_tokens_drops_stopwords():
    toks = _paper_guide_citation_lookup_query_tokens(
        "Which prior work is RVT attributed to in this paper, and what in-paper citation do they use?"
    )
    assert "rvt" in toks
    assert "which" not in toks
    assert "paper" not in toks


def test_extract_paper_guide_local_citation_lookup_refs_prefers_token_adjacent_refs():
    refs = _extract_paper_guide_local_citation_lookup_refs(
        "Specifically, we use the radial variance transform (RVT)[34], which converts an interferogram into an intensity-only map.",
        prompt="Which prior work is RVT attributed to in this paper?",
    )
    assert refs == [34]


def test_select_paper_guide_raw_target_hits_prefers_intext_citation_lookup_hit():
    hits = [
        {
            "score": 12.0,
            "text": "[33] Richardson, W. H. Bayesian-based iterative method of image restoration.",
            "meta": {
                "source_path": r"db\demo\paper.en.md",
                "heading_path": "References",
                "block_id": "blk_refs",
            },
        },
        {
            "score": 10.0,
            "text": "Specifically, we use the radial variance transform (RVT)[34], which converts an interferogram into an intensity-only map.",
            "meta": {
                "source_path": r"db\demo\paper.en.md",
                "heading_path": "Methods / RVT",
                "block_id": "blk_rvt_intro",
                "paper_guide_targeted_block": True,
            },
        },
    ]
    out = _select_paper_guide_raw_target_hits(
        hits_raw=hits,
        prompt="Which prior work is RVT attributed to in this paper, and what in-paper citation do they use when introducing it?",
        top_n=1,
        answer_hit_score=lambda hit, *, prompt: float(hit.get("score") or 0.0),
    )
    assert len(out) == 1
    assert str((out[0].get("meta") or {}).get("block_id") or "") == "blk_rvt_intro"


def test_build_paper_guide_direct_citation_lookup_answer_uses_reference_lookup():
    out = _build_paper_guide_direct_citation_lookup_answer(
        prompt="Which prior work is RVT attributed to in this paper, and what in-paper citation do they use when introducing it?",
        source_path=r"X:\demo.pdf",
        answer_hits=[
            {
                "text": "Specifically, we use the radial variance transform (RVT)[34], which converts an interferogram into an intensity-only map.",
                "meta": {"heading_path": "Results / APR"},
            }
        ],
        special_focus_block="",
        db_dir=Path("db"),
        extract_special_focus_excerpt=lambda block: "",
        reference_entry_lookup=lambda _src, ref_num, **_kwargs: {
            "title": "Precision single-particle localization using radial variance transform"
        }
        if int(ref_num) == 34
        else {},
    )
    assert "[34]" in out
    assert "Precision single-particle localization using radial variance transform" in out


def test_build_paper_guide_direct_citation_lookup_answer_prefers_author_attribution_ref_span():
    out = _build_paper_guide_direct_citation_lookup_answer(
        prompt="Which reference do the authors cite for single-pixel imaging via compressive sampling, and where is that stated exactly?",
        source_path=r"X:\demo.pdf",
        answer_hits=[
            {
                "text": (
                    "The original concept of the single-pixel imaging approach, demonstrated by Sen et al.$^{3,58}$, "
                    "was developed further in conjunction with compressive sensing$^{59}$ and reported soon after in a seminal paper by Duarte et al. at Rice University$^{4}$."
                ),
                "meta": {"heading_path": "Acquisition and image reconstruction strategies"},
            }
        ],
        special_focus_block="",
        db_dir=Path("db"),
        extract_special_focus_excerpt=lambda block: "",
        reference_entry_lookup=lambda _src, ref_num, **_kwargs: {
            "title": "Single-pixel imaging via compressive sampling"
        }
        if int(ref_num) == 4
        else {
            "title": "Sparsity and incoherence in compressive sampling"
        },
    )

    assert "The paper cites [4]" in out
    assert "Single-pixel imaging via compressive sampling" in out


def test_build_paper_guide_direct_citation_lookup_answer_focuses_on_target_ref_clause_for_sci():
    out = _build_paper_guide_direct_citation_lookup_answer(
        prompt="In the Abstract, which reference do they cite for video Snapshot Compressive Imaging (SCI), and where is it stated exactly?",
        source_path=r"X:\demo.pdf",
        answer_hits=[
            {
                "text": (
                    "Drawing inspiration from Compressed Sensing (CS) [5,8], "
                    "video Snapshot Compressive Imaging (SCI) [50] system has emerged to address these limitations."
                ),
                "meta": {"heading_path": "Abstract"},
            }
        ],
        special_focus_block="",
        db_dir=Path("db"),
        extract_special_focus_excerpt=lambda block: "",
        reference_entry_lookup=lambda _src, ref_num, **_kwargs: {
            "title": "Snapshot Compressive Imaging"
        }
        if int(ref_num) == 50
        else {"title": f"Reference {int(ref_num)}"},
    )

    assert "The paper cites [50]" in out
    assert "video Snapshot Compressive Imaging (SCI) [50]" in out
    assert "[5,8]" not in out


def test_build_paper_guide_direct_citation_lookup_answer_focuses_on_target_ref_clause_for_admm():
    out = _build_paper_guide_direct_citation_lookup_answer(
        prompt="In Related Work, which reference is cited for ADMM, and where is it stated exactly?",
        source_path=r"X:\demo.pdf",
        answer_hits=[
            {
                "text": (
                    "In SCI, different regularizers and priors have been used, including sparsity [47] and total variation (TV) [49]. "
                    "When solving the optimization problems, most methods employ alternating direction method of multipliers (ADMM) [4], "
                    "which leads to good results."
                ),
                "meta": {"heading_path": "2. Related Work"},
            }
        ],
        special_focus_block="",
        db_dir=Path("db"),
        extract_special_focus_excerpt=lambda block: "",
        reference_entry_lookup=lambda _src, ref_num, **_kwargs: {
            "title": "ADMM classic work"
        }
        if int(ref_num) == 4
        else {"title": f"Reference {int(ref_num)}"},
    )

    assert "The paper cites [4]" in out
    assert "ADMM) [4]" in out
    assert "[49]" not in out


def test_paper_guide_citation_lookup_signal_score_penalizes_reference_list_for_intext_query():
    intext_score = _paper_guide_citation_lookup_signal_score(
        prompt="Which prior work is RVT attributed to in this paper?",
        heading="Methods / RVT",
        text="Specifically, we use the radial variance transform (RVT)[34], which converts an interferogram into an intensity-only map.",
        inline_refs=[34],
        explicit_ref_list_request=False,
    )
    ref_score = _paper_guide_citation_lookup_signal_score(
        prompt="Which prior work is RVT attributed to in this paper?",
        heading="References",
        text="[34] Smith et al. Precision single-particle localization using radial variance transform.",
        inline_refs=[34],
        explicit_ref_list_request=False,
    )
    assert intext_score > ref_score
