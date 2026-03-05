from __future__ import annotations

from pathlib import Path

import kb.retrieval_engine as retrieval_engine
from kb.retrieval_engine import (
    _extract_explicit_anchor_hint,
    _group_hits_by_doc_for_refs,
    _postprocess_refs_pack,
)


def test_extract_explicit_anchor_hint_supports_figure_equation_and_theorem():
    fig = _extract_explicit_anchor_hint("LPR-2025.pdf这篇文章的第三张图讲了啥")
    assert fig["kind"] == "figure"
    assert fig["number"] == 3

    eq = _extract_explicit_anchor_hint("请解释这篇论文里的公式(11)")
    assert eq["kind"] == "equation"
    assert eq["number"] == 11

    thm = _extract_explicit_anchor_hint("what does theorem 2 mean in this paper")
    assert thm["kind"] == "theorem"
    assert thm["number"] == 2


def test_group_hits_by_doc_for_refs_prioritizes_anchor_snippet_for_explicit_doc(tmp_path: Path, monkeypatch):
    md = tmp_path / "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.md"
    md.write_text(
        "\n".join(
            [
                "# LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                "",
                "## 1. Introduction",
                "This survey reviews single-pixel imaging with deep learning and summarizes the overall motivation, scope, and background for the field.",
                "",
                "## 3. Fundamentals of Deep Learning",
                "![Figure 3](./assets/page_5_fig_2.png)",
                "**Figure 3.** The basic principles of neural networks. a) ANN. b) Convolution operation. c) Contraction network. d) Encoder-Decoder network. e) RNN. f) GAN. g) Transformer.",
                "The figure summarizes the neural-network building blocks referenced later in the survey and is the key visual explanation for this section.",
                "",
                "Equation (11) defines the reconstruction loss used by the optimization-based method.",
                "Theorem 2 gives the sufficient condition for convergence in the iterative reconstruction setting.",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval_engine, "_is_temp_source_path", lambda _src: False)

    hits_raw = [
        {
            "score": 9.0,
            "id": "h1",
            "text": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
            "meta": {
                "source_path": str(md),
                "heading_path": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
            },
        },
        {
            "score": 8.5,
            "id": "h2",
            "text": "This survey reviews single-pixel imaging with deep learning and summarizes the overall motivation, scope, and background for the field.",
            "meta": {
                "source_path": str(md),
                "heading_path": "1. Introduction",
            },
        },
    ]

    docs = _group_hits_by_doc_for_refs(
        hits_raw,
        prompt_text="LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf这篇文章的第三张图讲了啥",
        top_k_docs=3,
        deep_query="",
        deep_read=False,
        llm_rerank=False,
        settings=None,
    )

    assert len(docs) == 1
    doc = docs[0]
    assert "Figure 3" in str(doc.get("text") or "")
    meta = doc.get("meta", {}) or {}
    assert float(meta.get("explicit_doc_match_score") or 0.0) >= 6.0
    assert meta.get("anchor_target_kind") == "figure"
    assert meta.get("anchor_target_number") == 3
    show_snips = meta.get("ref_show_snippets") or []
    assert show_snips
    assert "Figure 3" in str(show_snips[0])


def test_group_hits_by_doc_for_refs_supports_latex_tagged_equation_anchor(tmp_path: Path, monkeypatch):
    md = tmp_path / "NatPhoton-2019-Principles and prospects for single-pixel imaging.md"
    md.write_text(
        "\n".join(
            [
                "# Principles and prospects for single-pixel imaging",
                "",
                "## Box 1 | The maths behind single-pixel imaging",
                "$$",
                r"\mathbf{I}_{\text{TC}} = \sum_{i=1}^N \left( \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2} \right) \tag{8}",
                "$$",
                "Equation (8) defines the total-curvature objective used in the reconstruction problem.",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval_engine, "_is_temp_source_path", lambda _src: False)

    hits_raw = [
        {
            "score": 8.8,
            "id": "h1",
            "text": "Principles and prospects for single-pixel imaging",
            "meta": {
                "source_path": str(md),
                "heading_path": "Principles and prospects for single-pixel imaging",
            },
        }
    ]

    docs = _group_hits_by_doc_for_refs(
        hits_raw,
        prompt_text="NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf里的公式8写的是什么",
        top_k_docs=3,
        deep_query="",
        deep_read=False,
        llm_rerank=False,
        settings=None,
    )

    assert len(docs) == 1
    doc = docs[0]
    text = str(doc.get("text") or "")
    assert "\\tag{8}" in text
    meta = doc.get("meta", {}) or {}
    assert meta.get("anchor_target_kind") == "equation"
    assert meta.get("anchor_target_number") == 8


def test_postprocess_refs_pack_overrides_conflicting_why_when_anchor_hit():
    docs = [
        {
            "text": "Equation content",
            "meta": {
                "source_path": "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
                "anchor_target_kind": "equation",
                "anchor_target_number": 8,
                "anchor_match_score": 15.2,
                "ref_section": "Box 1 | The maths behind single-pixel imaging",
                "ref_locs": [
                    {
                        "heading": "Box 1 | The maths behind single-pixel imaging",
                        "heading_path": "Box 1 | The maths behind single-pixel imaging",
                        "score": 10.0,
                        "quality": "high",
                    }
                ],
            },
        }
    ]
    result = {
        1: {
            "score": 82.0,
            "what": "这篇文献讨论单像素成像的数学基础。",
            "why": "问题询问公式8，但文档片段中未直接给出该公式表达式，因此无法确认。",
            "start": "",
            "gain": "只能提供部分背景。",
            "find": [],
            "section": "",
        }
    }
    out = _postprocess_refs_pack(result, docs, question="NatPhoton-2019这篇文章里公式8是什么")
    why = str(out[1]["why"] or "")
    assert "未直接给出" not in why
    assert "无法确认" not in why
    assert "公式8" in why
