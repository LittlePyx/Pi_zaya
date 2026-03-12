from __future__ import annotations

from pathlib import Path


def test_resolve_paper_guide_md_path_accepts_db_dir_as_md_root(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "papers" / "DemoPaper.pdf"
    md_root = tmp_path / "custom_md_root"
    md_dir = md_root / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text("# Demo\n\nParagraph evidence.", encoding="utf-8")

    resolved = task_runtime._resolve_paper_guide_md_path(str(source_pdf), db_dir=md_root)

    assert resolved is not None
    assert resolved.resolve(strict=False) == md_main.resolve(strict=False)


def test_segment_snippet_aliases_yields_sentence_level_aliases():
    from kb import task_runtime

    text = (
        "This first sentence is intentionally long enough for alias extraction. "
        "The second sentence adds extra context for matching behavior. "
        "A short tail."
    )

    aliases = task_runtime._segment_snippet_aliases(text)

    assert aliases
    assert len(aliases) <= 6
    assert aliases[0].startswith("this first sentence is intentionally")
    assert any("the second sentence adds extra context" in item for item in aliases)


def test_build_paper_guide_answer_provenance_contains_snippet_aliases(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "DemoPaper.pdf"
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    block_text = "Random mask encoding is used for compressive measurements."
    md_main.write_text(f"# Method\n\n{block_text}\n", encoding="utf-8")

    answer = block_text
    hits = [
        {
            "text": block_text,
            "meta": {
                "source_path": str(source_pdf),
                "ref_show_snippets": [block_text],
                "ref_best_heading_path": "Method",
            },
        }
    ]

    provenance = task_runtime._build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=str(source_pdf),
        bound_source_name="DemoPaper.pdf",
        db_dir=None,
        llm_rerank=False,
    )

    assert isinstance(provenance, dict)
    assert provenance.get("status") == "ready"
    assert provenance.get("mapping_mode") == "fast"
    assert int(provenance.get("llm_rerank_calls") or 0) == 0
    assert bool(provenance.get("llm_rerank_enabled")) is False
    segments = provenance.get("segments") or []
    assert segments
    assert isinstance(segments[0].get("snippet_aliases"), list)
    assert segments[0].get("snippet_aliases")
    assert str(segments[0].get("raw_markdown") or "").strip()
    direct_segment = next(
        (seg for seg in segments if str(seg.get("evidence_mode") or "").strip().lower() == "direct"),
        None,
    )
    assert isinstance(direct_segment, dict)
    assert str(direct_segment.get("primary_block_id") or "").strip()
    assert isinstance(direct_segment.get("support_block_ids"), list)
    assert str(direct_segment.get("evidence_quote") or "").strip()


def test_build_paper_guide_answer_provenance_prefers_supported_experiment_block_over_abstract(monkeypatch, tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "ScenePaper.pdf"
    md_dir = tmp_path / "ScenePaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "ScenePaper.en.md"
    md_main.write_text(
        (
            "# Abstract\n\n"
            "Our method exploits neural radiance fields (NeRF) to reconstruct scenes from SCI snapshots.\n\n"
            "# Experimental Setup\n\n"
            "We cannot estimate accurate poses from the SOTA outputs, so we use ground truth images to estimate camera poses instead.\n\n"
            "# Baselines\n\n"
            "NeRF+PnP-FFDNet is a two-stage pipeline: first reconstruct images, then train NeRF.\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    abstract_block = next(
        block for block in blocks
        if str(block.get("heading_path") or "").strip().lower() == "abstract"
        and "neural radiance fields" in str(block.get("text") or "").lower()
    )
    pose_block = next(
        block for block in blocks
        if "experimental setup" in str(block.get("heading_path") or "").strip().lower()
        and "ground truth images" in str(block.get("text") or "").lower()
    )

    class _DummyChat:
        def __init__(self, *_args, **_kwargs):
            pass

    def _fake_pick_blocks_with_llm(**kwargs):
        candidate_rows = kwargs.get("candidate_rows") or []
        for row in candidate_rows:
            block = row.get("block")
            if not isinstance(block, dict):
                continue
            if str(block.get("block_id") or "").strip() == str(abstract_block.get("block_id") or "").strip():
                return [str(abstract_block.get("block_id") or "").strip()]
        return []

    monkeypatch.setattr(task_runtime, "DeepSeekChat", _DummyChat)
    monkeypatch.setattr(task_runtime, "_pick_blocks_with_llm", _fake_pick_blocks_with_llm)

    answer = (
        "The paper says pose is not estimated from the SOTA reconstructed outputs; "
        "instead it uses ground truth images to estimate camera poses."
    )
    hits = [
        {
            "text": str(abstract_block.get("text") or ""),
            "meta": {
                "source_path": str(source_pdf),
                "ref_show_snippets": [
                    str(abstract_block.get("text") or ""),
                    str(pose_block.get("text") or ""),
                ],
                "ref_best_heading_path": "Abstract",
                "ref_locs": [
                    {"heading_path": "Experimental Setup"},
                ],
            },
        }
    ]

    provenance = task_runtime._build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=str(source_pdf),
        bound_source_name="ScenePaper.pdf",
        db_dir=None,
        settings_obj=object(),
        llm_rerank=True,
    )

    assert isinstance(provenance, dict)
    segments = provenance.get("segments") or []
    assert segments
    direct_segment = next(
        (seg for seg in segments if str(seg.get("evidence_mode") or "").strip().lower() == "direct"),
        None,
    )
    assert isinstance(direct_segment, dict)
    assert str(direct_segment.get("primary_block_id") or "").strip() == str(pose_block.get("block_id") or "").strip()
    assert str(direct_segment.get("primary_block_id") or "").strip() != str(abstract_block.get("block_id") or "").strip()
    assert "ground truth images" in str(direct_segment.get("evidence_quote") or "").lower()
    assert float(direct_segment.get("mapping_quality") or 0.0) >= 0.24
    assert str(direct_segment.get("mapping_source") or "").strip() in {"fast", "llm_refined"}


def test_build_paper_guide_answer_provenance_prefers_contribution_block_over_definition_block(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "SCIGS.pdf"
    md_dir = tmp_path / "SCIGS"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "SCIGS.en.md"
    contribution = (
        "The proposed SCIGS is the first to recover explicit 3D representations from a single "
        "snapshot compressed image within the 3D Gaussian Splating (3DGS) framework."
    )
    definition = "As the rendering primitive for 3DGS, a 3D Gaussian is defined as:"
    md_main.write_text(
        (
            "# Related Work\n\n"
            "Our main contributions can be summarized as follows:\n\n"
            f"- {contribution}\n\n"
            "# Method\n\n"
            f"{definition}\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    contribution_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "list_item"
        and "first to recover explicit 3d representations" in str(block.get("text") or "").lower()
    )
    definition_block = next(
        block for block in blocks
        if definition.lower() in str(block.get("text") or "").lower()
    )

    answer = (
        "首次实现从单帧快照压缩图像（SCI）重建显式3D高斯表示："
        "SCIGS 是首个在 3D Gaussian Splatting（3DGS）框架下，仅凭一张快照压缩图像"
        "即可恢复显式 3D 场景表示的方法。"
    )
    long_hit = (
        "Our main contributions can be summarized as follows:\n"
        f"- {contribution}\n"
        "- Introducing camera pose stamps and a Gaussian primitive-level transformation network.\n"
    )
    hits = [
        {
            "text": long_hit,
            "meta": {
                "source_path": str(source_pdf),
                "ref_show_snippets": [long_hit],
                "ref_best_heading_path": "Related Work",
            },
        }
    ]

    provenance = task_runtime._build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=str(source_pdf),
        bound_source_name="SCIGS.pdf",
        db_dir=None,
        llm_rerank=False,
    )

    segments = provenance.get("segments") or []
    direct_segment = next(
        (seg for seg in segments if str(seg.get("evidence_mode") or "").strip().lower() == "direct"),
        None,
    )
    assert isinstance(direct_segment, dict)
    assert str(direct_segment.get("primary_block_id") or "").strip() == str(contribution_block.get("block_id") or "")
    assert str(direct_segment.get("primary_block_id") or "").strip() != str(definition_block.get("block_id") or "")
    assert "first to recover explicit 3d representations" in str(direct_segment.get("evidence_quote") or "").lower()
    assert bool(direct_segment.get("must_locate")) is True


def test_build_paper_guide_answer_provenance_marks_quote_claim_as_must_locate(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "ScenePaper.pdf"
    md_dir = tmp_path / "ScenePaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "ScenePaper.en.md"
    md_main.write_text(
        (
            "# Abstract\n\n"
            "Our method exploits neural radiance fields (NeRF).\n\n"
            "# Experimental Setup\n\n"
            "We cannot estimate accurate poses from the SOTA outputs, so we use ground truth images to estimate camera poses instead.\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    pose_block = next(
        block for block in blocks
        if "ground truth images" in str(block.get("text") or "").lower()
    )

    answer = (
        '文中提到：“We cannot estimate accurate poses from the SOTA outputs, '
        'so we use ground truth images to estimate camera poses instead.” 说明：'
    )
    hits = [
        {
            "text": "Our method exploits neural radiance fields (NeRF).",
            "meta": {
                "source_path": str(source_pdf),
                "ref_show_snippets": ["Our method exploits neural radiance fields (NeRF)."],
                "ref_best_heading_path": "Abstract",
            },
        }
    ]

    provenance = task_runtime._build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=str(source_pdf),
        bound_source_name="ScenePaper.pdf",
        db_dir=None,
        llm_rerank=False,
    )

    segments = provenance.get("segments") or []
    assert segments
    seg = segments[0]
    assert str(seg.get("claim_type") or "") == "quote_claim"
    assert bool(seg.get("must_locate")) is True
    assert str(seg.get("anchor_kind") or "") == "quote"
    assert "ground truth images" in str(seg.get("anchor_text") or "").lower()
    assert str(seg.get("primary_block_id") or "") == str(pose_block.get("block_id") or "")


def test_build_paper_guide_answer_provenance_marks_blockquote_claim_as_must_locate(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "ScenePaper.pdf"
    md_dir = tmp_path / "ScenePaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "ScenePaper.en.md"
    quote_text = (
        "We noticed that due to the lack of high-quality details and multi-view consistency, "
        "we cannot estimate camera poses via SfM using the images reconstructed from those SCI image reconstruction methods."
    )
    md_main.write_text(f"# Experimental Setup\n\n{quote_text}\n", encoding="utf-8")
    blocks = task_runtime.load_source_blocks(md_main)
    quote_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "paragraph"
        and "multi-view consistency" in str(block.get("text") or "").lower()
    )

    answer = f"> {quote_text}"
    hits = [
        {
            "text": quote_text,
            "meta": {
                "source_path": str(source_pdf),
                "ref_show_snippets": [quote_text],
                "ref_best_heading_path": "Experimental Setup",
            },
        }
    ]

    provenance = task_runtime._build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=str(source_pdf),
        bound_source_name="ScenePaper.pdf",
        db_dir=None,
        llm_rerank=False,
    )

    segments = provenance.get("segments") or []
    assert segments
    seg = segments[0]
    assert int(provenance.get("provenance_schema_version") or 0) == 4
    assert bool(provenance.get("strict_identity_ready")) is True
    assert int(provenance.get("must_locate_count") or 0) >= 1
    assert int(provenance.get("strict_identity_count") or 0) >= 1
    assert str(seg.get("kind") or "") == "blockquote"
    assert str(seg.get("claim_type") or "") == "blockquote_claim"
    assert bool(seg.get("must_locate")) is True
    assert str(seg.get("primary_block_id") or "") == str(quote_block.get("block_id") or "")
    assert list(seg.get("evidence_block_ids") or []) == [str(quote_block.get("block_id") or "")]
    assert str(seg.get("anchor_kind") or "") == "blockquote"
    assert "multi-view consistency" in str(seg.get("anchor_text") or "").lower()
    assert "multi-view consistency" in str(seg.get("evidence_quote") or "").lower()


def test_build_paper_guide_answer_provenance_marks_formula_claim_as_must_locate(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "ImagingPaper.pdf"
    md_dir = tmp_path / "ImagingPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "ImagingPaper.en.md"
    formula = "$$Y = \\sum_{i=1}^{N} X_i \\odot M_i + Z \\tag{3}$$"
    md_main.write_text(
        (
            "# Method\n\n"
            f"{formula}\n\n"
            "where M_i is the DMD binary pattern for the i-th frame.\n"
        ),
        encoding="utf-8",
    )
    blocks = task_runtime.load_source_blocks(md_main)
    formula_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "equation"
        and int(block.get("number") or 0) == 3
    )

    answer = "公式(3) 给出成像模型：\n\n" + formula
    hits = [
        {
            "text": formula,
            "meta": {
                "source_path": str(source_pdf),
                "ref_show_snippets": [formula],
                "ref_best_heading_path": "Method",
                "anchor_target_kind": "equation",
                "anchor_target_number": 3,
            },
        }
    ]

    provenance = task_runtime._build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=str(source_pdf),
        bound_source_name="ImagingPaper.pdf",
        db_dir=None,
        llm_rerank=False,
    )

    segments = provenance.get("segments") or []
    direct_segments = [seg for seg in segments if str(seg.get("evidence_mode") or "") == "direct"]
    assert direct_segments
    formula_seg = next(seg for seg in direct_segments if str(seg.get("claim_type") or "") == "formula_claim")
    assert int(provenance.get("provenance_schema_version") or 0) == 4
    assert bool(provenance.get("strict_identity_ready")) is True
    assert int(provenance.get("must_locate_count") or 0) >= 1
    assert int(provenance.get("strict_identity_count") or 0) >= 1
    assert bool(formula_seg.get("must_locate")) is True
    assert str(formula_seg.get("primary_block_id") or "") == str(formula_block.get("block_id") or "")
    assert list(formula_seg.get("evidence_block_ids") or []) == [str(formula_block.get("block_id") or "")]
    assert str(formula_seg.get("anchor_kind") or "") == "equation"
    assert int(formula_seg.get("equation_number") or 0) == 3
    assert "X_i" in str(formula_seg.get("anchor_text") or "")
    assert str(formula_seg.get("anchor_text") or "").strip()


def test_segment_claim_meta_does_not_upgrade_explicit_generic_formula_supplement():
    from kb import task_runtime

    claim = task_runtime._segment_claim_meta(
        segment_text=(
            "补充说明（通用知识，非检索片段内容）： "
            "标准 NeRF 的体渲染公式为： $$\\hat{C}(\\mathbf{r}) = \\int_{t_n}^{t_f} T(t) \\sigma(\\mathbf{x}(t)) \\mathbf{c}(\\mathbf{x}(t), \\mathbf{d}) \\, dt$$ "
            "但该公式未出现在本文检索片段中。"
        ),
        raw_markdown=(
            "补充说明（通用知识，非检索片段内容）：\n"
            "$$\\hat{C}(\\mathbf{r}) = \\int_{t_n}^{t_f} T(t) \\sigma(\\mathbf{x}(t)) \\mathbf{c}(\\mathbf{x}(t), \\mathbf{d}) \\, dt$$\n"
            "但该公式未出现在本文检索片段中。"
        ),
        segment_kind="paragraph",
        evidence_mode="direct",
        primary_block={"block_id": "blk_eq7", "kind": "equation", "number": 7},
        evidence_quote="\\hat{C}(\\mathbf{r}) = \\int_{t_n}^{t_f} ...",
        mapping_quality=0.92,
    )

    assert str(claim.get("claim_type") or "") == "shell_sentence"
    assert bool(claim.get("must_locate")) is False
    assert str(claim.get("anchor_kind") or "") == ""


def test_apply_provenance_required_coverage_contract_hides_explicit_generic_formula_segment():
    from kb import task_runtime

    segments = task_runtime._apply_provenance_required_coverage_contract(
        [
            {
                "segment_id": "seg_008",
                "segment_index": 8,
                "kind": "paragraph",
                "segment_type": "fact",
                "text": (
                    "补充说明（通用知识，非检索片段内容）： "
                    "标准 NeRF 的体渲染公式为： $$\\hat{C}(\\mathbf{r}) = \\int_{t_n}^{t_f} ...$$ "
                    "但该公式未出现在本文检索片段中。"
                ),
                "raw_markdown": (
                    "补充说明（通用知识，非检索片段内容）：\n"
                    "$$\\hat{C}(\\mathbf{r}) = \\int_{t_n}^{t_f} ...$$\n"
                    "但该公式未出现在本文检索片段中。"
                ),
                "evidence_mode": "direct",
                "claim_type": "formula_claim",
                "must_locate": True,
                "anchor_kind": "equation",
                "equation_number": 7,
                "primary_block_id": "blk_eq7",
                "primary_anchor_id": "eq_00007",
                "primary_heading_path": "Method",
                "evidence_block_ids": ["blk_eq7"],
                "support_block_ids": [],
                "anchor_text": "\\hat{C}(\\mathbf{r}) = \\int_{t_n}^{t_f} ...",
                "evidence_quote": "\\hat{C}(\\mathbf{r}) = \\int_{t_n}^{t_f} ...",
            }
        ],
        block_lookup={"blk_eq7": {"order_index": 42}},
    )

    assert len(segments) == 1
    seg = segments[0]
    assert str(seg.get("locate_policy") or "") == "hidden"
    assert bool(seg.get("must_locate")) is False
    assert str(seg.get("claim_group_id") or "") == ""
    assert str(seg.get("claim_group_kind") or "") == ""


def test_apply_provenance_required_coverage_contract_propagates_non_source_scope_to_following_formula():
    from kb import task_runtime

    segments = task_runtime._apply_provenance_required_coverage_contract(
        [
            {
                "segment_id": "seg_009",
                "segment_index": 9,
                "kind": "list_item",
                "segment_type": "bullet",
                "text": "Supplementary note (generic knowledge, non-retrieved content):",
                "raw_markdown": "### Supplementary note (generic knowledge, non-retrieved content):",
                "evidence_mode": "direct",
                "claim_type": "shell_sentence",
                "must_locate": False,
                "anchor_kind": "",
                "primary_block_id": "",
                "evidence_block_ids": [],
            },
            {
                "segment_id": "seg_010",
                "segment_index": 10,
                "kind": "paragraph",
                "segment_type": "equation_explanation",
                "text": "$$ \\hat{C}(r) = \\sum_i T_i (1 - e^{-\\sigma_i \\delta_i}) c_i $$",
                "raw_markdown": "$$ \\hat{C}(r) = \\sum_i T_i (1 - e^{-\\sigma_i \\delta_i}) c_i $$",
                "evidence_mode": "direct",
                "claim_type": "formula_claim",
                "must_locate": True,
                "anchor_kind": "equation",
                "equation_number": 0,
                "primary_block_id": "blk_generic",
                "primary_anchor_id": "eq_generic",
                "primary_heading_path": "Method / Supplement",
                "evidence_block_ids": ["blk_generic"],
                "support_block_ids": [],
                "anchor_text": "$$ \\hat{C}(r) = \\sum_i T_i (1 - e^{-\\sigma_i \\delta_i}) c_i $$",
                "evidence_quote": "$$ \\hat{C}(r) = \\sum_i T_i (1 - e^{-\\sigma_i \\delta_i}) c_i $$",
            },
            {
                "segment_id": "seg_011",
                "segment_index": 11,
                "kind": "paragraph",
                "segment_type": "claim",
                "text": "Conclusion: retrieved equation below is direct evidence.",
                "raw_markdown": "Conclusion: retrieved equation below is direct evidence.",
                "evidence_mode": "direct",
                "claim_type": "formula_claim",
                "must_locate": True,
                "anchor_kind": "equation",
                "equation_number": 3,
                "primary_block_id": "blk_real",
                "primary_anchor_id": "eq_00003",
                "primary_heading_path": "Method",
                "evidence_block_ids": ["blk_real"],
                "support_block_ids": [],
                "anchor_text": "$$ Y = \\sum_i X_i \\odot M_i + Z \\tag{3} $$",
                "evidence_quote": "$$ Y = \\sum_i X_i \\odot M_i + Z \\tag{3} $$",
            },
        ],
        block_lookup={
            "blk_generic": {"order_index": 42},
            "blk_real": {"order_index": 80},
        },
    )

    assert len(segments) == 3
    supplement_formula = segments[1]
    direct_formula = segments[2]
    assert str(supplement_formula.get("locate_policy") or "") == "hidden"
    assert bool(supplement_formula.get("must_locate")) is False
    assert str(supplement_formula.get("claim_group_id") or "") == ""
    assert str(direct_formula.get("locate_policy") or "") == "required"
    assert bool(direct_formula.get("must_locate")) is True
    assert str(direct_formula.get("claim_group_kind") or "") == "formula_bundle"


def test_apply_provenance_strict_identity_contract_downgrades_invalid_must_locate_segment():
    from kb import task_runtime

    hardened, meta = task_runtime._apply_provenance_strict_identity_contract(
        [
            {
                "segment_id": "seg_001",
                "claim_type": "blockquote_claim",
                "must_locate": True,
                "primary_block_id": "",
                "evidence_block_ids": [],
                "anchor_kind": "blockquote",
                "anchor_text": "",
                "evidence_quote": "",
            }
        ]
    )

    assert len(hardened) == 1
    assert bool(hardened[0].get("must_locate")) is False
    assert set(hardened[0].get("strict_identity_missing_reasons") or []) == {
        "missing_primary_block_id",
        "missing_evidence_block_ids",
        "missing_anchor_text_or_evidence_quote",
    }
    assert int(meta.get("provenance_schema_version") or 0) == 4
    assert int(meta.get("must_locate_candidate_count") or 0) == 1
    assert int(meta.get("must_locate_count") or 0) == 0
    assert int(meta.get("strict_identity_count") or 0) == 0
    assert bool(meta.get("strict_identity_ready")) is False
    assert int((meta.get("identity_missing_reasons") or {}).get("missing_primary_block_id") or 0) == 1


def test_build_paper_guide_answer_provenance_does_not_force_section_title_quote(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "ScenePaper.pdf"
    md_dir = tmp_path / "ScenePaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "ScenePaper.en.md"
    quote_text = (
        "We noticed that due to the lack of high-quality details and multi-view consistency, "
        "we cannot estimate camera poses via SfM using the images reconstructed from those SCI image reconstruction methods."
    )
    md_main.write_text(
        (
            "# Baseline methods and evaluation metrics\n\n"
            f"{quote_text}\n"
        ),
        encoding="utf-8",
    )

    answer = (
        '结论: 该直接证据出自 "Baseline methods and evaluation metrics" 段落。\n\n'
        f'> "{quote_text}"\n'
    )
    hits = [
        {
            "text": quote_text,
            "meta": {
                "source_path": str(source_pdf),
                "ref_show_snippets": [quote_text],
                "ref_best_heading_path": "Baseline methods and evaluation metrics",
            },
        }
    ]

    provenance = task_runtime._build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=str(source_pdf),
        bound_source_name="ScenePaper.pdf",
        db_dir=None,
        llm_rerank=False,
    )

    segments = provenance.get("segments") or []
    assert segments
    heading_seg = segments[0]
    assert str(heading_seg.get("claim_type") or "") != "quote_claim"
    assert bool(heading_seg.get("must_locate")) is False


def test_build_paper_guide_answer_provenance_marks_figure_claim_as_must_locate(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "VisionPaper.pdf"
    md_dir = tmp_path / "VisionPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "VisionPaper.en.md"
    fig_caption = "Figure 3. Reconstruction quality comparison across baseline methods."
    md_main.write_text(
        (
            "# Results\n\n"
            f"{fig_caption}\n\n"
            "The figure shows SCINeRF preserves better view consistency.\n"
        ),
        encoding="utf-8",
    )

    answer = f"依据: {fig_caption}"
    hits = [
        {
            "text": fig_caption,
            "meta": {
                "source_path": str(source_pdf),
                "ref_show_snippets": [fig_caption],
                "ref_best_heading_path": "Results",
                "anchor_target_kind": "figure",
                "anchor_target_number": 3,
            },
        }
    ]

    provenance = task_runtime._build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=str(source_pdf),
        bound_source_name="VisionPaper.pdf",
        db_dir=None,
        llm_rerank=False,
    )

    segments = provenance.get("segments") or []
    direct_segments = [seg for seg in segments if str(seg.get("evidence_mode") or "") == "direct"]
    assert direct_segments
    fig_seg = next(seg for seg in direct_segments if str(seg.get("claim_type") or "") == "figure_claim")
    assert bool(fig_seg.get("must_locate")) is True
    assert str(fig_seg.get("anchor_kind") or "") == "figure"
    assert "Figure 3" in str(fig_seg.get("anchor_text") or "")


def test_build_paper_guide_answer_provenance_rebinds_figure_claim_to_figure_block(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "VisionPaper.pdf"
    md_dir = tmp_path / "VisionPaper"
    assets_dir = md_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "fig1.png").write_bytes(b"fake")
    md_main = md_dir / "VisionPaper.en.md"
    figure_caption = (
        "Figure 1. Given a single snapshot compressed image, our method is able to recover "
        "the underlying 3D scene representation."
    )
    method_para = (
        "Our method takes a single compressed image and encoding masks as input, and recovers "
        "the underlying 3D scene representation as well as camera poses."
    )
    md_main.write_text(
        (
            "# VisionPaper\n\n"
            "![Figure 1](./assets/fig1.png)\n"
            f"*{figure_caption}*\n\n"
            "## Method\n\n"
            f"{method_para}\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    figure_block = next(block for block in blocks if str(block.get("kind") or "") == "figure")
    caption_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "paragraph"
        and "single snapshot compressed image" in str(block.get("text") or "").lower()
    )
    method_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "paragraph"
        and "encoding masks as input" in str(block.get("text") or "").lower()
    )

    answer = "根据 Figure 1 的图注，论文展示了如何从单张 compressed image 恢复 3D scene representation。"
    hits = [
        {
            "text": method_para,
            "meta": {
                "source_path": str(source_pdf),
                "ref_show_snippets": [method_para],
                "ref_best_heading_path": "Method",
                "anchor_target_kind": "figure",
                "anchor_target_number": 1,
            },
        }
    ]

    provenance = task_runtime._build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=str(source_pdf),
        bound_source_name="VisionPaper.pdf",
        db_dir=None,
        llm_rerank=False,
    )

    segments = provenance.get("segments") or []
    fig_seg = next(seg for seg in segments if str(seg.get("claim_type") or "") == "figure_claim")
    evidence_block_ids = [str(item or "") for item in list(fig_seg.get("evidence_block_ids") or [])]

    assert str(fig_seg.get("primary_block_id") or "") == str(figure_block.get("block_id") or "")
    assert str(fig_seg.get("primary_anchor_id") or "") == str(figure_block.get("anchor_id") or "")
    assert str(fig_seg.get("primary_block_id") or "") != str(method_block.get("block_id") or "")
    assert str(caption_block.get("block_id") or "") in evidence_block_ids
    assert "single snapshot compressed image" in str(fig_seg.get("anchor_text") or "").lower()
    assert "single snapshot compressed image" in str(fig_seg.get("evidence_quote") or "").lower()


def test_build_paper_guide_answer_provenance_groups_formula_explanation_as_required(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "ImagingPaper.pdf"
    md_dir = tmp_path / "ImagingPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "ImagingPaper.en.md"
    formula = "$$Y = \\sum_{i=1}^{N} X_i \\odot M_i + Z \\tag{3}$$"
    explanation = "where M_i is the DMD binary pattern for the i-th frame."
    md_main.write_text(
        (
            "# Method\n\n"
            f"{formula}\n\n"
            f"{explanation}\n"
        ),
        encoding="utf-8",
    )
    blocks = task_runtime.load_source_blocks(md_main)
    formula_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "equation"
        and int(block.get("number") or 0) == 3
    )
    explanation_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "paragraph"
        and "binary pattern" in str(block.get("text") or "").lower()
    )

    answer = formula + "\n\n" + explanation
    hits = [
        {
            "text": formula,
            "meta": {
                "source_path": str(source_pdf),
                "ref_show_snippets": [formula, explanation],
                "ref_best_heading_path": "Method",
                "anchor_target_kind": "equation",
                "anchor_target_number": 3,
            },
        }
    ]

    provenance = task_runtime._build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=str(source_pdf),
        bound_source_name="ImagingPaper.pdf",
        db_dir=None,
        llm_rerank=False,
    )

    segments = provenance.get("segments") or []
    assert segments
    formula_seg = next(seg for seg in segments if str(seg.get("claim_type") or "") == "formula_claim")
    explanation_seg = next(seg for seg in segments if str(seg.get("claim_type") or "") == "equation_explanation_claim")
    assert int(provenance.get("provenance_schema_version") or 0) == 4
    assert bool(provenance.get("strict_identity_ready")) is True
    assert bool(formula_seg.get("must_locate")) is True
    assert bool(explanation_seg.get("must_locate")) is True
    assert str(formula_seg.get("locate_policy") or "") == "required"
    assert str(explanation_seg.get("locate_policy") or "") == "required"
    assert str(formula_seg.get("locate_surface_policy") or "") == "primary"
    assert str(explanation_seg.get("locate_surface_policy") or "") == "secondary"
    assert str(formula_seg.get("formula_origin") or "") == "source"
    assert str(explanation_seg.get("formula_origin") or "") == "explanation"
    assert str(formula_seg.get("claim_group_kind") or "") == "formula_bundle"
    assert str(explanation_seg.get("claim_group_kind") or "") == "formula_bundle"
    assert str(formula_seg.get("claim_group_id") or "")
    assert str(formula_seg.get("claim_group_id") or "") == str(explanation_seg.get("claim_group_id") or "")
    assert str(explanation_seg.get("primary_block_id") or "") == str(explanation_block.get("block_id") or "")
    assert str(explanation_seg.get("primary_anchor_id") or "") == str(explanation_block.get("anchor_id") or "")
    assert str(explanation_seg.get("anchor_kind") or "") == "sentence"
    assert int(explanation_seg.get("equation_number") or 0) == 3
    assert str(explanation_seg.get("anchor_text") or "").strip()
    evidence_ids = list(explanation_seg.get("evidence_block_ids") or [])
    assert str(formula_block.get("block_id") or "") in evidence_ids
    assert str(explanation_block.get("block_id") or "") in evidence_ids
    assert list(explanation_seg.get("related_block_ids") or []) == [str(formula_block.get("block_id") or "")]


def test_split_answer_segments_preserves_raw_markdown():
    from kb.source_blocks import split_answer_segments

    answer = (
        "结论: **随机掩码**用于压缩测量。\n\n"
        "- 依据: 原文明确写到 random mask encoding.\n"
        "- 下一步: 再核对实验段。\n"
    )

    segments = split_answer_segments(answer)

    assert len(segments) >= 3
    assert str(segments[0].get("raw_markdown") or "").startswith("结论:")
    assert str(segments[1].get("raw_markdown") or "").startswith("依据:")
    assert str(segments[2].get("raw_markdown") or "").startswith("下一步:")


def test_apply_provenance_required_coverage_contract_hides_duplicate_formula_surface_in_same_bundle():
    from kb import task_runtime

    segments = task_runtime._apply_provenance_required_coverage_contract(
        [
            {
                "segment_id": "seg_002",
                "segment_index": 2,
                "kind": "paragraph",
                "segment_type": "equation_explanation",
                "text": "$$ C(r) = \\int_{t_n}^{t_f} T(t) \\sigma(r(t)) c(r(t), d) \\, dt $$",
                "raw_markdown": "$$ C(r) = \\int_{t_n}^{t_f} T(t) \\sigma(r(t)) c(r(t), d) \\, dt $$",
                "evidence_mode": "direct",
                "claim_type": "formula_claim",
                "must_locate": True,
                "anchor_kind": "equation",
                "equation_number": 1,
                "primary_block_id": "blk_eq1",
                "primary_anchor_id": "eq_00001",
                "primary_heading_path": "Method",
                "evidence_block_ids": ["blk_eq1"],
                "support_block_ids": [],
                "anchor_text": "$$ C(r) = \\int_{t_n}^{t_f} T(t) \\sigma(r(t)) c(r(t), d) \\, dt $$",
                "evidence_quote": "$$ C(r) = \\int_{t_n}^{t_f} T(t) \\sigma(r(t)) c(r(t), d) \\, dt $$",
            },
            {
                "segment_id": "seg_008",
                "segment_index": 8,
                "kind": "paragraph",
                "segment_type": "equation_explanation",
                "text": "$$ C(r) = \\int_{t_n}^{t_f} T(t) \\sigma(r(t)) c(r(t), d) \\, dt $$",
                "raw_markdown": "$$ C(r) = \\int_{t_n}^{t_f} T(t) \\sigma(r(t)) c(r(t), d) \\, dt $$",
                "evidence_mode": "direct",
                "claim_type": "formula_claim",
                "must_locate": True,
                "anchor_kind": "equation",
                "equation_number": 1,
                "primary_block_id": "blk_eq1",
                "primary_anchor_id": "eq_00001",
                "primary_heading_path": "Method",
                "evidence_block_ids": ["blk_eq1"],
                "support_block_ids": [],
                "anchor_text": "$$ C(r) = \\int_{t_n}^{t_f} T(t) \\sigma(r(t)) c(r(t), d) \\, dt $$",
                "evidence_quote": "$$ C(r) = \\int_{t_n}^{t_f} T(t) \\sigma(r(t)) c(r(t), d) \\, dt $$",
            },
        ],
        block_lookup={
            "blk_eq1": {
                "block_id": "blk_eq1",
                "anchor_id": "eq_00001",
                "kind": "equation",
                "number": 1,
                "order_index": 24,
                "text": "$$ C(r) = \\int_{t_n}^{t_f} T(t) \\sigma(r(t)) c(r(t), d) \\, dt $$",
                "raw_text": "$$ C(r) = \\int_{t_n}^{t_f} T(t) \\sigma(r(t)) c(r(t), d) \\, dt $$",
            },
        },
    )

    assert len(segments) == 2
    primary_formula = next(seg for seg in segments if str(seg.get("segment_id") or "") == "seg_002")
    duplicate_formula = next(seg for seg in segments if str(seg.get("segment_id") or "") == "seg_008")
    assert bool(primary_formula.get("must_locate")) is True
    assert str(primary_formula.get("locate_policy") or "") == "required"
    assert str(primary_formula.get("locate_surface_policy") or "") == "primary"
    assert str(primary_formula.get("formula_origin") or "") == "source"
    assert bool(duplicate_formula.get("must_locate")) is False
    assert str(duplicate_formula.get("locate_policy") or "") == "hidden"
    assert str(duplicate_formula.get("locate_surface_policy") or "") == "hidden"
    assert str(duplicate_formula.get("formula_origin") or "") == "derived"


def test_apply_provenance_required_coverage_contract_hides_duplicate_quote_surface_in_same_bundle():
    from kb import task_runtime

    quote = (
        "where t_n and t_f are near and far bounds for volumetric rendering respectively, "
        "r(t) is the sampled 3D point along the ray."
    )
    segments = task_runtime._apply_provenance_required_coverage_contract(
        [
            {
                "segment_id": "seg_004",
                "segment_index": 4,
                "kind": "blockquote",
                "segment_type": "evidence",
                "text": quote,
                "raw_markdown": f"> {quote}",
                "evidence_mode": "direct",
                "claim_type": "blockquote_claim",
                "must_locate": True,
                "anchor_kind": "blockquote",
                "primary_block_id": "blk_quote",
                "primary_anchor_id": "p_00026",
                "primary_heading_path": "Background on NeRF",
                "evidence_block_ids": ["blk_quote"],
                "support_block_ids": [],
                "anchor_text": quote,
                "evidence_quote": quote,
            },
            {
                "segment_id": "seg_005",
                "segment_index": 5,
                "kind": "blockquote",
                "segment_type": "evidence",
                "text": quote,
                "raw_markdown": f"> - {quote}",
                "evidence_mode": "direct",
                "claim_type": "blockquote_claim",
                "must_locate": True,
                "anchor_kind": "blockquote",
                "primary_block_id": "blk_quote",
                "primary_anchor_id": "p_00026",
                "primary_heading_path": "Background on NeRF",
                "evidence_block_ids": ["blk_quote"],
                "support_block_ids": [],
                "anchor_text": quote,
                "evidence_quote": quote,
            },
        ],
        block_lookup={
            "blk_quote": {
                "block_id": "blk_quote",
                "anchor_id": "p_00026",
                "kind": "paragraph",
                "order_index": 26,
                "text": quote,
                "raw_text": quote,
            },
        },
    )

    assert len(segments) == 2
    keep = next(seg for seg in segments if str(seg.get("segment_id") or "") == "seg_004")
    hidden = next(seg for seg in segments if str(seg.get("segment_id") or "") == "seg_005")
    assert bool(keep.get("must_locate")) is True
    assert str(keep.get("locate_policy") or "") == "required"
    assert str(keep.get("locate_surface_policy") or "") == "primary"
    assert bool(hidden.get("must_locate")) is False
    assert str(hidden.get("locate_policy") or "") == "hidden"
    assert str(hidden.get("locate_surface_policy") or "") == "hidden"


def test_should_run_provenance_async_refine_requires_flags_and_api_key(monkeypatch):
    from kb import task_runtime

    class _Settings:
        api_key = "test-key"

    task = {
        "paper_guide_mode": True,
        "paper_guide_bound_source_path": "/tmp/demo.pdf",
        "llm_rerank": True,
        "settings_obj": _Settings(),
    }

    monkeypatch.setenv("KB_PROVENANCE_ASYNC_LLM", "1")
    assert task_runtime._should_run_provenance_async_refine(task) is True

    task_no_key = dict(task)
    task_no_key["settings_obj"] = object()
    assert task_runtime._should_run_provenance_async_refine(task_no_key) is False

    task_no_rerank = dict(task)
    task_no_rerank["llm_rerank"] = False
    assert task_runtime._should_run_provenance_async_refine(task_no_rerank) is False

    monkeypatch.setenv("KB_PROVENANCE_ASYNC_LLM", "0")
    assert task_runtime._should_run_provenance_async_refine(task) is False


def test_gen_store_answer_provenance_async_enables_llm_rerank(monkeypatch):
    from kb import task_runtime

    captured: dict[str, object] = {}

    def _fake_store(task: dict, *, answer: str, answer_hits: list[dict]) -> None:
        captured["task"] = dict(task)
        captured["answer"] = answer
        captured["answer_hits"] = list(answer_hits)

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, name=None):
            self._target = target

        def start(self):
            if callable(self._target):
                self._target()

    monkeypatch.setattr(task_runtime, "_gen_store_answer_provenance", _fake_store)
    monkeypatch.setattr(task_runtime.threading, "Thread", _ImmediateThread)

    task_in = {"llm_rerank": False, "paper_guide_mode": True, "paper_guide_bound_source_path": "/tmp/demo.pdf"}
    task_runtime._gen_store_answer_provenance_async(
        task_in,
        answer="demo answer",
        answer_hits=[{"text": "x"}],
    )

    assert captured["answer"] == "demo answer"
    assert captured["answer_hits"] == [{"text": "x"}]
    assert isinstance(captured["task"], dict)
    assert captured["task"].get("llm_rerank") is True
    assert task_in.get("llm_rerank") is False


def test_apply_provenance_required_coverage_contract_rebinds_excerpted_quote_to_true_source_block():
    from kb import task_runtime

    repo_root = Path(__file__).resolve().parents[2]
    md_main = (
        repo_root
        / "db"
        / "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image"
        / "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md"
    )
    blocks = task_runtime.load_source_blocks(md_main)
    block_lookup = {
        str(block.get("block_id") or "").strip(): dict(block)
        for block in blocks
        if isinstance(block, dict) and str(block.get("block_id") or "").strip()
    }
    conclusion_block = next(
        block for block in blocks
        if "scinerf exploits neural radiance fields as its underlying scene representation" in str(block.get("text") or "").lower()
    )
    wrong_method_block = next(
        block for block in blocks
        if "render $x_i$ to synthesize the compressed image $y$" in str(block.get("text") or "").lower()
    )

    segments = task_runtime._apply_provenance_required_coverage_contract(
        [
            {
                "segment_id": "seg_004",
                "segment_index": 4,
                "kind": "blockquote",
                "segment_type": "evidence",
                "text": (
                    "SCINeRF exploits neural radiance fields as its underlying scene representation [...] "
                    "Physical image formation process of an SCI image is exploited to formulate the training objective "
                    "for jointly NeRF training and camera poses optimization."
                ),
                "raw_markdown": (
                    '*"SCINeRF exploits neural radiance fields as its underlying scene representation [...] '
                    "Physical image formation process of an SCI image is exploited to formulate the training objective "
                    'for jointly NeRF training and camera poses optimization."*'
                ),
                "evidence_mode": "direct",
                "claim_type": "blockquote_claim",
                "must_locate": True,
                "anchor_kind": "blockquote",
                "primary_block_id": str(wrong_method_block.get("block_id") or ""),
                "primary_anchor_id": str(wrong_method_block.get("anchor_id") or ""),
                "primary_heading_path": str(wrong_method_block.get("heading_path") or ""),
                "evidence_block_ids": [str(wrong_method_block.get("block_id") or "")],
                "support_block_ids": [],
                "anchor_text": str(wrong_method_block.get("text") or ""),
                "evidence_quote": str(wrong_method_block.get("text") or ""),
            }
        ],
        block_lookup=block_lookup,
    )

    assert len(segments) == 1
    seg = segments[0]
    assert str(seg.get("primary_block_id") or "") == str(conclusion_block.get("block_id") or "")
    assert str(wrong_method_block.get("block_id") or "") in list(seg.get("support_block_ids") or [])
    assert str(conclusion_block.get("block_id") or "") in list(seg.get("evidence_block_ids") or [])
    assert "physical image formation process of an sci image is exploited" in str(seg.get("evidence_quote") or "").lower()
    assert "scinerf exploits neural radiance fields as its underlying scene representation" in str(seg.get("anchor_text") or "").lower()


def test_apply_provenance_required_coverage_contract_hides_non_exact_display_formula_summary():
    from kb import task_runtime

    repo_root = Path(__file__).resolve().parents[2]
    md_main = (
        repo_root
        / "db"
        / "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image"
        / "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md"
    )
    blocks = task_runtime.load_source_blocks(md_main)
    block_lookup = {
        str(block.get("block_id") or "").strip(): dict(block)
        for block in blocks
        if isinstance(block, dict) and str(block.get("block_id") or "").strip()
    }
    eq3_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "equation" and int(block.get("number") or 0) == 3
    )

    generated_segment = {
        "segment_id": "seg_006",
        "segment_index": 6,
        "kind": "paragraph",
        "segment_type": "fact",
        "text": (
            "指通过快照压缩成像系统一次性捕获的二维编码图像。"
            "$$ y = \\sum_{i=1}^N \\Phi_i \\odot x_i, $$"
            "其中 x_i 为第 i 帧真实图像。"
        ),
        "raw_markdown": (
            "指通过快照压缩成像系统一次性捕获的二维编码图像。\n"
            "$$ y = \\sum_{i=1}^N \\Phi_i \\odot x_i, $$\n"
            "其中 x_i 为第 i 帧真实图像。"
        ),
        "evidence_mode": "direct",
        "claim_type": "formula_claim",
        "must_locate": True,
        "anchor_kind": "equation",
        "equation_number": 3,
        "primary_block_id": str(eq3_block.get("block_id") or ""),
        "primary_anchor_id": str(eq3_block.get("anchor_id") or ""),
        "primary_heading_path": str(eq3_block.get("heading_path") or ""),
        "evidence_block_ids": [str(eq3_block.get("block_id") or "")],
        "support_block_ids": [],
        "anchor_text": "$$ y = \\sum_{i=1}^N \\Phi_i \\odot x_i, $$",
        "evidence_quote": str(eq3_block.get("raw_text") or eq3_block.get("text") or ""),
    }

    segments = task_runtime._apply_provenance_required_coverage_contract(
        [generated_segment],
        block_lookup=block_lookup,
    )

    assert len(segments) == 1
    seg = segments[0]
    assert str(seg.get("claim_type") or "") == "formula_claim"
    assert bool(seg.get("must_locate")) is False
    assert str(seg.get("locate_policy") or "") == "hidden"
    assert str(seg.get("claim_group_id") or "") == ""
    assert str(seg.get("claim_group_kind") or "") == ""
