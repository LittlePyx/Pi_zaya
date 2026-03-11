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
    assert str(seg.get("kind") or "") == "blockquote"
    assert str(seg.get("claim_type") or "") == "blockquote_claim"
    assert bool(seg.get("must_locate")) is True
    assert str(seg.get("anchor_kind") or "") == "blockquote"
    assert "multi-view consistency" in str(seg.get("anchor_text") or "").lower()


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
    assert bool(formula_seg.get("must_locate")) is True
    assert str(formula_seg.get("anchor_kind") or "") == "equation"
    assert int(formula_seg.get("equation_number") or 0) == 3
    assert "X_i" in str(formula_seg.get("anchor_text") or "")


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
