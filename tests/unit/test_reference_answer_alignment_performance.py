from pathlib import Path

import api.reference_ui as reference_ui


def test_chinese_people_prompt_keeps_biography_in_bounded_prefilter() -> None:
    blocks = [
        {
            "block_id": f"background-{index}",
            "kind": "paragraph",
            "heading_path": "3. Background",
            "text": (
                "Kai Song appears in this general background paragraph about "
                f"single-pixel imaging and reconstruction, item {index}."
            ),
        }
        for index in range(90)
    ]
    blocks.append(
        {
            "block_id": "kai-biography",
            "kind": "paragraph",
            "heading_path": "Author Biographies",
            "text": (
                "Kai Song received his degrees from Taiyuan University of Technology "
                "and researches single-photon and single-pixel imaging."
            ),
        }
    )

    shortlisted = reference_ui._prefilter_answer_aligned_source_blocks(
        blocks,
        terms=["Kai Song"],
        prompt="请根据作者简介说明 Kai Song 的教育经历、当前单位和研究方向。",
        source_path="paper.pdf",
        display_name="Example paper",
        limit=12,
    )

    assert shortlisted[0]["block_id"] == "kai-biography"
    assert len(shortlisted) == 12


def test_answer_alignment_shortlists_long_source_before_expensive_evidence_build(monkeypatch) -> None:
    blocks = [
        {
            "block_id": f"generic-{index}",
            "kind": "paragraph",
            "heading_path": "3. Background",
            "text": (
                "Single-pixel imaging (SPI) is an established computational imaging method "
                f"with generic background discussion number {index}."
            ),
        }
        for index in range(600)
    ]
    blocks.append(
        {
            "block_id": "target",
            "kind": "paragraph",
            "heading_path": "4. Distilled sensing",
            "text": (
                "The distilled sensing module performs measurement distillation before "
                "single-pixel image reconstruction and improves robustness."
            ),
        }
    )

    monkeypatch.setattr(reference_ui, "_resolve_source_md_path", lambda _path: Path("paper.md"))
    monkeypatch.setattr(reference_ui, "load_source_blocks", lambda _path: blocks)

    built_ids: list[str] = []

    def fake_build(*, block, source_path, display_name, **_kwargs):
        built_ids.append(str(block.get("block_id") or ""))
        text = str(block.get("text") or "")
        return {
            "source_path": source_path,
            "source_name": display_name,
            "block_id": block.get("block_id"),
            "heading_path": block.get("heading_path"),
            "snippet": text,
            "highlight_snippet": text,
            "strict_locate": True,
        }

    def fake_score(*, primary_evidence, terms, **_kwargs):
        surface = str(primary_evidence.get("snippet") or "").lower()
        matched = [term for term in terms if str(term or "").lower() in surface]
        return float(len(matched)), matched

    monkeypatch.setattr(reference_ui, "_source_block_to_answer_primary_evidence", fake_build)
    monkeypatch.setattr(reference_ui, "_score_primary_ref_evidence_against_answer", fake_score)

    primary, alignment = reference_ui._select_answer_aligned_source_block_primary_evidence(
        pack={
            "hits": [
                {
                    "meta": {"source_path": "paper.pdf"},
                    "ui_meta": {"display_name": "Example paper"},
                }
            ]
        },
        prompt="Explain distilled sensing and measurement distillation.",
        answer="SPI uses distilled sensing for measurement distillation.",
        terms=["SPI", "distilled sensing", "measurement distillation"],
    )

    assert primary["block_id"] == "target"
    assert alignment["selected_heading_path"] == "4. Distilled sensing"
    assert len(built_ids) <= 72
    assert "target" in built_ids
