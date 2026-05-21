from __future__ import annotations

from ui import refs_renderer


def test_structured_system_b_detail_prefers_source_markdown_context(monkeypatch, tmp_path):
    source_file = tmp_path / "paper.en.md"
    source_file.write_text(
        "\n".join(
            [
                "# Paper",
                "<!-- kb_page: 3 -->",
                "## 2. Related Work",
                "SCINeRF uses ADMM-based optimization [4] as prior reconstruction machinery.",
                "",
                "## References",
                "[4] Boyd S. Distributed Optimization and Statistical Learning via ADMM. 2011.",
            ]
        ),
        encoding="utf-8",
    )
    source_path = str(source_file)
    sid = refs_renderer._source_cite_id(source_path)

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 4:
            return None
        return {
            "source_path": source_path,
            "source_name": "paper.pdf",
            "ref_num": 4,
            "ref": {
                "authors": "Boyd S",
                "year": "2011",
                "title": "Distributed Optimization and Statistical Learning via ADMM",
                "raw": "[4] Boyd S. Distributed Optimization and Statistical Learning via ADMM. 2011.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "paper.pdf")
    monkeypatch.setattr(refs_renderer, "_is_temp_source_path", lambda _sp: False)

    md = f"ADMM is prior optimization machinery; open ADMM [[CITE:{sid}:4]] to follow the paper's citation trail."
    hits = [{"meta": {"source_path": source_path, "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[4](#kb-cite-" in out
    assert len(details) == 1
    detail = details[0]
    assert detail["is_inpaper"] is True
    assert detail["citation_context_source"] == "source_markdown"
    assert detail["evidence_source"] == "source_markdown"
    assert "ADMM-based optimization [4]" in detail["citation_context"]
    assert "Boyd S" not in detail["citation_context"]
    assert detail["heading_path"].endswith("2. Related Work")
    assert detail["page_start"] == 3
    assert "answer_context_only" not in detail["card_quality_flags"]
    assert "ADMM-based optimization [4]" in detail["card_evidence"]
    assert "ADMM 优化框架背景" in detail["card_takeaway"]
