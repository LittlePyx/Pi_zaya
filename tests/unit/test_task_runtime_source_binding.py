from __future__ import annotations

from pathlib import Path


def test_needs_bound_source_hint_for_inpaper_queries():
    from kb import task_runtime

    assert task_runtime._needs_bound_source_hint("这篇文章里的公式8是什么")
    assert task_runtime._needs_bound_source_hint("explain figure 3 in this paper")
    assert not task_runtime._needs_bound_source_hint("NatPhoton-2019-xxx.pdf 公式8是什么")


def test_pick_recent_bound_source_hints():
    from kb import task_runtime

    class FakeStore:
        def list_conversation_sources(self, conv_id: str, limit: int = 2):
            return [
                {"source_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf", "source_path": "/x/a.en.md"},
                {"source_name": "", "source_path": "/x/LPR-2025-Advances and Challenges.en.md"},
                {"source_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf", "source_path": "/x/a.en.md"},
            ]

    hints = task_runtime._pick_recent_bound_source_hints(conv_id="conv-1", chat_store=FakeStore(), limit=3)
    assert hints[0].startswith("NatPhoton-2019")
    assert any("LPR-2025" in h for h in hints)


def test_collect_doc_figure_assets_and_append_markdown(tmp_path: Path):
    from kb import task_runtime

    doc_dir = tmp_path / "NatPhoton-2019-Principles and prospects for single-pixel imaging"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig2 = assets_dir / "page_5_fig_2.png"
    fig2.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    md_path = doc_dir / "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
    md_path.write_text(
        "\n".join(
            [
                "## Box 1",
                "![Figure](./assets/page_5_fig_2.png)",
                "**Figure 2.** Single-pixel imaging setup and reconstruction principle.",
            ]
        ),
        encoding="utf-8",
    )

    assets = task_runtime._collect_doc_figure_assets(md_path)
    assert assets
    assert int(assets[0].get("number") or 0) == 2

    answer = "这是该文中的 Fig.2。"
    hits = [
        {
            "text": "Figure 2 ...",
            "meta": {
                "source_path": str(md_path),
                "anchor_target_kind": "figure",
                "anchor_target_number": 2,
                "anchor_match_score": 9.0,
            },
        }
    ]
    out = task_runtime._maybe_append_library_figure_markdown(answer, prompt="这篇文章的第二张图是什么", answer_hits=hits)
    assert "文献图示（库内截图）" in out
    assert "/api/references/asset?path=" in out
    assert "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf" in out


def test_append_library_figure_markdown_prefers_bound_source(tmp_path: Path):
    from kb import task_runtime

    doc_a = tmp_path / "NatPhoton-2019"
    doc_b = tmp_path / "LPR-2025"
    for d in (doc_a, doc_b):
        (d / "assets").mkdir(parents=True, exist_ok=True)

    (doc_a / "assets" / "page_5_fig_1.png").write_bytes(b"\x89PNG\r\n\x1a\nA")
    (doc_b / "assets" / "page_8_fig_1.png").write_bytes(b"\x89PNG\r\n\x1a\nB")

    md_a = doc_a / "NatPhoton-2019.en.md"
    md_b = doc_b / "LPR-2025.en.md"
    md_a.write_text(
        "\n".join(
            [
                "## Example",
                "![Fig. 3](./assets/page_5_fig_1.png)",
                "Fig. 3",
            ]
        ),
        encoding="utf-8",
    )
    md_b.write_text(
        "\n".join(
            [
                "## Example",
                "![Fig. 3](./assets/page_8_fig_1.png)",
                "Fig. 3",
            ]
        ),
        encoding="utf-8",
    )

    hits = [
        {
            "text": "Figure 3 ...",
            "meta": {
                "source_path": str(md_b),
                "anchor_target_kind": "figure",
                "anchor_target_number": 3,
                "anchor_match_score": 7.0,
                "explicit_doc_match_score": 0.0,
            },
        },
        {
            "text": "Figure 3 ...",
            "meta": {
                "source_path": str(md_a),
                "anchor_target_kind": "figure",
                "anchor_target_number": 3,
                "anchor_match_score": 8.0,
                "explicit_doc_match_score": 8.6,
            },
        },
    ]

    out = task_runtime._maybe_append_library_figure_markdown(
        "这是图3。",
        prompt="NatPhoton-2019.pdf 这篇文章的图3是什么",
        answer_hits=hits,
    )
    assert out.count("/api/references/asset?path=") == 1
    assert "NatPhoton-2019.pdf" in out
    assert "LPR-2025.pdf" not in out
