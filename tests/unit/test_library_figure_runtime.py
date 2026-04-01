from pathlib import Path

from kb.library_figure_runtime import (
    _build_doc_figure_card,
    _collect_doc_figure_assets,
    _maybe_append_library_figure_markdown,
)


def test_collect_doc_figure_assets_extracts_number_from_caption(tmp_path: Path):
    doc_dir = tmp_path / "paper"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig = assets_dir / "page_5_fig_2.png"
    fig.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    md_path = doc_dir / "paper.en.md"
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

    assets = _collect_doc_figure_assets(md_path, extract_figure_number=lambda text: 2 if "Figure 2" in str(text) else 0)
    assert assets
    assert int(assets[0].get("number") or 0) == 2


def test_build_doc_figure_card_formats_source_name_and_url(tmp_path: Path):
    md_path = tmp_path / "NatPhoton-2019.en.md"
    md_path.write_text("", encoding="utf-8")

    out = _build_doc_figure_card(
        source_path=str(md_path),
        figure_num=3,
        collect_doc_figure_assets=lambda _md: [{"path": str(tmp_path / "fig.png"), "number": 3, "label": "Figure 3 caption"}],
        source_name_from_md_path=lambda source_path: "NatPhoton-2019.pdf",
    )

    assert out["source_name"] == "NatPhoton-2019.pdf"
    assert out["figure_num"] == 3
    assert out["url"].startswith("/api/references/asset?path=")


def test_maybe_append_library_figure_markdown_uses_highest_scored_card():
    calls = []

    def _build_doc_figure_card_local(*, source_path: str, figure_num: int):
        return {
            "source_name": f"{source_path}.pdf",
            "figure_num": figure_num,
            "url": f"/api/references/asset?path={source_path}",
            "label": f"{source_path} label",
        }

    def _score_local(**kwargs):
        calls.append(kwargs["source_path"])
        return 10.0 if kwargs["source_path"] == "b.md" else 1.0

    out = _maybe_append_library_figure_markdown(
        "Figure 2 explanation.",
        prompt="Explain Figure 2.",
        answer_hits=[
            {"meta": {"source_path": "a.md"}},
            {"meta": {"source_path": "b.md"}},
        ],
        requested_figure_number=lambda prompt, hits: 2,
        build_doc_figure_card=_build_doc_figure_card_local,
        score_figure_card_source_binding=_score_local,
    )

    assert "### Library Figure" in out
    assert "b.md.pdf" in out
    assert "a.md.pdf" not in out
    assert calls == ["a.md", "b.md"]


def test_maybe_append_library_figure_markdown_prefers_bound_source_card():
    calls = []

    def _build_doc_figure_card_local(*, source_path: str, figure_num: int):
        if source_path not in {"bound.md", "old.md"}:
            return None
        return {
            "source_name": f"{source_path}.pdf",
            "figure_num": figure_num,
            "url": f"/api/references/asset?path={source_path}",
            "label": f"{source_path} label",
        }

    def _score_local(**kwargs):
        calls.append(kwargs["source_path"])
        return 99.0 if kwargs["source_path"] == "old.md" else 1.0

    out = _maybe_append_library_figure_markdown(
        "Figure 4 explanation.",
        prompt="Explain Figure 4.",
        answer_hits=[{"meta": {"source_path": "old.md"}}],
        bound_source_path="bound.md",
        requested_figure_number=lambda prompt, hits: 4,
        build_doc_figure_card=_build_doc_figure_card_local,
        score_figure_card_source_binding=_score_local,
    )

    assert "### Library Figure" in out
    assert "bound.md.pdf" in out
    assert "old.md.pdf" not in out
    assert calls == ["old.md"]
