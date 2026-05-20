from __future__ import annotations

from api.reference_ui import _filter_refs_hits_by_prompt_focus, enrich_refs_payload


def _hit(source_name: str, text: str) -> dict:
    return {
        "text": text,
        "meta": {
            "source_path": f"db/{source_name}/{source_name}.en.md",
            "source_name": source_name,
            "ref_pack_state": "ready",
            "ref_best_heading_path": "Abstract",
            "top_heading": "Abstract",
            "ref_show_snippets": [text],
            "ref_rank": {"score": 9.0, "display_score": 9.0, "llm": 90.0},
        },
    }


def test_multi_source_synthesis_does_not_bind_to_one_named_source() -> None:
    prompt = "显微成像这些 structured detection、interferometric、light-field 方法分别是在解决什么麻烦？"
    hits = [
        _hit("Structured detection for laser scanning microscopy", "structured detection improves optical sectioning"),
        _hit("Interferometric image scanning microscopy", "interferometric detection improves lateral resolution"),
        _hit("Quantum correlation light-field microscope", "light-field microscopy supports refocusing"),
    ]

    filtered = _filter_refs_hits_by_prompt_focus(prompt, hits)

    assert [item["meta"]["source_name"] for item in filtered] == [
        "Structured detection for laser scanning microscopy",
        "Interferometric image scanning microscopy",
        "Quantum correlation light-field microscope",
    ]


def test_fast_refs_payload_preserves_multi_source_synthesis_cards() -> None:
    prompt = "探测器综述和 physics-informed deep learning 这篇应该怎么搭配读？"
    hits = [
        _hit("Emerging single-photon detectors review", "SPAD detector noise and hardware background"),
        _hit("High-resolution single-photon imaging with physics-informed deep learning", "physics-informed noise model"),
    ]

    out = enrich_refs_payload(
        {
            7: {
                "user_msg_id": 7,
                "prompt": prompt,
                "hits": hits,
                "scores": [9.0, 9.0],
                "used_query": prompt,
                "used_translation": False,
            }
        },
        pdf_root=None,
        md_root=None,
        lib_store=None,
        allow_expensive_llm_for_ready=False,
        allow_exact_locate=False,
        render_variant="fast",
    )

    pack = out[7]
    assert pack["display_state"] == "ready"
    assert len(pack["hits"]) == 2
    assert pack["pipeline_debug"]["prompt_likely_multi_paper_synthesis"] is True


def test_ordinary_multi_source_synthesis_refs_are_bounded() -> None:
    prompt = "SCIGS 这篇到底想解决什么问题？它和 SCINeRF 的区别在哪里？"
    hits = [_hit(f"paper-{idx}", f"evidence {idx} about SCIGS and SCINeRF") for idx in range(1, 7)]

    out = enrich_refs_payload(
        {
            8: {
                "user_msg_id": 8,
                "prompt": prompt,
                "hits": hits,
                "scores": [9.0] * len(hits),
                "used_query": prompt,
                "used_translation": False,
            }
        },
        pdf_root=None,
        md_root=None,
        lib_store=None,
        allow_expensive_llm_for_ready=False,
        allow_exact_locate=False,
        render_variant="fast",
    )

    pack = out[8]
    assert len(pack["hits"]) == 4
    assert pack["pipeline_debug"]["display_cap"] == 4
