from __future__ import annotations

from types import SimpleNamespace

import pytest

from kb.path_safety import root_relative_file_id
from api.reference_ui import _attach_pack_primary_ref_evidence


def test_attach_pack_primary_evidence_prefers_answer_aligned_source_block(tmp_path):
    md_path = tmp_path / "qclfm.en.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Quantum correlation light-field microscope with extreme depth of field",
                "",
                "## I. INTRODUCTION",
                "",
                "Conventional light-field microscope designs typically make use of a microlens array.",
                "",
                "<!-- kb_page: 2 -->",
                "## Digital Refocusing Procedure",
                "",
                (
                    "Using the position and angular information of each photon, digital refocusing is achieved in two steps. "
                    "First, the trajectory of the photons is determined through ray tracing using a ray transfer matrix. "
                    "The image obtained after this first step is the diffraction pattern of the sample after propagating a distance z. "
                    "The second step reverses this diffraction by applying wave propagation of distance -z to bring the sample back into focus."
                ),
            ]
        ),
        encoding="utf-8",
    )
    intro_primary = {
        "source_path": str(md_path),
        "source_name": "Quantum correlation light-field microscope with extreme depth of field",
        "heading_path": "I. INTRODUCTION",
        "snippet": "Conventional light-field microscope designs typically make use of a microlens array.",
        "selection_reason": "answer_hit_top",
        "strict_locate": False,
    }
    pack = {
        "prompt": "这个 quantum correlation light-field microscope 是怎么把离焦样品重新对焦的？",
        "answer": (
            "It first uses ray tracing to recover photon trajectories, then treats the intermediate image "
            "as a diffraction pattern and applies wave propagation to refocus the sample."
        ),
        "hits": [
            {
                "text": "Conventional light-field microscope designs typically make use of a microlens array.",
                "meta": {"source_path": str(md_path)},
                "ui_meta": {
                    "source_path": str(md_path),
                    "display_name": "Quantum correlation light-field microscope with extreme depth of field",
                    "primary_evidence": intro_primary,
                },
            }
        ],
    }

    out = _attach_pack_primary_ref_evidence(pack)

    primary = out["primary_evidence"]
    assert "Digital Refocusing Procedure" in primary["heading_path"]
    assert "ray tracing" in primary["snippet"]
    assert "wave propagation" in primary["snippet"]
    assert primary["selection_reason"] == "answer_aligned_block"
    assert primary["page_start"] == 2
    assert primary["page_end"] == 2
    hit_primary = out["hits"][0]["ui_meta"]["primary_evidence"]
    assert hit_primary["block_id"] == primary["block_id"]
    assert hit_primary["anchor_id"] == primary["anchor_id"]
    assert "Digital Refocusing Procedure" in hit_primary["heading_path"]
    assert hit_primary["page_start"] == 2
    alignment = out["primary_evidence_alignment"]
    assert alignment["mismatch"] is False
    assert alignment["selected_source"] == "source_blocks"


def test_answer_alignment_keeps_question_specific_detector_and_speed_sentence(tmp_path):
    md_path = tmp_path / "three-d-video.en.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 2 -->",
                "## Abstract",
                (
                    "Performing high-speed structured illumination of a scene and sensing the reflected light "
                    "with four spatially-separated, single-pixel detectors, our system reconstructs continuous "
                    "real-time 3D video at approximately 8 frames per second for image resolutions of 64 by 64 pixels."
                ),
                "<!-- kb_page: 5 -->",
                "## Methods / Hadamard sampling",
                "A 32 by 32 resolution requires 1024 Hadamard patterns for complete sampling.",
            ]
        ),
        encoding="utf-8",
    )
    pack = {
        "prompt": "这篇 3D single-pixel video 为什么能实时重建？用了几个探测器，速度是多少？",
        "answer_text": "系统使用 4 个空间分离的单像素探测器，在 64×64 分辨率下约为 8 帧/秒。",
        "hits": [
            {
                "text": "A 32 by 32 resolution requires 1024 Hadamard patterns for complete sampling.",
                "meta": {"source_path": str(md_path)},
                "ui_meta": {
                    "source_path": str(md_path),
                    "display_name": "3D single-pixel video",
                    "primary_evidence": {
                        "source_path": str(md_path),
                        "source_name": "3D single-pixel video",
                        "heading_path": "Methods / Hadamard sampling",
                        "snippet": "A 32 by 32 resolution requires 1024 Hadamard patterns for complete sampling.",
                    },
                },
            }
        ],
    }

    out = _attach_pack_primary_ref_evidence(pack)

    primary = out["primary_evidence"]
    assert primary["heading_path"] == "Abstract"
    assert primary["page_start"] == 2
    assert "four spatially-separated" in primary["snippet"]
    assert "8 frames per second" in primary["snippet"]


def test_answer_alignment_skips_author_biography_for_method_question(tmp_path):
    md_path = tmp_path / "single-photon.en.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "## Abstract",
                (
                    "The SPAD detector operates in Geiger mode above breakdown voltage and "
                    "uses an active quenching circuit to reset after each photon event."
                ),
                "<!-- kb_page: 12 -->",
                "## Author Biographies",
                (
                    "The author's research interests include Geiger mode detectors, "
                    "breakdown voltage, active quenching circuits, and single-photon imaging."
                ),
            ]
        ),
        encoding="utf-8",
    )
    pack = {
        "prompt": "Why does a SPAD operate above breakdown voltage in Geiger mode and require quenching?",
        "answer": (
            "It operates in Geiger mode above breakdown voltage; after a photon triggers "
            "an avalanche, the active quenching circuit terminates it and resets the detector."
        ),
        "hits": [
            {
                "text": "The author's research interests include single-photon imaging.",
                "meta": {"source_path": str(md_path)},
                "ui_meta": {
                    "source_path": str(md_path),
                    "display_name": "Single-photon detector review",
                },
            }
        ],
    }

    out = _attach_pack_primary_ref_evidence(pack)

    primary = out["primary_evidence"]
    assert primary["heading_path"] == "Abstract"
    assert "active quenching circuit" in primary["snippet"]
    assert "research interests" not in primary["snippet"]


def test_spad_prompt_contract_prefers_principle_block_and_keeps_quenching(tmp_path):
    md_path = tmp_path / "single-photon.en.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "## 1 Introduction",
                (
                    "SPAD devices operate in Geiger mode above breakdown voltage. "
                    "Figure 2 also illustrates a quenching circuit."
                ),
                "<!-- kb_page: 2 -->",
                "## 2.1.1 Principle of single photon detection avalanche diode",
                (
                    "Single photon avalanche diode (SPAD) is a p-n junction that operates "
                    "in Geiger mode. The device uses a bias higher than its reverse bias "
                    "breakdown voltage. After avalanche current is detected, the quenching "
                    "circuit terminates the current and resets the detector."
                ),
            ]
        ),
        encoding="utf-8",
    )
    pack = {
        "prompt": "Why does a SPAD operate above breakdown voltage in Geiger mode and require quenching?",
        "hits": [
            {
                "text": "A generic detector review passage.",
                "meta": {
                    "source_path": str(md_path),
                    "ref_answer_citation_num": 1,
                },
                "ui_meta": {
                    "source_path": str(md_path),
                    "display_name": "Single-photon detector review",
                    "primary_evidence": {
                        "source_path": str(md_path),
                        "heading_path": "1 Introduction",
                        "snippet": "SPAD devices operate in Geiger mode.",
                        "block_id": "blk-intro",
                        "anchor_id": "p-intro",
                        "selection_reason": "answer_citation_grounded",
                        "strict_locate": True,
                    },
                },
            }
        ],
    }

    out = _attach_pack_primary_ref_evidence(pack)

    primary = out["primary_evidence"]
    assert "Principle of single photon detection avalanche diode" in primary["heading_path"]
    assert primary["page_start"] == 2
    assert "operates in Geiger mode" in primary["snippet"]
    assert "breakdown voltage" in primary["snippet"]
    assert "quenching circuit" in primary["snippet"]
    hit_primary = out["hits"][0]["ui_meta"]["primary_evidence"]
    assert "Principle of single photon detection avalanche diode" in hit_primary["heading_path"]
    assert hit_primary["selection_reason"] == "prompt_contract_block"


def test_answer_alignment_cannot_replace_complete_spad_prompt_contract(tmp_path):
    md_path = tmp_path / "single-photon.en.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 2 -->",
                "## 2.1.1 Principle of single photon detection avalanche diode",
                (
                    "Single photon avalanche diode (SPAD) is a p-n junction that operates "
                    "in Geiger mode. The device uses a bias higher than its reverse bias "
                    "breakdown voltage. The quenching circuit terminates the avalanche."
                ),
                "<!-- kb_page: 4 -->",
                "## 2.1.3 SPAD based on low-dimensional materials",
                (
                    "Low-dimensional materials improve single photon detection through "
                    "photogating and double-gate modulation."
                ),
            ]
        ),
        encoding="utf-8",
    )
    pack = {
        "prompt": "SPAD 为什么要工作在 Geiger 模式，雪崩后为什么需要淬灭电路？",
        "answer": "低维材料可通过光栅控效应提高单光子探测性能。",
        "hits": [
            {
                "text": "Low-dimensional materials improve single photon detection.",
                "meta": {"source_path": str(md_path)},
                "ui_meta": {
                    "source_path": str(md_path),
                    "display_name": "Single-photon detector review",
                },
            }
        ],
    }

    out = _attach_pack_primary_ref_evidence(pack)

    primary = out["primary_evidence"]
    assert "Principle of single photon detection avalanche diode" in primary["heading_path"]
    assert "operates in Geiger mode" in primary["snippet"]
    assert "breakdown voltage" in primary["snippet"]
    assert "quenching circuit" in primary["snippet"]
    assert out["primary_evidence_alignment"]["selected_source"] == "prompt_contract"


def test_converged_answer_primary_is_reused_without_rescoring(monkeypatch):
    from api import reference_ui

    primary = {
        "source_path": "paper.en.md",
        "heading_path": "Results",
        "snippet": "A strictly located answer-specific result.",
        "block_id": "blk-result",
        "anchor_id": "p-result",
        "strict_locate": True,
        "selection_reason": "answer_aligned_block",
    }
    pack = {
        "prompt": "What result does this paper report?",
        "answer": "The paper reports an answer-specific result.",
        "primary_evidence": primary,
        "hits": [
            {
                "text": primary["snippet"],
                "meta": {"source_path": "paper.en.md"},
                "ui_meta": {"source_path": "paper.en.md"},
            }
        ],
    }

    monkeypatch.setattr(
        reference_ui,
        "_select_answer_aligned_primary_ref_evidence",
        lambda **_kwargs: pytest.fail("converged primary should not be rescored"),
    )

    out = _attach_pack_primary_ref_evidence(pack)

    assert out["primary_evidence"]["block_id"] == "blk-result"
    assert out["primary_evidence_alignment"]["selected_source"] == "reused_converged_primary"


def test_prompt_only_cassi_card_uses_exact_architecture_block(tmp_path):
    md_path = tmp_path / "cassi.en.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "## Abstract",
                (
                    "The primary features of the system design are two dispersive elements, "
                    "arranged in opposition and surrounding a binary-valued aperture code. "
                    "This structure produces spatially-varying spectral filter functions."
                ),
                "<!-- kb_page: 8 -->",
                "## Conclusions",
                "We described a single-shot spectral imager based on compressive sensing ideas.",
            ]
        ),
        encoding="utf-8",
    )
    pack = {
        "prompt": "CASSI 的双色散结构怎么摆，为什么中间放二值孔径？",
        "hits": [
            {
                "text": "We described a single-shot spectral imager based on compressive sensing ideas.",
                "meta": {"source_path": str(md_path)},
                "ui_meta": {
                    "source_path": str(md_path),
                    "display_name": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                    "primary_evidence": {
                        "source_path": str(md_path),
                        "heading_path": "Conclusions",
                        "snippet": "We described a single-shot spectral imager based on compressive sensing ideas.",
                    },
                },
            }
        ],
    }

    out = _attach_pack_primary_ref_evidence(pack)

    primary = out["primary_evidence"]
    assert primary["heading_path"] == "Abstract"
    assert primary["page_start"] == 1
    assert "two dispersive elements" in primary["snippet"]
    assert "binary-valued aperture code" in primary["snippet"]
    assert out["primary_evidence_alignment"]["selected_source"] == "prompt_contract"
    ui = out["hits"][0]["ui_meta"]
    assert ui["heading_path"] == "Abstract"
    assert "CASSI" in ui["why_line"]


def test_prompt_only_cassi_card_resolves_public_source_id(tmp_path, monkeypatch):
    from api import reference_ui

    md_root = tmp_path / "md"
    db_root = tmp_path / "db"
    md_root.mkdir()
    db_root.mkdir()
    md_path = md_root / "CASSI" / "cassi.en.md"
    md_path.parent.mkdir()
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "## Abstract",
                (
                    "The primary features of the system design are two dispersive elements, "
                    "arranged in opposition and surrounding a binary-valued aperture code."
                ),
            ]
        ),
        encoding="utf-8",
    )
    source_id = root_relative_file_id(md_path, [md_root, tmp_path / "tmp"])
    monkeypatch.setattr(reference_ui, "load_settings", lambda: SimpleNamespace(db_dir=db_root))
    monkeypatch.setattr(reference_ui, "load_prefs", lambda: {"md_dir": str(md_root)})
    pack = {
        "prompt": "CASSI 的双色散结构怎么摆，为什么中间放二值孔径？",
        "hits": [
            {
                "text": "A generic conclusion.",
                "meta": {"source_path": source_id},
                "ui_meta": {
                    "source_path": source_id,
                    "display_name": "Single-shot compressive spectral imaging",
                },
            }
        ],
    }

    out = _attach_pack_primary_ref_evidence(pack)

    assert out["primary_evidence"]["heading_path"] == "Abstract"
    assert out["primary_evidence"]["source_path"] == source_id
    assert "two dispersive elements" in out["hits"][0]["ui_meta"]["summary_line"]
