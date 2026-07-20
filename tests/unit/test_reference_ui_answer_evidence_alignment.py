from __future__ import annotations

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
