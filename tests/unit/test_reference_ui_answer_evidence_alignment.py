from __future__ import annotations

from api.reference_ui import _attach_pack_primary_ref_evidence


def test_attach_pack_primary_evidence_prefers_answer_aligned_source_block(tmp_path):
    md_path = tmp_path / "qclfm.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Quantum correlation light-field microscope with extreme depth of field",
                "",
                "## I. INTRODUCTION",
                "",
                "Conventional light-field microscope designs typically make use of a microlens array.",
                "",
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
    alignment = out["primary_evidence_alignment"]
    assert alignment["mismatch"] is False
    assert alignment["selected_source"] == "source_blocks"
