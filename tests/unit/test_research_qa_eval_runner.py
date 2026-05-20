from __future__ import annotations

from tools.research_qa.run_research_qa_eval import (
    load_fixture,
    source_path_for_doc,
    validate_case,
)


def _case_by_id(fixture, case_id: str):
    for case in fixture.cases:
        if case.get("id") == case_id:
            return case
    raise AssertionError(f"missing case {case_id}")


def test_research_qa_fixture_loads_shared_docs_and_cases():
    fixture = load_fixture()

    assert len(fixture.docs) == 21
    assert len(fixture.cases) == 9
    assert source_path_for_doc(fixture, "scinerf").endswith(
        "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image/"
        "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md"
    )


def test_validate_case_accepts_grounded_system_b_answer():
    fixture = load_fixture()
    case = _case_by_id(fixture, "scinerf-admm-origin")
    scinerf_path = source_path_for_doc(fixture, "scinerf")
    answer = "ADMM 不是 SCINeRF 作者新提出的模块，而是已有的优化框架；论文只是把它放在相关工作里说明来源。"
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 101,
        "assistant_message": {
            "role": "assistant",
            "content": answer,
            "cite_details": [
                {
                    "source_path": scinerf_path,
                    "source_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                    "is_inpaper": True,
                    "title": "Distributed optimization and statistical learning via the alternating direction method of multipliers",
                    "raw": "Distributed optimization and statistical learning via ADMM.",
                }
            ],
        },
        "refs_payload": {
            "101": {
                "hits": [
                    {
                        "text": "SCINeRF discusses ADMM in Related Work.",
                        "ui_meta": {
                            "source_path": scinerf_path,
                            "display_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                            "summary_line": "这段 Related Work 把 ADMM 放在已有优化框架脉络里，而不是当作本文创新点。",
                            "why_line": "用户问 ADMM 的来源，这张卡定位到论文解释该方法背景和引用来源的位置。",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is True
    assert quality["system_b_count"] == 1


def test_validate_case_rejects_template_answer_and_missing_system_b():
    fixture = load_fixture()
    case = _case_by_id(fixture, "scinerf-admm-origin")
    scinerf_path = source_path_for_doc(fixture, "scinerf")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 102,
        "assistant_message": {
            "role": "assistant",
            "content": "The paper cites [4] for this point. This is stated in SCINeRF / Related Work.",
            "cite_details": [
                {
                    "source_path": scinerf_path,
                    "source_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                    "is_inpaper": False,
                }
            ],
        },
        "refs_payload": {
            "102": {
                "hits": [
                    {
                        "text": "SCINeRF Related Work mentions ADMM.",
                        "ui_meta": {
                            "source_path": scinerf_path,
                            "display_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                            "summary_line": "直接定义或解释了 ADMM。",
                            "why_line": "适合作为定位入口。",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert quality["ok"] is False
    assert "answer_no_template_phrase" in failed_names
    assert "system_b_present" in failed_names
    assert "refs_card_copy_quality" in failed_names


def test_validate_case_accepts_common_zh_en_synonyms():
    fixture = load_fixture()
    case = _case_by_id(fixture, "single-photon-pidl")
    source_path = source_path_for_doc(fixture, "pidl-single-photon")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 103,
        "assistant_message": {
            "role": "assistant",
            "content": "physics-informed deep learning 用 SPAD 物理模型处理多种噪声，并改善单光子成像质量。",
            "cite_details": [
                {
                    "source_path": source_path,
                    "source_name": "High-resolution single-photon imaging with physics-informed deep learning",
                    "is_inpaper": False,
                }
            ],
        },
        "refs_payload": {
            "103": {
                "hits": [
                    {
                        "text": "SPAD physical noise model.",
                        "ui_meta": {
                            "source_path": source_path,
                            "display_name": "High-resolution single-photon imaging with physics-informed deep learning",
                            "summary_line": "摘要说明 physics-informed deep learning 如何建模 SPAD 的多种物理噪声。",
                            "why_line": "用户问这种方法帮了什么，这张卡定位到噪声模型和训练数据的证据。",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is True


def test_validate_case_accepts_scientific_translation_synonyms():
    fixture = load_fixture()
    case = _case_by_id(fixture, "s2ism-thick-samples")
    source_path = source_path_for_doc(fixture, "s2ism")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 104,
        "assistant_message": {
            "role": "assistant",
            "content": "s2ISM 的权衡是空间分辨率、信噪比和光学层切能力；厚样本会带来离焦背景。",
            "cite_details": [
                {
                    "source_path": source_path,
                    "source_name": "Structured detection for high-SNR image scanning microscopy in thick samples",
                    "is_inpaper": False,
                }
            ],
        },
        "refs_payload": {
            "104": {
                "hits": [
                    {
                        "text": "Structured detection improves sectioning and SNR.",
                        "ui_meta": {
                            "source_path": source_path,
                            "display_name": "Structured detection for high-SNR image scanning microscopy in thick samples",
                            "summary_line": "摘要说明 s2ISM 处理厚样本时的分辨率、信噪比和层切权衡。",
                            "why_line": "用户问 trade-off，这张卡定位到论文直接解释厚样本成像瓶颈的证据。",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is True


def test_validate_case_accepts_foveated_chinese_paraphrase():
    fixture = load_fixture()
    case = _case_by_id(fixture, "foveated-dynamic-supersampling")
    source_path = source_path_for_doc(fixture, "foveated-spi")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 105,
        "assistant_message": {
            "role": "assistant",
            "content": "dynamic supersampling 会围绕重点区域做自适应采样，并通过 supersampling 提升细节。",
            "cite_details": [
                {
                    "source_path": source_path,
                    "source_name": "Adaptive foveated single-pixel imaging with dynamic supersampling",
                    "is_inpaper": False,
                }
            ],
        },
        "refs_payload": {
            "105": {
                "hits": [
                    {
                        "text": "Adaptive foveated single-pixel imaging with dynamic supersampling.",
                        "ui_meta": {
                            "source_path": source_path,
                            "display_name": "Adaptive foveated single-pixel imaging with dynamic supersampling",
                            "summary_line": "论文说明重点区域如何使用更高采样密度。",
                            "why_line": "用户问 dynamic supersampling 的直觉，这张卡定位到自适应采样机制。",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is True


def test_validate_case_checks_primary_evidence_terms():
    fixture = load_fixture()
    source_path = source_path_for_doc(fixture, "qclfm")
    case = {
        "id": "primary-evidence-smoke",
        "expected": {
            "requiredPrimaryEvidenceTerms": ["Digital Refocusing", "ray tracing", "wave propagation"],
            "forbiddenPrimaryEvidenceTerms": ["Conventional light-field microscope designs"],
        },
    }
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 201,
        "assistant_message": {
            "role": "assistant",
            "content": "The method uses ray tracing and then wave propagation to refocus the sample.",
        },
        "refs_payload": {
            "201": {
                "primary_evidence": {
                    "source_path": source_path,
                    "heading_path": "Digital Refocusing Procedure",
                    "snippet": "The trajectory of the photons is determined through ray tracing; the second step applies wave propagation.",
                },
                "hits": [
                    {
                        "ui_meta": {
                            "source_path": source_path,
                            "display_name": "Quantum correlation light-field microscope",
                            "summary_line": "Digital refocusing is described with ray tracing and wave propagation.",
                            "why_line": "This card points to the procedure that explains the answer.",
                        }
                    }
                ],
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is True


def test_validate_case_rejects_misaligned_primary_evidence():
    fixture = load_fixture()
    source_path = source_path_for_doc(fixture, "qclfm")
    case = {
        "id": "primary-evidence-smoke",
        "expected": {
            "requiredPrimaryEvidenceTerms": ["Digital Refocusing", "ray tracing", "wave propagation"],
            "forbiddenPrimaryEvidenceTerms": ["Conventional light-field microscope designs"],
        },
    }
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 202,
        "assistant_message": {
            "role": "assistant",
            "content": "The method uses ray tracing and then wave propagation to refocus the sample.",
        },
        "refs_payload": {
            "202": {
                "primary_evidence": {
                    "source_path": source_path,
                    "heading_path": "I. INTRODUCTION",
                    "snippet": "Conventional light-field microscope designs typically make use of a microlens array.",
                },
                "hits": [
                    {
                        "ui_meta": {
                            "source_path": source_path,
                            "display_name": "Quantum correlation light-field microscope",
                            "summary_line": "The introduction describes conventional light-field microscopes.",
                            "why_line": "This card points to broad background rather than the refocusing procedure.",
                        }
                    }
                ],
            }
        },
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert quality["ok"] is False
    assert "primary_evidence_contains_required_terms" in failed_names
    assert "primary_evidence_avoids_forbidden_terms" in failed_names
