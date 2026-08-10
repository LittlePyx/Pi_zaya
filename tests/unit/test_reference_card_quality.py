from __future__ import annotations

from api.reference_card_quality import (
    _repair_ref_card_copy_locale,
    attach_ref_card_polish_contract,
    attach_refs_pack_polish_contract,
    citation_detail_quality,
    citation_shelf_item_quality,
    ref_card_hit_quality,
    refs_pack_has_full_llm_copy,
    summarize_citation_detail_quality,
    summarize_citation_shelf_quality,
    summarize_ref_card_hit_quality,
)
from api.reference_card_payload import build_ref_card_ui_payload


def test_sequential_cs_card_gets_grounded_chinese_guide() -> None:
    evidence = (
        "The first stage involves log_2 log n steps and removes zero components. "
        "The second stage faces a lower dimensional problem, and support can be "
        "recovered exactly at much lower SNRs."
    )
    ui = _repair_ref_card_copy_locale(
        {
            "render_locale": "zh",
            "summary_kind": "guide",
            "summary_line": "",
            "why_line": "",
            "display_name": "SSP-2012-Sequentially designed compressed sensing",
            "heading_path": "II. MAIN RESULT",
            "primary_evidence": {"snippet": evidence},
        }
    )

    assert "第一阶段的筛除步数和剩余维度界" in ui["summary_line"]
    assert "低 SNR" in ui["summary_line"]
    assert ui["summary_generation"] == "deterministic_grounded"

    short_seed = _repair_ref_card_copy_locale(
        {
            "render_locale": "zh",
            "summary_kind": "guide",
            "summary_line": "共进行 \\log_2 \\log n 步，每步使用若干稀疏压缩感知矩阵",
            "why_line": "",
            "display_name": "Sequentially designed compressed sensing",
            "primary_evidence": {
                "snippet": "The first stage involves \\log_2 \\log n steps."
            },
        }
    )
    assert "量化了第一阶段的步数与稀疏测量预算" in short_seed["why_line"]
    assert short_seed["why_generation"] == "deterministic_grounded"


def test_qclfm_short_evidence_seed_gets_grounded_card_copy() -> None:
    evidence = (
        "We demonstrate an LFM design based on the inherent position and "
        "angular/momentum correlation of entangled photon pairs generated through SPDC."
    )
    ui = _repair_ref_card_copy_locale(
        {
            "render_locale": "zh",
            "summary_kind": "guide",
            "summary_line": "",
            "why_line": "",
            "display_name": "arXiv-Quantum correlation light-field microscope",
            "primary_evidence": {"snippet": evidence},
        }
    )

    assert "QCLFM" in ui["summary_line"]
    assert "位置—角度/动量关联" in ui["summary_line"]
    assert "双分辨率机制" in ui["why_line"]


def test_three_d_video_daq_block_gets_grounded_relevance_copy() -> None:
    evidence = (
        "The DAQ has a maximum acquisition rate of 250 kHz for all channels. "
        "Each of four channels is set to 62.5 kHz. Given that each pattern is "
        "displayed for 50 μs, there are approximately three samples per pattern."
    )
    ui = _repair_ref_card_copy_locale(
        {
            "render_locale": "zh",
            "summary_kind": "guide",
            "summary_line": "DAQ 总采样率按四通道均分，每通道为 62.5 kHz。",
            "why_line": "",
            "display_name": "Journal of Optics-2016-3D single-pixel video.pdf",
            "primary_evidence": {"snippet": evidence},
        }
    )

    assert "250 kHz 总采样预算" in ui["why_line"]
    assert "每个图案为何约得到 3 个样本" in ui["why_line"]


def test_fdm_non_awg_short_seed_gets_grounded_card_copy() -> None:
    evidence = (
        "Our frequency-division multiplexing (FDM) approach to imaging may realize "
        "additional advantages if the system noise is not AWG."
    )
    ui = _repair_ref_card_copy_locale(
        {
            "render_locale": "zh",
            "summary_kind": "guide",
            "summary_line": "",
            "why_line": "",
            "display_name": (
                "Optica-2016-Frequency-division-multiplexed single-pixel imaging "
                "with metamaterials"
            ),
            "primary_evidence": {"snippet": evidence},
        }
    )

    assert "系统噪声不是 AWG" in ui["summary_line"]
    assert "偏离 AWG 假设" in ui["why_line"]
    assert ui["summary_generation"] == "deterministic_grounded"
    assert ui["why_generation"] == "deterministic_grounded"


def test_ref_card_polish_contract_marks_full_llm_card():
    ui = attach_ref_card_polish_contract(
        {
            "summary_kind": "guide",
            "summary_generation": "llm_grounded",
            "why_generation": "llm_grounded",
        }
    )

    assert ui["polish_status"] == "full"
    assert ui["summary_polish_status"] == "full"
    assert ui["why_polish_status"] == "full"
    assert ui["polish_source"] == "llm"


def test_ref_card_polish_contract_marks_pending_before_llm_copy():
    ui = attach_ref_card_polish_contract(
        {
            "summary_kind": "guide",
            "summary_generation": "pending_section_seed",
            "why_generation": "pending_focus_seed",
            "score_pending": True,
        },
        hit_meta={"ref_pack_state": "pending"},
    )

    assert ui["polish_status"] == "pending"
    assert ui["summary_polish_status"] == "pending"
    assert ui["why_polish_status"] == "pending"


def test_ref_card_polish_contract_replaces_title_echo_with_specific_guide():
    title = "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning"
    ui = attach_ref_card_polish_contract(
        {
            "display_name": f"{title}.pdf",
            "citation_meta": {"title": title},
            "summary_kind": "guide",
            "summary_line": (
                "单像素成像综述（Advances and Challenges of Single-Pixel Imaging "
                "Based on Deep Learning）讨论了光子级成像的深度学习重建方法"
            ),
            "why_line": (
                "摘要明确指出迭代重建同时受图像质量和计算耗时限制，"
                "这是判断深度学习为何能改善单像素成像实用性的直接依据。"
            ),
            "summary_generation": "answer_citation_grounded",
            "why_generation": "deterministic_grounded",
        },
        hit_meta={"title": title},
    )

    assert ui["summary_line"] == "该综述指出迭代重建同时受图像质量和计算耗时限制。"
    assert title not in ui["summary_line"]
    assert ui["card_view"]["summary"] == ui["summary_line"]


def test_ref_card_polish_contract_replaces_metadata_led_summary():
    ui = attach_ref_card_polish_contract(
        {
            "display_name": "DL SPI review.pdf",
            "summary_kind": "guide",
            "summary_line": "（LPR, 2025）的摘要指出，深度学习在单像素成像中具有重要作用。",
            "why_line": (
                "摘要明确指出迭代重建同时受图像质量和计算耗时限制，"
                "这是判断深度学习为何能改善单像素成像实用性的直接依据。"
            ),
            "summary_generation": "answer_citation_grounded",
            "why_generation": "deterministic_grounded",
        }
    )

    assert ui["summary_line"] == "该综述指出迭代重建同时受图像质量和计算耗时限制。"


def test_refs_pack_polish_contract_counts_mixed_cards():
    pack = attach_refs_pack_polish_contract(
        {
            "display_state": "ready",
            "hits": [
                {
                    "ui_meta": {
                        "summary_kind": "guide",
                        "summary_generation": "llm_grounded",
                        "why_generation": "llm_grounded",
                    }
                },
                {
                    "ui_meta": {
                        "summary_kind": "guide",
                        "summary_generation": "deterministic_grounded",
                        "why_generation": "deterministic_grounded",
                    }
                },
            ],
        }
    )

    assert pack["polish_status"] == "heuristic"
    assert pack["polish_counts"]["full"] == 1
    assert pack["polish_counts"]["heuristic"] == 1
    assert refs_pack_has_full_llm_copy(pack) is False


def test_refs_pack_full_llm_copy_requires_all_visible_cards_full():
    assert refs_pack_has_full_llm_copy(
        {
            "hits": [
                {
                    "ui_meta": {
                        "summary_kind": "guide",
                        "summary_generation": "llm_grounded",
                        "why_generation": "llm_grounded",
                    }
                }
            ]
        }
    )


def test_ref_card_payload_builder_attaches_polish_contract():
    payload = build_ref_card_ui_payload(
        display_name="Demo.pdf",
        heading_path="2. Method",
        section_label="2. Method",
        subsection_label="",
        page_start=0,
        page_end=0,
        score=9.2,
        score_pending=False,
        score_tier="high",
        summary_line="A concise LLM-grounded summary.",
        summary_kind="guide",
        summary_surface={},
        summary_generation="llm_grounded",
        summary_basis_meta={},
        summary_source="prompt_aligned",
        primary_evidence_heading_path="2. Method",
        primary_evidence={},
        why_line="A concise LLM-grounded relevance note.",
        why_generation="llm_grounded",
        why_basis_meta={},
        anchor_target_kind="",
        anchor_target_number=0,
        anchor_match_score=0.0,
        explicit_doc_match_score=0.0,
        semantic_badges=[],
        can_open=True,
        citation_meta={},
        source_path="demo.en.md",
        reader_open={},
    )

    assert payload["polish_status"] == "full"
    assert payload["polish_contract_version"] == 1
    assert payload["card_view"]["quality"]["label"] == "full"
    assert payload["card_view"]["quality"]["source"] == "llm"
    sections = {section["id"]: section for section in payload["card_view"]["sections"]}
    assert sections["summary"]["text"] == "A concise LLM-grounded summary."
    assert sections["why"]["text"] == "A concise LLM-grounded relevance note."


def test_ref_card_polish_contract_unwraps_source_excerpt_summary():
    ui = attach_ref_card_polish_contract(
        {
            "display_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
            "source_path": "dl-spi-review.en.md",
            "summary_line": (
                "\u539f\u6587\u7247\u6bb5\u5199\u5230\uff1a\u201c"
                "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning: "
                "However, the limited image quality still hinders practical application."
                "\u201d"
            ),
            "why_line": "This card explains a concrete limitation of deep-learning single-pixel imaging.",
            "summary_generation": "deterministic_grounded",
            "why_generation": "deterministic_grounded",
        }
    )

    assert ui["summary_line"].startswith("However, the limited image quality")
    assert "\u539f\u6587\u7247\u6bb5\u5199\u5230" not in ui["summary_line"]
    assert "Advances and Challenges" not in ui["summary_line"]
    assert ui["card_view"]["summary"].startswith("However, the limited image quality")


def test_mostly_english_wrapper_is_localized_and_kept_as_raw_evidence():
    raw = (
        "原文片段写到：Choosing 333 unique patterns yields a reconstruction "
        "frame rate of 30 Hz for multiple image resolutions."
    )
    ui = attach_ref_card_polish_contract(
        {
            "summary_kind": "guide",
            "render_locale": "zh",
            "summary_line": raw,
        }
    )

    assert "333" in ui["summary_line"]
    assert "30 Hz" in ui["summary_line"]
    assert ui["summary_generation"] == "deterministic_grounded"
    assert ui["primary_evidence"]["snippet"].startswith("Choosing 333 unique patterns")
    assert ui["summary_display_role"] == "guide"
    assert ui["summary_label"] == "导读"


def test_ref_card_locale_contract_derives_chinese_guide_and_keeps_raw_evidence():
    evidence = (
        "This next-generation technique combines interferometric detection with image scanning "
        "microscopy to achieve about 120 nm lateral resolution while operating at lower power."
    )
    why = (
        "原文说明干涉检测与图像扫描显微镜结合后达到约 120 nm 横向分辨率，"
        "直接支撑干涉路线如何突破分辨率瓶颈。"
    )
    ui = attach_ref_card_polish_contract(
        {
            "display_name": "Interferometric Image Scanning Microscopy.pdf",
            "summary_kind": "guide",
            "render_locale": "zh",
            "summary_line": "Interferometric Image Scanning Microscopy for label-free imaging at 120 nm.",
            "why_line": why,
            "why_generation": "deterministic_grounded",
            "primary_evidence": {"snippet": evidence, "highlight_snippet": evidence},
        }
    )

    assert ui["summary_line"].startswith("该文将干涉检测与图像扫描显微镜")
    assert ui["why_line"] == why
    assert ui["primary_evidence"]["snippet"] == evidence
    sections = {section["id"]: section for section in ui["card_view"]["sections"]}
    assert set(sections) >= {"summary", "why"}
    assert sections["summary"]["text"] != sections["why"]["text"]


def test_ref_card_view_keeps_relevance_that_extends_the_guide():
    summary = "频分复用把多个空间编码并行到不同频率通道。"
    why = (
        "频分复用把多个空间编码并行到不同频率通道；这解释了为什么一次积分能够"
        "恢复多个测量分量，并明确对应速度提升的来源。"
    )
    ui = attach_ref_card_polish_contract(
        {
            "display_name": "FDM.pdf",
            "summary_kind": "guide",
            "render_locale": "zh",
            "summary_line": summary,
            "why_line": why,
            "summary_generation": "answer_citation_grounded",
            "why_generation": "answer_citation_grounded",
        }
    )

    assert ui["why_line"] == why
    sections = {section["id"]: section for section in ui["card_view"]["sections"]}
    assert sections["summary"]["text"] == summary
    assert sections["why"]["text"] == why


def test_ref_card_locale_contract_builds_relevance_for_compact_fdm_excerpt() -> None:
    evidence = (
        "The modulated light from the SLM is then multiplexed into a single-pixel detector, "
        "which produces a signal containing the phase and modulation frequency information."
    )
    ui = attach_ref_card_polish_contract(
        {
            "display_name": "Frequency-division-multiplexed single-pixel imaging.pdf",
            "heading_path": "B. Encoding",
            "summary_kind": "guide",
            "render_locale": "zh",
            "summary_line": "频分复用单像素成像将空间光调制器（SLM）的像素调制环节并行化。",
            "why_line": "",
            "primary_evidence": {
                "snippet": evidence,
                "highlight_snippet": evidence,
            },
        }
    )

    assert all(term in ui["why_line"] for term in ("SLM", "相位", "调制频率", "FDM"))
    assert ui["why_generation"] == "deterministic_grounded"
    sections = {section["id"]: section for section in ui["card_view"]["sections"]}
    assert sections["why"]["text"] == ui["why_line"]


def test_piln_metadata_guide_replacement_does_not_duplicate_relevance() -> None:
    evidence = (
        "The difference between the I_N(out) of the retrieved image and the "
        "I_N(real) captured by the SPD is used as a loss function to train ILNet. "
        "The generated 2D image serves as input for the subsequent image-loop iteration."
    )
    original_why = (
        "ILNet 在没有配对真值图像时，通过物理模型与 1D 信号标签形成"
        "自监督闭环，而不是依赖配对真值图像。"
    )
    ui = attach_ref_card_polish_contract(
        {
            "display_name": (
                "Part-based image-loop network for single-pixel imaging.pdf"
            ),
            "summary_kind": "guide",
            "render_locale": "zh",
            "summary_line": "已核对的原文要点：" + evidence,
            "why_line": original_why,
            "primary_evidence": {
                "snippet": evidence,
                "highlight_snippet": evidence,
            },
        }
    )

    assert ui["summary_line"] != ui["why_line"]
    assert "一致性损失" in ui["why_line"]
    assert "图像回环" in ui["why_line"]
    sections = {section["id"]: section for section in ui["card_view"]["sections"]}
    assert sections["summary"]["text"] != sections["why"]["text"]


def test_qclfm_card_explains_architecture_tradeoff_in_relevance() -> None:
    evidence = (
        "Since each degree of freedom can be measured on separate cameras, one does "
        "not need to sacrifice position resolution for angular resolution or vice versa."
    )
    ui = attach_ref_card_polish_contract(
        {
            "display_name": "Quantum correlation light-field microscope.pdf",
            "summary_kind": "guide",
            "render_locale": "zh",
            "summary_line": "QCLFM 同时保留位置与角度分辨率，避免传统 LFM 的权衡。",
            "why_line": "",
            "primary_evidence": {"snippet": evidence, "highlight_snippet": evidence},
        }
    )

    assert all(term in ui["why_line"] for term in ("独立相机", "采集架构", "权衡"))
    assert ui["summary_line"] != ui["why_line"]


def test_ref_card_locale_contract_keeps_spi_guide_and_relevance_complementary() -> None:
    evidence = (
        "The DMD can project patterns of light onto a scene, termed structured illumination, "
        "or structure detected image intensities, called structured detection. For the latter, "
        "the DMD is located in an image plane of the object. The modulation rate of the DMD "
        "is the acquisition-time bottleneck."
    )
    summary = (
        "同一段原文既区分了 DMD 在照明侧与探测像面的两种配置，"
        "也明确指出两者共同受 DMD 图案切换速率这一采集瓶颈限制。"
    )
    ui = attach_ref_card_polish_contract(
        {
            "display_name": "Principles and prospects for single-pixel imaging.pdf",
            "heading_path": "Camera architecture",
            "summary_kind": "guide",
            "render_locale": "zh",
            "summary_line": summary,
            "why_line": "",
            "primary_evidence": {"snippet": evidence},
        }
    )

    assert ui["summary_line"] == summary
    assert all(term in ui["why_line"] for term in ("DMD", "照明侧", "探测侧", "像面"))
    assert ui["why_line"] != ui["summary_line"]
    assert ui["why_generation"] == "deterministic_grounded"


def test_ref_card_locale_contract_explains_perovskite_scope_boundary() -> None:
    evidence = (
        "In this work, we demonstrate electrically driven lasing from a dual-cavity "
        "perovskite device, which integrates two microcavity sub-units in a vertically "
        "stacked structure and shows a low lasing threshold."
    )
    ui = attach_ref_card_polish_contract(
        {
            "display_name": (
                "Nature-2025-Electrically driven lasing from a dual-cavity "
                "perovskite device.pdf"
            ),
            "heading_path": "Abstract",
            "summary_kind": "guide",
            "render_locale": "zh",
            "summary_line": evidence,
            "why_line": "",
            "primary_evidence": {"snippet": evidence},
        }
    )

    assert all(term in ui["summary_line"] for term in ("双腔钙钛矿", "电驱动激光", "激射阈值"))
    assert all(term in ui["why_line"] for term in ("器件发光", "单像素成像", "编码", "重建"))
    assert ui["summary_generation"] == "deterministic_grounded"
    assert ui["why_generation"] == "deterministic_grounded"
    sections = {section["id"]: section for section in ui["card_view"]["sections"]}
    assert sections["summary"]["text"] == ui["summary_line"]
    assert sections["why"]["text"] == ui["why_line"]


def test_ref_card_copy_suppresses_located_quote_shell_without_grounded_replacement():
    why = (
        "“Abstract”中的原文直接支撑“频分复用并行采集”，"
        "可据此核对速度提升与探测器积分时间的关系。"
    )
    ui = attach_ref_card_polish_contract(
        {
            "display_name": "Frequency-division multiplexing.pdf",
            "summary_kind": "guide",
            "render_locale": "zh",
            "summary_line": "频分复用通过多频通道实现并行采集。",
            "why_line": why,
        }
    )

    assert ui["why_line"] == ""
    assert ui["why_generation"] == "locale_suppressed"
    assert all(section["id"] != "why" for section in ui["card_view"]["sections"])


def test_detector_review_gets_grounded_relevance_when_compacted_abstract_omits_list() -> None:
    ui = attach_ref_card_polish_contract(
        {
            "display_name": (
                "Emerging single-photon detection technique for high-performance "
                "photodetector.pdf"
            ),
            "heading_path": "Abstract",
            "summary_kind": "guide",
            "render_locale": "zh",
            "summary_line": (
                "\u8be5\u7efc\u8ff0\u8ba8\u8bba\u5355\u5149\u5b50\u63a2\u6d4b\u5668\u4e0e SPAD \u786c\u4ef6\u80cc\u666f\u3002"
            ),
            "why_line": "",
            "primary_evidence": {
                "snippet": (
                    "Conductors, superconductors, semiconductors, and nanowires "
                    "have all been discussed for single-photon detectors."
                )
            },
        }
    )

    assert "SPAD" in ui["why_line"]
    assert "\u786c\u4ef6" in ui["why_line"]
    sections = {section["id"]: section for section in ui["card_view"]["sections"]}
    assert sections["why"]["text"] == ui["why_line"]


def test_answer_citation_copy_cleanup_preserves_localized_guide_and_source_evidence_split():
    raw_summary = (
        "Clearly, the reconstructed results improve as illumination increases. "
        "HATNet generalizes in both low-light and high-light conditions."
    )
    localized_guide = "该实验表明，HATNet 在不同照度条件下均保持了良好的泛化能力。"
    ui = attach_ref_card_polish_contract(
        {
            "display_name": "HATNet.pdf",
            "source_path": "hatnet.en.md",
            "summary_source": "answer_citation",
            "primary_evidence_source": "answer_citation",
            "summary_line": localized_guide,
            "summary_generation": "section_grounded",
            "primary_evidence": {
                "snippet": raw_summary,
                "highlight_snippet": raw_summary,
            },
            "reader_open": {
                "snippet": raw_summary,
                "highlightSnippet": raw_summary,
                "primaryEvidence": {
                    "snippet": raw_summary,
                    "highlight_snippet": raw_summary,
                },
            },
        }
    )

    assert ui["summary_line"] == localized_guide
    assert ui["primary_evidence"]["snippet"] == raw_summary
    assert ui["reader_open"]["snippet"] == raw_summary
    assert ui["reader_open"]["primaryEvidence"]["snippet"] == raw_summary
    assert ui["summary_display_role"] == "guide"
    assert ui["summary_label"] == "导读"


def test_ref_card_hit_quality_accepts_grounded_openable_card():
    quality = ref_card_hit_quality(
        {
            "text": "The method maps low-dimensional measurements back to target images.",
            "meta": {"source_path": "demo.en.md", "ref_pack_state": "ready"},
            "ui_meta": {
                "display_name": "Demo SPI paper",
                "source_path": "demo.en.md",
                "heading_path": "4. Strategy and Advantages / Data-driven strategy",
                "summary_line": "This card explains how the model maps compressed measurements back to images.",
                "why_line": (
                    "The Data-driven strategy section links compressed measurements "
                    "to reconstruction quality under low sampling."
                ),
                "polish_status": "full",
                "can_open": True,
                "reader_open": {
                    "sourcePath": "demo.en.md",
                    "headingPath": "4. Strategy and Advantages",
                    "blockId": "blk-1",
                    "anchorId": "p-1",
                    "snippet": "The encoder samples the image into low-dimensional measurements.",
                },
            },
        }
    )

    assert quality["ok"] is True
    assert quality["score"] == 1.0


def test_citation_detail_quality_accepts_complete_has_attracted_evidence():
    quality = citation_detail_quality(
        {
            "num": 2,
            "anchor": "dl-a2",
            "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
            "source_path": "dl-spi-review.en.md",
            "heading_path": "Abstract",
            "card_evidence": (
                "Single-pixel imaging technology can capture images outside conventional focal plane arrays. "
                "Recently, single-pixel imaging based on deep learning has attracted a lot of attention "
                "due to its exceptional reconstruction quality and fast reconstruction speed."
            ),
            "card_claim": "The answer uses this review as the deep-learning SPI overview.",
        }
    )

    assert quality["ok"] is True


def test_ref_card_hit_quality_rejects_template_duplicate_and_broken_copy():
    quality = ref_card_hit_quality(
        {
            "text": "## Foveated single-pixel imaging has attrac...",
            "ui_meta": {
                "display_name": "Foveated SPI",
                "summary_line": "This hit is directly relevant to the user question.",
                "why_line": "This hit is directly relevant to the user question.",
                "polish_status": "sparkly",
                "reader_open": {"sourcePath": "demo.en.md", "snippet": "## Foveated single-pixel imaging has attrac..."},
            },
        },
        forbidden_phrases=["directly relevant"],
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "ref_card_template_phrase_visible" in names
    assert "ref_card_duplicate_summary_why" in names
    assert "ref_card_forbidden_phrase" in names
    assert "ref_card_unknown_polish_status" in names
    assert "ref_card_raw_markdown_visible" in names
    assert "ref_card_broken_evidence" in names


def test_ref_card_hit_quality_rejects_localized_generic_why_copy():
    quality = ref_card_hit_quality(
        {
            "meta": {"source_path": "SCINeRF.en.md"},
            "ui_meta": {
                "display_name": "SCINeRF.pdf",
                "summary_line": "The abstract introduces a physical SCI imaging process for NeRF reconstruction.",
                "why_line": "“Abstract”提供回答该问题所需的原文定位，卡片中的结论可在这里逐项核对。",
            },
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "ref_card_generic_why_visible" in names


def test_summarize_ref_card_hit_quality_indexes_failures():
    summary = summarize_ref_card_hit_quality(
        [
            {
                "ui_meta": {
                    "display_name": "Good",
                    "summary_line": "This card has a focused summary for the answer.",
                    "why_line": "It explains why this source belongs in the answer.",
                }
            },
            {"ui_meta": {"summary_line": "short", "why_line": "short"}},
        ]
    )

    assert summary["ok"] is False
    assert summary["count"] == 2
    assert summary["ok_count"] == 1
    assert any(item["index"] == 2 and item["name"] == "ref_card_summary_too_short" for item in summary["failures"])


def test_citation_detail_quality_accepts_grounded_system_a_card():
    quality = citation_detail_quality(
        {
            "num": 1,
            "anchor": "a1",
            "source_name": "Demo.pdf",
            "source_path": "demo.en.md",
            "heading_path": "2. Method / Reconstruction",
            "evidence_quote": "The method maps low-dimensional measurements back to target images with a learned decoder.",
            "answer_claim": "The method improves reconstruction from fewer measurements.",
            "support_relation": "The quoted sentence explains the encoder-decoder measurement mapping.",
        }
    )

    assert quality["ok"] is True
    assert quality["route"] == "system_a"


def test_citation_detail_quality_rejects_visible_weak_system_a_binding():
    quality = citation_detail_quality(
        {
            "num": 1,
            "anchor": "a1",
            "source_name": "3D single-pixel video.pdf",
            "source_path": "demo.en.md",
            "heading_path": "Methods / Photometric stereo",
            "evidence_quote": "Photometric stereo estimates surface orientation from different illumination directions.",
            "answer_claim": "Hadamard subsampling is useful for real-time low-sampling imaging.",
            "binding_status": "candidate",
            "binding_confidence": 0.35,
            "card_quality_flags": ["candidate_binding"],
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "system_a_weak_binding_visible" in names


def test_citation_detail_quality_rejects_raw_markdown_and_fragmented_evidence():
    quality = citation_detail_quality(
        {
            "num": 2,
            "anchor": "a2",
            "source_name": "Foveated SPI.pdf",
            "heading_path": "INTRODUCTION",
            "evidence_quote": "## Foveated single-pixel imaging has attrac...",
            "answer_claim": "This is a method card.",
            "support_relation": "This quote supports the answer.",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "raw_markdown_visible" in names
    assert "system_a_broken_evidence" in names


def test_citation_detail_quality_rejects_duplicate_visible_card_text():
    quality = citation_detail_quality(
        {
            "num": 1,
            "anchor": "a1",
            "source_name": "Demo.pdf",
            "heading_path": "2. Method",
            "card_takeaway": "深度学习模型把低维测量映射回目标图像，从而提升重建质量。",
            "card_evidence": "深度学习模型把低维测量映射回目标图像，从而提升重建质量。",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "duplicate_visible_card_text" in names


def test_citation_detail_quality_ignores_duplicate_hidden_card_sections():
    quality = citation_detail_quality(
        {
            "num": 1,
            "anchor": "a1",
            "source_name": "Demo.pdf",
            "heading_path": "2. Related Work",
            "location_label": "2. Related Work / paragraph",
            "card_claim": "Most existing methods employ ADMM [4].",
            "card_evidence": "Most existing methods employ ADMM [4].",
            "card_visible_sections": ["evidence", "locator"],
            "binding_status": "grounded",
            "binding_confidence": 0.9,
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert "duplicate_visible_card_text" not in names


def test_citation_detail_quality_rejects_metadata_repeated_in_card_copy():
    quality = citation_detail_quality(
        {
            "num": 2,
            "anchor": "r2",
            "is_inpaper": True,
            "source_name": "Current paper.pdf",
            "title": "Optical imaging by means of two-photon quantum entanglement",
            "venue": "Physical Review A",
            "year": "1995",
            "raw": "Pittman T, Shih Y. Optical imaging by means of two-photon quantum entanglement. Physical Review A, 1995.",
            "heading_path": "1. Introduction",
            "answer_claim": "单像素成像可以降低成像成本。",
            "citation_context": "Unlike traditional focal plane array detectors, SPI only adopts a SPD to collect echo signals.",
            "card_takeaway": "这篇发表于 Physical Review A 1995 的论文值得打开。",
            "system_b_trace_complete": True,
            "system_b_trace_score": 0.8,
            "system_b_trace_reference": "Pittman T, Shih Y. Optical imaging by means of two-photon quantum entanglement.",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "narrative_metadata_repeated" in names


def test_citation_detail_quality_accepts_system_b_support_relation_language():
    quality = citation_detail_quality(
        {
            "num": 4,
            "anchor": "r4",
            "is_inpaper": True,
            "source_name": "SCINeRF Neural Radiance Fields from a Snapshot Compressive Image.pdf",
            "title": "Distributed Optimization and Statistical Learning via ADMM",
            "raw": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
            "heading_path": "SCINeRF / 2. Related Work",
            "answer_claim": "ADMM is prior optimization background, not a new SCINeRF invention.",
            "citation_context": "Most existing methods employ ADMM-based optimization for snapshot compressive imaging.",
            "card_takeaway": "This upstream work provides the optimization framework behind the cited ADMM method.",
            "card_support_explanation": "It maps the answer claim back to a reference cited by the current paper.",
            "system_b_trace_complete": True,
            "system_b_trace_score": 0.82,
            "system_b_trace_reference": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is True
    assert "narrative_metadata_repeated" not in names


def test_citation_detail_quality_allows_answer_claim_to_name_reading_pair_title():
    quality = citation_detail_quality(
        {
            "num": 2,
            "anchor": "r2",
            "source_name": "NatCommun-2023-High-resolution single-photon imaging with physics-informed deep learning.pdf",
            "title": "High-resolution single-photon imaging with physics-informed deep learning",
            "heading_path": "Abstract",
            "answer_claim": (
                "结论：建议先读 Emerging single-photon detection technique for high-performance "
                "photodetector，再读 High-resolution single-photon imaging with physics-informed deep learning。"
            ),
            "evidence_quote": "The method incorporates physical noise models into deep learning reconstruction.",
            "card_claim": (
                "结论：建议先读 Emerging single-photon detection technique for high-performance "
                "photodetector，再读 High-resolution single-photon imaging with physics-informed deep learning。"
            ),
            "card_takeaway": "This evidence supports the second paper as an applied follow-up after detector background.",
            "card_support_explanation": "It grounds the reading order by connecting detector noise physics to reconstruction.",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is True
    assert "narrative_metadata_repeated" not in names


def test_ref_card_hit_quality_accepts_being_cited_as_prior_work_language():
    quality = ref_card_hit_quality(
        {
            "text": "Most existing SCI methods employ ADMM-based optimization.",
            "meta": {
                "source_path": "SCINeRF Neural Radiance Fields from a Snapshot Compressive Image.en.md",
                "title": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
            },
            "ui_meta": {
                "display_name": "SCINeRF Neural Radiance Fields from a Snapshot Compressive Image.pdf",
                "heading_path": "5. Conclusion",
                "summary_line": "结论部分说明 ADMM 在文中属于已有优化工具，而不是作者新提出的方法。",
                "why_line": "结论部分提及 ADMM，可验证该算法在文中是作为现有优化工具被引用，而非作者原创方法。",
                "card_view": {
                    "sections": [
                        {
                            "id": "summary",
                            "text": "结论部分说明 ADMM 在文中属于已有优化工具，而不是作者新提出的方法。",
                        },
                        {
                            "id": "why",
                            "text": "结论部分提及 ADMM，可验证该算法在文中是作为现有优化工具被引用，而非作者原创方法。",
                        },
                    ]
                },
                "reader_open": {"sourcePath": "SCINeRF.en.md", "headingPath": "5. Conclusion"},
            },
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is True
    assert "ref_card_narrative_metadata_repeated" not in names


def test_citation_detail_quality_accepts_grounded_system_b_card():
    quality = citation_detail_quality(
        {
            "num": 4,
            "anchor": "r4",
            "is_inpaper": True,
            "source_name": "SCINeRF Neural Radiance Fields from a Snapshot Compressive Image.pdf",
            "title": "Distributed Optimization and Statistical Learning via ADMM",
            "raw": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
            "heading_path": "SCINeRF / 2. Related Work / Snapshot Compressive Imaging",
            "answer_claim": "ADMM is prior optimization background, not a new SCINeRF invention.",
            "citation_context": "Most existing methods employ ADMM-based optimization for snapshot compressive imaging.",
            "upstream_work_role": "This upstream work provides the optimization framework behind the cited ADMM method.",
            "user_question_relation": "The citation shows ADMM is prior work rather than a new SCINeRF contribution.",
            "system_b_trace_complete": True,
            "system_b_trace_score": 0.82,
            "system_b_trace_steps": ["答案句", "当前论文引用处", "上游文献"],
            "system_b_trace_answer": "ADMM is prior optimization background, not a new SCINeRF invention.",
            "system_b_trace_context": "Most existing methods employ ADMM-based optimization for snapshot compressive imaging.",
            "system_b_trace_reference": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
        }
    )

    assert quality["ok"] is True
    assert quality["route"] == "system_b"


def test_citation_detail_quality_rejects_weak_system_b_card():
    quality = citation_detail_quality(
        {
            "ref_num": 3,
            "is_inpaper": True,
            "source_name": "Paper.pdf",
            "raw": "Missing cone problem and low-pass distortion.",
            "heading_path": "Unknown location",
            "citation_context": "Missing cone problem and low-pass distortion.",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "missing_click_anchor" in names
    assert "system_b_missing_takeaway" in names
    assert "system_b_missing_locator" in names
    assert "system_b_missing_answer_claim" in names


def test_summarize_citation_detail_quality_counts_routes_and_failures():
    summary = summarize_citation_detail_quality(
        [
            {
                "num": 1,
                "anchor": "a1",
                "source_name": "Demo.pdf",
                "heading_path": "Abstract",
                "evidence_quote": "A complete evidence sentence explains the answer in enough detail.",
            },
            {
                "is_inpaper": True,
                "source": "inline_marker",
                "ref_num": "4",
                "raw": "inline marker only",
            },
        ]
    )

    assert summary["ok"] is False
    assert summary["route_counts"] == {"system_a": 1, "system_b": 1}
    assert summary["ok_route_counts"]["system_a"] == 1
    assert any(item["name"] == "inline_marker_not_rendered" for item in summary["failures"])
    assert summary["system_b_audit"]["system_b_total"] == 1
    assert summary["system_b_audit"]["needs_review_count"] == 1
    assert summary["system_b_audit"]["review_examples"]


def test_summarize_citation_detail_quality_audits_system_b_sources():
    summary = summarize_citation_detail_quality(
        [
            {
                "num": 4,
                "anchor": "r4",
                "is_inpaper": True,
                "citation_route": "system_b",
                "routing_reason": "structured_cite",
                "source_name": "SCINeRF.pdf",
                "title": "Distributed Optimization and Statistical Learning via ADMM",
                "raw": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
                "answer_claim": "ADMM is prior optimization background.",
                "citation_context": "The current paper cites ADMM while discussing optimization background.",
                "citation_context_source": "source_markdown",
                "location_label": "Related Work",
                "card_takeaway": "这篇上游文献提供 ADMM 优化背景。",
                "system_b_trace_complete": True,
                "system_b_trace_score": 0.82,
                "system_b_trace_source": "source_markdown",
                "system_b_trace_flags": [],
            },
            {
                "num": 24,
                "anchor": "r24",
                "is_inpaper": True,
                "citation_route": "system_b",
                "routing_reason": "reference_index_fallback",
                "source_name": "SCI.pdf",
                "title": "Single-shot compressive spectral imaging",
                "raw": "Gehm et al. Single-shot compressive spectral imaging.",
                "answer_claim": "This is an upstream source.",
                "citation_context": "This is an upstream source.",
                "citation_context_source": "answer_context",
                "location_label": "SCI.pdf",
                "card_takeaway": "这篇文献提供单次压缩光谱成像背景。",
                "system_b_trace_complete": False,
                "system_b_trace_score": 0.32,
                "system_b_trace_source": "answer_context",
                "system_b_trace_flags": ["answer_context_only"],
            },
        ]
    )

    audit = summary["system_b_audit"]
    assert audit["system_b_total"] == 2
    assert audit["structured_cite_count"] == 1
    assert audit["reference_index_fallback_count"] == 1
    assert audit["source_markdown_count"] == 1
    assert audit["answer_context_only_count"] == 1
    assert audit["trace_complete_count"] == 1
    assert audit["needs_review_count"] == 1


def test_citation_shelf_item_quality_accepts_exportable_system_b_item():
    quality = citation_shelf_item_quality(
        {
            "num": 4,
            "anchor": "r4",
            "is_inpaper": True,
            "source_name": "SCINeRF.pdf",
            "title": "Distributed Optimization and Statistical Learning via ADMM",
            "authors": "Boyd et al.",
            "venue": "Foundations and Trends in Machine Learning",
            "year": "2011",
            "doi": "10.1561/2200000016",
            "raw": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
            "summary_line": "This upstream paper provides the ADMM optimization framework used by earlier SCI reconstruction methods.",
            "summary_quality": {"ok": True, "status": "grounded", "export_ready": True},
            "card_view": {
                "header": {
                    "title": "Distributed Optimization and Statistical Learning via ADMM",
                    "subtitle": "SCINeRF / Related Work",
                },
                "sections": [
                    {
                        "id": "takeaway",
                        "text": "This upstream paper provides the ADMM optimization framework used by earlier SCI reconstruction methods.",
                    }
                ],
                "summary": "This upstream paper provides the ADMM optimization framework used by earlier SCI reconstruction methods.",
                "quality": {"flags": []},
            },
        }
    )

    assert quality["ok"] is True
    assert quality["route"] == "system_b"
    assert quality["title"].startswith("Distributed Optimization")
    assert quality["metadata"]["export_ready"] is True
    assert quality["metadata"]["summary_export_ready"] is True
    assert quality["metadata"]["export_acceptance"]["export_ready"] is True


def test_citation_shelf_item_quality_requires_structured_export_fields_for_system_b():
    detail = {
        "num": 7,
        "anchor": "r7",
        "is_inpaper": True,
        "source_path": "refs/scigs.en.md",
        "source_name": "SCIGS references",
        "title": "Single-shot compressive spectral imaging",
        "raw": "Single-shot compressive spectral imaging.",
        "summary_line": "This upstream paper defines a single-shot compressive spectral imaging baseline used by later SCI systems.",
        "summary_quality": {"ok": True, "status": "grounded", "export_ready": True},
    }

    quality = citation_shelf_item_quality(detail)
    names = {item["name"] for item in quality["failures"]}

    assert quality["ok"] is False
    assert quality["metadata"]["export_ready"] is False
    assert quality["metadata"]["summary_export_ready"] is True
    assert quality["metadata"]["missing_export_fields"] == ["authors", "venue", "year", "doi"]
    assert {"shelf_export_missing_authors", "shelf_export_missing_venue", "shelf_export_missing_year", "shelf_export_missing_doi"}.issubset(names)

    summary = summarize_citation_shelf_quality([detail])
    assert summary["export_ready_count"] == 0
    assert summary["summary_export_ready_count"] == 1
    assert summary["doi_count"] == 0


def test_citation_shelf_item_quality_allows_partial_legacy_reference_without_review():
    detail = {
        "num": 15,
        "anchor": "r15",
        "is_inpaper": True,
        "source_path": "refs/qclfm.en.md",
        "source_name": "Quantum correlation light-field microscope.pdf",
        "title": "Pattern Analysis and Plenoptic Imaging",
        "venue": "Wave Optics",
        "year": "2013",
        "raw": "Pattern Analysis and Plenoptic Imaging: a mini review. Wave Optics, 2013.",
        "summary_line": "This upstream work is cited for light-field and plenoptic imaging background in microscopy.",
        "summary_quality": {"ok": True, "status": "grounded", "export_ready": True},
    }

    quality = citation_shelf_item_quality(detail)
    names = {item["name"] for item in quality["failures"]}
    summary = summarize_citation_shelf_quality([detail])

    assert quality["ok"] is True
    assert quality["metadata"]["metadata_ready"] is True
    assert quality["metadata"]["export_ready"] is False
    assert quality["metadata"]["review_needed"] is False
    assert "shelf_export_missing_authors" not in names
    assert "shelf_export_missing_doi" not in names
    assert summary["review_count"] == 0
    assert summary["metadata_ready_count"] == 1


def test_citation_shelf_item_quality_uses_raw_authors_and_softens_missing_doi():
    detail = {
        "num": 15,
        "anchor": "r15",
        "is_inpaper": True,
        "source_path": "refs/qclfm.en.md",
        "source_name": "Quantum correlation light-field microscope.pdf",
        "title": (
            "Pattern Analysis and Plenoptic Imaging: a mini review of light-field and "
            "ultra-slim light field microscopy that combines computation and light-field illumination"
        ),
        "venue": "Wave Optics, Oct",
        "year": "2013",
        "raw": (
            "Benjamin Sussman and Erik M. Gauger. Pattern Analysis and Plenoptic Imaging: "
            "a mini review of light-field and ultra-slim light field microscopy that combines "
            "computation and light-field illumination. Wave Optics, Oct 2013."
        ),
        "summary_line": "This upstream work is cited for light-field and plenoptic imaging background in microscopy.",
        "summary_quality": {"ok": True, "status": "grounded", "export_ready": True},
    }

    quality = citation_shelf_item_quality(detail)
    warning_names = {item["name"] for item in quality["warnings"]}

    assert quality["ok"] is True
    assert quality["metadata"]["has_author"] is True
    assert "shelf_missing_author_hint" not in warning_names
    assert "shelf_missing_doi" not in warning_names


def test_citation_shelf_item_quality_accepts_initial_surname_author():
    quality = citation_shelf_item_quality(
        {
            "num": 40,
            "anchor": "r40",
            "is_inpaper": True,
            "source_path": "refs/pidl.en.md",
            "source_name": "High-resolution single-photon imaging.pdf",
            "title": "Quantized Fourier ptychography with binary images from SPAD cameras",
            "authors": "X Yang",
            "venue": "Photon. Res.",
            "year": "2021",
            "doi": "10.1364/PRJ.427699",
            "raw": "Yang, X., Konda, P. C., Xu, S., Bian, L. & Horstmeyer, R. Quantized Fourier ptychography with binary images from SPAD cameras. Photon. Res. 9, 1958-1969 (2021).",
            "summary_line": "This upstream work demonstrates Fourier ptychography with binary SPAD camera images.",
            "summary_quality": {"ok": True, "status": "grounded", "export_ready": True},
        }
    )
    warning_names = {item["name"] for item in quality["warnings"]}

    assert quality["ok"] is True
    assert quality["metadata"]["export_ready"] is True
    assert "shelf_missing_author_hint" not in warning_names


def test_citation_shelf_item_quality_accepts_doi_export_without_authors():
    quality = citation_shelf_item_quality(
        {
            "num": 3,
            "anchor": "r3",
            "is_inpaper": True,
            "source_path": "refs/detector.en.md",
            "source_name": "Single photon detector review.pdf",
            "title": "Emerging single-photon detection technique for high-performance photodetector",
            "venue": "Frontiers of Physics",
            "year": "2024",
            "doi": "10.1007/s11467-024-1400-0",
            "raw": "Emerging single-photon detection technique for high-performance photodetector. Frontiers of Physics, 2024.",
            "summary_line": "This review builds detector background for single-photon imaging and photodetector choices.",
            "summary_quality": {"ok": True, "status": "grounded", "export_ready": True},
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is True
    assert quality["metadata"]["export_ready"] is True
    assert quality["metadata"]["review_needed"] is False
    assert "shelf_export_missing_authors" not in names


def test_citation_shelf_item_quality_accepts_single_word_journal_venue():
    quality = citation_shelf_item_quality(
        {
            "num": 124,
            "anchor": "ref-124",
            "is_inpaper": True,
            "citation_route": "system_b",
            "source_path": "refs/spd_review.en.md",
            "source_name": "Emerging single-photon detection technique for high-performance photodetector.pdf",
            "title": (
                "High-performance waveguide coupled Germanium-on-silicon single-photon avalanche diode "
                "with independently controllable absorption and multiplication"
            ),
            "authors": "H Wang",
            "venue": "Nanophotonics",
            "year": "2023",
            "volume": "12",
            "issue": "4",
            "pages": "705",
            "doi": "10.1515/nanoph-2022-0663",
            "raw": (
                "H. Wang, Y. Shi, Y. Zuo, Y. Yu, L. Lei, X. Zhang, and Z. Qian, "
                "High-performance waveguide coupled Germanium-on-silicon single-photon avalanche diode "
                "with independently controllable absorption and multiplication, "
                "Nanophotonics 12(4), 705 (2023)"
            ),
            "cite_fmt": (
                "H Wang. High-performance waveguide coupled Germanium-on-silicon single-photon avalanche diode "
                "with independently controllable absorption and multiplication. Nanophotonics, 12(4):705 (2023)."
            ),
            "summary_line": "This upstream detector paper provides a Ge-on-Si SPAD example relevant to single-photon imaging hardware choices.",
            "summary_quality": {"ok": True, "status": "grounded", "export_ready": True},
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is True
    assert quality["metadata"]["export_ready"] is True
    assert quality["metadata"]["export_acceptance"]["field_ready"]["venue"] is True
    assert "shelf_export_missing_venue" not in names


def test_citation_shelf_item_quality_does_not_show_candidate_review_when_identity_is_complete():
    quality = citation_shelf_item_quality(
        {
            "num": 4,
            "anchor": "r4",
            "is_inpaper": True,
            "source_name": "Distributed Optimization and Statistical Learning via ADMM.pdf",
            "title": "Distributed Optimization and Statistical Learning via ADMM",
            "authors": "Boyd et al.",
            "venue": "Foundations and Trends in Machine Learning",
            "year": "2011",
            "doi": "10.1561/2200000016",
            "raw": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
            "summary_line": "This upstream paper provides the ADMM optimization framework used by earlier SCI reconstruction methods.",
            "external_metadata_status": "candidate",
            "external_metadata_reason": "Low-similarity metrics candidate kept as a clue.",
            "external_doi": "10.1561/2200000016",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is True
    assert quality["metadata"]["metadata_ready"] is True
    assert quality["metadata"]["review_needed"] is False
    assert "shelf_untrusted_external_metadata_visible" not in names


def test_citation_shelf_item_quality_treats_system_a_raw_as_evidence_not_bibliography():
    quality = citation_shelf_item_quality(
        {
            "num": 1,
            "anchor": "a1",
            "source_name": "SCINeRF Neural Radiance Fields from a Snapshot Compressive Image.pdf",
            "source_path": "scinerf.en.md",
            "title": "3.2 Image Formation Model of Video SCI",
            "raw": "Due to the mask modulation, the image restoration problem is not ill-posed anymore.",
            "evidence_quote": "Due to the mask modulation, the image restoration problem is not ill-posed anymore.",
            "summary_line": "This evidence explains why the SCI forward model can recover virtual frames from a compressed image.",
        }
    )

    assert quality["ok"] is True
    assert quality["metadata"]["bibliographic"] is False
    assert quality["metadata"]["metadata_ready"] is True
    assert quality["metadata"]["review_needed"] is False


def test_citation_shelf_item_quality_accepts_system_a_doi_or_local_source_export() -> None:
    quality = citation_shelf_item_quality(
        {
            "num": 1,
            "anchor": "a1",
            "citation_route": "system_a",
            "is_inpaper": False,
            "source_name": "NatPhoton-2025-Structured detection.pdf",
            "source_path": "db/NatPhoton-2025-Structured detection/Structured detection.en.md",
            "bibliographic_title": "Structured detection for simultaneous super-resolution",
            "title": "Structured detection for simultaneous super-resolution",
            "heading_path": "Structured detection for simultaneous super-resolution / Abstract",
            "doi": "10.1038/example.structured",
            "summary_line": "The source describes structured detection as a way to retain optical sectioning and super-resolution.",
        }
    )

    assert quality["ok"] is True
    assert quality["metadata"]["export_ready"] is True
    assert quality["metadata"]["export_acceptance"]["export_mode"] == "system_a_doi"
    assert quality["metadata"]["has_author"] is False


def test_citation_shelf_item_quality_rejects_placeholder_summary_and_markdown():
    quality = citation_shelf_item_quality(
        {
            "num": 1,
            "anchor": "a1",
            "source_path": "demo.en.md",
            "title": "INTRODUCTION",
            "summary_line": "No summary available",
            "evidence_quote": "## Broken evidence has attrac...",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "shelf_template_phrase_visible" in names
    assert "shelf_summary_too_short" in names


def test_summarize_citation_shelf_quality_indexes_failures():
    summary = summarize_citation_shelf_quality(
        [
            {
                "num": 1,
                "anchor": "a1",
                "source_name": "Demo citation shelf paper.pdf",
                "source_path": "demo.en.md",
                "summary_line": "This shelf note explains why the cited evidence is useful for the answer.",
            },
            {"num": 2, "anchor": "a2", "summary_line": "short"},
        ]
    )

    assert summary["ok"] is False
    assert summary["count"] == 2
    assert summary["ok_count"] == 1
    assert any(item["index"] == 2 and item["name"] == "shelf_missing_source_identity" for item in summary["failures"])
