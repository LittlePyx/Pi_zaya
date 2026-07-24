from __future__ import annotations

from kb.citation_plan import build_citation_plan
from ui import refs_renderer


def test_system_a_binds_cassi_dual_disperser_aliases() -> None:
    binding = refs_renderer._assess_system_a_hit_binding(
        answer_claim="CASSI introduced a dual-disperser architecture with a coded aperture.",
        hit={"text": "The design uses two dispersive elements and a binary-valued aperture code."},
        meta={},
        heading="Abstract",
        evidence_quote="The design uses two dispersive elements and a binary-valued aperture code.",
        source_name="Single-shot compressive spectral imaging with a dual-disperser architecture",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "cassi" in binding["overlap_terms"]


def test_system_a_numeric_metric_binding_explains_relevance_without_repeating_table():
    evidence = "Table 2 shows SSIM results: VST+bicubic = 0.50, U-net = 0.60, Ours = 0.76."
    binding = refs_renderer._assess_system_a_hit_binding(
        answer_claim=(
            "该方法的 SSIM 达到 0.76，优于 VST+bicubic 的 0.50 和 U-net 的 0.60。"
        ),
        hit={"text": evidence},
        meta={},
        heading="Results / Noise model evaluation",
        evidence_quote=evidence,
        source_name="High-resolution single-photon imaging with physics-informed deep learning",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "直接量化" in binding["reason"]
    assert "0.76" not in binding["reason"]


def test_system_a_detector_table_binding_explains_hardware_boundary():
    evidence = (
        "Detector type: InGaAs/InAlAs-SPAD. Working parameter: wavelength = "
        "1310 nm; Performance = 61.2% DE at 200 K."
    )
    binding = refs_renderer._assess_system_a_hit_binding(
        answer_claim="该 InGaAs/InAlAs-SPAD 在 1310 nm、200 K 下达到 61.2% 探测效率。",
        hit={"text": evidence},
        meta={},
        heading="2.3 Superconducting",
        evidence_quote=evidence,
        source_name="Emerging single-photon detection technique",
    )

    assert binding["status"] == "grounded"
    assert "硬件性能边界" in binding["reason"]


def test_system_a_binds_chinese_sph_claim_to_english_mechanism() -> None:
    evidence = (
        "Instead of actively performing phase shifting, a beat frequency is introduced "
        "between the signal and reference beams, realizing phase stepping naturally in "
        "time through heterodyne holography."
    )
    binding = refs_renderer._assess_system_a_hit_binding(
        answer_claim="外差拍频让相移在时间上自然展开，从而提高采集吞吐量。",
        hit={"text": evidence},
        meta={},
        heading="Introduction",
        evidence_quote=evidence,
        source_name="Imaging biological tissue with high-throughput SPH",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "beat-frequency phase stepping" in binding["overlap_terms"]


def test_system_a_binds_bilingual_distilled_sensing_claim() -> None:
    evidence = (
        "The procedure is based on the principle of distilled sensing and uses sparse "
        "sensing matrices to identify irrelevant signal components."
    )
    binding = refs_renderer._assess_system_a_hit_binding(
        answer_claim="这一思想源于蒸馏感知（distilled sensing）原理。",
        hit={"text": evidence},
        meta={},
        heading="Abstract",
        evidence_quote=evidence,
        source_name="Sequentially Designed Compressed Sensing",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "distilled sensing" in binding["overlap_terms"]


def test_system_a_binds_chinese_s2ism_tradeoff_to_english_abstract() -> None:
    evidence = (
        "Current image scanning microscopy approaches do not provide optical sectioning "
        "and fail with thick samples, introducing a trade-off between optical sectioning "
        "and signal-to-noise ratio while spatial resolution remains constrained."
    )
    binding = refs_renderer._assess_system_a_hit_binding(
        answer_claim="s²ISM 同时改善空间分辨率、信噪比和光学切片能力。",
        hit={"text": evidence},
        meta={},
        heading="Abstract",
        evidence_quote=evidence,
        source_name="Structured detection for s2ISM",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "s2ism tradeoff" in binding["overlap_terms"]


def test_system_a_binds_chinese_spad_mechanism_to_english_principle() -> None:
    evidence = (
        "A single photon avalanche diode (SPAD) operates in Geiger mode above its reverse "
        "bias breakdown voltage and requires a quenching circuit after avalanche current."
    )
    binding = refs_renderer._assess_system_a_hit_binding(
        answer_claim="SPAD 在盖革模式下高于击穿电压工作，并由淬灭电路终止雪崩。",
        hit={"text": evidence},
        meta={},
        heading="Principle of single photon detection avalanche diode",
        evidence_quote=evidence,
        source_name="Single-photon detector review",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "spad geiger quenching" in binding["overlap_terms"]


def test_system_a_binds_piln_to_ilnet_and_sampling_rate_aliases() -> None:
    binding = refs_renderer._assess_system_a_hit_binding(
        answer_claim="PILN targets single-pixel imaging at a low sampling rate.",
        hit={
            "text": (
                "The part-based image-loop network (ILNet) reconstructs single-pixel images "
                "while reducing sample rates."
            )
        },
        meta={},
        heading="Abstract",
        evidence_quote=(
            "The part-based image-loop network (ILNet) reconstructs single-pixel images "
            "while reducing sample rates."
        ),
        source_name="Part-based image-loop network for single-pixel imaging",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert {"piln", "sampling ratio"}.issubset(set(binding["overlap_terms"]))


def test_system_a_does_not_ground_unrelated_cutoff_frequency_phrase() -> None:
    binding = refs_renderer._assess_system_a_hit_binding(
        answer_claim="The waveguide cut-off frequency sets the supported optical mode.",
        hit={"text": "The electrical low-pass filter has a measured cut-off frequency."},
        meta={},
        heading="Electronics",
        evidence_quote="The electrical low-pass filter has a measured cut-off frequency.",
        source_name="Detector readout electronics",
    )

    assert binding["status"] == "candidate"
    assert binding["suppress_link"] is True


def test_system_a_still_grounds_explicit_waveguide_evidence() -> None:
    binding = refs_renderer._assess_system_a_hit_binding(
        answer_claim="The waveguide geometry sets its supported optical modes.",
        hit={"text": "The waveguide geometry determines which optical modes are supported."},
        meta={},
        heading="Waveguide design",
        evidence_quote="The waveguide geometry determines which optical modes are supported.",
        source_name="Integrated optical waveguide design",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "waveguide" in binding["overlap_terms"]


def test_numeric_citation_is_hidden_when_local_doi_conflicts(monkeypatch):
    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 24:
            return None
        return {
            "source_path": "doc.en.md",
            "source_name": "doc.pdf",
            "ref_num": 24,
            "ref": {
                "authors": "Townsend P, Foster J",
                "year": "2003",
                "doi": "10.1000/wrong",
                "title": "Wrong ref",
                "raw": "[24] Townsend P, Foster J. Wrong ref. 2003. doi:10.1000/wrong",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = "This follows DOI 10.1364/OE.15.014013 [24]."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[24](#" not in out
    assert "[24]" not in out
    assert details == []


def test_numeric_citation_uses_doi_to_resolve_multi_source_ambiguity(monkeypatch):
    def fake_resolve(_index_data, source_path, ref_num, *, source_sha1=""):
        del _index_data, source_sha1
        if int(ref_num) != 24:
            return None
        if str(source_path).endswith("a.en.md"):
            return {
                "source_path": "a.en.md",
                "source_name": "a.pdf",
                "ref_num": 24,
                "ref": {
                    "authors": "Wrong A",
                    "year": "2010",
                    "doi": "10.1000/wrong-a",
                    "raw": "[24] Wrong A. 2010. doi:10.1000/wrong-a",
                },
            }
        if str(source_path).endswith("b.en.md"):
            return {
                "source_path": "b.en.md",
                "source_name": "b.pdf",
                "ref_num": 24,
                "ref": {
                    "authors": "Gehm M, Brady D",
                    "year": "2007",
                    "doi": "10.1364/OE.15.014013",
                    "raw": "[24] Gehm M, Brady D. 2007. doi:10.1364/OE.15.014013",
                },
            }
        return None

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda sp: "a.pdf" if sp.endswith("a.en.md") else "b.pdf")
    def fake_enrich(detail, *, source_path, ref_num, answer_context="", **_kwargs):
        del source_path, ref_num
        detail["answer_claim"] = answer_context
        detail["citation_context"] = "The current paper cites Gehm and Brady when tracing single-shot compressive spectral imaging."
        detail["citation_context_source"] = "source_markdown"
        detail["evidence_quote"] = detail["citation_context"]
        detail["location_label"] = "Related Work / Snapshot compressive imaging"
        return detail

    monkeypatch.setattr(refs_renderer, "enrich_inpaper_detail_context", fake_enrich)

    md = "This follows DOI 10.1364/OE.15.014013 [24]."
    hits = [
        {"meta": {"source_path": "a.en.md", "source_sha1": "aaa"}},
        {"meta": {"source_path": "b.en.md", "source_sha1": "bbb"}},
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[24](#" in out
    assert len(details) == 1
    assert details[0]["source_path"] == "b.en.md"


def test_numeric_citation_without_identity_signal_stays_clickable(monkeypatch):
    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 116:
            return None
        return {
            "source_path": "doc.en.md",
            "source_name": "doc.pdf",
            "ref_num": 116,
            "ref": {
                "authors": "Wang X, Li Y",
                "year": "2020",
                "title": "A paper",
                "raw": "[116] Wang X, Li Y. A paper. IEEE TCI, 2020.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")
    def fake_enrich(detail, *, source_path, ref_num, answer_context="", **_kwargs):
        del source_path, ref_num
        detail["answer_claim"] = answer_context
        detail["citation_context"] = "The current paper cites Wang and Li when discussing DenseNet reconstruction."
        detail["citation_context_source"] = "source_markdown"
        detail["evidence_quote"] = detail["citation_context"]
        detail["location_label"] = "Related Work / Reconstruction"
        return detail

    monkeypatch.setattr(refs_renderer, "enrich_inpaper_detail_context", fake_enrich)

    md = "Wang uses DenseNet for reconstruction [116]."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[116](#" in out
    assert len(details) == 1
    assert details[0]["system_b_trace_complete"] is True


def test_numeric_reference_index_fallback_stays_plain_when_citation_context_is_missing(monkeypatch):
    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 116:
            return None
        return {
            "source_path": "doc.en.md",
            "source_name": "doc.pdf",
            "ref_num": 116,
            "ref": {
                "authors": "Wang X, Li Y",
                "year": "2020",
                "title": "A paper",
                "raw": "[116] Wang X, Li Y. A paper. IEEE TCI, 2020.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = "Wang uses DenseNet for reconstruction [116]."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[116](#" not in out
    assert "[116]" in out
    assert details == []


def test_structured_cite_routes_to_system_b_for_upstream_origin_context(monkeypatch):
    source_path = "scinerf.en.md"

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 1:
            return None
        return {
            "source_path": source_path,
            "source_name": "SCINeRF.pdf",
            "ref_num": 1,
            "ref": {
                "authors": "Boyd S",
                "year": "2011",
                "title": "Distributed Optimization and Statistical Learning via ADMM",
                "raw": "[1] Boyd S. Distributed Optimization and Statistical Learning via ADMM. 2011.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "SCINeRF.pdf")

    sid = refs_renderer._source_cite_id(source_path)
    md = f"ADMM was not invented by this paper; it comes from prior optimization work [[CITE:{sid}:1]]."
    hits = [
        {
            "text": "SCINeRF uses ADMM for optimization in its reconstruction pipeline.",
            "meta": {"source_path": source_path, "source_sha1": "abc"},
        }
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[1](#kb-cite-" in out
    assert len(details) == 1
    detail = details[0]
    assert detail["is_inpaper"] is True
    assert detail["citation_route"] == "system_b"
    assert "ADMM" in " ".join([str(detail.get("title") or ""), str(detail.get("raw") or "")])
    assert detail["routing_reason"] == "structured_cite"


def test_same_number_system_a_and_system_b_keep_distinct_cards(monkeypatch):
    source_path = "scinerf.en.md"

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 1:
            return None
        return {
            "source_path": source_path,
            "source_name": "SCINeRF.pdf",
            "ref_num": 1,
            "ref": {
                "authors": "Boyd S",
                "year": "2011",
                "title": "Distributed Optimization and Statistical Learning via ADMM",
                "raw": "[1] Boyd S. Distributed Optimization and Statistical Learning via ADMM. 2011.",
            },
        }

    def fake_enrich(detail, *, source_path, ref_num, answer_context="", **_kwargs):
        del source_path, ref_num
        detail["answer_claim"] = answer_context
        detail["citation_context"] = "Existing methods employ ADMM [1]."
        detail["citation_context_source"] = "source_markdown"
        detail["evidence_quote"] = detail["citation_context"]
        detail["location_label"] = "Related Work"
        return detail

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "SCINeRF.pdf")
    monkeypatch.setattr(refs_renderer, "enrich_inpaper_detail_context", fake_enrich)

    sid = refs_renderer._source_cite_id(source_path)
    md = f"The current paper says existing methods employ ADMM [1].\nADMM is prior work [[CITE:{sid}:1]]."
    hits = [
        {
            "text": "The current paper explains that existing methods employ ADMM.",
            "meta": {"source_path": source_path, "source_sha1": "abc", "heading_path": "Related Work"},
        }
    ]
    plan = {"budget": {"system_a": 1, "system_b": 1}}

    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md,
        hits,
        anchor_ns="t",
        citation_plan=plan,
    )

    assert out.count("[1](#kb-cite-") == 2
    assert {detail["citation_route"] for detail in details} == {"system_a", "system_b"}
    assert len({detail["anchor"] for detail in details}) == 2


def test_numeric_router_keeps_good_system_a_for_generic_reference_word(monkeypatch):
    source_path = "single_pixel_review.en.md"

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 1:
            return None
        return {
            "source_path": source_path,
            "source_name": "Review.pdf",
            "ref_num": 1,
            "ref": {
                "authors": "Unrelated A",
                "year": "2001",
                "title": "An unrelated bibliography item",
                "raw": "[1] Unrelated A. An unrelated bibliography item. 2001.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "Review.pdf")

    md = "Use this single-pixel imaging review as a reference [1]; it explains reconstruction quality and sampling rate."
    hits = [
        {
            "text": "This review explains single-pixel imaging reconstruction quality, sampling rate, and deep learning methods.",
            "meta": {
                "source_path": source_path,
                "source_sha1": "abc",
                "heading_path": "Abstract",
                "evidence_quote": "This review explains single-pixel imaging reconstruction quality, sampling rate, and deep learning methods.",
            },
        }
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[1](#kb-cite-" in out
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["card_kind"] == "answer_evidence"


def test_system_a_keeps_contrastive_source_identity_citation(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        refs_renderer,
        "_display_source_name",
        lambda _sp: "Nature-2025-Electrically driven lasing from a dual-cavity perovskite device.pdf",
    )

    md = (
        "The Nature 2025 perovskite laser paper is not a single-pixel imaging paper; "
        "its core contribution is an electrically driven perovskite laser [1]."
    )
    hits = [
        {
            "text": "The device is constructed by integrating a low-threshold single-crystal perovskite microcavity with a high-power microcavity PeLED.",
            "meta": {
                "source_path": "perovskite.en.md",
                "source_sha1": "abc",
                "heading_path": "Conclusion",
            },
            "ui_meta": {
                "primary_evidence": {
                    "heading_path": "Conclusion",
                    "snippet": "The device is constructed by integrating a low-threshold single-crystal perovskite microcavity with a high-power microcavity PeLED.",
                    "block_id": "b1",
                    "anchor_id": "p1",
                    "anchor_kind": "paragraph",
                }
            },
        }
    ]

    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[1](#kb-cite-" in out
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["binding_status"] == "grounded"


def test_system_a_still_suppresses_unmatched_user_topic(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        refs_renderer,
        "_display_source_name",
        lambda _sp: "Nature-2025-Electrically driven lasing from a dual-cavity perovskite device.pdf",
    )

    md = "Single-pixel imaging relies on DMD measurement patterns and compressed sensing [1]."
    hits = [
        {
            "text": "The device is constructed by integrating a low-threshold single-crystal perovskite microcavity with a high-power microcavity PeLED.",
            "meta": {
                "source_path": "perovskite.en.md",
                "source_sha1": "abc",
                "heading_path": "Conclusion",
            },
        }
    ]

    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[1](#kb-cite-" not in out
    assert details == []


def test_system_a_treats_compressive_sensing_as_compressed_sensing_alias(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        refs_renderer,
        "_display_source_name",
        lambda _sp: "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
    )

    md = "早期 SCI 工作用压缩感知重建光谱数据立方体 [1]。"
    hits = [
        {
            "text": "The manuscript describes a new single-shot spectral imager based on compressive sensing ideas.",
            "meta": {
                "source_path": "cassi.en.md",
                "source_sha1": "abc",
                "heading_path": "5. Conclusions",
                "evidence_quote": "The manuscript describes a new single-shot spectral imager based on compressive sensing ideas.",
            },
        }
    ]

    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[1](#kb-cite-" in out
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"


def test_system_a_links_single_photon_detector_review_terms(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        refs_renderer,
        "_display_source_name",
        lambda _sp: "Frontiers of Physics-2024-Emerging single-photon detection technique for high-performance photodetector.pdf",
    )

    md = "探测器综述能帮你建立对 SPAD 和单光子探测器物理特性的系统认知 [1]。"
    hits = [
        {
            "text": "Single-photon detections represent a highly sensitive light detection technique capable of detecting individual photons.",
            "meta": {
                "source_path": "spd-review.en.md",
                "source_sha1": "abc",
                "heading_path": "3 Single photon detection parameter",
                "evidence_quote": "Single-photon detections represent a highly sensitive light detection technique capable of detecting individual photons.",
            },
        }
    ]

    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[1](#kb-cite-" in out
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"


def test_citation_plan_does_not_steal_bare_numeric_system_a_citation(monkeypatch):
    source_path = "scinerf.en.md"

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 1:
            return None
        return {
            "source_path": source_path,
            "source_name": "SCINeRF.pdf",
            "ref_num": 1,
            "ref": {
                "authors": "Boyd S",
                "year": "2011",
                "title": "Distributed Optimization and Statistical Learning via ADMM",
                "raw": "[1] Boyd S. Distributed Optimization and Statistical Learning via ADMM. 2011.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "SCINeRF.pdf")

    md = "ADMM is the optimization machinery discussed here [1]."
    hits = [
        {
            "text": "SCINeRF uses ADMM for optimization in its reconstruction pipeline.",
            "meta": {"source_path": source_path, "source_sha1": "abc"},
        }
    ]
    plan = {
        "intent": "beginner_overview",
        "budget": {"system_a": 2, "system_b": 1},
        "system_b_enabled": True,
        "slots": [{"preferred_system": "system_b", "candidate_refs": [1]}],
    }
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md,
        hits,
        anchor_ns="t",
        citation_plan=plan,
    )

    assert "[1](#kb-cite-" in out
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["routing_reason"] == "retrieval_hit"


def test_citation_router_applies_per_paragraph_budget_for_system_a(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = (
        "Structured detection improves optical sectioning [1], dynamic supersampling "
        "changes sampling [2], and light field microscopy captures 3D context [3]."
    )
    hits = [
        {
            "text": "Structured detection improves optical sectioning.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "A",
                "evidence_quote": "Structured detection improves optical sectioning.",
                "primary_block_id": "a",
            },
        },
        {
            "text": "Dynamic supersampling changes the sampling pattern.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "B",
                "evidence_quote": "Dynamic supersampling changes the sampling pattern.",
                "primary_block_id": "b",
            },
        },
        {
            "text": "Light field microscopy captures 3D context.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "C",
                "evidence_quote": "Light field microscopy captures 3D context.",
                "primary_block_id": "c",
            },
        },
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert out.count("#kb-cite-") == 2
    assert "[3](#kb-cite-" not in out
    assert len(details) == 2
    assert all(item["citation_route"] == "system_a" for item in details)


def test_multi_source_coverage_keeps_three_cards_without_three_per_paragraph(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda source_path: source_path)

    hits = [
        {
            "text": "Structured detection improves optical sectioning.",
            "meta": {
                "source_path": "structured.en.md",
                "heading_path": "Abstract",
                "evidence_quote": "Structured detection improves optical sectioning.",
                "primary_block_id": "structured",
            },
        },
        {
            "text": "Interferometric detection measures weak scattering signals.",
            "meta": {
                "source_path": "interferometric.en.md",
                "heading_path": "Abstract",
                "evidence_quote": "Interferometric detection measures weak scattering signals.",
                "primary_block_id": "interferometric",
            },
        },
        {
            "text": "Light-field microscopy captures three-dimensional context.",
            "meta": {
                "source_path": "light-field.en.md",
                "heading_path": "Abstract",
                "evidence_quote": "Light-field microscopy captures three-dimensional context.",
                "primary_block_id": "light-field",
            },
        },
    ]
    plan = build_citation_plan(
        prompt="structured detection、interferometric、light-field 分别解决什么问题？",
        prompt_family="method",
        answer_hits=hits,
    )

    separated = (
        "Structured detection improves optical sectioning [1].\n\n"
        "Interferometric detection measures weak scattering signals [2].\n\n"
        "Light-field microscopy captures three-dimensional context [3]."
    )
    separated_out, separated_details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        separated,
        hits,
        anchor_ns="multi-source-separated",
        citation_plan=plan,
    )

    assert plan["budget"]["system_a"] == 3
    assert plan["per_paragraph_budget"]["system_a"] == 2
    assert separated_out.count("#kb-cite-") == 3
    assert len(separated_details) == 3

    crowded = (
        "Structured detection improves optical sectioning [1], interferometric detection measures "
        "weak scattering signals [2], and light-field microscopy captures three-dimensional context [3]."
    )
    crowded_out, crowded_details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        crowded,
        hits,
        anchor_ns="multi-source-crowded",
        citation_plan=plan,
    )

    assert crowded_out.count("#kb-cite-") == 2
    assert len(crowded_details) == 2


def test_citation_router_reads_system_a_budget_from_citation_plan(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = "Structured detection improves sectioning [1], and dynamic supersampling changes sampling [2]."
    hits = [
        {
            "text": "Structured detection improves optical sectioning.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "A",
                "evidence_quote": "Structured detection improves optical sectioning.",
                "primary_block_id": "a",
            },
        },
        {
            "text": "Dynamic supersampling changes the sampling pattern.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "B",
                "evidence_quote": "Dynamic supersampling changes the sampling pattern.",
                "primary_block_id": "b",
            },
        },
    ]
    plan = {"intent": "origin_lookup", "budget": {"system_a": 1, "system_b": 0}, "slots": []}
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md,
        hits,
        anchor_ns="t",
        citation_plan=plan,
    )

    assert out.count("#kb-cite-") == 1
    assert "[2](#kb-cite-" not in out
    assert len(details) == 1


def test_structured_citation_is_hidden_when_context_doi_conflicts(monkeypatch):
    source_path = "doc.en.md"
    sid = refs_renderer._source_cite_id(source_path)

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 1:
            return None
        return {
            "source_path": source_path,
            "source_name": "doc.pdf",
            "ref_num": 1,
            "ref": {
                "authors": "Wrong A",
                "year": "2020",
                "doi": "10.1000/wrong",
                "title": "Wrong Reference",
                "raw": "[1] Wrong A. Wrong Reference. 2020. doi:10.1000/wrong",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = f"This follows DOI 10.1364/OE.15.014013 [[CITE:{sid}:1]]."
    hits = [{"meta": {"source_path": source_path, "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "#kb-cite-" not in out
    assert "[1]" not in out
    assert "Wrong Reference" not in out
    assert details == []


def test_unresolved_structured_citation_does_not_fall_back_to_hit_number(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = "Bad structured cite [[CITE:badbad:1]] and normal compressive imaging evidence [1]."
    hits = [
        {
            "text": "Compressive imaging evidence supports the reconstruction claim.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "Methods",
                "evidence_quote": "Compressive imaging evidence supports the reconstruction claim.",
            },
        }
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[[CITE:" not in out
    assert "Bad structured cite  and normal compressive imaging evidence [1](#" in out
    assert len(details) == 1
    assert details[0]["is_inpaper"] is False


def test_structured_citation_detail_points_to_context_matched_reference(monkeypatch):
    source_path = "doc.en.md"
    sid = refs_renderer._source_cite_id(source_path)

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 24:
            return None
        return {
            "source_path": source_path,
            "source_name": "doc.pdf",
            "ref_num": 24,
            "ref": {
                "authors": "Gehm M, Brady D",
                "year": "2007",
                "doi": "10.1364/OE.15.014013",
                "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                "raw": (
                    "[24] Gehm M, Brady D. Single-shot compressive spectral imaging with "
                    "a dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013"
                ),
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = f"This follows DOI 10.1364/OE.15.014013 [[CITE:{sid}:24]]."
    hits = [{"meta": {"source_path": source_path, "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[24](#kb-cite-" in out
    assert len(details) == 1
    detail = details[0]
    assert detail["num"] == 24
    assert detail["source_path"] == source_path
    assert detail["doi"] == "10.1364/OE.15.014013"
    assert "dual-disperser architecture" in detail["title"]
    assert "Wrong Reference" not in str(detail)


def test_structured_system_b_detail_carries_answer_context_and_role(monkeypatch):
    source_path = "doc.en.md"
    sid = refs_renderer._source_cite_id(source_path)

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 4:
            return None
        return {
            "source_path": source_path,
            "source_name": "doc.pdf",
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
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = f"ADMM is prior optimization machinery; open ADMM [[CITE:{sid}:4]] to follow the paper's citation trail."
    hits = [{"meta": {"source_path": source_path, "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[4](#kb-cite-" in out
    assert len(details) == 1
    detail = details[0]
    assert detail["is_inpaper"] is True
    assert "prior optimization machinery" in detail["answer_claim"]
    assert "prior optimization machinery" in detail["citation_context"]
    assert detail["citation_context_source"] == "answer_context"
    assert "prior optimization machinery" in detail["evidence_quote"]
    assert "prior work" in detail["upstream_work_role"].lower()
    assert "origin" in detail["user_question_relation"].lower()
    assert detail["support_relation"] == detail["user_question_relation"]
    assert detail["card_kind"] == "upstream_reference"
    assert detail["card_title"] == "Distributed Optimization and Statistical Learning via ADMM"
    assert detail["card_evidence_label"] == "回答里的线索"
    assert "ADMM 优化框架背景" in detail["card_takeaway"]
    assert "answer_context_only" in detail["card_quality_flags"]
    assert "完整引用语境" in detail["card_warning"]
    assert detail["system_b_trace_complete"] is False
    assert "answer_context_only" in detail["system_b_trace_flags"]
    assert detail["system_b_trace_steps"] == ["答案句", "引用语境待核对", "上游文献"]
