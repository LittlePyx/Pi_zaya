import kb.paper_guide_direct_answer_runtime as direct_answer_runtime
from kb.paper_guide_direct_answer_runtime import _build_paper_guide_direct_answer_override


def test_spad_noise_model_preflight_returns_grounded_localized_answer(tmp_path):
    source = tmp_path / "pidl.en.md"
    source.write_text(
        "<!-- kb_page: 2 -->\n"
        "The underlying limitation originates from the employed single-source Poisson noise model, "
        "which deviates from complex real SPAD noise containing a variety of different-model sources "
        "such as crosstalk, dark count rate, and so on. However, the technique also falls into the "
        "single-source based Poisson statistics that lead to degraded imaging quality without considering "
        "such multiple noise sources (as validated in the following experiments shown in Fig. 2b).\n\n"
        "**Figure 1.** Illustration. **a** The multi-source physical noise model of SPAD arrays, which "
        "consists of shot noise from photon incidence, fixed-pattern noise from SPAD array's photon "
        "absorption, dark count rate, afterpulsing and crosstalk noise from blind electron avalanche, "
        "and deadtime noise from the quenching circuit.\n",
        encoding="utf-8",
    )

    out = direct_answer_runtime._build_paper_guide_exact_answer_preflight(
        paper_guide_mode=True,
        prompt_family="method",
        prompt_for_user="为什么只用泊松噪声训练 SPAD 超分辨模型不够？模型纳入了哪些噪声？",
        source_path=str(source),
        db_dir=tmp_path,
    )

    assert "只用泊松噪声不够" in out["answer"]
    assert "串扰噪声" in out["answer"]
    assert "死时间噪声" in out["answer"]
    assert out["answer"].count("[1]") >= 6
    assert out["support_resolution"][0]["page_start"] == 2
    assert "multi-source physical noise model" in out["support_resolution"][0]["locate_anchor"]
    evidence = out["support_resolution"][0]["evidence_quote"]
    assert "single-source Poisson noise model" in evidence
    assert "multi-source physical noise model" in evidence
    assert "Mora-Mart" not in evidence
    assert "Figure 1." not in evidence
    assert "泊松噪声不够" in out["support_resolution"][0]["why_line"]
    assert "暗计数率" in out["support_resolution"][0]["guide_line"]
from kb.paper_guide.skills import PaperGuideSkillResult


def test_exact_answer_preflight_preserves_support_resolution(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_run_exact_support_skill",
        lambda **_kwargs: PaperGuideSkillResult(
            answer_text="ADMM is prior work [[CITE:s1234abcd:4]].",
            support_resolution=[
                {
                    "source_path": "paper.md",
                    "heading_path": "2. Related Work",
                    "block_id": "blk_admm",
                    "anchor_id": "p_admm",
                    "locate_anchor": "Most existing methods employ ADMM [4].",
                    "ref_nums": [4],
                }
            ],
        ),
    )

    out = direct_answer_runtime._build_paper_guide_exact_answer_preflight(
        paper_guide_mode=True,
        prompt_family="citation_lookup",
        prompt_for_user="Which reference is cited for ADMM, and where exactly?",
        source_path="paper.md",
        db_dir="db",
    )

    assert out["answer"].endswith("[[CITE:s1234abcd:4]].")
    assert out["support_resolution"][0]["block_id"] == "blk_admm"
    assert out["support_resolution"][0]["anchor_id"] == "p_admm"
    assert out["support_resolution"][0]["ref_nums"] == [4]


def test_build_paper_guide_direct_answer_override_prefers_abstract_path():
    calls = {}

    def _build_direct_abstract_answer(**kwargs):
        calls["abstract"] = kwargs
        return "ABSTRACT"

    def _build_direct_citation_lookup_answer(**kwargs):
        calls["citation"] = kwargs
        return "CITATION"

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="abstract",
        prompt_for_user="把摘要原文给出并翻译",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="",
        db_dir="db",
        llm="llm-client",
        build_direct_abstract_answer=_build_direct_abstract_answer,
        build_direct_citation_lookup_answer=_build_direct_citation_lookup_answer,
    )

    assert out == "ABSTRACT"
    assert calls["abstract"]["source_path"] == "direct.md"
    assert calls["abstract"]["llm"] == "llm-client"
    assert "citation" not in calls


def test_build_paper_guide_direct_answer_override_detects_natural_zh_reading_prompt(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_resolve_doc_map_records_from_source",
        lambda source_path, **_kwargs: [
            {
                "source_path": source_path,
                "heading_path": "Demo Paper",
                "locate_anchor": "Yunhao Li Xiaodong Wang Zhejiang University {author}@example.edu.cn",
            },
            {
                "source_path": source_path,
                "heading_path": "Abstract",
                "locate_anchor": "This paper connects snapshot compressive imaging with a NeRF representation.",
            },
            {
                "source_path": source_path,
                "heading_path": "3. Method",
                "locate_anchor": "The method optimizes the 3D representation from the compressed observation.",
            },
        ],
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="overview",
        prompt_for_user="我没看懂它怎么从一张压缩图变成可以转视角的3D场景，我应该先读哪几段？",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[{"meta": {"source_path": "bound.md"}}],
        special_focus_block="",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    assert out.startswith("可以先按这几处读：")
    assert "Yunhao Li" not in out
    assert "1. Abstract" in out
    assert "2. 3. Method" in out
    assert "snapshot compressive imaging" in out


def test_build_paper_guide_direct_answer_override_uses_targeted_box_hit(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_paper_guide_targeted_source_block_hits",
        lambda **_kwargs: [
            {
                "text": "Box 1",
                "meta": {
                    "heading_path": "Box 1",
                    "kind": "heading",
                },
            },
            {
                "text": "In the transform domain, the image can be reconstructed even if M < N.",
                "meta": {
                    "heading_path": "Box 1",
                    "kind": "paragraph",
                },
            },
        ],
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="box_only",
        prompt_for_user="From Box 1 only, what condition do the authors give for reconstructing the image in the transform domain?",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    assert "Relevant passage from Box 1" in out
    assert "M < N" in out


def test_build_paper_guide_direct_answer_override_uses_exact_citation_lookup_without_template(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        direct_answer_runtime,
        "_resolve_exact_citation_lookup_support_from_source",
        lambda source_path, **_kwargs: {
            "source_path": source_path,
            "heading_path": "Related Work",
            "locate_anchor": "The method follows ADMM [4].",
            "ref_nums": [4],
        },
    )

    def _build_direct_abstract_answer(**kwargs):
        calls["abstract"] = kwargs
        return "ABSTRACT"

    def _build_direct_citation_lookup_answer(**kwargs):
        calls["citation"] = kwargs
        return "CITATION"

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="citation_lookup",
        prompt_for_user="Which references are cited for RVT, and where is that stated exactly?",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[{"meta": {"source_path": "focus.md"}}],
        special_focus_block="focus",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=_build_direct_abstract_answer,
        build_direct_citation_lookup_answer=_build_direct_citation_lookup_answer,
    )

    assert "Use [[CITE:s6ce92c61:4]] as the cited source for this passage." in out
    assert "Source location: Related Work." in out
    assert "abstract" not in calls
    assert "citation" not in calls


def test_build_paper_guide_direct_answer_override_names_exact_reference_title(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_resolve_exact_citation_lookup_support_from_source",
        lambda source_path, **_kwargs: {
            "source_path": source_path,
            "heading_path": "Introduction",
            "locate_anchor": "Compressive sensing [7,9,14] has been adopted to reduce measurements.",
            "ref_nums": [9],
            "reference_title": "A single-pixel terahertz imaging system based on compressed sensing",
        },
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="citation_lookup",
        prompt_for_user=(
            "本文参考文献中的 “A single-pixel terahertz imaging system based on compressed sensing” "
            "是第几条？正文在哪里引用它？"
        ),
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[{"meta": {"source_path": "focus.md"}}],
        special_focus_block="focus",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    assert "本文参考文献 [[CITE:s6ce92c61:9]]" in out
    assert "A single-pixel terahertz imaging system based on compressed sensing" in out


def test_build_paper_guide_direct_answer_override_returns_deterministic_not_found(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_resolve_exact_citation_lookup_support_from_source",
        lambda source_path, **_kwargs: {
            "not_found": True,
            "requested_title": "First-Photon Imaging",
            "source_path": source_path,
        },
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="citation_lookup",
        prompt_for_user="本文参考文献里是否有 “First-Photon Imaging”？没有就明确说没有。",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[{"meta": {"source_path": "focus.md"}}],
        special_focus_block="focus",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    assert out == "本文参考文献中没有《First-Photon Imaging》。"


def test_build_paper_guide_direct_answer_override_skips_citation_lookup_when_targeted():
    calls = {}

    def _build_direct_abstract_answer(**kwargs):
        calls["abstract"] = kwargs
        return "ABSTRACT"

    def _build_direct_citation_lookup_answer(**kwargs):
        calls["citation"] = kwargs
        return "CITATION"

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="citation_lookup",
        prompt_for_user="From Figure 2 only, which references are cited for RVT?",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="focus",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=_build_direct_abstract_answer,
        build_direct_citation_lookup_answer=_build_direct_citation_lookup_answer,
    )

    assert out == ""
    assert calls == {}


def test_build_paper_guide_direct_answer_override_supports_exact_method_prompt_in_reproduce_family(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_resolve_exact_method_support_from_source",
        lambda source_path, **_kwargs: {
            "heading_path": "Experimental setup",
            "locate_anchor": "Thus, the beat frequency of these two beams is 62,500 Hz. The data acquisition card uses a sampling rate of 1.25 Ms/s.",
            "source_path": source_path,
        },
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="reproduce",
        prompt_for_user="In the Experimental setup section, what beat frequency do they use, and what sampling rate does the data acquisition card use? Point me to the exact supporting sentence.",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    assert "Experimental setup" in out
    assert "62,500 Hz" in out
    assert "1.25 Ms/s" in out


def test_build_paper_guide_direct_answer_override_supports_exact_equation_prompt(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_resolve_exact_equation_support_from_source",
        lambda source_path, **_kwargs: {
            "source_path": source_path,
            "heading_path": "3. Method / 3.1. Background on NeRF",
            "equation_number": 1,
            "equation_markdown": "$$\nC(r)=... \\tag{1}\n$$",
            "explanation_text": "where t_n and t_f are near and far bounds.",
        },
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="equation",
        prompt_for_user=(
            "What does Equation (1) define in this paper, and where do the authors define the variables "
            "like t_n and t_f? Point me to the exact supporting part."
        ),
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    assert "Equation (1) is stated in 3. Method / 3.1. Background on NeRF:" in out
    assert "C(r)=..." in out
    assert "where t_n and t_f are near and far bounds" in out


def test_build_paper_guide_direct_answer_override_builds_component_role_answer_for_beginner_prompt(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_extract_bound_paper_component_role_focus",
        lambda source_path, **_kwargs: (
            "To enable robust registration of the iISM pinhole stack independent of the interferometric phase, we applied RVT to it. "
            "RVT computes, for each pixel, the variance of intensity values along concentric circular areas of increasing radius and generates a new image in which pixel intensity encodes the degree of radial symmetry.\n\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one. "
            "Image registration of the RVT pinhole stack yielded shift vectors. "
            "These vectors were then applied back to the original iISM dataset."
        ),
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="method",
        prompt_for_user="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    low = out.lower()
    assert "in simple terms" in low
    assert "rvt converts each pinhole image into a radial-symmetry map" in low
    assert "apr uses phase-correlation registration to estimate shift vectors" in low


def test_build_paper_guide_direct_answer_override_uses_targeted_strength_limits_hit(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_paper_guide_targeted_source_block_hits",
        lambda **_kwargs: [
            {
                "text": "How a single-pixel camera works",
                "meta": {
                    "heading_path": "Abstract / How a single-pixel camera works",
                    "kind": "heading",
                },
            },
            {
                "text": (
                    "A detailed comparison shows that there is a trade-off between these advantages "
                    "and the dynamic range of the detector and associated quantization electronics."
                ),
                "meta": {
                    "heading_path": "Abstract / How a single-pixel camera works",
                    "kind": "paragraph",
                },
            },
        ],
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="strength_limits",
        prompt_for_user=(
            "In the 'How a single-pixel camera works' section only, what trade-off do the authors describe "
            "between the advantages of single-pixel imaging and the detector dynamic range?"
        ),
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    low = out.lower()
    assert "how a single-pixel camera works" in low
    assert "trade-off" in low
    assert "dynamic range" in low


def test_build_paper_guide_direct_answer_override_uses_targeted_discussion_hit(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_paper_guide_targeted_source_block_hits",
        lambda **_kwargs: [
            {
                "text": (
                    "The iISM approach introduced here demonstrates that coherent scattering signals can be "
                    "harnessed within the ISM framework to achieve both high resolution and high sensitivity."
                ),
                "meta": {
                    "heading_path": "Discussion",
                    "kind": "paragraph",
                },
            },
            {
                "text": (
                    "The implementation in live cells further highlights the potential of iISM as a label-free modality. "
                    "This opens new possibilities for hybrid strategies, where fluorescence ISM and label-free iISM can be combined side by side. "
                    "Parallelized detection schemes could accelerate imaging and extend iISM to dynamic processes at the millisecond scale."
                ),
                "meta": {
                    "heading_path": "Discussion",
                    "kind": "paragraph",
                },
            },
            {
                "text": (
                    "Beyond our implementation, the method is highly adaptable. The camera can be exchanged for a SPAD array, "
                    "which would provide higher temporal resolution analogous to recent fluorescence ISM advances. "
                    "Parallelized detection schemes could further accelerate imaging, extending iISM to dynamic processes at the millisecond scale."
                ),
                "meta": {
                    "heading_path": "Discussion",
                    "kind": "paragraph",
                },
            },
        ],
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="discussion_only",
        prompt_for_user="From the Discussion section only, what future directions or extensions do the authors suggest for iISM?",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    low = out.lower()
    assert "spad array" in low
    assert "parallelized detection schemes" in low
    assert "high resolution and high sensitivity" not in low


def test_build_paper_guide_direct_answer_override_prefers_discussion_limitation_hit_for_spad_calibration_prompt(monkeypatch):
    monkeypatch.setattr(
        direct_answer_runtime,
        "_paper_guide_targeted_source_block_hits",
        lambda **_kwargs: [
            {
                "text": (
                    "The state-of-the-art large-scale single-photon imaging performance of the reported technique "
                    "builds on several innovations, including the physical multi-source noise modeling of SPAD arrays."
                ),
                "meta": {
                    "heading_path": "Discussion",
                    "kind": "paragraph",
                },
            },
            {
                "text": (
                    "In the workflow, the multi-source noise model parameters were calibrated employing a set of collected "
                    "images acquired using a specific SPAD camera. Although the reported noise model is generalized and "
                    "applicable to various single-photon detection schemes, the noise parameters of different SPAD arrays "
                    "may deviate from each other. In this regard, the automatic calibration of different SPAD arrays is "
                    "worthy of further study. Besides, we consider that the transfer learning technique can be adapted to "
                    "other single-photon detection hardware and settings."
                ),
                "meta": {
                    "heading_path": "Discussion",
                    "kind": "paragraph",
                },
            },
            {
                "text": (
                    "The photon detection efficiency of the SPAD array is varied at different wavelengths. "
                    "Second, we can employ various photon efficiency to retrieve spectral information of incident light, "
                    "opening research avenues on single-photon multispectral imaging."
                ),
                "meta": {
                    "heading_path": "Discussion",
                    "kind": "paragraph",
                },
            },
        ],
    )

    out = _build_paper_guide_direct_answer_override(
        paper_guide_mode=True,
        prompt_family="strength_limits",
        prompt_for_user="From the Discussion section only, what limitation do the authors note about calibrating different SPAD arrays, and what follow-up do they suggest?",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        answer_hits=[],
        special_focus_block="",
        db_dir="db",
        llm=None,
        build_direct_abstract_answer=lambda **_kwargs: "",
        build_direct_citation_lookup_answer=lambda **_kwargs: "",
    )

    low = out.lower()
    assert "specific spad camera" in low
    assert "different spad arrays may deviate" in low
    assert "transfer learning" in low
    assert "multispectral imaging" not in low
