from __future__ import annotations

from tools.manual_regression import run_captured_pool_live_eval as live_eval
from tools.manual_regression.score_captured_replay_pool_rubric import _case_family, _score_case
from tests.replay.collect_paper_guide_replay_cases import _classify_case as _collect_classify_case
from tests.replay.curate_paper_guide_failure_pool_v1 import _classify_case as _curate_classify_case


def _build_role_case(answer: str, *, tag: str = "citation_lookup") -> dict:
    return {
        "id": "captured::role::1",
        "prompt": "I am new to this paper. What are RVT and APR doing here, in simple terms?",
        "intent_family": "",
        "tag": tag,
        "assistant_content": answer,
        "provenance": {
            "segments": [
                {
                    "evidence_mode": "direct",
                    "locate_policy": "required",
                    "must_locate": True,
                    "hit_level": "block",
                    "primary_heading_path": "Data analysis / Radial variance transform (RVT)",
                },
                {
                    "evidence_mode": "direct",
                    "locate_policy": "required",
                    "must_locate": True,
                    "hit_level": "exact",
                    "primary_heading_path": "Data analysis / Adaptive pixel-reassignment (APR)",
                },
            ]
        },
    }


def _build_discussion_case(prompt: str) -> dict:
    return {
        "id": "captured::discussion::1",
        "prompt": prompt,
        "intent_family": "",
        "tag": "",
        "assistant_content": "The discussion suggests future directions around faster hardware and better robustness.",
        "provenance": {"segments": []},
    }


def _build_strength_limits_case(prompt: str) -> dict:
    return {
        "id": "captured::strength_limits::1",
        "prompt": prompt,
        "intent_family": "",
        "tag": "",
        "assistant_content": "A key limitation is the remaining dependence on indirect evidence and trade-offs in robustness.",
        "provenance": {"segments": []},
    }


def test_role_prompt_family_overrides_legacy_citation_tag_for_eval_and_collection():
    case = _build_role_case("placeholder")

    assert _case_family(case) == "overview"
    assert live_eval._family_for_case(case) == "overview"
    assert _collect_classify_case(case) == "overview"
    assert _curate_classify_case(case) == "citation_lookup"


def test_discussion_prompt_family_classifies_consistently_across_scripts():
    case = _build_discussion_case(
        "From the Discussion section only, what future directions do the authors suggest for iISM?"
    )

    assert _case_family(case) == "discussion_only"
    assert live_eval._family_for_case(case) == "discussion_only"
    assert _collect_classify_case(case) == "discussion_only"
    assert _curate_classify_case(case) == "discussion_only"


def test_future_work_section_prompt_classifies_as_discussion_only_across_scripts():
    case = _build_discussion_case(
        "From the Future Work section only, what extension do the authors suggest next?"
    )

    assert _case_family(case) == "discussion_only"
    assert live_eval._family_for_case(case) == "discussion_only"
    assert _collect_classify_case(case) == "discussion_only"
    assert _curate_classify_case(case) == "discussion_only"


def test_strength_limits_prompt_family_classifies_consistently_across_scripts():
    case = _build_strength_limits_case(
        "What are the main limitations or trade-offs of this method according to the paper?"
    )

    assert _case_family(case) == "strength_limits"
    assert live_eval._family_for_case(case) == "strength_limits"
    assert _collect_classify_case(case) == "strength_limits"
    assert _curate_classify_case(case) == "strength_limits"


def test_score_case_rewards_grounded_beginner_role_explanation():
    old_case = _build_role_case(
        "- The only mention of **RVT** is the phrase “Radial variance transform (RVT)” under a section heading, "
        "with no further explanation given in the retrieved text.\n\n"
        "Implementation detail: APR was performed using image registration based on phase correlation of the off-axis "
        "raw images with respect to the central one, as detailed in [35]. [[CITE:s3583e628:35]]"
    )
    new_case = _build_role_case(
        "From the retrieved method evidence, in simple terms:\n"
        "- RVT converts each pinhole image into a radial-symmetry map so the registration step is more robust to "
        "interferometric phase.\n"
        "- APR uses phase-correlation registration to estimate shift vectors, then applies those shifts back to the "
        "original iISM data before summation."
    )

    old_score = _score_case(old_case)
    new_score = _score_case(new_case)

    assert new_score.overall_100 > old_score.overall_100
    assert new_score.question_hit > old_score.question_hit
    assert new_score.evidence_consistency > old_score.evidence_consistency
    assert new_score.uncertainty_handling > old_score.uncertainty_handling
    assert new_score.overall_100 >= 90.0
