from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.agent.citation_structure import inspect_citation_structure
from kb.claim_verification_shadow import arbitrate_shadow_claim, evaluate_support_strategy
from kb.retrieval_eval_fixture import sha256_file


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "claim_entailment_eval_v1.json"
DEFAULT_LEXICAL = REPO_ROOT / "docs" / "claim_entailment_lexical_baseline_v1.json"
DEFAULT_NLI = REPO_ROOT / "docs" / "claim_entailment_nli_baseline_v1.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "claim_verification_shadow_design_v1.json"
STRUCTURE_SOURCE = REPO_ROOT / "kb" / "agent" / "citation_structure.py"


def _cited_claim(claim: str) -> str:
    cited = re.sub(r"([.!?。！？])$", r" [1]\1", str(claim or "").strip())
    return cited if "[1]" in cited else f"{cited} [1]"


def build_report(
    fixture_path: Path = DEFAULT_FIXTURE,
    lexical_path: Path = DEFAULT_LEXICAL,
    nli_path: Path = DEFAULT_NLI,
) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    lexical = json.loads(lexical_path.read_text(encoding="utf-8"))
    nli = json.loads(nli_path.read_text(encoding="utf-8"))
    lexical_rows = {row["id"]: row for row in lexical["result"]["cases"]}
    nli_rows = {row["id"]: row for row in nli["result"]["cases"]}
    case_ids = [str(case.get("id") or "") for case in fixture.get("cases") or []]
    if set(case_ids) != set(lexical_rows) or set(case_ids) != set(nli_rows):
        raise ValueError("lexical and NLI reports must cover the frozen fixture exactly")

    structural: dict[str, dict[str, Any]] = {}
    selected_support: dict[str, bool] = {}
    naive_and: dict[str, bool] = {}
    contradiction_veto: dict[str, bool] = {}
    outcomes: dict[str, int] = {}
    for case in fixture.get("cases") or []:
        case_id = str(case.get("id") or "")
        structure = inspect_citation_structure(
            _cited_claim(str(case.get("claim") or "")), evidence_hit_count=1
        ).to_dict()
        structural[case_id] = structure
        nli_label = str(nli_rows[case_id].get("observed_nli_label") or "")
        shadow = arbitrate_shadow_claim(citation_structure=structure, nli_label=nli_label)
        selected_support[case_id] = bool(shadow["shadow_supported"])
        outcomes[shadow["outcome"]] = outcomes.get(shadow["outcome"], 0) + 1
        lexical_supported = bool(lexical_rows[case_id].get("observed_supported"))
        nli_supported = bool(nli_rows[case_id].get("observed_supported"))
        naive_and[case_id] = lexical_supported and nli_supported
        contradiction_veto[case_id] = lexical_supported and nli_label != "contradiction"

    structure_rows = list(structural.values())
    strategies = {
        "production_lexical_overlap": evaluate_support_strategy(
            fixture,
            {case_id: bool(row.get("observed_supported")) for case_id, row in lexical_rows.items()},
        ),
        "naive_lexical_and_nli": evaluate_support_strategy(fixture, naive_and),
        "lexical_with_nli_contradiction_veto": evaluate_support_strategy(
            fixture, contradiction_veto
        ),
        "selected_structural_binding_plus_nli": evaluate_support_strategy(
            fixture, selected_support
        ),
    }
    return {
        "schema_version": 1,
        "report_kind": "claim_verification_shadow_design",
        "inputs": {
            "fixture": {
                "path": fixture_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(fixture_path),
            },
            "lexical_report": {
                "path": lexical_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(lexical_path),
            },
            "nli_report": {
                "path": nli_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(nli_path),
            },
            "citation_structure_source": {
                "path": STRUCTURE_SOURCE.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(STRUCTURE_SOURCE),
            },
        },
        "structural_fixture": {
            "case_count": len(structure_rows),
            "structurally_bound_count": sum(
                bool(row.get("structurally_bound")) for row in structure_rows
            ),
            "unresolved_count": sum(
                not bool(row.get("structurally_bound")) for row in structure_rows
            ),
        },
        "strategies": strategies,
        "selected_shadow_outcomes": outcomes,
        "decision": {
            "selected": "structural_binding_plus_nli",
            "mode": "shadow_only",
            "production_effect": False,
            "ordinary_user_status_exposed": False,
            "product_enabled": False,
            "fallback": "current production lexical verifier remains authoritative",
            "reason": (
                "Naive lexical+NLI intersection removes all false support but drops entailment "
                "recall to 37.5%. Structural citation validity plus the frozen NLI argmax keeps "
                "87.5% entailment recall and 93.75% non-entailment specificity, while retaining "
                "the known TimesFM false-support case as an explicit shadow failure."
            ),
            "telemetry_contract": (
                "Persist only fixture/outcome IDs, aggregate counts, duration, model revision, "
                "and fallback reason; never claim text, evidence text, source paths, or credentials."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or validate the claim-verification shadow design.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--lexical", type=Path, default=DEFAULT_LEXICAL)
    parser.add_argument("--nli", type=Path, default=DEFAULT_NLI)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    expected = build_report(args.fixture.resolve(), args.lexical.resolve(), args.nli.resolve())
    rendered = json.dumps(expected, ensure_ascii=False, indent=2) + "\n"
    if args.write:
        args.report.resolve().write_text(rendered, encoding="utf-8")
    if args.validate_only:
        report_path = args.report.resolve()
        if not report_path.is_file() or report_path.read_text(encoding="utf-8") != rendered:
            print(f"[FAIL] claim shadow report differs from {report_path}")
            return 1
    selected = expected["strategies"]["selected_structural_binding_plus_nli"]["metrics"]
    print(
        "[OK] claim shadow design: "
        f"accuracy={selected['binary_support_accuracy']:.3f}, "
        f"entailment recall={selected['entailment_recall']:.3f}, "
        f"production_effect={expected['decision']['production_effect']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
