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

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_REGISTRY = REPO_ROOT / "docs" / "claim_entailment_nli_candidate_v1.json"
MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
REVISION = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
EXPECTED_ACCEPTANCE = {
    "three_way_accuracy_min": 0.75,
    "entailment_recall_min": 0.75,
    "non_entailment_specificity_min": 0.8125,
    "contradiction_false_support_rate_max": 0.125,
    "lexical_adversary_false_support_rate_max": 0.0,
    "insufficient_false_support_rate_max": 0.25,
    "cross_lingual_accuracy_min": 0.7,
    "truncated_pair_count_max": 0,
    "all_required": True,
}


def validate_registry(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1 or payload.get("curation_status") != "frozen_before_candidate_measurement":
        errors.append("candidate registry is not frozen before measurement")
    source = payload.get("fixture") if isinstance(payload.get("fixture"), dict) else {}
    fixture_path = REPO_ROOT / str(source.get("path") or "")
    if not fixture_path.is_file() or source.get("sha256") != sha256_file(fixture_path):
        errors.append("claim entailment fixture identity does not match")
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    if candidate.get("model_id") != MODEL_ID or candidate.get("revision") != REVISION:
        errors.append("NLI candidate identity does not match the preselection")
    if not re.fullmatch(r"[0-9a-f]{40}", str(candidate.get("revision") or "")):
        errors.append("candidate revision is not immutable")
    if candidate.get("declared_license") != "mit" or candidate.get("safe_weights_required") is not True:
        errors.append("candidate license or safe-weight requirement changed")
    if candidate.get("model_safetensors_bytes") != 557652046 or candidate.get(
        "model_safetensors_sha256"
    ) != "65af59b1ff4450b09ecbf13ca35c840dbf038b26ff8e10e5ea89ca724828ed1e":
        errors.append("candidate safe-weight artifact identity changed")
    if candidate.get("precision") != "float32" or candidate.get("max_length") != 512:
        errors.append("candidate numeric contract changed")
    if candidate.get("truncation") != "only_first":
        errors.append("claim-preserving pair truncation changed")
    if candidate.get("label_order") != ["entailment", "neutral", "contradiction"]:
        errors.append("NLI label order changed")
    if candidate.get("trust_remote_code") is not False or candidate.get("product_enabled") is not False:
        errors.append("candidate must remain product-disabled without remote code")

    privacy = payload.get("privacy_and_supply_chain")
    privacy = privacy if isinstance(privacy, dict) else {}
    if privacy.get("inference_location") != "local_only" or privacy.get("paper_text_uploaded") is not False:
        errors.append("local-only evidence privacy contract changed")
    if privacy.get("normal_runtime_downloads") is not False:
        errors.append("normal runtime must not download model weights")
    if privacy.get("benchmark_download_requires_explicit_flag") is not True:
        errors.append("benchmark download consent contract changed")
    if privacy.get("subsequent_measurement_cache_only") is not True:
        errors.append("frozen measurement must be cache-only")

    if payload.get("acceptance") != EXPECTED_ACCEPTANCE:
        errors.append("pre-registered NLI acceptance thresholds changed")
    policy = payload.get("measurement_policy")
    policy = policy if isinstance(policy, dict) else {}
    if policy.get("candidate_count") != 1 or policy.get("run_count_on_frozen_fixture") != 1:
        errors.append("single-candidate single-run policy changed")
    if policy.get("threshold_tuning_on_fixture") is not False or policy.get("promotion_on_pass") is not False:
        errors.append("frozen fixture must not tune thresholds or auto-promote")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the preselected multilingual NLI candidate.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    payload = json.loads(args.registry.resolve().read_text(encoding="utf-8"))
    errors = validate_registry(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print(f"[OK] frozen NLI candidate: {MODEL_ID}@{REVISION[:12]}, local-only, product-disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
