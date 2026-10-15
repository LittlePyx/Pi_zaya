from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_product_opt_in_contract import (
    DEFAULT_CONTRACT,
    validate_contract,
)


def _contract() -> dict:
    return json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))


def test_product_opt_in_contract_is_valid() -> None:
    assert validate_contract(_contract()) == []


def test_product_opt_in_contract_rejects_external_content() -> None:
    changed = copy.deepcopy(_contract())
    changed["product_boundary"]["external_web_content_allowed"] = True
    assert "product boundary changed" in validate_contract(changed)


def test_product_opt_in_contract_rejects_implicit_download() -> None:
    changed = copy.deepcopy(_contract())
    changed["runtime_contract"]["generation_request_downloads_model"] = True
    assert "runtime fail-open contract changed" in validate_contract(changed)


def test_product_opt_in_contract_rejects_dead_unavailable_option() -> None:
    changed = copy.deepcopy(_contract())
    changed["user_experience_contract"]["show_unavailable_hybrid_dead_option"] = True
    assert "user experience contract changed" in validate_contract(changed)


def test_product_opt_in_contract_rejects_blocking_fallback() -> None:
    changed = copy.deepcopy(_contract())
    changed["runtime_contract"]["fallback_blocks_answer"] = True
    assert "runtime fail-open contract changed" in validate_contract(changed)
