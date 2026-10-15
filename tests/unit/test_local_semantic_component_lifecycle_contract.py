from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_local_semantic_component_lifecycle_contract import (
    DEFAULT_CONTRACT,
    validate_contract,
)


def _contract() -> dict:
    return json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))


def test_component_lifecycle_contract_passes() -> None:
    assert validate_contract(_contract()) == []


def test_contract_rejects_implicit_query_time_consent() -> None:
    changed = copy.deepcopy(_contract())
    changed["consent"]["query_time_implicit_consent_allowed"] = True
    assert "local semantic component consent boundary changed" in validate_contract(changed)


def test_contract_rejects_unverified_archive_extraction() -> None:
    changed = copy.deepcopy(_contract())
    changed["component_package"]["archive_hash_checked_before_extraction"] = False
    assert "local semantic component package integrity changed" in validate_contract(changed)


def test_contract_rejects_user_owned_capacity_reclamation() -> None:
    changed = copy.deepcopy(_contract())
    changed["capacity"]["never_delete_library_pdfs_markdown_or_user_notes"] = False
    assert "local semantic component capacity boundary changed" in validate_contract(changed)


def test_contract_rejects_restart_that_requires_user_resume() -> None:
    changed = copy.deepcopy(_contract())
    changed["lifecycle"]["non_user_interruption_recovers_automatically_after_restart"] = False
    assert "local semantic component lifecycle transitions changed" in validate_contract(changed)


def test_contract_rejects_integrity_retry_handed_to_user() -> None:
    changed = copy.deepcopy(_contract())
    changed["automatic_recovery"]["integrity_failure_requires_user_retry"] = True
    assert "local semantic component automatic recovery changed" in validate_contract(changed)


def test_contract_rejects_raw_checksum_exposure() -> None:
    changed = copy.deepcopy(_contract())
    changed["public_contract"]["raw_exception_path_or_checksum_exposed"] = True
    assert "local semantic component public failure contract changed" in validate_contract(changed)


def test_contract_rejects_removing_library_data() -> None:
    changed = copy.deepcopy(_contract())
    changed["removal"]["preserve_library_pdfs_markdown_notes_chat_and_bm25_index"] = False
    assert "local semantic component removal boundary changed" in validate_contract(changed)


def test_contract_rejects_premature_product_enablement() -> None:
    changed = copy.deepcopy(_contract())
    changed["product_boundary"]["product_enabled"] = True
    assert "local semantic component promotion or product boundary changed" in validate_contract(
        changed
    )
