from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8016"
DEFAULT_GENERATE_BODY = {
    "top_k": 6,
    "deep_read": True,
    "max_tokens": 1400,
    "temperature": 0.15,
}
DEFAULT_REFS_WAIT_TIMEOUT_CAP_S = 45.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve_repo_path(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def load_suite(manifest_path: str) -> dict[str, Any]:
    path = Path(str(manifest_path or "").strip()).expanduser()
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    suite = dict(payload or {})
    suite["_manifest_abspath"] = str(path)
    cases = []
    for raw_case in list(suite.get("cases") or []):
        case = copy.deepcopy(raw_case or {})
        bound_source_path = _resolve_repo_path(str(case.get("bound_source_path") or ""))
        if bound_source_path:
            case["bound_source_path"] = bound_source_path
        case["_manifest_path"] = str(path)
        cases.append(case)
    suite["cases"] = cases
    return suite


def _parse_case_filter(values: list[str] | None) -> set[str]:
    out: set[str] = set()
    for raw in list(values or []):
        for item in str(raw or "").split(","):
            s = str(item or "").strip()
            if s:
                out.add(s)
    return out


def _filter_suite_cases(suite: dict[str, Any], case_ids: set[str]) -> dict[str, Any]:
    if not case_ids:
        return dict(suite or {})
    out = dict(suite or {})
    out["cases"] = [
        dict(case)
        for case in list(suite.get("cases") or [])
        if str((case or {}).get("id") or "").strip() in case_ids
    ]
    return out


def _post_json(base_url: str, path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    resp = requests.post(
        f"{base_url.rstrip('/')}{path}",
        json=payload,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def _get_json(base_url: str, path: str, *, timeout_s: float) -> dict[str, Any]:
    resp = requests.get(f"{base_url.rstrip('/')}{path}", timeout=timeout_s)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def _stream_done(base_url: str, session_id: str, *, timeout_s: float) -> dict[str, Any]:
    resp = requests.get(
        f"{base_url.rstrip('/')}/api/generate/{session_id}/stream",
        stream=True,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    final_payload: dict[str, Any] = {}
    t0 = time.time()
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if (time.time() - t0) > timeout_s:
                break
            line = str(raw or "").strip()
            if not line.startswith("data:"):
                continue
            data_txt = line[len("data:") :].strip()
            if not data_txt:
                continue
            try:
                payload = json.loads(data_txt)
            except Exception:
                continue
            if isinstance(payload, dict):
                final_payload = payload
                if bool(payload.get("done")):
                    break
    finally:
        resp.close()
    return final_payload


def _post_json_client(client: Any, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = client.post(path, json=payload)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def _get_json_client(client: Any, path: str) -> dict[str, Any]:
    resp = client.get(path)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def _stream_done_client(client: Any, session_id: str, *, timeout_s: float) -> dict[str, Any]:
    final_payload: dict[str, Any] = {}
    t0 = time.time()
    with client.stream("GET", f"/api/generate/{session_id}/stream") as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if (time.time() - t0) > timeout_s:
                break
            line = str(raw or "").strip()
            if not line.startswith("data:"):
                continue
            data_txt = line[len("data:") :].strip()
            if not data_txt:
                continue
            try:
                payload = json.loads(data_txt)
            except Exception:
                continue
            if isinstance(payload, dict):
                final_payload = payload
                if bool(payload.get("done")):
                    break
    return final_payload


def _select_refs_pack(refs_by_user: dict[str, Any] | dict[int, Any], user_msg_id: int) -> dict[str, Any]:
    if not isinstance(refs_by_user, dict):
        return {}
    for key in (user_msg_id, str(user_msg_id)):
        pack = refs_by_user.get(key)
        if isinstance(pack, dict):
            return pack
    return {}


def _pack_has_pending_hits(pack: dict[str, Any]) -> bool:
    if bool(pack.get("pending")):
        return True
    for hit in list(pack.get("hits") or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
            return True
    return False


def _wait_for_refs_pack(
    *,
    base_url: str,
    conv_id: str,
    user_msg_id: int,
    timeout_s: float,
) -> dict[str, Any]:
    t0 = time.time()
    last_pack: dict[str, Any] = {}
    stable_since = time.time()
    last_sig = ""
    while (time.time() - t0) <= timeout_s:
        refs_by_user = _get_json(base_url, f"/api/references/conversation/{conv_id}", timeout_s=min(10.0, timeout_s))
        pack = _select_refs_pack(refs_by_user, user_msg_id)
        sig = json.dumps(pack, ensure_ascii=False, sort_keys=True)
        if sig != last_sig:
            last_sig = sig
            stable_since = time.time()
        last_pack = pack
        if pack and (not _pack_has_pending_hits(pack)) and (time.time() - stable_since) >= 0.8:
            return pack
        time.sleep(0.6)
    return last_pack


def _observe_refs_pack_lifecycle(
    *,
    base_url: str,
    conv_id: str,
    user_msg_id: int,
    timeout_s: float,
) -> dict[str, Any]:
    t0 = time.time()
    last_pack: dict[str, Any] = {}
    last_stable_pack: dict[str, Any] = {}
    first_stable_pack: dict[str, Any] = {}
    stable_since = time.time()
    last_sig = ""
    while (time.time() - t0) <= timeout_s:
        refs_by_user = _get_json(base_url, f"/api/references/conversation/{conv_id}", timeout_s=min(10.0, timeout_s))
        pack = _select_refs_pack(refs_by_user, user_msg_id)
        sig = json.dumps(pack, ensure_ascii=False, sort_keys=True)
        if sig != last_sig:
            last_sig = sig
            stable_since = time.time()
        last_pack = pack
        if pack and (not _pack_has_pending_hits(pack)) and (time.time() - stable_since) >= 0.8:
            stable_pack = dict(pack)
            if not first_stable_pack:
                first_stable_pack = stable_pack
            last_stable_pack = stable_pack
            if _render_status_identity(stable_pack) == "full":
                break
        time.sleep(0.6)
    final_pack = last_stable_pack or last_pack
    return {
        "first_stable_pack": first_stable_pack or final_pack,
        "final_pack": final_pack,
    }


def _wait_for_refs_pack_client(
    *,
    client: Any,
    conv_id: str,
    user_msg_id: int,
    timeout_s: float,
) -> dict[str, Any]:
    t0 = time.time()
    last_pack: dict[str, Any] = {}
    stable_since = time.time()
    last_sig = ""
    while (time.time() - t0) <= timeout_s:
        refs_by_user = _get_json_client(client, f"/api/references/conversation/{conv_id}")
        pack = _select_refs_pack(refs_by_user, user_msg_id)
        sig = json.dumps(pack, ensure_ascii=False, sort_keys=True)
        if sig != last_sig:
            last_sig = sig
            stable_since = time.time()
        last_pack = pack
        if pack and (not _pack_has_pending_hits(pack)) and (time.time() - stable_since) >= 0.8:
            return pack
        time.sleep(0.6)
    return last_pack


def _observe_refs_pack_lifecycle_client(
    *,
    client: Any,
    conv_id: str,
    user_msg_id: int,
    timeout_s: float,
) -> dict[str, Any]:
    t0 = time.time()
    last_pack: dict[str, Any] = {}
    last_stable_pack: dict[str, Any] = {}
    first_stable_pack: dict[str, Any] = {}
    stable_since = time.time()
    last_sig = ""
    while (time.time() - t0) <= timeout_s:
        refs_by_user = _get_json_client(client, f"/api/references/conversation/{conv_id}")
        pack = _select_refs_pack(refs_by_user, user_msg_id)
        sig = json.dumps(pack, ensure_ascii=False, sort_keys=True)
        if sig != last_sig:
            last_sig = sig
            stable_since = time.time()
        last_pack = pack
        if pack and (not _pack_has_pending_hits(pack)) and (time.time() - stable_since) >= 0.8:
            stable_pack = dict(pack)
            if not first_stable_pack:
                first_stable_pack = stable_pack
            last_stable_pack = stable_pack
            if _render_status_identity(stable_pack) == "full":
                break
        time.sleep(0.6)
    final_pack = last_stable_pack or last_pack
    return {
        "first_stable_pack": first_stable_pack or final_pack,
        "final_pack": final_pack,
    }


def _messages_for_conversation(*, base_url: str, conv_id: str, timeout_s: float) -> list[dict[str, Any]]:
    payload = _get_json(
        base_url,
        f"/api/conversations/{conv_id}/messages?render_packet_only=1",
        timeout_s=min(10.0, timeout_s),
    )
    return [dict(item) for item in list(payload or []) if isinstance(item, dict)]


def _messages_for_conversation_client(*, client: Any, conv_id: str) -> list[dict[str, Any]]:
    payload = _get_json_client(client, f"/api/conversations/{conv_id}/messages?render_packet_only=1")
    return [dict(item) for item in list(payload or []) if isinstance(item, dict)]


def _top_hit(pack: dict[str, Any]) -> dict[str, Any]:
    hits = [hit for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
    return hits[0] if hits else {}


def _assistant_message(messages: list[dict[str, Any]] | None) -> dict[str, Any]:
    for msg in reversed(list(messages or [])):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role") or "").strip().lower() == "assistant":
            return dict(msg)
    return {}


def _assistant_message_model_failure(message: dict[str, Any] | None) -> bool:
    if not isinstance(message, dict):
        return False
    text = str(message.get("content") or "").strip().lower()
    if not text:
        return False
    markers = [
        "调用模型失败",
        "connection error",
        "deepseek_api_key",
        "base_url / model",
        "please check deepseek_api_key",
    ]
    return any(marker in text for marker in markers)


def _message_render_packet(message: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(message, dict):
        return {}
    meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
    contracts = meta.get("paper_guide_contracts") if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    for cand in (
        contracts.get("render_packet"),
        message.get("render_packet"),
    ):
        if isinstance(cand, dict) and cand:
            return dict(cand)
    return {}


def _normalize_evidence_surface(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out = {
        "source_path": str(raw.get("source_path") or raw.get("sourcePath") or "").strip() or None,
        "source_name": str(raw.get("source_name") or raw.get("sourceName") or "").strip() or None,
        "block_id": str(raw.get("block_id") or raw.get("blockId") or "").strip() or None,
        "anchor_id": str(raw.get("anchor_id") or raw.get("anchorId") or "").strip() or None,
        "heading_path": str(raw.get("heading_path") or raw.get("headingPath") or "").strip() or None,
        "snippet": (
            str(raw.get("snippet") or "").strip()
            or str(raw.get("highlight_snippet") or raw.get("highlightSnippet") or "").strip()
            or None
        ),
    }
    return {
        key: value
        for key, value in out.items()
        if value not in (None, "", [], {})
    }


def _source_identity(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    name = Path(text).name or text
    name = name.replace("\\", "/")
    name = name.rsplit("/", 1)[-1]
    for suffix in (".en.md", ".md", ".pdf"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return " ".join(name.replace("_", " ").replace("-", " ").split())


def _surface_source_identity(surface: dict[str, Any]) -> str:
    source_name = _source_identity(surface.get("source_name") or "")
    if source_name:
        return source_name
    return _source_identity(surface.get("source_path") or "")


def _surfaces_align(lhs: dict[str, Any], rhs: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    lhs_source = _surface_source_identity(lhs)
    rhs_source = _surface_source_identity(rhs)
    if lhs_source and rhs_source and lhs_source != rhs_source:
        mismatches.append("source")
    lhs_heading = str(lhs.get("heading_path") or "").strip().lower()
    rhs_heading = str(rhs.get("heading_path") or "").strip().lower()
    if lhs_heading and rhs_heading and lhs_heading != rhs_heading:
        mismatches.append("heading_path")
    lhs_block = str(lhs.get("block_id") or "").strip().lower()
    rhs_block = str(rhs.get("block_id") or "").strip().lower()
    if lhs_block and rhs_block and lhs_block != rhs_block:
        mismatches.append("block_id")
    lhs_anchor = str(lhs.get("anchor_id") or "").strip().lower()
    rhs_anchor = str(rhs.get("anchor_id") or "").strip().lower()
    if lhs_anchor and rhs_anchor and lhs_anchor != rhs_anchor:
        mismatches.append("anchor_id")
    return mismatches


def _pack_primary_evidence(refs_pack: dict[str, Any]) -> dict[str, Any]:
    pack_primary = _normalize_evidence_surface(refs_pack.get("primary_evidence") if isinstance(refs_pack, dict) else {})
    if pack_primary:
        return pack_primary
    top = _top_hit(refs_pack if isinstance(refs_pack, dict) else {})
    ui = top.get("ui_meta") if isinstance(top.get("ui_meta"), dict) else {}
    return _normalize_evidence_surface(ui.get("primary_evidence") if isinstance(ui.get("primary_evidence"), dict) else {})


def _hit_reader_primary_evidence(hit: dict[str, Any]) -> dict[str, Any]:
    ui = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
    reader_open = ui.get("reader_open") if isinstance(ui.get("reader_open"), dict) else {}
    reader_primary = reader_open.get("primaryEvidence") if isinstance(reader_open.get("primaryEvidence"), dict) else {}
    if isinstance(reader_primary, dict) and reader_primary:
        return _normalize_evidence_surface(reader_primary)
    return _normalize_evidence_surface(reader_open)


def _assistant_primary_evidence(message: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(message, dict):
        return {}
    meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
    contracts = meta.get("paper_guide_contracts") if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    contract_primary = _normalize_evidence_surface(contracts.get("primary_evidence") if isinstance(contracts.get("primary_evidence"), dict) else {})
    if contract_primary:
        return contract_primary
    packet = _message_render_packet(message)
    packet_primary = _normalize_evidence_surface(packet.get("primary_evidence") if isinstance(packet.get("primary_evidence"), dict) else {})
    if packet_primary:
        return packet_primary
    for cand in (message.get("provenance"), meta.get("provenance")):
        if isinstance(cand, dict):
            prov_primary = _normalize_evidence_surface(cand.get("primary_evidence") if isinstance(cand.get("primary_evidence"), dict) else {})
            if prov_primary:
                return prov_primary
    return {}


def _surface_heading(surface: dict[str, Any] | None) -> str:
    if not isinstance(surface, dict):
        return ""
    return str(surface.get("heading_path") or "").strip()


def _top_hit_heading(hit: dict[str, Any] | None) -> str:
    if not isinstance(hit, dict):
        return ""
    ui = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
    return str(ui.get("heading_path") or "").strip()


def _metric_summary_heading_consistent(*, top_hit: dict[str, Any], pack_primary: dict[str, Any]) -> dict[str, Any]:
    ui = top_hit.get("ui_meta") if isinstance(top_hit.get("ui_meta"), dict) else {}
    summary_line = str(ui.get("summary_line") or "").strip()
    card_heading = _top_hit_heading(top_hit)
    summary_primary = _normalize_evidence_surface(
        ui.get("primary_evidence") if isinstance(ui.get("primary_evidence"), dict) else {}
    ) or dict(pack_primary or {})
    primary_heading = _surface_heading(summary_primary)
    if (not summary_line) or (not card_heading) or (not primary_heading):
        return {
            "value": None,
            "reason": "insufficient_summary_heading_data",
            "card_heading": card_heading,
            "primary_heading": primary_heading,
        }
    return {
        "value": card_heading.strip().lower() == primary_heading.strip().lower(),
        "reason": "",
        "card_heading": card_heading,
        "primary_heading": primary_heading,
    }


def _metric_heading_reader_open_consistent(*, top_hit: dict[str, Any], reader_primary: dict[str, Any]) -> dict[str, Any]:
    ui = top_hit.get("ui_meta") if isinstance(top_hit.get("ui_meta"), dict) else {}
    reader_open = ui.get("reader_open") if isinstance(ui.get("reader_open"), dict) else {}
    card_heading = _top_hit_heading(top_hit)
    reader_heading = str(reader_open.get("headingPath") or "").strip() or _surface_heading(reader_primary)
    if (not card_heading) or (not reader_heading):
        return {
            "value": None,
            "reason": "insufficient_reader_heading_data",
            "card_heading": card_heading,
            "reader_heading": reader_heading,
        }
    return {
        "value": card_heading.strip().lower() == reader_heading.strip().lower(),
        "reason": "",
        "card_heading": card_heading,
        "reader_heading": reader_heading,
    }


def _render_status_identity(pack: dict[str, Any]) -> str:
    if not isinstance(pack, dict):
        return ""
    render_status = str(pack.get("render_status") or "").strip().lower()
    if render_status:
        return render_status
    payload_mode = str(pack.get("payload_mode") or "").strip().lower()
    if payload_mode:
        return payload_mode
    return ""


def _metric_fast_to_full_primary_block_same(
    *,
    first_pack: dict[str, Any],
    final_pack: dict[str, Any],
) -> dict[str, Any]:
    first_primary = _pack_primary_evidence(first_pack)
    final_primary = _pack_primary_evidence(final_pack)
    first_status = _render_status_identity(first_pack)
    final_status = _render_status_identity(final_pack)
    first_sig = json.dumps(first_primary, ensure_ascii=False, sort_keys=True)
    final_sig = json.dumps(final_primary, ensure_ascii=False, sort_keys=True)
    transition_observed = bool(first_sig and final_sig and (first_sig != final_sig or first_status != final_status))
    full_observed = final_status == "full"
    if not first_primary or not final_primary:
        return {
            "value": None,
            "reason": "missing_fast_or_full_primary_evidence",
            "first_primary": first_primary,
            "final_primary": final_primary,
            "first_status": first_status,
            "final_status": final_status,
            "transition_observed": transition_observed,
        }
    if (not transition_observed) and (not full_observed):
        return {
            "value": None,
            "reason": "full_transition_not_observed",
            "first_primary": first_primary,
            "final_primary": final_primary,
            "first_status": first_status,
            "final_status": final_status,
            "transition_observed": transition_observed,
        }
    mismatches = _surfaces_align(first_primary, final_primary)
    return {
        "value": not mismatches,
        "reason": "" if not mismatches else f"mismatch:{','.join(mismatches)}",
        "first_primary": first_primary,
        "final_primary": final_primary,
        "first_status": first_status,
        "final_status": final_status,
        "transition_observed": transition_observed,
    }


def _source_surface(hit: dict[str, Any]) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    ui = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
    return " | ".join(
        bit for bit in [
            str((meta or {}).get("source_path") or "").strip(),
            str((ui or {}).get("display_name") or "").strip(),
        ] if bit
    )


def _check_contains_any(text: str, needles: list[str]) -> bool:
    hay = str(text or "")
    hay_low = hay.lower()
    aliases = {
        "matched section": [
            "matched section",
            "matched section evidence",
            "section-grounded",
            "section grounded",
            "命中章节",
            "定位证据",
            "命中章节/定位证据",
            "关键词对齐",
        ],
        "llm": [
            "llm",
            "large language model",
            "模型",
            "大模型",
        ],
    }
    for needle in list(needles or []):
        raw = str(needle or "").strip()
        if not raw:
            continue
        variants = [raw]
        for key, extra in aliases.items():
            if raw.lower() == key:
                variants.extend(extra)
                break
        if any(str(variant or "").strip() and str(variant).lower() in hay_low for variant in variants):
            return True
    return False


def _evaluate_case(
    case: dict[str, Any],
    *,
    refs_pack: dict[str, Any],
    assistant_message: dict[str, Any] | None = None,
    first_refs_pack: dict[str, Any] | None = None,
    final_refs_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = case.get("checks") if isinstance(case.get("checks"), dict) else {}
    hits = [hit for hit in list(refs_pack.get("hits") or []) if isinstance(hit, dict)]
    guide_filter = refs_pack.get("guide_filter") if isinstance(refs_pack.get("guide_filter"), dict) else {}
    pipeline_debug = refs_pack.get("pipeline_debug") if isinstance(refs_pack.get("pipeline_debug"), dict) else {}
    gate_results: dict[str, Any] = {}
    failures: list[str] = []

    hit_checks = checks.get("hits") if isinstance(checks.get("hits"), dict) else {}
    if hit_checks:
        min_hits = int(hit_checks.get("min") or 0)
        max_hits = int(hit_checks.get("max") or 0)
        hit_count = len(hits)
        status = "PASS"
        reasons: list[str] = []
        if min_hits and hit_count < min_hits:
            status = "FAIL"
            reasons.append(f"expected_at_least_{min_hits}_hits_got_{hit_count}")
        if max_hits >= 0 and hit_count > max_hits:
            status = "FAIL"
            reasons.append(f"expected_at_most_{max_hits}_hits_got_{hit_count}")
        gate_results["hits"] = {"status": status, "count": hit_count, "reasons": reasons}
        failures.extend(reasons)

    top_checks = checks.get("top_hit") if isinstance(checks.get("top_hit"), dict) else {}
    if top_checks:
        top = _top_hit(refs_pack)
        ui = top.get("ui_meta") if isinstance(top.get("ui_meta"), dict) else {}
        source_text = _source_surface(top)
        status = "PASS"
        reasons: list[str] = []
        if not top:
            status = "FAIL"
            reasons.append("missing_top_hit")
        else:
            if top_checks.get("source_contains_any") and (not _check_contains_any(source_text, list(top_checks.get("source_contains_any") or []))):
                status = "FAIL"
                reasons.append("top_hit_source_mismatch")
            if top_checks.get("source_not_contains_any") and _check_contains_any(source_text, list(top_checks.get("source_not_contains_any") or [])):
                status = "FAIL"
                reasons.append("top_hit_source_forbidden")
            if top_checks.get("summary_contains_any") and (not _check_contains_any(str((ui or {}).get("summary_line") or ""), list(top_checks.get("summary_contains_any") or []))):
                status = "FAIL"
                reasons.append("summary_line_not_specific_enough")
            if top_checks.get("why_contains_any") and (not _check_contains_any(str((ui or {}).get("why_line") or ""), list(top_checks.get("why_contains_any") or []))):
                status = "FAIL"
                reasons.append("why_line_not_specific_enough")
            if top_checks.get("summary_basis_contains_any") and (not _check_contains_any(str((ui or {}).get("summary_basis") or ""), list(top_checks.get("summary_basis_contains_any") or []))):
                status = "FAIL"
                reasons.append("summary_basis_mismatch")
            if top_checks.get("why_basis_contains_any") and (not _check_contains_any(str((ui or {}).get("why_basis") or ""), list(top_checks.get("why_basis_contains_any") or []))):
                status = "FAIL"
                reasons.append("why_basis_mismatch")
        gate_results["top_hit"] = {
            "status": status,
            "source": source_text,
            "summary_line": str((ui or {}).get("summary_line") or "") if top else "",
            "why_line": str((ui or {}).get("why_line") or "") if top else "",
            "summary_basis": str((ui or {}).get("summary_basis") or "") if top else "",
            "why_basis": str((ui or {}).get("why_basis") or "") if top else "",
            "reasons": reasons,
        }
        failures.extend(reasons)

    filter_checks = checks.get("guide_filter") if isinstance(checks.get("guide_filter"), dict) else {}
    if filter_checks:
        status = "PASS"
        reasons: list[str] = []
        for key in ("active", "hidden_self_source"):
            if key in filter_checks:
                expected = bool(filter_checks.get(key))
                actual = bool(guide_filter.get(key))
                if expected != actual:
                    status = "FAIL"
                    reasons.append(f"guide_filter_{key}_expected_{expected}_got_{actual}")
        gate_results["guide_filter"] = {
            "status": status,
            "guide_filter": guide_filter,
            "reasons": reasons,
        }
        failures.extend(reasons)

    pipeline_checks = checks.get("pipeline_debug") if isinstance(checks.get("pipeline_debug"), dict) else {}
    if pipeline_checks or pipeline_debug:
        status = "PASS"
        reasons: list[str] = []
        raw_hit_count = int(pipeline_debug.get("raw_hit_count") or 0)
        post_score_gate_hit_count = int(pipeline_debug.get("post_score_gate_hit_count") or 0)
        post_focus_filter_hit_count = int(pipeline_debug.get("post_focus_filter_hit_count") or 0)
        post_llm_filter_hit_count = int(pipeline_debug.get("post_llm_filter_hit_count") or 0)
        final_hit_count = int(pipeline_debug.get("final_hit_count") or len(hits))
        if bool(pipeline_checks.get("require_raw_hits")) and raw_hit_count <= 0:
            status = "FAIL"
            reasons.append("retrieval_empty_before_ui_filters")
        if bool(pipeline_checks.get("require_post_score_hits")) and post_score_gate_hit_count <= 0:
            status = "FAIL"
            reasons.append("all_hits_removed_by_score_gate")
        if bool(pipeline_checks.get("require_post_focus_hits")) and post_focus_filter_hit_count <= 0:
            status = "FAIL"
            reasons.append("all_hits_removed_by_focus_filter")
        if bool(pipeline_checks.get("require_post_llm_hits")) and post_llm_filter_hit_count <= 0:
            status = "FAIL"
            reasons.append("all_hits_removed_by_llm_filter")
        if "max_raw_to_final_drop" in pipeline_checks:
            max_drop = max(0, int(pipeline_checks.get("max_raw_to_final_drop") or 0))
            actual_drop = max(0, raw_hit_count - final_hit_count)
            if actual_drop > max_drop:
                status = "FAIL"
                reasons.append(f"raw_to_final_drop_too_large:{actual_drop}>{max_drop}")
        gate_results["pipeline_debug"] = {
            "status": status,
            "pipeline_debug": pipeline_debug,
            "reasons": reasons,
        }
        failures.extend(reasons)

    identity_checks = checks.get("evidence_identity") if isinstance(checks.get("evidence_identity"), dict) else {}
    if identity_checks:
        top = _top_hit(refs_pack)
        pack_primary = _pack_primary_evidence(refs_pack)
        hit_primary = _normalize_evidence_surface(
            ((top.get("ui_meta") or {}).get("primary_evidence") if isinstance((top.get("ui_meta") or {}).get("primary_evidence"), dict) else {})
            if isinstance(top, dict)
            else {}
        )
        reader_primary = _hit_reader_primary_evidence(top)
        assistant_message_dict = dict(assistant_message or {})
        assistant_primary = _assistant_primary_evidence(assistant_message_dict)
        assistant_model_failure = _assistant_message_model_failure(assistant_message_dict)
        status = "PASS"
        reasons: list[str] = []
        if bool(identity_checks.get("require_pack_primary")) and not pack_primary:
            status = "FAIL"
            reasons.append("missing_pack_primary_evidence")
        if bool(identity_checks.get("require_hit_reader_sync")):
            if not reader_primary:
                status = "FAIL"
                reasons.append("missing_reader_primary_evidence")
            elif not pack_primary:
                status = "FAIL"
                reasons.append("missing_pack_primary_for_reader_sync")
            else:
                mismatches = _surfaces_align(pack_primary, reader_primary)
                if mismatches:
                    status = "FAIL"
                    reasons.append(f"pack_reader_primary_mismatch:{','.join(mismatches)}")
        if bool(identity_checks.get("require_assistant_primary")):
            if assistant_model_failure and not assistant_primary:
                pass
            elif not assistant_primary:
                status = "FAIL"
                reasons.append("missing_assistant_primary_evidence")
            elif not pack_primary:
                status = "FAIL"
                reasons.append("missing_pack_primary_for_assistant_sync")
            else:
                mismatches = _surfaces_align(pack_primary, assistant_primary)
                if mismatches:
                    status = "FAIL"
                    reasons.append(f"pack_assistant_primary_mismatch:{','.join(mismatches)}")
        gate_results["evidence_identity"] = {
            "status": status,
            "pack_primary": pack_primary,
            "hit_primary": hit_primary,
            "reader_primary": reader_primary,
            "assistant_primary": assistant_primary,
            "assistant_model_failure": assistant_model_failure,
            "reasons": reasons,
        }
        failures.extend(reasons)

    consistency_checks = checks.get("consistency_metrics") if isinstance(checks.get("consistency_metrics"), dict) else {}
    if consistency_checks:
        top = _top_hit(refs_pack)
        pack_primary = _pack_primary_evidence(refs_pack)
        reader_primary = _hit_reader_primary_evidence(top)
        summary_heading = _metric_summary_heading_consistent(top_hit=top, pack_primary=pack_primary)
        heading_reader = _metric_heading_reader_open_consistent(top_hit=top, reader_primary=reader_primary)
        fast_full = _metric_fast_to_full_primary_block_same(
            first_pack=dict(first_refs_pack or refs_pack),
            final_pack=dict(final_refs_pack or refs_pack),
        )
        status = "PASS"
        reasons: list[str] = []

        require_summary_heading = bool(consistency_checks.get("require_summary_heading_consistent"))
        if require_summary_heading:
            if summary_heading.get("value") is not True:
                status = "FAIL"
                reasons.append(
                    "summary_heading_consistency_unavailable"
                    if summary_heading.get("value") is None
                    else "summary_heading_inconsistent"
                )

        require_heading_reader = bool(consistency_checks.get("require_heading_reader_open_consistent"))
        if require_heading_reader:
            if heading_reader.get("value") is not True:
                status = "FAIL"
                reasons.append(
                    "heading_reader_open_consistency_unavailable"
                    if heading_reader.get("value") is None
                    else "heading_reader_open_inconsistent"
                )

        require_fast_full = bool(consistency_checks.get("require_fast_to_full_primary_block_same"))
        if require_fast_full:
            if fast_full.get("value") is not True:
                status = "FAIL"
                reasons.append(
                    "fast_to_full_primary_block_same_unavailable"
                    if fast_full.get("value") is None
                    else "fast_to_full_primary_block_drift"
                )

        gate_results["consistency_metrics"] = {
            "status": status,
            "summary_heading_consistent": summary_heading.get("value"),
            "heading_reader_open_consistent": heading_reader.get("value"),
            "fast_to_full_primary_block_same": fast_full.get("value"),
            "summary_heading": summary_heading,
            "heading_reader_open": heading_reader,
            "fast_to_full": fast_full,
            "reasons": reasons,
        }
        failures.extend(reasons)

    return {
        "status": "PASS" if not failures else "FAIL",
        "gate_results": gate_results,
        "failures": failures,
    }


def evaluate_suite(*, base_url: str, suite: dict[str, Any], timeout_s: float = 120.0) -> dict[str, Any]:
    cases = [case for case in list(suite.get("cases") or []) if isinstance(case, dict)]
    default_generate = dict(DEFAULT_GENERATE_BODY)
    default_generate.update(dict(suite.get("generate_body") or {}))
    results: list[dict[str, Any]] = []
    for case in cases:
        conv_title = str(case.get("title") or case.get("id") or "refs-regression").strip()
        mode = str(case.get("mode") or "normal").strip().lower() or "normal"
        conv_body = {
            "title": conv_title,
            "mode": mode,
            "bound_source_path": str(case.get("bound_source_path") or "").strip(),
            "bound_source_name": str(case.get("bound_source_name") or "").strip(),
            "bound_source_ready": bool(case.get("bound_source_path")),
        }
        conv = _post_json(base_url, "/api/conversations", conv_body, timeout_s=timeout_s)
        conv_id = str(conv.get("id") or "").strip()
        gen_body = dict(default_generate)
        gen_body.update(dict(case.get("generate_body") or {}))
        gen_body["conv_id"] = conv_id
        gen_body["prompt"] = str(case.get("prompt") or "").strip()
        started = _post_json(base_url, "/api/generate", gen_body, timeout_s=timeout_s)
        session_id = str(started.get("session_id") or "").strip()
        user_msg_id = int(started.get("user_msg_id") or 0)
        final = _stream_done(base_url, session_id, timeout_s=timeout_s)
        refs_observation = _observe_refs_pack_lifecycle(
            base_url=base_url,
            conv_id=conv_id,
            user_msg_id=user_msg_id,
            timeout_s=min(timeout_s, DEFAULT_REFS_WAIT_TIMEOUT_CAP_S),
        )
        refs_pack = dict(refs_observation.get("final_pack") or {})
        first_refs_pack = dict(refs_observation.get("first_stable_pack") or refs_pack)
        messages = _messages_for_conversation(
            base_url=base_url,
            conv_id=conv_id,
            timeout_s=timeout_s,
        )
        assistant_message = _assistant_message(messages)
        evaluated = _evaluate_case(
            case,
            refs_pack=refs_pack,
            assistant_message=assistant_message,
            first_refs_pack=first_refs_pack,
            final_refs_pack=refs_pack,
        )
        results.append(
            {
                "id": str(case.get("id") or ""),
                "title": str(case.get("title") or ""),
                "mode": mode,
                "prompt": str(case.get("prompt") or ""),
                "status": evaluated["status"],
                "answer_preview": str(final.get("answer") or "")[:240],
                "refs_first_pack": first_refs_pack,
                "refs_pack": refs_pack,
                "assistant_message_id": int(assistant_message.get("id") or 0) if isinstance(assistant_message, dict) else 0,
                "gate_results": evaluated["gate_results"],
                "failures": evaluated["failures"],
                "conv_id": conv_id,
                "user_msg_id": user_msg_id,
            }
        )

    fail_count = sum(1 for item in results if str(item.get("status") or "") != "PASS")
    return {
        "suite_id": str(suite.get("suite_id") or ""),
        "title": str(suite.get("title") or ""),
        "case_count": len(results),
        "fail_count": fail_count,
        "overall_status": "PASS" if fail_count == 0 else "FAIL",
        "results": results,
    }


def evaluate_suite_inprocess(*, suite: dict[str, Any], timeout_s: float = 120.0) -> dict[str, Any]:
    from fastapi.testclient import TestClient
    from api import deps as api_deps
    from api.main import app

    cases = [case for case in list(suite.get("cases") or []) if isinstance(case, dict)]
    default_generate = dict(DEFAULT_GENERATE_BODY)
    default_generate.update(dict(suite.get("generate_body") or {}))
    results: list[dict[str, Any]] = []
    prev_chat_db = os.environ.get("KB_CHAT_DB")
    with TemporaryDirectory(prefix="refs-bench-") as tmpdir:
        os.environ["KB_CHAT_DB"] = str((Path(tmpdir) / "chat.sqlite3").resolve())
        api_deps.get_settings.cache_clear()
        api_deps.get_chat_store.cache_clear()
        try:
            with TestClient(app) as client:
                for case in cases:
                    conv_title = str(case.get("title") or case.get("id") or "refs-regression").strip()
                    mode = str(case.get("mode") or "normal").strip().lower() or "normal"
                    conv_body = {
                        "title": conv_title,
                        "mode": mode,
                        "bound_source_path": str(case.get("bound_source_path") or "").strip(),
                        "bound_source_name": str(case.get("bound_source_name") or "").strip(),
                        "bound_source_ready": bool(case.get("bound_source_path")),
                    }
                    conv = _post_json_client(client, "/api/conversations", conv_body)
                    conv_id = str(conv.get("id") or "").strip()
                    gen_body = dict(default_generate)
                    gen_body.update(dict(case.get("generate_body") or {}))
                    gen_body["conv_id"] = conv_id
                    gen_body["prompt"] = str(case.get("prompt") or "").strip()
                    started = _post_json_client(client, "/api/generate", gen_body)
                    session_id = str(started.get("session_id") or "").strip()
                    user_msg_id = int(started.get("user_msg_id") or 0)
                    final = _stream_done_client(client, session_id, timeout_s=timeout_s)
                    refs_observation = _observe_refs_pack_lifecycle_client(
                        client=client,
                        conv_id=conv_id,
                        user_msg_id=user_msg_id,
                        timeout_s=min(timeout_s, DEFAULT_REFS_WAIT_TIMEOUT_CAP_S),
                    )
                    refs_pack = dict(refs_observation.get("final_pack") or {})
                    first_refs_pack = dict(refs_observation.get("first_stable_pack") or refs_pack)
                    messages = _messages_for_conversation_client(client=client, conv_id=conv_id)
                    assistant_message = _assistant_message(messages)
                    evaluated = _evaluate_case(
                        case,
                        refs_pack=refs_pack,
                        assistant_message=assistant_message,
                        first_refs_pack=first_refs_pack,
                        final_refs_pack=refs_pack,
                    )
                    results.append(
                        {
                            "id": str(case.get("id") or ""),
                            "title": str(case.get("title") or ""),
                            "mode": mode,
                            "prompt": str(case.get("prompt") or ""),
                            "status": evaluated["status"],
                            "answer_preview": str(final.get("answer") or "")[:240],
                            "refs_first_pack": first_refs_pack,
                            "refs_pack": refs_pack,
                            "assistant_message_id": int(assistant_message.get("id") or 0) if isinstance(assistant_message, dict) else 0,
                            "gate_results": evaluated["gate_results"],
                            "failures": evaluated["failures"],
                            "conv_id": conv_id,
                            "user_msg_id": user_msg_id,
                        }
                    )
        finally:
            if prev_chat_db is None:
                os.environ.pop("KB_CHAT_DB", None)
            else:
                os.environ["KB_CHAT_DB"] = prev_chat_db
            api_deps.get_settings.cache_clear()
            api_deps.get_chat_store.cache_clear()

    fail_count = sum(1 for item in results if str(item.get("status") or "") != "PASS")
    return {
        "suite_id": str(suite.get("suite_id") or ""),
        "title": str(suite.get("title") or ""),
        "case_count": len(results),
        "fail_count": fail_count,
        "overall_status": "PASS" if fail_count == 0 else "FAIL",
        "results": results,
    }


def _write_outputs(summary: dict[str, Any], *, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {str(summary.get('title') or summary.get('suite_id') or 'Reference Locate Benchmark')}",
        "",
        f"- suite_id: `{str(summary.get('suite_id') or '')}`",
        f"- case_count: `{int(summary.get('case_count') or 0)}`",
        f"- fail_count: `{int(summary.get('fail_count') or 0)}`",
        f"- overall_status: `{str(summary.get('overall_status') or '')}`",
        "",
    ]
    for item in list(summary.get("results") or []):
        lines.extend(
            [
                f"## {str(item.get('id') or '')} - {str(item.get('status') or '')}",
                "",
                f"- prompt: {str(item.get('prompt') or '')}",
                f"- failures: {', '.join(list(item.get('failures') or [])) or '(none)'}",
                f"- answer_preview: {str(item.get('answer_preview') or '')}",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path, report_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run reference locate quality benchmark against the local API.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument(
        "--manifest",
        default="tools/manual_regression/manifests/reference_locate_quality_v1.json",
    )
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--in-process", action="store_true")
    ap.add_argument("--case-id", action="append", default=[], help="Run only selected case ids; may be repeated or comma-separated.")
    args = ap.parse_args(argv)

    suite = load_suite(str(args.manifest))
    suite = _filter_suite_cases(suite, _parse_case_filter(list(args.case_id or [])))
    if bool(args.in_process):
        summary = evaluate_suite_inprocess(
            suite=suite,
            timeout_s=float(args.timeout_s),
        )
    else:
        summary = evaluate_suite(
            base_url=str(args.base_url).strip().rstrip("/"),
            suite=suite,
            timeout_s=float(args.timeout_s),
        )
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_arg = str(args.out_dir or "").strip()
    out_dir = Path(out_dir_arg).expanduser() if out_dir_arg else None
    if out_dir is None:
        out_dir = _repo_root() / "tmp" / "reference_locate_benchmark" / stamp
    summary_path, report_path = _write_outputs(summary, out_dir=out_dir)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"overall_status={summary.get('overall_status')} fail_count={summary.get('fail_count')} case_count={summary.get('case_count')}")
    return 0 if str(summary.get("overall_status") or "") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
