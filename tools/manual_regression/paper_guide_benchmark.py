from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import requests

# NOTE:
# - This file defines a *benchmark manifest* (golden test cases) used to evaluate paper-guide behavior.
# - The per-case checks like `expected_ref_nums` are *assertions* on the model/runtime output for a specific
#   paper fixture (so we can detect hallucinated/mismatched in-paper citations and broken locate/jump).
# - These values are not used by the runtime to generate answers; they exist only for regression testing.

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_PREF_PATCH = {
    "answer_contract_v1": True,
    "answer_depth_auto": True,
    "answer_mode_hint": "reading",
    "answer_output_mode": "reading_guide",
}
DEFAULT_GENERATE_BODY = {
    "top_k": 6,
    "deep_read": True,
    "max_tokens": 2600,
    "temperature": 0.15,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_file(rel_path: str) -> str:
    return str((_repo_root() / rel_path).resolve())


def _builtin_suites() -> dict[str, dict[str, Any]]:
    lsa_rel = (
        "db/LSA-2026-Interferometric Image Scanning...lateral resolution inside live cells/"
        "LSA-2026-Interferometric Image Scanning...lateral resolution inside live cells.en.md"
    )
    nat2019_rel = (
        "db/NatPhoton-2019-Principles and prospects for single-pixel imaging/"
        "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
    )
    lsa_name = "LSA-2026-Interferometric Image Scanning Microscopy.pdf"
    nat2019_name = "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf"
    return {
        "cross_paper_v1": {
            "suite_id": "cross_paper_v1",
            "title": "Paper Guide Cross-Paper Benchmark",
            "prefs_patch": dict(DEFAULT_PREF_PATCH),
            "generate_body": dict(DEFAULT_GENERATE_BODY),
            "default_checks": {
                "structured_markers": {
                    "forbid_raw_structured_cites": True,
                    "forbid_raw_support_markers": True,
                },
            },
            "papers": [
                {
                    "id": "lsa_2026",
                    "title": "Interferometric Image Scanning Microscopy for label-free imaging at 120 nm lateral resolution inside live cells",
                    "source_path": _repo_file(lsa_rel),
                    "source_name": lsa_name,
                    "cases": [
                        {
                            "id": "LSA_ABSTRACT",
                            "title": "Abstract translation",
                            "prompt_family": "abstract",
                            "tags": ["abstract", "cross_paper", "core"],
                            "prompt": "Give the abstract text and translate it into Chinese.",
                            "checks": {
                                "answer": {
                                    "contains_all": ["Light microscopy remains indispensable"],
                                    "not_contains_any": [
                                        "Conclusion:",
                                        "Evidence:",
                                        "Next Steps:",
                                        "Interferometric Image Scanning Microscopy for label-free imaging",
                                    ],
                                },
                                "quality": {"minimum_ok_required": True},
                            },
                        },
                        {
                            "id": "LSA_DOC_MAP",
                            "title": "Doc map / outline (verbatim anchors)",
                            "prompt_family": "overview",
                            "tags": ["doc_map", "cross_paper"],
                            "prompt": "先把整篇 Markdown 按章节给我一个文档地图：每个章节用一条原文句子概括，方便我查找，并给出可定位跳转的支持证据。",
                            "checks": {
                                "answer": {
                                    "contains_any": ["Doc map (verbatim anchors by section):"],
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["Introduction", "Methods", "Results", "Discussion"],
                                    "claim_types": ["doc_map"],
                                },
                            },
                        },
                        {
                            "id": "LSA_BEGINNER_RVT_APR_ROLE",
                            "title": "RVT/APR beginner role explanation",
                            "prompt_family": "overview",
                            "tags": ["overview", "beginner", "cross_paper", "core"],
                            "prompt": "I am new to this paper. What are RVT and APR doing here, in simple terms?",
                            "checks": {
                                "answer": {
                                    "contains_all": ["RVT", "APR"],
                                    "contains_any": [
                                        "radial-symmetry map",
                                        "phase-correlation",
                                        "shift vectors",
                                        "registration",
                                    ],
                                    "not_contains_any": [
                                        "not stated",
                                        "does not specify",
                                        "no further explanation",
                                        "only mention",
                                    ],
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["Data analysis", "RVT", "APR"],
                                    "anchor_contains_any": ["RVT", "phase correlation", "shift vectors"],
                                    "claim_types": ["method_detail"],
                                },
                            },
                        },
                        {
                            "id": "LSA_CITATION_RVT",
                            "title": "RVT citation lookup",
                            "prompt_family": "citation_lookup",
                            "tags": ["citation_lookup", "cross_paper", "core"],
                            "prompt": "Which prior work do the authors cite for the RVT step, and where is that stated exactly?",
                            "checks": {
                                "answer": {
                                    "contains_any": ["RVT", "radial variance transform"],
                                },
                                "citation": {
                                    "required": True,
                                    "expected_ref_nums": [34],
                                    "allow_only_expected": True,
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["RVT", "Methods", "Adaptive pixel-reassignment"],
                                    "anchor_contains_any": ["radial variance transform", "RVT"],
                                },
                            },
                        },
                        {
                            "id": "LSA_CITATION_RVT_ZH",
                            "title": "RVT citation lookup (Chinese)",
                            "prompt_family": "citation_lookup",
                            "tags": ["citation_lookup", "zh", "cross_paper"],
                            "prompt": "RVT 这一步作者引用了哪篇 prior work？原文哪里明确写到的？请给出可定位的原文支持。",
                            "checks": {
                                "answer": {
                                    "contains_any": ["RVT", "radial variance transform"],
                                },
                                "citation": {
                                    "required": True,
                                    "expected_ref_nums": [34],
                                    "allow_only_expected": True,
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["RVT", "Methods", "Adaptive pixel-reassignment"],
                                    "anchor_contains_any": ["radial variance transform", "RVT"],
                                },
                            },
                        },
                        {
                            "id": "LSA_METHOD_REAPPLY",
                            "title": "APR exact support",
                            "prompt_family": "method",
                            "tags": ["method_exact_support", "cross_paper", "core"],
                            "prompt": "In the APR pipeline, where do the authors say the shift vectors are re-applied to the original iISM dataset? Point me to the exact supporting part of the paper.",
                            "checks": {
                                "answer": {
                                    "contains_any": [
                                        "original iISM dataset",
                                        "applied back to the original iISM dataset",
                                    ],
                                    "not_contains_any": ["not stated"],
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["Adaptive pixel-reassignment", "Methods", "APR"],
                                    "anchor_contains_any": ["original iISM dataset", "shift vectors"],
                                    "claim_types": ["method_detail"],
                                },
                            },
                        },
                        {
                            "id": "LSA_METHOD_REAPPLY_ZH",
                            "title": "APR exact support (Chinese)",
                            "prompt_family": "method",
                            "tags": ["method_exact_support", "zh", "cross_paper"],
                            "prompt": "在 APR 流水线里，作者在原文哪里说 shift vectors 被重新 applied back 到 original iISM dataset？请指向 exact supporting part（原句）并可定位跳转。",
                            "checks": {
                                "answer": {
                                    "contains_any": [
                                        "original iISM dataset",
                                        "applied back to the original iISM dataset",
                                    ],
                                    "not_contains_any": ["not stated"],
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["Adaptive pixel-reassignment", "Methods", "APR"],
                                    "anchor_contains_any": ["original iISM dataset", "shift vectors"],
                                    "claim_types": ["method_detail"],
                                },
                            },
                        },
                        {
                            "id": "LSA_FIGURE_PANEL_FG",
                            "title": "Figure 1 panels f/g",
                            "prompt_family": "figure_walkthrough",
                            "tags": ["figure_panel_locate", "cross_paper", "core"],
                            "prompt": "For Figure 1 panels (f) and (g), what exactly does each panel show? Point me to the exact supporting part of the paper.",
                            "checks": {
                                "answer": {
                                    "contains_any": ["APR", "adaptive pixel-reassignment", "line profiles"],
                                },
                                "figure_panel": {
                                    "required": True,
                                    "panel_letters": ["f", "g"],
                                    "heading_includes_any": ["Figure 1"],
                                    "anchor_contains_any": ["APR", "line profiles", "Resulting iPSF"],
                                    "claim_types": ["figure_panel"],
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["Figure 1"],
                                    "anchor_contains_any": ["APR", "line profiles", "Resulting iPSF"],
                                    "claim_types": ["figure_panel"],
                                },
                                "jump": {"required": True},
                            },
                        },
                        {
                            "id": "LSA_DISCUSSION_ONLY",
                            "title": "Discussion-only future directions",
                            "prompt_family": "discussion_only",
                            "tags": ["discussion_only", "cross_paper", "core"],
                            "prompt": "From the Discussion section only, what future directions or extensions do the authors suggest for iISM?",
                            "checks": {
                                "answer": {
                                    "contains_any": [
                                        "SPAD array",
                                        "parallelized detection",
                                        "single-molecule fluorescence ISM",
                                        "commercial confocal fluorescence ISM systems",
                                    ],
                                },
                                "section_target": {
                                    "required": True,
                                    "exclusive": True,
                                    "heading_includes_any": ["Discussion"],
                                    "heading_excludes_any": [
                                        "Introduction",
                                        "Results",
                                        "Materials",
                                        "Methods",
                                        "References",
                                    ],
                                },
                            },
                        },
                    ],
                },
                {
                    "id": "natphoton_2019_spi",
                    "title": "Principles and prospects for single-pixel imaging",
                    "source_path": _repo_file(nat2019_rel),
                    "source_name": nat2019_name,
                    "cases": [
                        {
                            "id": "NP2019_ABSTRACT",
                            "title": "Abstract translation",
                            "prompt_family": "abstract",
                            "tags": ["abstract", "cross_paper"],
                            "prompt": "Give the abstract text and translate it into Chinese.",
                            "checks": {
                                "answer": {
                                    "contains_all": ["Modern digital cameras employ silicon focal plane array"],
                                    "not_contains_any": [
                                        "Conclusion:",
                                        "Evidence:",
                                        "Next Steps:",
                                        "Principles and prospects for single-pixel imaging",
                                    ],
                                },
                                "quality": {"minimum_ok_required": True},
                            },
                        },
                        {
                            "id": "NP2019_BEGINNER_OVERVIEW_APPLICATIONS",
                            "title": "Beginner overview: why SPI and which applications",
                            "prompt_family": "overview",
                            "tags": ["overview", "beginner", "cross_paper", "core"],
                            "prompt": "I am just getting started. Why is single-pixel imaging interesting, and what kinds of applications does this review emphasize?",
                            "checks": {
                                "answer": {
                                    "contains_any": [
                                        "cost per megapixel",
                                        "infrared",
                                        "microscopy",
                                        "remote sensing",
                                        "quantum state tomography",
                                    ],
                                    "not_contains_any": ["not stated", "does not specify", "cannot be determined"],
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["Abstract", "Applications and future potential", "Acquisition and image reconstruction strategies"],
                                    "anchor_contains_any": ["cost per megapixel", "infrared", "remote sensing", "quantum state tomography"],
                                },
                            },
                        },
                        {
                            "id": "NP2019_STRENGTH_LIMITS_TRADEOFF",
                            "title": "Trade-off in single-pixel imaging",
                            "prompt_family": "strength_limits",
                            "tags": ["strength_limits", "section_targeted", "cross_paper", "core"],
                            "prompt": "In the 'How a single-pixel camera works' section only, what trade-off do the authors describe between the advantages of single-pixel imaging and the detector dynamic range?",
                            "checks": {
                                "answer": {
                                    "contains_any": ["trade-off", "dynamic range", "quantization electronics", "mean square error"],
                                    "not_contains_any": ["not stated", "does not specify", "cannot be determined"],
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["How a single-pixel camera works"],
                                    "anchor_contains_any": ["trade-off", "dynamic range", "quantization electronics", "mean square error"],
                                },
                                "section_target": {
                                    "required": True,
                                    "exclusive": True,
                                    "heading_includes_any": ["How a single-pixel camera works"],
                                    "heading_excludes_any": [
                                        "Camera architecture",
                                        "Applications and future potential",
                                        "References",
                                    ],
                                },
                            },
                        },
                        {
                            "id": "NP2019_BOX1_ONLY",
                            "title": "Box 1 exact support",
                            "prompt_family": "box_only",
                            "tags": ["section_targeted", "box_only", "cross_paper"],
                            "prompt": "From Box 1 only, what condition do the authors give for reconstructing the image in the transform domain? Point me to the exact supporting part.",
                            "checks": {
                                "answer": {
                                    "contains_any": ["K log(N/K)", "M >= O(K log(N/K))", "M ≥ O(K log(N/K))"],
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["Box 1"],
                                    "anchor_contains_any": ["K log(N/K)", "M >=", "M ≥"],
                                },
                                "section_target": {
                                    "required": True,
                                    "exclusive": True,
                                    "heading_includes_any": ["Box 1"],
                                    "heading_excludes_any": [
                                        "How a single-pixel camera works",
                                        "Acquisition and image reconstruction strategies",
                                        "Applications and future potential",
                                        "References",
                                    ],
                                },
                            },
                        },
                        {
                            "id": "NP2019_CITATION_DUARTE",
                            "title": "Duarte citation lookup",
                            "prompt_family": "citation_lookup",
                            "tags": ["citation_lookup", "cross_paper"],
                            "prompt": "Which reference do the authors cite for single-pixel imaging via compressive sampling, and where is that stated exactly?",
                            "checks": {
                                "answer": {
                                    "contains_any": ["Duarte", "compressive sampling"],
                                },
                                "citation": {
                                    "required": True,
                                    "expected_ref_nums": [4],
                                    "allow_only_expected": True,
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["Acquisition and image reconstruction strategies"],
                                    "anchor_contains_any": ["Duarte", "compressive sensing", "seminal paper"],
                                },
                            },
                        },
                        {
                            "id": "NP2019_FIGURE3_PANEL_F",
                            "title": "Figure 3 panel f",
                            "prompt_family": "figure_walkthrough",
                            "tags": ["figure_panel_locate", "cross_paper"],
                            "prompt": "For Figure 3 panel (f), what exactly does that panel correspond to? Point me to the exact supporting caption clause.",
                            "checks": {
                                "answer": {
                                    "contains_any": ["methane imaging using SPC", "SPC", "single-pixel camera"],
                                },
                                "figure_panel": {
                                    "required": True,
                                    "panel_letters": ["f"],
                                    "heading_includes_any": ["Figure 3"],
                                    "anchor_contains_any": ["methane imaging using SPC", "SPC"],
                                    "claim_types": ["figure_panel"],
                                },
                                "locate": {
                                    "required": True,
                                    "exact": True,
                                    "heading_includes_any": ["Figure 3"],
                                    "anchor_contains_any": ["methane imaging using SPC", "SPC"],
                                    "claim_types": ["figure_panel"],
                                },
                                "jump": {"required": True},
                            },
                        },
                    ],
                },
            ],
        }
    }


def _now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_json(resp: requests.Response) -> dict[str, Any]:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_json(base_url: str, path: str, *, timeout_s: float) -> dict[str, Any]:
    resp = requests.get(f"{base_url}{path}", timeout=timeout_s)
    resp.raise_for_status()
    return _safe_json(resp)


def _patch_settings(base_url: str, patch: dict[str, Any], *, timeout_s: float) -> None:
    if not patch:
        return
    resp = requests.patch(f"{base_url}/api/settings", json=patch, timeout=timeout_s)
    resp.raise_for_status()


def _create_conversation(
    base_url: str,
    *,
    source_path: str,
    source_name: str,
    title: str,
    timeout_s: float,
) -> str:
    source = str(source_path or "").strip()
    if not source:
        raise RuntimeError("empty source_path")
    body = {
        "title": title,
        "mode": "paper_guide",
        "bound_source_path": source,
        "bound_source_name": str(source_name or Path(source).name).strip() or Path(source).name,
        "bound_source_ready": True,
    }
    resp = requests.post(f"{base_url}/api/conversations", json=body, timeout=timeout_s)
    resp.raise_for_status()
    payload = _safe_json(resp)
    conv_id = str(payload.get("id") or "").strip()
    if not conv_id:
        raise RuntimeError("failed to create conversation")
    return conv_id


def _start_generation(
    base_url: str,
    *,
    conv_id: str,
    prompt: str,
    timeout_s: float,
    generate_body: dict[str, Any],
) -> str:
    body = dict(generate_body or {})
    body.update({"conv_id": conv_id, "prompt": prompt})
    resp = requests.post(f"{base_url}/api/generate", json=body, timeout=timeout_s)
    resp.raise_for_status()
    payload = _safe_json(resp)
    session_id = str(payload.get("session_id") or "").strip()
    if not session_id:
        raise RuntimeError("failed to start generation")
    return session_id


def _stream_generation(base_url: str, *, session_id: str, timeout_s: float) -> dict[str, Any]:
    final_payload: dict[str, Any] = {}
    deadline = time.time() + timeout_s
    with requests.get(f"{base_url}/api/generate/{session_id}/stream", stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if time.time() >= deadline:
                raise RuntimeError("generation stream timeout")
            if not raw:
                continue
            line = str(raw).strip()
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
                    return final_payload
    return final_payload


def _list_messages(base_url: str, *, conv_id: str, timeout_s: float) -> list[dict[str, Any]]:
    resp = requests.get(f"{base_url}/api/conversations/{conv_id}/messages", timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, list) else []


def _max_message_id(messages: list[dict[str, Any]]) -> int:
    best = 0
    for msg in messages:
        try:
            mid = int((msg or {}).get("id") or 0)
        except Exception:
            mid = 0
        if mid > best:
            best = mid
    return best


def _is_live_task_text(text: str) -> bool:
    return str(text or "").strip().startswith("__KB_LIVE_TASK__:")


def _last_assistant_message(
    messages: list[dict[str, Any]],
    *,
    min_message_id: int = 0,
    allow_live: bool = False,
) -> dict[str, Any]:
    fallback: dict[str, Any] = {}
    for msg in reversed(messages):
        if str((msg or {}).get("role") or "").strip() != "assistant":
            continue
        try:
            mid = int((msg or {}).get("id") or 0)
        except Exception:
            mid = 0
        if mid <= int(min_message_id or 0):
            continue
        if not fallback:
            fallback = msg
        if allow_live or (not _is_live_task_text(str((msg or {}).get("content") or ""))):
            return msg
    return fallback


def _answer_text(message: dict[str, Any], payload: dict[str, Any]) -> str:
    for cand in (
        message.get("content"),
        payload.get("answer"),
        payload.get("partial"),
    ):
        if isinstance(cand, str) and cand.strip():
            text = cand.strip()
            if text.startswith("__KB_LIVE_TASK__:"):
                continue
            return text
    return ""


def _fetch_answer_bundle(
    base_url: str,
    *,
    conv_id: str,
    payload: dict[str, Any],
    timeout_s: float,
    min_message_id: int = 0,
    attempts: int = 20,
    delay_s: float = 1.0,
) -> tuple[dict[str, Any], str]:
    last_message: dict[str, Any] = {}
    answer_text = ""
    tries = max(1, int(attempts))
    for idx in range(tries):
        messages = _list_messages(base_url, conv_id=conv_id, timeout_s=timeout_s)
        last_message = _last_assistant_message(messages, min_message_id=min_message_id)
        if not last_message:
            last_message = _last_assistant_message(messages, min_message_id=min_message_id, allow_live=True)
        answer_text = _answer_text(last_message, payload)
        if answer_text.strip():
            return last_message, answer_text
        if idx + 1 < tries:
            time.sleep(max(0.1, float(delay_s)))
    return last_message, answer_text


def _render_cache(message: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    for cand in (message.get("render_cache"), payload.get("render_cache")):
        if isinstance(cand, dict):
            return cand
    return {}


def _provenance(message: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    for cand in (message.get("provenance"), payload.get("provenance")):
        if isinstance(cand, dict):
            return cand
    return {}


def _message_render_packet(message: dict[str, Any]) -> dict[str, Any]:
    meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
    contracts = meta.get("paper_guide_contracts") if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    for cand in (
        contracts.get("render_packet"),
        message.get("render_packet"),
    ):
        if isinstance(cand, dict):
            return cand
    return {}


def _contains_any(text: str, needles: list[str]) -> bool:
    hay = str(text or "").lower()
    return any(str(needle or "").lower() in hay for needle in list(needles or []))


def _contains_all(text: str, needles: list[str]) -> bool:
    hay = str(text or "").lower()
    return all(str(needle or "").lower() in hay for needle in list(needles or []))


def _regex_count(text: str, patterns: list[str]) -> int:
    hay = str(text or "")
    return sum(1 for pattern in patterns if re.search(pattern, hay, flags=re.IGNORECASE))


def _has_jump_asset(answer_text: str, provenance: dict[str, Any]) -> bool:
    if "/api/references/asset?" in str(answer_text or ""):
        return True
    source_path = str(provenance.get("source_path") or "").strip()
    segments = provenance.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            for value in (seg.get("display_markdown"), seg.get("raw_markdown"), seg.get("text")):
                if "/api/references/asset?" in str(value or ""):
                    return True
            if not source_path:
                continue
            if str(seg.get("evidence_mode") or "").strip().lower() != "direct":
                continue
            if str(seg.get("locate_policy") or "").strip().lower() == "hidden":
                continue
            block_ids = [
                str(seg.get("primary_block_id") or "").strip(),
                *[
                    str(item or "").strip()
                    for item in list(seg.get("support_block_ids") or [])
                ],
                *[
                    str(item or "").strip()
                    for item in list(seg.get("evidence_block_ids") or [])
                ],
            ]
            has_block = any(block_ids)
            has_anchor = bool(
                str(seg.get("anchor_text") or "").strip()
                or str(seg.get("support_locate_anchor") or "").strip()
                or str(seg.get("primary_heading_path") or "").strip()
            )
            if has_block and has_anchor:
                return True
    return False


def _mentions_figure_panel(answer_text: str, panel: str) -> bool:
    token = str(panel or "").strip().lower()
    if not token:
        return False
    return bool(
        re.search(
            rf"(?:panel[s]?\s*\({re.escape(token)}\)|\({re.escape(token)}\)|\b{re.escape(token)}\s*[/,&]\s*[a-z]\b|\b[a-z]\s*[/,&]\s*{re.escape(token)}\b|panel\s*\({re.escape(token)},\s*[a-z]\)|panel\s*\([a-z],\s*{re.escape(token)}\))",
            str(answer_text or ""),
            flags=re.IGNORECASE,
        )
    )


def _rendered_citation_count(message: dict[str, Any]) -> int:
    rendered = str(message.get("rendered_body") or message.get("rendered_content") or "")
    if not rendered:
        return 0
    anchors = set(re.findall(r"#kb-cite-[A-Za-z0-9_-]+", rendered))
    return len(anchors)


def _raw_structured_citation_count(answer_text: str) -> int:
    text = str(answer_text or "")
    if not text:
        return 0
    markers = set(re.findall(r"\[\[CITE:[^:\]]+:\d+\]\]", text, flags=re.IGNORECASE))
    return len(markers)


def _raw_support_marker_count(answer_text: str) -> int:
    text = str(answer_text or "")
    if not text:
        return 0
    markers = set(re.findall(r"\[\[SUPPORT:[^\]]+\]\]", text, flags=re.IGNORECASE))
    return len(markers)


def _minimum_ok_label(answer_quality: dict[str, Any] | None) -> str:
    if not isinstance(answer_quality, dict):
        return "n/a"
    if "minimum_ok" not in answer_quality:
        return "n/a"
    return str(bool(answer_quality.get("minimum_ok"))).lower()


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _message_cite_details(message: dict[str, Any]) -> list[dict[str, Any]]:
    details = message.get("cite_details")
    if isinstance(details, list) and details:
        return [item for item in details if isinstance(item, dict)]
    packet = _message_render_packet(message)
    details = packet.get("cite_details")
    if isinstance(details, list) and details:
        return [item for item in details if isinstance(item, dict)]
    return []


def _message_cite_ref_nums(message: dict[str, Any]) -> list[int]:
    nums: list[int] = []
    for item in _message_cite_details(message):
        if not isinstance(item, dict):
            continue
        ref_num = _to_int(item.get("ref_num") or item.get("num"))
        if ref_num > 0:
            nums.append(ref_num)
    return nums


def _segment_heading(seg: dict[str, Any]) -> str:
    return (
        str(seg.get("primary_heading_path") or "").strip()
        or str(seg.get("heading_path") or "").strip()
    )


def _segment_anchor(seg: dict[str, Any]) -> str:
    return (
        str(seg.get("support_locate_anchor") or "").strip()
        or str(seg.get("anchor_text") or "").strip()
        or str(seg.get("evidence_quote") or "").strip()
        or str(seg.get("text") or "").strip()
    )


def _segment_claim_type(seg: dict[str, Any]) -> str:
    return (
        str(seg.get("support_slot_claim_type") or "").strip()
        or str(seg.get("claim_type") or "").strip()
    )


def _segment_hit_level(seg: dict[str, Any]) -> str:
    raw = str(seg.get("hit_level") or "").strip().lower()
    if raw in {"exact", "block", "heading", "none"}:
        return raw
    if str(seg.get("primary_block_id") or "").strip():
        if str(seg.get("primary_anchor_id") or "").strip() or str(seg.get("anchor_kind") or "").strip():
            return "exact"
        return "block"
    if _segment_heading(seg):
        return "heading"
    return "none"


def _render_target_heading(target: dict[str, Any]) -> str:
    return str(target.get("headingPath") or "").strip()


def _render_target_anchor(target: dict[str, Any]) -> str:
    return (
        str(target.get("anchorText") or "").strip()
        or str(target.get("evidenceQuote") or "").strip()
        or str(target.get("highlightSnippet") or "").strip()
        or str(target.get("snippet") or "").strip()
    )


def _render_target_snippet(target: dict[str, Any]) -> str:
    return (
        str(target.get("highlightSnippet") or "").strip()
        or str(target.get("snippet") or "").strip()
        or str(target.get("anchorText") or "").strip()
        or str(target.get("evidenceQuote") or "").strip()
    )


def _render_packet_primary_target(render_packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(render_packet, dict):
        return {}, {}
    locate_target = render_packet.get("locate_target") if isinstance(render_packet.get("locate_target"), dict) else {}
    reader_open = render_packet.get("reader_open") if isinstance(render_packet.get("reader_open"), dict) else {}
    if locate_target:
        return dict(locate_target), dict(reader_open)
    nested_locate = reader_open.get("locateTarget") if isinstance(reader_open.get("locateTarget"), dict) else {}
    if nested_locate:
        return dict(nested_locate), dict(reader_open)
    if reader_open:
        return dict(reader_open), dict(reader_open)
    return {}, {}


def _normalized_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _text_token_overlap(lhs: str, rhs: str) -> float:
    lhs_tokens = {token for token in re.findall(r"[a-z0-9]{3,}", str(lhs or "").lower()) if token}
    rhs_tokens = {token for token in re.findall(r"[a-z0-9]{3,}", str(rhs or "").lower()) if token}
    if not lhs_tokens or not rhs_tokens:
        return 0.0
    return float(len(lhs_tokens.intersection(rhs_tokens))) / float(min(len(lhs_tokens), len(rhs_tokens)))


def _looks_like_heading_only_surface(target: dict[str, Any]) -> bool:
    snippet = _normalized_text_key(_render_target_snippet(target))
    if not snippet:
        return True
    anchor_kind = str(target.get("anchorKind") or "").strip().lower()
    if anchor_kind == "equation":
        return False
    if len(snippet) < 18:
        return True
    heading = _render_target_heading(target)
    heading_leaf = str(heading.split("/")[-1] if heading else "").strip()
    heading_key = _normalized_text_key(heading_leaf)
    if heading_key and snippet == heading_key:
        return True
    return False


def _render_target_matches_rules(target: dict[str, Any], rules: dict[str, Any]) -> bool:
    heading = _render_target_heading(target)
    anchor = _render_target_anchor(target)
    snippet = _render_target_snippet(target)
    claim_type = str(target.get("claimType") or "").strip()
    if list(rules.get("heading_includes_any") or []) and not _contains_any(heading, list(rules.get("heading_includes_any") or [])):
        return False
    if list(rules.get("heading_excludes_any") or []) and _contains_any(heading, list(rules.get("heading_excludes_any") or [])):
        return False
    if list(rules.get("anchor_contains_any") or []) and not _contains_any(anchor, list(rules.get("anchor_contains_any") or [])):
        return False
    if list(rules.get("anchor_not_contains_any") or []) and _contains_any(anchor, list(rules.get("anchor_not_contains_any") or [])):
        return False
    if list(rules.get("snippet_contains_any") or []) and not _contains_any(snippet, list(rules.get("snippet_contains_any") or [])):
        return False
    if list(rules.get("snippet_not_contains_any") or []) and _contains_any(snippet, list(rules.get("snippet_not_contains_any") or [])):
        return False
    if list(rules.get("claim_types") or []):
        allowed = {str(item or "").strip().lower() for item in list(rules.get("claim_types") or []) if str(item or "").strip()}
        if str(claim_type or "").strip().lower() not in allowed:
            return False
    if bool(rules.get("exact")) and not str(target.get("blockId") or "").strip():
        return False
    return True


def _render_target_matches_segment(target: dict[str, Any], seg: dict[str, Any]) -> bool:
    target_segment_id = str(target.get("sourceSegmentId") or target.get("segmentId") or "").strip()
    seg_segment_id = str(seg.get("segment_id") or "").strip()
    if target_segment_id and seg_segment_id and target_segment_id == seg_segment_id:
        return True
    target_block = str(target.get("blockId") or "").strip()
    target_anchor_id = str(target.get("anchorId") or "").strip()
    seg_block = str(
        ((seg.get("locate_target") or {}).get("blockId") if isinstance(seg.get("locate_target"), dict) else "")
        or seg.get("primary_block_id")
        or ""
    ).strip()
    seg_anchor_id = str(
        ((seg.get("locate_target") or {}).get("anchorId") if isinstance(seg.get("locate_target"), dict) else "")
        or seg.get("primary_anchor_id")
        or ""
    ).strip()
    if target_block and seg_block and target_block == seg_block:
        if (not target_anchor_id) or (not seg_anchor_id) or target_anchor_id == seg_anchor_id:
            return True
    target_heading = _render_target_heading(target)
    seg_heading = _segment_heading(seg)
    target_anchor = _render_target_anchor(target)
    seg_anchor = _segment_anchor(seg)
    if target_heading and seg_heading and target_heading == seg_heading:
        if _text_token_overlap(target_anchor, seg_anchor) >= 0.45:
            return True
    return False


def _visible_direct_segments(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in list((provenance or {}).get("segments") or []):
        if not isinstance(seg, dict):
            continue
        if str(seg.get("evidence_mode") or "").strip().lower() != "direct":
            continue
        if str(seg.get("locate_policy") or "").strip().lower() == "hidden":
            continue
        out.append(seg)
    return out


def _provenance_hit_level_counts(provenance: dict[str, Any]) -> dict[str, int]:
    counts = {"exact": 0, "block": 0, "heading": 0, "none": 0}
    for seg in _visible_direct_segments(provenance):
        level = _segment_hit_level(seg)
        if level not in counts:
            level = "none"
        counts[level] = counts[level] + 1
    return counts


def _segment_matches(seg: dict[str, Any], rules: dict[str, Any]) -> bool:
    heading = _segment_heading(seg)
    anchor = _segment_anchor(seg)
    claim_type = _segment_claim_type(seg)
    if list(rules.get("heading_includes_any") or []) and not _contains_any(heading, list(rules.get("heading_includes_any") or [])):
        return False
    if list(rules.get("heading_excludes_any") or []) and _contains_any(heading, list(rules.get("heading_excludes_any") or [])):
        return False
    if list(rules.get("anchor_contains_any") or []) and not _contains_any(anchor, list(rules.get("anchor_contains_any") or [])):
        return False
    if list(rules.get("anchor_not_contains_any") or []) and _contains_any(anchor, list(rules.get("anchor_not_contains_any") or [])):
        return False
    if list(rules.get("claim_types") or []):
        allowed = {str(item or "").strip().lower() for item in list(rules.get("claim_types") or []) if str(item or "").strip()}
        if str(claim_type or "").strip().lower() not in allowed:
            return False
    if bool(rules.get("exact")):
        if not str(seg.get("primary_block_id") or "").strip():
            return False
        if not anchor:
            return False
    return True


def _matching_segments(provenance: dict[str, Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    return [seg for seg in _visible_direct_segments(provenance) if _segment_matches(seg, rules)]


def _normalized_case_checks(case: dict[str, Any], suite_defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    defaults = copy.deepcopy(suite_defaults or {})
    checks = copy.deepcopy(case.get("checks") or {})
    for gate, gate_defaults in defaults.items():
        if gate not in checks and isinstance(gate_defaults, dict):
            checks[gate] = dict(gate_defaults)
        elif isinstance(gate_defaults, dict) and isinstance(checks.get(gate), dict):
            merged = dict(gate_defaults)
            merged.update(checks.get(gate) or {})
            checks[gate] = merged
    return checks


def _evaluate_answer_gate(answer_text: str, rules: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if list(rules.get("contains_all") or []) and not _contains_all(answer_text, list(rules.get("contains_all") or [])):
        reasons.append("answer_missing_contains_all")
    if list(rules.get("contains_any") or []) and not _contains_any(answer_text, list(rules.get("contains_any") or [])):
        reasons.append("answer_missing_contains_any")
    if list(rules.get("not_contains_any") or []) and _contains_any(answer_text, list(rules.get("not_contains_any") or [])):
        reasons.append("answer_contains_forbidden_text")
    if list(rules.get("contains_regex_any") or []) and _regex_count(answer_text, list(rules.get("contains_regex_any") or [])) <= 0:
        reasons.append("answer_missing_contains_regex_any")
    if list(rules.get("not_contains_regex_any") or []) and _regex_count(answer_text, list(rules.get("not_contains_regex_any") or [])) > 0:
        reasons.append("answer_contains_forbidden_regex")
    return {"status": "FAIL" if reasons else "PASS", "reasons": reasons}


def _evaluate_structured_marker_gate(answer_text: str, rules: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    raw_count = _raw_structured_citation_count(answer_text)
    raw_support_count = _raw_support_marker_count(answer_text)
    if bool(rules.get("forbid_raw_structured_cites")) and raw_count > 0:
        reasons.append(f"raw_structured_cite_count={raw_count}")
    if bool(rules.get("forbid_raw_support_markers")) and raw_support_count > 0:
        reasons.append(f"raw_support_marker_count={raw_support_count}")
    return {
        "status": "FAIL" if reasons else "PASS",
        "reasons": reasons,
        "raw_structured_cite_count": raw_count,
        "raw_support_marker_count": raw_support_count,
    }


def _evaluate_quality_gate(answer_quality: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if bool(rules.get("minimum_ok_required")) and (not bool((answer_quality or {}).get("minimum_ok"))):
        reasons.append("minimum_ok_false")
    return {"status": "FAIL" if reasons else "PASS", "reasons": reasons}


def _evaluate_citation_gate(
    message: dict[str, Any],
    provenance: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    # Allow plain numeric citations like "[15]" in the rendered answer to satisfy citation checks.
    # This keeps the user-facing output clean while still validating in-paper ref grounding.
    answer_text = str(message.get("content") or "").strip()
    inline_nums = [int(n) for n in re.findall(r"\[(\d{1,4})\]", answer_text) if str(n).isdigit() and int(n) > 0]
    rendered_ref_nums = _message_cite_ref_nums(message)
    resolved_ref_nums = [
        _to_int(seg.get("resolved_ref_num"))
        for seg in list((provenance or {}).get("segments") or [])
        if isinstance(seg, dict) and _to_int(seg.get("resolved_ref_num")) > 0
    ]
    strict_only = bool(rules.get("allow_only_expected"))
    structured_nums = [n for n in (rendered_ref_nums + resolved_ref_nums) if int(n) > 0]
    # If a case is explicitly strict about "only expected", prefer structured signals.
    # But if structured signals are missing entirely, fall back to inline "[n]" tokens.
    if strict_only and structured_nums:
        inline_nums = []
    matched_ref_nums = sorted({num for num in (structured_nums + inline_nums) if num > 0})
    if bool(rules.get("required")) and not matched_ref_nums:
        reasons.append("missing_citation_ref_num")
    expected_ref_nums = sorted({_to_int(num) for num in list(rules.get("expected_ref_nums") or []) if _to_int(num) > 0})
    if expected_ref_nums and not set(matched_ref_nums).intersection(expected_ref_nums):
        reasons.append(f"expected_ref_nums_missing={expected_ref_nums}")
    if bool(rules.get("allow_only_expected")) and expected_ref_nums:
        unexpected = [num for num in matched_ref_nums if num not in expected_ref_nums]
        if unexpected:
            reasons.append(f"unexpected_ref_nums={unexpected}")
    cite_details = _message_cite_details(message)
    details_by_num: dict[int, list[dict[str, Any]]] = {}
    for item in cite_details:
        ref_num = _to_int(item.get("ref_num") or item.get("num"))
        if ref_num <= 0:
            continue
        details_by_num.setdefault(ref_num, []).append(item)
    expected_or_matched = expected_ref_nums or matched_ref_nums
    detail_records = [item for num in expected_or_matched for item in details_by_num.get(num, [])]
    if expected_ref_nums and bool(rules.get("required", True)):
        missing_details = [num for num in expected_ref_nums if num in matched_ref_nums and num not in details_by_num]
        if missing_details:
            reasons.append(f"missing_reference_details={missing_details}")
    detail_text = "\n".join(
        " ".join(
            part
            for part in [
                str(item.get("title") or "").strip(),
                str(item.get("authors") or "").strip(),
                str(item.get("venue") or "").strip(),
                str(item.get("year") or "").strip(),
                str(item.get("raw") or "").strip(),
                str(item.get("cite_fmt") or "").strip(),
            ]
            if part
        ).strip()
        for item in detail_records
    ).strip()
    if expected_ref_nums and bool(rules.get("required", True)) and expected_or_matched and (not detail_text):
        reasons.append("reference_detail_text_missing")
    if list(rules.get("expected_reference_text_contains_any") or []) and not _contains_any(
        detail_text,
        list(rules.get("expected_reference_text_contains_any") or []),
    ):
        reasons.append("expected_reference_text_missing_any")
    if list(rules.get("expected_reference_text_contains_all") or []) and not _contains_all(
        detail_text,
        list(rules.get("expected_reference_text_contains_all") or []),
    ):
        reasons.append("expected_reference_text_missing_all")
    if list(rules.get("reference_text_not_contains_any") or []) and _contains_any(
        detail_text,
        list(rules.get("reference_text_not_contains_any") or []),
    ):
        reasons.append("reference_detail_contains_forbidden_text")
    return {
        "status": "FAIL" if reasons else "PASS",
        "reasons": reasons,
        "matched_ref_nums": matched_ref_nums,
        "detail_ref_nums": sorted(details_by_num.keys()),
        "detail_preview": detail_text[:240],
    }


def _evaluate_locate_like_gate(
    provenance: dict[str, Any],
    render_packet: dict[str, Any],
    rules: dict[str, Any],
    *,
    gate_name: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    matched = _matching_segments(provenance, rules)
    render_target, reader_open = _render_packet_primary_target(render_packet)
    render_target_checked = bool(render_target)
    render_target_ok = False
    if render_target_checked:
        if not _render_target_matches_rules(render_target, rules):
            reasons.append(f"{gate_name}_render_target_mismatch")
        else:
            render_target_ok = True
            if matched and not any(_render_target_matches_segment(render_target, seg) for seg in matched):
                reasons.append(f"{gate_name}_render_target_not_matched_segment")
        if bool(rules.get("exact")):
            if not str(render_target.get("blockId") or "").strip():
                reasons.append(f"{gate_name}_render_target_missing_block_id")
            if not str(render_target.get("anchorId") or "").strip():
                reasons.append(f"{gate_name}_render_target_missing_anchor_id")
            if _looks_like_heading_only_surface(render_target):
                reasons.append(f"{gate_name}_render_target_surface_too_thin")
        if bool(reader_open) and bool(rules.get("exact")) and not bool(reader_open.get("strictLocate")):
            reasons.append(f"{gate_name}_reader_open_not_strict")
    if bool(rules.get("required")) and (not matched) and (not render_target_ok):
        reasons.append(f"{gate_name}_missing_matching_segment")
    if bool(rules.get("exclusive")):
        visible = _visible_direct_segments(provenance)
        non_matching = [seg for seg in visible if not _segment_matches(seg, rules)]
        if non_matching:
            reasons.append(f"{gate_name}_has_non_target_segment")
    return {
        "status": "FAIL" if reasons else "PASS",
        "reasons": reasons,
        "matched_segment_count": len(matched),
        "matched_headings": [_segment_heading(seg) for seg in matched[:3]],
        "matched_anchors": [_segment_anchor(seg)[:240] for seg in matched[:3]],
        "render_target_checked": render_target_checked,
        "render_target_ok": render_target_ok,
        "render_target_heading": _render_target_heading(render_target)[:240] if render_target else "",
        "render_target_anchor": _render_target_anchor(render_target)[:240] if render_target else "",
        "render_target_snippet": _render_target_snippet(render_target)[:240] if render_target else "",
    }


def _evaluate_figure_panel_gate(
    answer_text: str,
    provenance: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    matched = _matching_segments(provenance, rules)
    for panel in list(rules.get("panel_letters") or []):
        if not _mentions_figure_panel(answer_text, str(panel or "")):
            reasons.append(f"answer_missing_panel_{str(panel or '').lower()}")
    if bool(rules.get("required")) and not matched:
        reasons.append("figure_panel_missing_matching_segment")
    return {
        "status": "FAIL" if reasons else "PASS",
        "reasons": reasons,
        "matched_segment_count": len(matched),
        "matched_headings": [_segment_heading(seg) for seg in matched[:3]],
        "matched_anchors": [_segment_anchor(seg)[:240] for seg in matched[:3]],
    }


def _evaluate_jump_gate(
    answer_text: str,
    provenance: dict[str, Any],
    render_packet: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    jump_found = _has_jump_asset(answer_text, provenance)
    if bool(rules.get("required")) and (not jump_found):
        reasons.append("jump_asset_missing")
    render_target, reader_open = _render_packet_primary_target(render_packet)
    if bool(rules.get("required")) and (not render_target):
        reasons.append("jump_render_target_missing")
    elif render_target:
        if list(rules.get("heading_includes_any") or []) or list(rules.get("anchor_contains_any") or []) or bool(rules.get("exact")):
            if not _render_target_matches_rules(render_target, rules):
                reasons.append("jump_render_target_mismatch")
        if bool(rules.get("exact")):
            if not str(render_target.get("blockId") or "").strip():
                reasons.append("jump_render_target_missing_block_id")
            if not str(render_target.get("anchorId") or "").strip():
                reasons.append("jump_render_target_missing_anchor_id")
            if _looks_like_heading_only_surface(render_target):
                reasons.append("jump_render_target_surface_too_thin")
        if matched := _matching_segments(provenance, rules):
            if not any(_render_target_matches_segment(render_target, seg) for seg in matched):
                reasons.append("jump_render_target_not_matched_segment")
        if bool(reader_open) and bool(rules.get("exact")) and not bool(reader_open.get("strictLocate")):
            reasons.append("jump_reader_open_not_strict")
    return {
        "status": "FAIL" if reasons else "PASS",
        "reasons": reasons,
        "jump_found": bool(jump_found),
        "render_target_heading": _render_target_heading(render_target)[:240] if render_target else "",
        "render_target_anchor": _render_target_anchor(render_target)[:240] if render_target else "",
    }


def _evaluate_case(
    case: dict[str, Any],
    *,
    suite_defaults: dict[str, Any] | None,
    answer_text: str,
    answer_quality: dict[str, Any],
    render_cache: dict[str, Any],
    provenance: dict[str, Any],
    message: dict[str, Any],
) -> dict[str, Any]:
    del render_cache
    checks = _normalized_case_checks(case, suite_defaults=suite_defaults)
    answer = str(answer_text or "").strip()
    visible_segments = _visible_direct_segments(provenance)
    render_packet = _message_render_packet(message)
    hit_level_counts = _provenance_hit_level_counts(provenance)
    primary_hit_level = _segment_hit_level(visible_segments[0]) if visible_segments else "none"
    raw_structured_cite_count = _raw_structured_citation_count(answer)
    rendered_cite_count = max(
        int(len(message.get("cite_details") or [])) if isinstance(message.get("cite_details"), list) else 0,
        _rendered_citation_count(message),
    )
    gate_results: dict[str, dict[str, Any]] = {}
    if isinstance(checks.get("answer"), dict):
        gate_results["answer"] = _evaluate_answer_gate(answer, dict(checks.get("answer") or {}))
    if isinstance(checks.get("structured_markers"), dict):
        gate_results["structured_markers"] = _evaluate_structured_marker_gate(answer, dict(checks.get("structured_markers") or {}))
    if isinstance(checks.get("quality"), dict):
        gate_results["quality"] = _evaluate_quality_gate(answer_quality, dict(checks.get("quality") or {}))
    if isinstance(checks.get("citation"), dict):
        gate_results["citation"] = _evaluate_citation_gate(message, provenance, dict(checks.get("citation") or {}))
    if isinstance(checks.get("locate"), dict):
        gate_results["locate"] = _evaluate_locate_like_gate(
            provenance,
            render_packet,
            dict(checks.get("locate") or {}),
            gate_name="locate",
        )
    if isinstance(checks.get("section_target"), dict):
        gate_results["section_target"] = _evaluate_locate_like_gate(
            provenance,
            render_packet,
            dict(checks.get("section_target") or {}),
            gate_name="section_target",
        )
    if isinstance(checks.get("figure_panel"), dict):
        gate_results["figure_panel"] = _evaluate_figure_panel_gate(answer, provenance, dict(checks.get("figure_panel") or {}))
    if isinstance(checks.get("jump"), dict):
        gate_results["jump"] = _evaluate_jump_gate(answer, provenance, render_packet, dict(checks.get("jump") or {}))

    reasons: list[str] = []
    for gate_name, gate in gate_results.items():
        for reason in list(gate.get("reasons") or []):
            reasons.append(f"{gate_name}:{reason}")
    status = "PASS"
    if any(str(gate.get("status") or "") == "FAIL" for gate in gate_results.values()):
        status = "FAIL"

    return {
        "id": str(case.get("id") or "").strip(),
        "title": str(case.get("title") or "").strip(),
        "paper_id": str(case.get("paper_id") or "").strip(),
        "paper_title": str(case.get("paper_title") or "").strip(),
        "prompt_family": str(case.get("prompt_family") or "").strip(),
        "prompt": str(case.get("prompt") or "").strip(),
        "tags": list(case.get("tags") or []),
        "status": status,
        "reasons": reasons,
        "gate_results": gate_results,
        "answer_quality": dict(answer_quality or {}),
        "cite_count": rendered_cite_count,
        "primary_hit_level": primary_hit_level,
        "visible_direct_segment_count": len(visible_segments),
        "hit_level_counts": hit_level_counts,
        "raw_structured_cite_count": raw_structured_cite_count,
        "answer_preview": answer[:900],
    }


def _summary_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(items),
        "pass": sum(1 for item in items if str(item.get("status") or "") == "PASS"),
        "fail": sum(1 for item in items if str(item.get("status") or "") == "FAIL"),
    }


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _summary_hit_levels(items: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"exact": 0, "block": 0, "heading": 0, "none": 0}
    total_segments = 0
    cases_with_direct = 0
    cases_with_exact = 0
    for item in items:
        counts0 = item.get("hit_level_counts")
        if not isinstance(counts0, dict):
            counts0 = {}
        case_segments = 0
        for key in ("exact", "block", "heading", "none"):
            value = max(0, _to_int(counts0.get(key)))
            totals[key] = totals[key] + value
            case_segments += value
        total_segments += case_segments
        if case_segments > 0:
            cases_with_direct += 1
        if _to_int(counts0.get("exact")) > 0:
            cases_with_exact += 1

    return {
        "counts": totals,
        "total_segments": total_segments,
        "cases_with_direct": cases_with_direct,
        "cases_with_exact": cases_with_exact,
        "rates": {
            "exact_share": _rate(totals["exact"], total_segments),
            "exact_or_block_share": _rate(totals["exact"] + totals["block"], total_segments),
            "heading_or_none_share": _rate(totals["heading"] + totals["none"], total_segments),
            "cases_with_direct_share": _rate(cases_with_direct, len(items)),
            "cases_with_exact_share": _rate(cases_with_exact, len(items)),
        },
    }


def _summary_primary_hit_levels(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"exact": 0, "block": 0, "heading": 0, "none": 0}
    for item in items:
        hit = str(item.get("primary_hit_level") or "").strip().lower()
        if hit not in counts:
            hit = "none"
        counts[hit] = int(counts.get(hit) or 0) + 1
    total = sum(counts.values())
    return {
        "counts": counts,
        "total_cases": total,
        "rates": {
            "exact_share": _rate(counts["exact"], total),
            "exact_or_block_share": _rate(counts["exact"] + counts["block"], total),
            "heading_or_none_share": _rate(counts["heading"] + counts["none"], total),
        },
    }


def _gate_result(item: dict[str, Any], gate_name: str) -> dict[str, Any] | None:
    gate = (item.get("gate_results") or {}).get(gate_name)
    return gate if isinstance(gate, dict) else None


def _gate_cases(items: list[dict[str, Any]], gate_name: str) -> list[dict[str, Any]]:
    return [item for item in items if isinstance(_gate_result(item, gate_name), dict)]


def _gate_pass_count(items: list[dict[str, Any]], gate_name: str) -> int:
    return sum(1 for item in items if str((_gate_result(item, gate_name) or {}).get("status") or "") == "PASS")


def _first_click_locate_ok(item: dict[str, Any]) -> bool:
    gate = _gate_result(item, "locate") or {}
    if str(gate.get("status") or "") != "PASS":
        return False
    return str(item.get("primary_hit_level") or "").strip().lower() in {"exact", "block"}


def _exact_first_click_ok(item: dict[str, Any]) -> bool:
    gate = _gate_result(item, "locate") or {}
    if str(gate.get("status") or "") != "PASS":
        return False
    return str(item.get("primary_hit_level") or "").strip().lower() == "exact"


def _build_scorecard(items: list[dict[str, Any]]) -> dict[str, Any]:
    totals = _summary_counts(items)
    locate_cases = _gate_cases(items, "locate")
    citation_cases = _gate_cases(items, "citation")
    figure_panel_cases = _gate_cases(items, "figure_panel")
    section_target_cases = _gate_cases(items, "section_target")
    structured_cases = _gate_cases(items, "structured_markers")
    quality_cases = _gate_cases(items, "quality")

    locate_pass = _gate_pass_count(locate_cases, "locate")
    first_click_ok = sum(1 for item in locate_cases if _first_click_locate_ok(item))
    exact_first_click = sum(1 for item in locate_cases if _exact_first_click_ok(item))
    heading_or_none = sum(
        1
        for item in locate_cases
        if str(item.get("primary_hit_level") or "").strip().lower() in {"heading", "none"}
    )
    locate_primary = _summary_primary_hit_levels(locate_cases)

    cases_with_direct = sum(1 for item in items if _to_int(item.get("visible_direct_segment_count")) > 0)
    raw_cite_leak_cases = sum(1 for item in items if _to_int(item.get("raw_structured_cite_count")) > 0)

    return {
        "overall_pass_rate": _rate(totals["pass"], totals["total"]),
        "direct_evidence_coverage": _rate(cases_with_direct, totals["total"]),
        "locate": {
            "cases": len(locate_cases),
            "pass": locate_pass,
            "pass_rate": _rate(locate_pass, len(locate_cases)),
            "first_click_ok": first_click_ok,
            "first_click_rate": _rate(first_click_ok, len(locate_cases)),
            "exact_first_click": exact_first_click,
            "exact_first_click_rate": _rate(exact_first_click, len(locate_cases)),
            "heading_or_none": heading_or_none,
            "heading_or_none_rate": _rate(heading_or_none, len(locate_cases)),
            "primary_hit_levels": locate_primary,
        },
        "citation": {
            "cases": len(citation_cases),
            "pass": _gate_pass_count(citation_cases, "citation"),
            "ref_num_accuracy": _rate(_gate_pass_count(citation_cases, "citation"), len(citation_cases)),
        },
        "figure_panel": {
            "cases": len(figure_panel_cases),
            "pass": _gate_pass_count(figure_panel_cases, "figure_panel"),
            "pass_rate": _rate(_gate_pass_count(figure_panel_cases, "figure_panel"), len(figure_panel_cases)),
        },
        "section_target": {
            "cases": len(section_target_cases),
            "pass": _gate_pass_count(section_target_cases, "section_target"),
            "pass_rate": _rate(_gate_pass_count(section_target_cases, "section_target"), len(section_target_cases)),
        },
        "structured_markers": {
            "cases": len(structured_cases),
            "pass": _gate_pass_count(structured_cases, "structured_markers"),
            "clean_rate": _rate(_gate_pass_count(structured_cases, "structured_markers"), len(structured_cases)),
            "raw_cite_leak_cases": raw_cite_leak_cases,
            "raw_cite_leak_rate": _rate(raw_cite_leak_cases, totals["total"]),
        },
        "quality": {
            "cases": len(quality_cases),
            "pass": _gate_pass_count(quality_cases, "quality"),
            "minimum_ok_rate": _rate(_gate_pass_count(quality_cases, "quality"), len(quality_cases)),
        },
    }


def _build_family_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        family = str(item.get("prompt_family") or "").strip() or "unknown"
        groups.setdefault(family, []).append(item)

    out: dict[str, Any] = {}
    for family in sorted(groups.keys()):
        rows = groups.get(family) or []
        totals = _summary_counts(rows)
        hit_summary = _summary_hit_levels(rows)
        primary_summary = _summary_primary_hit_levels(rows)
        locate_cases = _gate_cases(rows, "locate")
        citation_cases = _gate_cases(rows, "citation")
        out[family] = {
            "total": totals["total"],
            "pass": totals["pass"],
            "fail": totals["fail"],
            "pass_rate": _rate(totals["pass"], totals["total"]),
            "cases_with_direct": _to_int(hit_summary.get("cases_with_direct")),
            "cases_with_exact": _to_int(hit_summary.get("cases_with_exact")),
            "locate_cases": len(locate_cases),
            "locate_pass": _gate_pass_count(locate_cases, "locate"),
            "locate_first_click_ok": sum(1 for item in locate_cases if _first_click_locate_ok(item)),
            "citation_cases": len(citation_cases),
            "citation_pass": _gate_pass_count(citation_cases, "citation"),
            "hit_level_counts": hit_summary.get("counts") if isinstance(hit_summary.get("counts"), dict) else {},
            "primary_hit_levels": primary_summary.get("counts") if isinstance(primary_summary.get("counts"), dict) else {},
        }
    return out


def _lookup_metric_path(payload: dict[str, Any], dotted_path: str) -> float | int | str | None:
    cur: Any = payload
    for part in [seg for seg in str(dotted_path or "").split(".") if str(seg or "").strip()]:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    if isinstance(cur, (int, float, str)):
        return cur
    return None


def _evaluate_scorecard_thresholds(scorecard: dict[str, Any], thresholds: dict[str, Any] | None) -> dict[str, Any]:
    checks_out: list[dict[str, Any]] = []
    for metric, rules in dict(thresholds or {}).items():
        if not isinstance(rules, dict):
            continue
        actual = _lookup_metric_path(scorecard, str(metric or "").strip())
        min_value = rules.get("min")
        max_value = rules.get("max")
        ok = True
        if isinstance(actual, (int, float)):
            if isinstance(min_value, (int, float)) and float(actual) < float(min_value):
                ok = False
            if isinstance(max_value, (int, float)) and float(actual) > float(max_value):
                ok = False
        else:
            ok = False
        checks_out.append(
            {
                "metric": str(metric or "").strip(),
                "actual": actual,
                "min": min_value,
                "max": max_value,
                "ok": bool(ok),
            }
        )
    return {
        "enabled": bool(checks_out),
        "status": "FAIL" if any(not bool(item.get("ok")) for item in checks_out) else "PASS",
        "checks": checks_out,
    }


def _build_summary_payload(
    *,
    base_url: str,
    suite: dict[str, Any],
    results: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    totals = _summary_counts(results)
    hit_summary = _summary_hit_levels(results)
    primary_summary = _summary_primary_hit_levels(results)
    scorecard = _build_scorecard(results)
    threshold_summary = _evaluate_scorecard_thresholds(scorecard, suite.get("scorecard_thresholds"))
    return {
        "timestamp": started_at,
        "base_url": base_url,
        "suite_id": str(suite.get("suite_id") or "").strip(),
        "title": str(suite.get("title") or "").strip(),
        "counts": totals,
        "hit_levels": hit_summary,
        "primary_hit_levels": primary_summary,
        "scorecard": scorecard,
        "by_family": _build_family_summary(results),
        "thresholds": threshold_summary,
        "overall_status": (
            "FAIL"
            if (totals["fail"] > 0 or (bool(threshold_summary.get("enabled")) and str(threshold_summary.get("status")) == "FAIL"))
            else "PASS"
        ),
    }


def _render_gate_status(item: dict[str, Any], gate_name: str) -> str:
    gate = (item.get("gate_results") or {}).get(gate_name)
    if not isinstance(gate, dict):
        return "n/a"
    return str(gate.get("status") or "n/a")


def _render_report(
    *,
    base_url: str,
    suite: dict[str, Any],
    results: list[dict[str, Any]],
    started_at: str,
    summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# {str(suite.get('title') or 'Paper Guide Benchmark')}")
    lines.append("")
    lines.append(f"- Time: `{started_at}`")
    lines.append(f"- Base URL: `{base_url}`")
    lines.append(f"- Suite: `{str(suite.get('suite_id') or '').strip()}`")
    lines.append("")
    totals = summary.get("counts") if isinstance(summary.get("counts"), dict) else _summary_counts(results)
    lines.append("## Summary")
    lines.append(f"- total_cases: `{totals['total']}`")
    lines.append(f"- pass_cases: `{totals['pass']}`")
    lines.append(f"- failed_cases: `{totals['fail']}`")
    lines.append(f"- overall: **{str(summary.get('overall_status') or ('PASS' if totals['fail'] == 0 else 'FAIL'))}**")
    hit_summary = summary.get("hit_levels") if isinstance(summary.get("hit_levels"), dict) else _summary_hit_levels(results)
    hit_counts = hit_summary.get("counts") if isinstance(hit_summary.get("counts"), dict) else {}
    hit_rates = hit_summary.get("rates") if isinstance(hit_summary.get("rates"), dict) else {}
    lines.append("")
    scorecard = summary.get("scorecard") if isinstance(summary.get("scorecard"), dict) else {}
    locate_score = scorecard.get("locate") if isinstance(scorecard.get("locate"), dict) else {}
    citation_score = scorecard.get("citation") if isinstance(scorecard.get("citation"), dict) else {}
    structured_score = scorecard.get("structured_markers") if isinstance(scorecard.get("structured_markers"), dict) else {}
    quality_score = scorecard.get("quality") if isinstance(scorecard.get("quality"), dict) else {}
    lines.append("### Scorecard")
    lines.append(f"- overall_pass_rate: `{float(scorecard.get('overall_pass_rate') or 0.0):.3f}`")
    lines.append(f"- direct_evidence_coverage: `{float(scorecard.get('direct_evidence_coverage') or 0.0):.3f}`")
    lines.append(
        "- locate: "
        f"pass=`{int(locate_score.get('pass') or 0)}/{int(locate_score.get('cases') or 0)}` "
        f"(`{float(locate_score.get('pass_rate') or 0.0):.3f}`), "
        f"first_click=`{int(locate_score.get('first_click_ok') or 0)}/{int(locate_score.get('cases') or 0)}` "
        f"(`{float(locate_score.get('first_click_rate') or 0.0):.3f}`), "
        f"exact_first_click=`{float(locate_score.get('exact_first_click_rate') or 0.0):.3f}`, "
        f"heading_or_none=`{float(locate_score.get('heading_or_none_rate') or 0.0):.3f}`"
    )
    lines.append(
        "- citation: "
        f"pass=`{int(citation_score.get('pass') or 0)}/{int(citation_score.get('cases') or 0)}` "
        f"(`{float(citation_score.get('ref_num_accuracy') or 0.0):.3f}`)"
    )
    lines.append(
        "- structured_markers: "
        f"clean_rate=`{float(structured_score.get('clean_rate') or 0.0):.3f}`, "
        f"raw_cite_leak_rate=`{float(structured_score.get('raw_cite_leak_rate') or 0.0):.3f}`"
    )
    lines.append(
        "- quality: "
        f"minimum_ok_rate=`{float(quality_score.get('minimum_ok_rate') or 0.0):.3f}`"
    )
    threshold_summary = summary.get("thresholds") if isinstance(summary.get("thresholds"), dict) else {}
    if bool(threshold_summary.get("enabled")):
        lines.append("")
        lines.append("### Threshold Checks")
        lines.append(f"- status: **{str(threshold_summary.get('status') or 'PASS')}**")
        for check in list(threshold_summary.get("checks") or []):
            if not isinstance(check, dict):
                continue
            lines.append(
                f"- {str(check.get('metric') or '')}: actual=`{check.get('actual')}` "
                f"min=`{check.get('min')}` max=`{check.get('max')}` ok=`{bool(check.get('ok'))}`"
            )
    lines.append("")
    lines.append("### Locate Hit Levels")
    lines.append(f"- direct_segments_total: `{int(hit_summary.get('total_segments') or 0)}`")
    lines.append(
        "- counts: "
        f"exact=`{_to_int(hit_counts.get('exact'))}`, "
        f"block=`{_to_int(hit_counts.get('block'))}`, "
        f"heading=`{_to_int(hit_counts.get('heading'))}`, "
        f"none=`{_to_int(hit_counts.get('none'))}`"
    )
    lines.append(
        "- rates: "
        f"exact_share=`{float(hit_rates.get('exact_share') or 0.0):.3f}`, "
        f"exact_or_block_share=`{float(hit_rates.get('exact_or_block_share') or 0.0):.3f}`, "
        f"heading_or_none_share=`{float(hit_rates.get('heading_or_none_share') or 0.0):.3f}`"
    )
    lines.append(
        "- case_coverage: "
        f"cases_with_direct=`{_to_int(hit_summary.get('cases_with_direct'))}/{len(results)}`, "
        f"cases_with_exact=`{_to_int(hit_summary.get('cases_with_exact'))}/{len(results)}`"
    )
    family_summary = summary.get("by_family") if isinstance(summary.get("by_family"), dict) else {}
    if family_summary:
        lines.append("")
        lines.append("## By Family")
        lines.append("")
        lines.append("| Family | Cases | Pass | Locate | First Click | Citation | Primary Hit Levels |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for family in sorted(family_summary.keys()):
            rec = family_summary.get(family) if isinstance(family_summary.get(family), dict) else {}
            primary = rec.get("primary_hit_levels") if isinstance(rec.get("primary_hit_levels"), dict) else {}
            lines.append(
                f"| {family} | {int(rec.get('total') or 0)} | {int(rec.get('pass') or 0)} | "
                f"{int(rec.get('locate_pass') or 0)}/{int(rec.get('locate_cases') or 0)} | "
                f"{int(rec.get('locate_first_click_ok') or 0)}/{int(rec.get('locate_cases') or 0)} | "
                f"{int(rec.get('citation_pass') or 0)}/{int(rec.get('citation_cases') or 0)} | "
                f"E:{_to_int(primary.get('exact'))} B:{_to_int(primary.get('block'))} "
                f"H:{_to_int(primary.get('heading'))} N:{_to_int(primary.get('none'))} |"
            )
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump | hit_level |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for item in results:
        lines.append(
            f"| {item.get('id')} | {item.get('paper_id')} | {item.get('prompt_family')} | {item.get('status')} | "
            f"{_render_gate_status(item, 'answer')} | {_render_gate_status(item, 'citation')} | "
            f"{_render_gate_status(item, 'locate')} | {_render_gate_status(item, 'section_target')} | "
            f"{_render_gate_status(item, 'figure_panel')} | {_render_gate_status(item, 'jump')} | "
            f"{str(item.get('primary_hit_level') or 'none')} |"
        )
    lines.append("")
    lines.append("## Findings")
    for item in results:
        lines.append(f"### {item.get('id')} {item.get('title')}")
        lines.append(f"- paper: `{item.get('paper_id')}`")
        lines.append(f"- family: `{item.get('prompt_family')}`")
        lines.append(f"- status: **{item.get('status')}**")
        lines.append(f"- minimum_ok: `{_minimum_ok_label(item.get('answer_quality') if isinstance(item.get('answer_quality'), dict) else None)}`")
        lines.append(f"- cite_count: `{int(item.get('cite_count') or 0)}`")
        lines.append(f"- primary_hit_level: `{str(item.get('primary_hit_level') or 'none')}`")
        lines.append(f"- visible_direct_segment_count: `{_to_int(item.get('visible_direct_segment_count'))}`")
        hit_counts_case = item.get("hit_level_counts") if isinstance(item.get("hit_level_counts"), dict) else {}
        lines.append(
            "- hit_level_counts: "
            f"exact=`{_to_int(hit_counts_case.get('exact'))}`, "
            f"block=`{_to_int(hit_counts_case.get('block'))}`, "
            f"heading=`{_to_int(hit_counts_case.get('heading'))}`, "
            f"none=`{_to_int(hit_counts_case.get('none'))}`"
        )
        lines.append(f"- raw_structured_cite_count: `{int(item.get('raw_structured_cite_count') or 0)}`")
        gate_results = item.get("gate_results") or {}
        for gate_name in ("answer", "citation", "locate", "section_target", "figure_panel", "jump", "quality", "structured_markers"):
            gate = gate_results.get(gate_name)
            if not isinstance(gate, dict):
                continue
            lines.append(f"- gate[{gate_name}]: `{gate.get('status')}`")
            for reason in list(gate.get("reasons") or []):
                lines.append(f"- gate_reason[{gate_name}]: `{reason}`")
            for label in ("matched_ref_nums", "matched_headings", "matched_anchors"):
                value = gate.get(label)
                if value:
                    lines.append(f"- {gate_name}_{label}: `{value}`")
        if list(item.get("reasons") or []):
            for reason in list(item.get("reasons") or []):
                lines.append(f"- reason: `{reason}`")
        else:
            lines.append("- reason: `none`")
        preview = str(item.get("answer_preview") or "").strip()
        if preview:
            lines.append("- answer_preview:")
            lines.append("")
            lines.append("```text")
            lines.append(preview)
            lines.append("```")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _normalize_suite(raw_suite: dict[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    suite = copy.deepcopy(raw_suite or {})
    base = Path(base_dir or _repo_root()).resolve()
    papers_out: list[dict[str, Any]] = []
    for raw_paper in list(suite.get("papers") or []):
        if not isinstance(raw_paper, dict):
            continue
        source_path = str(raw_paper.get("source_path") or "").strip()
        if source_path and (not Path(source_path).is_absolute()):
            source_path = str((base / source_path).resolve())
        paper_id = str(raw_paper.get("id") or "").strip()
        paper_title = str(raw_paper.get("title") or "").strip()
        source_name = str(raw_paper.get("source_name") or "").strip() or Path(source_path).name
        cases_out: list[dict[str, Any]] = []
        for raw_case in list(raw_paper.get("cases") or []):
            if not isinstance(raw_case, dict):
                continue
            case = copy.deepcopy(raw_case)
            case["paper_id"] = paper_id
            case["paper_title"] = paper_title
            case["source_path"] = source_path
            case["source_name"] = source_name
            cases_out.append(case)
        paper = dict(raw_paper)
        paper["id"] = paper_id
        paper["title"] = paper_title
        paper["source_path"] = source_path
        paper["source_name"] = source_name
        paper["cases"] = cases_out
        papers_out.append(paper)
    suite["papers"] = papers_out
    return suite


def _load_manifest(manifest_path: str) -> dict[str, Any]:
    path = Path(str(manifest_path or "")).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("manifest must be a JSON object")
    return _normalize_suite(data, base_dir=path.parent)


def load_suite(*, suite_name: str = "cross_paper_v1", manifest_path: str = "") -> dict[str, Any]:
    if str(manifest_path or "").strip():
        return _load_manifest(str(manifest_path or "").strip())
    suites = _builtin_suites()
    if suite_name not in suites:
        raise RuntimeError(f"unknown suite: {suite_name}")
    return _normalize_suite(suites[suite_name], base_dir=_repo_root())


def _iter_selected_cases(
    suite: dict[str, Any],
    *,
    paper_filter: str = "",
    case_filter: str = "",
) -> list[dict[str, Any]]:
    paper_re = re.compile(paper_filter, flags=re.IGNORECASE) if str(paper_filter or "").strip() else None
    case_re = re.compile(case_filter, flags=re.IGNORECASE) if str(case_filter or "").strip() else None
    selected: list[dict[str, Any]] = []
    for paper in list(suite.get("papers") or []):
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("id") or "").strip()
        paper_title = str(paper.get("title") or "").strip()
        if paper_re and (not (paper_re.search(paper_id) or paper_re.search(paper_title))):
            continue
        for case in list(paper.get("cases") or []):
            if not isinstance(case, dict):
                continue
            case_id = str(case.get("id") or "").strip()
            case_title = str(case.get("title") or "").strip()
            if case_re and (not (case_re.search(case_id) or case_re.search(case_title))):
                continue
            selected.append(case)
    return selected


def _run_case(
    *,
    base_url: str,
    case: dict[str, Any],
    timeout_s: float,
    generate_body: dict[str, Any],
    suite_defaults: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    conv_id = _create_conversation(
        base_url,
        source_path=str(case.get("source_path") or ""),
        source_name=str(case.get("source_name") or ""),
        title=f"paper-guide-benchmark-{case.get('paper_id')}-{case.get('id')}-{uuid.uuid4().hex[:8]}",
        timeout_s=timeout_s,
    )
    before_messages = _list_messages(base_url, conv_id=conv_id, timeout_s=timeout_s)
    min_message_id = _max_message_id(before_messages)
    session_id = _start_generation(
        base_url,
        conv_id=conv_id,
        prompt=str(case.get("prompt") or ""),
        timeout_s=timeout_s,
        generate_body=generate_body,
    )
    payload = _stream_generation(base_url, session_id=session_id, timeout_s=timeout_s)
    time.sleep(0.6)
    message, answer_text = _fetch_answer_bundle(
        base_url,
        conv_id=conv_id,
        payload=payload,
        timeout_s=timeout_s,
        min_message_id=min_message_id,
    )
    answer_quality = payload.get("answer_quality") if isinstance(payload.get("answer_quality"), dict) else {}
    render_cache = _render_cache(message, payload)
    provenance = _provenance(message, payload)
    evaluation = _evaluate_case(
        case,
        suite_defaults=suite_defaults,
        answer_text=answer_text,
        answer_quality=answer_quality if isinstance(answer_quality, dict) else {},
        render_cache=render_cache,
        provenance=provenance,
        message=message,
    )
    return evaluation, {
        "conv_id": conv_id,
        "case": case,
        "payload": payload,
        "message": message,
        "evaluation": evaluation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a cross-paper benchmark for paper-guide grounding and locate behavior.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend base URL.")
    parser.add_argument("--suite", default="cross_paper_v1", help="Builtin suite name.")
    parser.add_argument("--manifest", default="", help="Optional external JSON manifest path.")
    parser.add_argument("--paper-filter", default="", help="Regex filter for paper id/title.")
    parser.add_argument("--case-filter", default="", help="Regex filter for case id/title.")
    parser.add_argument("--timeout-s", type=float, default=180.0, help="HTTP timeout seconds.")
    parser.add_argument("--out-dir", default="test_results/paper_guide_benchmark", help="Output directory.")
    parser.add_argument("--no-restore-prefs", action="store_true", help="Do not restore answer settings after the run.")
    parser.add_argument("--list-suites", action="store_true", help="List builtin suite names and exit.")
    args = parser.parse_args(argv)

    if bool(args.list_suites):
        for name in sorted(_builtin_suites().keys()):
            print(name)
        return 0

    base_url = str(args.base_url or "").rstrip("/")
    suite = load_suite(suite_name=str(args.suite or "").strip() or "cross_paper_v1", manifest_path=str(args.manifest or "").strip())
    selected_cases = _iter_selected_cases(
        suite,
        paper_filter=str(args.paper_filter or ""),
        case_filter=str(args.case_filter or ""),
    )
    if not selected_cases:
        print("[ERROR] no benchmark cases selected", file=sys.stderr)
        return 2

    started_at = dt.datetime.now().isoformat(timespec="seconds")
    out_dir = Path(args.out_dir) / _now_str()
    out_dir.mkdir(parents=True, exist_ok=True)

    backup_prefs: dict[str, Any] | None = None
    try:
        cur = _get_json(base_url, "/api/settings", timeout_s=float(args.timeout_s))
        prefs = cur.get("prefs") if isinstance(cur, dict) else {}
        prefs = prefs if isinstance(prefs, dict) else {}
        backup_prefs = {
            "answer_contract_v1": prefs.get("answer_contract_v1"),
            "answer_depth_auto": prefs.get("answer_depth_auto"),
            "answer_mode_hint": prefs.get("answer_mode_hint"),
            "answer_output_mode": prefs.get("answer_output_mode"),
        }
    except Exception as exc:
        print(f"[WARN] failed to read current settings: {exc}", file=sys.stderr)

    results: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    suite_defaults = suite.get("default_checks") if isinstance(suite.get("default_checks"), dict) else {}
    generate_body = dict(DEFAULT_GENERATE_BODY)
    generate_body.update(dict(suite.get("generate_body") or {}))
    try:
        _patch_settings(
            base_url,
            dict(DEFAULT_PREF_PATCH) | dict(suite.get("prefs_patch") or {}),
            timeout_s=float(args.timeout_s),
        )
        for case in selected_cases:
            evaluation, raw_row = _run_case(
                base_url=base_url,
                case=case,
                timeout_s=float(args.timeout_s),
                generate_body=generate_body,
                suite_defaults=suite_defaults if isinstance(suite_defaults, dict) else {},
            )
            results.append(evaluation)
            raw_rows.append(raw_row)
            print(
                f"[{evaluation['id']}] {evaluation['status']} | paper={evaluation['paper_id']} "
                f"| cite={_render_gate_status(evaluation, 'citation')} "
                f"| locate={_render_gate_status(evaluation, 'locate')} "
                f"| section={_render_gate_status(evaluation, 'section_target')} "
                f"| panel={_render_gate_status(evaluation, 'figure_panel')} "
                f"| hit={str(evaluation.get('primary_hit_level') or 'none')}"
            )
    finally:
        if backup_prefs is not None and (not bool(args.no_restore_prefs)):
            try:
                _patch_settings(base_url, backup_prefs, timeout_s=float(args.timeout_s))
            except Exception as exc:
                print(f"[WARN] failed to restore settings: {exc}", file=sys.stderr)

    summary = _build_summary_payload(
        base_url=base_url,
        suite=suite,
        results=results,
        started_at=started_at,
    )
    report_md = _render_report(
        base_url=base_url,
        suite=suite,
        results=results,
        started_at=started_at,
        summary=summary,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")
    (out_dir / "raw_results.json").write_text(json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "suite_resolved.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary JSON written: {out_dir / 'summary.json'}")
    print(f"Report written: {out_dir / 'report.md'}")
    print(f"Raw JSON written: {out_dir / 'raw_results.json'}")
    return 0 if str(summary.get("overall_status") or "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
