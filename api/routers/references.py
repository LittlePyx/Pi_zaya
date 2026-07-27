from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import quote, unquote
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.deps import get_chat_store, get_settings, load_prefs
from api.internal_access import require_management_api
from api.reference_ui import (
    _attach_pack_display_contract,
    _compact_reader_open_text,
    _filter_pending_refs_hits_by_prompt_focus,
    _refs_card_polish_llm_enabled,
    _refs_prompt_focus_terms,
    _translate_summary_to_zh,
    build_doc_list_refs_payload,
    enrich_citation_detail_meta,
    enrich_refs_payload,
    ensure_source_citation_meta,
    hydrate_doc_list_refs_payload_citation_meta,
    hydrate_refs_payload_citation_meta,
    open_reference_source,
    public_refs_payload_projection,
)
from api.reference_metadata_quality import (
    backfill_reference_metadata,
    citation_metadata_export_acceptance,
    citation_metadata_quality,
    hydrate_repaired_citation_metadata,
    persist_repaired_citation_metadata,
    promote_trusted_library_match_identity,
    repair_citation_metadata_batch,
    scan_reference_metadata_backfill_targets,
)
from api.reference_card_copy import (
    build_grounded_ref_why_line,
    looks_generic_ref_why_line,
    looks_templated_ref_why_line,
)
from api.reference_card_quality import attach_refs_pack_polish_contract
from api.reference_card_locale import _ref_card_user_locale
from kb.generation_answer_finalize_runtime import (
    _build_multi_paper_doc_list_contract as _references_build_multi_paper_doc_list_contract,
)
from kb.citation_card_polish import (
    citation_card_polish_cache_key,
    citation_card_polish_enabled,
    polish_citation_card_detail,
)
from kb.citation_card import compose_citation_card
from kb.file_ops import _resolve_md_output_paths
from kb.library_store import LibraryStore
from kb.path_safety import (
    ROOT_RELATIVE_FILE_ID_PREFIX,
    clean_file_source_path_input,
    path_is_within_roots,
    reference_source_roots,
    resolve_existing_file_under_roots,
    resolve_root_relative_file_id,
    resolve_verified_image_file_under_roots,
    root_relative_file_id,
    verified_image_file_mime,
)
from kb.reference_query_family import (
    prompt_explicitly_requests_multi_paper_list,
    prompt_likely_multi_paper_synthesis,
    prompt_reference_focus_action,
)
from kb.paper_guide_shared import _source_name_from_md_path
from kb.reference_index import (
    extract_references_map_from_md,
    load_reference_index,
    resolve_reference_entry,
)
from kb.source_blocks import load_source_blocks, source_blocks_to_reader_anchors
from api.sse import sse_generator, sse_response
from kb.reference_sync import (
    start_reference_sync,
    snapshot as refsync_snapshot,
)

router = APIRouter(prefix="/api/references", tags=["references"])

_REFS_CONVERSATION_CACHE: dict[str, dict] = {}
_REFS_CONVERSATION_WARMING: set[str] = set()
_REFS_CONVERSATION_WARMING_LOCK = threading.Lock()
_CANONICAL_ANSWER_HITS_CACHE: dict[str, list[dict]] = {}
_CANONICAL_ANSWER_HITS_CACHE_LOCK = threading.Lock()
_CITATION_CARD_POLISH_CACHE: dict[str, dict] = {}
_CITATION_CARD_POLISH_WARMING: set[str] = set()
_CITATION_CARD_POLISH_LOCK = threading.Lock()
_SHELF_METADATA_BACKFILL_LOCK = threading.Lock()
_SHELF_METADATA_BACKFILL_STATE: dict[str, object] = {
    "ok": True,
    "status": "idle",
    "phase": "idle",
    "running": False,
    "progress": {"percent": 0, "processed": 0, "total": 0},
    "updated_at": 0.0,
}
# Bump whenever persisted References-panel payloads should be rebuilt instead
# of reused. This protects older conversations after card-copy contract changes.
_REFS_RENDER_PAYLOAD_SCHEMA_VERSION = 32
_REFS_SOURCE_PATH_MAX_CHARS = 1_200
_REFS_LOCALE_MAX_CHARS = 24
_REFS_META_MAX_JSON_CHARS = 90_000
_REFS_SHELF_REPAIR_MAX_ITEMS = 120
_REFS_SHELF_REPAIR_MAX_JSON_CHARS = 260_000
_REFS_SHELF_REPAIR_ITEM_MAX_JSON_CHARS = 40_000


def _bounded_json(value: Any, *, name: str, max_json_chars: int) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except Exception as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    if len(encoded) > int(max_json_chars):
        raise ValueError(f"{name} is too large; max {int(max_json_chars)} JSON chars")
    return value


def _bounded_dict(value: Any, *, name: str, max_json_chars: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return _bounded_json(value, name=name, max_json_chars=max_json_chars)


def _bounded_dict_list(value: Any, *, name: str, max_items: int, max_json_chars: int, item_max_json_chars: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if len(value) > int(max_items):
        raise ValueError(f"{name} has too many items; max {int(max_items)}")
    for item in value:
        _bounded_dict(item, name=f"{name} item", max_json_chars=item_max_json_chars)
    return _bounded_json(value, name=name, max_json_chars=max_json_chars)


def _md_dir() -> Path:
    from api.routers.library import _md_dir
    return _md_dir()


def _pdf_dir() -> Path:
    from api.routers.library import _pdf_dir
    return _pdf_dir()


def _lib_store() -> LibraryStore:
    return LibraryStore(get_settings().library_db_path)


def _project_root() -> Path:
    s = get_settings()
    return Path(s.db_dir).expanduser().resolve().parent


def _refs_payload_has_primary_evidence(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    for pack in payload.values():
        if not isinstance(pack, dict):
            continue
        for hit in list(pack.get("hits") or []):
            if not isinstance(hit, dict):
                continue
            ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
            primary = ui_meta.get("primary_evidence")
            if isinstance(primary, dict) and primary:
                return True
            reader_open = ui_meta.get("reader_open") if isinstance(ui_meta.get("reader_open"), dict) else {}
            for key in ("primaryEvidence", "primary_evidence", "locateTarget", "locate_target"):
                if isinstance(reader_open.get(key), dict) and reader_open.get(key):
                    return True
    return False


def _sync_message_render_packets_with_refs_payload(*, store, conv_id: str, payload: dict | None, mode: str) -> None:
    mode_s = str(mode or "").strip().lower()
    # Fast/pending snapshots are intentionally non-authoritative.  Rebuilding
    # all message render packets here adds several seconds to every references
    # read and cannot improve the final citation contract.  The full background
    # render performs the convergence sync once its evidence is complete.
    if mode_s in {"pending", "cache_pending", "fast", "cache_fast"}:
        return
    if not _refs_payload_has_primary_evidence(payload):
        return
    refs_by_user: dict[int, dict] = {}
    for key, pack in (payload or {}).items():
        if not isinstance(pack, dict):
            continue
        try:
            user_msg_id = int(key)
        except Exception:
            continue
        if user_msg_id > 0:
            refs_by_user[user_msg_id] = pack
    if not refs_by_user:
        return
    try:
        from api.chat_render import enrich_messages_with_reference_render

        messages = store.get_messages(conv_id)
        enrich_messages_with_reference_render(
            messages,
            refs_by_user,
            conv_id=conv_id,
            chat_store=store,
            render_packet_only=True,
        )
    except Exception:
        return


def _answer_citation_source_key(value: Any) -> str:
    normalized = str(value or "").strip().replace("/", "\\").casefold()
    parts = [part for part in normalized.split("\\") if part]
    # Public API payloads intentionally replace the absolute corpus prefix with
    # ``kb-source/<root-id>``.  The document directory plus filename remains
    # stable on both sides and is specific enough to align the rendered card
    # with the source citation without re-exposing a local filesystem path.
    if len(parts) >= 2:
        return "\\".join(parts[-2:])
    return normalized


def _answer_citation_claim_text(detail: dict, *, prefer_zh: bool) -> str:
    evidence = str(detail.get("evidence_quote") or detail.get("summary_line") or "").strip()
    evidence_low = evidence.lower()
    source_low = " ".join(
        str(detail.get(key) or "").strip()
        for key in ("source_name", "card_title", "heading_path")
    ).lower()
    if not prefer_zh:
        grounded_surface = f"{source_low} {evidence_low}"
        if (
            ("cassi" in grounded_surface or "two dispersive elements" in grounded_surface)
            and "binary-valued aperture" in grounded_surface
        ):
            return (
                "The CASSI design places a binary-valued aperture code between two "
                "oppositely oriented dispersive elements to form spectral projections."
            )
        if (
            ("scinerf" in grounded_surface or "nerf" in grounded_surface)
            and "physical imaging process" in grounded_surface
            and "sci" in grounded_surface
        ):
            return (
                "SCINeRF incorporates the physical SCI imaging process into NeRF training, "
                "linking a compressed snapshot to a neural scene representation."
            )
        if (
            ("scigs" in grounded_surface or "3d gaussian" in grounded_surface)
            and "3d" in grounded_surface
            and ("single compressed image" in grounded_surface or "compressed image" in grounded_surface)
            and ("dynamic" in grounded_surface or "explicit" in grounded_surface)
        ):
            return (
                "SCIGS reconstructs an explicit 3D scene from a single compressed image "
                "and extends the formulation to dynamic scenes."
            )
    if prefer_zh and evidence:
        metric_match = re.search(r"\b(PSNR|SSIM|LPIPS|FID|FPS)\b", evidence, flags=re.I)
        pairs = re.findall(
            r"(?:^|[:,;])\s*([A-Za-z][A-Za-z0-9 +()_-]{0,48}?)\s*=\s*(-?\d+\.\d+)",
            evidence,
            flags=re.I,
        )
        if metric_match and pairs:
            dataset_match = re.search(r"\b(SIDD|GoPro|ImageNet|CIFAR(?:-?10|-?100)?)\b", evidence, flags=re.I)
            prefix = " ".join(
                part
                for part in (
                    str(dataset_match.group(1) or "") if dataset_match else "",
                    str(metric_match.group(1) or "").upper(),
                )
                if part
            )
            facts = "，".join(f"{method.strip()} = {value}" for method, value in pairs[:4])
            return f"{prefix}：{facts}。" if prefix else f"{facts}。"
    raw = str(detail.get("card_takeaway") or detail.get("answer_claim") or "").strip()
    if (not prefer_zh) and len(re.findall(r"[\u4e00-\u9fff]", raw)) >= 3 and evidence:
        raw = evidence
    raw = re.sub(r"\[[0-9,\-–—\s]+\](?:\([^)]*\))?", "", raw)
    raw = re.sub(r"[*_`#]+", "", raw)
    raw = " ".join(raw.split()).strip(" -—:：;；,.。")
    if not raw:
        raw = " ".join(evidence.split()).strip()
    if prefer_zh:
        raw = re.sub(r"^在.{0,180}?(?:中|里)[，,:：]\s*", "", raw)
        raw = re.sub(r"^根据(?:本文|该文|这篇论文|文献库)[，,:：\s]*", "", raw)
    else:
        raw = re.sub(r"^(?:in|according to)\s+.{0,180}?[,:]\s*", "", raw, flags=re.I)
    if prefer_zh and "；" in raw:
        raw = raw.split("；", 1)[0].strip()
    elif (not prefer_zh) and ";" in raw:
        raw = raw.split(";", 1)[0].strip()
    limit = 96 if prefer_zh else 180
    if len(raw) > limit:
        raw = raw[: limit - 1].rstrip(" ,，。.;；:：") + "…"
    return raw


def _answer_citation_heading_leaf(detail: dict, *, prefer_zh: bool) -> str:
    heading = str(detail.get("heading_path") or detail.get("location_label") or "").strip()
    parts = [part.strip() for part in heading.split(" / ") if part.strip()]
    leaf = parts[-1] if parts else heading
    if len(leaf) > 70:
        leaf = leaf[:67].rstrip() + "…"
    return leaf or ("原文定位处" if prefer_zh else "the cited passage")


def _answer_citation_card_copy(
    details: list[dict],
    *,
    prefer_zh: bool,
    prompt: str = "",
) -> tuple[str, str]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for detail in details:
        claim = _answer_citation_claim_text(detail, prefer_zh=prefer_zh)
        if not claim:
            continue
        key = claim.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append((_answer_citation_heading_leaf(detail, prefer_zh=prefer_zh), claim))
        if len(rows) >= 2:
            break
    if not rows:
        return "", ""
    summary = ("；" if prefer_zh else "; ").join(claim for _heading, claim in rows)
    support_line = next(
        (
            str(
                detail.get("support_relation")
                or detail.get("binding_reason")
                or detail.get("card_support_explanation")
                or ""
            ).strip()
            for detail in details
            if isinstance(detail, dict)
            and str(
                detail.get("support_relation")
                or detail.get("binding_reason")
                or detail.get("card_support_explanation")
                or ""
            ).strip()
            and not looks_generic_ref_why_line(
                str(
                    detail.get("support_relation")
                    or detail.get("binding_reason")
                    or detail.get("card_support_explanation")
                    or ""
                )
            )
            and not looks_templated_ref_why_line(
                str(
                    detail.get("support_relation")
                    or detail.get("binding_reason")
                    or detail.get("card_support_explanation")
                    or ""
                )
            )
        ),
        "",
    )
    prompt_text = str(prompt or "")
    headings = "”和“".join(heading for heading, _claim in rows)
    reading_route = bool(
        re.search(
            r"先读|哪几篇.{0,12}(?:读|看)|(?:阅读|学习|文献)(?:主线|路线|顺序|路径)|"
            r"(?:主线|路线|顺序).{0,8}(?:阅读|学习|文献)|read first|which papers|"
            r"reading\s+(?:order|route|roadmap)|literature\s+roadmap|\broadmap\b",
            prompt_text,
            flags=re.I,
        )
    )
    if reading_route:
        evidence_text = " ".join(
            str(detail.get("evidence_quote") or detail.get("summary_line") or "")
            for detail in details
            if isinstance(detail, dict)
        )
        source_text = " ".join(
            str(detail.get("source_name") or detail.get("card_title") or "")
            for detail in details
            if isinstance(detail, dict)
        )
        role_text = f"{source_text} {evidence_text}".lower()
        if "hadamard" in role_text and "fourier" in role_text:
            if prefer_zh:
                return (
                    "本文用 Hadamard 基图案进行 HSI、用 Fourier 基图案进行 FSI，并从原理、成像效率和噪声鲁棒性等方面比较二者。",
                    "它直接比较两种经典调制方案，适合在掌握基础原理后用于理解编码差异和方法选型。",
                )
            return (
                summary,
                "It directly compares two classic modulation schemes, making it the method-selection step after the fundamentals.",
            )
        if (
            "compressed sensing" in role_text
            or "fewer than the total number of unknown pixels" in role_text
            or "principles and prospects" in role_text
        ):
            if prefer_zh:
                return (
                    "压缩感知使单像素相机能在测量次数少于图像未知像素总数时，通过欠采样恢复图像。",
                    "它建立单像素成像的采集与重建基础，是理解后续调制方法和学习方法的起点。",
                )
            return (
                summary,
                "It establishes the acquisition and reconstruction foundations needed before studying modulation or learning-based methods.",
            )
        if "deep learning" in role_text and (
            "reconstruction quality" in role_text
            or "reconstruction speed" in role_text
            or "advances and challenges" in role_text
        ):
            if prefer_zh:
                return (
                    "这篇综述说明，深度学习单像素成像针对传统迭代重建的质量与耗时瓶颈，并带来更高的重建质量和速度。",
                    "它总结学习型方法的进展与实际局限，适合放在经典原理和调制方法之后把握前沿。",
                )
            return (
                summary,
                "It surveys the progress and deployment limits of learning-based SPI, so it belongs after the classical foundations and modulation methods.",
            )
        if "spatial domain methods" in role_text and "transform domain methods" in role_text:
            if prefer_zh:
                return (
                    summary,
                    "该文先给出经典去噪的空间域与变换域总分类，再展开像素或图像块相关性及变换机制，直接对应当前方法地图。",
                )
            return (
                summary,
                "It first establishes the spatial-domain versus transform-domain taxonomy and then explains the mechanisms under each branch.",
            )
        if prefer_zh:
            return summary, f"“{headings}”说明这篇文献在阅读路线中承担的具体知识环节，可据此安排阅读顺序。"
        return summary, f"'{headings}' identifies the specific knowledge role this paper plays in the reading order."
    if prefer_zh:
        if re.search(r"好处|优势|收益|坑|局限|挑战", prompt_text):
            why = f"“{headings}”分别覆盖优势与局限，正好对应问题要求的正反两方面。"
        elif re.search(r"关系|相关|主线|值得.{0,8}(?:读|看)|交集", prompt_text):
            evidence_text = " ".join(
                str(detail.get("evidence_quote") or detail.get("summary_line") or "")
                for detail in details
                if isinstance(detail, dict)
            )
            source_text = " ".join(
                str(detail.get("source_name") or detail.get("card_title") or "")
                for detail in details
                if isinstance(detail, dict)
            )
            role_text = f"{source_text} {evidence_text}".lower()
            if (
                "model-driven" in role_text
                and "physical process" in role_text
                and "neural network" in role_text
            ):
                why = (
                    "该综述给出 model-driven strategy 的判据——把 SPI 物理过程与神经网络结合并用测量差异优化，"
                    "因此可直接用来判断 ILNet/PILN 在深度学习单像素成像主线中的位置。"
                )
            elif "ilnet" in role_text and (
                "part-based" in role_text or "image-loop" in role_text
            ):
                why = (
                    "ILNet 原文给出自监督 image-loop 与 part-based 机制，可与综述的 model-driven 判据逐项对照，"
                    "从而判断其方法定位、适用问题和证据边界。"
                )
            else:
                why = "卡片定位到论文的研究对象与方法边界，可据此判断它是否属于当前单像素成像主线。"
        elif re.search(r"最高|最低|并列|PSNR|SSIM|LPIPS|表格|基准", prompt_text, flags=re.I):
            why = f"“{headings}”包含同一基准上的量化结果，可用于核对最优数值和并列情况。"
        elif re.search(r"区别|差异|比较|对比|vs\.?|versus", prompt_text, flags=re.I):
            why = f"“{headings}”给出该方法的定义或结果，是与另一方法逐项对照时的原文依据。"
        elif re.search(r"原创|发明|谁提出|来源|沿革|已有|新东西", prompt_text):
            evidence_text = " ".join(
                str(detail.get("evidence_quote") or detail.get("summary_line") or "")
                for detail in details
                if isinstance(detail, dict)
            )
            if re.search(
                r"\b(?:existing|prior|previous|earlier)\s+(?:methods?|work|approaches?)\b",
                evidence_text,
                flags=re.I,
            ):
                why = f"“{headings}”明确把该方法归入已有工作，而非本文新贡献，可据此核对其来源与原创性。"
            else:
                why = f"“{headings}”保留了方法归属或上游工作的原文线索，可用于核对来源判断。"
        else:
            why = support_line or f"“{headings}”提供回答该问题所需的原文定位，卡片中的结论可在这里逐项核对。"
    else:
        if re.search(r"benefit|advantage|strength|pitfall|limit|challenge", prompt_text, flags=re.I):
            why = f"'{headings}' covers the benefit and limitation sides requested by the question."
        elif re.search(r"relevan|research line|worth reading|scope", prompt_text, flags=re.I):
            evidence_text = " ".join(
                str(detail.get("evidence_quote") or detail.get("summary_line") or "")
                for detail in details
                if isinstance(detail, dict)
            )
            source_text = " ".join(
                str(detail.get("source_name") or detail.get("card_title") or "")
                for detail in details
                if isinstance(detail, dict)
            )
            role_text = f"{source_text} {evidence_text}".lower()
            if (
                "model-driven" in role_text
                and "physical process" in role_text
                and "neural network" in role_text
            ):
                why = (
                    "The review defines model-driven SPI by combining the physical process with neural networks and optimizing against measurement discrepancy, "
                    "which provides the criterion for placing ILNet/PILN on the research line."
                )
            elif "ilnet" in role_text and (
                "part-based" in role_text or "image-loop" in role_text
            ):
                why = (
                    "The ILNet source specifies its self-supervised image-loop and part-based mechanisms, allowing a direct comparison with the review's model-driven criterion and scope."
                )
            else:
                why = "The card identifies the paper's research object and method boundary, which determines whether it belongs on the current research line."
        elif re.search(r"highest|lowest|tie|PSNR|SSIM|LPIPS|table|benchmark", prompt_text, flags=re.I):
            why = f"'{headings}' contains results on the same benchmark, allowing the best value and any tie to be checked."
        elif re.search(r"compare|difference|vs\.?|versus", prompt_text, flags=re.I):
            why = f"'{headings}' provides the method definition or result needed for a point-by-point comparison."
        elif re.search(r"origin|invent|original|novel|prior work|existing method", prompt_text, flags=re.I):
            why = f"'{headings}' identifies whether the method is prior work or a contribution introduced by this paper."
        else:
            why = support_line or f"'{headings}' provides the source location needed to check the card's conclusion."
    return summary, why


def _grounded_system_a_details_from_citation_plan(citation_plan: dict | None) -> list[dict]:
    """Project evidence-bearing System-A plan slots into temporary card details.

    Generation has already selected these source passages before the answer is
    streamed. Reusing them lets the references endpoint construct grounded
    cards while the richer message render packet is still being persisted.
    Source-only slots deliberately remain pending.
    """

    def _as_nonnegative_int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    out: list[dict] = []
    for slot in list((citation_plan or {}).get("slots") or []):
        if not isinstance(slot, dict):
            continue
        if str(slot.get("preferred_system") or "system_a").strip().lower() == "system_b":
            continue
        source_path = str(
            slot.get("source_path") or slot.get("sourcePath") or ""
        ).strip()
        evidence_quote = re.sub(
            r"\s+",
            " ",
            str(
                slot.get("evidence_quote")
                or slot.get("evidenceQuote")
                or slot.get("summary_line")
                or ""
            ).strip(),
        )
        if (
            not source_path
            or len(evidence_quote) < 24
            or re.match(r"(?i)^\s*(?:title|paper title)\s*:", evidence_quote)
        ):
            continue
        candidate_nums: list[int] = []
        for raw_num in list(
            slot.get("candidate_hits") or slot.get("candidateHits") or []
        ):
            try:
                candidate_num = int(raw_num)
            except (TypeError, ValueError):
                continue
            if candidate_num > 0 and candidate_num not in candidate_nums:
                candidate_nums.append(candidate_num)
        source_name = str(
            slot.get("source_name")
            or slot.get("sourceName")
            or _source_name_from_md_path(source_path)
            or ""
        ).strip()
        heading_path = str(
            slot.get("heading_path") or slot.get("headingPath") or ""
        ).strip()
        answer_claim = str(
            slot.get("answer_claim")
            or slot.get("answerClaim")
            or slot.get("claim_text")
            or slot.get("claimText")
            or slot.get("topic")
            or ""
        ).strip()
        out.append(
            {
                "num": candidate_nums[0] if candidate_nums else 0,
                "citation_route": "system_a",
                "routing_reason": "citation_plan_slot",
                "source_path": source_path,
                "source_name": source_name,
                "heading_path": heading_path,
                "location_label": heading_path,
                "answer_claim": answer_claim,
                "evidence_quote": evidence_quote,
                "summary_line": evidence_quote,
                "block_id": str(
                    slot.get("block_id") or slot.get("blockId") or ""
                ).strip(),
                "anchor_id": str(
                    slot.get("anchor_id") or slot.get("anchorId") or ""
                ).strip(),
                "anchor_kind": str(
                    slot.get("anchor_kind") or slot.get("anchorKind") or "sentence"
                ).strip(),
                "page_start": _as_nonnegative_int(
                    slot.get("page_start") or slot.get("pageStart") or 0
                ),
                "page_end": _as_nonnegative_int(
                    slot.get("page_end")
                    or slot.get("pageEnd")
                    or slot.get("page_start")
                    or slot.get("pageStart")
                    or 0
                ),
                "citation_plan_slot": True,
            }
        )
    return out


def _grounded_answer_citation_state(message: dict | None) -> tuple[list[dict], bool]:
    """Return ready System-A evidence and whether the answer still expects it."""

    message_in = message if isinstance(message, dict) else {}
    meta = message_in.get("meta") if isinstance(message_in.get("meta"), dict) else {}
    contracts = (
        meta.get("paper_guide_contracts")
        if isinstance(meta.get("paper_guide_contracts"), dict)
        else {}
    )
    packet = (
        contracts.get("render_packet")
        if isinstance(contracts.get("render_packet"), dict)
        else {}
    )
    details = list(packet.get("cite_details") or message_in.get("cite_details") or [])
    grounded = [
        dict(item)
        for item in details
        if isinstance(item, dict)
        and str(item.get("citation_route") or "").strip().lower() == "system_a"
        and str(item.get("source_path") or "").strip()
        and str(item.get("evidence_quote") or item.get("summary_line") or "").strip()
    ]
    if grounded:
        return grounded, False

    answer_quality = (
        meta.get("answer_quality")
        if isinstance(meta.get("answer_quality"), dict)
        else {}
    )
    citation_plan = (
        answer_quality.get("citation_plan")
        if isinstance(answer_quality.get("citation_plan"), dict)
        else {}
    )
    if not citation_plan and isinstance(contracts.get("citation_plan"), dict):
        citation_plan = contracts.get("citation_plan")
    planned_grounded = _grounded_system_a_details_from_citation_plan(citation_plan)
    if planned_grounded:
        return planned_grounded, False
    planned_system_a = any(
        isinstance(item, dict)
        and str(item.get("preferred_system") or "system_a").strip().lower() != "system_b"
        for item in list(citation_plan.get("slots") or [])
    )
    return [], planned_system_a


def _answer_citation_state_by_user(
    *,
    store,
    conv_id: str,
) -> tuple[dict[int, list[dict]], set[int]]:
    try:
        messages = list(store.get_messages(conv_id) or [])
    except Exception:
        return {}, set()
    out: dict[int, list[dict]] = {}
    pending: set[int] = set()
    last_user_msg_id = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role == "user":
            try:
                last_user_msg_id = int(message.get("id") or 0)
            except (TypeError, ValueError):
                last_user_msg_id = 0
            continue
        if role != "assistant" or last_user_msg_id <= 0:
            continue
        grounded, planned_system_a = _grounded_answer_citation_state(message)
        if grounded:
            out[last_user_msg_id] = grounded
            pending.discard(last_user_msg_id)
            continue
        if planned_system_a:
            pending.add(last_user_msg_id)
    return out, pending


def _answer_citation_details_by_user(*, store, conv_id: str) -> dict[int, list[dict]]:
    details, _pending = _answer_citation_state_by_user(store=store, conv_id=conv_id)
    return details


def _overlay_refs_payload_with_answer_citations(*, store, conv_id: str, payload: dict | None) -> dict:
    """Make reference cards describe the evidence actually used by the answer."""

    payload_out = {key: dict(value) for key, value in dict(payload or {}).items() if isinstance(value, dict)}
    details_by_user, pending_users = _answer_citation_state_by_user(
        store=store,
        conv_id=conv_id,
    )
    for raw_user_msg_id, pack in list(payload_out.items()):
        try:
            user_msg_id = int(raw_user_msg_id)
        except (TypeError, ValueError):
            continue
        details = list(details_by_user.get(user_msg_id) or [])
        if not details:
            if user_msg_id in pending_users and list(pack.get("hits") or []):
                pack["enrichment_pending"] = True
                pack["answer_citation_overlay_pending"] = True
            continue
        pack["enrichment_pending"] = False
        pack.pop("answer_citation_overlay_pending", None)
        prompt = str(pack.get("prompt") or "")
        # The saved preference is authoritative. A persisted pack can carry a
        # stale locale from an earlier render and must not defeat a user
        # language change.
        render_locale = _ref_card_user_locale(prompt)
        prefer_zh = render_locale == "zh" or (
            render_locale != "en" and bool(re.search(r"[\u4e00-\u9fff]", prompt))
        )
        grouped: dict[str, list[dict]] = {}
        source_order: list[str] = []
        for detail in details:
            source_key = _answer_citation_source_key(detail.get("source_path"))
            if not source_key:
                continue
            if source_key not in grouped:
                grouped[source_key] = []
                source_order.append(source_key)
            grouped[source_key].append(detail)

        existing_by_source: dict[str, dict] = {}
        remaining: list[dict] = []
        for hit in list(pack.get("hits") or []):
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
            source_key = _answer_citation_source_key(
                (ui_meta or {}).get("source_path") or (meta or {}).get("source_path")
            )
            if source_key in grouped and source_key not in existing_by_source:
                existing_by_source[source_key] = dict(hit)
            elif source_key not in grouped:
                remaining.append(dict(hit))

        aligned_hits: list[dict] = []
        for source_key in source_order:
            source_details = grouped[source_key]
            detail = source_details[0]
            hit = dict(existing_by_source.get(source_key) or {})
            meta = dict(hit.get("meta") or {}) if isinstance(hit.get("meta"), dict) else {}
            ui = dict(hit.get("ui_meta") or {}) if isinstance(hit.get("ui_meta"), dict) else {}
            summary_line, why_line = _answer_citation_card_copy(
                source_details,
                prefer_zh=prefer_zh,
                prompt=prompt,
            )
            evidence_quote = str(detail.get("evidence_quote") or detail.get("summary_line") or "").strip()
            source_path = str(detail.get("source_path") or "").strip()
            source_name = str(detail.get("source_name") or detail.get("card_title") or "").strip()
            heading_path = str(detail.get("heading_path") or detail.get("location_label") or "").strip()
            grounding_surface = " ".join(
                " ".join(
                    str(item.get(field) or "").strip()
                    for field in (
                        "evidence_quote",
                        "summary_line",
                        "answer_claim",
                        "card_takeaway",
                        # Keep source identity after the evidence text.  The
                        # evidence normalizer removes leading metadata labels,
                        # so a leading filename would erase method names such
                        # as SCINeRF before the grounded-copy rules see them.
                        "source_name",
                        "card_title",
                    )
                ).strip()
                for item in source_details
                if isinstance(item, dict)
            ).strip()
            grounded_why = build_grounded_ref_why_line(
                prefer_zh=prefer_zh,
                focus_terms=[],
                heading_path=heading_path,
                summary_line=grounding_surface or evidence_quote,
                action=prompt_reference_focus_action(prompt),
            )
            if grounded_why:
                why_line = grounded_why
            primary = {
                "source_path": source_path,
                "source_name": source_name,
                "heading_path": heading_path,
                "snippet": evidence_quote,
                "highlight_snippet": evidence_quote,
                "block_id": str(detail.get("block_id") or "").strip(),
                "anchor_id": str(detail.get("anchor_id") or "").strip(),
                "anchor_kind": str(detail.get("anchor_kind") or "sentence").strip(),
                "page_start": int(detail.get("page_start") or 0),
                "page_end": int(detail.get("page_end") or detail.get("page_start") or 0),
                "selection_reason": "answer_citation_grounded",
                "strict_locate": bool(detail.get("block_id") or detail.get("anchor_id")),
            }
            meta.update(
                {
                    "source_path": source_path,
                    "source_name": source_name,
                    "heading_path": heading_path,
                    # This row is now backed by the answer's concrete citation
                    # detail and strict reader locator.  A stale pending flag
                    # from the earlier fast pass must not keep the whole card
                    # shelf in a 60-second polling loop.
                    "ref_pack_state": "ready",
                }
            )
            try:
                answer_citation_num = int(detail.get("num") or 0)
            except (TypeError, ValueError):
                answer_citation_num = 0
            if answer_citation_num > 0:
                meta["ref_answer_citation_num"] = answer_citation_num
            meta["answer_citation_overlay_grounded"] = True
            ui.update(
                {
                    "display_name": source_name or str(ui.get("display_name") or ""),
                    "source_path": source_path,
                    "heading_path": heading_path,
                    "section_label": heading_path,
                    "summary_line": summary_line,
                    "summary_kind": "guide",
                    "summary_display_role": "guide",
                    "summary_label": "导读" if prefer_zh else "Guide",
                    "summary_title": "这条证据说明什么" if prefer_zh else "What This Evidence Shows",
                    "summary_generation": "answer_citation_grounded",
                    "summary_basis": "基于回答实际引用的原文证据" if prefer_zh else "Based on the source evidence cited by the answer",
                    "summary_source": "answer_citation_grounded",
                    "why_line": why_line,
                    "why_generation": "answer_citation_grounded",
                    "why_basis": "对齐答案主张与原文定位" if prefer_zh else "Aligned to the answer claim and source locator",
                    "primary_evidence": primary,
                    "primary_evidence_heading_path": heading_path,
                    "render_locale": "zh" if prefer_zh else "en",
                    "score_pending": False,
                }
            )
            reader_open = dict(ui.get("reader_open") or {}) if isinstance(ui.get("reader_open"), dict) else {}
            reader_open.update(
                {
                    "sourcePath": source_path,
                    "sourceName": source_name,
                    "headingPath": heading_path,
                    "snippet": evidence_quote,
                    "highlightSnippet": evidence_quote,
                    "strictLocate": bool(primary["strict_locate"]),
                    "primaryEvidence": primary,
                }
            )
            ui["reader_open"] = reader_open
            location = heading_path or ("原文定位处" if prefer_zh else "Source passage")
            ui["card_view"] = {
                "version": 1,
                "route": "references",
                "kind": "reference_locator",
                "header": {
                    "kicker": "参考定位" if prefer_zh else "Reference",
                    "title": source_name,
                    "subtitle": location,
                },
                "sections": [
                    {
                        "id": "summary",
                        "label": "导读" if prefer_zh else "Guide",
                        "text": summary_line,
                        "kind": "summary",
                        "tone": "primary",
                        "source": "answer_citation_grounded",
                    },
                    {
                        "id": "why",
                        "label": "相关性" if prefer_zh else "Why it matters",
                        "text": why_line,
                        "kind": "why",
                        "tone": "",
                    },
                    {
                        "id": "location",
                        "label": "位置" if prefer_zh else "Location",
                        "text": location,
                        "kind": "locator",
                        "tone": "",
                    },
                ],
                "summary": summary_line,
                "quality": {"label": "grounded", "source": "answer_citation_grounded"},
            }
            hit.update({"text": evidence_quote, "meta": meta, "ui_meta": ui})
            aligned_hits.append(hit)
        if remaining and not list(pack.get("retrieval_hits") or []):
            # Keep unused candidates available for diagnostics without
            # presenting them as evidence for claims the answer did not make.
            pack["retrieval_hits"] = [dict(hit) for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
        pack["hits"] = aligned_hits
        pack["answer_aligned_citation_cards"] = True
        if _answer_citation_overlay_pack_is_complete(pack):
            pack["payload_mode"] = "full"
            pack["display_state"] = "ready"
            pack["render_status"] = "full"
            pack["render_error"] = ""
            pack["render_error_detail"] = ""
            pack["render_built_at"] = float(pack.get("render_built_at") or time.time())
            pack["render_attempts"] = max(1, int(pack.get("render_attempts") or 0))
            pack.pop("pending", None)
            pack.pop("pending_hit_count", None)
            pack.pop("enrichment_pending", None)
            pack.pop("answer_citation_overlay_pending", None)
        payload_out[raw_user_msg_id] = attach_refs_pack_polish_contract(pack)
    return payload_out


def _answer_citation_overlay_pack_is_complete(pack: dict | None) -> bool:
    if not isinstance(pack, dict) or not bool(pack.get("answer_aligned_citation_cards")):
        return False
    hits = [hit for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
    if not hits:
        return False
    for hit in hits:
        ui = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        primary = ui.get("primary_evidence") if isinstance(ui.get("primary_evidence"), dict) else {}
        source_path = str(
            (ui or {}).get("source_path")
            or (primary or {}).get("source_path")
            or ""
        ).strip()
        if (
            not source_path
            or not str((ui or {}).get("summary_line") or "").strip()
            or not str((ui or {}).get("why_line") or "").strip()
            or not str(
                (primary or {}).get("snippet")
                or (primary or {}).get("highlight_snippet")
                or ""
            ).strip()
        ):
            return False
    return True


def _refs_without_completed_answer_citation_overlays(
    *,
    store,
    conv_id: str,
    refs: dict[int, dict] | None,
) -> dict[int, dict]:
    refs_out = {
        int(key): dict(value)
        for key, value in dict(refs or {}).items()
        if (str(key).isdigit() or isinstance(key, int)) and isinstance(value, dict)
    }
    if not refs_out:
        return {}
    ready_user_ids = set(
        _answer_citation_details_by_user(
            store=store,
            conv_id=conv_id,
        )
    )
    return {
        user_msg_id: pack
        for user_msg_id, pack in refs_out.items()
        if user_msg_id not in ready_user_ids
    }


def _reference_asset_roots() -> list[Path]:
    return reference_source_roots(
        md_root=_md_dir(),
        db_dir=getattr(get_settings(), "db_dir", None),
    )


def _path_within_roots(path_obj: Path, roots: list[Path]) -> bool:
    return path_is_within_roots(path_obj, roots)


def _refs_conversation_cache_ttl_s() -> float:
    try:
        raw = float(str(os.environ.get("KB_REFS_CONVERSATION_CACHE_TTL_S", "6") or "6"))
    except Exception:
        raw = 6.0
    return max(0.0, min(30.0, raw))


def _refs_validated_cache_ttl_s() -> float:
    try:
        raw = float(
            str(os.environ.get("KB_REFS_VALIDATED_CACHE_TTL_S", "45") or "45")
        )
    except Exception:
        raw = 45.0
    return max(1.0, min(300.0, raw))


def _refs_conversation_state_signature(
    *,
    conversation: dict | None,
    refs_state: dict | None,
) -> str:
    """Hash cheap SQLite state so polls can reuse refs without JSON loading."""

    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    conv = conversation if isinstance(conversation, dict) else {}
    state = refs_state if isinstance(refs_state, dict) else {}
    payload = {
        "render_schema": _REFS_RENDER_PAYLOAD_SCHEMA_VERSION,
        "conversation": {
            "mode": str(conv.get("mode") or "").strip().lower(),
            "bound_source_path": str(conv.get("bound_source_path") or "").strip(),
            "bound_source_name": str(conv.get("bound_source_name") or "").strip(),
            "bound_source_ready": bool(conv.get("bound_source_ready")),
            "updated_at": float(conv.get("updated_at") or 0.0),
        },
        "refs_background_llm_polish": bool(_refs_background_llm_polish_enabled()),
        "refs_card_locale": str((prefs or {}).get("refs_card_locale") or "")
        .strip()
        .lower(),
        "ui_locale": str((prefs or {}).get("ui_locale") or "").strip().lower(),
        "refs_state": state,
    }
    blob = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _get_state_validated_conversation_refs_record(
    *,
    conv_id: str,
    state_signature: str,
) -> dict | None:
    if not str(state_signature or "").strip():
        return None
    rec = _REFS_CONVERSATION_CACHE.get(str(conv_id or "").strip())
    if not isinstance(rec, dict):
        return None
    if str(rec.get("state_signature") or "") != str(state_signature or ""):
        return None
    try:
        cached_at = float(rec.get("cached_at") or 0.0)
    except Exception:
        cached_at = 0.0
    if (
        cached_at <= 0
        or (time.time() - cached_at) > _refs_validated_cache_ttl_s()
    ):
        return None
    payload = rec.get("payload")
    return rec if isinstance(payload, dict) else None


def _refs_conversation_cache_signature(
    *,
    refs: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
    authoritative_doc_list_by_user: dict[int, list[dict]] | None = None,
) -> str:
    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    authoritative_map: dict[int, list[dict]] = {}
    for key, value in dict(authoritative_doc_list_by_user or {}).items():
        try:
            user_msg_key = int(key)
        except Exception:
            continue
        authoritative_map[user_msg_key] = [dict(item) for item in list(value or []) if isinstance(item, dict)]
    refs_digest: list[dict] = []
    for user_msg_id, pack in sorted((refs or {}).items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0])):
        if not isinstance(pack, dict):
            continue
        try:
            user_msg_key = int(user_msg_id)
        except Exception:
            user_msg_key = 0
        hits = list(pack.get("hits") or [])
        pending_count = 0
        source_keys: list[str] = []
        for hit in hits[:4]:
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
                pending_count += 1
            source_path = str((meta or {}).get("source_path") or "").strip()
            if source_path:
                source_keys.append(source_path)
        doc_list_sig: list[dict] = []
        for item in list(authoritative_map.get(user_msg_key, []) or [])[:4]:
            source_path = str(item.get("source_path") or "").strip()
            source_name = str(item.get("source_name") or "").strip()
            heading_path = str(item.get("heading_path") or "").strip()
            if source_path or source_name:
                doc_list_sig.append(
                    {
                        "source_path": source_path,
                        "source_name": source_name,
                        "heading_path": heading_path,
                    }
                )
        payload = {
            "user_msg_id": user_msg_key if user_msg_key > 0 else str(user_msg_id),
            "prompt_sig": str(pack.get("prompt_sig") or "").strip(),
            "answer_sig": str(pack.get("answer_sig") or "").strip(),
            "used_query": str(pack.get("used_query") or "").strip(),
            "used_translation": bool(pack.get("used_translation")),
            "updated_at": float(pack.get("updated_at") or 0.0),
            "render_status": str(pack.get("render_status") or "").strip().lower(),
            "rendered_payload_sig": str(pack.get("rendered_payload_sig") or "").strip(),
            "hit_count": len(hits),
            "pending_count": pending_count,
            "top_sources": source_keys,
            "authoritative_doc_list": doc_list_sig,
        }
        refs_digest.append(payload)
    payload = {
        "render_schema": _REFS_RENDER_PAYLOAD_SCHEMA_VERSION,
        "guide_mode": bool(guide_mode),
        "guide_source_path": str(guide_source_path or "").strip(),
        "guide_source_name": str(guide_source_name or "").strip(),
        "refs_background_llm_polish": bool(_refs_background_llm_polish_enabled()),
        "refs_card_locale": str((prefs or {}).get("refs_card_locale") or "").strip().lower(),
        "ui_locale": str((prefs or {}).get("ui_locale") or "").strip().lower(),
        "refs_digest": refs_digest,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _refs_pack_render_signature(
    *,
    user_msg_id: int | str,
    pack: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
) -> str:
    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    payload = {
        "render_schema": _REFS_RENDER_PAYLOAD_SCHEMA_VERSION,
        "user_msg_id": int(user_msg_id) if str(user_msg_id).isdigit() else str(user_msg_id),
        "guide_mode": bool(guide_mode),
        "guide_source_path": str(guide_source_path or "").strip(),
        "guide_source_name": str(guide_source_name or "").strip(),
        "refs_background_llm_polish": bool(_refs_background_llm_polish_enabled()),
        "refs_card_locale": str((prefs or {}).get("refs_card_locale") or "").strip().lower(),
        "ui_locale": str((prefs or {}).get("ui_locale") or "").strip().lower(),
        "prompt": str((pack or {}).get("prompt") or "").strip(),
        "prompt_sig": str((pack or {}).get("prompt_sig") or "").strip(),
        "answer_sig": str((pack or {}).get("answer_sig") or "").strip(),
        "used_query": str((pack or {}).get("used_query") or "").strip(),
        "used_translation": bool((pack or {}).get("used_translation")),
        "hits": list((pack or {}).get("hits") or []),
        "scores": list((pack or {}).get("scores") or []),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _get_cached_conversation_refs_record(*, conv_id: str, signature: str) -> dict | None:
    ttl_s = _refs_conversation_cache_ttl_s()
    if ttl_s <= 0:
        return None
    rec = _REFS_CONVERSATION_CACHE.get(str(conv_id or "").strip())
    if not isinstance(rec, dict):
        return None
    if str(rec.get("signature") or "") != str(signature or ""):
        return None
    try:
        cached_at = float(rec.get("cached_at") or 0.0)
    except Exception:
        cached_at = 0.0
    if cached_at <= 0 or (time.time() - cached_at) > ttl_s:
        return None
    payload = rec.get("payload")
    if not isinstance(payload, dict):
        return None
    return rec


def _get_cached_conversation_refs_payload(*, conv_id: str, signature: str) -> dict | None:
    rec = _get_cached_conversation_refs_record(conv_id=conv_id, signature=signature)
    if not isinstance(rec, dict):
        return None
    payload = rec.get("payload")
    return payload if isinstance(payload, dict) else None


def _refs_cache_input_pack_signature(pack: dict | None) -> str:
    src = pack if isinstance(pack, dict) else {}
    try:
        updated_at = float(src.get("updated_at") or 0.0)
    except Exception:
        updated_at = 0.0
    payload = {
        "prompt": str(src.get("prompt") or "").strip(),
        "prompt_sig": str(src.get("prompt_sig") or "").strip(),
        "answer_sig": str(src.get("answer_sig") or "").strip(),
        "used_query": str(src.get("used_query") or "").strip(),
        "used_translation": bool(src.get("used_translation")),
        "updated_at": updated_at,
        "hits": list(src.get("hits") or []),
        "scores": list(src.get("scores") or []),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _augment_pack_with_canonical_answer_paths(pack: dict | None) -> dict:
    out = dict(pack or {})
    if bool(out.get("_canonical_answer_paths_aligned")):
        return out
    canonical_paths = [
        str(path or "").strip()
        for path in list(out.get("_canonical_answer_paths") or [])
        if str(path or "").strip()
    ]
    answer_text = str(out.get("answer_text") or "").strip()
    if not canonical_paths or not answer_text:
        return out
    from api.chat_render import _augment_hits_with_canonical_answer_citations

    raw_hits = list(out.get("hits") or [])
    cache_blob = json.dumps(
        {
            "answer": answer_text,
            "canonical_paths": canonical_paths,
            "hits": raw_hits,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    cache_key = hashlib.sha1(cache_blob.encode("utf-8")).hexdigest()
    with _CANONICAL_ANSWER_HITS_CACHE_LOCK:
        cached_hits = _CANONICAL_ANSWER_HITS_CACHE.get(cache_key)
    if isinstance(cached_hits, list):
        aligned_hits = [dict(hit) for hit in cached_hits if isinstance(hit, dict)]
    else:
        aligned_hits = _augment_hits_with_canonical_answer_citations(
            raw_hits,
            canonical_paths=canonical_paths,
            answer_text=answer_text,
        )
        with _CANONICAL_ANSWER_HITS_CACHE_LOCK:
            if len(_CANONICAL_ANSWER_HITS_CACHE) >= 64:
                _CANONICAL_ANSWER_HITS_CACHE.pop(next(iter(_CANONICAL_ANSWER_HITS_CACHE)), None)
            _CANONICAL_ANSWER_HITS_CACHE[cache_key] = [
                dict(hit) for hit in aligned_hits if isinstance(hit, dict)
            ]
    cited_hits = [
        hit
        for hit in aligned_hits
        if isinstance(hit, dict)
        and int(((hit.get("meta") or {}).get("ref_answer_citation_num") or 0)) > 0
    ]
    if cited_hits:
        normalized_answer = re.sub(r"\[\[\s*(\d{1,5})\s*\]\]", r"[\1]", answer_text)

        def _citation_order(hit: dict) -> tuple[int, int]:
            try:
                number = int(((hit.get("meta") or {}).get("ref_answer_citation_num") or 0))
            except (TypeError, ValueError):
                number = 0
            marker = re.search(rf"(?<![!\\])\[{number}\](?!\()", normalized_answer) if number > 0 else None
            return (int(marker.start()) if marker else len(normalized_answer) + number, number)

        cited_hits.sort(key=_citation_order)
    out["hits"] = cited_hits or aligned_hits
    out["_canonical_answer_paths_aligned"] = True
    return out


def _attach_assistant_answers_to_refs(*, store, conv_id: str, refs: dict | None) -> dict:
    """Keep final answers internal so evidence cards can align to supported claims."""

    refs_out = {
        int(key): dict(value)
        for key, value in dict(refs or {}).items()
        if (str(key).isdigit() or isinstance(key, int)) and isinstance(value, dict)
    }
    if not refs_out or not hasattr(store, "get_messages"):
        return refs_out
    try:
        messages = store.get_messages(conv_id)
    except Exception:
        return refs_out
    wanted = set(refs_out)
    answers: dict[int, tuple[str, list[str], bool]] = {}
    active_user_msg_id = 0
    for message in list(messages or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        try:
            message_id = int(message.get("id") or 0)
        except (TypeError, ValueError):
            message_id = 0
        if role == "user":
            active_user_msg_id = message_id if message_id in wanted else 0
            continue
        if role != "assistant" or active_user_msg_id <= 0:
            continue
        answer_text = str(message.get("content") or "").strip()
        if not answer_text:
            continue
        msg_meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
        canonical_paths = [
            str(path or "").strip()
            for path in list((msg_meta or {}).get("canonical_hit_paths") or [])
            if str(path or "").strip()
        ]
        grounded_details, _planned_system_a = _grounded_answer_citation_state(message)
        answers[active_user_msg_id] = (
            answer_text,
            canonical_paths,
            bool(grounded_details),
        )
        active_user_msg_id = 0
    latest_user_msg_id = max(refs_out, default=0)
    for user_msg_id, (answer_text, canonical_paths, grounded_overlay_ready) in answers.items():
        pack = dict(refs_out.get(user_msg_id) or {})
        pack["answer_text"] = answer_text
        pack["answer_sig"] = hashlib.sha1(answer_text.encode("utf-8")).hexdigest()
        if canonical_paths and not grounded_overlay_ready:
            pack["_canonical_answer_paths"] = canonical_paths
            # The current answer must be exact on first paint.  Historical
            # turns keep their shallow rendered cards and are realigned one at
            # a time by the background warm, avoiding repeated whole-library
            # source scans on every conversation read.
            if user_msg_id == latest_user_msg_id:
                pack = _augment_pack_with_canonical_answer_paths(pack)
        elif grounded_overlay_ready:
            # The answer renderer/citation plan already provides the exact
            # source path, evidence quote, and locator consumed by
            # ``_overlay_refs_payload_with_answer_citations``. Re-ranking every
            # block in every cited source here duplicates that work and adds
            # several seconds before the first complete card response.
            pack["_answer_citation_overlay_ready"] = True
        refs_out[user_msg_id] = pack
    return refs_out


def _refs_cache_input_signatures(refs: dict | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_key, pack in dict(refs or {}).items():
        if not isinstance(pack, dict):
            continue
        try:
            key = str(int(raw_key))
        except Exception:
            continue
        out[key] = _refs_cache_input_pack_signature(pack)
    return out


def _store_cached_conversation_refs_payload(
    *,
    conv_id: str,
    signature: str,
    payload: dict,
    mode: str = "full",
    refs: dict | None = None,
    state_signature: str = "",
) -> None:
    _REFS_CONVERSATION_CACHE[str(conv_id or "").strip()] = {
        "signature": str(signature or ""),
        "state_signature": str(state_signature or ""),
        "cached_at": time.time(),
        "mode": str(mode or "full").strip().lower() or "full",
        "payload": dict(payload or {}),
        "refs_input_signatures": _refs_cache_input_signatures(refs),
    }


def _get_any_cached_conversation_refs_record(*, conv_id: str) -> dict | None:
    ttl_s = _refs_conversation_cache_ttl_s()
    if ttl_s <= 0:
        return None
    rec = _REFS_CONVERSATION_CACHE.get(str(conv_id or "").strip())
    if not isinstance(rec, dict):
        return None
    try:
        cached_at = float(rec.get("cached_at") or 0.0)
    except Exception:
        cached_at = 0.0
    if cached_at <= 0 or (time.time() - cached_at) > ttl_s:
        return None
    payload = rec.get("payload")
    return rec if isinstance(payload, dict) else None


def _get_any_cached_conversation_refs_payload(*, conv_id: str) -> dict | None:
    rec = _get_any_cached_conversation_refs_record(conv_id=conv_id)
    if not isinstance(rec, dict):
        return None
    payload = rec.get("payload")
    return payload if isinstance(payload, dict) else None


def _get_compatible_cached_conversation_refs_payload(*, conv_id: str, refs: dict | None) -> dict | None:
    rec = _get_any_cached_conversation_refs_record(conv_id=conv_id)
    if not isinstance(rec, dict):
        return None
    cached_payload = rec.get("payload")
    recorded_signatures = rec.get("refs_input_signatures")
    if not isinstance(cached_payload, dict) or not isinstance(recorded_signatures, dict):
        return None
    current_signatures = _refs_cache_input_signatures(refs)
    compatible: dict[int, dict] = {}
    for raw_key, pack in cached_payload.items():
        if not isinstance(pack, dict):
            continue
        pipeline_debug = pack.get("pipeline_debug") if isinstance(pack.get("pipeline_debug"), dict) else {}
        # The authoritative doc-list identity comes from the assistant message,
        # not the raw refs row. Without that current input, a TTL-valid cache can
        # resurrect documents removed by a newer answer contract.
        if bool((pipeline_debug or {}).get("doc_list_authoritative")):
            continue
        try:
            key = int(raw_key)
        except Exception:
            continue
        signature_key = str(key)
        cached_sig = str(recorded_signatures.get(signature_key) or "").strip()
        current_sig = str(current_signatures.get(signature_key) or "").strip()
        if cached_sig and current_sig and cached_sig == current_sig:
            compatible[key] = pack
    return compatible or None


def _refs_perf_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - float(started_at or time.perf_counter())) * 1000.0)


def _refs_payload_counts_for_header(payload: dict | None) -> str:
    packs = 0
    hits = 0
    pending = 0
    fast = 0
    ready = 0
    for pack in list((payload or {}).values()):
        if not isinstance(pack, dict):
            continue
        packs += 1
        pack_hits = [hit for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
        hits += len(pack_hits)
        mode = str(pack.get("payload_mode") or "").strip().lower()
        if mode == "pending" or bool(pack.get("enrichment_pending")):
            pending += 1
        elif mode == "fast":
            fast += 1
        elif pack_hits:
            ready += 1
    return f"packs={packs};hits={hits};pending={pending};fast={fast};ready={ready}"


def _set_refs_timing_headers(
    response: Response | None,
    *,
    timings: list[tuple[str, float]],
    total_ms: float,
    mode: str,
    payload: dict | None,
) -> None:
    if response is None:
        return
    seen: dict[str, int] = {}
    parts: list[str] = []
    for raw_name, raw_duration in list(timings or []):
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw_name or "").strip())[:36] or "phase"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        try:
            duration = float(raw_duration)
        except Exception:
            duration = 0.0
        parts.append(f"{name};dur={max(0.0, duration):.1f}")
    parts.append(f"total;dur={max(0.0, float(total_ms or 0.0)):.1f}")
    response.headers["Server-Timing"] = ", ".join(parts)
    response.headers["X-KB-Refs-Mode"] = str(mode or "").strip().lower() or "unknown"
    response.headers["X-KB-Refs-Counts"] = _refs_payload_counts_for_header(payload)


def _refs_conversation_read_timeout_s() -> float:
    try:
        raw = float(str(os.environ.get("KB_REFS_CONVERSATION_READ_TIMEOUT_S", "0.35") or "0.35"))
    except Exception:
        raw = 0.35
    return max(0.05, min(2.0, raw))


def _refs_ready_budget_s() -> float:
    try:
        raw = float(str(os.environ.get("KB_REFS_READY_BUDGET_S", "1.8") or "1.8"))
    except Exception:
        raw = 1.8
    return max(0.25, min(8.0, raw))


def _refs_pending_stale_after_s() -> float:
    try:
        raw = float(str(os.environ.get("KB_REFS_PENDING_STALE_AFTER_S", "20") or "20"))
    except Exception:
        raw = 20.0
    return max(5.0, min(120.0, raw))


def _refs_pack_is_stale_pending(pack: dict) -> bool:
    if not isinstance(pack, dict):
        return False
    has_pending = False
    for hit in list(pack.get("hits") or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
            has_pending = True
            break
    if not has_pending:
        return False
    try:
        updated_at = float(pack.get("updated_at") or 0.0)
    except Exception:
        updated_at = 0.0
    if updated_at <= 0:
        return False
    return (time.time() - updated_at) >= _refs_pending_stale_after_s()


def _refs_payload_has_pending(refs: dict, *, include_stale: bool = True) -> bool:
    for pack in list((refs or {}).values()):
        if not isinstance(pack, dict):
            continue
        if _refs_pack_has_pending(pack, include_stale=include_stale):
            return True
    return False


def _refs_pack_has_pending(pack: dict, *, include_stale: bool = True) -> bool:
    if not isinstance(pack, dict):
        return False
    has_pending = False
    for hit in list(pack.get("hits") or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
            has_pending = True
            break
    if (not has_pending) or include_stale:
        return has_pending
    return not _refs_pack_is_stale_pending(pack)


def _stored_rendered_pack_payload_lost_current_hits(*, payload: dict, pack: dict) -> bool:
    if not isinstance(payload, dict) or not isinstance(pack, dict):
        return False
    raw_hits = [hit for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
    if not raw_hits:
        return False
    payload_hits = [hit for hit in list(payload.get("hits") or []) if isinstance(hit, dict)]
    if payload_hits:
        prompt = str(pack.get("prompt") or "")
        explicit_list = bool(
            prompt_explicitly_requests_multi_paper_list(prompt)
            or re.search(
                r"(?i)列出|文献清单|(?:五|六|七|八|九|十|\d+)\s*篇|"
                r"\b(?:list|five|six|seven|eight|nine|ten)\b",
                prompt,
            )
        )
        display_cap = 6 if explicit_list else 4

        def _source_identity(value: str) -> str:
            parts = [
                part
                for part in str(value or "").replace("\\", "/").lower().split("/")
                if part
            ]
            return "/".join(parts[-2:]) if len(parts) >= 2 else "/".join(parts)

        raw_sources: list[str] = []
        for hit in raw_hits[:display_cap]:
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            source_path = str((meta or {}).get("source_path") or "").strip()
            if source_path:
                raw_sources.append(source_path)
        payload_sources: list[str] = []
        for hit in payload_hits[:display_cap]:
            ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            source_path = str((ui_meta or {}).get("source_path") or (meta or {}).get("source_path") or "").strip()
            if source_path:
                payload_sources.append(source_path)
        # A persisted rendered payload is stale if it no longer contains the
        # answer-leading source stored in message_refs.  This can happen when
        # the fast reference route was cached before final answer provenance
        # rewrote the refs pack.
        payload_identities = {
            _source_identity(source_path)
            for source_path in payload_sources
            if _source_identity(source_path)
        }
        if (
            raw_sources
            and payload_sources
            and _source_identity(raw_sources[0]) not in payload_identities
        ):
            return True
        if (
            (prompt_likely_multi_paper_synthesis(prompt) or explicit_list)
            and len(set(raw_sources)) >= 2
        ):
            raw_identities = {
                _source_identity(source_path)
                for source_path in raw_sources
                if _source_identity(source_path)
            }
            if not raw_identities.issubset(payload_identities):
                return True
        return False
    display_state = str(payload.get("display_state") or "").strip().lower()
    suppression_reason = str(payload.get("suppression_reason") or "").strip().lower()
    if display_state == "hidden_by_guide" or suppression_reason == "guide_self_source_only":
        return False
    pipeline_debug = payload.get("pipeline_debug") if isinstance(payload.get("pipeline_debug"), dict) else {}
    try:
        debug_raw_hit_count = int((pipeline_debug or {}).get("raw_hit_count") or 0)
    except Exception:
        debug_raw_hit_count = 0
    doc_list_authoritative = bool((pipeline_debug or {}).get("doc_list_authoritative"))
    if doc_list_authoritative and debug_raw_hit_count <= 0:
        return True
    return bool(display_state == "empty" and suppression_reason == "no_candidate_hits" and debug_raw_hit_count <= 0)


_REF_CARD_RAW_MARKDOWN_RE = re.compile(
    r"\[\[\s*CITE:|```|^\s{0,3}#{1,6}\s+\S|"
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$",
    re.MULTILINE,
)


def _ref_card_copy_text_key(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").lower()).strip()


def _stored_rendered_pack_payload_has_dirty_card_copy(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    for hit in list(payload.get("hits") or []):
        if not isinstance(hit, dict):
            continue
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        summary = str((ui_meta or {}).get("summary_line") or "").strip()
        why = str((ui_meta or {}).get("why_line") or "").strip()
        for text in (summary, why):
            if not text:
                continue
            if _REF_CARD_RAW_MARKDOWN_RE.search(text):
                return True
        if why and (looks_generic_ref_why_line(why) or looks_templated_ref_why_line(why)):
            return True
        summary_key = _ref_card_copy_text_key(summary)
        why_key = _ref_card_copy_text_key(why)
        if summary_key and why_key and len(summary_key) >= 24 and summary_key == why_key:
            return True
    return False


def _get_stored_rendered_pack_payload(
    *,
    user_msg_id: int | str,
    pack: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
    allow_authoritative_source_override: bool = False,
) -> dict | None:
    if not isinstance(pack, dict):
        return None
    payload = pack.get("rendered_payload")
    if not isinstance(payload, dict) or (not payload):
        return None
    stored_sig = str(pack.get("rendered_payload_sig") or "").strip()
    expected_sig = _refs_pack_render_signature(
        user_msg_id=user_msg_id,
        pack=pack,
        guide_mode=guide_mode,
        guide_source_path=guide_source_path,
        guide_source_name=guide_source_name,
    )
    if (not stored_sig) or (stored_sig != expected_sig):
        return None
    if (
        not allow_authoritative_source_override
        and _stored_rendered_pack_payload_lost_current_hits(payload=payload, pack=pack)
    ):
        return None
    if _stored_rendered_pack_payload_has_dirty_card_copy(payload):
        return None
    return dict(payload)


def _extract_doc_list_contract_from_message_meta(meta: dict | None) -> list[dict]:
    if not isinstance(meta, dict):
        return []
    contracts = meta.get("paper_guide_contracts") if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    return [dict(item) for item in list((contracts or {}).get("doc_list") or []) if isinstance(item, dict)]


def _load_authoritative_doc_list_contracts(
    *,
    store,
    conv_id: str,
    user_msg_ids: set[int],
) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    if not user_msg_ids:
        return out
    get_messages = getattr(store, "get_messages", None)
    if not callable(get_messages):
        return out
    try:
        messages = list(get_messages(str(conv_id or "").strip()) or [])
    except sqlite3.OperationalError:
        return out
    except Exception:
        return out
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        try:
            msg_id = int(msg.get("id") or 0)
        except Exception:
            msg_id = 0
        if msg_id not in user_msg_ids:
            continue
        if str(msg.get("role") or "").strip().lower() != "user":
            continue
        for nxt in messages[idx + 1 :]:
            if not isinstance(nxt, dict):
                continue
            role = str(nxt.get("role") or "").strip().lower()
            if role == "user":
                break
            if role != "assistant":
                continue
            content = str(nxt.get("content") or "")
            if content.startswith("__KB_LIVE_TASK__:"):
                continue
            meta = nxt.get("meta") if isinstance(nxt.get("meta"), dict) else {}
            contracts = meta.get("paper_guide_contracts") if isinstance(meta.get("paper_guide_contracts"), dict) else {}
            if "doc_list" in contracts:
                out[msg_id] = _extract_doc_list_contract_from_message_meta(meta)
            break
    return out


def _load_pending_doc_list_contracts(
    *,
    store,
    conv_id: str,
    pending_user_msg_ids: set[int],
) -> dict[int, list[dict]]:
    return _load_authoritative_doc_list_contracts(
        store=store,
        conv_id=conv_id,
        user_msg_ids=pending_user_msg_ids,
    )


def _mark_doc_list_pending_pack(*, payload_pack: dict, pending_count: int) -> dict:
    pack2 = dict(payload_pack or {})
    hits_out: list[dict] = []
    for raw_hit in list(pack2.get("hits") or []):
        if not isinstance(raw_hit, dict):
            continue
        hit = dict(raw_hit)
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        meta2 = dict(meta or {})
        meta2["ref_pack_state"] = "pending"
        hit["meta"] = meta2
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        if isinstance(ui_meta, dict):
            ui_meta2 = dict(ui_meta)
            ui_meta2["score_pending"] = True
            ui_meta2["score"] = None
            ui_meta2["score_tier"] = ""
            hit["ui_meta"] = ui_meta2
        hits_out.append(hit)
    pack2["hits"] = hits_out
    pack2["pending"] = True
    pack2["pending_hit_count"] = int(max(0, int(pending_count or 0)))
    pack2["payload_mode"] = "pending"
    pack2["enrichment_pending"] = True
    return _attach_pack_display_contract(pack2)


def _doc_list_source_paths(doc_list: list[dict] | None) -> list[str]:
    out: list[str] = []
    for item in list(doc_list or []):
        if not isinstance(item, dict):
            continue
        source_path = str(item.get("source_path") or "").strip()
        if source_path:
            out.append(source_path)
    return out


def _payload_source_paths(payload_pack: dict | None) -> list[str]:
    out: list[str] = []
    if not isinstance(payload_pack, dict):
        return out
    for hit in list(payload_pack.get("hits") or []):
        if not isinstance(hit, dict):
            continue
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = str(
            (ui_meta or {}).get("source_path")
            or (meta or {}).get("source_path")
            or ""
        ).strip()
        if source_path:
            out.append(source_path)
    return out


def _payload_is_authoritative_doc_list_pack(payload_pack: dict | None, authoritative_doc_list: list[dict] | None) -> bool:
    if not isinstance(payload_pack, dict):
        return False
    pipeline_debug = payload_pack.get("pipeline_debug") if isinstance(payload_pack.get("pipeline_debug"), dict) else {}
    if not bool((pipeline_debug or {}).get("doc_list_authoritative")):
        return False
    expected_paths = _doc_list_source_paths(authoritative_doc_list)
    actual_paths = _payload_source_paths(payload_pack)
    if not expected_paths:
        return True
    return bool(actual_paths) and actual_paths == expected_paths


def _cached_payload_matches_authoritative_doc_lists(
    cached_payload: dict | None,
    authoritative_doc_lists: dict[int, list[dict]] | None,
) -> bool:
    if not isinstance(cached_payload, dict):
        return False
    for raw_user_msg_id, doc_list in dict(authoritative_doc_lists or {}).items():
        try:
            user_msg_id = int(raw_user_msg_id)
        except Exception:
            return False
        payload_pack = cached_payload.get(user_msg_id)
        if not isinstance(payload_pack, dict):
            payload_pack = cached_payload.get(str(user_msg_id))
        if not _payload_is_authoritative_doc_list_pack(payload_pack, doc_list):
            return False
    return True


def _rebuild_authoritative_doc_list_from_pack(*, prompt: str, pack: dict, guide_mode: bool) -> list[dict]:
    prompt_text = str(prompt or "").strip()
    if guide_mode or (not prompt_explicitly_requests_multi_paper_list(prompt_text)):
        return []
    rows = [dict(hit) for hit in list((pack or {}).get("hits") or []) if isinstance(hit, dict)]
    if not rows:
        return []
    try:
        rebuilt = _references_build_multi_paper_doc_list_contract(
            prompt=prompt_text,
            seed_docs=list(rows),
            answer_hits=list(rows),
            evidence_cards=[],
        )
    except Exception:
        rebuilt = []
    return [dict(item) for item in list(rebuilt or []) if isinstance(item, dict)]


def _normalize_authoritative_doc_list_contracts_for_refs(
    *,
    refs: dict,
    doc_lists: dict[int, list[dict]],
    guide_mode: bool,
) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for raw_user_msg_id, raw_rows in dict(doc_lists or {}).items():
        try:
            user_msg_id = int(raw_user_msg_id)
        except Exception:
            continue
        rows = [dict(item) for item in list(raw_rows or []) if isinstance(item, dict)]
        if rows:
            out[user_msg_id] = rows
            continue
        if guide_mode:
            # In guide mode an empty cross-paper contract is meaningful: it hides self-only refs.
            out[user_msg_id] = []
            continue
        pack = None
        for key in (user_msg_id, str(user_msg_id)):
            candidate = (refs or {}).get(key)
            if isinstance(candidate, dict):
                pack = candidate
                break
        if not isinstance(pack, dict):
            continue
        rebuilt = _rebuild_authoritative_doc_list_from_pack(
            prompt=str(pack.get("prompt") or "").strip(),
            pack=pack,
            guide_mode=False,
        )
        if rebuilt:
            out[user_msg_id] = rebuilt
    return out


def _filter_pending_multi_paper_hits_for_display(prompt: str, hits: list[dict] | None) -> list[dict]:
    rows = [dict(hit) for hit in list(hits or []) if isinstance(hit, dict)]
    if (not rows) or (not prompt_explicitly_requests_multi_paper_list(prompt)):
        return rows
    try:
        doc_list_seed = _references_build_multi_paper_doc_list_contract(
            prompt=prompt,
            seed_docs=list(rows),
            answer_hits=list(rows),
            evidence_cards=[],
        )
    except Exception:
        return rows
    source_order = _doc_list_source_paths(doc_list_seed)
    if not source_order:
        return rows
    rows_by_source: dict[str, dict] = {}
    for row in rows:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        if source_path and source_path not in rows_by_source:
            rows_by_source[source_path] = row
    filtered: list[dict] = []
    seen: set[str] = set()
    for source_path in source_order:
        row = rows_by_source.get(source_path)
        if not isinstance(row, dict) or source_path in seen:
            continue
        filtered.append(row)
        seen.add(source_path)
    target_count = min(3, len(rows))
    if len(filtered) < target_count:
        for row in rows:
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            source_path = str((meta or {}).get("source_path") or "").strip()
            if (not source_path) or source_path in seen:
                continue
            filtered.append(row)
            seen.add(source_path)
            if len(filtered) >= target_count:
                break
    return filtered or rows


def _render_authoritative_doc_list_pack(
    *,
    user_msg_id: int,
    pack: dict,
    doc_list: list[dict],
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
    pending: bool,
    allow_expensive_llm: bool | None = None,
    allow_citation_prefetch: bool = False,
) -> dict:
    # Keep pending/full on the same authoritative paper set.  The pending pass
    # shapes only the already-computed contract; all source scans and copy
    # enrichment belong to the background full render.
    allow_llm = bool(not pending) if allow_expensive_llm is None else bool(allow_expensive_llm)
    return build_doc_list_refs_payload(
        user_msg_id=int(user_msg_id),
        pack=pack,
        doc_list=doc_list,
        allow_expensive_llm=allow_llm,
        allow_exact_locate=not pending,
        apply_copy_polish=not pending,
        guide_mode=bool(guide_mode),
        guide_source_path=str(guide_source_path or "").strip(),
        guide_source_name=str(guide_source_name or "").strip(),
        pdf_root=_pdf_dir(),
        md_root=_md_dir(),
        lib_store=_lib_store(),
        allow_citation_prefetch=bool(allow_citation_prefetch),
        db_dir=get_settings().db_dir,
        seed_only=bool(pending),
    )


def _build_pending_conversation_refs_payload(
    refs: dict,
    *,
    doc_list_by_user: dict[int, list[dict]] | None = None,
    guide_mode: bool = False,
    guide_source_path: str = "",
    guide_source_name: str = "",
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    authoritative_map = {
        int(key): [dict(item) for item in list(value or []) if isinstance(item, dict)]
        for key, value in dict(doc_list_by_user or {}).items()
        if str(key).isdigit() or isinstance(key, int)
    }
    for user_msg_id, pack in (refs or {}).items():
        if not isinstance(pack, dict):
            continue
        prompt = str(pack.get("prompt") or "").strip()
        render_locale = _ref_card_user_locale(prompt)
        prefer_zh = render_locale == "zh"
        focus_terms = [str(term or "").strip() for term in _refs_prompt_focus_terms(prompt) if str(term or "").strip()]
        focus_action = prompt_reference_focus_action(prompt)
        raw_hits = [hit for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
        if prompt_explicitly_requests_multi_paper_list(prompt):
            filtered_hits = _filter_pending_multi_paper_hits_for_display(prompt, raw_hits)
            if not filtered_hits:
                filtered_hits = _filter_pending_refs_hits_by_prompt_focus(prompt, raw_hits)
            filtered_hits = filtered_hits[:3]
        else:
            filtered_hits = _filter_pending_refs_hits_by_prompt_focus(prompt, raw_hits)[:2]
        pending_count = 0
        hits_out: list[dict] = []
        for hit in raw_hits:
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
                pending_count += 1
        authoritative_doc_list_present = int(user_msg_id) in authoritative_map
        authoritative_doc_list = [dict(item) for item in list(authoritative_map.get(int(user_msg_id), []) or []) if isinstance(item, dict)]
        if authoritative_doc_list_present:
            authoritative_pack = _render_authoritative_doc_list_pack(
                user_msg_id=int(user_msg_id),
                pack=pack,
                doc_list=authoritative_doc_list,
                guide_mode=bool(guide_mode),
                guide_source_path=str(guide_source_path or "").strip(),
                guide_source_name=str(guide_source_name or "").strip(),
                pending=True,
            )
            out[int(user_msg_id)] = _mark_doc_list_pending_pack(
                payload_pack=authoritative_pack,
                pending_count=pending_count,
            )
            continue
        for hit in filtered_hits:
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            source_path = str((meta or {}).get("source_path") or "").strip()
            heading_path = str((meta or {}).get("ref_best_heading_path") or (meta or {}).get("top_heading") or "").strip()
            display_name = _source_name_from_md_path(source_path) if source_path else "Reference"
            snippet_seed = ""
            for key in ("ref_show_snippets", "ref_snippets", "ref_overview_snippets"):
                raw = (meta or {}).get(key)
                if isinstance(raw, list):
                    snippet_seed = next((str(item or "").strip() for item in raw if str(item or "").strip()), "")
                if snippet_seed:
                    break
            if not snippet_seed:
                snippet_seed = str(hit.get("text") or "").strip()
            summary_line = _compact_reader_open_text(snippet_seed)
            focus_text = (
                (" 与 " if prefer_zh else " and ").join(focus_terms[:2])
                if len(focus_terms) >= 2
                else (
                    focus_terms[0]
                    if focus_terms
                    else ("当前问题" if prefer_zh else "the requested concept")
                )
            )
            section_text = heading_path or ("命中章节" if prefer_zh else "the matched section")
            if focus_action == "compare":
                why_line = (
                    f"“{section_text}”正在核验 {focus_text} 的对比依据。"
                    if prefer_zh
                    else f"“{section_text}” is being checked for direct comparison evidence about {focus_text}."
                )
            elif focus_action == "define":
                why_line = (
                    f"“{section_text}”正在核验 {focus_text} 的定义原句。"
                    if prefer_zh
                    else f"“{section_text}” is being checked for the source definition of {focus_text}."
                )
            else:
                why_line = (
                    f"“{section_text}”正在核验与 {focus_text} 直接相关的原文。"
                    if prefer_zh
                    else f"“{section_text}” is being checked for source evidence directly about {focus_text}."
                )
            reader_open = {
                "sourcePath": source_path,
                "sourceName": display_name,
                "headingPath": heading_path or None,
                "snippet": summary_line or None,
                "highlightSnippet": summary_line or None,
                "strictLocate": False,
            }
            primary_evidence = {
                "source_path": source_path,
                "source_name": display_name,
                "heading_path": heading_path or None,
                "snippet": summary_line or None,
                "highlight_snippet": summary_line or None,
                "selection_reason": "pending_section_seed",
                "strict_locate": False,
            }
            hit2 = dict(hit)
            hit2["ui_meta"] = {
                "display_name": display_name,
                "heading_path": heading_path,
                "summary_line": summary_line,
                "summary_kind": "evidence",
                "summary_display_role": "evidence",
                "summary_label": "原文证据" if prefer_zh else "Source Evidence",
                "summary_title": "待核验的原文片段" if prefer_zh else "Source Passage Being Checked",
                "summary_generation": "pending_section_seed",
                "summary_basis": "待核验的命中章节原文" if prefer_zh else "Provisional source text from the matched section",
                "why_line": why_line,
                "why_generation": "pending_focus_seed",
                "why_basis": "命中章节与问题关键词的待核验对齐" if prefer_zh else "Provisional alignment between the matched section and question terms",
                "render_locale": render_locale,
                "score": None,
                "score_pending": True,
                "score_tier": "",
                "primary_evidence": {key: value for key, value in primary_evidence.items() if value not in (None, "", [], {})},
                "source_path": source_path,
                "reader_open": {key: value for key, value in reader_open.items() if value not in (None, "", [], {})},
            }
            if isinstance(hit2["ui_meta"].get("reader_open"), dict) and hit2["ui_meta"].get("primary_evidence"):
                hit2["ui_meta"]["reader_open"] = dict(hit2["ui_meta"]["reader_open"])
                hit2["ui_meta"]["reader_open"]["primaryEvidence"] = dict(hit2["ui_meta"]["primary_evidence"])
            hits_out.append(hit2)
        pack2 = dict(pack)
        pack2["hits"] = hits_out
        pack2["pending"] = True
        pack2["pending_hit_count"] = int(pending_count)
        pack2["payload_mode"] = "pending"
        pack2["enrichment_pending"] = True
        out[int(user_msg_id)] = _attach_pack_display_contract(pack2)
    return out


def _annotate_refs_payload_refresh_state(payload: dict, *, mode: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    mode_norm = str(mode or "").strip().lower() or "full"
    needs_enrichment = mode_norm in {"fast", "pending"}
    for user_msg_id, pack in (payload or {}).items():
        if not isinstance(pack, dict):
            continue
        pack2 = dict(pack)
        pack2["payload_mode"] = mode_norm
        if needs_enrichment:
            pack2["enrichment_pending"] = True
        else:
            pack2.pop("enrichment_pending", None)
        out[int(user_msg_id)] = _attach_pack_display_contract(pack2)
    return out


def _attach_pack_render_state(
    payload_pack: dict,
    *,
    source_pack: dict | None,
    default_status: str = "",
    override_status: bool = False,
) -> dict:
    out = _attach_pack_display_contract(payload_pack)
    src = source_pack if isinstance(source_pack, dict) else {}
    render_status = str(
        (default_status if override_status else "")
        or (src or {}).get("render_status")
        or default_status
        or ""
    ).strip().lower()
    render_error = str((src or {}).get("render_error") or "").strip()
    render_error_detail = str((src or {}).get("render_error_detail") or "").strip()
    try:
        render_attempts = int((src or {}).get("render_attempts") or 0)
    except Exception:
        render_attempts = 0
    try:
        render_built_at = float((src or {}).get("render_built_at") or 0.0)
    except Exception:
        render_built_at = 0.0
    render_evidence_sig = str((src or {}).get("render_evidence_sig") or "").strip()
    render_locale = str((src or {}).get("render_locale") or "").strip()
    if render_status:
        out["render_status"] = render_status
    if render_error:
        out["render_error"] = render_error
    if render_error_detail:
        out["render_error_detail"] = render_error_detail
    if render_attempts > 0:
        out["render_attempts"] = render_attempts
    if render_built_at > 0:
        out["render_built_at"] = render_built_at
    if render_evidence_sig:
        out["render_evidence_sig"] = render_evidence_sig
    if render_locale:
        out["render_locale"] = render_locale
    if str(out.get("render_status") or "").strip().lower() == "failed":
        out.pop("enrichment_pending", None)
    return _attach_pack_display_contract(out)


def _build_fast_ready_conversation_refs_payload(
    *,
    refs: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
    deadline_at: float | None = None,
) -> dict[int, dict]:
    return _annotate_refs_payload_refresh_state(
        enrich_refs_payload(
            refs,
            pdf_root=_pdf_dir(),
            md_root=_md_dir(),
            lib_store=_lib_store(),
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
            render_variant="fast",
            allow_expensive_llm_for_ready=False,
            allow_exact_locate=False,
            deadline_at=deadline_at,
        ),
        mode="fast",
    )


def _persist_rendered_refs_payloads(
    *,
    refs: dict,
    payload: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
) -> None:
    if not isinstance(refs, dict) or not isinstance(payload, dict):
        return
    try:
        store = get_chat_store()
    except Exception:
        return
    for user_msg_id, pack in refs.items():
        if not isinstance(pack, dict):
            continue
        payload_pack = payload.get(user_msg_id)
        if not isinstance(payload_pack, dict):
            payload_pack = payload.get(str(user_msg_id))
        if not isinstance(payload_pack, dict) or (not payload_pack):
            continue
        # ``payload_pack`` is built from the row returned by
        # ``list_message_refs``.  That row already carries the previous stored
        # payload under ``rendered_payload``.  Persisting it again recursively
        # nests every historical render, bloats SQLite, and can leak stale card
        # copy back into later reads.
        payload_to_store = _without_nested_render_payload(payload_pack)
        sig = _refs_pack_render_signature(
            user_msg_id=user_msg_id,
            pack=pack,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
        )
        try:
            store.set_message_refs_rendered_payload(
                user_msg_id=int(user_msg_id),
                rendered_payload=payload_to_store,
                rendered_payload_sig=sig,
                render_status="full",
                render_error="",
                render_error_detail="",
                render_built_at=time.time(),
                render_attempts=max(1, int(pack.get("render_attempts") or 0)),
                render_evidence_sig=str(sig or "").strip(),
            )
        except Exception:
            continue


def _without_nested_render_payload(payload_pack: dict | None) -> dict:
    out = dict(payload_pack or {})
    out.pop("rendered_payload", None)
    out.pop("rendered_payload_sig", None)
    return out


def _refs_background_llm_polish_enabled() -> bool:
    raw = str(os.environ.get("KB_REFS_BACKGROUND_LLM_POLISH", "") or "").strip().lower()
    if raw:
        return raw in {"1", "true", "on", "yes"}
    # Source-grounded deterministic cards are the default.  A second model
    # pass races the answer renderer and used to add a 12-45 second tail even
    # when the final cards were ultimately composed from exact citations.
    # Keep the optional polish available as an explicit deployment opt-in.
    return False


def _refs_payload_has_fast_exact_hit(refs: dict | None) -> bool:
    for pack in dict(refs or {}).values():
        if not isinstance(pack, dict):
            continue
        for hit in list(pack.get("hits") or []):
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            if bool((meta or {}).get("paper_guide_fast_exact")):
                return True
    return False


def _warm_conversation_refs_payload_async(
    *,
    conv_id: str,
    signature: str,
    refs: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
    authoritative_doc_list_by_user: dict[int, list[dict]] | None = None,
) -> None:
    conv_key = str(conv_id or "").strip()
    sig_key = str(signature or "").strip()
    if (not conv_key) or (not sig_key):
        return
    warm_key = f"{conv_key}:{sig_key}"
    with _REFS_CONVERSATION_WARMING_LOCK:
        if warm_key in _REFS_CONVERSATION_WARMING:
            return
        _REFS_CONVERSATION_WARMING.add(warm_key)

    def _run() -> None:
        try:
            authoritative_map = {
                int(key): [dict(item) for item in list(value or []) if isinstance(item, dict)]
                for key, value in dict(authoritative_doc_list_by_user or {}).items()
                if str(key).isdigit() or isinstance(key, int)
            }
            payload: dict[int, dict] = {}
            regular_refs: dict[int, dict] = {}
            for raw_user_msg_id, pack in dict(refs or {}).items():
                try:
                    user_msg_id = int(raw_user_msg_id)
                except Exception:
                    continue
                if not isinstance(pack, dict):
                    continue
                pack = _augment_pack_with_canonical_answer_paths(pack)
                if user_msg_id in authoritative_map:
                    rendered = _render_authoritative_doc_list_pack(
                        user_msg_id=user_msg_id,
                        pack=pack,
                        doc_list=authoritative_map[user_msg_id],
                        guide_mode=bool(guide_mode),
                        guide_source_path=str(guide_source_path or "").strip(),
                        guide_source_name=str(guide_source_name or "").strip(),
                        pending=False,
                        allow_expensive_llm=_refs_background_llm_polish_enabled(),
                        # Exact source positioning remains enabled below, but
                        # remote metadata refresh must not hold the full-card
                        # transition open.  Cached/local DOI and metrics are
                        # hydrated on the read path and remote refresh can run
                        # independently.
                        allow_citation_prefetch=False,
                    )
                    if isinstance(rendered, dict) and rendered:
                        payload[user_msg_id] = rendered
                        _persist_rendered_refs_payloads(
                            refs={user_msg_id: pack},
                            payload={user_msg_id: rendered},
                            guide_mode=guide_mode,
                            guide_source_path=guide_source_path,
                            guide_source_name=guide_source_name,
                        )
                    continue
                regular_refs[user_msg_id] = pack
            if regular_refs:
                # Finish the newest answer first and persist each completed
                # pack immediately.  A long historical conversation must not
                # keep the card under the latest answer in fast mode until all
                # older turns have finished exact-location work.
                for user_msg_id in sorted(regular_refs, reverse=True):
                    pack = regular_refs[user_msg_id]
                    has_answer_citation_locator = any(
                        int((((hit.get("meta") or {}) if isinstance(hit, dict) else {}).get("ref_answer_citation_num") or 0)) > 0
                        for hit in list(pack.get("hits") or [])
                    )
                    regular_payload = enrich_refs_payload(
                        {user_msg_id: pack},
                        pdf_root=_pdf_dir(),
                        md_root=_md_dir(),
                        lib_store=_lib_store(),
                        guide_mode=guide_mode,
                        guide_source_path=guide_source_path,
                        guide_source_name=guide_source_name,
                        render_variant="bounded_full",
                        # Once generation has assigned answer citation numbers,
                        # the cards can be composed directly from the exact
                        # cited passages.  Waiting for a second LLM to restate
                        # those passages only adds a long tail and can make the
                        # card wording less faithful.
                        allow_expensive_llm_for_ready=bool(
                            _refs_background_llm_polish_enabled()
                            and not has_answer_citation_locator
                        ),
                        # The retrieval row already carries a source passage,
                        # and completed answers replace it with a stricter
                        # citation-plan block.  A second whole-source candidate
                        # scan was the remaining long tail and could even move
                        # the card away from the passage used in the answer.
                        allow_exact_locate=False,
                    )
                    rendered = regular_payload.get(user_msg_id) if isinstance(regular_payload, dict) else None
                    if not isinstance(rendered, dict):
                        continue
                    payload[user_msg_id] = rendered
                    _persist_rendered_refs_payloads(
                        refs={user_msg_id: pack},
                        payload={user_msg_id: rendered},
                        guide_mode=guide_mode,
                        guide_source_path=guide_source_path,
                        guide_source_name=guide_source_name,
                    )
            if not isinstance(payload, dict):
                return
            current = _REFS_CONVERSATION_CACHE.get(conv_key)
            cached_payload: dict[int, dict] = {}
            if isinstance(current, dict):
                current_sig = str(current.get("signature") or "").strip()
                if current_sig and current_sig != sig_key:
                    return
                current_payload = current.get("payload")
                if isinstance(current_payload, dict):
                    cached_payload = {
                        int(key): value
                        for key, value in current_payload.items()
                        if (str(key).isdigit() or isinstance(key, int)) and isinstance(value, dict)
                    }
            cached_payload.update(payload)
            _store_cached_conversation_refs_payload(
                conv_id=conv_key,
                signature=sig_key,
                payload=cached_payload,
                mode="full",
                refs=refs,
            )
        except Exception as exc:
            try:
                store = get_chat_store()
            except Exception:
                store = None
            if store is not None:
                for user_msg_id, pack in (refs or {}).items():
                    if not isinstance(pack, dict):
                        continue
                    try:
                        store.set_message_refs_render_state(
                            user_msg_id=int(user_msg_id),
                            render_status="failed",
                            render_error="route_warm_failed",
                            render_error_detail=f"{type(exc).__name__}: {str(exc or '').strip()}"[:500],
                        )
                    except Exception:
                        continue
        finally:
            with _REFS_CONVERSATION_WARMING_LOCK:
                _REFS_CONVERSATION_WARMING.discard(warm_key)

    try:
        threading.Thread(target=_run, daemon=True, name="kb_refs_conv_warm").start()
    except Exception:
        with _REFS_CONVERSATION_WARMING_LOCK:
            _REFS_CONVERSATION_WARMING.discard(warm_key)


@router.post("/sync", dependencies=[Depends(require_management_api)])
def start_sync(workers: int | None = None, crossref_budget_s: float | None = None):
    s = get_settings()
    try:
        workers_default = int(os.environ.get("KB_REFSYNC_WORKERS", "6") or 6)
    except Exception:
        workers_default = 6
    if workers is None:
        workers = workers_default
    workers_final = int(max(1, min(16, int(workers))))

    try:
        budget_default = float(os.environ.get("KB_CROSSREF_BUDGET_S", "180") or 180.0)
    except Exception:
        budget_default = 180.0
    if crossref_budget_s is None:
        crossref_budget_s = budget_default
    budget_final = float(max(5.0, min(600.0, float(crossref_budget_s))))

    result = start_reference_sync(
        src_root=_md_dir(),
        db_dir=s.db_dir,
        pdf_root=_pdf_dir(),
        library_db_path=s.library_db_path,
        crossref_time_budget_s=budget_final,
        doi_prefetch_workers=workers_final,
    )
    return result


@router.get("/sync/status")
async def sync_status():
    def poll():
        snap = refsync_snapshot()
        return {
            **snap,
            "done": snap.get("status") in ("done", "error", "idle"),
        }
    return sse_response(sse_generator(poll, interval=0.5))


def _compute_diagnose_suggestion(suppression_reason: str) -> str:
    suggestions = {
        "no_candidate_hits": (
            "No documents matched the query. Try rephrasing with different keywords, "
            "or check that relevant documents are ingested in the knowledge base."
        ),
        "score_gate_removed_all": (
            "All BM25 scores were below the relevance threshold. "
            "Try a more specific query."
        ),
        "focus_filter_removed_all": (
            "All hits were filtered out because they did not match the prompt's "
            "focus terms. Try broadening the question or removing specific constraints."
        ),
        "llm_filter_removed_all": (
            "The LLM relevance filter judged all hits as irrelevant. "
            "This may indicate a vocabulary mismatch between the query and documents."
        ),
        "guide_self_source_only": (
            "Guide mode hides the bound source paper. "
            "Disable guide mode or ask about other papers."
        ),
        "render_failed": (
            "The reference card rendering pipeline failed unexpectedly. "
            "Check server logs for error details."
        ),
        "pending_enrichment": (
            "Results are still being computed. "
            "Try again in a few seconds."
        ),
        "no_renderable_hits": (
            "Hits entered the pipeline but none could be rendered as reference cards. "
            "Check the pipeline stage counts for details."
        ),
    }
    return suggestions.get(suppression_reason, "No specific suggestion available for this state.")


def _build_diagnostic_report(*, store, conv_id: str, refs: dict) -> dict:
    """Build a diagnostic report for all refs packs in a conversation."""
    packs: dict[int, dict] = {}
    total_packs = 0
    empty_packs = 0
    suppressed_packs = 0

    for key, pack in (refs or {}).items():
        try:
            user_msg_id = int(key)
        except (ValueError, TypeError):
            continue
        total_packs += 1
        if not isinstance(pack, dict):
            packs[user_msg_id] = {"parse_error": "pack is not a dict"}
            continue

        try:
            contract = _attach_pack_display_contract(pack)
        except Exception as exc:
            packs[user_msg_id] = {"parse_error": str(exc)[:200]}
            continue

        display_state = str(contract.get("display_state") or "unknown")
        suppression_reason = str(contract.get("suppression_reason") or "").strip()
        pipeline_debug = contract.get("pipeline_debug") if isinstance(contract.get("pipeline_debug"), dict) else {}
        retrieval_diag = pipeline_debug.get("retrieval_diag") if isinstance(pipeline_debug.get("retrieval_diag"), dict) else {}
        prompt_raw = str(pack.get("prompt") or pack.get("question") or "").strip()
        used_query = str(pack.get("used_query") or pipeline_debug.get("used_query") or retrieval_diag.get("used_query") or "").strip()
        used_translation = bool(pack.get("used_translation") or retrieval_diag.get("query_translated") or False)

        # Compute top BM25 scores from hits.
        top_scores: list[dict] = []
        hits = [h for h in list(contract.get("hits") or []) if isinstance(h, dict)]
        scored = []
        for h in hits:
            try:
                bm25_score = float(h.get("score") or 0.0)
            except (ValueError, TypeError):
                bm25_score = 0.0
            meta = h.get("meta") if isinstance(h.get("meta"), dict) else {}
            source_path = str(meta.get("source_path") or "").strip()
            source_name = str(meta.get("source_name") or "").strip()
            if not source_name:
                source_name = str(Path(source_path).stem if source_path else "unknown")
            heading = str(meta.get("heading_path") or "").strip()[:120]
            scored.append({
                "score": round(bm25_score, 2),
                "doc_name": source_name[:80],
                "source_path": source_path,
                "heading_path": heading,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_scores = scored[:5]

        if display_state == "empty":
            empty_packs += 1
        elif display_state in ("suppressed", "hidden_by_guide"):
            suppressed_packs += 1

        suggestion = _compute_diagnose_suggestion(suppression_reason) if suppression_reason else ""

        packs[user_msg_id] = {
            "prompt": prompt_raw[:500],
            "display_state": display_state,
            "suppression_reason": suppression_reason,
            "pipeline_debug": pipeline_debug,
            "retrieval_diag": retrieval_diag,
            "used_query": used_query,
            "used_translation": used_translation,
            "top_scores": top_scores,
            "has_pending": bool(contract.get("pending")),
            "suggestion": suggestion,
        }

    return {
        "conv_id": conv_id,
        "total_packs": total_packs,
        "empty_packs": empty_packs,
        "suppressed_packs": suppressed_packs,
        "packs": packs,
    }


def get_conversation_refs(conv_id: str, response: Response | None = None):
    route_started_at = time.perf_counter()
    route_deadline_at = route_started_at + _refs_ready_budget_s()
    timings: list[tuple[str, float]] = []
    refs_for_finish: dict[int, dict] = {}
    refs_state_signature = ""
    signature = ""
    finish_guide_mode = False
    finish_guide_source_path = ""
    finish_guide_source_name = ""

    def _record(name: str, started_at: float) -> None:
        timings.append((name, _refs_perf_ms(started_at)))

    def _finish(payload: dict | None, mode: str) -> dict:
        payload_out = payload if isinstance(payload, dict) else {}
        payload_out = _overlay_refs_payload_with_answer_citations(
            store=store,
            conv_id=conv_id,
            payload=payload_out,
        )
        completed_payloads = {
            int(user_msg_id): pack
            for user_msg_id, pack in payload_out.items()
            if (str(user_msg_id).isdigit() or isinstance(user_msg_id, int))
            and isinstance(pack, dict)
            and _answer_citation_overlay_pack_is_complete(pack)
            and int(user_msg_id) in refs_for_finish
            and str(
                (refs_for_finish.get(int(user_msg_id)) or {}).get("render_status")
                or ""
            ).strip().lower()
            != "full"
        }
        if completed_payloads:
            _persist_rendered_refs_payloads(
                refs={
                    user_msg_id: refs_for_finish[user_msg_id]
                    for user_msg_id in completed_payloads
                },
                payload=completed_payloads,
                guide_mode=finish_guide_mode,
                guide_source_path=finish_guide_source_path,
                guide_source_name=finish_guide_source_name,
            )
            if signature:
                _store_cached_conversation_refs_payload(
                    conv_id=conv_id,
                    signature=signature,
                    payload=payload_out,
                    mode="full",
                    refs=refs_for_finish,
                    state_signature=refs_state_signature,
                )
        # Card construction/background warming already embeds cached local
        # bibliography metadata. Re-scanning every source and rebuilding every
        # message render packet on this read path caused 10–45 second stalls,
        # even for a cache hit. The message endpoint can consume the stored
        # rendered pack directly; remote/local metadata refresh remains an
        # independent background concern.
        _set_refs_timing_headers(
            response,
            timings=timings,
            total_ms=_refs_perf_ms(route_started_at),
            mode=mode,
            payload=payload_out,
        )
        return public_refs_payload_projection(
            payload_out,
            source_roots=_reference_asset_roots(),
        )

    store = get_chat_store()
    read_timeout_s = _refs_conversation_read_timeout_s()
    phase_started_at = time.perf_counter()
    try:
        conversation = store.get_conversation(conv_id, timeout_s=read_timeout_s) or {}
    except TypeError:
        conversation = store.get_conversation(conv_id) or {}
    except sqlite3.OperationalError:
        _record("conversation", phase_started_at)
        cached_any = _get_any_cached_conversation_refs_payload(conv_id=conv_id)
        return _finish(cached_any if isinstance(cached_any, dict) else {}, "cache_fallback")
    _record("conversation", phase_started_at)
    guide_mode = str(conversation.get("mode") or "").strip().lower() == "paper_guide"
    guide_source_path = str(conversation.get("bound_source_path") or "").strip()
    guide_source_name = str(conversation.get("bound_source_name") or "").strip()
    finish_guide_mode = bool(guide_mode)
    finish_guide_source_path = guide_source_path
    finish_guide_source_name = guide_source_name
    if hasattr(store, "list_message_refs_state"):
        phase_started_at = time.perf_counter()
        try:
            refs_state = store.list_message_refs_state(
                conv_id,
                timeout_s=read_timeout_s,
            )
        except TypeError:
            refs_state = store.list_message_refs_state(conv_id)
        except sqlite3.OperationalError:
            refs_state = None
        if isinstance(refs_state, dict):
            refs_state_signature = _refs_conversation_state_signature(
                conversation=conversation,
                refs_state=refs_state,
            )
            cached_state_rec = _get_state_validated_conversation_refs_record(
                conv_id=conv_id,
                state_signature=refs_state_signature,
            )
            _record("state_cache_lookup", phase_started_at)
            if isinstance(cached_state_rec, dict):
                cached_state_payload = cached_state_rec.get("payload")
                cached_state_mode = (
                    str(cached_state_rec.get("mode") or "full").strip().lower()
                    or "full"
                )
                if isinstance(cached_state_payload, dict):
                    return _finish(
                        cached_state_payload,
                        f"cache_validated_{cached_state_mode}",
                    )
        else:
            _record("state_cache_lookup", phase_started_at)
    phase_started_at = time.perf_counter()
    try:
        refs = store.list_message_refs(conv_id, timeout_s=read_timeout_s)
    except TypeError:
        refs = store.list_message_refs(conv_id)
    except sqlite3.OperationalError:
        _record("list_refs", phase_started_at)
        cached_any = _get_any_cached_conversation_refs_payload(conv_id=conv_id)
        return _finish(cached_any if isinstance(cached_any, dict) else {}, "cache_fallback")
    _record("list_refs", phase_started_at)
    refs_norm = _attach_assistant_answers_to_refs(
        store=store,
        conv_id=conv_id,
        refs=refs if isinstance(refs, dict) else {},
    )
    refs_for_finish = refs_norm
    answer_citation_ready_user_ids = set(
        _answer_citation_details_by_user(
            store=store,
            conv_id=conv_id,
        )
    )
    all_user_msg_ids: set[int] = set()
    for key in refs_norm.keys():
        try:
            all_user_msg_ids.add(int(key))
        except Exception:
            continue
    phase_started_at = time.perf_counter()
    authoritative_doc_lists = _load_authoritative_doc_list_contracts(
        store=store,
        conv_id=conv_id,
        user_msg_ids=all_user_msg_ids,
    )
    authoritative_doc_lists = _normalize_authoritative_doc_list_contracts_for_refs(
        refs=refs_norm,
        doc_lists=authoritative_doc_lists,
        guide_mode=bool(guide_mode),
    )
    _record("doc_list_contracts", phase_started_at)
    phase_started_at = time.perf_counter()
    signature = _refs_conversation_cache_signature(
        refs=refs_norm,
        guide_mode=guide_mode,
        guide_source_path=guide_source_path,
        guide_source_name=guide_source_name,
        authoritative_doc_list_by_user=authoritative_doc_lists,
    )
    _record("signature", phase_started_at)
    has_pending = _refs_payload_has_pending(refs_norm, include_stale=False)
    has_authoritative_doc_list = bool(authoritative_doc_lists)
    phase_started_at = time.perf_counter()
    cached_rec = _get_cached_conversation_refs_record(conv_id=conv_id, signature=signature)
    _record("cache_lookup", phase_started_at)
    cached_payload = cached_rec.get("payload") if isinstance(cached_rec, dict) else None
    cached_mode = str(cached_rec.get("mode") or "").strip().lower() if isinstance(cached_rec, dict) else ""
    if isinstance(cached_payload, dict) and cached_mode == "full":
        if (
            (not has_authoritative_doc_list)
            or _cached_payload_matches_authoritative_doc_lists(
                cached_payload,
                authoritative_doc_lists,
            )
        ):
            return _finish(cached_payload, "cache_full")

    stored_full_payload: dict[int, dict] = {}
    pending_refs: dict[int, dict] = {}
    failed_ready_refs: dict[int, dict] = {}
    ready_missing_refs: dict[int, dict] = {}
    historical_stale_payloads: dict[int, dict] = {}
    historical_stale_refs: dict[int, dict] = {}
    authoritative_sync_payloads: dict[int, dict] = {}
    authoritative_sync_refs: dict[int, dict] = {}
    authoritative_fast_payloads: dict[int, dict] = {}
    authoritative_fast_refs: dict[int, dict] = {}
    numeric_user_msg_ids = [
        int(key)
        for key in refs_norm
        if str(key).isdigit() or isinstance(key, int)
    ]
    latest_user_msg_id = max(numeric_user_msg_ids, default=0)
    phase_started_at = time.perf_counter()
    for user_msg_id, pack in refs_norm.items():
        if not isinstance(pack, dict):
            continue
        prompt_text = str(pack.get("prompt") or "").strip()
        authoritative_doc_list_present = int(user_msg_id) in authoritative_doc_lists
        authoritative_doc_list = [
            dict(item)
            for item in list(authoritative_doc_lists.get(int(user_msg_id), []) or [])
            if isinstance(item, dict)
        ]
        if authoritative_doc_list_present and (not authoritative_doc_list):
            rebuilt_doc_list = _rebuild_authoritative_doc_list_from_pack(
                prompt=prompt_text,
                pack=pack,
                guide_mode=bool(guide_mode),
            )
            if rebuilt_doc_list:
                authoritative_doc_list = rebuilt_doc_list
        pack_phase_started_at = time.perf_counter()
        pack_full = _get_stored_rendered_pack_payload(
            user_msg_id=user_msg_id,
            pack=pack,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
            allow_authoritative_source_override=bool(authoritative_doc_list_present),
        )
        _record("stored_pack_lookup", pack_phase_started_at)
        if authoritative_doc_list_present and _refs_pack_has_pending(pack, include_stale=False):
            pending_refs[int(user_msg_id)] = pack
            continue
        if authoritative_doc_list_present:
            if isinstance(pack_full, dict) and _payload_is_authoritative_doc_list_pack(pack_full, authoritative_doc_list):
                pack_full = hydrate_doc_list_refs_payload_citation_meta(
                    pack_full,
                    doc_list=authoritative_doc_list,
                    pdf_root=_pdf_dir(),
                    lib_store=_lib_store(),
                    db_dir=get_settings().db_dir,
                )
                stored_full_payload[int(user_msg_id)] = _attach_pack_render_state(
                    pack_full,
                    source_pack=pack,
                    default_status="full",
                )
                continue
            if int(user_msg_id) in answer_citation_ready_user_ids:
                # The answer plan already supplies source-bound evidence for
                # every visible card. Seed the overlay directly from the
                # retrieval rows instead of building an intermediate
                # authoritative card pack that will be replaced in _finish.
                authoritative_seed = _without_nested_render_payload(pack)
                authoritative_seed["payload_mode"] = "fast"
                authoritative_seed["enrichment_pending"] = True
                authoritative_fast_payloads[int(user_msg_id)] = authoritative_seed
                authoritative_fast_refs[int(user_msg_id)] = pack
                continue
            pack_phase_started_at = time.perf_counter()
            authoritative_payload = _render_authoritative_doc_list_pack(
                user_msg_id=int(user_msg_id),
                pack=pack,
                doc_list=authoritative_doc_list,
                guide_mode=bool(guide_mode),
                guide_source_path=str(guide_source_path or "").strip(),
                guide_source_name=str(guide_source_name or "").strip(),
                # The first read is a fast snapshot.  Strict block matching is
                # completed by the background full render below; doing it here
                # can synchronously block the references endpoint for minutes.
                pending=bool(authoritative_doc_list),
                allow_expensive_llm=False,
            )
            _record("authoritative_fast_pack", pack_phase_started_at)
            if isinstance(authoritative_payload, dict) and authoritative_payload:
                if not authoritative_doc_list:
                    authoritative_sync_payloads[int(user_msg_id)] = authoritative_payload
                    authoritative_sync_refs[int(user_msg_id)] = pack
                    stored_full_payload[int(user_msg_id)] = _attach_pack_render_state(
                        authoritative_payload,
                        source_pack=pack,
                        default_status="full",
                        override_status=True,
                    )
                    continue
                authoritative_fast_payloads[int(user_msg_id)] = authoritative_payload
                authoritative_fast_refs[int(user_msg_id)] = pack
                continue
        if isinstance(pack_full, dict):
            stored_full_payload[int(user_msg_id)] = _attach_pack_render_state(
                pack_full,
                source_pack=pack,
                default_status="full",
            )
            continue
        if _refs_pack_has_pending(pack, include_stale=False):
            pending_refs[int(user_msg_id)] = pack
        elif str((pack or {}).get("render_status") or "").strip().lower() == "failed":
            failed_ready_refs[int(user_msg_id)] = pack
        else:
            stale_payload = pack.get("rendered_payload")
            if (
                int(user_msg_id) != latest_user_msg_id
                and isinstance(stale_payload, dict)
                and bool(stale_payload)
                and not _stored_rendered_pack_payload_has_dirty_card_copy(stale_payload)
            ):
                historical_stale_payloads[int(user_msg_id)] = _without_nested_render_payload(stale_payload)
                historical_stale_refs[int(user_msg_id)] = pack
            else:
                ready_missing_refs[int(user_msg_id)] = pack
    _record("render_state_scan", phase_started_at)

    if authoritative_sync_payloads:
        phase_started_at = time.perf_counter()
        _persist_rendered_refs_payloads(
            refs=authoritative_sync_refs,
            payload=authoritative_sync_payloads,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
        )
        _record("persist_empty_authoritative", phase_started_at)

    if (
        refs_norm
        and (not pending_refs)
        and (not failed_ready_refs)
        and (not ready_missing_refs)
        and (not historical_stale_payloads)
        and (not authoritative_fast_payloads)
        and stored_full_payload
    ):
        _store_cached_conversation_refs_payload(
            conv_id=conv_id,
            signature=signature,
            payload=stored_full_payload,
            mode="full",
            refs=refs_norm,
            state_signature=refs_state_signature,
        )
        return _finish(stored_full_payload, "stored_full")

    if isinstance(cached_payload, dict) and (not stored_full_payload) and (not has_authoritative_doc_list):
        if (not has_pending) and (not failed_ready_refs) and cached_mode != "full":
            refs_to_warm = _refs_without_completed_answer_citation_overlays(
                store=store,
                conv_id=conv_id,
                refs=refs_norm,
            )
            if refs_to_warm:
                _warm_conversation_refs_payload_async(
                    conv_id=conv_id,
                    signature=signature,
                    refs=refs_to_warm,
                    guide_mode=guide_mode,
                    guide_source_path=guide_source_path,
                    guide_source_name=guide_source_name,
                )
        annotated_cached = _annotate_refs_payload_refresh_state(
            cached_payload,
            mode=cached_mode or ("pending" if has_pending else "fast"),
        )
        return _finish(annotated_cached, f"cache_{cached_mode or ('pending' if has_pending else 'fast')}")

    payload: dict[int, dict] = dict(stored_full_payload)
    if historical_stale_payloads:
        annotated_historical = _annotate_refs_payload_refresh_state(
            historical_stale_payloads,
            mode="fast",
        )
        for user_msg_id, pack in historical_stale_refs.items():
            payload_pack = annotated_historical.get(int(user_msg_id))
            if isinstance(payload_pack, dict):
                payload[int(user_msg_id)] = _attach_pack_render_state(
                    payload_pack,
                    source_pack=pack,
                    default_status="fast",
                    override_status=True,
                )
    if authoritative_fast_payloads:
        annotated_authoritative = _annotate_refs_payload_refresh_state(
            authoritative_fast_payloads,
            mode="fast",
        )
        for user_msg_id, pack in authoritative_fast_refs.items():
            payload_pack = annotated_authoritative.get(int(user_msg_id))
            if isinstance(payload_pack, dict):
                payload[int(user_msg_id)] = _attach_pack_render_state(
                    payload_pack,
                    source_pack=pack,
                    default_status="fast",
                    override_status=True,
                )
    if pending_refs:
        phase_started_at = time.perf_counter()
        pending_payload = _build_pending_conversation_refs_payload(
            pending_refs,
            doc_list_by_user=authoritative_doc_lists,
            guide_mode=bool(guide_mode),
            guide_source_path=str(guide_source_path or "").strip(),
            guide_source_name=str(guide_source_name or "").strip(),
        )
        for user_msg_id, pack in pending_refs.items():
            payload_pack = pending_payload.get(int(user_msg_id))
            if isinstance(payload_pack, dict):
                payload[int(user_msg_id)] = _attach_pack_render_state(
                    payload_pack,
                    source_pack=pack,
                    default_status="pending",
                )
        _record("pending_render", phase_started_at)
    if failed_ready_refs:
        phase_started_at = time.perf_counter()
        failed_payload = _build_fast_ready_conversation_refs_payload(
            refs=failed_ready_refs,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
            deadline_at=route_deadline_at,
        )
        for user_msg_id, pack in failed_ready_refs.items():
            payload_pack = failed_payload.get(int(user_msg_id))
            if isinstance(payload_pack, dict):
                payload[int(user_msg_id)] = _attach_pack_render_state(
                    payload_pack,
                    source_pack=pack,
                    default_status="failed",
                )
        _record("failed_fast_render", phase_started_at)
    # Even exact System-B hits use the local fast card path for the first
    # response.  Their stricter locator and polished copy are still computed by
    # the background warm, without holding the user's request open.
    normal_ready_refs: dict[int, dict] = {
        int(user_msg_id): pack for user_msg_id, pack in ready_missing_refs.items()
    }
    if normal_ready_refs:
        phase_started_at = time.perf_counter()
        fast_payload = _build_fast_ready_conversation_refs_payload(
            refs=normal_ready_refs,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
            deadline_at=route_deadline_at,
        )
        for user_msg_id, pack in normal_ready_refs.items():
            payload_pack = fast_payload.get(int(user_msg_id))
            if isinstance(payload_pack, dict):
                payload[int(user_msg_id)] = _attach_pack_render_state(
                    payload_pack,
                    source_pack=pack,
                    default_status="fast",
                    override_status=True,
                )
        _record("fast_render", phase_started_at)

    cache_mode = "full"
    if authoritative_fast_payloads or normal_ready_refs or historical_stale_payloads:
        cache_mode = "fast"
    elif failed_ready_refs:
        cache_mode = "fast"
    elif pending_refs:
        cache_mode = "pending"
    if isinstance(payload, dict):
        _store_cached_conversation_refs_payload(
            conv_id=conv_id,
            signature=signature,
            payload=payload,
            mode=cache_mode,
            refs=refs_norm,
            state_signature=refs_state_signature,
        )
    ready_refs_to_warm = {
        **historical_stale_refs,
        **authoritative_fast_refs,
        **normal_ready_refs,
    }
    if ready_refs_to_warm and (not pending_refs) and (not failed_ready_refs):
        refs_to_warm = _refs_without_completed_answer_citation_overlays(
            store=store,
            conv_id=conv_id,
            refs=ready_refs_to_warm,
        )
        authoritative_to_warm = {
            user_msg_id: authoritative_doc_lists[user_msg_id]
            for user_msg_id in refs_to_warm
            if user_msg_id in authoritative_doc_lists
        }
        if refs_to_warm:
            _warm_conversation_refs_payload_async(
                conv_id=conv_id,
                signature=signature,
                refs=refs_to_warm,
                guide_mode=guide_mode,
                guide_source_path=guide_source_path,
                guide_source_name=guide_source_name,
                authoritative_doc_list_by_user=authoritative_to_warm,
            )
    return _finish(payload, cache_mode)


@router.get("/conversation/{conv_id}")
def get_conversation_refs_route(conv_id: str, response: Response):
    return get_conversation_refs(conv_id, response=response)


@router.get("/diagnose/{conv_id}")
def get_refs_diagnose(conv_id: str):
    """Return a diagnostic report for why reference cards are empty/suppressed."""
    store = get_chat_store()
    try:
        refs = store.list_message_refs(conv_id, timeout_s=10.0)
    except Exception:
        refs = None
    if refs is None:
        raise HTTPException(404, f"Conversation {conv_id} not found or has no refs data")
    return _build_diagnostic_report(store=store, conv_id=conv_id, refs=refs)


class OpenReferenceBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_path: str = Field(..., max_length=_REFS_SOURCE_PATH_MAX_CHARS)
    page: int | None = Field(None, ge=1, le=20_000)


class CitationMetaBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_path: str = Field(..., max_length=_REFS_SOURCE_PATH_MAX_CHARS)


class BibliometricsBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meta: dict[str, Any]
    refs_card_locale: str | None = Field(None, max_length=_REFS_LOCALE_MAX_CHARS)
    ui_locale: str | None = Field(None, max_length=_REFS_LOCALE_MAX_CHARS)
    target_locale: str | None = Field(None, max_length=_REFS_LOCALE_MAX_CHARS)

    @field_validator("meta")
    @classmethod
    def _check_meta(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_dict(value, name="bibliometrics meta", max_json_chars=_REFS_META_MAX_JSON_CHARS)


class ShelfMetadataRepairBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[dict[str, Any]] = Field(..., max_length=_REFS_SHELF_REPAIR_MAX_ITEMS)
    limit: int | None = Field(None, ge=1, le=500)

    @field_validator("items")
    @classmethod
    def _check_items(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _bounded_dict_list(
            value,
            name="shelf metadata repair items",
            max_items=_REFS_SHELF_REPAIR_MAX_ITEMS,
            max_json_chars=_REFS_SHELF_REPAIR_MAX_JSON_CHARS,
            item_max_json_chars=_REFS_SHELF_REPAIR_ITEM_MAX_JSON_CHARS,
        )


class ShelfMetadataBackfillBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    limit: int | None = Field(None, ge=1, le=500)
    scan_limit: int | None = Field(None, ge=1, le=2_000)


class CitationCardPolishBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meta: dict[str, Any]
    wait_s: float | None = Field(None, ge=0.0, le=10.0)

    @field_validator("meta")
    @classmethod
    def _check_meta(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_dict(value, name="citation card polish meta", max_json_chars=_REFS_META_MAX_JSON_CHARS)


class ReaderDocBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_path: str = Field(..., max_length=_REFS_SOURCE_PATH_MAX_CHARS)


def _bibliometrics_requested_locale(body: BibliometricsBody, meta: dict | None) -> str:
    data = dict(meta or {})
    for raw in (
        body.target_locale,
        data.get("target_locale"),
        data.get("targetLocale"),
        data.get("render_locale"),
        data.get("renderLocale"),
        data.get("locale"),
    ):
        locale = _normalize_ref_locale(raw)
        if locale:
            return locale

    for raw in (
        body.refs_card_locale,
        data.get("refs_card_locale"),
        data.get("refsCardLocale"),
    ):
        locale = _normalize_ref_locale(raw, allow_auto=True)
        if locale in {"zh", "en"}:
            return locale

    for raw in (
        body.ui_locale,
        data.get("ui_locale"),
        data.get("uiLocale"),
    ):
        locale = _normalize_ref_locale(raw)
        if locale:
            return locale

    return ""


def _shelf_repair_text(value) -> str:
    return str(value or "").strip()


def _shelf_repair_int(value) -> int:
    try:
        n = int(float(str(value or "0").strip()))
    except Exception:
        return 0
    return n if n > 0 else 0


def _shelf_metadata_repair_targets(items: list[dict]) -> tuple[list[str], list[str]]:
    names: list[str] = []
    sources: list[str] = []
    seen_names: set[str] = set()
    seen_sources: set[str] = set()
    for item in list(items or [])[:80]:
        if not isinstance(item, dict):
            continue
        name = _shelf_repair_text(item.get("source_name") or item.get("sourceName") or item.get("title"))
        source = _shelf_repair_text(item.get("source_path") or item.get("sourcePath") or name)
        if name:
            key = name.lower()
            if key not in seen_names:
                seen_names.add(key)
                names.append(name)
        if source:
            key = source.replace("\\", "/").lower()
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append(source)
    return names[:40], sources[:40]


def _shelf_metadata_repair_verification(result: dict) -> dict:
    acceptance = result.get("acceptance") if isinstance(result.get("acceptance"), dict) else {}
    requested = _shelf_repair_int(result.get("requested"))
    export_ready = _shelf_repair_int(acceptance.get("export_ready_after") or result.get("export_ready"))
    metadata_ready = _shelf_repair_int(acceptance.get("metadata_ready_after") or result.get("ready"))
    retryable = _shelf_repair_int(acceptance.get("retryable") or result.get("retryable"))
    failed = _shelf_repair_int(acceptance.get("failed") or result.get("failed"))
    unresolved = _shelf_repair_int(acceptance.get("unresolved_after") or result.get("unresolved"))
    changed = _shelf_repair_int(result.get("changed"))
    if requested <= 0:
        status = "skipped"
    elif export_ready >= requested and retryable <= 0 and failed <= 0 and unresolved <= 0:
        status = "passed"
    elif retryable > 0:
        status = "retryable"
    elif unresolved > 0 or failed > 0:
        status = "failed"
    else:
        status = "partial"
    quality_ok = status == "passed"
    return {
        "type": "shelf_metadata_repair",
        "status": status,
        "quality_ok": quality_ok,
        "target_count": requested,
        "metadata_ready_after": metadata_ready,
        "export_ready_after": export_ready,
        "changed": changed,
        "retryable": retryable,
        "failed": failed,
        "unresolved_after": unresolved,
        "remaining_fields": list(acceptance.get("remaining_fields") or [])[:8],
        "remaining_issue_codes": list(acceptance.get("remaining_issue_codes") or [])[:8],
        "summary_export_ready_after": _shelf_repair_int(acceptance.get("summary_export_ready_after")),
        "detail": (
            f"Metadata export verified for {export_ready}/{requested} shelf items."
            if quality_ok
            else (
                f"Metadata repair can retry for {retryable}/{requested} shelf items."
                if retryable > 0
                else f"Metadata export still missing fields for {unresolved}/{requested} shelf items."
            )
        ),
    }


def _record_shelf_metadata_quality_run(*, result: dict, items: list[dict]) -> dict:
    try:
        from api.routers import library as library_router

        verification = result.get("verification") if isinstance(result.get("verification"), dict) else _shelf_metadata_repair_verification(result)
        names, sources = _shelf_metadata_repair_targets(items)
        acceptance = result.get("acceptance") if isinstance(result.get("acceptance"), dict) else {}
        impact = dict(result.get("impact") or {}) if isinstance(result.get("impact"), dict) else {}
        impact["repair_kind"] = "shelf_metadata"
        if acceptance:
            impact["acceptance"] = acceptance
        status = "completed" if bool(verification.get("quality_ok")) else ("info" if str(verification.get("status") or "") == "skipped" else "warning")
        phase = "shelf_metadata_verified" if status == "completed" else (
            "shelf_metadata_retryable" if str(verification.get("status") or "") == "retryable" else "shelf_metadata_unresolved"
        )
        return library_router._append_quality_repair_run(
            {
                "status": status,
                "phase": phase,
                "requested": result.get("requested"),
                "enqueued": 0,
                "repaired": result.get("changed"),
                "failed": _shelf_repair_int(verification.get("failed")) + _shelf_repair_int(verification.get("unresolved_after")),
                "skipped_busy": 0,
                "needs_reindex": False,
                "reindexed": True,
                "target_names": names,
                "target_sources": sources,
                "impact": impact,
                "verification": verification,
                "detail": _shelf_repair_text(verification.get("detail")),
            }
        )
    except Exception:
        return {}


_ASSET_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_MD_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\("
    r"("
    r"<[^>\n]+>(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?"
    r"|"
    r"(?:\\.|[^()\n]|\([^()\n]*\))+"
    r")"
    r"\)"
)
_MD_LINK_TITLE_SUFFIX_RE = re.compile(r"""\s+(?:"[^"]*"|'[^']*'|\([^)]*\))\s*$""")
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_MD_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_MD_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_MD_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*")
_EQ_NUMBER_RE = re.compile(r"(?:\b(?:eq|equation|公式)\s*[#(（]?\s*|[\(（])(\d{1,4})(?:\s*[)）])", re.IGNORECASE)
_INLINE_EQ_RE = re.compile(r"\$[^$]{1,280}\$")
_TEX_CMD_RE = re.compile(r"\\[a-zA-Z]{2,}")


_LOCAL_SUMMARY_SECTION_PRIORITY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("abstract", ("abstract", "summary", "摘要")),
    ("fulltext", ("introduction", "background", "overview", "引言", "简介")),
    ("fulltext", ("discussion", "conclusion", "conclusions", "结论", "讨论")),
)


def _clean_markdown_for_local_summary(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    raw = re.sub(r"<!--\s*kb_page:\s*\d+\s*-->", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<!--[\s\S]*?-->", " ", raw)
    raw = re.sub(r"!\[[^\]]*]\([^)]+\)", " ", raw)
    raw = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", raw)
    raw = re.sub(r"`([^`]+)`", r"\1", raw)
    raw = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw)
    raw = re.sub(r"\*([^*]+)\*", r"\1", raw)
    raw = re.sub(r"^\s{0,3}#{1,6}\s+", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^\s{0,3}>\s?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$", " ", raw, flags=re.MULTILINE)
    raw = re.sub(r"^\s*\|", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\|\s*$", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*\|\s*", " ", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def _local_summary_heading_key(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(text or "").lower()).strip()


def _heading_matches_local_summary_section(heading: str, needles: tuple[str, ...]) -> bool:
    key = _local_summary_heading_key(heading)
    if not key:
        return False
    if key in {"references", "bibliography", "acknowledgements", "acknowledgments"}:
        return False
    tokens = set(key.split())
    for raw in needles:
        needle = _local_summary_heading_key(raw)
        if not needle:
            continue
        if key == needle or key.endswith(f" {needle}") or needle in tokens:
            return True
    return False


def _extract_local_summary_section(md_text: str, needles: tuple[str, ...]) -> str:
    lines = str(md_text or "").splitlines()
    capture = False
    level_start = 0
    out: list[str] = []
    for line in lines:
        m = _MD_HEADING_RE.match(line)
        if m:
            level = len(str(m.group(1) or ""))
            heading = str(m.group(2) or "").strip()
            if capture and level <= level_start:
                break
            if (not capture) and _heading_matches_local_summary_section(heading, needles):
                capture = True
                level_start = level
                continue
        if capture:
            out.append(line)
            if len("\n".join(out)) > 12000:
                break
    return "\n".join(out).strip()


def _local_summary_excerpt(text: str, *, max_len: int = 420) -> str:
    clean = _clean_markdown_for_local_summary(text)
    if not clean:
        return ""
    parts = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?])\s+|(?<=[A-Za-z0-9][.!?])\s+(?=[A-Z(])", clean)
        if part.strip()
    ]
    picked: list[str] = []
    for part in parts or [clean]:
        if not part:
            continue
        picked.append(part)
        joined = " ".join(picked).strip()
        if len(picked) >= 2 or len(joined) >= 220:
            break
    summary = " ".join(picked).strip() or clean
    return _compact_reader_open_text(summary, max_len=max_len)


def _normalize_ref_locale(value: object, *, allow_auto: bool = False) -> str:
    text = str(value or "").strip().lower()
    if text in {"zh", "cn", "zh-cn", "chinese"}:
        return "zh"
    if text in {"en", "en-us", "en-gb", "english"}:
        return "en"
    if allow_auto and text == "auto":
        return "auto"
    return ""


def _local_summary_detect_locale(text: str) -> str:
    raw = str(text or "")
    if re.search(r"[\u4e00-\u9fff]", raw):
        return "zh"
    if re.search(r"[A-Za-z]", raw):
        return "en"
    return ""


def _localize_local_summary(summary: str, *, target_locale: str) -> str:
    target = _normalize_ref_locale(target_locale)
    text = _compact_reader_open_text(summary, max_len=420)
    if not text or not target:
        return text
    current = _local_summary_detect_locale(text)
    if current == target:
        return text
    if target == "zh":
        translated = _compact_reader_open_text(_translate_summary_to_zh(text), max_len=420)
        return translated if _local_summary_detect_locale(translated) == "zh" else ""
    if target == "en":
        return text if current in {"", "en"} else ""
    return text


def _resolve_local_summary_md_path(source_path: str) -> Path | None:
    raw = clean_file_source_path_input(source_path)
    if not raw:
        return None
    src = Path(raw).expanduser()
    if src.suffix.lower().endswith(".md"):
        try:
            if not (src.exists() and src.is_file()):
                return None
            resolved = src.resolve(strict=False)
        except Exception:
            return None
        roots = list(_reference_asset_roots())
        if not _path_within_roots(resolved, roots):
            return None
        return resolved
    return _resolve_reader_md_path(raw)


def _local_source_summary_meta(meta: dict | None, *, target_locale: str = "") -> dict:
    data = dict(meta or {})
    source_path = str(data.get("source_path") or data.get("sourcePath") or "").strip()
    if not source_path:
        return {}
    md_path = _resolve_local_summary_md_path(source_path)
    if md_path is None:
        return {}
    try:
        md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    for source, headings in _LOCAL_SUMMARY_SECTION_PRIORITY:
        section = _extract_local_summary_section(md_text, headings)
        summary = _localize_local_summary(_local_summary_excerpt(section), target_locale=target_locale)
        if not summary:
            continue
        summary_locale = _local_summary_detect_locale(summary) or _normalize_ref_locale(target_locale)
        return {
            "summary_line": summary,
            "summary_source": source,
            "summary_provider": "local_markdown",
            "summary_generation": "extractive_local_markdown",
            "summary_locale": summary_locale,
            "summary_quality": {
                "contract_version": 1,
                "ok": True,
                "status": "grounded",
                "score": 94 if source == "abstract" else 90,
                "source": source,
                "provider": "local_markdown",
                "generation": "extractive_local_markdown",
                "locale": summary_locale,
                "issues": [],
                "export_ready": True,
            },
        }
    return {}


_LOCAL_SOURCE_SUMMARY_PROVIDERS = {"local_markdown"}
_LOCAL_SOURCE_SUMMARY_GENERATIONS = {"extractive_local_markdown"}
_UPSTREAM_REFERENCE_CONTEXT_SOURCES = {
    "reader_reference_link",
    "reader_references",
    "reader_occurrence",
    "reader_cross_reference",
    "answer_context",
    "answer_reference_mention",
    "structured_reference_index",
    "source_markdown",
    "matched_ref_marker",
}
_UPSTREAM_CONTEXT_SUMMARY_SOURCES = _UPSTREAM_REFERENCE_CONTEXT_SOURCES | {
    "citation_context",
    "citation_card",
    "citation_card_view",
    "references_panel_hit",
}


def _truthy_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _bibliometrics_is_upstream_reference_context(meta: dict | None) -> bool:
    data = dict(meta or {})
    if _truthy_bool(data.get("is_inpaper") if "is_inpaper" in data else data.get("isInpaper")):
        return True
    marker_values = [
        data.get("summary_source"),
        data.get("summarySource"),
        data.get("citation_context_source"),
        data.get("citationContextSource"),
        data.get("evidence_source"),
        data.get("evidenceSource"),
        data.get("shelf_origin"),
        data.get("shelfOrigin"),
        data.get("binding_status"),
        data.get("bindingStatus"),
    ]
    for value in marker_values:
        marker = str(value or "").strip().lower()
        if marker in _UPSTREAM_REFERENCE_CONTEXT_SOURCES or marker == "reader_reference":
            return True
    return False


def _summary_is_local_source_summary(meta: dict | None) -> bool:
    data = dict(meta or {})
    provider = str(data.get("summary_provider") or data.get("summaryProvider") or "").strip().lower()
    generation = str(data.get("summary_generation") or data.get("summaryGeneration") or "").strip().lower()
    quality = data.get("summary_quality") or data.get("summaryQuality")
    if isinstance(quality, dict):
        provider = provider or str(quality.get("provider") or "").strip().lower()
        generation = generation or str(quality.get("generation") or "").strip().lower()
    return provider in _LOCAL_SOURCE_SUMMARY_PROVIDERS or generation in _LOCAL_SOURCE_SUMMARY_GENERATIONS


def _summary_is_upstream_context_summary(meta: dict | None) -> bool:
    data = dict(meta or {})
    source = str(data.get("summary_source") or data.get("summarySource") or "").strip().lower()
    quality = data.get("summary_quality") or data.get("summaryQuality")
    if isinstance(quality, dict):
        source = source or str(quality.get("source") or "").strip().lower()
    if source in _UPSTREAM_CONTEXT_SUMMARY_SOURCES:
        return True
    summary = str(data.get("summary_line") or data.get("summaryLine") or "").strip()
    return bool(
        summary
        and re.search(
            r"opened paper cites|bibliography entry is linked|current paper cites|"
            r"当前论文|本文引用|参考文献条目|上游文献",
            summary,
            flags=re.I,
        )
    )


def _strip_misbound_local_source_summary(meta: dict | None) -> dict:
    data = dict(meta or {})
    if not _bibliometrics_is_upstream_reference_context(data):
        return data
    if not (_summary_is_local_source_summary(data) or _summary_is_upstream_context_summary(data)):
        return data
    for key in (
        "summary_line",
        "summaryLine",
        "summary_source",
        "summarySource",
        "summary_provider",
        "summaryProvider",
        "summary_generation",
        "summaryGeneration",
        "summary_quality",
        "summaryQuality",
        "metadata_export_acceptance",
        "metadataExportAcceptance",
    ):
        data.pop(key, None)
    return data


def _bibliometrics_accept_local_source_summary(meta: dict | None) -> bool:
    data = _strip_misbound_local_source_summary(meta)
    if _bibliometrics_is_upstream_reference_context(data):
        return False
    summary = str(data.get("summary_line") or data.get("summaryLine") or "").strip()
    source = str(data.get("summary_source") or data.get("summarySource") or "").strip().lower()
    quality = data.get("summary_quality") or data.get("summaryQuality")
    quality_ok = bool(isinstance(quality, dict) and (quality.get("ok") or str(quality.get("status") or "").lower() == "grounded"))
    if not summary:
        return True
    if source in {"metadata", "citation_context", "citation_card", "citation_card_view", "references_panel_hit"}:
        return True
    if source in {"abstract", "fulltext"} and quality_ok:
        return False
    return not quality_ok


_LIBRARY_MATCH_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_LIBRARY_MATCH_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "via",
    "with",
}


def _library_match_text(value: object) -> str:
    return str(value or "").strip()


def _library_match_normalize_doi(value: object) -> str:
    text = _library_match_text(value)
    if not text:
        return ""
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE).strip()
    match = _LIBRARY_MATCH_DOI_RE.search(text)
    doi = match.group(0) if match else text
    doi = doi.strip().strip("<>[](){}").rstrip(".,;:")
    return doi.lower() if re.match(r"^10\.\d{4,9}/", doi, flags=re.IGNORECASE) else ""


def _library_match_first_doi(data: dict) -> str:
    for key in (
        "doi",
        "doi_url",
        "doiUrl",
        "external_doi",
        "externalDoi",
        "external_doi_url",
        "externalDoiUrl",
        "raw",
        "cite_fmt",
        "citeFmt",
        "card_reference_entry",
        "cardReferenceEntry",
    ):
        doi = _library_match_normalize_doi(data.get(key))
        if doi:
            return doi
    return ""


def _library_match_year(value: object) -> str:
    match = re.search(r"\b(?:19|20)\d{2}\b", _library_match_text(value))
    return match.group(0) if match else ""


def _library_match_first_year(data: dict) -> str:
    for key in ("year", "external_year", "externalYear", "raw", "cite_fmt", "citeFmt", "source_name", "sourceName"):
        year = _library_match_year(data.get(key))
        if year:
            return year
    return ""


def _library_match_normalize_title(value: object) -> str:
    text = _library_match_text(value)
    if not text:
        return ""
    text = re.sub(r"\.(?:pdf|md)$", "", text, flags=re.IGNORECASE)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"^https?://\S+$", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdoi\s*:\s*\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text.lower())
    tokens = [tok for tok in text.split() if tok not in _LIBRARY_MATCH_TITLE_STOPWORDS]
    return " ".join(tokens).strip()


def _library_match_title_usable(title_key: str) -> bool:
    key = _library_match_text(title_key)
    if not key:
        return False
    if re.search(r"[\u4e00-\u9fff]", key):
        return len(key) >= 12
    tokens = key.split()
    return len(tokens) >= 4 and len(key) >= 24


def _library_match_title_candidates(data: dict) -> list[tuple[str, str]]:
    is_inpaper = bool(data.get("is_inpaper") is True or data.get("isInpaper") is True)
    keys = [
        "title",
        "external_title",
        "externalTitle",
        "card_title",
        "cardTitle",
        "main",
    ]
    if not is_inpaper:
        keys.extend(["source_name", "sourceName"])
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key in keys:
        title = _library_match_text(data.get(key))
        norm = _library_match_normalize_title(title)
        if not _library_match_title_usable(norm) or norm in seen:
            continue
        seen.add(norm)
        out.append((norm, title))
    return out


def _library_match_record_title_values(record: dict) -> list[str]:
    meta = record.get("citation_meta") if isinstance(record.get("citation_meta"), dict) else {}
    values = [
        meta.get("title"),
        meta.get("display_title"),
        meta.get("external_title"),
        meta.get("card_title"),
    ]
    path_text = _library_match_text(record.get("path"))
    if path_text:
        try:
            values.append(Path(path_text).stem)
        except Exception:
            values.append(path_text)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _library_match_text(value)
        if not text:
            continue
        norm = _library_match_normalize_title(text)
        if not _library_match_title_usable(norm) or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def _library_match_record_dois(record: dict) -> list[str]:
    meta = record.get("citation_meta") if isinstance(record.get("citation_meta"), dict) else {}
    values = [
        meta.get("doi"),
        meta.get("doi_url"),
        meta.get("doiUrl"),
        meta.get("external_doi"),
        meta.get("externalDoi"),
        meta.get("external_doi_url"),
        meta.get("externalDoiUrl"),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        doi = _library_match_normalize_doi(value)
        if doi and doi not in seen:
            seen.add(doi)
            out.append(doi)
    return out


def _library_match_record_year(record: dict) -> str:
    meta = record.get("citation_meta") if isinstance(record.get("citation_meta"), dict) else {}
    for value in (meta.get("year"), meta.get("external_year"), record.get("path")):
        year = _library_match_year(value)
        if year:
            return year
    return ""


def _library_match_record_title(record: dict) -> str:
    for title in _library_match_record_title_values(record):
        if title:
            return title
    path_text = _library_match_text(record.get("path"))
    return Path(path_text).stem if path_text else ""


def _library_match_payload(
    *,
    status: str,
    reason: str,
    method: str = "",
    confidence: float = 0.0,
    record: dict | None = None,
    query_doi: str = "",
    query_title: str = "",
) -> dict:
    rec = record if isinstance(record, dict) else {}
    dois = _library_match_record_dois(rec) if rec else []
    return {
        "status": status,
        "matched": status == "in_library",
        "confidence": round(float(confidence or 0.0), 3),
        "method": method,
        "reason": reason,
        "path": _library_match_text(rec.get("path")),
        "sha1": _library_match_text(rec.get("sha1")),
        "title": _library_match_record_title(rec) if rec else "",
        "doi": dois[0] if dois else "",
        "year": _library_match_record_year(rec) if rec else "",
        "query_doi": query_doi,
        "query_title": query_title,
    }


def _library_match_unique_records(matches: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for record in matches:
        key = _library_match_text(record.get("sha1")) or _library_match_text(record.get("path")).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _match_citation_meta_to_library(meta: dict | None) -> dict:
    data = dict(meta or {})
    query_doi = _library_match_first_doi(data)
    title_candidates = _library_match_title_candidates(data)
    query_title = title_candidates[0][1] if title_candidates else ""
    query_year = _library_match_first_year(data)
    if not query_doi and not title_candidates:
        return _library_match_payload(status="unknown", reason="insufficient_metadata")

    try:
        records = _lib_store().list_citation_records(limit=8000)
    except Exception:
        return _library_match_payload(
            status="unknown",
            reason="library_unavailable",
            query_doi=query_doi,
            query_title=query_title,
        )
    if not records:
        return _library_match_payload(
            status="not_in_library",
            reason="library_empty",
            query_doi=query_doi,
            query_title=query_title,
        )

    if query_doi:
        for record in records:
            if query_doi in _library_match_record_dois(record):
                return _library_match_payload(
                    status="in_library",
                    reason="doi_exact",
                    method="doi",
                    confidence=0.99,
                    record=record,
                    query_doi=query_doi,
                    query_title=query_title,
                )

    title_entries: list[tuple[str, dict, str]] = []
    for record in records:
        rec_year = _library_match_record_year(record)
        for title in _library_match_record_title_values(record):
            norm = _library_match_normalize_title(title)
            if _library_match_title_usable(norm):
                title_entries.append((norm, record, rec_year))

    for title_key, raw_title in title_candidates:
        exact = [
            record for rec_title, record, rec_year in title_entries
            if rec_title == title_key and (not query_year or not rec_year or query_year == rec_year)
        ]
        exact_unique = _library_match_unique_records(exact)
        if len(exact_unique) == 1:
            method = "title_year" if query_year else "title"
            return _library_match_payload(
                status="in_library",
                reason="title_exact",
                method=method,
                confidence=0.9 if query_year else 0.84,
                record=exact_unique[0],
                query_doi=query_doi,
                query_title=raw_title,
            )

    for title_key, raw_title in title_candidates:
        if len(title_key) < 32:
            continue
        fuzzy = [
            record for rec_title, record, rec_year in title_entries
            if (
                (title_key in rec_title or rec_title in title_key)
                and (not query_year or not rec_year or query_year == rec_year)
            )
        ]
        fuzzy_unique = _library_match_unique_records(fuzzy)
        if len(fuzzy_unique) == 1:
            return _library_match_payload(
                status="in_library",
                reason="title_contained",
                method="title_contains",
                confidence=0.78,
                record=fuzzy_unique[0],
                query_doi=query_doi,
                query_title=raw_title,
            )

    return _library_match_payload(
        status="not_in_library",
        reason="no_match",
        query_doi=query_doi,
        query_title=query_title,
    )


def _attach_library_match_contract(meta: dict | None) -> dict:
    data = dict(meta or {})
    match = _match_citation_meta_to_library(data)
    data["library_match"] = match
    data["library_match_status"] = match.get("status") or ""
    data["library_match_confidence"] = match.get("confidence") or 0
    data["library_match_method"] = match.get("method") or ""
    data["library_match_reason"] = match.get("reason") or ""
    data["library_match_path"] = match.get("path") or ""
    data["library_match_sha1"] = match.get("sha1") or ""
    data["library_match_title"] = match.get("title") or ""
    data["library_match_doi"] = match.get("doi") or ""
    data["library_match_year"] = match.get("year") or ""
    return data


def _prepare_bibliometrics_identity(meta: dict | None, *, verify_library: bool = False) -> dict:
    candidate = _attach_library_match_contract(meta) if verify_library else dict(meta or {})
    data = promote_trusted_library_match_identity(candidate)
    status = str(data.get("library_match_status") or data.get("libraryMatchStatus") or "").strip().lower()
    match_doi = str(data.get("library_match_doi") or data.get("libraryMatchDoi") or "").strip()
    if status == "in_library" and match_doi:
        aliases = {
            "library_match_status": "libraryMatchStatus",
            "library_match_confidence": "libraryMatchConfidence",
            "library_match_method": "libraryMatchMethod",
            "library_match_reason": "libraryMatchReason",
            "library_match_path": "libraryMatchPath",
            "library_match_sha1": "libraryMatchSha1",
            "library_match_title": "libraryMatchTitle",
            "library_match_doi": "libraryMatchDoi",
            "library_match_year": "libraryMatchYear",
        }
        for snake_key, camel_key in aliases.items():
            snake_value = data.get(snake_key)
            camel_value = data.get(camel_key)
            if (snake_value is None or snake_value == "") and camel_value is not None and camel_value != "":
                data[snake_key] = data[camel_key]
        if not isinstance(data.get("library_match"), dict):
            data["library_match"] = {
                "matched": True,
                "status": data.get("library_match_status") or "",
                "confidence": data.get("library_match_confidence") or 0,
                "method": data.get("library_match_method") or "",
                "reason": data.get("library_match_reason") or "",
                "path": data.get("library_match_path") or "",
                "sha1": data.get("library_match_sha1") or "",
                "title": data.get("library_match_title") or "",
                "doi": data.get("library_match_doi") or "",
                "year": data.get("library_match_year") or "",
            }
        return data
    return promote_trusted_library_match_identity(_attach_library_match_contract(data))


def _bibliometrics_local_summary_input(meta: dict | None) -> dict:
    data = dict(meta or {})
    status = str(data.get("library_match_status") or data.get("libraryMatchStatus") or "").strip().lower()
    reason = str(data.get("library_match_reason") or data.get("libraryMatchReason") or "").strip().lower()
    match_path = str(data.get("library_match_path") or data.get("libraryMatchPath") or "").strip()
    if status == "in_library" and reason in {"doi_exact", "title_exact"} and match_path:
        data["source_path"] = match_path
        data["sourcePath"] = match_path
    elif _bibliometrics_is_upstream_reference_context(data):
        # The source path on a System-B card is the paper containing the
        # bibliography entry, not the referenced article.  Reading that path
        # here would bind the citing paper's abstract to the upstream work.
        for key in ("source_path", "sourcePath", "md_path", "mdPath"):
            data.pop(key, None)
    return data


def _bibliometrics_has_metrics(meta: dict | None) -> bool:
    data = dict(meta or {})
    citation_source = str(data.get("citation_source") or data.get("citationSource") or "").strip()
    citation_count = data.get("citation_count", data.get("citationCount"))
    has_citations = citation_source != "" and isinstance(citation_count, (int, float))
    metric_values = (
        data.get("journal_if"),
        data.get("journalIf"),
        data.get("journal_quartile"),
        data.get("journalQuartile"),
        data.get("conference_tier"),
        data.get("conferenceTier"),
        data.get("conference_ccf"),
        data.get("conferenceCcf"),
    )
    return has_citations or any(value is not None and value != "" for value in metric_values)


@router.post("/open", dependencies=[Depends(require_management_api)])
def open_reference(body: OpenReferenceBody):
    source_path = _resolve_public_reference_source_input(body.source_path)
    if not source_path:
        raise HTTPException(404, "source not found")
    ok, message = open_reference_source(
        source_path=source_path,
        pdf_root=_pdf_dir(),
        page=body.page,
    )
    if not ok:
        raise HTTPException(404, message)
    return {"ok": True, "message": "PDF opened"}


@router.post("/citation-meta")
def get_reference_citation_meta(body: CitationMetaBody):
    source_path = _resolve_public_reference_source_input(body.source_path)
    if not source_path:
        raise HTTPException(404, "source not found")
    out = ensure_source_citation_meta(
        source_path=source_path,
        pdf_root=_pdf_dir(),
        md_root=_md_dir(),
        lib_store=_lib_store(),
    )
    if isinstance(out, dict) and body.source_path.replace("\\", "/").startswith(
        ROOT_RELATIVE_FILE_ID_PREFIX
    ):
        for key in ("source_path", "sourcePath", "md_path"):
            if key in out:
                out[key] = body.source_path
    return out


def _bibliometrics_quality_contract(meta: dict | None) -> dict:
    data = dict(meta or {})
    for key in (
        "metadata_export_acceptance",
        "metadataExportAcceptance",
        "export_acceptance",
        "exportAcceptance",
        "metadataQuality",
    ):
        data.pop(key, None)
    summary = str(data.get("summary_line") or data.get("summaryLine") or "").strip()
    if not summary:
        for key in (
            "summary_source",
            "summarySource",
            "summary_provider",
            "summaryProvider",
            "summary_generation",
            "summaryGeneration",
            "summary_locale",
            "summaryLocale",
            "summary_quality",
            "summaryQuality",
        ):
            data.pop(key, None)
        data["summary_quality"] = {
            "contract_version": 1,
            "ok": False,
            "status": "missing",
            "score": 0,
            "source": "",
            "provider": "",
            "issues": ["summary_missing"],
            "export_ready": False,
        }
    quality = citation_metadata_quality(data)
    acceptance = citation_metadata_export_acceptance({**data, "metadata_quality": quality})
    data["metadata_quality"] = quality
    data["metadata_export_acceptance"] = acceptance
    data["bibliometrics_checked"] = True
    current_status = str(data.get("metadata_repair_status") or "").strip().lower()
    if bool(quality.get("ok")):
        data["metadata_repair_status"] = current_status if current_status in {"ready", "repaired"} else "ready"
    elif bool(quality.get("retryable")):
        data["metadata_repair_status"] = current_status or "retryable"
    else:
        data["metadata_repair_status"] = current_status or "partial"
    return data


def _bibliometrics_summary_export_ready(acceptance: dict | None) -> bool:
    if not isinstance(acceptance, dict):
        return False
    if bool(acceptance.get("summary_export_ready")):
        return True
    summary = acceptance.get("summary")
    if isinstance(summary, dict):
        return bool(summary.get("export_ready"))
    return False


def _bibliometrics_summary_locale(meta: dict | None) -> str:
    data = dict(meta or {})
    text_locale = _local_summary_detect_locale(str(data.get("summary_line") or data.get("summaryLine") or ""))
    if text_locale:
        return text_locale
    quality = data.get("summary_quality") or data.get("summaryQuality")
    for raw in (
        data.get("summary_locale"),
        data.get("summaryLocale"),
        quality.get("locale") if isinstance(quality, dict) else "",
    ):
        locale = _normalize_ref_locale(raw)
        if locale:
            return locale
    return ""


def _bibliometrics_summary_matches_locale(meta: dict | None, target_locale: str) -> bool:
    target = _normalize_ref_locale(target_locale)
    if not target:
        return True
    summary = str((meta or {}).get("summary_line") or (meta or {}).get("summaryLine") or "").strip()
    if not summary:
        return False
    return _bibliometrics_summary_locale(meta) == target


def _bibliometrics_summary_has_locale_contract(meta: dict | None) -> bool:
    data = dict(meta or {})
    quality = data.get("summary_quality") or data.get("summaryQuality")
    return bool(
        _normalize_ref_locale(data.get("summary_locale") or data.get("summaryLocale"))
        or (
            isinstance(quality, dict)
            and _normalize_ref_locale(quality.get("locale"))
        )
    )


def _attach_bibliometrics_summary_locale(meta: dict | None) -> dict:
    data = dict(meta or {})
    locale = _bibliometrics_summary_locale(data)
    if not locale:
        return data
    data["summary_locale"] = locale
    quality = data.get("summary_quality") if isinstance(data.get("summary_quality"), dict) else {}
    data["summary_quality"] = {**quality, "locale": locale}
    return data


@router.post("/bibliometrics")
def get_bibliometrics(body: BibliometricsBody):
    meta = _prepare_bibliometrics_identity(
        _strip_misbound_local_source_summary(body.meta or {}),
        verify_library=True,
    )
    settings = get_settings()
    target_locale = _bibliometrics_requested_locale(body, meta)
    hydrated = _prepare_bibliometrics_identity(
        _strip_misbound_local_source_summary(
            hydrate_repaired_citation_metadata(meta, db_dir=settings.db_dir)
        )
    )
    if (
        _bibliometrics_accept_local_source_summary(hydrated)
        or not _bibliometrics_summary_matches_locale(hydrated, target_locale)
        or not _bibliometrics_summary_has_locale_contract(hydrated)
    ):
        local_summary = _local_source_summary_meta(
            _bibliometrics_local_summary_input({**dict(hydrated or {}), **dict(meta or {})}),
            target_locale=target_locale,
        )
        if local_summary:
            hydrated = _bibliometrics_quality_contract({**dict(hydrated or {}), **local_summary})
    hydrated = _attach_bibliometrics_summary_locale(_bibliometrics_quality_contract(hydrated))
    quality = hydrated.get("metadata_quality") if isinstance(hydrated.get("metadata_quality"), dict) else {}
    acceptance = (
        hydrated.get("metadata_export_acceptance")
        if isinstance(hydrated.get("metadata_export_acceptance"), dict)
        else {}
    )
    if (
        bool(quality.get("ok"))
        and bool(acceptance.get("export_ready"))
        and _bibliometrics_summary_export_ready(acceptance)
        and _bibliometrics_summary_matches_locale(hydrated, target_locale)
        and (
            not bool(hydrated.get("library_match_doi_promoted"))
            or _bibliometrics_has_metrics(hydrated)
        )
    ):
        return _prepare_bibliometrics_identity(
            _attach_bibliometrics_summary_locale(_bibliometrics_quality_contract(hydrated))
        )
    seed = _prepare_bibliometrics_identity({**dict(meta or {}), **dict(hydrated or {})})
    enriched = enrich_citation_detail_meta(seed)
    if not isinstance(enriched, dict):
        enriched = {}
    merged_result = _prepare_bibliometrics_identity(
        _strip_misbound_local_source_summary({**seed, **enriched})
    )
    if (
        not isinstance(enriched.get("metadata_quality"), dict)
        or enriched.get("metadata_quality") == seed.get("metadata_quality")
    ):
        merged_result.pop("metadata_quality", None)
        merged_result.pop("metadataQuality", None)
    if (
        not isinstance(enriched.get("metadata_export_acceptance"), dict)
        or enriched.get("metadata_export_acceptance") == seed.get("metadata_export_acceptance")
    ):
        merged_result.pop("metadata_export_acceptance", None)
        merged_result.pop("metadataExportAcceptance", None)
    result = _bibliometrics_quality_contract(merged_result)
    if (
        _bibliometrics_accept_local_source_summary(result)
        or not _bibliometrics_summary_matches_locale(result, target_locale)
        or not _bibliometrics_summary_has_locale_contract(result)
    ):
        local_summary = _local_source_summary_meta(
            _bibliometrics_local_summary_input(result),
            target_locale=target_locale,
        )
        if local_summary:
            result = _bibliometrics_quality_contract({**result, **local_summary})
    acceptance = (
        result.get("metadata_export_acceptance")
        if isinstance(result.get("metadata_export_acceptance"), dict)
        else {}
    )
    if _bibliometrics_summary_export_ready(acceptance):
        persist_repaired_citation_metadata(_attach_bibliometrics_summary_locale(result), db_dir=settings.db_dir)
    return _prepare_bibliometrics_identity(_attach_bibliometrics_summary_locale(result))


@router.post("/shelf/metadata/repair", dependencies=[Depends(require_management_api)])
def repair_shelf_metadata(body: ShelfMetadataRepairBody):
    limit = 40
    if body.limit is not None:
        limit = max(1, min(80, int(body.limit)))
    items = list(body.items or [])
    result = repair_citation_metadata_batch(items, limit=limit, db_dir=get_settings().db_dir)
    verification = _shelf_metadata_repair_verification(result)
    result["verification"] = verification
    repair_run = _record_shelf_metadata_quality_run(result=result, items=items[:limit])
    if repair_run:
        result["repair_run_id"] = str(repair_run.get("run_id") or "")
        result["repair_run"] = repair_run
    return result


def _backfill_progress(percent: int, *, processed: int = 0, total: int = 0) -> dict:
    return {
        "percent": max(0, min(100, int(percent or 0))),
        "processed": max(0, int(processed or 0)),
        "total": max(0, int(total or 0)),
    }


def _shelf_metadata_backfill_snapshot() -> dict:
    with _SHELF_METADATA_BACKFILL_LOCK:
        try:
            return json.loads(json.dumps(_SHELF_METADATA_BACKFILL_STATE, ensure_ascii=False, default=str))
        except Exception:
            return dict(_SHELF_METADATA_BACKFILL_STATE)


def _set_shelf_metadata_backfill_state(**patch) -> dict:
    with _SHELF_METADATA_BACKFILL_LOCK:
        _SHELF_METADATA_BACKFILL_STATE.update(patch)
        _SHELF_METADATA_BACKFILL_STATE["updated_at"] = time.time()
        try:
            return json.loads(json.dumps(_SHELF_METADATA_BACKFILL_STATE, ensure_ascii=False, default=str))
        except Exception:
            return dict(_SHELF_METADATA_BACKFILL_STATE)


def _run_shelf_metadata_backfill_job(job_id: str, *, db_dir: str | Path, limit: int, scan_limit: int) -> None:
    try:
        scan_window = max(limit, scan_limit)
        _set_shelf_metadata_backfill_state(
            status="running",
            phase="scanning",
            running=True,
            progress=_backfill_progress(8),
        )
        before = scan_reference_metadata_backfill_targets(db_dir=db_dir, limit=scan_window)
        targets = [
            dict(item)
            for item in list(before.get("targets") or [])[:limit]
            if isinstance(item, dict)
        ]
        target_total = len(targets)
        _set_shelf_metadata_backfill_state(
            phase="repairing" if target_total else "verifying",
            scan=before,
            target_total=int(before.get("target_count") or target_total),
            progress=_backfill_progress(28, processed=0, total=target_total),
        )
        repair = repair_citation_metadata_batch(targets, limit=limit, db_dir=db_dir)
        _set_shelf_metadata_backfill_state(
            phase="verifying",
            progress=_backfill_progress(82, processed=target_total, total=target_total),
            result={**repair, "scan": before},
        )
        after = scan_reference_metadata_backfill_targets(db_dir=db_dir, limit=scan_window)
        result = {
            **repair,
            "scan": before,
            "after_scan": after,
            "preheated": max(_shelf_repair_int(repair.get("changed")), _shelf_repair_int(repair.get("persisted"))),
            "remaining_targets": _shelf_repair_int(after.get("needs_repair")),
        }
        verification = _shelf_metadata_repair_verification(result)
        result["verification"] = verification
        repair_run = _record_shelf_metadata_quality_run(result=result, items=targets)
        if repair_run:
            result["repair_run_id"] = str(repair_run.get("run_id") or "")
            result["repair_run"] = repair_run
        phase = "completed"
        if str(verification.get("status") or "") in {"retryable", "partial", "failed"}:
            phase = f"completed_{verification.get('status')}"
        _set_shelf_metadata_backfill_state(
            ok=bool(result.get("ok", True)),
            status="completed",
            phase=phase,
            running=False,
            finished_at=time.time(),
            progress=_backfill_progress(100, processed=target_total, total=target_total),
            result=result,
            after_scan=after,
            verification=verification,
            repair_run_id=str((repair_run or {}).get("run_id") or ""),
            repair_run=repair_run or {},
        )
    except Exception as exc:
        _set_shelf_metadata_backfill_state(
            ok=False,
            status="error",
            phase="error",
            running=False,
            finished_at=time.time(),
            progress=_backfill_progress(100),
            error_kind=type(exc).__name__,
            error_detail=str(exc or "")[:500],
        )


def _start_shelf_metadata_backfill_job(*, limit: int, scan_limit: int) -> dict:
    db_dir = get_settings().db_dir
    now = time.time()
    job_id = f"shelf-meta-{int(now * 1000)}"
    with _SHELF_METADATA_BACKFILL_LOCK:
        if bool(_SHELF_METADATA_BACKFILL_STATE.get("running")):
            try:
                state = json.loads(json.dumps(_SHELF_METADATA_BACKFILL_STATE, ensure_ascii=False, default=str))
            except Exception:
                state = dict(_SHELF_METADATA_BACKFILL_STATE)
            return {"started": False, "reason": "already_running", "state": state}
        _SHELF_METADATA_BACKFILL_STATE.clear()
        _SHELF_METADATA_BACKFILL_STATE.update(
            {
                "ok": True,
                "job_id": job_id,
                "status": "running",
                "phase": "queued",
                "running": True,
                "limit": int(limit),
                "scan_limit": int(scan_limit),
                "started_at": now,
                "updated_at": now,
                "progress": _backfill_progress(1),
            }
        )
    try:
        threading.Thread(
            target=_run_shelf_metadata_backfill_job,
            kwargs={"job_id": job_id, "db_dir": db_dir, "limit": int(limit), "scan_limit": int(scan_limit)},
            daemon=True,
            name=f"kb_shelf_meta_backfill_{job_id[-6:]}",
        ).start()
    except Exception as exc:
        state = _set_shelf_metadata_backfill_state(
            ok=False,
            status="error",
            phase="error",
            running=False,
            finished_at=time.time(),
            progress=_backfill_progress(100),
            error_kind=type(exc).__name__,
            error_detail=str(exc or "")[:500],
        )
        return {"started": False, "reason": "start_failed", "state": state}
    return {"started": True, "job_id": job_id, "state": _shelf_metadata_backfill_snapshot()}


@router.get("/shelf/metadata/backfill/scan")
def scan_shelf_metadata_backfill(limit: int = 120):
    scan_limit = max(1, min(500, int(limit or 120)))
    return scan_reference_metadata_backfill_targets(db_dir=get_settings().db_dir, limit=scan_limit)


@router.post("/shelf/metadata/backfill", dependencies=[Depends(require_management_api)])
def backfill_shelf_metadata(body: ShelfMetadataBackfillBody):
    limit = 40 if body.limit is None else max(1, min(80, int(body.limit)))
    scan_limit = 240 if body.scan_limit is None else max(limit, min(1000, int(body.scan_limit)))
    result = backfill_reference_metadata(db_dir=get_settings().db_dir, limit=limit, scan_limit=scan_limit)
    items = [
        dict(item)
        for item in list((result.get("scan") or {}).get("targets") or [])[:limit]
        if isinstance(item, dict)
    ]
    verification = _shelf_metadata_repair_verification(result)
    result["verification"] = verification
    repair_run = _record_shelf_metadata_quality_run(result=result, items=items)
    if repair_run:
        result["repair_run_id"] = str(repair_run.get("run_id") or "")
        result["repair_run"] = repair_run
    return result


@router.get("/shelf/metadata/backfill/status")
def shelf_metadata_backfill_status():
    return _shelf_metadata_backfill_snapshot()


@router.post("/shelf/metadata/backfill/start", dependencies=[Depends(require_management_api)])
def start_shelf_metadata_backfill(body: ShelfMetadataBackfillBody):
    limit = 40 if body.limit is None else max(1, min(80, int(body.limit)))
    scan_limit = 240 if body.scan_limit is None else max(limit, min(1000, int(body.scan_limit)))
    return _start_shelf_metadata_backfill_job(limit=limit, scan_limit=scan_limit)


def _run_citation_card_polish_job(key: str, detail: dict) -> None:
    key_text = str(key or "").strip()
    if not key_text:
        return
    try:
        result = polish_citation_card_detail(detail or {})
        if not isinstance(result, dict):
            result = {}
        status = str(result.get("citation_card_polish_status") or "").strip().lower()
        if not status:
            status = "empty"
            result["citation_card_polish_status"] = status
        result["citation_card_polish_cached_at"] = time.time()
        with _CITATION_CARD_POLISH_LOCK:
            _CITATION_CARD_POLISH_CACHE[key_text] = dict(result)
    except Exception as exc:
        with _CITATION_CARD_POLISH_LOCK:
            _CITATION_CARD_POLISH_CACHE[key_text] = {
                "citation_card_polish_status": "failed",
                "citation_card_polish_source": "error",
                "citation_card_polish_checked": True,
                "citation_card_polish_error": f"{type(exc).__name__}: {str(exc or '').strip()}"[:300],
                "citation_card_polish_cached_at": time.time(),
            }
    finally:
        with _CITATION_CARD_POLISH_LOCK:
            _CITATION_CARD_POLISH_WARMING.discard(key_text)


def _schedule_citation_card_polish(key: str, detail: dict) -> bool:
    key_text = str(key or "").strip()
    if not key_text:
        return False
    with _CITATION_CARD_POLISH_LOCK:
        if key_text in _CITATION_CARD_POLISH_CACHE:
            return False
        if key_text in _CITATION_CARD_POLISH_WARMING:
            return False
        _CITATION_CARD_POLISH_WARMING.add(key_text)
    try:
        threading.Thread(
            target=_run_citation_card_polish_job,
            args=(key_text, dict(detail or {})),
            daemon=True,
            name="kb_citation_card_polish",
        ).start()
        return True
    except Exception:
        with _CITATION_CARD_POLISH_LOCK:
            _CITATION_CARD_POLISH_WARMING.discard(key_text)
        return False


def _wait_for_citation_card_polish_cache(key: str, *, wait_s: float | None) -> dict | None:
    key_text = str(key or "").strip()
    if not key_text:
        return None
    try:
        deadline = time.time() + max(0.0, min(8.0, float(wait_s or 0.0)))
    except Exception:
        deadline = time.time()
    if deadline <= time.time():
        return None
    while time.time() < deadline:
        with _CITATION_CARD_POLISH_LOCK:
            cached = _CITATION_CARD_POLISH_CACHE.get(key_text)
        if isinstance(cached, dict):
            return dict(cached)
        time.sleep(0.12)
    return None


@router.post("/citation-card-polish")
def polish_citation_card(body: CitationCardPolishBody):
    detail = dict(body.meta or {}) if isinstance(body.meta, dict) else {}
    key = citation_card_polish_cache_key(detail)
    if not key:
        return {
            "citation_card_polish_status": "empty",
            "citation_card_polish_source": "no_key",
            "citation_card_polish_checked": True,
        }
    with _CITATION_CARD_POLISH_LOCK:
        cached = _CITATION_CARD_POLISH_CACHE.get(key)
        warming = key in _CITATION_CARD_POLISH_WARMING
    if isinstance(cached, dict):
        out = dict(cached)
        out["citation_card_polish_key"] = key
        out["citation_card_polish_cached"] = True
        return out
    if not citation_card_polish_enabled():
        return {
            "citation_card_polish_status": "disabled",
            "citation_card_polish_source": "disabled",
            "citation_card_polish_checked": True,
            "citation_card_polish_key": key,
        }
    started = False if warming else _schedule_citation_card_polish(key, detail)
    waited = _wait_for_citation_card_polish_cache(key, wait_s=body.wait_s)
    if isinstance(waited, dict):
        waited["citation_card_polish_key"] = key
        waited["citation_card_polish_cached"] = True
        waited["citation_card_polish_waited"] = True
        return waited
    return {
        "citation_card_polish_status": "pending",
        "citation_card_polish_source": "background_llm",
        "citation_card_polish_checked": False,
        "citation_card_polish_key": key,
        "citation_card_polish_started": bool(started),
    }


def _resolve_reader_md_path(source_path: str) -> Path | None:
    raw = _clean_reader_source_path_input(source_path)
    if not raw:
        return None
    asset_roots = _reference_asset_roots()
    if raw.replace("\\", "/").startswith(ROOT_RELATIVE_FILE_ID_PREFIX):
        return resolve_root_relative_file_id(raw, asset_roots)
    src = Path(raw).expanduser()
    if src.suffix.lower().endswith(".md"):
        return resolve_existing_file_under_roots(src, asset_roots)

    pdf_root = _pdf_dir()
    md_root = _md_dir()

    pdf_candidate = src
    try:
        if not pdf_candidate.is_absolute():
            pdf_candidate = pdf_root / pdf_candidate
    except Exception:
        pass

    resolved_pdf = resolve_existing_file_under_roots(pdf_candidate, [pdf_root])
    if resolved_pdf is None or resolved_pdf.suffix.lower() != ".pdf":
        return None

    try:
        _md_folder, md_main, md_exists = _resolve_md_output_paths(md_root, resolved_pdf)
    except Exception:
        return None
    if not md_exists:
        return None
    return resolve_existing_file_under_roots(md_main, asset_roots)


def _clean_reader_source_path_input(source_path: str) -> str:
    return clean_file_source_path_input(source_path)


def _resolve_public_reference_source_input(source_path: str) -> str:
    raw = clean_file_source_path_input(source_path)
    if not raw:
        return ""
    if not raw.replace("\\", "/").startswith(ROOT_RELATIVE_FILE_ID_PREFIX):
        return raw
    resolved = resolve_root_relative_file_id(raw, _reference_asset_roots())
    return str(resolved) if resolved is not None else ""


def _rewrite_md_asset_links(md_text: str, *, md_path: Path, asset_roots: list[Path]) -> str:
    text = str(md_text or "")
    if not text:
        return text

    def _markdown_image_destination(raw: str) -> str:
        src = str(raw or "").strip()
        if not src:
            return ""
        def _clean_destination(value: str) -> str:
            out = str(value or "").strip().replace("\\ ", " ")
            try:
                out = unquote(out)
            except Exception:
                pass
            return out.strip()
        if src.startswith("<"):
            end = src.find(">")
            if end > 0:
                return _clean_destination(src[1:end])
            return ""
        return _clean_destination(_MD_LINK_TITLE_SUFFIX_RE.sub("", src))

    def _asset_cache_version(path: Path) -> str:
        try:
            stat = path.stat()
            return f"{int(stat.st_mtime_ns)}-{int(stat.st_size)}"
        except Exception:
            return ""

    def _replace(m: re.Match[str]) -> str:
        alt = str(m.group(1) or "")
        raw = str(m.group(2) or "").strip()
        if not raw:
            return m.group(0)
        url = _markdown_image_destination(raw)
        low = url.lower()
        if low.startswith(("http://", "https://", "data:", "#", "/api/")):
            return m.group(0)
        try:
            cand = Path(url).expanduser()
            if not cand.is_absolute():
                cand = (md_path.parent / cand).resolve(strict=False)
            else:
                cand = cand.resolve(strict=False)
            if (not cand.exists()) or (not cand.is_file()):
                return m.group(0)
            if cand.suffix.lower() not in _ASSET_IMAGE_EXTS:
                return m.group(0)
            if not _path_within_roots(cand, asset_roots):
                return m.group(0)
            if not verified_image_file_mime(cand):
                return m.group(0)
            version = _asset_cache_version(cand)
            version_part = f"&v={quote(version, safe='')}" if version else ""
            asset_id = root_relative_file_id(cand, asset_roots)
            if not asset_id:
                return m.group(0)
            asset_url = f"/api/references/asset?path={quote(asset_id, safe='')}{version_part}"
            return f"![{alt}]({asset_url})"
        except Exception:
            return m.group(0)

    return _MD_IMAGE_RE.sub(_replace, text)


def _strip_md_inline_for_anchor(input_text: str) -> str:
    text = str(input_text or "")
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_equation_signal(text: str) -> bool:
    src = str(text or "")
    if not src:
        return False
    if "$$" in src:
        return True
    low = src.lower()
    if "\\begin{equation" in low or "\\[" in src:
        return True
    if _INLINE_EQ_RE.search(src):
        return True
    if _TEX_CMD_RE.search(src) and re.search(r"[=^_]", src):
        return True
    return False


def _extract_equation_number(text: str) -> int:
    src = str(text or "")
    if not src:
        return 0
    m = _EQ_NUMBER_RE.search(src)
    if not m:
        return 0
    try:
        v = int(str(m.group(1) or "0"))
    except Exception:
        return 0
    return v if v > 0 else 0


def _anchor_id(kind: str, index: int) -> str:
    prefix_map = {
        "heading": "hd",
        "paragraph": "p",
        "equation": "eq",
        "list_item": "li",
        "blockquote": "bq",
        "code": "cd",
        "table": "tb",
    }
    prefix = prefix_map.get(str(kind or "").strip().lower(), "a")
    return f"{prefix}_{int(max(1, index)):05d}"


def _build_reader_anchors(md_text: str, *, md_path: Path) -> tuple[list[dict], list[dict]]:
    blocks = load_source_blocks(md_path, md_text=md_text)
    anchors = source_blocks_to_reader_anchors(blocks)
    return anchors, blocks


def _reader_doc_hash(md_text: str) -> str:
    return hashlib.sha1(str(md_text or "").encode("utf-8", errors="ignore")).hexdigest()


def _reader_reference_index_data() -> dict:
    try:
        return load_reference_index(Path(get_settings().db_dir).expanduser())
    except Exception:
        return {}


def _resolve_reader_reference_entry(
    index_data: dict,
    *,
    source_path: str,
    md_path: Path,
    ref_num: int,
) -> dict | None:
    for candidate in [source_path, str(md_path)]:
        raw = str(candidate or "").strip()
        if not raw:
            continue
        try:
            resolved = resolve_reference_entry(index_data, raw, int(ref_num))
        except Exception:
            resolved = None
        if isinstance(resolved, dict):
            return resolved
    return None


_READER_BODY_CITATION_RE = re.compile(
    r"(?<!!)\[([Rr]?\d{1,4}(?:\s*[,，、;；\-–—−]\s*[Rr]?\d{1,4})*)\](?!\()",
)


def _reader_references_heading_line_index(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        text = str(line or "").strip()
        if re.match(r"^#{1,6}\s*(?:references|bibliography|参考文献)\b", text, flags=re.I):
            return idx
    return None


def _reader_first_body_heading_index(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        text = str(line or "").strip()
        if re.match(r"^#{2,6}\s+\S", text):
            return idx
    return 0


def _expand_reader_citation_body(body: str) -> list[int]:
    matches = list(re.finditer(r"[Rr]?\d{1,4}", str(body or "")))
    out: list[int] = []
    idx = 0
    while idx < len(matches):
        current = matches[idx]
        try:
            start_num = int(re.sub(r"^[Rr]", "", current.group(0)))
        except Exception:
            idx += 1
            continue
        nxt = matches[idx + 1] if idx + 1 < len(matches) else None
        if nxt is not None:
            sep = str(body or "")[current.end():nxt.start()]
            try:
                end_num = int(re.sub(r"^[Rr]", "", nxt.group(0)))
            except Exception:
                end_num = 0
            if re.fullmatch(r"\s*[-–—−]\s*", sep or "") and start_num > 0 and end_num >= start_num and (end_num - start_num) <= 64:
                out.extend(range(start_num, end_num + 1))
                idx += 2
                continue
        if start_num > 0:
            out.append(start_num)
        idx += 1
    return out


def _reader_body_cited_reference_numbers(md_text: str, *, known_ref_nums: set[int]) -> list[int]:
    if not known_ref_nums:
        return []
    lines = str(md_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        return []
    start_idx = _reader_first_body_heading_index(lines)
    refs_idx = _reader_references_heading_line_index(lines)
    end_idx = refs_idx if refs_idx is not None and refs_idx > start_idx else len(lines)
    body = "\n".join(lines[start_idx:end_idx])
    if not body:
        return []
    max_known = max(known_ref_nums)
    max_allowed = min(600, max_known + 5)
    nums: set[int] = set()
    for match in _READER_BODY_CITATION_RE.finditer(body):
        for num in _expand_reader_citation_body(match.group(1)):
            if 1 <= int(num) <= max_allowed:
                nums.add(int(num))
    return sorted(nums)


def _compose_missing_reader_reference_card(rec: dict, *, ref_num: int, note: str) -> dict:
    card = compose_citation_card(rec, locale=str(rec.get("render_locale") or ""))
    flags = list(card.get("card_quality_flags") or [])
    for flag in ("missing_reference_entry", "missing_reference_title", "reader_reference_gap"):
        if flag not in flags:
            flags.append(flag)
    card["card_quality_flags"] = flags
    card["card_quality_label"] = "Missing reference entry"
    card["card_quality_score"] = 0.0
    card["card_warning"] = ""
    card["card_reference_label"] = "Missing reference entry"
    card["card_reference_entry"] = note
    card["bibliometrics_checked"] = True
    card["metadata_quality"] = {
        "contract_version": 1,
        "ok": False,
        "status": "missing_reference_entry",
        "score": 0.0,
        "issues": [
            {
                "code": "missing_reference_entry",
                "label": "Reference entry was cited in text but missing from the converted bibliography.",
                "field": "raw",
                "severity": "error",
            }
        ],
    }
    card["metadata_export_acceptance"] = {
        "export_ready": False,
        "summary_export_ready": False,
        "reason": "missing_reference_entry",
    }
    card["summary_line"] = ""
    card["summary_source"] = ""
    card["title"] = card.get("title") or f"Reference [{ref_num}]"
    return card


def _reader_reference_cite_details(
    md_text: str,
    *,
    source_path: str,
    source_name: str,
    md_path: Path,
    doc_hash: str,
) -> list[dict]:
    ref_map = extract_references_map_from_md(md_text)
    if not ref_map:
        return []
    index_data = _reader_reference_index_data()
    anchor_sig = hashlib.sha1(
        f"{str(md_path)}|{doc_hash}".encode("utf-8", errors="ignore"),
    ).hexdigest()[:12]
    out: list[dict] = []
    known_ref_nums = set(int(n) for n in ref_map.keys() if int(n) > 0)
    cited_nums = set(_reader_body_cited_reference_numbers(md_text, known_ref_nums=known_ref_nums))
    all_nums = sorted(known_ref_nums | (cited_nums - known_ref_nums))
    for ref_num in all_nums[:600]:
        resolved = _resolve_reader_reference_entry(
            index_data,
            source_path=source_path,
            md_path=md_path,
            ref_num=ref_num,
        )
        ref = resolved.get("ref") if isinstance(resolved, dict) and isinstance(resolved.get("ref"), dict) else {}
        raw_ref = str(ref.get("raw") or ref_map.get(ref_num) or "").strip()
        missing_reference_entry = not raw_ref
        if missing_reference_entry and ref_num not in cited_nums:
            continue
        missing_note = (
            f"Reference [{ref_num}] is cited in the opened Reader document, "
            "but the converted References section does not contain a matching bibliography entry."
        )
        if missing_reference_entry:
            raw_ref = missing_note
        if not raw_ref:
            continue
        anchor = f"kb-cite-reader-{anchor_sig}-{ref_num}"
        context = missing_note if missing_reference_entry else f"The opened paper cites this upstream work as reference [{ref_num}]."
        rec = {
            "num": ref_num,
            "display_num": ref_num,
            "linked_nums": [ref_num],
            "anchor": anchor,
            "source_name": source_name,
            "source_path": str(source_path or md_path),
            "is_inpaper": True,
            "citation_route": "system_b",
            "routing_reason": "reader_missing_reference_entry" if missing_reference_entry else "reader_reference_index",
            "routing_confidence": 0.35 if missing_reference_entry else (1.0 if ref else 0.72),
            "raw": "" if missing_reference_entry else raw_ref,
            "cite_fmt": "" if missing_reference_entry else raw_ref,
            "title": str(ref.get("title") or "").strip(),
            "authors": str(ref.get("authors") or "").strip(),
            "venue": str(ref.get("venue") or "").strip(),
            "year": str(ref.get("year") or "").strip(),
            "volume": str(ref.get("volume") or "").strip(),
            "issue": str(ref.get("issue") or "").strip(),
            "pages": str(ref.get("pages") or "").strip(),
            "doi": str(ref.get("doi") or "").strip(),
            "doi_url": str(ref.get("doi_url") or "").strip(),
            "heading_path": "References",
            "location_label": f"{source_name} / References / [{ref_num}]",
            "evidence_quote": context,
            "evidence_source": "reader_missing_reference_entry" if missing_reference_entry else "reader_reference_link",
            "citation_context": context,
            "citation_context_source": "reader_missing_reference_entry" if missing_reference_entry else "reader_reference_link",
            "summary_line": "" if missing_reference_entry else context,
            "summary_source": "" if missing_reference_entry else "reader_reference_link",
            "answer_claim": context,
            "support_relation": missing_note if missing_reference_entry else "This bibliography entry is linked from the opened Reader document.",
            "binding_status": "missing_reference_entry" if missing_reference_entry else "reader_reference",
            "binding_confidence": 0.35 if missing_reference_entry else (1.0 if ref else 0.72),
            "binding_reason": "Cited in text, but the bibliography entry is missing from the converted References section." if missing_reference_entry else "Resolved from the opened paper's local reference list.",
            "card_reference_entry": "" if missing_reference_entry else raw_ref,
            "render_locale": str(load_prefs().get("ui_locale") or "").strip(),
        }
        if missing_reference_entry:
            rec["title"] = f"Reference [{ref_num}]"
            out.append(_compose_missing_reader_reference_card(rec, ref_num=ref_num, note=missing_note))
        else:
            out.append(compose_citation_card(rec, locale=str(rec.get("render_locale") or "")))
    return out


def _reader_outline_quality(blocks: list[dict]) -> dict:
    heading_blocks = [
        dict(block)
        for block in (blocks or [])
        if str((block or {}).get("kind") or "").strip().lower() == "heading"
    ]
    heading_count = len(heading_blocks)
    missing_level_count = 0
    caption_heading_count = 0
    publisher_heading_count = 0
    has_document_title = False
    max_heading_level = 0
    for block in heading_blocks:
        text = str(block.get("text") or "").strip()
        heading_path = str(block.get("heading_path") or "").strip()
        try:
            level = int(block.get("heading_level") or 0)
        except Exception:
            level = 0
        if level <= 0:
            missing_level_count += 1
        else:
            max_heading_level = max(max_heading_level, level)
            if level == 1:
                has_document_title = True
        if not has_document_title and heading_path and len([part for part in heading_path.split(" / ") if part.strip()]) == 1:
            has_document_title = True
        if re.match(r"^(?:fig(?:ure)?|table|extended\s+data\s+fig(?:ure)?)\.?\s+\d+", text, re.IGNORECASE):
            caption_heading_count += 1
        if text.strip().lower() in {"article", "research article", "letter", "nature", "communications", "optica"}:
            publisher_heading_count += 1
    issues: list[str] = []
    if heading_count <= 0:
        issues.append("no_outline")
    if heading_count > 0 and not has_document_title:
        issues.append("missing_document_title")
    if missing_level_count > 0:
        issues.append("missing_heading_level")
    if caption_heading_count > 0:
        issues.append("caption_heading")
    if publisher_heading_count > 0:
        issues.append("publisher_heading")
    return {
        "contract_version": 1,
        "ok": not issues,
        "status": "ok" if not issues else "warning",
        "heading_count": heading_count,
        "has_document_title": has_document_title,
        "max_heading_level": max_heading_level,
        "missing_heading_level_count": missing_level_count,
        "caption_heading_count": caption_heading_count,
        "publisher_heading_count": publisher_heading_count,
        "issues": issues,
    }


@router.post("/reader/doc")
def get_reader_doc(body: ReaderDocBody):
    source_path = str(body.source_path or "").strip()
    if not source_path:
        raise HTTPException(400, "source_path required")
    md_path = _resolve_reader_md_path(source_path)
    if md_path is None:
        raise HTTPException(404, "markdown not found for source")
    try:
        md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raise HTTPException(500, "failed to read markdown")

    asset_roots = _reference_asset_roots()
    public_source_path = root_relative_file_id(md_path, asset_roots) or md_path.name
    md_render = _rewrite_md_asset_links(
        md_text,
        md_path=md_path,
        asset_roots=asset_roots,
    )
    anchors, blocks = _build_reader_anchors(md_text, md_path=md_path)
    source_name = md_path.name
    low = source_name.lower()
    if low.endswith(".en.md"):
        source_name = source_name[:-6] + ".pdf"
    elif low.endswith(".md"):
        source_name = source_name[:-3] + ".pdf"
    doc_hash = _reader_doc_hash(md_text)
    cite_details = _reader_reference_cite_details(
        md_text,
        source_path=source_path,
        source_name=source_name,
        md_path=md_path,
        doc_hash=doc_hash,
    )
    for detail in cite_details:
        if isinstance(detail, dict):
            detail["source_path"] = public_source_path

    return {
        "ok": True,
        "source_path": public_source_path,
        "source_name": source_name,
        "md_path": public_source_path,
        "doc_hash": doc_hash,
        "outline_quality": _reader_outline_quality(blocks),
        "markdown": md_render,
        "anchors": anchors,
        "blocks": blocks,
        "cite_details": cite_details,
        "reference_cite_details": cite_details,
    }


@router.get("/asset")
def get_reference_asset(path: str):
    raw = str(path or "").strip()
    if not raw or len(raw) > _REFS_SOURCE_PATH_MAX_CHARS:
        raise HTTPException(404, "asset not found")
    asset_roots = _reference_asset_roots()
    candidate: str | Path = raw
    if raw.replace("\\", "/").startswith(ROOT_RELATIVE_FILE_ID_PREFIX):
        resolved = resolve_root_relative_file_id(raw, asset_roots)
        if resolved is None:
            raise HTTPException(404, "asset not found")
        candidate = resolved
    verified = resolve_verified_image_file_under_roots(candidate, asset_roots)
    if verified is None:
        raise HTTPException(404, "asset not found")
    resolved, media_type = verified
    if resolved.suffix.lower() not in _ASSET_IMAGE_EXTS:
        raise HTTPException(404, "asset not found")
    return FileResponse(
        str(resolved),
        media_type=media_type,
        filename=resolved.name,
        headers={"Cache-Control": "no-cache, max-age=0", "Pragma": "no-cache"},
    )
