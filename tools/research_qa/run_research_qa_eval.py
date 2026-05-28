from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import parse, request

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.reference_card_quality import (
    summarize_citation_detail_quality,
    summarize_citation_shelf_quality,
    summarize_ref_card_hit_quality,
)
from kb.citation_audit import summarize_system_b_citation_audit


DEFAULT_FIXTURE = Path("web/src/testing/researchQaData.json")
DEFAULT_OUT_DIR = Path("test_results/research_qa_eval")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class ResearchQaFixture:
    db_root: str
    docs: list[dict[str, Any]]
    cases: list[dict[str, Any]]
    forbidden_phrases: list[str]

    @property
    def docs_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(item.get("id") or ""): item for item in self.docs}


def _post_json(base_url: str, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")


def _get_json(base_url: str, path: str, timeout_s: float) -> Any:
    req = request.Request(f"{base_url}{path}", method="GET")
    with request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    return json.loads(raw or "{}")


def _stream_generation(base_url: str, session_id: str, timeout_s: float) -> dict[str, Any]:
    req = request.Request(f"{base_url}/api/generate/{parse.quote(session_id)}/stream", method="GET")
    final_payload: dict[str, Any] = {}
    with request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
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
                if payload.get("done"):
                    break
    return final_payload


def load_fixture(path: Path | str = DEFAULT_FIXTURE) -> ResearchQaFixture:
    fixture_path = Path(path)
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    docs = data.get("docs") if isinstance(data, dict) else []
    cases = data.get("cases") if isinstance(data, dict) else []
    forbidden = data.get("forbiddenPhrases") if isinstance(data, dict) else []
    return ResearchQaFixture(
        db_root=str(data.get("dbRoot") or "") if isinstance(data, dict) else "",
        docs=[item for item in list(docs or []) if isinstance(item, dict)],
        cases=[item for item in list(cases or []) if isinstance(item, dict)],
        forbidden_phrases=[str(item) for item in list(forbidden or []) if str(item or "").strip()],
    )


def source_path_for_doc(fixture: ResearchQaFixture, doc_id: str) -> str:
    doc = fixture.docs_by_id.get(str(doc_id or ""))
    if not isinstance(doc, dict):
        return ""
    directory = str(doc.get("dir") or "").strip()
    file_stem = str(doc.get("file") or directory).strip()
    if not directory or not file_stem:
        return ""
    return f"{fixture.db_root.rstrip('/')}/{directory}/{file_stem}.en.md"


def _norm(value: Any) -> str:
    text = str(value or "").replace("\\", "/").lower()
    return " ".join(text.split())


def _walk_strings(value: Any, *, max_depth: int = 8) -> list[str]:
    if max_depth <= 0:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        out: list[str] = []
        for key, child in value.items():
            out.extend(_walk_strings(key, max_depth=max_depth - 1))
            out.extend(_walk_strings(child, max_depth=max_depth - 1))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for child in value:
            out.extend(_walk_strings(child, max_depth=max_depth - 1))
        return out
    return []


def _payload_text(value: Any) -> str:
    return "\n".join(_walk_strings(value))


def _contains_term(haystack: Any, term: str) -> bool:
    hay = _norm(haystack)
    for needle in _term_aliases(term):
        needle_norm = _norm(needle)
        if needle_norm and needle_norm in hay:
            return True
    return False


def _term_aliases(term: str) -> list[str]:
    raw = str(term or "").strip()
    aliases = {
        "已有": ["已有", "现成", "成熟", "经典", "既有", "existing", "prior", "previous", "not new", "background"],
        "不是": ["不是", "不大", "关系不大", "并非", "不属于", "not", "not a", "different"],
        "foveated": ["foveated", "foveation", "注视", "焦点", "重要区域", "重点区域", "感兴趣区域", "空间变分辨率", "自适应采样"],
        "重要": ["重要", "重点", "重点区域", "关键", "变化剧烈", "边缘", "纹理", "salient", "important"],
        "wave": ["wave", "wave optics", "波动", "波前", "光场", "角度信息"],
        "重聚焦": ["重聚焦", "重新对焦", "重对焦", "refocus", "refocusing"],
        "resolution": ["resolution", "分辨率", "超分辨", "超分辨率", "空间分辨", "横向分辨率", "轴向分辨率"],
        "noise": ["noise", "噪声"],
        "ray tracing": ["ray tracing", "ray-tracing", "ray transfer matrix", "射线追踪"],
        "SNR": ["SNR", "signal-to-noise", "signal to noise", "信噪比"],
        "optical sectioning": ["optical sectioning", "sectioning", "光学层切", "光学切片"],
        "perovskite": ["perovskite", "钙钛矿"],
    }
    aliases.update(
        {
            "已有": [
                "已有",
                "现有",
                "既有",
                "前人",
                "前人的",
                "成熟",
                "经典",
                "已被",
                "existing",
                "prior",
                "previous",
                "not new",
                "background",
            ],
            "不是": ["不是", "并非", "不属于", "不应该", "not", "not a", "not the", "different"],
        }
    )
    extra_aliases = {
        "single-pixel imaging": [
            "single-pixel imaging",
            "single pixel imaging",
            "SPI",
            "单像素成像",
            "單像素成像",
        ],
        "deep learning": ["deep learning", "DL", "深度学习", "深度學習"],
        "snapshot compressive imaging": [
            "snapshot compressive imaging",
            "snapshot compressed image",
            "SCI",
            "压缩快照成像",
            "快照压缩成像",
        ],
        "CASSI": [
            "CASSI",
            "coded aperture snapshot spectral imaging",
            "compressive spectral imaging",
            "压缩光谱成像",
        ],
        "3DGS": ["3DGS", "3D Gaussian", "Gaussian splatting", "Gaussians Splatting"],
        "SPAD": ["SPAD", "single-photon avalanche diode", "单光子雪崩二极管"],
        "detector": ["detector", "photodetector", "探测器", "光探测器"],
        "generalization": [
            "generalization",
            "generalisation",
            "泛化",
            "未知",
            "通用框架",
            "通用性",
            "未知环境",
            "unseen",
            "out-of-distribution",
            "robustness",
        ],
        "PILN": ["PILN", "Part-based image-loop network", "image-loop network"],
        "structured detection": ["structured detection", "结构化探测", "结构探测"],
        "测量": [
            "测量",
            "采样",
            "测量次数",
            "测量预算",
            "measurement",
            "measurements",
            "sampling",
            "acquisition",
            "DMD",
            "调制",
        ],
        "wave": ["\u6ce2\u52a8", "\u6ce2\u52a8\u5149\u5b66", "\u884d\u5c04", "\u4f20\u64ad"],
        "ray tracing": [
            "\u5149\u7ebf\u8ffd\u8ff9",
            "\u5149\u7ebf\u8ffd\u8e2a",
            "\u5c04\u7ebf\u8ffd\u8ff9",
            "\u5c04\u7ebf\u8ffd\u8e2a",
            "\u5149\u7ebf\u4f20\u9012\u77e9\u9635",
        ],
        "refocus": [
            "\u91cd\u805a\u7126",
            "\u91cd\u65b0\u5bf9\u7126",
            "\u91cd\u5bf9\u7126",
            "\u79bb\u7126",
            "\u666f\u6df1",
            "\u5927\u666f\u6df1",
            "refocusing",
            "light-field",
            "light field",
            "depth of field",
        ],
    }
    out = [raw]
    out.extend(aliases.get(raw, []))
    out.extend(extra_aliases.get(raw, []))
    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        key = str(item or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(str(item))
    return deduped


def _doc_match_candidates(fixture: ResearchQaFixture, doc_id: str) -> list[str]:
    doc = fixture.docs_by_id.get(str(doc_id or "")) or {}
    return [
        str(doc.get("id") or ""),
        str(doc.get("title") or ""),
        str(doc.get("shortLabel") or ""),
        str(doc.get("dir") or ""),
        source_path_for_doc(fixture, doc_id),
    ]


def _doc_matches_payload(fixture: ResearchQaFixture, doc_id: str, payload: Any) -> bool:
    text = _norm(_payload_text(payload))
    return any(_contains_term(text, candidate) for candidate in _doc_match_candidates(fixture, doc_id))


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _messages_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("messages", "items", "data"):
            child = payload.get(key)
            if isinstance(child, list):
                return [item for item in child if isinstance(item, dict)]
    return []


def _latest_assistant_message(messages_payload: Any) -> dict[str, Any]:
    messages = _messages_list(messages_payload)
    for item in reversed(messages):
        if str(item.get("role") or "").lower() == "assistant":
            return item
    return {}


def _assistant_message_by_id(messages_payload: Any, msg_id: Any) -> dict[str, Any]:
    try:
        target_id = int(msg_id or 0)
    except Exception:
        target_id = 0
    if target_id <= 0:
        return _latest_assistant_message(messages_payload)
    for item in reversed(_messages_list(messages_payload)):
        if str(item.get("role") or "").lower() != "assistant":
            continue
        try:
            current_id = int(item.get("id") or 0)
        except Exception:
            current_id = 0
        if current_id == target_id:
            return item
    return _latest_assistant_message(messages_payload)


def _answer_text(result: dict[str, Any]) -> str:
    message = result.get("assistant_message")
    if isinstance(message, dict):
        meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
        packet = meta.get("paper_guide_contracts") if isinstance(meta, dict) else {}
        render_packet = packet.get("render_packet") if isinstance(packet, dict) and isinstance(packet.get("render_packet"), dict) else {}
        for key in ("rendered_body", "rendered_content", "answer_markdown", "copy_markdown", "copy_text"):
            text = str(render_packet.get(key) or "").strip()
            if text:
                return text
        for key in ("rendered_body", "content", "copy_markdown", "copy_text"):
            text = str(message.get(key) or "").strip()
            if text:
                return text
    final_payload = result.get("final_payload")
    if isinstance(final_payload, dict):
        return str(final_payload.get("answer") or "").strip()
    return str(result.get("answer") or "").strip()


def _inline_inpaper_citation_details(answer: str) -> list[dict[str, Any]]:
    text = str(answer or "")
    if "[[CITE:" not in text:
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in re.finditer(r"\[\[CITE:([^:\]\s]+):(\d{1,4})\]\]", text):
        sid = str(match.group(1) or "").strip()
        ref_num = str(match.group(2) or "").strip()
        start = max(
            text.rfind("\n", 0, match.start()),
            text.rfind("。", 0, match.start()),
            text.rfind(".", 0, match.start()),
            text.rfind("；", 0, match.start()),
            text.rfind(";", 0, match.start()),
        )
        end_candidates = [
            idx for idx in (
                text.find("\n", match.end()),
                text.find("。", match.end()),
                text.find(".", match.end()),
                text.find("；", match.end()),
                text.find(";", match.end()),
            )
            if idx >= 0
        ]
        end = min(end_candidates) if end_candidates else min(len(text), match.end() + 180)
        context = text[start + 1 : end + 1].strip()
        key = (sid, ref_num, context.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "is_inpaper": True,
                "sid": sid,
                "ref_num": ref_num,
                "raw": context,
                "context": context,
                "source": "inline_marker",
            }
        )
    return out


def _citation_details(result: dict[str, Any]) -> list[dict[str, Any]]:
    message = result.get("assistant_message")
    details: list[Any] = []
    if isinstance(message, dict):
        details.extend(_as_list(message.get("cite_details")))
        meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
        packet = meta.get("paper_guide_contracts") if isinstance(meta, dict) else {}
        if isinstance(packet, dict):
            render_packet = packet.get("render_packet") if isinstance(packet.get("render_packet"), dict) else {}
            details.extend(_as_list(render_packet.get("cite_details")))
    final_payload = result.get("final_payload")
    if isinstance(final_payload, dict):
        details.extend(_as_list(final_payload.get("cite_details")))
    details.extend(_inline_inpaper_citation_details(_answer_text(result)))
    return [item for item in details if isinstance(item, dict)]


def _extract_ref_packs(refs_payload: Any, user_msg_id: int | str | None = None) -> list[dict[str, Any]]:
    if not isinstance(refs_payload, dict):
        return []
    if user_msg_id is not None:
        for key in (user_msg_id, str(user_msg_id)):
            pack = refs_payload.get(key)
            if isinstance(pack, dict):
                return [pack]
    if "hits" in refs_payload and isinstance(refs_payload.get("hits"), list):
        return [refs_payload]
    return [item for item in refs_payload.values() if isinstance(item, dict)]


def _extract_ref_hits(refs_payload: Any, user_msg_id: int | str | None = None) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for pack in _extract_ref_packs(refs_payload, user_msg_id=user_msg_id):
        hits.extend([item for item in _as_list(pack.get("hits")) if isinstance(item, dict)])
    return hits


def _extract_primary_evidence_payloads(refs_payload: Any, user_msg_id: int | str | None = None) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for pack in _extract_ref_packs(refs_payload, user_msg_id=user_msg_id):
        if isinstance(pack.get("primary_evidence"), dict):
            payloads.append(pack["primary_evidence"])
        if isinstance(pack.get("primary_evidence_alignment"), dict):
            payloads.append(pack["primary_evidence_alignment"])
        for hit in _as_list(pack.get("hits")):
            if not isinstance(hit, dict):
                continue
            ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
            if isinstance(ui_meta.get("primary_evidence"), dict):
                payloads.append(ui_meta["primary_evidence"])
            reader_open = ui_meta.get("reader_open") if isinstance(ui_meta.get("reader_open"), dict) else {}
            if isinstance(reader_open.get("primaryEvidence"), dict):
                payloads.append(reader_open["primaryEvidence"])
            if isinstance(reader_open.get("locateTarget"), dict):
                payloads.append(reader_open["locateTarget"])
    return payloads


def _expected_int(expected: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return max(0, int(expected.get(key, default) or default))
    except Exception:
        return max(0, int(default or 0))


def _unique_doc_ids_in_payload(fixture: ResearchQaFixture, payload: Any) -> list[str]:
    return sorted(doc_id for doc_id in fixture.docs_by_id if _doc_matches_payload(fixture, doc_id, payload))


def _ref_card_quality_summary(
    refs_payload: Any,
    forbidden_phrases: list[str],
    user_msg_id: int | str | None = None,
) -> dict[str, Any]:
    hits = _extract_ref_hits(refs_payload, user_msg_id=user_msg_id)
    return summarize_ref_card_hit_quality(hits[:5], forbidden_phrases=forbidden_phrases)


def _ref_card_quality_failures(ref_card_quality: dict[str, Any]) -> list[str]:
    summary = ref_card_quality if isinstance(ref_card_quality, dict) else {}
    return [
        f"ref_card_{item.get('index')}_{item.get('name')}"
        + (f":{item.get('detail')}" if item.get("detail") else "")
        for item in _as_list(summary.get("failures"))
        if isinstance(item, dict)
    ]


def _ref_pack_state_failures(refs_payload: Any, user_msg_id: int | str | None = None) -> list[str]:
    failures: list[str] = []
    for pack_idx, pack in enumerate(_extract_ref_packs(refs_payload, user_msg_id=user_msg_id), start=1):
        display_state = str(pack.get("display_state") or "").strip().lower()
        if display_state and display_state != "ready":
            failures.append(f"pack_{pack_idx}_display_state:{display_state}")
        if bool(pack.get("pending")):
            failures.append(f"pack_{pack_idx}_pending:true")
        for hit_idx, hit in enumerate(_as_list(pack.get("hits")), start=1):
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            ref_state = str(meta.get("ref_pack_state") or "").strip().lower()
            if ref_state and ref_state != "ready":
                failures.append(f"pack_{pack_idx}_hit_{hit_idx}_ref_pack_state:{ref_state}")
    return failures


def _ref_polish_failures(
    refs_payload: Any,
    *,
    user_msg_id: int | str | None = None,
    require_status: bool = False,
    allowed_statuses: list[str] | None = None,
) -> list[str]:
    failures: list[str] = []
    allowed = {str(item or "").strip().lower() for item in list(allowed_statuses or []) if str(item or "").strip()}
    for idx, hit in enumerate(_extract_ref_hits(refs_payload, user_msg_id=user_msg_id), start=1):
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        status = str(ui_meta.get("polish_status") or "").strip().lower()
        if require_status and not status:
            failures.append(f"ref_card_{idx}_missing_polish_status")
            continue
        if status and allowed and status not in allowed:
            failures.append(f"ref_card_{idx}_polish_status:{status}")
    return failures


def _should_check_citation_card_quality(expected: dict[str, Any]) -> bool:
    return bool(
        expected.get("requireCitationCardQuality")
        or expected.get("requireSystemB")
        or _expected_int(expected, "minSystemBCount") > 0
        or bool(expected.get("requireRefsReady"))
        or bool(expected.get("requirePolishStatus"))
        or _expected_int(expected, "minSystemAQualityCount") > 0
        or _expected_int(expected, "minSystemBQualityCount") > 0
    )


def _citation_quality_failures(
    citation_details: list[dict[str, Any]],
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    summary = summarize_citation_detail_quality(citation_details)
    failures: list[str] = []
    if not bool(summary.get("ok")):
        failures.extend(
            f"citation_{item.get('index')}_{item.get('name')}"
            for item in _as_list(summary.get("failures"))
            if isinstance(item, dict)
        )
    min_system_a = _expected_int(expected, "minSystemAQualityCount")
    min_system_b = _expected_int(
        expected,
        "minSystemBQualityCount",
        _expected_int(expected, "minSystemBCount", 1 if bool(expected.get("requireSystemB")) else 0),
    )
    ok_route_counts = summary.get("ok_route_counts") if isinstance(summary.get("ok_route_counts"), dict) else {}
    actual_system_a = int(ok_route_counts.get("system_a") or 0)
    actual_system_b = int(ok_route_counts.get("system_b") or 0)
    if min_system_a and actual_system_a < min_system_a:
        failures.append(f"system_a_quality_count:{actual_system_a}<{min_system_a}")
    if min_system_b and actual_system_b < min_system_b:
        failures.append(f"system_b_quality_count:{actual_system_b}<{min_system_b}")
    return summary, failures


def _shelf_quality_failures(
    citation_details: list[dict[str, Any]],
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    summary = summarize_citation_shelf_quality(citation_details)
    failures: list[str] = []
    if not bool(summary.get("ok")):
        failures.extend(
            f"shelf_{item.get('index')}_{item.get('name')}"
            for item in _as_list(summary.get("failures"))
            if isinstance(item, dict)
        )
    min_ok = _expected_int(expected, "minCitationShelfQualityCount", _expected_int(expected, "minCitationCount", 1))
    ok_count = int(summary.get("ok_count") or 0)
    if min_ok and ok_count < min_ok:
        failures.append(f"shelf_quality_count:{ok_count}<{min_ok}")
    min_metadata_ready = _expected_int(expected, "minCitationShelfMetadataReadyCount")
    metadata_ready = int(summary.get("metadata_ready_count") or 0)
    if min_metadata_ready and metadata_ready < min_metadata_ready:
        failures.append(f"shelf_metadata_ready_count:{metadata_ready}<{min_metadata_ready}")
    min_export_ready = _expected_int(expected, "minCitationShelfExportReadyCount")
    export_ready = int(summary.get("export_ready_count") or 0)
    if min_export_ready and export_ready < min_export_ready:
        failures.append(f"shelf_export_ready_count:{export_ready}<{min_export_ready}")
    min_doi = _expected_int(expected, "minCitationShelfDoiCount")
    doi_count = int(summary.get("doi_count") or 0)
    if min_doi and doi_count < min_doi:
        failures.append(f"shelf_doi_count:{doi_count}<{min_doi}")
    min_source_click = _expected_int(expected, "minCitationShelfSourceClickCount")
    source_clickable = int(summary.get("source_clickable_count") or 0)
    if min_source_click and source_clickable < min_source_click:
        failures.append(f"shelf_source_clickable_count:{source_clickable}<{min_source_click}")
    max_review = _expected_optional_int(expected, "maxCitationShelfMetadataReviewCount")
    review_count = int(summary.get("review_count") or 0)
    if max_review is not None and review_count > max_review:
        failures.append(f"shelf_metadata_review_count:{review_count}>{max_review}")
    return summary, failures


def _system_b_audit_expected(expected: dict[str, Any]) -> bool:
    return bool(
        expected.get("requireSystemBTraceComplete")
        or expected.get("forbidSystemBAnswerContextOnly")
        or expected.get("forbidSystemBReferenceIndexFallback")
        or "maxSystemBNeedsReviewCount" in expected
        or "maxSystemBAnswerContextOnlyCount" in expected
        or "maxSystemBReferenceIndexFallbackCount" in expected
        or "minSystemBCompleteRate" in expected
    )


def _expected_float(expected: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(expected.get(key) if key in expected else default)
    except Exception:
        value = float(default)
    if value != value:
        return float(default)
    return value


def _expected_optional_int(expected: dict[str, Any], key: str) -> int | None:
    if key not in expected:
        return None
    try:
        return max(0, int(expected.get(key) or 0))
    except Exception:
        return None


def _system_b_audit_failures(audit: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not isinstance(audit, dict):
        return ["system_b_audit_missing"]

    total = int(audit.get("system_b_total") or 0)
    needs_review = int(audit.get("needs_review_count") or 0)
    answer_context_only = int(audit.get("answer_context_only_count") or 0)
    reference_index_fallback = int(audit.get("reference_index_fallback_count") or 0)
    complete_rate = float(audit.get("complete_rate") or 0.0)

    if bool(expected.get("requireSystemBTraceComplete")):
        if total <= 0:
            failures.append("system_b_audit_no_system_b")
        if needs_review > 0:
            failures.append(f"system_b_trace_needs_review:{needs_review}")
    if bool(expected.get("forbidSystemBAnswerContextOnly")) and answer_context_only > 0:
        failures.append(f"system_b_answer_context_only:{answer_context_only}")
    if bool(expected.get("forbidSystemBReferenceIndexFallback")) and reference_index_fallback > 0:
        failures.append(f"system_b_reference_index_fallback:{reference_index_fallback}")

    max_needs_review = _expected_optional_int(expected, "maxSystemBNeedsReviewCount")
    if max_needs_review is not None and needs_review > max_needs_review:
        failures.append(f"system_b_needs_review:{needs_review}>{max_needs_review}")
    max_answer_context_only = _expected_optional_int(expected, "maxSystemBAnswerContextOnlyCount")
    if max_answer_context_only is not None and answer_context_only > max_answer_context_only:
        failures.append(f"system_b_answer_context_only:{answer_context_only}>{max_answer_context_only}")
    max_reference_index_fallback = _expected_optional_int(expected, "maxSystemBReferenceIndexFallbackCount")
    if max_reference_index_fallback is not None and reference_index_fallback > max_reference_index_fallback:
        failures.append(f"system_b_reference_index_fallback:{reference_index_fallback}>{max_reference_index_fallback}")

    min_complete_rate = _expected_float(expected, "minSystemBCompleteRate", 0.0)
    if min_complete_rate > 0 and complete_rate < min_complete_rate:
        failures.append(f"system_b_complete_rate:{complete_rate}<{min_complete_rate}")
    return failures


def validate_case(
    case: dict[str, Any],
    fixture: ResearchQaFixture,
    result: dict[str, Any],
) -> dict[str, Any]:
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    answer = _answer_text(result)
    refs_payload = result.get("refs_payload")
    user_msg_id = result.get("user_msg_id")
    citation_details = _citation_details(result)
    ref_hits = _extract_ref_hits(refs_payload, user_msg_id=user_msg_id)
    ref_packs = _extract_ref_packs(refs_payload, user_msg_id=user_msg_id)
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: Any = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    status = str(result.get("status") or "").strip().lower()
    add_check("generation_done", bool(result.get("done")) and status not in {"error", "canceled"}, status)

    forbidden_hits = [phrase for phrase in fixture.forbidden_phrases if _contains_term(answer, phrase)]
    add_check("answer_no_template_phrase", not forbidden_hits, forbidden_hits)

    direct_starts = ("the paper cites", "this is stated in", "该文在", "文献中提到")
    add_check("answer_directly_addresses_question", not _norm(answer).startswith(direct_starts), answer[:120])

    required_terms = [str(item) for item in _as_list(expected.get("requiredAnswerTerms")) if str(item or "").strip()]
    missing_answer_terms = [term for term in required_terms if not _contains_term(answer, term)]
    add_check("answer_contains_required_terms", not missing_answer_terms, missing_answer_terms)

    required_ref_doc_ids = [str(item) for item in _as_list(expected.get("requiredRefDocIds")) if str(item or "").strip()]
    missing_ref_docs = [doc_id for doc_id in required_ref_doc_ids if not _doc_matches_payload(fixture, doc_id, refs_payload)]
    add_check("refs_include_required_docs", not missing_ref_docs, missing_ref_docs)

    min_ref_hits = _expected_int(expected, "minRefHits")
    if min_ref_hits:
        add_check("refs_min_hit_count", len(ref_hits) >= min_ref_hits, {"actual": len(ref_hits), "min": min_ref_hits})

    min_ref_doc_count = _expected_int(expected, "minRefDocCount")
    if min_ref_doc_count:
        ref_doc_ids = _unique_doc_ids_in_payload(fixture, refs_payload)
        add_check(
            "refs_min_doc_count",
            len(ref_doc_ids) >= min_ref_doc_count,
            {"actual": len(ref_doc_ids), "min": min_ref_doc_count, "doc_ids": ref_doc_ids},
        )

    if bool(expected.get("requireRefsReady")):
        ready_failures = _ref_pack_state_failures(refs_payload, user_msg_id=user_msg_id)
        add_check("refs_ready", not ready_failures and bool(ref_packs), ready_failures)

    require_polish_status = bool(expected.get("requirePolishStatus"))
    allowed_polish_statuses = [
        str(item).strip().lower()
        for item in _as_list(expected.get("allowedRefPolishStatuses"))
        if str(item or "").strip()
    ]
    if require_polish_status or allowed_polish_statuses:
        polish_failures = _ref_polish_failures(
            refs_payload,
            user_msg_id=user_msg_id,
            require_status=require_polish_status,
            allowed_statuses=allowed_polish_statuses,
        )
        add_check("refs_card_polish_status", not polish_failures and bool(ref_hits), polish_failures)

    required_citation_doc_ids = [
        str(item) for item in _as_list(expected.get("requiredCitationDocIds")) if str(item or "").strip()
    ]
    missing_citation_docs = [
        doc_id for doc_id in required_citation_doc_ids if not _doc_matches_payload(fixture, doc_id, citation_details)
    ]
    add_check("citations_include_required_docs", not missing_citation_docs, missing_citation_docs)

    min_citation_count = _expected_int(expected, "minCitationCount")
    if min_citation_count:
        add_check(
            "citations_min_count",
            len(citation_details) >= min_citation_count,
            {"actual": len(citation_details), "min": min_citation_count},
        )

    min_citation_doc_count = _expected_int(expected, "minCitationDocCount")
    if min_citation_doc_count:
        citation_doc_ids = _unique_doc_ids_in_payload(fixture, citation_details)
        add_check(
            "citations_min_doc_count",
            len(citation_doc_ids) >= min_citation_doc_count,
            {"actual": len(citation_doc_ids), "min": min_citation_doc_count, "doc_ids": citation_doc_ids},
        )

    inpaper_details = [item for item in citation_details if bool(item.get("is_inpaper"))]
    min_system_b_count = _expected_int(expected, "minSystemBCount", 1 if bool(expected.get("requireSystemB")) else 0)
    if min_system_b_count:
        add_check(
            "system_b_min_count",
            len(inpaper_details) >= min_system_b_count,
            {"actual": len(inpaper_details), "min": min_system_b_count},
        )
    if bool(expected.get("requireSystemB")):
        required_system_b_terms = [
            str(item) for item in _as_list(expected.get("requiredSystemBTerms")) if str(item or "").strip()
        ]
        missing_system_b_terms = [term for term in required_system_b_terms if not _contains_term(inpaper_details, term)]
        add_check("system_b_present", bool(inpaper_details), len(inpaper_details))
        add_check("system_b_contains_required_terms", not missing_system_b_terms, missing_system_b_terms)
    elif min_system_b_count:
        required_system_b_terms = [
            str(item) for item in _as_list(expected.get("requiredSystemBTerms")) if str(item or "").strip()
        ]
        if required_system_b_terms:
            answer_for_system_b = _answer_text(result)
            missing_system_b_terms = [
                term
                for term in required_system_b_terms
                if not (_contains_term(inpaper_details, term) or _contains_term(answer_for_system_b, term))
            ]
            add_check("system_b_contains_required_terms", not missing_system_b_terms, missing_system_b_terms)

    required_system_b_doc_ids = [
        str(item) for item in _as_list(expected.get("requiredSystemBDocIds")) if str(item or "").strip()
    ]
    if required_system_b_doc_ids:
        missing_system_b_docs = [
            doc_id for doc_id in required_system_b_doc_ids if not _doc_matches_payload(fixture, doc_id, inpaper_details)
        ]
        add_check("system_b_includes_required_docs", not missing_system_b_docs, missing_system_b_docs)

    citation_quality: dict[str, Any] = {}
    shelf_quality: dict[str, Any] = {}
    system_b_audit = summarize_system_b_citation_audit(citation_details)
    if _should_check_citation_card_quality(expected):
        citation_quality, citation_quality_failures = _citation_quality_failures(citation_details, expected)
        if isinstance(citation_quality.get("system_b_audit"), dict):
            system_b_audit = dict(citation_quality.get("system_b_audit") or {})
        add_check(
            "citation_card_quality",
            not citation_quality_failures and bool(citation_details),
            citation_quality_failures or citation_quality,
        )
    if bool(expected.get("requireCitationShelfQuality")):
        shelf_quality, shelf_quality_failures = _shelf_quality_failures(citation_details, expected)
        add_check(
            "citation_shelf_quality",
            not shelf_quality_failures and bool(citation_details),
            shelf_quality_failures or shelf_quality,
        )
    if _system_b_audit_expected(expected):
        system_b_audit_failures = _system_b_audit_failures(system_b_audit, expected)
        add_check(
            "system_b_audit",
            not system_b_audit_failures,
            system_b_audit_failures or system_b_audit,
        )

    ref_card_quality = _ref_card_quality_summary(
        refs_payload,
        fixture.forbidden_phrases,
        user_msg_id=user_msg_id,
    )
    card_failures = _ref_card_quality_failures(ref_card_quality)
    add_check("refs_card_copy_quality", not card_failures, card_failures)

    required_primary_terms = [
        str(item) for item in _as_list(expected.get("requiredPrimaryEvidenceTerms")) if str(item or "").strip()
    ]
    if required_primary_terms:
        primary_payloads = _extract_primary_evidence_payloads(refs_payload, user_msg_id=user_msg_id)
        missing_primary_terms = [term for term in required_primary_terms if not _contains_term(primary_payloads, term)]
        add_check("primary_evidence_contains_required_terms", not missing_primary_terms, missing_primary_terms)

    forbidden_primary_terms = [
        str(item) for item in _as_list(expected.get("forbiddenPrimaryEvidenceTerms")) if str(item or "").strip()
    ]
    if forbidden_primary_terms:
        primary_payloads = _extract_primary_evidence_payloads(refs_payload, user_msg_id=user_msg_id)
        present_forbidden_primary_terms = [term for term in forbidden_primary_terms if _contains_term(primary_payloads, term)]
        add_check("primary_evidence_avoids_forbidden_terms", not present_forbidden_primary_terms, present_forbidden_primary_terms)

    failures = [item for item in checks if not item.get("ok")]
    return {
        "ok": not failures,
        "checks": checks,
        "failures": failures,
        "answer_preview": answer[:360],
        "citation_count": len(citation_details),
        "system_b_count": len(inpaper_details),
        "ref_hit_count": len(ref_hits),
        "ref_doc_ids": _unique_doc_ids_in_payload(fixture, refs_payload),
        "citation_doc_ids": _unique_doc_ids_in_payload(fixture, citation_details),
        "citation_quality": citation_quality,
        "citation_shelf_quality": shelf_quality,
        "ref_card_quality": ref_card_quality,
        "system_b_audit": system_b_audit,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _build_report(rows: list[dict[str, Any]], *, fixture_path: Path, base_url: str, output_dir: Path) -> str:
    passed = [row for row in rows if bool((row.get("quality") or {}).get("ok"))]
    failed = [row for row in rows if not bool((row.get("quality") or {}).get("ok"))]
    lines = [
        "# Research QA Eval Report",
        "",
        f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Base URL: `{base_url}`",
        f"- Fixture: `{fixture_path}`",
        f"- Output: `{output_dir}`",
        f"- Cases: {len(rows)}",
        f"- Passed: {len(passed)}",
        f"- Failed: {len(failed)}",
        "",
        "## Failures",
        "",
    ]
    if not failed:
        lines.append("- None")
    for row in failed:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        failure_names = [str(item.get("name") or "") for item in _as_list(quality.get("failures")) if isinstance(item, dict)]
        lines.append(f"- `{row.get('id')}`: {', '.join(failure_names) or 'unknown'}")
    card_rows: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        ref_card_quality = quality.get("ref_card_quality") if isinstance(quality.get("ref_card_quality"), dict) else {}
        if ref_card_quality:
            card_rows.append((str(row.get("id") or ""), ref_card_quality))
    lines.extend(["", "## Ref Card Quality", ""])
    if not card_rows:
        lines.append("- None")
    for case_id, card_quality in card_rows:
        failures = [item for item in _as_list(card_quality.get("failures")) if isinstance(item, dict)]
        warnings = [item for item in _as_list(card_quality.get("warnings")) if isinstance(item, dict)]
        lines.append(
            "- "
            f"`{case_id}`: ok={bool(card_quality.get('ok'))}, "
            f"cards={int(card_quality.get('count') or 0)}, "
            f"ok_cards={int(card_quality.get('ok_count') or 0)}, "
            f"failures={len(failures)}, warnings={len(warnings)}, "
            f"min_score={float(card_quality.get('min_score') or 0.0):.3f}"
        )
        for failure in failures[:5]:
            lines.append(
                "  - "
                f"card {int(failure.get('index') or 0)}: "
                f"{failure.get('name')}"
                + (f" ({failure.get('field')})" if failure.get("field") else "")
                + (f" - {failure.get('detail')}" if failure.get("detail") else "")
            )
    shelf_rows: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        shelf_quality = quality.get("citation_shelf_quality") if isinstance(quality.get("citation_shelf_quality"), dict) else {}
        if shelf_quality:
            shelf_rows.append((str(row.get("id") or ""), shelf_quality))
    lines.extend(["", "## Citation Shelf Quality", ""])
    if not shelf_rows:
        lines.append("- None")
    for case_id, shelf_quality in shelf_rows:
        failures = [item for item in _as_list(shelf_quality.get("failures")) if isinstance(item, dict)]
        warnings = [item for item in _as_list(shelf_quality.get("warnings")) if isinstance(item, dict)]
        lines.append(
            "- "
            f"`{case_id}`: ok={bool(shelf_quality.get('ok'))}, "
            f"items={int(shelf_quality.get('count') or 0)}, "
            f"ok_items={int(shelf_quality.get('ok_count') or 0)}, "
            f"metadata_ready={int(shelf_quality.get('metadata_ready_count') or 0)}, "
            f"doi={int(shelf_quality.get('doi_count') or 0)}, "
            f"source_clickable={int(shelf_quality.get('source_clickable_count') or 0)}, "
            f"failures={len(failures)}, warnings={len(warnings)}, "
            f"min_score={float(shelf_quality.get('min_score') or 0.0):.3f}"
        )
        for failure in failures[:5]:
            lines.append(
                "  - "
                f"item {int(failure.get('index') or 0)}: "
                f"{failure.get('name')}"
                + (f" ({failure.get('field')})" if failure.get("field") else "")
                + (f" - {failure.get('detail')}" if failure.get("detail") else "")
            )
    audit_rows: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        audit = quality.get("system_b_audit") if isinstance(quality.get("system_b_audit"), dict) else {}
        if int(audit.get("system_b_total") or 0) > 0:
            audit_rows.append((str(row.get("id") or ""), audit))
    lines.extend(["", "## System B Audit", ""])
    if not audit_rows:
        lines.append("- None")
    for case_id, audit in audit_rows:
        lines.append(
            "- "
            f"`{case_id}`: total={int(audit.get('system_b_total') or 0)}, "
            f"complete={int(audit.get('trace_complete_count') or 0)}, "
            f"review={int(audit.get('needs_review_count') or 0)}, "
            f"answer_context_only={int(audit.get('answer_context_only_count') or 0)}, "
            f"fallback={int(audit.get('reference_index_fallback_count') or 0)}"
        )
    return "\n".join(lines) + "\n"


def run_case(
    *,
    base_url: str,
    fixture: ResearchQaFixture,
    case: dict[str, Any],
    timeout_s: float,
    top_k: int,
    max_tokens: int,
) -> dict[str, Any]:
    case_id = str(case.get("id") or "").strip()
    question = str(case.get("question") or "").strip()
    preferred_sources = [source_path_for_doc(fixture, str(doc_id)) for doc_id in _as_list(case.get("docIds"))]
    preferred_sources = [item for item in preferred_sources if item]
    started = time.perf_counter()
    conv = _post_json(
        base_url,
        "/api/conversations",
        {"title": f"research-qa-{case_id}", "project_id": None},
        timeout_s=timeout_s,
    )
    conv_id = str(conv.get("id") or "").strip()
    if not conv_id:
        raise RuntimeError("conversation creation returned no id")

    gen = _post_json(
        base_url,
        "/api/generate",
        {
            "conv_id": conv_id,
            "prompt": question,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "preferred_sources": preferred_sources,
        },
        timeout_s=timeout_s,
    )
    session_id = str(gen.get("session_id") or "").strip()
    if not session_id:
        raise RuntimeError("generation returned no session_id")
    final_payload = _stream_generation(base_url, session_id, timeout_s=timeout_s)
    _get_json(base_url, f"/api/conversations/{parse.quote(conv_id)}/messages?render_packet_only=1", timeout_s)
    refs_payload = _get_json(base_url, f"/api/references/conversation/{parse.quote(conv_id)}", timeout_s)
    # References may refine primary evidence and backfill message render packets;
    # validate the converged message after the references endpoint has run.
    messages = _get_json(base_url, f"/api/conversations/{parse.quote(conv_id)}/messages?render_packet_only=1", timeout_s)
    assistant_message = _assistant_message_by_id(messages, gen.get("assistant_msg_id"))
    user_msg_id = gen.get("user_msg_id")
    row = {
        "id": case_id,
        "conv_id": conv_id,
        "user_msg_id": user_msg_id,
        "assistant_msg_id": gen.get("assistant_msg_id"),
        "status": str(final_payload.get("status") or ""),
        "done": bool(final_payload.get("done")),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "question": question,
        "expected": case.get("expected") if isinstance(case.get("expected"), dict) else {},
        "final_payload": final_payload,
        "assistant_message": assistant_message,
        "refs_payload": refs_payload,
    }
    row["quality"] = validate_case(case, fixture, row)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real research-user QA checks against the local API.")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE), help="Shared research QA fixture JSON.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory root.")
    parser.add_argument("--timeout-s", type=float, default=180.0, help="HTTP timeout seconds.")
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit.")
    parser.add_argument("--case-id", action="append", default=[], help="Run one or more case ids.")
    parser.add_argument("--top-k", type=int, default=6, help="Retrieval top_k sent to /api/generate.")
    parser.add_argument("--max-tokens", type=int, default=1800, help="Max tokens sent to /api/generate.")
    parser.add_argument("--dry-run", action="store_true", help="Load fixture and print planned cases without calling API.")
    parser.add_argument("--fail-on-quality", action="store_true", help="Exit 1 when any quality check fails.")
    args = parser.parse_args(argv)

    fixture_path = Path(args.fixture)
    fixture = load_fixture(fixture_path)
    selected_cases = list(fixture.cases)
    wanted_ids = {str(item) for item in list(args.case_id or []) if str(item or "").strip()}
    if wanted_ids:
        selected_cases = [item for item in selected_cases if str(item.get("id") or "") in wanted_ids]
    if int(args.limit or 0) > 0:
        selected_cases = selected_cases[: int(args.limit)]
    if not selected_cases:
        print("[ERROR] no cases selected", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"[OK] fixture: {fixture_path}")
        print(f"[OK] docs: {len(fixture.docs)}")
        print(f"[OK] cases: {len(selected_cases)}")
        for idx, case in enumerate(selected_cases, start=1):
            doc_ids = ", ".join(str(item) for item in _as_list(case.get("docIds")))
            print(f"{idx:02d}. {case.get('id')} [{doc_ids}] {case.get('question')}")
        return 0

    base_url = str(args.base_url or DEFAULT_BASE_URL).rstrip("/")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(selected_cases, start=1):
        case_id = str(case.get("id") or f"case-{idx}")
        try:
            row = run_case(
                base_url=base_url,
                fixture=fixture,
                case=case,
                timeout_s=float(args.timeout_s),
                top_k=int(args.top_k),
                max_tokens=int(args.max_tokens),
            )
        except Exception as exc:
            row = {
                "id": case_id,
                "status": "error",
                "done": True,
                "error": str(exc),
                "quality": {"ok": False, "failures": [{"name": "runner_error", "detail": str(exc)}]},
            }
        rows.append(row)
        ok = bool((row.get("quality") or {}).get("ok")) if isinstance(row.get("quality"), dict) else False
        print(f"[{idx}/{len(selected_cases)}] {case_id} -> {'pass' if ok else 'fail'} ({row.get('latency_ms', 0)} ms)")

    summary = {
        "total": len(rows),
        "passed": sum(1 for row in rows if bool((row.get("quality") or {}).get("ok"))),
        "failed": sum(1 for row in rows if not bool((row.get("quality") or {}).get("ok"))),
        "base_url": base_url,
        "fixture": str(fixture_path),
        "output_dir": str(output_dir),
    }
    _write_jsonl(output_dir / "raw_results.jsonl", rows)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        _build_report(rows, fixture_path=fixture_path, base_url=base_url, output_dir=output_dir),
        encoding="utf-8",
    )
    print(f"[OK] research QA eval finished: {output_dir}")
    if bool(args.fail_on_quality) and summary["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
