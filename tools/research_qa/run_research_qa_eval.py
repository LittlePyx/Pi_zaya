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
DEFAULT_REPLAY = Path("docs/research_qa_grounded_replay_v1.jsonl")
DEFAULT_OUT_DIR = Path("test_results/research_qa_eval")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_DB_ROOT = Path("db")

REQUIRED_EVALUATION_FOCUSES = {
    "paper_summary",
    "method_detail",
    "method_comparison",
    "multi_paper_synthesis",
    "upstream_reference",
    "scope_boundary",
}


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


def _stream_generation(
    base_url: str,
    session_id: str,
    timeout_s: float,
    *,
    started_at: float | None = None,
) -> dict[str, Any]:
    req = request.Request(f"{base_url}/api/generate/{parse.quote(session_id)}/stream", method="GET")
    final_payload: dict[str, Any] = {}
    origin = float(started_at) if started_at is not None else time.perf_counter()
    first_answer_ms: float | None = None
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
                visible_answer = str(payload.get("partial") or payload.get("answer") or "").strip()
                if visible_answer and first_answer_ms is None:
                    first_answer_ms = round((time.perf_counter() - origin) * 1000.0, 2)
                if payload.get("done"):
                    break
    final_payload = dict(final_payload or {})
    final_payload["_eval_timing"] = {
        "first_answer_ms": first_answer_ms,
        "answer_complete_ms": round((time.perf_counter() - origin) * 1000.0, 2),
    }
    return final_payload


def _refs_payload_is_full(refs_payload: Any, *, user_msg_id: int | str | None = None) -> bool:
    packs = _extract_ref_packs(refs_payload, user_msg_id=user_msg_id)
    if not packs:
        return False
    for pack in packs:
        if bool(pack.get("pending") or pack.get("enrichment_pending")):
            return False
        render_status = str(pack.get("render_status") or "").strip().lower()
        payload_mode = str(pack.get("payload_mode") or "").strip().lower()
        if render_status and render_status != "full":
            return False
        if payload_mode and payload_mode not in {"full", "ready"}:
            return False
    return True


def _case_requires_full_refs_wait(expected: dict[str, Any] | None) -> bool:
    contract = expected if isinstance(expected, dict) else {}
    return bool(
        "maxCardsCompleteMs" in contract
        or contract.get("requireRefsReady")
        or contract.get("requirePolishStatus")
        or contract.get("requireCitationShelfQuality")
    )


def _generation_should_wait_for_full_refs(
    final_payload: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> bool:
    payload = final_payload if isinstance(final_payload, dict) else {}
    return bool(
        payload.get("done")
        and str(payload.get("status") or "").strip().lower() == "done"
        and _case_requires_full_refs_wait(expected)
    )


def _latency_budget_checks(expected: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for expected_key, result_key in (
        ("maxFirstAnswerMs", "first_answer_ms"),
        ("maxAnswerCompleteMs", "answer_complete_ms"),
        ("maxCardsCompleteMs", "cards_complete_ms"),
    ):
        if expected_key not in expected or result_key not in result:
            continue
        try:
            budget_ms = float(expected.get(expected_key))
            actual_ms = float(result.get(result_key))
        except (TypeError, ValueError):
            checks.append(
                {
                    "name": f"latency_{result_key}",
                    "ok": False,
                    "detail": {"actual_ms": result.get(result_key), "max_ms": expected.get(expected_key)},
                }
            )
            continue
        checks.append(
            {
                "name": f"latency_{result_key}",
                "ok": actual_ms <= budget_ms,
                "detail": {"actual_ms": actual_ms, "max_ms": budget_ms},
            }
        )
    return checks


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


def load_replay(path: Path | str = DEFAULT_REPLAY) -> list[dict[str, Any]]:
    target = Path(path)
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{target}:{line_no}: replay row must be an object")
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def validate_fixture_contracts(fixture: ResearchQaFixture) -> list[str]:
    errors: list[str] = []
    known_docs = set(fixture.docs_by_id)
    focus_cases: dict[str, list[str]] = {}
    for case in fixture.cases:
        case_id = str(case.get("id") or "").strip() or "<missing-id>"
        focus = str(case.get("evaluationFocus") or "").strip()
        if focus:
            focus_cases.setdefault(focus, []).append(case_id)
        expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
        doc_ids = [str(item or "").strip() for item in _as_list(case.get("docIds")) if str(item or "").strip()]
        unknown_docs = sorted(set(doc_ids) - known_docs)
        if unknown_docs:
            errors.append(f"{case_id}: unknown doc ids: {', '.join(unknown_docs)}")
        if bool(case.get("sourceGrounded")):
            source_contracts = [
                item
                for item in _as_list(expected.get("claimEvidenceContracts"))
                if isinstance(item, dict)
            ]
            locate_contracts = [
                item
                for item in _as_list(expected.get("requiredLocateContracts"))
                if isinstance(item, dict)
            ]
            if not source_contracts:
                errors.append(f"{case_id}: source-grounded case requires claimEvidenceContracts")
            if not locate_contracts:
                errors.append(f"{case_id}: source-grounded case requires requiredLocateContracts")
            for contract in [*source_contracts, *locate_contracts]:
                contract_id = str(contract.get("id") or "<missing-contract-id>")
                if not str(contract.get("docId") or "").strip():
                    errors.append(f"{case_id}/{contract_id}: source-grounded contract requires docId")
                if _contract_source_page(contract) is None:
                    errors.append(f"{case_id}/{contract_id}: source-grounded contract requires sourcePage")
                if not _contract_terms(contract.get("evidenceTerms")):
                    errors.append(f"{case_id}/{contract_id}: source-grounded contract requires evidenceTerms")
        if not focus:
            continue
        if not _as_list(expected.get("allowedRefDocIds")):
            errors.append(f"{case_id}: focused case requires allowedRefDocIds")
        if not _as_list(expected.get("claimEvidenceContracts")):
            errors.append(f"{case_id}: focused case requires claimEvidenceContracts")
        if not isinstance(expected.get("requiredRouteCounts"), dict):
            errors.append(f"{case_id}: focused case requires requiredRouteCounts")
        if not _as_list(expected.get("requiredLocateContracts")):
            errors.append(f"{case_id}: focused case requires requiredLocateContracts")
    missing_focuses = sorted(REQUIRED_EVALUATION_FOCUSES - set(focus_cases))
    if missing_focuses:
        errors.append(f"fixture missing evaluation focuses: {', '.join(missing_focuses)}")
    return errors


def source_path_for_doc(
    fixture: ResearchQaFixture,
    doc_id: str,
    *,
    db_root: Path | str | None = None,
) -> str:
    doc = fixture.docs_by_id.get(str(doc_id or ""))
    if not isinstance(doc, dict):
        return ""
    directory = str(doc.get("dir") or "").strip()
    file_stem = str(doc.get("file") or directory).strip()
    if not directory or not file_stem:
        return ""
    if db_root is None:
        root = fixture.db_root.rstrip("/\\")
        return f"{root}/{directory}/{file_stem}.en.md"
    return str(Path(db_root) / directory / f"{file_stem}.en.md")


_PAGE_MARKER_RE = re.compile(r"<!--\s*kb_page:\s*(\d+)\s*-->", flags=re.IGNORECASE)


def _source_pages(markdown: str) -> dict[int, str]:
    matches = list(_PAGE_MARKER_RE.finditer(str(markdown or "")))
    pages: dict[int, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        pages[int(match.group(1))] = markdown[start:end]
    return pages


def _contract_source_page(contract: dict[str, Any]) -> int | None:
    try:
        page = int(contract.get("sourcePage") or 0)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def validate_fixture_sources(
    fixture: ResearchQaFixture,
    *,
    db_root: Path | str = DEFAULT_DB_ROOT,
) -> list[str]:
    """Verify reviewed source-grounded contracts against the current Markdown corpus."""

    errors: list[str] = []
    source_cache: dict[str, dict[int, str]] = {}
    for case in fixture.cases:
        if not bool(case.get("sourceGrounded")):
            continue
        case_id = str(case.get("id") or "<missing-id>")
        expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
        contracts = [
            ("claim", item)
            for item in _as_list(expected.get("claimEvidenceContracts"))
            if isinstance(item, dict)
        ]
        contracts.extend(
            ("locate", item)
            for item in _as_list(expected.get("requiredLocateContracts"))
            if isinstance(item, dict)
        )
        for contract_kind, contract in contracts:
            contract_id = str(contract.get("id") or contract_kind)
            doc_id = str(contract.get("docId") or "").strip()
            page = _contract_source_page(contract)
            source_path = Path(source_path_for_doc(fixture, doc_id, db_root=db_root))
            if not source_path.is_file():
                errors.append(f"{case_id}/{contract_id}: source markdown not found: {source_path}")
                continue
            cache_key = str(source_path.resolve())
            if cache_key not in source_cache:
                source_cache[cache_key] = _source_pages(source_path.read_text(encoding="utf-8", errors="ignore"))
            pages = source_cache[cache_key]
            if page not in pages:
                errors.append(f"{case_id}/{contract_id}: source page {page} not found in {source_path.name}")
                continue
            page_text = pages[page]
            evidence_terms = _contract_terms(contract.get("evidenceTerms"))
            missing_terms = [term for term in evidence_terms if not _contains_term(page_text, term)]
            if missing_terms:
                errors.append(
                    f"{case_id}/{contract_id}: page {page} missing evidence terms: {', '.join(missing_terms)}"
                )
    return errors


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
    surface = _payload_text(haystack) if isinstance(haystack, (dict, list)) else haystack
    hay = _norm(surface)
    hay_compact = re.sub(r"\s+", "", hay)
    for needle in _term_aliases(term):
        needle_norm = _norm(needle)
        if needle_norm and (
            needle_norm in hay
            or re.sub(r"\s+", "", needle_norm) in hay_compact
        ):
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
    structured: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    detailed_system_b_nums: set[int] = set()
    for raw in details:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        route = _citation_route(item)
        try:
            num = int(item.get("num") or item.get("ref_num") or item.get("refNum") or 0)
        except Exception:
            num = 0
        source_path = str(item.get("source_path") or item.get("sourcePath") or "").strip().lower()
        title = str(item.get("title") or item.get("card_title") or "").strip().lower()
        doi = str(item.get("doi") or "").strip().lower()
        key = (route, num, source_path, title, doi)
        if key in seen:
            continue
        seen.add(key)
        structured.append(item)
        if route == "system_b" and num > 0:
            detailed_system_b_nums.add(num)
    for item in _inline_inpaper_citation_details(_answer_text(result)):
        try:
            ref_num = int(item.get("ref_num") or 0)
        except Exception:
            ref_num = 0
        if ref_num > 0 and ref_num in detailed_system_b_nums:
            continue
        structured.append(item)
    return structured


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


def _system_a_ref_evidence_details(
    refs_payload: Any,
    *,
    answer: str,
    user_msg_id: int | str | None = None,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def _append(raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        source_path = str(raw.get("source_path") or raw.get("sourcePath") or "").strip()
        heading_path = str(raw.get("heading_path") or raw.get("headingPath") or "").strip()
        snippet = str(
            raw.get("snippet")
            or raw.get("highlight_snippet")
            or raw.get("highlightSnippet")
            or ""
        ).strip()
        if not source_path or not snippet:
            return
        key = (source_path.lower(), heading_path.lower(), snippet[:160].lower())
        if key in seen:
            return
        seen.add(key)
        details.append(
            {
                "citation_route": "system_a",
                "is_inpaper": False,
                "source_path": source_path,
                "source_name": str(raw.get("source_name") or raw.get("sourceName") or "").strip(),
                "heading_path": heading_path,
                "location_label": heading_path,
                "answer_claim": answer,
                "evidence_quote": snippet,
                "support_relation": str(raw.get("selection_reason") or "reference_card_primary_evidence"),
                "block_id": str(raw.get("block_id") or raw.get("blockId") or "").strip(),
                "anchor_id": str(raw.get("anchor_id") or raw.get("anchorId") or "").strip(),
                "anchor_kind": str(raw.get("anchor_kind") or raw.get("anchorKind") or "").strip(),
            }
        )

    for pack in _extract_ref_packs(refs_payload, user_msg_id=user_msg_id):
        _append(pack.get("primary_evidence"))
        for hit in _as_list(pack.get("hits")):
            if not isinstance(hit, dict):
                continue
            ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
            reader_open = ui_meta.get("reader_open") if isinstance(ui_meta.get("reader_open"), dict) else {}
            _append(reader_open.get("primaryEvidence"))
    return details


def _expected_int(expected: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return max(0, int(expected.get(key, default) or default))
    except Exception:
        return max(0, int(default or 0))


def _citation_route(detail: dict[str, Any]) -> str:
    route = str(detail.get("citation_route") or detail.get("citationRoute") or "").strip().lower()
    if route in {"system_a", "system_b"}:
        return route
    return "system_b" if bool(detail.get("is_inpaper")) else "system_a"


def _strict_phrase_hits(payload: Any, phrases: list[str]) -> list[str]:
    haystack = _norm(_payload_text(payload))
    return [phrase for phrase in phrases if _norm(phrase) and _norm(phrase) in haystack]


def _detail_evidence_payload(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        key: detail.get(key)
        for key in (
            "evidence_quote",
            "citation_context",
            "card_evidence",
            "support_relation",
            "upstream_work_role",
            "user_question_relation",
            "raw",
        )
        if detail.get(key)
    }


def _detail_locator_payload(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        key: detail.get(key)
        for key in (
            "heading_path",
            "location_label",
            "card_locator",
            "block_id",
            "anchor_id",
            "anchor_kind",
            "page_start",
            "page_end",
            "source_path",
        )
        if detail.get(key) not in (None, "")
    }


def _matching_citation_details(
    fixture: ResearchQaFixture,
    citation_details: list[dict[str, Any]],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    doc_id = str(contract.get("docId") or "").strip()
    route = str(contract.get("route") or "").strip().lower()
    matches: list[dict[str, Any]] = []
    for detail in citation_details:
        if doc_id and not _doc_matches_payload(fixture, doc_id, detail):
            continue
        if route and _citation_route(detail) != route:
            continue
        matches.append(detail)
    return matches


def _contract_terms(value: Any) -> list[str]:
    return [str(item or "").strip() for item in _as_list(value) if str(item or "").strip()]


def _missing_term_groups(payload: Any, value: Any) -> list[list[str]]:
    groups = [
        _contract_terms(group)
        for group in _as_list(value)
        if _contract_terms(group)
    ]
    return [group for group in groups if not any(_contains_term(payload, term) for term in group)]


def _detail_matches_source_page(detail: dict[str, Any], source_page: int | None) -> bool:
    if source_page is None:
        return True
    try:
        page_start = int(detail.get("page_start") or detail.get("pageStart") or 0)
        page_end = int(detail.get("page_end") or detail.get("pageEnd") or page_start)
    except (TypeError, ValueError):
        return False
    return page_start > 0 and page_start <= source_page <= max(page_start, page_end)


def _claim_evidence_contract_failures(
    fixture: ResearchQaFixture,
    citation_details: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    *,
    answer: str = "",
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, contract in enumerate(contracts, start=1):
        contract_id = str(contract.get("id") or f"claim-{index}").strip()
        answer_terms = _contract_terms(contract.get("answerTerms"))
        answer_term_groups = _as_list(contract.get("answerTermGroups"))
        evidence_terms = _contract_terms(contract.get("evidenceTerms"))
        source_page = _contract_source_page(contract)
        answer_scope = str(contract.get("answerScope") or "citation_claim").strip().lower()
        candidates = _matching_citation_details(fixture, citation_details, contract)
        matched = False
        for detail in candidates:
            if not _detail_matches_source_page(detail, source_page):
                continue
            claim_payload: Any
            if answer_scope in {"response", "full_answer", "answer"}:
                claim_payload = answer
            else:
                claim_payload = {
                    "answer_claim": detail.get("answer_claim"),
                    "support_relation": detail.get("support_relation"),
                    "user_question_relation": detail.get("user_question_relation"),
                }
            if answer_terms and not all(_contains_term(claim_payload, term) for term in answer_terms):
                continue
            if _missing_term_groups(claim_payload, answer_term_groups):
                continue
            evidence_payload = _detail_evidence_payload(detail)
            if evidence_terms and not all(_contains_term(evidence_payload, term) for term in evidence_terms):
                continue
            matched = True
            break
        if not matched:
            failures.append(
                {
                    "id": contract_id,
                    "doc_id": str(contract.get("docId") or ""),
                    "route": str(contract.get("route") or ""),
                    "candidate_count": len(candidates),
                    "answer_terms": answer_terms,
                    "answer_term_groups": answer_term_groups,
                    "answer_scope": answer_scope,
                    "evidence_terms": evidence_terms,
                    "source_page": source_page,
                }
            )
    return failures


def _locate_contract_failures(
    fixture: ResearchQaFixture,
    citation_details: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, contract in enumerate(contracts, start=1):
        contract_id = str(contract.get("id") or f"locate-{index}").strip()
        locator_terms = _contract_terms(contract.get("locatorTerms"))
        evidence_terms = _contract_terms(contract.get("evidenceTerms"))
        source_page = _contract_source_page(contract)
        candidates = _matching_citation_details(fixture, citation_details, contract)
        matched = False
        for detail in candidates:
            if not _detail_matches_source_page(detail, source_page):
                continue
            locator_payload = _detail_locator_payload(detail)
            if locator_terms and not all(_contains_term(locator_payload, term) for term in locator_terms):
                continue
            evidence_payload = _detail_evidence_payload(detail)
            if evidence_terms and not all(_contains_term(evidence_payload, term) for term in evidence_terms):
                continue
            if not str(detail.get("source_path") or "").strip():
                continue
            matched = True
            break
        if not matched:
            failures.append(
                {
                    "id": contract_id,
                    "doc_id": str(contract.get("docId") or ""),
                    "route": str(contract.get("route") or ""),
                    "candidate_count": len(candidates),
                    "locator_terms": locator_terms,
                    "evidence_terms": evidence_terms,
                    "source_page": source_page,
                }
            )
    return failures


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
    contract_citation_details = list(citation_details)
    if bool(expected.get("allowSystemARefEvidence")):
        contract_citation_details.extend(
            _system_a_ref_evidence_details(
                refs_payload,
                answer=answer,
                user_msg_id=user_msg_id,
            )
        )
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: Any = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    status = str(result.get("status") or "").strip().lower()
    add_check("generation_done", bool(result.get("done")) and status not in {"error", "canceled"}, status)
    checks.extend(_latency_budget_checks(expected, result))

    forbidden_hits = [phrase for phrase in fixture.forbidden_phrases if _contains_term(answer, phrase)]
    add_check("answer_no_template_phrase", not forbidden_hits, forbidden_hits)

    assistant_message = result.get("assistant_message") if isinstance(result.get("assistant_message"), dict) else {}
    assistant_meta = assistant_message.get("meta") if isinstance(assistant_message.get("meta"), dict) else {}
    guide_contracts = (
        assistant_meta.get("paper_guide_contracts")
        if isinstance(assistant_meta.get("paper_guide_contracts"), dict)
        else {}
    )
    render_packet = (
        guide_contracts.get("render_packet")
        if isinstance(guide_contracts.get("render_packet"), dict)
        else {}
    )
    has_rendered_answer = any(
        str(render_packet.get(key) or "").strip()
        for key in ("rendered_body", "rendered_content", "answer_markdown")
    )
    unresolved_citation_markers = re.findall(
        r"\[\[(?:CITE|SUPPORT):[^\]]+\]\]",
        answer,
        flags=re.IGNORECASE,
    )
    if has_rendered_answer:
        unresolved_citation_markers.extend(
            re.findall(r"(?<!\[)\[(?:R)?\d+\](?!\s*\()", answer, flags=re.IGNORECASE)
        )
    add_check(
        "answer_no_unresolved_citation_markers",
        not unresolved_citation_markers,
        unresolved_citation_markers,
    )

    direct_starts = ("the paper cites", "this is stated in", "该文在", "文献中提到")
    add_check("answer_directly_addresses_question", not _norm(answer).startswith(direct_starts), answer[:120])

    required_terms = [str(item) for item in _as_list(expected.get("requiredAnswerTerms")) if str(item or "").strip()]
    missing_answer_terms = [term for term in required_terms if not _contains_term(answer, term)]
    add_check("answer_contains_required_terms", not missing_answer_terms, missing_answer_terms)
    required_term_groups = _as_list(expected.get("requiredAnswerTermGroups"))
    missing_answer_term_groups = _missing_term_groups(answer, required_term_groups)
    if required_term_groups:
        add_check(
            "answer_contains_required_term_groups",
            not missing_answer_term_groups,
            missing_answer_term_groups,
        )

    forbidden_answer_terms = _contract_terms(expected.get("forbiddenAnswerTerms"))
    if forbidden_answer_terms:
        present_forbidden_answer_terms = _strict_phrase_hits(answer, forbidden_answer_terms)
        add_check(
            "answer_avoids_forbidden_claims",
            not present_forbidden_answer_terms,
            present_forbidden_answer_terms,
        )

    required_ref_doc_ids = [str(item) for item in _as_list(expected.get("requiredRefDocIds")) if str(item or "").strip()]
    missing_ref_docs = [doc_id for doc_id in required_ref_doc_ids if not _doc_matches_payload(fixture, doc_id, refs_payload)]
    add_check("refs_include_required_docs", not missing_ref_docs, missing_ref_docs)

    allowed_ref_doc_ids = {
        str(item or "").strip()
        for item in _as_list(expected.get("allowedRefDocIds"))
        if str(item or "").strip()
    }
    if allowed_ref_doc_ids:
        ref_doc_ids = set(_unique_doc_ids_in_payload(fixture, refs_payload))
        unexpected_ref_docs = sorted(ref_doc_ids - allowed_ref_doc_ids)
        max_unexpected_ref_docs = _expected_int(expected, "maxUnexpectedRefDocCount")
        add_check(
            "refs_avoid_unexpected_docs",
            len(unexpected_ref_docs) <= max_unexpected_ref_docs,
            {
                "unexpected": unexpected_ref_docs,
                "allowed": sorted(allowed_ref_doc_ids),
                "max": max_unexpected_ref_docs,
            },
        )

    min_ref_hits = _expected_int(expected, "minRefHits")
    if min_ref_hits:
        add_check("refs_min_hit_count", len(ref_hits) >= min_ref_hits, {"actual": len(ref_hits), "min": min_ref_hits})
    max_ref_hits = _expected_optional_int(expected, "maxRefHits")
    if max_ref_hits is not None:
        add_check(
            "refs_max_hit_count",
            len(ref_hits) <= max_ref_hits,
            {"actual": len(ref_hits), "max": max_ref_hits},
        )

    min_ref_doc_count = _expected_int(expected, "minRefDocCount")
    if min_ref_doc_count:
        ref_doc_ids = _unique_doc_ids_in_payload(fixture, refs_payload)
        add_check(
            "refs_min_doc_count",
            len(ref_doc_ids) >= min_ref_doc_count,
            {"actual": len(ref_doc_ids), "min": min_ref_doc_count, "doc_ids": ref_doc_ids},
        )
    max_ref_doc_count = _expected_optional_int(expected, "maxRefDocCount")
    if max_ref_doc_count is not None:
        ref_doc_ids = _unique_doc_ids_in_payload(fixture, refs_payload)
        add_check(
            "refs_max_doc_count",
            len(ref_doc_ids) <= max_ref_doc_count,
            {"actual": len(ref_doc_ids), "max": max_ref_doc_count, "doc_ids": ref_doc_ids},
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

    allowed_citation_doc_ids = {
        str(item or "").strip()
        for item in _as_list(expected.get("allowedCitationDocIds"))
        if str(item or "").strip()
    }
    if allowed_citation_doc_ids:
        citation_doc_ids = set(_unique_doc_ids_in_payload(fixture, citation_details))
        unexpected_citation_docs = sorted(citation_doc_ids - allowed_citation_doc_ids)
        max_unexpected_citation_docs = _expected_int(expected, "maxUnexpectedCitationDocCount")
        add_check(
            "citations_avoid_unexpected_docs",
            len(unexpected_citation_docs) <= max_unexpected_citation_docs,
            {
                "unexpected": unexpected_citation_docs,
                "allowed": sorted(allowed_citation_doc_ids),
                "max": max_unexpected_citation_docs,
            },
        )

    required_route_counts = expected.get("requiredRouteCounts") if isinstance(expected.get("requiredRouteCounts"), dict) else {}
    if required_route_counts:
        actual_route_counts = {
            route: sum(1 for detail in contract_citation_details if _citation_route(detail) == route)
            for route in ("system_a", "system_b")
        }
        missing_route_counts = {
            route: {"actual": actual_route_counts.get(route, 0), "min": _expected_int(required_route_counts, route)}
            for route in ("system_a", "system_b")
            if actual_route_counts.get(route, 0) < _expected_int(required_route_counts, route)
        }
        add_check("citations_match_required_routes", not missing_route_counts, missing_route_counts or actual_route_counts)

    claim_evidence_contracts = [
        item for item in _as_list(expected.get("claimEvidenceContracts")) if isinstance(item, dict)
    ]
    if claim_evidence_contracts:
        claim_contract_failures = _claim_evidence_contract_failures(
            fixture,
            contract_citation_details,
            claim_evidence_contracts,
            answer=answer,
        )
        add_check("claims_have_matching_evidence", not claim_contract_failures, claim_contract_failures)

    locate_contracts = [
        item for item in _as_list(expected.get("requiredLocateContracts")) if isinstance(item, dict)
    ]
    if locate_contracts:
        locate_failures = _locate_contract_failures(fixture, contract_citation_details, locate_contracts)
        add_check("citations_have_expected_locators", not locate_failures, locate_failures)

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
    max_citation_doc_count = _expected_optional_int(expected, "maxCitationDocCount")
    if max_citation_doc_count is not None:
        citation_doc_ids = _unique_doc_ids_in_payload(fixture, citation_details)
        add_check(
            "citations_max_doc_count",
            len(citation_doc_ids) <= max_citation_doc_count,
            {"actual": len(citation_doc_ids), "max": max_citation_doc_count, "doc_ids": citation_doc_ids},
        )

    inpaper_details = [item for item in citation_details if bool(item.get("is_inpaper"))]
    max_system_b_count = _expected_optional_int(expected, "maxSystemBCount")
    if max_system_b_count is not None:
        add_check(
            "system_b_max_count",
            len(inpaper_details) <= max_system_b_count,
            {"actual": len(inpaper_details), "max": max_system_b_count},
        )
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
            f"export_ready={int(shelf_quality.get('export_ready_count') or 0)}, "
            f"summary_export_ready={int(shelf_quality.get('summary_export_ready_count') or 0)}, "
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


def evaluate_replay_rows(
    fixture: ResearchQaFixture,
    replay_rows: list[dict[str, Any]],
    *,
    selected_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    case_by_id = {str(case.get("id") or "").strip(): case for case in fixture.cases}
    selected = set(selected_case_ids or [])
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for row in replay_rows:
        case_id = str(row.get("case_id") or row.get("id") or "").strip()
        if selected and case_id not in selected:
            continue
        if not case_id:
            errors.append(f"line {row.get('_line_no')}: missing case_id")
            continue
        if case_id in seen:
            errors.append(f"{case_id}: duplicate replay row")
            continue
        seen.add(case_id)
        case = case_by_id.get(case_id)
        if not isinstance(case, dict):
            errors.append(f"{case_id}: replay references unknown fixture case")
            continue
        if str(row.get("review_status") or "").strip().lower() != "accepted":
            errors.append(f"{case_id}: replay row must be human-reviewed and accepted")
            continue
        result = row.get("result") if isinstance(row.get("result"), dict) else row
        result = dict(result)
        result.setdefault("id", case_id)
        quality = validate_case(case, fixture, result)
        results.append({"id": case_id, "quality": quality})

    expected_ids = {
        str(case.get("id") or "").strip()
        for case in fixture.cases
        if str(case.get("evaluationFocus") or "").strip()
        and (not selected or str(case.get("id") or "").strip() in selected)
    }
    missing_rows = sorted(expected_ids - seen)
    if missing_rows:
        errors.append(f"missing focused replay rows: {', '.join(missing_rows)}")
    failed_results = [row for row in results if not bool((row.get("quality") or {}).get("ok"))]
    for row in failed_results:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        names = [
            str(item.get("name") or "")
            for item in _as_list(quality.get("failures"))
            if isinstance(item, dict)
        ]
        errors.append(f"{row.get('id')}: {', '.join(names) or 'quality check failed'}")
    return {
        "ok": not errors,
        "total": len(results),
        "passed": len(results) - len(failed_results),
        "failed": len(failed_results),
        "errors": errors,
        "cases": results,
    }


def _selected_context_pack(fixture: ResearchQaFixture, case: dict[str, Any]) -> dict[str, Any] | None:
    items: list[dict[str, Any]] = []
    for doc_id_raw in _as_list(case.get("docIds")):
        doc_id = str(doc_id_raw or "").strip()
        doc = fixture.docs_by_id.get(doc_id) or {}
        source_path = source_path_for_doc(fixture, doc_id)
        if not source_path:
            continue
        items.append(
            {
                "key": f"research-qa:{doc_id}",
                "kind": "source",
                "sourcePath": source_path,
                "sourceName": str(doc.get("title") or doc.get("shortLabel") or doc_id),
                "title": str(doc.get("title") or doc_id),
            }
        )
    if not items:
        return None
    return {
        "version": 1,
        "id": f"research-qa:{case.get('id')}",
        "count": len(items),
        "items": items,
    }


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
    query_scope = str(case.get("queryScope") or "library").strip().lower()
    source_lock_path = preferred_sources[0] if query_scope == "current_paper" and len(preferred_sources) == 1 else ""
    source_lock_name = ""
    if source_lock_path:
        doc_id = str(_as_list(case.get("docIds"))[0] or "").strip()
        doc = fixture.docs_by_id.get(doc_id) or {}
        source_lock_name = str(doc.get("title") or doc.get("shortLabel") or "").strip()
    prompt_context = _selected_context_pack(fixture, case) if query_scope == "selected_context" else None
    started = time.perf_counter()
    conv = _post_json(
        base_url,
        "/api/conversations",
        {
            "title": f"research-qa-{case_id}",
            "project_id": None,
            "mode": "paper_guide" if source_lock_path else "normal",
            "bound_source_path": source_lock_path,
            "bound_source_name": source_lock_name,
            "bound_source_ready": bool(source_lock_path),
        },
        timeout_s=timeout_s,
    )
    conv_id = str(conv.get("id") or "").strip()
    if not conv_id:
        raise RuntimeError("conversation creation returned no id")

    generation_started = time.perf_counter()
    gen = _post_json(
        base_url,
        "/api/generate",
        {
            "conv_id": conv_id,
            "prompt": question,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "preferred_sources": preferred_sources,
            "source_lock_path": source_lock_path,
            "source_lock_name": source_lock_name,
            "query_scope": query_scope,
            "prompt_context": prompt_context,
        },
        timeout_s=timeout_s,
    )
    session_id = str(gen.get("session_id") or "").strip()
    if not session_id:
        raise RuntimeError("generation returned no session_id")
    final_payload = _stream_generation(
        base_url,
        session_id,
        timeout_s=timeout_s,
        started_at=generation_started,
    )
    eval_timing = (
        dict(final_payload.get("_eval_timing") or {})
        if isinstance(final_payload.get("_eval_timing"), dict)
        else {}
    )
    refs_payload = _get_json(base_url, f"/api/references/conversation/{parse.quote(conv_id)}", timeout_s)
    cards_complete_ms: float | None = None
    if _refs_payload_is_full(refs_payload, user_msg_id=gen.get("user_msg_id")):
        cards_complete_ms = round((time.perf_counter() - generation_started) * 1000.0, 2)
    elif _generation_should_wait_for_full_refs(
        final_payload,
        case.get("expected") if isinstance(case.get("expected"), dict) else {},
    ):
        card_wait_deadline = time.perf_counter() + min(45.0, max(1.0, float(timeout_s)))
        while time.perf_counter() < card_wait_deadline:
            time.sleep(0.35)
            refs_payload = _get_json(
                base_url,
                f"/api/references/conversation/{parse.quote(conv_id)}",
                timeout_s,
            )
            if _refs_payload_is_full(refs_payload, user_msg_id=gen.get("user_msg_id")):
                cards_complete_ms = round((time.perf_counter() - generation_started) * 1000.0, 2)
                break
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
        "first_answer_ms": eval_timing.get("first_answer_ms"),
        "answer_complete_ms": eval_timing.get("answer_complete_ms"),
        "cards_complete_ms": cards_complete_ms,
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
    parser.add_argument(
        "--replay",
        default="",
        help="Validate a human-reviewed deterministic JSONL replay without calling the API.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load fixture and print planned cases without calling API.")
    parser.add_argument(
        "--validate-sources",
        action="store_true",
        help="Check source-grounded contracts against page-marked Markdown without calling the API.",
    )
    parser.add_argument(
        "--db-root",
        default=str(DEFAULT_DB_ROOT),
        help="Markdown corpus root used by --validate-sources (default: db).",
    )
    parser.add_argument("--fail-on-quality", action="store_true", help="Exit 1 when any quality check fails.")
    args = parser.parse_args(argv)

    fixture_path = Path(args.fixture)
    fixture = load_fixture(fixture_path)
    fixture_errors = validate_fixture_contracts(fixture)
    if fixture_errors:
        for error in fixture_errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    selected_cases = list(fixture.cases)
    wanted_ids = {str(item) for item in list(args.case_id or []) if str(item or "").strip()}
    if wanted_ids:
        selected_cases = [item for item in selected_cases if str(item.get("id") or "") in wanted_ids]
    if int(args.limit or 0) > 0:
        selected_cases = selected_cases[: int(args.limit)]
    if not selected_cases:
        print("[ERROR] no cases selected", file=sys.stderr)
        return 2

    if args.validate_sources:
        source_fixture = ResearchQaFixture(
            db_root=fixture.db_root,
            docs=fixture.docs,
            cases=selected_cases,
            forbidden_phrases=fixture.forbidden_phrases,
        )
        source_errors = validate_fixture_sources(source_fixture, db_root=args.db_root)
        if source_errors:
            for error in source_errors:
                print(f"[ERROR] {error}", file=sys.stderr)
            return 1
        grounded_count = sum(1 for case in selected_cases if bool(case.get("sourceGrounded")))
        print(
            f"[OK] source grounding: cases={grounded_count} "
            f"db_root={Path(args.db_root).resolve()}"
        )
        return 0

    if args.replay:
        summary = evaluate_replay_rows(
            fixture,
            load_replay(args.replay),
            selected_case_ids=wanted_ids or None,
        )
        for row in summary["cases"]:
            quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
            print(f"[replay] {row.get('id')} -> {'pass' if quality.get('ok') else 'fail'}")
        for error in summary["errors"]:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(
            f"[OK] reviewed replay: total={summary['total']} "
            f"passed={summary['passed']} failed={summary['failed']}"
        )
        return 0 if summary["ok"] or not bool(args.fail_on_quality) else 1

    if args.dry_run:
        print(f"[OK] fixture: {fixture_path}")
        print(f"[OK] docs: {len(fixture.docs)}")
        print(f"[OK] cases: {len(selected_cases)}")
        print(f"[OK] source-grounded cases: {sum(1 for case in selected_cases if bool(case.get('sourceGrounded')))}")
        focus_counts: dict[str, int] = {}
        for idx, case in enumerate(selected_cases, start=1):
            doc_ids = ", ".join(str(item) for item in _as_list(case.get("docIds")))
            print(f"{idx:02d}. {case.get('id')} [{doc_ids}] {case.get('question')}")
            focus = str(case.get("evaluationFocus") or "").strip()
            if focus:
                focus_counts[focus] = focus_counts.get(focus, 0) + 1
        print(f"[OK] evaluation focuses: {json.dumps(focus_counts, ensure_ascii=False, sort_keys=True)}")
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
