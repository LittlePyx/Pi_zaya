from __future__ import annotations

import re


_AGENT_TRACE_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s*)?(?:[*_`~\s]*)?(?:research\s+agent\s+trace|agent\s+trace)(?:[*_`~\s]*)?:?\s*$",
    flags=re.IGNORECASE,
)
_AGENT_TRACE_KEY_RE = re.compile(r'^\s*(?:"?agent_trace"?|agentTrace)\s*[:=]')
_AGENT_TRACE_JSON_RE = re.compile(r'"(?:agent_trace|agentTrace)"\s*:', flags=re.IGNORECASE)


def _line_looks_like_agent_trace_boundary(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    return bool(_AGENT_TRACE_HEADING_RE.match(text) or _AGENT_TRACE_KEY_RE.match(text))


def _fenced_block_looks_like_agent_trace(lines: list[str], index: int) -> bool:
    line = str(lines[index] or "").strip()
    if not (line.startswith("```") or line.startswith("~~~")):
        return False
    lookahead = "\n".join(lines[index + 1 : min(len(lines), index + 12)])
    return bool(_AGENT_TRACE_JSON_RE.search(lookahead))


def _json_block_looks_like_agent_trace(lines: list[str], index: int) -> bool:
    line = str(lines[index] or "").strip()
    if not line.startswith("{"):
        return False
    lookahead = "\n".join(lines[index : min(len(lines), index + 12)])
    return bool(_AGENT_TRACE_JSON_RE.search(lookahead))


def _find_agent_trace_boundary(text: str) -> int:
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for idx, line in enumerate(lines):
        if (
            _line_looks_like_agent_trace_boundary(line)
            or _fenced_block_looks_like_agent_trace(lines, idx)
            or _json_block_looks_like_agent_trace(lines, idx)
        ):
            return idx
    return -1


def clean_assistant_answer_presentation_text(value: object) -> str:
    """Remove appended Research Agent debug payloads from user-facing answer text."""
    text = str(value or "")
    if not text.strip():
        return text
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    boundary = _find_agent_trace_boundary(normalized)
    if boundary < 0:
        return text
    return "\n".join(normalized.split("\n")[:boundary]).rstrip()
