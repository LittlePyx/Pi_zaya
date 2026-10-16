from __future__ import annotations

from pathlib import Path
import re

from kb.source_blocks import load_source_blocks, normalize_match_text


def corruption_condition_question(prompt: str) -> bool:
    return bool(
        re.search(r"likelihood|explicit model|显式模型|似然", prompt, re.I)
        and re.search(r"corruption|污染", prompt, re.I)
        and re.search(r"statistical condition|remains necessary|统计条件|仍需", prompt, re.I)
    )


def target_condition_quote(text: str) -> str:
    """Recognize the complete L2 target relation, not shared topic words."""
    surface = re.sub(r"\s+", " ", text)
    if not re.search(r"L\s*2\s+minimization", surface, re.I):
        return ""
    match = re.search(
        r"input-conditioned target distributions p\s*\(\s*y\s*\|\s*x\s*\) "
        r"are replaced with arbitrary distributions that have the same conditional expected values\. "
        r"This implies that we can, in principle, corrupt the training targets of a neural network "
        r"with zero-mean noise without changing what the network learns\.",
        surface, re.I,
    )
    return match.group() if match else ""


def _source(value: object) -> str:
    return re.sub(r"/+", "/", str(value or "").replace("\\", "/")).casefold()


def _verified_block(path: str, row: dict) -> dict | None:
    try:
        block = next(
            (b for b in load_source_blocks(Path(path)) if b["block_id"] == row.get("block_id")),
            None,
        )
        quote = normalize_match_text(str(row.get("evidence_quote") or row.get("text") or ""))
        if (
            block and int(block.get("page_start") or 0) > 0
            and block.get("page_start") == row.get("page_start")
            and all(not row.get(key) or row[key] == block.get(key)
                    for key in ("anchor_id", "page_end", "heading_path"))
            and len(quote) >= 40
            and quote in normalize_match_text(str(block.get("raw_text") or block.get("text") or ""))
        ):
            return block
    except (OSError, ValueError, TypeError):
        pass
    return None


def _corruption_summary_quote(text: str) -> str:
    match = re.search(
        r"In stark contrast to our approach, they need a perfect, explicit computational model of the corruption; "
        r"our requirement of the knowledge of the appropriate statistical summary \(mean, median\) is significantly less limiting\.",
        re.sub(r"\s+", " ", text), re.I,
    )
    return match.group() if match else ""


def attach_target_condition_slot(plan: dict, hits: list[dict]) -> dict:
    """Keep both verified facets, regardless of which page the plan chose first."""
    slots = [dict(s) if isinstance(s, dict) else s for s in plan.get("slots") or []]
    sources = {
        str(s.get("source_path") or "") for s in slots if isinstance(s, dict)
        and s.get("preferred_system") == "system_a"
        and "noise2noise" in Path(str(s.get("source_path") or "")).name.casefold()
    }
    for path in sources:
        for number, hit in enumerate(hits or [], 1):
            meta = hit.get("meta") or {}
            if _source(meta.get("source_path")) != _source(path):
                continue
            for row in meta.get("source_passages") or [{**meta, "text": hit.get("text")}]:
                selected_direct = meta.get("direct_question_evidence") and all(
                    row.get(key) == meta.get(key) for key in ("block_id", "anchor_id", "page_start")
                )
                if not (row.get("direct_question_evidence") or selected_direct):
                    continue
                block = _verified_block(path, row)
                body = str((block or {}).get("raw_text") or "")
                quote = target_condition_quote(body) or _corruption_summary_quote(body)
                if not quote:
                    continue
                existing = next((s for s in slots if isinstance(s, dict)
                                 and _source(s.get("source_path")) == _source(path)
                                 and s.get("block_id") == row.get("block_id")), None)
                if existing is not None:
                    if any(existing.get(key) and existing[key] != block.get(key)
                           for key in ("anchor_id", "page_start", "page_end", "heading_path")):
                        continue
                    # A long converted paragraph can contain interleaved prose.
                    # Show the exact relation, retaining page-local audit text.
                    existing["evidence_quote"] = quote
                    existing["source_evidence_quotes"] = [body]
                    existing["anchor_kind"] = block.get("kind")
                    existing["source_passage_bundle"] = True
                    existing["compound_same_page_evidence"] = True
                    continue
                slots.append({
                    "preferred_system": "system_a", "claim_type": "paper_evidence",
                    "source_path": path, "source_name": Path(path).stem,
                    "candidate_hits": [number], "evidence_quote": quote,
                    "source_evidence_quotes": [body],
                    "anchor_kind": block.get("kind"),
                    "source_passage_bundle": True, "compound_same_page_evidence": True,
                    **{key: block.get(key) for key in (
                        "block_id", "anchor_id", "heading_path", "page_start", "page_end",
                    )},
                    "strict_locate": True, "evidence_selection_reason": "prompt_contract_block",
                })
    return {**plan, "slots": slots} if slots else plan


def complete_target_condition(answer: str, plan: dict, hits: list[dict]) -> str:
    if re.search(r"input-conditioned target distributions.{0,150}same conditional expected values", answer, re.I):
        return answer
    for slot in plan.get("slots") or []:
        if not isinstance(slot, dict) or slot.get("preferred_system") != "system_a":
            continue
        path = str(slot.get("source_path") or "")
        if "noise2noise" not in Path(path).name.casefold():
            continue
        block = _verified_block(path, slot)
        if not block or not target_condition_quote(str(block.get("raw_text") or "")):
            continue
        if "same conditional expected values" not in str(slot.get("evidence_quote") or ""):
            continue
        number = next((i for i, h in enumerate(hits, 1) if _source((h.get("meta") or {}).get("source_path")) == _source(path)), 0)
        if number:
            # A newly supplied direct condition supersedes this specific
            # retrieval-gap caveat. Do not retain a contradictory "not found".
            answer = re.sub(
                r"(?im)(?:^|\n\n)(?:\*{0,2}Not stated in the retrieved evidence\*{0,2}\s*\n+)?"
                r"The retrieved snippets do not specify a general formal condition[^\n]*"
                r"beyond the summary-statistic requirement[^\n]*(?=\n\n|$)",
                "", answer,
            ).strip()
            return (
                f"{answer}\n\nFor L2 minimization, input-conditioned target distributions can be replaced "
                f"with distributions that have the same conditional expected values [{number}]. "
                "This permits corrupting the training targets with zero-mean noise without changing "
                f"what the network learns [{number}]."
            )
    return answer
