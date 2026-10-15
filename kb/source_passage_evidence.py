from __future__ import annotations

import re


def bind_source_passage_primary(card: dict, primary: dict, hits: list[dict]) -> dict:
    """Bind a compound context excerpt to its first verified source passage."""
    quote = str(
        primary.get("snippet")
        or primary.get("evidence_quote")
        or card.get("snippet")
        or card.get("evidence_quote")
        or ""
    )
    match = re.match(
        r"^\[Source passage \d+: .*?\bp\.\s*(\d+)\]\s*(.*?)(?=\[Source passage \d+:|\Z)",
        quote,
        re.S,
    )
    if not match:
        return primary
    source = re.sub(
        r"/+",
        "/",
        str(primary.get("source_path") or card.get("source_path") or "").replace(
            "\\", "/"
        ),
    ).casefold()
    body = match.group(2).strip()
    for hit in hits:
        meta = hit.get("meta") or {}
        if (
            re.sub(
                r"/+", "/", str(meta.get("source_path") or "").replace("\\", "/")
            ).casefold()
            != source
        ):
            continue
        for passage in meta.get("source_passages") or []:
            actual = str(passage.get("text") or "").strip()
            if (
                len(actual) < 40
                or re.sub(r"\s+", " ", body)[:100] != re.sub(r"\s+", " ", actual)[:100]
            ):
                continue
            if int(passage.get("page_start") or 0) != int(match.group(1)):
                continue
            return {
                **primary,
                **passage,
                "source_path": primary.get("source_path") or card.get("source_path"),
                "snippet": actual,
                "evidence_quote": actual,
            }
    return primary
