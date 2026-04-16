from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.chat_render import enrich_messages_with_reference_render
from tests._paper_guide_fixtures import (
    build_paper_guide_runtime_fixture,
    build_scinerf_like_fixture,
)
from tests.replay.curate_paper_guide_failure_pool_v1 import _case_signature, _interesting_score


_CAPTURED_PATHS = [
    Path(__file__).resolve().parent.parent / "replay" / "paper_guide_failure_pool_captured_failures.jsonl",
    Path(__file__).resolve().parent.parent / "replay" / "paper_guide_failure_pool_captured_promoted.jsonl",
    Path(__file__).resolve().parent.parent / "replay" / "paper_guide_failure_pool_captured.jsonl",
]


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for ln in path.read_text(encoding="utf-8-sig").splitlines():
        s = str(ln or "").strip()
        if not s or s.startswith("#"):
            continue
        rec = json.loads(s)
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _captured_case_index(paths: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for rec in _read_jsonl(path):
            key = str(rec.get("id") or "").strip()
            if key and key not in out:
                out[key] = dict(rec)
    return out


def _resolve_case_reference(rec: dict, *, captured_index: dict[str, dict]) -> dict:
    fixture_name = str(rec.get("fixture") or "").strip().lower()
    if fixture_name != "captured_ref":
        return dict(rec)
    captured_case_id = str(rec.get("captured_case_id") or rec.get("source_case_id") or "").strip()
    if not captured_case_id:
        raise ValueError(f"captured_ref case missing captured_case_id: {rec}")
    base = dict(captured_index.get(captured_case_id) or {})
    if not base:
        raise ValueError(f"captured_ref points to unknown captured case: {captured_case_id}")
    merged = dict(base)
    for key, value in dict(rec).items():
        if key in {"fixture", "captured_case_id", "source_case_id"}:
            continue
        merged[key] = value
    merged["fixture"] = "captured"
    merged["captured_case_id"] = captured_case_id
    merged["id"] = str(rec.get("id") or captured_case_id).strip() or captured_case_id
    return merged


def _load_jsonl(path: Path) -> list[dict]:
    captured_index = _captured_case_index(_CAPTURED_PATHS)
    return [
        _resolve_case_reference(rec, captured_index=captured_index)
        for rec in _read_jsonl(path)
    ]


def _normalized_case_family(case: dict) -> str:
    raw = str(case.get("intent_family") or case.get("tag") or "").strip().lower()
    if "figure_walkthrough" in raw:
        return "figure_walkthrough"
    if "box" in raw:
        return "box_only"
    if "abstract" in raw:
        return "abstract"
    if "discussion" in raw:
        return "discussion_only"
    if "strength" in raw or "limitation" in raw or "trade" in raw:
        return "strength_limits"
    if "citation" in raw:
        return "citation_lookup"
    if "equation" in raw:
        return "equation"
    if "figure" in raw:
        return "figure"
    if raw in {"method", "reproduce"}:
        return "method"
    return "overview"


_POOL_PATH = Path(__file__).resolve().parent.parent / "replay" / "paper_guide_failure_pool_v1.jsonl"


def test_paper_guide_failure_replay_pool_v1_resolves_captured_refs():
    cases = _load_jsonl(_POOL_PATH)
    captured = [case for case in cases if str(case.get("captured_case_id") or "").strip()]
    assert captured, "expected curated captured_ref cases in replay pool v1"
    sample = captured[0]
    assert str(sample.get("fixture") or "").strip() == "captured"
    assert isinstance(sample.get("provenance"), dict)
    assert isinstance(sample.get("hits"), list)
    if bool(sample.get("expect_locate")):
        assert str(sample.get("expect_block_id") or "").strip()


def test_paper_guide_failure_replay_pool_v1_keeps_broad_family_coverage():
    cases = _load_jsonl(_POOL_PATH)
    families = {_normalized_case_family(case) for case in cases}
    captured = [case for case in cases if str(case.get("captured_case_id") or "").strip()]
    captured_families = {_normalized_case_family(case) for case in captured}
    captured_docs = {
        str(((case.get("provenance") or {}).get("doc_id") or "")).strip()
        for case in captured
        if isinstance(case.get("provenance"), dict)
    }
    assert len(cases) >= 18
    assert {"overview", "method", "equation", "figure", "citation_lookup", "discussion_only", "strength_limits", "abstract", "box_only", "figure_walkthrough"}.issubset(families)
    assert len(captured) >= 14
    assert {"overview", "method", "figure", "citation_lookup", "discussion_only", "strength_limits", "abstract", "box_only", "figure_walkthrough"}.issubset(captured_families)
    assert len({doc_id for doc_id in captured_docs if doc_id}) >= 4


def test_paper_guide_failure_replay_pool_v1_keeps_best_captured_case_per_signature():
    pool_rows = _read_jsonl(_POOL_PATH)
    captured_index = _captured_case_index(_CAPTURED_PATHS)
    best_by_signature: dict[str, dict] = {}
    for case in captured_index.values():
        sig = _case_signature(case)
        prev = best_by_signature.get(sig)
        if prev is None or _interesting_score(case) > _interesting_score(prev):
            best_by_signature[sig] = case

    upgrades: list[str] = []
    for row in pool_rows:
        if str(row.get("fixture") or "").strip().lower() != "captured_ref":
            continue
        captured_case_id = str(row.get("captured_case_id") or row.get("source_case_id") or "").strip()
        if not captured_case_id:
            continue
        case = captured_index.get(captured_case_id)
        if not isinstance(case, dict):
            upgrades.append(f"{row.get('id')}: missing captured case {captured_case_id}")
            continue
        best = best_by_signature.get(_case_signature(case)) or {}
        best_case_id = str(best.get("id") or "").strip()
        if best_case_id and best_case_id != captured_case_id:
            upgrades.append(f"{row.get('id')}: {captured_case_id} -> {best_case_id}")

    assert not upgrades, "expected replay pool v1 to keep best captured case per signature:\n" + "\n".join(upgrades)


@pytest.mark.parametrize("case", _load_jsonl(_POOL_PATH), ids=lambda c: str(c.get("id") or "case"))
def test_paper_guide_failure_replay_pool_v1(case: dict, tmp_path: Path):
    fixture_name = str(case.get("fixture") or "").strip()
    prompt = str(case.get("prompt") or "").strip()
    assistant_content = str(case.get("assistant_content") or "").strip()
    expect_locate = bool(case.get("expect_locate"))
    expect_block = str(case.get("expect_block") or "").strip()
    expect_block_id = str(case.get("expect_block_id") or "").strip()
    expect_anchor_id = str(case.get("expect_anchor_id") or "").strip()

    if fixture_name == "scinerf_like":
        fx = build_scinerf_like_fixture(tmp_path)
        md_main = fx["md_main"]
        blocks = list(fx["blocks"])
        expected_block = fx.get(expect_block) if expect_block else None
        expected_block_id = str((expected_block or {}).get("block_id") or "").strip() or expect_block_id
        expected_anchor_id = str((expected_block or {}).get("anchor_id") or "").strip() or expect_anchor_id
        provenance = None
        # Only attach provenance segments when the replay case expects locating.
        if expect_locate:
            provenance = {
                "source_path": str(md_main),
                "source_name": Path(str(md_main)).name.replace(".md", ".pdf"),
                "segments": [
                    {
                        "segment_id": f"seg::{case.get('id')}",
                        "text": assistant_content or prompt,
                        "locate_policy": "required",
                        "locate_surface_policy": "primary",
                        "primary_heading_path": str((expected_block or {}).get("heading_path") or "Methods").strip(),
                        "primary_block_id": expected_block_id,
                        "primary_anchor_id": expected_anchor_id,
                        "anchor_kind": str((expected_block or {}).get("kind") or "paragraph"),
                        "claim_type": str(case.get("intent_family") or "overview") + "_claim",
                    }
                ],
            }
        hits = [{"text": "dummy", "meta": {"source_path": str(md_main)}}]
    elif fixture_name == "paper_guide_runtime":
        fx = build_paper_guide_runtime_fixture(tmp_path)
        # Keep these cases as "non-locate smoke" for now; the fixture is for
        # reference index and box parsing, not deterministic locate assertions.
        provenance = None
        hits = [{"text": "dummy", "meta": {"source_path": str(fx.get("nat_md") or fx.get("lsa_md") or "")}}]
    elif fixture_name == "captured":
        provenance = case.get("provenance") if isinstance(case.get("provenance"), dict) else None
        hits_raw = case.get("hits")
        hits = [dict(item) for item in list(hits_raw or []) if isinstance(item, dict)]
        expected_block_id = expect_block_id
        expected_anchor_id = expect_anchor_id
    else:
        raise AssertionError(f"unknown fixture: {fixture_name}")

    messages = [
        {"id": 1, "role": "user", "content": prompt or "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": assistant_content or "answer",
            "provenance": provenance,
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": str(case.get("intent_family") or "").strip()},
                }
            },
        },
    ]
    refs_by_user = {1: {"hits": hits}}

    rendered = enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]

    # In render_packet_only mode, legacy projections must be stripped.
    assert "rendered_body" not in msg
    assert "rendered_content" not in msg
    assert "copy_text" not in msg
    assert "copy_markdown" not in msg
    assert "cite_details" not in msg

    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    rendered_body = str(packet.get("rendered_body") or "")
    rendered_content = str(packet.get("rendered_content") or "")
    copy_text = str(packet.get("copy_text") or "")
    copy_md = str(packet.get("copy_markdown") or "")

    assert rendered_body.strip() or rendered_content.strip()
    assert copy_text.strip() or copy_md.strip()

    # Leak guards: user-visible output must not include internal structured tokens.
    hay = "\n".join([rendered_body, rendered_content, copy_text, copy_md])
    assert "[[CITE:" not in hay
    assert "[SID:" not in hay

    if expect_locate:
        reader_open = packet.get("reader_open") or {}
        locate_target = packet.get("locate_target") or {}
        assert isinstance(reader_open, dict)
        assert isinstance(locate_target, dict)
        assert str(reader_open.get("blockId") or "").strip()
        assert str(reader_open.get("anchorId") or "").strip()
        assert str(locate_target.get("blockId") or "").strip()
        assert str(locate_target.get("anchorId") or "").strip()
        if expected_block_id:
            got_reader_block = str(reader_open.get("blockId") or "").strip()
            got_locate_block = str(locate_target.get("blockId") or "").strip()
            if fixture_name == "captured":
                assert got_reader_block in {expected_block_id, got_locate_block}
                assert got_locate_block in {expected_block_id, got_reader_block}
            else:
                assert got_reader_block == expected_block_id
                assert got_locate_block == expected_block_id
        if expected_anchor_id:
            got_reader_anchor = str(reader_open.get("anchorId") or "").strip()
            got_locate_anchor = str(locate_target.get("anchorId") or "").strip()
            if fixture_name == "captured":
                assert got_reader_anchor in {expected_anchor_id, got_locate_anchor}
                assert got_locate_anchor in {expected_anchor_id, got_reader_anchor}
            else:
                assert got_reader_anchor == expected_anchor_id
                assert got_locate_anchor == expected_anchor_id
    else:
        # When a case is explicitly marked as non-locate, it should not emit a
        # strict/primary locate payload. Some flows may still include soft reader
        # open hints; allow them as long as they are not strict.
        reader_open = packet.get("reader_open") or {}
        locate_target = packet.get("locate_target") or {}
        if isinstance(reader_open, dict) and reader_open:
            assert bool(reader_open.get("strictLocate")) is False
        if isinstance(locate_target, dict) and locate_target:
            # These are camelCase keys in the packet payload.
            assert str(locate_target.get("locatePolicy") or "").strip().lower() != "required"
            assert str(locate_target.get("locateSurfacePolicy") or "").strip().lower() != "primary"
