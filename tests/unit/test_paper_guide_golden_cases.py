from __future__ import annotations

import json
from pathlib import Path


def _golden_cases_path() -> Path:
    return Path(__file__).resolve().parents[2] / "web" / "tests" / "fixtures" / "paper-guide-golden-cases.json"


def test_paper_guide_golden_cases_schema_and_regression_anchors():
    cases = json.loads(_golden_cases_path().read_text(encoding="utf-8"))

    assert isinstance(cases, list)
    assert len(cases) >= 12
    ids = [str(item.get("id") or "").strip() for item in cases]
    assert len(ids) == len(set(ids))

    for item in cases:
        assert str(item.get("id") or "").strip()
        assert str(item.get("question") or "").strip()
        assert isinstance(item.get("answerContainsAny"), list)
        assert isinstance(item.get("answerNotContains"), list)
        assert "retrieved context" in [str(token).lower() for token in item["answerNotContains"]]
        assert int(item.get("minLocateButtons") or 0) >= 1

    serialized = json.dumps(cases, ensure_ascii=False)
    for bad_fragment in ("�", "杩欑瘒", "鏂囩珷", "闃呰", "鍗曞儚", "锛?", "銆"):
        assert bad_fragment not in serialized

    q01 = next(item for item in cases if item["id"] == "nat-single-pixel-q01-core-problem")
    assert "blk_ce8a1e326e8b_00049" in set(q01["locateBlockIdsAny"])

    q03 = next(item for item in cases if item["id"] == "nat-single-pixel-q03-reconstruction-tradeoffs")
    assert "三类主流重建方法" in q03["answerNotContains"]
    assert {
        "blk_ce8a1e326e8b_00022",
        "blk_ce8a1e326e8b_00025",
        "blk_ce8a1e326e8b_00026",
    }.issubset(set(q03["locateBlockIdsAny"]))
