from __future__ import annotations

from tools.manual_regression import reference_locate_benchmark as benchmark


def test_parse_case_filter_accepts_repeated_and_comma_separated_values():
    out = benchmark._parse_case_filter(["A,B", "C", " B "])
    assert out == {"A", "B", "C"}


def test_filter_suite_cases_keeps_only_requested_case_ids():
    suite = {
        "suite_id": "demo",
        "cases": [
            {"id": "A", "prompt": "a"},
            {"id": "B", "prompt": "b"},
            {"id": "C", "prompt": "c"},
        ],
    }

    out = benchmark._filter_suite_cases(suite, {"B", "C"})

    assert [case["id"] for case in out["cases"]] == ["B", "C"]


def test_pack_has_pending_hits_recognizes_top_level_pending_flag():
    assert benchmark._pack_has_pending_hits({"pending": True, "hits": []}) is True
