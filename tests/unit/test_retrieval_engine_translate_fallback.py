from __future__ import annotations


from kb.retrieval_engine import _translate_query_for_search


class _SettingsNoKey:
    api_key = None
    timeout_s = 60.0
    max_retries = 0


def test_translate_query_for_search_heuristic_works_without_api_key() -> None:
    s = _SettingsNoKey()
    out = _translate_query_for_search(s, "这篇论文的主要贡献是什么？")
    assert isinstance(out, (str, type(None)))
    assert out is not None
    assert "contribution" in out.lower()

