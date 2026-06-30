import json

from kb.agent import tools as agent_tools


def test_retrieve_references_resolves_upstream_reference_from_index(monkeypatch, tmp_path):
    hits = [
        {
            "text": "The paper follows frequency multiplexed illumination [7].",
            "score": 3.0,
            "meta": {
                "source_path": "paper-a.md",
                "source_sha1": "sha-a",
                "heading_path": "Introduction",
            },
        }
    ]
    grouped_doc = {
        "score": 4.0,
        "text": "The paper follows frequency multiplexed illumination [7].",
        "meta": {
            "source_path": "paper-a.md",
            "source_sha1": "sha-a",
            "ref_best_heading_path": "Introduction",
            "ref_show_snippets": ["The paper follows frequency multiplexed illumination [7]."],
        },
    }
    captured = {}

    def fake_group(hits_raw, prompt_text, top_k_docs, **kwargs):
        captured["prompt_text"] = prompt_text
        captured["top_k_docs"] = top_k_docs
        return [grouped_doc]

    def fake_load(db_dir):
        captured["db_dir"] = db_dir
        return {"docs": {"paper-a.md": {"refs": {"7": {}}}}}

    def fake_resolve(index_data, source_path, ref_num, *, source_sha1=""):
        captured["resolve"] = (index_data, source_path, ref_num, source_sha1)
        return {
            "source_path": source_path,
            "source_name": "Current Paper",
            "ref_num": int(ref_num),
            "ref": {
                "num": 7,
                "title": "Fast hyperspectral single-pixel imaging",
                "authors": "Jiang X, Li Z",
                "year": "2022",
                "venue": "Optics Express",
                "doi": "10.1364/oe.458742",
                "doi_url": "https://doi.org/10.1364/oe.458742",
                "raw": "[7] Jiang X, Li Z. Fast hyperspectral single-pixel imaging. Optics Express, 2022.",
                "metadata_status": "crossref_enriched",
            },
        }

    monkeypatch.setattr(agent_tools, "_group_hits_by_doc_for_refs", fake_group)
    monkeypatch.setattr(agent_tools, "load_reference_index", fake_load)
    monkeypatch.setattr(agent_tools, "resolve_reference_entry", fake_resolve)

    result = agent_tools.retrieve_references(
        "Which upstream frequency multiplexed illumination paper should I read?",
        hits,
        db_dir=tmp_path,
        top_k=3,
    )

    refs = result["references"]
    assert result["reference_index_available"] is True
    assert result["resolved_reference_count"] == 1
    assert captured["resolve"][1:] == ("paper-a.md", 7, "sha-a")
    assert refs[0]["ref_num"] == 7
    assert refs[0]["title"] == "Fast hyperspectral single-pixel imaging"
    assert refs[0]["doi"] == "10.1364/oe.458742"
    assert refs[0]["source_paper"] == "Current Paper"
    assert refs[0]["reference_index_available"] is True
    assert refs[0]["anchor"]
    assert refs[0]["num"] == 7
    assert refs[0]["is_inpaper"] is True
    assert refs[0]["cite_fmt"].startswith("[7] Jiang")
    assert refs[0]["shelf_item_kind"] == "reference"
    assert refs[0]["shelf_origin"] == "agent_trace"
    assert refs[0]["card_reference_entry"].startswith("[7] Jiang")
    assert "frequency" in refs[0]["why_relevant"]
    json.dumps(result)


def test_retrieve_references_falls_back_to_source_summary_without_index(monkeypatch, tmp_path):
    grouped_doc = {
        "score": 2.5,
        "text": "The retrieved paper discusses prior work but no local reference index is available.",
        "meta": {
            "source_path": "paper-b.md",
            "ref_best_heading_path": "Related Work",
        },
    }

    monkeypatch.setattr(agent_tools, "_group_hits_by_doc_for_refs", lambda *args, **kwargs: [grouped_doc])
    monkeypatch.setattr(agent_tools, "load_reference_index", lambda db_dir: {})

    result = agent_tools.retrieve_references(
        "Which prior work should I follow up?",
        [],
        db_dir=tmp_path,
        top_k=3,
    )

    assert result["reference_index_available"] is False
    assert result["resolved_reference_count"] == 0
    assert result["references"][0]["source_path"] == "paper-b.md"
    assert result["references"][0]["heading_path"] == "Related Work"
    assert result["references"][0]["reference_index_available"] is False
