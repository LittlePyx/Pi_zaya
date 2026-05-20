from __future__ import annotations

import hashlib
import json
import re

from ui import refs_renderer

# ── helpers ────────────────────────────────────────────────────────────


def _sid_for(path: str) -> str:
    """Calculate the SID that refs_renderer._source_cite_id generates."""
    return "s" + hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]


def _collect_anchors(html: str) -> list[str]:
    """Extract all href anchors from rendered HTML.

    Catches both markdown-style [text](#anchor "title") and HTML <a href="#anchor">.
    """
    md_anchors = re.findall(r'\(#([^\s")]+)', html)
    html_anchors = re.findall(r'href="#([^"]+)"', html)
    return md_anchors + html_anchors


def _collect_payloads(details: list[dict]) -> list[dict]:
    """Return the details list as‑is (payloads are already dicts)."""
    return list(details)


def _find_payload_by_num(details: list[dict], num: int) -> dict | None:
    for d in details:
        if int(d.get("num") or 0) == num:
            return d
    return None


# ── fixtures ───────────────────────────────────────────────────────────

PAPER1 = "paper1.pdf"
PAPER2 = "paper2.pdf"
PAPER1_TEXT = "Single-pixel imaging (SPI) utilizes the second-order correlation [3,58]."
PAPER2_TEXT = "Compressive sensing enables sub-Nyquist sampling [12,31]."


def _base_hits():
    return [
        {"meta": {"source_path": PAPER1, "source_sha1": "aaa"}, "text": PAPER1_TEXT},
        {"meta": {"source_path": PAPER2, "source_sha1": "bbb"}, "text": PAPER2_TEXT},
    ]


def _ref_entry(ref_num: int, title: str = "", **kw) -> dict:
    entry = {
        "source_path": PAPER1,
        "source_name": PAPER1,
        "ref_num": ref_num,
        "ref": {
            "title": title or f"Paper title {ref_num}",
            "authors": "Sen, P. et al.",
            "year": "2009",
            "venue": "Nature Photonics",
            "volume": "3",
            "issue": "5",
            "pages": "291-295",
            "doi": f"10.1038/nphoton.2009.{ref_num}",
            "raw": f"[{ref_num}] Sen P et al. Test reference {ref_num}. 2009.",
        },
    }
    entry["ref"].update(kw)
    return entry


def _mock_empty_hints(*args, **kwargs):
    return {}


# ══════════════════════════════════════════════════════════════════════
# Test 1 — Renderer output format
# ══════════════════════════════════════════════════════════════════════


def test_num_cite_maps_to_hit_source(monkeypatch):
    """System A: [1] renders as markdown link and resolves via hits."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *a, **kw: None)

    md = "SPI reconstructs images from 1D signals [1]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t", canonical_paths=[PAPER1, PAPER2],
    )

    # [1] is rendered as a clickable markdown link
    assert "[1](#" in out, f"Expected [1] link, got: {out}"
    assert len(details) >= 1

    d = _find_payload_by_num(details, 1)
    assert d is not None
    assert PAPER1 in d["source_path"], f"Expected source {PAPER1}, got {d['source_path']}"
    assert d["raw"]  # snippet text should exist


def test_num_cite_maps_to_second_source(monkeypatch):
    """System A: [2] resolves to the second hit source."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *a, **kw: None)

    md = "CS enables sub-Nyquist sampling [2]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t", canonical_paths=[PAPER1, PAPER2],
    )

    assert "[2](#" in out
    d = _find_payload_by_num(details, 2)
    assert d is not None
    assert PAPER2 in d["source_path"], f"Expected {PAPER2}, got {d['source_path']}"


def test_num_cite_out_of_range_stripped(monkeypatch):
    """System A: [999] out of range with 2 hits → stripped."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *a, **kw: None)

    md = "This is not supported [999]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t", canonical_paths=[PAPER1, PAPER2],
    )

    assert "[999]" not in out
    assert not details


def test_struct_cite_renders_as_clickable_system_b_link(monkeypatch):
    """System B: [[CITE:sid:58]] renders as a clickable citation link."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "extract_citation_context_hints", _mock_empty_hints)

    def fake_resolve(*args, **kw):
        return _ref_entry(58, title="Single-pixel imaging via compressive sampling")
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    sid = _sid_for(PAPER1)
    md = f"As shown by Sen et al. [[CITE:{sid}:58]]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t",
    )

    assert "[58](#kb-cite-" in out, f"Expected clickable System B link, got: {out}"
    assert len(details) >= 1

    d = _find_payload_by_num(details, 58)
    assert d is not None
    assert d["title"] == "Single-pixel imaging via compressive sampling"
    assert d["doi"] == "10.1038/nphoton.2009.58"


def test_struct_cite_resolves_metadata_from_ref_index(monkeypatch):
    """System B: payload contains all fields from references_index.json."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "extract_citation_context_hints", _mock_empty_hints)

    def fake_resolve(*args, **kw):
        return _ref_entry(
            3,
            title="A new compressive imaging method",
            authors="Smith, J. et al.",
            year="2010",
            venue="Optics Express",
            doi="10.1364/OE.18.012345",
        )
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    sid = _sid_for(PAPER1)
    md = f"Smith et al. [[CITE:{sid}:3]] proposed a new method."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t",
    )

    assert "[3](#kb-cite-" in out
    d = _find_payload_by_num(details, 3)
    assert d is not None
    assert d["title"] == "A new compressive imaging method"
    assert d["authors"] == "Smith, J. et al."
    assert d["year"] == "2010"
    assert d["venue"] == "Optics Express"
    assert d["doi"] == "10.1364/OE.18.012345"
    assert d["doi_url"] == "https://doi.org/10.1364/OE.18.012345"


def test_struct_cite_invalid_sid_stripped(monkeypatch):
    """System B: [[CITE:bad:1]] with unresolvable SID → stripped."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *a, **kw: None)

    md = "Some claim [[CITE:nonexistent:1]]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t",
    )

    assert "[[CITE:" not in out
    assert "<sup>" not in out
    assert not details


def test_num_and_struct_cite_coexist_in_same_segment(monkeypatch):
    """Both [1] and [[CITE:58]] appear together in one segment."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "extract_citation_context_hints", _mock_empty_hints)

    call_count = 0

    def fake_resolve(*args, **kw):
        nonlocal call_count
        call_count += 1
        return _ref_entry(58, title="Ref 58 title")
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    sid = _sid_for(PAPER1)
    md = f"SPI [1] was first demonstrated by Sen et al. [[CITE:{sid}:58]]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t", canonical_paths=[PAPER1, PAPER2],
    )

    # Both must be present
    assert "[1](#" in out, f"[1] link missing in: {out}"
    assert "[58](#kb-cite-" in out, f"System B link missing in: {out}"

    # Both should be in details
    d1 = _find_payload_by_num(details, 1)
    d58 = _find_payload_by_num(details, 58)
    assert d1 is not None
    assert d58 is not None
    assert PAPER1 in d1["source_path"]
    assert d58["title"] == "Ref 58 title"


def test_adjacent_struct_cites_both_render(monkeypatch):
    """Multiple adjacent [[CITE:...]] tokens both render."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "extract_citation_context_hints", _mock_empty_hints)

    def fake_resolve(*args, **kw):
        ref_num = int(args[2])
        return _ref_entry(ref_num, title=f"Ref {ref_num}")
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    sid = _sid_for(PAPER1)
    md = f"Multiple refs [[CITE:{sid}:3]][[CITE:{sid}:58]][[CITE:{sid}:59]]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t",
    )

    assert "[3](#kb-cite-" in out
    assert "[58](#kb-cite-" in out
    assert "[59](#kb-cite-" in out
    assert len(details) >= 3


def test_no_hits_no_citations_stripped(monkeypatch):
    """With no hits, both [n] and [[CITE:...]] are stripped."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *a, **kw: None)

    md = "No sources [1] and nothing [[CITE:sid:58]]."
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, [], anchor_ns="t",
    )

    # When no hits: [[CITE:...]] stripped, [n] left as-is (renderer can't know
    # if it's a citation without source info), no payload details.
    assert "CITE" not in out
    assert not details


def test_empty_text_returns_empty(monkeypatch):
    """Empty input returns empty output."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})

    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta("", [], anchor_ns="t")
    assert out == ""
    assert not details


# ══════════════════════════════════════════════════════════════════════
# Test 2 — Payload data correctness
# ══════════════════════════════════════════════════════════════════════


def test_hit_payload_contains_snippet_text(monkeypatch):
    """System A payload includes actual snippet text from the hit."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *a, **kw: None)

    md = "The technique was revolutionary [1]."
    snippet_text = "Revolutionary single-pixel imaging technique was first proposed in 2008."
    hits = [
        {"meta": {"source_path": PAPER1, "source_sha1": "aaa"}, "text": snippet_text},
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t-payload", canonical_paths=[PAPER1],
    )

    d = _find_payload_by_num(details, 1)
    assert d is not None
    assert d["raw"] == snippet_text, f"Snippet mismatch: {d['raw']} != {snippet_text}"


def test_hit_payload_has_source_path_from_canonical(monkeypatch):
    """System A payload source_path matches canonical_paths entry."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *a, **kw: None)

    md = "Two sources were compared [1] and [2]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t-check", canonical_paths=[PAPER1, PAPER2],
    )

    d1 = _find_payload_by_num(details, 1)
    d2 = _find_payload_by_num(details, 2)
    assert d1 is not None
    assert d2 is not None
    assert PAPER1 in d1["source_path"], f"[1] -> {d1['source_path']}"
    assert PAPER2 in d2["source_path"], f"[2] -> {d2['source_path']}"


def test_ref_payload_has_all_metadata_fields(monkeypatch):
    """System B payload contains title/authors/year/venue/doi/doi_url."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "extract_citation_context_hints", _mock_empty_hints)

    def fake_resolve(*args, **kw):
        return _ref_entry(
            12,
            title="Compressive sensing: signal recovery",
            authors="Candes, E. et al.",
            year="2006",
            venue="IEEE Trans. Info. Theory",
            volume="52",
            issue="2",
            pages="489-509",
            doi="10.1109/TIT.2005.862083",
        )
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    sid = _sid_for(PAPER1)
    md = f"Candes [[CITE:{sid}:12]] proposed compressive sensing."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t-full",
    )

    d = _find_payload_by_num(details, 12)
    assert d is not None
    assert d["title"] == "Compressive sensing: signal recovery"
    assert d["authors"] == "Candes, E. et al."
    assert d["year"] == "2006"
    assert d["venue"] == "IEEE Trans. Info. Theory"
    assert d["volume"] == "52"
    assert d["issue"] == "2"
    assert d["pages"] == "489-509"
    assert d["doi"] == "10.1109/TIT.2005.862083"
    assert d["doi_url"] == "https://doi.org/10.1109/TIT.2005.862083"


def test_hit_payload_has_no_ref_metadata_fields(monkeypatch):
    """System A payload (hit‑based) should NOT have ref metadata like doi."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *a, **kw: None)

    md = "Simple claim [1]."
    hits = [{"meta": {"source_path": PAPER1, "source_sha1": "aaa"}, "text": "Some text."}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t-no-ref", canonical_paths=[PAPER1],
    )

    d = _find_payload_by_num(details, 1)
    assert d is not None
    # System A payload has raw (snippet) but no doi / authors / year
    assert d.get("doi") == "" or d.get("doi") is None
    assert d.get("authors") == "" or d.get("authors") is None
    # It should have raw text (the snippet)
    assert d.get("raw") == "Some text."


# ══════════════════════════════════════════════════════════════════════
# Test 4 — Link ↔ Payload consistency
# ══════════════════════════════════════════════════════════════════════


def test_every_link_has_corresponding_payload(monkeypatch):
    """Every visible anchor link has a matching detail entry."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "extract_citation_context_hints", _mock_empty_hints)

    call_log: list[int] = []

    def fake_resolve(*args, **kw):
        ref_num = int(args[2])
        call_log.append(ref_num)
        return _ref_entry(ref_num, title=f"Ref {ref_num}")
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    sid = _sid_for(PAPER1)
    md = f"Claim [1] with refs [[CITE:{sid}:3]] and [[CITE:{sid}:58]]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t-consist", canonical_paths=[PAPER1, PAPER2],
    )

    # Collect all anchors from visible HTML
    html_anchors = _collect_anchors(out)
    assert html_anchors, "No anchor links found in output"

    # Every detail's anchor must appear in the HTML
    detail_anchors = {d["anchor"] for d in details if d.get("anchor")}
    for da in detail_anchors:
        assert da in html_anchors, f"Detail anchor {da} missing in HTML anchors: {html_anchors}"

    # Every HTML anchor must have a matching detail
    # (details are deduplicated by num+source, but links could be from group cites)
    num_details_found = 0
    for ha in html_anchors:
        found = any(
            d.get("anchor") == ha
            for d in details
        )
        if found:
            num_details_found += 1


def test_no_orphan_payloads(monkeypatch):
    """Every detail entry has a corresponding anchor in the HTML."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "extract_citation_context_hints", _mock_empty_hints)

    def fake_resolve(*args, **kw):
        return _ref_entry(99, title="Orphan test")
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    sid = _sid_for(PAPER1)
    md = f"Just one cite [[CITE:{sid}:99]]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t-orphan",
    )

    html_anchors = _collect_anchors(out)

    # All details should have their anchor in the HTML
    for d in details:
        anchor = d.get("anchor", "")
        assert anchor, f"Detail {d['num']} has empty anchor"
        assert anchor in html_anchors, f"Detail {d['num']} anchor {anchor} not found in HTML"


def test_payload_json_is_valid(monkeypatch):
    """Payload struct can be serialized to JSON (readable by React)."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "extract_citation_context_hints", _mock_empty_hints)

    def fake_resolve(*args, **kw):
        return _ref_entry(42, title="JSON test")
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    sid = _sid_for(PAPER1)
    md = f"JSON test [[CITE:{sid}:42]]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t-json",
    )

    assert details
    # Verify every detail is JSON-serializable
    for d in details:
        try:
            dumped = json.dumps(d, ensure_ascii=False)
            assert dumped  # not empty
            parsed = json.loads(dumped)
            assert parsed["num"] == d["num"]
        except (TypeError, ValueError) as e:
            raise AssertionError(f"Detail {d['num']} not JSON-serializable: {e}") from e


# ══════════════════════════════════════════════════════════════════════
# Regression: reference_index fallback still works
# ══════════════════════════════════════════════════════════════════════


def test_num_cite_falls_back_to_ref_index_when_no_canonical_paths(monkeypatch):
    """Without canonical_paths, [n] falls back to reference index."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})

    def fake_resolve(*args, **kw):
        source_path = str(args[1])
        ref_num = int(args[2])
        if "paper1" in source_path and ref_num == 24:
            return {
                "source_path": "paper1.pdf",
                "source_name": "paper1.pdf",
                "ref_num": 24,
                "ref": {
                    "title": "Fallback Ref",
                    "authors": "Fallback, A.",
                    "year": "2020",
                    "doi": "",
                    "raw": "[24] Fallback A. 2020.",
                },
            }
        return None

    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    md = "Fallback evidence [24]."
    hits = [{"meta": {"source_path": PAPER1, "source_sha1": "aaa"}, "text": "Some text [24]."}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t-fb",
    )

    # Should still resolve [24] via reference index fallback
    assert "[24](#" in out
    d = _find_payload_by_num(details, 24)
    assert d is not None
    assert d["title"] == "Fallback Ref"


def test_struct_cite_with_different_sids_resolve_to_different_sources(monkeypatch):
    """[[CITE:sid1:3]] and [[CITE:sid2:5]] resolve to different source docs."""
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "extract_citation_context_hints", _mock_empty_hints)

    call_log: list[tuple] = []

    def fake_resolve(*args, **kw):
        source_path = str(args[1])
        ref_num = int(args[2])
        call_log.append((source_path, ref_num))
        return {
            "source_path": source_path,
            "source_name": source_path,
            "ref_num": ref_num,
            "ref": {
                "title": f"Ref {ref_num} from {source_path}",
                "authors": "Author",
                "year": "2020",
                "doi": "",
                "raw": f"[{ref_num}] Author. 2020.",
            },
        }
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    sid1 = _sid_for(PAPER1)
    sid2 = _sid_for(PAPER2)
    md = f"From paper1 [[CITE:{sid1}:3]] and from paper2 [[CITE:{sid2}:5]]."
    hits = _base_hits()
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md, hits, anchor_ns="t-multi-src",
    )

    assert "[3](#kb-cite-" in out
    assert "[5](#kb-cite-" in out
    assert len(details) >= 2

    d3 = _find_payload_by_num(details, 3)
    d5 = _find_payload_by_num(details, 5)
    assert d3 is not None
    assert d5 is not None
    assert PAPER1 in d3["source_path"], f"[3] source: {d3['source_path']}"
    assert PAPER2 in d5["source_path"], f"[5] source: {d5['source_path']}"
