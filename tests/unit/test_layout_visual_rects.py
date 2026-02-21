import pytest

try:
    import fitz
except Exception:  # pragma: no cover
    fitz = None

from kb.converter.layout_analysis import _merge_nearby_visual_rects


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_merge_stacked_multi_panel_figure():
    page_w, page_h = 600.0, 900.0
    rects = [
        fitz.Rect(100, 120, 380, 260),  # top panel group
        fitz.Rect(130, 298, 350, 450),  # bottom panel / chart
    ]
    out = _merge_nearby_visual_rects(rects, page_w=page_w, page_h=page_h)
    assert len(out) == 1
    r = out[0]
    assert r.x0 <= 100 and r.y0 <= 120
    assert r.x1 >= 380 and r.y1 >= 450


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_do_not_merge_distant_stacked_figures():
    page_w, page_h = 600.0, 900.0
    rects = [
        fitz.Rect(100, 120, 380, 260),
        fitz.Rect(130, 430, 350, 580),  # too far below
    ]
    out = _merge_nearby_visual_rects(rects, page_w=page_w, page_h=page_h)
    assert len(out) == 2
