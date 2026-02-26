import fitz
import pytest

from kb.converter.geometry_utils import (
    _bbox_width,
    _bbox_height,
    _rect_area,
    _rect_intersection_area,
    _overlap_1d,
    _union_rect
)

def test_bbox_width_height():
    # bbox in format: (x0, y0, x1, y1)
    bbox = (10.0, 20.0, 50.0, 80.0)
    assert _bbox_width(bbox) == 40.0
    assert _bbox_height(bbox) == 60.0

def test_rect_area():
    rect = fitz.Rect(0.0, 0.0, 10.0, 10.0)
    assert _rect_area(rect) == 100.0

def test_rect_intersection_area():
    rect_a = fitz.Rect(0.0, 0.0, 10.0, 10.0)
    rect_b = fitz.Rect(5.0, 5.0, 15.0, 15.0)
    
    # Intersecting area should be from (5,5) to (10,10) -> width 5, height 5 -> area 25
    assert _rect_intersection_area(rect_a, rect_b) == 25.0
    
    # Disjoint rects
    rect_c = fitz.Rect(20.0, 20.0, 30.0, 30.0)
    assert _rect_intersection_area(rect_a, rect_c) == 0.0
    
    # One rect completely inside another
    rect_d = fitz.Rect(2.0, 2.0, 8.0, 8.0)
    assert _rect_intersection_area(rect_a, rect_d) == 36.0

def test_overlap_1d():
    # Overlapping lines
    assert _overlap_1d(0, 10, 5, 15) == 5.0
    
    # Non-overlapping lines
    assert _overlap_1d(0, 5, 10, 15) == 0.0
    
    # Completely inside
    assert _overlap_1d(0, 20, 5, 10) == 5.0

def test_union_rect():
    # Empty list
    assert _union_rect([]) is None
    
    # Single rect
    r1 = fitz.Rect(10, 10, 20, 20)
    assert _union_rect([r1]) == fitz.Rect(10, 10, 20, 20)
    
    # Multiple rects
    r2 = fitz.Rect(30, 30, 40, 40)
    r3 = fitz.Rect(5, 5, 15, 15)
    union = _union_rect([r1, r2, r3])
    
    # The minimum x0, y0 should be 5, 5. The maximum x1, y1 should be 40, 40.
    assert union == fitz.Rect(5, 5, 40, 40)
