from kb.converter.post_processing import postprocess_markdown
import re


def test_unwrap_text_wrapped_display_math_prose_fragments():
    src = """
# 2. Introduction
In the past decade, the spectral response range of SPI has been expanded from

$$
\\text{range of SPI has been expanded from visible light}[8]\\text{ to UV,}[9,10]\\text{ infrared,}[11,12]
$$

and even terahertz wavelengths.
"""
    out = postprocess_markdown(src)
    # The pseudo display-math should be unwrapped into normal text.
    assert "range of SPI has been expanded from visible light" in out
    assert "$$" not in out


def test_keep_real_display_math_equation():
    src = """
$$
T(x,y)^* = \\frac{1}{N} \\sum_{i=1}^{N} (\\Delta B_i - \\langle \\Delta B_i \\rangle)
$$
"""
    out = postprocess_markdown(src)
    # Real equation should stay in display math.
    assert "$$" in out
    assert "\\frac{1}{N}" in out


def test_unwrap_text_wrapped_display_math_with_citation_ranges():
    src = """
These advancements have significantly enhanced its application in medical

$$
\\text{imaging},[16--19]\\text{ biological imaging},[20--24]
$$

and related areas.
"""
    out = postprocess_markdown(src)
    assert "$$" not in out
    assert "imaging,[16--19] biological imaging,[20--24]" in out


def test_unwrap_single_line_display_math_that_is_actually_prose():
    src = """
$$range of SPI has been expanded from visible light[8] to UV,[9,10] infrared,[11,12]$$
"""
    out = postprocess_markdown(src)
    assert "$$" not in out
    assert "range of SPI has been expanded from visible light" in out
    assert "$^{[8]}$" in out


def test_keep_real_display_math_superscript_not_converted_to_hat():
    src = """
$$T(x,y)^* = \\frac{1}{N} \\sum_{i=1}^N (\\Delta B_i - \\langle \\Delta B_i \\rangle)$$
"""
    out = postprocess_markdown(src)
    assert "$$" in out
    assert "^N" in out
    assert "\\hat{N}" not in out


def test_fix_malformed_code_fence_inline_closer():
    src = """
A paragraph before a short code-like block.

```
110010101001 001111000101
101011001000111 ```

## 3. Methods
Body paragraph.
"""
    out = postprocess_markdown(src)
    assert "101011001000111 ```" not in out
    standalone_fences = [ln for ln in out.splitlines() if re.match(r"^\s*```\s*$", ln)]
    assert len(standalone_fences) == 2
    assert re.search(r"(?m)^#{1,6}\s+3\.\s+Methods\b", out)
    assert "Body paragraph." in out


def test_fix_malformed_code_fence_auto_close_before_heading():
    src = """
```
some code-ish line
still in fence
## 4. Results
Normal paragraph.
"""
    out = postprocess_markdown(src)
    # Heading should not stay trapped inside an unterminated fence.
    assert re.search(r"(?m)^#{1,6}\s+4\.\s+Results\b", out)
    standalone_fences = [ln for ln in out.splitlines() if re.match(r"^\s*```\s*$", ln)]
    assert len(standalone_fences) >= 2
