from kb.converter.post_processing import postprocess_markdown
from kb.converter.post_math_rules import contains_bare_tagged_display_math
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
    assert "[8]" in out
    assert "$^{[8]}$" not in out


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


def test_normalize_common_unit_rendering_artifacts():
    src = r"""
Cover glasses (high precision, $22 \times 22\,mm$ No. 1.5H $170\,\mum \pm 5\,\mum$ thickness) were heated to 37 掳C.
"""
    out = postprocess_markdown(src)
    assert r"\mum" not in out
    assert r"\mu\mathrm{m}" in out
    assert "37 °C" in out


def test_normalize_glued_microsecond_latex_units():
    src = r"""
Under each illumination, the integration time is $0.02\,\mus$, and the frame time is $20\,\mus$.
"""
    out = postprocess_markdown(src)
    assert r"\mus" not in out
    assert r"$0.02\,\mu\mathrm{s}$" in out
    assert r"$20\,\mu\mathrm{s}$" in out


def test_normalize_unit_wrapping_and_common_ocr_word_splits():
    src = r"""
**Figure 3.** Example at 0.5 μ W and scale bar 10 μ m. Line pro fi les on fl at- fi elded background at ( t 0 − t 4 ).
The system runs with $0.5\,\mu$W illumination and 150 $\mu$l sample volume.
This improves photon collection but sacri ﬁ ces contrast at low inci- dent powers.
"""
    out = postprocess_markdown(src)
    assert "$0.5\\,\\mu\\mathrm{W}$" in out
    assert "$10\\,\\mu\\mathrm{m}$" in out
    assert "$150\\,\\mu\\mathrm{l}$" in out
    assert "profiles" in out
    assert "flat-fielded" in out
    assert "(t0-t4)" in out or "(t0–t4)" in out
    assert "sacrifices contrast" in out
    assert "incident powers" in out


def test_unwrap_inline_math_line_that_is_actually_prose_fragment():
    src = """
displace-

$ments(see eq.(2)).$
"""
    out = postprocess_markdown(src)
    assert "$ments(see eq.(2)).$" not in out
    assert "ments (see eq.(2))." in out


def test_wrap_bare_latex_inline_comparisons_in_prose():
    src = r"""
FFDNet is inferior when the noise level is low (e.g., \sigma \leq 25), but outperforms DnCNN as the noise level increases (e.g., \sigma > 25).
The existing inline math $F(\cdot)$ should stay unchanged.
"""
    out = postprocess_markdown(src)
    assert r"$\sigma \leq 25$" in out
    assert r"$\sigma > 25$" in out
    assert r"$F(\cdot)$" in out


def test_fix_cdot_loss_ocr_and_stray_citation_dollar():
    src = r"""
where loss( cdot) denotes the loss function. loss( cdot) is used near $\hat{x}$.
The methods use variances $\sigma \in [30,50,75]. The BSD68 dataset comes from BSD68 [110]$ and Set12. The first images are $256 \times 256$.
"""
    out = postprocess_markdown(src)
    assert r"loss( cdot)" not in out
    assert out.count(r"$\operatorname{loss}(\cdot)$") == 2
    assert r"$\sigma \in [30,50,75]$." in out
    assert "[110]$" not in out
    assert "[110] and Set12" in out
    assert r"$256 \times 256$" in out


def test_caption_cleanup_keeps_repaired_ocr_words_as_separate_words():
    src = """
**Figure 1.** Interferometric ISM (iISM) principle. FM fl ip mirror, EF emission fi lter, FC fi ber coupler, PM SMF polarization-maintaining single-mode fi ber. Side-by- side comparison. g Line pro fi les in the three con fi gurations.
"""
    out = postprocess_markdown(src)
    assert "FM flip mirror" in out
    assert "emission filter" in out
    assert "FC fiber coupler" in out
    assert "single-mode fiber" in out
    assert "Side-by-side comparison" in out
    assert "Line profiles" in out
    assert "three configurations" in out


def test_unwrap_sentence_style_display_math_with_text_macros_and_units():
    src = r"""
$$
about 1.4 \text{ Airy units (AU, with } 1\,\text{AU} = 1.22\,\lambda/(2\,\text{NA})\text{), which} \\ \text{for our parameters (}\lambda = 445\,\text{nm},\ \text{NA} = 1.4,\ 1\,\text{AU} = 194
$$
"""
    out = postprocess_markdown(src)
    assert "$$" not in out
    assert "about 1.4 Airy units" in out
    assert "1 AU = 1.22" in out
    assert "445" in out


def test_restore_ocr_stripped_tagged_display_equation():
    src = "widehat{s} = underset{s in mathcal{S}}{ arg min} left { - log p(d|W widehat{s}) + pen( widetilde{s}) right }, tag{13}"

    out = postprocess_markdown(src)

    assert out.startswith("$$\n")
    assert r"\widehat{s}" in out
    assert r"\underset{s \in \mathcal{S}}{\arg \min}" in out
    assert r"\left\{" in out
    assert r"\right\}" in out
    assert r"\tag{13}" in out
    assert contains_bare_tagged_display_math(out) is False


def test_tagged_math_restoration_does_not_wrap_a_natural_language_sentence():
    src = "The estimator widehat{s} = underset{s in mathcal{S}}{ arg min} left { - log p(d|s) right }, tag{13}"

    out = postprocess_markdown(src)

    assert out == src
    assert not out.startswith("$$")
