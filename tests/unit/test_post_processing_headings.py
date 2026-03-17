from kb.converter.post_processing import postprocess_markdown


def test_split_inline_heading_marker_to_new_line():
    src = """
Some trailing footer text 2401397 (20 of 21) ## ADVANCED SCIENCE NEWS www.advancedsciencenews.com
"""
    out = postprocess_markdown(src)
    assert "20 of 21) ## ADVANCED SCIENCE NEWS" not in out
    assert "\n## ADVANCED SCIENCE NEWS" in out or out.startswith("## ADVANCED SCIENCE NEWS")


def test_split_inline_all_caps_title_after_page_marker():
    src = """
[225] ... 2401397 (20 of 21) ADVANCED SCIENCE NEWS www.advancedsciencenews.com
"""
    out = postprocess_markdown(src)
    assert "(20 of 21) ADVANCED SCIENCE NEWS" not in out
    assert "(20 of 21)\nADVANCED SCIENCE NEWS" in out


def test_numbered_sections_are_demoted_under_document_title():
    src = """
# Advances and Challenges of Single-Pixel Imaging Based on Deep Learning
# 1. Introduction
## 2.1. Structures of Single-Pixel Imaging
"""
    out = postprocess_markdown(src)
    assert "# Advances and Challenges of Single-Pixel Imaging Based on Deep Learning" in out
    assert "## 1. Introduction" in out
    assert "### 2.1. Structures of Single-Pixel Imaging" in out


def test_promote_bare_numbered_headings_and_demote_caption_headings():
    src = """
# Paper Title

2.1. Structures of Single-Pixel Imaging
3.1.2. Training and Testing of Deep Neural Network
## Figure 6
"""
    out = postprocess_markdown(src)
    assert "### 2.1. Structures of Single-Pixel Imaging" in out
    assert "#### 3.1.2. Training and Testing of Deep Neural Network" in out
    assert "## Figure 6" not in out
    assert "\n**Figure 6.**\n" in f"\n{out}\n"


def test_drop_rogue_unnumbered_headings_inside_numbered_doc():
    src = """
# Paper Title
### 5.4. Optical Encryption
Body paragraph.
## SPI optical encryption
Figure 8. Caption text.
### 5.6. Image-Free Sensing
## Image-Free Sensing
More body text.
"""
    out = postprocess_markdown(src)
    assert "### 5.4. Optical Encryption" in out
    assert "### 5.6. Image-Free Sensing" in out
    assert "## SPI optical encryption" not in out
    assert "## Image-Free Sensing" not in out


def test_do_not_promote_reference_like_numbered_initials_as_headings():
    src = """
# Paper Title

4. J. Hunt, T. Driscoll, A. Mrozack, Science 339, 310-313 (2013).
5. W. L. Chan, K. Charan, Appl. Phys. Lett. 93, 121105 (2008).
"""
    out = postprocess_markdown(src)
    assert "## 4. J. Hunt" not in out
    assert "## 5. W. L. Chan" not in out


def test_demote_custom_h2_inside_structural_section():
    src = """
# Paper Title

## Methods

### Noise modeling of sensors
Body text.

## Network structure
Body text.

## Deep feature fusion
More body text.

## References
[1] A. Author, J. Test 2024, 1, 1.
"""
    out = postprocess_markdown(src)
    assert "## Methods" in out
    assert "### Network structure" in out
    assert "### Deep feature fusion" in out
    assert "\n## Network structure\n" not in f"\n{out}\n"
    assert "\n## Deep feature fusion\n" not in f"\n{out}\n"


def test_drop_empty_duplicate_heading_when_caption_repeats_it():
    src = """
# Paper Title

## Methods

### Network structure
Some paragraph.

## Workflow and structure of the reported deep transformer network

## Deep feature fusion
More body text.

**Figure 6.** Illustration of the reported deep transformer network. The workflow and structure of the reported deep transformer network is shown here.
![Figure 6](./assets/page_10_fig_1.png)
    """
    out = postprocess_markdown(src)
    assert "## Workflow and structure of the reported deep transformer network" not in out
    assert "### Workflow and structure of the reported deep transformer network" not in out
    assert "workflow and structure of the reported deep transformer network" in out.lower()
