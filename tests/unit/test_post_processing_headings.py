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


def test_drop_empty_duplicate_h1_heading_when_caption_repeats_it():
    src = """
# Paper Title

## Methods

### Network structure
Some paragraph.

# Workflow and structure of the reported deep transformer network

![Figure 6](./assets/page_10_fig_1.png)
**Figure 6.** Illustration of the reported deep transformer network. The workflow and structure of the reported deep transformer network is shown here.
"""
    out = postprocess_markdown(src)
    assert "# Workflow and structure of the reported deep transformer network" not in out


def test_do_not_treat_caption_like_phrase_as_common_structural_heading():
    src = """
# Paper Title

## Experiment results of large-scale single-photon imaging

![Figure 2](./assets/page_3_fig_1.png)
**Figure 2.** Experiment results of large-scale single-photon imaging. a Optical setup.

## Results
Body text.
"""
    out = postprocess_markdown(src)
    assert "## Experiment results of large-scale single-photon imaging" not in out
    assert "## Results" in out


def test_drop_standalone_journal_metadata_line():
    src = """
# Paper Title

Paragraph above.

Optics EXPRESS

![Figure 5](./assets/page_12_fig_1.png)
**Figure 5.** Comparison results for Lena.
"""
    out = postprocess_markdown(src)
    assert "Optics EXPRESS" not in out


def test_drop_affiliation_metadata_directly_below_abstract_heading():
    src = """
# Sequentially Designed Compressed Sensing

## Abstract

$^{1}$ Dept. of Electrical and Computer Engineering; University of Minnesota; Minneapolis, MN 55455 $^{2}$ Dept. of Electrical and Computer Engineering; Rice University; Houston, TX 77005

*Abstract* A sequential adaptive compressed sensing procedure for signal support recovery is proposed and analyzed.

*Index Terms* adaptive sensing, compressed sensing, support recovery
"""
    out = postprocess_markdown(src)
    assert "Dept. of Electrical and Computer Engineering" not in out
    assert "University of Minnesota" not in out
    assert "*Abstract* A sequential adaptive compressed sensing procedure" in out
    assert "*Index Terms* adaptive sensing" in out


def test_demote_formulaish_heading_fragment_after_equation():
    src = """
## Results

$$
\\operatorname{obj} \\ast (h_{det} \\cdot h_{ill})
$$

## I = obj ⊛ h_det h_ill
with h_det and h_ill denoting the detection and illumination amplitude PSFs.
"""
    out = postprocess_markdown(src)
    assert "## I = obj ⊛ h_det h_ill" not in out
    assert "\nI = obj ⊛ h_det h_ill\n" in f"\n{out}\n"
    assert "with h_det and h_ill denoting the detection and illumination amplitude PSFs." in out


def test_do_not_promote_numeric_table_rows_as_numbered_headings():
    src = """
# Paper Title

22.85 .4057 .4986 22.35 .7663 .3179 21.77 .4321 .6031
0.125 31.93 .9562 .0520 0.25 33.61 .9599 .0420
"""
    out = postprocess_markdown(src)
    assert "## 22.85 .4057 .4986" not in out
    assert "## 0.125 31.93 .9562 .0520" not in out
    assert "22.85 .4057 .4986 22.35 .7663 .3179 21.77 .4321 .6031" in out
    assert "0.125 31.93 .9562 .0520 0.25 33.61 .9599 .0420" in out


def test_do_not_promote_formula_fragments_as_numbered_headings():
    src = """
# Paper Title

1 T N )) , (4)
"""
    out = postprocess_markdown(src)
    assert "## 1 T N )) , (4)" not in out
    assert "\n1 T N )) , (4)\n" in f"\n{out}\n"


def test_drop_running_journal_header_line_with_page_marker():
    src = """
# Paper Title

Kuppers and Moerner Light: Science & Applications (2026) 15:129 Page 3 of 13

Body paragraph.
"""
    out = postprocess_markdown(src)
    assert "Light: Science & Applications" not in out
    assert "Body paragraph." in out


def test_promote_known_plain_subheading_network_training():
    src = """
# Paper Title

## Methods

Network training

We implemented the network with the Pytorch framework.
"""
    out = postprocess_markdown(src)
    assert "### Network training" in out
    assert "We implemented the network with the Pytorch framework." in out


def test_promote_plain_document_title_line_when_followed_by_section_headings():
    src = """
Interferometric Image Scanning Microscopy for label-free imaging at 120 nm lateral resolution inside live cells

## Abstract
Body text.

## Introduction
More body text.
"""
    out = postprocess_markdown(src)
    assert out.lstrip().startswith("# Interferometric Image Scanning Microscopy for label-free imaging at 120 nm lateral resolution inside live cells")
    assert "## Abstract" in out
    assert "## Introduction" in out


def test_do_not_promote_affiliation_like_leading_line_as_document_title():
    src = """
Department of Physics, University of Example, Sample City, Country

## Abstract
Body text.
    """
    out = postprocess_markdown(src)
    assert not out.lstrip().startswith("# Department of Physics")
    assert "# Abstract" in out or "## Abstract" in out


def test_rebalance_custom_headings_within_nested_structural_section():
    src = """
## Materials and Methods

### Data analysis

#### Confocal measurement
Body text.

### Contrast-to-noise ratio (CNR)
Body text.

    ### Resolution
    Body text.
"""
    out = postprocess_markdown(src)
    heading_levels = {}
    for line in out.splitlines():
        if line.startswith("#"):
            marks, title = line.split(" ", 1)
            heading_levels[title.strip()] = len(marks)
    assert "Data analysis" in heading_levels
    assert "Confocal measurement" in heading_levels
    assert "Contrast-to-noise ratio (CNR)" in heading_levels
    assert "Resolution" in heading_levels
    assert heading_levels["Confocal measurement"] == heading_levels["Data analysis"] + 1
    assert heading_levels["Contrast-to-noise ratio (CNR)"] == heading_levels["Data analysis"] + 1
    assert heading_levels["Resolution"] == heading_levels["Data analysis"] + 1
