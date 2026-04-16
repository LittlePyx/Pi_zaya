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


def test_insert_missing_abstract_and_introduction_for_natcommun_like_frontmatter():
    src = """
# ARTICLE

## Imaging biological tissue with high-throughput single-pixel compressive holography

Daixuan Wu$^{1,2}$, Jiawei Luo$^{1,2}$, Yuecheng Shen$^{1,2}$ & Zhaohui Li$^{1,2}$

Single-pixel holography is capable of generating holographic images with rich spatial information by employing only a single-pixel detector. Thanks to the relatively low dark-noise production, high sensitivity, large bandwidth, and cheap price of single-pixel detectors in comparison to pixel-array detectors, this system becomes attractive for biophotonics and related applications.

Pixel-array detectors, such as CCD and CMOS cameras, were commonly used in the traditional imaging scheme. However, these detectors are only cost-effective and maintain good performance within a certain spectrum range. In contrast, single-pixel detectors have lower dark-noise production, higher sensitivity, faster response time, and a much cheaper price, motivating the broader introduction to the field.

## Results

Body text.
"""
    out = postprocess_markdown(src)
    assert "## Abstract" in out
    assert "## Introduction" in out
    assert out.index("## Abstract") < out.index("Single-pixel holography is capable")
    assert out.index("## Introduction") < out.index("Pixel-array detectors, such as CCD")
    assert out.index("## Introduction") < out.index("## Results")


def test_insert_missing_abstract_before_explicit_roman_introduction_for_arxiv_like_layout():
    src = """
# Quantum correlation light-field microscope with extreme depth of field

Yingwen Zhang, Duncan England, Antony Orth, Ebrahim Karimi, and Benjamin Sussman

Light-field microscopy is a 3D microscopy technique whereby volumetric information of a sample is gained in a single shot by simultaneously capturing both position and angular information of light emanating from a sample. In this work, we demonstrate a design that does not require the conventional depth-resolution trade-off.

### I. INTRODUCTION

Utilizing the properties of quantum entangled photons to enhance the performance of sensing and imaging techniques has been an active area of research in recent decades.
"""
    out = postprocess_markdown(src)
    assert "## Abstract" in out
    assert "### I. INTRODUCTION" in out
    assert "## Introduction" not in out
    assert out.index("## Abstract") < out.index("### I. INTRODUCTION")


def test_normalize_spaced_abstract_heading_from_elsevier_like_front_page():
    src = """
# Optics and Laser Technology

## Part-based image-loop network for single-pixel imaging

## A B S T R A C T

In this study, we proposed a self-supervised image-loop neural network with a part-based model for single-pixel imaging.

## 1. Introduction

Single-pixel imaging utilizes the second-order correlation of classical or quantum light to reconstruct one-dimensional intensity signals into two-dimensional images.
"""
    out = postprocess_markdown(src)
    assert "## A B S T R A C T" not in out
    assert "## Abstract" in out
    assert "## 1. Introduction" in out


def test_preserve_optica_like_implicit_abstract_with_copyright_tail():
    src = """
# Frequency-division-multiplexed single-pixel imaging with metamaterials

**CLAIRE M. WATTS,**$^{1}$ **CHRISTIAN C. NADELL,**$^{2}$ **JOHN MONTOYA,**$^{2,3}$ **SANJAY KRISHNA,**$^{3}$ **AND WILLIE J. PADILLA**$^{1,2,*}$

We propose and experimentally realize the concept of frequency-division-multiplexed single-pixel imaging. Our technique relies entirely on metamaterial spatial light modulators, the advent of which has permitted advanced modulation techniques difficult to achieve with alternative approaches. So far, implementations of single-pixel imaging have used a single encoding frequency, making them sensitive to narrowband noise. Here, we implement frequency-division methods to parallelize the single-pixel imaging process at 3.2 THz. Our technique enables a trade-off between signal-to-noise ratio and acquisition speed without altering detector integration time, thus realizing a key development due to the limitations imposed by slow thermal detectors in terahertz and far IR. In addition, our technique yields high image fidelity and marries communications concepts to single-pixel imaging, opening a new path forward for future imaging systems. © 2016 Optical Society of America

*OCIS codes:* (110.1758) Computational imaging; (110.6795) Terahertz imaging; (160.3918) Metamaterials; (060.4230) Multiplexing; (060.5060) Phase modulation.

## 1. INTRODUCTION

Imaging with a single pixel was first shown in 1976, when a hyperspectral imager was fashioned using a mechanically scanned coded aperture wheel.
"""
    out = postprocess_markdown(src)
    assert "## Abstract" in out
    assert "## 1. INTRODUCTION" in out
    assert "We propose and experimentally realize the concept of frequency-division-multiplexed single-pixel imaging." in out
    assert out.index("## Abstract") < out.index("We propose and experimentally realize")
    assert out.index("## Abstract") < out.index("## 1. INTRODUCTION")


def test_do_not_insert_introduction_for_nature_like_multi_paragraph_abstract_only():
    src = """
# Electrically driven lasing from a dual-cavity perovskite device

Chen Zou, Zhixiang Ren, Kangshuo Hui, Zixiang Wang, Yangning Fan, Yichen Yang, Bo Yuan, Baodan Zhao & Dawei Di

Solution-processed semiconductor lasers promise lightweight, wearable and scalable optoelectronic applications. Among the gain media for solution-processed lasers, metal halide perovskites stand out as an exceptional class because of their ability to achieve wavelength-adjustable, low-threshold lasing under optical pumping.

Metal halide perovskites are an emerging class of semiconductors combining remarkable optoelectronic properties with cost-effective solution processability. Apart from the rapid advances in solar cells and LEDs, the unique attributes of high gain coefficients, long carrier lifetimes and tunable emission wavelengths have made perovskites excellent optical gain media for lasing applications.

The success of conventional semiconductor lasers builds on the ability of electrically driving the lasing action, allowing them to be easily integrated with a range of optoelectronic device platforms. However, for halide perovskites, the realization of electrically driven lasing remains a great challenge because of the inability to achieve intense electrical injection into high-quality perovskite resonant cavities.

In this work, we demonstrate electrically driven lasing from a dual-cavity perovskite device, which integrates a low-threshold perovskite single-crystal microcavity sub-unit with a high-power microcavity PeLED sub-unit to form a vertically stacked multi-layer structure.

## Structure of the integrated dual-cavity perovskite laser

The structure of the integrated dual-cavity device is shown in Fig. 1a.
"""
    out = postprocess_markdown(src)
    assert "## Abstract" in out
    assert "## Introduction" not in out
    assert out.index("## Abstract") < out.index("Solution-processed semiconductor lasers promise")
    assert out.index("## Structure of the integrated dual-cavity perovskite laser") > out.index("In this work, we demonstrate electrically driven lasing")


def test_strip_jopt_like_reader_service_block_before_real_title_and_abstract():
    src = """
# 3D single-pixel video

To cite this article: Yiwei Zhang et al 2016 J. Opt. 18 035203

View the [article online](https://doi.org/10.1088/2040-8978/18/3/035203) for updates and enhancements.

## You may also like

- Design and modeling of an acoustically excited double-paddle scanner
Khaled M Ahmida and Luiz Otavio S Ferreira

- Graphene hydrate: theoretical prediction of a new insulating form of graphene
Wei L Wang and Efthimios Kaxiras

![Ampheia advertisement](./assets/page_1_fig_2.png)

## 3D single-pixel video

Yiwei Zhang, Matthew P Edgar, Baoqing Sun, Neal Radwell, Graham M Gibson and Miles J Padgett

## Abstract

Photometric stereo is an established three-dimensional imaging technique for estimating surface shape and reflectivity using multiple images.

## Introduction

Three-dimensional imaging is a heavily explored research field.
"""
    out = postprocess_markdown(src)
    assert "To cite this article:" not in out
    assert "View the [article online]" not in out
    assert "## You may also like" not in out
    assert "Ampheia advertisement" not in out
    assert out.count("# 3D single-pixel video") == 1
    assert "## Abstract" in out
    assert "## Introduction" in out


def test_strip_author_frontmatter_lines_and_insert_abstract_for_sciadv_like_first_page():
    src = """
# Adaptive foveated single-pixel imaging with dynamic supersampling

David B. Phillips, 1 * Ming-Jie Sun, 1,2 * Jonathan M. Taylor, 1 Matthew P. Edgar, 1

Stephen M. Barnett, 1 Graham M. Gibson, 1 Miles J. Padgett 1

In contrast to conventional multipixel cameras, single-pixel cameras capture images using a single detector that measures the correlations between the scene and a set of patterns. However, these systems typically exhibit low frame rates, because to fully sample a scene in this way requires at least the same number of correlation measurements as the number of pixels in the reconstructed image.

## RESULTS
"""
    out = postprocess_markdown(src)
    assert "David B. Phillips" not in out
    assert "Stephen M. Barnett" not in out
    assert "## Abstract" in out
    assert out.index("## Abstract") < out.index("In contrast to conventional multipixel cameras")
    assert out.index("## Abstract") < out.index("## RESULTS")
