from kb.converter.post_processing import postprocess_markdown


def test_insert_visible_caption_from_image_alt_when_missing():
    src = """
Paragraph above.
![Fig. 8. Experimental results from real-world objects under broadband illumination.](./assets/page_13_fig_1.png)
Paragraph below.
"""
    out = postprocess_markdown(src)
    assert "**Figure 8.** Experimental results from real-world objects under broadband illumination." in out
    assert "Paragraph below." in out


def test_caption_line_after_image_is_italicized():
    src = """
![Figure](./assets/page_4_fig_1.png)
Fig. 3. Reconstructing images with a spatially variant effective exposure time.
"""
    out = postprocess_markdown(src)
    assert "**Figure 3.** Reconstructing images with a spatially variant effective exposure time." in out


def test_adjacent_duplicate_image_refs_are_collapsed():
    src = """
![Figure 5](./assets/page_12_fig_1.png)

![Figure 5](./assets/page_12_fig_1.png)
![Figure 5](./assets/page_12_fig_1.png)

**Figure 5.** Comparison results for Lena.
"""
    out = postprocess_markdown(src)
    assert out.count("![Figure 5](./assets/page_12_fig_1.png)") == 1


def test_caption_cleanup_strips_duplicate_bold_prefix_inside_tail():
    src = """
![Figure 1](./assets/page_4_fig_2.png)
**Figure 1.** ** Illustration of differential HSI.
"""
    out = postprocess_markdown(src)
    assert "**Figure 1.** Illustration of differential HSI." in out


def test_dedupe_nearby_repeated_caption_without_image_between():
    src = """
![Figure 5](./assets/page_12_fig_1.png)

**Figure 5.** Comparison results for Lena.


**Figure 5.** Comparison results for Lena.
"""
    out = postprocess_markdown(src)
    assert out.count("**Figure 5.** Comparison results for Lena.") == 1


def test_dedupe_nearby_repeated_caption_prefers_copy_nearest_matching_image():
    src = """
**Figure 16.** Comparison results for single-pixel microscopy.

Paragraph about setup.

![Figure 16](./assets/page_20_fig_1.png)

**Figure 16.** Comparison results for single-pixel microscopy.
"""
    out = postprocess_markdown(src)
    assert out.count("**Figure 16.** Comparison results for single-pixel microscopy.") == 1
    assert out.index("![Figure 16](./assets/page_20_fig_1.png)") < out.index("**Figure 16.** Comparison results for single-pixel microscopy.")


def test_caption_cleanup_strips_leading_pipe_from_tail():
    src = """
![Figure 2](./assets/page_3_fig_1.png)
**Figure 2.** | Experiment results of large-scale single-photon imaging.
"""
    out = postprocess_markdown(src)
    assert "**Figure 2.** Experiment results of large-scale single-photon imaging." in out


def test_dedupe_same_figure_number_when_short_caption_repeats_near_image():
    src = """
## Experiment results of large-scale single-photon imaging

![Figure 2](./assets/page_3_fig_1.png)

**Figure 2.** Experiment results of large-scale single-photon imaging. a Optical setup.
"""
    out = postprocess_markdown(src)
    assert "## Experiment results of large-scale single-photon imaging" not in out
    assert out.count("**Figure 2.**") == 1


def test_dedupe_same_figure_number_prefers_later_caption_with_continuation_block():
    src = """
![Figure 4](./assets/page_8_fig_1.png)
**Figure 4.** Performance of the high-throughput SPH with unstained tissue from mouse brains in high-resolution mode. garbled c鈥揺 text.
**Figure 4.** Performance of the high-throughput SPH with unstained tissue from mouse brains in high-resolution mode.
**a** Better continuation with math $\\Delta\\varphi = 1.795$.
"""
    out = postprocess_markdown(src)
    assert out.count("**Figure 4.**") == 1
    assert "garbled c鈥揺 text" not in out
    assert "Better continuation with math" in out


def test_normalize_plural_figure_reference_line_to_plain_text():
    src = """
**Figure s.** 16 and 17 show the comparison results for single-pixel microscopy.
"""
    out = postprocess_markdown(src)
    assert "**Figure s.**" not in out
    assert "Figures 16 and 17 show the comparison results for single-pixel microscopy." in out


def test_fill_empty_image_alt_from_following_caption():
    src = """
![](./assets/page_3_fig_1.png)

**Figure 2.** Experiment results of large-scale single-photon imaging.
"""
    out = postprocess_markdown(src)
    assert "![Figure 2](./assets/page_3_fig_1.png)" in out


def test_tighten_image_caption_spacing():
    src = """
![Figure 3](./assets/page_10_fig_1.png)




**Figure 3.** Comparison results for USAF 1951 test chart pattern reconstruction by HSI and FSI for different sampling ratios.
"""
    out = postprocess_markdown(src)
    assert "![Figure 3](./assets/page_10_fig_1.png)\n\n**Figure 3.**" in out


def test_move_image_up_to_matching_previous_caption_when_body_text_slips_between():
    src = """
**Figure 4.** Qualitative evaluations on the synthetic dataset.

High compression ratio We study the performance of our model under different compression ratios.

![Figure 4](./assets/page_7_fig_1.png)
"""
    out = postprocess_markdown(src)
    assert out.index("![Figure 4](./assets/page_7_fig_1.png)") < out.index("**Figure 4.** Qualitative evaluations on the synthetic dataset.")
    assert "High compression ratio We study the performance of our model under different compression ratios." in out


def test_do_not_rewrite_body_subfigure_reference_sentence_as_caption():
    src = """
Figure 1b, c compare the effects of linear and circular polarization on the illumination PSF.
"""
    out = postprocess_markdown(src)
    assert "**Figure 1b.**" not in out
    assert "Figure 1b, c compare the effects of linear and circular polarization on the illumination PSF." in out


def test_repair_malformed_captionized_body_subfigure_reference_sentence():
    src = """
**Figure 1b.** , c compare the effects of linear and circular polarization on the illumination PSF.
"""
    out = postprocess_markdown(src)
    assert "**Figure 1b.**" not in out
    assert "Figure 1b, c compare the effects of linear and circular polarization on the illumination PSF." in out


def test_repair_sentence_split_by_figure_block_moves_continuation_above_float_figure():
    src = """
Major organelles are readily distinguishable, despite the

![Figure](./assets/page_6_fig_1.png)

**Figure 3.** Example caption text.

absence of labels. Notably, individual structures exhibit positive and negative interference contrast.

Next paragraph starts here.
"""
    out = postprocess_markdown(src)
    assert "Major organelles are readily distinguishable, despite the absence of labels." in out
    assert out.index("Major organelles are readily distinguishable, despite the absence of labels.") < out.index("![Figure](./assets/page_6_fig_1.png)")


def test_repair_sentence_split_by_figure_block_with_orphan_prose_fragment_and_wrapped_paragraph():
    src = """
Major organelles are readily distinguishable, despite the

![Figure](./assets/page_6_fig_1.png)

**Figure 3.** Example caption text.

absence of labels, highlighting nanoscale displace-

$ments(see eq.(2)).$

To benchmark the performance against conventional confocal iSCAT, we compared the same

    square region of interest reconstructed with a closed pinhole.
"""
    out = postprocess_markdown(src)
    assert "despite the absence of labels, highlighting nanoscale displacements (see eq.(2))." in out
    assert "To benchmark the performance against conventional confocal iSCAT, we compared the same square region of interest reconstructed with a closed pinhole." in out
    assert "\neq.(2)\n" not in f"\n{out}\n"
    assert out.index("despite the absence of labels, highlighting nanoscale displacements (see eq.(2)).") < out.index("![Figure](./assets/page_6_fig_1.png)")
    assert out.index("To benchmark the performance against conventional confocal iSCAT, we compared the same square region of interest reconstructed with a closed pinhole.") > out.index("**Figure 3.** Example caption text.")


def test_caption_cleanup_removes_spaces_before_punctuation_without_touching_caption_content():
    src = """
**Figure 4.** Qualitative evaluations on the synthetic dataset. Top to bottom shows the results for different scenes, including Cozy2room , Tanabata , Factory and Vender .
"""
    out = postprocess_markdown(src)
    assert "Cozy2room, Tanabata, Factory and Vender." in out
    assert "Cozy2room ," not in out
    assert "Vender ." not in out


def test_merge_panel_continuation_paragraph_into_caption_block():
    src = """
![Figure 6](./assets/page_10_fig_1.png)

**Figure 6.** Illustration of the reported deep transformer network for high-fidelity large-scale single-photon imaging.

a The workflow and structure of the reported network. b The enhancement comparison between CNN-based U-net network and the reported transformer-based network.

### Shallow feature extraction
"""
    out = postprocess_markdown(src)
    assert "a The workflow and structure of the reported network. b The enhancement comparison" in out
    assert out.count("**Figure 6.**") == 1
    assert "\n\na The workflow and structure of the reported network." not in out


def test_lift_embedded_panel_clauses_from_body_into_following_caption():
    src = """
### Network structure

Compared with the conventional convolutional networks, the gated fusion transformer network maintains the a The workflow and structure of the reported network. b The enhancement comparison between CNN-based U-net network and the reported transformer-based network. Benefiting from the transformer structure, our network realizes spatially varying convolution that helps pay more attention to the regions of interest. following advantages: 1) The content-based interactions between image content and attention weights can be interpreted as spatially varying convolution.

![Figure 6](./assets/page_10_fig_1.png)

**Figure 6.** Illustration of the reported deep transformer network for high-fidelity large-scale single-photon imaging.
"""
    out = postprocess_markdown(src)
    assert "maintains the following advantages: 1) The content-based interactions" in out
    assert "following advantages: 1) The content-based interactions between image content and attention weights can be interpreted as spatially varying convolution. Benefiting from the transformer structure" in out
    assert "**Figure 6.** Illustration of the reported deep transformer network for high-fidelity large-scale single-photon imaging. a The workflow and structure of the reported network. b The enhancement comparison between CNN-based U-net network and the reported transformer-based network." in out
    assert "maintains the a The workflow" not in out
    assert "maintains the Benefiting from the transformer structure" not in out
    assert "reported transformer-based network. Benefiting from the transformer structure" not in out
