import re

from kb.converter.post_processing import postprocess_markdown
from kb.converter.reference_markdown import normalize_references_page_text


def _refs_tail(md: str) -> str:
    lines = md.splitlines()
    ref_i = -1
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("## references"):
            ref_i = i
            break
    if ref_i < 0:
        return ""
    return "\n".join(lines[ref_i + 1 :]).strip()


def test_normalize_references_page_text_drops_body_before_references_heading():
    src = """
Body text from the final article page should not become reference entry 1.

REFERENCES

1. ALPHA, A. A real reference. Journal, 1950, 1, 1-2.
2. BETA, B. Another real reference. Journal, 1951, 2, 3-4.
"""
    out = normalize_references_page_text(src)
    assert out.startswith("# References")
    assert "Body text from the final article page" not in out
    assert "1. ALPHA" in out
    assert "2. BETA" in out


def test_references_unwrap_math_wrappers():
    src = """
# Paper Title

## References

$$
[141] D. Singh, M. Kaur, M. Y. Jabarulla, V. Kumar.
$$
[142] $S. S. Afzal$, W. Akbar, O. Rodriguez, Nat. Commun. 2022, 13, 5546.
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    assert "$" not in refs
    assert "[141]" in refs
    assert "[142]" in refs


def test_references_split_collapsed_items_and_strip_math():
    src = """
## References
[130] Q. Meng, W. Lai, Opt. Laser Eng. 2024, 181, 108257. [131] $D. A. B. Miller$, Science 2023, 379, 41.
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    lines = [ln.strip() for ln in refs.splitlines() if ln.strip()]
    assert any(ln.startswith("[130] ") for ln in lines)
    assert any(ln.startswith("[131] ") for ln in lines)
    assert "$" not in refs


def test_references_infer_heading_when_missing():
    src = """
# Main Body
Some normal paragraph.

[1] A. Author, Journal, 2020. [2] B. Author, Journal, 2021.
[3] C. Author, Journal, 2022.
[4] D. Author, Journal, 2023.
[5] E. Author, Journal, 2024.
[6] $F. Author$, Journal, 2025.
"""
    out = postprocess_markdown(src)
    assert "## References" in out
    refs = _refs_tail(out)
    assert refs
    assert "$" not in refs


def test_references_do_not_infer_bibliography_from_citation_dense_figure_caption():
    caption = (
        "Figure 9. Color SPI. a) First panel. Reproduced with permission.[184] "
        "Copyright 2022, The Optical Society. b) Second panel.[181] Copyright 2013, "
        "The Optical Society. c) Third panel.[31] Copyright 2024, Springer Nature. "
        "d) Fourth panel.[182] Copyright 2021, The Optical Society. e) Fifth "
        "panel.[183] Copyright 2018, The Optical Society."
    )
    refs = [
        f"[{idx}] Author {idx}. Complete reference {idx}. Journal of Tests 2024, {idx}, {1000 + idx}."
        for idx in range(1, 9)
    ]
    src = "\n".join(
        [
            "# Complete Paper",
            "",
            "<!-- kb_page: 14 -->",
            caption,
            "",
            "## 5.6. Image-Free Sensing",
            "Body text that must remain before the bibliography.",
            "",
            "<!-- kb_page: 16 -->",
            "## 6. Challenges and Outlooks",
            "The final body section also must remain before the bibliography.",
            "",
            "<!-- kb_page: 17 -->",
            *refs,
        ]
    )

    out = postprocess_markdown(src)

    assert out.count("## References") == 1
    assert out.index("Color SPI.") < out.index("Image-Free Sensing")
    assert out.index("Challenges and Outlooks") < out.index("## References")
    assert all(re.search(rf"(?m)^\[{idx}\]\s+", out) for idx in range(1, 9))


def test_references_infer_heading_for_bare_numbered_reference_tail():
    src = """
# Main Body
The final paragraph cites several works [1,2] before the list.

1. A. Lovelace, Example Journal 12, 34-39 (2024).
2. A. M. Turing and C. Shannon, Proceedings of Testing 3, 44-48 (2025).
3. G. Hopper, IEEE Computer 1, 2-8 (2026).
"""
    out = postprocess_markdown(src)
    assert "## References" in out
    refs = _refs_tail(out)
    assert refs
    ref_lines = [ln.strip() for ln in refs.splitlines() if ln.strip()]
    assert ref_lines[0].startswith("[1] A. Lovelace")
    assert any(ln.startswith("[2] A. M. Turing") for ln in ref_lines)
    assert any(ln.startswith("[3] G. Hopper") for ln in ref_lines)
    assert "1. A. Lovelace" not in refs


def test_references_trim_prose_tail_after_citation_terminus():
    src = """
## References
[8] L. Pan, Y. Shen, J. Qi, J. Shi, X. Feng, Opt. Express 2023 , 31 , 13943. resulting in limited utilization capability of the available information.
[9] J. T. Ye, C. Yu, W. Li, Z.-P. Li, H. Lu, R. Zhang, Appl. Phys. Lett. 2023 , 123 , 024005.
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    assert "[8] " in refs
    assert "13943." in refs
    assert "resulting in limited utilization capability" not in refs
    assert "[9] " in refs


def test_references_trim_footer_and_ack_noise():
    src = """
## References
[24] L. Y. Dou, D. Z. Cao, L. Gao, X. B. Song, Opt. Express 2020 , 28 , 37167. Acknowledgements K.S. and Y.B. contributed equally to this work. This work was supported by ... 2401397 (17 of 21) www.advancedsciencenews.com www.lpr-journal.org
[25] M. J. Sun, M. P. Edgar, G. M. Gibson, Nat. Commun. 2016 , 7 , 12010.
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    assert "[24] " in refs
    assert "37167." in refs
    assert "Acknowledgements" not in refs
    assert "advancedsciencenews.com" not in refs
    assert "[25] " in refs


def test_references_trim_generic_prose_tail():
    src = """
## References
[30] D. Wu, J. Luo, G. Huang, Y. Feng, X. Feng, Nat. Commun. 2021 , 12 , 4712. this paragraph explains why the model can improve performance in dynamic scenes and challenging environments.
[31] S. Author, A. Author, Opt. Lett. 2022 , 47 , 3363.
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    assert "[30] " in refs
    assert "4712." in refs
    assert "this paragraph explains why the model" not in refs
    assert "[31] " in refs


def test_references_keep_doi_tail():
    src = """
## References
[40] A. Author, B. Author, Laser Photonics Rev. 2024 , 9 , 2401101. doi:10.1002/lpor.202401101
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    assert "[40] " in refs
    assert "doi:10.1002/lpor.202401101" in refs


def test_references_trim_merged_tail_for_parenthesized_year_style():
    src = """
## References
[5] B. Wang, M. Y. Zheng, J. J. Han, X. Huang, Phys. Rev. Lett. 127(5), 053602 (2021) Z. P. Li, J. T. Ye, X. Huang, IEEE J. Sel. Top. Quantum Electron. 28, 3804210 (2022)
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    assert "[5] " in refs
    assert "053602 (2021)" in refs
    assert "Z. P. Li" not in refs


def test_references_pull_prelude_lines_before_references_heading():
    src = """
# Main Body
Some normal paragraph.

[1] [2] [3] [4] [1] A. Author, J. Test 2022, 1, 1. [2] B. Author, J. Test 2023, 2, 2.
## References
[25] C. Author, J. Test 2024, 3, 3.
"""
    out = postprocess_markdown(src)
    assert "## References" in out
    refs = _refs_tail(out)
    assert refs
    ref_lines = [ln.strip() for ln in refs.splitlines() if ln.strip()]
    assert ref_lines[0].startswith("[1] ")
    assert any(ln.startswith("[2] ") for ln in ref_lines)
    assert any(ln.startswith("[25] ") for ln in ref_lines)
    assert "[1] [2] [3] [4]" not in out


def test_references_pull_dense_block_across_blank_gap_before_heading():
    src = """
# Main Body
Some normal paragraph.

[1] A. Author, J. Test 2020, 1, 1. [2] B. Author, J. Test 2021, 2, 2. [3] C. Author, J. Test 2022, 3, 3.

## References

[4] D. Author, J. Test 2023, 4, 4.
[5] E. Author, J. Test 2024, 5, 5.
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    ref_lines = [ln.strip() for ln in refs.splitlines() if ln.strip()]
    assert ref_lines[0].startswith("[1] ")
    assert any(ln.startswith("[2] ") for ln in ref_lines)
    assert any(ln.startswith("[3] ") for ln in ref_lines)
    assert any(ln.startswith("[4] ") for ln in ref_lines)
    assert any(ln.startswith("[5] ") for ln in ref_lines)
    assert "[1] A. Author" not in out.split("## References", 1)[0]


def test_references_front_keeps_following_body_sections():
    src = """
# Paper Title

## References
[1] A. Author, J. Test 2001, 1, 1.
[2] B. Author, J. Test 2002, 2, 2.
[3] C. Author, J. Test 2003, 3, 3.
[4] D. Author, J. Test 2004, 4, 4.
[5] E. Author, J. Test 2005, 5, 5.
[6] F. Author, J. Test 2006, 6, 6.
[7] G. Author, J. Test 2007, 7, 7.
[8] H. Author, J. Test 2008, 8, 8.

## 2. System design
This section must remain after references formatting.
"""
    out = postprocess_markdown(src)
    assert "## References" in out
    assert "## 2. System design" in out
    assert "This section must remain after references formatting." in out


def test_references_drop_incomplete_visible_placeholders():
    src = """
## References
[1] P. Sen, B. Chen, G. Garg, ACM Trans. Graph. 24, 745-755 (2005).
4. J. R. (incomplete visible)
5. W. (incomplete visible)
[4] J. Hunt, T. Driscoll, A. Mrozack, Science 339, 310-313 (2013).
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    assert "incomplete visible" not in out.lower()
    assert "## 4. J. R." not in out
    assert "[4] J. Hunt" in refs


def test_references_keep_year_backref_lines_with_previous_entry_and_do_not_spawn_fake_year_refs():
    src = """
## References
[16] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization.
arXiv preprint arXiv:1412.6980,
2014. 5
[17] Moyang Li, Peng Wang, Lingzhe Zhao, Bangyan Liao, and Peidong Liu. USB-NeRF: Unrolling shutter bundle adjusted
neural radiance fields. In arXiv preprint arXiv:2310.02687,
2023. 4
[48] Lin Yen-Chen. Nerf-pytorch. https://github.com/
yenchenlin/nerf-pytorch/, 2020. 5
[50] Xin Yuan, David J Brady, and Aggelos K Katsaggelos. Snapshot compressive imaging: Theory, algorithms, and applications.
IEEE Signal Processing Magazine, 38(2):65-88,
2021. 1
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    assert "[16] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014." in refs
    assert "[17] Moyang Li, Peng Wang, Lingzhe Zhao, Bangyan Liao, and Peidong Liu. USB-NeRF: Unrolling shutter bundle adjusted neural radiance fields. In arXiv preprint arXiv:2310.02687, 2023." in refs
    assert "[48] Lin Yen-Chen. Nerf-pytorch. https://github.com/yenchenlin/nerf-pytorch/, 2020." in refs
    assert "[50] Xin Yuan, David J Brady, and Aggelos K Katsaggelos. Snapshot compressive imaging: Theory, algorithms, and applications. IEEE Signal Processing Magazine, 38(2):65-88, 2021." in refs
    assert "[2014]" not in refs
    assert "[2020]" not in refs
    assert "[2021]" not in refs
    assert "[2023]" not in refs


def test_references_preserve_author_year_style_without_fake_year_markers():
    src = """
# Paper Title

## References

Kara-Ali Aliev, Artem Sevastopolsky, Maria Kolos, Dmitry Ulyanov, and Victor Lem-
pitsky. 2020. Neural Point-Based Graphics. In Computer Vision - ECCV 2020. 696-712.
Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-
Brualla, and Pratul P Srinivasan. 2021. Mip-nerf: A multiscale representation.
Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman.
2022. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. CVPR.

<!-- kb_page: 14 -->

3D Gaussian Splatting for Real-Time Radiance Field Rendering
139:13
Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson. 2020. Synsin:
End-to-end view synthesis from a single image. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 7467-7477.

## A Details
Appendix text must remain outside the references.
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)

    assert refs
    assert "Victor Lempitsky. 2020. Neural Point-Based Graphics." in refs
    assert "Jonathan T Barron" in refs
    assert "<!-- kb_page: 14 -->" in refs
    assert "Olivia Wiles" in refs
    assert "[2020]" not in refs
    assert "[2021]" not in refs
    assert "[2022]" not in refs
    assert "## A Details" in out
    assert out.index("Olivia Wiles") < out.index("## A Details")


def test_references_keep_author_year_page_marker_on_own_line():
    src = """
## References

Angtian Wang, Peng Wang, Jian Sun, Adam Kortylewski, and Alan Yuille. 2023. VoGE: A Differentiable Volume Renderer. <!-- kb_page: 14 --> Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson. 2020. Synsin: End-to-end view synthesis from a single image.
"""
    out = postprocess_markdown(src)
    lines = [line.strip() for line in out.splitlines() if line.strip()]

    assert "<!-- kb_page: 14 -->" in lines
    marker_idx = lines.index("<!-- kb_page: 14 -->")
    assert lines[marker_idx - 1].startswith("Angtian Wang")
    assert lines[marker_idx + 1].startswith("Olivia Wiles")


def test_references_drop_standalone_page_number_lines_inside_references():
    src = """
## References
[54] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric.
In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586-595,
2018. 5
11
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    assert "[54] Richard Zhang" in refs
    assert "2018." in refs
    assert " 11" not in refs


def test_references_split_bare_numbered_following_item_after_period():
    src = """
## References
[20] Sun, Q. et al. End-to-End Learned, Optically Coded SuperResolution SPAD Camera. ACM Trans. Graph. 39, 1-14 (2020). 21. Bolduc, E., Agnew, M. & Leach, J. Video-rate denoising of low-lightlevel images acquired with a SPAD camera. In 2016 Photonics North (PN), p. 1-1 (2016).
[22] Chandramouli, P. et al. A bit too much? high speed imaging from sparse photon counts. In 2019 IEEE International Conference on Computational Photography (ICCP), p. 1-1 (2019).
"""
    out = postprocess_markdown(src)
    refs = _refs_tail(out)
    assert refs
    ref_lines = [ln.strip() for ln in refs.splitlines() if ln.strip()]
    assert any(ln.startswith("[20] Sun, Q. et al.") for ln in ref_lines)
    assert any(ln.startswith("[21] Bolduc, E., Agnew, M. & Leach, J.") for ln in ref_lines)
    assert any(ln.startswith("[22] Chandramouli, P. et al.") for ln in ref_lines)
    assert "(2020). 21." not in refs


def test_references_preserve_sequential_entries_and_page_anchors_through_full_postprocess():
    refs_page_9 = [
        f"[{idx}] Author {idx}, Complete reference {idx}. Journal of Tests 2024, {idx}, {1000 + idx}."
        for idx in range(1, 9)
    ]
    refs_page_10 = [
        f"[{idx}] Author {idx}, Complete reference {idx}. Journal of Tests 2025, {idx}, {1000 + idx}."
        for idx in range(9, 13)
    ]
    src = "\n".join(
        [
            "# Complete Paper",
            "",
            "## References",
            "",
            "<!-- kb_page: 9 -->",
            *refs_page_9,
            "<!-- kb_page: 10 -->",
            *refs_page_10,
        ]
    )

    out = postprocess_markdown(src)
    ref_lines = [line.strip() for line in _refs_tail(out).splitlines() if line.strip()]
    extracted_numbers = [
        int(match.group(1))
        for line in ref_lines
        if (match := re.match(r"^\[(\d+)\]\s+", line))
    ]

    assert extracted_numbers == list(range(1, 13))
    assert ref_lines.index("<!-- kb_page: 9 -->") < ref_lines.index("[1] Author 1, Complete reference 1. Journal of Tests 2024, 1, 1001.")
    assert ref_lines.index("<!-- kb_page: 10 -->") < ref_lines.index("[9] Author 9, Complete reference 9. Journal of Tests 2025, 9, 1009.")


def test_inferred_references_heading_keeps_following_supplementary_material():
    refs = [
        f"[{idx}] Author {idx}, Complete reference {idx}. Journal of Tests 2024, {idx}, {1000 + idx}."
        for idx in range(1, 9)
    ]
    src = "\n".join(
        [
            "# Complete Paper",
            "",
            "## Abstract",
            "This article has enough main-body text before its bibliography to infer the missing heading safely. " * 8,
            "",
            "<!-- kb_page: 9 -->",
            *refs[:4],
            "<!-- kb_page: 10 -->",
            *refs[4:],
            "",
            "## Supplementary Materials",
            "",
            "### Refocusing Operation",
            "",
            "$$",
            r"I(x,y,z) = \mathcal{F}^{-1}\{F(k_x,k_y)e^{ik_z z}\}. \tag{8}",
            "$$",
            "",
            "<!-- kb_page: 11 -->",
            "The physical operation shifts each image before wave propagation.",
            "",
            "<!-- kb_page: 12 -->",
            "![Figure 7](./assets/page_12_fig_1.png)",
            "",
            "Figure 7. Full supplementary dataset.",
        ]
    )

    out = postprocess_markdown(src)

    assert "## References" in out
    assert all(re.search(rf"(?m)^\[{idx}\]\s+", out) for idx in range(1, 9))
    assert re.search(r"(?m)^## Supplementary Materials$", out)
    assert not re.search(r"(?m)^### Supplementary Materials$", out)
    assert "### Refocusing Operation" in out
    assert r"\tag{8}" in out
    assert "<!-- kb_page: 11 -->" in out
    assert "<!-- kb_page: 12 -->" in out
    assert "![Figure 7](./assets/page_12_fig_1.png)" in out
    assert out.index("[8] Author 8") < out.index("## Supplementary Materials")


def test_inferred_references_keeps_supplementary_words_inside_a_reference_entry():
    src = "\n".join(
        [
            "# Complete Paper",
            "",
            "## Abstract",
            "This article has enough main-body text before its bibliography to infer the missing heading safely. " * 8,
            "",
            "[1] A. Author. First source. Journal of Tests 2020, 1, 10.",
            "[2] B. Author. Second source. Journal of Tests 2021, 2, 20.",
            "[3] C. Author.",
            "Supplementary material for the third source.",
            "Journal of Tests 2022, 3, 30.",
            "[4] D. Author. Fourth source. Journal of Tests 2023, 4, 40.",
        ]
    )

    out = postprocess_markdown(src)

    assert "## Supplementary material for the third source" not in out
    assert re.search(r"(?m)^\[3\]\s+C\. Author\. Supplementary material for the third source\.", out)
    assert re.search(r"(?m)^\[4\]\s+D\. Author\.", out)


def test_references_preserve_unheaded_author_biography_page_and_are_idempotent():
    src = "\n".join(
        [
            "# Complete Paper",
            "",
            "<!-- kb_page: 20 -->",
            "## References",
            "",
            "[235] A. Author. Complete reference 235. Journal of Tests 2024, 20, 2350.",
            "[236] B. Author. Complete reference 236. Journal of Tests 2024, 20, 2360.",
            "<!-- kb_page: 21 -->",
            "**Kai Song** received his B.S. degree in 2019 and M.S. degree in 2022.",
            "",
            "**Yaoxing Bian** received his Ph.D. degree in 2022. His research interests include random lasers.",
            "",
            "**Liantuan Xiao** received his Ph.D. degree in 2001. His research interests include single-photon imaging.",
        ]
    )

    out = postprocess_markdown(src)

    assert out.index("[236] B. Author") < out.index("<!-- kb_page: 21 -->")
    assert out.index("<!-- kb_page: 21 -->") < out.index("**Kai Song** received")
    assert out.index("<!-- kb_page: 21 -->") < out.index("## Author Biographies") < out.index("**Kai Song** received")
    assert out.count("## Author Biographies") == 1
    assert "**Yaoxing Bian** received his Ph.D. degree in 2022." in out
    assert "**Liantuan Xiao** received his Ph.D. degree in 2001." in out
    assert postprocess_markdown(out) == out


def test_references_preserve_explicit_author_biography_heading_without_degree_boilerplate():
    src = "\n".join(
        [
            "# Complete Paper",
            "",
            "<!-- kb_page: 20 -->",
            "## References",
            "",
            "[235] A. Author. Complete reference 235. Journal of Tests 2024, 20, 2350.",
            "[236] B. Author. Complete reference 236. Journal of Tests 2024, 20, 2360.",
            "<!-- kb_page: 21 -->",
            "## Author Biographies",
            "",
            "Kai Song is a professor working on computational imaging.",
        ]
    )

    out = postprocess_markdown(src)

    assert out.index("[236] B. Author") < out.index("<!-- kb_page: 21 -->")
    assert "Kai Song is a professor working on computational imaging." in out
    assert out.count("## Author Biographies") == 1
    assert postprocess_markdown(out) == out


def test_inferred_references_preserve_author_biography_page():
    refs = [
        f"[{idx}] Author {idx}. Complete reference {idx}. Journal of Tests 2024, {idx}, {1000 + idx}."
        for idx in range(1, 9)
    ]
    src = "\n".join(
        [
            "# Complete Paper",
            "",
            "## Abstract",
            "This paper contains enough body prose before the bibliography to make inference safe. " * 8,
            "<!-- kb_page: 20 -->",
            *refs,
            "<!-- kb_page: 21 -->",
            "**Kai Song** received his B.S. degree in 2019. His research interests include single-pixel imaging.",
        ]
    )

    out = postprocess_markdown(src)

    assert "## References" in out
    assert out.index("[8] Author 8") < out.index("<!-- kb_page: 21 -->")
    assert out.index("<!-- kb_page: 21 -->") < out.index("## Author Biographies") < out.index("**Kai Song** received")
    assert postprocess_markdown(out) == out
