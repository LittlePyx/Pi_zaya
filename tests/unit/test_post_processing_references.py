from kb.converter.post_processing import postprocess_markdown


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
