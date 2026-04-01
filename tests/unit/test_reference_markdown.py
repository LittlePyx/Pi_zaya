from kb.converter.reference_markdown import (
    format_references_block,
    normalize_references_page_text,
)


def test_format_references_block_merges_cross_page_continuation_and_dehyphenates():
    ref_lines = [
        (1, "[26] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view syn-"),
        (2, ""),
        (3, "thesis. Communications of the ACM, 65(1):99-106, 2021. 1, 2, 3, 5"),
        (4, "[27] Thomas Muller, Alex Evans. Instant neural graphics primitives. ACM Trans. Graph., 41(4):102:1-102:15, July 2022. 5"),
    ]

    out = format_references_block(ref_lines)

    assert len(out) == 2
    assert out[0].startswith("[26] Ben Mildenhall")
    assert "view synthesis. Communications of the ACM, 65(1):99-106, 2021." in out[0]
    assert not out[0].endswith("1, 2, 3, 5")
    assert out[1].startswith("[27] Thomas Muller")


def test_normalize_references_page_text_drops_heading_page_number_and_www_noise():
    page_text = (
        "References\n"
        "[1] A. Author. First paper. Journal, 2020. 3\n"
        "10\n"
        "ADVANCED SCIENCE NEWS www.advancedsciencenews.com\n"
        "[2] B. Author. Second paper. Conference, 2021. 4\n"
    )

    out = normalize_references_page_text(page_text)

    assert out.startswith("# References")
    assert "[1] A. Author. First paper. Journal, 2020. 3" in out
    assert "[2] B. Author. Second paper. Conference, 2021. 4" in out
    assert "\n10\n" not in f"\n{out}\n"
    assert "advancedsciencenews.com" not in out.lower()


def test_format_references_block_does_not_treat_year_backref_line_as_new_reference():
    ref_lines = [
        (1, "[50] Xin Yuan, David J Brady, and Aggelos K Katsaggelos. Snapshot compressive imaging: Theory, algorithms, and appli-"),
        (2, "cations. IEEE Signal Processing Magazine, 38(2):65-88,"),
        (3, "2021. 1"),
        (4, "[51] Next Author. Next paper. Journal, 2022. 3"),
    ]

    out = format_references_block(ref_lines)

    assert len(out) == 2
    assert out[0].startswith("[50] Xin Yuan, David J Brady")
    assert "38(2):65-88, 2021." in out[0]
    assert not out[0].endswith("2021. 1")
    assert out[1].startswith("[51] Next Author")
