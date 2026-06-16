from kb.converter.reference_markdown import (
    fix_references_format,
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


def test_references_format_keeps_standalone_numbered_url_entry_and_avoids_year_refs():
    page_text = "\n".join(
        [
            "References",
            "18.",
            "Hirose, Y. et al. 5.6 A 400x400-pixel vertical avalanche",
            "photodiodes image sensor. In 2019 IEEE International Solid-State Circuits Conference",
            "(ISSCC), p. 104-106 (2019).",
            "19.",
            "S.r.l., M. P. D. Micro Photon Devices. http://www.micro-photon-",
            "devices.com/.",
            "20. Sun, Q. et al. End-to-End Learned SPAD Camera. ACM Trans. Graph. 39, 1-14 (2020).",
            "48. Popescu, G. Large-scale phase retrieval. Light: Sci. Appl. 10,",
            "175 (2021).",
            "Acknowledgements",
            "This section must not be appended to the final reference.",
        ]
    )

    out = fix_references_format(normalize_references_page_text(page_text))

    assert "[18] Hirose, Y. et al." in out
    assert "[19] S.r.l., M. P. D. Micro Photon Devices. http://www.micro-photondevices.com/." in out
    assert "[20] Sun, Q. et al." in out
    assert "[48] Popescu, G. Large-scale phase retrieval. Light: Sci. Appl. 10, 175 (2021)." in out
    assert "[2019]" not in out
    assert "[175]" not in out
    assert "Acknowledgements" not in out


def test_references_format_preserves_author_year_references_without_fake_numbers():
    page_text = "\n".join(
        [
            "REFERENCES",
            "Kara-Ali Aliev, Artem Sevastopolsky, Maria Kolos, Dmitry Ulyanov, and Victor Lem-",
            "pitsky. 2020. Neural Point-Based Graphics. In Computer Vision - ECCV 2020. 696-712.",
            "Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-",
            "Brualla, and Pratul P Srinivasan. 2021. Mip-nerf: A multiscale representation.",
            "Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman.",
            "2022. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. CVPR.",
        ]
    )

    out = fix_references_format(normalize_references_page_text(page_text))
    ref_lines = [line.strip() for line in out.splitlines() if line.strip() and not line.startswith("#")]

    assert len(ref_lines) == 3
    assert ref_lines[0].startswith("Kara-Ali Aliev")
    assert "Victor Lempitsky. 2020. Neural Point-Based Graphics." in ref_lines[0]
    assert ref_lines[1].startswith("Jonathan T Barron")
    assert ref_lines[2].startswith("Jonathan T. Barron")
    assert "[2020]" not in out
    assert "[2021]" not in out
    assert "[2022]" not in out
