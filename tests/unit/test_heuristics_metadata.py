from kb.converter.heuristics import _is_non_body_metadata_text


def test_detect_affiliation_contact_block_as_metadata():
    text = (
        "K. Song, Y. Bian, D. Wang. College of Physics and Optoelectronics Engineering, "
        "Taiyuan University of Technology, Taiyuan 030024, China. "
        "E-mail: bianyaoxing@tyut.edu.cn; xlt@sxu.edu.cn"
    )
    assert _is_non_body_metadata_text(
        text,
        page_index=0,
        y0=460,
        y1=620,
        page_height=900,
        max_font_size=8.5,
        body_font_size=11.0,
        is_references_page=False,
    )


def test_detect_footer_boilerplate_as_metadata():
    text = "Laser Photonics Rev. 2025, 19, 2401397 (1 of 21) www.lpr-journal.org © 2024 Wiley-VCH GmbH"
    assert _is_non_body_metadata_text(
        text,
        page_index=2,
        y0=845,
        y1=888,
        page_height=900,
        max_font_size=8.0,
        body_font_size=11.0,
        is_references_page=False,
    )


def test_do_not_drop_regular_body_sentence():
    text = (
        "In this work, we evaluate the method on a university campus dataset and show "
        "that reconstruction accuracy improves under low-light conditions."
    )
    assert not _is_non_body_metadata_text(
        text,
        page_index=4,
        y0=260,
        y1=320,
        page_height=900,
        max_font_size=11.0,
        body_font_size=11.0,
        is_references_page=False,
    )


def test_keep_reference_entries_on_references_page():
    text = "[88] A. Author, B. Author, Journal Name 2023, 31, 13943. doi:10.1002/example.123456"
    assert not _is_non_body_metadata_text(
        text,
        page_index=10,
        y0=420,
        y1=445,
        page_height=900,
        max_font_size=10.5,
        body_font_size=10.5,
        is_references_page=True,
    )
