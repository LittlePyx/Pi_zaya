from kb.converter.heuristics import _is_non_body_metadata_text, _looks_like_author_name_line


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


def test_detect_first_page_author_name_line_as_metadata():
    text = "David B. Phillips, 1 * Ming-Jie Sun, 1,2 * Jonathan M. Taylor, 1 Matthew P. Edgar, 1"
    assert _looks_like_author_name_line(
        text,
        page_index=0,
        y0=120.0,
        page_height=792.0,
    )
    assert _is_non_body_metadata_text(
        text,
        page_index=0,
        y0=120.0,
        y1=135.0,
        page_height=792.0,
        max_font_size=10.0,
        body_font_size=9.0,
        is_references_page=False,
    )


def test_detect_article_history_received_date_as_metadata():
    text = "Received 12 March 2023; Accepted 4 July 2023; Published online 15 August 2023"
    assert _is_non_body_metadata_text(
        text,
        page_index=0,
        y0=820,
        y1=845,
        page_height=900,
        max_font_size=8.0,
        body_font_size=10.0,
        is_references_page=False,
    )


def test_do_not_drop_first_page_body_paragraph_with_received_word_near_bottom():
    text = (
        "Single-photon avalanche diode (SPAD) array has received wide attention due to "
        "its excellent single-photon sensitivity 1-4. Such a single-photon imaging "
        "sensor has been widely applied in various fields such as fluorescence lifetime "
        "imaging 5, fluorescence fluctuation spectroscopy 6, time-of-flight imaging 7-9, "
        "quantum communication and computing 10, 11, and so on 12, 13. Compared with "
        "EMCCD and sCMOS cameras that also maintain high detection sensitivity, SPAD "
        "arrays acquire photon-level light signals at a low-noise level, and perform "
        "direct photon-digital conversion that can effectively eliminate readout noise "
        "and enhance readout speed 14."
    )
    assert not _is_non_body_metadata_text(
        text,
        page_index=0,
        y0=573.8,
        y1=678.5,
        page_height=686.0,
        max_font_size=8.3,
        body_font_size=9.0,
        is_references_page=False,
    )
