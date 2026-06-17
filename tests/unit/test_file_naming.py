from kb.file_naming import sanitize_filename_component


def test_sanitize_filename_component_protects_windows_reserved_names():
    assert sanitize_filename_component("CON") == "CON-paper"
    assert sanitize_filename_component("LPT1") == "LPT1-paper"


def test_sanitize_filename_component_strips_illegal_path_chars():
    assert sanitize_filename_component("bad:name / paper?.pdf") == "bad-name - paper-.pdf"
