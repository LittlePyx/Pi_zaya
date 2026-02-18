from kb.converter.post_processing import postprocess_markdown


def test_insert_visible_caption_from_image_alt_when_missing():
    src = """
Paragraph above.
![Fig. 8. Experimental results from real-world objects under broadband illumination.](./assets/page_13_fig_1.png)
Paragraph below.
"""
    out = postprocess_markdown(src)
    assert "*Fig. 8. Experimental results from real-world objects under broadband illumination.*" in out
    assert "Paragraph below." in out


def test_caption_line_after_image_is_italicized():
    src = """
![Figure](./assets/page_4_fig_1.png)
Fig. 3. Reconstructing images with a spatially variant effective exposure time.
"""
    out = postprocess_markdown(src)
    assert "*Fig. 3. Reconstructing images with a spatially variant effective exposure time.*" in out
