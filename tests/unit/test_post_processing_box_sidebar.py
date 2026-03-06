from kb.converter.post_processing import postprocess_markdown


def test_postprocess_extracts_box_heading_into_sidebar_block():
    src = """# Title

Some intro line.

## Box 1 | The maths behind single-pixel imaging

We can consider the problem.

$$
S = PI
$$

## Applications

Body text.
"""

    out = postprocess_markdown(src)

    assert "## Box 1 | The maths behind single-pixel imaging" not in out
    assert "<!-- box:start id=1 -->" in out
    assert "**[Box 1 - The maths behind single-pixel imaging]**" in out
    assert "$$\nS = PI\n$$" in out
    assert "## Applications" in out


def test_postprocess_extracts_plain_box_line_into_sidebar_block():
    src = """Title

Box 2: Extra notes

line a
line b

## Next
"""

    out = postprocess_markdown(src)

    assert "Box 2: Extra notes" not in out
    assert "<!-- box:start id=2 -->" in out
    assert "**[Box 2 - Extra notes]**" in out
    assert "line a" in out and "line b" in out

