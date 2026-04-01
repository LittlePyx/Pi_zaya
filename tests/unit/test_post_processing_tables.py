from kb.converter.post_processing import postprocess_markdown


def _table_widths(md: str) -> list[int]:
    widths: list[int] = []
    for line in md.splitlines():
        if not line.strip().startswith("|"):
            continue
        widths.append(len(line.strip().strip("|").split("|")))
    return widths


def test_postprocess_markdown_normalizes_sparse_multilevel_table_headers():
    src = "\n".join(
        [
            "**Table 1.** Example",
            "",
            "| | | Sampling ratio |",
            "|---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
            "| | Strategy | 1% | 5% | 10% | 15% | 20% | 40% | 60% | 80% |",
            "| **PSNR (dB)** | circular | 11.00 | 12.45 | 13.17 | 13.64 | 14.25 | 15.45 | 19.42 | 27.24 |",
            "| | Hadamard | square | 10.93 | 12.39 | 13.04 | 13.49 | 13.84 | 15.09 | 16.68 | 20.94 |",
        ]
    )
    out = postprocess_markdown(src)
    widths = _table_widths(out)
    assert len(set(widths)) == 1
    assert "|  |  |  | Sampling ratio |" in out
    assert "| **PSNR (dB)** |  | circular | 11.00 | 12.45 |" in out


def test_postprocess_markdown_expands_multicolumn_cells():
    src = "\n".join(
        [
            "**Table 4.** Comparison between HSI and FSI",
            "",
            "| | HSI | \\multicolumn{2}{c}{FSI} |",
            "|---|---|---|---|",
            "| | | Original FSI | Binary FSI |",
            "| Perfect reconstruction | Yes | Yes | No |",
        ]
    )
    out = postprocess_markdown(src)
    widths = _table_widths(out)
    assert len(set(widths)) == 1
    assert "\\multicolumn" not in out
    assert "|  | HSI | FSI |  |" in out
