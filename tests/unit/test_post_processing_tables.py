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


def test_postprocess_markdown_expands_aligned_html_break_rows():
    src = "\n".join(
        [
            "| Method | Airplants<br>PSNR↑ SSIM↑ LPIPS↓ | Hotdog<br>PSNR↑ SSIM↑ LPIPS↓ |",
            "| --- | --- | --- |",
            "| GAP-TV<br>PnP-FFDNet<br>EfficientSCI | 22.85 .4057 .4986<br>27.79 .9117 .1817<br>30.13 .9425 .1129 | 22.35 .7663 .3179<br>29.00 .9765 .0511<br>30.75 .9568 .0461 |",
            "| ours | 30.69 .9335 .0728 | 31.35 .9878 .0310 |",
        ]
    )

    out = postprocess_markdown(src)

    assert "<br>" not in out
    assert "| Method | Airplants |  |  | Hotdog |  |  |" in out
    assert "|  | PSNR↑ | SSIM↑ | LPIPS↓ | PSNR↑ | SSIM↑ | LPIPS↓ |" in out
    assert "| GAP-TV | 22.85 | .4057 | .4986 | 22.35 | .7663 | .3179 |" in out
    assert "| PnP-FFDNet | 27.79 | .9117 | .1817 | 29.00 | .9765 | .0511 |" in out
    assert "| EfficientSCI | 30.13 | .9425 | .1129 | 30.75 | .9568 | .0461 |" in out


def test_postprocess_markdown_prefers_nearby_structured_duplicate_table():
    src = "\n".join(
        [
            "| Method | Airplants<br>PSNR↑SSIM↑LPIPS↓ | Hotdog<br>PSNR↑SSIM↑LPIPS↓ |",
            "| --- | --- | --- |",
            "| GAP-TV<br>PnP-FFDNet | 22.85 .4057 .4986<br>27.79 .9117 .1817 | 22.35 .7663 .3179<br>29.00 .9765 .0511 |",
            "| ours | 30.69 .9335 .0728 | 31.35 .9878 .0310 |",
            "",
            "<!-- kb_page: 6 -->",
            "",
            "**Table 1.** Quantitative comparison.",
            "",
            "| Method | Airplants |  |  | Hotdog |  |  |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            "|  | PSNR↑ | SSIM↑ | LPIPS↓ | PSNR↑ | SSIM↑ | LPIPS↓ |",
            "| GAP-TV | 22.85 | .4057 | .4986 | 22.35 | .7663 | .3179 |",
            "| PnP-FFDNet | 27.79 | .9117 | .1817 | 29.00 | .9765 | .0511 |",
            "| ours | 30.69 | .9335 | .0728 | 31.35 | .9878 | .0310 |",
        ]
    )

    out = postprocess_markdown(src)

    assert out.count("Airplants") == 1
    assert out.count("GAP-TV") == 1
    assert "| GAP-TV | 22.85 | .4057 | .4986 | 22.35 | .7663 | .3179 |" in out
    assert "**Table 1.** Quantitative comparison." in out


def test_postprocess_markdown_keeps_nearby_tables_with_different_values():
    src = "\n".join(
        [
            "| Method | Airplants | Hotdog |",
            "| --- | --- | --- |",
            "| GAP-TV | 22.85 | 22.35 |",
            "| ours | 30.69 | 31.35 |",
            "",
            "**Table 2.** Novel-view comparison.",
            "",
            "| Method | Airplants | Hotdog |",
            "| --- | --- | --- |",
            "| NeRF+GAP-TV | 23.72 | 23.80 |",
            "| ours | 30.61 | 30.59 |",
        ]
    )

    out = postprocess_markdown(src)

    assert out.count("Airplants") == 2
    assert "22.85" in out
    assert "23.72" in out
