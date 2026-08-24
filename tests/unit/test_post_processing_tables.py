from kb.converter.post_processing import postprocess_markdown
from kb.converter.tables import markdown_table_issue_spans


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


def test_postprocess_markdown_reattaches_detached_rows_with_missing_leading_cell():
    src = "\n".join(
        [
            "| Model | Setting | A | B |",
            "| --- | --- | --- | --- |",
            "| **Base** | Batch Size | 16 | 32 |",
            "|  |",
            "Epochs | 30 | 60 |",
            "|  | Learning Rate | 5E-04 | 4E-04 |",
        ]
    )

    out = postprocess_markdown(src)

    assert "|  | Epochs | 30 | 60 |" in out
    assert "\nEpochs |" not in out
    assert "|  |\n" not in out


def test_table_issue_spans_include_fragmented_header_and_following_body():
    src = "\n".join(
        [
            "<!-- kb_page: 3 -->",
            "| Method |",
            "Trainable Parameters | WikiSQL | MNLI-m |",
            "| Fine-Tune | 175B | 73.8 | 89.5 |",
            "| --- | --- | --- | --- |",
            "| LoRA | 4.7M | 73.4 | 91.7 |",
        ]
    )

    assert markdown_table_issue_spans(src) == [
        {
            "start": 1,
            "end": 6,
            "collapsed_row_count": 0,
            "ambiguous_break_row_count": 1,
        }
    ]


def test_table_issue_spans_include_blank_fragmented_header_cell():
    src = "\n".join(
        [
            "<!-- kb_page: 4 -->",
            "|  |",
            "of Trainable Parameters = 18M | | | |",
            "| Weight Type | Wq | Wk | Wv |",
            "| --- | --- | --- | --- |",
            "| Rank | 8 | 8 | 8 |",
        ]
    )

    assert markdown_table_issue_spans(src)[0]["start"] == 1
    assert markdown_table_issue_spans(src)[0]["end"] == 6


def test_table_issue_spans_include_singleton_across_page_marker():
    src = "\n".join(
        [
            "<!-- kb_page: 18 -->",
            "|  | mIoU at 1 point | 3 points |  |",
            "",
            "<!-- kb_page: 19 -->",
            "",
            "mIoU at 1 point | 3 points |",
            "| group | one point | three points |",
            "| --- | --- | --- |",
        ]
    )

    spans = markdown_table_issue_spans(src)

    assert any(item["start"] == 1 and item["end"] == 2 for item in spans)


def test_table_issue_spans_ignore_isolated_equation_pipe_fragment():
    src = "\n".join(["The objective is", "| y | X", "X", "$$", "sum_t log p", "$$"])

    assert markdown_table_issue_spans(src) == []


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


def test_postprocess_markdown_keeps_ambiguous_break_rows_unmodified():
    src = "\n".join(
        [
            "| Method | PSNR | SSIM |",
            "| --- | --- | --- |",
            "| A<br>B<br>C | 30.1<br>31.2 | .91<br>.92<br>.93 |",
        ]
    )

    out = postprocess_markdown(src)

    assert "A<br>B<br>C" in out
    assert "30.1<br>31.2" in out
    assert "A · B · C" not in out


def test_postprocess_markdown_flattens_non_numeric_multiline_cells():
    src = "\n".join(
        [
            "| Method | Notes | PSNR |",
            "| --- | --- | --- |",
            "| NAFNet<br>(ours) | high quality<br>low noise<br>fast inference | 40.30 |",
        ]
    )

    out = postprocess_markdown(src)

    assert "<br>" not in out
    assert "NAFNet · (ours)" in out
    assert "high quality · low noise · fast inference" in out
    assert "40.30" in out


def test_postprocess_markdown_keeps_legitimate_wide_tables():
    tables = [
        [
            "| Method | PSNR | SSIM | Dice | IoU | EPE | WER | SAM |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            *[f"| M{i} | {39 + i} | .9{i} | .8{i} | .7{i} | .6{i} | .5{i} | .4{i} |" for i in range(4)],
        ],
        [
            "| Method | Setting | 5 | 10 | 25 | 50 | 100 | 200 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            *[f"| N{i} | base | {39 + i} | .9{i} | {25 + i} | {50 + i} | {100 + i} | {200 + i} |" for i in range(4)],
        ],
        [
            "| Method | PSNR | SSIM | Size | Depth | Batch | Epoch | Time |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            *[f"| P{i} | {39 + i} | .9{i} | 256 | 8 | 16 | 100 | {20 + i} |" for i in range(4)],
        ],
    ]
    src = "\n\n".join("\n".join(table) for table in tables)

    out = postprocess_markdown(src)

    assert out.count("| Method |") == 3
    assert "| M3 | 42 | .93 |" in out
    assert "| N3 | base | 42 | .93 |" in out
    assert "| P3 | 42 | .93 |" in out


def test_postprocess_markdown_keeps_fragmented_table_with_unique_numbers():
    src = "\n".join(
        [
            "| Model | blocks | SIDD PSNR SSIM | GoPro PSNR SSIM | Latency-256 | Latency-720 |",
            "| --- | --- | --- | --- | --- | --- |",
            "| NAFNet | 9 | 39.78 0.959 | 31.79 0.951 | 11.8 | 154.7 |",
            "| NAFNet | 18 | 39.90 0.960 | 32.64 0.951 | 19.9 | 151.7 |",
            "| NAFNet | 36 | 39.96 0.960 | 32.85 0.959 | 39.1 | 177.1 |",
            "| NAFNet | 72 | 39.95 0.960 | 32.88 0.961 | 73.8 | 230.1 |",
            "",
            "| sigma | SIDD PSNR SSIM | GoPro PSNR SSIM |",
            "| --- | --- | --- |",
            "| Identity | 39.96 0.960 | 32.85 0.960 |",
            "| ReLU | 39.98 0.960 | 32.59 0.958 |",
            "| GELU | 39.97 0.960 | 32.72 0.959 |",
            "| Sigmoid | 39.99 0.960 | 32.50 0.958 |",
            "",
            "| Model | blocks | SIDD PSNR SSIM P | GoPro SNR SSIM | ate | ncy- | metric | Latenc |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            "| WideA | 9 | 39 | .96 | 31.79 | 0.951 | 101 | 201 |",
            "| WideB | 18 | 39 | .96 | 32.64 | 0.951 | 202 | 302 |",
            "| WideC | 36 | 39 | .96 | 32.85 | 0.959 | 303 | 403 |",
            "| WideD | 72 | 39 | .96 | 32.88 | 0.961 | 304 | 406 |",
        ]
    )

    out = postprocess_markdown(src)

    assert "ncy-" in out
    assert "WideD" in out
    assert "406" in out


def test_postprocess_markdown_drops_fragmented_aggregate_duplicate():
    src = "\n".join(
        [
            "| Model | blocks | SIDD PSNR SSIM | GoPro PSNR SSIM | Latency-256 | Latency-720 |",
            "| --- | --- | --- | --- | --- | --- |",
            "| NAFNet | 9 | 39.78 0.959 | 31.79 0.951 | 11.8 | 154.7 |",
            "|  | 18 | 39.90 0.960 | 32.64 0.951 | 19.9 | 151.7 |",
            "|  | 36 | 39.96 0.960 | 32.85 0.959 | 39.1 | 177.1 |",
            "|  | 72 | 39.95 0.960 | 32.88 0.961 | 73.8 | 230.1 |",
            "",
            "| sigma | SIDD PSNR SSIM | GoPro PSNR SSIM |",
            "| --- | --- | --- |",
            "| Identity (ours) | 39.96 0.960 | 32.85 0.960 |",
            "| ReLU | 39.98 0.960 | 32.59 0.958 |",
            "| GELU | 39.97 0.960 | 32.72 0.959 |",
            "| Sigmoid | 39.99 0.960 | 32.50 0.958 |",
            "| SiLU | 39.96 0.960 | 32.74 0.960 |",
            "",
            "|  | blocks | SIDD PSNR SSIM P | GoPro L SNR SSIM | ate | ncy- | 25 | 6 | Latenc |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "|  | 9 | 39.78 0.959 | 31.79 0.951 |  | 11.8 |  |  | 154 |",
            "|  | 18 | 39.90 0.960 | 32.64 0.951 |  | 19.9 |  |  | 151 |",
            "|  | 36 | 39.96 0.960 | 32.85 0.959 |  | 39.1 |  |  | 177 |",
            "|  | 72 | 39.95 0.960 | 32.88 0.961 |  | 73.8 |  |  | 230 |",
            "|  | variants |  | Table 5 variants | of | sigma in | Si | mpl | eGate |",
            "| patches | TLC |  | sigma | PS | SIDD NR S | SI | M | GoPro PSNR |",
            "| NAFNet | 3 3 3 |  |  |  |  |  |  |  |",
            "|  |  |  | Identity | 39 | .96 0 | .9 | 60 | 32.85 |",
            "|  |  |  | ReLU | 39 | .98 0 | .9 | 60 | 32.59 |",
            "|  |  |  | GELU | 39 | .97 0 | .9 | 60 | 32.72 |",
            "|  |  |  | Sigmoid | 39 | .99 0 | .9 | 60 | 32.50 |",
            "|  |  |  | SiLU | 39 | .96 0 | .9 | 60 | 32.74 |",
        ]
    )

    out = postprocess_markdown(src)

    assert "| ate | ncy- | 25 | 6 | Latenc |" not in out
    assert out.count("Identity (ours)") == 1
    assert "| NAFNet | 9 | 39.78 0.959 | 31.79 0.951 | 11.8 | 154.7 |" in out


def test_postprocess_markdown_drops_truncated_decimal_duplicate_table():
    src = "\n".join(
        [
            "**Table 3.** Comparison on two datasets.",
            "",
            "| Datasets | Indexes | AE | HSCNN | IS |",
            "| --- | --- | --- | --- | --- |",
            "| Harvard | PSNR | 29.20 | 27.60 | 29. |",
            "|  | SSIM | 0.912 | 0.900 | 0. |",
            "|  | ERGAS | 91.62 | 105.90 | 83. |",
            "| KAIST | PSNR | 26.03 | 21.06 | 29. |",
            "|  | SSIM | 0.920 | 0.814 | 0. |",
            "|  | ERGAS | 172.32 | 273.37 | 15. |",
            "",
            "Short discussion between the duplicate representations.",
            "",
            "| Datasets | Indexes | AE | HSCNN | ISTA | Ours |",
            "| --- | --- | --- | --- | --- | --- |",
            "| Harvard | PSNR | 29.20 | 27.60 | 29.87 | 31.14 |",
            "|  | SSIM | 0.912 | 0.900 | 0.913 | 0.932 |",
            "|  | ERGAS | 91.62 | 105.90 | 85.21 | 74.92 |",
            "| KAIST | PSNR | 26.03 | 21.06 | 28.57 | 34.87 |",
            "|  | SSIM | 0.920 | 0.814 | 0.909 | 0.962 |",
            "|  | ERGAS | 172.32 | 273.37 | 151.49 | 37.40 |",
        ]
    )

    out = postprocess_markdown(src)

    assert "| Datasets | Indexes | AE | HSCNN | IS |" not in out
    assert out.count("| Datasets | Indexes |") == 1
    assert "| KAIST | PSNR | 26.03 | 21.06 | 28.57 | 34.87 |" in out


def test_postprocess_markdown_prefers_complete_table_after_lossy_vision_duplicate():
    lossy_header = "|  |  |  |  | us. | Orb | it. | Avg. | Deg. |  | Clus. | O | rbit. |  |"
    src = "\n".join(
        [
            "<!-- kb_page: 17 -->",
            lossy_header,
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "| EDP- | EDP-GN<br>GNN (st | N<br>ep=120) | 0.053 0.1<br>0.586 0.2 | 44<br>53 | 0.0<br>0.7 | 26<br>05 | 0.074<br>0.515 | 0.052<br>0.141 |  | 0.093<br>0.114 | 0.<br>0. | 007<br>036 | 0.050 0.062<br>0.097 0.306 |",
            "|  | Ours |  | 0.004 0.1 | 04 | 0.00 | 1 | 0.036 | 0.019 |  | 0.047 | 0. | 005 | 0.024 0.030 |",
            "| Unif<br>Small<br>Unif<br>Large<br>Dec<br>No | Ta<br>orm<br>Noise<br>orm<br>Noise<br>ayed<br>ise<br>Figure | ble 5: C<br>6: Com | omparing FH<br>pare the gener | D<br>at | M with<br>ed segm | E<br>e | DP-GNN<br>ntation ma | with si<br>ps wit | m<br>h | ilar di<br>differ | ffus<br>ent n | ion step<br>oise sc | s.<br>hedule. |",
            "",
            "| Method | Community-small |  |  |  | Ego-small |  |  |  | Avg |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "|  | Deg. | Clus. | Orbit. | Avg. | Deg. | Clus. | Orbit. | Avg. |  |",
            "| EDP-GNN | 0.053 | 0.144 | 0.026 | 0.074 | 0.052 | 0.093 | 0.007 | 0.050 | 0.062 |",
            "| EDP-GNN (step=120) | 0.586 | 0.253 | 0.705 | 0.515 | 0.141 | 0.114 | 0.036 | 0.097 | 0.306 |",
            "| Ours | **0.004** | **0.104** | **0.001** | **0.036** | **0.019** | **0.047** | **0.005** | **0.024** | **0.030** |",
        ]
    )

    out = postprocess_markdown(src)

    assert lossy_header not in out
    assert "<br>" not in out
    assert out.count("| Method | Community-small |") == 1
    assert "| EDP-GNN (step=120) | 0.586 | 0.253 | 0.705 |" in out
