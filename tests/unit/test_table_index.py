from __future__ import annotations

import json

from ingest import _incremental_chunks_are_usable
from kb.chunking import CHUNK_SCHEMA_VERSION, chunk_markdown
from kb.converter.structured_indices import STRUCTURED_INDEX_VERSION, rebuild_structured_indices_for_markdown
from kb.paper_guide_structured_index_runtime import load_paper_guide_table_index
from kb.retriever import BM25Retriever
from kb.store import compute_doc_id, write_doc_chunks


def _comparison_markdown() -> str:
    return "\n".join(
        [
            "# Results",
            "<!-- kb_page: 7 -->",
            "**Table 2. Quantitative comparison.**",
            "",
            "| Method | Airplants<br>PSNR↑ SSIM↑ LPIPS↓ | Hotdog<br>PSNR↑ SSIM↑ LPIPS↓ | Cozyroom<br>PSNR↑ SSIM↑ LPIPS↓ |",
            "|---|---|---|---|",
            "| GAP-TV [49]<br>PnP-FFDNet [51]<br>PnP-FastDVDNet [52]<br>EfficientSCI [38] | 22.85 .4057 .4986<br>27.79 .9117 .1817<br>28.18 .9092 .1757<br>30.13 .9425 .1129 | 22.35 .7663 .3179<br>29.00 .9765 .0511<br>29.93 .9728 .0522<br>30.75 .9568 .0461 | 21.77 .4321 .6031<br>28.98 .8910 .0984<br>30.19 .9132 .0793<br>30.75 .9327 .0476 |",
            "| ours | 30.69 .9335 .0728 | 31.35 .9878 .0310 | 33.23 .9492 .0445 |",
        ]
    )


def test_chunking_expands_br_merged_rows_and_builds_metric_comparison_chunks() -> None:
    chunks = chunk_markdown(_comparison_markdown(), source_path="paper.md", overlap=0)
    table_rows = [chunk for chunk in chunks if chunk["meta"].get("structured_kind") == "table_row"]
    table_metrics = [chunk for chunk in chunks if chunk["meta"].get("structured_kind") == "table_metric"]

    assert len(table_rows) == 5
    assert len(table_metrics) == 9
    gap_row = next(chunk for chunk in table_rows if chunk["meta"]["table_row_label"] == "GAP-TV [49]")
    assert "Airplants PSNR ↑ = 22.85" in gap_row["text"]
    assert "Cozyroom LPIPS ↓ = .6031" in gap_row["text"]
    assert gap_row["meta"]["page_start"] == 7
    assert gap_row["meta"]["table_number"] == 2

    psnr = next(chunk for chunk in table_metrics if chunk["meta"]["table_metric_label"] == "Airplants PSNR ↑")
    assert "higher is better" in psnr["text"]
    assert "GAP-TV [49] = 22.85" in psnr["text"]
    assert "ours = 30.69" in psnr["text"]


def test_table_query_prefers_complete_metric_series_over_raw_table() -> None:
    chunks = chunk_markdown(_comparison_markdown(), source_path="paper.md", overlap=0)
    chunks.extend(
        {
            "text": (
                "The ablation table discusses the best and highest PSNR results for SIDD and Airplants. "
                "This prose mentions comparison repeatedly but does not contain the complete metric series."
            ),
            "meta": {"source_path": "paper.md", "chunk_schema_version": CHUNK_SCHEMA_VERSION},
        }
        for _ in range(3)
    )
    hits = BM25Retriever(chunks).search("Which method has the highest Airplants PSNR?", top_k=3)

    assert hits
    assert hits[0]["meta"]["structured_kind"] == "table_metric"
    assert hits[0]["meta"]["table_metric_label"] == "Airplants PSNR ↑"
    assert "GAP-TV [49] = 22.85" in hits[0]["text"]
    assert "ours = 30.69" in hits[0]["text"]

    chinese_hits = BM25Retriever(chunks).search("哪种方法的 Airplants PSNR 最高？", top_k=3)
    assert chinese_hits[0]["meta"]["structured_kind"] == "table_metric"
    assert chinese_hits[0]["meta"]["table_metric_label"] == "Airplants PSNR ↑"


def test_multi_level_headers_keep_dataset_method_and_sampling_ratio_together() -> None:
    md = "\n".join(
        [
            "# Experiments",
            "| Dataset | Method | Sampling Ratio (SR) |  |  |  |",
            "| --- | --- | --- | --- | --- | --- |",
            "|  |  | 4% | 10% | 25% | 50% |",
            "| Set11 | ReconNet [22] | 20.93/0.5897 | 24.38/0.7301 | 28.44/0.8531 | 32.25/0.9177 |",
            "|  | ISTA-Net+ [69] | 21.32/0.6037 | 26.64/0.8087 | 32.59/0.9254 | 38.11/0.9707 |",
        ]
    )

    chunks = chunk_markdown(md, source_path="dual-scale.md", overlap=0)
    rows = [chunk for chunk in chunks if chunk["meta"].get("structured_kind") == "table_row"]
    metrics = [chunk for chunk in chunks if chunk["meta"].get("structured_kind") == "table_metric"]

    assert [row["meta"]["table_row_label"] for row in rows] == [
        "Set11 / ReconNet [22]",
        "Set11 / ISTA-Net+ [69]",
    ]
    ratio_4 = next(chunk for chunk in metrics if chunk["meta"]["table_metric_label"] == "Sampling Ratio (SR) 4%")
    assert "Set11 / ReconNet [22] = 20.93/0.5897" in ratio_4["text"]
    assert "Set11 / ISTA-Net+ [69] = 21.32/0.6037" in ratio_4["text"]


def test_multi_level_metric_headers_label_paired_values() -> None:
    md = "\n".join(
        [
            "# Ablation",
            "| Model | lr | LN | SIDD | GoPro |",
            "| --- | --- | --- | --- | --- |",
            "|  |  |  | PSNR SSIM | PSNR SSIM |",
            "| Baseline | 1e-3 | yes | 39.85 0.959 | 32.35 0.956 |",
        ]
    )

    chunks = chunk_markdown(md, source_path="restoration.md", overlap=0)
    row = next(chunk for chunk in chunks if chunk["meta"].get("structured_kind") == "table_row")

    assert row["meta"]["table_row_label"] == "Baseline / 1e-3 / yes"
    assert "SIDD PSNR = 39.85" in row["text"]
    assert "SIDD SSIM = 0.959" in row["text"]
    assert "GoPro PSNR = 32.35" in row["text"]


def test_exact_dataset_metric_label_beats_generic_metric_column() -> None:
    md = "\n".join(
        [
            "# Results",
            "| Model | PSNR |",
            "| --- | --- |",
            "| GenericNet | 40.05 |",
            "",
            "| Model | SIDD | GoPro |",
            "| --- | --- | --- |",
            "|  | PSNR SSIM | PSNR SSIM |",
            "| Baseline | 39.85 0.959 | 32.35 0.956 |",
            "| NAFNet | 39.96 0.960 | 32.85 0.967 |",
        ]
    )
    chunks = chunk_markdown(md, source_path="restoration.md", overlap=0)

    hit = BM25Retriever(chunks).search("Which model has the highest SIDD PSNR?", top_k=1)[0]

    assert hit["meta"]["structured_kind"] == "table_metric"
    assert hit["meta"]["table_metric_label"] == "SIDD PSNR"
    assert "Baseline = 39.85" in hit["text"]
    assert "NAFNet = 39.96" in hit["text"]


def test_transposed_metric_table_beats_ablation_for_dataset_best_query() -> None:
    md = "\n".join(
        [
            "# Results",
            "**Table 3.** The effect of the number of blocks.",
            "",
            "| Model | # of blocks | SIDD PSNR SSIM |",
            "| --- | --- | --- |",
            "| NAFNet | 36 | 39.96 0.960 |",
            "|  | 72 | 39.95 0.960 |",
            "",
            "**Table 6.** Image Denoising Results on SIDD [1]",
            "",
            "| Method | MPRNet | Restormer | Baseline ours | NAFNet ours |",
            "| --- | --- | --- | --- | --- |",
            "| PSNR | 39.71 | 40.02 | 40.30 | 40.30 |",
            "| SSIM | 0.958 | 0.960 | 0.962 | 0.962 |",
        ]
    )
    chunks = chunk_markdown(md, source_path="restoration.md", overlap=0)

    hit = BM25Retriever(chunks).search("Which model has the highest SIDD PSNR?", top_k=1)[0]

    assert hit["meta"]["structured_kind"] == "table_metric"
    assert hit["meta"]["table_number"] == 6
    assert hit["meta"]["table_metric_label"] == "SIDD PSNR"
    assert hit["meta"]["table_subject_kind"] == "method"
    assert "MPRNet = 39.71" in hit["text"]
    assert "Baseline ours = 40.30" in hit["text"]
    assert "NAFNet ours = 40.30" in hit["text"]

    concise_hit = BM25Retriever(chunks).search("What is the highest SIDD PSNR?", top_k=1)[0]
    assert concise_hit["meta"]["table_number"] == 6

    variant_hit = BM25Retriever(chunks).search("Which block count has the highest SIDD PSNR?", top_k=1)[0]
    assert variant_hit["meta"]["table_number"] == 3

    chinese_hit = BM25Retriever(chunks).search(
        "SIDD 基准测试里 PSNR 最高的模型是谁？如果并列请全部列出。",
        top_k=1,
    )[0]
    assert chinese_hit["meta"]["table_number"] == 6
    assert chinese_hit["meta"]["table_subject_kind"] == "method"

    english_ablation_hit = BM25Retriever(chunks).search(
        "Which ablation setting has the highest SIDD PSNR?",
        top_k=1,
    )[0]
    assert english_ablation_hit["meta"]["table_number"] == 3

    chinese_ablation_hit = BM25Retriever(chunks).search(
        "SIDD 基准消融实验里 PSNR 最高是多少？",
        top_k=1,
    )[0]
    assert chinese_ablation_hit["meta"]["table_number"] == 3


def test_uncaptioned_table_does_not_invent_authored_table_number() -> None:
    md = "\n".join(
        [
            "# Results",
            "| Method | PSNR |",
            "| --- | --- |",
            "| ours | 40.30 |",
        ]
    )

    metric = next(
        chunk
        for chunk in chunk_markdown(md, source_path="restoration.md", overlap=0)
        if chunk["meta"].get("structured_kind") == "table_metric"
    )

    assert metric["meta"]["table_number"] == 0
    assert metric["text"].startswith("Table data.")
    assert not metric["text"].startswith("Table 1.")


def test_comparison_words_do_not_make_unrelated_tables_searchable() -> None:
    chunks = chunk_markdown(_comparison_markdown(), source_path="paper.md", overlap=0)

    hits = BM25Retriever(chunks).search("Which quantum entanglement protocol has the highest fidelity?", top_k=3)

    assert hits == []


def test_structured_rebuild_writes_traceable_table_index_and_runtime_loader(tmp_path) -> None:
    md_path = tmp_path / "paper.en.md"
    assets_dir = tmp_path / "assets"
    md_path.write_text(_comparison_markdown(), encoding="utf-8")

    result = rebuild_structured_indices_for_markdown(md_path, assets_dir=assets_dir)
    payload = json.loads((assets_dir / "table_index.json").read_text(encoding="utf-8"))
    loaded = load_paper_guide_table_index(md_path)

    assert result["table_index"]["table_count"] == 1
    assert payload["version"] == STRUCTURED_INDEX_VERSION
    assert payload["row_count"] == 5
    assert loaded[0]["table_number"] == 2
    assert loaded[0]["block_id"]
    assert loaded[0]["anchor_id"]
    assert loaded[0]["page_start"] == 7
    assert loaded[0]["rows"][-1]["row_label"] == "ours"

    chunks = chunk_markdown(_comparison_markdown(), source_path=str(md_path), overlap=0)
    table_row = next(chunk for chunk in chunks if chunk["meta"].get("structured_kind") == "table_row")
    assert table_row["meta"]["block_id"] == loaded[0]["block_id"]


def test_incremental_ingest_rebuilds_chunks_when_schema_changes(tmp_path) -> None:
    db_dir = tmp_path / "db"
    md_path = tmp_path / "paper.md"
    md_path.write_text("# Paper\n\nStable text.", encoding="utf-8")
    doc_id = compute_doc_id(md_path)
    current_chunk = {
        "text": "Stable text.",
        "meta": {"source_path": str(md_path), "chunk_schema_version": CHUNK_SCHEMA_VERSION},
    }
    write_doc_chunks(db_dir, doc_id, [current_chunk])

    assert _incremental_chunks_are_usable(
        db_dir,
        doc_id,
        {"num_chunks": 1, "chunk_schema_version": CHUNK_SCHEMA_VERSION - 1},
    ) is False
    assert _incremental_chunks_are_usable(
        db_dir,
        doc_id,
        {"num_chunks": 1, "chunk_schema_version": CHUNK_SCHEMA_VERSION},
    ) is True
