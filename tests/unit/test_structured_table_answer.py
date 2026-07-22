from __future__ import annotations

from kb.structured_table_answer import build_structured_table_extreme_answer


def _table_hit(*, direction: str = "up") -> dict:
    return {
        "text": (
            "Table 6. Image Denoising Results on SIDD. SIDD PSNR: "
            "MPRNet [37] = 39.71; Restormer [39] = 40.02; "
            "Baseline ours = 40.30; NAFNet ours = 40.30"
        ),
        "meta": {
            "source_path": r"C:\db\ECCV-2022-Simple Baselines for Image Restoration\ECCV-2022-Simple Baselines for Image Restoration.en.md",
            "structured_kind": "table_metric",
            "structured_evidence_locked": True,
            "table_number": 6,
            "table_metric": "PSNR",
            "table_metric_label": "SIDD PSNR",
            "table_metric_direction": direction,
        },
    }


def test_structured_table_extreme_answer_lists_all_tied_winners_in_user_language():
    answer = build_structured_table_extreme_answer(
        "ECCV-2022 Simple Baselines 论文的 SIDD 基准测试里，PSNR 最高的模型是谁？如果并列请全部列出。",
        [_table_hit()],
    )

    assert "表 6" in answer
    assert "SIDD PSNR" in answer
    assert "Baseline ours 和 NAFNet ours" in answer
    assert "40.30" in answer
    assert "并列" in answer


def test_structured_table_extreme_answer_accepts_raw_table_index_hit():
    hit = _table_hit()
    hit["meta"].pop("structured_evidence_locked")

    answer = build_structured_table_extreme_answer(
        "SIDD 的 PSNR 最高模型是谁？",
        [hit],
    )

    assert "Baseline ours 和 NAFNet ours" in answer


def test_structured_table_extreme_answer_uses_metric_direction_for_best_query():
    answer = build_structured_table_extreme_answer(
        "Which model has the best SIDD PSNR? Include ties.",
        [_table_hit(direction="up")],
    )

    assert "Baseline ours and NAFNet ours tie at the highest SIDD PSNR value of 40.30" in answer


def test_structured_table_extreme_answer_rejects_broad_explanation_questions():
    answer = build_structured_table_extreme_answer(
        "请解释这张表中的各项 SIDD PSNR 结果为什么不同。",
        [_table_hit()],
    )

    assert answer == ""


def test_structured_table_extreme_answer_rejects_metric_mismatch():
    answer = build_structured_table_extreme_answer(
        "这张表里 LPIPS 最高的模型是什么？",
        [_table_hit()],
    )

    assert answer == ""
