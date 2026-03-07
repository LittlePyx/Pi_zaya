# 回答优化评测运行手册（A0）

更新时间：2026-03-06

## 1. 目的

用固定评测集批量跑 `/api/generate`，输出：

1. 质量指标：结构完整率、证据覆盖率、下一步覆盖率。
2. 时延指标：平均时延、P95 时延。
3. 异常指标：`error/canceled` 占比。

## 2. 输入文件

1. 评测集：`docs/answer_eval_dataset_v1.jsonl`
2. 执行脚本：`tools/answer_eval/run_eval.py`

## 3. 运行命令

```bash
python tools/answer_eval/run_eval.py --base-url http://127.0.0.1:8000
```

说明：默认会在评测期间“冻结回答偏好”（`answer_contract_v1=on`、`answer_depth_auto=on`、`answer_mode_hint=''`），并在结束后自动恢复原偏好，避免评测过程中被 UI 设置改动污染。

快速冒烟（前 8 条）：

```bash
python tools/answer_eval/run_eval.py --base-url http://127.0.0.1:8000 --limit 8
```

如果你要保留当前 UI 偏好不干预：

```bash
python tools/answer_eval/run_eval.py --base-url http://127.0.0.1:8000 --no-freeze-answer-prefs
```

## 4. 输出目录

默认输出到：`test_results/answer_eval/<timestamp>/`

包含：

1. `raw_results.jsonl`：逐条样本原始结果（含 `answer_quality`）。
2. `summary.json`：汇总指标。
3. `report.md`：可直接贴到周报/PR 的简版结论。

## 5. Go/No-Go 建议阈值

1. `minimum_ok_rate >= 95%`
2. `evidence_ok_rate >= 90%`（在 `evidence_required=true` 子集上）
3. `error == 0`

满足上面三条可判定 `go`，否则 `no-go` 并回看失败样本。
