# Answer Optimization Eval Report（2026-03-06）

- Time: 2026-03-06 16:06:50
- Base URL: `http://127.0.0.1:8000`
- Dataset: `docs/answer_eval_dataset_v1.jsonl`
- Eval output: `test_results/answer_eval/20260306_160650`

## KPI

| Metric | Value |
|---|---:|
| Total cases | 37 |
| Done | 37 |
| Error | 0 |
| Canceled | 0 |
| Avg latency (ms) | 18404.61 |
| P95 latency (ms) | 27005.27 |
| Avg answer chars | 974.57 |
| Minimum-ok rate | 100.00% |
| Core section coverage avg | 100.00% |
| Evidence-required count | 37 |
| Evidence-ok rate | 100.00% |

## Decision

- Go/No-Go: `go`
- Reasons:
1. 质量指标达到阈值：`minimum_ok_rate >= 95%`、`evidence_ok_rate >= 90%`。
2. 异常率达标：`error == 0`、`canceled == 0`。
3. 覆盖场景完整：`reading/compare/idea/experiment/troubleshoot/writing` 六类问题均通过。

## Notes

1. 评测时已冻结回答偏好（`answer_contract_v1=on`、`answer_depth_auto=on`、`answer_mode_hint=''`），避免运行过程中被 UI 偏好改写污染结果。
2. 历史一次低分结果（`20260306_155059`）已确认是偏好漂移导致的评测污染，不作为版本结论依据。

## Next Actions

1. 进入 Week2 D4 手工回归（重点检查引用弹窗、文献篮、结构化可读性）。
2. 开始 Week2 D5：整理灰度复盘并给出是否全量开关 `answer_contract_v1` 的发布建议。
