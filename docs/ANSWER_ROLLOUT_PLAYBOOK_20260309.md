# 回答优化灰度发布与回滚预案（D5）

更新时间：2026-03-09  
状态：`ready_for_canary`

## 1. 输入证据

1. 历史全量评测（37 条）  
`test_results/answer_eval/20260306_160650/summary.json`
   - `minimum_ok_rate=1.0`
   - `evidence_ok_rate=1.0`
   - `error=0`
   - `avg_latency_ms=18404.61`
   - `p95_latency_ms=27005.27`

2. 当前版本快速评测（8 条，覆盖 6 类意图）  
`test_results/answer_eval/20260309_162427/summary.json`
   - `minimum_ok_rate=1.0`
   - `evidence_ok_rate=1.0`
   - `error=0`
   - `avg_latency_ms=50089.32`
   - `p95_latency_ms=58174.96`

3. 自动回归（健康检查 + 质量汇总接口 + 生成流）  
`test_results/manual_regression/answer_manual_regression_20260309_162419.md`
   - 自动检查全部 PASS

4. 运行时质量汇总（最近 41 条样本）  
`/api/generate/quality/summary?limit=50`
   - `failed_rate=0`
   - `structure_complete_rate=1.0`
   - `evidence_coverage_rate=1.0`
   - `next_steps_coverage_rate=1.0`
   - `minimum_ok_rate=1.0`

## 2. 结论

质量维度已满足发布条件，但时延相对 2026-03-06 基线有明显上升。  
发布结论：`go_with_guardrail`（可灰度，但必须带时延闸门）。

## 3. 灰度策略（建议 3 阶段）

1. 阶段 A（10% 流量，4-8 小时）
   - 开启：`answer_contract_v1=on`、`answer_depth_auto=on`
   - 观察：`minimum_ok_rate`、`evidence_ok_rate`、`error/canceled`、`p95_latency_ms`
2. 阶段 B（50% 流量，1 天）
   - 阶段 A 指标达标后推进
   - 重点观察高峰期 `p95/p99` 与失败分桶
3. 阶段 C（100% 流量）
   - 阶段 B 连续稳定后全量

## 4. 指标闸门

1. 质量闸门（必须同时满足）
   - `minimum_ok_rate >= 0.95`
   - `evidence_ok_rate >= 0.90`
   - `failed_rate <= 0.02`
2. 稳定性闸门
   - `error + canceled <= 0.02`
3. 时延闸门（相对基线 20260306_160650）
   - `p95_latency_ms <= baseline_p95 * 1.8`（即 `<= 48609ms`）
   - 若连续 30 分钟超阈值，停止升量

## 5. 回滚触发条件

满足任一条立即回滚：

1. `minimum_ok_rate < 0.93`（连续 15 分钟）
2. `evidence_ok_rate < 0.85`（连续 15 分钟）
3. `failed_rate > 0.05` 或 `error_rate > 0.03`
4. `p95_latency_ms > baseline_p95 * 2.0`（连续 15 分钟）

## 6. 回滚动作

1. 立刻关闭：
   - `answer_contract_v1=off`
2. 保留：
   - 质量探针与质量汇总面板（仅观测，不影响答复链路）
3. 现场处置：
   - 导出最近 50 条失败样本
   - 按 `intent/depth/fail_reasons` 分桶定位

## 7. 明日执行清单（可直接照做）

1. 先补一次人工回归（交互式）  
`python tools/manual_regression/run_answer_manual_regression.py --base-url http://127.0.0.1:8000`
2. 以 10% 灰度开启 `answer_contract_v1`
3. 每 30 分钟记录一次：
   - `/api/generate/quality/summary?limit=50`
4. 若触发回滚条件，按第 6 节执行

