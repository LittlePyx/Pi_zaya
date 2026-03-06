# P0 Execution Plan

更新时间：2026-03-06  
负责人：当前迭代主线  
状态：`in_progress`（WP0 completed, WP1 completed, WP2 in progress）

## 1. P0 目标

1. 封住转换链路图/公式映射错误，确保引用定位可验证。
2. 封住文献篮状态丢失/降级/串写问题（切会话、切页面、刷新）。
3. 封住 Chat 与 Library 往返切换卡顿/假死问题。
4. 建立可重复回归门禁，避免同类问题反复出现。

## 2. 工作包与排期（建议 6 天）

### WP0：基线与复现框架（D1，0.5-1 天）

状态：`completed`（2026-03-06）

任务：

1. 固化复现入口与环境检查。
2. 产出 `P0_REPRO.md` 与 `P0_CHECKLIST.md`。
3. 明确基线指标：
   - 文献篮丢失率（目标：0）
   - 切换假死率（目标：0）
   - 关键测试通过率（目标：100%）

交付：

1. [P0_REPRO.md](/f:/research-papers/2026/Jan/else/kb_chat/docs/P0_REPRO.md)
2. [P0_CHECKLIST.md](/f:/research-papers/2026/Jan/else/kb_chat/docs/P0_CHECKLIST.md)
3. [P0_BASELINE.md](/f:/research-papers/2026/Jan/else/kb_chat/docs/P0_BASELINE.md)

---

### WP1：转换链路映射正确性（D2-D3，1.5 天）

状态：`completed`（2026-03-06）

任务：

1. 审计 figure/equation 映射链路，补映射决策日志（debug）。
2. 对 NatPhoton 类样本做固定回归。
3. 防止“语义命中正确但资源图错误”的 remap 退化。

建议改动点：

1. `kb/converter/layout_analysis.py`
2. `kb/converter/heuristics.py`
3. `kb/converter/post_processing.py`
4. `tests/unit/test_pipeline_figure_remap.py`

验收：

1. 目标样本映射正确。
2. 相关单测全部通过。
3. 映射调试日志可按需开启（用于线上复盘）。
4. 扩展多论文风格回归（NatPhoton/CVPR/Nature/NatCommun/Optica）并通过 `tests/unit` 全量回归。

---

### WP2：文献篮状态一致性（D3-D4，2 天）

状态：`in_progress`（2026-03-06）

进展：

1. 已完成文献篮快照版本化（`version/revision/updatedAt`）与损坏 JSON 容错清理。
2. 已完成会话切换前强制 flush、按 key 维护 revision、去抖持久化，降低并发覆盖与切换抖动。
3. 已补 `Case-07`（revision 单调性）复现步骤，用于排查“用户可复现而本地难复现”场景。
4. 已补 `window.__kbDebug.runSwitchStress()` 与 `window.__kbSwitchPerf` 观测入口，开始推进 WP3 压测与卡顿定位。

任务：

1. 统一文献篮数据模型，禁止结构化信息被弱数据覆盖。
2. 统一 hydrate/flush 时机：
   - 切会话
   - 切页面
   - 刷新前后
3. 写保护：
   - revision/timestamp
   - 过期异步结果丢弃
4. localStorage 容错：
   - 损坏 JSON
   - 超配额
   - 不可用场景 fallback

建议改动点：

1. `web/src/components/chat/MessageList.tsx`
2. `web/src/stores/chatStore.ts`
3. `web/src/pages/ChatPage.tsx`
4. `web/src/pages/LibraryPage.tsx`

验收：

1. 文献篮跨切换不丢失、不降级、不串写。
2. 4 个核心场景回归通过（见 `P0_REPRO.md`）。

---

### WP3：切换卡顿/假死治理（D5，1.5 天）

任务：

1. 审计切换路径上的阻塞 `await` 和重复请求。
2. 补全异步过期保护（token/abort/revision）。
3. 减少同步重写（localStorage 节流、批处理）。
4. 做“连续切换压测”并记录结果。

建议改动点：

1. `web/src/stores/chatStore.ts`
2. `web/src/components/layout/AppSider.tsx`
3. `web/src/components/chat/MessageList.tsx`

验收：

1. 连续 50 次切换无假死。
2. 切换交互可用性无明显阻塞。

---

### WP4：回归门禁与发布准入（D6，1 天）

任务：

1. 固化单测 + 手工回归 + CI 的统一准入。
2. 把关键失败场景纳入文档化“必须项”。
3. 发布前跑 3 轮完整 checklist。

建议改动点：

1. `.github/workflows/ci.yml`
2. `tests/unit/test_task_runtime_source_binding.py`
3. `tests/unit/test_reference_ui_score_calibration.py`

验收：

1. CI 全绿。
2. P0 Checklist 三轮通过。

## 3. 依赖与风险

主要依赖：

1. 可复现的失败样本（特别是图映射错位样本）。
2. 稳定测试环境（Python/Node 版本一致）。

主要风险与应对：

1. 风险：并发回写覆盖新状态。  
   应对：统一 revision 写保护 + 过期丢弃。
2. 风险：跨平台路径差异导致 CI 失败。  
   应对：统一 source path 规范化与跨平台测试。
3. 风险：回归脚本不一致导致“本地过、线上挂”。  
   应对：所有提测基于同一 checklist。

## 4. DoD（完成定义）

1. `tests/unit` 全部通过。
2. CI 全绿且无新增不稳定项。
3. 文献篮核心场景连续回归通过。
4. 切换稳定性压测通过（50 次切换无卡死）。
