# 会话回答优化手工回归清单（Week2 D4）

更新时间：2026-03-06

## 0. 可执行脚本（推荐）

先启动后端服务后执行：

```bash
python tools/manual_regression/run_answer_manual_regression.py --base-url http://127.0.0.1:8000
```

说明：

1. 脚本会先执行自动检查（`/api/health`、`/api/generate/quality/summary`、可选生成流校验）。
2. 然后进入手工检查交互（`p/f/s`），最后自动输出报告到 `test_results/manual_regression/`。
3. 若仅跑自动检查，可加：

```bash
python tools/manual_regression/run_answer_manual_regression.py --non-interactive --skip-generation
```

## 1. 回答结构完整性

1. 在 `answer_contract_v1=on` 下提问 5 类问题（阅读/对比/idea/实验/排障）。
2. 期望每条回答包含：`结论 + 依据/证据 + 下一步`。
3. 对“依据/证据/限制/局限”多种标题别名，前端都应正确分块渲染。

## 2. 引用与弹窗

1. 点击回答中的引用编号，弹窗可打开且字段完整（标题、source、DOI/指标）。
2. DOI 存在时优先展示标题，不应退化成整条题录。
3. 回答中不出现内部标签（如 `[SID:xxxx]`、`[[CITE:...]]` 原始 token）。

## 3. 文献篮一致性

1. 在会话中加入 2 条文献后，执行：
   - 切换到文献管理页 -> 切回会话
   - 切换到其他会话 -> 再切回
   - 浏览器刷新
2. 期望文献篮条目不丢失、不降级、不串会话。
3. 连续往返切换 20 次，不应出现明显卡死或空白。

## 4. 流式体验与复制一致性

1. 回答流式输出时，段落显示不闪烁、不重复插入。
2. “复制文本 / Copy Markdown”与页面展示语义一致。
3. 结构化块中公式、代码、表格渲染正常。

## 5. 异常与降级

1. 构造“未命中知识库”问题，回答应明确提示并给可执行下一步建议。
2. 中断生成后，状态应变为 `canceled`，不出现脏状态残留。
3. 设置 `answer_contract_v1=off` 后，回答应退回旧样式但不影响可用性。

## 6. 验收记录模板

每条用例如下记录：

1. 用例 ID
2. 输入问题
3. 实际结果
4. 期望结果
5. 结论（Pass/Fail）
6. 备注（截图路径/复现步骤）
