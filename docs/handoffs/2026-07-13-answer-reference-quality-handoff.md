# Pi_zaya 回答与引用质量改进交接

更新时间：2026-07-13
仓库：`F:\research-papers\2026\Jan\else\kb_chat`
分支：`main`
当前 HEAD：`a55dcdd Speed up exact paper evidence answers`

## 新会话从这里开始

请先阅读本文件和仓库根目录的 `AGENTS.md`，然后检查 `git status` 与当前 diff。当前工作区包含一批尚未提交的改动，不要重置、覆盖或假定这些改动已经通过完整 CI。

当前最重要的任务不是继续增加功能，而是修复下面两个真实用户问题：

1. 保留 LLM 生成的高质量阅读路线，不要因为内部 `DOC_*` 标签或局部契约问题，把完整回答整体替换成机械的三篇文献清单。
2. 文献篮、引用详情和回答中的推荐文献必须严格对齐；回答只推荐 3 篇时，不能展示 6 张看似也被推荐的卡片。

## 用户目标与不可破坏项

用户希望把产品持续改进为更优秀的学术 AI agent，重点是：

- 回答应由 LLM 做真正的概括、比较和阅读引导，不能为了提速移除 LLM 概括。
- 首屏回答应尽快开始输出，但文献卡片的完整润色可以异步完成。
- 引用文章必须真正相关，System A / System B 触发正确，定位和跳转正确。
- 文献篮卡片不能丢失 DOI、被引次数、影响因子、分区等重要信息。
- 回答正文保持简洁，不把 trace/debug 信息暴露到正文。
- 前后端契约应清晰，回答、引用标记、引用详情、文献篮和导出结果不能错配。
- legacy Streamlit 已移除，不使用 `app.py`、`streamlit run` 或 8501 端口。

## 当前未提交改动

本轮有约 32 个已修改文件，约 2980 行新增、188 行删除，尚未 commit、push，也没有针对这批改动等待 GitHub CI。

主要改动如下。

### 1. LLM 延迟与失败降级

- `kb/llm.py`
  - SDK client 默认禁用隐式重试。
  - 完善流式主/备用 provider 路由和主流超时边界。
  - 非流式 `chat` 支持可选 `timeout_s`、`max_retries`。
- `kb/task_runtime.py`
  - 流式失败后先做一次有时间上限的非流式 LLM 重试，再使用 grounded fallback。
  - 默认重试时间由 `KB_GENERATION_LLM_RETRY_TIMEOUT_S=18` 控制。
  - 保留了 LLM 概括，没有把 LLM 生成从产品中移除。

### 2. 两阶段引用卡片

- 回答生成不再被昂贵的引用卡片 LLM 润色阻塞。
- 先返回权威 document list 构造的快速卡片，再在后台完成完整润色。
- 本地已有的 DOI、`citation_count`、`journal_if`、`quartile` 会写入引用卡片。
- 评测在契约要求完整引用状态时，会等待完整 refs，而不是只检查快速态。

### 3. 引用展示和数据一致性

- 引用详情会从 `ui_meta.citation_meta` 复制文献计量字段。
- 重复 source hit 中优先选取元数据最完整的一条。
- 改善 evidence Markdown 标题清理和引用片段可读性。
- 检测 summary 与 why 的重复；保留权威证据，移除无意义重复说明。
- `api/chat_render.py` 改善 authoritative pack 选择，并避免渲染过程修改输入 refs map。

### 4. System B 触发策略

- 泛化的“初学者阅读路线”不再强制触发 System B。
- System B 只用于明确询问来源、起源、上游依据等问题。
- `spi-roadmap-beginner` 评测夹具已按此调整。
- 上游来源类问题仍必须验证完整 System B trace、文章相关性和跳转目标。

### 5. 真实评测工具

- `tools/research_qa/run_research_qa_eval.py` 会在 refs ready、polish 或 shelf 契约要求完整态时等待卡片完成。
- 回答首 token、回答完成和引用完成耗时分别记录，避免把不同阶段混成一个指标。

### 6. 最近加入但仍需修正的对齐逻辑

- `kb/task_runtime.py` 新增 `_align_multi_paper_doc_list_contract_with_display_hits`。
- 目标是让回答显示命中与 document list 对齐。
- 当前实现会把 3 篇回答扩展成 6 篇卡片，说明选择边界仍然过宽，不能直接视为完成。

## 当前修改文件

开始工作前请重新执行 `git status --short`。交接时的修改文件为：

```text
api/chat_render.py
api/reference_doc_list.py
api/reference_intent.py
api/reference_ui.py
api/routers/references.py
kb/citation_plan.py
kb/generation_answer_finalize_runtime.py
kb/llm.py
kb/paper_guide_answer_post_runtime.py
kb/paper_guide_grounding_runtime.py
kb/paper_guide_prompting.py
kb/paper_guide_retrieval_runtime.py
kb/research_answer_plan.py
kb/task_runtime.py
tests/unit/test_chat_render_reference_notes.py
tests/unit/test_citation_plan.py
tests/unit/test_generation_answer_finalize_runtime.py
tests/unit/test_llm_multimodal_stream.py
tests/unit/test_paper_guide_answer_post_runtime.py
tests/unit/test_paper_guide_grounding_runtime.py
tests/unit/test_paper_guide_prompting.py
tests/unit/test_reference_doc_list.py
tests/unit/test_reference_intent.py
tests/unit/test_reference_ui_score_calibration.py
tests/unit/test_references_router_cache.py
tests/unit/test_refs_renderer_system_a_popover.py
tests/unit/test_research_qa_eval_runner.py
tests/unit/test_task_runtime_answer_contract.py
tests/unit/test_task_runtime_bg_task.py
tools/research_qa/run_research_qa_eval.py
ui/refs_renderer.py
web/src/testing/researchQaData.json
```

## 已完成验证

本轮多组定向后端测试已通过，曾得到以下结果：

- refs / document list / renderer：87 passed。
- LLM / chat render / document list：155 passed。
- research eval runner / LLM / chat render：113 passed。
- document list / router / renderer / chat render：186 passed。
- chat render / router / document list：170 passed。
- task runtime / generation finalize：218 passed，1 skipped。

最近执行的是：

```powershell
python -m py_compile kb/task_runtime.py
python -m pytest tests/unit/test_task_runtime_bg_task.py tests/unit/test_task_runtime_refs_display_merge.py tests/unit/test_generation_answer_finalize_runtime.py -q
```

结果为 `218 passed, 1 skipped`。

尚未完成：

- 本批改动的广泛后端回归测试。
- 前端 `lint`、`build` 和指定 E2E smoke。
- 浏览器里的真实用户操作审查。
- commit、push 和当前改动对应的 GitHub CI。

## 真实评测结果

### 失败运行：`20260713_153747`

路径：`test_results/research_qa_eval/20260713_153747`

- 流失败后只得到 fragment list fallback。
- 回答质量不合格。

### 内容较好的运行：`20260713_160124`

路径：`test_results/research_qa_eval/20260713_160124`
conversation id：`d8cb8e1070e0440a840b065601509922`

- LLM 生成了真正有阅读引导价值的三篇文献路线。
- 首段约 7.7 秒，回答完成约 19.5 秒。
- 当时 DOI 缺失、卡片重复和元数据问题导致评测失败。
- 这份回答可作为“应该保留的回答风格”回归样本。

### 自动评测通过但用户体验退化：`20260713_163944`

路径：`test_results/research_qa_eval/20260713_163944`
conversation id：`fd43e45de6bb4698a88a469433a6fd47`

- 自动 evaluator：PASS。
- 首段：约 7.31 秒。
- 回答完成：约 18.13 秒。
- 总耗时：约 73.47 秒，主要是评测工具等待完整 refs。
- 引用卡质量：5/5。
- 文献篮质量：6/6。
- DOI / export ready：6。
- System B：0，针对泛化阅读路线这是正确的。

但这一运行仍有两个严重问题：

1. 最终回答退化成模板化的“以下 3 篇文章”列表，只包含定位和依据，丢失了上一运行中 LLM 生成的“为什么先读、主要看什么、阅读顺序”等高价值内容。
2. 回答只显示 3 篇文章，`paper_guide_contracts.doc_list` 和文献篮却扩展为 6 篇，用户会误以为额外 3 篇也被回答正式推荐。

因此，不能因为这次 evaluator PASS 就提交。

## 问题定位

### A. LLM 回答被整体重建

重点检查 `kb/generation_answer_finalize_runtime.py` 中调用 `_format_multi_paper_list_answer_v2` 的分支。

当前逻辑在以下情况可能整体重建答案：

- `raw_answer_had_internal_doc_labels`
- `_multi_paper_answer_needs_contract_rebuild(...)`

最近一次运行很可能只是因为原始回答带内部 `DOC_*` 标签，就把完整 LLM 阅读路线替换为确定性模板。正确方向是：

- 如果回答内容完整，只清理内部标签并修复引用映射。
- 尽量保留 LLM 的章节、比较、阅读顺序和解释。
- 只有回答结构确实缺失或不可修复时，才使用整体 fallback formatter。

需要新增一个回归测试，以 `20260713_160124` 的丰富三段式回答为样本，断言：

- 输出仍保留“为什么先读 / 主要看什么 / 阅读顺序”一类内容。
- 输出不含内部 `DOC_*` 标签。
- 引用标记仍能映射到正确文章。

### B. 3 篇回答扩展出 6 张卡片

重点检查：

- `kb/task_runtime.py::_align_multi_paper_doc_list_contract_with_display_hits`
- `_merge_refs_display_docs_with_answer_hits`
- final answer 与 numeric citations 的解析顺序。

最新运行中 `answer_hits` 数量为 6，导致 contract 被扩展为 6。应改为：

- 最终回答已有数字引用时，以最终回答实际引用的文献为权威集合。
- 只保留明确推荐或引用的文献，不用未引用 seed hit 填满卡片。
- 文献顺序与回答顺序一致。
- 对 3 篇路线，回答、citation details、doc list、文献篮和导出都应严格为同 3 篇。

现有 `tests/unit/test_task_runtime_refs_display_merge.py` 中有测试期待“引用 1、3 后继续补 seed”。需要重新判断这个期望是否适用于 authoritative answer cards；很可能应修改为有明确引用时只返回被引用集合。

## 最新 6 张卡片的元数据

元数据恢复已经生效，不要在修复对齐时丢掉：

| 文献 | DOI | 被引 | IF / 分区 |
|---|---|---:|---|
| LPR | `10.1002/lpor.202401397` | 37 | 10.0 / Q1 |
| OE | `10.1364/oe.25.019619` | 531 | 3.3 / Q2 |
| Nature Photonics | `10.1038/s41566-018-0300-7` | 910 | 32.9 / Q1 |
| PILN | `10.1016/j.optlastec.2023.109917` | 19 | 5.0 / Q1 |
| Science Advances | `10.1126/sciadv.1601782` | 268 | 12.5 / Q1 |
| Optica | `10.1364/optica.3.000133` | 27 | 8.5 / Q1 |

修复后的三篇卡片仍应显示对应 DOI、被引次数、影响因子和分区。

## 建议执行顺序

1. 检查 dirty diff，重点阅读 finalizer、task runtime 和相关测试，确认没有覆盖现有用户改动。
2. 修复完整 LLM 回答被整体模板重建的问题，先写回归测试，再做最小实现修改。
3. 修复回答、citation details、doc list、文献篮的文章集合和顺序对齐。
4. 重跑一轮真实 `spi-roadmap-beginner`，人工阅读回答正文；不能只看 evaluator PASS。
5. 运行一个明确询问上游来源的真实案例，例如 `scinerf-admm-origin`，验证 System B 会触发、文章真正相关、定位和跳转正确。
6. 启动 React 前端，以真实用户方式操作聊天、引用 popover、文献篮、来源定位和导出。
7. 运行广泛后端测试和全部要求的前端检查。
8. 检查 diff 与工作区，确认没有临时日志或测试产物，再 commit、push，并等待 GitHub CI 通过。

## 浏览器真实审查清单

进行浏览器操作前，先阅读 browser skill：

`F:\ai_coding\plugins\cache\openai-bundled\browser\26.707.61608\skills\control-in-app-browser\SKILL.md`

需要实际检查：

- 回答首屏是否及时出现，流式输出是否自然。
- 回答是不是 LLM 的有效概括，而不是证据片段拼接或模板清单。
- 正文引用序号与 popover 文章一致。
- 引用 popover 和文献篮显示 DOI、被引、IF、分区。
- summary、why、引用语境没有重复或错位。
- 回答推荐 3 篇时，文献篮也只有这 3 篇且顺序一致。
- System A 定位能跳到正确文章和正确证据位置。
- 上游来源问题触发 System B，泛化路线问题不触发。
- System B 展示的上游文章真正支持当前论点，跳转目标正确。
- 导出内容与屏幕所见文章集合一致。

## 最终验证命令

根据 `AGENTS.md`，至少执行：

```powershell
cd web
npm run lint
npm run build
npm run test:e2e:smoke -- agent-trace-actions.spec.ts
```

后端应先运行受影响模块的定向测试，再运行覆盖面更广的测试。当前没有证据表明这批改动已通过完整 CI。

## 本地运行状态

交接时 FastAPI 曾运行在 `127.0.0.1:8000`，PID 为 `55812`；开始前请重新检查进程是否仍存在。前端 5173 当时未启动。

正确启动方式：

```powershell
.\run_new.ps1 -StopExisting
```

或分别启动：

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

```powershell
cd web
npm run dev -- --host 127.0.0.1 --port 5173
```

## Git 与 CI

- 当前分支：`main`。
- 当前 HEAD：`a55dcdd`。
- 本轮工作未 commit、未 push。
- 不要宣称当前 dirty diff 已通过 CI。
- 用户此前提供的 CI 链接属于更早状态：`https://github.com/LittlePyx/Pi_zaya/actions/runs/28842500149`。
- 完成上述修复和验证后，再提交并推送 `main`，随后等待新的 GitHub Actions 运行通过。

## 完成标准

只有同时满足以下条件，才可以把本轮工作标记为完成：

- LLM 阅读路线保留有效概括、比较和阅读建议。
- 不暴露内部 `DOC_*` 标签。
- 回答、引用详情、文献篮和导出文章集合严格一致。
- DOI、被引次数、影响因子、分区仍完整显示。
- System A / System B 触发、相关性、定位和跳转均经真实操作验证。
- 回答延迟合理，引用异步完成不会阻塞正文。
- 后端测试、前端 lint/build/E2E 和新一轮 GitHub CI 均通过。
