# Reference Locate Handoff 2026-04-24

## 1. 交接范围

本次交接聚焦于真实 UI 链路下的 4 个问题：

1. 参考定位卡片出现过晚
2. 回答与参考定位卡片引用的文献不一致
3. 部分问题回答本身还可以，但卡片为空、错排或定位错误
4. 回答后处理会删坏部分有效信息，留下格式残缺

用户本轮明确关注的评估维度包括：

- 命中的文献是否准确
- 参考定位卡片是否和回答中的文献对应
- 卡片内容是不是高质量总结，还是泛化空话
- 速度是否过慢，慢在后端还是前端展示
- 回答质量是否稳定
- 后处理是否删掉了重要信息

## 2. 本轮已完成的修复

### 2.1 已修复的 UI 阻塞

已完成以下 3 个前端修复：

1. `web/src/components/refs/RefsPanel.tsx`
   - 当 refs 仍处于 `pending`，但后端已经返回可用 `hits` 时，不再只显示“正在筛选”，而是直接展示临时卡片
   - 同时显示“参考定位正在精修中”的提示

2. `web/src/stores/chatStore.ts`
   - refs 轮询窗口从约 1 分钟延长为更稳妥的渐进轮询
   - 降低“长任务还没完成，前端先停止拉取”的概率

3. `web/src/components/chat/MessageList.tsx`
   - 新增“最后一条 user 消息后，assistant 仍在流式生成时，也允许先渲染 refs 行”
   - 这是本轮发现的第二个 UI 阻塞点，修复后 refs 不必再等 assistant 消息正式落库

### 2.2 已补充的回归测试

新增或补充了以下回归覆盖：

- `web/tests/e2e/refs-panel-regression.spec.ts`
  - `refs panel renders provisional cards while refs enrichment is pending`

- `web/tests/e2e/message-list-locate-primary.spec.ts`
  - `refs render after the latest user message while assistant is still streaming`

- `web/src/pages/RefsPanelRegressionPage.tsx`
- `web/src/pages/MessageListRegressionPage.tsx`

### 2.3 中文问号乱码问题已定位

“测试提问变成 `????`”不是后端 UTF-8 存储坏掉，而是：

- 使用 PowerShell 直接传中文命令字面量时，中文在进入 HTTP 请求前就被转成了 `?`

本轮验证后确认：

- 使用 Python / UTF-8 安全方式发请求，数据库中存储正常
- 后续做中文自动化提问时，不要再用 PowerShell 直接嵌入中文 prompt

## 3. 已验证的命令与结果

本轮已通过的检查：

- `npm run build`
- `python -m pytest tests/unit/test_references_router_cache.py tests/unit/test_chat_store_rendered_refs.py`
- `npx playwright test tests/e2e/refs-panel-regression.spec.ts --project=chromium`
- `npx playwright test tests/e2e/message-list-locate-primary.spec.ts --project=chromium`

其中最近一次 `message-list-locate-primary` 为 `10 passed`。

## 4. 真实 UI 测试集与结论

本轮基于库内现有文献做了 5 组真实问题测试。

### 4.1 测试题 1

问题：

- `哪些文章提到了NeRF？它们分别如何和SCI或三维表示结合？`

预期：

- 至少命中 `SCINeRF`
- 至少命中 `SCIGS`

结果：

- 回答基本正确
- refs 卡片也对应 `SCINeRF` 与 `SCIGS`

结论：

- 这是当前最稳定的一类“跨文献列举题”
- 回答与卡片的一致性较好

补充说明：

- 修复前，该类问题存在“后端已产生 refs，但 UI 直到回答快结束才显示”的问题
- 修复后，同类问题已可做到“refs 先出，回答后到”

### 4.2 测试题 2

问题：

- `SCINeRF的真实硬件实验装置包含哪些部件？请对应到原文图3或实验设置。`

预期：

- 命中 `SCINeRF` 原文 Figure 3
- 关键部件应包含：
  - `CCD camera`
  - `primary and relay lens`
  - `DMD`

结果：

- 回答错误地说“未找到”
- refs 卡片也没有定位到 Figure 3，而是偏到了：
  - `4.2 Additional Study / Figure 5`
  - `SCIGS` 对比段落

结论：

- 这是当前最明确的高优先级准确性问题
- 原文证据是存在的，但检索/定位没有召回到真正需要的图注和实验设置块

原文可确认的证据：

- `Figure 3. Experimental setup for real dataset collection.`
- `This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.`

### 4.3 测试题 3

问题：

- `哪篇文章对Hadamard单像素成像和Fourier单像素成像做了理论对比？主要结论是什么？`

预期：

- 首篇命中应为 `OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging`

结果：

- 回答正确
- 首卡正确
- 后续卡片混入一些弱相关的单像素成像文献

结论：

- 当前系统对“明确比较题”已经能给出对的主答案
- 但 refs 卡片排序还不够干净，候选集过宽

质量观察：

- 回答本身信息量较高
- refs 卡片里的 `summary_line` 仍偏泛化，很多只是“相关内容位于某章节”

### 4.4 测试题 4

问题：

- `哪些文献讨论了单像素成像中的深度学习？请概括它解决了什么问题，又有哪些挑战。`

预期：

- 至少命中：
  - `LPR-2025-Advances and Challenges of Single‐Pixel Imaging Based on Deep Learning`
  - `Optics & Laser Technology-2024-Part-based image-loop network for single-pixel imaging`

结果：

- 回答内容看起来较完整，且提到了正确文献
- 但 refs 最终被渲染成 `empty`
- 同时 `hits_json` 原始命中其实是强命中，不是没有结果

关键现象：

- `used_query = single-pixel deep learning discussion imaging`
- `used_translation = True`
- 原始 `hits_json` 有 6 条命中
- 最终 `rendered_payload_json`：
  - `display_state = empty`
  - `suppression_reason = no_candidate_hits`

结论：

- 这是“回答可用，但 refs 被错误压空”的典型问题
- 问题不在检索本身，而在 raw hits 到 final refs payload 的过滤/抑制链路

### 4.5 测试题 5

问题：

- `结构化探测在激光扫描显微中主要解决什么矛盾？与传统ISM或共聚焦的权衡有什么不同？`

预期：

- 主文献应为：
  - `NatPhoton-2025-Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy`

结果：

- 回答主体基本正确
- 但 refs 卡片排序不对
- 正确文献没有排到第 1，而是被排到了第 6

关键现象：

- raw retrieval 第 1 名就是 `NatPhoton-2025`
- final refs card 排序时却被其他弱相关文献挤到了后面

结论：

- 当前问题不在“能不能搜到”
- 而在“最终 card 排序与回答主证据没有强绑定”

## 5. 时间线观察

### 5.1 修复前的现象

几类真实问题中，曾观察到：

- 后端很早已有 hits
- 但 UI refs 面板要等到回答结束前后才出现

典型例子：

- Hadamard/Fourier 题：
  - `firstBackendHitMs ≈ 733ms`
  - `firstPanelMs ≈ 12902ms`
  - `answerDoneMs ≈ 13298ms`

- 结构化探测题：
  - `firstBackendHitMs ≈ 4097ms`
  - `firstPanelMs ≈ 20563ms`
  - `answerDoneMs ≈ 20564ms`

说明：

- 问题不是后端完全没算出来
- 而是 UI 在生成中途没有把已到达的 refs 展示出来

### 5.2 修复后的复测

Hadamard/Fourier 题复测结果：

- `firstBackendHitMs ≈ 683ms`
- `firstPanelMs ≈ 683ms`
- `firstCardMs ≈ 683ms`
- `answerDoneMs ≈ 81763ms`

说明：

- 当前 refs 提前展示的 UI 阻塞已被修掉
- 后续若用户仍感觉“很慢”，就更可能是回答生成本身慢，而不是 refs 面板展示卡住

## 6. 归纳后的主要问题

### P0：精确定位召回失败

代表案例：

- `SCINeRF 图3/实验装置` 问题

现状：

- 原文明确有图注与部件列表
- 但系统没有把 `Figure 3 + Experimental setup + CCD/DMD/lens` 作为高优先级证据召回

需要改进：

- 对 `Figure 3 / 图3 / 实验设置 / hardware / 装置 / 部件` 这类 prompt 增加显式 focus match
- 对 figure caption、实验设置章节、图号精确匹配增加排序加权

### P0：回答与 refs 使用了不同的权威证据面

代表案例：

- `单像素成像中的深度学习`
- `结构化探测 vs ISM`

现状：

- raw hits 可能是对的
- 回答主文献也可能是对的
- 但 final refs payload 可能被压空、错排，或与回答主证据脱节

需要改进：

- 在 answer render 与 refs render 之间建立同一个 authoritative evidence contract
- 如果回答已经明确采用某篇文献作为主证据，refs top card 不应换成别的弱相关文献

### P0：refs 空结果抑制逻辑过强

代表案例：

- `display_state = empty`
- `suppression_reason = no_candidate_hits`
- 但 `hits_json` 实际上有强命中

需要改进：

- 把 `raw_hit_count / post_score_gate / post_focus_filter / post_llm_filter / final_hit_count` 做成更明确的 debug 输出
- 排查哪一步把有效 hits 清零
- 如果回答已经引用到这些文献，refs 不应同时变成 empty

### P1：card copy 质量不稳定

现状：

- 很多卡片是 `section_grounded`
- 常见 summary 只有：
  - `相关内容位于……`
  - `该文在某章节讨论了……`

这类卡片可读性低，不足以支撑“为什么是这篇”

需要改进：

- 优先输出具体句子级摘要
- 如果没有高质量一句话摘要，至少用原文 snippet 做轻量压缩
- 避免只回显章节名

### P1：后处理会删坏答案

已观察到的残留：

- `****`
- `基于）`
- `和 分别讨论……`

说明：

- 清理引用标记、Markdown 或内部结构标记的后处理还不够稳
- 部分 token 被删掉后留下残缺句式

需要改进：

- 为 sanitizer / markdown cleanup 增加中文回答回归样例
- 特别覆盖：
  - 删引用后语句仍通顺
  - 中英文括号成对
  - 列表项不残留星号

### P2：可观测性还不够

现状：

- 需要临时查数据库和 payload 才能分辨“慢在哪”

建议增加：

- `generation_started_at`
- `first_raw_hit_at`
- `first_rendered_refs_at`
- `assistant_first_token_at`
- `assistant_done_at`
- `ui_refs_first_visible_at`

如果这些指标可直接从 debug 接口或日志中拿到，后续定位会快很多

## 7. 建议的后续优化顺序

### 第一阶段：先修正确性

目标：

- 先把“回答说的文献”和“卡片展示的文献”统一起来

建议顺序：

1. 修 `Figure / Experimental Setup / Hardware` 类问题的精确召回
2. 修 `raw hits 有结果，但 final refs 变 empty` 的过滤链路
3. 绑定回答主证据与 refs top card 的一致性

验收标准：

- `SCINeRF 图3` 问题必须稳定命中 Figure 3
- `单像素成像中的深度学习` 问题不允许回答提到文献而 refs 为空
- `结构化探测` 问题的 top card 必须是 `NatPhoton-2025`

### 第二阶段：再修卡片质量

目标：

- 提升 refs 卡片的可用性，而不是只显示章节回声

建议顺序：

1. 改善 `section_grounded` summary 生成策略
2. 将 card `why_line` 与用户问题的焦点词对齐
3. 为 figure、实验设置、对比题、综述题分别做更合适的卡片模版

验收标准：

- 卡片 summary 不能只剩“相关内容位于……”
- 用户应能仅靠卡片判断“为什么是这篇”

### 第三阶段：修后处理与可观测性

目标：

- 避免回答表面可看，但细节被清洗坏

建议顺序：

1. 为答案后处理加入中文高频回归集
2. 增加每轮问题的时间线指标
3. 对真实 UI 测试结果自动产出一份结构化 JSON 报告

## 8. 建议长期保留的真实测试问题

建议把下面几题固化成长期 smoke / manual regression 集：

1. `哪些文章提到了NeRF？它们分别如何和SCI或三维表示结合？`
2. `SCINeRF的真实硬件实验装置包含哪些部件？请对应到原文图3或实验设置。`
3. `哪篇文章对Hadamard单像素成像和Fourier单像素成像做了理论对比？主要结论是什么？`
4. `哪些文献讨论了单像素成像中的深度学习？请概括它解决了什么问题，又有哪些挑战。`
5. `结构化探测在激光扫描显微中主要解决什么矛盾？与传统ISM或共聚焦的权衡有什么不同？`
6. 继续保留已有负例：
   - `Which paper in my library most directly discusses ADMM?`

## 9. 给下一位接手同事的注意事项

1. 终端里直接 `Get-Content` 某些 UTF-8 中文文件时，PowerShell 显示可能乱码；文件本身不一定坏
2. 如果要做中文自动化提问，不要直接在 PowerShell 命令行里嵌中文 prompt
3. 更稳妥的做法：
   - 用 Python `read_text(encoding='utf-8')`
   - 用 Python / Playwright 发 UTF-8 请求
   - 文档文件名优先使用 ASCII，正文再写中文

## 10. 本轮最值得继续追的 3 个点

如果下一个迭代只能做 3 件事，建议优先做：

1. 修 `SCINeRF 图3` 这类“明明原文有，系统却说没有”的精确定位问题
2. 修 “回答有文献、refs 却 empty / 错排” 的 answer-refs 不一致问题
3. 修答案后处理残缺，确保不会再出现 `****`、`基于）`、`和 分别讨论……` 这类输出
