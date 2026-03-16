# 文献阅读指导证据定位重构计划

## 1. 当前现象

当前版本已经解决了“入口完全不显示”的问题，但仍存在两个核心失败点：

1. 入口挂点不准。
   入口虽然出现在回答正文里，但经常不是挂在真正被证据支持的那句话旁边。
2. 跳转高亮不准。
   点击后能打开 reader，但高亮块和回答里的句子强相关性不足，用户感知为“根本没匹配上”。

这说明问题已经从“可见性”转移到了“正确性”。

## 2. 根因判断

根因不是样式，而是链路架构仍然错误：

1. 前端仍在自行切回答并做二次匹配。
   当前 `MessageList` 会把 `bodyContent` 重新拆成若干渲染块，再用文本相似度去猜它对应哪条 provenance segment。
2. 点击后仍会重新按 snippet 对候选块排序。
   当前 strict 模式虽然已经有 `provenance.segments`，但点击时仍存在“前端重排”行为，会把后端给出的主证据块换成一个“更像但不一定正确”的块。
3. Reader 仍以整篇 markdown 渲染为主。
   虽然 reader 已经拿到了 `blocks[]`，但渲染主视图仍然以整篇 markdown 为主，再把 block identity 映射回 DOM，导致 block -> DOM 不是严格一一对应。
4. 后端 segment 信息还不够完整。
   当前 `provenance.segments` 只有 `text / evidence_block_ids / confidence`，还不够支撑“按 segment 渲染 + 直达主证据块”。

结论：不能继续围绕前端 heuristic 调阈值。主线必须切换到“后端权威 segment + block identity 精确跳转”。

## 3. 本次重构目标

### 3.1 产品目标

1. 入口必须贴在真正被证据支持的回答 segment 旁边。
2. 点击后优先跳到后端判定的主证据块，而不是前端重新猜测的块。
3. 高亮优先对准证据短语或证据句，而不是整篇模糊搜索。
4. 对没有 direct evidence 的 synthesis 段，不显示定位入口。

### 3.2 工程目标

1. 回答渲染以后端 segment 为主，不再由前端自行切分正文。
2. `provenance.segments` 必须携带前端可直接消费的显示字段与证据字段。
3. strict 模式点击时不再重排 primary block，只允许在后端给定候选范围内降级。
4. Reader 逐步转为 block-driven 渲染，但本轮先完成“block identity 精确滚动”链路。

## 4. 数据契约调整

本轮扩展 `message.provenance.segments[]`，每个 segment 至少包含以下字段：

1. `segment_id`
2. `segment_index`
3. `kind`
4. `segment_type`
5. `text`
6. `raw_markdown`
7. `display_markdown`
8. `cite_details`
9. `evidence_mode`
10. `evidence_confidence`
11. `primary_block_id`
12. `primary_anchor_id`
13. `primary_heading_path`
14. `support_block_ids`
15. `evidence_quote`

字段职责：

1. `raw_markdown`
   后端切 segment 时保留原始 markdown，前端不再自己拆。
2. `display_markdown`
   API 层完成 citation / equation 注释后的最终渲染文本。
3. `cite_details`
   segment 自己的 citation hover 元数据，避免依赖整条消息共享 anchor。
4. `primary_block_id / primary_anchor_id`
   作为 strict 跳转的首选目标。
5. `support_block_ids`
   作为候补证据块，不允许前端跳出这个集合自由搜索。
6. `evidence_quote`
   用于 reader 内二次高亮证据句，而不是整块泛高亮。

## 5. 实施分期

### Phase 1：后端补全权威 segment

目标：让 segment 真正可渲染、可跳转、可高亮。

实施项：

1. `split_answer_segments()` 保留 `raw_text`。
2. provenance builder 输出 `raw_markdown / primary_block_id / support_block_ids / evidence_quote`。
3. API 渲染层为每个 segment 生成 `display_markdown + cite_details`。
4. 类型定义补齐到前端 API 层。

完成标准：

1. 一条 paper guide assistant 消息的 `provenance.segments[]` 已经足以单独渲染回答。
2. direct segment 至少有 `primary_block_id`。

### Phase 2：聊天区按 segment 渲染

目标：入口不再挂在“前端猜出来的句子”上，而是挂在后端 segment 上。

实施项：

1. strict 模式优先按 `provenance.segments[].display_markdown` 渲染。
2. 入口只对 direct segment 显示。
3. 入口数量控制为 1-5 个，按 direct evidence 置信度筛选，但渲染顺序保持原回答顺序。
4. 旧 heuristic 匹配保留为非 strict 模式 fallback。

完成标准：

1. 入口总是出现在 segment 对应句子旁。
2. 不再需要 `splitAnswerRenderSegments(bodyContent)` 才能决定入口位置。

### Phase 3：strict 跳转改为 exact-first

目标：点击后先相信后端的主证据块，而不是前端重排。

实施项：

1. `openReaderByStructuredEntry()` 默认直接使用 `primary_block_id` 对应块。
2. `support_block_ids` 仅作为后备候选。
3. 前端不再按 snippet 重新给 alternatives 排序。
4. `payload.snippet` 优先传 `evidence_quote`，其次传 `segment.text`。

完成标准：

1. strict 模式点击时，不会因为前端相似度重排而跳离后端选中的主块。

### Phase 4：reader 证据高亮精修

目标：高亮从“整块”提升到“块内证据句”。

实施项：

1. reader 收到 `block_id + evidence_quote` 后，先精确滚动到 block。
2. 在 block 内搜索 `evidence_quote`，优先高亮 quote。
3. quote 命中失败时，再回退到整块高亮。
4. 若 exact identity 失效，只允许在 `primary + support + neighbor` 中做 constrained fallback。

完成标准：

1. 用户看到的高亮短语必须与回答句子强相关。

## 6. 本轮实施边界

这轮先完成 Phase 1 + Phase 2 + Phase 3 的主线，不在本轮做以下事情：

1. 不重写整套 reader 为 block-only 组件。
2. 不做历史消息批量迁移。
3. 不引入新的大模型自由搜索定位。

## 7. 风险与约束

1. segment 级渲染可能改变当前 markdown 的部分视觉分组。
   处理：先保证定位正确性，列表视觉再做二次整理。
2. 一些老消息没有新增字段。
   处理：没有 `display_markdown` 时回退到旧渲染路径。
3. `evidence_quote` 抽取可能不稳定。
   处理：reader 中 quote 高亮失败时只退到 primary block，不退到全文模糊搜。

## 8. 验收标准

### 8.1 功能验收

1. 有 direct evidence 的 segment 可以在段边看到 `❞` 入口。
2. synthesis segment 不显示入口。
3. 点击后优先跳到 `primary_block_id` 对应块。
4. 高亮内容与回答句子显著相关。

### 8.2 指标验收

1. 每条回答定位入口数：1-5。
2. strict 模式首次 exact block 命中率 > 90%。
3. 因前端重排导致的错误跳转应降为 0。

## 9. 当前执行顺序

1. 写入本计划文档。
2. 落 Phase 1：后端 segment 协议补齐。
3. 落 Phase 2：聊天区按 segment 渲染。
4. 落 Phase 3：strict 跳转 exact-first。
5. 构建与单测回归。
