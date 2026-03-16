# 会话打开性能与读路径架构改造计划

## 1. 目标

这份计划解决的是一个明确问题：

- 当会话消息很多时，点击打开会话后，主内容区域会卡一段时间才刷新出来。

这不是单一的前端卡顿问题，而是整条读路径的架构问题。当前链路里同时存在：

1. 全量消息读取
2. 后端按读请求现算 assistant 渲染结果
3. 前端在切会话时同步重算时间线、paper-guide locate、provenance 映射

本计划的目标是把“切会话”从：

- 打开一个重计算页面

改成：

- 打开一个以最近消息为主、逐步补全的读优化界面

## 2. 当前现象

从当前实现看，切会话的主要问题有四层：

### 2.1 前端切换时旧内容停留太久

当前 `selectConversation` 在切换时会先更新：

- `activeConvId`
- `activeConversation`
- `refs`
- `generation`

但不会立即清空或替换 `messages` 显示区。

相关文件：

- `web/src/stores/chatStore.ts`
- `web/src/pages/ChatPage.tsx`

结果：

- 用户点击了新会话，但屏幕上还挂着旧会话内容
- 新会话消息要等后端全量返回后才整体替换
- 主观感受就是“点了没反应”

### 2.2 后端每次全量读取整个会话

当前前端调用：

- `web/src/api/chat.ts`

使用的是：

- `GET /api/conversations/{conv_id}/messages`

后端实现：

- `api/routers/chat.py`
- `kb/chat_store.py`

当前 `get_messages(conv_id, limit=None)` 使用：

- `ORDER BY id ASC`

也就是说当前读路径默认是“整段会话全量取回”，而不是“优先最近一页”。

### 2.3 后端对整段 assistant 消息做读时渲染

当前 `/messages` 返回前还会走：

- `api/chat_render.py`

里面会对整段 assistant 消息做：

1. in-paper citation 注释
2. copy text / copy markdown 生成
3. rendered body / rendered content 生成
4. ref 相关 enrich

这意味着：

- 会话越长
- assistant 消息越多
- 切会话的后端返回越慢

### 2.4 前端拿到消息后还会同步做重计算

当前 `ChatPage` 与 `MessageList` 在收到 `messages` 后还会同步做：

1. 时间线节点推导
2. timeline 映射表构建
3. refs / cite map 计算
4. structured provenance locate entry 构建
5. render slot / locate fallback 相关推导

相关文件：

- `web/src/pages/ChatPage.tsx`
- `web/src/components/chat/MessageList.tsx`

其中 paper-guide 会话由于 `message.provenance` 更重，成本更明显。

## 3. 根因判断

根因不是“某一个组件慢”，而是当前读路径架构默认了：

1. 会话读取是全量读取
2. assistant 展示结果是读时现算
3. timeline 是前端临时推导
4. provenance 是渲染时临时编译
5. 页面首屏要等全部辅助信息准备好

这对短会话还能接受，但对长会话会变成：

- 网络传输慢
- JSON 解析重
- React 首次渲染重
- 侧边时间线和 paper-guide 计算放大主线程压力

## 4. 成功标准

本改造完成后，至少要满足以下结果：

1. 点击会话后，主内容区域在极短时间内出现明确加载态，不再停留旧会话内容。
2. 最近一页消息优先显示，不等待整段历史。
3. 长会话打开时，timeline / refs / paper-guide 辅助层不会阻塞主消息区首屏。
4. 再次切回最近访问过的会话时，能优先命中前端缓存。
5. 后端读接口的主要成本不再随整段会话线性增长得这么明显。

建议指标：

1. 最近一页消息首屏时间 `P50 < 300ms`
2. 最近一页消息首屏时间 `P95 < 800ms`
3. 缓存命中会话切回时间 `P50 < 120ms`
4. timeline 首次可交互不阻塞主消息内容显示

## 5. 非目标

这轮改造不以这些为目标：

1. 重做聊天产品交互
2. 重写整个 provenance 系统
3. 把普通聊天页直接改成一个完全虚拟化 IDE
4. 立刻做消息全文检索或跨会话搜索架构

## 6. 架构方向

## 6.1 核心原则

会话打开应该是：

1. `tail-first`
2. `cache-first`
3. `primary-content-first`
4. `secondary-panels-later`

翻成产品行为就是：

1. 先看最近消息
2. 先把主消息区显示出来
3. 时间线、refs、paper-guide 元信息稍后补
4. 历史消息按需加载

## 6.2 推荐架构

### A. 引入会话读模型

不要把“原始消息存储”直接当“聊天页读取模型”。

建议拆成：

1. 写模型
   - 原始 message
   - refs
   - provenance

2. 读模型
   - rendered assistant payload
   - timeline summary
   - recent page metadata
   - optional paper-guide summary fields

这本质上是轻量的 `CQRS / read-model` 思路，不需要一次性重做所有后端，只要先把读路径从“现算”改成“面向页面消费”即可。

### B. tail-first 会话打开接口

新增一个面向聊天页的读接口，例如：

- `GET /api/conversations/{conv_id}/open?tail_limit=24`

返回结构建议：

```json
{
  "conversation": {},
  "messages": [],
  "has_more_before": true,
  "oldest_loaded_id": 123,
  "timeline": [],
  "refs_state": "idle|loading|ready"
}
```

这样聊天页不需要再：

1. 先调 conversation
2. 再调 messages
3. 再自己临时从全量 messages 推 timeline

### C. 历史消息分页接口

新增分页接口，例如：

- `GET /api/conversations/{conv_id}/messages?before_id=123&limit=24`

注意这里必须是：

1. 先按 `id DESC LIMIT ?`
2. 再在服务端反转成 ASC 返回

而不是当前 `ORDER BY id ASC LIMIT ?` 的“最早 N 条”。

### D. assistant 渲染结果缓存

当前 `api/chat_render.py` 的结果应该被缓存，而不是每次切会话都重算。

建议缓存这些字段：

1. `rendered_content`
2. `rendered_body`
3. `copy_text`
4. `copy_markdown`
5. `cite_details`
6. `render_cache_key`

缓存失效条件可以基于：

1. message content
2. refs payload version
3. render pipeline version

### E. timeline 独立为二级读模型

timeline 不应该由 `ChatPage` 每次用整段 `messages` 临时推导。

建议后端直接返回：

1. `order`
2. `user_msg_id`
3. `target_msg_id`
4. `question_preview`
5. `has_answer`

前端只负责：

1. 渲染
2. 高亮联动
3. 跳转

### F. 前端会话级缓存

当前 store 是：

- `activeConvId + messages + refs`

建议改成：

1. `conversationMetaById`
2. `messagePagesByConvId`
3. `timelineByConvId`
4. `refsByConvId`
5. `loadingStateByConvId`
6. `openPerfByConvId`

这样切回最近会话时不必重做整段加载。

## 7. 分阶段实施计划

### Phase 0. UX 快速止血

目标：

- 先去掉“点了没反应”的体感

工作项：

1. 在 `chatStore` 增加 `conversationLoading`
2. `selectConversation` 开始时立即清除旧消息主区展示或切换为 skeleton
3. `ChatPage` 在 loading 时显示会话级占位 UI
4. timeline 在 loading 时延迟挂载

涉及文件：

- `web/src/stores/chatStore.ts`
- `web/src/pages/ChatPage.tsx`

验收：

1. 点击会话后，旧消息不会继续假装是当前会话内容
2. 用户能立刻感知“正在打开新会话”

### Phase 1. tail-first 消息读取

目标：

- 先只取最近一页消息

工作项：

1. 新增后端 tail API
2. 修正消息分页 SQL，支持真正的“最后 N 条”
3. 前端切会话改用 tail API
4. 保留“展开更早消息”能力，但改为分页追加

涉及文件：

- `kb/chat_store.py`
- `api/routers/chat.py`
- `web/src/api/chat.ts`
- `web/src/stores/chatStore.ts`
- `web/src/pages/ChatPage.tsx`

验收：

1. 长会话打开只请求最近一页
2. “显示更早消息”走分页，不再依赖整段已加载

### Phase 2. timeline 与辅助面板解耦

目标：

- 主消息先出，侧边辅助信息后出

工作项：

1. timeline 改为 tail API 一并返回或二次异步返回
2. 前端 timeline 改为低优先级更新
3. timeline 列表做虚拟化或至少做条数上限
4. refs / paper-guide meta 延迟装配，不阻塞主区

涉及文件：

- `web/src/pages/ChatPage.tsx`
- `web/src/components/chat/MessageList.tsx`
- `web/src/components/layout/*`
- 可选新增 timeline API

验收：

1. 主消息区首屏不被 timeline 阻塞
2. 长会话右侧 timeline 打开不会显著拖慢切换

### Phase 3. assistant render cache

目标：

- 后端不再对整段 assistant 每次读都现算

工作项：

1. 引入 message render cache 结构
2. 在消息生成完成时写入缓存
3. refs 更新时按需失效并重建
4. `/open` 和 `/messages` 优先返回缓存结果

涉及文件：

- `api/chat_render.py`
- `kb/chat_store.py`
- 可能新增缓存表或缓存字段
- `api/routers/chat.py`

验收：

1. 同一个长会话重复打开时，后端耗时显著下降
2. message render 不再随着总消息量线性放大

### Phase 4. 前端按会话缓存与增量更新

目标：

- 切回最近会话尽量秒开

工作项：

1. store 改成按会话存消息页和 timeline
2. 最近打开会话保留热缓存
3. 切换时先命中缓存，再后台 refresh
4. 与生成中会话的流式更新并存

涉及文件：

- `web/src/stores/chatStore.ts`
- `web/src/pages/ChatPage.tsx`
- `web/src/api/chat.ts`

验收：

1. 同一批近期会话来回切换明显更快
2. 不会因为缓存把旧会话内容错误展示为新会话内容

### Phase 5. paper-guide 与 provenance 读优化

目标：

- 收掉长 assistant + heavy provenance 的附加成本

工作项：

1. `MessageList` 里重的 provenance prep 按 `message.id + render_cache_key` memo
2. 对 paper-guide message 的 structured locate summary 做预处理字段
3. timeline 与 locate 只消费轻量 summary，不在 render 时重编译整段 provenance

涉及文件：

- `web/src/components/chat/MessageList.tsx`
- `kb/paper_guide_provenance.py`
- `kb/task_runtime.py`
- 相关 API 输出结构

验收：

1. paper-guide 长会话打开成本不再明显高于普通会话
2. locate 能力不回退

## 8. 数据与 API 改造建议

### 8.1 新增打开接口

建议新增：

- `GET /api/conversations/{conv_id}/open`

参数：

- `tail_limit`
- 可选 `include_timeline`
- 可选 `include_refs_state`

返回：

1. conversation meta
2. recent messages
3. has_more_before
4. oldest_loaded_id
5. recent timeline
6. refs readiness state

### 8.2 新增历史分页接口

建议形式：

- `GET /api/conversations/{conv_id}/messages?before_id=...&limit=...`

必要约束：

1. 返回顺序仍保持 ASC，方便前端直接 prepend
2. 底层查询必须先倒序取尾页，再反转

### 8.3 可选新增 timeline 接口

如果不想先把 timeline 塞进 `/open`，可以独立给：

- `GET /api/conversations/{conv_id}/timeline`

## 9. 前端设计建议

### 9.1 切会话状态机

建议把会话页状态拆成：

1. `idle`
2. `switching`
3. `ready`
4. `backfilling_history`
5. `refreshing_secondary_panels`

不要再只靠：

- `activeConvId`
- `messages`

两个字段隐式推状态。

### 9.2 主内容优先

首屏只保证：

1. 最近消息
2. 输入框
3. 基础滚动定位

以下内容允许晚一拍：

1. timeline
2. refs panel
3. provenance-heavy affordance
4. paper-guide 额外统计信息

### 9.3 timeline 退居二级

timeline 是辅助导航，不应阻塞主内容。

建议：

1. 切换时先渲染简版 timeline 或 skeleton
2. 条目多时只渲染视窗区
3. 联动逻辑保持，但测量与渲染降优先级

## 10. 风险

1. 分页后，jump-to-message、timeline 跳转、reader 引用可能依赖未加载历史消息。
   - 需要在跳转时自动补齐对应消息页。
2. assistant render cache 引入后，refs 更新与缓存失效必须严格一致。
3. 前端会话缓存如果做得不干净，容易出现串会话数据。
4. paper-guide 的 structured locate 不能因为读优化而破坏 contract。

## 11. 验收与观测

### 11.1 前端埋点

建议记录：

1. 点击会话时间
2. tail API 返回时间
3. 首屏消息可见时间
4. timeline ready 时间
5. refs ready 时间
6. 是否命中前端缓存

### 11.2 后端埋点

建议记录：

1. 会话消息查询耗时
2. render enrich 耗时
3. timeline 构建耗时
4. render cache 命中率

### 11.3 核心回归

至少覆盖：

1. 切会话不再显示旧消息残留
2. 分页加载历史消息顺序正确
3. timeline 仍能跳转到正确回答
4. paper-guide locate 不受 tail-first 影响
5. 长会话切回缓存命中时内容正确

## 12. 推荐实施顺序

如果只按收益排优先级，建议严格按这个顺序做：

1. `Phase 0` conversation loading 与 skeleton
2. `Phase 1` tail-first API 与分页
3. `Phase 2` timeline 与辅助面板降级
4. `Phase 3` assistant render cache
5. `Phase 4` 前端按会话缓存
6. `Phase 5` paper-guide/provenance 读优化

原因：

1. Phase 0 先修观感
2. Phase 1 先砍掉最大读放大器
3. Phase 3 再砍掉后端现算成本
4. 最后再优化复杂的 paper-guide 增量路径

## 13. Immediate Next Task

建议下一步直接做 `Phase 0 + Phase 1` 的实施设计。

最小起步任务：

1. 在 `chatStore` 引入 `conversationLoading`
2. 在 `ChatPage` 增加切会话 skeleton
3. 后端新增真正的 tail 消息读取 SQL
4. 前端切会话改成优先加载最近 24 条

这四步做完之后，用户体感会先明显变好，后面再继续做 timeline 和 read-model 才有稳定基线。
