# 文献阅读指导定位功能回归修复计划

## 1. 当前回归现象

本轮改造后，入口可见性提升了，但出现了更严重的质量回归：

1. 回答排版明显变差。
   原本一条结构化回答被拆成大量松散段落，阅读流被破坏。
2. 入口经常挂在“废话句”或“引导句”上。
   例如“表明：”“说明：”“这进一步证实：”这类承上启下的壳句，而不是真正承载证据的内容句。
3. 跳转高亮落到宽泛、空泛、弱相关的原文块。
   用户点击的是一个具体判断句，reader 却高亮到类似 “Our method exploits NeRF” 这种泛表述。

这说明当前实现虽然在技术上“有入口、有跳转”，但产品上是失败的。

## 2. 失败原因复盘

### 2.1 显示层失败：把 provenance segment 当成 display segment

当前 strict 模式已经开始按 provenance segment 直接渲染回答。
但 provenance segment 的职责本来是“证据映射单元”，不是“最终展示单元”。

直接后果：

1. 列表、层级、节标题被打散。
2. 同一句话既出现在结论段，也出现在依据段时，会被重复渲染。
3. 原始回答里的 section 结构丢失，排版变成“扁平化碎片列表”。

结论：

1. `display unit` 和 `evidence unit` 必须分离。
2. provenance segment 不能直接替代完整回答渲染。

### 2.2 选点失败：命中了“修辞壳”，没命中“内容核”

当前可定位入口的 direct segment 里，存在大量“修辞引导句”，例如：

1. “表格标题与方法命名明确为……表明：”
2. “文中提到……说明：”
3. “这进一步证实：”

这些句子的问题不是错误，而是信息密度太低：

1. 它们主要起过渡作用。
2. 真正有证据价值的内容，往往在后续子弹点或后续一句。
3. 用户点击这种壳句时，系统很容易跳到一个泛相关块，而不是强证据块。

结论：

1. 可定位入口不能挂在 rhetorical shell 上。
2. 系统必须优先给“内容核句”打点，而不是给“承接句”打点。

### 2.3 reader 失败：高亮策略仍然过于宽松

虽然现在 strict 模式已开始走 `primary_block_id`，但 reader 仍存在两个问题：

1. quote 不够强时，会回退到宽泛块级高亮。
2. 一旦块内搜索失败，系统仍容易把用户带到“主题相近但证据不对”的句子。

结论：

1. strict 模式应禁止“全文自由模糊搜”。
2. 如果 exact quote 失败，最多只能在 `primary + support + neighbor` 范围内降级。

## 3. 修复原则

1. 先恢复阅读质量，再恢复定位精度。
2. 先减少错误入口，再追求入口数量。
3. 入口必须服务于“证据验证”，不能破坏正文本身。
4. strict 模式宁可少，也不能错。

## 4. 新的结构设计

### 4.1 分离三种单元

后续系统中必须明确区分三类对象：

1. `display blocks`
   最终给用户看的回答结构，保留结论/依据/补充/列表层次。
2. `evidence claims`
   真正允许挂定位入口的内容核句。
3. `source blocks`
   reader 中可精确滚动和高亮的原文块。

关系应为：

1. 一个 `display block` 可以包含 0 到多个 `evidence claims`
2. 一个 `evidence claim` 对应 1 个 `primary source block`，以及少量 `support source blocks`

### 4.2 UI 层恢复策略

回答区不再直接按 provenance segment 逐条渲染，而是：

1. 保留原始 `rendered_body` 作为主显示内容。
2. 在主显示内容内部，为少量高质量 `evidence claim` 注入入口。
3. 入口仍放在句子右边，但句子来自 display block，不来自 segment 碎片。

这样可以同时保住：

1. 排版
2. section 结构
3. 入口就近性

## 5. 选点策略重做

### 5.1 新增“壳句过滤器”

以下类型默认不允许挂入口，除非后面没有更具体内容句：

1. 以“表明：”“说明：”“可见：”“因此：”“这说明：”“进一步证实：”结尾的承接句
2. 主要由引用导语组成的句子，例如“文中提到：”
3. 只负责展开后续列表的 lead sentence

过滤规则建议：

1. 若句子末尾是冒号，且后续存在同级或下级列表项，则优先把入口移动到后续内容项。
2. 若句子中实体词密度低、动作词少、证据短语缺失，则不打点。
3. 若句子只提供推论框架，不提供事实内容，则不打点。

### 5.2 新增“内容核评分”

每个可选 claim 增加内容核评分，评分项包括：

1. 是否包含具体对象
   例如方法名、变量名、公式编号、实验设置、原句引用。
2. 是否包含可验证断言
   例如“使用 ground-truth 图像估计位姿”。
3. 是否能独立成立
   单独读这一句，用户是否知道它在说什么。
4. 是否存在强 evidence quote
   原文里是否有可直接高亮的短语或句子。

默认策略：

1. `direct evidence claims` 不再做稀疏抽样，只要来自原文且可 strict 定位，就必须暴露入口。
2. `optional fuzzy claims` 才做压缩与限额。
3. 旧的“每条回答最多显示 2-4 个入口，硬上限 5”只适用于 optional fuzzy 入口，不适用于 direct quote / blockquote / formula / 公式解释。

### 5.3 列表项合并策略

对于这种结构：

1. lead sentence: “表明：”
2. children:
   - 先用 SCI 重建恢复图像
   - 再将图像输入 NeRF

应将其视为一个 claim group，而不是三个独立可定位项。

推荐处理：

1. 入口挂在第一个真正有信息量的子项上。
2. hover / title 中说明该入口对应一个 grouped evidence claim。
3. 点击后在 reader 中同时提供主块和相邻支持块。

## 6. 回答生成策略修正

当前回答内容本身也在制造定位噪音，主要表现在：

1. 结论在“结论”和“依据”里重复出现。
2. “下一步”模板被不合时宜地插入事实型问答。
3. 依据段中混入泛化建议，而不是纯证据支撑。

所以需要从生成端收口：

### 6.1 区分回答模式

1. `fact_answer`
   只输出：结论 + 依据 + 可选补充
2. `reading_guide`
   输出：结论 + 依据 + 下一步
3. `critical_review`
   输出：结论 + 依据 + 风险/问题

像你截图这种“具体事实求证”场景，应强制使用 `fact_answer`，不再自动附带泛泛的“下一步”。

### 6.2 去重规则

1. `依据` 不允许简单重复 `结论` 原句。
2. `依据` 必须比 `结论` 更具体，要么给原文短句，要么给分条事实。
3. 若 `依据` 第一条与 `结论` 相似度过高，则自动压缩或重写。

## 7. reader 高亮修复

### 7.1 strict 模式高亮链路

新链路应为：

1. 用 `primary_block_id` 精确滚动到块
2. 在块内匹配 `evidence_quote`
3. quote 命中失败时，只在 `support_block_ids + adjacent blocks` 中尝试
4. 若仍失败，提示“仅定位到证据块，未命中证据短语”

### 7.2 禁止宽泛回退

strict 模式下，以下行为应禁用：

1. 全文节点级自由模糊搜索
2. 因为共享关键词而跳到文首泛定义句
3. 因为出现相同术语（如 NeRF）而高亮无关句

## 8. 分阶段修复顺序

### Phase A：先止损

目标：恢复回答排版，移除最明显的废话挂点。

实施项：

1. strict 模式停止直接按 provenance segment 扁平渲染回答
2. 恢复以 `rendered_body` 为主显示
3. 暂时禁用 rhetorical shell 入口
4. 暂时把入口数降到 1-3

验收：

1. 排版恢复正常
2. 明显废话句不再有入口

### Phase B：引入 claim group

目标：让入口挂在“内容核句”上。

实施项：

1. 从 answer/provenance 中抽 `evidence claims`
2. 增加 claim group 和内容核评分
3. 列表 lead sentence 与子项做合并

验收：

1. 入口主要出现在有事实含量的句子上
2. “表明：”“说明：”不再成为默认挂点

### Phase C：reader exact highlight

目标：高亮短语和点击句强相关。

实施项：

1. exact quote 高亮
2. constrained fallback
3. UI 上明确显示 exact / degraded 命中状态

验收：

1. 点击后的第一眼高亮内容能直接解释回答里的那句话

### Phase D：生成端收口

目标：减少定位噪音源。

实施项：

1. 事实型问答关闭“下一步”模板
2. 去重结论与依据
3. 依据区只保留支撑性内容

验收：

1. 回答本身更短、更稳、更易打点

## 9. 验收指标

新增三类指标：

1. 排版回归指标
   正文 section 结构保持完整，重复渲染率接近 0
2. 废话挂点指标
   入口落在 rhetorical shell 上的比例 < 5%
3. 强相关高亮指标
   用户点击后首屏看到的高亮句与点击句强相关率 > 90%

## 10. 下一步执行建议

下一步不应该继续在现有实现上调阈值，而应按下面顺序改：

1. 先做 Phase A：回退扁平 segment 渲染，恢复排版
2. 再做 Phase B：增加 shell filter 和 claim group
3. 然后做 Phase C：reader exact quote 高亮
4. 最后做 Phase D：收紧回答生成模板

## 11. 必须有入口的场景

以下场景不再属于“可选挂点”，而属于“默认必须挂点”：

### 11.1 原文关键句直引

满足以下任一条件时，必须给入口：

1. 回答中出现明显原文直引，尤其是英文原句或长中文原句。
2. 回答中出现带引号的关键结论句，且该句本身就是证据，而不是对证据的解释。
3. 回答中出现 blockquote 形式的直接证据。

对应到你给的样例：

1. `we cannot estimate accurate poses ... we use ground truth images ...`
2. “due to the lack of high-quality details ...”
3. `providing reasonable results`

这些都不应该把入口挂在“文中提到：”“说明：”“直接证据：”之类壳句上，而应该：

1. 直接挂在引号句本身右侧；
2. 若是 blockquote，则挂在 blockquote 右上或末尾；
3. 点击后优先跳到原文中对应句子，而不是只跳到相关段落。

### 11.2 原文公式与编号公式

满足以下任一条件时，必须给入口：

1. 回答中展示了 display formula。
2. 回答中明确提到“公式(3) / Equation 3 / Eq. (3)”。
3. 回答中围绕某个公式解释变量含义、观测模型、损失函数或推导关系。

对应规则：

1. 公式块本体必须有入口。
2. 若公式下面还有一句“其中 ... 表示 ...”，该解释句也必须有入口。
3. 公式与解释句可以共用同一个 claim group / locate target。
4. 点击公式入口时，reader 必须优先定位到同编号公式或同一公式块，不能只跳到提到相同变量名的普通段落。

### 11.3 高价值事实句

以下句子在证据充分时，应优先成为入口承载句：

1. 实验设置中的关键限制、前提、例外条件。
2. 方法流程中的关键操作步骤。
3. 对比基线的关键差异句。
4. 带具体对象的判断句：
   例如方法名、变量名、公式号、表号、图号、硬件名、超参数、数据来源。

### 11.4 明确不应承载入口的句子

以下句子即使附近有证据，也不应直接挂入口：

1. `文中提到：`
2. `说明：`
3. `表明：`
4. `可见：`
5. `因此：`
6. `直接证据（唯一且明确）：`
7. `注意：`
8. `延伸思考题`
9. `下一步`

处理方式不是“删掉入口”，而是“把入口迁移到后面的内容核句、block quote 或公式块上”。

## 12. 后续具体修改步骤

下面的顺序是我建议你接下来真正执行的顺序，不再继续做零散调参。

### Step 1：把入口资格从“通用句子匹配”改成“claim 类型驱动”

在 provenance 里新增 claim 类型，而不是只存一个泛化 segment：

1. `quote_claim`
2. `blockquote_claim`
3. `formula_claim`
4. `critical_fact_claim`
5. `shell_sentence`

其中：

1. `quote_claim / blockquote_claim / formula_claim` 默认允许挂入口；
2. `shell_sentence` 默认禁止挂入口；
3. `critical_fact_claim` 只有在 `primary_block_id + evidence_quote` 可靠时才允许挂入口。

后端需要新增字段：

1. `claim_type`
2. `must_locate`
3. `primary_block_id`
4. `support_block_ids`
5. `evidence_quote`
6. `anchor_kind`：`quote | blockquote | equation | sentence`
7. `anchor_text`
8. `mapping_quality`

验收标准：

1. 你截图里的英文原句和公式块，后端产物里必须是 `must_locate = true`。

### Step 2：重做 claim 抽取规则，先抓“必须入口对象”

抽取顺序不要再按普通段落平均处理，而应按优先级处理：

1. 先抓 blockquote
2. 再抓显式引号句
3. 再抓 display formula / equation mention
4. 最后才抓普通事实句

这样做的原因很直接：

1. 你最在意的入口，本来就集中在这三类对象上；
2. 它们也是最容易做“强定位”的对象；
3. 先把这三类做好，整体体验会立刻改善。

实现要求：

1. 回答 display block 内如果包含 blockquote，就优先把入口挂给 blockquote；
2. 若同一 block 内有长引号句，则入口优先挂给该引号句；
3. 若同一 block 内有公式，则公式入口优先级高于普通解释句；
4. 同一 display block 不再强制“最多 1 个主入口”。
5. 只要 block 内存在多个来自原文的 direct evidence 对象，就应全部保留入口。
6. 对“公式 + 公式解释句”允许组成同一个 `claim group`，共享同一个 locate target；如果前端把它们渲染成两个独立显示单元，则两个显示单元都要有入口，但可以跳到同一个 grouped target。

### Step 3：把前端挂点对象从 paragraph/li 扩展到 quote 和 formula block

当前正文里已经能挂 `paragraph / list_item / blockquote`，下一步要明确把以下 DOM 作为一等挂点对象：

1. `blockquote`
2. `.katex-display`
3. 公式外层容器
4. 引号句所在的句级 span

这一步的目标不是再做一次文本猜测，而是：

1. 后端告诉前端“入口该挂给哪类对象”；
2. 前端只负责把 `❞` 放到那个对象右边；
3. 不允许再把 blockquote 的入口迁移到上面的栏目句。

验收标准：

1. blockquote 的右侧可见入口；
2. 公式块右侧可见入口；
3. 壳句旁不再抢占入口位置。

### Step 4：reader 增加按锚点类型的 strict 定位

不同 claim 类型需要不同的 reader 定位链路：

1. `quote_claim / blockquote_claim`
   - 先到 `primary_block_id`
   - 再按 `evidence_quote` 做块内 exact highlight
   - quote 失败时，只在 support + neighbor 中降级

2. `formula_claim`
   - 先按 `equation number`
   - 再按公式 token overlap
   - 再按同块解释文本
   - 禁止跳到只共享变量名的普通段落

3. `critical_fact_claim`
   - 先 exact quote
   - 再小范围 constrained fallback

这一步需要把现在的 “strictLocate + blockId” 升级成：

1. `blockId`
2. `highlightSnippet`
3. `anchorKind`
4. `anchorText`
5. `equationNumber`

### Step 5：把“入口必须存在”与“入口必须准确”拆成双指标

后面验收不能只看“有没有入口”，要同时看两组指标：

1. 入口覆盖率
   - direct 引号关键句覆盖率
   - direct blockquote 覆盖率
   - direct 公式块覆盖率
   - direct 公式解释句覆盖率

2. 入口准确率
   - 入口是否挂在目标对象右侧
   - 点击后是否命中原文强相关句
   - 首屏高亮是否就是用户想验证的内容

推荐新的验收门槛：

1. direct 引号关键句入口覆盖率 = 100%
2. direct blockquote 入口覆盖率 = 100%
3. direct 公式块入口覆盖率 = 100%
4. direct 公式解释句入口覆盖率 = 100%
5. strict 点击首屏强相关命中率 > 90%

### Step 6：最后再收紧生成模板

等入口规则稳定后，再回到生成端继续收口：

1. `fact_answer` 模式默认关闭“下一步”
2. `依据` 优先输出 blockquote、直引、编号公式、原文短句
3. 降低“文中提到：”“说明：”“表明：”这类外壳句的生成概率
4. 如果 evidence 不足，直接说“未命中可直接定位原句”，而不是生成一个壳句再配弱入口

## 13. 我建议你下一阶段实际开工的顺序

如果按投入产出比排优先级，我建议下一阶段这样做：

1. 先做 `Step 1 + Step 2`
   目标：把“关键引号句 / blockquote / 公式块”稳定识别成必须入口对象。

2. 再做 `Step 3`
   目标：前端真的把入口挂到 blockquote 和公式块右边，而不是挂到说明句上。

3. 再做 `Step 4`
   目标：reader 对 quote 和 formula 分开走 strict 定位。

4. 最后做 `Step 6`
   目标：减少回答里继续制造壳句和弱证据。

换句话说，接下来不要再以“普通句子挂点”作为主线了，而要切到：

1. 原文直引优先
2. blockquote 优先
3. 公式优先
4. 其他高价值事实句次之

## 14. 进展更新（2026-03-11）

本轮先按 C1 hardening 执行，未继续做阈值微调或样式微调。

### 14.1 已完成：后端 provenance hardening

已落地：

1. `must_locate=true` 现在要求同时满足：
   - `primary_block_id`
   - 非空 `evidence_block_ids`
   - `anchor_kind`
   - `anchor_text` 或 `evidence_quote`
2. 不满足上述条件的 segment 会在落库前自动降级，不再以 strict 入口形态暴露给前端。
3. provenance 根级新增：
   - `provenance_schema_version = 3`
   - `strict_identity_ready`
   - `must_locate_candidate_count`
   - `must_locate_count`
   - `strict_identity_count`
   - `identity_missing_reasons`
   - `identity_missing_segments`

已验证：

1. `blockquote_claim` 的 strict segment 会持久化 `primary_block_id / evidence_block_ids / anchor_kind / evidence_quote`
2. `formula_claim` 的 strict segment 会持久化 `primary_block_id / evidence_block_ids / anchor_kind / equation_number`
3. 缺 strict identity 的 must-locate segment 会被降级

### 14.2 已完成：reader equation visible-anchor binder

已落地：

1. reader 在 markdown 渲染后会收集可见 `.katex-display`
2. 按 `ReaderDocBlock(kind=equation)` 文档顺序绑定：
   - `data-kb-block-id`
   - `data-kb-anchor-id`
   - `data-kb-anchor-kind=equation`
   - `data-kb-anchor-number`
3. 有公式编号时优先按编号绑定，编号不足时再按顺序回退
4. strict formula locate 现在优先解析到可见公式块，不再满足于“只打开文档”

### 14.3 当前状态

已完成阶段：

1. `PAPER_GUIDE_LOCATE_C1_HARDENING_PLAN` 的 Step 1
2. `PAPER_GUIDE_LOCATE_C1_HARDENING_PLAN` 的 Step 2

下一阶段仍按原计划继续：

1. 补 `exact / block / weak` 结果态
2. 继续收紧 chat 侧 fallback containment
3. 最后再回到入口位置与生成模板收口

### 14.4 新发现并已修复：direct 证据句被可选入口配额吞掉

真实回放消息 `1394` 后确认，截图里那类“明显原文句子但没有入口”的根因不是原文缺块，也不是样式问题，而是 chat 侧 fallback 配额策略错误：

1. 历史消息还没有根级 `strict_identity_ready`，所以会走 legacy fallback locate。
2. 该 fallback 有全局 `5` 个入口上限。
3. 旧逻辑在判断当前 snippet 是否属于 direct provenance 之前，就先检查这个上限。
4. 结果是前面的总结句 / 概括句先占满名额，后面的 direct quote 虽然已经有精确 provenance，也会被直接压掉。

已收口的规则：

1. `5` 个入口上限现在只约束 optional fuzzy 入口。
2. direct provenance-backed 的 quote / formula 入口优先于这个上限判断。
3. 因此“依据”区里的原文直引句不会再因为前文已有若干普通入口而消失。

本地验证：

1. 对真实消息 `1394` 做 SSR 回放。
2. 注入会抢普通配额的 refs 候选后，仍可渲染出 `8` 个入口。
3. 其中包含截图中 `依据` 区的三条英文原句入口。

### 14.5 下一步计划调整：从“稀疏入口”切到“direct evidence 全覆盖”

结合你现在的目标、当前截图、以及本轮已经完成的测试/回放，下一阶段不应再以“减少入口数量”为主目标，而应切换为：

1. 每一个来自原文的 direct 句子都必须有入口。
2. 每一个来自原文的公式块都必须有入口。
3. 每一个来自原文的公式解释句都必须有入口；允许与公式共用同一个 grouped target。
4. optional fuzzy 入口继续保留，但只作为补充层，且仍可限额。

下一阶段的具体工作拆分如下。

#### 14.5.1 后端 contract 扩充

在现有 `must_locate / strict_identity_ready` 之上，再补一层“覆盖 contract”：

1. 新增 `locate_policy`：
   - `required`
   - `optional`
   - `hidden`
2. `quote_claim / blockquote_claim / formula_claim` 默认进入 `required`。
3. 新增 `equation_explanation_claim`，用于识别公式后紧邻的解释句、变量定义句、`where ... denotes ...` 一类句子。
4. 新增 `claim_group_id / claim_group_kind`：
   - `formula_bundle`
   - `quote_bundle`
5. 对“公式 + 解释句”：
   - 可作为同一 `formula_bundle`
   - 共享 `primary_block_id`
   - 共享 locate target
   - 但前端仍可在多个显示单元上暴露入口

#### 14.5.2 前端渲染规则重写

`MessageList` 下一阶段要从“抢 5 个入口”改成“双轨渲染”：

1. `required` 入口全部渲染，不参与全局 cap。
2. `optional` 入口才受 cap 控制。
3. 同一个 grouped target 可以被多个显示单元复用：
   - 公式块一个入口
   - 紧邻解释句一个入口
   - 二者点击后落到同一个 grouped locate target
4. dedupe 只允许压掉“同一显示单元上的完全重复入口”，不能再按 block 级粗暴折叠。

#### 14.5.3 Reader 行为补齐

reader 侧下一步不只是“落到公式块”，还要支持 grouped formula locate：

1. 先落到可见公式块。
2. 再高亮公式邻近的解释块或解释句。
3. 若点击的是解释句入口，也应先定位到同一公式 bundle，而不是只跳到普通段落。
4. quote / blockquote 继续按 exact quote 优先。

#### 14.5.4 测试计划补齐

当前已有测试主要覆盖 provenance 落库 contract，但还缺“渲染覆盖率”测试。下一阶段要补四类测试：

1. 后端单测：
   - `equation_explanation_claim` 能被识别成 `required`
   - `formula_bundle` 具备稳定 `claim_group_id`
2. 前端 SSR/渲染回归：
   - 真实消息 `1394` 的三条英文原句都必须渲染入口
   - 同段多 direct quote 时，入口数必须等于 claim 数
3. 公式回归：
   - 用真实公式消息（如 `1382`）验证公式块入口存在
   - 若回答中同时出现公式与解释句，两者都应有入口，且可共享同一 grouped target
4. 端到端人工回归：
   - 点击 direct quote 首屏命中原句
   - 点击公式/解释句首屏命中可见公式块

#### 14.5.5 新的阶段目标

下一阶段完成标志不再是“入口更少更干净”，而是：

1. direct evidence 覆盖完整。
2. grouped formula locate 可稳定工作。
3. optional fuzzy 入口不会再干扰 required 入口。

### 14.6 已完成：required/group contract 第一阶段落地

本阶段已先把后端 contract 与 strict 前端消费链路打通：

1. provenance schema 已升级到 `3`。
2. segment 新增：
   - `locate_policy`
   - `claim_group_id`
   - `claim_group_kind`
3. 公式邻近解释句现在会被升级为 `equation_explanation_claim`。
4. `formula_claim` 与 `equation_explanation_claim` 会共享同一个 `formula_bundle`。
5. 解释句会保留自己的显示入口，但点击时可共享同一个公式 bundle target。

本轮已验证：

1. Python 单测补充了 `equation_explanation_claim` 与 `formula_bundle` 回归。
2. `pytest tests/unit/test_task_runtime_provenance.py -q` 通过。
3. `web/` 下 `npm run build` 通过。

当前还没做的部分：

1. 真实消息 `1382` 的前端 SSR/渲染覆盖回归还没补成自动化。
2. grouped formula locate 的 reader 联动高亮还可以再细化到“公式块 + 邻近解释块”双高亮。

### 14.7 已完成：创新点 summary claim 不再误跳到定义段

最新真实回放里，消息 `1398` 的第一条创新点原先会错误跳到：

1. `blk_707c81a1c1ff_00028`
2. 文本是 `As the rendering primitive for 3DGS, a 3D Gaussian is defined as:`
3. 这和回答里的“首次实现从单帧 SCI 重建显式 3D 高斯表示”并不对应

本轮确认的根因不是 reader，而是后端 provenance 选块仍被词面重叠带偏：

1. `answer_hits` 里的长英文 contribution 段没有被拆成更细的 bullet snippet
2. segment 侧对中文 summary claim 做 block matching 时，真实 contribution list item 会被 `match_source_blocks` 的 raw floor 提前过滤
3. 即便前面选中了 contribution block，后面还有一层旧的 generic-heading guard 会再次把它清空
4. `defined as / proposed in [13] / related work` 这类方法定义段又因为 `3DGS / Gaussian` 词面重叠被错误抬高

已落地的修复：

1. `_collect_paper_guide_block_pool` 现在会把长 hit 文本拆成 bullet / sentence 级 snippet，再入 pool
2. summary/result claim 在 segment 侧允许更低 raw floor，再交给后续 support + semantic rerank 收口
3. 新增 summary-aware rerank：
   - 提升 `first / introducing / contribution / experiments demonstrate` 一类真实贡献块
   - 压制 `defined as / denotes / parameterized as` 一类定义块
4. generic heading 的 guard 现在对 summary claim 改成“必须有足够 semantic bonus 才能放宽”，不再无条件清空真实 contribution block

本地真实验证：

1. 直接重放消息 `1398`
2. 修复后第一个创新点落到 `blk_707c81a1c1ff_00021`
3. 其 evidence quote 为：
   - `The proposed SCIGS is the first to recover explicit 3D representations ...`
4. 不再跳到 `3D Gaussian is defined as`

本轮也新增了一个后端回归单测，专门防止“创新点总结跳去定义段”的回归。

### 14.8 下一步计划调整：先补 inline 公式入口与首击成功链路

最新真实回放 `1402` 说明，当前剩余问题已经更具体，不应再泛泛写成“继续优化公式 locate”，而应拆成三个明确缺口：

1. 回答里出现了来自原文的长公式，但 chat 侧没有给公式 token 本体挂入口。
2. reader 的公式绑定和 locate 仍然存在 ready-state race，导致部分入口需要点多次才首屏命中高亮。
3. “通用知识 / 非检索片段内容”这类非原文公式说明，仍可能被后端误升成 `formula_claim required`。

#### 14.8.1 前端优先项：inline 公式入口覆盖

下一步先把“公式本体有没有入口”补齐，而不是继续微调按钮位置：

1. `MarkdownRenderer` 增加 `inline_math` 级别的 locate token。
2. 识别对象至少包括：
   - `$...$` inline math
   - rehype/KaTeX 生成的 inline math span
   - 明确的 `公式(1) / Eq. (1) / Equation 1`
3. 只要该公式对应 provenance 中的 `formula_claim required`，公式 token 本体就必须暴露入口。
4. 不允许再只给整段说明句挂入口、而公式本体本身没有入口。

#### 14.8.2 输出层配合：长公式优先转成行间

长公式改成行间不是根修，但应该作为生成侧配合策略落入计划：

1. 只要答案在解释“公式(1)/(2)/Eq. (n)”或公式本身较长，就优先输出为 display math。
2. 公式解释句尽量单独成句或单独成段，不和长公式挤在同一行。
3. 短变量、短表达式仍可保留 inline。
4. 该策略只是提高可挂点和可见锚点稳定性，不能替代 inline 公式入口支持。

#### 14.8.3 Reader 优先项：首击成功的 locate 生命周期

下一步 reader 不应继续依赖“打开后立刻 locate + 少量 retry”，而要改成显式阶段链路：

1. `drawer open`
2. `markdown mounted`
3. `reader blocks ready`
4. `visible equation anchors bound`
5. `strict locate run`
6. `highlight ack`

对应实现要求：

1. 给每次点击分配新的 `locateNonce / requestId`。
2. 在 equation binder 完成前，不允许 strict formula locate 直接失败。
3. 验收标准不再是“多点几次最终能跳到”，而是“首次点击即命中可见公式块并高亮”。

#### 14.8.4 后端优先项：非原文公式 claim 降级

后端 contract 还要补一层“非检索公式不得进入 required”：

1. 明确标注为“通用知识”“非检索片段内容”“外部常识补充”的 segment，默认降级为 `hidden` 或 `optional`，不得进入 strict formula 入口。
2. `formula_claim required` 必须同时满足：
   - 可追溯的 evidence block / equation identity
   - 与 `anchor_text / evidence_quote / equation_number` 一致
3. 若只有“变量名重叠”而没有真实公式证据块，不允许产出 required formula locate。

#### 14.8.5 下一阶段测试与验收重写

下一步测试不再只看“有没有入口”，而要把 `1402` 纳入真实回归：

1. 前端渲染回归：
   - 真实消息 `1402` 中的长 inline 公式本体必须有独立入口
   - 同段解释句若单独显示，也必须有入口
2. reader 交互回归：
   - 单击一次即可落到 `eq_00001`
   - 首屏高亮命中可见公式块，不需要重复点击
3. 后端回归：
   - `1402` 中“通用知识，非检索片段内容”不得再产出 `formula_claim required`
4. 生成侧回归：
   - 长公式问答默认优先输出 display math
   - 公式解释句与长公式分行显示

下一阶段完成标志改为：

1. 每个 direct quote / blockquote / formula / formula-explanation 都有入口。
2. 长公式默认以更适合定位的 display 形态输出。
3. 公式 locate 首击成功，不再依赖重复点击。
4. 非原文公式说明不再混入 strict locate。

### 14.9 已完成：chat 侧 inline 公式入口覆盖第一阶段

本轮先只落地 `14.8.1`，目标是把“公式本体没有入口”的缺口先补上，不与 reader lifecycle 混做。

已落地：

1. `MarkdownRenderer` 新增 `inline_math` locate token。
2. raw text 中的 `$...$` inline 公式现在会被识别成公式级入口候选。
3. rehype/KaTeX 产出的可见 inline 公式节点也会被识别，并直接在公式 token 本体右侧挂入口。
4. `equation_ref / inline_math` 入口会以 `equation` 类型进入现有 strict slot 解析，不再按普通段落 token 对待。
5. 对于带公式 token 的段落 / 列表项，前端现在优先保留公式 token 自己的入口，不再总是退回成整段入口。

本轮验证：

1. `web/` 下 `npm run build` 通过。
2. 用真实消息 `1402` 中的长公式片段做本地临时 SSR 回放：
   - 公式 token 本体右侧已产出 `kb-md-locate-inline-btn`
   - 列表项本身不再退回成整段 locate 入口

仍待下一步：

1. reader 首击成功链路尚未改成 `bind ready -> locate -> ack`
2. `1402` 中“通用知识 / 非检索片段内容”误升为 required formula claim 的后端问题尚未收口
3. 长公式默认转行间的生成侧策略尚未接入输出链路

### 14.10 已完成：reader 首击成功链路第一阶段

本轮继续落实 `14.8.3`，先把 reader 里的“新请求”和“公式锚点绑定 ready”分离出来，不再让 strict formula locate 抢跑。

已落地：

1. `ChatPage` 在每次打开 reader 时都会生成新的 `locateRequestId`。
2. `PaperGuideReaderDrawer` 现在会把每次点击视为独立 locate request，而不是复用旧 effect 状态。
3. 对公式类定位，reader 会先等待 visible equation binder 进入 ready，再运行 strict locate。
4. 新请求进来时会先清掉上一次的高亮和 inline hit，避免旧结果残留。
5. drawer 顶部增加了短暂的绑定状态提示：
   - `正在绑定公式锚点`
   - 绑定到可见公式节点后再继续 locate

本轮验证：

1. `web/` 下 `npm run build` 通过。
2. payload/request 链路已贯通：
   - `ReaderOpenPayload` 新增 `locateRequestId`
   - `ChatPage -> PaperGuideReaderDrawer` 已按每次点击刷新请求 id

当前仍待下一步：

1. 后端还需把 `1402` 里“通用知识 / 非检索片段内容”这类 segment 降级，禁止进入 required formula locate。
2. 生成侧“长公式优先转行间、解释句分行”策略还未接入。

### 14.11 已完成：generic / non-source 公式段降级

本轮继续落实 `14.8.4`，把显式“通用知识 / 非检索片段内容”段落从 formula strict locate 链路中踢出去。

已落地：

1. `kb/task_runtime.py` 新增 explicit non-source segment detector。
2. 只要 segment 明确包含：
   - `通用知识`
   - `非检索片段内容`
   - `未出现在本文检索片段中`
   - `generic knowledge / non-retrieved` 等显式免责声明
   就不再允许升级成 strict formula locate。
3. segment claim 初判里，这类段落会被降为非定位型 segment，不再升级成 `formula_claim must_locate`。
4. coverage contract 层再次兜底：
   - 即便历史或旧路径里已经被判成 `formula_claim`
   - 也会被强制降成 `locate_policy = hidden`
   - 同时清空 `formula_bundle` 归组
5. 前端 `MessageList` 也补了 `hidden` 语义消费：
   - structured locate entries 直接跳过 `locate_policy=hidden`
   - provenance direct fallback 也不再把 hidden segment 纳入候选

本轮验证：

1. `pytest tests/unit/test_task_runtime_provenance.py -q` 通过，`17 passed`。
2. `web/` 下 `npm run build` 通过。
3. 用真实消息 `1402` 的旧 provenance 第 8 段回放：
   - 原先是 `formula_claim must_locate=true`
   - 现在经 contract 处理后变成：
     - `must_locate = false`
     - `locate_policy = hidden`
     - 无 `formula_bundle`

当前只剩下一步：

1. 生成侧把长公式默认转成行间，并把解释句尽量拆成独立句/段。

### 14.12 已完成：生成侧长公式输出约束补强

本轮把 `14.8.2` 的输出约束写进了 `paper guide` 生成提示，避免模型继续把长公式塞在一句话中间。

已落地：

1. `kb/task_runtime.py` 的系统提示现在明确要求：
   - 解释 `公式(n) / Eq. (n)` 时，优先输出行间公式
   - 只要公式包含 `\\frac / \\sum / \\int / \\mathcal / \\mathbf` 等复杂结构，就优先转成 `$$...$$`
   - 长公式尽量单独成行，不要塞进普通句子中间
   - `where` 句、变量定义句、解释句尽量另起一句或另起一条
2. `paper-guide formula grounding` 规则也同步加强：
   - 检索到的长公式或编号公式，公式本体与解释句分开输出
   - 不要把“长公式 + 解释”压缩成一个混合 prose 句子

本轮验证：

1. `pytest tests/unit/test_task_runtime_provenance.py -q` 通过，`17 passed`。

当前剩余风险：

1. 这一步属于 prompt guidance，不是强制后处理，因此仍需要你用真实问题继续手测输出形态。
2. 重点手测对象仍是：
   - `1402` 类长公式回答
   - `1382` 类“公式 + 解释句”回答
### 14.13 已完成：公式锚点检索不再把 References `[1]` 误当成 `公式(1)`

最新真实失败样本 `1405 -> 1406` 说明，当前坏结果不是“论文里没有公式”，而是 retrieval anchor 规则把参考文献编号 `[1]` 错当成了 `equation (1)`。

真实情况：

1. `chat.sqlite3 / message_refs.user_msg_id=1405` 的问句是：
   - `SCINeRF 的 NeRF 体渲染公式是哪条？请解释公式(1)以及后面的 where 句`
2. 对应 SCINeRF markdown 实际包含：
   - `C(\\mathbf{r}) = ... \\tag{1}`
   - 紧随其后的 `where t_n and t_f ...`
3. 旧规则却允许 `equation` anchor 同时匹配 `[(1)]` 与 `[1]`
4. 结果是 `References [1]` 被错误抬成 anchor-focused snippet，生成端才会说成“只检索到标题/参考文献，没有公式正文”

本轮已落地修复：

1. `kb/retrieval_engine.py`
   - `equation` anchor bonus / regex 不再把方括号 `[1]` 当成公式编号命中
   - 只保留 `Eq. 1 / Equation (1) / 公式(1) / 式(1) / \\tag{1}` 这类公式锚点
2. 同文件新增非 reference 问题下的 reference-snippet 过滤：
   - `References / Bibliography` heading 下的 snippet 不再混入公式问答的 answer pack
   - 多条 `[n]` 参考文献列表片段也会被过滤
3. 新增 retrieval 回归测试：
   - `tests/unit/test_retrieval_engine_doc_anchor.py`
   - 验证“同一文档同时存在 Eq.(1) 与 References [1]”时，最终 answer doc 必须优先返回公式正文和 where 句，而不是 reference item

本轮验证：

1. `pytest tests/unit/test_retrieval_engine_doc_anchor.py -q` 通过
2. `pytest tests/unit/test_task_runtime_provenance.py -q` 通过
3. 用真实 SCINeRF markdown 进行等价 anchor query 重放后：
   - 主 snippet 已切回 `\\tag{1}` 对应的公式正文
   - `ref_show_snippets` 不再混入 reference list，而是保留 `Eq.(1) + where` 与其所在方法段落

这一步完成后，paper-guide 在“编号公式问答”上的验收口径进一步明确为：

1. 公式编号只能命中真正的 equation anchor，不得被 citation index 劫持
2. 回答上下文里必须优先出现 `公式正文 + where 解释`
3. `References [1]` 只能在用户明确问参考文献时进入 answer pack

### 14.14 已完成：关闭 inline 小公式入口，并按 formula bundle 收敛重复入口

最新真实消息 `1408` 暴露出的主问题已经不是“能不能定位”，而是“定位入口粒度错误”：

1. 回答里很多 `$C(r)$ / $t_n,t_f$ / $T(t)$` 这类行内变量都出现了入口
2. 多个入口最终都指向同一个 `Eq. (1)` 可见公式块
3. `1408` 的 provenance 里，`seg_004 / seg_008 / seg_016` 都落到同一个 `formula_bundle:blk_29cad7662df5_00025`
4. 这会造成用户视角上的“满屏很多按钮，但都跳去同一个地方”

本轮已落地修复：

1. `web/src/components/chat/MarkdownRenderer.tsx`
   - 停用 raw `$...$` inline math token 的 locate 入口
   - 停用可见 inline KaTeX 节点的 locate 按钮
   - chat 侧只保留：
     - 编号公式引用（如 `公式(1) / Eq. (1)`）
     - display math / block formula
     - quote / blockquote / figure 这些真正的 evidence 单元
2. `web/src/components/chat/MessageList.tsx`
   - `buildStructuredProvenanceLocateEntries()` 现在会按 `formula_bundle` 收敛代表项
   - 同一 bundle 中优先保留真正的 `formula_claim`
   - 同一 strict render slot / formula bundle 现在只允许显示一个入口，不再让多个 token 同时占用
3. `kb/task_runtime.py`
   - `补充说明（generic knowledge / non-retrieved content）` 会打开一个 non-source scope
   - 该 scope 下后续相邻公式段也会一起降级为 `hidden`
   - 避免“补充说明里的离散化通用公式”再次漏成 `required`

本轮验证：

1. `pytest tests/unit/test_task_runtime_provenance.py -q` 通过，`18 passed`
2. `pytest tests/unit/test_retrieval_engine_doc_anchor.py -q` 通过，`5 passed`
3. `web/` 下 `npm run build` 通过
4. 开发服务已重启，`http://127.0.0.1:8000/openapi.json` 返回 `200`

这一阶段后的产品口径调整为：

1. 行内小公式/变量本身不再单独挂入口
2. 同一个公式 bundle 在同一答案视图中只保留一个代表性入口
3. “补充说明 / 通用知识”中的公式不再混入 strict locate

### 14.15 已完成：paper-guide 不再强制结构化输出

这轮把 `answer_contract_v1` 收回到“尊重用户开关”：

1. `paper_guide_mode` 不再把 `answer_contract_v1=false` 强制改回 `true`
2. `paper-guide formula grounding` 在 contract 关闭时只保留 evidence / formula grounding 约束
3. contract 关闭时，不再继续注入：
   - `3-4 sections`
   - `Evidence / Limits` 固定骨架
4. `KB miss` 场景下的自动 `Next Steps:` 也纳入同一个开关
   - contract 开启：仍允许自动补结构化 next steps
   - contract 关闭：保留自然回答，不再后处理拼回结构化段落

本轮验证：

1. `pytest tests/unit/test_task_runtime_answer_contract.py -q` 通过，`25 passed`
2. `pytest tests/unit/test_task_runtime_provenance.py -q` 通过，`18 passed`

新的产品口径：

1. `paper_guide` 仍然要求答案基于检索证据
2. 但是否输出 `Conclusion / Evidence / Limits / Next Steps` 由 `answer_contract_v1` 决定
3. 用户关闭结构化输出后，后端不得再以 paper-guide 为由恢复固定模板

### 14.16 已完成：reader 首击定位稳定化

这轮不改样式，只收 `reader` 里的点击时序问题。根因是：

1. drawer 打开动画、markdown 挂载、公式 `.katex-display` 绑定、strict locate 这几步之前仍然是并行 race
2. 老逻辑只做了少量 `requestAnimationFrame` 重试，超时过早
3. 第一次点击经常在 DOM 还没稳定前就停止定位，用户再点一次其实只是重新发起一次 locate

本轮已落地：

1. `PaperGuideReaderDrawer.tsx`
   - 新增 `drawerReady`，通过 `Drawer.afterOpenChange` 只在抽屉真正打开后开始 bind / locate
2. equation binder 改为：
   - deadline 窗口内持续重试
   - 监听 DOM subtree 变化自动重绑
   - 等可见公式数稳定后再标记 `equationBindingReady`
3. locate 流程改为：
   - deadline 窗口内持续重试
   - 监听 `data-kb-block-id / data-kb-anchor-id / class` 变化自动重跑
   - 成功后停止 observer/retry
   - 滚动改成双 RAF，减少抽屉打开瞬间 scroll 失效

本轮验证：

1. `web/` 下 `npm run build` 通过

### 14.19 已完成：equation binder 不再清空原始公式锚点

对最新真实消息 `1412` 的排查确认了一个更底层的根因：

1. `1412` 的 provenance 完整，`eq_00001 / blk_29cad7662df5_00025` 与 `p_00015 / blk_29cad7662df5_00026` 都存在
2. 同一篇 SCINeRF markdown 走前端 `MarkdownRenderer variant=reader` 的 SSR 后，目标 `data-kb-block-id / data-kb-anchor-id` 也确实存在
3. 问题出在 reader 的 `bindVisibleEquationAnchors()`：
   - 旧逻辑会先对所有 `[data-kb-anchor-kind=\"equation\"]` 执行清空
   - 这会把 markdown 渲染阶段原本就存在的公式 wrapper 锚点一起抹掉
   - 一旦后续 `.katex-display` 重绑稍慢，strict direct 就会在这段时间里完全失去目标

本轮修复：

1. binder 清理阶段现在只清自己加过的可见公式绑定
2. 原始 markdown wrapper 上的公式锚点不再被清空
3. 即使 `.katex-display` 重绑稍晚，strict direct 也仍可命中原始公式节点

本轮验证：

1. `web/` 下 `npm run build` 通过
2. 对同一篇 SCINeRF 文档做 SSR 检查，目标 block/anchor 仍存在于 reader DOM 中

### 14.18 已完成：strict locate 自动降级到 fuzzy（避免“只打开文件不跳转”）

针对“定位按钮只打开文件、不滚动高亮”的持续问题，这轮进一步把 strict locate 的失败策略改为可用性优先：

1. strict exact 在 deadline 内仍优先重试
2. 超时后不再直接停止，而是自动降级到 fuzzy locate
3. fuzzy 仍失败时，最后兜底滚动到文档首个可读块（heading/paragraph/equation/figure）

目的：

1. 避免因为 identity 轻微漂移导致“完全不跳转”
2. 保证每次点击至少有可见滚动反馈
3. 后续再继续收紧 exact 命中率，而不是牺牲基本可用性

本轮验证：

1. `web/` 下 `npm run build` 通过

当前验收口径补充为：

1. strict locate 应尽量首击成功
2. 公式 locate 不应依赖用户重复点击来等 binder 就绪
3. 高亮/滚动应在 drawer 打开完成后执行，而不是抢在动画阶段

### 14.20 已完成：reader 滚动定位改为显式滚动容器 + 公式来源注释去乱码

这轮针对最新真实消息 `1414` 收了两个仍然影响手测的缺口：

1. 公式来源注释不是模型输出乱码，而是 `ui/refs_renderer.py` 追加注释时写入了坏字符串。
2. reader 打开后“卡一下但不跳”的剩余风险，不再继续依赖 `scrollIntoView` 单一路径，而是改成先对 `kb-reader-content` 做显式滚动，再保留浏览器原生滚动作为补充。

本轮落地：

1. `ui/refs_renderer.py`
   - display math 的来源注释统一输出为 `（式(n) 对应命中的库内文献：filename.pdf）`
   - 不再输出旧的 `Open/Page` / 坏编码串
2. `api/chat_render.py`
   - 新增公式来源 label 清洗
   - 旧消息里已经落下来的 mojibake note 也会在渲染时自动抽取出真正的 `*.pdf` 文件名
3. `web/src/components/chat/PaperGuideReaderDrawer.tsx`
   - 公式块优先聚焦 `.katex-display`
   - locate 成功后先对 reader 自身滚动容器计算 `scrollTop`，再调用原生 `scrollIntoView`
   - 清理 reader 内剩余几处用户可见乱码文案

本轮验证：

1. `pytest tests/unit/test_chat_render_reference_notes.py -q`
2. `pytest tests/unit/test_task_runtime_provenance.py -q`
3. `web/` 下 `npm run build`

### 14.21 已完成：live browser 下的公式滚动失效根因定位与修复

这轮不再停留在静态分析，而是直接对最新真实消息 `1416` 做了 headless browser 回放。运行时证据表明，之前“打开 drawer 但完全不滚”的根因有两层：

1. `MarkdownRenderer` 在 reader 中会尝试对 display math 做静态 line-based anchor 分配，但 live browser 下这条路径会把公式块错误绑定成相邻 paragraph block。
   - 例如第一个 `.katex-display` 被错绑成 `blk_29cad7662df5_00034 / p_00020`
   - 而不是目标 `blk_29cad7662df5_00025 / eq_00001`
2. locate 流程在真正滚动之前就执行了 `setState`
   - `locateHint` 等 state 更新会触发 rerender
   - rerender 会把刚刚命中的公式 DOM 换掉
   - 结果就是滚动前的 target/focus node 失效，用户侧表现为“只打开、不滚动”

本轮修复：

1. `MarkdownRenderer.tsx`
   - reader 里的 display math 不再走静态 line-based anchor 分配
   - 公式块只交给 runtime visible-equation binder 处理，避免错绑到 paragraph
2. `PaperGuideReaderDrawer.tsx`
   - 每次真正执行 locate 前，都会对当前 DOM 重新执行一次 `bindVisibleEquationAnchors`
   - locate 过程中的 `locateHint` 更新后移，不再抢在滚动之前触发 rerender
   - 这样即使前面有过 DOM 波动，滚动时拿到的仍是当前 live equation block

本轮浏览器级验证（真实消息 `1416`）：

1. 点击公式入口后，reader `scrollTop` 从 `0` 变为 `4596`
2. 截图显示 drawer 已落到：
   - `3.1 Background on NeRF`
   - 公式 `(1)` 可见区域
3. 因此“完全不滚”的主问题已经在运行时回放中复现并收口

### 14.22 已完成：跳转后高亮保持

在 `14.21` 之后，浏览器级回放继续暴露了一个剩余问题：虽然已经能滚到目标区域，但高亮会在后续 rerender 后掉掉。

根因：

1. 高亮原先只依赖那一次 locate 成功瞬间加上的 `.kb-reader-focus`
2. 后续的 `locateHint` / binder / React rerender 会把当次命中的公式 DOM 替换掉
3. 对公式块尤其明显，因为它们没有稳定的静态 line-bound anchor，只能靠 runtime 重新解析

本轮修复：

1. `PaperGuideReaderDrawer.tsx`
   - 记录最后一次命中的 sticky highlight 信息：
     - `blockId / anchorId / anchorKind`
     - `anchorNumber`
     - `highlightSeed / highlightQueries`
   - reader 在后续 rerender 后会自动重新解析并补回高亮
   - 对 equation sticky highlight 增加一层基于公式内容和公式编号的回查，不再只依赖 blockId
2. `index.css`
   - reader focus 样式补了更明显的 `box-shadow`
   - exact phrase 命中 `.kb-reader-inline-hit` 也补了持久高亮底色

本轮浏览器级验证：

1. 对真实消息 `1416` 再次 headless 回放
2. 点击后 `scrollTop = 4596`
3. 5.5 秒后 `focusCount = 1`
4. 说明跳转后高亮仍然保留

### 14.17 已完成：reader locate 卡死回归热修

你反馈“点击定位按钮只能打开文件，不再跳转高亮”后，这轮做了回归热修，先恢复可用性：

1. `Drawer.afterOpenChange` 仍保留，但新增 `open` 后 240ms 的 ready 兜底
   - 目的：兼容某些环境里 `afterOpenChange` 未触发导致 locate effect 永远不跑
2. strict locate 的终止分支补上 `finishLocate()`
   - 目的：失败后明确停止 observer/retry，避免卡在重复重试循环
3. 以上改动不涉及 UI 样式，仅修复定位执行链路

本轮验证：

1. `web/` 下 `npm run build` 通过
### 14.23 In-Progress: `1418` figure jump + quote entrance audit

Real replay was run against message `1418` (`Figure 1` + abstract question).

Observed:

1. The stored provenance for `1418` had `figure_claim` bound to method paragraph `blk_29cad7662df5_00022`, not the figure/caption block.
2. The first direct quote blockquote had no entrance because the chat renderer used the whole blockquote (`label + quote`) as the strict snippet.
3. The second displayed quote still has no entrance because current stored provenance does not contain a direct quote segment for that sentence at all.

Implemented in this phase:

1. Backend figure-claim rebinding:
   - `kb/task_runtime.py`
   - direct `figure_claim` segments are rebound toward figure/caption blocks during coverage hardening
   - added regression test `test_build_paper_guide_answer_provenance_rebinds_figure_claim_to_figure_block`
2. Old-message normalization on read:
   - `api/chat_render.py`
   - message fetch now reapplies provenance hardening so existing messages can pick up figure rebinding without regeneration
3. Blockquote entrance recovery:
   - `web/src/components/chat/MarkdownRenderer.tsx`
   - blockquote locate now prefers the actual quoted span instead of the whole labeled blockquote
   - `web/src/components/chat/MessageList.tsx`
   - render-segment splitter now merges consecutive `>` lines into one blockquote segment
4. Figure highlight visibility:
   - `web/src/styles/index.css`
   - figure-anchor focus now styles the visible image wrapper

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit/test_chat_render_reference_notes.py -q`

### 14.25 已完成：strict inline locate 改成纯函数门控，修复真实消息 `1420` 的“入口被 render 吃掉”

这轮问题已经按真实页面回放定因，不再是 provenance 缺失。

真实根因：

1. `web/src/components/chat/MessageList.tsx` 里的 `canLocateSnippet()` 在 React render 期间用 `shownStructuredInlineKeys` 做去重。
2. 对真实消息 `1420` 的浏览器回放显示：
   - strict resolver 实际已经能为 `Figure 1` 和三条 direct blockquote 分别解析到 `seg_001/seg_002/seg_003/seg_004`
   - 但同一轮渲染里，第一次 render 先把这些 entry 标记为“已显示”
   - 第二次 render/commit 时又被 `blocked-shown` 拦掉，导致最终 DOM 里没有按钮
3. 这属于 render-time side effect，不是数据 contract 问题，所以继续调 provenance 也修不好。

本轮实现：

1. `web/src/components/chat/MessageList.tsx`
   - 去掉 strict locate 渲染期的可变 `Set` 去重，strict 入口改成纯函数门控
   - `blockquote_claim / quote_claim` 只允许挂在 `blockquote` 容器本身，不再让内部 `p` 抢到同一入口
   - `figure_claim` 只允许挂在显式 `Figure n / 图 n` 文本引用，不再让 caption / filename alt 混入同一 strict entry
2. 没有再改 UI 样式，也没有回退 strict data contract

真实回放验证：

1. 最新真实消息 `1420`
   - 三条 direct blockquote 现在都各有 1 个入口
   - 点击后 reader 命中分别为：
     - 第一条 quote -> `blk_29cad7662df5_00004`（Fig. 1 caption block）
     - 第二条 quote -> `blk_29cad7662df5_00085`
     - 第三条 quote -> `blk_29cad7662df5_00085`
   - 第二、第三条 quote 的 `.kb-reader-inline-hit` 都命中了对应英文原句
2. `Figure 1` 文本入口重新出现
   - 点击后 reader focus 命中 `blk_29cad7662df5_00003 / fg_00001`
   - focus 节点为带 `data-kb-anchor-kind="figure"` 的图像 wrapper

本轮验证：

1. `npm run build` in `web/`
2. `pytest tests/unit/test_task_runtime_provenance.py -q`
3. `pytest tests/unit/test_chat_render_reference_notes.py -q`
4. 浏览器级真实回放：
   - 检查 `1420` 的 quote/figure 入口数量
   - 实点 latest quote/figure 按钮并核对 reader focus/highlight
3. `npm run build` in `web/`
4. Browser replay confirms:
   - the first quote in `1418` now has a locate entrance
   - old-message provenance for `1418` is normalized from method paragraph to figure-caption evidence on fetch

Still open:

1. `1418` second displayed quote still has no entrance because its direct provenance segment was never persisted.
2. `Figure 1` mention in the answer text still needs a more robust strict-entry fallback when the render slot cannot be aligned from old stored content.

Next step:

1. backend multi-quote coverage for displayed direct quotes not yet captured as direct segments
2. strict figure-entry fallback for old messages when slot alignment fails but required figure provenance exists

### 14.24 已完成：`1418` direct quote 回绑 + 旧消息全文 block 归一化

这轮针对你刚才指出的两个具体症状继续收口：

1. 第一条原文直接引用跳转错位
2. 第二条原文直接引用没有入口

真实根因已经确认：

1. `1418` 的第二条 blockquote 其实有 direct segment（`seg_004`），但它被错误绑定到了 `blk_29cad7662df5_00037`
   - 该 block 是 `Given the NeRF representation ... synthesize the compressed image Y`
   - 与回答里显示的 conclusion quote 不匹配
2. 旧消息读取时 `api/chat_render.py` 只拿存量 `block_map`
   - 就算后端 hardening 能识别更好的 block
   - 也找不到不在旧 `block_map` 里的真实目标块，因此旧消息刷新后仍然修不回来

本轮实现：

1. `kb/task_runtime.py`
   - 为 `quote_claim / blockquote_claim` 增加 excerpt-aware rebind
   - 支持带 `[...]` / `...` 的跨句引文
   - 命中后重写：
     - `primary_block_id`
     - `primary_anchor_id`
     - `primary_heading_path`
     - `evidence_block_ids / support_block_ids`
     - `anchor_text / evidence_quote`
2. `api/chat_render.py`
   - 旧消息 display enrichment 时不再只依赖存量 `block_map`
   - 会按 `md_path` 加载全文 source blocks 参与 provenance hardening
   - 并把本次 hardening 重新引用到的 block 合并回 `block_map`

真实验证：

1. 对真实消息 `1418` 的本地重放结果
   - `seg_004` 已从错误的 `blk_29cad7662df5_00037` 改绑到 `blk_29cad7662df5_00085`
   - 该 block 即论文 `5. Conclusion` 中包含
     - `SCINeRF exploits neural radiance fields as its underlying scene representation`
     - `Physical image formation process of an SCI image is exploited ...`
2. 归一化后的 `block_map` 已包含 `blk_29cad7662df5_00085`
   - 因此旧消息刷新后，前端 strict/fallback 都有正确 target 可用

新增回归：

1. `tests/unit/test_task_runtime_provenance.py`
   - `test_apply_provenance_required_coverage_contract_rebinds_excerpted_quote_to_true_source_block`
2. `tests/unit/test_chat_render_reference_notes.py`
   - `test_enrich_provenance_segments_for_display_loads_md_blocks_for_quote_rebind`

本轮验证：

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit/test_chat_render_reference_notes.py -q`
### 15.1 Progress landed (2026-03-12, formula-bundle primary/secondary/hidden contract)

This phase closes the current formula-locate UX problem where several entrances inside one answer all land on the same equation block.

Implemented:

1. `kb/task_runtime.py`
   - provenance schema upgraded to `4`
   - formula-related segments now carry:
     - `formula_origin = source | explanation | derived`
     - `locate_surface_policy = primary | secondary | hidden`
     - `related_block_ids`
   - only one `formula_claim` per `formula_bundle` remains `required + primary`
   - duplicate / rewritten formula surfaces in the same bundle are downgraded to `hidden`
   - `equation_explanation_claim` now keeps its own prose block as primary target when it exists, instead of always collapsing back to the equation block
2. `web/src/components/chat/MessageList.tsx`
   - strict locate entry building now consumes `locate_surface_policy / formula_origin / related_block_ids`
   - hidden / derived formula surfaces are excluded from visible entrances
3. `web/src/components/chat/PaperGuideReaderDrawer.tsx`
   - reader payload now accepts `relatedBlockIds`
   - landing on an explanation can keep the related formula block highlighted as secondary context
4. `web/src/styles/index.css`
   - added secondary highlight styling for related bundle blocks

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit/test_chat_render_reference_notes.py -q`
3. `npm run build` in `web/`
4. offline replay on real messages:
   - `1416`: only the source Eq. (1) formula remains `primary`; the repeated surface is now hidden
   - `1420`: synthesized formula surfaces are downgraded instead of producing misleading entrances

Next focus:

1. browser-level replay on the latest live messages after refresh/reask
2. reduce `llm_refined` calls by moving formula / figure / direct-quote mapping to deterministic bundle-level matching

### 15.2 Progress landed (2026-03-12, quote-bundle dedupe + display-math-only formula entrances)

This phase closes the latest live-message regression where one answer still showed many repeated entrances even after the formula-bundle contract split.

Implemented:

1. `kb/task_runtime.py`
   - added duplicate-surface dedupe for `quote_bundle`
   - if multiple direct quote / blockquote claims in the same bundle point to the same block with the same normalized quote text, only one remains `required + primary`
   - the rest are downgraded to `hidden`
2. `web/src/components/chat/MessageList.tsx`
   - `formula_claim` is no longer allowed to bind onto paragraph / list-item render slots
   - strict equation locate no longer falls back from an unmatched display-math block to a generic formula entry
3. `web/src/components/chat/MarkdownRenderer.tsx`
   - removed inline `equation_ref` locate tokens
   - formula entrances now come from display-math blocks only

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit/test_chat_render_reference_notes.py -q`
3. `npm run build` in `web/`
4. offline replay on real message `1424` now shows:
   - only one visible `formula_claim` for Eq. (1)
   - the repeated `where ...` quote bundle is collapsed to one primary entrance
   - the corrected `T(t)` display formula remains hidden instead of reusing the Eq. (1) entrance
