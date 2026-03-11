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

1. 每个 display block 只取最高分 claim。
2. 每条回答最多显示 2-4 个入口，硬上限 5。

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
2. 若公式下面还有一句“其中 ... 表示 ...”，该解释句可以与公式共用同一个 claim group。
3. 点击公式入口时，reader 必须优先定位到同编号公式或同一公式块，不能只跳到提到相同变量名的普通段落。

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
4. 同一 display block 最多 1 个主入口，除非存在“公式 + 关键引号句”两个都必须保留的情况，此时允许 2 个。

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
   - 引号关键句覆盖率
   - blockquote 覆盖率
   - 公式块覆盖率

2. 入口准确率
   - 入口是否挂在目标对象右侧
   - 点击后是否命中原文强相关句
   - 首屏高亮是否就是用户想验证的内容

推荐新的验收门槛：

1. 关键引号句入口覆盖率 > 95%
2. blockquote 入口覆盖率 > 95%
3. 公式块入口覆盖率 > 95%
4. strict 点击首屏强相关命中率 > 90%

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
