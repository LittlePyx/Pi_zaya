# Paper Guide Question Coverage Plan (Beginner + Expert)

目标：让用户不管是“泛泛地读个大概”还是“精准定位某一句/某个引用/某个图注”，都能得到可靠结果，并且满足三条硬约束：
1) 内容必须来自绑定 Markdown（不编） 2) 文内引用必须可验证（不乱给） 3) 定位跳转必须落在回答声称的那一段（不漂移）

本文把“用户会问什么”先建模，再给出落地的系统方案、测试矩阵与验收门禁。

---

## 1. 两类用户画像与典型提问

### 1.1 科研初学者（Beginner）
核心诉求：快速理解、降低认知负担、知道怎么读、知道去哪找。

常见问题（示例）：
1. 这篇论文在解决什么问题？为什么重要？（overview）
2. 主要贡献/创新点是什么？一句话总结。（overview）
3. 方法整体怎么工作？能不能用“步骤+直觉”讲清楚？（method）
4. 关键名词/缩写（例如 RVT/APR/NEC/CNR）分别是什么意思？（definition，偏“在文中怎么用/怎么定义”）
5. 作者做了哪些实验来证明有效？结论是什么？（reproduce/strength_limits）
6. 图 1/表 1 在讲什么？每个 panel 的含义是什么？（figure_walkthrough）
7. 我应该按什么顺序读？哪些段落最关键？（overview + doc map）
8. 我记不清在哪里写了 X，能直接定位到原文吗？（locate）

### 1.2 科研老手（Expert）
核心诉求：精确、可复现、可对照、能抓住细节与证据链。

常见问题（示例）：
1. 这个结论的“精确支撑句”是哪一句？给我原文并定位。（method/strength_limits + exact support）
2. RVT/APR/某个 trick 引用的 prior work 是哪条 ref？在文中哪里归因？（citation_lookup）
3. 这个实验设置（样本、参数、硬件、对照、统计）具体怎么做的？（reproduce）
4. 作者的评价指标/公式是怎么定义的？变量含义在哪里写？（method + equation/definition）
5. 哪些 failure cases/limitations 被承认？哪些没说但你能从证据推断？（strength_limits，推断必须标注为推断）
6. 与某类 baseline 的 trade-off（分辨率/噪声/速度/光毒性）结论分别来自哪些图表/段落？（compare + locate）
7. 这篇工作的可复现性风险点有哪些？（reproduce + risk checklist）

---

## 2. “问题类型”到系统能力的覆盖矩阵

Paper Guide 的核心不是“让模型看全文”，而是把回答拆成两部分：
1) **回答内容**（可以是解释/总结/对比）  
2) **证据绑定**（support_resolution）：每条关键断言绑定到某个 SourceBlock（heading/block_id/anchor_id + locate_anchor），以保证可定位与可验真。

### 2.1 问题类型分类（推荐的 Router）
1. **Doc Map / 目录地图**：按章节给“一句话原文锚点”，用于快速浏览与查找入口。
2. **Overview**：问题/贡献/结果概览，要求给 2-4 条证据锚点，避免纯作文。
3. **Method**：原理与流程解释；当用户要求“exact supporting part”时，必须输出原文句子并定位（确定性更好）。
4. **Figure Walkthrough**：图/表/面板解释，定位必须落在 caption 或紧邻解释段落；面板字母必须一致。
5. **Citation Lookup**：X 的归因/ref 号；必须在同一句附近出现 inline ref，并输出 ref 条目（或至少 ref 号 + locate）。
6. **Reproduce**：复现要素（数据/硬件/参数/步骤）；允许结构化清单，但每个关键要素都要有证据绑定或明确“文中未给出”。
7. **Compare**：优势/代价/对比；每个对比结论需绑定到结果段落/图表证据。
8. **Strength & Limits**：证据最强处、局限、缺口；对“缺口”必须能指出“文中没有写到哪里”或“证据不足”。
9. **Locate-only**（任意家族的子模式）：用户问“在哪里写了 X”，优先返回 locate_anchor 而不是扩写解释。

### 2.2 输出策略（Beginner/Expert 都适用）
1. **证据闸门（Evidence Gate）**：当用户提问属于“事实/定位/归因/具体细节”时，如果检索不到可绑定证据，必须回答“文中找不到/未检索到”，并给出最接近的片段与跳转，而不是猜。
2. **确定性模式优先**：对于 doc map、exact method support、citation lookup 的关键路径，用解析后的 SourceBlock 做确定性提取，减少模型漂移。
3. **把“解释”和“证据”分层**：解释可以更自由，但证据必须是可匹配的原文锚点（locate_anchor），并能落到同一 block。

---

## 3. 架构落地计划（可执行里程碑）

### P0（已具备或本周可完成）：可靠性底座
1. 绑定论文后提供 **Doc Map（verbatim anchors）**：每个章节 1 条原文锚点 + 可跳转。
2. “exact supporting part”类 method 问题走确定性句子提取。
3. 引用与跳转统一从 `support_resolution` 生成与校验，杜绝“模型口头说有但定位不到”。
4. 中文查询至少保证能触发有效检索兜底（翻译后的 used_query 用于 deep-read/扫描）。

验收指标：
1. 对核心回归集（overview/method/citation/figure/locate/doc map）PASS 率 >= 95%（新能力先从 LSA/NatPhoton 两篇跑通）。
2. “硬约束问题”（定位/引用/归因）0 条“无证据硬答”。

### P1：覆盖更多“泛泛问题”，同时不牺牲证据
1. Beginner 友好输出模板：阅读顺序、关键术语表、3 句核心 takeaway（每句带证据锚点）。
2. 章节级索引（Doc Index）持久化：
   - 每个 heading 选 1-2 条“高信息句”（原文锚点）作为 section card
   - 检索先命中 section card，再落到具体 block（提高召回与导航）
3. “Definition/术语”能力：优先在文中找定义句（e.g. “we define…”），找不到就明确说明并给最接近使用上下文。

验收指标：
1. 初学者问题集（见第 4 节）中，overview/method/figure 的“理解性”人工评分 >= 4/5。
2. 证据绑定覆盖：回答中的关键句（自动检测）>= 80% 能匹配到 direct evidence segment。

### P2：专家细节（可复现与可对照）
1. Reproduce 模式强化：把“复现 checklist”结构化输出（数据/代码/参数/硬件/采集/重建/统计）。
2. Compare 模式：对比结论拆成“结论 + 证据 + 适用条件/限制”，并要求至少 1 个结果证据锚点。
3. Equation/公式解释子能力：
   - 公式块、变量定义、式号定位
   - 解释时引用变量定义段落，避免自由发挥

验收指标：
1. reproduction/compare/equation 套件 PASS 率 >= 90%。
2. 关键参数/设置类问题：若文中没有给出，必须明确标注“not stated”并给出最近证据上下文。

### P3：跨论文能力（可选）
1. cross-paper compare：同一问题在多篇文章里怎么做，输出对照表，但每行都要能回跳到各自证据。
2. 统一引用与 bibliographic identity（DOI/author-year 与 ref num 的一致性校验）。

---

## 4. 测试设计（自动化 + 人工）

### 4.1 自动化回归（Regression）
工具：`tools/manual_regression/paper_guide_benchmark.py`

建议建立三套套件：
1. **core_grounding_v1**：overview/method/citation_lookup/figure_walkthrough/locate/doc_map（必须全 PASS）
2. **beginner_reading_v1**：阅读顺序、术语表、3 句 takeaway（允许更主观，但要求至少 N 条证据锚点存在）
3. **expert_precision_v1**：exact support、reproduce checklist、equation 定位、compare 对比项（严格 locate/citation）

每条 case 至少包含：
1. `answer.contains_any/contains_all`（确保输出形态正确）
2. `locate.required + exact`（确保跳转能落到 block）
3. `citation`（需要时必须有，且 ref num 必须在候选集合内）
4. `structured_markers`（禁止泄漏 `[[CITE:...]]`/`[[SUPPORT:...]]`）

### 4.2 人工验收（QA Checklist）
每次发布前抽测 2 篇论文，每篇覆盖 10 个问题：
1. Doc map：章节锚点是否都能跳转到正确位置？
2. Overview：每条关键结论是否都能回到原文锚点？
3. Citation lookup：ref 号是否真在原文附近出现？条目是否匹配？
4. Method exact：是否给到“原文句子”，并且 locate 落在同一句附近？
5. Figure walkthrough：panel 字母与 caption 是否一致？跳转是否落在图注或说明段？
6. Reproduce：参数/设置缺失时是否明确说“文中未给出”？有没有“脑补”？
7. 中文提问：同一问题中英各问一次，证据绑定是否一致（允许解释不同，但定位必须一致）？

输出记录：每个问题记录 PASS/FAIL、失败原因（检索漏/定位漂/引用错/胡编/语言问题）。

---

## 5. 验收门禁（Release Criteria）

### 5.1 必须满足（硬门禁）
1. 任何“定位/引用/归因/具体细节”问题：不允许无证据硬答。
2. `core_grounding_v1` 全 PASS。
3. 跳转一致性：locate_anchor 必须能在绑定 Markdown 中匹配到同一 block（或明确记录 locate-only 策略）。

### 5.2 建议满足（软门禁）
1. Beginner 套件人工评分平均 >= 4/5。
2. Expert 套件中 reproduction/compare 的“缺失项标注”准确率 >= 95%。

---

## 6. 需要持续迭代的风险点
1. “中文问法多样性”导致 router 误判：需要持续补充中文触发词，并用回归集覆盖。
2. 章节结构不规整（heading 噪声、合并段落、图表转写异常）影响 doc map 与定位：需要在 converter 侧持续提升 SourceBlock 质量。
3. 多进程/热重载导致“改了不生效”：开发模式建议默认关闭 backend reload，或限制 reload-dir 避免被测试输出触发重载。

