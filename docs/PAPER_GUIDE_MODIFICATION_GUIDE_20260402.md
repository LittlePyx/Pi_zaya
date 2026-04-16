# Paper Guide 新修改指导文档（2026-04-02）

## 1. 目标定义：什么叫“更好”
“更好”不是回答更长，而是以下 6 个维度同时提升且可量化：

1. 问得到：用户问的点在文内明明存在时，不再被误判为“未命中”。
2. 引得准：文内引用号、引用条目、归因句子三者一致。
3. 跳得准：点击证据后优先命中正确 block/anchor，不漂移到相邻无关段。
4. 说得真：低置信时明确给出不确定性与原因，不伪造结论。
5. 跑得稳：不同论文、不同问法（中英、自然问法、精确问法）表现稳定。
6. 双轨增益：阅读指导提升的同时，转换质量不下降。

## 2. 当前基线（本轮已达成）

1. 检索链路已加入“置信度快照 + 分级救援”，不再只看粗检索是否有 hits。
2. citation lookup 已修复一类典型误判（Duarte 问题被 LiDAR[83] 抢分）。
3. 回归结果：
   - `pytest -q`：`758 passed`
   - `paper_guide_baseline_p0_v1`：`17/17 PASS`
   - 关键 E2E（reader/locate）通过

## 3. 下一阶段北极星指标（建议）

1. `first-click locate` 命中率：`>= 90%`
2. `citation ref_num` 正确率：`>= 95%`
3. “有文却未命中”率：`<= 5%`
4. baseline 套件：持续 `100% PASS`
5. 改动后全量单测：持续通过

## 4. 修改总策略（先后顺序）

### P1. 检索真值化（优先）
目标：减少“明明有内容却回答未命中”。

1. 保持“粗检索 -> 定向救援 -> 深读救援”分层，不允许粗检索失败即直接 `not_stated`。
2. 按 prompt family 使用不同闸门阈值：
   - `method / citation_lookup / equation / figure_walkthrough` 用严格闸门。
   - `overview / compare / strength_limits` 用宽松闸门。
3. 对每次救援记录 reason code（如 `target_miss`、`weak_signal`、`strict_family_sparse_hits`），用于后续阈值校准。

实施位置：
- `kb/paper_guide_retrieval_runtime.py`
- `kb/task_runtime.py`

验收：
- `baseline_p0` 全过。
- 新增失败回放样例不过度回退。

### P2. 引用判别专用化（第二优先）
目标：避免“引用号看起来合理但不是问题所指”的误选。

1. 对 `citation_lookup` 单独打分，不直接复用通用检索排序。
2. 强化焦点词邻域判别：
   - `cite for X` 里 X 的命中优先级高于标题泛词命中。
3. 保留可解释低置信输出：
   - 主答案给“首选 ref + 证据句”。
   - 低置信时附“候选 refs + 原因”。

实施位置：
- `kb/paper_guide_retrieval_runtime.py`
- `kb/paper_guide_answer_post_runtime.py`（低置信提示渲染）

验收：
- citation 套件正确率提升。
- 不引入 reference-list 误吸附回归。

### P3. 定位契约收敛（第三优先）
目标：同一条证据在 UI 内点击定位稳定可复现。

1. 后端统一输出定位契约字段：
   - `block_id`, `anchor_id`, `heading_path`, `claim_type`, `evidence_confidence`, `hit_level`
2. 前端严格按三段式降级：
   - 精确命中（block+anchor）-> block 命中 -> heading 命中
3. UI 明示命中级别，用户可见“是否精确命中”。

实施位置：
- 后端 provenance 组装链路
- `web/src/components/chat/reader/useReaderLocateEngine.ts`

验收：
- `paper-guide-locate-flow` 回归通过。
- “跳错但不自知”样例数下降。

### P4. 转换协同优化（需你审批后再改）
目标：提升阅读指导效果，同时保障转换质量不倒退。

1. 图像身份稳定：
   - 转换阶段建立 `figure_index -> asset_name` 稳定映射。
2. 引用结构稳定：
   - reference entry 解析后持久化规范字段（title/authors/year/doi）。
3. 文本完整性守卫：
   - 对正文截断、公式断裂、引用编号断链做质控检查并报告。

说明：
- 本项涉及转换代码，按你的要求必须先审方案后实施。

## 5. 每次改动的固定流程（执行纪律）

1. 先选 1-2 个“真实失败样例”定义目标，不盲改。
2. 先补最小回归测试，再改实现，再跑回归。
3. 每步提交“计划对照反馈”：
   - 改了什么
   - 为什么改
   - 验证结果
   - 剩余风险
4. 若改动触达转换链路，必须先走审批。

## 6. 推荐测试矩阵（每轮至少跑）

1. 单测：
   - `pytest -q tests/unit/test_paper_guide_retrieval_runtime.py`
   - `pytest -q tests/unit/test_task_runtime_bg_task.py`
2. 全量：
   - `pytest -q`
3. Paper-guide 回归：
   - `paper_guide_baseline_p0_v1`
   - `paper_guide_smoke_scinerf2024_v1`
   - `paper_guide_smoke_general_v1`
   - 一键回放池：`python tools/manual_regression/run_paper_guide_failure_replay_v1.py --base-url http://127.0.0.1:8016`
4. 前端定位：
   - `reader-regression.spec.ts`
   - `paper-guide-locate-flow.spec.ts`（建议限题先跑）

## 7. 下一步建议（可直接开工）

1. 建立“失败回放池 v1”：
   - 收集 20 条真实失败问句，覆盖 citation/method/figure/equation/overview。
2. 增加“低置信可解释输出”：
   - 不只在日志里有低置信，回答层也要带轻量提示。
3. 加入“命中级别埋点看板”：
   - 每日统计 exact/block/heading fallback 占比。
4. 若你批准，再进入转换协同项（P4）。

---

## 8. 下一阶段技术重构方案（2026-04-03）

### 8.1 重构目标
目标不是继续在后处理里堆更多 `if/else`，而是把 Paper Guide 收敛成一条强类型、可回归、可观测的流水线：

`GuideRequest -> Intent -> RetrievalBundle -> SupportPack -> GroundingTrace -> RenderPacket`

对应定义：
- `GuideRequest`：用户问题、当前论文、会话上下文、语言偏好。
- `Intent`：问题家族与目标对象，例如 `overview / method / figure_walkthrough / equation / citation_lookup / supplement`。
- `RetrievalBundle`：检索命中、support slots、cards、结构化索引命中。
- `SupportPack`：本次回答的正文、support records、可见引用策略、定位策略。
- `GroundingTrace`：回答每一段最终落到论文里的哪个 `block / anchor / figure / equation`。
- `RenderPacket`：前端直接消费的渲染结果、`cite_details`、canonical locate target。

### 8.2 推荐架构
建议把现有逻辑拆成“router + skills + contracts + grounder + converter indices”五层：

1. Router 层
   - 负责 prompt family、exact-support、beginner 模式、target figure/equation/panel 的判定。
   - 目标文件：`kb/paper_guide/router.py`

2. Skills 层
   - 每类问题由独立 skill 负责，不再共用一套大后处理逻辑。
   - 推荐 skill：
   - `OverviewSkill`
   - `MethodSupportSkill`
   - `FigurePanelSkill`
   - `EquationSkill`
   - `CitationLookupSkill`
   - `BeginnerSupplementSkill`
   - 目标文件：`kb/paper_guide/skills.py`

3. Contracts 层
   - 所有核心数据结构使用 `Pydantic v2` 强类型化，避免字段漂移、前后端语义不一致。
   - 目标文件：`kb/paper_guide/contracts.py`

4. Grounder 层
   - 统一 support resolution、segment grounding、canonical locate target 输出。
   - 后端直接产出稳定定位合同，前端尽量不再做语义补救。
   - 目标文件：`kb/paper_guide/grounder.py`

5. Converter Indices 层
   - 转换后不只产出 `.md`，还同时产出 `figure/equation/reference/anchor` 结构化索引。
   - 目标文件：`kb/converter/structured_indices.py`

### 8.3 核心 Contracts 设计
建议至少先定义以下模型：

```python
class GuideIntent(BaseModel):
    family: Literal["overview", "method", "figure_walkthrough", "equation", "citation_lookup", "supplement"]
    exact_support: bool = False
    beginner_mode: bool = False
    target_figure: int = 0
    target_panels: list[str] = []
    target_equation: int = 0

class SupportRecord(BaseModel):
    claim_type: str
    cite_policy: Literal["prefer_ref", "locate_only", "hidden"] = "locate_only"
    locate_policy: Literal["required", "optional", "hidden"] = "optional"
    source_path: str
    heading_path: str = ""
    block_id: str = ""
    anchor_id: str = ""
    locate_anchor: str = ""
    resolved_ref_num: int = 0
    candidate_refs: list[int] = []
    figure_number: int = 0
    panel_letters: list[str] = []
    equation_number: int = 0

class SupportPack(BaseModel):
    family: str
    answer_markdown: str
    support_records: list[SupportRecord] = []
    needs_supplement: bool = False

class GroundingTraceSegment(BaseModel):
    segment_id: str
    text: str
    primary_block_id: str = ""
    primary_anchor_id: str = ""
    heading_path: str = ""
    anchor_kind: str = ""
    anchor_number: int = 0
    claim_type: str = ""
    cite_policy: str = "locate_only"
    locate_policy: str = "optional"
```

要求：
- `SupportRecord` 是后端内部唯一真相来源。
- `cite_details`、`provenance segments`、前端 locate entry 都从 `SupportRecord / GroundingTrace` 派生，不再各自拼装语义。
- `raw [[CITE:...]]` 仅作为内部过渡标记，最终用户态只保留渲染后的 `[n]` 或 `locate-only`。

### 8.4 Skills 设计与显示策略
每个 skill 都要同时定义“回答职责”和“显示策略”。

1. `OverviewSkill`
   - 职责：解释论文解决什么问题、核心思路、主线贡献。
   - 默认策略：`cite_policy = hidden or locate_only`
   - 不主动展示外部 `[n]`，优先给段落定位。

2. `MethodSupportSkill`
   - 职责：解释 pipeline、训练细节、实现细节、数据处理步骤。
   - 默认策略：`locate_only`
   - 只有用户明确问“引用了哪篇方法”时才允许显式 `[n]`。

3. `FigurePanelSkill`
   - 职责：Figure N / panel (a)(b)(c) 的图注、子图解释、局部结论。
   - 默认策略：`locate_only`
   - 优先消费 `figure_index.json`，不优先依赖通用 BM25。

4. `EquationSkill`
   - 职责：公式、变量、相邻 prose 解释、公式所在章节。
   - 默认策略：`locate_only`
   - 优先消费 `equation_index.json`。

5. `CitationLookupSkill`
   - 职责：回答“某一点引用了哪条参考文献”。
   - 默认策略：`prefer_ref`
   - 这是唯一默认显式展示 `[n]` 的 skill。

6. `BeginnerSupplementSkill`
   - 职责：当原文提得很少时补一段 AI 自己的话。
   - 默认策略：不可伪装成文内证据。
   - 输出必须显式标注“以下为基于论文内容的补充理解，并非论文原文直接陈述”。

### 8.5 Converter 升级方向
转换链需要从“只生成 markdown”升级为“生成 markdown + 结构化索引”。

推荐新增产物：
- `assets/anchor_index.json`
- `assets/figure_index.json`
- `assets/equation_index.json`
- `assets/reference_index.json`

其中 `figure_index.json` 至少包含：
- `figure_number`
- `heading_path`
- `caption`
- `caption_continuation`
- `panel_clauses`
- `block_ids`
- `anchor_id`
- `page`

`equation_index.json` 至少包含：
- `equation_number`
- `equation_markdown`
- `normalized_tex`
- `context_before`
- `context_after`
- `block_id`
- `anchor_id`
- `page`

Converter 的优先规则：
1. 继续保留并增强 `Figure N` 后续 `a/b/c` continuation 合并。
2. 继续修复 OCR caption 断词、panel range 断裂。
3. 继续识别并还原“伪公式块”。
4. 为 figure / box / equation 统一生成稳定 `heading_path + anchor_id`。

### 8.6 前后端职责边界
后端负责：
- 意图判定
- 证据抽取
- support resolution
- canonical locate target
- 引用显示策略

前端负责：
- 渲染 `[n]` popup
- 根据后端返回的 canonical locate target 打开 reader
- 在可选备选项之间做有限 UI 降级

原则：
- 不再把“figure / equation / panel 语义判断”留给前端兜底。
- `MessageList.tsx` 最终应尽量只消费稳定的 `RenderPacket` 和 `locateTarget`。

### 8.7 推荐技术选型
必选：
- `Pydantic v2`：contracts 与运行时校验。
- `RapidFuzz`：caption clause、equation context、anchor 文本对齐。
- `pytest`：单测与回归。
- `Playwright`：reader locate 与文内参考渲染 E2E。

可选：
- `LangGraph`
  - 适用于把 `classify -> retrieve -> exact_resolve -> answer -> ground -> render` 做成可观测状态图。
  - 在 contracts 稳定前不建议先引入。

不建议：
- 过早引入重型 agent 编排来替代 deterministic grounding。
- 让 LLM 直接决定最终 `block_id / anchor_id`，而没有结构化索引和 deterministic 校验。
 
### 8.8 两周落地顺序
第一阶段，2-3 天：
- 新建 `contracts.py`
- 把现有 runtime 输出适配到 `GuideIntent / SupportRecord / SupportPack / GroundingTraceSegment`
- 这一阶段只做类型适配，不改外部行为

第二阶段，3-4 天：
- 新建 `structured_indices.py`
- 让 converter 输出 `figure_index / equation_index / anchor_index / reference_index`
- 把 `Figure N + panel continuation`、伪公式块、caption OCR 修复并入统一后处理

第三阶段，3-4 天：
- 新建 `router.py + skills.py`
- 先把 `citation_lookup / figure_walkthrough / equation` 三类 skill 独立出来
- 保持对现有 runtime 的兼容入口

第四阶段，2-3 天：
- 新建 `grounder.py`
- 把 support resolution、segment grounding、canonical locate target 统一到 grounder
- 收缩前端语义补救逻辑

第五阶段，1-2 天：
- 建立 beginner broad prompts、figure panel、equation、method exact-support、citation lookup 的回归池
- 加入 raw structured cite leak 检查
- 加入 PDF 对照抽检

### 8.9 验收指标
1. 图 / 公式类 `first-click locate` 命中率：`>= 90%`
2. `citation_lookup` 的 `ref_num` 正确率：`>= 95%`
3. broad beginner overview 的外部显式 `[n]` 噪声率：`< 5%`
4. raw `[[CITE:...]]` 最终展示泄漏：`= 0`
5. baseline + smoke + failure replay 回归持续通过
6. 抽检 PDF 页面时，caption / equation / panel 对齐错误持续下降

### 8.10 当前最推荐的立即行动项
如果只做一个最优先动作，推荐先落地：

1. `contracts.py`
2. `router.py`
3. converter 的 `structured_indices.py`

原因：
- 这三项能同时提升“文内参考准确性”“跳转定位稳定性”“转换对 guide 的友好程度”。
- 它们会把后续工作从“修补 bug”变成“调 policy”。
- 它们对现有功能最容易做到增量接入，不需要一次性推翻现有链路。

本文档用于指导下一轮实现，原则是：先稳住真值与定位，再做体验装饰；先可解释，再追求更激进召回。

### 8.11 当前实现进度（截至 2026-04-08）
已完成（合同收敛与前端消费方向）：
- 后端已将 Paper Guide 的关键渲染输出收敛进 `meta.paper_guide_contracts.render_packet`，并在渲染阶段补齐 `cite_details / locate_target / reader_open`，同时保留对旧字段的兼容投影。
- Render cache 已支持持久化与恢复 `render_packet`，避免缓存命中时丢失合同字段。
- API 支持 `render_packet_only`（paper_guide 模式下默认开启，且可通过 query 显式覆盖开/关），可在不破坏 UI 的前提下逐步“去旧字段”。
- 前端 `MessageList.tsx` 已优先消费 `RenderPacket`（正文渲染、复制文本、引用详情、定位打开 reader、notice）。
- 已加入 E2E 回归用例覆盖 `render_packet` 驱动的 locate 和 notice 展示。
- 前端消息拉取接口已支持显式传 `render_packet_only=1/0`，paper_guide 场景下会主动开启 `render_packet_only=1`，确保 contract-first 路径持续被压测。

待完成（两周落地顺序中的后续项）：
- 将 router/skills/grounder 继续按“可替换模块”拆分（目前仍以 `paper_guide_*.py` 为主，尚未完全迁移到 `kb/paper_guide/*.py` 布局）。
- Converter 已能产出 `structured_indices.py` 的 `figure/equation/anchor/reference_index.json`（建议继续完善字段与回归覆盖），并在 grounder/render 阶段优先消费。
- 建立失败回放池与更系统的回归集（beginner、figure panel、equation、method exact-support、citation lookup）。
  - 已新增失败回放池 v1 框架：`tests/replay/paper_guide_failure_pool_v1.jsonl` + `tests/unit/test_paper_guide_failure_replay_pool_v1.py`（可持续往 jsonl 追加真实失败样本）。
  - 已新增“自动采集回放用例”脚本：`python tests/replay/collect_paper_guide_replay_cases.py` 会从 `chat.sqlite3` 抽取 paper_guide 会话并追加到 `tests/replay/paper_guide_failure_pool_captured.jsonl`，用于快速积累真实样本。
    - 支持 `--failures-only --summary`：只抓“值得修复”的失败候选，并打印按 tag 的统计（must_locate/no_target/fallback）。推荐把失败候选单独写到 `tests/replay/paper_guide_failure_pool_captured_failures.jsonl`。
    - 可选运行 captured 回放：设置环境变量 `KB_RUN_CAPTURED_REPLAY=1` 后执行 `pytest -q tests/unit/test_paper_guide_failure_replay_captured.py`（注意：依赖本机 chat db 内容，默认跳过）。

备注：当前 contracts 的主实现仍在 `kb/paper_guide_contracts.py`，已新增 `kb/paper_guide/contracts.py` 作为兼容 import 路径，便于后续按文档结构逐步迁移。
同理，已补齐以下兼容 import 路径（均为薄封装 re-export，不改变现有逻辑）：
- `kb/paper_guide/router.py` -> `kb/paper_guide_router.py`
- `kb/paper_guide/grounder.py` -> `kb/paper_guide_grounding_runtime.py`
- `kb/paper_guide/skills.py` 目前为占位模块，待后续真正拆分 skills。
