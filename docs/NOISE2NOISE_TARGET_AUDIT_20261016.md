# Noise2Noise目标关系的证据校验交接

2026-10-16。仅处理库内论文阅读和证据追溯，保持BM25默认及全库路径；混合明确开启且
限选中文献/文献篮。保留上轮未提交的卡片数学修复，不改题目、标准、融合参数或reranker。

## 基线和确定性复现

main仍为 `2c454e29ab0fb4bea9dbfca63abab1f8071975c8`，
[CI 38767047986](https://github.com/LittlePyx/Pi_zaya/actions/runs/38767047986)成功。
本轮在已有 `codex/citation-display-math-r32` 分支继续，原有未提交修改全部保留。

从r33追踪文件精确取第47、248行的实际 `audit_and_repair_claim_evidence` 输入及参数，
重放两模式失败；不是重新生成一个更好的回答。第51、252行的完整finalizer输入也保留。
记录在 `test_results/citation_binding_20261014/n2n-target-r34-original-audit-events.json`、
`n2n-target-r34-baseline-replay.json`。后者初始历史比较字段受tuple/list容器差异影响，
不据该字段判定文本变更；`n2n-target-r34-baseline-comparison.json` 的序列化比较确认
两份完整审计结果均与历史一致，目标关系删除、支撑分数为0实际复现。

实际第2页块 `blk_6a93682babb1_00030`、anchor `p_00018` 长1401字符，包含L2限定、
输入条件目标分布、相同条件期望及训练目标零均值关系；冻结转换的分栏混入文本保持原样。
校验叠加却只留下292字符的条件摘录，L2限定已不在校验证据中，因此正确答案的 `$L_2$`
被数值校验拒绝。完整段落仍在 `source_passages`，该行的直接证据标记已在整理中丢失，
而顶层选中命中保留了标记。已有计划槽更新精确摘录时，也没有像新增槽一样保留完整上下文。

内存对照仅补入同一个实际源块的完整原文，分数从0变为8，原门槛5不变，两份关系均保留；
见 `n2n-target-r34-context-counterfactual.json`。这不代表已修改产品或新的生成验收。

## 修复与负例

`kb/n2n_target_condition.py` 仅为选中直接命中中block、anchor、页码完全相同的主段落继承
直接身份；其他未标记段落不能借用。重读实际源块，验证完整关系、源文及定位后，已有槽
也保留完整 `source_evidence_quotes`；展示仍使用精确关系原句。过时槽的页码、anchor或
heading不符时不授予新上下文，不跨来源补证据。

新增上下文暴露出普通数字匹配会借用章节1/3等数字支持错误L1/L3的风险，因此同时在
`kb/theory_evidence.py` 保守核对损失标识、噪声均值、条件/无条件期望、期望方向、训练
输入/目标及结果变化方向。这些是额外拒绝约束，没有降低分数、支撑门槛或跳过校验。

`n2n-target-r34-red.log` 为10失败、6通过；随后损失函数、反向/否定期望、错误均值、
训练输入替换、错误来源、主段落与过时计划定位负例，以及原有回归共203项通过。
源条件不完整、实际文件改变、同名不同路径和空候选等此前负例保持执行。

首轮完整后端5910通过、43跳过后，再将实际段落末尾的“corrupted inputs”纳入负例，
另发现输入描述、英文changing方向、源文not the same否定3处边界，见
`n2n-target-r34-input-negative-red.log`（3失败、19通过）。随后区分输入描述与替换动作、
保留目标替换定理的角色、检查明确否定及变化方向，定向最终205项通过。
首轮完整回归属于这一最后收窄前的代码，保留并另跑最终全回归，不将两者混为一轮。

用原r33两份完整finalizer输入重放，再运行最终渲染，两份“干净目标—条件期望相同—
最优参数不变”均保留、本地标号存在，卡片绑定实际第2页并展示完整条件及数学竖线。
见 `n2n-target-r34-full-finalizer-replay.json`、`n2n-target-r34-fixed-render.json`。
这是历史生成重放，不替代后续原24题的新鲜完整复测。

## 冻结复测

题集仍为 `docs/research_qa_independent_denoising_v1.json`，SHA256
`a4b01a3731fb3450f95bf764d20751823b504c5df5681d362cc1b39ca49b107b`。
语料 `F:/pz-independent-20261013` 不改写、不重转；测试副本仍为
`F:/pz-citation-product-20261014`。使用新的唯一轮次 `citation-binding-r34`，305个Python
文件已同步，清单SHA256为 `50d63800c1d284cf23a29390514b0b735bb1358afe0e408a4ffaee1213496251`。
85个前端产物、357个源文件与r33已测试构建逐字节相同，未改前端或重建；新核对清单
SHA256为 `41da88d656f09b98a945eaacec17825717601ec9acbe5f61dcb4555c2a233adf`。
分别见 `stage-source-manifest-r34.json`、`stage-frontend-manifest-r34.json`。
运行期间源码与前端保持固定，最终实测结果在实际完成后记录，不用预期替代通过。

r34最后收窄后的后端5912通过、43跳过，浏览器核心131、101条质量门禁通过。
但同义正例在实际第2页完整源块上确定性复现：
“target distributions can be changed ... same conditional expected values” 被误判成改变参数，
分数为0，见 `n2n-target-r34-distribution-change-failure.json`。
因此停止r34精确runner及其后端直接子进程并等待实际退出，未覆盖运行中源码。
14个已完成BM25回答（含目标题）均通过原自动合同，但混合尚未运行，不算完整配对。
305源码及85构建/357前端文件在停止前后核验一致，见 `r34-interrupted-integrity.json`、
`r34-interrupted-bm25-cases.json`、`r34-source-overlay/`；所有数据保留。

新增正反例先得到3失败、23通过，见 `n2n-target-r35-outcome-role-red.log`。
变化检查随后限定为“改变最优参数／所学映射”，并识别该动作自身的否定；改变目标分布
不再被当成改变最优参数，条件期望不变也不能借此支持参数改变。定向最终210项通过。

最终重新同步并使用唯一轮次 `citation-binding-r35`，记录 `stage-source-manifest-r35.json`、
`stage-frontend-manifest-r35.json`。前端仍与已测试构建逐字节相同；未改变题目、词组、
workers或超时。r35完整运行及实际点击进程结束后再核对散列、记录结果，不用r34已完成
的14题代替最终原24题。

## r35最终结果与剩余失败

r35原48次回答、双模式阅读器和全部附加进程实际完成、退出。305个Python源码、85个
产物、357个前端源文件、3个验收/追踪工具及题集散列均保持固定，语料未变；最终核对
及源码快照为 `r35-final-integrity-check.json`、`r35-source-overlay/`。
源码清单SHA256为 `956a0a4a5bd85b988888e94c7ed15a1345e2eb54c650c9f848dfca03db99b8b8`，
前端清单SHA256为 `02e23a1c94020735272f33e4e8b1afafca55778547c109a1d1454b12670ff01e`。

| r35原24题检查 | BM25 | 明确请求局部混合 |
|---|---:|---:|
| 自动回答/引用/卡片合同 | 23/24 | 24/24 |
| 答案期望词组完整性 | 23/24 | 24/24 |
| 所需论文均有直接本地卡片 | 24/24 | 24/24 |
| 显示/复制未绑定号 | 0 | 0 |
| 阅读器精确定位 | 24/24 | 24/24 |
| 每题一条原PDF精确短语/页码 | 24/24 | 24/24 |
| 实际使用混合路径 | 0/24 | 24/24 |

BM25 `n2n-targets` 缺冻结的“干净／清洁／无噪”词组：初稿就用了“真实目标”，校验前后
都没有这些词，不是r33那种先有关系再被删。该失败及逐阶段输入输出保存在
`n2n-target-r35-bm25-failure-trace.json`，不改标准、不补字或重新生成凑通过。
混合实际初稿有这些词，最终仍保留；见 `n2n-target-r35-live-target-chain.json`。
最终r35源码另重放原r33两份生成并渲染，原目标关系、完整摘录及第2页绑定仍保留，见
`n2n-target-r35-full-finalizer-replay.json`、`n2n-target-r35-fixed-render.json`。

附加实际界面检查见 `n2n-target-r35-fresh-reader/`、
`n2n-target-r35-passage-coverage-audit.json`，四个场景不计入原冻结24题合同：

- BM25目标题在原5秒的冻结词组断言处失败，其第2页独立reader检查精确；不据能定位
  将词组失败改为通过。
- BM25污染模型题两张第2/8页卡片、混合目标题一张第2页卡片，显示、剪贴板及实际聊天
  点击精确定位共3张通过。混合目标题的带噪/干净目标句在实际回答中链接到该卡片，
  卡片的 `answer_claims` 列表保留此绑定，不能只检查首句预览 `answer_claim`。
- 混合污染模型题的引用计划已有第2、8页，但最终卡片只有第8页，条件句仍复用第8页
  卡片，缺第2页条件卡片。附加reader检查为 `no_local_card`；聊天脚本因所需第2页
  卡片不存在而报 `undefined anchor`。截图、实际卡片和失败日志保留，不算通过。

这说明论文级来源覆盖48/48、每题首卡定位48/48不等于逐句条件覆盖。
83张本地卡片路径均存在、范围正确；50张带块ID的实际块存在且页码一致，33张为短语
定位，见 `r35-card-block-summary.json`。这些元数据检查不验证每张卡片是否选对支撑句。
7次含来源公式的实际校验未丢公式，见 `r35-math-retention-audit.json`，不等于N2S冻结
语料中的残缺条件或图片公式已恢复，也不是人工正确率或完整entailment。

最终后端5917通过、43跳过，相关210项通过；最终源码的浏览器核心131项、101条质量
命令、Ruff、架构预算和空白检查通过。受保护文件相对main未增长。完整索引为
`frozen-24-r35-summary.json`、`frozen-24-r35-report.json`、`binding-audit-r35.json`、
`pdf-audit-r35/`。r32-r35全部中间失败及原始记录不覆盖。

本轮未提交、推送、合入main或发布，远端main仍为 `2c454e29`。
原始删除路径已修复，但完整验收仍有生成词组及条件卡片覆盖失败，因此暂不更新main。
下一步优先追踪混合污染模型题的引用计划到最终卡片映射，再处理目标角色生成完整性；
不把本轮比较题通过说成其错误编号波动已经修复。
