# 文内参考双系统方案

## 概述

在回答中同时支持两种引用，用不同的标记格式区分：

| 标记 | 系统 | 含义 | 点击弹出 |
|---|---|---|---|
| `[1]` | A — 片段定位 | "这段信息来自库里的第 1 篇论文" | 来源论文名 + 章节 + 片段预览 |
| `[[CITE:sid:58]]` | B — 参考文献追踪 | "这篇论文自身引用了第 58 条参考文献" | 该参考文献的标题/作者/年份/DOI |

---

## 系统 A：`[n]` 片段定位（已完成 ✅）

### 工作流程

```
LLM context:
  DOC-1 [SID:s9a2b3c4] NatPhoton-2019 | Abstract
  Single-pixel imaging (SPI) utilizes the second-order correlation...

LLM answer:
  SPI is a technique that reconstructs images from 1D signals [1].

Renderer:
  [1] → _resolve_n_from_hits(1) → canonical_paths[0] → NatPhoton-2019 Abstract
       → markdown: [1](#kb-cite-xxx "NatPhoton-2019 | Abstract")

用户点击 [1]:
  弹出卡片: 来源: NatPhoton-2019, 章节: Abstract, 片段预览...
```

### 已就绪的组件

| 组件 | 文件 | 状态 |
|---|---|---|
| 条件化 prompt（经典 RAG 用 [n]） | `paper_guide_message_builder.py:49-76` | ✅ 已修复 |
| canonical_hit_paths 存储 | `task_runtime.py` | ✅ 已修复 |
| `_resolve_n_from_hits` 解析 | `refs_renderer.py:3631` | ✅ 已实现 |
| strict 剥离（越界 [n] 自动隐藏） | `refs_renderer.py` | ✅ 已实现 |
| 测试 | 772 测试通过 | ✅ 已验证 |

---

## 系统 B：`[[CITE:sid:ref_num]]` 参考文献追踪（待实现）

### 工作流程

```
LLM context:
  DOC-1 [SID:s9a2b3c4] NatPhoton-2019 | Abstract | candidate refs: 3, 12, 31, 46, 58, 59
  ...as demonstrated by Sen et al. at Stanford University [3,58]...

LLM answer:
  SPI was first demonstrated by Sen et al.
  [[CITE:s9a2b3c4:3]][[CITE:s9a2b3c4:58]].

Renderer:
  [[CITE:s9a2b3c4:3]] → _resolve_struct_token → sid=s9a2b3c4, ref_num=3
       → lookup references_index.json[docs][source_key][refs]["3"]
       → markdown: [3](#kb-cite-xxx "Sen et al. 2009, Stanford University")

用户点击 ³:
  弹出卡片: Sen et al., 2009, Stanford University, DOI: 10.xxxx/xxxxx
              [查看详情] [加入文献篮]
```

### 视觉区分

| 系统 | 原始标记 | 渲染后 | 含义 |
|---|---|---|---|
| A — 片段定位 | `[1]` | `[1]`（蓝色方括号链接） | "这段内容来自库里的第 1 篇论文" |
| B — 参考文献追踪 | `[[CITE:sid:58]]` | ⁵⁸（上标链接） | "这篇论文引用了第 58 条参考文献" |

**实现方式：** `_resolve_struct_token` 末尾将 `_mk_cite_link_md` 替换为 HTML `<a href="#anchor" title="..."><sup>58</sup></a>`。Streamlit 渲染 markdown 中的 HTML 标签时，`<sup>` 正常生效。

### 现有基础设施

| 组件 | 文件 | 状态 |
|---|---|---|
| `candidate_refs` 提取 | `inpaper_citation_grounding.py:58` | ✅ 已有函数 |
| `_cite_source_id` (path→SID) | `paper_guide_shared.py` | ✅ 已有 |
| `references_index.json` | `db/references_index.json` | ✅ 已有（21 篇论文） |
| `resolve_reference_entry` | `reference_index.py:1908` | ✅ 已有 |
| `_resolve_struct_token` | `refs_renderer.py:3569` | ✅ 已有（paper_guide 用） |
| `_normalize_reference_for_popup` | `refs_renderer.py:3249` | ✅ 已有 |
| `_citation_hover_title` | `refs_renderer.py:2880` | ✅ 已有 |

### 待实施

#### B1. Context 中开启 candidate_refs 提取

**文件：** `paper_guide_context_runtime.py:147`

**当前：**
```python
if paper_guide_mode and src:
    candidate_refs = extract_candidate_ref_nums_from_hits(...)
```

**改为：**
```python
if src:  # 移除了 paper_guide_mode 条件
    candidate_refs = extract_candidate_ref_nums_from_hits(...)
    cue_texts = extract_candidate_ref_cue_texts(hit, ...)
    if candidate_refs:
        refs_txt = ", ".join(str(int(n)) for n in candidate_refs[:6])
        header += f" | candidate refs: {refs_txt}"
```

这样经典 RAG 的 context header 也会显示：
```
DOC-1 [SID:s9a2b3c4] NatPhoton-2019 | Abstract | candidate refs: 3, 12, 31, 46, 58, 59
```

#### B2. Prompt 中增加 [[CITE:...]] 指令

**文件：** `paper_guide_message_builder.py`，在 `else`（经典 RAG）分支中追加

```python
else:
    system += (
        "\nCitation rule:\n"
        "- Retrieved snippets are labeled DOC-1, DOC-2, etc. in the context.\n"
        "- When citing information from a snippet, use [1] [2] markers matching the DOC number.\n"
        "- Example: \"The method achieves state-of-the-art results [1].\"\n"
        "- Never mention DOC-k labels directly in the visible answer; use [n] markers instead.\n"
    )
    # + 新增的参考文献追踪指令
    system += (
        "\nPaper reference tracking:\n"
        "- Retrieved snippets may contain in-paper reference markers like [3,58] from the original paper's bibliography.\n"
        "- When you mention or repeat such a reference number from the paper text, "
        "MUST use [[CITE:<sid>:<ref_num>]] format.\n"
        "- Example: write \"as shown by Sen et al. [[CITE:s9a2b3c4:3]]\" instead of \"as shown by Sen et al. [3]\".\n"
        "- Only use [[CITE:...]] for numbers that already appear as reference markers in the original paper text; "
        "do NOT use it for your own snippet citations (which still use [n]).\n"
    )
```

#### B3. 渲染器：解除 [[CITE:...]] 与 [n] 的互斥

**文件：** `refs_renderer.py:3727-3729`

**当前：**
```python
if structured_seen:
    return seg2  # ← 只处理了 [[CITE:...]]，[n] 被跳过
return _INPAPER_CITE_ANY_RE.sub(_repl_any, seg2)
```

**改为：**
```python
# [[CITE:...]] 和 [n] 各自处理，互不阻塞
if structured_seen:
    seg2 = _INPAPER_CITE_ANY_RE.sub(_repl_any, seg2)
    return seg2
return _INPAPER_CITE_ANY_RE.sub(_repl_any, seg2)
```

这样同一段文本可以同时包含 `[[CITE:sid:3]]` 和 `[1]`，各自按自己的逻辑解析。

#### B4. 参考文献弹窗卡片与文献篮

**状态：已有完整实现，无需改动 ✅**

弹出卡片和文献篮功能已存在：
- **弹出卡片：** `web/src/components/chat/CitationPopover.tsx` — 显示标题/作者/来源/DOI，包含"加入文献篮"按钮
- **文献篮面板：** `web/src/components/chat/CiteShelf.tsx` — 完整的收藏/去重/搜索/排序/标签/笔记/快照/导出功能
- **前端状态管理：** `web/src/components/chat/citationState.ts` — localStorage 持久化，跨标签页同步
- **后端数据嵌入：** `_render_inpaper_citation_details`（`refs_renderer.py:3785`）将引用元数据以 JSON payload 嵌入页面

`[n]` 和 `[[CITE:...]]` 生成的弹出卡片共用同一个渲染机制，区别在于：
- `[n]` → `_resolve_n_from_hits` → 弹出卡片显示来源论文名 + 章节 + 片段
- `[[CITE:...]]` → `_resolve_reference_entry_from_index` → 弹出卡片显示该参考文献的标题/作者/年份/DOI

#### B5. 验证方案（详细设计）

### 设计目标

确保用户网页上点击引用链接看到的效果，与内部数据完全一致，没有错误的、缺失的或乱给的链接。

### 核心验证逻辑

渲染器产生两类输出，网页端读取它们来显示弹窗：

```
渲染器输出
 ├── 可见部分: markdown/HTML（用户看到的 [1] 和 ⁵⁸ 链接）
 └── 隐藏部分: <div class="kb-cite-data" data-kb-cite="anchor" data-kb-payload="{...}">
                 ↑ 网页 JS 读取这个 JSON 来渲染弹窗卡片
```

所以验证的关键是：**验证隐藏 payload 中的数据正确**，因为网页弹窗内容完全由 payload 决定。可见链接只要锚点 ID 匹配，点击就一定弹出正确内容。

---

### 测试 1：渲染器输出格式测试（纯单元测试）

**输入：** 含有 `[1]` 和 `[[CITE:sid:58]]` 的模拟答案文本 + 模拟 hits 数据

**步骤：**
1. 调用 `_annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="test")`
2. 解析返回的 `(out_html, cite_details)` 元组

**断言：**

| 检查项 | 方法 |
|---|---|
| `[1]` 渲染为 markdown 链接 | `out_html` 中包含 `[1](#kb-cite-test-...)` 模式 |
| `[[CITE:sid:58]]` 渲染为 HTML 上标 | `out_html` 中包含 `<a href="#kb-cite-test-..."><sup>58</sup></a>` 模式 |
| 同一段中两种标记共存 | 同一行内同时匹配上述两种模式 |
| 锚点 ID 格式正确 | 所有 `href="#..."` 匹配 `#kb-cite-test-[a-z0-9]+$` |
| 每个锚点都有对应 payload | `cite_details` 中每个 anchor 都能在 `out_html` 的 `data-kb-cite` 中找到 |
| payload 中的 num 正确 | `detail["num"] == n` |
| 越界 [999] 被剥离 | `out_html` 中不含 `[999]` |
| 无效 SID [[CITE:bad:1]] 被剥离 | `out_html` 中不含 `[[CITE:bad:1]]` |

**测试用例矩阵：**

```
用例 1: "[1]" → 3 hits, canonical_paths=[A,B,C]
  预期: [1] → hits[0] 的 source_path == A ✅

用例 2: "[2]" → 3 hits, canonical_paths=[A,B,C]
  预期: [2] → hits[1] 的 source_path == B ✅

用例 3: "[1] and [[CITE:s9a2b3c4:58]]" → same segment
  预期: [1] 以 markdown 链接存在，<sup>58</sup> 同时存在 ✅

用例 4: "[[CITE:s9a2b3c4:3]][[CITE:s9a2b3c4:58]]" → adjacent superscripts
  预期: 两个相邻 <sup>3</sup><sup>58</sup> 链接 ✅

用例 5: "[999]" → 只有 3 个 hits
  预期: 被 strict 剥离，输出中没有 [999] ✅

用例 6: "[[CITE:nonexistent:1]]" → SID 不存在
  预期: 被剥离 ✅

用例 7: (空字符串) → 无 hit
  预期: 输出为空字符串 ✅
```

---

### 测试 2：Payload 数据正确性测试（数据完整性验证）

这是最关键的测试——保证网页弹窗显示的数据是对的。

**验证 pipeline：**

```
                   ┌──────────────────────────────┐
                   │  cite_details (Python dict)    │
                   │  [{num, source_name, title,   │
                   │    authors, year, doi, ...}]   │
                   └──────────────┬───────────────┘
                                  ↓
                   ┌──────────────────────────────┐
                   │  JSON 序列化 → payload 字符串  │
                   │  data-kb-payload="{...}"      │
                   └──────────────┬───────────────┘
                                  ↓
                   ┌──────────────────────────────┐
                   │  React 解析 JSON → 弹窗卡片    │
                   │  CitationPopover.tsx           │
                   └──────────────────────────────┘
```

**验证方法：**

对于 System A `[n]` → 片段定位 payload：
```
{
  "num": 1,
  "source_name": "NatPhoton-2019.pdf",     ← 来自 hits[0].meta.source_path → display name
  "source_path": ".../NatPhoton-2019.pdf",  ← 来自 canonical_paths[0] 或 hits[0]
  "raw": "Single-pixel imaging...",          ← 片段文本
  "cite_fmt": "",                            ← 片段定位没有论文元数据
  "title": "", "authors": "", ...            ← 空，因为没有参考条目
}
```

验证：每个 `[n]` 对应的 payload 中：
- `num == n`
- `source_path` 等于 `canonical_paths[n-1]`（或 `hits[n-1].meta.source_path`）
- `source_name` 等于 `source_path` 的文件名
- `raw` 非空（有实际片段内容）

对于 System B `[[CITE:sid:ref_num]]` → 参考文献 payload：
```
{
  "num": 58,
  "source_name": "NatPhoton-2019.pdf",
  "source_path": ".../NatPhoton-2019.pdf",
  "raw": "Sen et al., 2009, Stanford University...",
  "cite_fmt": "Sen et al. (2009). Single-pixel...",
  "title": "Single-pixel imaging via compressive sampling",
  "authors": "Sen, P. et al.",
  "venue": "Nature Photonics",
  "year": "2009",
  "volume": "3",
  "issue": "5",
  "pages": "291-295",
  "doi": "10.1038/nphoton.2009.58",
  "doi_url": "https://doi.org/10.1038/nphoton.2009.58"
}
```

验证：将 payload 中的每个字段与 `references_index.json` 中对应的条目逐一比对：
- `num == ref_num`
- `source_path` 与 SID→source_path 映射一致
- `title`、`authors`、`year`、`doi` 与索引条目完全匹配

---

### 测试 3：端到端管线测试（含真实 LLM 调用）

**输入：** 一个经典的 RAG 对话，包含带参考文献标记的论文片段

**步骤：**
1. 构造一个测试 prompt，context 中包含 `candidate refs: 3, 12, 31, 46, 58, 59`
2. 调用 LLM 获取原始回答
3. 运行 finalization（citation stripping with `preserve_numeric_markers=True`）
4. 运行渲染器获取 `(out_html, cite_details)`
5. 对每一个输出执行测试 1 和测试 2 的验证

**关键检查：**
- LLM 是否正确生成 `[n]` 用于片段引用
- LLM 是否正确生成 `[[CITE:<sid>:<ref_num>]]` 用于参考文献引用
- 两种标记没有混淆（不会对参考文献用 `[n]`，也不会对片段用 `[[CITE:...]]`）
- finalization 没有错误地剥离 `[n]` 或 `[[CITE:...]]`

---

### 测试 4：Payload ↔ 可见链接一致性测试

保证用户点击的链接和弹出的 payload 之间存在一一对应关系。

**方法：**
```
for each anchor href in out_html:              # 如 #kb-cite-test-a1b2c3
  find matching data-kb-cite="a1b2c3" in html  # 找到对应的隐藏 payload
  assert: payload exists                        # 不能有链接但无数据
  assert: payload.num contains correct number   # 数字正确

for each data-kb-cite in hidden html:           # 遍历所有隐藏数据
  find matching anchor link in visible html     # 找到对应的可见链接
  assert: link exists                           # 不能有数据但无链接（孤立数据）
```

保证**无孤立链接、无孤立数据**。

---

### 测试 5：引用老对话回归测试

用 `verify_citation_trace.py` 跑已有的对话数据库，确保：
- 已有功能（`[n]` 解析）没被本次改动破坏
- canonical_hit_paths 缺失时，positional fallback 仍正常工作

---

### 测试工具设计

建议在 `tools/verify_citation_trace.py` 基础上扩展，或创建 `tools/verify_citation_rendering.py`：

```
tools/verify_citation_rendering.py
├── render_and_parse(text, hits, ref_index)
│   ├── 调用 _annotate_inpaper_citations_with_hover_meta
│   ├── 解析 out_html 提取所有可见链接
│   ├── 解析 hidden payloads 提取所有数据
│   └── 返回 (links, payloads)
│
├── verify_links_vs_payloads(links, payloads)
│   ├── 每个 link → 有对应 payload ✓
│   └── 每个 payload → 有对应 link ✓
│
├── verify_payload_against_hits(payload, hits, canonical_paths)
│   ├── [n] → payload.num == n
│   ├── payload.source_path == canonical_paths[n-1]
│   ├── payload.raw 是 hits 中的文本
│   └── 返回 PASS / FAIL + 差异详情
│
├── verify_payload_against_ref_index(payload, ref_index, sid_map)
│   ├── [[CITE:...]] → payload.title == ref_index.title
│   ├── payload.authors == ref_index.authors
│   ├── payload.doi == ref_index.doi
│   └── 返回 PASS / FAIL + 差异详情
│
└── test_main()
    ├── 运行所有预设用例（测试 1）
    ├── 运行 payload 数据校验（测试 2）
    └── 汇总报告
```

### 验证标准

| 等级 | 条件 | 判定 |
|---|---|---|
| ✅ PASS | 所有 `[n]` 和 `[[CITE:...]]` 正确解析，payload 数据与来源完全一致 | 通过 |
| ⚠️ WARNING | 有 `[n]` 使用 fallback 解析（无 canonical_paths），但内容基本正确 | 警告 |
| ❌ FAIL | 任何 `[n]`/`[[CITE:...]]` 未解析、payload 数据不匹配、孤立链接或孤立数据 | 失败 |

---

## 风险与注意事项

1. **LLM 可能混淆两种标记**：需要在 prompt 中清晰说明何时用 `[n]` 何时用 `[[CITE:...]]`。如果 LLM 错误地对参考文献也使用 `[n]`（如 `[58]`），`_resolve_n_from_hits(58)` 会越界，strict 模式会剥离。这可以接受——宁缺毋滥。

2. **references_index 覆盖不全**：当前只有 21 篇论文有参考索引。对于没有索引的论文，`[[CITE:...]]` 会解析失败并被 strict 模式剥离，行为是安全的。

3. **渲染性能**：`_annotate_inpaper_citations_with_hover_meta` 每段调用一次。加入 `[[CITE:...]]` 解析后负载增加，但参考索引已缓存在 `_load_reference_index_cached()` 中，且论文参考文献数量通常有限（每篇 <100 条）。

4. **`candidate refs` 提取范围**：`extract_candidate_ref_nums_from_hits` 扫描 hit 文本中的 `[N]` 模式。如果原文有 "in 2025 [12, 13]" 会被正确提取，但方括号内的非参考文献数字（如化学式、坐标）可能被误提取。

---

## 实施顺序

1. **B1**（context 开启 candidate_refs）— ✅ 已完成
2. **B3**（渲染器解除互斥 + 上标渲染）— ✅ 已完成
3. **B2**（prompt 增加 [[CITE:...]] 指令）— ✅ 已完成
4. **B5**（验证）— ✅ 测试通过（772/773，预存 1 失败）
5. **B4**（文献篮 UI）— 已有完整实现，无需改动
