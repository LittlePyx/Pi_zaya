# 冗余与重构建议

基于对当前代码库的扫描，下面列出**冗余最严重、最值得优先重构**的部分。

---

## 1. 主题 / CSS：`theme_legacy.py`（约 5271 行）

### 1.1 重复的 theme tokens（高）

- **位置**：`_init_theme_css()` 里 `mode == "dark"` 的 `tokens` 与 `_sync_theme_with_browser_preference()` 里的 `dark_tokens` 几乎完全相同（约 60 行 CSS 变量）。
- **建议**：抽成一份“暗色主题变量”数据（如 `_DARK_THEME_TOKENS`），两处都引用；若以后要加 `_LIGHT_THEME_TOKENS` 也可一起集中管理。

### 1.2 重复的 CSS 选择器前缀（高）

- **现象**：同一条规则经常写成多行，仅前缀不同，例如：
  - `section[data-testid="stSidebar"]` 与 `section.stSidebar`
  - `div[data-testid="element-container"]` 与 `div[data-testid="stElementContainer"]`
- **规模**：`section[data-testid="stSidebar"]` 出现约 484 次，`!important` 约 1812 次。
- **建议**：
  - 用 Python 生成“选择器列表”，例如：  
    `SIDEBAR = ("section[data-testid=\"stSidebar\"]", "section.stSidebar")`  
    然后写 `",\n".join(SIDEBAR) + " " + common_selector { ... }` 拼出整条规则。
  - 或把整块 CSS 拆成“逻辑块”（sidebar / dock / chat / stale 等），每块用一个小函数生成，再 `"\n".join()` 组成最终样式，便于去重和复用前缀。

### 1.3 单一大字符串（中）

- **现象**：`css = """..."""` 体量极大（数千行），难以定位和局部修改。
- **建议**：按功能拆成多个字符串或从多函数返回，例如：  
  `_css_base()`, `_css_sidebar()`, `_css_dock()`, `_css_chat()`, … 再在 `_init_theme_css()` 里拼成一份 `css`。

---

## 2. 历史侧栏 CSS：`theme_history_overrides.py`（约 601 行）

### 2.1 多组选择器变体重复（高）

- **现象**：同一条规则经常带 4～8 个选择器变体：  
  `section[data-testid="stSidebar"]`、`section.stSidebar`、  
  `element-container` / `stElementContainer`、  
  `column` / `stColumn`、`+ div` / `+ div div[data-testid="stHorizontalBlock"]` 等。
- **建议**：
  - 定义若干“选择器组”常量或小函数，例如：  
    `sidebar()`, `row_block()`, `row_btn()`, `actions_block()`，每个返回 `(sel1, sel2, ...)`。
  - 写一个 `_rule(selectors, declarations)`，内部用 `",\n".join(selectors) + " { " + declarations + " }"` 生成单条规则，再在文件中用这些规则拼接整段 CSS，可显著减少重复行数并统一兼容写法。

---

## 3. Chat / Dock 运行时：`chat_runtime.py`（约 650 行）

### 3.1 重复的 iframe 脚本结构（中）

- **现象**：多处 `components.html("<script>(function(){ try { ... } catch(e){} })();</script>", height=0)`，且逻辑类似：  
  `host = window.parent || window`、`doc = host.document`、在 parent 或 doc 上操作。
- **建议**：抽一个 `_inject_script(js_body: str)`（或 `_inject_script_html(js_body)`），内部统一包成 IIFE + try/catch 并调用 `components.html(..., height=0)`，各调用点只传核心逻辑的 JS 字符串。

### 3.2 内联大段 JS 字符串（中）

- **现象**：`_inject_chat_dock_into_parent()` 内 `code = r"""..."""` 体量很大，且与 `chat_dock_runtime.js` 及 `_chat_dock_same_document_script_html()` 有功能重叠。
- **建议**：  
  - 若需在 parent 中执行，可考虑把这段逻辑移入 `assets/` 下某 .js，通过 `Path.read_text()` 注入（与现有 `_CHAT_DOCK_JS_PATH` 类似），避免在 Python 里维护大段字符串。  
  - 与 `chat_dock_runtime.js`、same-document 脚本做一次职责划分，避免三处重复实现“找 form / 改样式 / 监听 resize”。

---

## 4. 参考文献渲染：`refs_renderer.py`（约 3500+ 行）

### 4.1 大量仅一两处使用的 `_xxx_ui` 小函数（中）

- **现象**：存在大量 `_xxx_ui()` 辅助函数（如 `_looks_like_doc_title_heading_ui`, `_trim_clause_ui`, `_pick_term_from_sentence_ui` 等），多数只在单一调用链里用一次。
- **建议**：  
  - 将“纯文本/逻辑、无状态”的辅助函数迁到单独模块（如 `ui/refs_ui_helpers.py`），按主题分组（heading / sentence / anchor / display），便于测试和复用。  
  - 对确实只在一处使用的，可评估是否内联到调用处，减少跳转和文件长度。

### 4.2 `_render_refs` 过长（中）

- **现象**：`_render_refs()` 承担整块引用列表的渲染，体量大，分支多。
- **建议**：按“区块”拆成子函数，例如：  
  `_render_refs_header()`, `_render_refs_list()`, `_render_refs_item()` 等，由 `_render_refs()` 按顺序调用，便于阅读和单测。

---

## 5. 主入口与页面：`app.py`（约 3000+ 行）

### 5.1 `_page_chat` 过长（中）

- **现象**：`_page_chat()` 包含侧栏、消息区、输入区、流式处理、轮询等，单函数体量很大。
- **建议**：拆成语义清晰的子函数，例如：  
  `_page_chat_sidebar()`, `_page_chat_messages()`, `_page_chat_form()`, `_page_chat_streaming_loop()` 等，`_page_chat()` 只做顺序编排与共享参数传递。

### 5.2 `main()` 与 `_page_library`（低）

- **现象**：`main()` 里初始化与路由较多；`_page_library()` 也较长。
- **建议**：  
  - `main()` 可拆出 `_init_app_state()`, `_route_pages()` 等。  
  - `_page_library()` 可按“区域”拆成 `_library_upload_section()`, `_library_table_section()` 等，与 `_page_chat` 的拆分方式一致。

---

## 6. 运行时 UI 脚本：`runtime_ui.py`（约 1359 行）

### 6.1 单一大段 JS 字符串（中）

- **现象**：`_inject_runtime_ui_fixes()` 等通过一个很大的 f-string 注入整段 JavaScript，可读性和可维护性都较差。
- **建议**：  
  - 将主要逻辑移到 `ui/assets/` 下独立 .js 文件，在 Python 中只做“读取 + 可选的参数替换 + components.html()”，便于格式化、语法高亮和复用。  
  - 若需动态插入 conv_id、theme 等，可保留少量 f-string 或 `json.dumps()` 注入，其余用 .js 文件。

---

## 优先级小结

| 优先级 | 位置 | 问题 | 预期收益 |
|--------|------|------|----------|
| 高 | theme_legacy.py | 重复的 dark tokens + 重复的 CSS 选择器前缀 | 减少数百行重复，后续改主题/兼容性更稳 |
| 高 | theme_history_overrides.py | 多组选择器变体重复 | 减少大量重复行，统一侧栏选择器策略 |
| 中 | chat_runtime.py | 重复的 iframe 脚本结构 + 大段内联 JS | 更清晰的注入方式，减少三处 dock 逻辑重叠 |
| 中 | refs_renderer.py | 大量 _xxx_ui 小函数 + _render_refs 过长 | 文件更易导航，便于单测和复用 |
| 中 | app.py | _page_chat / main / _page_library 过长 | 更易维护和加功能 |
| 中 | runtime_ui.py | 单一大段 JS 字符串 | 可读性、可维护性提升 |

建议从 **theme_legacy 的 tokens 去重** 和 **theme_history_overrides 的选择器生成** 入手，改动集中、收益明显；再逐步处理 chat_runtime、refs_renderer 和 app 的拆分。
