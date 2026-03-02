# 聊天输入框前端（固定底部 + 玻璃样式）

## 输入框是怎么来的

- **Python 渲染**：`app.py` 里 `_render_chat_prompt_form_ui()`（约 933–962 行）  
  - 用 `st.form(key=...)` 包一层  
  - 里面是 `st.text_area(" ", height=96, key="prompt_text")` 和 `st.file_uploader(...)`、`st.form_submit_button(...)`  
- **DOM**：Streamlit 会生成 `form` 或外层 `div[data-testid="stForm"]`，内部有 `div[data-testid="stTextArea"]`、`div[data-testid="stFileUploader"]` 等。

## 样式与脚本注入

| 作用 | 位置 | 方式 |
|------|------|------|
| 全局主题 + 输入框 CSS | `ui/runtime_patches_parts/theme_legacy.py` | `_init_theme_css()` → `st.markdown(<style>)`，在 **主文档** |
| 输入框 dock 的 class 样式 | 同上，`.kb-input-dock`、`.kb-chat-dock-root` 及结构选择器 | 同上，主文档 |
| 原 dock 脚本（找表单、定位、按钮等） | `ui/assets/chat_dock_runtime.js` | `_inject_chat_dock_runtime()` → **components.html(script)** → 脚本在 **iframe 里** 跑 |
| 新增：同文档内给表单加 class + 固定/玻璃 | `ui/runtime_patches_parts/chat_runtime.py` → `_chat_dock_same_document_script_html()` | **st.markdown(script)** → 脚本在 **主文档** 跑 |

## 为什么不生效（根本原因）

- **dock 大逻辑**在 `chat_dock_runtime.js` 里，是通过 `components.html("<script>...")` 注入的。  
- `components.html()` 会创建一个 **iframe**，脚本只在这个 iframe 的 document 里执行。  
- 表单在 **主文档**（Streamlit 画出来的那一页），不在 iframe 里。  
- 脚本里用 `root = (window.parent || window).document` 去“主文档”找表单；若 Streamlit 把整块主内容也放在另一个 iframe 里，那 `parent.document` 其实是“外层 host”，**根本不包含表单**，所以怎么找都找不到，class 和样式都加不上，看起来就“不生效”。

## 修复方式

- 增加一条**和表单在同一文档里执行的路径**：用 **st.markdown** 注入一段脚本（不经过 components.html，所以不在 iframe 里）。  
- 这段脚本在 **主文档** 里：  
  - 用 `document.querySelectorAll('form')` 找到带 `[data-testid="stTextArea"]` 和 `[data-testid="stFileUploader"]` 的那个 form；  
  - 取 `form` 或 `form.closest('[data-testid="stForm"]')` 作为根；  
  - 给根加上 class `kb-input-dock`、`kb-dock-positioned`，并设置 `position:fixed`、`bottom`、`z-index`、`backdrop-filter` 等内联样式。  
- 这样不依赖 iframe 里的脚本能否访问到主文档，**只要主文档里有这个表单，就能被固定到底部并呈现玻璃样式**。  
- 调用链：聊天页在 `app.py` 里 `if not disable_hooks` 时先 `_inject_chat_dock_runtime()`，再 `st.markdown(_chat_dock_same_document_script_html(), unsafe_allow_html=True)`。

## 相关文件一览

| 文件 | 作用 |
|------|------|
| `app.py` | 渲染 form、调用 `_inject_chat_dock_runtime()` 和 `st.markdown(_chat_dock_same_document_script_html())` |
| `ui/runtime_patches_parts/chat_runtime.py` | `_inject_chat_dock_runtime()`（iframe 脚本）、`_chat_dock_same_document_script_html()`（同文档脚本 HTML） |
| `ui/assets/chat_dock_runtime.js` | iframe 内 dock 逻辑（找表单、placeDock、宽度随侧边栏等）；若主文档在另一 iframe 可能拿不到表单 |
| `ui/runtime_patches_parts/theme_legacy.py` | `.kb-input-dock` / `.kb-chat-dock-root` 及结构选择器 CSS（固定、玻璃、圆角等） |

## 小结

- **输入框前端** = Streamlit 的 form + text_area + file_uploader（由 `app.py` 渲染）。  
- **不生效** = 原来只靠 iframe 里的脚本去“主文档”找表单，在 Streamlit 的 iframe 结构下经常拿不到。  
- **修复** = 用 `st.markdown` 在主文档里再跑一段脚本，直接在该文档里找表单并加 class + 内联样式，保证固定底部和玻璃样式一定生效。
