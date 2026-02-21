# kb_chat 代码框架梳理（给协作者）

## 1. 项目定位
`kb_chat` 是一个基于 Streamlit 的本地论文知识库问答应用，主链路是：

1. PDF 转 Markdown
2. Markdown 分块入库（`db/chunks/*.jsonl`）
3. BM25 检索 + 检索增强
4. LLM 生成回答
5. 回答旁显示“参考定位”（可打开 PDF / 跳页）

应用入口：`app.py`

---

## 2. 目录与分层

### 2.1 页面编排层
- `app.py`
  - `main()`：应用入口、侧边栏设置、页面路由、retriever 热加载
  - `_page_chat(...)`：聊天页（消息渲染、提交、流式轮询）
  - `_page_library(...)`：文献管理页（目录选择、转换、队列、上传、重建索引）

### 2.2 UI 组件层（`ui/`）
- `ui/runtime_patches.py`
  - 全局主题 CSS
  - 运行时 JS 注入（复制、输入框 dock、自动刷新）
- `ui/chat_widgets.py`
  - 标题渲染、AI 生成状态头、复制按钮、数学 Markdown 规范化
- `ui/refs_renderer.py`
  - 参考定位渲染（卡片、score、page、Open PDF / Go Page）
- `ui/strings.py`
  - 文案常量
- `ui/assets/chat_dock_runtime.js`
  - 聊天输入框悬浮/对齐逻辑

### 2.3 核心业务层（`kb/`）
- `kb/task_runtime.py`
  - 对话生成任务状态机（`GEN_TASKS`）
  - 后台 PDF 转换队列（`BG_STATE`）
  - 线程 worker（聊天生成 / 转换）
- `kb/retrieval_engine.py`
  - 检索增强（中文转英文检索词、分文档聚合、deep-read、LLM rerank）
- `kb/retrieval_heuristics.py`
  - 规则与打分函数（关键词、去噪、章节偏好、token 打分）
- `kb/retriever.py`
  - BM25 封装（空库保护）
- `kb/file_ops.py`
  - 文件路径/上传/目录选择/扫描工具

### 2.4 持久化层
- `kb/chat_store.py`
  - `chat.sqlite3`：`conversations`、`messages`、`message_refs`
- `kb/library_store.py`
  - `library.sqlite3`：PDF sha1 去重索引
- `kb/store.py`
  - 知识库索引文件与 chunks 读写（`docs.json` + `chunks/*.jsonl`）

### 2.5 配置与模型调用
- `kb/config.py`：环境变量 -> `Settings`
- `kb/llm.py`：OpenAI 兼容调用封装（chat / stream）

---

## 3. 两条核心数据流

## 3.1 对话流（Chat）
1. 用户提交问题（`_page_chat`）
2. 先写入 user 消息和 assistant 占位消息
3. `task_runtime._gen_start_task(...)` 启动后台线程
4. worker 内执行：
   - 检索（BM25 + fallback）
   - 参考定位聚合（按文档分组、章节定位）
   - 组装上下文并调用 LLM 流式输出
5. 流式增量更新 `messages`，完成后落最终答案
6. `message_refs` 保存该条 user 消息对应的参考定位
7. 前端轮询刷新并渲染消息 + refs 卡片

## 3.2 文献管理流（Library）
1. 选择 PDF 根目录与 MD 输出目录
2. 扫描 PDF、识别“已转换/未转换”
3. 入后台转换队列（可批量）
4. worker 执行 `run_pdf_to_md(...)`
5. 转换成功后自动 ingest 到知识库
6. DB 变更触发 retriever 热重载

---

## 4. 状态管理（关键）

## 4.1 Streamlit `st.session_state`
用于页面态与短期缓存，例如：
- 当前会话 `conv_id`
- UI 开关（如 `show_context`、`deep_read`）
- 当前页、主题、最近消息缓存

## 4.2 进程级共享状态（`kb/runtime_state.py`）
- `GEN_TASKS`：聊天生成任务状态
- `BG_STATE`：后台转换队列状态
- `CACHE`：跨 rerun 缓存（翻译词、深读文本、rerank 结果等）

设计目的：Streamlit rerun 频繁，任务状态不能只放在页面局部变量。

---

## 5. 检索与参考定位策略（当前）
- 基础检索：BM25
- 中文问题：优先尝试翻译为英文检索词再检索
- 结果聚合：按文档聚合，生成“文档级 refs”
- 可选增强：
  - deep-read（读取 MD 全文补片段）
  - LLM rerank（语义重排）
- 显示层：refs 卡片 + 打开 PDF / 跳页

---

## 6. 数据文件与产物
- `chat.sqlite3`：聊天记录与 refs
- `library.sqlite3`：上传去重索引
- `db/docs.json`：文档索引
- `db/chunks/*.jsonl`：检索分块
- `tmp/`：临时转换与调试产物

---

## 7. 协作时优先阅读顺序（建议）
1. `app.py`（先看 `main()`，再看 `_page_chat`、`_page_library`）
2. `kb/task_runtime.py`（任务状态机与后台队列）
3. `kb/retrieval_engine.py`（检索增强逻辑）
4. `ui/refs_renderer.py`（参考定位最终呈现）
5. `kb/chat_store.py` + `kb/store.py`（持久化结构）

---

  
