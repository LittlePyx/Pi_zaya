# Pi_zaya / kb_chat

学术 PDF 知识库助手 —— 基于 RAG 的问答系统，支持可追溯引用。

前端：Vite + React 18 + Ant Design 5 + TailwindCSS v4
后端：FastAPI（包装 `kb/` 模块）+ Uvicorn

## 你可以用它做什么

- 在「文献管理」页批量上传 PDF，并转换为 Markdown（含图片、公式、参考文献处理）。
- 一键「更新知识库」，把 Markdown 分块索引到本地 DB。
- 在「对话」页基于知识库检索回答，LLM 流式输出（SSE），并显示可点击的文内引用 `[n]`。
- 点击引用可查看文献信息（来源、题录、DOI 链接）。
- 参考文献索引支持后台 Crossref 同步，不阻塞页面使用。
- 支持 Dark / Light 主题切换。

---

## 当前默认模型（我现在使用）

本项目当前优先使用 **Qwen**（OpenAI 兼容接口）：

- 默认 `QWEN_BASE_URL`: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- 默认 `QWEN_MODEL`: `qwen3-vl-plus`

代码逻辑是：

1. 优先读 `QWEN_API_KEY`
2. 否则回退 `DEEPSEEK_API_KEY`
3. 再回退 `OPENAI_API_KEY`

所以你只要设置 `QWEN_API_KEY`，就会走 Qwen。

---

## 快速启动

### 1) 克隆项目

```bash
git clone https://github.com/LittlePyx/Pi_zaya.git
cd Pi_zaya
```

### 2) 设置 API Key

macOS / Linux：

```bash
export QWEN_API_KEY="你的key"
```

Windows (PowerShell)：

```powershell
$env:QWEN_API_KEY="你的key"
```

### 3) 安装后端依赖

macOS / Linux：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell)：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4) 安装前端依赖并构建

```bash
cd web
npm install
npm run build
cd ..
```

### 5) 启动

```bash
python server.py
# 访问 http://localhost:8000
```

生产模式下 FastAPI 同时提供 API 和前端静态文件。

**开发模式**（前后端分离热更新）：

```bash
# 终端 1：后端
python server.py

# 终端 2：前端（Vite dev server，自动代理 /api → localhost:8000）
cd web && npm run dev
# 访问 http://localhost:5173
```

---

## 标准使用流程

1. 打开「文献管理」页。
2. 设置：
   - `文献目录（PDF）`
   - `输出目录（Markdown）`
3. 上传 PDF（支持批量）。
4. 选择转换模式：
   - `normal`：质量优先（截图识别 + VL）
   - `ultra_fast`：更快，质量略降
   - `no_llm`：不使用多模态模型（基础提取）
5. 点击「更新知识库」。
6. 回到「对话」页提问。

---

## 引用与文献篮

在回答里看到 `[n]` 后：

- 点击 `[n]` 会弹出引用详情（支持拖动）。
- 可直接打开 DOI。
- 可「加入文献篮」。
- 文献篮会在右侧汇总，支持定位与高亮条目。

说明：不是每条参考文献都一定有 DOI（历史文献、会议条目、源数据缺失时常见）。

---

## 常用环境变量

### 模型相关

- `QWEN_API_KEY`
- `QWEN_BASE_URL`
- `QWEN_MODEL`
- `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`（回退）
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`（回退）

### 路径与服务

- `KB_PDF_DIR`：默认 PDF 根目录
- `KB_MD_DIR`：默认 Markdown 目录
- `KB_DB_DIR`：知识库索引目录
- `KB_CHAT_DB`：对话数据库路径
- `KB_LIBRARY_DB`：文献库数据库路径

### 参考文献索引

- `KB_CROSSREF_BUDGET_S`：Crossref 后台同步预算秒数（默认 45）

---

## 常见问题

### 1) 页面打不开或接口报错

1. 确认后端已启动：`python server.py`
2. 生产模式确认已执行 `cd web && npm run build`
3. 检查终端日志中的错误信息

### 2) 为什么有的引用没有 DOI

可能原因：

- 文献本身未注册 DOI
- 参考文献条目不完整或噪声大
- Crossref 未命中

### 3) 更新知识库后对话没变化

请确认你更新的是当前使用的 `DB` 目录，并且 Markdown 文件确实已生成到对应目录。

---

## 给使用者的说明

- 这个项目目前是我持续迭代中的版本，界面和细节会更新。
- 你遇到“可复现”的问题时，附上：PDF 名称、页面截图、生成的 `.md` 片段，我可以更快定位并修复。
