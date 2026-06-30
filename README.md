# Pi_zaya

Pi_zaya 是一个本地优先的学术文献库问答工作台。它把 PDF 文献转换成 Markdown、结构化索引和可追溯引用，让你在同一个界面里完成阅读、提问、证据定位、文献收集和引用导出。

它不只是“对一篇 PDF 聊天”。你可以围绕当前论文深入追问，也可以让系统从整个文献库里找出相关论文，并把答案落回原文页码、章节、引用或片段。

> 当前产品入口是 **FastAPI + React**。旧 Streamlit 入口已经移除，不使用 `app.py`、`streamlit run` 或端口 `8501` 作为产品入口。

## 适合场景

- 读单篇论文时，快速追问创新点、方法、实验、图表、公式和上下文。
- 做课题调研时，从一批本地文献中找相关论文，而不是只依赖单篇 PDF。
- 写综述、开题、论文或报告时，保留可回溯的证据和引用格式。
- 管理长期积累的 PDF 文献库，并持续修复转换质量、同步参考文献和重建索引。

## 核心功能

| 功能 | 说明 |
|---|---|
| 文献库问答 | 在当前论文、文献篮或全库范围内检索回答，适合从问题出发找相关研究。 |
| 阅读指导 | 围绕 Reader 中打开的论文解释结构、创新点、方法细节和关键证据。 |
| 参考定位 | 回答中的证据可以定位到原文片段、页码、章节、文内引用或参考文献条目。 |
| 文献篮 | 收集研究摘录和候选文献，保存本机快照，导出 BibTeX、RIS、Markdown、GB/T 等格式。 |
| 文献管理 | 批量上传 PDF，转换 Markdown，管理元数据，执行质量检查和引用同步。 |
| 转换质量 | 对 PDF 转 Markdown 的结果做质量扫描、修复、重试和索引重建。 |
| 维护能力 | 支持自动备份、手动备份、恢复预演、受控恢复、诊断包导出和版本更新提醒。 |

## 快速开始

### 环境要求

- Python `3.10.11`，见 `.python-version`
- Node.js `24.13.0`，见 `.nvmrc`
- 两类模型 API：文本问答 API 和视觉/图片 API

### 模型 API 配置

Pi_zaya 现在建议配置两类 API，分工如下：

| 类型 | 用途 | 推荐配置 |
|---|---|---|
| 文本问答 API | 普通问答、总结、阅读指导、检索改写、参考卡片生成 | `DEEPSEEK_API_KEY` |
| 视觉/图片 API | 图片提问、PDF 页面理解、图表/公式处理、多模态转换 | `QWEN_API_KEY` |

`OPENAI_API_KEY` 可以作为可选回退配置。只配置文本 API 时，纯文本问答可以工作；但涉及图片、图表、扫描页、多模态 PDF 转换时会降级或失败。正式使用和上线前建议同时配置文本 API 与视觉 API。

有两种配置方式。

**方式一：在项目环境变量或 `.env` 中配置**

适合部署、多人共享或长期固定运行。可以直接设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY="your-deepseek-key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
$env:DEEPSEEK_MODEL="deepseek-chat"

$env:QWEN_API_KEY="your-qwen-key"
$env:QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:QWEN_MODEL="qwen3-vl-plus"
```

也可以复制生产模板后填写：

```powershell
Copy-Item .env.production.example .env
```

然后在 `.env` 中填写：

```dotenv
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

QWEN_API_KEY=your-qwen-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-vl-plus
```

**方式二：在项目设置页中配置**

适合本机使用、临时测试或不想手动改 `.env` 的情况。

1. 启动项目并打开 `http://127.0.0.1:5173/`。
2. 进入设置页。
3. 在“连接”区域填写“文本问答模型”的 API key、Base URL 和模型 ID。
4. 填写“视觉/图片模型”的 API key、Base URL 和模型 ID。
5. 分别点击测试文本连接和测试视觉连接。
6. 测试通过后点击保存 API 设置。

设置页保存的 key 会写入本地 `user_prefs.json`，不会在设置接口中明文返回。需要注意：环境变量和 `.env` 的优先级高于设置页；如果你同时配置了 `.env` 和设置页，实际运行会优先使用 `.env`/环境变量中的值。

### 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd web
npm ci
cd ..
```

### 启动开发环境

```powershell
.\run_new.ps1 -StopExisting
```

本地开发启动默认保持公开模式；即使 `.env` 里误留了 `KB_PRIVATE_INSTANCE_AUTH=1`、`KB_REQUIRE_AUTH=1` 或 `KB_ENABLE_AUTH_GATE=1`，普通 `run_new.ps1` 也会在本进程内关掉令牌门禁，用户打开页面不需要访问令牌。只有调试私有实例鉴权时才使用 `.\run_new.ps1 -AllowAuthGate`。

`run.ps1` 是等价的便捷入口：

```powershell
.\run.ps1 -StopExisting
```

默认地址：

- React app: `http://127.0.0.1:5173/`
- FastAPI backend: `http://127.0.0.1:8000/`

### 单服务运行

```powershell
cd web
npm run build
cd ..
python server.py
```

默认访问：`http://127.0.0.1:8000/`

## 日常使用流程

1. 打开文献管理页。
2. 设置 PDF 目录、Markdown 目录和知识库目录。
3. 上传或选择 PDF，执行转换。
4. 检查转换结果，必要时运行质量修复或重新转换。
5. 点击更新知识库，生成检索索引和参考文献索引。
6. 回到对话页提问，按需要切换本文、文献篮或全库范围。
7. 点击回答中的证据、引用或定位入口，在 Reader 中查看原文依据。
8. 将重要文献加入文献篮，按写作需要导出引用格式。

## 生产部署

生产部署建议先复制环境模板：

```powershell
Copy-Item .env.production.example .env
```

重点确认：

- 面向普通用户的公开部署保持 `KB_PRIVATE_INSTANCE_AUTH=0`、`KB_ENABLE_AUTH_GATE=0` 和 `KB_REQUIRE_AUTH=0`，用户打开应用不需要访问令牌。
- 只有私有/内部实例才同时设置 `KB_PRIVATE_INSTANCE_AUTH=1`、`KB_ENABLE_AUTH_GATE=1`、`KB_REQUIRE_AUTH=1`，并配置 `KB_ACCESS_TOKEN` 或 `KB_ACCESS_TOKEN_SHA256`。
- 仅在同时开启 `KB_PRIVATE_INSTANCE_AUTH=1`、`KB_ENABLE_AUTH_GATE=1` 和 `KB_REQUIRE_AUTH=1` 时，按部署方式配置 `KB_AUTH_COOKIE_SECURE`。
- `KB_API_ALLOW_ORIGINS` 只包含允许访问的前端来源。
- `KB_DB_DIR`、`KB_CHAT_DB`、`KB_LIBRARY_DB`、`KB_BACKUP_DIR` 使用稳定的绝对路径。
- 文本问答 API 和视觉/图片 API 都已配置并测试通过。
- `KB_AUTO_BACKUP=1`，让高风险操作前自动创建快照。
- `KB_APP_VERSION` 与 GitHub Release tag 对齐，版本提醒才能判断是否有更新。

构建并启动：

```powershell
cd web
npm run build
cd ..
python server.py
```

启动后执行上线检查：

```powershell
python tools\check_production_readiness.py --base-url http://127.0.0.1:8000
```

私有/内部实例如果同时启用了 `KB_PRIVATE_INSTANCE_AUTH=1`、`KB_ENABLE_AUTH_GATE=1` 和 `KB_REQUIRE_AUTH=1`，再追加 `--token $env:KB_ACCESS_TOKEN`。

完整部署说明见 `docs/DEPLOYMENT.md`。

## 维护与数据安全

Pi_zaya 的设置页只向普通用户展示必要状态：API 是否可用、数据保护是否开启、是否有恢复后待复查。备份列表、恢复控制、审计时间线和诊断包导出位于高级维护/管理员工具中。

数据恢复建议流程：

1. 先创建或确认可用备份。
2. 运行恢复预演，检查是否存在阻塞项。
3. 执行受控恢复。
4. 重启服务。
5. 完成恢复后复查确认。

本地业务数据默认包括：

| 数据 | 说明 |
|---|---|
| `chat.sqlite3` | 会话、消息、检索引用和文献篮状态 |
| `library.sqlite3` | 文献库、转换记录、元数据和来源状态 |
| `db/` | 文档索引、分块、参考文献索引和 Crossref 缓存 |
| `backups/` | 自动和手动备份 |

这些文件是本地运行数据，不应提交到 Git。

## 开发命令

后端开发：

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

前端开发：

```powershell
cd web
npm run dev -- --host 127.0.0.1 --port 5173
```

前端检查：

```powershell
cd web
npm run lint
npm run build
```

后端测试：

```powershell
python -m pytest tests/unit -q
```

回归 fixture 检查：

```powershell
python tools\research_qa\run_research_qa_eval.py --dry-run
python tools\converter_quality\run_converter_quality_eval.py --dry-run
```

GitHub Actions 会在 push 和 pull request 时执行前端 lint/build、轻量 Playwright smoke、后端 unit tests、后端 sanity/API 回归测试，以及两组 dry-run 回归检查。

## 项目结构

| 路径 | 说明 |
|---|---|
| `api/main.py` | FastAPI 应用入口 |
| `api/routers/chat.py` | 对话、流式回答和检索集成 |
| `api/routers/library.py` | 文献管理、转换、质量、元数据和索引接口 |
| `api/reference_ui.py` | 引用卡片、参考定位和文献篮数据整形 |
| `server.py` | 本地生产/单服务入口 |
| `web/src/main.tsx` | React 前端入口 |
| `web/src/pages/ChatPage.tsx` | 主问答体验 |
| `web/src/pages/LibraryPage.tsx` | 文献管理与转换工作台 |
| `web/src/components/refs/RefsPanel.tsx` | 研究侧栏、文献篮和参考定位 |
| `kb/converter/pipeline.py` | PDF 到 Markdown 转换流水线 |
| `kb/retrieval_engine.py` | 查询扩展、检索、过滤和缓存 |
| `kb/reference_index.py` | 参考文献抽取与索引 |
| `kb/task_runtime.py` | 后台任务运行时 |

## 本地清理与体检

安全清理运行缓存、日志和临时文件：

```powershell
.\tools\stability\reset_state.ps1
```

默认不会删除 `chat.sqlite3`、`library.sqlite3`、`db/` 或 `backups/`。清空业务数据需要显式传参，执行前请先备份。

环境体检：

```powershell
.\tools\stability\doctor.ps1
.\tools\stability\doctor.ps1 -Strict
```

## 开发原则

- 产品只使用 FastAPI + React 入口。
- 不使用 `app.py`、`streamlit run` 或端口 `8501` 作为产品入口。
- Markdown 页码标记 `<!-- kb_page: N -->` 必须在转换和修复中保留。
- React API 合同应放在 `web/src/api` 中并保持类型化。
- 质量问题优先从转换、索引和数据链路修复，而不是只隐藏前端状态。
- 高风险操作遵循“先备份、再执行、再复查”。
