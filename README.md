# Pi_zaya / kb_chat

学术 PDF 知识库助手（RAG）：上传文献 -> 转换与建库 -> 对话检索回答（含可追溯引用）。

## 1. 项目入口

当前推荐入口：

- `server.py` + `web/`（FastAPI + React）

兼容保留入口（旧版）：

- `app.py`（Streamlit）

Windows 下可直接用脚本：

```powershell
# 新版前后端（推荐）
.\run_new.ps1 -StopExisting

# 旧版兼容入口
.\run_old.ps1
```

说明：

- `run_new.ps1` 默认不会自动安装依赖，需要时加 `-InstallBackendDeps` / `-InstallFrontendDeps`。
- `run_new.ps1` 默认端口：后端 `8000`，前端 `5173`。

## 2. 你能做什么

- 批量上传 PDF，并生成 Markdown。
- 在文献管理页执行转换、重命名建议、引用同步、更新知识库。
- 在对话页进行基于知识库的问答（流式输出）。
- 点击回答中的引用 `[n]` 查看来源信息、DOI、文献篮条目。

## 3. 快速开始

### 3.1 克隆项目

```bash
git clone https://github.com/LittlePyx/Pi_zaya.git
cd Pi_zaya
```

### 3.2 配置 API Key（以 Qwen 为例）

macOS / Linux:

```bash
export QWEN_API_KEY="你的key"
```

Windows PowerShell:

```powershell
$env:QWEN_API_KEY="你的key"
```

### 3.3 安装依赖

后端：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

前端：

```bash
cd web
npm install
cd ..
```

### 3.4 启动（两种模式）

开发模式（推荐用于调试）：

```powershell
.\run_new.ps1 -StopExisting
```

然后访问：`http://localhost:5173`

生产本地模式（单服务）：

```bash
cd web && npm run build && cd ..
python server.py
```

然后访问：`http://localhost:8000`

## 4. 标准使用流程（给终端用户）

1. 打开「文献管理」。
2. 在目录设置中配置：
   - `PDF 目录`
   - `MD 目录`
3. 上传 PDF（支持批量）。
4. 在上传工作台检查失败项，必要时重试。
5. 点击「更新知识库」。
6. 回到「对话」开始提问。

建议：首次导入大量文献时，先完成一轮转换 + 更新知识库，再进入对话页。

## 5. 重要使用说明

- 聊天侧设置中仅保留核心参数：`Top-K`、`温度`、`最大输出 tokens`。
- `深读 MD` 已固定为默认开启（不再在设置中提供开关）。
- `显示片段全文` 功能已移除（无设置按钮）。
- 前端默认使用非“极速扫描”模式（更稳定）。

## 6. 常见问题

### 6.1 页面打不开

- 开发模式请确认 `run_new.ps1` 两个端口都启动成功。
- 生产模式请确认已先执行 `web` 的 `npm run build`。

### 6.2 上传后无法检索到内容

- 确认已执行「更新知识库」。
- 确认 PDF 对应 Markdown 已生成在当前 `MD 目录`。

### 6.3 引用里没有 DOI

不一定是错误，常见原因：

- 原始文献条目无 DOI。
- 条目不完整，Crossref 未命中。

## 7. 常用环境变量

模型：

- `QWEN_API_KEY`
- `QWEN_BASE_URL`
- `QWEN_MODEL`
- `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`（回退）
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`（回退）

路径与数据库：

- `KB_PDF_DIR`
- `KB_MD_DIR`
- `KB_DB_DIR`
- `KB_CHAT_DB`
- `KB_LIBRARY_DB`

引用同步：

- `KB_CROSSREF_BUDGET_S`（Crossref 后台同步预算秒数）

## 8. 开发者补充

- 前端目录：`web/`
- API 入口：`api/main.py`
- 生产启动文件：`server.py`
- React 迁移计划：`docs/REACT_MIGRATION_PLAN.md`

## 9. 稳定性基线（推荐）

为减少“我这边正常、用户那边异常”的环境差异，建议统一以下约束：

- Python 版本锚点：`.python-version`（当前 `3.10.11`）
- Node 版本锚点：`.nvmrc`（当前 `24.13.0`）
- 前端安装优先用 `npm ci`（锁定 `web/package-lock.json`）

### 9.1 环境体检（doctor）

新增脚本：`tools/stability/doctor.ps1`

```powershell
# 输出版本、git 状态、锁文件 hash、KB_* 环境变量
.\tools\stability\doctor.ps1

# 严格模式：版本不匹配时返回非 0
.\tools\stability\doctor.ps1 -Strict
```

### 9.2 运行态重置（reset）

新增脚本：`tools/stability/reset_state.ps1`

```powershell
# 默认安全重置（运行临时文件 + 日志 + Python 缓存）
.\tools\stability\reset_state.ps1

# 仅预览将删除内容
.\tools\stability\reset_state.ps1 -DryRun

# 如需清空数据库，必须显式指定（谨慎）
.\tools\stability\reset_state.ps1 -ClearChatDb -ClearLibraryDb

# 需要全仓深度日志扫描时（较慢）
.\tools\stability\reset_state.ps1 -ClearLogs -DeepLogScan

# 需要深度扫描所有 __pycache__ 时（较慢）
.\tools\stability\reset_state.ps1 -ClearPyCaches -DeepPyCacheScan
```

说明：默认不会删除聊天库/文献库，避免误清空业务数据。

### 9.3 CI 基线

仓库新增 GitHub Actions：`.github/workflows/ci.yml`，在 push / PR 时执行：

1. 按 `.nvmrc` 安装 Node 并构建前端
2. 使用 Python `3.10.11` 安装后端依赖
3. 执行 `tests/unit` 回归测试
