# Pi_zaya

[English](README.md) | 简体中文

Pi_zaya 是一个面向学术 PDF 的本地优先、证据可追溯阅读助手。它把 PDF 转换为
带页码和结构锚点的 Markdown，建立 BM25 词法与结构化索引，回答关于库内论文的问题，
展示可定位到原文的引用，并帮助用户比较、核验和整理证据。

Pi_zaya 的边界是辅助阅读用户已经放入文献库的论文和寻找证据，不帮助撰写、
改写或润色论文，也不提供投稿准备流程。默认回答只使用文献库；用户可按会话
明确选择在库内证据不足时使用模型知识或网页检索补充。

产品入口是 FastAPI + React。旧版 Streamlit 入口已经移除；不要使用
`app.py`、`streamlit run` 或端口 `8501` 启动产品。

## 下载 Windows 测试版

当前已发布的 Windows 测试版是
[`v0.1.0-beta.13`](https://github.com/LittlePyx/Pi_zaya/releases/tag/v0.1.0-beta.13)。

### 应该下载哪个文件

| Release 文件 | 用途 | 使用方法 |
|---|---|---|
| [`Pi_zaya-v0.1.0-beta.13-windows-x64-setup.exe`](https://github.com/LittlePyx/Pi_zaya/releases/download/v0.1.0-beta.13/Pi_zaya-v0.1.0-beta.13-windows-x64-setup.exe) | 推荐普通 Windows 用户使用 | 先核对 SHA-256，再运行安装向导。安装后从开始菜单或可选的桌面快捷方式打开 Pi_zaya。只安装到当前用户，不需要管理员权限。 |
| [`Pi_zaya-v0.1.0-beta.13-windows-x64.zip`](https://github.com/LittlePyx/Pi_zaya/releases/download/v0.1.0-beta.13/Pi_zaya-v0.1.0-beta.13-windows-x64.zip) | 免安装、移动目录或排查问题 | 先核对 SHA-256，把 ZIP **完整解压**到普通文件夹，再双击 `Pi_zaya.exe`。 |
| `*.sha256` | 校验下载是否完整 | 这是文本校验文件，不是软件。使用 `Get-FileHash` 计算对应 EXE 或 ZIP 的 SHA-256，并与文件中的值比较。 |
| `*.manifest.json` | 查看构建来源和签名状态 | 这是机器可读的版本、提交、许可证、包类型和签名记录，不是软件。 |
| GitHub 自动生成的 Source code 压缩包 | 开发者从源码构建 | 这不是可直接运行的 Windows 软件。普通用户请选择安装器或便携 ZIP。 |

安装版和便携版功能相同，都已包含 Python 运行时和构建好的 React 前端，
不要求用户安装 Python、Node.js 或 npm。启动后浏览器会打开本地页面，Pi_zaya
图标会留在系统托盘；托盘菜单可以重新打开页面、查看日志和数据目录并安全退出。

当前 Release 没有受信任的代码签名，因此 Windows 可能显示“未知发布者”。
请只从项目正式 GitHub Release 下载并先核对 SHA-256。只有 manifest 中
`signed` 为 `true`，并且 Windows 验证的发布者与发布说明一致时，才应视为
受信任签名版本。

更详细的安装、首次配置、更新、卸载和故障恢复步骤见
[Windows 中文使用说明](packaging/windows/README-中文.md)。

### 首次使用

1. 启动 Pi_zaya，浏览器会自动打开本地页面。
2. 按首次使用引导进入“设置”，填写文本模型 API Key。
3. 系统只会在能够可靠识别服务商时读取模型列表；普通 `sk-` Key 不会被发送
   给多家服务尝试。
4. 无法自动识别时，从下拉框选择 Qwen、DeepSeek、OpenAI 或自定义
   OpenAI 兼容服务，再选择模型或手动输入模型 ID。
5. 模型发现和连接测试都有短超时且不自动重试；网络不可用或模型不匹配时会
   回退到内置选项或手动输入，不会一直卡在加载状态。
6. 在“文库”导入 PDF，等待转换和索引完成，然后进入“问答”提出问题。
7. 点击回答中的引用，可在阅读器中定位原文证据。

AI 问答和研究工作流需要文本模型。复杂扫描件、公式和图片较多的论文建议另外
配置支持图片输入的 Qwen 视觉模型，以获得更好的 PDF 转换质量。

### 数据、更新和转换恢复

- 用户数据库、PDF、转换后的 Markdown、偏好、备份、运行状态和日志默认位于
  `%LOCALAPPDATA%\Pi_zaya`，与安装或解压目录分离。
- 安装版升级时先从托盘退出，再运行新版安装器。卸载程序只删除软件和快捷方式，
  不删除个人文库。
- 便携版升级时先退出，把新版 ZIP 解压到新的程序目录，再启动新版。
- 如果转换期间退出或后台异常结束，重启后任务会显示为已中断，不会一直显示
  “转换中”，也不会自动继续调用付费模型。
- 点击“继续转换”或“全部继续”后，软件会检查源 PDF 和视觉 API 配置，并复用
  已完成且校验有效的页面。源文件或 Key 缺失时会保留明确提示，不会无限重试。
- 自动更新尚未实现；当前版本应明确视为 beta，而不是正式稳定版。

## 主要能力

| 能力 | 作用 |
|---|---|
| 带锚点的 PDF 转换 | 转换为 Markdown，同时尽量保留页码标记、章节、图片、公式和原文定位信息。 |
| 可恢复转换任务 | 持久化排队和运行中的转换任务，重启后由用户明确继续，并复用已完成页面，不自动消费模型额度。 |
| 基于证据的问答 | 在本地索引中检索证据，构造 RAG 上下文并生成有来源约束的回答。 |
| 引用追踪 | 展示回答证据、来源卡片、参考文献上下文和阅读器定位目标。 |
| 项目文献篮 | 收集论文与摘录，维持项目研究上下文并导出引用。 |
| 研究笔记 | 保存带精确原文位置的文字、表格、公式与图片证据摘录；支持个人批注、标签、检索、置顶、归档、回到来源、跨论文证据整理，以及 Markdown/Word 备份。导出用于继续阅读和证据复核，不定位为投稿材料。 |
| 证据矩阵 | 按项目和版本整理方法、实验、指标、结果与限制；有事实的单元格绑定同一论文的精确证据，缺失内容保持为空。 |
| 来源补充选择 | 默认仅使用文献库；也可按会话选择“模型补充”或“网页补充”。两者只在库内证据不足时启用，网页失败不会偷偷切换为模型知识。 |
| Research Agent | 默认在问答流程中执行规划、来源策略、工具轨迹、证据充分性和逐句引用支持核验。 |
| 质量工具 | 扫描转换质量、执行安全修复、重建索引并跟踪元数据和参考文献同步。 |

## 架构

```mermaid
flowchart LR
  A["PDF 文库"] --> B["PDF 转 Markdown"]
  B --> C["分块与结构化索引"]
  C --> D["BM25 + 结构化检索"]
  D --> E["Research Agent Runtime"]
  E --> P["规划与意图路由"]
  E --> R["证据检索"]
  E --> X["参考文献解析"]
  E --> V["主张核验"]
  L["项目文献篮"] --> EM["持久化证据矩阵"]
  P --> O["有证据的回答与引用界面"]
  R --> O
  X --> O
  V --> O
  EM --> O
```

整体数据流：

1. PDF 转换为保留页码和来源位置的 Markdown。
2. Markdown 被分块，并建立正文、参考文献、图片和阅读器导航索引。
3. 检索根据当前论文、文献篮或全库范围返回候选证据。
4. Research Agent 默认规划工具调用、判断证据是否充分、生成回答并核验引用支持。
   只有用户明确选择时，证据门禁才会在库内证据不足后使用模型知识或网页检索。
5. 项目文献篮可生成持久化、版本化证据矩阵；已填事实必须来自同一论文并具有
   可定位证据。

当前发布版的候选召回以 BM25、查询翻译/扩展和结构化索引为主，尚未启用
embedding、Dense 候选召回或向量数据库。代码库正在独立开发 BM25 + Dense
融合能力；只有完成索引持久化、用户授权、隐私提示和检索评测后才会进入产品链路。

主要后端入口：

- `api/main.py`：FastAPI 应用。
- `api/routers/chat.py`：聊天、消息和 Research Agent API。
- `api/routers/library.py`：文库、转换、质量、元数据和索引 API。
- `api/routers/evidence_matrices.py`：证据矩阵、版本、审计和导出。
- `kb/task_runtime.py`：后台问答和转换运行时。
- `kb/evidence_matrix.py`：结构化证据比较与质量契约。

主要前端入口：

- `web/src/main.tsx`：React 入口。
- `web/src/pages/ChatPage.tsx`：问答工作区。
- `web/src/pages/LibraryPage.tsx`：PDF 和文库工作区。
- `web/src/components/chat/AgentTracePanel.tsx`：Agent 轨迹界面。
- `web/src/components/chat/EvidenceMatrixWorkspace.tsx`：证据矩阵工作区。

## Research Agent 与来源补充

Research Agent 是问答的默认运行方式，不替换检索、提示构造或引用卡片。回答可
携带问题类型、查询范围、规划步骤、工具调用、证据矩阵和逐句支持核验；主回答
保持简洁，执行细节默认收起，需要时再展开检查。

输入框中的“补充来源”按会话保存，有三个选项：

- “仅文献库”（默认）：证据不足时直接说明不足。
- “模型补充”：只有库内证据不足时才允许模型知识，不发起网页检索。
- “网页补充”：只有库内证据不足时才尝试网页检索；不可用或失败时不改用模型知识。

所有外部内容都会与库内证据区分并明确披露，不会伪装成文献库已经验证的结论。
没有文本模型 Key 时，Agent 仍可降级返回检索证据和来源核对结果。

## 从源码开发

本节只面向贡献者。使用 Windows 安装版或便携 ZIP 的用户不需要安装 Python、
Node.js 或 npm。

源码环境要求：

- Python `3.10.11`，见 `.python-version`。
- Node.js `24.13.0`，见 `.nvmrc`。
- 完整问答至少配置一个文本模型 Key：`QWEN_API_KEY`、
  `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`。

安装依赖：

```powershell
Copy-Item .env.example .env

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd web
npm ci
cd ..
```

启动开发界面和 API：

```powershell
.\run_new.ps1 -StopExisting
```

也可以使用等价包装：

```powershell
.\run.ps1 -StopExisting
```

开发地址：

- React：`http://127.0.0.1:5173/`
- FastAPI：`http://127.0.0.1:8000/`

生产式单服务本地运行：

```powershell
cd web
npm run build
cd ..
python server.py
```

## 常用配置

- `QWEN_API_KEY`、`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`：文本模型访问。
- `QWEN_BASE_URL`、`DEEPSEEK_BASE_URL`、`OPENAI_BASE_URL`：可选服务地址。
- `QWEN_TEXT_MODEL`、`QWEN_VISION_MODEL`：Qwen 文本和 PDF 视觉转换模型。
- `KB_PDF_DIR`：源 PDF 目录。
- `KB_MD_DIR`：转换后 Markdown 目录。
- `KB_DB_DIR`：检索和索引目录。
- `KB_CHAT_DB`：聊天 SQLite 路径。
- `KB_LIBRARY_DB`：文库 SQLite 路径。
- `KB_CROSSREF_BUDGET_S`：Crossref 同步时间预算。

开发环境可复制 `.env.example`，生产式部署才使用 `.env.production.example`。
设置界面也可把本地 API 偏好保存到 `user_prefs.json`；环境变量和 `.env` 优先。

## 典型研究流程

1. 打开“文库”，上传或选择 PDF。
2. 转换为 Markdown 并检查转换质量。
3. 必要时执行修复或重新转换，再重建知识库索引。
4. 在当前论文、文献篮或全库范围提问。
5. 从回答引用卡片进入阅读器，核对精确原文。
6. 把重要论文加入项目文献篮，生成并核验证据矩阵，缺失证据保持为空。
7. 将带原文位置的证据保存为研究笔记，或将矩阵导出为 Markdown、CSV 或 XLSX，
   用于继续阅读和复核。

## 质量与发布

项目维护后端单元和 sanity 测试、Ruff、研究问答回放、证据比较、项目旅程、
转换结构质量、前端 lint/build、浏览器 smoke、核心引用/阅读器回归和普通用户
界面隔离门禁。Windows 发布还必须通过无标签预检、嵌入式运行时打包、干净配置
启动、安装器覆盖升级、卸载、用户数据保留、校验和与签名声明检查。

评估维度和可复现命令见 [docs/EVAL_DASHBOARD.md](docs/EVAL_DASHBOARD.md)，
发布边界和历史验收见 [docs/RELEASE_RUNBOOK.md](docs/RELEASE_RUNBOOK.md)。文档不会
把未测量的数据写成性能结论。

## 本地数据文件

| 路径 | 用途 |
|---|---|
| `chat.sqlite3` | 会话、消息、引用和聊天元数据。 |
| `library.sqlite3` | 文库元数据、来源和转换任务记录。 |
| `db/` | 文档、分块、参考文献索引和 Crossref 缓存。 |
| `backups/` | 手动和自动备份。 |

这些运行数据不应提交到 Git。

## 开发注意事项

- 保留 Markdown 页码标记 `<!-- kb_page: N -->`。
- React API 契约应在 `web/src/api` 中保持类型化。
- 质量问题应优先在转换器、检索或数据流源头修复，不要只隐藏界面状态。
- 后台任务和共享状态必须保持线程安全。
- 高风险操作前先创建或验证备份。

## 开发与测试

- 开发：LittlePyx
- 测试：Izaya

## 许可证

Pi_zaya 使用 MIT License，完整条款见 [LICENSE](LICENSE)。
