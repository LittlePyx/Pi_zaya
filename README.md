# kb_chat

这份 README 是我写给你（合作者）看的：帮你在**你自己的电脑**上跑起来 kb_chat，并且用**你自己的 PDF/目录**，同时我这边更新代码后你也能很快用到最新版。

---

## 0) 你需要准备什么

在 Windows 上：
- Git（能在 PowerShell 里运行 `git`）
- Python（建议 3.10+；Anaconda 也可以，只要命令行里能用 `python`）
- 一个 DeepSeek API Key（我不会让你把 key 写进代码）

---

## 1) 一键启动（推荐）

第一次启动会自动：
- 创建虚拟环境 `.venv`
- 安装依赖
- 启动网页（Streamlit）

### 1.1 克隆代码

```powershell
git clone https://github.com/LittlePyx/Pi_zaya.git
cd Pi_zaya
```

### 1.2 配置你的 API Key（PowerShell）

```powershell
$env:DEEPSEEK_API_KEY="你的key"
```

（可选）如果你想改模型/地址：

```powershell
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

### 1.3 启动

双击 `run.cmd` 即可。  
或在 PowerShell 里运行：

```powershell
.\run.ps1
```

启动后在浏览器打开提示的地址（默认是 `http://127.0.0.1:8501`）。

---

## 2) 用你自己的 PDF/目录（每个人独立）

打开网页后，在左侧「设置」里把路径改成你自己的：
- `PDF 路径`：你存放 PDF 的根目录
- `DB 路径 / MD 路径`：你自己的知识库 Markdown 目录（或你建库时使用的 db 目录）

（可选）你也可以用环境变量直接指定默认 PDF 目录（不改 UI 也能生效）：

```powershell
$env:KB_PDF_DIR="D:\\papers"
```

这些设置只会写在你电脑本地的 `user_prefs.json`，不会影响我，也不会被更新覆盖。

---

## 2.2) 第一次建库（不然会提示 DB 为空）

第一次跑起来后，知识库 DB 默认是空的（还没有 chunks），这是正常的。你需要：
1) 去左侧切到「文献管理」
2) 设置你的 `PDF 路径`、`输出目录（Markdown）`、`DB 路径`
3) 先把 PDF 转成 MD（单篇或批量）
4) 点「更新知识库」

完成后回到「对话」页，再问问题就能正常做“参考定位”了。

---

## 2.1) 转 MD 的效果怎么和我一致

这个项目的 PDF→Markdown 转换器已经跟代码一起放在仓库里：`test2.py`。  
你在网页里点“转换/批量转换”时，会优先调用它，所以只要你拉到最新版，转换逻辑和我的是同一份代码。

想要“效果和我一样”，关键是你在同一个开关组合下跑：
- 如果我这边也勾选了“**不用 LLM（更快）**”，那你也勾选（不需要 API Key 也能转，但数学公式/表格可能没那么干净）。
- 如果我这边取消了“**不用 LLM（更快）**”（更准），那你也要取消，并在启动前设置：
  - `DEEPSEEK_API_KEY`
  - （可选）`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`

---

## 3) 我更新代码后，你怎么拿到最新版

你以后每次启动都建议走 `run.cmd` / `run.ps1`：
- 如果你的目录是通过 `git clone` 得来的，`run.ps1` 会在启动前自动 `git pull`（有 git 的情况下）
- 然后再启动 Streamlit

如果你想手动更新也可以（推荐在启动前做一次）：

```powershell
git pull --rebase
```

如果你也参与一起改代码（协作开发），请用 “分支 + PR” 的方式，避免直接在 `main` 上改：

```powershell
# 先切到 main 并拉最新
git checkout main
git pull --rebase

# 开新分支做改动
git checkout -b feature/your-change

# 改完提交并推送分支
git add .
git commit -m "your change"
git push -u origin feature/your-change
```

然后去 GitHub 页面发起 Pull Request（PR），我会 review 后合并。

---

## 4)（可选）让同一局域网的其它设备也能访问你这台电脑

默认只允许本机访问（安全）：`127.0.0.1:8501`  
如果你需要同一 Wi‑Fi 下其它设备访问（不建议在公共 Wi‑Fi 开）：

```powershell
$env:KB_STREAMLIT_ADDR="0.0.0.0"
.\run.ps1
```

端口也可改：

```powershell
$env:KB_STREAMLIT_PORT="8501"
.\run.ps1
```

---

## 常见问题

### 1) 报错 401 / key 无效

检查你是否在当前 PowerShell 里设置了 key：

```powershell
echo $env:DEEPSEEK_API_KEY
```

再重新设置：

```powershell
$env:DEEPSEEK_API_KEY="你的正确key"
```

### 2) 运行 `run.cmd` 说找不到 python

说明你的系统 PATH 里没有 `python`。解决方式：
- 安装 Python 后勾选 “Add Python to PATH”
- 或用 Anaconda Prompt 启动（确保 `python` 可用）

### 3) 运行 `run.ps1` 提示执行策略阻止

你可以继续用 `run.cmd`（它会用 Bypass 启动 PowerShell），或在管理员 PowerShell 里设置一次：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```


