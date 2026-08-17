# Pi_zaya Windows portable package

## Start

1. Extract the entire ZIP to a normal folder. Do not run it inside the ZIP preview.
2. Double-click `Start-Pi-zaya.cmd`.
3. Pi_zaya opens `http://127.0.0.1:8000/` in your browser.
4. Configure a text-model API key in Settings. Configure a Qwen vision key for the best PDF conversion quality.

The package includes its own Python runtime and prebuilt React frontend. Node.js and a system Python installation are not required.

## Stop and update

- Double-click `Stop-Pi-zaya.cmd` before replacing or moving the application folder.
- To update, stop Pi_zaya, extract the new ZIP into a new application folder, and start it. User data is not stored in the application folder, so it remains available.
- Verify the downloaded ZIP with the adjacent `.sha256` file before extracting it.

## Data and logs

User data is stored under `%LOCALAPPDATA%\Pi_zaya` by default, including the library databases, uploaded PDFs, converted Markdown, backups, preferences, runtime state, and logs. Deleting the extracted application folder does not delete user data.

Pi_zaya binds only to `127.0.0.1` in desktop mode. It is not configured as a public network service.

Pi_zaya is distributed under the MIT License. See `LICENSE` in this folder.

---

# Pi_zaya Windows 便携版

## 启动

1. 将 ZIP 完整解压到普通文件夹，不要在压缩包预览窗口内直接运行。
2. 双击 `Start-Pi-zaya.cmd`。
3. Pi_zaya 会在浏览器中打开 `http://127.0.0.1:8000/`。
4. 在“设置”中填写文本模型 API Key；若需最佳 PDF 转换质量，请同时配置 Qwen 视觉模型 Key。

软件包已经包含 Python 运行时和构建好的 React 前端，不要求用户另行安装 Node.js 或 Python。

## 停止与更新

- 替换或移动程序文件夹前，双击 `Stop-Pi-zaya.cmd`。
- 更新时先停止旧版，将新版 ZIP 解压到新的程序文件夹后再启动。用户数据不在程序目录中，不会因替换程序目录而丢失。
- 解压前请使用随附的 `.sha256` 文件核对下载包。

## 数据与日志

用户数据库、上传的 PDF、转换后的 Markdown、备份、偏好设置、运行状态和日志默认保存在 `%LOCALAPPDATA%\Pi_zaya`。删除解压后的程序目录不会删除用户数据。

桌面模式只监听 `127.0.0.1`，不会自动作为公网服务开放。

Pi_zaya 使用 MIT 许可证分发，完整条款见本目录中的 `LICENSE`。
