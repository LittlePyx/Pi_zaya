# Pi_zaya Windows portable package

中文用户请直接阅读同目录的 `README-中文.md`。

## Start

1. Extract the entire ZIP to a normal folder. Do not run it inside the ZIP preview.
2. Double-click `Pi_zaya.exe`.
3. Pi_zaya opens in your default browser and stays available from the Windows system tray. It prefers `http://127.0.0.1:8000/` and automatically selects another loopback port if that port is occupied.
4. On a new profile, use the first-run guide to configure a text-model API key. Provider/model detection uses a short timeout; ambiguous keys require an explicit provider choice and are never probed across multiple services. The model dropdown always permits a manual model ID. Configure a Qwen vision key for the best PDF conversion quality. You can dismiss the guide and continue using local library and reader tools first.

The package includes its own Python runtime and prebuilt React frontend. Node.js and a system Python installation are not required.

Double-clicking `Pi_zaya.exe` again opens the already-running application instead of starting a duplicate backend. The tray menu can open Pi_zaya, the log folder, or the user-data folder, and safely stop the application. `Start-Pi-zaya.cmd` and `Stop-Pi-zaya.cmd` remain available as diagnostic fallbacks.

If Pi_zaya stops during PDF conversion, the Library shows the task as interrupted after restart. It never resumes paid-model work automatically. Use **Continue conversion** or **Continue all** to validate the source/API settings and reuse completed page-cache artifacts. A missing source or API key remains an actionable recovery notice instead of retrying indefinitely.

## Stop and update

- Use **Exit Pi_zaya** from the system-tray menu before replacing or moving the application folder. If you used the command fallback, run `Stop-Pi-zaya.cmd`.
- To update, stop Pi_zaya, extract the new ZIP into a new application folder, and start it. User data is not stored in the application folder, so it remains available.
- Verify the downloaded ZIP with the adjacent `.sha256` file before extracting it.

## Data and logs

User data is stored under `%LOCALAPPDATA%\Pi_zaya` by default, including the library databases, uploaded PDFs, converted Markdown, backups, preferences, runtime state, and logs. Deleting the extracted application folder does not delete user data.

Pi_zaya binds only to `127.0.0.1` in desktop mode. It is not configured as a public network service.

Pi_zaya is distributed under the MIT License. See `LICENSE` in this folder.

---

# Pi_zaya Windows 便携版

更完整的中文安装、API 模型配置、更新和校验说明见同目录的 `README-中文.md`。

## 启动

1. 将 ZIP 完整解压到普通文件夹，不要在压缩包预览窗口内直接运行。
2. 双击 `Pi_zaya.exe`。
3. Pi_zaya 会在默认浏览器中打开，并常驻 Windows 系统托盘。默认使用 `http://127.0.0.1:8000/`；端口被占用时会自动选择其他本地端口。
4. 新用户可按首次启动引导填写文本模型 API Key。系统只会在可靠识别供应商后读取模型；无法识别时请选择供应商，再从模型下拉框选择或手动输入模型 ID。识别过程有短超时，不会无限等待。若需最佳 PDF 转换质量，请同时配置 Qwen 视觉模型 Key。也可以暂时关闭引导，先使用本地文库与阅读功能。

软件包已经包含 Python 运行时和构建好的 React 前端，不要求用户另行安装 Node.js 或 Python。

再次双击 `Pi_zaya.exe` 只会打开已经运行的页面，不会重复启动后台服务。托盘菜单可以打开 Pi_zaya、日志目录和数据目录，也可以安全退出。`Start-Pi-zaya.cmd` 与 `Stop-Pi-zaya.cmd` 继续作为诊断兜底。

如果 PDF 转换过程中退出或后台异常停止，重启后“文库”会把任务显示为已中断，不会自动调用付费模型。点击“继续转换”或“全部继续”后，软件会检查源文件和 API 配置，并优先复用已完成页面；源 PDF 或 API Key 缺失时会保留明确的可恢复提示，不会无限重试。

## 停止与更新

- 替换或移动程序文件夹前，请从系统托盘选择“退出 Pi_zaya”；若使用 `.cmd` 兜底启动，则双击 `Stop-Pi-zaya.cmd`。
- 更新时先停止旧版，将新版 ZIP 解压到新的程序文件夹后再启动。用户数据不在程序目录中，不会因替换程序目录而丢失。
- 解压前请使用随附的 `.sha256` 文件核对下载包。

## 数据与日志

用户数据库、上传的 PDF、转换后的 Markdown、备份、偏好设置、运行状态和日志默认保存在 `%LOCALAPPDATA%\Pi_zaya`。删除解压后的程序目录不会删除用户数据。

桌面模式只监听 `127.0.0.1`，不会自动作为公网服务开放。

Pi_zaya 使用 MIT 许可证分发，完整条款见本目录中的 `LICENSE`。
