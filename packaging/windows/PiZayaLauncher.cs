using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("Pi_zaya")]
[assembly: System.Reflection.AssemblyProduct("Pi_zaya Windows Launcher")]
[assembly: System.Reflection.AssemblyCompany("LittlePyx")]
[assembly: System.Reflection.AssemblyCopyright("Copyright (c) 2026 LittlePyx")]

namespace PiZaya.WindowsLauncher
{
    internal sealed class LaunchOptions
    {
        public bool NoBrowser;
        public bool NoTray;
        public string DataDirectory;
        public int Port;

        public static LaunchOptions Parse(string[] args)
        {
            LaunchOptions options = new LaunchOptions();
            for (int index = 0; index < args.Length; index++)
            {
                string arg = args[index] ?? String.Empty;
                if (arg.Equals("--no-browser", StringComparison.OrdinalIgnoreCase))
                {
                    options.NoBrowser = true;
                }
                else if (arg.Equals("--no-tray", StringComparison.OrdinalIgnoreCase))
                {
                    options.NoTray = true;
                }
                else if (arg.Equals("--data-dir", StringComparison.OrdinalIgnoreCase))
                {
                    if (++index >= args.Length)
                    {
                        throw new ArgumentException("--data-dir requires a path.");
                    }
                    options.DataDirectory = args[index];
                }
                else if (arg.Equals("--port", StringComparison.OrdinalIgnoreCase))
                {
                    if (++index >= args.Length || !Int32.TryParse(args[index], out options.Port) || options.Port < 0 || options.Port > 65535)
                    {
                        throw new ArgumentException("--port requires a number from 0 to 65535.");
                    }
                }
                else
                {
                    throw new ArgumentException("Unknown launcher option: " + arg);
                }
            }
            return options;
        }
    }

    internal sealed class LaunchResult
    {
        public bool Success;
        public string Url;
        public string Error;
        public string DataDirectory;
    }

    internal static class BackendController
    {
        private const int StartTimeoutMilliseconds = 60000;
        private const int StopTimeoutMilliseconds = 20000;

        public static string AppRoot
        {
            get { return Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory); }
        }

        public static string ResolveDataDirectory(string explicitDirectory)
        {
            string configured = explicitDirectory;
            if (String.IsNullOrWhiteSpace(configured))
            {
                configured = Environment.GetEnvironmentVariable("KB_APP_DATA_DIR");
            }
            if (String.IsNullOrWhiteSpace(configured))
            {
                configured = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Pi_zaya");
            }
            return Path.GetFullPath(Environment.ExpandEnvironmentVariables(configured));
        }

        public static LaunchResult Start(LaunchOptions options)
        {
            string dataDirectory = ResolveDataDirectory(options.DataDirectory);
            string script = Path.Combine(AppRoot, "Start-Pi-zaya.ps1");
            if (!File.Exists(script))
            {
                return Failed(dataDirectory, "启动脚本不存在。请完整解压安装包后重试。\n\n" + script);
            }

            StringBuilder command = new StringBuilder();
            command.Append("& ").Append(PowerShellLiteral(script)).Append(" -NoBrowser");
            if (!String.IsNullOrWhiteSpace(options.DataDirectory))
            {
                command.Append(" -DataDir ").Append(PowerShellLiteral(dataDirectory));
            }
            if (options.Port > 0)
            {
                command.Append(" -Port ").Append(options.Port);
            }
            command.Append("; exit $LASTEXITCODE");

            ProcessResult process = RunPowerShell(command.ToString(), StartTimeoutMilliseconds);
            if (process.TimedOut)
            {
                TryStop(dataDirectory);
                return Failed(dataDirectory, "启动超过 60 秒，已停止等待。请查看日志后重试。\n\n" + LogPath(dataDirectory));
            }
            if (process.ExitCode != 0)
            {
                return Failed(dataDirectory, "Pi_zaya 启动失败。\n\n" + UsefulError(process) + "\n\n日志：" + LogPath(dataDirectory));
            }

            int port;
            if (!TryReadRecordedPort(dataDirectory, out port))
            {
                TryStop(dataDirectory);
                return Failed(dataDirectory, "服务已启动，但无法读取实际端口。请查看日志。\n\n" + LogPath(dataDirectory));
            }
            LaunchResult result = new LaunchResult();
            result.Success = true;
            result.Url = "http://127.0.0.1:" + port + "/";
            result.DataDirectory = dataDirectory;
            return result;
        }

        public static string Stop(string dataDirectory)
        {
            string script = Path.Combine(AppRoot, "Stop-Pi-zaya.ps1");
            if (!File.Exists(script))
            {
                return "停止脚本不存在：" + script;
            }
            string command = "& " + PowerShellLiteral(script) + " -DataDir " + PowerShellLiteral(dataDirectory) + "; exit $LASTEXITCODE";
            ProcessResult process = RunPowerShell(command, StopTimeoutMilliseconds);
            if (process.TimedOut)
            {
                return "安全停止超过 20 秒。服务可能仍在运行，请查看日志。";
            }
            if (process.ExitCode != 0)
            {
                return "无法安全停止 Pi_zaya。\n\n" + UsefulError(process);
            }
            return null;
        }

        public static void Open(string target)
        {
            ProcessStartInfo info = new ProcessStartInfo();
            info.FileName = target;
            info.UseShellExecute = true;
            Process.Start(info);
        }

        public static string LogPath(string dataDirectory)
        {
            return Path.Combine(dataDirectory, "logs");
        }

        private static LaunchResult Failed(string dataDirectory, string error)
        {
            LaunchResult result = new LaunchResult();
            result.Success = false;
            result.Error = error;
            result.DataDirectory = dataDirectory;
            return result;
        }

        private static void TryStop(string dataDirectory)
        {
            try { Stop(dataDirectory); }
            catch { }
        }

        private static bool TryReadRecordedPort(string dataDirectory, out int port)
        {
            port = 0;
            string recordPath = Path.Combine(dataDirectory, "runtime", "server-process.json");
            if (!File.Exists(recordPath))
            {
                return false;
            }
            string json = File.ReadAllText(recordPath, Encoding.UTF8);
            Match match = Regex.Match(json, "\\\"port\\\"\\s*:\\s*(?<port>[0-9]+)", RegexOptions.CultureInvariant);
            return match.Success && Int32.TryParse(match.Groups["port"].Value, out port) && port > 0 && port <= 65535;
        }

        private static string UsefulError(ProcessResult process)
        {
            string text = String.IsNullOrWhiteSpace(process.Error) ? process.Output : process.Error;
            if (String.IsNullOrWhiteSpace(text))
            {
                return "启动脚本未返回详细错误。";
            }
            text = text.Trim();
            return text.Length <= 1600 ? text : text.Substring(text.Length - 1600);
        }

        private static string PowerShellLiteral(string value)
        {
            return "'" + (value ?? String.Empty).Replace("'", "''") + "'";
        }

        private static ProcessResult RunPowerShell(string command, int timeoutMilliseconds)
        {
            string encoded = Convert.ToBase64String(Encoding.Unicode.GetBytes(command));
            string powershell = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "WindowsPowerShell", "v1.0", "powershell.exe");
            if (!File.Exists(powershell))
            {
                powershell = "powershell.exe";
            }

            ProcessStartInfo info = new ProcessStartInfo();
            info.FileName = powershell;
            info.Arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand " + encoded;
            info.WorkingDirectory = AppRoot;
            info.UseShellExecute = false;
            info.CreateNoWindow = true;
            info.WindowStyle = ProcessWindowStyle.Hidden;
            info.RedirectStandardOutput = true;
            info.RedirectStandardError = true;

            StringBuilder output = new StringBuilder();
            StringBuilder error = new StringBuilder();
            using (Process process = new Process())
            {
                process.StartInfo = info;
                process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs args) { if (args.Data != null) output.AppendLine(args.Data); };
                process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs args) { if (args.Data != null) error.AppendLine(args.Data); };
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                if (!process.WaitForExit(timeoutMilliseconds))
                {
                    try { process.Kill(); }
                    catch { }
                    return new ProcessResult(-1, true, output.ToString(), error.ToString());
                }
                // Do not call the unbounded WaitForExit() overload here. A detached
                // backend can temporarily retain inherited pipe handles after the
                // PowerShell parent exits, which would make the GUI launcher hang.
                Thread.Sleep(100);
                return new ProcessResult(process.ExitCode, false, output.ToString(), error.ToString());
            }
        }

        private sealed class ProcessResult
        {
            public readonly int ExitCode;
            public readonly bool TimedOut;
            public readonly string Output;
            public readonly string Error;

            public ProcessResult(int exitCode, bool timedOut, string output, string error)
            {
                ExitCode = exitCode;
                TimedOut = timedOut;
                Output = output;
                Error = error;
            }
        }
    }

    internal sealed class TrayApplicationContext : ApplicationContext
    {
        private readonly LaunchOptions options;
        private readonly EventWaitHandle openSignal;
        private readonly NotifyIcon trayIcon;
        private readonly ToolStripMenuItem openItem;
        private readonly ToolStripMenuItem statusItem;
        private readonly ToolStripMenuItem exitItem;
        private readonly System.Windows.Forms.Timer timer;
        private Task<LaunchResult> startupTask;
        private Task<string> stopTask;
        private LaunchResult launch;
        private bool openRequested;
        private bool stopRequested;

        public TrayApplicationContext(LaunchOptions launchOptions, EventWaitHandle signal)
        {
            options = launchOptions;
            openSignal = signal;

            openItem = new ToolStripMenuItem("打开 Pi_zaya");
            openItem.Enabled = false;
            openItem.Click += delegate { OpenApplication(); };

            statusItem = new ToolStripMenuItem("正在启动…");
            statusItem.Enabled = false;

            ToolStripMenuItem logsItem = new ToolStripMenuItem("打开日志目录");
            logsItem.Click += delegate { OpenDirectory(BackendController.LogPath(CurrentDataDirectory())); };
            ToolStripMenuItem dataItem = new ToolStripMenuItem("打开数据目录");
            dataItem.Click += delegate { OpenDirectory(CurrentDataDirectory()); };

            exitItem = new ToolStripMenuItem("退出 Pi_zaya");
            exitItem.Click += delegate { BeginStop(); };

            ContextMenuStrip menu = new ContextMenuStrip();
            menu.Items.Add(openItem);
            menu.Items.Add(statusItem);
            menu.Items.Add(new ToolStripSeparator());
            menu.Items.Add(logsItem);
            menu.Items.Add(dataItem);
            menu.Items.Add(new ToolStripSeparator());
            menu.Items.Add(exitItem);

            Icon icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath) ?? SystemIcons.Application;
            trayIcon = new NotifyIcon();
            trayIcon.Icon = icon;
            trayIcon.Text = "Pi_zaya 正在启动";
            trayIcon.ContextMenuStrip = menu;
            trayIcon.Visible = true;
            trayIcon.DoubleClick += delegate { OpenApplication(); };

            timer = new System.Windows.Forms.Timer();
            timer.Interval = 250;
            timer.Tick += OnTimer;
            timer.Start();

            startupTask = Task.Factory.StartNew(delegate { return BackendController.Start(options); });
        }

        private void OnTimer(object sender, EventArgs args)
        {
            if (openSignal.WaitOne(0))
            {
                openRequested = true;
                if (!stopRequested)
                {
                    OpenApplication();
                }
            }
            if (startupTask != null && startupTask.IsCompleted)
            {
                LaunchResult completed;
                try { completed = startupTask.Result; }
                catch (Exception exception)
                {
                    completed = new LaunchResult();
                    completed.Success = false;
                    completed.Error = exception.Message;
                    completed.DataDirectory = BackendController.ResolveDataDirectory(options.DataDirectory);
                }
                startupTask = null;
                launch = completed;
                if (stopRequested)
                {
                    StartStopTask();
                    return;
                }
                if (completed.Success)
                {
                    statusItem.Text = "运行中：" + completed.Url;
                    openItem.Enabled = true;
                    trayIcon.Text = "Pi_zaya 正在运行";
                    trayIcon.ShowBalloonTip(2500, "Pi_zaya", "Pi_zaya 已启动。关闭时请使用托盘菜单“退出 Pi_zaya”。", ToolTipIcon.Info);
                    if (!options.NoBrowser || openRequested)
                    {
                        OpenApplication();
                    }
                }
                else
                {
                    statusItem.Text = "启动失败";
                    trayIcon.Text = "Pi_zaya 启动失败";
                    MessageBox.Show(completed.Error, "Pi_zaya 启动失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
            if (stopTask != null && stopTask.IsCompleted)
            {
                string error;
                try { error = stopTask.Result; }
                catch (Exception exception) { error = exception.Message; }
                stopTask = null;
                if (String.IsNullOrWhiteSpace(error))
                {
                    trayIcon.Visible = false;
                    ExitThread();
                }
                else
                {
                    statusItem.Text = launch != null && launch.Success ? "运行中：" + launch.Url : "停止失败";
                    stopRequested = false;
                    exitItem.Enabled = true;
                    MessageBox.Show(error, "Pi_zaya 无法安全退出", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private string CurrentDataDirectory()
        {
            return launch != null && !String.IsNullOrWhiteSpace(launch.DataDirectory)
                ? launch.DataDirectory
                : BackendController.ResolveDataDirectory(options.DataDirectory);
        }

        private void OpenApplication()
        {
            if (launch == null || !launch.Success || String.IsNullOrWhiteSpace(launch.Url))
            {
                return;
            }
            try { BackendController.Open(launch.Url); }
            catch (Exception exception) { MessageBox.Show(exception.Message, "无法打开浏览器", MessageBoxButtons.OK, MessageBoxIcon.Warning); }
        }

        private void OpenDirectory(string path)
        {
            try
            {
                Directory.CreateDirectory(path);
                BackendController.Open(path);
            }
            catch (Exception exception) { MessageBox.Show(exception.Message, "无法打开目录", MessageBoxButtons.OK, MessageBoxIcon.Warning); }
        }

        private void BeginStop()
        {
            if (stopRequested)
            {
                return;
            }
            stopRequested = true;
            exitItem.Enabled = false;
            openItem.Enabled = false;
            if (startupTask != null)
            {
                statusItem.Text = "启动完成后将安全退出…";
                return;
            }
            StartStopTask();
        }

        private void StartStopTask()
        {
            statusItem.Text = "正在安全停止…";
            stopTask = Task.Factory.StartNew(delegate { return BackendController.Stop(CurrentDataDirectory()); });
        }

        protected override void ExitThreadCore()
        {
            timer.Stop();
            trayIcon.Visible = false;
            trayIcon.Dispose();
            base.ExitThreadCore();
        }
    }

    internal static class Program
    {
        private const string MutexName = "Local\\Pi_zaya.WindowsLauncher";
        private const string OpenEventName = "Local\\Pi_zaya.WindowsLauncher.Open";

        [STAThread]
        private static int Main(string[] args)
        {
            LaunchOptions options;
            try { options = LaunchOptions.Parse(args); }
            catch (Exception exception)
            {
                MessageBox.Show(exception.Message, "Pi_zaya 启动参数错误", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 2;
            }

            if (options.NoTray)
            {
                LaunchResult result = BackendController.Start(options);
                if (!result.Success)
                {
                    return 3;
                }
                if (!options.NoBrowser)
                {
                    try { BackendController.Open(result.Url); }
                    catch { return 4; }
                }
                return 0;
            }

            bool ownsMutex;
            using (Mutex mutex = new Mutex(true, MutexName, out ownsMutex))
            {
                if (!ownsMutex)
                {
                    SignalExistingLauncher();
                    return 0;
                }
                using (EventWaitHandle openSignal = new EventWaitHandle(false, EventResetMode.AutoReset, OpenEventName))
                {
                    Application.EnableVisualStyles();
                    Application.SetCompatibleTextRenderingDefault(false);
                    Application.Run(new TrayApplicationContext(options, openSignal));
                }
            }
            return 0;
        }

        private static void SignalExistingLauncher()
        {
            for (int attempt = 0; attempt < 20; attempt++)
            {
                try
                {
                    using (EventWaitHandle signal = EventWaitHandle.OpenExisting(OpenEventName))
                    {
                        signal.Set();
                        return;
                    }
                }
                catch (WaitHandleCannotBeOpenedException)
                {
                    Thread.Sleep(50);
                }
            }
        }
    }
}
