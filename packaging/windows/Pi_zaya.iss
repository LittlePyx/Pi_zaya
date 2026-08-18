#define AppVersion GetEnv("PI_ZAYA_INSTALLER_VERSION")
#define AppNumericVersion GetEnv("PI_ZAYA_INSTALLER_NUMERIC_VERSION")
#define StageRoot GetEnv("PI_ZAYA_INSTALLER_STAGE")
#define OutputRoot GetEnv("PI_ZAYA_INSTALLER_OUTPUT")
#define OutputBaseName GetEnv("PI_ZAYA_INSTALLER_BASENAME")
#define SigningEnabled GetEnv("PI_ZAYA_INSTALLER_SIGNING")

#if AppVersion == ""
  #error PI_ZAYA_INSTALLER_VERSION is required
#endif
#if StageRoot == ""
  #error PI_ZAYA_INSTALLER_STAGE is required
#endif

[Setup]
AppId={{BDA27978-68AE-4C98-8E35-C85D11872562}
AppName=Pi_zaya
AppVersion={#AppVersion}
AppVerName=Pi_zaya {#AppVersion}
AppPublisher=LittlePyx
AppPublisherURL=https://github.com/LittlePyx/Pi_zaya
AppSupportURL=https://github.com/LittlePyx/Pi_zaya/issues
AppUpdatesURL=https://github.com/LittlePyx/Pi_zaya/releases
DefaultDirName={localappdata}\Programs\Pi_zaya
DefaultGroupName=Pi_zaya
DisableProgramGroupPage=auto
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputRoot}
OutputBaseFilename={#OutputBaseName}
SetupIconFile={#StageRoot}\Pi_zaya.ico
LicenseFile={#StageRoot}\LICENSE
UninstallDisplayIcon={app}\Pi_zaya.exe
UninstallDisplayName=Pi_zaya
VersionInfoVersion={#AppNumericVersion}
VersionInfoCompany=LittlePyx
VersionInfoDescription=Pi_zaya Windows 安装程序
VersionInfoProductName=Pi_zaya
VersionInfoProductVersion={#AppNumericVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
AppMutex=Local\Pi_zaya.WindowsLauncher
SetupLogging=yes
#if SigningEnabled == "1"
SignTool=PiZayaAuthenticode
SignedUninstaller=yes
#else
SignedUninstaller=no
#endif

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："; Flags: unchecked

[Files]
Source: "{#StageRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Pi_zaya"; Filename: "{app}\Pi_zaya.exe"; WorkingDir: "{app}"
Name: "{group}\Pi_zaya 中文使用说明"; Filename: "{app}\README-中文.md"
Name: "{group}\卸载 Pi_zaya"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Pi_zaya"; Filename: "{app}\Pi_zaya.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Pi_zaya.exe"; Description: "启动 Pi_zaya"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ""{app}\Stop-Pi-zaya.ps1"""; WorkingDir: "{app}"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "StopPiZayaBackend"

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
  StopScript: String;
begin
  Result := '';
  StopScript := ExpandConstant('{app}\Stop-Pi-zaya.ps1');
  if FileExists(StopScript) then
  begin
    if not Exec(
      ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
      '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + StopScript + '"',
      ExpandConstant('{app}'),
      SW_HIDE,
      ewWaitUntilTerminated,
      ResultCode
    ) then
      Result := '无法停止已安装的 Pi_zaya 后台服务。请退出系统托盘中的 Pi_zaya 后重试。'
    else if ResultCode <> 0 then
      Result := '已安装的 Pi_zaya 后台服务未能安全停止。请退出系统托盘中的 Pi_zaya 后重试。';
  end;
end;
