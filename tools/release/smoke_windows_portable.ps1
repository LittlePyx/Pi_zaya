[CmdletBinding(DefaultParameterSetName = "Bundle")]
param(
    [Parameter(Mandatory = $true, ParameterSetName = "Bundle")]
    [string]$BundleRoot,
    [Parameter(Mandatory = $true, ParameterSetName = "Archive")]
    [string]$ArchivePath,
    [string]$DataDir = "",
    [switch]$KeepData,
    [switch]$AllowDirty,
    [switch]$CleanProfile
)

$ErrorActionPreference = "Stop"
$bundle = ""
$archiveExtractRoot = ""
$cleanProfileRoot = ""
$dataRoot = ""
$launchAttempted = $false
$portBlocker = $null
$explicitDataDir = [bool]$DataDir
$ownsDataDir = -not $explicitDataDir
$environmentNames = @(
    "PATH",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "TEMP",
    "TMP",
    "QWEN_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "KB_AGENT_WEB_SEARCH_API_KEY",
    "KB_RELEASE_MODE",
    "KB_APP_DATA_DIR",
    "KB_APP_ENV",
    "KB_APP_VERSION",
    "KB_BUILD_COMMIT",
    "KB_BUILD_TIME",
    "KB_SERVER_HOST",
    "KB_SERVER_PORT",
    "KB_SERVER_RELOAD",
    "KB_STARTUP_PREFLIGHT",
    "KB_STARTUP_STRICT",
    "KB_UPDATE_CHECK_ENABLED"
)
$savedEnvironment = @{}
foreach ($name in $environmentNames) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$releaseSmokeTempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar

function Assert-GeneratedTempPath([string]$Path, [string]$Label) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($releaseSmokeTempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label escaped the system temp directory: $resolved"
    }
}

try {
    if ($PSCmdlet.ParameterSetName -eq "Archive") {
        $archive = [IO.Path]::GetFullPath($ArchivePath)
        if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
            throw "Portable archive was not found: $archive"
        }
        $checksumPath = "$archive.sha256"
        if (-not (Test-Path -LiteralPath $checksumPath -PathType Leaf)) {
            throw "Portable archive checksum was not found: $checksumPath"
        }
        $expectedHash = (((Get-Content -LiteralPath $checksumPath -Raw).Trim() -split '\s+')[0]).ToLowerInvariant()
        if ($expectedHash -notmatch '^[0-9a-f]{64}$') {
            throw "Portable archive checksum is malformed."
        }
        $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $expectedHash) {
            throw "Portable archive checksum mismatch before extraction."
        }

        $archiveExtractRoot = Join-Path ([IO.Path]::GetTempPath()) ("pi-zaya-release-archive-" + [guid]::NewGuid().ToString("N"))
        Assert-GeneratedTempPath $archiveExtractRoot "Archive extraction directory"
        New-Item -ItemType Directory -Force -Path $archiveExtractRoot | Out-Null
        Expand-Archive -LiteralPath $archive -DestinationPath $archiveExtractRoot
        $expectedBundleName = [IO.Path]::GetFileNameWithoutExtension($archive)
        $bundle = Join-Path $archiveExtractRoot $expectedBundleName
        if (-not (Test-Path -LiteralPath $bundle -PathType Container)) {
            throw "Portable archive did not contain the expected root directory '$expectedBundleName'."
        }
    }
    else {
        $bundle = [IO.Path]::GetFullPath($BundleRoot)
    }

    foreach ($required in @("VERSION", "LICENSE", "release-manifest.json", "README-PORTABLE.md", "README-中文.md", "Pi_zaya.exe", "Pi_zaya.ico", "Start-Pi-zaya.cmd", "Stop-Pi-zaya.cmd", "Start-Pi-zaya.ps1", "Stop-Pi-zaya.ps1", "web\dist\index.html")) {
        if (-not (Test-Path -LiteralPath (Join-Path $bundle $required))) {
            throw "Portable bundle is missing $required"
        }
    }

    $manifest = Get-Content -LiteralPath (Join-Path $bundle "release-manifest.json") -Raw | ConvertFrom-Json
    if ($manifest.license -ne "MIT" -or $manifest.license_status -ne "included") {
        throw "Portable bundle does not declare an included MIT license."
    }
    if ([bool]$manifest.source_dirty -and -not $AllowDirty) {
        throw "Portable bundle was built from a dirty source tree."
    }
    if ($manifest.entrypoint -ne "Pi_zaya.exe" -or $manifest.fallback_entrypoint -ne "Start-Pi-zaya.cmd") {
        throw "Portable bundle does not declare the native launcher and command fallback."
    }
    if ($manifest.launcher -ne "native_windows_tray") {
        throw "Portable bundle does not declare the expected native tray launcher."
    }
    if ($CleanProfile -and $manifest.python_runtime -ne "embedded") {
        throw "Clean-profile acceptance requires the embedded Python runtime."
    }

    if ($CleanProfile) {
        $cleanProfileRoot = Join-Path ([IO.Path]::GetTempPath()) ("pi-zaya-clean-profile-" + [guid]::NewGuid().ToString("N"))
        Assert-GeneratedTempPath $cleanProfileRoot "Clean profile directory"
        $cleanUserRoot = Join-Path $cleanProfileRoot "User"
        $cleanLocalAppData = Join-Path $cleanUserRoot "AppData\Local"
        $cleanRoamingAppData = Join-Path $cleanUserRoot "AppData\Roaming"
        $cleanTemp = Join-Path $cleanProfileRoot "Temp"
        New-Item -ItemType Directory -Force -Path $cleanUserRoot, $cleanLocalAppData, $cleanRoamingAppData, $cleanTemp | Out-Null

        $env:USERPROFILE = $cleanUserRoot
        $env:LOCALAPPDATA = $cleanLocalAppData
        $env:APPDATA = $cleanRoamingAppData
        $env:TEMP = $cleanTemp
        $env:TMP = $cleanTemp
        $env:QWEN_API_KEY = $null
        $env:DEEPSEEK_API_KEY = $null
        $env:OPENAI_API_KEY = $null
        $env:KB_AGENT_WEB_SEARCH_API_KEY = $null
        $env:KB_APP_DATA_DIR = $null

        $systemPaths = @(
            (Join-Path $env:SystemRoot "System32"),
            $env:SystemRoot,
            (Join-Path $env:SystemRoot "System32\Wbem"),
            (Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0")
        ) | Where-Object { Test-Path -LiteralPath $_ }
        $env:PATH = $systemPaths -join [IO.Path]::PathSeparator

        foreach ($externalRuntime in @("python.exe", "python3.exe", "node.exe", "npm.cmd")) {
            if (Get-Command $externalRuntime -CommandType Application -ErrorAction SilentlyContinue) {
                throw "Clean profile unexpectedly exposes $externalRuntime on PATH."
            }
        }
    }

    if ($explicitDataDir) {
        $dataRoot = [IO.Path]::GetFullPath($DataDir)
    }
    elseif ($CleanProfile) {
        $dataRoot = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "Pi_zaya"))
    }
    else {
        $dataRoot = Join-Path ([IO.Path]::GetTempPath()) ("pi-zaya-release-smoke-" + [guid]::NewGuid().ToString("N"))
    }
    if ($ownsDataDir) {
        Assert-GeneratedTempPath $dataRoot "Generated smoke data directory"
    }

    # Hold the preferred port open while the launcher starts. This verifies that
    # the packaged entrypoint automatically selects another loopback port rather
    # than hanging or failing when the default/configured port is occupied.
    $portBlocker = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $portBlocker.Start()
    $occupiedPort = ([Net.IPEndPoint]$portBlocker.LocalEndpoint).Port
    $env:KB_SERVER_PORT = [string]$occupiedPort
    if ($explicitDataDir -or -not $CleanProfile) {
        $env:KB_APP_DATA_DIR = $dataRoot
    }
    $launchAttempted = $true
    $launcherInfo = [Diagnostics.ProcessStartInfo]::new()
    $launcherInfo.FileName = Join-Path $bundle "Pi_zaya.exe"
    $launcherInfo.WorkingDirectory = $bundle
    $launcherInfo.UseShellExecute = $false
    $launcherInfo.CreateNoWindow = $true
    [void]$launcherInfo.ArgumentList.Add("--no-browser")
    [void]$launcherInfo.ArgumentList.Add("--no-tray")
    $launcherProcess = [Diagnostics.Process]::new()
    $launcherProcess.StartInfo = $launcherInfo
    if (-not $launcherProcess.Start()) {
        throw "Native launcher process did not start."
    }
    if (-not $launcherProcess.WaitForExit(65000)) {
        $launcherProcess.Kill()
        throw "Native launcher did not finish its bounded startup within 65 seconds."
    }
    if ($launcherProcess.ExitCode -ne 0) {
        throw "Native launcher failed with exit code $($launcherProcess.ExitCode)."
    }

    $processRecordPath = Join-Path $dataRoot "runtime\server-process.json"
    if (-not (Test-Path -LiteralPath $processRecordPath)) {
        throw "Native launcher did not write its process record to the user data directory."
    }
    $processRecord = Get-Content -LiteralPath $processRecordPath -Raw | ConvertFrom-Json
    $port = [int]$processRecord.port
    if ($port -lt 1 -or $port -gt 65535) {
        throw "Native launcher recorded an invalid port '$port'."
    }
    if ($port -eq $occupiedPort) {
        throw "Native launcher did not avoid the occupied preferred port $occupiedPort."
    }
    $portBlocker.Stop()
    $portBlocker = $null

    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/health" -TimeoutSec 10
    $build = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/app/version" -TimeoutSec 10
    $settings = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/settings" -TimeoutSec 10
    $onboarding = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/app/onboarding-status" -TimeoutSec 10
    $frontPage = Invoke-WebRequest -Uri "http://127.0.0.1:$port/" -TimeoutSec 10 -UseBasicParsing
    $expectedVersion = (Get-Content -LiteralPath (Join-Path $bundle "VERSION") -Raw).Trim()
    if ($health.status -ne "ok") { throw "Health endpoint did not return ok." }
    if ($health.version -ne $expectedVersion) { throw "Health version '$($health.version)' did not match '$expectedVersion'." }
    if ($build.version -ne $expectedVersion) { throw "Build version '$($build.version)' did not match '$expectedVersion'." }
    if ($frontPage.StatusCode -ne 200) { throw "Frontend root returned HTTP $($frontPage.StatusCode)." }
    if ($CleanProfile) {
        $expectedDataRoot = if ($explicitDataDir) {
            [IO.Path]::GetFullPath($DataDir)
        }
        else {
            [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "Pi_zaya"))
        }
        if (-not $dataRoot.Equals($expectedDataRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Clean-profile launch did not use the expected isolated data directory."
        }
        if ([bool]$settings.has_api_key) {
            throw "Clean-profile launch unexpectedly inherited a text-model API key."
        }
        if ([string]$onboarding.current_step -ne "connect_model" -or [bool]$onboarding.completed) {
            throw "Clean-profile onboarding did not start at text-model setup."
        }
        $expectedPdfDir = [IO.Path]::GetFullPath((Join-Path $expectedDataRoot "pdfs"))
        $expectedMarkdownDir = [IO.Path]::GetFullPath((Join-Path $expectedDataRoot "markdown"))
        $actualPdfDir = [IO.Path]::GetFullPath([string]$settings.library_paths.pdf_dir)
        $actualMarkdownDir = [IO.Path]::GetFullPath([string]$settings.library_paths.md_dir)
        if (-not $actualPdfDir.Equals($expectedPdfDir, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Clean-profile PDF directory did not use the application-managed default."
        }
        if (-not $actualMarkdownDir.Equals($expectedMarkdownDir, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Clean-profile Markdown directory did not use the application-managed default."
        }
    }
    $mode = if ($CleanProfile) { "Clean-profile archive" } else { "Portable" }
    Write-Host "$mode smoke passed for Pi-zaya $expectedVersion on port $port."
}
finally {
    $stopError = $null
    if ($portBlocker) {
        $portBlocker.Stop()
        $portBlocker = $null
    }
    if ($launchAttempted -and $bundle) {
        try {
            $stopArgs = @{}
            if ($explicitDataDir -or -not $CleanProfile) {
                $stopArgs.DataDir = $dataRoot
            }
            & (Join-Path $bundle "Stop-Pi-zaya.ps1") @stopArgs
        }
        catch {
            $stopError = $_
        }
    }

    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], "Process")
    }

    if ($ownsDataDir -and -not $KeepData -and -not $CleanProfile -and $dataRoot -and (Test-Path -LiteralPath $dataRoot)) {
        Assert-GeneratedTempPath $dataRoot "Smoke data cleanup directory"
        Remove-Item -LiteralPath $dataRoot -Recurse -Force
    }
    if ($cleanProfileRoot -and (-not $KeepData -or $explicitDataDir) -and (Test-Path -LiteralPath $cleanProfileRoot)) {
        Assert-GeneratedTempPath $cleanProfileRoot "Clean profile cleanup directory"
        Remove-Item -LiteralPath $cleanProfileRoot -Recurse -Force
    }
    if ($archiveExtractRoot -and (Test-Path -LiteralPath $archiveExtractRoot)) {
        Assert-GeneratedTempPath $archiveExtractRoot "Archive extraction cleanup directory"
        Remove-Item -LiteralPath $archiveExtractRoot -Recurse -Force
    }
    if ($stopError) {
        throw "Packaged stop command failed during cleanup: $($stopError.Exception.Message)"
    }
}
