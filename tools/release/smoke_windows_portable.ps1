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

    foreach ($required in @("VERSION", "LICENSE", "release-manifest.json", "Start-Pi-zaya.cmd", "Stop-Pi-zaya.cmd", "Start-Pi-zaya.ps1", "Stop-Pi-zaya.ps1", "web\dist\index.html")) {
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

    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = ([Net.IPEndPoint]$listener.LocalEndpoint).Port
    $listener.Stop()

    $startArgs = @{ Port = $port; NoBrowser = $true }
    if ($explicitDataDir -or -not $CleanProfile) {
        $startArgs.DataDir = $dataRoot
    }
    $launchAttempted = $true
    & (Join-Path $bundle "Start-Pi-zaya.ps1") @startArgs

    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/health" -TimeoutSec 10
    $build = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/app/version" -TimeoutSec 10
    $settings = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/settings" -TimeoutSec 10
    $frontPage = Invoke-WebRequest -Uri "http://127.0.0.1:$port/" -TimeoutSec 10 -UseBasicParsing
    $expectedVersion = (Get-Content -LiteralPath (Join-Path $bundle "VERSION") -Raw).Trim()
    if ($health.status -ne "ok") { throw "Health endpoint did not return ok." }
    if ($health.version -ne $expectedVersion) { throw "Health version '$($health.version)' did not match '$expectedVersion'." }
    if ($build.version -ne $expectedVersion) { throw "Build version '$($build.version)' did not match '$expectedVersion'." }
    if ($frontPage.StatusCode -ne 200) { throw "Frontend root returned HTTP $($frontPage.StatusCode)." }
    if (-not (Test-Path -LiteralPath (Join-Path $dataRoot "runtime\server-process.json"))) {
        throw "Launcher did not write its process record to the user data directory."
    }
    if ($CleanProfile) {
        $expectedDataRoot = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "Pi_zaya"))
        if (-not $dataRoot.Equals($expectedDataRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Clean-profile launch did not use the default LOCALAPPDATA data directory."
        }
        if ([bool]$settings.has_api_key) {
            throw "Clean-profile launch unexpectedly inherited a text-model API key."
        }
    }
    $mode = if ($CleanProfile) { "Clean-profile archive" } else { "Portable" }
    Write-Host "$mode smoke passed for Pi-zaya $expectedVersion on port $port."
}
finally {
    $stopError = $null
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
    if ($cleanProfileRoot -and -not $KeepData -and (Test-Path -LiteralPath $cleanProfileRoot)) {
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
