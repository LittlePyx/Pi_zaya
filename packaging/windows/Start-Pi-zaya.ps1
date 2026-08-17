[CmdletBinding()]
param(
    [string]$DataDir = "",
    [int]$Port = 0,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$version = (Get-Content -LiteralPath (Join-Path $appRoot "VERSION") -Raw).Trim()
$manifestPath = Join-Path $appRoot "release-manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

if (-not $DataDir) {
    if ($env:KB_APP_DATA_DIR) {
        $DataDir = $env:KB_APP_DATA_DIR
    }
    elseif ($env:LOCALAPPDATA) {
        $DataDir = Join-Path $env:LOCALAPPDATA "Pi_zaya"
    }
    else {
        $DataDir = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "Pi_zaya"
    }
}
$dataRoot = [IO.Path]::GetFullPath($DataDir)
$runtimeDir = Join-Path $dataRoot "runtime"
$logsDir = Join-Path $dataRoot "logs"
New-Item -ItemType Directory -Force -Path $dataRoot, $runtimeDir, $logsDir | Out-Null

$bundledPython = Join-Path $appRoot "runtime\python\python.exe"
if (Test-Path -LiteralPath $bundledPython) {
    $python = $bundledPython
}
elseif ($manifest.python_runtime -eq "system") {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "This development bundle requires Python $($manifest.python_version). Build an embedded-runtime bundle for end users."
    }
    $python = $pythonCommand.Source
}
else {
    throw "The bundled Python runtime is missing or incomplete. Re-download the release ZIP."
}

if ($Port -le 0) {
    $Port = if ($env:KB_SERVER_PORT) { [int]$env:KB_SERVER_PORT } else { 8000 }
}
if ($Port -lt 1 -or $Port -gt 65535) {
    throw "Port must be between 1 and 65535."
}

$env:KB_RELEASE_MODE = "1"
$env:KB_APP_DATA_DIR = $dataRoot
$env:KB_APP_ENV = "desktop"
$env:KB_APP_VERSION = $version
$env:KB_BUILD_COMMIT = [string]$manifest.commit
$env:KB_BUILD_TIME = [string]$manifest.built_at
$env:KB_SERVER_HOST = "127.0.0.1"
$env:KB_SERVER_PORT = [string]$Port
$env:KB_SERVER_RELOAD = "0"
$env:KB_STARTUP_PREFLIGHT = "1"
$env:KB_STARTUP_STRICT = "0"
$env:KB_UPDATE_CHECK_ENABLED = "1"

$processInfoPath = Join-Path $runtimeDir "server-process.json"
if (Test-Path -LiteralPath $processInfoPath) {
    try {
        $existing = Get-Content -LiteralPath $processInfoPath -Raw | ConvertFrom-Json
        $existingProcess = Get-Process -Id ([int]$existing.pid) -ErrorAction SilentlyContinue
        if ($existingProcess) {
            $existingUrl = "http://127.0.0.1:$([int]$existing.port)/"
            $recordedExecutable = [IO.Path]::GetFullPath([string]$existing.executable)
            $actualExecutable = [IO.Path]::GetFullPath([string]$existingProcess.Path)
            $existingHealth = Invoke-RestMethod -Uri "${existingUrl}api/health" -TimeoutSec 2
            if (
                $actualExecutable.Equals($recordedExecutable, [StringComparison]::OrdinalIgnoreCase) -and
                $existingHealth.status -eq "ok" -and
                $existingHealth.version -eq [string]$existing.version
            ) {
                if (-not $NoBrowser) {
                    Start-Process $existingUrl
                }
                Write-Host "Pi-zaya is already running at $existingUrl"
                exit 0
            }
        }
    }
    catch {
        # A stale or partial process record is replaced below.
    }
    Remove-Item -LiteralPath $processInfoPath -Force -ErrorAction SilentlyContinue
}

$stdoutPath = Join-Path $logsDir "server-stdout.log"
$stderrPath = Join-Path $logsDir "server-stderr.log"
$server = Start-Process `
    -FilePath $python `
    -ArgumentList @("server.py") `
    -WorkingDirectory $appRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$processInfo = [ordered]@{
    pid = $server.Id
    executable = [IO.Path]::GetFullPath($python)
    port = $Port
    started_at = [DateTimeOffset]::UtcNow.ToString("o")
    version = $version
}
$processInfo | ConvertTo-Json | Set-Content -LiteralPath $processInfoPath -Encoding UTF8

$healthUrl = "http://127.0.0.1:$Port/api/health"
$appUrl = "http://127.0.0.1:$Port/"
$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if ($server.HasExited) {
        break
    }
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        $server.Refresh()
        if (-not $server.HasExited -and $health.status -eq "ok" -and $health.version -eq $version) {
            $ready = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}

if (-not $ready) {
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $processInfoPath -Force -ErrorAction SilentlyContinue
    $tail = if (Test-Path -LiteralPath $stderrPath) { (Get-Content -LiteralPath $stderrPath -Tail 20) -join [Environment]::NewLine } else { "No error log was written." }
    throw "Pi-zaya did not become ready. See $stderrPath`n$tail"
}

if (-not $NoBrowser) {
    Start-Process $appUrl
}
Write-Host "Pi-zaya $version is running at $appUrl"
Write-Host "User data: $dataRoot"
Write-Host "Run Stop-Pi-zaya.ps1 to stop the local server."
