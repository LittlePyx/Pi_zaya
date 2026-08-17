[CmdletBinding()]
param(
    [string]$DataDir = ""
)

$ErrorActionPreference = "Stop"
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
$processInfoPath = Join-Path $dataRoot "runtime\server-process.json"
if (-not (Test-Path -LiteralPath $processInfoPath)) {
    Write-Host "Pi-zaya is not running (no process record found)."
    exit 0
}

$info = Get-Content -LiteralPath $processInfoPath -Raw | ConvertFrom-Json
$process = Get-Process -Id ([int]$info.pid) -ErrorAction SilentlyContinue
if (-not $process) {
    Remove-Item -LiteralPath $processInfoPath -Force
    Write-Host "Removed a stale Pi-zaya process record."
    exit 0
}

$recordedExecutable = [IO.Path]::GetFullPath([string]$info.executable)
$actualExecutable = ""
try {
    $actualExecutable = [IO.Path]::GetFullPath([string]$process.Path)
}
catch {
    throw "Could not verify process $($info.pid); it was not stopped."
}
if (-not $actualExecutable.Equals($recordedExecutable, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Process $($info.pid) no longer matches Pi-zaya's recorded executable; it was not stopped."
}

Stop-Process -Id $process.Id
[void]$process.WaitForExit(10000)
Remove-Item -LiteralPath $processInfoPath -Force -ErrorAction SilentlyContinue
Write-Host "Pi-zaya has stopped."
