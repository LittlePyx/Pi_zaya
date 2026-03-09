[CmdletBinding()]
param(
  [switch]$StopExisting,
  [switch]$InstallBackendDeps,
  [switch]$InstallFrontendDeps,
  [string]$BackendHost = "127.0.0.1",
  [int]$BackendPort = 8000,
  [string]$FrontendHost = "127.0.0.1",
  [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host "[kb_chat:new-ui] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[kb_chat:new-ui] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[kb_chat:new-ui] $msg" -ForegroundColor Red }

function Test-PortListening([int]$Port) {
  try {
    $conn = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop | Select-Object -First 1
    return $null -ne $conn
  } catch {
    return $false
  }
}

function Get-PortPids([int[]]$Ports) {
  $out = @()
  foreach ($p in $Ports) {
    try {
      $conns = Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction Stop
      foreach ($c in $conns) {
        if ($c.OwningProcess -and ($out -notcontains [int]$c.OwningProcess)) {
          $out += [int]$c.OwningProcess
        }
      }
    } catch {
      # ignore missing port
    }
  }
  return $out
}

function Tail-IfExists([string]$Path, [int]$Lines = 60) {
  if (Test-Path $Path) {
    Write-Host "---- $Path ----" -ForegroundColor DarkGray
    Get-Content $Path -Tail $Lines
  }
}

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$webDir = Join-Path $here "web"
if (!(Test-Path $webDir)) {
  throw "web/ not found under $here"
}
Set-Location $here

$venvPython = Join-Path $here ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
  $pythonExe = $venvPython
} else {
  $pyCmd = Get-Command python -ErrorAction SilentlyContinue
  if ($null -eq $pyCmd) { throw "python not found in PATH and .venv\\Scripts\\python.exe is missing." }
  $pythonExe = $pyCmd.Source
}

$npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $npmCmd) { $npmCmd = Get-Command npm -ErrorAction SilentlyContinue }
if ($null -eq $npmCmd) { throw "npm not found in PATH." }
$npmExe = $npmCmd.Source

$targetPorts = @($BackendPort, $FrontendPort)
if ($StopExisting) {
  $pids = Get-PortPids -Ports $targetPorts
  if (@($pids).Count -gt 0) {
    Write-Info "Stopping existing processes on ports $($targetPorts -join ', '): $($pids -join ', ')"
    Stop-Process -Id $pids -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
  }
} else {
  foreach ($p in $targetPorts) {
    if (Test-PortListening -Port $p) {
      throw "Port $p is already in use. Re-run with -StopExisting or stop the process manually."
    }
  }
}

if ($InstallBackendDeps) {
  Write-Info "Installing backend dependencies from requirements.txt ..."
  & $pythonExe -m pip install -r (Join-Path $here "requirements.txt") | Out-Host
}

if ($InstallFrontendDeps) {
  Write-Info "Installing frontend dependencies in web/ ..."
  Push-Location $webDir
  try {
    & $npmExe install | Out-Host
  } finally {
    Pop-Location
  }
}

$fastapiOut = Join-Path $here ".tmp_fastapi_stdout.log"
$fastapiErr = Join-Path $here ".tmp_fastapi_stderr.log"
$viteOut = Join-Path $here ".tmp_vite_stdout.log"
$viteErr = Join-Path $here ".tmp_vite_stderr.log"
Remove-Item $fastapiOut, $fastapiErr, $viteOut, $viteErr -ErrorAction SilentlyContinue

Write-Info "Starting backend (uvicorn) on http://$BackendHost`:$BackendPort ..."
$backendProc = Start-Process `
  -FilePath $pythonExe `
  -ArgumentList @('-m', 'uvicorn', 'api.main:app', '--host', $BackendHost, '--port', "$BackendPort", '--reload') `
  -WorkingDirectory $here `
  -PassThru `
  -RedirectStandardOutput $fastapiOut `
  -RedirectStandardError $fastapiErr

Write-Info "Starting frontend (vite) on http://$FrontendHost`:$FrontendPort ..."
$frontendProc = Start-Process `
  -FilePath $npmExe `
  -ArgumentList @('run', 'dev', '--', '--host', $FrontendHost, '--port', "$FrontendPort") `
  -WorkingDirectory $webDir `
  -PassThru `
  -RedirectStandardOutput $viteOut `
  -RedirectStandardError $viteErr

Start-Sleep -Seconds 6

$backendOk = Test-PortListening -Port $BackendPort
$frontendOk = Test-PortListening -Port $FrontendPort

Write-Host ""
Write-Info "Backend PID:  $($backendProc.Id)  (port ${BackendPort}: $(if ($backendOk) { 'UP' } else { 'DOWN' }))"
Write-Info "Frontend PID: $($frontendProc.Id)  (port ${FrontendPort}: $(if ($frontendOk) { 'UP' } else { 'DOWN' }))"
Write-Info "Frontend URL: http://localhost:$FrontendPort"
Write-Info "Backend URL:  http://localhost:$BackendPort"
Write-Info "Logs: $fastapiErr, $viteOut"

if (-not $backendOk -or -not $frontendOk) {
  Write-Err "One or more services did not start correctly. Recent logs:"
  Tail-IfExists -Path $fastapiErr
  Tail-IfExists -Path $fastapiOut
  Tail-IfExists -Path $viteErr
  Tail-IfExists -Path $viteOut
  exit 1
}

Write-Info "New UI dev mode started. Press Ctrl+C only affects this shell; use Stop-Process to stop the background PIDs."
Write-Info "Tip: run_new.ps1 -StopExisting to restart cleanly."
