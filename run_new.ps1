[CmdletBinding()]
param(
  [switch]$StopExisting,
  [switch]$InstallBackendDeps,
  [switch]$InstallFrontendDeps,
  [switch]$NoBackendReload,
  [switch]$AllowAuthGate,
  [string]$BackendHost = "127.0.0.1",
  [int]$BackendPort = 8000,
  [string]$FrontendHost = "127.0.0.1",
  [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host "[kb_chat:new-ui] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[kb_chat:new-ui] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[kb_chat:new-ui] $msg" -ForegroundColor Red }

function Test-EnvTruthy([string]$Name) {
  $value = [Environment]::GetEnvironmentVariable($Name, "Process")
  if ([string]::IsNullOrWhiteSpace($value)) { return $false }
  return @("1", "true", "yes", "on") -contains $value.Trim().ToLowerInvariant()
}

function Test-PortListening([int]$Port) {
  try {
    $conn = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop | Select-Object -First 1
    return $null -ne $conn
  } catch {
    return $false
  }
}

function Wait-PortListening([int]$Port, [int]$TimeoutSeconds = 20, [int]$IntervalMs = 500) {
  $deadline = (Get-Date).AddSeconds([Math]::Max(1, $TimeoutSeconds))
  while ((Get-Date) -lt $deadline) {
    if (Test-PortListening -Port $Port) {
      return $true
    }
    Start-Sleep -Milliseconds ([Math]::Max(100, $IntervalMs))
  }
  return (Test-PortListening -Port $Port)
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

function Get-ProjectDevPids([string]$Root, [int[]]$Ports) {
  $ids = @(Get-PortPids -Ports $Ports)
  try {
    $procs = @(Get-CimInstance Win32_Process | Where-Object {
      $_.CommandLine -and
      $_.CommandLine.Contains($Root) -and
      (
        ($_.Name -match 'python' -and $_.CommandLine -match 'uvicorn|multiprocessing\.spawn') -or
        ($_.Name -match 'node|cmd' -and $_.CommandLine -match 'vite|npm.*run dev')
      )
    })
    $ids += @($procs | Select-Object -ExpandProperty ProcessId)
    $ids += @(
      $procs |
        Where-Object { $_.CommandLine -match 'multiprocessing\.spawn|vite|npm.*run dev' } |
        Select-Object -ExpandProperty ParentProcessId
    )
  } catch {
    # Fall back to port-based stopping when process metadata is unavailable.
  }
  return @($ids | Where-Object { $_ -and $_ -ne $PID } | Sort-Object -Unique)
}

function Tail-IfExists([string]$Path, [int]$Lines = 60) {
  if (Test-Path $Path) {
    Write-Host "---- $Path ----" -ForegroundColor DarkGray
    Get-Content $Path -Tail $Lines
  }
}

function Test-BackendDepsReady([string]$PythonExe) {
  $code = @'
import importlib.util
required = ('fastapi', 'uvicorn')
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
'@
  & $PythonExe -c $code *> $null
  return ($LASTEXITCODE -eq 0)
}

function Test-FrontendDepsReady([string]$WebDir) {
  $viteCmd = Join-Path $WebDir "node_modules\.bin\vite.cmd"
  $viteBin = Join-Path $WebDir "node_modules\vite"
  return (Test-Path $viteCmd) -or (Test-Path $viteBin)
}

function Install-FrontendDeps([string]$WebDir, [string]$NpmExe) {
  Push-Location $WebDir
  try {
    if (Test-Path (Join-Path $WebDir "package-lock.json")) {
      & $NpmExe ci | Out-Host
    } else {
      & $NpmExe install | Out-Host
    }
  } finally {
    Pop-Location
  }
}

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$webDir = Join-Path $here "web"
if (!(Test-Path $webDir)) {
  throw "web/ not found under $here"
}
Set-Location $here

$envPath = Join-Path $here ".env"
if (Test-Path $envPath) {
  Write-Info "Loading environment from .env ..."
  Get-Content $envPath | ForEach-Object {
    $line = $_.Trim()
    if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#") -or ($line -notmatch "=")) {
      return
    }
    $parts = $line.Split("=", 2)
    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    if (-not [string]::IsNullOrWhiteSpace($name)) {
      [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
  }
}

if (-not $AllowAuthGate) {
  if ((Test-EnvTruthy -Name "KB_REQUIRE_AUTH") -or (Test-EnvTruthy -Name "KB_ENABLE_AUTH_GATE") -or (Test-EnvTruthy -Name "KB_PRIVATE_INSTANCE_AUTH")) {
    Write-Warn "Access-token gate settings detected. The local user app stays public by default, so this launcher is disabling the gate for this process. Use -AllowAuthGate only when testing a private access-token gate."
  }
  [Environment]::SetEnvironmentVariable("KB_REQUIRE_AUTH", "0", "Process")
  [Environment]::SetEnvironmentVariable("KB_ENABLE_AUTH_GATE", "0", "Process")
  [Environment]::SetEnvironmentVariable("KB_PRIVATE_INSTANCE_AUTH", "0", "Process")
  [Environment]::SetEnvironmentVariable("KB_ALLOW_LOCAL_AUTH_GATE", "0", "Process")
  [Environment]::SetEnvironmentVariable("VITE_ENABLE_AUTH_GATE", "0", "Process")
  [Environment]::SetEnvironmentVariable("VITE_PRIVATE_INSTANCE_AUTH", "0", "Process")
  [Environment]::SetEnvironmentVariable("VITE_ALLOW_LOCAL_AUTH_GATE", "0", "Process")
} else {
  [Environment]::SetEnvironmentVariable("KB_ENABLE_AUTH_GATE", "1", "Process")
  [Environment]::SetEnvironmentVariable("KB_PRIVATE_INSTANCE_AUTH", "1", "Process")
  [Environment]::SetEnvironmentVariable("KB_ALLOW_LOCAL_AUTH_GATE", "1", "Process")
  [Environment]::SetEnvironmentVariable("VITE_ENABLE_AUTH_GATE", "1", "Process")
  [Environment]::SetEnvironmentVariable("VITE_PRIVATE_INSTANCE_AUTH", "1", "Process")
  [Environment]::SetEnvironmentVariable("VITE_ALLOW_LOCAL_AUTH_GATE", "1", "Process")
}
if ((Test-EnvTruthy -Name "KB_PRIVATE_INSTANCE_AUTH") -and (Test-EnvTruthy -Name "KB_ENABLE_AUTH_GATE") -and (Test-EnvTruthy -Name "KB_REQUIRE_AUTH")) {
  Write-Warn "Access-token gate: ON for this local run."
} else {
  Write-Info "Access-token gate: OFF; users do not need a token."
}

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
  for ($i = 0; $i -lt 3; $i++) {
    $pids = Get-ProjectDevPids -Root $here -Ports $targetPorts
    if (@($pids).Count -eq 0) {
      break
    }
    Write-Info "Stopping existing project dev processes on ports $($targetPorts -join ', '): $($pids -join ', ')"
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
  Install-FrontendDeps -WebDir $webDir -NpmExe $npmExe
}

if (-not (Test-BackendDepsReady -PythonExe $pythonExe)) {
  throw "Backend dependencies are missing. Run .\run_new.ps1 -InstallBackendDeps -InstallFrontendDeps -StopExisting, or run: pip install -r requirements.txt"
}

if (-not (Test-FrontendDepsReady -WebDir $webDir)) {
  throw "Frontend dependencies are missing. Run .\run_new.ps1 -InstallFrontendDeps -StopExisting, or run: cd web; npm ci"
}

$fastapiOut = Join-Path $here ".tmp_fastapi_stdout.log"
$fastapiErr = Join-Path $here ".tmp_fastapi_stderr.log"
$viteOut = Join-Path $here ".tmp_vite_stdout.log"
$viteErr = Join-Path $here ".tmp_vite_stderr.log"
Remove-Item $fastapiOut, $fastapiErr, $viteOut, $viteErr -ErrorAction SilentlyContinue

$backendPrePids = Get-PortPids -Ports @($BackendPort)
$frontendPrePids = Get-PortPids -Ports @($FrontendPort)

Write-Info "Starting backend (uvicorn) on http://$BackendHost`:$BackendPort ..."
$backendArgs = @('-m', 'uvicorn', 'api.main:app', '--host', $BackendHost, '--port', "$BackendPort")
if (-not $NoBackendReload) {
  # Avoid reload loops caused by benchmark outputs / logs written under repo root.
  $backendArgs += @(
    '--reload',
    '--reload-dir', (Join-Path $here 'api'),
    '--reload-dir', (Join-Path $here 'kb')
  )
}
$backendProc = Start-Process `
  -FilePath $pythonExe `
  -ArgumentList $backendArgs `
  -WorkingDirectory $here `
  -WindowStyle Hidden `
  -PassThru `
  -RedirectStandardOutput $fastapiOut `
  -RedirectStandardError $fastapiErr

Write-Info "Starting frontend (vite) on http://$FrontendHost`:$FrontendPort ..."
# Keep the browser runtime on same-origin /api while telling Vite's dev proxy
# where the backend lives. This avoids stale VITE_BACKEND_URL values from .env
# making local clone users call an old or cross-origin backend.
if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("VITE_BACKEND_URL", "Process"))) {
  Write-Warn "VITE_BACKEND_URL detected. Local dev mode clears it so the browser uses same-origin /api; Vite proxy will target the backend below."
}
[Environment]::SetEnvironmentVariable("VITE_BACKEND_URL", "", "Process")
[Environment]::SetEnvironmentVariable("VITE_BACKEND_PROXY_TARGET", "http://$BackendHost`:$BackendPort", "Process")
$frontendProc = Start-Process `
  -FilePath $npmExe `
  -ArgumentList @('run', 'dev', '--', '--host', $FrontendHost, '--port', "$FrontendPort") `
  -WorkingDirectory $webDir `
  -WindowStyle Hidden `
  -PassThru `
  -RedirectStandardOutput $viteOut `
  -RedirectStandardError $viteErr

$backendListening = Wait-PortListening -Port $BackendPort -TimeoutSeconds 25 -IntervalMs 500
$frontendListening = Wait-PortListening -Port $FrontendPort -TimeoutSeconds 25 -IntervalMs 500
$backendPostPids = Get-PortPids -Ports @($BackendPort)
$frontendPostPids = Get-PortPids -Ports @($FrontendPort)
$backendNewPids = @($backendPostPids | Where-Object { $backendPrePids -notcontains $_ })
$frontendNewPids = @($frontendPostPids | Where-Object { $frontendPrePids -notcontains $_ })
$backendOk = $backendListening
$frontendOk = $frontendListening

Write-Host ""
Write-Info "Backend PID:  $($backendProc.Id)  (port ${BackendPort}: $(if ($backendOk) { 'UP' } else { 'DOWN' }))"
Write-Info "Frontend PID: $($frontendProc.Id)  (port ${FrontendPort}: $(if ($frontendOk) { 'UP' } else { 'DOWN' }))"
Write-Info "Frontend URL: http://localhost:$FrontendPort"
Write-Info "Backend URL:  http://localhost:$BackendPort"
Write-Info "Backend reload: $(if ($NoBackendReload) { 'OFF' } else { 'ON' })"
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
