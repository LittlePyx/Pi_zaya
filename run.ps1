$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$newUiScript = Join-Path $here "run_new.ps1"

if (!(Test-Path $newUiScript)) {
  throw "run_new.ps1 not found: $newUiScript"
}

Write-Host "[kb_chat] Launching FastAPI + React entry via run_new.ps1" -ForegroundColor Cyan

# Forward any extra args to the current launcher.
& $newUiScript @args
