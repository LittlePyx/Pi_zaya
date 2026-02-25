$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$legacyScript = Join-Path $here "run.ps1"

if (!(Test-Path $legacyScript)) {
  throw "run.ps1 not found: $legacyScript"
}

Write-Host "[kb_chat] Launching legacy Streamlit entry (app.py) via run.ps1" -ForegroundColor Cyan

# Forward any extra args to the existing launcher.
& $legacyScript @args

