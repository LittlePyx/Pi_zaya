$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$webDir = Join-Path $here "web"
if (!(Test-Path $webDir)) {
  throw "web/ not found under $here"
}

$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if ($null -eq $nodeCmd) {
  throw "node not found in PATH."
}
$nodeExe = $nodeCmd.Source

$viteOut = Join-Path $here ".tmp_vite_stdout.log"
$viteErr = Join-Path $here ".tmp_vite_stderr.log"
Remove-Item $viteOut, $viteErr -ErrorAction SilentlyContinue

$viteScript = Join-Path $webDir "node_modules\vite\bin\vite.js"
$full = 'start "" /b "' + $nodeExe + '" "' + $viteScript + '" --host 127.0.0.1 --port 5173 1>>"' + $viteOut + '" 2>>"' + $viteErr + '"'
Start-Process `
  -FilePath "cmd.exe" `
  -ArgumentList "/c", $full `
  -WorkingDirectory $webDir
