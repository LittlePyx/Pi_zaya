[CmdletBinding()]
param(
  [switch]$All,
  [switch]$ClearRuntimeTemp,
  [switch]$ClearLogs,
  [switch]$DeepLogScan,
  [switch]$ClearPyCaches,
  [switch]$DeepPyCacheScan,
  [switch]$ClearFrontendBuild,
  [switch]$ClearChatDb,
  [switch]$ClearLibraryDb,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $repoRoot

$explicitFlags = @(
  $All,
  $ClearRuntimeTemp,
  $ClearLogs,
  $ClearPyCaches,
  $ClearFrontendBuild,
  $ClearChatDb,
  $ClearLibraryDb
)
$hasExplicit = $false
foreach ($f in $explicitFlags) {
  if ($f) {
    $hasExplicit = $true
    break
  }
}

# Safe defaults when no explicit scope is given.
if (-not $hasExplicit) {
  $ClearRuntimeTemp = $true
  $ClearLogs = $true
  $ClearPyCaches = $true
}

if ($All) {
  $ClearRuntimeTemp = $true
  $ClearLogs = $true
  $ClearPyCaches = $true
  $ClearFrontendBuild = $true
}

function Remove-PathSafe([string]$Path) {
  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $false
  }

  if (-not (Test-Path $Path)) {
    return $false
  }

  if ($DryRun) {
    Write-Host "[dry-run] remove $Path"
    return $true
  }

  Remove-Item -Path $Path -Recurse -Force -ErrorAction SilentlyContinue
  if (Test-Path $Path) {
    Write-Warning "Failed to fully remove: $Path"
    return $false
  }
  Write-Host "[removed] $Path"
  return $true
}

$removedCount = 0

if ($ClearRuntimeTemp) {
  Write-Host '[reset] ClearRuntimeTemp enabled'
  $paths = @(
    '.tmp_fastapi_stdout.log',
    '.tmp_fastapi_stderr.log',
    '.tmp_vite_stdout.log',
    '.tmp_vite_stderr.log',
    '.tmp_streamlit_stdout.log',
    '.tmp_streamlit_stderr.log',
    '.logs',
    'tmp'
  )

  foreach ($p in $paths) {
    if (Remove-PathSafe $p) { $removedCount += 1 }
  }

  $tmpFiles = Get-ChildItem -Path . -Filter '.tmp_*' -File -ErrorAction SilentlyContinue
  foreach ($f in $tmpFiles) {
    if (Remove-PathSafe $f.FullName) { $removedCount += 1 }
  }
}

if ($ClearLogs) {
  Write-Host '[reset] ClearLogs enabled'
  if ($DeepLogScan) {
    $logFiles = Get-ChildItem -Path . -Recurse -File -Include '*.log' -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -notmatch '\\.venv|\\node_modules\\|\\.git\\' }
  } else {
    # Fast path: only scan common runtime/log folders, avoid full-tree stalls.
    $logFiles = @()
    $logFiles += Get-ChildItem -Path . -File -Filter '*.log' -ErrorAction SilentlyContinue
    foreach ($dir in @('.logs', 'tmp', 'test_results')) {
      if (Test-Path $dir) {
        $logFiles += Get-ChildItem -Path $dir -Recurse -File -Filter '*.log' -ErrorAction SilentlyContinue
      }
    }
  }
  foreach ($f in $logFiles) {
    if (Remove-PathSafe $f.FullName) { $removedCount += 1 }
  }
}

if ($ClearPyCaches) {
  Write-Host '[reset] ClearPyCaches enabled'
  if (Remove-PathSafe '.pytest_cache') { $removedCount += 1 }
  if ($DeepPyCacheScan) {
    $pyCacheDirs = Get-ChildItem -Path . -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -notmatch '\\.venv|\\.git\\' }
  } else {
    # Fast path: scan only source/test folders.
    $pyCacheDirs = New-Object System.Collections.Generic.List[System.IO.DirectoryInfo]
    if (Test-Path '__pycache__') {
      $pyCacheDirs.Add((Get-Item '__pycache__'))
    }
    foreach ($d in @('api', 'kb', 'tests', 'ui')) {
      if (Test-Path $d) {
        $dirs = Get-ChildItem -Path $d -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue
        foreach ($x in $dirs) {
          $pyCacheDirs.Add($x)
        }
      }
    }
  }
  foreach ($d in $pyCacheDirs) {
    if (Remove-PathSafe $d.FullName) { $removedCount += 1 }
  }
}

if ($ClearFrontendBuild) {
  Write-Host '[reset] ClearFrontendBuild enabled'
  if (Remove-PathSafe 'web/dist') { $removedCount += 1 }
}

$chatDbPath = [Environment]::GetEnvironmentVariable('KB_CHAT_DB')
if ([string]::IsNullOrWhiteSpace($chatDbPath)) {
  $chatDbPath = Join-Path $repoRoot 'chat.sqlite3'
}
$chatCandidates = @($chatDbPath, (Join-Path $repoRoot 'chat.db'))

if ($ClearChatDb) {
  Write-Host '[reset] ClearChatDb enabled'
  foreach ($db in $chatCandidates) {
    foreach ($suffix in @('', '-shm', '-wal')) {
      $candidate = "$db$suffix"
      if (Remove-PathSafe $candidate) { $removedCount += 1 }
    }
  }
}

$libraryDbPath = [Environment]::GetEnvironmentVariable('KB_LIBRARY_DB')
if ([string]::IsNullOrWhiteSpace($libraryDbPath)) {
  $libraryDbPath = Join-Path $repoRoot 'library.sqlite3'
}
$libraryCandidates = @($libraryDbPath, (Join-Path $repoRoot 'library.db'))

if ($ClearLibraryDb) {
  Write-Host '[reset] ClearLibraryDb enabled'
  foreach ($db in $libraryCandidates) {
    foreach ($suffix in @('', '-shm', '-wal')) {
      $candidate = "$db$suffix"
      if (Remove-PathSafe $candidate) { $removedCount += 1 }
    }
  }
}

Write-Host "[reset] done. Removed targets: $removedCount"
if (-not $hasExplicit) {
  Write-Host '[reset] Default scope used: RuntimeTemp + Logs + PyCaches'
}
if ($DryRun) {
  Write-Host '[reset] Dry-run mode: no files were deleted.'
}
