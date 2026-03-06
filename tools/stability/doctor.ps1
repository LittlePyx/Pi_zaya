[CmdletBinding()]
param(
  [switch]$Json,
  [switch]$Strict
)

$ErrorActionPreference = 'Stop'

function Read-FirstLine([string]$Path) {
  if (Test-Path $Path) {
    $line = Get-Content $Path -TotalCount 1
    if ($null -ne $line) {
      return "$line".Trim()
    }
  }
  return ''
}

function Get-ExeVersion([string]$ExePath, [string[]]$ArgList) {
  if ([string]::IsNullOrWhiteSpace($ExePath) -or -not (Test-Path $ExePath)) {
    return ''
  }
  try {
    $out = & $ExePath @ArgList 2>$null
    if ($LASTEXITCODE -ne 0) {
      return ''
    }
    return ("$out".Trim())
  } catch {
    return ''
  }
}

function Normalize-Version([string]$Text) {
  if ([string]::IsNullOrWhiteSpace($Text)) {
    return ''
  }
  return $Text.Trim().TrimStart('v')
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $repoRoot

$requiredPython = Read-FirstLine '.python-version'
$requiredNode = Read-FirstLine '.nvmrc'

$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
$pythonPath = ''
if (Test-Path $venvPython) {
  $pythonPath = $venvPython
} else {
  $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
  if ($null -ne $pythonCmd) {
    $pythonPath = $pythonCmd.Source
  }
}

$pythonVersion = ''
if (-not [string]::IsNullOrWhiteSpace($pythonPath)) {
  try {
    $pythonVersion = & $pythonPath -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
    $pythonVersion = Normalize-Version $pythonVersion
  } catch {
    $pythonVersion = ''
  }
}

$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
$nodePath = ''
if ($null -ne $nodeCmd) {
  $nodePath = $nodeCmd.Source
}
$nodeVersion = Normalize-Version (Get-ExeVersion $nodePath @('-v'))

$npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $npmCmd) {
  $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
}
$npmPath = ''
if ($null -ne $npmCmd) {
  $npmPath = $npmCmd.Source
}
$npmVersion = Normalize-Version (Get-ExeVersion $npmPath @('-v'))

$gitCmd = Get-Command git -ErrorAction SilentlyContinue
$gitInfo = [ordered]@{
  available = $false
  branch = ''
  commit = ''
  dirty = $false
}
if ($null -ne $gitCmd) {
  $gitInfo.available = $true
  $gitInfo.branch = (& git rev-parse --abbrev-ref HEAD 2>$null | Out-String).Trim()
  $gitInfo.commit = (& git rev-parse --short HEAD 2>$null | Out-String).Trim()
  $dirtyLines = & git status --porcelain 2>$null
  if ($null -ne $dirtyLines -and @($dirtyLines).Count -gt 0) {
    $gitInfo.dirty = $true
  }
}

function Hash-IfExists([string]$Path) {
  if (Test-Path $Path) {
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLower()
  }
  return ''
}

$hashes = [ordered]@{
  requirements_txt = Hash-IfExists 'requirements.txt'
  package_lock_json = Hash-IfExists 'web/package-lock.json'
}

$envKeys = @('KB_DB_DIR', 'KB_CHAT_DB', 'KB_LIBRARY_DB', 'KB_PDF_DIR', 'KB_MD_DIR')
$envMap = [ordered]@{}
foreach ($k in $envKeys) {
  $v = [Environment]::GetEnvironmentVariable($k)
  if ([string]::IsNullOrWhiteSpace($v)) {
    $envMap[$k] = '<unset>'
  } else {
    $envMap[$k] = $v
  }
}

$pythonMatch = $true
if (-not [string]::IsNullOrWhiteSpace($requiredPython)) {
  $pythonMatch = ($pythonVersion -eq $requiredPython)
}

$nodeMatch = $true
if (-not [string]::IsNullOrWhiteSpace($requiredNode)) {
  $nodeMatch = ($nodeVersion -eq $requiredNode)
}

$report = [ordered]@{
  timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz')
  repo_root = "$repoRoot"
  os = [Environment]::OSVersion.VersionString
  timezone = (Get-TimeZone).Id
  versions = [ordered]@{
    python = [ordered]@{
      required = $requiredPython
      actual = $pythonVersion
      exe = $pythonPath
      match = $pythonMatch
    }
    node = [ordered]@{
      required = $requiredNode
      actual = $nodeVersion
      exe = $nodePath
      match = $nodeMatch
    }
    npm = [ordered]@{
      actual = $npmVersion
      exe = $npmPath
    }
  }
  git = $gitInfo
  file_hashes = $hashes
  env = $envMap
}

if ($Json) {
  $report | ConvertTo-Json -Depth 8
} else {
  Write-Host "[doctor] Repo: $($report.repo_root)"
  Write-Host "[doctor] Time: $($report.timestamp) ($($report.timezone))"
  Write-Host "[doctor] Python required/actual: $requiredPython / $pythonVersion"
  Write-Host "[doctor] Node required/actual:   $requiredNode / $nodeVersion"
  Write-Host "[doctor] NPM actual:             $npmVersion"
  Write-Host "[doctor] Git branch/commit:      $($gitInfo.branch) / $($gitInfo.commit)"
  Write-Host "[doctor] Git dirty:              $($gitInfo.dirty)"
  Write-Host "[doctor] SHA256 requirements.txt: $($hashes.requirements_txt)"
  Write-Host "[doctor] SHA256 web/package-lock.json: $($hashes.package_lock_json)"
  Write-Host "[doctor] Env (KB_*):"
  foreach ($k in $envKeys) {
    Write-Host ("  - {0}={1}" -f $k, $envMap[$k])
  }

  if (-not $pythonMatch) {
    Write-Warning "Python version mismatch. Expected $requiredPython but got $pythonVersion"
  }
  if (-not $nodeMatch) {
    Write-Warning "Node version mismatch. Expected $requiredNode but got $nodeVersion"
  }
}

if ($Strict -and ((-not $pythonMatch) -or (-not $nodeMatch))) {
  exit 1
}
