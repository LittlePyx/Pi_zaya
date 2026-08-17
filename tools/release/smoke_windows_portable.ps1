[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BundleRoot,
    [string]$DataDir = "",
    [switch]$KeepData,
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"
$bundle = [IO.Path]::GetFullPath($BundleRoot)
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

$ownsDataDir = -not [bool]$DataDir
if ($ownsDataDir) {
    $DataDir = Join-Path ([IO.Path]::GetTempPath()) ("pi-zaya-release-smoke-" + [guid]::NewGuid().ToString("N"))
}
$dataRoot = [IO.Path]::GetFullPath($DataDir)
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if ($ownsDataDir -and -not $dataRoot.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Generated smoke data directory escaped the system temp directory: $dataRoot"
}

$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
$listener.Start()
$port = ([Net.IPEndPoint]$listener.LocalEndpoint).Port
$listener.Stop()

try {
    & (Join-Path $bundle "Start-Pi-zaya.ps1") -DataDir $dataRoot -Port $port -NoBrowser

    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/health" -TimeoutSec 10
    $build = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/app/version" -TimeoutSec 10
    $frontPage = Invoke-WebRequest -Uri "http://127.0.0.1:$port/" -TimeoutSec 10 -UseBasicParsing
    $expectedVersion = (Get-Content -LiteralPath (Join-Path $bundle "VERSION") -Raw).Trim()
    if ($health.status -ne "ok") { throw "Health endpoint did not return ok." }
    if ($health.version -ne $expectedVersion) { throw "Health version '$($health.version)' did not match '$expectedVersion'." }
    if ($build.version -ne $expectedVersion) { throw "Build version '$($build.version)' did not match '$expectedVersion'." }
    if ($frontPage.StatusCode -ne 200) { throw "Frontend root returned HTTP $($frontPage.StatusCode)." }
    if (-not (Test-Path -LiteralPath (Join-Path $dataRoot "runtime\server-process.json"))) {
        throw "Launcher did not write its process record to the user data directory."
    }
    Write-Host "Portable smoke passed for Pi-zaya $expectedVersion on port $port."
}
finally {
    & (Join-Path $bundle "Stop-Pi-zaya.ps1") -DataDir $dataRoot
    if ($ownsDataDir -and -not $KeepData -and (Test-Path -LiteralPath $dataRoot)) {
        Remove-Item -LiteralPath $dataRoot -Recurse -Force
    }
}
