[CmdletBinding()]
param(
    [string]$StageRoot = "",
    [string]$OutputDir = "",
    [string]$InnoSetupCompiler = "",
    [string]$SigningThumbprint = $env:PI_ZAYA_SIGNING_THUMBPRINT,
    [string]$SignToolPath = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [switch]$RequireSignature,
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
. (Join-Path $PSScriptRoot "Authenticode.ps1")

$version = (Get-Content -LiteralPath (Join-Path $repoRoot "VERSION") -Raw).Trim()
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "release"
}
$outputRoot = [IO.Path]::GetFullPath($OutputDir)
if (-not $StageRoot) {
    $StageRoot = Join-Path $outputRoot "Pi_zaya-v$version-windows-x64"
}
$stage = [IO.Path]::GetFullPath($StageRoot)
if (-not (Test-Path -LiteralPath $stage -PathType Container)) {
    throw "Portable staging directory was not found: $stage"
}

$requiredFiles = @("VERSION", "LICENSE", "release-manifest.json", "README-中文.md", "Pi_zaya.exe", "Pi_zaya.ico", "runtime\python\python.exe", "web\dist\index.html")
foreach ($required in $requiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $stage $required) -PathType Leaf)) {
        throw "Installer staging directory is missing $required"
    }
}
$stagedVersion = (Get-Content -LiteralPath (Join-Path $stage "VERSION") -Raw).Trim()
if ($stagedVersion -ne $version) {
    throw "Staged version '$stagedVersion' does not match repository version '$version'."
}
$portableManifest = Get-Content -LiteralPath (Join-Path $stage "release-manifest.json") -Raw | ConvertFrom-Json
if ($portableManifest.package_type -ne "portable_zip" -or $portableManifest.python_runtime -ne "embedded") {
    throw "The installer must be built from the embedded-runtime portable stage."
}
if ($portableManifest.license -ne "MIT" -or $portableManifest.license_status -ne "included") {
    throw "The installer stage does not include the approved MIT license."
}
if ([bool]$portableManifest.source_dirty -and -not $AllowDirty) {
    throw "The installer stage was built from a dirty source tree. Use -AllowDirty only for a non-distributable acceptance build."
}

if (-not $InnoSetupCompiler) {
    $compilerCommand = Get-Command ISCC.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($compilerCommand) {
        $InnoSetupCompiler = $compilerCommand.Source
    }
    else {
        $knownPaths = @(
            (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 7\ISCC.exe"),
            (Join-Path $env:ProgramFiles "Inno Setup 7\ISCC.exe"),
            (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
        ) | Where-Object { $_ }
        $InnoSetupCompiler = $knownPaths | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    }
}
if (-not $InnoSetupCompiler) {
    throw "ISCC.exe was not found. Install Inno Setup 7 or pass -InnoSetupCompiler."
}
$compiler = [IO.Path]::GetFullPath($InnoSetupCompiler)
if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) {
    throw "ISCC.exe was not found: $compiler"
}

$signingEnabled = -not [string]::IsNullOrWhiteSpace($SigningThumbprint)
if ($RequireSignature -and -not $signingEnabled) {
    throw "-RequireSignature needs a trusted code-signing certificate thumbprint."
}
$launcherState = Get-PiZayaAuthenticodeState -Path (Join-Path $stage "Pi_zaya.exe")
$signTool = ""
$certificate = $null
if ($signingEnabled) {
    $certificate = Get-PiZayaSigningCertificate -Thumbprint $SigningThumbprint
    $signTool = Find-PiZayaSignTool -ExplicitPath $SignToolPath
    if (-not $launcherState.Signed -or $launcherState.Thumbprint -ne $certificate.Thumbprint -or -not $launcherState.Timestamped) {
        throw "The staged Pi_zaya.exe is not signed and timestamped with the selected certificate. Rebuild the portable stage with the same signing options."
    }
}
elseif ($launcherState.Signed) {
    throw "The staged launcher is signed, but no SigningThumbprint was supplied to verify and sign the installer consistently."
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$baseName = "Pi_zaya-v$version-windows-x64-setup"
$installerPath = Join-Path $outputRoot "$baseName.exe"
foreach ($oldPath in @($installerPath, "$installerPath.sha256", (Join-Path $outputRoot "$baseName.manifest.json"))) {
    if (Test-Path -LiteralPath $oldPath) {
        Remove-Item -LiteralPath $oldPath -Force
    }
}

$coreParts = @(($version -split '[-+]')[0] -split '\.')
$revision = 0
if ($version -match '-[^+]*?(?<revision>\d+)(?:\+.*)?$') {
    $revision = [int]$Matches.revision
}
$numericVersion = "$($coreParts[0]).$($coreParts[1]).$($coreParts[2]).$revision"
$environmentNames = @("PI_ZAYA_INSTALLER_VERSION", "PI_ZAYA_INSTALLER_NUMERIC_VERSION", "PI_ZAYA_INSTALLER_STAGE", "PI_ZAYA_INSTALLER_OUTPUT", "PI_ZAYA_INSTALLER_BASENAME", "PI_ZAYA_INSTALLER_SIGNING")
$savedEnvironment = @{}
foreach ($name in $environmentNames) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
try {
    $env:PI_ZAYA_INSTALLER_VERSION = $version
    $env:PI_ZAYA_INSTALLER_NUMERIC_VERSION = $numericVersion
    $env:PI_ZAYA_INSTALLER_STAGE = $stage
    $env:PI_ZAYA_INSTALLER_OUTPUT = $outputRoot
    $env:PI_ZAYA_INSTALLER_BASENAME = $baseName
    $env:PI_ZAYA_INSTALLER_SIGNING = if ($signingEnabled) { "1" } else { "0" }

    $compilerArgs = @("--quiet-progress", "--no-ide-signtools")
    if ($signingEnabled) {
        $signCommand = '"' + $signTool + '" sign /sha1 ' + $certificate.Thumbprint + ' /fd SHA256 /td SHA256 /tr "' + $TimestampUrl + '" /d "Pi_zaya" $f'
        $compilerArgs += "--signtool=PiZayaAuthenticode=$signCommand"
    }
    $compilerArgs += (Join-Path $repoRoot "packaging\windows\Pi_zaya.iss")
    & $compiler @compilerArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $installerPath -PathType Leaf)) {
        throw "Inno Setup failed to build the Pi_zaya installer."
    }
}
finally {
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], "Process")
    }
}

$installerState = Get-PiZayaAuthenticodeState -Path $installerPath
if ($signingEnabled) {
    if (-not $installerState.Signed -or $installerState.Thumbprint -ne $certificate.Thumbprint -or -not $installerState.Timestamped) {
        throw "The compiled installer is not signed and timestamped with the selected certificate."
    }
}
elseif ($installerState.Signed) {
    throw "The unsigned build unexpectedly produced a signed installer."
}

$hash = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([IO.Path]::GetFileName($installerPath))" | Set-Content -LiteralPath "$installerPath.sha256" -Encoding ASCII
$builtAt = [DateTimeOffset]::UtcNow.ToString("o")
$manifest = [ordered]@{
    schema_version = 1
    product = "Pi_zaya"
    version = $version
    tag = "v$version"
    artifact = [IO.Path]::GetFileName($installerPath)
    package_type = "windows_installer"
    platform = "windows"
    architecture = "x64"
    install_scope = "current_user"
    default_install_dir = '%LOCALAPPDATA%\Programs\Pi_zaya'
    user_data_default = '%LOCALAPPDATA%\Pi_zaya'
    uninstall_preserves_user_data = $true
    size_bytes = (Get-Item -LiteralPath $installerPath).Length
    sha256 = $hash
    built_at = $builtAt
    commit = [string]$portableManifest.commit
    source_dirty = [bool]$portableManifest.source_dirty
    license = "MIT"
    signed = [bool]$installerState.Signed
    signature_status = [string]$installerState.Status
    certificate_thumbprint = [string]$installerState.Thumbprint
    certificate_subject = [string]$installerState.Subject
    timestamped = [bool]$installerState.Timestamped
    launcher_signed = [bool]$launcherState.Signed
    launcher_signature_status = [string]$launcherState.Status
    launcher_certificate_thumbprint = [string]$launcherState.Thumbprint
    launcher_timestamped = [bool]$launcherState.Timestamped
    signed_uninstaller = [bool]$signingEnabled
}
$manifestPath = Join-Path $outputRoot "$baseName.manifest.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "Built $installerPath"
Write-Host "SHA256 $hash"
Write-Host "Authenticode $($installerState.Status)"
