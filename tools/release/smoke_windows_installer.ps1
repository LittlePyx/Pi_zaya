[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InstallerPath,
    [switch]$AllowDirty,
    [switch]$RequireSignature,
    [switch]$KeepData
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
. (Join-Path $PSScriptRoot "Authenticode.ps1")

$installer = [IO.Path]::GetFullPath($InstallerPath)
if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    throw "Windows installer was not found: $installer"
}
$checksumPath = "$installer.sha256"
$manifestPath = Join-Path (Split-Path -Parent $installer) "$([IO.Path]::GetFileNameWithoutExtension($installer)).manifest.json"
foreach ($required in @($checksumPath, $manifestPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Installer acceptance file was not found: $required"
    }
}

$expectedHash = (((Get-Content -LiteralPath $checksumPath -Raw).Trim() -split '\s+')[0]).ToLowerInvariant()
if ($expectedHash -notmatch '^[0-9a-f]{64}$') {
    throw "Installer checksum is malformed."
}
$actualHash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) {
    throw "Installer checksum mismatch."
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.package_type -ne "windows_installer" -or $manifest.license -ne "MIT") {
    throw "Installer manifest does not declare the expected Windows/MIT release contract."
}
if ([bool]$manifest.source_dirty -and -not $AllowDirty) {
    throw "Installer was built from a dirty source tree. Use -AllowDirty only for local acceptance."
}
if (-not [bool]$manifest.uninstall_preserves_user_data -or $manifest.install_scope -ne "current_user") {
    throw "Installer manifest does not promise current-user install and preserved user data."
}
$signature = Get-PiZayaAuthenticodeState -Path $installer
if ([bool]$manifest.signed -ne [bool]$signature.Signed -or [string]$manifest.signature_status -ne [string]$signature.Status) {
    throw "Installer Authenticode state does not match its manifest."
}
if ($RequireSignature -and (-not $signature.Signed -or -not $signature.Timestamped)) {
    throw "Installer acceptance requires a valid timestamped Authenticode signature."
}
if ($signature.Signed -and ([string]$manifest.certificate_thumbprint -ne [string]$signature.Thumbprint -or -not $signature.Timestamped)) {
    throw "Installer signature identity or timestamp does not match its manifest."
}

$appId = "{BDA27978-68AE-4C98-8E35-C85D11872562}_is1"
$uninstallKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$appId"
if (Test-Path -LiteralPath $uninstallKey) {
    throw "A real Pi_zaya installation already exists for this Windows user. Refusing to overwrite it during smoke acceptance."
}

$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$smokeRoot = Join-Path ([IO.Path]::GetTempPath()) ("pi-zaya-installer-smoke-" + [guid]::NewGuid().ToString("N"))
$installRoot = Join-Path $smokeRoot "Programs\Pi_zaya"
$dataRoot = Join-Path $smokeRoot "UserData\Pi_zaya"
$sentinel = Join-Path $dataRoot "preserve-after-uninstall.txt"
$installed = $false

function Assert-SmokePath([string]$Path, [string]$Label) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label escaped the system temp directory: $resolved"
    }
}

function Invoke-SetupInstall {
    New-Item -ItemType Directory -Force -Path $installRoot | Out-Null
    $arguments = @(
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/SP-",
        "/NOICONS",
        "/CURRENTUSER",
        ('/DIR="' + $installRoot + '"')
    )
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        throw "Silent installer exited with code $($process.ExitCode)."
    }
}

try {
    Assert-SmokePath $smokeRoot "Installer smoke root"
    New-Item -ItemType Directory -Force -Path $dataRoot | Out-Null
    "Pi_zaya installer acceptance data - do not remove" | Set-Content -LiteralPath $sentinel -Encoding UTF8

    Invoke-SetupInstall
    $installed = $true
    foreach ($required in @("Pi_zaya.exe", "README-中文.md", "LICENSE", "runtime\python\python.exe", "web\dist\index.html", "unins000.exe")) {
        if (-not (Test-Path -LiteralPath (Join-Path $installRoot $required) -PathType Leaf)) {
            throw "Installed application is missing $required"
        }
    }
    $installedLauncherSignature = Get-PiZayaAuthenticodeState -Path (Join-Path $installRoot "Pi_zaya.exe")
    if ([bool]$manifest.launcher_signed -ne [bool]$installedLauncherSignature.Signed -or [string]$manifest.launcher_signature_status -ne [string]$installedLauncherSignature.Status) {
        throw "Installed launcher Authenticode state does not match the installer manifest."
    }
    if ($installedLauncherSignature.Signed -and ([string]$manifest.launcher_certificate_thumbprint -ne [string]$installedLauncherSignature.Thumbprint -or -not $installedLauncherSignature.Timestamped)) {
        throw "Installed launcher signature identity or timestamp does not match the installer manifest."
    }
    $installedUninstallerSignature = Get-PiZayaAuthenticodeState -Path (Join-Path $installRoot "unins000.exe")
    if ([bool]$manifest.signed_uninstaller) {
        if (-not $installedUninstallerSignature.Signed -or $installedUninstallerSignature.Thumbprint -ne [string]$manifest.certificate_thumbprint -or -not $installedUninstallerSignature.Timestamped) {
            throw "Installed uninstaller is not signed and timestamped with the installer certificate."
        }
    }
    elseif ($installedUninstallerSignature.Signed) {
        throw "Unsigned installer manifest unexpectedly installed a signed uninstaller."
    }

    & (Join-Path $repoRoot "tools\release\smoke_windows_portable.ps1") -BundleRoot $installRoot -DataDir $dataRoot -CleanProfile -KeepData -AllowDirty:$AllowDirty
    if (-not (Test-Path -LiteralPath $sentinel -PathType Leaf)) {
        throw "Runtime acceptance unexpectedly removed the user-data sentinel."
    }

    # Reinstalling the same artifact exercises the stable AppId and in-place
    # upgrade path without weakening the active-process protection.
    Invoke-SetupInstall
    if (-not (Test-Path -LiteralPath $sentinel -PathType Leaf)) {
        throw "In-place installer upgrade removed user data."
    }

    $uninstaller = Join-Path $installRoot "unins000.exe"
    $uninstallProcess = Start-Process -FilePath $uninstaller -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART") -PassThru -Wait
    if ($uninstallProcess.ExitCode -ne 0) {
        throw "Silent uninstaller exited with code $($uninstallProcess.ExitCode)."
    }
    $installed = $false
    $deadline = [DateTimeOffset]::UtcNow.AddSeconds(15)
    while ((Test-Path -LiteralPath (Join-Path $installRoot "Pi_zaya.exe")) -and [DateTimeOffset]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 200
    }
    if (Test-Path -LiteralPath (Join-Path $installRoot "Pi_zaya.exe")) {
        throw "Uninstaller did not remove the installed program."
    }
    if (-not (Test-Path -LiteralPath $sentinel -PathType Leaf)) {
        throw "Uninstaller removed Pi_zaya user data."
    }
    if (Test-Path -LiteralPath $uninstallKey) {
        throw "Uninstaller left its current-user registration behind."
    }
    Write-Host "Installer clean-profile, in-place upgrade, and data-preserving uninstall smoke passed for Pi_zaya $($manifest.version)."
}
finally {
    if ($installed) {
        $uninstaller = Join-Path $installRoot "unins000.exe"
        if (Test-Path -LiteralPath $uninstaller -PathType Leaf) {
            Start-Process -FilePath $uninstaller -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART") -Wait -ErrorAction SilentlyContinue | Out-Null
        }
    }
    if (-not $KeepData -and (Test-Path -LiteralPath $smokeRoot)) {
        Assert-SmokePath $smokeRoot "Installer smoke cleanup root"
        Remove-Item -LiteralPath $smokeRoot -Recurse -Force
    }
}
