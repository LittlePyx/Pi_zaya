$ErrorActionPreference = "Stop"

function Find-PiZayaSignTool {
    param([string]$ExplicitPath = "")

    if ($ExplicitPath) {
        $resolved = [IO.Path]::GetFullPath($ExplicitPath)
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            throw "signtool.exe was not found at the explicit path: $resolved"
        }
        return $resolved
    }

    $command = Get-Command signtool.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $kitRoots = @(
        (Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"),
        (Join-Path $env:ProgramFiles "Windows Kits\10\bin")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) }
    $candidates = foreach ($root in $kitRoots) {
        Get-ChildItem -LiteralPath $root -Filter signtool.exe -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '[\\/]x64[\\/]signtool\.exe$' }
    }
    $selected = $candidates | Sort-Object FullName -Descending | Select-Object -First 1
    if (-not $selected) {
        throw "signtool.exe was not found. Install the Windows SDK or pass -SignToolPath."
    }
    return $selected.FullName
}

function Get-PiZayaSigningCertificate {
    param([Parameter(Mandatory = $true)][string]$Thumbprint)

    $normalized = ($Thumbprint -replace '\s', '').ToUpperInvariant()
    if ($normalized -notmatch '^[0-9A-F]{40,64}$') {
        throw "The Authenticode certificate thumbprint is malformed."
    }
    $certificate = @(
        Get-ChildItem Cert:\CurrentUser\My -ErrorAction SilentlyContinue
        Get-ChildItem Cert:\LocalMachine\My -ErrorAction SilentlyContinue
    ) | Where-Object {
        ($_.Thumbprint -replace '\s', '').ToUpperInvariant() -eq $normalized
    } | Select-Object -First 1
    if (-not $certificate) {
        throw "The Authenticode certificate was not found in the CurrentUser or LocalMachine personal store."
    }
    if (-not $certificate.HasPrivateKey) {
        throw "The Authenticode certificate does not expose its private key."
    }
    if ($certificate.NotAfter -le (Get-Date)) {
        throw "The Authenticode certificate has expired."
    }
    return $certificate
}

function Get-PiZayaAuthenticodeState {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "Authenticode target was not found: $resolved"
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $resolved
    [pscustomobject]@{
        Signed = $signature.Status -eq [Management.Automation.SignatureStatus]::Valid
        Status = [string]$signature.Status
        Thumbprint = if ($signature.SignerCertificate) { [string]$signature.SignerCertificate.Thumbprint } else { "" }
        Subject = if ($signature.SignerCertificate) { [string]$signature.SignerCertificate.Subject } else { "" }
        Timestamped = [bool]$signature.TimeStamperCertificate
    }
}

function Invoke-PiZayaAuthenticodeSign {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Thumbprint,
        [string]$SignToolPath = "",
        [string]$TimestampUrl = "http://timestamp.digicert.com"
    )

    $resolvedPath = [IO.Path]::GetFullPath($Path)
    $certificate = Get-PiZayaSigningCertificate -Thumbprint $Thumbprint
    $signTool = Find-PiZayaSignTool -ExplicitPath $SignToolPath
    & $signTool sign /sha1 $certificate.Thumbprint /fd SHA256 /td SHA256 /tr $TimestampUrl /d "Pi_zaya" $resolvedPath
    if ($LASTEXITCODE -ne 0) {
        throw "signtool.exe failed to sign $resolvedPath."
    }
    & $signTool verify /pa /all $resolvedPath
    if ($LASTEXITCODE -ne 0) {
        throw "signtool.exe could not verify $resolvedPath after signing."
    }
    $state = Get-PiZayaAuthenticodeState -Path $resolvedPath
    if (-not $state.Signed -or $state.Thumbprint -ne $certificate.Thumbprint -or -not $state.Timestamped) {
        throw "The Authenticode signature did not validate with the selected certificate and RFC 3161 timestamp."
    }
    return $state
}
