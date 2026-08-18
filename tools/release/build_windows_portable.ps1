[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [ValidateSet("Embedded", "System")]
    [string]$PythonRuntime = "Embedded",
    [switch]$SkipFrontendBuild,
    [switch]$KeepStage,
    [switch]$AllowDirty,
    [switch]$AllowMissingLicense,
    [string]$SigningThumbprint = $env:PI_ZAYA_SIGNING_THUMBPRINT,
    [string]$SignToolPath = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [switch]$RequireSignature
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
. (Join-Path $PSScriptRoot "Authenticode.ps1")
if ($RequireSignature -and [string]::IsNullOrWhiteSpace($SigningThumbprint)) {
    throw "-RequireSignature needs a trusted code-signing certificate thumbprint."
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "release"
}
$outputRoot = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$version = (Get-Content -LiteralPath (Join-Path $repoRoot "VERSION") -Raw).Trim()
if ($version -notmatch '^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$') {
    throw "VERSION is not valid SemVer: $version"
}
$gitStatus = @(& git -C $repoRoot status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect the Git working tree before packaging."
}
$sourceDirty = $gitStatus.Count -gt 0
if ($sourceDirty -and -not $AllowDirty) {
    throw "The Git working tree is not clean. Commit the exact release sources, or use -AllowDirty only for a non-distributable acceptance build."
}
$packageName = "Pi_zaya-v$version-windows-x64"
$stageRoot = [IO.Path]::GetFullPath((Join-Path $outputRoot $packageName))
$expectedPrefix = $outputRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $stageRoot.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to stage outside the selected output directory: $stageRoot"
}
if (Test-Path -LiteralPath $stageRoot) {
    Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null

$licensePath = Join-Path $repoRoot "LICENSE"
if (-not (Test-Path -LiteralPath $licensePath) -and -not $AllowMissingLicense) {
    throw "LICENSE is missing. Choose and add the project's distribution license before building a release artifact."
}
if (Test-Path -LiteralPath $licensePath) {
    $licenseText = Get-Content -LiteralPath $licensePath -Raw
    if ($licenseText -notmatch '^MIT License\s' -or $licenseText -notmatch 'Copyright \(c\) [^\r\n]+ LittlePyx' -or $licenseText -notmatch 'Permission is hereby granted') {
        throw "LICENSE does not match the expected MIT license contract for LittlePyx."
    }
}

$webPackage = Get-Content -LiteralPath (Join-Path $repoRoot "web\package.json") -Raw | ConvertFrom-Json
if ([string]$webPackage.version -ne $version) {
    throw "web/package.json version '$($webPackage.version)' does not match VERSION '$version'."
}
if (-not $SkipFrontendBuild) {
    Push-Location (Join-Path $repoRoot "web")
    try {
        npm ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    }
    finally {
        Pop-Location
    }
}
$distRoot = Join-Path $repoRoot "web\dist"
if (-not (Test-Path -LiteralPath (Join-Path $distRoot "index.html"))) {
    throw "web/dist/index.html is missing. Build the frontend or omit -SkipFrontendBuild."
}

function Copy-RuntimeTree([string]$Source, [string]$Destination) {
    $sourceRoot = [IO.Path]::GetFullPath($Source)
    Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | Where-Object {
        $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and $_.Extension -ne '.pyc'
    } | ForEach-Object {
        $relative = $_.FullName.Substring($sourceRoot.Length).TrimStart('\', '/')
        $target = Join-Path $Destination $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $target
    }
}

function New-LauncherIcon([string]$SourcePng, [string]$DestinationIcon) {
    Add-Type -AssemblyName System.Drawing
    $sourceImage = [Drawing.Image]::FromFile($SourcePng)
    $canvas = [Drawing.Bitmap]::new(256, 256, [Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [Drawing.Graphics]::FromImage($canvas)
    $pngStream = [IO.MemoryStream]::new()
    try {
        $graphics.Clear([Drawing.Color]::Transparent)
        $graphics.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $scale = [Math]::Min(224.0 / $sourceImage.Width, 224.0 / $sourceImage.Height)
        $width = [int][Math]::Round($sourceImage.Width * $scale)
        $height = [int][Math]::Round($sourceImage.Height * $scale)
        $left = [int][Math]::Floor((256 - $width) / 2.0)
        $top = [int][Math]::Floor((256 - $height) / 2.0)
        $graphics.DrawImage($sourceImage, $left, $top, $width, $height)
        $canvas.Save($pngStream, [Drawing.Imaging.ImageFormat]::Png)
        $pngBytes = $pngStream.ToArray()
    }
    finally {
        $pngStream.Dispose()
        $graphics.Dispose()
        $canvas.Dispose()
        $sourceImage.Dispose()
    }

    $fileStream = [IO.File]::Create($DestinationIcon)
    $writer = [IO.BinaryWriter]::new($fileStream)
    try {
        # ICONDIR followed by one 256x256 PNG-compressed image entry.
        $writer.Write([uint16]0)
        $writer.Write([uint16]1)
        $writer.Write([uint16]1)
        $writer.Write([byte]0)
        $writer.Write([byte]0)
        $writer.Write([byte]0)
        $writer.Write([byte]0)
        $writer.Write([uint16]1)
        $writer.Write([uint16]32)
        $writer.Write([uint32]$pngBytes.Length)
        $writer.Write([uint32]22)
        $writer.Write($pngBytes)
    }
    finally {
        $writer.Dispose()
        $fileStream.Dispose()
    }
}

function Build-WindowsLauncher([string]$DestinationExe, [string]$DestinationIcon) {
    $sourcePath = Join-Path $repoRoot "packaging\windows\PiZayaLauncher.cs"
    $logoPath = Join-Path $repoRoot "assets\pi_logo.png"
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Windows launcher source is missing: $sourcePath"
    }
    if (-not (Test-Path -LiteralPath $logoPath -PathType Leaf)) {
        throw "Windows launcher logo is missing: $logoPath"
    }
    $frameworkRoot = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319"
    $compiler = Join-Path $frameworkRoot "csc.exe"
    if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) {
        $frameworkRoot = Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319"
        $compiler = Join-Path $frameworkRoot "csc.exe"
    }
    if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) {
        throw "The Windows .NET Framework C# compiler is required to build Pi_zaya.exe."
    }

    New-LauncherIcon $logoPath $DestinationIcon
    $coreVersion = ($version -split '[-+]')[0]
    $coreParts = @($coreVersion -split '\.')
    $revision = 0
    if ($version -match '-[^+]*?(?<revision>\d+)(?:\+.*)?$') {
        $revision = [int]$Matches.revision
    }
    $assemblyVersion = "$($coreParts[0]).$($coreParts[1]).$($coreParts[2]).0"
    $fileVersion = "$($coreParts[0]).$($coreParts[1]).$($coreParts[2]).$revision"
    $versionSource = Join-Path (Split-Path -Parent $DestinationExe) ".PiZayaLauncherVersion.cs"
    @"
[assembly: System.Reflection.AssemblyVersion("$assemblyVersion")]
[assembly: System.Reflection.AssemblyFileVersion("$fileVersion")]
[assembly: System.Reflection.AssemblyInformationalVersion("$version")]
"@ | Set-Content -LiteralPath $versionSource -Encoding UTF8
    $compilerArgs = @(
        "/nologo",
        "/target:winexe",
        "/optimize+",
        "/platform:anycpu",
        "/warn:4",
        "/warnaserror+",
        "/out:$DestinationExe",
        "/win32icon:$DestinationIcon",
        "/reference:$(Join-Path $frameworkRoot 'System.dll')",
        "/reference:$(Join-Path $frameworkRoot 'System.Core.dll')",
        "/reference:$(Join-Path $frameworkRoot 'System.Drawing.dll')",
        "/reference:$(Join-Path $frameworkRoot 'System.Windows.Forms.dll')",
        $sourcePath,
        $versionSource
    )
    try {
        & $compiler @compilerArgs
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $DestinationExe -PathType Leaf)) {
            throw "Compiling the native Windows launcher failed."
        }
    }
    finally {
        Remove-Item -LiteralPath $versionSource -Force -ErrorAction SilentlyContinue
    }
}

Copy-RuntimeTree (Join-Path $repoRoot "api") (Join-Path $stageRoot "api")
Copy-RuntimeTree (Join-Path $repoRoot "kb") (Join-Path $stageRoot "kb")
# The React product still shares framework-neutral citation/Markdown helpers
# from ui/. The removed Streamlit entrypoint is not packaged or launched.
Copy-RuntimeTree (Join-Path $repoRoot "ui") (Join-Path $stageRoot "ui")
Copy-RuntimeTree $distRoot (Join-Path $stageRoot "web\dist")
foreach ($relativePath in @(
    "VERSION",
    "README.md",
    "CHANGELOG.md",
    ".env.production.example",
    "requirements.txt",
    "requirements-release.txt",
    "server.py",
    "ingest.py",
    "pdf_to_md.py"
)) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $relativePath) -Destination (Join-Path $stageRoot $relativePath)
}
if (Test-Path -LiteralPath $licensePath) {
    Copy-Item -LiteralPath $licensePath -Destination (Join-Path $stageRoot "LICENSE")
}
Copy-Item -LiteralPath (Join-Path $repoRoot "packaging\windows\Start-Pi-zaya.ps1") -Destination $stageRoot
Copy-Item -LiteralPath (Join-Path $repoRoot "packaging\windows\Stop-Pi-zaya.ps1") -Destination $stageRoot
Copy-Item -LiteralPath (Join-Path $repoRoot "packaging\windows\Start-Pi-zaya.cmd") -Destination $stageRoot
Copy-Item -LiteralPath (Join-Path $repoRoot "packaging\windows\Stop-Pi-zaya.cmd") -Destination $stageRoot
Copy-Item -LiteralPath (Join-Path $repoRoot "packaging\windows\README-PORTABLE.md") -Destination $stageRoot
Copy-Item -LiteralPath (Join-Path $repoRoot "packaging\windows\README-中文.md") -Destination $stageRoot
Build-WindowsLauncher (Join-Path $stageRoot "Pi_zaya.exe") (Join-Path $stageRoot "Pi_zaya.ico")
$launcherPath = Join-Path $stageRoot "Pi_zaya.exe"
$launcherSignature = if ([string]::IsNullOrWhiteSpace($SigningThumbprint)) {
    Get-PiZayaAuthenticodeState -Path $launcherPath
}
else {
    Invoke-PiZayaAuthenticodeSign -Path $launcherPath -Thumbprint $SigningThumbprint -SignToolPath $SignToolPath -TimestampUrl $TimestampUrl
}
if ($RequireSignature -and (-not $launcherSignature.Signed -or -not $launcherSignature.Timestamped)) {
    throw "The native Windows launcher must have a valid timestamped Authenticode signature."
}

$pythonVersion = (Get-Content -LiteralPath (Join-Path $repoRoot ".python-version") -Raw).Trim()
$runtimeKind = $PythonRuntime.ToLowerInvariant()
if ($PythonRuntime -eq "Embedded") {
    $pythonRuntimeDir = Join-Path $stageRoot "runtime\python"
    New-Item -ItemType Directory -Force -Path $pythonRuntimeDir | Out-Null
    $pythonArchive = Join-Path $outputRoot "python-$pythonVersion-embed-amd64.zip"
    $pythonUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-embed-amd64.zip"
    Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonArchive
    try {
        Expand-Archive -LiteralPath $pythonArchive -DestinationPath $pythonRuntimeDir -Force
    }
    finally {
        Remove-Item -LiteralPath $pythonArchive -Force -ErrorAction SilentlyContinue
    }
    $pythonMinor = ($pythonVersion.Split('.')[0..1] -join '')
    $pthPath = Join-Path $pythonRuntimeDir "python$pythonMinor._pth"
    if (-not (Test-Path -LiteralPath $pthPath)) {
        throw "Embedded Python path configuration is missing: $pthPath"
    }
    $pthLines = Get-Content -LiteralPath $pthPath | Where-Object { $_ -notmatch '^\s*#?\s*import site\s*$' }
    # Embedded Python ignores the process working directory when a _pth file is
    # present. runtime/python is two levels below the staged application root.
    @($pthLines + "..\.." + "Lib/site-packages" + "import site") | Set-Content -LiteralPath $pthPath -Encoding ASCII
    $getPipPath = Join-Path $pythonRuntimeDir "get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipPath
    $embeddedPython = Join-Path $pythonRuntimeDir "python.exe"
    & $embeddedPython $getPipPath --disable-pip-version-check --no-warn-script-location
    if ($LASTEXITCODE -ne 0) { throw "Installing pip into the embedded runtime failed." }
    $releaseRequirements = Join-Path $repoRoot "requirements-release.txt"
    & $embeddedPython -m pip install --disable-pip-version-check --no-warn-script-location --no-compile -r $releaseRequirements
    if ($LASTEXITCODE -ne 0) { throw "Installing backend dependencies into the embedded runtime failed." }
    & $embeddedPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw "The embedded runtime has incompatible dependencies." }
    & $embeddedPython -m pip freeze --all | Set-Content -LiteralPath (Join-Path $stageRoot "THIRD_PARTY_PACKAGES.txt") -Encoding UTF8
    Remove-Item -LiteralPath $getPipPath -Force
}

$commit = (git -C $repoRoot rev-parse --short=12 HEAD 2>$null)
if ($LASTEXITCODE -ne 0) { $commit = "" }
$builtAt = [DateTimeOffset]::UtcNow.ToString("o")
$manifest = [ordered]@{
    schema_version = 1
    product = "Pi_zaya"
    version = $version
    tag = "v$version"
    platform = "windows"
    architecture = "x64"
    package_type = "portable_zip"
    python_runtime = $runtimeKind
    python_version = $pythonVersion
    dependencies_lock = if ($PythonRuntime -eq "Embedded") { "requirements-release.txt" } else { "" }
    frontend_prebuilt = $true
    user_data_default = '%LOCALAPPDATA%\Pi_zaya'
    entrypoint = "Pi_zaya.exe"
    fallback_entrypoint = "Start-Pi-zaya.cmd"
    stop_command = "Stop-Pi-zaya.cmd"
    launcher = "native_windows_tray"
    launcher_runtime = "windows_dotnet_framework_4"
    launcher_signed = [bool]$launcherSignature.Signed
    launcher_signature_status = [string]$launcherSignature.Status
    launcher_certificate_thumbprint = [string]$launcherSignature.Thumbprint
    launcher_certificate_subject = [string]$launcherSignature.Subject
    launcher_timestamped = [bool]$launcherSignature.Timestamped
    commit = [string]$commit
    source_dirty = [bool]$sourceDirty
    built_at = $builtAt
    license = if (Test-Path -LiteralPath $licensePath) { "MIT" } else { "" }
    license_status = if (Test-Path -LiteralPath $licensePath) { "included" } else { "missing-release-blocked" }
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stageRoot "release-manifest.json") -Encoding UTF8

$zipPath = Join-Path $outputRoot "$packageName.zip"
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -LiteralPath $stageRoot -DestinationPath $zipPath -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
$hashLine = "$hash  $([IO.Path]::GetFileName($zipPath))"
$hashPath = "$zipPath.sha256"
$hashLine | Set-Content -LiteralPath $hashPath -Encoding ASCII
$artifactManifest = [ordered]@{
    product = "Pi_zaya"
    version = $version
    artifact = [IO.Path]::GetFileName($zipPath)
    size_bytes = (Get-Item -LiteralPath $zipPath).Length
    sha256 = $hash
    python_runtime = $runtimeKind
    built_at = $builtAt
    commit = [string]$commit
    source_dirty = [bool]$sourceDirty
    license = if (Test-Path -LiteralPath $licensePath) { "MIT" } else { "" }
    launcher_signed = [bool]$launcherSignature.Signed
    launcher_signature_status = [string]$launcherSignature.Status
    launcher_certificate_thumbprint = [string]$launcherSignature.Thumbprint
    launcher_timestamped = [bool]$launcherSignature.Timestamped
}
$artifactManifestPath = Join-Path $outputRoot "$packageName.manifest.json"
$artifactManifest | ConvertTo-Json | Set-Content -LiteralPath $artifactManifestPath -Encoding UTF8

if (-not $KeepStage) {
    Remove-Item -LiteralPath $stageRoot -Recurse -Force
}

Write-Host "Built $zipPath"
Write-Host "SHA256 $hash"
