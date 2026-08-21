param(
    [string]$RemoteHost = "iboy",
    [string]$RemoteRoot = "/lavender/VideoTrace"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$syncDir = Join-Path $root ".sync"
New-Item -ItemType Directory -Force -Path $syncDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$archive = Join-Path $syncDir "videotrace-code-$stamp.tar.gz"
$remoteArchive = "/tmp/videotrace-code-$stamp.tar.gz"

$include = @(
    ".gitignore",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "start.bat",
    "start.ps1",
    "assets",
    "configs",
    "data/preference",
    "data/verifier",
    "data/supervision",
    "data/regression_cases.json",
    "docs",
    "scripts",
    "src",
    "tests"
)

try {
    & tar -czf $archive `
        --exclude="__pycache__" `
        --exclude="*.pyc" `
        --exclude=".pytest_cache" `
        -C $root @include
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }
    & scp $archive "${RemoteHost}:$remoteArchive"
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed with exit code $LASTEXITCODE"
    }
    $remoteCommand = "mkdir -p '$RemoteRoot' && tar -xzf '$remoteArchive' -C '$RemoteRoot' && rm -f '$remoteArchive'"
    & ssh $RemoteHost $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "remote extraction failed with exit code $LASTEXITCODE"
    }
    Write-Host "Synced code to ${RemoteHost}:$RemoteRoot without touching raw/SFT data, outputs, caches, or models."
}
finally {
    if (Test-Path -LiteralPath $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
}
