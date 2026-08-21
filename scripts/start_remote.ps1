param(
    [string]$RemoteHost = "iboy",
    [string]$RemoteRoot = "/lavender/VideoTrace",
    [ValidateRange(1, 65535)]
    [int]$LocalPort = 7860,
    [ValidateRange(1, 65535)]
    [int]$RemotePort = 7860,
    [ValidateRange(0, 86400)]
    [int]$GpuWaitSeconds = 1800
)

$ErrorActionPreference = "Stop"

function Get-HealthyVideoTraceService([int]$Port, [string]$ExpectedSource = "") {
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 3
        $healthRoot = ([string]$health.root).TrimEnd('/')
        $sourceMatches = [string]::IsNullOrWhiteSpace($ExpectedSource) -or ([string]$health.source_sha256 -eq $ExpectedSource)
        if ($health.ok -and [int]$health.product_version -ge 3 -and $healthRoot -eq $RemoteRoot.TrimEnd('/') -and $sourceMatches) {
            return $health
        }
    } catch {
        return $null
    }
    return $null
}

function Get-LocalListener([int]$Port) {
    try {
        return Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
    } catch {
        return $null
    }
}

$existingListener = Get-LocalListener $LocalPort
if ($existingListener) {
    $existingHealth = Get-HealthyVideoTraceService $LocalPort
    if (-not $existingHealth) {
        throw "Local port $LocalPort is already used by PID $($existingListener.OwningProcess), and it is not a healthy VideoTrace tunnel. Choose another -LocalPort."
    }
}

Write-Host "Starting the remote VideoTrace Web service on $RemoteHost ..."
$remoteCommand = "cd '$RemoteRoot' && VIDEOTRACE_WEB_PORT=$RemotePort VIDEOTRACE_GPU_WAIT_SECONDS=$GpuWaitSeconds VIDEOTRACE_RESTART_STALE=1 bash scripts/remote/start_web_service.sh"
& ssh $RemoteHost $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "Remote Web service failed to start. Inspect $RemoteRoot/outputs_runtime/web/server.log on $RemoteHost."
}

$remoteHealthText = & ssh $RemoteHost "curl -fsS http://127.0.0.1:$RemotePort/api/health"
if ($LASTEXITCODE -ne 0) {
    throw "Remote Web service started but its health endpoint is not reachable on $RemoteHost port $RemotePort."
}
try {
    $remoteHealth = ($remoteHealthText -join "`n") | ConvertFrom-Json
} catch {
    throw "Remote Web service returned invalid health JSON on $RemoteHost port $RemotePort."
}
$remoteHealthRoot = ([string]$remoteHealth.root).TrimEnd('/')
$expectedSource = ([string]$remoteHealth.source_sha256).Trim()
if (-not $remoteHealth.ok -or [int]$remoteHealth.product_version -lt 3 -or $remoteHealthRoot -ne $RemoteRoot.TrimEnd('/') -or [string]::IsNullOrWhiteSpace($expectedSource)) {
    throw "Remote Web health does not match the requested VideoTrace root/product/source contract."
}

if ($existingListener) {
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $existingHealth = Get-HealthyVideoTraceService $LocalPort $expectedSource
        if ($existingHealth) {
            Write-Host "VideoTrace is already ready: http://127.0.0.1:$LocalPort"
            Write-Host "Remote root: $($existingHealth.root)"
            Write-Host "Source SHA-256: $($existingHealth.source_sha256)"
            exit 0
        }
        Start-Sleep -Milliseconds 500
    }
    throw "The existing VideoTrace tunnel did not become healthy after the remote service restart."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeDir = Join-Path $projectRoot "outputs_runtime\tunnels"
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
$safeHost = $RemoteHost -replace '[^A-Za-z0-9_.-]', '_'
$pidRecord = Join-Path $runtimeDir "$safeHost-$LocalPort.json"
$tunnelArgs = @(
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-N",
    "-L", "${LocalPort}:127.0.0.1:${RemotePort}",
    $RemoteHost
)
$tunnel = Start-Process -FilePath "ssh" -ArgumentList $tunnelArgs -WindowStyle Hidden -PassThru
$ready = $false

try {
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        if ($tunnel.HasExited) {
            throw "SSH tunnel exited with code $($tunnel.ExitCode) before becoming ready."
        }
        $health = Get-HealthyVideoTraceService $LocalPort $expectedSource
        if ($health) {
            $ready = $true
            break
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        throw "SSH tunnel did not expose the remote Web service at http://127.0.0.1:$LocalPort within 10 seconds."
    }

    $record = [ordered]@{
        schema_version = "videotrace-ssh-tunnel-v1"
        pid = $tunnel.Id
        local_url = "http://127.0.0.1:$LocalPort"
        remote_host = $RemoteHost
        remote_root = $RemoteRoot
        remote_port = $RemotePort
        started_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    $tempRecord = "$pidRecord.tmp"
    $record | ConvertTo-Json | Set-Content -LiteralPath $tempRecord -Encoding UTF8
    Move-Item -LiteralPath $tempRecord -Destination $pidRecord -Force
} catch {
    if (-not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $pidRecord) {
        Remove-Item -LiteralPath $pidRecord -Force
    }
    throw
}

Write-Host "VideoTrace is ready: http://127.0.0.1:$LocalPort"
Write-Host "Source SHA-256: $expectedSource"
Write-Host "SSH tunnel PID: $($tunnel.Id)"
Write-Host "Tunnel record: $pidRecord"
Write-Host "Remote logs: $RemoteHost`:$RemoteRoot/outputs_runtime/web/server.log"
