from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_remote_launcher_preflights_port_and_reuses_healthy_tunnel():
    script = (ROOT / "scripts/start_remote.ps1").read_text(encoding="utf-8")
    assert "Get-HealthyVideoTraceService" in script
    assert "VideoTrace is already ready" in script
    assert script.index("Get-LocalListener $LocalPort") < script.index(
        'Write-Host "Starting the remote VideoTrace Web service'
    )
    assert "it is not a healthy VideoTrace tunnel" in script
    assert "VIDEOTRACE_RESTART_STALE=1" in script
    assert "VIDEOTRACE_WEB_PORT=$RemotePort" in script
    assert "Get-HealthyVideoTraceService $LocalPort $expectedSource" in script
    assert "source_sha256" in script
    assert "Remote Web health does not match the requested VideoTrace root/product/source contract" in script
    assert script.index("Get-LocalListener $LocalPort") < script.index(
        'Write-Host "Starting the remote VideoTrace Web service'
    )
    assert script.index('Write-Host "Starting the remote VideoTrace Web service') < script.index(
        'Write-Host "VideoTrace is already ready'
    )


def test_remote_launcher_owns_and_cleans_up_new_ssh_tunnel():
    script = (ROOT / "scripts/start_remote.ps1").read_text(encoding="utf-8")
    assert '"ExitOnForwardFailure=yes"' in script
    assert '"ServerAliveInterval=30"' in script
    assert "-PassThru" in script
    assert "Stop-Process -Id $tunnel.Id" in script
    assert 'schema_version = "videotrace-ssh-tunnel-v1"' in script
    assert 'outputs_runtime\\tunnels' in script
    assert "[ValidateRange(1, 65535)]" in script
    assert 'remote_port = $RemotePort' in script


def test_remote_web_wrapper_selects_gpus_exactly_once_and_keeps_runtime_audit_separate():
    wrapper = (ROOT / "scripts/remote/run_web.sh").read_text(encoding="utf-8")
    service = (ROOT / "scripts/remote/start_web_service.sh").read_text(encoding="utf-8")
    stop = (ROOT / "scripts/remote/stop_web_service.sh").read_text(encoding="utf-8")

    assert "select_gpus.py" not in wrapper
    assert service.count("select_gpus.py") == 1
    assert 'outputs_runtime/web' in service
    assert 'gpu_selection_audit.json' in service
    assert 'gpu_selection_canonical.json' not in service
    assert '--preferred-qwen-index' in service
    assert '--preferred-siglip-index' in service
    assert 'VIDEOTRACE_REQUIRE_IDLE_STOP=1' in service
    assert 'scripts/run_web.py' in stop
    assert 'kill -TERM "$pid"' in stop
    assert "SIGKILL" in stop


def test_remote_pytest_launcher_never_injects_system_site_packages():
    script = (ROOT / "scripts/remote/run_pytest.py").read_text(encoding="utf-8")
    assert "/usr/local/lib/python3.10/dist-packages" not in script
    assert "sys.path.append" not in script
    assert "VIDEOTRACE_TEST_PYTHON" in script
    assert "/linyuanping/miniconda3/envs/wyf_vm/bin/python" in script
    assert "pytest, sklearn and a usable MP4 encoder" in script
