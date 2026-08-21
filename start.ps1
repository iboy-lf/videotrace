$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "src"
python scripts\start.py
