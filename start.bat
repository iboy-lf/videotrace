@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=src
python scripts\start.py
pause
