@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m contaedes interface
) else (
  python -m contaedes interface
)
if errorlevel 1 pause
