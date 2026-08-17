@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Pi-zaya.ps1"
if errorlevel 1 (
  echo.
  echo Pi-zaya failed to start. Check the logs under %%LOCALAPPDATA%%\Pi_zaya\logs.
  pause
  exit /b 1
)
