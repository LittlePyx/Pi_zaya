@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Stop-Pi-zaya.ps1"
if errorlevel 1 (
  echo.
  echo Pi-zaya could not be stopped safely. See the message above.
  pause
  exit /b 1
)
