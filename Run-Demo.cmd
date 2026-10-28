@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Run-Demo.ps1"
if errorlevel 1 (echo Demo failed. & pause & exit /b 1)
pause
