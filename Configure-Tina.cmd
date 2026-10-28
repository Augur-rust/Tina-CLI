@echo off
setlocal
"%~dp0bin\tina.exe" configure %*
if errorlevel 1 (echo Configuration failed. & pause & exit /b 1)
pause
