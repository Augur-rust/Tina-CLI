@echo off
setlocal
"%~dp0bin\tina.exe" %*
if errorlevel 1 (echo Tina exited with an error. & pause & exit /b 1)
