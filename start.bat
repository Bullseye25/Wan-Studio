@echo off
title Wan 2.2 Studio Launcher
cd /d "%~dp0"

echo ============================================================
echo Starting Wan 2.2 Studio Runner...
echo ============================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your PATH.
    echo Please ensure Python is installed and added to PATH.
    pause
    exit /b 1
)

python start.py

if %errorlevel% neq 0 (
    echo.
    echo Process finished with code %errorlevel%.
    pause
)
