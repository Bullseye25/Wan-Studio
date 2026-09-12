@echo off
title Smart Studio - Wan 2.2 & Hermes 3 Runner
cd /d "%~dp0\Smart Studio"

echo ============================================================
echo Launching Smart Studio (Wan 2.2 + Hermes 3 LLM)...
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
