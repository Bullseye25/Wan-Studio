@echo off
setlocal enabledelayedexpansion
title Video Combiner with 1s Fade-to-Black Transitions

echo ===================================================================
echo     Video Transition and Combiner Tool
echo ===================================================================
echo.

cd /d "%~dp0"

REM Check for python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please ensure Python is installed and accessible.
    pause
    exit /b 1
)

python process_videos.py

echo.
echo ===================================================================
echo     Finished! Press any key to exit.
echo ===================================================================
pause >nul
