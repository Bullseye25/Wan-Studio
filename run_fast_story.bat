@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ========================================================================================
echo   WAN 2.2 FAST ANIMATION STUDIO · 30-SECOND PRODUCTION CLI
echo ========================================================================================
echo.

set PYTHON_EXE=C:\Users\Ammad Raza\AppData\Local\Python\pythoncore-3.14-64\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

if "%~1"=="" (
    "%PYTHON_EXE%" fast_story_cli.py
) else if "%~2"=="" (
    "%PYTHON_EXE%" fast_story_cli.py "%~1"
) else (
    "%PYTHON_EXE%" fast_story_cli.py "%~1" --story-folder "%~2"
)

echo.
echo Press any key to exit...
pause >nul
