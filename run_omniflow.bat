@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ========================================================================================
echo   OmniFlow-30: Next-Gen 30s 9:16 Vertical Story-to-Animation Production Engine
echo ========================================================================================
echo.

set PYTHON_EXE=C:\Users\Ammad Raza\AppData\Local\Python\pythoncore-3.14-64\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

if "%~1"=="" (
    "%PYTHON_EXE%" omniflow_cli.py
) else if "%~2"=="" (
    "%PYTHON_EXE%" omniflow_cli.py "%~1"
) else (
    "%PYTHON_EXE%" omniflow_cli.py "%~1" --story-folder "%~2"
)

echo.
echo Press any key to exit...
pause >nul
