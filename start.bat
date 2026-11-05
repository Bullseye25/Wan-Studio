@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
title Wan 2.2 AI Studio Launcher ^& Modal Deployer
cd /d "%~dp0"

echo ========================================================================================
echo                 Wan 2.2 AI Studio — Launcher, Setup ^& Modal Deployer
echo ========================================================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your PATH.
    echo Please ensure Python is installed and added to PATH.
    pause
    exit /b 1
)

REM Allow direct CLI flag execution (e.g., start.bat --deploy-all, start.bat --check-setup)
if not "%~1"=="" (
    python start.py %*
    exit /b %errorlevel%
)

echo.
echo   [1] Start Studio Frontend Locally (http://localhost:7860)
echo   [2] Pre-flight Setup ^& Deployment Verification (Modal Auth + Budget Cap + LLMs)
echo   [3] Deploy ALL 4 Studio Backends to Modal (Wan 2.2 + Qwen + Z-Image + Kokoro Narrator)
echo   [4] Deploy Wan 2.2 Video Backend Only (deploy_wan.py)
echo   [5] Deploy Qwen Image Edit Backend Only (deploy_qwen.py)
echo   [6] Deploy Z-Image Turbo Character Backend Only (deploy_z_image.py)
echo   [7] Deploy Kokoro-82M Neural Narrator Backend Only (deploy_narrator.py)
echo   [8] Pre-cache ALL Model Weights on Modal Volumes
echo   [9] Run OmniFlow-30 Fast Story CLI (Fast 30s Movie CLI)
echo  [10] Start Studio Frontend on Local Network (Mobile / LAN)
echo  [11] Switch / Manage Modal Account Profiles
echo  [12] Clear Local Outputs Folder (outputs/)
echo   [0] Open Full Interactive Python Runner Menu (start.py)
echo.
set /p BAT_CHOICE="Select option [1-12, 0] (default 1): "

if "%BAT_CHOICE%"=="" set BAT_CHOICE=1

if "%BAT_CHOICE%"=="1" (
    python start.py --run
) else if "%BAT_CHOICE%"=="2" (
    python start.py --check-setup
) else if "%BAT_CHOICE%"=="3" (
    python start.py --deploy-all
) else if "%BAT_CHOICE%"=="4" (
    python start.py --deploy-wan
) else if "%BAT_CHOICE%"=="5" (
    python start.py --deploy-qwen
) else if "%BAT_CHOICE%"=="6" (
    python start.py --deploy-z-image
) else if "%BAT_CHOICE%"=="7" (
    python start.py --deploy-narrator
) else if "%BAT_CHOICE%"=="8" (
    python start.py --cache-all
) else if "%BAT_CHOICE%"=="9" (
    python start.py --fast-story
) else if "%BAT_CHOICE%"=="10" (
    python start.py --lan
) else if "%BAT_CHOICE%"=="11" (
    python start.py --profile
) else if "%BAT_CHOICE%"=="12" (
    python start.py --clear-outputs
) else (
    python start.py
)

if %errorlevel% neq 0 (
    echo.
    echo Process finished with code %errorlevel%.
    pause
) else if not "%BAT_CHOICE%"=="1" if not "%BAT_CHOICE%"=="10" (
    echo.
    pause
)
