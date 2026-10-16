@echo off
setlocal EnableDelayedExpansion
title Map Proxy Setup & Server

echo ============================================
echo   Map Proxy - Auto Setup & Launcher
echo ============================================
echo.

set "RAW=https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe"

:: ── Check if Node.js is installed ──────────────
echo [1/4] Checking for Node.js...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   Node.js is NOT installed.
    echo   Attempting to install Node.js automatically...
    echo.

    where winget >nul 2>&1
    if !errorlevel! equ 0 (
        echo   Using winget to install Node.js LTS...
        winget install OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
        set "PATH=%LOCALAPPDATA%\Programs\node;%ProgramFiles%\nodejs;!PATH!"
    )

    where node >nul 2>&1
    if !errorlevel! neq 0 (
        echo   Downloading Node.js installer...
        set "NODE_URL=https://nodejs.org/dist/v20.15.1/node-v20.15.1-x64.msi"
        set "NODE_INSTALLER=%TEMP%\node-install.msi"
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%NODE_URL%' -OutFile '%NODE_INSTALLER%' -UseBasicParsing"
        if !errorlevel! neq 0 (
            echo.
            echo   ERROR: Failed to download Node.js installer.
            echo   Please install manually from: https://nodejs.org
            echo.
            pause
            exit /b 1
        )
        echo   Installing Node.js ^(this may require admin rights^)...
        msiexec /i "%NODE_INSTALLER%" /qn /norestart
        set "PATH=%LOCALAPPDATA%\Programs\node;%ProgramFiles%\nodejs;!PATH!"
        del "%NODE_INSTALLER%" 2>nul
    )

    where node >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo   ERROR: Node.js installation failed.
        echo   Please install Node.js manually from: https://nodejs.org
        echo.
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%v in ('node --version 2^>nul') do set "NODE_VER=%%v"
echo   Node.js found: !NODE_VER!
echo.

:: ── Check if proxy.js exists ──────────────────
echo [2/4] Checking for proxy script...
set "SCRIPT_DIR=%~dp0"
echo   Script directory: "!SCRIPT_DIR!"

if not exist "!SCRIPT_DIR!proxy.js" (
    echo   proxy.js not found locally. Downloading from GitHub...
    where curl.exe >nul 2>&1
    if !errorlevel! equ 0 (
        curl.exe -fsSL "!RAW!/proxy.js" -o "!SCRIPT_DIR!proxy.js"
    ) else (
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!RAW!/proxy.js' -OutFile '!SCRIPT_DIR!proxy.js' -UseBasicParsing"
    )
)

:: Validate the download — a 404 from GitHub saves an HTML/JSON page that
:: node cannot run (this used to break the Windows setup silently).
findstr /C:"PROXY_VERSION" "!SCRIPT_DIR!proxy.js" >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo   ERROR: proxy.js is missing or invalid ^(download failed^).
    echo   Please download manually:
    echo     !RAW!/proxy.js
    echo   and save it as: !SCRIPT_DIR!proxy.js
    echo.
    pause
    exit /b 1
)
echo   proxy.js OK.

:: ── Port note ────────────────────────────────
echo [3/4] Port 8080 - the proxy stops an older copy of itself automatically.
echo.

:: ── Start the proxy ──────────────────────────
echo [4/4] Starting the KML proxy (plain HTTP 8080)...
echo.
echo   ============================================
echo   Proxy + map:  http://localhost:8080/map.html
echo   Health check: http://localhost:8080/ping
echo   ============================================
echo.
echo   Plain HTTP - no certificate, nothing to accept.
echo   Leave this window open and go back to the map.
echo.
echo   Close this window or press Ctrl+C to stop.
echo.

node "!SCRIPT_DIR!proxy.js"
pause
