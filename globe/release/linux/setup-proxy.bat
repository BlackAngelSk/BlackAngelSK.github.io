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

:: ── Certificate for the HTTPS listener ────────
echo   Checking HTTPS certificate...
if not exist "!SCRIPT_DIR!.proxy-cert.pem" (
    where curl.exe >nul 2>&1
    if !errorlevel! equ 0 (
        curl.exe -fsSL "!RAW!/.proxy-cert.pem" -o "!SCRIPT_DIR!.proxy-cert.pem"
    ) else (
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!RAW!/.proxy-cert.pem' -OutFile '!SCRIPT_DIR!.proxy-cert.pem' -UseBasicParsing"
    )
)
if not exist "!SCRIPT_DIR!.proxy-key.pem" (
    where curl.exe >nul 2>&1
    if !errorlevel! equ 0 (
        curl.exe -fsSL "!RAW!/.proxy-key.pem" -o "!SCRIPT_DIR!.proxy-key.pem"
    ) else (
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!RAW!/.proxy-key.pem' -OutFile '!SCRIPT_DIR!.proxy-key.pem' -UseBasicParsing"
    )
)
echo   Certificate ready.
echo.

:: ── Check for existing proxy process ──────────
echo [3/4] Checking ports 8080 and 8443...
for %%P in (8080 8443) do (
    netstat -ano | findstr ":%%P" | findstr "LISTENING" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   Port %%P is in use - freeing it...
        for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%%P" ^| findstr "LISTENING"') do (
            taskkill /PID %%p /F >nul 2>&1
        )
    )
)
timeout /t 1 /nobreak >nul
echo.

:: ── Start the proxy ──────────────────────────
echo [4/4] Starting KML proxy ^(HTTP 8080 + HTTPS 8443^)...
echo.
echo   ============================================
echo   Proxy:  http://localhost:8080/ping
echo   Proxy:  https://localhost:8443/ping
echo   ============================================
echo.
echo   The map on https://blackangelsk.github.io is an HTTPS page, so it can
echo   only reach the proxy over HTTPS. One-time step:
echo.
echo     1. Open  https://localhost:8443/ping  in your browser
echo     2. The certificate warning appears - click "Advanced" then
echo        "Continue to localhost (unsafe)"
echo     3. You should see "pong" - now reload the map and import again
echo.
echo   Close this window or press Ctrl+C to stop.
echo.

node "!SCRIPT_DIR!proxy.js"
pause
