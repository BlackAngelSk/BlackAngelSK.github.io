@echo off
setlocal EnableDelayedExpansion
title Map Proxy Setup & Server

echo ============================================
echo   Map Proxy - Auto Setup & Launcher
echo ============================================
echo.

:: ── Check if Node.js is installed ──────────────
echo [1/4] Checking for Node.js...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   Node.js is NOT installed.
    echo   Attempting to install Node.js automatically...
    echo.

    :: Try winget first (Windows 10/11 built-in)
    where winget >nul 2>&1
    if !errorlevel! equ 0 (
        echo   Using winget to install Node.js LTS...
        winget install OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
        if !errorlevel! neq 0 (
            echo   winget install failed. Trying direct download...
            goto :downloadNode
        )
        :: Refresh PATH
        set "PATH=%LOCALAPPDATA%\Programs\node;!PATH!"
    ) else (
        goto :downloadNode
    )

    :: Verify install
    where node >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo   ERROR: Node.js installation failed.
        echo   Please install Node.js manually from: https://nodejs.org
        echo.
        pause
        exit /b 1
    )
    goto :nodeInstalled
)

goto :nodeInstalled

:downloadNode
echo   Downloading Node.js installer...
set "NODE_URL=https://nodejs.org/dist/v20.15.1/node-v20.15.1-x64.msi"
set "NODE_INSTALLER=%TEMP%\node-install.msi"
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%NODE_URL%' -OutFile '%NODE_INSTALLER%' -UseBasicParsing"
if %errorlevel% neq 0 (
    echo   ERROR: Failed to download Node.js installer.
    echo   Please install manually from: https://nodejs.org
    pause
    exit /b 1
)
echo   Installing Node.js (this may require admin rights)...
msiexec /i "%NODE_INSTALLER%" /qn /norestart
if %errorlevel% neq 0 (
    echo   ERROR: Node.js installation failed. You may need to run this script as Administrator.
    echo   Or install manually from: https://nodejs.org
    pause
    exit /b 1
)
set "PATH=%LOCALAPPDATA%\Programs\node;%ProgramFiles%\nodejs;!PATH!"
del "%NODE_INSTALLER%" 2>nul

:nodeInstalled
:: Get Node.js version
for /f "tokens=*" %%v in ('node --version 2^>nul') do set "NODE_VER=%%v"
echo   Node.js found: !NODE_VER!
echo.

:: ── Check if proxy.js exists ──────────────────
echo [2/4] Checking for proxy script...
set "SCRIPT_DIR=%~dp0"
echo   Script directory: "!SCRIPT_DIR!"
echo   Looking for: "!SCRIPT_DIR!proxy.js"

if exist "!SCRIPT_DIR!proxy.js" (
    echo   proxy.js found.
    goto :proxyReady
)

echo   proxy.js not found locally. Attempting download...

set "REPO_RAW=https://github.com/BlackAngelSk/BlackAngelSK.github.io/main/globe/proxy.js
echo   Download URL: !REPO_RAW!

:: Try curl.exe first (built into Windows 10+)
where curl.exe >nul 2>&1
if !errorlevel! equ 0 (
    echo   Downloading using curl.exe...
    curl.exe -fsSL -o "!SCRIPT_DIR!proxy.js" "!REPO_RAW!"
    if !errorlevel! equ 0 (
        echo   Downloaded proxy.js successfully via curl.
        goto :verifyProxy
    )
    echo   curl download failed. Trying PowerShell...
)

:: Fallback to PowerShell
echo   Downloading using PowerShell...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!REPO_RAW!' -OutFile '!SCRIPT_DIR!proxy.js' -UseBasicParsing"
if !errorlevel! neq 0 (
    echo.
    echo   ERROR: Failed to download proxy.js
    echo   Tried both curl.exe and PowerShell.
    echo   Please download manually from:
    echo   !REPO_RAW!
    echo   And save it to: !SCRIPT_DIR!proxy.js
    echo.
    pause
    exit /b 1
)

:verifyProxy
:: Verify the downloaded file is valid (not a 404 HTML page)
set "FILE_SIZE=0"
for %%A in ("!SCRIPT_DIR!proxy.js") do set "FILE_SIZE=%%~zA"
echo   Downloaded file size: !FILE_SIZE! bytes
if !FILE_SIZE! lss 50 (
    echo.
    echo   ERROR: proxy.js seems too small (!FILE_SIZE! bytes).
    echo   The download may have failed (e.g., 404 error from GitHub).
    echo   Check your internet connection and try again.
    echo   Or download manually from:
    echo   !REPO_RAW!
    echo.
    pause
    exit /b 1
)
echo   proxy.js downloaded successfully.

:proxyReady
echo.

:: ── Check for existing proxy process ──────────
echo [3/4] Checking port 8080...
netstat -ano | findstr ":8080" | findstr "LISTENING" >nul 2>&1
if !errorlevel! equ 0 (
    echo   Port 8080 is already in use (proxy may already be running).
    echo   Attempting to free port...
    for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8080" ^| findstr "LISTENING"') do (
        taskkill /PID %%p /F >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
)
echo.

:: ── Start the proxy ──────────────────────────
echo [4/4] Starting KML proxy on port 8080...
echo.
echo   ============================================
echo   Proxy is running at: http://localhost:8080
echo   ============================================
echo.
echo   Use this URL in the map's Import ^> URL tab:
echo   http://localhost:8080/kml?mid=YOUR_MAP_ID
echo.
echo   Close this window or press Ctrl+C to stop.
echo.

node "!SCRIPT_DIR!proxy.js"