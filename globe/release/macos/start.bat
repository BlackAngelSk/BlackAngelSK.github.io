@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo   Globe/Map - Starting servers
echo ============================================
echo.

where node >nul 2>nul
if %errorlevel%==0 (
    echo Node.js found - starting globe-server.js ...
    node "%~dp0globe-server.js"
) else (
    echo Node.js not found - using globe-server.exe ...
    if exist "%~dp0globe-server.exe" (
        "%~dp0globe-server.exe"
    ) else (
        echo ERROR: neither Node.js nor globe-server.exe found in:
        echo   %~dp0
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Server stopped.
pause
