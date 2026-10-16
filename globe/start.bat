@echo off
setlocal
cd /d "%~dp0"
title Globe/Map server (plain HTTP)

echo ============================================
echo   Globe/Map - starting the proxy (plain HTTP)
echo ============================================
echo.

where node >nul 2>nul
if %errorlevel%==0 goto usenode
if exist "%~dp0globe-server.exe" goto usebin

echo ERROR: neither Node.js nor globe-server.exe was found in:
echo   %~dp0
echo.
echo Fix it with either of these:
echo   * install Node.js from https://nodejs.org and run this file again
echo   * or unzip globe-windows.zip (portable build, no Node.js needed)
echo     from https://blackangelsk.github.io/globe/
goto done

:usenode
echo Node.js found - starting globe-server.js ...
echo.
node "%~dp0globe-server.js" %*
goto done

:usebin
echo Node.js not found - starting globe-server.exe ...
echo.
"%~dp0globe-server.exe" %*
goto done

:done
echo.
echo Server stopped. Press any key to close this window.
pause >nul
endlocal
