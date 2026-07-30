@echo off
echo ============================================
echo   KML CORS Proxy
echo ============================================
echo.

echo Starting KML proxy on port 8080...
echo.
echo   Proxy: http://localhost:8080
echo.
echo   Close this window to stop the proxy.
echo.

node "%~dp0proxy.js"