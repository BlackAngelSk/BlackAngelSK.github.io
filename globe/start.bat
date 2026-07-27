@echo off
echo ============================================
echo   Interactive Map - Starting Servers
echo ============================================
echo.

REM Start the KML CORS proxy in a separate window
echo Starting KML proxy on port 8080...
start "KML Proxy - localhost:8080" node "%~dp0proxy.js"

REM Wait a moment for the proxy to start
timeout /t 2 /nobreak >nul

echo Starting map server on port 8000...
echo.
echo   Map:  http://localhost:8000/map.html
echo   Globe: http://localhost:8000/index.html
echo.
echo   Close this window to stop the map server.
echo   Close the proxy window to stop the proxy.
echo.

REM Start the HTTP server in this window
python -m http.server 8000 --directory "%~dp0"
