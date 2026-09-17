Globe/Map Server - Windows
==========================

Quick Start:
  1. Double-click globe-server.exe
  2. Browser opens automatically to http://localhost:8000
  3. Press Ctrl+C to stop

Manual Usage:
  globe-server.exe                  Start proxy + HTTP server
  globe-server.exe --proxy          Start only proxy
  globe-server.exe --port 3000      Custom HTTP port

The CORS proxy runs on port 8080.
The map/globe server runs on port 8000.

You can also use proxy.js with Node.js if you prefer:
  node proxy.js
  node start.js
