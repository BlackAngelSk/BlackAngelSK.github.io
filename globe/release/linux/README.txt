Globe/Map Server - Linux
========================

Quick Start:
  1. Open a terminal
  2. Navigate to this folder: cd ~/path/to/globe
  3. Make executable: chmod +x globe-server
  4. Run: ./globe-server
  5. Browser opens automatically to http://localhost:8000
  6. Press Ctrl+C to stop

Manual Usage:
  ./globe-server                    Start proxy + HTTP server
  ./globe-server --proxy            Start only proxy
  ./globe-server --port 3000        Custom HTTP port

You can also use the Node.js scripts:
  node proxy.js
  node start.js
