Globe/Map Server - macOS
========================

Quick Start:
  1. Open Terminal
  2. Navigate to this folder: cd ~/path/to/globe
  3. Run: ./globe-server-x64.bin (Intel Mac) or ./globe-server-arm64.bin (Apple Silicon)
  4. Browser opens automatically to http://localhost:8000
  5. Press Ctrl+C to stop

Note: On first run, macOS may block the executable. Go to:
  System Settings > Privacy & Security > click "Allow Anyway"

Manual Usage:
  ./globe-server-x64.bin            Start proxy + HTTP server (Intel)
  ./globe-server-arm64.bin          Start proxy + HTTP server (Apple Silicon)
  ./globe-server-x64.bin --proxy    Start only proxy
  ./globe-server-x64.bin --port 3000  Custom HTTP port

You can also use the Node.js scripts:
  node proxy.js
  node start.js
