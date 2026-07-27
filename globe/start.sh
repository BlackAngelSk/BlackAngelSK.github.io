#!/bin/bash
echo "============================================"
echo "  Interactive Map - Starting Servers"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill any existing processes on our ports
echo "Checking for existing servers on ports 8000/8080..."
fuser -k 8000/tcp 2>/dev/null
fuser -k 8080/tcp 2>/dev/null
sleep 1

# Start KML CORS proxy in background
echo "Starting KML proxy on port 8080..."
node "$SCRIPT_DIR/proxy.js" &
PROXY_PID=$!

# Wait for proxy to start
sleep 1

echo "Starting map server on port 8000..."
echo
echo "  Map:   http://localhost:8000/map.html"
echo "  Globe: http://localhost:8000/index.html"
echo
echo "  Press Ctrl+C to stop both servers."
echo

# Cleanup on exit
cleanup() {
    echo
    echo "Stopping servers..."
    kill $PROXY_PID 2>/dev/null
    fuser -k 8000/tcp 2>/dev/null
    fuser -k 8080/tcp 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start HTTP server in foreground
python3 -m http.server 8000 --directory "$SCRIPT_DIR"

