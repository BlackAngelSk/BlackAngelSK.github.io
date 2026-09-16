#!/bin/bash
echo "============================================"
echo "  Interactive Map - Starting Servers"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill any existing processes on our ports (cross-platform: works on Linux + macOS)
echo "Checking for existing servers on ports 8000/8080..."

# lsof -ti works on both Linux and macOS
kill_port() {
    local port=$1
    if command -v lsof &> /dev/null; then
        local pids=$(lsof -ti :$port 2>/dev/null)
        if [ -n "$pids" ]; then
            echo "  Killing processes on port $port: $pids"
            echo "$pids" | xargs kill -9 2>/dev/null
        fi
    fi
    # Also try fuser on Linux (if available)
    if command -v fuser &> /dev/null; then
        fuser -k $port/tcp 2>/dev/null
    fi
}

kill_port 8000
kill_port 8080
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
    kill_port 8000
    kill_port 8080
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start HTTP server in foreground (use python3 if available, fallback to python)
if command -v python3 &> /dev/null; then
    python3 -m http.server 8000 --directory "$SCRIPT_DIR"
elif command -v python &> /dev/null; then
    python -m http.server 8000 --directory "$SCRIPT_DIR"
else
    echo "ERROR: Python not found. Install Python 3 and try again."
    exit 1
fi