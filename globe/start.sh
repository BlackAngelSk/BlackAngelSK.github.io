#!/bin/bash
# Globe/Map — Terminal Startup
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Globe/Map Server — Starting...         ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Kill existing servers
for port in 8000 8080; do
    pids=$(lsof -ti :$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Killing processes on port $port..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
done
sleep 1

echo "Starting CORS proxy on port 8080..."
node proxy.js &
PROXY_PID=$!

sleep 1

echo "Starting HTTP server on port 8000..."
echo ""
echo "  Map:   http://localhost:8000/map.html"
echo "  Globe: http://localhost:8000/index.html"
echo ""
echo "  Press Ctrl+C to stop both servers."
echo ""

# Open browser after 2s
(sleep 2 && xdg-open http://localhost:8000 2>/dev/null || open http://localhost:8000 2>/dev/null || true) &

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $PROXY_PID 2>/dev/null || true
    for port in 8000 8080; do
        pids=$(lsof -ti :$port 2>/dev/null || true)
        [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null || true
    done
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start HTTP server in foreground
python3 -m http.server 8000 --directory "$SCRIPT_DIR"
