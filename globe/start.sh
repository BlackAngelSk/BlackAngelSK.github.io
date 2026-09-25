#!/bin/bash
# Globe/Map — launcher (proxy 8080+8443, local file server 8000)
set -e
cd "$(dirname "$0")"

echo ""
echo "  Globe/Map — starting (proxy 8080 + 8443, files 8000)"
echo ""

# macOS: a binary that came out of a downloaded zip carries the quarantine flag
# and refuses to start ("cannot be opened because the developer cannot be
# verified"). Clearing it here makes a plain double-click work.
if [ "$(uname -s)" = "Darwin" ]; then
    xattr -dr com.apple.quarantine . 2>/dev/null || true
    chmod +x ./*.bin ./globe-server ./*.command 2>/dev/null || true
fi

if command -v node >/dev/null 2>&1; then
    exec node globe-server.js "$@"
fi

# No Node.js — use the bundled binary for this machine's architecture
ARCH="$(uname -m)"
case "$ARCH" in
    arm64|aarch64) BINS="globe-server-arm64.bin globe-server globe-server-x64.bin" ;;
    *)             BINS="globe-server-x64.bin globe-server globe-server-arm64.bin" ;;
esac

for bin in $BINS; do
    if [ -f "$bin" ]; then
        chmod +x "$bin" 2>/dev/null || true
        exec "./$bin" "$@"
    fi
done

echo "  ERROR: neither Node.js nor a globe-server binary found."
echo "  Install Node.js from https://nodejs.org and run this again."
exit 1
