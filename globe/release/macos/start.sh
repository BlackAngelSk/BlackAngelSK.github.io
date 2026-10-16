#!/usr/bin/env bash
# ============================================================
#   Globe/Map — start the proxy + local server
#   Linux and macOS.  Plain HTTP — nothing to install, nothing
#   to accept, no certificate warning.
#
#   Usage:  ./start.sh              proxy + local map
#           ./start.sh --proxy      proxy only
#           ./start.sh --no-open    do not open the browser
#           ./start.sh --https      also listen on https (Safari)
# ============================================================

cd "$(dirname "$0")" || exit 1
OS="$(uname -s)"

# macOS: a file that came out of a downloaded zip carries the quarantine flag,
# and Gatekeeper refuses to run it until the flag is cleared.
if [ "$OS" = "Darwin" ]; then
    xattr -dr com.apple.quarantine . 2>/dev/null || true
    chmod +x ./globe-server ./globe-server-*.bin ./start.sh ./*.command 2>/dev/null || true
fi

# ── 1. Node.js — the normal path ───────────────────────────
if command -v node >/dev/null 2>&1; then
    exec node globe-server.js "$@"
fi

# ── 2. No Node.js — use the binary built for this machine ──
case "$(uname -m)" in
    arm64|aarch64) BINS="globe-server-arm64.bin globe-server" ;;
    *)             BINS="globe-server globe-server-x64.bin" ;;
esac

for bin in $BINS; do
    if [ -f "$bin" ]; then
        chmod +x "$bin" 2>/dev/null || true
        exec "./$bin" "$@"
    fi
done

# ── 3. Nothing to run ──────────────────────────────────────
echo ""
echo "  ERROR: neither Node.js nor a globe-server binary was found in:"
echo "    $(pwd)"
echo ""
echo "  Fix it with either of these:"
echo "    * install Node.js  →  https://nodejs.org   then run this file again"
echo "    * or download the portable build for your OS from"
echo "      https://blackangelsk.github.io/globe/ (globe-linux.zip, globe-macos.zip,"
echo "      globe-windows.zip) and start that instead"
echo ""
exit 1
