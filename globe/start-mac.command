#!/bin/bash
# ============================================================
#   Globe/Map — macOS double-click launcher
#   Finder runs .command files in Terminal; this just prepares
#   the folder (Gatekeeper quarantine) and calls start.sh.
# ============================================================

cd "$(dirname "$0")" || exit 1

xattr -dr com.apple.quarantine . 2>/dev/null || true
chmod +x ./start.sh ./globe-server ./globe-server-*.bin 2>/dev/null || true

./start.sh "$@"

echo
echo "The server has stopped. Press Enter to close this window."
read -r _
