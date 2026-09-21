#!/bin/bash
# Globe/Map — launcher (proxy 8080+8443 + local file server 8000)
set -e
cd "$(dirname "$0")"

echo ""
echo "  Globe/Map — starting (proxy 8080 + 8443, files 8000)"
echo ""

if command -v node >/dev/null 2>&1; then
    exec node globe-server.js "$@"
fi

# No Node.js — use a bundled binary if this folder has one
for bin in globe-server globe-server-arm64.bin globe-server-x64.bin; do
    if [ -x "$bin" ]; then
        exec "./$bin" "$@"
    fi
done

echo "  ERROR: neither Node.js nor a globe-server binary found."
echo "  Install Node.js from https://nodejs.org and run this again."
exit 1
