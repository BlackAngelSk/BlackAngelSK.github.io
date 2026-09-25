#!/bin/bash
# Double-clickable launcher for macOS (Finder runs .command files in Terminal).
cd "$(dirname "$0")"
xattr -dr com.apple.quarantine . 2>/dev/null || true
chmod +x ./*.bin ./globe-server ./start.sh 2>/dev/null || true
exec ./start.sh "$@"
