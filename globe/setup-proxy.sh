#!/bin/bash
# ============================================
#   Map Proxy - Auto Setup & Launcher
# ============================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  Map Proxy - Auto Setup & Launcher${NC}"
echo -e "${CYAN}============================================${NC}"
echo

# ── Step 1: Check for Node.js ──────────────────
echo -e "[1/4] Checking for Node.js..."

install_node_linux() {
    echo -e "  ${YELLOW}Node.js not found. Installing...${NC}"
    
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        echo "  Using apt (Debian/Ubuntu)..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif command -v dnf &> /dev/null; then
        echo "  Using dnf (Fedora/RHEL)..."
        sudo dnf install -y nodejs
    elif command -v yum &> /dev/null; then
        echo "  Using yum (CentOS/RHEL)..."
        curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
        sudo yum install -y nodejs
    elif command -v pacman &> /dev/null; then
        echo "  Using pacman (Arch)..."
        sudo pacman -S --noconfirm nodejs npm
    elif command -v apk &> /dev/null; then
        echo "  Using apk (Alpine)..."
        sudo apk add nodejs npm
    else
        echo -e "  ${RED}ERROR: No supported package manager found.${NC}"
        echo "  Please install Node.js manually from: https://nodejs.org"
        exit 1
    fi
}

install_node_mac() {
    echo -e "  ${YELLOW}Node.js not found. Installing...${NC}"
    
    if command -v brew &> /dev/null; then
        echo "  Using Homebrew..."
        brew install node@20
    else
        echo "  Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        brew install node@20
    fi
}

if command -v node &> /dev/null; then
    NODE_VER=$(node --version)
    echo -e "  ${GREEN}Node.js found: ${NODE_VER}${NC}"
else
    # Detect OS
    case "$(uname -s)" in
        Linux*)  install_node_linux ;;
        Darwin*) install_node_mac ;;
        *)
            echo -e "  ${RED}ERROR: Unknown OS. Please install Node.js from https://nodejs.org${NC}"
            exit 1
            ;;
    esac
    
    # Verify
    if ! command -v node &> /dev/null; then
        echo -e "  ${RED}ERROR: Node.js installation failed.${NC}"
        echo "  Please install manually from: https://nodejs.org"
        exit 1
    fi
    NODE_VER=$(node --version)
    echo -e "  ${GREEN}Node.js installed: ${NODE_VER}${NC}"
fi
echo

# ── Step 2: Check for proxy.js ──────────────────
echo -e "[2/4] Checking for proxy script..."

if [ ! -f "$SCRIPT_DIR/proxy.js" ]; then
    echo "  proxy.js not found locally. Downloading from GitHub..."
    REPO_RAW="https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/main/globe/proxy.js"
    curl -fsSL "$REPO_RAW" -o "$SCRIPT_DIR/proxy.js"
    if [ $? -ne 0 ]; then
        echo -e "  ${RED}ERROR: Failed to download proxy.js${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}Downloaded proxy.js successfully.${NC}"
else
    echo -e "  ${GREEN}proxy.js found.${NC}"
fi
echo

# ── Step 3: Check port 8080 ─────────────────────
echo -e "[3/4] Checking port 8080..."

if lsof -i :8080 -t &> /dev/null 2>&1 || ss -tlnp 2>/dev/null | grep -q ':8080'; then
    echo "  Port 8080 is already in use. Attempting to free it..."
    PID=$(lsof -i :8080 -t 2>/dev/null | head -1)
    if [ -n "$PID" ]; then
        kill -9 "$PID" 2>/dev/null
        sleep 1
    fi
fi
echo

# ── Step 4: Start the proxy ─────────────────────
echo -e "[4/4] Starting KML proxy on port 8080..."
echo
echo -e "  ${GREEN}============================================${NC}"
echo -e "  ${GREEN}Proxy is running at: http://localhost:8080${NC}"
echo -e "  ${GREEN}============================================${NC}"
echo
echo "  Use this URL in the map's Import > URL tab:"
echo "  http://localhost:8080/kml?mid=YOUR_MAP_ID"
echo
echo "  Press Ctrl+C to stop."
echo

node "$SCRIPT_DIR/proxy.js"