#!/bin/bash
# ============================================
#   Map Proxy - Auto Setup & Launcher
# ============================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Always use raw.githubusercontent.com + the master branch: github.com/.../main/...
# returns 404 HTML, and saving that as proxy.js makes node die with a syntax error.
REPO_RAW="https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/proxy.js"
CERT_RAW="https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/.proxy-cert.pem"
KEY_RAW="https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/.proxy-key.pem"

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  Map Proxy - Auto Setup & Launcher${NC}"
echo -e "${CYAN}============================================${NC}"
echo

# ── macOS preparation ──────────────────────────
if [ "$(uname -s)" = "Darwin" ]; then
    # A file that came out of a downloaded zip is quarantined and Gatekeeper
    # refuses to run it until the flag is cleared.
    xattr -dr com.apple.quarantine "$SCRIPT_DIR" 2>/dev/null || true
    chmod +x "$SCRIPT_DIR"/*.bin "$SCRIPT_DIR"/globe-server "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.command 2>/dev/null || true
    OSSL_VER="$(openssl version 2>/dev/null || echo 'not found')"
    echo -e "  ${CYAN}macOS openssl: ${OSSL_VER}${NC}"
    case "$OSSL_VER" in
        LibreSSL\ 2.*|LibreSSL\ 3.0.*)
            echo -e "  ${YELLOW}Note: this openssl has no -addext, so it cannot put subjectAltName${NC}"
            echo -e "  ${YELLOW}into a self-signed certificate. The proxy detects that and installs${NC}"
            echo -e "  ${YELLOW}its built-in certificate instead — just leave .proxy-cert.pem and${NC}"
            echo -e "  ${YELLOW}.proxy-key.pem in this folder.${NC}" ;;
    esac
    echo
fi

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
echo -e "  Script directory: ${SCRIPT_DIR}"
echo -e "  Looking for: ${SCRIPT_DIR}/proxy.js"

if [ -f "$SCRIPT_DIR/proxy.js" ]; then
    echo -e "  ${GREEN}proxy.js found.${NC}"
else
    echo -e "  proxy.js not found locally. Attempting download..."
    echo -e "  Download URL: $REPO_RAW"

    # Try curl first, then wget
    DOWNLOADED=0
    if command -v curl &> /dev/null; then
        echo "  Downloading using curl..."
        curl -fsSL "$REPO_RAW" -o "$SCRIPT_DIR/proxy.js"
        DOWNLOADED=$?
    elif command -v wget &> /dev/null; then
        echo "  Downloading using wget..."
        wget -q "$REPO_RAW" -O "$SCRIPT_DIR/proxy.js"
        DOWNLOADED=$?
    else
        echo -e "  ${RED}ERROR: Neither curl nor wget found.${NC}"
        echo "  Please install curl or wget, or download proxy.js manually."
        exit 1
    fi

    if [ $DOWNLOADED -ne 0 ]; then
        echo -e "  ${RED}ERROR: Failed to download proxy.js${NC}"
        echo "  Please download manually from:"
        echo "  $REPO_RAW"
        echo "  And save it to: $SCRIPT_DIR/proxy.js"
        rm -f "$SCRIPT_DIR/proxy.js" 2>/dev/null
        exit 1
    fi

    # Verify the download is really JavaScript (a 404 from GitHub would save
    # an HTML/JSON page that node cannot run)
    FILE_SIZE=$(wc -c < "$SCRIPT_DIR/proxy.js" 2>/dev/null || echo 0)
    echo -e "  Downloaded file size: ${FILE_SIZE} bytes"
    if [ "$FILE_SIZE" -lt 200 ] || ! grep -q "PROXY_VERSION" "$SCRIPT_DIR/proxy.js"; then
        echo -e "  ${RED}ERROR: proxy.js looks invalid (not the proxy script).${NC}"
        echo "  Tried: $REPO_RAW"
        echo "  Check your internet connection, or download it manually."
        rm -f "$SCRIPT_DIR/proxy.js" 2>/dev/null
        exit 1
    fi
    echo -e "  ${GREEN}Downloaded proxy.js successfully.${NC}"
fi

# Certificate for the HTTPS listener (needed when the page is served over https)
cert_ok() {
    [ -s "$SCRIPT_DIR/.proxy-cert.pem" ] && [ -s "$SCRIPT_DIR/.proxy-key.pem" ] || return 1
    if command -v node &> /dev/null; then
        node -e '
            const fs = require("fs"), crypto = require("crypto");
            try {
                const x = new crypto.X509Certificate(fs.readFileSync(process.argv[1]));
                process.exit(/DNS:localhost|IP Address:127\.0\.0\.1/.test(String(x.subjectAltName || "")) ? 0 : 1);
            } catch (e) { process.exit(1); }
        ' "$SCRIPT_DIR/.proxy-cert.pem" && return 0
        return 1
    fi
    openssl x509 -in "$SCRIPT_DIR/.proxy-cert.pem" -noout -text 2>/dev/null | grep -q "DNS:localhost"
}

if [ ! -f "$SCRIPT_DIR/.proxy-cert.pem" ] || [ ! -f "$SCRIPT_DIR/.proxy-key.pem" ]; then
    curl -fsSL "$CERT_RAW" -o "$SCRIPT_DIR/.proxy-cert.pem" 2>/dev/null || true
    curl -fsSL "$KEY_RAW"  -o "$SCRIPT_DIR/.proxy-key.pem"  2>/dev/null || true
fi

if cert_ok; then
    echo -e "  ${GREEN}HTTPS certificate ready (.proxy-cert.pem).${NC}"
else
    echo -e "  ${YELLOW}Certificate missing or without subjectAltName — removing it,${NC}"
    echo -e "  ${YELLOW}the proxy will install its built-in one on start-up.${NC}"
    rm -f "$SCRIPT_DIR/.proxy-cert.pem" "$SCRIPT_DIR/.proxy-key.pem"
fi
echo

# ── Step 3: Check the ports ─────────────────────
echo -e "[3/4] Checking ports 8080 and 8443..."
for port in 8080 8443; do
    if command -v lsof &> /dev/null && lsof -nP -iTCP:"$port" -sTCP:LISTEN &> /dev/null; then
        owner=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NR==2{print $1" (PID "$2")"}')
        if curl -fsS --max-time 2 "http://localhost:$port/ping" 2>/dev/null | grep -q pong; then
            echo -e "  Port $port: an older copy of this proxy is running ($owner) — the new one will take it over."
        else
            echo -e "  ${YELLOW}Port $port is used by another program: $owner${NC}"
            echo -e "  ${YELLOW}The proxy will skip that port and keep running. Free it with:${NC}"
            echo "    lsof -ti :$port | xargs kill -9"
        fi
    else
        echo -e "  ${GREEN}Port $port is free.${NC}"
    fi
done
echo

# ── Step 4: Start the proxy ─────────────────────
echo -e "[4/4] Starting KML proxy (HTTP 8080 + HTTPS 8443)..."
echo
echo -e "  ${GREEN}============================================${NC}"
echo -e "  ${GREEN}Proxy:  http://localhost:8080/ping${NC}"
echo -e "  ${GREEN}Proxy:  https://localhost:8443/ping${NC}"
echo -e "  ${GREEN}============================================${NC}"
echo
echo "  If the map page is HTTPS (blackangelsk.github.io), open"
echo "  https://localhost:8443/ping once, click Advanced -> Proceed to accept"
echo "  the self-signed certificate, then reload the map."
echo
echo "  Press Ctrl+C to stop."
echo

node "$SCRIPT_DIR/proxy.js"