#!/usr/bin/env node
/**
 * setup-proxy.js — Cross-platform proxy setup & launcher.
 * Works on Windows, Linux, and macOS without any OS-specific commands.
 *
 * Usage:
 *   node setup-proxy.js        Auto-install Node.js if needed, check port, start proxy
 *   node setup-proxy.js --help Show help
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

const SCRIPT_DIR = __dirname;
const PROXY_SCRIPT = path.join(SCRIPT_DIR, 'proxy.js');
const PROXY_PORT = 8080;
const REPO_RAW = 'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/proxy.js';
const CERT_RAW = 'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/.proxy-cert.pem';
const KEY_RAW  = 'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/.proxy-key.pem';

// ── Colors ─────────────────────────────────────────────────────
const C = {
    red:    '\x1b[31m',
    green:  '\x1b[32m',
    yellow: '\x1b[33m',
    cyan:   '\x1b[36m',
    reset:  '\x1b[0m',
    bold:   '\x1b[1m',
};

function log(color, msg) { console.log(`${color}${msg}${C.reset}`); }
function err(msg) { log(C.red, `  ERROR: ${msg}`); }

/** Run a command, return output or null on failure */
function run(cmd, opts) {
    try { return execSync(cmd, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'], ...opts }); }
    catch { return null; }
}

/** Check if a command exists */
function cmdExists(cmd) {
    const isWin = process.platform === 'win32';
    const check = isWin ? `where ${cmd}` : `command -v ${cmd}`;
    try { execSync(check, { stdio: 'pipe' }); return true; }
    catch { return false; }
}

/** Kill process on a given port */
function killPort(port) {
    try {
        if (process.platform === 'win32') {
            const result = run(`netstat -ano | findstr ":${port}" | findstr "LISTENING"`);
            if (result) {
                result.trim().split('\n').filter(Boolean).forEach(line => {
                    const pid = line.trim().split(/\s+/).pop();
                    if (pid && pid !== '0') run(`taskkill /PID ${pid} /F`);
                });
            }
        } else {
            const result = run(`lsof -ti :${port}`);
            if (result && result.trim()) {
                result.trim().split('\n').filter(Boolean).forEach(pid => {
                    run(`kill -9 ${pid}`);
                });
            }
        }
    } catch { /* ignore */ }
}

/** Download a file */
function download(url, dest) {
    if (cmdExists('curl')) {
        const rc = run(`curl -fsSL -o "${dest}" "${url}"`);
        if (rc !== null) return true;
    }
    if (cmdExists('wget')) {
        const rc = run(`wget -q "${url}" -O "${dest}"`);
        if (rc !== null) return true;
    }
    if (process.platform === 'win32') {
        try {
            execSync(
                `powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '${url}' -OutFile '${dest}' -UseBasicParsing"`,
                { stdio: 'pipe' }
            );
            return true;
        } catch { return false; }
    }
    return false;
}

// ── Main ───────────────────────────────────────────────────────
log(C.cyan, '============================================');
log(C.cyan, '  Map Proxy — Cross-Platform Setup & Launcher');
log(C.cyan, '============================================');
console.log();

const isWin = process.platform === 'win32';
const isMac = process.platform === 'darwin';
const isLinux = process.platform === 'linux';

// ── Step 1: Check Node.js ─────────────────────────────────────
log(C.cyan, '[1/4] Checking for Node.js...');
const nodeVersion = run('node --version');
if (nodeVersion && nodeVersion.trim()) {
    log(C.green, `  Node.js found: ${nodeVersion.trim()}`);
} else {
    log(C.yellow, '  Node.js not found. Attempting to install...');

    if (isWin) {
        // Windows: try winget first
        if (cmdExists('winget')) {
            log(C.cyan, '  Using winget...');
            run('winget install OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements');
        } else {
            err('winget not found. Please install Node.js from https://nodejs.org');
            process.exit(1);
        }
    } else if (isMac) {
        // macOS: use Homebrew
        if (cmdExists('brew')) {
            log(C.cyan, '  Using Homebrew...');
            run('brew install node@20');
        } else {
            log(C.cyan, '  Installing Homebrew first...');
            run('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"');
            run('brew install node@20');
        }
    } else if (isLinux) {
        // Linux: detect package manager
        if (cmdExists('apt-get')) {
            log(C.cyan, '  Using apt (Debian/Ubuntu)...');
            run('curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -');
            run('sudo apt-get install -y nodejs');
        } else if (cmdExists('dnf')) {
            log(C.cyan, '  Using dnf (Fedora/RHEL)...');
            run('sudo dnf install -y nodejs');
        } else if (cmdExists('pacman')) {
            log(C.cyan, '  Using pacman (Arch)...');
            run('sudo pacman -S --noconfirm nodejs npm');
        } else if (cmdExists('apk')) {
            log(C.cyan, '  Using apk (Alpine)...');
            run('sudo apk add nodejs npm');
        } else {
            err('No supported package manager found. Install Node.js from https://nodejs.org');
            process.exit(1);
        }
    }

    // Verify
    const newVer = run('node --version');
    if (!newVer) {
        err('Node.js installation failed. Install manually from https://nodejs.org');
        process.exit(1);
    }
    log(C.green, `  Node.js installed: ${newVer.trim()}`);
}
console.log();

// ── Step 2: Check proxy.js ─────────────────────────────────────
log(C.cyan, '[2/4] Checking for proxy script...');
log(C.reset, `  Script directory: ${SCRIPT_DIR}`);
log(C.reset, `  Looking for: ${PROXY_SCRIPT}`);

if (fs.existsSync(PROXY_SCRIPT)) {
    log(C.green, '  proxy.js found.');
} else {
    log(C.yellow, '  proxy.js not found locally. Attempting download...');
    log(C.reset, `  Download URL: ${REPO_RAW}`);

    const ok = download(REPO_RAW, PROXY_SCRIPT);
    if (!ok) {
        err('Failed to download proxy.js');
        err(`Download manually from: ${REPO_RAW}`);
        err(`Save to: ${PROXY_SCRIPT}`);
        process.exit(1);
    }

    // Verify it is really JavaScript — a 404 saves an HTML/JSON page that
    // node refuses to run.
    const stat = fs.statSync(PROXY_SCRIPT);
    log(C.reset, `  Downloaded file size: ${stat.size} bytes`);
    const head = fs.readFileSync(PROXY_SCRIPT, 'utf8');
    if (stat.size < 200 || !head.includes('PROXY_VERSION')) {
        err('Downloaded file is not the proxy script (404 page?).');
        err('Check your internet connection and try again.');
        try { fs.unlinkSync(PROXY_SCRIPT); } catch {}
        process.exit(1);
    }
    log(C.green, '  Downloaded proxy.js successfully.');
}
console.log();

// ── Step 3: Check port ─────────────────────────────────────────
log(C.cyan, `[3/4] Checking port ${PROXY_PORT}...`);
killPort(PROXY_PORT);
console.log();

// ── Step 4: Start proxy ────────────────────────────────────────
log(C.green, `[4/4] Starting KML proxy on port ${PROXY_PORT}...`);
console.log();
log(C.green, '  ============================================');
log(C.green, `  Proxy:  http://localhost:${PROXY_PORT}/ping`);
log(C.green, '  Proxy:  https://localhost:8443/ping');
log(C.green, '  ============================================');
console.log();
log(C.reset, '  If the map page is HTTPS (blackangelsk.github.io), open');
log(C.reset, '  https://localhost:8443/ping once and accept the self-signed');
log(C.reset, '  certificate (Advanced -> Proceed), then reload the map.');
console.log();

if (isWin) {
    log(C.reset, '  Close this window or press Ctrl+C to stop.');
} else {
    log(C.reset, '  Press Ctrl+C to stop.');
}
console.log();

// Best-effort: fetch the HTTPS certificate next to proxy.js
for (const [url, file] of [[CERT_RAW, '.proxy-cert.pem'], [KEY_RAW, '.proxy-key.pem']]) {
    const dest = path.join(SCRIPT_DIR, file);
    if (!fs.existsSync(dest)) {
        if (download(url, dest)) log(C.dim, `  Certificate: ${file}`);
    }
}

// Start the proxy
const proxy = spawn(process.execPath, [PROXY_SCRIPT], {
    cwd: SCRIPT_DIR,
    stdio: ['ignore', 'inherit', 'inherit'],
});

proxy.on('error', (e) => { err(`Proxy failed to start: ${e.message}`); });
proxy.on('exit', (code) => {
    if (code !== 0) log(C.yellow, `Proxy exited with code ${code}`);
});

// Cleanup on exit
function cleanup() {
    console.log();
    log(C.yellow, 'Stopping proxy...');
    if (!proxy.killed) proxy.kill('SIGTERM');
    process.exit(0);
}

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

if (isWin) {
    const readline = require('readline');
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.on('SIGINT', () => cleanup());
}