#!/usr/bin/env node
/**
 * start.js — Cross-platform launcher for the KML proxy + map HTTP server.
 * Works on Windows, Linux, and macOS without any OS-specific commands.
 *
 * Usage:
 *   node start.js              Start both proxy (8080) + HTTP server (8000)
 *   node start.js --proxy      Start only the proxy (8080)
 *   node start.js --port 3000  Use a custom HTTP server port
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// ── Configuration ──────────────────────────────────────────────
const PROXY_PORT = 8080;
const SCRIPT_DIR = __dirname;
const PROXY_SCRIPT = path.join(SCRIPT_DIR, 'proxy.js');

// ── Parse arguments ────────────────────────────────────────────
const args = process.argv.slice(2);
let HTTP_PORT = 8000;
let proxyOnly = false;

for (let i = 0; i < args.length; i++) {
    if (args[i] === '--proxy') {
        proxyOnly = true;
    } else if (args[i] === '--port' && args[i + 1]) {
        HTTP_PORT = parseInt(args[i + 1], 10);
        i++;
    } else if (args[i] === '--help' || args[i] === '-h') {
        console.log(`
  start.js — Cross-platform launcher

  Usage:
    node start.js              Start proxy + HTTP server
    node start.js --proxy      Start only proxy
    node start.js --port 3000  Custom HTTP port
    node start.js --help       Show this help
`);
        process.exit(0);
    }
}

// ── Helpers ────────────────────────────────────────────────────
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

/** Kill a process on a given port (cross-platform). */
function killPort(port) {
    try {
        const isWin = process.platform === 'win32';
        if (isWin) {
            const result = execSync(`netstat -ano | findstr ":${port}" | findstr "LISTENING"`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
            const lines = result.trim().split('\n').filter(Boolean);
            for (const line of lines) {
                const parts = line.trim().split(/\s+/);
                const pid = parts[parts.length - 1];
                if (pid && pid !== '0') {
                    execSync(`taskkill /PID ${pid} /F`, { stdio: 'pipe' });
                    log(C.yellow, `  Killed PID ${pid} on port ${port}`);
                }
            }
        } else {
            // Linux & macOS — use lsof (works on both)
            try {
                const result = execSync(`lsof -ti :${port}`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
                const pids = result.trim().split('\n').filter(Boolean);
                for (const pid of pids) {
                    execSync(`kill -9 ${pid}`, { stdio: 'pipe' });
                    log(C.yellow, `  Killed PID ${pid} on port ${port}`);
                }
            } catch {
                // No process on that port — fine
            }
        }
    } catch {
        // Port not in use or kill failed — not critical
    }
}

// ── MIME types for the HTTP server ─────────────────────────────
const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.css':  'text/css; charset=utf-8',
    '.js':   'application/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.xml':  'text/xml; charset=utf-8',
    '.png':  'image/png',
    '.jpg':  'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif':  'image/gif',
    '.svg':  'image/svg+xml',
    '.ico':  'image/x-icon',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.ttf':  'font/ttf',
    '.mp4':  'video/mp4',
    '.webm': 'video/webm',
    '.kml':  'application/vnd.google-earth.kml+xml',
};

// ── Start HTTP server (for the map files) ──────────────────────
function startHTTPServer(port) {
    const server = http.createServer((req, res) => {
        let urlPath = new URL(req.url, 'http://localhost').pathname;
        if (urlPath === '/') urlPath = '/index.html';

        const filePath = path.join(SCRIPT_DIR, urlPath);

        // Security: prevent directory traversal
        if (!filePath.startsWith(SCRIPT_DIR)) {
            res.writeHead(403); res.end('Forbidden');
            return;
        }

        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end('Not Found');
                return;
            }
            const ext = path.extname(filePath).toLowerCase();
            const contentType = MIME[ext] || 'application/octet-stream';
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(data);
        });
    });

    server.listen(port, '0.0.0.0', () => {
        log(C.green, `  HTTP server running at: http://localhost:${port}`);
        log(C.reset,  `    Map:   http://localhost:${port}/map.html`);
        log(C.reset,  `    Globe: http://localhost:${port}/index.html`);
    });

    server.on('error', (e) => {
        if (e.code === 'EADDRINUSE') {
            err(`Port ${port} is already in use.`);
        } else {
            err(`HTTP server error: ${e.message}`);
        }
        process.exit(1);
    });

    return server;
}

// ── Start the proxy (spawns proxy.js in a child process) ───────
function startProxy() {
    if (!fs.existsSync(PROXY_SCRIPT)) {
        err(`proxy.js not found at ${PROXY_SCRIPT}`);
        err('Download it from: https://github.com/BlackAngelSk/BlackAngelSK.github.io/main/globe/proxy.js');
        process.exit(1);
    }

    const { spawn } = require('child_process');
    const child = spawn(process.execPath, [PROXY_SCRIPT], {
        cwd: SCRIPT_DIR,
        stdio: ['ignore', 'inherit', 'inherit'],
        detached: false,
    });

    child.on('error', (e) => {
        err(`Proxy failed to start: ${e.message}`);
    });

    child.on('exit', (code) => {
        if (code !== 0) {
            log(C.yellow, `  Proxy exited with code ${code}`);
        }
    });

    return child;
}

// ── Main ───────────────────────────────────────────────────────
log(C.cyan,  '============================================');
log(C.cyan,  '  Interactive Map — Cross-Platform Launcher');
log(C.cyan,  '============================================');
log(C.reset, `  Platform: ${process.platform} (${process.arch})`);
log(C.reset, `  Node.js:  ${process.version}`);
log(C.reset, '');

// Kill existing processes on our ports
log(C.yellow, 'Checking for existing servers...');
killPort(PROXY_PORT);
if (!proxyOnly) killPort(HTTP_PORT);
log(C.reset, '');

// Start proxy
log(C.green, `[1] Starting KML proxy on port ${PROXY_PORT}...`);
const proxyProcess = startProxy();
log(C.reset, '');

// Start HTTP server
if (!proxyOnly) {
    log(C.green, `[2] Starting HTTP server on port ${HTTP_PORT}...`);
    const httpServer = startHTTPServer(HTTP_PORT);
    log(C.reset, '');
}

log(C.green, '  ============================================');
log(C.green, `  Servers are running!`);
log(C.green, `  Proxy:  http://localhost:${PROXY_PORT}`);
if (!proxyOnly) {
    log(C.green, `  Map:    http://localhost:${HTTP_PORT}/map.html`);
    log(C.green, `  Globe:  http://localhost:${HTTP_PORT}/index.html`);
}
log(C.green, '  ============================================');
log(C.reset, '');

// Cleanup on exit
function cleanup() {
    log(C.yellow, '\nStopping servers...');
    if (proxyProcess && !proxyProcess.killed) {
        proxyProcess.kill('SIGTERM');
    }
    process.exit(0);
}

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

// Windows: handle Ctrl+C
if (process.platform === 'win32') {
    const readline = require('readline');
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.on('SIGINT', () => cleanup());
}