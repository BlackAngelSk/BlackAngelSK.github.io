#!/usr/bin/env node
/**
 * globe-server.js — All-in-one server for the Globe/Map project.
 * Embeds both the CORS proxy (port 8080) and the HTTP file server (port 8000).
 * No external dependencies beyond Node.js stdlib.
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { spawn } = require('child_process');
const readline = require('readline');

// ── Configuration ──────────────────────────────────────────────
const PROXY_PORT = 8080;
const HTTP_PORT = 8000;
const PROXY_VERSION = 2;

// ── Parse CLI arguments ────────────────────────────────────────
const args = process.argv.slice(2);
let customHttpPort = HTTP_PORT;
let proxyOnly = false;

for (let i = 2; i < args.length; i++) {
    if (args[i] === '--proxy') proxyOnly = true;
    else if (args[i] === '--port' && args[i + 1]) { customHttpPort = parseInt(args[i + 1], 10); i++; }
    else if (args[i] === '--help' || args[i] === '-h') {
        console.log(`
  globe-server.js — All-in-one Globe/Map server

  Usage:
    globe-server.js                  Start proxy + HTTP server
    globe-server.js --proxy          Start only proxy
    globe-server.js --port 3000      Custom HTTP port
    globe-server.js --help           Show this help
`);
        process.exit(0);
    }
}

// ── Colors ─────────────────────────────────────────────────────
const C = {
    red: '\x1b[31m', green: '\x1b[32m', yellow: '\x1b[33m',
    cyan: '\x1b[36m', reset: '\x1b[0m', bold: '\x1b[1m',
};
function log(color, msg) { console.log(`${color}${msg}${C.reset}`); }

// ── Kill process on a port ─────────────────────────────────────
function killPort(port) {
    try {
        if (process.platform === 'win32') {
            const r = execSync(`netstat -ano | findstr ":${port}" | findstr "LISTENING"`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
            r.trim().split('\n').filter(Boolean).forEach(line => {
                const pid = line.trim().split(/\s+/).pop();
                if (pid && pid !== '0') execSync(`taskkill /PID ${pid} /F`, { stdio: 'pipe' });
            });
        } else {
            const r = execSync(`lsof -ti :${port}`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
            r.trim().split('\n').filter(Boolean).forEach(pid => {
                execSync(`kill -9 ${pid}`, { stdio: 'pipe' });
            });
        }
    } catch { /* port not in use */ }
}

// ══════════════════════════════════════════════════════════════
//  CORS PROXY (from proxy.js)
// ══════════════════════════════════════════════════════════════

function proxyFetch(url, maxRedirects) {
    maxRedirects = maxRedirects || 5;
    return new Promise(function (resolve, reject) {
        if (maxRedirects <= 0) return reject(new Error('Too many redirects'));
        var mod = url.startsWith('https') ? https : http;
        var req = mod.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' } }, function (res) {
            if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                var redirectUrl = res.headers.location;
                if (redirectUrl.startsWith('/')) {
                    var base = new URL(url);
                    redirectUrl = base.origin + redirectUrl;
                }
                res.resume();
                return proxyFetch(redirectUrl, maxRedirects - 1).then(resolve).catch(reject);
            }
            var chunks = [];
            res.on('data', function (c) { chunks.push(c); });
            res.on('end', function () {
                resolve({ statusCode: res.statusCode, headers: res.headers, data: Buffer.concat(chunks) });
            });
            res.on('error', reject);
        });
        req.on('error', reject);
        req.setTimeout(15000, function () { req.destroy(); reject(new Error('Request timed out')); });
    });
}

function detectContentType(targetUrl) {
    if (/google\.com\/maps\/d\//i.test(targetUrl)) return 'application/vnd.google-earth.kml+xml; charset=utf-8';
    if (/yandex\.(ru|com).*maps/i.test(targetUrl)) return 'text/html; charset=utf-8';
    return null;
}

function startProxyServer() {
    const server = http.createServer(function (req, res) {
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
        if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

        var parsedUrl = new URL(req.url, 'http://localhost');
        var targetUrl = parsedUrl.searchParams.get('url');

        if (!targetUrl) {
            res.writeHead(400, { 'Content-Type': 'text/plain' });
            res.end('Missing ?url= parameter.\n\nUsage:\n  /proxy?url=ENCODED_URL\n');
            return;
        }
        if (!targetUrl.startsWith('https://')) {
            res.writeHead(400, { 'Content-Type': 'text/plain' });
            res.end('Only HTTPS URLs supported.');
            return;
        }

        console.log('Proxying:', targetUrl.slice(0, 150));

        proxyFetch(targetUrl).then(function (r) {
            var ct = detectContentType(targetUrl) || r.headers['content-type'] || 'application/octet-stream';
            res.writeHead(r.statusCode, { 'Content-Type': ct, 'Content-Length': r.data.length, 'Access-Control-Allow-Origin': '*' });
            res.end(r.data);
        }).catch(function (err) {
            console.error('Proxy error:', err.message);
            res.writeHead(502, { 'Content-Type': 'text/plain' });
            res.end('Proxy fetch failed: ' + err.message);
        });
    });

    server.listen(PROXY_PORT, () => {
        log(C.green, `  CORS Proxy v${PROXY_VERSION} running at http://localhost:${PROXY_PORT}`);
    });

    server.on('error', (e) => {
        if (e.code === 'EADDRINUSE') log(C.yellow, `  Proxy port ${PROXY_PORT} already in use (continuing)`);
        else log(C.red, `  Proxy error: ${e.message}`);
    });

    return server;
}

// ══════════════════════════════════════════════════════════════
//  HTTP FILE SERVER (from start.js)
// ══════════════════════════════════════════════════════════════

const MIME = {
    '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8',
    '.xml': 'text/xml; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon', '.woff': 'font/woff', '.woff2': 'font/woff2',
    '.ttf': 'font/ttf', '.mp4': 'video/mp4', '.webm': 'video/webm',
    '.kml': 'application/vnd.google-earth.kml+xml',
};

function getScriptDir() {
    // When compiled with pkg, __dirname points to the snapshot filesystem.
    // Use process.cwd() or the executable's directory as fallback.
    if (__dirname === '/' || __dirname.startsWith('/snapshot')) {
        return process.cwd();
    }
    return __dirname;
}

function startHTTPServer(port) {
    const scriptDir = getScriptDir();
    const server = http.createServer((req, res) => {
        let urlPath = new URL(req.url, 'http://localhost').pathname;
        if (urlPath === '/') urlPath = '/index.html';

        const filePath = path.join(scriptDir, urlPath);
        if (!filePath.startsWith(scriptDir)) {
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
        log(C.green, `  HTTP server running at http://localhost:${port}`);
        log(C.reset,  `    Map:   http://localhost:${port}/map.html`);
        log(C.reset,  `    Globe: http://localhost:${port}/index.html`);
    });

    server.on('error', (e) => {
        if (e.code === 'EADDRINUSE') log(C.yellow, `  HTTP port ${port} already in use`);
        else log(C.red, `  HTTP server error: ${e.message}`);
    });

    return server;
}

// ══════════════════════════════════════════════════════════════
//  MAIN
// ══════════════════════════════════════════════════════════════

log(C.cyan,  '============================================');
log(C.cyan,  '  Globe/Map Server — All-in-One');
log(C.cyan,  '============================================');
log(C.reset, `  Platform: ${process.platform} (${process.arch})`);
log(C.reset, `  Node.js:  ${process.version}`);
log(C.reset, '');

log(C.yellow, 'Checking for existing servers...');
killPort(PROXY_PORT);
if (!proxyOnly) killPort(customHttpPort);
log(C.reset, '');

// Start proxy
log(C.green, `[1] Starting CORS proxy on port ${PROXY_PORT}...`);
const proxyServer = startProxyServer();
log(C.reset, '');

// Start HTTP server
let httpServer = null;
if (!proxyOnly) {
    log(C.green, `[2] Starting HTTP server on port ${customHttpPort}...`);
    httpServer = startHTTPServer(customHttpPort);
    log(C.reset, '');
}

log(C.green, '  ============================================');
log(C.green, `  Servers are running!`);
log(C.green, `  Proxy:  http://localhost:${PROXY_PORT}`);
if (!proxyOnly) {
    log(C.green, `  Map:    http://localhost:${customHttpPort}/map.html`);
    log(C.green, `  Globe:  http://localhost:${customHttpPort}/index.html`);
}
log(C.green, '  ============================================');
log(C.reset, '');

// Try to open browser
if (!proxyOnly) {
    const openUrl = `http://localhost:${customHttpPort}`;
    setTimeout(() => {
        try {
            if (process.platform === 'darwin') execSync(`open "${openUrl}"`, { stdio: 'pipe' });
            else if (process.platform === 'linux') execSync(`xdg-open "${openUrl}"`, { stdio: 'pipe' });
            else if (process.platform === 'win32') execSync(`start "" "${openUrl}"`, { stdio: 'pipe' });
        } catch { /* browser open failed — not critical */ }
    }, 1000);
}

// Cleanup on exit
function cleanup() {
    log(C.yellow, '\nStopping servers...');
    if (proxyServer && !proxyServer.closed) proxyServer.close();
    if (httpServer && !httpServer.closed) httpServer.close();
    process.exit(0);
}

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);
if (process.platform === 'win32') {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.on('SIGINT', () => cleanup());
}
