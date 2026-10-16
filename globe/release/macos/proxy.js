#!/usr/bin/env node
/**
 * proxy.js — standalone CORS proxy (plain HTTP, no dependencies).
 *
 *   node proxy.js              proxy + static files on http://localhost:8080
 *   node proxy.js --port 9000  custom port
 *   node proxy.js --no-open    do not open the browser
 *   node proxy.js --help       this text
 *
 * HTTP ONLY — no certificate, no warning to click through.
 * Chrome, Edge, Firefox and Brave treat http://localhost as a trustworthy
 * origin, so a page served over https:// (blackangelsk.github.io) may call
 * http://localhost:8080 directly. Safari does not: use globe-server.js with
 * --https there.
 *
 * Self-updates from the GitHub repo on start-up (network is optional — if it
 * cannot reach GitHub, the local copy just runs as-is).
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

const PORT = 8080;
const PROXY_VERSION = 7;
const SCRIPT_DIR = __dirname;
const SELF_PATH = path.join(SCRIPT_DIR, 'proxy.js');
const SELF_URL = 'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/proxy.js';

// ── Arguments ──────────────────────────────────────────────────
const args = process.argv.slice(2);
let port = PORT;
let openBrowser = true;
for (let i = 0; i < args.length; i++) {
    if (args[i] === '--port' && args[i + 1]) { port = parseInt(args[i + 1], 10); i++; }
    else if (args[i] === '--no-open') openBrowser = false;
    else if (args[i] === '--help' || args[i] === '-h') {
        console.log(`
  proxy.js ${PROXY_VERSION} — standalone CORS proxy (plain HTTP)

  Usage:
    node proxy.js               proxy + static files on http://localhost:${PORT}
    node proxy.js --port 9000   custom port
    node proxy.js --no-open     do not open the browser
    node proxy.js --help        this text

  Endpoints:
    /ping                  health check (answers "pong")
    /proxy?url=ENCODED     proxy any HTTPS URL
    /kml?url=ENCODED       same, forced KML content type
    /kml?mid=MY_MAPS_ID    Google My Maps id shortcut
    /*                     static files next to this script
`);
        process.exit(0);
    }
}

// ── Colors ─────────────────────────────────────────────────────
const C = { red: '\x1b[31m', green: '\x1b[32m', yellow: '\x1b[33m', cyan: '\x1b[36m', dim: '\x1b[2m', reset: '\x1b[0m' };
function log(color, msg) { console.log(`${color}${msg}${C.reset}`); }

/* ═══ SELF-UPDATE ═══════════════════════════════════
   Optional: if GitHub has a newer proxy.js, install it and restart. A machine
   with no internet simply keeps the local copy. */
function fetchUrl(url) {
    return new Promise((resolve, reject) => {
        https.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' } }, (res) => {
            if (res.statusCode !== 200) { res.resume(); return reject(new Error('HTTP ' + res.statusCode)); }
            let body = '';
            res.on('data', (c) => body += c);
            res.on('end', () => resolve(body));
        }).on('error', reject);
    });
}

function checkForUpdates(cb) {
    fetchUrl(SELF_URL).then((remote) => {
        const m = remote.match(/PROXY_VERSION\s*=\s*(\d+)/);
        const remoteVer = m ? parseInt(m[1], 10) : 0;
        if (remoteVer > PROXY_VERSION && remote.includes('const http = require')) {
            log(C.yellow, `[proxy] Updating to v${remoteVer} (current v${PROXY_VERSION})…`);
            try {
                fs.writeFileSync(SELF_PATH, remote, 'utf8');
                spawn(process.execPath, [SELF_PATH, ...args], { stdio: 'inherit', detached: true }).unref();
                process.exit(0);
            } catch (e) {
                log(C.yellow, '[proxy] Could not save the update: ' + e.message);
            }
        } else {
            log(C.dim, `[proxy] Up to date (v${PROXY_VERSION})`);
        }
        cb();
    }).catch((e) => {
        log(C.dim, `[proxy] Update check skipped (${e.message})`);
        cb();
    });
}

// ── Proxy core ─────────────────────────────────────────────────
function proxyFetch(url, maxRedirects) {
    maxRedirects = maxRedirects || 5;
    return new Promise(function (resolve, reject) {
        if (maxRedirects <= 0) return reject(new Error('Too many redirects'));
        var mod = url.startsWith('https') ? https : http;
        var req = mod.get(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
                'Accept': 'application/vnd.google-earth.kml+xml,application/xml,text/xml,*/*',
            },
        }, function (res) {
            if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                var redirectUrl = res.headers.location;
                if (redirectUrl.startsWith('/')) redirectUrl = new URL(url).origin + redirectUrl;
                else if (!/^https?:/i.test(redirectUrl)) redirectUrl = new URL(redirectUrl, url).href;
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
        req.setTimeout(20000, function () { req.destroy(); reject(new Error('Request timed out')); });
    });
}

const MIME = {
    '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8',
    '.xml': 'text/xml; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon', '.woff': 'font/woff', '.woff2': 'font/woff2',
    '.ttf': 'font/ttf', '.kml': 'application/vnd.google-earth.kml+xml',
    '.zip': 'application/zip', '.txt': 'text/plain; charset=utf-8',
};

function handler(req, res) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.setHeader('Access-Control-Allow-Private-Network', 'true');

    if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

    var parsedUrl = new URL(req.url, 'http://localhost');
    var pathname = parsedUrl.pathname;
    var targetUrl = parsedUrl.searchParams.get('url');
    var mid = parsedUrl.searchParams.get('mid');

    if (pathname === '/ping' || pathname === '/health') {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('pong');
        return;
    }

    if (mid) targetUrl = 'https://www.google.com/maps/d/u/0/kml?mid=' + mid + '&forcekml=1';

    if (targetUrl) {
        if (!/^https:\/\//i.test(targetUrl)) {
            res.writeHead(400, { 'Content-Type': 'text/plain' });
            res.end('Only HTTPS URLs supported.');
            return;
        }
        console.log('  → proxying ' + targetUrl.slice(0, 140));
        proxyFetch(targetUrl).then(function (r) {
            var ct = (/google\.[a-z.]+\.[a-z]+\/maps\/d\//i.test(targetUrl) && 'application/vnd.google-earth.kml+xml; charset=utf-8')
                || r.headers['content-type'] || 'application/octet-stream';
            res.writeHead(r.statusCode, { 'Content-Type': ct, 'Content-Length': r.data.length });
            res.end(r.data);
        }).catch(function (err) {
            console.log('  ! proxy error: ' + err.message);
            res.writeHead(502, { 'Content-Type': 'text/plain' });
            res.end('Proxy fetch failed: ' + err.message);
        });
        return;
    }

    // static files next to this script
    if (pathname === '/') pathname = '/index.html';
    var resolved = path.resolve(path.join(SCRIPT_DIR, pathname));
    if (resolved.startsWith(path.resolve(SCRIPT_DIR))) {
        try {
            if (fs.statSync(resolved).isFile()) {
                res.writeHead(200, { 'Content-Type': MIME[path.extname(resolved).toLowerCase()] || 'application/octet-stream' });
                fs.createReadStream(resolved).pipe(res);
                return;
            }
        } catch { /* fall through to 404 */ }
    }

    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not found.\n\nEndpoints:\n  /ping                  — health check\n  /proxy?url=ENCODED_URL — proxy a URL\n  /kml?url=ENCODED_URL   — proxy a URL, forced KML content type\n  /kml?mid=MY_MAPS_ID    — Google My Maps shortcut\n  /*                     — static files\n');
}

// ── Start ──────────────────────────────────────────────────────
function portOwner(port) {
    try {
        if (process.platform === 'win32') {
            const out = execSync(`netstat -ano | findstr "LISTENING" | findstr ":${port} "`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
            const pid = out.trim().split(/\s+/).pop();
            return pid ? { pid, cmd: '?' } : null;
        }
        const out = execSync(`lsof -nP -iTCP:${port} -sTCP:LISTEN -Fpc`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
        const m = out.match(/^p(\d+)/m), c = out.match(/^c(.+)$/m);
        return m ? { pid: m[1], cmd: c ? c[1].trim() : '?' } : null;
    } catch { return null; }
}

/* Does whatever holds the port answer our own /ping with "pong"? If yes it is
   an older copy of this proxy and it is stopped so the new one can start. */
function pingPort(port) {
    return new Promise((resolve) => {
        const req = http.get({ host: '127.0.0.1', port, path: '/ping', timeout: 1500 }, (res) => {
            let b = '';
            res.on('data', (c) => b += c);
            res.on('end', () => resolve(b.indexOf('pong') !== -1));
        });
        req.on('error', () => resolve(false));
        req.on('timeout', () => { req.destroy(); resolve(false); });
    });
}

function killPid(pid) {
    try {
        execSync(process.platform === 'win32' ? `taskkill /PID ${pid} /F` : `kill -9 ${pid}`, { stdio: 'pipe' });
        return true;
    } catch { return false; }
}

async function startProxy() {
    const owner = portOwner(port);
    if (owner && owner.pid !== String(process.pid)) {
        if (await pingPort(port)) {
            log(C.dim, `  Port ${port}: stopping an older proxy (PID ${owner.pid})`);
            killPid(owner.pid);
            await new Promise((r) => setTimeout(r, 500));
        } else {
            log(C.yellow, `  Port ${port} is used by ${owner.cmd} (PID ${owner.pid}) — leaving it alone`);
        }
    }

    const server = http.createServer(handler);

    server.on('error', (e) => {
        if (e.code === 'EADDRINUSE') {
            log(C.red, `  Port ${port} is already in use by another program.`);
            log(C.reset, '  Free the port and run this again:');
            log(C.reset, process.platform === 'win32'
                ? `    netstat -ano | findstr ":${port}"   then   taskkill /PID <pid> /F`
                : `    lsof -ti :${port} | xargs kill -9`);
            process.exit(1);
        }
        log(C.red, '  Server error: ' + e.message);
        process.exit(1);
    });

    const hosts = process.platform === 'win32' ? ['0.0.0.0'] : ['127.0.0.1', '::1'];
    let listening = 0;
    for (const host of hosts) {
        const s = host === hosts[0] ? server : http.createServer(handler);
        s.on('error', () => { /* that address family is unavailable — not fatal */ });
        s.listen(port, host, () => {
            listening++;
            if (listening > 1) return;
            log(C.green, '  ============================================');
            log(C.green, '  Proxy is running — nothing else to do.');
            log(C.green, `  Proxy + map:  http://localhost:${port}/map.html`);
            log(C.green, `  Health check: http://localhost:${port}/ping`);
            log(C.green, '  ============================================');
            log(C.dim, '  Press Ctrl+C to stop.');
            console.log('');
            if (openBrowser) {
                const url = `http://localhost:${port}/map.html`;
                try {
                    if (process.platform === 'darwin') execSync(`open "${url}"`, { stdio: 'pipe' });
                    else if (process.platform === 'linux') execSync(`xdg-open "${url}"`, { stdio: 'pipe' });
                    else spawn('cmd', ['/c', 'start', '', url], { detached: true, stdio: 'ignore' }).unref();
                } catch { /* not critical */ }
            }
        });
    }

    function cleanup() {
        log(C.yellow, '\n[proxy] Stopping…');
        try { server.close(); } catch { }
        setTimeout(() => process.exit(0), 200);
    }
    process.on('SIGINT', cleanup);
    process.on('SIGTERM', cleanup);
    process.on('uncaughtException', (e) => log(C.yellow, '  Unhandled error: ' + e.message + ' (proxy keeps running)'));
}

log(C.cyan, `proxy.js v${PROXY_VERSION} — plain HTTP, no certificate needed`);
checkForUpdates(startProxy);
