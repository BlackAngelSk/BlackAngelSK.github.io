#!/usr/bin/env node
/**
 * globe-server.js — Globe/Map all-in-one server.
 *
 *   • CORS proxy + static files on  http://localhost:8080
 *   • static files + CORS proxy on  http://localhost:8000
 *
 * PLAIN HTTP BY DEFAULT — no certificate, no warning to click through,
 * nothing to accept. Just start it and it works.
 *
 *   Chrome, Edge, Firefox and Brave treat http://localhost as a trustworthy
 *   origin, so a page served from https://blackangelsk.github.io may call
 *   http://localhost:8080 directly (no mixed-content block). That is why no
 *   HTTPS listener is needed.
 *
 *   Safari does NOT have that exception. If you use Safari, start the server
 *   with --https: it then also listens on https://localhost:8443 with a
 *   self-signed certificate that has to be accepted once.
 *
 * No external dependencies — Node.js stdlib only.
 *
 * Usage:
 *   globe-server                 proxy + static server, opens the browser
 *   globe-server --proxy         proxy only (no second static server)
 *   globe-server --no-open       do not open the browser
 *   globe-server --https         ALSO listen on https://localhost:8443 (Safari)
 *   globe-server --port 3000     custom static-server port
 *   globe-server --help          this text
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execSync, spawn } = require('child_process');

// ── Configuration ──────────────────────────────────────────────
const PROXY_PORT = 8080;      // CORS proxy + static files
const HTTPS_PORT = 8443;      // only with --https
const HTTP_PORT = 8000;       // second static server
const SERVER_VERSION = 7;

const CERT_HOSTS = 'DNS:localhost,DNS:*.localhost,IP:127.0.0.1,IP:::1';
const CERT_URLS = [
    'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe',
    'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/main/globe',
];

/* ═══ BUNDLED FALLBACK CERTIFICATE ══════════════════
   Only used with --https, on a machine without openssl and without internet.
   Self-signed, localhost-only.
   ═══════════════════════════════════════════════════ */
const BUNDLED_CERT = `-----BEGIN CERTIFICATE-----
MIIDRjCCAi6gAwIBAgIUMdg7GIPvrlckYkhjf4NkmkxnNYUwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDkyMTE2MDUxOFoXDTM2MDkx
ODE2MDUxOFowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAzThFcLBbmG22jGu+itXbdlayj0+tfUQpYdBRrQSV2B2K
5YdLFQ7XddliKQ3RwEKN6k1o4i9BCGJd/kVUD339GLZFPzw+Tv0UimNDNS9dVW1Z
nErrhBkaMxa6X1KuLhlBkV5EvOxFtQkIVdp/YP8jrZjT9PZd88TU4lA5/gvZnrjP
ZnwvPdskPkwwvK7IjscLnrsVR/IiTrehypG5PMYJPQYOQqzFGoj2ACJgQeD70WVw
XrUi8tk986MpKJ2fG1YjjusrynZ2xvIqMRXunlgRVamvcxMh+qLkHaSgwIh43n1m
CFuhaXernwBJVLX/uY3u/3So/51d2xJJ2fsHdd493wIDAQABo4GPMIGMMB0GA1Ud
DgQWBBRjahXA0lRy5bL9GAsXJTkExgqm8TAfBgNVHSMEGDAWgBRjahXA0lRy5bL9
GAsXJTkExgqm8TAPBgNVHRMBAf8EBTADAQH/MDkGA1UdEQQyMDCCCWxvY2FsaG9z
dIILKi5sb2NhbGhvc3SHBH8AAAGHEAAAAAAAAAAAAAAAAAAAAAEwDQYJKoZIhvcN
AQELBQADggEBAHp6DLUOHTPJqwNiPhebgFPA81P62dXeAz9jjY96SUlgC18Ni7My
ZihvQWqXAQTuRU/bW6FMJsf7S3xrY9kULp0sf0AwujiioXI7hg3L2pc5e/H2UD/7
wWkcp19IFtuSS3geNESo4bpye0UsnGs8phVd7OMHq8LCKICbJ3eMenM1LUaemcbL
PuAYvHsteBU437MgJs/PER0bh3MtEEDp35Oqnb94/A8ODJ1Nw0/xVyTc/S88owuM
T5+C4IaJisCftngmeUiZlybWonK+dHfi//eik6IFdEEsWzZyduJjX6+N3zgoe/lX
XuSs1ZR2G/N5ZJIZkSajJxkRG5U97+eMnPQ=
-----END CERTIFICATE-----`;
const BUNDLED_KEY  = `-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDNOEVwsFuYbbaM
a76K1dt2VrKPT619RClh0FGtBJXYHYrlh0sVDtd12WIpDdHAQo3qTWjiL0EIYl3+
RVQPff0YtkU/PD5O/RSKY0M1L11VbVmcSuuEGRozFrpfUq4uGUGRXkS87EW1CQhV
2n9g/yOtmNP09l3zxNTiUDn+C9meuM9mfC892yQ+TDC8rsiOxwueuxVH8iJOt6HK
kbk8xgk9Bg5CrMUaiPYAImBB4PvRZXBetSLy2T3zoykonZ8bViOO6yvKdnbG8iox
Fe6eWBFVqa9zEyH6ouQdpKDAiHjefWYIW6Fpd6ufAElUtf+5je7/dKj/nV3bEknZ
+wd13j3fAgMBAAECggEAWfuSdxbdErkIkgmsQrQCxqC9KpmFOKfqfTkalwKrsVdC
z+HCyjj3wHTQ1a95iROlplbK5mJR4ZtAG33KngBgObWyJ0gDCh9uIj/j+C4Mfqiz
4hP3sLRRCJWuop1eQlhQp6zt9uiip/6N6RclQyKcQkIArihYicqEjbcftoT7ptwY
xJoDYtBJVvofgta7O1wlY8r0gCq3R08B6nv64AX9Fbd+2V6PbyqQpB4JfTr5ol1J
Ic0tVKrNLFtllPl0Xs8t8JFMIERH8GwAGBhidaa3h9ZpQKOx8zaWXr4W+dKR+oZ6
RGSwiQtfjuLlxg6Gp4hRhy978vk15nD4V4aApvl+AQKBgQDyP+ACV8ado/LONuFs
Cptc/GmS+x8W+x8jhdI0GwS+OnEWZRpQFqt32H3QnOwwh2qa8COpk5aFeQHbYE5O
tjT9yYway5pXbX9Mh5dQg1VWpX3wegOmZTUsUXmoBpajWS/cBiH76hdmLgOK0b8G
eROxtMigBqX5tTAGtug8RdUSLQKBgQDY3lFp7wcE0+dwO8TThIhTaN9C6ejJUB5l
z1CTtNj68QJBQ/ArV7tI/+LfOSgwg+55EyUCemnCiRe8iC0FfQteN/q/l4P11CPA
jDLV2xJKhdrqq2dn/ovZ0R3zGMpjX3Ro18IDKnWTUZtZOqi/wTMIIQqnKzyWCiuJ
wro0QuczuwKBgQCXK7cdnyrpesLtXqADbQcQ2s8DEsSO8by3zS2kqGuoTg3+HK9z
5QGxsx6oTRSzH/4tm8eVbe1Tc3TuMkYLpnahHOHaYs342vLCjial7p8ViAZ0R5/5
eVTaSCmz6hCc9O4Bzq3sZ2OctwAs48DiSDI7BgbgneV7U08aEZAQ9L8HLQKBgFvS
JV5Y2fvtTHx0ke2Mm7vVF1JZfzpdNcGdeVxQod391RqVDUcMqjtKPGCO7rk74uhY
dr8J3WWEbgxhC87bFskZoo0kNbcLzudjdNKfIqty6TGayPl7CPN8WtXmlSpl6//H
+lkkzQtG7iNldrVZs6Wpd703zTspqy10ySn/eaypAoGBANdzYWCiAXzNwJemkeYi
TGRMj+PfIWJ/jabYc+oKH4gx1KD4WuO3FL8NPEz+MoJs+sVDWdQTbu3r1tpL2fCy
+w+4n6LZoeBdQ+5QgCdxtfhJRjcqqDj63fx5on70NYemQeMDLSpqONmn2xOiu7Rx
Gxt7BY5zfSND9ETnADmmNDwj
-----END PRIVATE KEY-----`;


// ── Script directory (works both as plain .js and as a pkg-compiled binary) ──
function scriptDir() {
    if (process.pkg) return path.dirname(process.execPath);
    if (typeof __dirname === 'string' && __dirname && !__dirname.startsWith('/snapshot')) return __dirname;
    return process.cwd();
}
const SCRIPT_DIR = scriptDir();
const CERT_PATH = path.join(SCRIPT_DIR, '.proxy-cert.pem');
const KEY_PATH = path.join(SCRIPT_DIR, '.proxy-key.pem');

// ── Parse CLI arguments ────────────────────────────────────────
const args = process.argv.slice(2);
let customHttpPort = HTTP_PORT;
let proxyOnly = false;
let wantHTTPS = false;
let openBrowser = true;

for (let i = 0; i < args.length; i++) {
    if (args[i] === '--proxy') proxyOnly = true;
    else if (args[i] === '--no-open') openBrowser = false;
    else if (args[i] === '--https') wantHTTPS = true;
    else if (args[i] === '--port' && args[i + 1]) { customHttpPort = parseInt(args[i + 1], 10); i++; }
    else if (args[i] === '--help' || args[i] === '-h') {
        console.log(`
  globe-server ${SERVER_VERSION} — Globe/Map all-in-one server (plain HTTP)

  Usage:
    globe-server                     proxy + static server, opens the browser
    globe-server --proxy             proxy only
    globe-server --no-open           do not open the browser
    globe-server --https             ALSO listen on https://localhost:8443
    globe-server --port 3000         custom static-server port
    globe-server --help              this text

  Ports:
    8080  proxy + local copy of the map   (http://localhost:8080/map.html)
    8000  second static server            (http://localhost:8000/map.html)
    8443  only with --https               (https://localhost:8443/ping)
`);
        process.exit(0);
    }
}

// ── Colors ─────────────────────────────────────────────────────
const C = {
    red: '\x1b[31m', green: '\x1b[32m', yellow: '\x1b[33m',
    cyan: '\x1b[36m', dim: '\x1b[2m', reset: '\x1b[0m', bold: '\x1b[1m',
};
function log(color, msg) { console.log(`${color}${msg}${C.reset}`); }

// ── Ports ──────────────────────────────────────────────────────
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

/* Does something on this port answer our own /ping with "pong"? */
function pingLocal(port, tls) {
    return new Promise((resolve) => {
        const mod = tls ? https : http;
        const opts = { host: '127.0.0.1', port, path: '/ping', timeout: 1500 };
        if (tls) opts.rejectUnauthorized = false;
        const req = mod.get(opts, (res) => {
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

function freePortHint(port) {
    if (process.platform === 'win32') return `    netstat -ano | findstr ":${port}"   then   taskkill /PID <pid> /F`;
    if (process.platform === 'darwin') return `    lsof -ti :${port} | xargs kill -9   (or: sudo lsof -ti :${port} | xargs kill -9)`;
    return `    lsof -ti :${port} | xargs kill -9   (or: ss -ltnp | grep :${port})`;
}

/* Free a port only when an older copy of THIS server holds it. Another
   program's server is never killed — we report it and carry on. */
async function freePort(port, label) {
    const owner = portOwner(port);
    if (!owner || owner.pid === String(process.pid)) return true;
    if (await pingLocal(port, port === HTTPS_PORT)) {
        log(C.dim, `  ${label} port ${port}: stopping an older copy (PID ${owner.pid})`);
        killPid(owner.pid);
        await new Promise((r) => setTimeout(r, 500));
        return true;
    }
    log(C.yellow, `  ${label} port ${port} is used by ${owner.cmd} (PID ${owner.pid}) — leaving it alone`);
    return false;
}

// ══════════════════════════════════════════════════════════════
//  CERTIFICATE (only for --https)
// ══════════════════════════════════════════════════════════════

function certStatus() {
    /* Parsed with Node itself, so it also works on Windows without openssl.
       Chrome/Edge reject a certificate with no subjectAltName for localhost
       (a legacy CN-only cert), so that is checked properly. */
    let pem, key;
    try { pem = fs.readFileSync(CERT_PATH, 'utf8'); } catch { return { ok: false, why: 'missing' }; }
    try { key = fs.readFileSync(KEY_PATH, 'utf8'); } catch { return { ok: false, why: 'key missing' }; }
    if (!/BEGIN CERTIFICATE/.test(pem) || !/PRIVATE KEY/.test(key)) return { ok: false, why: 'unreadable files' };

    let x;
    try { x = new crypto.X509Certificate(pem); } catch (e) { return { ok: false, why: 'not a valid certificate (' + e.message + ')' }; }

    const san = String(x.subjectAltName || '');
    if (!/DNS:localhost|IP Address:127\.0\.0\.1/i.test(san)) return { ok: false, why: 'no subjectAltName for localhost (browsers reject this)' };

    const now = Date.now();
    const from = Date.parse(x.validFrom), to = Date.parse(x.validTo);
    if (isNaN(from) || isNaN(to)) return { ok: false, why: 'unparsable validity dates' };
    if (now < from) return { ok: false, why: 'not valid yet (' + x.validFrom + ')' };
    if (now > to) return { ok: false, why: 'expired on ' + x.validTo };

    return { ok: true, until: x.validTo };
}

function downloadSync(url, dest) {
    const target = path.join(dest, path.basename(url));
    const cmd = process.platform === 'win32'
        ? `powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = 'Tls12'; Invoke-WebRequest -Uri '${url}' -OutFile '${target}' -UseBasicParsing"`
        : `curl -fsSL -m 30 -o "${target}" "${url}"`;
    try { execSync(cmd, { stdio: 'pipe' }); return fs.existsSync(target); }
    catch { return false; }
}

function downloadCert() {
    for (const base of CERT_URLS) {
        if (downloadSync(base + '/.proxy-cert.pem', SCRIPT_DIR) &&
            downloadSync(base + '/.proxy-key.pem', SCRIPT_DIR)) {
            const st = certStatus();
            if (st.ok) return true;
            log(C.yellow, '  Downloaded certificate rejected: ' + st.why);
        }
    }
    return false;
}

function generateCert() {
    /* macOS ships LibreSSL as /usr/bin/openssl and only LibreSSL >= 3.1 knows
       -addext. Older versions reject the flag and the fallback builds a legacy
       CN-only certificate that Chrome/Safari refuse, so every generated
       certificate is validated and an unusable one is discarded. */
    const variants = [
        'openssl req -x509 -newkey rsa:2048 -nodes ' +
            '-keyout "' + KEY_PATH + '" -out "' + CERT_PATH + '" ' +
            '-days 3650 -subj "/CN=localhost" ' +
            '-addext "subjectAltName=' + CERT_HOSTS + '"',
        'openssl req -x509 -newkey rsa:2048 -nodes -keyout "' + KEY_PATH + '" -out "' + CERT_PATH +
            '" -days 3650 -subj "/CN=localhost"',
    ];
    for (const cmd of variants) {
        try { execSync(cmd, { stdio: 'pipe' }); } catch { continue; }
        const st = certStatus();
        if (st.ok) return true;
        log(C.yellow, '  openssl produced an unusable certificate (' + st.why + ') — trying another way');
    }
    return false;
}

function ensureCert() {
    const st = certStatus();
    if (st.ok) {
        log(C.dim, '  SSL certificate found (valid until ' + st.until + ')');
        return true;
    }
    if (st.why !== 'missing') log(C.yellow, '  Existing certificate unusable: ' + st.why);

    if (generateCert()) { log(C.green, '  Self-signed SSL certificate created'); return true; }

    log(C.yellow, '  openssl not available — using the bundled certificate');
    try {
        fs.writeFileSync(CERT_PATH, BUNDLED_CERT);
        fs.writeFileSync(KEY_PATH, BUNDLED_KEY);
        if (certStatus().ok) { log(C.green, '  Bundled certificate installed'); return true; }
    } catch (e) { log(C.yellow, '  Could not write certificate files: ' + e.message); }

    log(C.yellow, '  Trying to download the certificate…');
    if (downloadCert()) { log(C.green, '  Certificate downloaded'); return true; }

    log(C.red, '  No usable certificate — the --https listener stays off.');
    return false;
}

// ══════════════════════════════════════════════════════════════
//  CORS PROXY
// ══════════════════════════════════════════════════════════════

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

/** Google My Maps id → full KML export URL */
function midToKmlUrl(mid) {
    return 'https://www.google.com/maps/d/u/0/kml?mid=' + mid + '&forcekml=1';
}

function detectContentType(targetUrl) {
    if (/google\.[a-z.]+\.[a-z]+\/maps\/d\//i.test(targetUrl)) return 'application/vnd.google-earth.kml+xml; charset=utf-8';
    if (/yandex\.[a-z.]+\/maps/i.test(targetUrl)) return 'text/html; charset=utf-8';
    return null;
}

function corsHeaders(res) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    /* Chrome "Private Network Access": a public HTTPS page reaching a local
       server sends a preflight that only passes with this header. */
    res.setHeader('Access-Control-Allow-Private-Network', 'true');
}

/** Handles /ping, /proxy?url=…, /kml?url=…, /kml?mid=… . Returns true if it answered. */
function handleProxyRoute(req, res, parsedUrl) {
    var pathname = parsedUrl.pathname;
    var targetUrl = parsedUrl.searchParams.get('url');
    var mid = parsedUrl.searchParams.get('mid');

    if (pathname === '/ping' || pathname === '/health') {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('pong');
        return true;
    }

    if (mid) targetUrl = midToKmlUrl(mid);

    if (!targetUrl) return false;

    if (!/^https:\/\//i.test(targetUrl)) {
        res.writeHead(400, { 'Content-Type': 'text/plain' });
        res.end('Only HTTPS URLs supported.');
        return true;
    }

    console.log('  → proxying ' + targetUrl.slice(0, 140));
    proxyFetch(targetUrl).then(function (r) {
        var ct = detectContentType(targetUrl) || r.headers['content-type'] || 'application/octet-stream';
        res.writeHead(r.statusCode, { 'Content-Type': ct, 'Content-Length': r.data.length });
        res.end(r.data);
    }).catch(function (err) {
        console.log('  ! proxy error: ' + err.message);
        res.writeHead(502, { 'Content-Type': 'text/plain' });
        res.end('Proxy fetch failed: ' + err.message);
    });
    return true;
}

// ══════════════════════════════════════════════════════════════
//  STATIC FILES
// ══════════════════════════════════════════════════════════════

const MIME = {
    '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8',
    '.xml': 'text/xml; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon', '.woff': 'font/woff', '.woff2': 'font/woff2',
    '.ttf': 'font/ttf', '.mp4': 'video/mp4', '.webm': 'video/webm',
    '.kml': 'application/vnd.google-earth.kml+xml', '.kmz': 'application/vnd.google-earth.kmz',
    '.zip': 'application/zip', '.txt': 'text/plain; charset=utf-8', '.md': 'text/plain; charset=utf-8',
};

function serveStatic(res, pathname) {
    if (pathname === '/') pathname = '/index.html';
    var resolved = path.resolve(path.join(SCRIPT_DIR, pathname));
    if (!resolved.startsWith(path.resolve(SCRIPT_DIR))) return false;
    try { if (!fs.statSync(resolved).isFile()) return false; } catch { return false; }
    var ext = path.extname(resolved).toLowerCase();
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    fs.createReadStream(resolved).pipe(res);
    return true;
}

/* Both ports answer the proxy API AND serve the map/globe files, so
   http://localhost:8080/map.html works as documented. */
function makeHandler() {
    return function (req, res) {
        corsHeaders(res);
        if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

        var parsedUrl;
        try { parsedUrl = new URL(req.url, 'http://localhost'); }
        catch { res.writeHead(400); res.end('Bad request'); return; }

        var target = parsedUrl.searchParams.get('url') || parsedUrl.searchParams.get('mid');
        console.log('[proxy] ' + (req.socket.encrypted ? 'HTTPS' : 'HTTP ') + ' ' + req.method + ' ' +
            parsedUrl.pathname + (target ? ' → ' + target.slice(0, 100) : ''));

        if (handleProxyRoute(req, res, parsedUrl)) return;
        if (serveStatic(res, parsedUrl.pathname)) return;

        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('Not found.\n\nEndpoints:\n  /ping                   — health check (expects "pong")\n' +
            '  /proxy?url=ENCODED_URL  — proxy any HTTPS URL\n' +
            '  /kml?url=ENCODED_URL    — same, forced KML content type\n' +
            '  /kml?mid=MY_MAPS_ID     — Google My Maps id shortcut\n' +
            '  /*                      — static files\n');
    };
}

// ══════════════════════════════════════════════════════════════
//  SERVERS
// ══════════════════════════════════════════════════════════════

function listenOn(server, port, host) {
    return new Promise(function (resolve) {
        server.once('error', function (e) { resolve({ ok: false, err: e, server }); });
        server.listen(port, host, function () { resolve({ ok: true, server }); });
    });
}

/* One listener per loopback family. macOS resolves "localhost" to ::1 before
   127.0.0.1, so binding only IPv4 makes http://localhost fail there.
   Loopback only keeps the macOS firewall quiet. */
async function listenDual(factory, port, label, servers) {
    const res = { port, label, where: [], err: null };
    const busy = portOwner(port);
    if (busy) {
        res.err = new Error(`port ${port} is already in use by ${busy.cmd || '?'} (PID ${busy.pid})`);
        return res;
    }
    const r4 = await listenOn(factory(), port, '127.0.0.1');
    if (r4.ok) { res.where.push('127.0.0.1'); servers.push(r4.server); } else { res.err = r4.err; }
    if (process.platform !== 'win32') {
        const r6 = await listenOn(factory(), port, '::1');
        if (r6.ok) { res.where.push('::1'); servers.push(r6.server); } else if (!res.err) res.err = r6.err;
    }
    return res;
}

// ══════════════════════════════════════════════════════════════
//  MAIN
// ══════════════════════════════════════════════════════════════

(async function main() {
    log(C.cyan, '============================================');
    log(C.cyan, `  Globe/Map Server v${SERVER_VERSION} — plain HTTP`);
    log(C.cyan, '============================================');
    log(C.reset, `  Platform: ${process.platform} (${process.arch})`);
    log(C.reset, `  Node.js:  ${process.version}`);
    log(C.reset, `  Folder:   ${SCRIPT_DIR}`);
    log(C.reset, '');

    log(C.yellow, 'Checking for existing servers…');
    await freePort(PROXY_PORT, 'Proxy');
    if (!proxyOnly) await freePort(customHttpPort, 'Static');
    if (wantHTTPS) await freePort(HTTPS_PORT, 'HTTPS');
    log(C.reset, '');

    let httpsOpts = null;
    if (wantHTTPS) {
        log(C.dim, '  --https requested: preparing the certificate…');
        if (ensureCert()) {
            try { httpsOpts = { key: fs.readFileSync(KEY_PATH), cert: fs.readFileSync(CERT_PATH) }; }
            catch (e) { log(C.red, '  Certificate unreadable: ' + e.message + ' — HTTPS disabled'); }
        }
    }

    const servers = [];

    // Proxy + files over plain HTTP — this is the whole point.
    const httpRes = await listenDual(() => http.createServer(makeHandler()), PROXY_PORT, 'Proxy', servers);

    // Optional HTTPS, only with --https
    let httpsRes = null;
    if (httpsOpts) {
        httpsRes = await listenDual(() => https.createServer(httpsOpts, makeHandler()), HTTPS_PORT, 'HTTPS', servers);
    }

    // Second static server
    let staticRes = null;
    if (!proxyOnly) {
        staticRes = await listenDual(() => http.createServer(makeHandler()), customHttpPort, 'Static', servers);
    }

    for (const r of [httpRes, httpsRes, staticRes]) {
        if (r && !r.where.length) log(C.red, `  ${r.label} on port ${r.port} did NOT start: ${r.err ? r.err.message : 'unknown error'}`);
    }

    const ok = (r) => !!(r && r.where.length);

    console.log('');
    if (ok(httpRes)) {
        log(C.green, '  ============================================');
        log(C.green, '  Proxy is running — nothing else to do.');
        log(C.green, `  Proxy + map:  http://localhost:${PROXY_PORT}/map.html`);
        log(C.green, `  Health check: http://localhost:${PROXY_PORT}/ping`);
        if (ok(staticRes)) log(C.green, `  Also on:      http://localhost:${customHttpPort}/map.html`);
        log(C.green, '  ============================================');
    } else {
        log(C.red, '  The proxy could NOT start on port ' + PROXY_PORT + '.');
        if (httpRes && httpRes.err) log(C.yellow, '  Reason: ' + httpRes.err.message);
        log(C.reset, '  Free the port and run again:');
        log(C.reset, freePortHint(PROXY_PORT));
    }

    if (wantHTTPS && ok(httpsRes)) {
        console.log('');
        log(C.yellow, '  --https is on. One-time step (only needed for Safari):');
        log(C.reset, `    1. Open https://localhost:${HTTPS_PORT}/ping in your browser`);
        log(C.reset, '    2. Click "Advanced" → "Continue to localhost (unsafe)"');
    } else if (wantHTTPS) {
        console.log('');
        log(C.yellow, '  --https was requested but port ' + HTTPS_PORT + ' did not start — plain HTTP still works.');
    }

    if (!ok(httpRes)) {
        console.log('');
        process.exit(1);
    }

    log(C.dim, '  Press Ctrl+C to stop.');
    console.log('');

    // Open the browser once, after everything is up.
    if (openBrowser && !proxyOnly) {
        const openUrl = `http://localhost:${PROXY_PORT}/map.html`;
        setTimeout(() => {
            try {
                if (process.platform === 'darwin') execSync(`open "${openUrl}"`, { stdio: 'pipe' });
                else if (process.platform === 'linux') execSync(`xdg-open "${openUrl}"`, { stdio: 'pipe' });
                else if (process.platform === 'win32') spawn('cmd', ['/c', 'start', '', openUrl], { detached: true, stdio: 'ignore' }).unref();
            } catch { /* not critical */ }
        }, 1000);
    }

    function cleanup() {
        log(C.yellow, '\nStopping servers…');
        servers.forEach(s => { try { s.close(); } catch { } });
        setTimeout(() => process.exit(0), 200);
    }
    process.on('SIGINT', cleanup);
    process.on('SIGTERM', cleanup);
    process.on('uncaughtException', (e) => log(C.yellow, '  Unhandled error: ' + e.message + ' (server keeps running)'));
})();
