const { createServer } = require('http');
const httpsMod = require('https');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const HTTP_PORT  = 8080;
const HTTPS_PORT = 8443;
const PROXY_VERSION = 6;
const SCRIPT_DIR = __dirname;
const CERT_PATH = path.join(SCRIPT_DIR, '.proxy-cert.pem');
const KEY_PATH  = path.join(SCRIPT_DIR, '.proxy-key.pem');
const CERT_HOSTS = 'DNS:localhost,DNS:*.localhost,IP:127.0.0.1,IP:::1';

/* ═══ SELF-UPDATE ═══════════════════════════════════ */
const SELF_URL = 'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/proxy.js';
const SELF_CERT_URL = 'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/.proxy-cert.pem';
const SELF_KEY_URL  = 'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/.proxy-key.pem';
const SELF_PATH = path.join(SCRIPT_DIR, 'proxy.js');

/* ═══ BUNDLED FALLBACK CERTIFICATE ══════════════════
   Embedded so HTTPS always works, even on a machine without openssl and
   without internet (the .proxy-*.pem files are shipped next to this file
   anyway; this is only the last-resort copy). Self-signed, localhost-only.
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


function fetchUrl(url) {
    return new Promise((resolve, reject) => {
        httpsMod.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' } }, (res) => {
            if (res.statusCode !== 200) { res.resume(); return reject(new Error('HTTP ' + res.statusCode)); }
            let body = '';
            res.on('data', (c) => body += c);
            res.on('end', () => resolve(body));
        }).on('error', reject);
    });
}

(function checkForUpdates() {
    console.log('[proxy] Checking for updates…');
    fetchUrl(SELF_URL).then((remote) => {
        const m = remote.match(/PROXY_VERSION\s*=\s*(\d+)/);
        const remoteVer = m ? parseInt(m[1], 10) : 0;
        if (remoteVer > PROXY_VERSION) {
            console.log('[proxy] Updating to v' + remoteVer + ' (current v' + PROXY_VERSION + ')…');
            fs.writeFileSync(SELF_PATH, remote, 'utf8');
            const { spawn } = require('child_process');
            spawn('node', [SELF_PATH], { stdio: 'inherit', detached: true }).unref();
            process.exit(0);
        }
        console.log('[proxy] Up to date (v' + PROXY_VERSION + ')');
        startProxy();
    }).catch((e) => {
        console.log('[proxy] Update check skipped (' + e.message + ')');
        startProxy();
    });
})();
/* ═══ END SELF-UPDATE ═══════════════════════════════ */

/* ═══ MIME TYPES ════════════════════════════════════ */
var MIME = {
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
    '.woff2':'font/woff2',
    '.ttf':  'font/ttf',
    '.mp4':  'video/mp4',
    '.webm': 'video/webm',
    '.kml':  'application/vnd.google-earth.kml+xml',
};

/* ═══ CERTIFICATE GENERATION ════════════════════════ */
function certStatus() {
    /* Parsed with Node itself, so it also works on Windows without openssl.
       A legacy CN-only certificate (no subjectAltName) is rejected by
       Chrome/Edge and must be replaced. */
    let pem, key;
    try { pem = fs.readFileSync(CERT_PATH, 'utf8'); } catch (e) { return { ok: false, why: 'missing' }; }
    try { key = fs.readFileSync(KEY_PATH, 'utf8'); } catch (e) { return { ok: false, why: 'key missing' }; }
    if (!/BEGIN CERTIFICATE/.test(pem) || !/PRIVATE KEY/.test(key)) return { ok: false, why: 'unreadable files' };

    let x;
    try { x = new crypto.X509Certificate(pem); } catch (e) { return { ok: false, why: 'invalid certificate (' + e.message + ')' }; }
    if (!/DNS:localhost|IP Address:127\.0\.0\.1/i.test(String(x.subjectAltName || '')))
        return { ok: false, why: 'no subjectAltName for localhost (browsers reject this)' };

    const now = Date.now(), from = Date.parse(x.validFrom), to = Date.parse(x.validTo);
    if (isNaN(from) || isNaN(to)) return { ok: false, why: 'unparsable validity dates' };
    if (now < from) return { ok: false, why: 'not valid yet (' + x.validFrom + ')' };
    if (now > to) return { ok: false, why: 'expired on ' + x.validTo };
    return { ok: true, until: x.validTo };
}

function generateCert() {
    /* macOS ships LibreSSL as /usr/bin/openssl and only LibreSSL >= 3.1 knows
       -addext. On anything older the flag is rejected, so the first command
       below MUST NOT be trusted just because it exited 0 — the fallback builds
       a legacy CN-only certificate that Chrome/Safari refuse. Every generated
       certificate is therefore validated and an unusable one is discarded in
       favour of the bundled pair. */
    const variants = [
        'openssl req -x509 -newkey rsa:2048 -nodes ' +
            '-keyout "' + KEY_PATH + '" -out "' + CERT_PATH + '" ' +
            '-days 3650 -subj "/CN=localhost" ' +
            '-addext "subjectAltName=' + CERT_HOSTS + '"',
        'openssl req -x509 -newkey rsa:2048 -nodes -keyout "' + KEY_PATH + '" -out "' + CERT_PATH +
            '" -days 3650 -subj "/CN=localhost"',
    ];
    for (const cmd of variants) {
        try { execSync(cmd, { stdio: 'pipe' }); } catch (e) { continue; }
        const st = certStatus();
        if (st.ok) return true;
        console.log('[proxy] openssl made an unusable certificate (' + st.why + ') — trying another way');
    }
    return false;
}

function downloadFile(url, dest) {
    try {
        execSync('curl -fsSL -m 30 -o "' + dest + '" "' + url + '"', { stdio: 'pipe' });
        return fs.existsSync(dest);
    } catch (e) { return false; }
}

function downloadCert() {
    if (downloadFile(SELF_CERT_URL, CERT_PATH) && downloadFile(SELF_KEY_URL, KEY_PATH)) {
        const st = certStatus();
        if (st.ok) return true;
        console.log('[proxy] Downloaded certificate rejected: ' + st.why);
    }
    return false;
}

function ensureCert() {
    const st = certStatus();
    if (st.ok) {
        console.log('[proxy] SSL certificate found (valid until ' + st.until + ')');
        return true;
    }
    if (st.why !== 'missing') console.log('[proxy] Existing certificate unusable: ' + st.why);

    console.log('[proxy] Generating self-signed SSL certificate…');
    if (generateCert()) {
        console.log('[proxy] SSL certificate created');
        return true;
    }
    console.log('[proxy] openssl not available — using the bundled certificate');
    try {
        fs.writeFileSync(CERT_PATH, BUNDLED_CERT);
        fs.writeFileSync(KEY_PATH, BUNDLED_KEY);
        if (certStatus().ok) {
            console.log('[proxy] Bundled certificate installed');
            return true;
        }
    } catch (e) { console.log('[proxy] Could not write certificate files: ' + e.message); }

    console.log('[proxy] Trying to download the certificate…');
    if (downloadCert()) {
        console.log('[proxy] Certificate downloaded');
        return true;
    }
    console.log('[proxy] No usable certificate — HTTPS disabled');
    return false;
}

/* ═══ REQUEST HANDLER ═══════════════════════════════ */
function proxyFetch(url, maxRedirects) {
  maxRedirects = maxRedirects || 5;
  return new Promise(function (resolve, reject) {
    if (maxRedirects <= 0) return reject(new Error('Too many redirects'));
    var mod = url.startsWith('https') ? httpsMod : require('http');
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

function midToKmlUrl(mid) {
  return 'https://www.google.com/maps/d/u/0/kml?mid=' + mid + '&forcekml=1';
}

function detectContentType(targetUrl) {
  if (/google\.[a-z.]+\/maps\/d\//i.test(targetUrl)) return 'application/vnd.google-earth.kml+xml; charset=utf-8';
  if (/yandex\.[a-z.]+\/maps/i.test(targetUrl)) return 'text/html; charset=utf-8';
  return null;
}

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  /* Chrome Private Network Access — public https page → local server */
  res.setHeader('Access-Control-Allow-Private-Network', 'true');
}

function serveStatic(req, res, pathname) {
  if (req.method !== 'GET' && req.method !== 'HEAD') return false;
  if (pathname === '/') pathname = '/index.html';
  var resolved = path.resolve(path.join(SCRIPT_DIR, pathname));
  if (!resolved.startsWith(path.resolve(SCRIPT_DIR))) return false;
  try {
    var stat = fs.statSync(resolved);
    if (!stat.isFile()) return false;
  } catch (e) { return false; }
  var ext = path.extname(resolved).toLowerCase();
  var ct = MIME[ext] || 'application/octet-stream';
  res.writeHead(200, { 'Content-Type': ct, 'Content-Length': stat.size, 'Cache-Control': 'no-cache' });
  fs.createReadStream(resolved).pipe(res);
  return true;
}

function handler(req, res) {
  var parsedUrl = new URL(req.url, 'http://localhost');
  var pathname = parsedUrl.pathname;
  var targetUrl = parsedUrl.searchParams.get('url');
  var mid = parsedUrl.searchParams.get('mid');

  cors(res);

  console.log((req.socket.encrypted ? 'HTTPS' : 'HTTP ') + ' ' + req.method + ' ' + pathname + (targetUrl ? ' → ' + targetUrl.slice(0, 80) : ''));

  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  /* Health check */
  if (pathname === '/ping' || pathname === '/health') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('pong');
    return;
  }

  if (mid) targetUrl = midToKmlUrl(mid);

  /* Proxy request */
  if (targetUrl) {
    if (!/^https:\/\//i.test(targetUrl)) {
      res.writeHead(400, { 'Content-Type': 'text/plain' });
      res.end('Only HTTPS URLs supported.');
      return;
    }
    proxyFetch(targetUrl).then(function (r) {
      var ct = detectContentType(targetUrl) || r.headers['content-type'] || 'application/octet-stream';
      res.writeHead(r.statusCode, { 'Content-Type': ct, 'Content-Length': r.data.length });
      res.end(r.data);
    }).catch(function (err) {
      console.error('Proxy error:', err.message);
      res.writeHead(502, { 'Content-Type': 'text/plain' });
      res.end('Proxy fetch failed: ' + err.message);
    });
    return;
  }

  /* Static files */
  if (serveStatic(req, res, pathname)) return;

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not found.\n\nEndpoints:\n  /ping                  — health check\n  /proxy?url=ENCODED_URL — proxy a URL\n  /kml?url=ENCODED_URL   — proxy a URL, forced KML content type\n  /kml?mid=MY_MAPS_ID    — Google My Maps shortcut\n  /*                     — static files\n');
}

/* ═══ PORT HELPERS ══════════════════════════════════
   macOS specifics handled here:
   • "localhost" resolves to ::1 first, so binding 127.0.0.1 alone makes the
     browser's https://localhost:8443 request fail. Both loopback addresses
     are served.
   • A second launch used to die with an unhandled EADDRINUSE (the old proxy
     kept running in the background) — the stale listener is now detected and
     stopped, and a foreign program on the port is reported instead of
     crashing the process.                                          */

function portOwner(port) {
  try {
    if (process.platform === 'win32') {
      var out = execSync('netstat -ano | findstr "LISTENING" | findstr ":' + port + ' "', { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
      var pid = (out.trim().split(/\s+/).pop() || '');
      return pid ? { pid: pid, cmd: '?' } : null;
    }
    var lsof = execSync('lsof -nP -iTCP:' + port + ' -sTCP:LISTEN -Fpc', { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
    var m = lsof.match(/^p(\d+)/m), c = lsof.match(/^c(.+)$/m);
    return m ? { pid: m[1], cmd: c ? c[1].trim() : '?' } : null;
  } catch (e) { return null; }
}

function pingLoopback(port, tls) {
  return new Promise(function (resolve) {
    var mod = tls ? httpsMod : require('http');
    var opts = { host: '127.0.0.1', port: port, path: '/ping', timeout: 1500 };
    if (tls) opts.rejectUnauthorized = false;
    var req = mod.get(opts, function (res) {
      var b = '';
      res.on('data', function (c) { b += c; });
      res.on('end', function () { resolve(b.indexOf('pong') !== -1); });
    });
    req.on('error', function () { resolve(false); });
    req.on('timeout', function () { req.destroy(); resolve(false); });
  });
}

function killPid(pid) {
  try {
    execSync(process.platform === 'win32' ? 'taskkill /PID ' + pid + ' /F' : 'kill -9 ' + pid, { stdio: 'pipe' });
    return true;
  } catch (e) { return false; }
}

/* Free a port only when the program holding it is an older copy of this proxy
   (it answers "pong"). Someone else's server is never killed. */
async function freeOurStaleListener(port, label) {
  var owner = portOwner(port);
  if (!owner) return;
  if (owner.pid === String(process.pid)) return;   /* never kill ourselves */
  if (await pingLoopback(port, port === HTTPS_PORT)) {
    console.log('[proxy] ' + label + ' port ' + port + ' is held by an older proxy (PID ' + owner.pid + ') — stopping it');
    killPid(owner.pid);
    await new Promise(function (r) { setTimeout(r, 500); });
  } else {
    console.log('[proxy] ' + label + ' port ' + port + ' is used by another program: ' + owner.cmd + ' (PID ' + owner.pid + ')');
  }
}

/* ═══ START SERVERS ═════════════════════════════════ */
var SERVERS = [];

function listenOn(server, port, host) {
  return new Promise(function (resolve) {
    server.once('error', function (e) { resolve({ ok: false, err: e, server: server }); });
    server.listen(port, host, function () { resolve({ ok: true, server: server }); });
  });
}

/* One listener per loopback family — loopback only, so the macOS firewall
   never asks whether the proxy may accept incoming connections. */
async function listenDual(factory, port, label) {
  var res = { port: port, label: label, where: [], err: null };
  var busy = portOwner(port);
  if (busy) {
    res.err = new Error('port ' + port + ' is already in use by ' + (busy.cmd || '?') + ' (PID ' + busy.pid + ')');
    return res;
  }
  var r4 = await listenOn(factory(), port, '127.0.0.1');
  if (r4.ok) { res.where.push('127.0.0.1'); SERVERS.push(r4.server); } else { res.err = r4.err; }
  if (process.platform !== 'win32') {
    var r6 = await listenOn(factory(), port, '::1');
    if (r6.ok) { res.where.push('::1'); SERVERS.push(r6.server); } else if (!res.err) res.err = r6.err;
  }
  return res;
}

async function startProxy() {
  var hasHTTPS = ensureCert();

  await freeOurStaleListener(HTTP_PORT, 'HTTP');
  await freeOurStaleListener(HTTPS_PORT, 'HTTPS');

  var httpRes = await listenDual(function () { return createServer(handler); }, HTTP_PORT, 'HTTP');
  var httpsRes = null;
  if (hasHTTPS) {
    var opts;
    try {
      opts = { key: fs.readFileSync(KEY_PATH), cert: fs.readFileSync(CERT_PATH) };
    } catch (e) {
      opts = null;
      console.log('[proxy] HTTPS certificate unreadable: ' + e.message + ' — HTTPS disabled');
    }
    if (opts) {
      var tlsHandler = handler;
      httpsRes = await listenDual(function () { return httpsMod.createServer(opts, tlsHandler); }, HTTPS_PORT, 'HTTPS');
    }
  }

  function state(res) {
    if (!res) return false;
    if (res.where.length) return true;
    return false;
  }
  var okHTTP = state(httpRes), okHTTPS = state(httpsRes);

  console.log('');
  console.log('╔═══════════════════════════════════════════════════╗');
  console.log('║  CORS Proxy + Static Server v' + PROXY_VERSION + '                 ║');
  console.log('╠═══════════════════════════════════════════════════╣');
  console.log('║  HTTP:  ' + (okHTTP ? 'http://localhost:' + HTTP_PORT + '                 ' : 'NOT RUNNING (port busy)        ') + '║');
  console.log('║  HTTPS: ' + (okHTTPS ? 'https://localhost:' + HTTPS_PORT + '  (self-signed)' : 'NOT RUNNING                    ') + '║');
  console.log('╚═══════════════════════════════════════════════════╝');
  console.log('');
  if (okHTTPS) {
    console.log('  One-time step (only if the map still imports nothing):');
    console.log('  1. Open https://localhost:' + HTTPS_PORT + '/ping in your browser');
    console.log('  2. Click "Advanced" → "Proceed to localhost (unsafe)"');
    console.log('');
  } else {
    console.log('  !! HTTPS 8443 is NOT running — a map opened over https://');
    console.log('     (blackangelsk.github.io) cannot import anything.');
    if (httpsRes && httpsRes.err) console.log('     Reason: ' + httpsRes.err.message);
    console.log('     Free the port and start again:');
    console.log('       lsof -ti :8443 | xargs kill -9');
    console.log('');
  }
  if (!okHTTP) {
    console.log('  Local (http) map is unavailable — the map at');
    console.log('     http://localhost:' + HTTP_PORT + '/map.html needs port ' + HTTP_PORT + '.');
    if (httpRes && httpRes.err) console.log('     Reason: ' + httpRes.err.message);
    console.log('');
  }
  if (okHTTP || okHTTPS) {
    console.log('  Press Ctrl+C to stop.');
    console.log('');
  } else {
    console.log('  Nothing could be started. Free the ports and run again:');
    console.log('    lsof -ti :' + HTTP_PORT + ' | xargs kill -9');
    console.log('    lsof -ti :' + HTTPS_PORT + ' | xargs kill -9');
    console.log('');
    process.exit(1);
  }

  function cleanup() {
    console.log('');
    console.log('[proxy] Stopping…');
    SERVERS.forEach(function (s) { try { s.close(); } catch (e) { } });
    setTimeout(function () { process.exit(0); }, 200);
  }
  process.on('SIGINT', cleanup);
  process.on('SIGTERM', cleanup);
  process.on('uncaughtException', function (e) {
    console.log('[proxy] Unhandled error: ' + e.message + ' (proxy keeps running)');
  });
}
