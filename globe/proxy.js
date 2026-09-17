const { createServer } = require('http');
const httpsMod = require('https');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const HTTP_PORT  = 8080;
const HTTPS_PORT = 8443;
const PROXY_VERSION = 4;
const SCRIPT_DIR = __dirname;
const CERT_PATH = path.join(SCRIPT_DIR, '.proxy-cert.pem');
const KEY_PATH  = path.join(SCRIPT_DIR, '.proxy-key.pem');

/* ═══ SELF-UPDATE ═══════════════════════════════════ */
const SELF_URL = 'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/proxy.js';
const SELF_PATH = path.join(SCRIPT_DIR, 'proxy.js');

(function checkForUpdates() {
    console.log('[proxy] Checking for updates…');
    httpsMod.get(SELF_URL, { headers: { 'User-Agent': 'Mozilla/5.0' } }, (res) => {
        if (res.statusCode !== 200) {
            console.log('[proxy] Update check: HTTP ' + res.statusCode + ' — starting');
            startProxy();
            return;
        }
        let remote = '';
        res.on('data', (c) => remote += c);
        res.on('end', () => {
            const m = remote.match(/PROXY_VERSION\s*=\s*(\d+)/);
            const remoteVer = m ? parseInt(m[1], 10) : 0;
            if (remoteVer > PROXY_VERSION) {
                console.log('[proxy] ┌─────────────────────────────────────────');
                console.log('[proxy] │ NEW VERSION: v' + remoteVer + ' (current: v' + PROXY_VERSION + ')');
                console.log('[proxy] │ Downloading…');
                try {
                    fs.writeFileSync(SELF_PATH, remote, 'utf8');
                    console.log('[proxy] │ Saved. Restarting…');
                    console.log('[proxy] └─────────────────────────────────────────');
                    const { spawn } = require('child_process');
                    spawn('node', [SELF_PATH], { stdio: 'inherit', detached: true }).unref();
                    process.exit(0);
                } catch (e) {
                    console.error('[proxy] │ Update failed: ' + e.message);
                    console.log('[proxy] └─────────────────────────────────────────');
                    startProxy();
                }
            } else {
                console.log('[proxy] Up to date (v' + PROXY_VERSION + ')');
                startProxy();
            }
        });
    }).on('error', () => { console.log('[proxy] Update check skipped (offline)'); startProxy(); });
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
function ensureCert() {
    if (fs.existsSync(CERT_PATH) && fs.existsSync(KEY_PATH)) {
        console.log('[proxy] SSL certificate found');
        return true;
    }
    console.log('[proxy] Generating self-signed SSL certificate…');
    try {
        execSync(
            'openssl req -x509 -newkey rsa:2048 -nodes ' +
            '-keyout ' + KEY_PATH + ' -out ' + CERT_PATH + ' ' +
            '-days 365 -subj "/CN=localhost" 2>/dev/null',
            { stdio: 'pipe' }
        );
        console.log('[proxy] SSL certificate created');
        return true;
    } catch (e) {
        console.log('[proxy] SSL generation failed (' + e.message + ') — HTTPS disabled');
        return false;
    }
}

/* ═══ REQUEST HANDLER ═══════════════════════════════ */
function proxyFetch(url, maxRedirects) {
  maxRedirects = maxRedirects || 5;
  return new Promise(function (resolve, reject) {
    if (maxRedirects <= 0) return reject(new Error('Too many redirects'));
    var mod = url.startsWith('https') ? httpsMod : require('http');
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
  if (/yandex\.(ru|com).*maps/i.targetUrl) return 'text/html; charset=utf-8';
  return null;
}

function serveStatic(req, res, pathname) {
  if (req.method !== 'GET') return false;
  if (pathname === '/') pathname = '/index.html';
  var filePath = path.join(SCRIPT_DIR, pathname);
  var resolved = path.resolve(filePath);
  if (!resolved.startsWith(path.resolve(SCRIPT_DIR))) return false;
  try {
    var stat = fs.statSync(resolved);
    if (!stat.isFile()) return false;
  } catch (e) { return false; }
  var ext = path.extname(resolved).toLowerCase();
  var ct = MIME[ext] || 'application/octet-stream';
  res.writeHead(200, { 'Content-Type': ct, 'Content-Length': stat.size, 'Access-Control-Allow-Origin': '*' });
  fs.createReadStream(resolved).pipe(res);
  return true;
}

function handler(req, res) {
  var parsedUrl = new URL(req.url, 'http://localhost');
  var pathname = parsedUrl.pathname;
  var targetUrl = parsedUrl.searchParams.get('url');

  console.log((req.socket.encrypted ? 'HTTPS' : 'HTTP ') + ' ' + req.method + ' ' + pathname + (targetUrl ? ' → proxy: ' + targetUrl.slice(0, 80) : ''));

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  /* Health check */
  if (pathname === '/ping') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('pong');
    return;
  }

  /* Proxy request */
  if (targetUrl) {
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
    return;
  }

  /* Static files */
  if (serveStatic(req, res, pathname)) return;

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not found.\n\nEndpoints:\n  /ping                  — health check\n  /proxy?url=ENCODED_URL — proxy a URL\n  /*                      — static files\n');
}

/* ═══ START SERVERS ═════════════════════════════════ */
function startProxy() {
  var hasHTTPS = ensureCert();

  /* HTTP server */
  createServer(handler).listen(HTTP_PORT, function () {
    console.log('');
    console.log('╔═══════════════════════════════════════════════════╗');
    console.log('║  CORS Proxy + Static Server v' + PROXY_VERSION + '                 ║');
    console.log('╠═══════════════════════════════════════════════════╣');
    console.log('║  HTTP:  http://localhost:' + HTTP_PORT + '                     ║');
    if (hasHTTPS) {
    console.log('║  HTTPS: https://localhost:' + HTTPS_PORT + '  (self-signed)  ║');
    }
    console.log('╠═══════════════════════════════════════════════════╣');
    console.log('║  Local map:   http://localhost:' + HTTP_PORT + '/map.html       ║');
    if (hasHTTPS) {
    console.log('║  From site:   https://localhost:' + HTTPS_PORT + '/map.html    ║');
    }
    console.log('╚═══════════════════════════════════════════════════╝');
    console.log('');
    if (hasHTTPS) {
    console.log('  HTTPS first-time setup (one click):');
    console.log('  1. Open https://localhost:' + HTTPS_PORT + '/ping in your browser');
    console.log('  2. Click "Advanced" → "Proceed to localhost (unsafe)"');
    console.log('  3. Done — your site can now reach the proxy');
    console.log('');
    }
  });

  /* HTTPS server */
  if (hasHTTPS) {
    try {
      var opts = {
        key:  fs.readFileSync(KEY_PATH),
        cert: fs.readFileSync(CERT_PATH),
      };
      httpsMod.createServer(opts, handler).listen(HTTPS_PORT, function () {
        /* server is ready */
      });
    } catch (e) {
      console.log('[proxy] HTTPS startup failed: ' + e.message + ' — HTTPS disabled');
    }
  }
}
