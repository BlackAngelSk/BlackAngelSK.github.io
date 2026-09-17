const { createServer } = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

const PORT = 8080;
const PROXY_VERSION = 3;  // bump this when you push updates
const SCRIPT_DIR = __dirname;

/* ═══ SELF-UPDATE ═══════════════════════════════════ */
const SELF_URL = 'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/proxy.js';
const SELF_PATH = path.join(SCRIPT_DIR, 'proxy.js');

(function checkForUpdates() {
    console.log('[proxy] Checking for updates…');
    https.get(SELF_URL, { headers: { 'User-Agent': 'Mozilla/5.0' } }, (res) => {
        if (res.statusCode !== 200) {
            console.log('[proxy] Update check: HTTP ' + res.statusCode + ' — starting with current version');
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
                console.log('[proxy] │ NEW VERSION AVAILABLE: v' + remoteVer + ' (current: v' + PROXY_VERSION + ')');
                console.log('[proxy] │ Downloading update…');
                try {
                    fs.writeFileSync(SELF_PATH, remote, 'utf8');
                    console.log('[proxy] │ Saved to: ' + SELF_PATH);
                    console.log('[proxy] │ Restarting with new version…');
                    console.log('[proxy] └─────────────────────────────────────────');
                    const { spawn } = require('child_process');
                    spawn('node', [SELF_PATH], { stdio: 'inherit', detached: true }).unref();
                    process.exit(0);
                } catch (e) {
                    console.error('[proxy] │ Update failed: ' + e.message);
                    console.log('[proxy] │ Continuing with current version…');
                    console.log('[proxy] └─────────────────────────────────────────');
                    startProxy();
                }
            } else {
                console.log('[proxy] Up to date (v' + PROXY_VERSION + ')');
                startProxy();
            }
        });
    }).on('error', (e) => {
        console.log('[proxy] Update check skipped (' + e.code + ') — starting with current version');
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

function startProxy() {

function proxyFetch(url, maxRedirects) {
  maxRedirects = maxRedirects || 5;
  return new Promise(function (resolve, reject) {
    if (maxRedirects <= 0) return reject(new Error('Too many redirects'));
    var mod = url.startsWith('https') ? https : require('http');
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

/* Serve a static file from SCRIPT_DIR. Returns true if served. */
function serveStatic(req, res, pathname) {
  // Only allow GET
  if (req.method !== 'GET') return false;

  // Default to index.html
  if (pathname === '/') pathname = '/index.html';

  // Security: resolve and check the path stays within SCRIPT_DIR
  var filePath = path.join(SCRIPT_DIR, pathname);
  var resolved = path.resolve(filePath);
  if (!resolved.startsWith(path.resolve(SCRIPT_DIR))) return false;

  // Check file exists
  try {
    var stat = fs.statSync(resolved);
    if (!stat.isFile()) return false;
  } catch (e) {
    return false;
  }

  // Serve it
  var ext = path.extname(resolved).toLowerCase();
  var ct = MIME[ext] || 'application/octet-stream';
  res.writeHead(200, {
    'Content-Type': ct,
    'Content-Length': stat.size,
    'Access-Control-Allow-Origin': '*',
  });
  fs.createReadStream(resolved).pipe(res);
  return true;
}

createServer(function (req, res) {
  var parsedUrl = new URL(req.url, 'http://localhost');
  var pathname = parsedUrl.pathname;
  var targetUrl = parsedUrl.searchParams.get('url');

  /* Log every request */
  console.log(req.method + ' ' + pathname + (targetUrl ? ' → proxy: ' + targetUrl.slice(0, 80) : ''));

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  /* ── Health check ──────────────────────────────── */
  if (pathname === '/ping') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('pong');
    return;
  }

  /* ── Proxy request (has ?url= param) ───────────── */
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

  /* ── Static file fallback (map, globe, etc.) ───── */
  if (serveStatic(req, res, pathname)) return;

  /* ── Nothing matched ───────────────────────────── */
  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not found.\n\nEndpoints:\n  /ping                  — health check\n  /proxy?url=ENCODED_URL — proxy a URL\n  /*                      — static files from globe/\n');
}).listen(PORT, function () {
  console.log('');
  console.log('╔═══════════════════════════════════════════════╗');
  console.log('║  CORS Proxy + Static Server v' + PROXY_VERSION + '             ║');
  console.log('║  http://localhost:' + PORT + '                        ║');
  console.log('╠═══════════════════════════════════════════════╣');
  console.log('║  Map:   http://localhost:' + PORT + '/map.html        ║');
  console.log('║  Globe: http://localhost:' + PORT + '/index.html     ║');
  console.log('║  Ping:  http://localhost:' + PORT + '/ping           ║');
  console.log('╠═══════════════════════════════════════════════╣');
  console.log('║  Proxy URL:                                 ║');
  console.log('║  http://localhost:' + PORT + '/proxy?url=ENCODED_URL ║');
  console.log('╚═══════════════════════════════════════════════╝');
  console.log('');
});

} /* end startProxy */
