const { createServer } = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

const PORT = 8080;
const PROXY_VERSION = 2;  // bump this when you push updates

/* ═══ SELF-UPDATE ═══════════════════════════════════ */
const SELF_URL = 'https://raw.githubusercontent.com/BlackAngelSK/BlackAngelSK.github.io/master/globe/proxy.js';
const SELF_PATH = path.join(__dirname, 'proxy.js');

(function checkForUpdates() {
    https.get(SELF_URL, { headers: { 'User-Agent': 'Mozilla/5.0' } }, (res) => {
        if (res.statusCode !== 200) { startProxy(); return; }
        let remote = '';
        res.on('data', (c) => remote += c);
        res.on('end', () => {
            // Extract version from remote
            const m = remote.match(/PROXY_VERSION\s*=\s*(\d+)/);
            const remoteVer = m ? parseInt(m[1], 10) : 0;
            if (remoteVer > PROXY_VERSION) {
                console.log('[proxy] v' + remoteVer + ' available (current: v' + PROXY_VERSION + ') — updating…');
                fs.writeFileSync(SELF_PATH, remote, 'utf8');
                console.log('[proxy] Updated to v' + remoteVer + '. Restarting…');
                const { spawn } = require('child_process');
                spawn('node', [SELF_PATH], { stdio: 'inherit', detached: true }).unref();
                process.exit(0);
            }
            console.log('[proxy] Up to date (v' + PROXY_VERSION + ')');
            startProxy();
        });
    }).on('error', () => { console.log('[proxy] Update check skipped (offline)'); startProxy(); });
})();
/* ═══ END SELF-UPDATE ═══════════════════════════════ */

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

createServer(function (req, res) {
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
}).listen(PORT, function () {
  console.log('\n===== CORS Proxy v' + PROXY_VERSION + ' at http://localhost:' + PORT + ' =====\n');
  console.log('Self-update: checks GitHub on startup (bump PROXY_VERSION to push updates)');
  console.log('Supported:');
  console.log('  1. Google Maps KML : /proxy?url=https://www.google.com/maps/d/...?mid=ID');
  console.log('  2. Yandex Maps     : /proxy?url=https://yandex.ru/maps/...');
  console.log('  3. Any HTTPS URL   : /proxy?url=https://example.com/data.json\n');
});

} /* end startProxy */
