const { createServer } = require('http');
const https = require('https');

const PORT = 8080;

/**
 * proxyFetch - Fetches a URL with redirect following.
 * Returns a Promise that resolves with { statusCode, headers, data }.
 */
function proxyFetch(url, maxRedirects = 5) {
  return new Promise((resolve, reject) => {
    if (maxRedirects <= 0) return reject(new Error('Too many redirects'));

    const mod = url.startsWith('https') ? https : require('http');
    const req = mod.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        let redirectUrl = res.headers.location;
        // Handle relative redirects
        if (redirectUrl.startsWith('/')) {
          const base = new URL(url);
          redirectUrl = base.origin + redirectUrl;
        }
        res.resume(); // Consume response to free memory
        return proxyFetch(redirectUrl, maxRedirects - 1).then(resolve).catch(reject);
      }
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        resolve({
          statusCode: res.statusCode,
          headers: res.headers,
          data: Buffer.concat(chunks),
        });
      });
      res.on('error', reject);
    });
    req.on('error', reject);
    req.setTimeout(15000, () => {
      req.destroy();
      reject(new Error('Request timed out'));
    });
  });
}

/**
 * Detects the expected content type based on the URL pattern.
 */
function detectContentType(targetUrl) {
  if (/google\.com\/maps\/d\//i.test(targetUrl)) {
    return 'application/vnd.google-earth.kml+xml; charset=utf-8';
  }
  if (/yandex\.(ru|com).*maps/i.test(targetUrl)) {
    return 'text/html; charset=utf-8';
  }
  // Default: let the remote server decide, but prefer application/octet-stream
  // so the browser treats it as a downloadable/parseable resource.
  return null;
}

createServer((req, res) => {
  // CORS headers for all responses
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const parsedUrl = new URL(req.url, 'http://localhost');
  const targetUrl = parsedUrl.searchParams.get('url');

  if (!targetUrl) {
    res.writeHead(400, { 'Content-Type': 'text/plain' });
    res.end(
      'Missing ?url= parameter.\n\n' +
      'Usage examples:\n' +
      '  Google Maps KML:  /proxy?url=https://www.google.com/maps/d/...?mid=YOUR_MAP_ID\n' +
      '  Yandex Maps:      /proxy?url=https://yandex.ru/maps/...\n' +
      '  Any HTTPS URL:    /proxy?url=https://example.com/data.json\n'
    );
    return;
  }

  // Validate HTTPS
  if (!targetUrl.startsWith('https://')) {
    res.writeHead(400, { 'Content-Type': 'text/plain' });
    res.end('Only HTTPS URLs are supported. Received: ' + targetUrl.slice(0, 200));
    return;
  }

  console.log('Proxying:', targetUrl.slice(0, 150));

  proxyFetch(targetUrl)
    .then(({ statusCode, headers, data }) => {
      // Determine content type
      let contentType = detectContentType(targetUrl);
      if (!contentType) {
        // Fall back to what the remote server reported
        contentType = headers['content-type'] || 'application/octet-stream';
      }

      const responseHeaders = {
        'Content-Type': contentType,
        'Content-Length': data.length,
        'Access-Control-Allow-Origin': '*',
      };

      res.writeHead(statusCode, responseHeaders);
      res.end(data);
    })
    .catch((err) => {
      console.error('Proxy error:', err.message);
      res.writeHead(502, { 'Content-Type': 'text/plain' });
      res.end('Proxy fetch failed: ' + err.message);
    });
}).listen(PORT, () => {
  console.log(`\n===== CORS Proxy running at http://localhost:${PORT} =====\n`);
  console.log('Supported URL types:');
  console.log(`  1. Google Maps KML : /proxy?url=https://www.google.com/maps/d/...?mid=YOUR_MAP_ID`);
  console.log(`  2. Yandex Maps     : /proxy?url=https://yandex.ru/maps/...`);
  console.log(`  3. Any HTTPS URL   : /proxy?url=https://example.com/data.json`);
  console.log(`\nExample for map import:`);
  console.log(`  http://localhost:${PORT}/proxy?url=YOUR_KML_URL_HERE`);
  console.log('');
});
