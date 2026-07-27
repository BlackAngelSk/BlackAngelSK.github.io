const { createServer } = require('http');
const https = require('https');

const PORT = 8080;

createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET');
    if (req.method === 'OPTIONS') { res.writeHead(200); res.end(); return; }

    const url = new URL(req.url, 'http://localhost').searchParams.get('url');
    if (!url || !url.startsWith('https://www.google.com/maps/d/')) {
        res.writeHead(400); res.end('Missing or invalid ?url= parameter'); return;
    }

    console.log('Proxying:', url.slice(0, 120));
    https.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' } }, proxyRes => {
        if (proxyRes.statusCode >= 300 && proxyRes.statusCode < 400 && proxyRes.headers.location) {
            https.get(proxyRes.headers.location, { headers: { 'User-Agent': 'Mozilla/5.0' } }, finalRes => {
                res.writeHead(finalRes.statusCode, { 'Content-Type': 'text/xml; charset=utf-8' });
                finalRes.pipe(res);
            }).on('error', e => { res.writeHead(502); res.end('Redirect failed: ' + e.message); });
            return;
        }
        res.writeHead(proxyRes.statusCode, { 'Content-Type': 'text/xml; charset=utf-8' });
        proxyRes.pipe(res);
    }).on('error', e => { res.writeHead(502); res.end('Fetch failed: ' + e.message); });
}).listen(PORT, () => console.log('KML proxy running at http://localhost:' + PORT + '\nPaste this URL in the map\'s Import → URL tab:\n  http://localhost:' + PORT + '/kml?mid=YOUR_MAP_ID'));
