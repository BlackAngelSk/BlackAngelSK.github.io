Globe/Map Server - Linux
========================

QUICK START
-----------
  1. chmod +x globe-server
  2. ./globe-server
  3. Your browser opens http://localhost:8000
  4. ONE-TIME step so the map on https://blackangelsk.github.io works:
       open  https://localhost:8443/ping  and accept the self-signed
       certificate ("Advanced" -> "Proceed"); you should see "pong"
  5. Import a Google Maps / Yandex Maps link in the map

WHY STEP 4 IS NEEDED
--------------------
The map is published over HTTPS, and browsers refuse to let an HTTPS page call
http://localhost:8080 (mixed content). The proxy therefore also listens on
HTTPS 8443 with a self-signed certificate that must be accepted once.
Without it the proxy prints "running" but the page silently imports nothing.

Both lines must appear at start-up:

    Proxy:   http://localhost:8080/ping
    Proxy:   https://localhost:8443/ping   <- use this from the website

PORTS
-----
  8080  CORS proxy (HTTP)
  8443  CORS proxy + files (HTTPS, self-signed)
  8000  local copy of the map/globe

USAGE
-----
  ./globe-server                  proxy + local file server
  ./globe-server --proxy          proxy only
  ./globe-server --port 3000      custom local-server port
  ./globe-server --help           help

TROUBLESHOOTING
---------------
"Proxy is running, but nothing is imported"
  * Check that the start-up banner lists https://localhost:8443. If it does
    not, the certificate is missing: keep .proxy-cert.pem and .proxy-key.pem
    next to the binary (the server also regenerates them with openssl).
  * Open https://localhost:8443/ping and accept the certificate.

Port already in use
  * lsof -ti :8080 | xargs kill -9   (same for 8443 and 8000)

Node.js instead of the binary
  * node globe-server.js   — same thing
  * node proxy.js          — proxy only (8080 + 8443), self-updates from GitHub
  * node start.js          — proxy 8080 + python http.server on 8000
