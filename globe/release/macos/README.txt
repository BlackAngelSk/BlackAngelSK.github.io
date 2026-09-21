Globe/Map Server - macOS
========================

QUICK START
-----------
  1. ./globe-server-arm64.bin   (Apple Silicon)
     ./globe-server-x64.bin     (Intel)
  2. Your browser opens http://localhost:8000
  3. ONE-TIME step so the map on https://blackangelsk.github.io works:
       open  https://localhost:8443/ping  and accept the self-signed
       certificate ("Show Details" -> "visit this website"); expect "pong"
  4. Import a Google Maps / Yandex Maps link in the map

First run may be blocked by Gatekeeper:
  System Settings > Privacy & Security > "Allow Anyway", or right-click the
  binary > Open.

IMPORTANT for Apple Silicon (M1/M2/M3): the binaries carry an ad-hoc code
signature, which is the minimum macOS requires to launch them. Because they
came from the internet, macOS still flags them as quarantined, so clear that
flag once:

    xattr -dr com.apple.quarantine .
    ./globe-server-arm64.bin

If macOS still refuses to start the binary, run the same server through
Node.js instead:

    brew install node        # if node is missing
    node globe-server.js     # or simply: ./start.sh

WHY STEP 3 IS NEEDED
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
  ./globe-server-arm64.bin --proxy        proxy only
  ./globe-server-arm64.bin --port 3000    custom local-server port
  ./globe-server-arm64.bin --help         help

TROUBLESHOOTING
---------------
"Proxy is running, but nothing is imported"
  * Make sure the banner lists https://localhost:8443. If not, keep
    .proxy-cert.pem and .proxy-key.pem next to the binary (the server can also
    regenerate them with openssl, which ships with macOS).
  * Open https://localhost:8443/ping and accept the certificate.

Port already in use
  * lsof -ti :8080 | xargs kill -9   (same for 8443 and 8000)

Node.js instead of the binary
  * node globe-server.js   — same thing
  * node proxy.js          — proxy only (8080 + 8443), self-updates from GitHub
