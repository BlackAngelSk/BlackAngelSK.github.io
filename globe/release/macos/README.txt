Globe/Map Server - macOS
========================

QUICK START (one command)
-------------------------
    ./start-mac.command        <- double-click this in Finder, or:
    ./start.sh                 <- in Terminal

Both clear the macOS quarantine flag themselves, pick the right binary for
your Mac (Apple Silicon / Intel), prefer Node.js when it is installed, and
start everything:

    Proxy:   http://localhost:8080/ping
    Proxy:   https://localhost:8443/ping   <- use this one from the website
    Map:     http://localhost:8000/map.html

Then open the map (https://blackangelsk.github.io/globe/map.html) and import
a Google Maps / Yandex Maps link.

ONE-TIME step for the hosted map
--------------------------------
The website is HTTPS and a browser refuses to let an HTTPS page call
http://localhost (mixed content), so the proxy also listens on HTTPS 8443 with
a self-signed certificate. Accept that certificate once:

    open https://localhost:8443/ping      ->  "Advanced" -> "Proceed to
                                              localhost (unsafe)"  -> "pong"

Without this the proxy prints "running" but the map silently imports nothing.

IF SOMETHING DOES NOT WORK
--------------------------
"A port is already in use" / EADDRINUSE
  * An earlier copy of the proxy is usually still running. The current version
    detects it, stops it and takes the port over by itself — just start again.
  * Another program on the port is reported and left alone; only 8443 matters
    for the hosted map. To free a port manually:
        lsof -ti :8080 | xargs kill -9
        lsof -ti :8443 | xargs kill -9

"The binary will not open / cannot be verified"
  * Gatekeeper quarantine. ./start.sh does this automatically; otherwise:
        xattr -dr com.apple.quarantine .
        chmod +x globe-server-arm64.bin

"Proxy is running, but nothing is imported"
  * The banner must list BOTH http://localhost:8080 and
    https://localhost:8443 — otherwise port 8443 is busy (see above).
  * Open https://localhost:8443/ping and accept the certificate.
  * Node.js alternative (identical server, no Gatekeeper involved):
        brew install node
        node globe-server.js

macOS openssl is LibreSSL
  * /usr/bin/openssl on older macOS has no -addext and cannot put
    subjectAltName into a self-signed certificate; Chrome and Safari reject
    such a certificate. The server detects this, throws the bad certificate
    away and installs its built-in one. Keep .proxy-cert.pem and
    .proxy-key.pem next to the server and nothing has to be generated.

RUNNING WITHOUT THE BINARY
--------------------------
    node globe-server.js            all-in-one (proxy + files)
    node globe-server.js --proxy    proxy only
    node globe-server.js --port 3000
    node proxy.js                   proxy only, self-updates from GitHub

PORTS
-----
  8080  CORS proxy (HTTP)
  8443  CORS proxy + files (HTTPS, self-signed)
  8000  local copy of the map/globe

The servers listen on the loopback interface only (127.0.0.1 and ::1), so
macOS never asks about incoming connections and nothing is exposed to your
local network.
