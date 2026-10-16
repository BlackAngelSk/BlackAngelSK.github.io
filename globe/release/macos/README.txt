Globe/Map Server — macOS
========================

QUICK START (one double-click)
------------------------------
    start-mac.command          <- double-click this in Finder

or, in Terminal:

    ./start.sh

The script clears the macOS quarantine flag, picks the right binary for your
Mac (Apple Silicon / Intel) or uses Node.js when it is installed, and starts
everything:

    Proxy + map:  http://localhost:8080/map.html
    Health check: http://localhost:8080/ping

Your browser opens automatically. Then import a Google Maps / Yandex Maps link
in the map.

Plain HTTP — no certificate, nothing to accept.

SAFARI USERS — READ THIS
------------------------
The website is https://blackangelsk.github.io. Chrome, Edge, Firefox and Brave
treat http://localhost as a trustworthy origin, so that page may call
http://localhost:8080 directly. Safari has no such exception and blocks it.

  Fix: start with HTTPS as well, in Terminal:
      ./start.sh --https
  then open https://localhost:8443/ping once and accept the self-signed
  certificate ("Advanced" → "Continue to localhost"). You should see "pong".

IF SOMETHING DOES NOT WORK
--------------------------
"A port is already in use" / EADDRINUSE
  * An earlier copy of the server is usually still running. The current version
    detects it, stops it and takes the port over by itself — just start again.
  * Another program on the port is reported and left alone. To free it:
        lsof -ti :8080 | xargs kill -9

"The binary will not open / cannot be verified"
  * Gatekeeper quarantine. ./start.sh does this automatically; otherwise:
        xattr -dr com.apple.quarantine .
        chmod +x globe-server-arm64.bin

"Proxy is running, but nothing is imported"
  * Check http://localhost:8080/ping — it must answer "pong".
  * Safari: use --https (see above).
  * Node.js alternative (no Gatekeeper involved):
        brew install node
        node globe-server.js

Nothing happens when double-clicking
  * Open Terminal in this folder and run  ./start.sh  to read the message.

RUNNING WITHOUT THE START SCRIPT
--------------------------------
    node globe-server.js            all-in-one (proxy + files)
    node globe-server.js --proxy    proxy only
    node globe-server.js --port 3000
    node proxy.js                   proxy only, self-updates from GitHub

PORTS
-----
  8080  proxy + local copy of the map   (http://localhost:8080/map.html)
  8000  second static server            (http://localhost:8000/map.html)
  8443  only with --https               (https://localhost:8443/ping)

The servers listen on the loopback interface only (127.0.0.1 and ::1), so
macOS never asks about incoming connections and nothing is exposed to your
local network.
