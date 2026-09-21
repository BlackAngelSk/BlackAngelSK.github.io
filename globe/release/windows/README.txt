Globe/Map Server - Windows
==========================

QUICK START
-----------
  1. Unzip the whole folder somewhere (keep all files together - the
     certificate files .proxy-cert.pem / .proxy-key.pem must sit next to
     globe-server.exe).
  2. Double-click  globe-server.exe
  3. A console window opens and your browser opens http://localhost:8000
  4. ONE-TIME step so the map on https://blackangelsk.github.io works:
       - open  https://localhost:8443/ping  in the same browser
       - the browser warns about a self-signed certificate
       - click "Advanced" -> "Continue to localhost (unsafe)"
       - you should see "pong"
  5. Import a Google Maps / Yandex Maps link in the map - it now works.


WHY STEP 4 IS NEEDED
--------------------
The map is published over HTTPS. Browsers refuse to let an HTTPS page call
http://localhost:8080 (mixed content), so the proxy also listens on
HTTPS 8443 with a self-signed certificate, and that certificate must be
accepted once. Without this the proxy prints "running" but the page silently
imports nothing - that is the exact symptom of a missing HTTPS listener.

Both endpoints must be listed when the server starts:

    Proxy:   http://localhost:8080/ping
    Proxy:   https://localhost:8443/ping   <- use this from the website

If the second line is missing, the server printed
"WARNING: HTTPS 8443 is NOT running" - see TROUBLESHOOTING below.


PORTS
-----
  8080  CORS proxy (HTTP)
  8443  CORS proxy + files (HTTPS, self-signed)
  8000  local copy of the map/globe


USAGE
-----
  globe-server.exe                  Start proxy + local file server
  globe-server.exe --proxy          Start only the proxy
  globe-server.exe --port 3000      Custom local-server port
  globe-server.exe --help           Show help

Test the proxy in a browser:  http://localhost:8080/ping
                              https://localhost:8443/ping   -> "pong"


TROUBLESHOOTING
---------------
"Proxy is running, but nothing is imported"
  * The page is HTTPS and the HTTPS listener is not up. Re-check that the
    startup banner shows https://localhost:8443 - if not, the certificate
    is missing.
  * Fix: open https://localhost:8443/ping and accept the certificate.
  * Still failing? Check that the two hidden files .proxy-cert.pem and
    .proxy-key.pem are present next to globe-server.exe (Windows Explorer:
    View -> Show -> Hidden items). If they are missing, the server downloads
    them from the project's GitHub repo on startup - that needs internet.

Port already in use / server will not start
  * The server kills leftovers automatically. If that fails, run in cmd:
      netstat -ano | findstr ":8080"
      taskkill /PID <pid> /F
  * Note the exe resolves files relative to its own folder, so start it from
    the extracted folder rather than a shortcut with another working dir.

"Windows protected your PC" / SmartScreen
  * Click "More info" -> "Run anyway" (the exe is not code-signed).

Prefer Node.js (no exe)
  * With Node.js 18+ installed:
      node globe-server.js
      node proxy.js         (proxy only - 8080 + 8443)
      node start.js         (proxy 8080 + file server 8000)
    proxy.js updates itself from GitHub on every start.


FILES
-----
  globe-server.exe / globe-server.js   all-in-one server (proxy + files)
  proxy.js                             standalone proxy (8080 + 8443)
  map.html / map-new.html              the maps
  .proxy-cert.pem / .proxy-key.pem     self-signed localhost certificate