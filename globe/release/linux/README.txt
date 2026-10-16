Globe/Map Server — Linux
========================

QUICK START
-----------
  1. Keep all the files in this folder together
  2. Run:   ./start.sh
  3. The proxy starts and your browser opens http://localhost:8080/map.html
  4. Import a Google Maps / Yandex Maps link in the map

That is all. The server runs over plain HTTP — no certificate, no browser
warning to click through, nothing to install.

No Node.js on this machine? The folder also contains a self-contained binary
(globe-server); start.sh uses it automatically when node is missing.

PORTS
-----
  8080  proxy + local copy of the map   (http://localhost:8080/map.html)
  8000  second static server            (http://localhost:8000/map.html)

WHY NO HTTPS
------------
The map is published as https://blackangelsk.github.io. Chrome, Edge, Firefox
and Brave treat http://localhost as a trustworthy origin, so that page may
call http://localhost:8080 directly — no mixed-content block, no certificate.
Safari has no such exception: there run  ./start.sh --https  and open
https://localhost:8443/ping once to accept the self-signed certificate.

USAGE
-----
  ./start.sh                 proxy + local file server (opens the browser)
  ./start.sh --proxy         proxy only
  ./start.sh --no-open       do not open the browser
  ./start.sh --https         also listen on https://localhost:8443 (Safari)
  ./start.sh --port 3000     custom local-server port
  ./start.sh --help          help

  With Node.js you can also run the server directly:
    node globe-server.js     same options as above
    node proxy.js            the proxy alone (self-updates from GitHub)
    node start.js            same as globe-server.js

TROUBLESHOOTING
---------------
"Proxy is running, but nothing is imported"
  * Use Chrome, Edge, Firefox or Brave (Safari: see WHY NO HTTPS above).
  * Open http://localhost:8080/ping — it must answer "pong".
  * Reload the map page, then import again.

Port already in use
  * The server replaces an older copy of itself automatically. If a different
    program holds the port:  lsof -ti :8080 | xargs kill -9

"Permission denied" when running start.sh
  * chmod +x start.sh globe-server
