Globe/Map Server — Windows
==========================

QUICK START
-----------
  1. Unzip the whole folder somewhere (keep all the files together).
  2. Double-click  start.bat
  3. The proxy starts and your browser opens http://localhost:8080/map.html
  4. Import a Google Maps / Yandex Maps link in the map

That is all. The server runs over plain HTTP — no certificate, no browser
warning to click through, nothing to install.

No Node.js on this machine? The folder also contains globe-server.exe;
start.bat uses it automatically when node is missing.

PORTS
-----
  8080  proxy + local copy of the map   (http://localhost:8080/map.html)
  8000  second static server            (http://localhost:8000/map.html)

WHY NO HTTPS
------------
The map is published as https://blackangelsk.github.io. Chrome, Edge, Firefox
and Brave treat http://localhost as a trustworthy origin, so that page may
call http://localhost:8080 directly — no mixed-content block, no certificate,
nothing to accept. Safari is not available on Windows, so --https is never
needed here.

USAGE
-----
  start.bat                          proxy + local file server + browser
  globe-server.exe --proxy           proxy only
  globe-server.exe --port 3000       custom local-server port
  globe-server.exe --help            help

  With Node.js 18+ installed you can run the same server either way:
    node globe-server.js             proxy + local file server
    node start.js                    same thing
    node proxy.js                    the proxy alone (self-updates from GitHub)

Test the proxy in a browser:  http://localhost:8080/ping  ->  "pong"

TROUBLESHOOTING
---------------
"Proxy is running, but nothing is imported"
  * Open http://localhost:8080/ping — it must answer "pong".
  * Reload the map page, then import again.

Port already in use
  * The server replaces an older copy of itself automatically. If a different
    program holds the port, run in cmd:
        netstat -ano | findstr ":8080"      then     taskkill /PID <pid> /F
  * The exe resolves files relative to its own folder — start it from the
    extracted folder rather than a shortcut with another working directory.

"Windows protected your PC" / SmartScreen
  * Click "More info" -> "Run anyway" (the exe is not code-signed).

The window flashes and closes
  * start.bat keeps itself open and prints the reason. For the full message
    open cmd, cd to this folder and run  start.bat.

FILES
-----
  start.bat                            the launcher (double-click this)
  globe-server.exe / globe-server.js   all-in-one server (proxy + files)
  proxy.js                             standalone proxy (port 8080)
  setup-proxy.bat / .js / .sh          auto-install Node.js, then start
  map.html / map-new.html              the maps
