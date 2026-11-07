# 🌍 Earth Globe — TODO

Feature ideas and improvements for globe/index.html.

Last upgrade: **v2 — live layers pack** (cursor coordinates, keyboard control, graticule,
day/night terminator with a real sun position, flight arcs, live ISS + earthquakes,
great-circle measurement, compass, shareable URL state, PNG export, FPS meter).

---

## 🔥 High Priority

- [x] **Coordinates Display** — Cursor lat/lon in real time (bottom-left HUD); shows the
      great-circle distance and bearing to the first measure point while measuring
- [x] **Keyboard Shortcuts** — Arrows spin/tilt, +/- zoom, digits toggle layers, `?` help sheet
- [x] **Equator / Tropics / Arctic Circles** — Graticule toggle: 0°, ±23.44°, ±66.56°, prime
      meridian + a faint 30° reference grid
- [x] **Fullscreen Button** — Toggle browser fullscreen mode
- [x] **URL State** — Camera, globe rotation and every layer live in the hash; the address
      bar is a shareable view (restored on load)
- [x] **Compass Rose** — Heading readout under the clock, rose rotates with the camera

## 🟡 Medium Priority

- [x] **ISS / Satellite Tracker** — Live position, altitude, velocity, visibility footprint
      and a ground trail from wheretheiss.at (CORS-friendly, refreshed every 5 s). Not SGP4
      from TLE — the API does the propagation for us and is honest about staleness
- [x] **Flight Path Arcs** — 20 great-circle routes between major cities with animated
      travelling dots
- [x] **Day/Night Terminator Line** — Glowing great circle from the real sub-solar point;
      the key light is moved onto that same sun direction so the shading is not a lie
- [x] **Distance Measurement** — Click two points: km, nautical miles, bearing, great-circle arc
- [ ] **Country Highlight on Hover** — Highlight the hovered country border in a different color
- [x] **Earthquake Markers** — Live USGS M4.5+ (last 24 h) as pulsing rings sized by magnitude,
      hover tooltip, colour-coded by magnitude
- [ ] **Mini-map** — Small 2D overview map in corner showing current camera view orientation
- [x] **Screenshot / Export** — Save the current globe view as PNG
- [ ] **Custom Annotations** — Drop markers with custom labels and colors, saved in localStorage
- [ ] **Favorite/Pinned Locations** — Let users pin locations to the globe (persisted in localStorage)

## 🟢 Nice to Have

- [ ] **Weather Overlay** — Toggle a weather tile layer (OpenWeatherMap or similar)
- [ ] **Ocean Currents** — Animated flowing lines showing major ocean currents
- [ ] **Tectonic Plate Boundaries** — Overlay showing plate edges
- [ ] **Population Density Heatmap** — Color-coded overlay showing population density
- [ ] **Time Zones** — Show timezone boundaries as lines
- [x] **Major Rivers** — Overlay major world river paths (19 rivers with toggle)
- [ ] **Airport Markers** — Show major international airports with IATA codes
- [ ] **GDP / HDI Choropleth** — Color countries by economic or development data
- [x] **Live Clock** — Show current UTC + local time for the point facing the camera
- [x] **Sun Position Indicator** — Sun marker + label at the true sub-solar point (with the
      terminator layer)
- [ ] **Country Flags** — Show small flag icons next to city labels
- [ ] **Wikipedia Quick Info** — On country click, fetch brief summary from Wikipedia API

## 🔴 Conflict Visualization

- [x] **Conflict Hotspot Markers** — 28 conflict locations with pulsing markers at exact coordinates
- [x] **Conflict Data Structure** — Structured data with name, lat/lon, status, severity, type, start date, parties, description
- [x] **Severity-Based Visuals** — Critical (red fast pulse), High (orange), Medium (amber), Low/Frozen (gray)
- [x] **Conflict Info Panel** — Click a marker to see full conflict details (left-side panel)
- [x] **Conflict Toggle Button** — ⚔ Conflicts button to show/hide the conflict layer
- [x] **Conflict Legend** — Color-coded legend showing severity levels
- [x] **Conflict Hover Tooltip** — Red-themed tooltip showing conflict name, severity, and status
- [x] **Click-to-Zoom** — Clicking a conflict marker zooms to its location
- [x] **Country Border Highlight** — Conflict countries shown with red-tinted borders (existing feature)

## 🔧 Technical Improvements

- [x] **LOD (Level of Detail)** — 192/128/64/32 segment spheres swapped by camera distance,
      with an on-screen LOD readout
- [ ] **Texture Streaming** — Load higher-res textures only when zoomed in
- [ ] **Web Workers** — Offload GeoJSON parsing to a web worker for smoother loading
- [x] **Performance Monitoring** — FPS counter next to the LOD readout
- [x] **Touch Gestures** — Pinch-to-zoom / two-finger rotate (OrbitControls), compact HUD on mobile
- [ ] **Right-click Context Menu** — Copy coordinates, get info, zoom to location
- [ ] **Settings Panel** — Collapsible panel grouping all toggle options instead of individual buttons
- [ ] **Lightning Strikes** — Random animated flashes in storm regions
- [ ] **Bump Map Enhancement** — Real DEM-based elevation for mountain ranges to pop visually

---

## ✅ Completed

- [x] Earth globe with day/night textures
- [x] Cloud layer with toggle
- [x] Atmosphere glow shader
- [x] Country borders from GeoJSON
- [x] 150+ city markers with labels
- [x] Star background
- [x] Auto-rotation toggle
- [x] Night mode toggle
- [x] Search cities and countries
- [x] Country click → zoom + info panel
- [x] Hover tooltips on cities
- [x] Smooth camera animations
- [x] Dynamic label scaling based on zoom
- [x] Loading screen with progress
- [x] Link to 2D map page
- [x] Conflict visualization layer with 28 hotspots, severity levels, pulsing markers, info panel, toggle, legend
- [x] Live layer pack v2 (see the list above)

## 🐞 Fixed in v2

- Clock HUD latitude was inverted (`asin` instead of `acos`) and its longitude ignored the
  globe's rotation, so the "local time for the point facing you" was wrong; it now uses the
  camera position in the globe's own frame.
- The `#lod-hud` element was being overwritten wholesale (`textContent = …`), which wiped the
  FPS readout; the LOD text now has its own node.
- The temporary "Development has been stopped" banner is gone.
- The control bar sized itself to half the viewport (fixed-position shrink-to-fit at
  `left:50%`), which stacked every button into a narrow column on phones.
