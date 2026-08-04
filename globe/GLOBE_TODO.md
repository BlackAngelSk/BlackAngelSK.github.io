# 🌍 Earth Globe — TODO

Feature ideas and improvements for globe/index.html.

---

## 🔥 High Priority

- [ ] **Coordinates Display** — Show lat/lon of cursor position in real-time (bottom-left HUD)
- [ ] **Keyboard Shortcuts** — Arrow keys to rotate, +/- to zoom, number keys to toggle layers
- [ ] **Equator / Tropics / Arctic Circles** — Toggle guide lines (great circles at 0°, ±23.5°, ±66.5°)
- [ ] **Fullscreen Button** — Toggle browser fullscreen mode
- [ ] **URL State** — Encode camera position, active layers in URL hash so views are shareable via link
- [ ] **Compass Rose** — Show N/S/E/W orientation indicator that updates as globe rotates

## 🟡 Medium Priority

- [ ] **ISS / Satellite Tracker** — Show the International Space Station orbiting the globe in real-time using TLE data + SGP4 propagation
- [ ] **Flight Path Arcs** — Curved arcs between major cities showing airline routes with animated dots traveling along them
- [ ] **Day/Night Terminator Line** — A glowing line showing where sunrise/sunset currently is on Earth
- [ ] **Distance Measurement** — Click two points to measure great-circle distance between them
- [ ] **Country Highlight on Hover** — Highlight the hovered country border in a different color
- [ ] **Earthquake Markers** — Live data from USGS API showing recent seismic activity as pulsing dots
- [ ] **Mini-map** — Small 2D overview map in corner showing current camera view orientation
- [ ] **Screenshot / Export** — Save the current globe view as PNG
- [ ] **Custom Annotations** — Drop markers with custom labels and colors, saved in localStorage
- [ ] **Favorite/Pinned Locations** — Let users pin locations to the globe (persisted in localStorage)

## 🟢 Nice to Have

- [ ] **Weather Overlay** — Toggle a weather tile layer (OpenWeatherMap or similar)
- [ ] **Ocean Currents** — Animated flowing lines showing major ocean currents
- [ ] **Tectonic Plate Boundaries** — Overlay showing plate edges
- [ ] **Population Density Heatmap** — Color-coded overlay showing population density
- [ ] **Time Zones** — Show timezone boundaries as lines
- [ ] **Major Rivers** — Overlay major world river paths
- [ ] **Airport Markers** — Show major international airports with IATA codes
- [ ] **GDP / HDI Choropleth** — Color countries by economic or development data
- [x] **Live Clock** — Show current UTC + local time for the point facing the camera
- [ ] **Sun Position Indicator** — Show where the sun is right now relative to Earth
- [ ] **Country Flags** — Show small flag icons next to city labels
- [ ] **Wikipedia Quick Info** — On country click, fetch brief summary from Wikipedia API

## 🔧 Technical Improvements

- [ ] **LOD (Level of Detail)** — Reduce polygon count when zoomed out, increase when zoomed in
- [ ] **Texture Streaming** — Load higher-res textures only when zoomed in
- [ ] **Web Workers** — Offload GeoJSON parsing to a web worker for smoother loading
- [ ] **Performance Monitoring** — FPS counter toggle in corner
- [ ] **Touch Gestures** — Pinch-to-zoom, two-finger rotate for mobile
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