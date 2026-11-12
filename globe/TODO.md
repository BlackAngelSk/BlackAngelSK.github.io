# 🗺️ Interactive Map — TODO

Feature ideas and improvements for globe/map.html.

Legend: `[x]` = done on map.html · `[~]` = implemented and tested on **map-new.html** only (still to be ported)

---

## 🔥 High Priority

- [~] **Polygon/Fill Tool** — Draw filled regions (zones, areas of interest) with configurable fill color and opacity (L.polygon) · live area readout · closes on double-click / Enter / Finish
- [~] **Circle Tool** — Draw radius-based circles from a center point (L.circle), show radius distance · live radius + area while sizing
- [~] **Annotation List Panel** — Modal listing all annotations: zoom to, show/hide, rename, delete, zoom-to-fit-all, export
- [~] **Geocoding / Search Bar** — Search for a place by name and fly to it (Nominatim, no API key needed)

## 🟡 Medium Priority

- [x] **Right-click Context Menu** — Copy coordinates, delete annotation, edit properties, zoom to location (ghost arrow)
- [~] **Coordinate Display on Hover** — Show lat/lng + zoom in the status bar as the cursor moves (click to copy)
- [ ] **Annotation Editing** — Click existing annotations to change color, weight, or label text after creation (renaming exists in the list panel)
- [ ] **Multi-select and Group** — Select multiple annotations and move/rotate/delete as a group
- [x] **Image Overlay** — Attach custom images to map regions (screenshots, old maps, overlays)
- [ ] **Snap-to-Grid** — Optionally snap points to a grid or to existing vertices for precision drawing
- [ ] **Print-friendly / PDF Export** — Generate a clean printable version without UI chrome

## 🟢 Nice to Have

- [x] **Fullscreen Mode** — One-click fullscreen toggle (hide browser chrome)
- [ ] **GPS / My Location** — Button to center map on the user real-time GPS position
- [ ] **Weather/Radar Overlay** — Toggle a weather or precipitation tile layer
- [ ] **Share via URL** — Encode annotations in the URL hash/query so annotated views can be shared via link
- [ ] **Elevation Profile** — Show elevation chart when measuring distance
- [ ] **Drawing Replay** — Record drawing actions and replay them as a timelapse
- [ ] **Dark/Light Theme Toggle** — Switch UI between the current dark theme and a light theme

---

## 🔁 Parity — what map.html had and map-new.html was missing (now ported)

map-new.html is now a superset of map.html. Ported from `map.js`:

- [x] **NATO member overlay** — `#btn-nato`, 32 members, blue outline + fill, hover name, centroid labels
- [x] **EU member overlay** — `#btn-eu`, amber; 26 of 27 match the border data (Malta is not in `world.geojson`)
- [x] **Yandex Maps import** — `ll / z / spn / text / pt / rtext` URL parsing → pins, route polyline + waypoints,
      plus `?um=constructor:HASH` user maps fetched through the local proxy and parsed out of the page
- [x] **Import naming** — `#import-name-input` in the preview, `window._pendingImportName`, auto-name from
      preset / saved map / source kind ("Google Maps KML", "Yandex Map", "GPX Import", "GeoJSON Import")
- [x] **Imported Maps panel** — one row per import with show/hide checkbox, feature list with colour dots and
      Collapse All; it replaced map-new's ad-hoc second `L.control.layers` (which duplicated the base-layer control)
- [x] **Single-click country select** — click selects (300 ms debounce), double click still selects and drills into
      regions; border weights/hover and the `_hiddenBySel` guard restored to the map.html values

Still different on purpose: map.html keeps the draggable floating toolbar + bottom bar layout, map-new keeps the
top bar + right sidebar with 9 themes (classic, minimal, sidebar-left, dark-pro, cyber, terminal, daylight, amber, glass).

## Bugs fixed on the way (map-new)

- Areas and circles now survive the export → import round trip: a GeoJSON `Polygon` used to be flattened
  into outline lines on import, and the polygon import branch itself misread the ring array.
- `fire` annotations were exported as a one-point `LineString` (invalid GeoJSON) — now a `Point`.
- Export and "save map" used two copies of the same converter; both use `annToFeature()` now, so they can
  no longer drift apart.
- **NATO/EU labels never rendered** (fixed in both pages): `_polygonCentroid()` called `layer.eachLayer()`, but
  `onEachFeature` hands over the `L.Polygon` itself, so it threw for every country and the `try { … } catch (_) {}`
  around the label code swallowed it. It now accepts a group *or* a single path.
- **"Import Selected" could not be clicked** with a long preview list: the import modal had no `max-height`
  and the preview list no scroll cap, so the button was pushed below the fold (a 20-feature preset put it at
  y=1030 on a 900 px window). The modal now caps at 90vh, the list scrolls at 280 px and the action row is sticky.
- **KML polygons aborted the whole import**: `confirmImport` assumed the nested GeoJSON ring
  (`coordinates[0]`), while KML hands over a flat ring, so `L.polygon()` threw, the modal stayed open and the
  partially imported features were never registered. Both shapes are accepted now, each feature is wrapped in
  its own try/catch, and a failure can no longer stop the modal from closing.
- **Unnamed KML was named after its first pin**: `getElementsByTagName('name')[0]` walked into the first
  `<Placemark>`, so an unnamed document pre-filled the import name with e.g. "Pin A". Only a direct child
  `<name>` counts now.
- **Base map switcher was invisible in map-new**: the Leaflet layers control sat at `topright`, under the
  fixed top bar and behind the theme-switcher button (z-index 1000), so Standard/Topographic/Satellite could
  not be reached. It is now a "Base map" section in the sidebar (the control itself is gone).

## Next steps (not implemented yet)

- Port the five shipped features from map-new.html back to map.html after a visual pass on all 9 themes.
- Share via URL (encode the `annToFeature()` output in the location hash — the converter already exists for it).
- Elevation profile for the measure tool (OpenTopoData or a public DEM API).
- GPS / my-location button (`navigator.geolocation`, works over https or localhost).
