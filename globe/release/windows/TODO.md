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

## Bugs fixed on the way (map-new)

- Areas and circles now survive the export → import round trip: a GeoJSON `Polygon` used to be flattened
  into outline lines on import, and the polygon import branch itself misread the ring array.
- `fire` annotations were exported as a one-point `LineString` (invalid GeoJSON) — now a `Point`.
- Export and "save map" used two copies of the same converter; both use `annToFeature()` now, so they can
  no longer drift apart.

## Next steps (not implemented yet)

- Port the five shipped features from map-new.html back to map.html after a visual pass on all 9 themes.
- Share via URL (encode the `annToFeature()` output in the location hash — the converter already exists for it).
- Elevation profile for the measure tool (OpenTopoData or a public DEM API).
- GPS / my-location button (`navigator.geolocation`, works over https or localhost).
