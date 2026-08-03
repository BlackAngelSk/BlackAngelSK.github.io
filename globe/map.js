/* =====================================================
   Interactive Detailed Map with Custom Line Annotations
   ===================================================== */
// @ts-nocheck
/* global L, html2canvas */
'use strict';

/* ── State ────────────────────────────────────────── */
const S = {
    tool: 'pan',
    color: '#ff0000',
    weight: 3,
    dashStyle: 'solid',
    drawing: false,
    annotations: [],
    nextId: 1,
    /* freehand */
    fhPoints: [],
    fhLine: null,
    /* polyline */
    plPoints: [],
    plLine: null,
    plPreview: null,
    /* eraser */
    eraserHandlers: [],
    eraserDragging: false,
    eraserErasedIds: new Set(),
    /* touch fallback */
    lastLatLng: null,
    /* undo/redo */
    undoStack: [],
    redoStack: [],
    /* arrow */
    arrowStart: null,
    arrowLine: null,
    arrowHead: null,
    arrowDrawing: false,
    flag: null,
    /* measure */
    msPoints: [],
    msLine: null,
    msLabels: [],
    msPreview: null,
    /* label input */
    labelLatLng: null,
    /* region highlight */
    highlighted: null,
    highlightedOrigStyle: null
};
const UNDO_LIMIT = 50;

const COLORS = [
    '#ff0000', '#ff8800', '#ffdd00', '#00cc44', '#00bbff',
    '#0066ff', '#9900ff', '#ff00aa', '#ffffff', '#333333'
];
const DASH = { solid: null, dashed: '12, 8', dotted: '4, 8' };
const STATUS = {
    pan: 'Ready — Pan mode',
    freehand: 'Click and drag to draw',
    polyline: 'Click to add points · Dbl-click / Enter to finish · Esc to cancel',
    arrow: 'Click and drag to draw an arrow',
    marker: 'Click to drop a pin on the map',
    label: 'Click to place a text label',
    measure: 'Click to measure distance · Dbl-click / Enter to finish · Esc to cancel',
    eraser: 'Click an annotation to erase it',
    fire: 'Click to place a fire on the map'
};

/* ── DOM shortcuts ────────────────────────────────── */
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

/* ── Map init ─────────────────────────────────────── */
const canvasRenderer = L.canvas({ padding: 0.5 });
const map = L.map('map', {
    center: [48.15, 17.11],
    zoom: 6,
    minZoom: 2,
    maxZoom: 19,
    zoomControl: true,
    doubleClickZoom: false,
    preferCanvas: true
});

/* Keep Leaflet's viewport in sync when the browser viewport or the map
   container changes size (fullscreen, mobile browser chrome, split view,
   or a resized window). */
const mapElement = document.getElementById('map');
let mapResizeFrame = 0;
function requestMapResize() {
    if (mapResizeFrame) cancelAnimationFrame(mapResizeFrame);
    mapResizeFrame = requestAnimationFrame(() => {
        mapResizeFrame = 0;
        map.invalidateSize({ pan: false, debounceMoveend: true });
        if (typeof clampDraggablePanels === 'function') clampDraggablePanels();
    });
}
if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(requestMapResize).observe(mapElement);
}
window.addEventListener('resize', requestMapResize);
window.addEventListener('orientationchange', requestMapResize);
if (window.visualViewport) window.visualViewport.addEventListener('resize', requestMapResize);

/* ── Tile layers (zoom-dependent density) ─────────── */
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, crossOrigin: true,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
});
const topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    maxZoom: 17, crossOrigin: true,
    attribution: '&copy; <a href="https://opentopomap.org">OpenTopoMap</a>'
});
const sat = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 18, crossOrigin: true,
    attribution: '&copy; <a href="https://www.esri.com">Esri</a>'
});
/* ── Fullscreen Button (injected into zoom control) ─── */
map.whenReady(function () {
    var zoomCtrl = document.querySelector('#map .leaflet-control-zoom');
    if (zoomCtrl) {
        var btn = document.createElement('button');
        btn.id = 'btn-fullscreen';
        btn.className = 'leaflet-control-fullscreen';
        btn.title = 'Toggle fullscreen (F)';
        btn.textContent = '\u26F6';
        zoomCtrl.appendChild(btn);
    }
});

osm.addTo(map);
L.control.layers({
    '\u{1F5FA} Standard': osm,
    '\u{1F3D4} Topographic': topo,
    '\u{1F6F0} Satellite': sat
}, null, { position: 'topright' }).addTo(map);

/* ── Map Label Languages ─────────────────────────── */
const LANGUAGES = [
    { code: 'EN', flag: '🇬🇧', url: 'https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png', max: 19, attr: '© CARTO © OSM' },
    { code: 'SK', flag: '🇸🇰', url: 'https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png', max: 19, attr: '© CARTO © OSM' },
    { code: 'DE', flag: '🇩🇪', url: 'https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png', max: 19, attr: '© CARTO © OSM' },
    { code: 'JP', flag: '🇯🇵', url: 'https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png', max: 19, attr: '© CARTO © OSM' },
    { code: 'OFF', flag: '🚫', url: '', max: 19, attr: '' }
];
let langIdx = -1;
let langOverlay = null;

function setMapLanguage(idx) {
    langIdx = idx % LANGUAGES.length;
    const lang = LANGUAGES[langIdx];
    if (langOverlay) { map.removeLayer(langOverlay); langOverlay = null; }
    if (lang.url) {
        langOverlay = L.tileLayer(lang.url, { maxZoom: lang.max, crossOrigin: true, attribution: lang.attr, pane: 'overlayPane' });
        langOverlay.addTo(map);
    }
    $('#btn-lang').textContent = lang.flag + ' ' + lang.code;
}
$('#btn-lang').addEventListener('click', () => setMapLanguage(langIdx + 1));

/* Use the base map's native labels by default. Load the optional label overlay only on demand. */
$('#btn-lang').textContent = LANGUAGES[0].flag + ' ' + LANGUAGES[0].code;

/* =====================================================
   Tool Management
   ===================================================== */
function setTool(tool) {
    /* finish any in-progress drawing */
    if (S.tool === 'polyline' && S.plPoints.length > 0) finishPolyline();
    if (S.tool === 'freehand' && S.drawing) finishFreehand();
    if (S.tool === 'arrow' && S.arrowDrawing) finishArrow();
    if (S.tool === 'measure' && S.msPoints.length > 0) finishMeasure();

    /* hide label input if switching away */
    if (S.tool === 'label') hideLabelInput();

    /* leave previous tool */
    if (S.tool === 'eraser') disableEraser();

    S.tool = tool;
    S.drawing = false;

    /* map dragging — disabled for eraser so drag-to-erase works */
    if (tool === 'pan') map.dragging.enable();
    else map.dragging.disable();

    /* enter new tool */
    if (tool === 'eraser') enableEraser();

    /* UI */
    $$('.tool-btn').forEach(b => b.classList.toggle('active', b.dataset.tool === tool));
    $('#status-text').textContent = STATUS[tool];
    const icons = {
        pan: '\u{1F4CD}', freehand: '\u270F\uFE0F', polyline: '\u{1F4D0}',
        arrow: '\u27A1\uFE0F', marker: '\u{1F4CC}', label: '\u{1F3F7}\uFE0F', measure: '\u{1F4CF}', eraser: '\u{1F9F9}'
    };
    $('#status-icon').textContent = icons[tool] || '\u{1F4CD}';
    Array.from(document.body.classList)
        .filter(className => className.indexOf('tool-') === 0)
        .forEach(className => document.body.classList.remove(className));
    document.body.classList.add('tool-' + tool);
    if (tool !== 'polyline' && tool !== 'measure') $('#btn-finish').classList.add('hidden');
    /* show/hide flag panel when arrow tool is active */
    const flagPanel = $('#flag-panel');
    if (flagPanel) flagPanel.classList.toggle('hidden', tool !== 'arrow');
    updateUndoRedoUI();
}

/* =====================================================
   Freehand Drawing
   ===================================================== */
function startFreehand(e) {
    if (S.tool !== 'freehand') return;
    S.drawing = true;
    S.fhPoints = [[e.latlng.lat, e.latlng.lng]];
    S.fhLine = L.polyline(S.fhPoints, {
        color: S.color, weight: S.weight,
        dashArray: DASH[S.dashStyle], opacity: 1,
        lineCap: 'round', lineJoin: 'round',
        smoothFactor: 1.5
    }).addTo(map);
}

function moveFreehand(e) {
    if (!S.drawing || !S.fhLine) return;
    S.lastLatLng = e.latlng;
    S.fhPoints.push([e.latlng.lat, e.latlng.lng]);
    S.fhLine.addLatLng(e.latlng);
}

function finishFreehand() {
    if (!S.fhLine) { resetFreehand(); return; }
    if (S.fhPoints.length < 2) {
        map.removeLayer(S.fhLine);
        resetFreehand();
        return;
    }
    /* smooth the freehand stroke */
    const simplified = rdpSimplify(S.fhPoints, 0.00005);
    const smoothed = simplified.length >= 3
        ? catmullRom(simplified, 8) : simplified;
    S.fhLine.setLatLngs(smoothed);
    storeAnnotation(S.fhLine, 'freehand', smoothed);
    resetFreehand();
}

function resetFreehand() {
    S.fhPoints = []; S.fhLine = null; S.drawing = false;
}

/* =====================================================
   Polyline Drawing
   ===================================================== */
function addPolylinePoint(e) {
    if (S.tool !== 'polyline') return;
    S.plPoints.push([e.latlng.lat, e.latlng.lng]);
    if (!S.plLine) {
        S.plLine = L.polyline(S.plPoints, {
            color: S.color, weight: S.weight,
            dashArray: DASH[S.dashStyle], opacity: 1,
            lineCap: 'round', lineJoin: 'round'
        }).addTo(map);
    } else {
        S.plLine.addLatLng(e.latlng);
    }
    if (S.plPoints.length >= 2) $('#btn-finish').classList.remove('hidden');
    updatePlPreview(e);
}

function updatePlPreview(e) {
    if (S.plPoints.length === 0) return;
    const last = S.plPoints[S.plPoints.length - 1];
    const cur = [e.latlng.lat, e.latlng.lng];
    if (!S.plPreview) {
        S.plPreview = L.polyline([last, cur], {
            color: S.color, weight: S.weight,
            dashArray: '6, 10', opacity: 0.5, interactive: false
        }).addTo(map);
    } else {
        S.plPreview.setLatLngs([last, cur]);
    }
}

function movePolyPreview(e) {
    if (S.tool === 'polyline' && S.plPoints.length > 0) {
        S.lastLatLng = e.latlng;
        updatePlPreview(e);
    }
}

function finishPolyline() {
    if (S.plPreview) { map.removeLayer(S.plPreview); S.plPreview = null; }
    if (S.plPoints.length < 2) { cancelPolyline(); return; }
    storeAnnotation(S.plLine, 'polyline', [...S.plPoints]);
    S.plPoints = []; S.plLine = null;
    $('#btn-finish').classList.add('hidden');
}

function cancelPolyline() {
    if (S.plLine) map.removeLayer(S.plLine);
    if (S.plPreview) map.removeLayer(S.plPreview);
    S.plPoints = []; S.plLine = null; S.plPreview = null;
    $('#btn-finish').classList.add('hidden');
}

/* =====================================================
   Arrow Drawing (click-and-drag)
   ===================================================== */
function computeArrowAngle(startLatLng, endLatLng) {
    var dLng = (endLatLng.lng - startLatLng.lng) * Math.PI / 180;
    var lat1 = startLatLng.lat * Math.PI / 180;
    var lat2 = endLatLng.lat * Math.PI / 180;
    var y = Math.sin(dLng) * Math.cos(lat2);
    var x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLng);
    return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

function createArrowheadIcon(color, preview, flag) {
    var size = 24;
    var svgNs = 'http://www.w3.org/2000/svg';
    var flagHtml = flag
        ? '<span class="arrowhead-flag">' + flag + '</span>'
        : '';
    return L.divIcon({
        className: '',
        html: '<div class="arrowhead-wrap' + (preview ? ' arrowhead-preview' : '') + '">'
            + '<svg xmlns="' + svgNs + '" width="' + size + '" height="' + size + '" viewBox="0 0 24 24">'
            + '<polygon points="12,1 22,22 12,17 2,22" fill="' + color + '" '
            + 'stroke="rgba(255,255,255,0.45)" stroke-width="1.2" stroke-linejoin="round"/>'
            + '</svg>' + flagHtml + '</div>',
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2]
    });
}

function removeArrowHead(ann) {
    if (ann && ann.extra && ann.extra.headLayer) {
        map.removeLayer(ann.extra.headLayer);
    }
}

function addArrowHeadToAnnotation(ann, marker) {
    if (!ann || !marker) return;
    marker._arrowId = ann.id;
    marker._isArrowHead = true;
    ann.extra = ann.extra || {};
    ann.extra.headLayer = marker;
}

function startArrow(e) {
    if (S.tool !== 'arrow') return;
    S.arrowDrawing = true;
    S.arrowStart = e.latlng;
    S.arrowLine = L.polyline([e.latlng, e.latlng], {
        color: S.color, weight: S.weight,
        dashArray: DASH[S.dashStyle], opacity: 1,
        lineCap: 'round', lineJoin: 'round'
    }).addTo(map);
    S.arrowHead = L.marker(e.latlng, {
        icon: createArrowheadIcon(S.color, true, S.flag),
        interactive: false
    }).addTo(map);
}

function moveArrow(e) {
    if (!S.arrowDrawing || !S.arrowLine) return;
    S.lastLatLng = e.latlng;
    S.arrowLine.setLatLngs([S.arrowStart, e.latlng]);
    S.arrowHead.setLatLng(e.latlng);
    var angle = computeArrowAngle(S.arrowStart, e.latlng);
    var el = S.arrowHead.getElement();
    if (el) {
        var inner = el.querySelector('.arrowhead-wrap');
        if (inner) inner.style.transform = 'rotate(' + angle + 'deg)';
        var flagEl = el.querySelector('.arrowhead-flag');
        if (flagEl) flagEl.style.transform = 'translateY(-50%) rotate(' + (-angle) + 'deg)';
    }
}

function finishArrow() {
    if (!S.arrowLine || !S.arrowStart) { resetArrow(); return; }
    if (S.arrowHead) { map.removeLayer(S.arrowHead); S.arrowHead = null; }
    var latlngs = S.arrowLine.getLatLngs();
    if (latlngs.length < 2 || (latlngs[0].lat === latlngs[1].lat && latlngs[0].lng === latlngs[1].lng)) {
        map.removeLayer(S.arrowLine); resetArrow(); return;
    }
    var startLL = latlngs[0], endLL = latlngs[1];
    var angle = computeArrowAngle(startLL, endLL);
    var coords = [[startLL.lat, startLL.lng], [endLL.lat, endLL.lng]];
    var currentFlag = S.flag;
    var headIcon = createArrowheadIcon(S.color, false, currentFlag);
    /* Keep the head visual-only so the line receives eraser clicks at its end. */
    var headMarker = L.marker(endLL, { icon: headIcon, interactive: false }).addTo(map);
    var headEl = headMarker.getElement();
    if (headEl) {
        var inner = headEl.querySelector('.arrowhead-wrap');
        if (inner) inner.style.transform = 'rotate(' + angle + 'deg)';
        var flagEl = headEl.querySelector('.arrowhead-flag');
        if (flagEl) flagEl.style.transform = 'translateY(-50%) rotate(' + (-angle) + 'deg)';
    }
    storeAnnotation(S.arrowLine, 'arrow', coords, { headLatLng: [endLL.lat, endLL.lng], angle: angle, flag: currentFlag || null });
    addArrowHeadToAnnotation(S.annotations[S.annotations.length - 1], headMarker);
    resetArrow();
}

function resetArrow() {
    S.arrowStart = null; S.arrowLine = null; S.arrowHead = null; S.arrowDrawing = false;
}

/* =====================================================
   Marker / Pin Tool
   ===================================================== */
function placeMarker(e) {
    if (S.tool !== 'marker') return;
    const icon = L.divIcon({
        className: '',
        html: '<div class="marker-pin" style="background:' + S.color + '"></div>',
        iconSize: [18, 22], iconAnchor: [9, 22]
    });
    const marker = L.marker(e.latlng, { icon }).addTo(map);
    storeAnnotation(marker, 'marker', [e.latlng.lat, e.latlng.lng]);
}

/* =====================================================
   Text Label Tool
   ===================================================== */
const labelWrap   = $('#label-input-wrap');
const labelInput  = $('#label-text-input');
const labelOkBtn  = $('#label-ok');
const labelCancelBtn = $('#label-cancel');

function showLabelInput(latlng, screenPt) {
    S.labelLatLng = latlng;
    labelWrap.style.left = screenPt.x + 'px';
    labelWrap.style.top  = (screenPt.y - 50) + 'px';
    labelWrap.classList.remove('hidden');
    labelInput.value = '';
    labelInput.focus();
}
function hideLabelInput() { labelWrap.classList.add('hidden'); S.labelLatLng = null; }

function placeLabel() {
    const text = labelInput.value.trim();
    if (!text || !S.labelLatLng) { hideLabelInput(); return; }
    const icon = L.divIcon({
        className: '',
        html: '<div class="label-marker" style="border-color:' + S.color + ';color:#ffffff">' + escapeHtml(text) + '</div>',
        iconSize: null, iconAnchor: [0, 0]
    });
    const m = L.marker(S.labelLatLng, { icon, interactive: true }).addTo(map);
    storeAnnotation(m, 'label', [S.labelLatLng.lat, S.labelLatLng.lng], { text });
    hideLabelInput();
}

/* =====================================================
   Fire Emoji Tool
   ===================================================== */
function placeFire(e) {
    if (S.tool !== 'fire') return;
    const icon = L.divIcon({
        className: '',
        html: '<div class="fire-marker">🔥</div>',
        iconSize: [32, 32], iconAnchor: [16, 16]
    });
    const marker = L.marker(e.latlng, { icon }).addTo(map);
    storeAnnotation(marker, 'fire', [e.latlng.lat, e.latlng.lng]);
}
function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
labelOkBtn.addEventListener('click', placeLabel);
labelCancelBtn.addEventListener('click', hideLabelInput);
labelInput.addEventListener('keydown', e => {
    if (e.key === 'Enter')  { e.preventDefault(); placeLabel(); }
    if (e.key === 'Escape') { e.preventDefault(); hideLabelInput(); }
    e.stopPropagation();
});

/* =====================================================
   Distance Measurement Tool
   ===================================================== */
function startMeasure(e) {
    if (S.tool !== 'measure') return;
    S.msPoints.push(e.latlng);
    if (!S.msLine) {
        S.msLine = L.polyline(S.msPoints, {
            color: '#ffe066', weight: 2, dashArray: '8, 6', opacity: 0.9
        }).addTo(map);
    } else { S.msLine.addLatLng(e.latlng); }
    if (S.msPoints.length >= 2) $('#btn-finish').classList.remove('hidden');
    updateMeasureLabels();
    updateMeasurePreview(e);
}

function updateMeasurePreview(e) {
    if (S.msPoints.length === 0) return;
    const last = S.msPoints[S.msPoints.length - 1];
    if (!S.msPreview) {
        S.msPreview = L.polyline([last, e.latlng], {
            color: '#ffe066', weight: 1, dashArray: '4, 8', opacity: 0.5, interactive: false
        }).addTo(map);
    } else { S.msPreview.setLatLngs([last, e.latlng]); }
}

function moveMeasurePreview(e) {
    if (S.tool === 'measure' && S.msPoints.length > 0) updateMeasurePreview(e);
}

function updateMeasureLabels() {
    S.msLabels.forEach(l => map.removeLayer(l));
    S.msLabels = [];
    let total = 0;
    for (let i = 1; i < S.msPoints.length; i++) {
        const seg = S.msPoints[i - 1].distanceTo(S.msPoints[i]);
        total += seg;
        const mid = L.latLng(
            (S.msPoints[i - 1].lat + S.msPoints[i].lat) / 2,
            (S.msPoints[i - 1].lng + S.msPoints[i].lng) / 2
        );
        const lbl = L.tooltip({ permanent: true, direction: 'center', className: 'measure-label', interactive: false })
            .setContent(fmtDist(seg)).setLatLng(mid).addTo(map);
        S.msLabels.push(lbl);
    }
    if (S.msPoints.length >= 2) {
        const last = S.msPoints[S.msPoints.length - 1];
        const totalLbl = L.tooltip({ permanent: true, direction: 'top', offset: [0, -10],
            className: 'measure-label measure-total', interactive: false })
            .setContent('\u2248 ' + fmtDist(total)).setLatLng(last).addTo(map);
        S.msLabels.push(totalLbl);
    }
    if (S.msPoints.length >= 2) $('#status-text').textContent = 'Distance: ' + fmtDist(total) + ' (Enter to finish)';
}

function fmtDist(m) { return m >= 1000 ? (m / 1000).toFixed(2) + ' km' : m.toFixed(0) + ' m'; }

function finishMeasure() {
    if (S.msPreview) { map.removeLayer(S.msPreview); S.msPreview = null; }
    S.msLabels.forEach(l => map.removeLayer(l)); S.msLabels = [];
    if (S.msPoints.length < 2 || !S.msLine) { cancelMeasure(); return; }
    let total = 0;
    for (let i = 1; i < S.msPoints.length; i++) total += S.msPoints[i - 1].distanceTo(S.msPoints[i]);
    const last = S.msPoints[S.msPoints.length - 1];
    const totalLbl = L.tooltip({ permanent: true, direction: 'top', offset: [0, -10],
        className: 'measure-label measure-total', interactive: false })
        .setContent('\u2248 ' + fmtDist(total)).setLatLng(last).addTo(map);
    S.msLine._totalLabel = totalLbl;
    storeAnnotation(S.msLine, 'measure', S.msPoints.map(p => [p.lat, p.lng]), { total });
    S.msPoints = []; S.msLine = null; S.msPreview = null;
    $('#btn-finish').classList.add('hidden');
}

function cancelMeasure() {
    if (S.msPreview) map.removeLayer(S.msPreview);
    S.msLabels.forEach(l => map.removeLayer(l));
    if (S.msLine) map.removeLayer(S.msLine);
    S.msPoints = []; S.msLine = null; S.msLabels = []; S.msPreview = null;
    $('#btn-finish').classList.add('hidden');
}

/* =====================================================
   Eraser
   ===================================================== */
function enableEraser() {
    S.annotations.forEach(ann => {
        const isPoint = ann.type === 'marker' || ann.type === 'label';
        const origWeight = ann.weight;
        /* Force interactive mode so eraser can receive clicks (measurement lines are non-interactive by default) */
        if (ann.layer && ann.layer.options) {
            ann._origInteractive = ann.layer.options.interactive;
            ann.layer.options.interactive = true;
            if (ann.layer.getElement) {
                const el = ann.layer.getElement();
                if (el) el.style.pointerEvents = 'auto';
            }
        }
        /* Make lines thicker for easier clicking during erase mode */
        if (!isPoint && ann.layer && ann.layer.setStyle) {
            ann.layer.setStyle({ weight: Math.max(8, origWeight + 5) });
            ann.layer.bringToFront();
        }
        const h = {
            over: function () {
                if (isPoint) {
                    const el = this.getElement();
                    if (el) { const inner = el.querySelector('.marker-pin, .label-marker'); if (inner) inner.classList.add('eraser-hover'); }
                } else if (this.setStyle) {
                    this.setStyle({ color: '#ff4444', weight: Math.max(10, origWeight + 6) });
                    this.bringToFront();
                }
            },
            out: function () {
                if (isPoint) {
                    const el = this.getElement();
                    if (el) { const inner = el.querySelector('.marker-pin, .label-marker'); if (inner) inner.classList.remove('eraser-hover'); }
                } else if (this.setStyle) {
                    this.setStyle({ color: ann.color, weight: Math.max(8, origWeight + 5) });
                }
            },
            click: function (ev) {
                if (ev && ev.originalEvent) L.DomEvent.stop(ev.originalEvent);
                removeAnnotation(ann.id);
            }
        };
        ann.layer.on('mouseover', h.over);
        ann.layer.on('mouseout', h.out);
        ann.layer.on('click', h.click);
        S.eraserHandlers.push({ ann, h });
    });
}

function disableEraser() {
    S.eraserHandlers.forEach(({ ann, h }) => {
        restoreEraserState(ann, h);
    });
    S.eraserHandlers = [];
}

function restoreEraserState(ann, h) {
    if (!ann || !ann.layer) return;
    ann.layer.off('mouseover', h.over);
    ann.layer.off('mouseout', h.out);
    ann.layer.off('click', h.click);
    /* Restore original style (weight, color) that was changed for eraser visibility */
    const isPoint = ann.type === 'marker' || ann.type === 'label';
    if (isPoint && ann.layer.getElement) {
        const el = ann.layer.getElement();
        if (el) {
            const inner = el.querySelector('.marker-pin, .label-marker');
            if (inner) inner.classList.remove('eraser-hover');
        }
    }
    if (!isPoint && ann.layer.setStyle) {
        ann.layer.setStyle({ weight: ann.weight, color: ann.color });
        ann.layer.bringToBack();
    }
    /* Restore original interactive state */
    if (ann.layer.options && ann._origInteractive !== undefined) {
        ann.layer.options.interactive = ann._origInteractive;
        if (ann.layer.getElement) {
            const el = ann.layer.getElement();
            if (el) el.style.pointerEvents = '';
        }
        delete ann._origInteractive;
    }
}

function removeEraserHandlersFor(ann) {
    S.eraserHandlers = S.eraserHandlers.filter(({ ann: current, h }) => {
        if (current !== ann) return true;
        restoreEraserState(ann, h);
        return false;
    });
}

/* ── Eraser drag-to-erase ──────────────────────── */
const ERASE_RADIUS_PX = 20;

function eraserDragStart() {
    if (S.tool !== 'eraser') return;
    S.eraserDragging = true;
    S.eraserErasedIds = new Set();
}

function eraserDragMove(e) {
    if (!S.eraserDragging || S.tool !== 'eraser') return;
    var containerPoint = map.latLngToContainerPoint(e.latlng);
    S.annotations.slice().forEach(function (ann) {
        if (S.eraserErasedIds.has(ann.id)) return;
        var hit = false;
        var isPoint = ann.type === 'marker' || ann.type === 'label';
        if (isPoint) {
            /* For point annotations, check distance from cursor to the annotation's latlng */
            var annLatLng = ann.coords;
            if (Array.isArray(annLatLng) && annLatLng.length === 2) {
                var annPoint = map.latLngToContainerPoint(L.latLng(annLatLng[0], annLatLng[1]));
                var dist = Math.hypot(containerPoint.x - annPoint.x, containerPoint.y - annPoint.y);
                if (dist <= ERASE_RADIUS_PX) hit = true;
            }
        } else {
            /* For line annotations, check distance from cursor to each segment */
            var coords = ann.coords;
            if (Array.isArray(coords)) {
                for (var i = 0; i < coords.length - 1; i++) {
                    var p1 = map.latLngToContainerPoint(L.latLng(coords[i][0], coords[i][1]));
                    var p2 = map.latLngToContainerPoint(L.latLng(coords[i + 1][0], coords[i + 1][1]));
                    var d = pointToSegmentDist(containerPoint.x, containerPoint.y, p1.x, p1.y, p2.x, p2.y);
                    if (d <= ERASE_RADIUS_PX) { hit = true; break; }
                }
            }
        }
        if (hit) {
            S.eraserErasedIds.add(ann.id);
            removeAnnotation(ann.id);
        }
    });
}

function eraserDragEnd() {
    S.eraserDragging = false;
    S.eraserErasedIds = new Set();
}

/* Helper: distance from point (px,py) to segment (ax,ay)-(bx,by) */
function pointToSegmentDist(px, py, ax, ay, bx, by) {
    var dx = bx - ax, dy = by - ay;
    var lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(px - ax, py - ay);
    var t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/* =====================================================
   Undo / Redo
   ===================================================== */
function pushUndo(action) {
    S.undoStack.push(action);
    if (S.undoStack.length > UNDO_LIMIT) S.undoStack.shift();
    S.redoStack = [];
    updateUndoRedoUI();
}

function undo() {
    if (S.undoStack.length === 0) return;
    const a = S.undoStack.pop();
    if (a.type === 'add') {
        map.removeLayer(a.ann.layer);
        removeArrowHead(a.ann);
        S.annotations = S.annotations.filter(x => x.id !== a.ann.id);
    } else if (a.type === 'remove') {
        a.ann.layer.addTo(map);
        if (a.ann.extra && a.ann.extra.headLayer) a.ann.extra.headLayer.addTo(map);
        if (a.ann.layer._totalLabel) a.ann.layer._totalLabel.addTo(map);
        S.annotations.push(a.ann);
    } else if (a.type === 'clear') {
        a.anns.forEach(x => {
            x.layer.addTo(map);
            if (x.extra && x.extra.headLayer) x.extra.headLayer.addTo(map);
            if (x.layer._totalLabel) x.layer._totalLabel.addTo(map);
            S.annotations.push(x);
        });
    }
    if (S.tool === 'eraser') { disableEraser(); enableEraser(); }
    S.redoStack.push(a);
    updateCount(); updateUndoRedoUI();
}

function redo() {
    if (S.redoStack.length === 0) return;
    const a = S.redoStack.pop();
    if (a.type === 'add') {
        a.ann.layer.addTo(map);
        if (a.ann.extra && a.ann.extra.headLayer) a.ann.extra.headLayer.addTo(map);
        S.annotations.push(a.ann);
    } else if (a.type === 'remove') {
        map.removeLayer(a.ann.layer);
        removeArrowHead(a.ann);
        if (a.ann.layer._totalLabel) map.removeLayer(a.ann.layer._totalLabel);
        S.annotations = S.annotations.filter(x => x.id !== a.ann.id);
    } else if (a.type === 'clear') {
        a.anns.forEach(x => {
            map.removeLayer(x.layer);
            removeArrowHead(x);
            if (x.layer._totalLabel) map.removeLayer(x.layer._totalLabel);
        });
        S.annotations = [];
    }
    if (S.tool === 'eraser') { disableEraser(); enableEraser(); }
    S.undoStack.push(a);
    updateCount(); updateUndoRedoUI();
}

function updateUndoRedoUI() {
    $('#btn-undo').disabled = S.undoStack.length === 0;
    $('#btn-redo').disabled = S.redoStack.length === 0;
}

/* =====================================================
   Annotation Management
   ===================================================== */
function storeAnnotation(layer, type, coords, extra) {
    const ann = {
        id: S.nextId++,
        layer, type, coords,
        color: S.color,
        weight: S.weight,
        dashStyle: S.dashStyle,
        dashArray: DASH[S.dashStyle],
        extra: extra || null
    };
    layer._annId = ann.id;
    S.annotations.push(ann);
    pushUndo({ type: 'add', ann });
    updateCount();
}

function removeAnnotation(id) {
    const i = S.annotations.findIndex(a => a.id === id);
    if (i === -1) return;
    const ann = S.annotations[i];
    removeEraserHandlersFor(ann);
    map.removeLayer(ann.layer);
    removeArrowHead(ann);
    /* Also remove the total label tooltip for measurement annotations */
    if (ann.layer._totalLabel) {
        map.removeLayer(ann.layer._totalLabel);
    }
    S.annotations.splice(i, 1);
    pushUndo({ type: 'remove', ann });
    updateCount();
}

function clearAll() {
    if (S.annotations.length === 0) return;
    if (!confirm('Clear all annotations? You can undo this.')) return;
    disableEraser();
    const snapshot = S.annotations.slice();
    S.annotations.forEach(a => {
        map.removeLayer(a.layer);
        removeArrowHead(a);
        if (a.layer._totalLabel) map.removeLayer(a.layer._totalLabel);
    });
    S.annotations = [];
    pushUndo({ type: 'clear', anns: snapshot });
    updateCount();
}

function updateCount() {
    $('#ann-count').textContent = S.annotations.length + ' annotation' + (S.annotations.length !== 1 ? 's' : '');
}

/* =====================================================
   Region Highlight — double-click toggle
   ===================================================== */
function toggleHighlight(layer) {
    /* If clicking the already-highlighted polygon, un-highlight it */
    if (S.highlighted === layer) {
        layer.setStyle(S.highlightedOrigStyle);
        layer.bringToBack();
        S.highlighted = null;
        S.highlightedOrigStyle = null;
        return;
    }
    /* Un-highlight previous if any */
    if (S.highlighted) {
        S.highlighted.setStyle(S.highlightedOrigStyle);
        S.highlighted.bringToBack();
    }
    /* Store original style and apply highlight */
    S.highlightedOrigStyle = Object.assign({}, layer.options);
    S.highlighted = layer;
    layer.setStyle({
        color: '#ffdd00',
        weight: 7,
        opacity: 1,
        fillOpacity: 0.35,
        dashArray: ''
    });
    layer.bringToFront();
}

/* Attach double-click highlight to a polygon layer */
function attachHighlightHandler(layer) {
    layer.on('dblclick', function (e) {
        if (e && e.originalEvent) L.DomEvent.stop(e.originalEvent);
        toggleHighlight(this);
    });
}

/* =====================================================
   Smoothing — Ramer-Douglas-Peucker + Catmull-Rom
   ===================================================== */
function rdpSimplify(pts, tol) {
    if (pts.length <= 2) return pts.slice();
    let maxD = 0, maxI = 0;
    const a = pts[0], b = pts[pts.length - 1];
    for (let i = 1; i < pts.length - 1; i++) {
        const d = ptSegDist(pts[i], a, b);
        if (d > maxD) { maxD = d; maxI = i; }
    }
    if (maxD > tol) {
        const l = rdpSimplify(pts.slice(0, maxI + 1), tol);
        const r = rdpSimplify(pts.slice(maxI), tol);
        return l.slice(0, -1).concat(r);
    }
    return [a, b];
}

function ptSegDist(p, a, b) {
    const dy = b[0] - a[0], dx = b[1] - a[1];
    const lenSq = dy * dy + dx * dx;
    if (lenSq === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
    let t = ((p[0] - a[0]) * dy + (p[1] - a[1]) * dx) / lenSq;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(p[0] - (a[0] + t * dy), p[1] - (a[1] + t * dx));
}

function catmullRom(pts, seg) {
    if (pts.length < 3) return pts.slice();
    const out = [];
    for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[Math.max(0, i - 1)];
        const p1 = pts[i];
        const p2 = pts[Math.min(pts.length - 1, i + 1)];
        const p3 = pts[Math.min(pts.length - 1, i + 2)];
        for (let j = 0; j < seg; j++) {
            const t = j / seg, t2 = t * t, t3 = t2 * t;
            out.push([
                0.5 * (2*p1[0] + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3),
                0.5 * (2*p1[1] + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            ]);
        }
    }
    out.push(pts[pts.length - 1]);
    return out;
}

/* =====================================================
   Export — PNG (via html2canvas)
   ===================================================== */
function exportPNG() {
    const btn = $('#btn-png');
    btn.disabled = true;
    btn.textContent = '⏳ …';
    /* temporarily show finish btn hidden */
    const finishWasHidden = $('#btn-finish').classList.contains('hidden');
    $('#btn-finish').classList.add('hidden');

    html2canvas(document.getElementById('map'), {
        useCORS: true, allowTaint: true, scale: 1
    }).then(canvas => {
        const a = document.createElement('a');
        a.download = 'map-' + Date.now() + '.png';
        a.href = canvas.toDataURL('image/png');
        a.click();
    }).catch(err => {
        console.error('PNG export failed:', err);
        alert('PNG export failed. Try using your browser\'s screenshot feature instead.');
    }).finally(() => {
        btn.disabled = false;
        btn.textContent = '\uD83D\uDCF7 PNG';
        if (!finishWasHidden && S.plPoints.length >= 2) {
            $('#btn-finish').classList.remove('hidden');
        }
    });
}

/* =====================================================
   Export — GeoJSON
   ===================================================== */
function exportGeoJSON() {
    if (S.annotations.length === 0) {
        alert('No annotations to export.');
        return;
    }
    const features = S.annotations.map(a => {
        const isPoint = a.type === 'marker' || a.type === 'label';
        return {
            type: 'Feature',
            properties: {
                id: a.id, type: a.type, color: a.color,
                weight: a.weight, dashStyle: a.dashStyle,
                text: (a.extra && a.extra.text) || undefined
            },
            geometry: isPoint
                ? { type: 'Point', coordinates: [a.coords[1], a.coords[0]] }
                : { type: 'LineString', coordinates: a.coords.map(c => [c[1], c[0]]) }
        };
    });
    const geojson = {
        type: 'FeatureCollection',
        crs: { type: 'name', properties: { name: 'urn:ogc:def:crs:OGC:1.3:CRS84' } },
        features
    };
    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.download = 'annotations-' + Date.now() + '.geojson';
    a.href = url;
    a.click();
    URL.revokeObjectURL(url);
}

/* =====================================================
   Import Modal — State & Helpers
   ===================================================== */
let importPendingFeatures = [];

/* ── Built-in Presets ────────────────────────── */
const BUILT_IN_PRESETS = [
    /* ── 1. WWI Western Front ──────────────────── */
    {
        name: 'WWI Western Front',
        desc: 'Allied & German front lines, Hindenburg Line, key battles 1914-1918',
        geojson:{type:'FeatureCollection',features:[
            /* Entente (Allied) front line — Nieuport to Swiss border (approx 1917-18) */
            {type:'Feature',properties:{type:'polyline',color:'#ff4444',weight:3,dashStyle:'dashed',text:'Entente Front Line (1917-18)'},geometry:{type:'LineString',coordinates:[
                [2.92,51.13],[2.86,51.03],[2.89,50.85],[2.89,50.77],
                [2.85,50.61],[2.77,50.43],[2.77,50.35],[2.85,50.20],
                [2.93,50.09],[3.07,50.01],[3.18,49.93],[2.95,49.80],
                [3.10,49.65],[3.55,49.44],[3.82,49.25],[4.50,49.20],
                [5.38,49.16],[5.80,49.13],[6.18,49.12],[6.50,48.95],
                [6.90,48.78],[7.30,48.58],[7.75,48.58],[7.50,48.20],
                [7.34,47.75],[6.86,47.64],[7.58,47.56]
            ]}},
            /* Central Powers front line — offset east of Allied line */
            {type:'Feature',properties:{type:'polyline',color:'#4488ff',weight:3,dashStyle:'dashed',text:'Central Powers Front (1917-18)'},geometry:{type:'LineString',coordinates:[
                [3.12,51.13],[3.06,51.03],[3.09,50.85],[3.09,50.77],
                [3.05,50.61],[2.97,50.43],[2.97,50.35],[3.05,50.20],
                [3.13,50.09],[3.27,50.01],[3.38,49.93],[3.15,49.80],
                [3.30,49.65],[3.75,49.44],[4.02,49.25],[4.70,49.20],
                [5.58,49.16],[6.00,49.13],[6.38,49.12],[6.70,48.95],
                [7.10,48.78],[7.50,48.58],[7.95,48.58],[7.70,48.20],
                [7.54,47.75],[7.06,47.64],[7.78,47.56]
            ]}},
            /* Hindenburg Line (Siegfriedstellung) — German defensive position */
            {type:'Feature',properties:{type:'polyline',color:'#ffaa00',weight:2,dashStyle:'dotted',text:'Hindenburg Line (Siegfriedstellung)'},geometry:{type:'LineString',coordinates:[
                [3.20,51.00],[3.25,50.90],[3.15,50.70],[3.10,50.55],
                [3.10,50.40],[3.15,50.25],[3.35,50.05],[3.50,49.95],
                [3.40,49.85],[3.55,49.70],[3.85,49.50],[4.10,49.35],
                [4.40,49.28],[5.10,49.22],[5.70,49.18]
            ]}},
            /* Battle markers */
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Nieuport (1914, 1918)'},geometry:{type:'Point',coordinates:[2.92,51.13]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Ypres (1914-18, 3 battles)'},geometry:{type:'Point',coordinates:[2.87,50.85]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Passchendaele (1917)'},geometry:{type:'Point',coordinates:[2.95,50.90]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Messines Ridge (1917)'},geometry:{type:'Point',coordinates:[2.89,50.77]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Loos (1915)'},geometry:{type:'Point',coordinates:[2.77,50.43]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Arras (1917)'},geometry:{type:'Point',coordinates:[2.78,50.29]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Vimy Ridge (1917)'},geometry:{type:'Point',coordinates:[2.77,50.35]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Somme (1916)'},geometry:{type:'Point',coordinates:[2.70,50.01]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Cambrai (1917, first tanks)'},geometry:{type:'Point',coordinates:[3.24,50.18]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Chemin des Dames (1917)'},geometry:{type:'Point',coordinates:[3.55,49.44]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'2nd Battle of the Marne (1918)'},geometry:{type:'Point',coordinates:[3.60,49.08]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Verdun (1916, 300 days)'},geometry:{type:'Point',coordinates:[5.38,49.16]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Metz'},geometry:{type:'Point',coordinates:[6.18,49.12]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Strasbourg'},geometry:{type:'Point',coordinates:[7.75,48.58]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Belfort'},geometry:{type:'Point',coordinates:[6.86,47.64]}},
            /* Area labels */
            {type:'Feature',properties:{type:'label',color:'#ff6644',text:'Allied-held France'},geometry:{type:'Point',coordinates:[2.5,49.5]}},
            {type:'Feature',properties:{type:'label',color:'#4488ff',text:'German-occupied Belgium & France'},geometry:{type:'Point',coordinates:[4.5,50.0]}}
        ]}
    },
    /* ── 2. WWII Eastern Front ──────────────────── */
    {
        name: 'WWII Eastern Front',
        desc: 'Barbarossa, max German advance (1942), Soviet counter-offensive to Berlin',
        geojson:{type:'FeatureCollection',features:[
            /* Barbarossa start line — Baltic to Black Sea (Jun 22, 1941) */
            {type:'Feature',properties:{type:'polyline',color:'#ff2222',weight:3,dashStyle:'dashed',text:'Barbarossa Line (Jun 1941)'},geometry:{type:'LineString',coordinates:[
                [21.1,55.7],[22.8,54.1],[23.2,53.2],[23.7,52.1],
                [24.0,51.0],[24.0,50.0],[24.5,49.2],[25.0,48.5],
                [25.9,48.3],[27.6,47.1],[28.3,46.0]
            ]}},
            /* Max German advance — Leningrad to Caucasus (autumn 1942) */
            {type:'Feature',properties:{type:'polyline',color:'#ff8800',weight:3,dashStyle:'solid',text:'Max German Advance (autumn 1942)'},geometry:{type:'LineString',coordinates:[
                [30.0,59.9],[33.0,57.5],[34.5,56.2],[37.0,55.8],
                [36.5,54.0],[37.5,52.5],[39.2,51.7],[41.0,49.5],
                [43.5,48.7],[42.0,46.5],[44.0,44.0]
            ]}},
            /* Soviet counter-offensive path to Berlin (1943-45) */
            {type:'Feature',properties:{type:'polyline',color:'#44aaff',weight:3,dashStyle:'dashed',text:'Soviet Advance to Berlin (1943-45)'},geometry:{type:'LineString',coordinates:[
                [43.5,48.7],[41.0,47.5],[38.5,47.0],[36.0,46.5],
                [33.5,46.5],[31.5,48.5],[30.5,50.5],[28.0,51.5],
                [24.0,53.0],[21.0,52.2],[18.5,52.0],[15.0,52.5],
                [13.4,52.5]
            ]}},
            /* City & battle markers */
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Leningrad (sieged 872 days)'},geometry:{type:'Point',coordinates:[30.3,59.93]}},
            {type:'Feature',properties:{type:'marker',color:'#ff8800',text:'Moscow (1941-42)'},geometry:{type:'Point',coordinates:[37.6,55.75]}},
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Stalingrad (1942-43)'},geometry:{type:'Point',coordinates:[43.5,48.7]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Kursk (1943, largest tank battle)'},geometry:{type:'Point',coordinates:[36.2,51.7]}},
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Minsk (1941, liberated 1944)'},geometry:{type:'Point',coordinates:[27.6,53.9]}},
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Smolensk (1941)'},geometry:{type:'Point',coordinates:[32.0,54.8]}},
            {type:'Feature',properties:{type:'marker',color:'#ff8800',text:'Kiev (1941, liberated 1943)'},geometry:{type:'Point',coordinates:[30.5,50.45]}},
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Sevastopol (1941-42)'},geometry:{type:'Point',coordinates:[33.5,44.6]}},
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Rostov-on-Don'},geometry:{type:'Point',coordinates:[39.7,47.2]}},
            {type:'Feature',properties:{type:'marker',color:'#ff8800',text:'Brest-Litovsk'},geometry:{type:'Point',coordinates:[23.7,52.1]}},
            {type:'Feature',properties:{type:'marker',color:'#ff8800',text:'Warsaw (1944 uprising)'},geometry:{type:'Point',coordinates:[21.0,52.2]}},
            {type:'Feature',properties:{type:'marker',color:'#ff8800',text:'Vienna (1945)'},geometry:{type:'Point',coordinates:[16.4,48.2]}},
            {type:'Feature',properties:{type:'marker',color:'#ff8800',text:'Budapest (1944-45 siege)'},geometry:{type:'Point',coordinates:[19.0,47.5]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Berlin (May 1945)'},geometry:{type:'Point',coordinates:[13.4,52.5]}},
            /* Labels */
            {type:'Feature',properties:{type:'label',color:'#ff6644',text:'Nazi-occupied Europe (1942)'},geometry:{type:'Point',coordinates:[28,50]}},
            {type:'Feature',properties:{type:'label',color:'#44aaff',text:'Soviet Union'},geometry:{type:'Point',coordinates:[50,53]}}
        ]}
    },
    /* ── 3. Napoleonic Campaigns ──────────────────── */
    {
        name: 'Napoleonic Campaigns',
        desc: 'Grande Armee routes & key battles, 1805-1815',
        geojson:{type:'FeatureCollection',features:[
            /* March to Moscow (Jun-Sep 1812) */
            {type:'Feature',properties:{type:'polyline',color:'#ffaa00',weight:3,dashStyle:'dotted',text:'March to Moscow (Jun-Sep 1812)'},geometry:{type:'LineString',coordinates:[
                [20.5,54.7],[21.0,54.7],[22.8,54.1],[23.7,52.1],
                [24.0,53.7],[26.0,54.5],[27.6,53.9],[28.5,53.7],
                [30.0,54.5],[32.0,54.8],[34.0,55.0],[36.0,55.5],
                [37.6,55.75]
            ]}},
            /* Retreat from Moscow (Oct-Dec 1812) */
            {type:'Feature',properties:{type:'polyline',color:'#ff4444',weight:3,dashStyle:'solid',text:'Retreat from Moscow (Oct-Dec 1812)'},geometry:{type:'LineString',coordinates:[
                [37.6,55.75],[36.0,55.3],[34.5,54.5],[33.0,54.0],
                [31.0,53.5],[29.0,53.0],[27.0,52.5],[25.0,52.0],
                [24.0,53.7],[23.0,54.0],[21.5,54.7],[20.5,54.7]
            ]}},
            /* Earlier campaign routes */
            {type:'Feature',properties:{type:'polyline',color:'#ff8800',weight:2,dashStyle:'dashed',text:'Austerlitz Campaign (1805)'},geometry:{type:'LineString',coordinates:[
                [2.35,48.86],[5.0,48.5],[7.0,48.0],[8.5,48.5],
                [10.0,48.4],[13.0,48.8],[15.0,49.0],[16.76,49.13]
            ]}},
            {type:'Feature',properties:{type:'polyline',color:'#ff8800',weight:2,dashStyle:'dashed',text:'Jena-Auerstedt Campaign (1806)'},geometry:{type:'LineString',coordinates:[
                [8.0,49.0],[9.0,49.5],[10.0,50.0],[11.0,50.5],[11.59,50.93]
            ]}},
            /* Battle markers */
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Austerlitz (Dec 1805)'},geometry:{type:'Point',coordinates:[16.76,49.13]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Jena-Auerstedt (Oct 1806)'},geometry:{type:'Point',coordinates:[11.59,50.93]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Friedland (Jun 1807)'},geometry:{type:'Point',coordinates:[20.89,54.40]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Borodino (Sep 1812)'},geometry:{type:'Point',coordinates:[35.82,55.53]}},
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Moscow (burned Sep 1812)'},geometry:{type:'Point',coordinates:[37.62,55.75]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Dresden (Aug 1813)'},geometry:{type:'Point',coordinates:[13.74,51.05]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Leipzig (Oct 1813, Battle of Nations)'},geometry:{type:'Point',coordinates:[12.37,51.34]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Lutzen (May 1813)'},geometry:{type:'Point',coordinates:[12.15,51.25]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Bautzen (May 1813)'},geometry:{type:'Point',coordinates:[14.42,51.18]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Waterloo (Jun 1815)'},geometry:{type:'Point',coordinates:[4.40,50.71]}},
            /* City markers */
            {type:'Feature',properties:{type:'marker',color:'#00bbff',text:'Paris (capital)'},geometry:{type:'Point',coordinates:[2.35,48.86]}},
            {type:'Feature',properties:{type:'marker',color:'#00bbff',text:'Bologne-sur-Mer (camp 1805)'},geometry:{type:'Point',coordinates:[1.61,50.73]}},
            {type:'Feature',properties:{type:'marker',color:'#00bbff',text:'Ulm (surrendered Oct 1805)'},geometry:{type:'Point',coordinates:[9.99,48.40]}},
            {type:'Feature',properties:{type:'marker',color:'#00bbff',text:'Tilsit (treaty Jul 1807)'},geometry:{type:'Point',coordinates:[21.88,55.08]}},
            /* Labels */
            {type:'Feature',properties:{type:'label',color:'#ffaa00',text:'Grande Armee'},geometry:{type:'Point',coordinates:[30,53]}},
            {type:'Feature',properties:{type:'label',color:'#4488ff',text:'Russian Empire'},geometry:{type:'Point',coordinates:[45,57]}}
        ]}
    },
    /* ── 4. Cold War Iron Curtain ──────────────────── */
    {
        name: 'Cold War Iron Curtain',
        desc: 'Division of Europe — NATO vs Warsaw Pact, 1945-1991',
        geojson:{type:'FeatureCollection',features:[
            /* Inner German border (East-West Germany) */
            {type:'Feature',properties:{type:'polyline',color:'#ff3333',weight:4,dashStyle:'dotted',text:'Inner German Border (1945-90)'},geometry:{type:'LineString',coordinates:[
                [10.87,53.96],[10.60,53.70],[10.30,53.40],[10.10,53.10],
                [9.90,52.80],[9.80,52.50],[9.75,52.20],[9.70,51.90],
                [9.80,51.60],[10.00,51.30],[10.20,51.10],[10.50,50.90],
                [10.80,50.70],[11.00,50.50],[11.20,50.35],[11.50,50.20],
                [11.80,50.05]
            ]}},
            /* Full Iron Curtain — Baltic to Adriatic */
            {type:'Feature',properties:{type:'polyline',color:'#ff3333',weight:4,dashStyle:'dotted',text:'Iron Curtain (full line)'},geometry:{type:'LineString',coordinates:[
                [10.87,53.96],[14.00,54.20],[14.30,53.90],[14.80,53.40],
                [15.00,53.00],[14.60,52.50],[14.70,52.00],[15.00,51.10],
                [15.50,50.70],[16.00,50.20],[16.50,49.60],[17.00,49.20],
                [17.50,48.80],[18.00,48.50],[18.50,48.20],[19.00,48.00],
                [20.00,47.80],[21.00,47.50],[22.00,47.00],[23.00,46.50],
                [24.00,46.00],[25.00,45.50],[26.00,45.00],[27.00,44.50],
                [28.00,44.00],[29.00,43.50]
            ]}},
            /* Oder-Neisse line (Poland western border) */
            {type:'Feature',properties:{type:'polyline',color:'#ffaa00',weight:2,dashStyle:'dashed',text:'Oder-Neisse Line'},geometry:{type:'LineString',coordinates:[
                [14.30,53.90],[14.20,53.50],[14.10,53.10],[14.00,52.70],
                [14.50,52.00],[14.70,51.60],[15.00,51.10]
            ]}},
            /* Berlin Wall */
            {type:'Feature',properties:{type:'polyline',color:'#ff0000',weight:3,dashStyle:'solid',text:'Berlin Wall (1961-89)'},geometry:{type:'LineString',coordinates:[
                [13.30,52.55],[13.25,52.52],[13.30,52.48],[13.38,52.45],
                [13.42,52.47],[13.46,52.50],[13.43,52.53],[13.38,52.55],
                [13.30,52.55]
            ]}},
            /* City markers */
            {type:'Feature',properties:{type:'marker',color:'#ff0000',text:'Berlin (divided city)'},geometry:{type:'Point',coordinates:[13.40,52.52]}},
            {type:'Feature',properties:{type:'marker',color:'#4488ff',text:'Vienna (withdrawn 1955)'},geometry:{type:'Point',coordinates:[16.37,48.21]}},
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Prague (1968 invasion)'},geometry:{type:'Point',coordinates:[14.42,50.08]}},
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Budapest (1956 uprising)'},geometry:{type:'Point',coordinates:[19.04,47.50]}},
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Warsaw (Pact HQ)'},geometry:{type:'Point',coordinates:[21.01,52.23]}},
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Moscow (Kremlin)'},geometry:{type:'Point',coordinates:[37.62,55.75]}},
            {type:'Feature',properties:{type:'marker',color:'#4488ff',text:'Helsinki (Finland, neutral)'},geometry:{type:'Point',coordinates:[24.94,60.17]}},
            {type:'Feature',properties:{type:'marker',color:'#4488ff',text:'Belgrade (Non-Aligned)'},geometry:{type:'Point',coordinates:[20.45,44.82]}},
            /* Labels */
            {type:'Feature',properties:{type:'label',color:'#4488ff',text:'NATO'},geometry:{type:'Point',coordinates:[8,50]}},
            {type:'Feature',properties:{type:'label',color:'#ff4444',text:'Warsaw Pact'},geometry:{type:'Point',coordinates:[25,50]}},
            {type:'Feature',properties:{type:'label',color:'#44aa44',text:'Non-Aligned / Neutral'},geometry:{type:'Point',coordinates:[18,46]}}
        ]}
    },
    /* ── 5. Modern Ukraine Conflict ──────────────────── */
    {
        name: 'Modern Ukraine Conflict',
        desc: 'Front lines & key cities, 2022-present (approx late 2024)',
        geojson:{type:'FeatureCollection',features:[
            /* Front line (approx late 2024) */
            {type:'Feature',properties:{type:'polyline',color:'#ff4444',weight:3,dashStyle:'dashed',text:'Front Line (approx late 2024)'},geometry:{type:'LineString',coordinates:[
                [37.3,49.9],[37.2,49.5],[37.0,49.0],[37.3,48.7],
                [37.8,48.4],[38.0,48.1],[37.8,47.8],[37.5,47.5],
                [37.0,47.2],[36.5,46.9],[36.0,46.5],[35.5,46.2],
                [35.0,46.0],[34.5,46.3],[34.0,46.6],[33.7,46.8]
            ]}},
            /* Crimea annexation outline */
            {type:'Feature',properties:{type:'polyline',color:'#ff8800',weight:2,dashStyle:'dotted',text:'Crimea (annexed 2014)'},geometry:{type:'LineString',coordinates:[
                [33.5,46.0],[33.8,45.5],[34.0,45.0],[34.5,44.5],
                [35.0,44.2],[35.5,44.0],[36.0,44.2],[36.5,44.5],
                [36.6,45.0],[36.5,45.5],[36.0,45.8],[35.5,46.0],
                [35.0,46.1],[34.5,46.2],[34.0,46.1],[33.5,46.0]
            ]}},
            /* Russian border */
            {type:'Feature',properties:{type:'polyline',color:'#888888',weight:2,dashStyle:'solid',text:'International border'},geometry:{type:'LineString',coordinates:[
                [38.0,52.1],[37.5,51.5],[37.0,51.0],[36.5,50.5],
                [36.0,50.0],[35.5,49.5],[35.0,49.0],[36.5,48.5],
                [37.0,48.0],[36.5,47.5],[35.5,47.0],[35.0,46.5],[34.5,46.0]
            ]}},
            /* City markers */
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Kyiv (capital, defended)'},geometry:{type:'Point',coordinates:[30.52,50.45]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Kharkiv (2nd city, near front)'},geometry:{type:'Point',coordinates:[36.23,49.99]}},
            {type:'Feature',properties:{type:'marker',color:'#ff6644',text:'Mariupol (fell May 2022)'},geometry:{type:'Point',coordinates:[37.54,47.10]}},
            {type:'Feature',properties:{type:'marker',color:'#ff6644',text:'Bakhmut (fell May 2023)'},geometry:{type:'Point',coordinates:[38.00,48.60]}},
            {type:'Feature',properties:{type:'marker',color:'#ff6644',text:'Avdiivka (fell Feb 2024)'},geometry:{type:'Point',coordinates:[37.75,48.13]}},
            {type:'Feature',properties:{type:'marker',color:'#ff6644',text:'Donetsk (occupied since 2014)'},geometry:{type:'Point',coordinates:[37.80,48.00]}},
            {type:'Feature',properties:{type:'marker',color:'#ff6644',text:'Luhansk (occupied since 2014)'},geometry:{type:'Point',coordinates:[39.30,48.57]}},
            {type:'Feature',properties:{type:'marker',color:'#ff6644',text:'Zaporizhzhia (frontline city)'},geometry:{type:'Point',coordinates:[35.15,47.84]}},
            {type:'Feature',properties:{type:'marker',color:'#ff8800',text:'Kherson (liberated Nov 2022)'},geometry:{type:'Point',coordinates:[32.62,46.64]}},
            {type:'Feature',properties:{type:'marker',color:'#ff6644',text:'Sevastopol (naval base)'},geometry:{type:'Point',coordinates:[33.53,44.62]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Odesa (port city)'},geometry:{type:'Point',coordinates:[30.73,46.48]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Lviv (western hub)'},geometry:{type:'Point',coordinates:[24.03,49.84]}},
            {type:'Feature',properties:{type:'label',color:'#44aaff',text:'Ukrainian-controlled'},geometry:{type:'Point',coordinates:[32,49]}},
            {type:'Feature',properties:{type:'label',color:'#ff4444',text:'Russian-occupied'},geometry:{type:'Point',coordinates:[38,48.5]}}
        ]}
    },
    /* ── 6. WWII Western Front & D-Day ──────────────── */
    {
        name: 'WWII Western Front & D-Day',
        desc: 'Atlantic Wall, D-Day landings, Battle of the Bulge 1944-45',
        geojson:{type:'FeatureCollection',features:[
            /* Atlantic Wall — occupied European coastline */
            {type:'Feature',properties:{type:'polyline',color:'#666666',weight:2,dashStyle:'dashed',text:'Atlantic Wall (Fortifications)'},geometry:{type:'LineString',coordinates:[
                [-5.0,48.5],[-2.0,48.7],[0.0,49.3],[1.0,49.5],
                [1.6,50.0],[1.8,50.5],[2.9,51.1],[3.5,51.4],
                [4.0,51.9],[4.5,52.2],[5.0,52.7],[5.5,53.2],
                [6.0,53.5],[7.0,54.0],[8.0,54.5],[9.0,55.0],
                [10.0,55.5],[12.0,56.0],[12.5,56.5]
            ]}},
            /* D-Day front line — Normandy beaches */
            {type:'Feature',properties:{type:'polyline',color:'#44aaff',weight:3,dashStyle:'solid',text:'D-Day Front Line (Jun 6, 1944)'},geometry:{type:'LineString',coordinates:[
                [-1.2,49.7],[-0.8,49.6],[-0.5,49.4],[-0.3,49.3],
                [0.0,49.2],[0.3,49.15],[0.6,49.10]
            ]}},
            /* Breakout from Normandy */
            {type:'Feature',properties:{type:'polyline',color:'#44aaff',weight:2,dashStyle:'dashed',text:'Breakout from Normandy (Jul-Aug 1944)'},geometry:{type:'LineString',coordinates:[
                [-0.3,49.3],[0.5,49.0],[1.5,48.5],[2.0,48.2],
                [2.5,48.0],[3.0,48.5],[3.5,49.0],[4.0,49.5]
            ]}},
            /* Battle of the Bulge salient */
            {type:'Feature',properties:{type:'polyline',color:'#ff8800',weight:3,dashStyle:'dashed',text:'Battle of the Bulge (Dec 1944)'},geometry:{type:'LineString',coordinates:[
                [5.5,50.5],[5.8,50.3],[6.0,50.1],[5.8,49.9],
                [6.0,49.7],[6.3,49.8],[6.5,50.0],[6.3,50.2],
                [6.0,50.4],[5.5,50.5]
            ]}},
            /* D-Day landing beach markers */
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Utah Beach'},geometry:{type:'Point',coordinates:[-1.17,49.42]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Omaha Beach'},geometry:{type:'Point',coordinates:[-0.87,49.36]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Gold Beach'},geometry:{type:'Point',coordinates:[-0.25,49.36]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Juno Beach'},geometry:{type:'Point',coordinates:[-0.20,49.33]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Sword Beach'},geometry:{type:'Point',coordinates:[-0.10,49.31]}},
            /* Key city markers */
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Cherbourg (captured Jun 1944)'},geometry:{type:'Point',coordinates:[-1.62,49.64]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Caen (fierce fighting Jul 1944)'},geometry:{type:'Point',coordinates:[-0.37,49.18]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Paris (liberated Aug 25, 1944)'},geometry:{type:'Point',coordinates:[2.35,48.86]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Brussels (liberated Sep 1944)'},geometry:{type:'Point',coordinates:[4.35,50.85]}},
            {type:'Feature',properties:{type:'marker',color:'#ff8800',text:'Bastogne (Bulge siege)'},geometry:{type:'Point',coordinates:[5.72,50.00]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Remagen Bridge (Mar 1945)'},geometry:{type:'Point',coordinates:[7.77,50.58]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Berlin (fell May 2, 1945)'},geometry:{type:'Point',coordinates:[13.40,52.52]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Rhineland crossing (Mar 1945)'},geometry:{type:'Point',coordinates:[6.80,51.00]}},
            /* Labels */
            {type:'Feature',properties:{type:'label',color:'#ff6644',text:'German-occupied France'},geometry:{type:'Point',coordinates:[2.0,50.0]}},
            {type:'Feature',properties:{type:'label',color:'#44aaff',text:'Allied advance'},geometry:{type:'Point',coordinates:[5.5,49.0]}}
        ]}
    }
];

/* ── localStorage helpers ──────────────────────── */
const LS_KEY = 'map_saved_maps';
function getSavedMaps() {
    try { return JSON.parse(localStorage.getItem(LS_KEY)) || []; } catch { return []; }
}
function saveMapToStorage(name, annotations) {
    const saved = getSavedMaps();
    const features = annotations.map(a => {
        const isPoint = a.type === 'marker' || a.type === 'label';
        return {
            type: 'Feature',
            properties: { id: a.id, type: a.type, color: a.color, weight: a.weight, dashStyle: a.dashStyle, text: (a.extra && a.extra.text) || undefined },
            geometry: isPoint
                ? { type: 'Point', coordinates: [a.coords[1], a.coords[0]] }
                : { type: 'LineString', coordinates: a.coords.map(c => [c[1], c[0]]) }
        };
    });
    const entry = { name, date: new Date().toISOString(), annotationCount: annotations.length, geojson: { type: 'FeatureCollection', features } };
    const idx = saved.findIndex(s => s.name === name);
    if (idx >= 0) saved[idx] = entry; else saved.push(entry);
    localStorage.setItem(LS_KEY, JSON.stringify(saved));
}
function deleteSavedMap(name) {
    const saved = getSavedMaps().filter(s => s.name !== name);
    localStorage.setItem(LS_KEY, JSON.stringify(saved));
}

/* ── Import Modal Open / Close ─────────────────── */
function openImportModal() {
    $('#import-modal').classList.remove('hidden');
    importPendingFeatures = [];
    $('#import-preview').classList.add('hidden');
    $('#import-actions').classList.add('hidden');
    $('#import-url-input').value = '';
    refreshPresetsTab();
}
function closeImportModal() {
    $('#import-modal').classList.add('hidden');
    importPendingFeatures = [];
}

/* ── Expand Multi* geometries ──────────────────── */
function expandGeometry(geom) {
    if (geom.type === 'MultiLineString') return geom.coordinates.map(c => ({type:'LineString',coordinates:c}));
    if (geom.type === 'Polygon') return geom.coordinates.map(ring => ({type:'LineString',coordinates:ring}));
    if (geom.type === 'MultiPolygon') {
        const out = [];
        geom.coordinates.forEach(poly => poly.forEach(ring => out.push({type:'LineString',coordinates:ring})));
        return out;
    }
    if (geom.type === 'MultiPoint') return geom.coordinates.map(c => ({type:'Point',coordinates:c}));
    return [geom];
}

/* ── Parse GeoJSON into preview features ──────── */
function parseGeoJSONForPreview(data) {
    const features = data.type === 'FeatureCollection' ? data.features
        : data.type === 'Feature' ? [data] : [];
    const parsed = [];
    features.forEach((f, idx) => {
        if (!f.geometry) return;
        const p = f.properties || {};
        const geoms = expandGeometry(f.geometry);
        geoms.forEach(g => {
            parsed.push({
                id: idx + '_' + parsed.length,
                name: p.text || p.name || p.label || (g.type === 'LineString' ? 'Line #' + (parsed.length + 1) : 'Pin #' + (parsed.length + 1)),
                type: p.type || (g.type === 'Point' ? (p.text ? 'label' : 'marker') : 'polyline'),
                color: p.color || '#ff0000',
                weight: p.weight || 3,
                dashStyle: p.dashStyle || 'solid',
                geometryType: g.type,
                geometry: g,
                text: p.text || undefined,
                checked: true
            });
        });
    });
    return parsed;
}

/* ── Render Preview List ──────────────────────── */
function renderPreview(features) {
    const list = $('#import-preview-list');
    list.innerHTML = '';
    const hasFolders = features.some(f => f.folder);
    if (hasFolders) {
        /* Group by folder for KML data */
        const groups = {};
        features.forEach((f, i) => {
            const g = f.folder || 'Other';
            if (!groups[g]) groups[g] = [];
            groups[g].push({ feat: f, idx: i });
        });
        Object.keys(groups).forEach(folderName => {
            const grp = groups[folderName];
            const hdr = document.createElement('div');
            hdr.className = 'preview-folder';
            hdr.innerHTML = '<input type="checkbox" class="folder-toggle" data-folder="' + escapeHtml(folderName) + '" checked>'
                + '<span class="folder-name">' + escapeHtml(folderName) + '</span>'
                + '<span class="folder-count">' + grp.length + ' features</span>';
            hdr.querySelector('.folder-toggle').addEventListener('change', e => {
                const checked = e.target.checked;
                grp.forEach(g => { features[g.idx].checked = checked; });
                list.querySelectorAll('.preview-item[data-folder="' + CSS.escape(folderName) + '"] input[type="checkbox"]')
                    .forEach(cb => cb.checked = checked);
                updateSelectAllState();
            });
            list.appendChild(hdr);
            grp.forEach(g => {
                const div = document.createElement('div');
                div.className = 'preview-item';
                div.dataset.folder = folderName;
                div.innerHTML = '<input type="checkbox" data-idx="' + g.idx + '"' + (g.feat.checked ? ' checked' : '') + '>'
                    + '<span class="preview-item-color" style="background:' + g.feat.color + '"></span>'
                    + '<span class="preview-item-label">' + escapeHtml(g.feat.name) + '</span>'
                    + '<span class="preview-item-type">' + g.feat.geometryType + '</span>';
                div.querySelector('input').addEventListener('change', e => {
                    features[g.idx].checked = e.target.checked;
                    const allChecked = grp.every(x => features[x.idx].checked);
                    list.querySelector('.folder-toggle[data-folder="' + CSS.escape(folderName) + '"]').checked = allChecked;
                    updateSelectAllState();
                });
                list.appendChild(div);
            });
        });
    } else {
        /* Flat list for GeoJSON data */
        features.forEach((f, i) => {
            const div = document.createElement('div');
            div.className = 'preview-item';
            div.innerHTML = '<input type="checkbox" data-idx="' + i + '"' + (f.checked ? ' checked' : '') + '>'
                + '<span class="preview-item-color" style="background:' + f.color + '"></span>'
                + '<span class="preview-item-label">' + escapeHtml(f.name) + '</span>'
                + '<span class="preview-item-type">' + f.geometryType + '</span>';
            div.querySelector('input').addEventListener('change', e => {
                features[i].checked = e.target.checked;
                updateSelectAllState();
            });
            list.appendChild(div);
        });
    }
    $('#import-preview-count').textContent = features.length + ' feature' + (features.length !== 1 ? 's' : '') + ' found';
    $('#import-preview').classList.remove('hidden');
    $('#import-actions').classList.remove('hidden');
    $('#import-select-all').checked = features.every(f => f.checked);
}

/* ── Select All Checkbox Sync ──────────────────── */
function updateSelectAllState() {
    const all = importPendingFeatures;
    $('#import-select-all').checked = all.length > 0 && all.every(f => f.checked);
}

/* ── Confirm Import ────────────────────────────── */
function confirmImport() {
    const toImport = importPendingFeatures.filter(f => f.checked);
    if (toImport.length === 0) { alert('No features selected.'); return; }

    /* Group by folder for layer control */
    const folderGroups = {};
    let count = 0;
    toImport.forEach(f => {
        const folder = f.folder || 'Imported';
        if (!folderGroups[folder]) folderGroups[folder] = L.layerGroup();
        const grp = folderGroups[folder];

        if (f.geometryType === 'LineString') {
            const coords = f.geometry.coordinates.map(c => [c[1], c[0]]);
            const layer = L.polyline(coords, {
                color: f.color, weight: f.weight, dashArray: DASH[f.dashStyle], opacity: 1,
                lineCap: 'round', lineJoin: 'round', renderer: canvasRenderer
            });
            /* Hover tooltip with name */
            layer.bindTooltip(f.name || '', { sticky: true, className: 'import-tooltip' });
            /* Store annotation */
            const prevColor = S.color, prevW = S.weight, prevDS = S.dashStyle;
            S.color = f.color; S.weight = f.weight; S.dashStyle = f.dashStyle;
            storeAnnotation(layer, f.type, coords);
            S.color = prevColor; S.weight = prevW; S.dashStyle = prevDS;
            layer.addTo(grp);
            count++;
        } else if (f.geometryType === 'Polygon') {
            const coords = f.geometry.coordinates.map(c => [c[1], c[0]]);
            const layer = L.polygon(coords, {
                color: f.color, fillColor: f.fillColor || f.color,
                fillOpacity: f.fillOpacity || 0.3,
                weight: Math.max(1, f.weight || 2), renderer: canvasRenderer
            });
            layer.bindTooltip(f.name || '', { sticky: true, className: 'import-tooltip' });
            const prevColor = S.color;
            S.color = f.fillColor || f.color;
            storeAnnotation(layer, 'polygon', f.geometry.coordinates, { text: f.text || f.name });
            S.color = prevColor;
            layer.addTo(grp);
            attachHighlightHandler(layer);
            count++;
        } else if (f.geometryType === 'Point') {
            const ll = [f.geometry.coordinates[1], f.geometry.coordinates[0]];
            const latlng = L.latLng(ll);
            const text = f.text || f.name || '';
            const marker = L.circleMarker(latlng, {
                radius: 5, fillColor: f.color, color: '#fff',
                weight: 1.5, fillOpacity: 0.9, renderer: canvasRenderer
            });
            if (text) marker.bindPopup('<b>' + escapeHtml(text) + '</b>', { maxWidth: 250 });
            const prevColor = S.color;
            S.color = f.color;
            storeAnnotation(marker, f.type || 'marker', ll, { text });
            S.color = prevColor;
            marker.addTo(grp);
            count++;
        }
    });

    /* Add all folder groups to map */
    const importedFolders = {};
    Object.keys(folderGroups).forEach(name => {
        folderGroups[name].addTo(map);
        importedFolders[name] = folderGroups[name];
    });

    /* Add to layer control */
    if (Object.keys(importedFolders).length > 0) {
        const base = { '🗺️ Standard': osm, '🏔 Topographic': topo, '🛰 Satellite': sat };
        /* Preserve existing overlay layers from the control */
        map._layers_control_overlays = map._layers_control_overlays || {};
        Object.assign(map._layers_control_overlays, importedFolders);
        if (window._importCtrl) map.removeControl(window._importCtrl);
        window._importCtrl = L.control.layers(base, map._layers_control_overlays, { collapsed: false, position: 'topright' }).addTo(map);
    }

    closeImportModal();
    if (count > 0) {
        $('#status-text').textContent = 'Imported ' + count + ' annotation' + (count !== 1 ? 's' : '') + ' from map.';
    }
}

/* ── Presets Tab Rendering ─────────────────────── */
function refreshPresetsTab() {
    const saved = getSavedMaps();
    const savedSection = $('#preset-saved-section');
    const savedList = $('#preset-saved-list');
    const builtInList = $('#preset-built-in-list');

    if (saved.length > 0) {
        savedSection.classList.remove('hidden');
        savedList.innerHTML = '';
        saved.forEach(s => {
            const div = document.createElement('div');
            div.className = 'preset-item';
            div.innerHTML = '<div class="preset-item-info"><div class="preset-item-name">' + escapeHtml(s.name) + '</div>'
                + '<div class="preset-item-desc">' + s.annotationCount + ' annotations</div></div>'
                + '<div class="preset-item-actions">'
                + '<button class="preset-item-btn load-preset" data-type="saved" data-name="' + escapeHtml(s.name) + '">Load</button>'
                + '<button class="preset-item-btn delete-btn delete-preset" data-name="' + escapeHtml(s.name) + '">✕</button>'
                + '</div>';
            savedList.appendChild(div);
        });
    } else {
        savedSection.classList.add('hidden');
    }

    builtInList.innerHTML = '';
    BUILT_IN_PRESETS.forEach((p, i) => {
        const div = document.createElement('div');
        div.className = 'preset-item';
        div.innerHTML = '<div class="preset-item-info"><div class="preset-item-name">' + escapeHtml(p.name) + '</div>'
            + '<div class="preset-item-desc">' + escapeHtml(p.desc) + '</div></div>'
            + '<div class="preset-item-actions">'
            + '<button class="preset-item-btn load-preset" data-type="builtin" data-idx="' + i + '">Load</button>'
            + '</div>';
        builtInList.appendChild(div);
    });

    builtInList.querySelectorAll('.load-preset').forEach(btn => btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.idx);
        if (BUILT_IN_PRESETS[idx]) loadPresetIntoPreview(BUILT_IN_PRESETS[idx].geojson);
    }));
    savedList.querySelectorAll('.load-preset').forEach(btn => btn.addEventListener('click', () => {
        const name = btn.dataset.name;
        const mapData = getSavedMaps().find(s => s.name === name);
        if (mapData && mapData.geojson) loadPresetIntoPreview(mapData.geojson);
    }));
    savedList.querySelectorAll('.delete-preset').forEach(btn => btn.addEventListener('click', () => {
        const name = btn.dataset.name;
        if (confirm('Delete saved map "' + name + '"?')) { deleteSavedMap(name); refreshPresetsTab(); }
    }));
}

function loadPresetIntoPreview(geojson) {
    importPendingFeatures = parseGeoJSONForPreview(geojson);
    if (importPendingFeatures.length === 0) { alert('No compatible features found.'); return; }
    renderPreview(importPendingFeatures);
}

/* ── Import Modal Event Wiring ─────────────────── */
function wireImportModal() {
    $('#btn-import').addEventListener('click', openImportModal);
    $('#import-modal-close').addEventListener('click', closeImportModal);
    $('#import-modal').addEventListener('click', e => {
        if (e.target === $('#import-modal')) closeImportModal();
    });

    $$('.import-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.import-tab').forEach(t => t.classList.remove('active'));
            $$('.import-tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            const target = $('#import-tab-' + tab.dataset.tab);
            if (target) target.classList.add('active');
            importPendingFeatures = [];
            $('#import-preview').classList.add('hidden');
            $('#import-actions').classList.add('hidden');
        });
    });

    $('#import-browse-btn').addEventListener('click', () => $('#file-input').click());
    $('#file-input').addEventListener('change', e => {
        if (e.target.files.length > 0) { handleFileImport(e.target.files[0]); e.target.value = ''; }
    });

    const dz = $('#import-dropzone');
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
    dz.addEventListener('drop', e => {
        e.preventDefault(); dz.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleFileImport(e.dataTransfer.files[0]);
    });

    $('#import-url-btn').addEventListener('click', handleUrlImport);
    $('#import-url-input').addEventListener('keydown', e => { if (e.key === 'Enter') handleUrlImport(); });

    $('#import-select-all').addEventListener('change', e => {
        const checked = e.target.checked;
        importPendingFeatures.forEach(f => f.checked = checked);
        $$('#import-preview-list input[type="checkbox"]').forEach(cb => cb.checked = checked);
    });

    $('#import-cancel-btn').addEventListener('click', closeImportModal);
    $('#import-confirm-btn').addEventListener('click', confirmImport);
}

function handleFileImport(file) {
    const nameLC = file.name.toLowerCase();
    const isKML = nameLC.endsWith('.kml');
    const isKMZ = nameLC.endsWith('.kmz');
    const isGPX = nameLC.endsWith('.gpx');

    /* ── KMZ (zipped KML) — read as binary ────── */
    if (isKMZ) {
        if (typeof JSZip === 'undefined') { alert('JSZip library not loaded. Please refresh the page and try again.'); return; }
        const reader = new FileReader();
        reader.onload = function (ev) {
            JSZip.loadAsync(ev.target.result).then(zip => {
                const kmlFile = Object.keys(zip.files).find(f => f.toLowerCase().endsWith('.kml'));
                if (!kmlFile) { alert('No KML file found inside the KMZ archive.'); return; }
                return zip.files[kmlFile].async('string');
            }).then(kmlText => {
                if (!kmlText) return;
                try {
                    importPendingFeatures = parseKML(kmlText);
                    if (importPendingFeatures.length === 0) { alert('No compatible features found in KMZ.'); return; }
                    renderPreview(importPendingFeatures);
                } catch (err) {
                    if (err && err.isNetworkLink) {
                        const netUrl = err.url;
                        alert('This KMZ contains a Google Maps NetworkLink.\nFetching the full map data…');
                        const proxies = ['http://localhost:8080/kml?url=', '', 'https://corsproxy.io/?', 'https://api.allorigins.win/raw?url='];
                        let attempt = 0;
                        function tryFetch() {
                            const target = proxies[attempt] ? proxies[attempt] + encodeURIComponent(netUrl) : netUrl;
                            return fetch(target).then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
                                .then(t => { if (!t.includes('<Placemark') && !t.includes('<Document')) throw new Error('Not KML'); return t; })
                                .catch(e => { attempt++; if (attempt < proxies.length) return tryFetch(); throw e; });
                        }
                        tryFetch().then(fullKml => {
                            importPendingFeatures = parseKML(fullKml);
                            if (importPendingFeatures.length === 0) { alert('No features found after fetching full KML.'); return; }
                            renderPreview(importPendingFeatures);
                        }).catch(fetchErr => {
                            console.error('NetworkLink fetch failed:', fetchErr);
                            openProxyModal('https://www.google.com/maps/d/viewer?mid=' + encodeURIComponent(netUrl));
                            startProxyPolling();
                        });
                    } else { alert('Failed to parse KML inside KMZ.\n' + (err.message || err)); }
                }
            }).catch(err => { console.error('KMZ unzip failed:', err); alert('Failed to open KMZ file.\n' + err.message); });
        };
        reader.readAsArrayBuffer(file);
        return;
    }

    /* ── GPX — read as text ────────────────────── */
    if (isGPX) {
        const reader = new FileReader();
        reader.onload = function (ev) {
            try {
                importPendingFeatures = parseGPX(ev.target.result);
                if (importPendingFeatures.length === 0) { alert('No compatible features found in GPX file.'); return; }
                renderPreview(importPendingFeatures);
            } catch (err) { console.error('GPX import failed:', err); alert('Failed to parse GPX file.\n' + (err.message || err)); }
        };
        reader.readAsText(file);
        return;
    }

    /* ── KML / GeoJSON — read as text ─────────── */
    const reader = new FileReader();
    reader.onload = function (ev) {
        const text = ev.target.result;
        if (isKML) {
            try {
                importPendingFeatures = parseKML(text);
                if (importPendingFeatures.length === 0) { alert('No compatible features found.'); return; }
                renderPreview(importPendingFeatures);
            } catch (err) {
                if (err && err.isNetworkLink) {
                    /* Google My Maps stub — fetch the real KML from the NetworkLink URL */
                    const netUrl = err.url;
                    const mapName = err.mapName || 'Google Maps';
                    alert('This KML is a Google Maps shortcut file.\nFetching the full map data…');
                    const proxies = [
                        'http://localhost:8080/kml?url=',
                        '',
                        'https://corsproxy.io/?',
                        'https://api.allorigins.win/raw?url='
                    ];
                    let attempt = 0;
                    function tryFetch() {
                        const target = proxies[attempt] ? proxies[attempt] + encodeURIComponent(netUrl) : netUrl;
                        return fetch(target).then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
                            .then(t => { if (!t.includes('<Placemark') && !t.includes('<Document')) throw new Error('Not KML data'); return t; })
                            .catch(err => { attempt++; if (attempt < proxies.length) return tryFetch(); throw err; });
                    }
                    tryFetch().then(fullKml => {
                        importPendingFeatures = parseKML(fullKml);
                        if (importPendingFeatures.length === 0) { alert('No features found after fetching full KML.'); return; }
                        renderPreview(importPendingFeatures);
                    }).catch(fetchErr => {
                        console.error('NetworkLink fetch failed:', fetchErr);
                        openProxyModal('https://www.google.com/maps/d/viewer?mid=' + encodeURIComponent(netUrl));
                        startProxyPolling();
                    });
                } else {
                    alert('Failed to parse file.\n' + (err.message || err));
                }
            }
            return;
        }
        /* GeoJSON */
        try {
            importPendingFeatures = parseGeoJSONForPreview(JSON.parse(text));
            if (importPendingFeatures.length === 0) { alert('No compatible features found.'); return; }
            renderPreview(importPendingFeatures);
        } catch (err) {
            console.error('Import failed:', err);
            alert('Failed to parse file.\n' + err.message);
        }
    };
    reader.readAsText(file);
}

/* =====================================================
   Google My Maps KML Import
   ===================================================== */
function parseGoogleMapsUrl(url) {
    const p = /google\.com\/maps\/d\/(?:u\/\d+\/)?(?:viewer|edit|kml|embed(?:ded)?)\?.*mid=([a-zA-Z0-9_-]+)/;
    const m = url.match(p);
    return m ? m[1] : null;
}
function kmlColorToHex(kmlColor) {
    if (!kmlColor || kmlColor.length < 6) return '#888888';
    return '#' + (kmlColor.slice(6, 8) || '88') + (kmlColor.slice(4, 6) || '88') + (kmlColor.slice(2, 4) || '88');
}
function kmlWidthToWeight(w) {
    const n = parseFloat(w);
    if (isNaN(n) || n <= 0) return 2;
    if (n < 0.1) return Math.max(2, Math.round(n * 1000));
    if (n < 1) return Math.max(2, Math.round(n * 20));
    return Math.min(10, Math.max(2, Math.round(n)));
}
function resolveKmlStyle(styleUrl, styleMap) {
    if (!styleUrl) return { color: '#888888', width: 2, type: 'unknown' };
    let id = styleUrl.replace(/^#/, '');
    if (styleMap[id] && styleMap[id].normal) id = styleMap[id].normal;
    return styleMap[id] || { color: '#888888', width: 2, type: 'unknown' };
}
function kmlAlphaToOpacity(kmlColor) {
    if (!kmlColor || kmlColor.length < 2) return 0.3;
    const alpha = parseInt(kmlColor.slice(0, 2), 16);
    return isNaN(alpha) ? 0.3 : alpha / 255;
}

function buildKmlStyleMap(doc) {
    const styles = {};
    const styleEls = doc.getElementsByTagName('Style');
    Array.from(styleEls).forEach(s => {
        const id = s.getAttribute('id'); if (!id) return;
        const entry = {};
        const lineEl = s.getElementsByTagName('LineStyle')[0];
        const iconEl = s.getElementsByTagName('IconStyle')[0];
        const polyEl = s.getElementsByTagName('PolyStyle')[0];
        if (lineEl) {
            const c = lineEl.getElementsByTagName('color')[0];
            const w = lineEl.getElementsByTagName('width')[0];
            entry.color = c ? kmlColorToHex(c.textContent.trim()) : '#888888';
            entry.width = w ? kmlWidthToWeight(w.textContent.trim()) : 2;
            entry.type = 'line';
        } else if (iconEl) {
            const c = iconEl.getElementsByTagName('color')[0];
            entry.color = c ? kmlColorToHex(c.textContent.trim()) : '#888888';
            entry.width = 3; entry.type = 'icon';
        }
        if (polyEl) {
            const pc = polyEl.getElementsByTagName('color')[0];
            if (pc) {
                const raw = pc.textContent.trim();
                entry.fillColor = kmlColorToHex(raw);
                entry.fillOpacity = kmlAlphaToOpacity(raw);
            }
            if (!entry.color) entry.color = entry.fillColor || '#888888';
            if (!entry.type) entry.type = 'poly';
        }
        styles[id] = entry;
    });
    const styleMaps = doc.getElementsByTagName('StyleMap');
    Array.from(styleMaps).forEach(sm => {
        const id = sm.getAttribute('id'); if (!id) return;
        const pairs = {};
        Array.from(sm.children).forEach(p => {
            if (p.localName !== 'Pair') return;
            const k = p.getElementsByTagName('key')[0];
            const ref = p.getElementsByTagName('styleUrl')[0];
            if (k && ref) pairs[k.textContent.trim()] = ref.textContent.trim().replace(/^#/, '');
        });
        styles[id] = pairs.normal ? (styles[pairs.normal] || { normal: pairs.normal }) : pairs;
    });
    return styles;
}
function parseKmlCoords(text) {
    if (!text) return [];
    return text.trim().split(/\s+/).map(pair => {
        const p = pair.split(',');
        return [parseFloat(p[0]), parseFloat(p[1])]; /* KML lon,lat → store as lon,lat (GeoJSON convention) */
    }).filter(c => !isNaN(c[0]) && !isNaN(c[1]));
}
function extractKmlFeatures(node, styleMap, folderName) {
    const features = [];
    const children = Array.from(node.children || []);
    children.forEach(el => {
        if (el.localName === 'Placemark') {
            const nameEl = el.querySelector('name');
            const styleEl = el.querySelector('styleUrl');
            const name = (nameEl ? nameEl.textContent.trim() : '') || folderName || 'Unnamed';
            const style = resolveKmlStyle(styleEl ? styleEl.textContent.trim() : '', styleMap);

            const multiGeo = el.querySelector('MultiGeometry');
            if (multiGeo) { features.push(...extractKmlFeatures(multiGeo, styleMap, name)); return; }

            const pt = el.querySelector('Point > coordinates');
            if (pt) {
                const coords = parseKmlCoords(pt.textContent);
                if (coords.length > 0) features.push({ id:'kml_'+features.length, name, type: style.type==='icon'?'marker':'label',
                    color: style.color, weight: 3, dashStyle:'solid', geometryType:'Point',
                    geometry:{type:'Point',coordinates:[coords[0][0],coords[0][1]]}, text:name, checked:true, folder:folderName||'Ungrouped' });
                return;
            }
            const ls = el.querySelector('LineString > coordinates');
            if (ls) {
                const coords = parseKmlCoords(ls.textContent);
                if (coords.length > 1) features.push({ id:'kml_'+features.length, name, type:'polyline',
                    color: style.color, weight: style.width||2, dashStyle:'solid', geometryType:'LineString',
                    geometry:{type:'LineString',coordinates:coords}, text:name, checked:true, folder:folderName||'Ungrouped' });
                return;
            }
            const poly = el.querySelector('Polygon');
            if (poly) {
                const outer = poly.querySelector('outerBoundaryIs > LinearRing > coordinates');
                if (outer) {
                    const coords = parseKmlCoords(outer.textContent);
                    if (coords.length > 1) features.push({ id:'kml_'+features.length, name, type:'polygon',
                        color: style.color, fillColor: style.fillColor || style.color,
                        fillOpacity: style.fillOpacity || 0.3,
                        weight: Math.max(1, style.width||2), dashStyle:'solid', geometryType:'Polygon',
                        geometry:{type:'Polygon',coordinates:coords}, text:name, checked:true, folder:folderName||'Ungrouped' });
                }
            }
        } else if (el.localName === 'Folder' || el.localName === 'Document') {
            const subName = (el.querySelector('name') || {}).textContent || folderName || '';
            features.push(...extractKmlFeatures(el, styleMap, subName || folderName));
        }
    });
    return features;
}

function parseKML(kmlText) {
    /* Strip default KML namespace so getElementsByTagName works in browsers */
    kmlText = kmlText.replace(/xmlns="http:\/\/www\.opengis\.net\/kml\/2\.2"/g, '');
    const doc = new DOMParser().parseFromString(kmlText, 'application/xml');
    const parseErr = doc.querySelector('parsererror');
    if (parseErr) { console.error('KML parse error:', parseErr.textContent.slice(0, 300)); throw new Error('Invalid KML/XML'); }
    const root = doc.getElementsByTagName('Document')[0] || doc.getElementsByTagName('Folder')[0] || doc.documentElement;
    const nameEl = root.getElementsByTagName('name')[0];
    const docName = nameEl ? nameEl.textContent.trim() : 'Google Maps';

    /* Check for NetworkLink stub (Google My Maps download quirk) */
    const netLink = root.getElementsByTagName('NetworkLink')[0];
    if (netLink) {
        const linkHref = (netLink.getElementsByTagName('href')[0] || {}).textContent
            || (netLink.getElementsByTagName('link')[0] && netLink.getElementsByTagName('link')[0].getElementsByTagName('href')[0] || {}).textContent
            || '';
        if (linkHref) {
            throw { isNetworkLink: true, url: linkHref.trim(), mapName: docName };
        }
    }

    const styleMap = buildKmlStyleMap(doc);
    const features = extractKmlFeatures(root, styleMap, docName);
    features.forEach(f => { f.mapName = docName; });
    return features;
}

/* =====================================================
   GPX Import (GPS Exchange Format)
   ===================================================== */
function parseGPX(gpxText) {
    gpxText = gpxText.replace(/xmlns="http:\/\/www\.topografix\.com\/GPX\/1\/[12]"/g, '');
    const doc = new DOMParser().parseFromString(gpxText, 'application/xml');
    const parseErr = doc.querySelector('parsererror');
    if (parseErr) { console.error('GPX parse error:', parseErr.textContent.slice(0, 300)); throw new Error('Invalid GPX/XML'); }
    const features = [];
    let idx = 0;

    function getTagText(parent, tag) {
        const el = parent.getElementsByTagName(tag)[0];
        return el ? el.textContent.trim() : '';
    }

    /* Waypoints → Markers */
    const wpts = doc.getElementsByTagName('wpt');
    for (let i = 0; i < wpts.length; i++) {
        const wpt = wpts[i];
        const lat = parseFloat(wpt.getAttribute('lat'));
        const lon = parseFloat(wpt.getAttribute('lon'));
        if (isNaN(lat) || isNaN(lon)) continue;
        const name = getTagText(wpt, 'name') || getTagText(wpt, 'desc') || 'Waypoint ' + (idx + 1);
        const desc = getTagText(wpt, 'desc');
        features.push({ id: 'gpx_' + idx, name, type: 'label', color: '#ffcc00', weight: 3, dashStyle: 'solid',
            geometryType: 'Point', geometry: { type: 'Point', coordinates: [lon, lat] },
            text: desc ? name + '\n' + desc : name, checked: true, folder: 'Waypoints' });
        idx++;
    }

    /* Routes → Polylines */
    const routes = doc.getElementsByTagName('rte');
    for (let i = 0; i < routes.length; i++) {
        const rte = routes[i];
        const rteName = getTagText(rte, 'name') || 'Route ' + (i + 1);
        const rtepts = rte.getElementsByTagName('rtept');
        const coords = [];
        for (let j = 0; j < rtepts.length; j++) {
            const lat = parseFloat(rtepts[j].getAttribute('lat'));
            const lon = parseFloat(rtepts[j].getAttribute('lon'));
            if (!isNaN(lat) && !isNaN(lon)) coords.push([lon, lat]);
        }
        if (coords.length >= 2) {
            features.push({ id: 'gpx_' + idx, name: rteName, type: 'polyline', color: '#ff8800', weight: 3, dashStyle: 'dashed',
                geometryType: 'LineString', geometry: { type: 'LineString', coordinates: coords },
                text: rteName, checked: true, folder: 'Routes' });
            idx++;
        }
    }

    /* Tracks → Polylines */
    const tracks = doc.getElementsByTagName('trk');
    for (let i = 0; i < tracks.length; i++) {
        const trk = tracks[i];
        const trkName = getTagText(trk, 'name') || 'Track ' + (i + 1);
        const trksegs = trk.getElementsByTagName('trkseg');
        for (let s = 0; s < trksegs.length; s++) {
            const trkpts = trksegs[s].getElementsByTagName('trkpt');
            const coords = [];
            for (let j = 0; j < trkpts.length; j++) {
                const lat = parseFloat(trkpts[j].getAttribute('lat'));
                const lon = parseFloat(trkpts[j].getAttribute('lon'));
                if (!isNaN(lat) && !isNaN(lon)) coords.push([lon, lat]);
            }
            if (coords.length >= 2) {
                const segName = trksegs.length > 1 ? trkName + ' (segment ' + (s + 1) + ')' : trkName;
                features.push({ id: 'gpx_' + idx, name: segName, type: 'polyline', color: '#44aaff', weight: 3, dashStyle: 'solid',
                    geometryType: 'LineString', geometry: { type: 'LineString', coordinates: coords },
                    text: segName, checked: true, folder: 'Tracks' });
                idx++;
            }
        }
    }

    return features;
}

function fetchGoogleKML(mid, btn) {
    const kmlUrl = 'https://www.google.com/maps/d/u/0/kml?mid=' + mid + '&forcekml=1';
    const proxies = [
        'http://localhost:8080/kml?url=',
        '',
        'https://corsproxy.io/?',
        'https://api.allorigins.win/raw?url='
    ];
    let attempt = 0;
    function tryFetch() {
        const target = proxies[attempt] ? proxies[attempt] + encodeURIComponent(kmlUrl) : kmlUrl;
        btn.textContent = attempt === 0 ? 'Fetching KML…' : 'Trying proxy…';
        return fetch(target).then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
            .then(t => { if (!t.includes('<kml') && !t.includes('<Document')) throw new Error('Not KML'); return t; })
            .catch(err => { attempt++; if (attempt < proxies.length) return tryFetch(); throw err; });
    }
    return tryFetch();
}

function handleUrlImport() {
    const url = $('#import-url-input').value.trim();
    if (!url) { alert('Please enter a URL.'); return; }
    const btn = $('#import-url-btn');
    btn.disabled = true; btn.textContent = 'Fetching…';

    /* ── Google Maps URL → KML proxy flow ────── */
    const mid = parseGoogleMapsUrl(url);
    if (mid) {
        fetchGoogleKML(mid, btn).then(kmlText => {
            importPendingFeatures = parseKML(kmlText);
            if (importPendingFeatures.length === 0) { alert('No features found in this Google Map.'); return; }
            renderPreview(importPendingFeatures);
        }).catch(err => {
            console.error('KML import failed:', err);
            openProxyModal(url);
            startProxyPolling();
        }).finally(() => { btn.disabled = false; btn.textContent = 'Fetch & Import'; });
        return;
    }

    /* ── Detect format from URL extension ────── */
    const urlLower = url.split('?')[0].toLowerCase();
    const isGPXUrl = urlLower.endsWith('.gpx');
    const isKmlUrl = urlLower.endsWith('.kml');
    const isKmzUrl = urlLower.endsWith('.kmz');

    function fetchAsText() {
        return fetch(url).then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); });
    }
    function fetchAsBinary() {
        return fetch(url).then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.arrayBuffer(); });
    }

    function handleError(err) {
        console.error('URL import failed:', err);
        alert('Failed to fetch data from URL.\n' + err.message);
        btn.disabled = false; btn.textContent = 'Fetch & Import';
    }

    /* ── GPX URL ─────────────────────────────── */
    if (isGPXUrl) {
        fetchAsText().then(gpxText => {
            importPendingFeatures = parseGPX(gpxText);
            if (importPendingFeatures.length === 0) { alert('No compatible features found in GPX.'); return; }
            renderPreview(importPendingFeatures);
        }).catch(handleError).finally(() => { btn.disabled = false; btn.textContent = 'Fetch & Import'; });
        return;
    }

    /* ── KML URL ─────────────────────────────── */
    if (isKmlUrl) {
        fetchAsText().then(kmlText => {
            importPendingFeatures = parseKML(kmlText);
            if (importPendingFeatures.length === 0) { alert('No compatible features found in KML.'); return; }
            renderPreview(importPendingFeatures);
        }).catch(handleError).finally(() => { btn.disabled = false; btn.textContent = 'Fetch & Import'; });
        return;
    }

    /* ── KMZ URL (zipped) ────────────────────── */
    if (isKmzUrl) {
        if (typeof JSZip === 'undefined') { alert('JSZip library not loaded. Please refresh the page and try again.'); btn.disabled = false; btn.textContent = 'Fetch & Import'; return; }
        fetchAsBinary().then(buf => JSZip.loadAsync(buf))
            .then(zip => {
                const kmlFile = Object.keys(zip.files).find(f => f.toLowerCase().endsWith('.kml'));
                if (!kmlFile) throw new Error('No KML file found inside KMZ');
                return zip.files[kmlFile].async('string');
            }).then(kmlText => {
                importPendingFeatures = parseKML(kmlText);
                if (importPendingFeatures.length === 0) { alert('No compatible features found in KMZ.'); return; }
                renderPreview(importPendingFeatures);
            }).catch(handleError).finally(() => { btn.disabled = false; btn.textContent = 'Fetch & Import'; });
        return;
    }

    /* ── Generic: try GeoJSON → KML → GPX ────── */
    const CORS_PROXIES = ['', 'https://corsproxy.io/?', 'https://api.allorigins.win/raw?url='];

    function tryFetchWithProxy(proxyIdx) {
        const target = CORS_PROXIES[proxyIdx] ? CORS_PROXIES[proxyIdx] + encodeURIComponent(url) : url;
        return fetch(target).then(r => {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.text().then(t => {
                if (t.includes('<kml') || t.includes('<Document') || t.includes('<Placemark')) return { text: t, format: 'kml' };
                if (t.includes('<gpx')) return { text: t, format: 'gpx' };
                try { JSON.parse(t); return { text: t, format: 'geojson' }; } catch (_) {}
                return { text: t, format: 'unknown' };
            });
        });
    }

    function attemptImport(proxyIdx) {
        if (proxyIdx >= CORS_PROXIES.length) {
            btn.disabled = false; btn.textContent = 'Fetch & Import';
            alert('Could not import data from this URL.\nMake sure the URL points to a GeoJSON, KML, or GPX file.');
            return;
        }
        tryFetchWithProxy(proxyIdx).then(result => {
            let parsed = false;
            if (result.format === 'geojson') {
                try { importPendingFeatures = parseGeoJSONForPreview(JSON.parse(result.text)); parsed = importPendingFeatures.length > 0; } catch (_) {}
            } else if (result.format === 'kml') {
                try { importPendingFeatures = parseKML(result.text); parsed = importPendingFeatures.length > 0; } catch (_) {}
            } else if (result.format === 'gpx') {
                try { importPendingFeatures = parseGPX(result.text); parsed = importPendingFeatures.length > 0; } catch (_) {}
            }
            if (parsed) { renderPreview(importPendingFeatures); btn.disabled = false; btn.textContent = 'Fetch & Import'; }
            else { attemptImport(proxyIdx + 1); }
        }).catch(() => attemptImport(proxyIdx + 1));
    }

    attemptImport(0);
}

/* ── Save Modal ────────────────────────────────── */
function wireSaveModal() {
    $('#btn-save-map').addEventListener('click', () => {
        if (S.annotations.length === 0) { alert('No annotations to save.'); return; }
        $('#save-modal').classList.remove('hidden');
        $('#save-name-input').value = '';
        setTimeout(() => $('#save-name-input').focus(), 100);
    });
    $('#save-cancel-btn').addEventListener('click', () => $('#save-modal').classList.add('hidden'));
    $('#save-modal').addEventListener('click', e => {
        if (e.target === $('#save-modal')) $('#save-modal').classList.add('hidden');
    });
    $('#save-confirm-btn').addEventListener('click', () => {
        const name = $('#save-name-input').value.trim();
        if (!name) { alert('Please enter a name.'); return; }
        saveMapToStorage(name, S.annotations);
        $('#save-modal').classList.add('hidden');
        $('#status-text').textContent = 'Map saved as "' + name + '"';
    });
    $('#save-name-input').addEventListener('keydown', e => {
        if (e.key === 'Enter') $('#save-confirm-btn').click();
        if (e.key === 'Escape') $('#save-modal').classList.add('hidden');
    });
}

/* =====================================================
   Map Events
   ===================================================== */
let mouseDown = false;

map.on('mousedown', e => {
    mouseDown = true;
    if (S.tool === 'freehand') startFreehand(e);
    if (S.tool === 'arrow') startArrow(e);
    if (S.tool === 'eraser') eraserDragStart();
});

map.on('mousemove', e => {
    S.lastLatLng = e.latlng;
    if (S.tool === 'freehand' && S.drawing) moveFreehand(e);
    if (S.tool === 'arrow' && S.arrowDrawing) moveArrow(e);
    if (S.tool === 'eraser' && S.eraserDragging) eraserDragMove(e);
    if (S.tool === 'polyline') movePolyPreview(e);
    if (S.tool === 'measure') moveMeasurePreview(e);
});

map.on('mouseup', () => {
    mouseDown = false;
    if (S.tool === 'freehand' && S.drawing) finishFreehand();
    if (S.tool === 'arrow' && S.arrowDrawing) finishArrow();
    if (S.tool === 'eraser') eraserDragEnd();
});

map.on('click', e => {
    if (S.tool === 'polyline') addPolylinePoint(e);
    if (S.tool === 'marker') placeMarker(e);
    if (S.tool === 'fire') placeFire(e);
    if (S.tool === 'measure') startMeasure(e);
    if (S.tool === 'label') {
        const pt = map.latLngToContainerPoint(e.latlng);
        showLabelInput(e.latlng, { x: pt.x + 100, y: pt.y + document.getElementById('map').offsetTop });
    }
});

map.on('dblclick', e => {
    if (S.tool === 'polyline') {
        if (e && e.originalEvent) L.DomEvent.stop(e.originalEvent);
        if (S.plPoints.length > 0) {
            S.plPoints.pop();
            if (S.plLine) { const ll = S.plLine.getLatLngs(); if (ll.length > 0) ll.pop(); S.plLine.setLatLngs(ll); }
        }
        finishPolyline();
    }
    if (S.tool === 'measure') {
        if (e && e.originalEvent) L.DomEvent.stop(e.originalEvent);
        finishMeasure();
    }
    /* Clear highlight when double-clicking empty map area (only in pan mode) */
    if (S.tool === 'pan' && S.highlighted) {
        S.highlighted.setStyle(S.highlightedOrigStyle);
        S.highlighted.bringToBack();
        S.highlighted = null;
        S.highlightedOrigStyle = null;
    }
});

/* touch support (Leaflet normalises most, but explicit is safer) */
map.on('touchstart', e => {
    if (S.tool === 'freehand' && e.latlng) startFreehand(e);
});
map.on('touchmove', e => {
    if (S.tool === 'freehand' && S.drawing && e.latlng) moveFreehand(e);
});
map.on('touchend', () => {
    if (S.tool === 'freehand' && S.drawing) finishFreehand();
});

map.on('zoomend', () => {
    $('#zoom-display').textContent = 'Zoom: ' + map.getZoom();
});

/* =====================================================
   UI — Tool Buttons
   ===================================================== */
$$('.tool-btn').forEach(btn => {
    btn.addEventListener('click', () => setTool(btn.dataset.tool));
});

/* =====================================================
   UI — Color Swatches (generated dynamically)
   ===================================================== */
COLORS.forEach((c, i) => {
    const el = document.createElement('button');
    el.className = 'color-swatch' + (i === 0 ? ' active' : '');
    el.dataset.color = c;
    el.style.background = c;
    if (c === '#333333') el.style.boxShadow = 'inset 0 0 0 1px #666';
    el.title = c;
    el.addEventListener('click', () => setColor(c));
    $('#color-grid').appendChild(el);
});

function setColor(c) {
    S.color = c;
    $$('.color-swatch').forEach(s => s.classList.toggle('active', s.dataset.color === c));
    $('#hex-input').value = c.replace('#', '');
}

$('#hex-input').addEventListener('input', e => {
    const val = e.target.value.replace(/[^0-9a-fA-F]/g, '');
    if (val.length === 6) {
        const c = '#' + val.toLowerCase();
        S.color = c;
        $$('.color-swatch').forEach(s => s.classList.toggle('active', s.dataset.color === c));
    }
});

/* =====================================================
   UI — Weight Slider
   ===================================================== */
$('#weight-slider').addEventListener('input', e => {
    S.weight = parseInt(e.target.value, 10);
    $('#weight-val').textContent = S.weight;
});

/* =====================================================
   UI — Style Buttons
   ===================================================== */
$$('.style-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        S.dashStyle = btn.dataset.style;
        $$('.style-btn').forEach(b => b.classList.toggle('active', b.dataset.style === S.dashStyle));
    });
});

/* =====================================================
   UI — Action Buttons
   ===================================================== */
$('#btn-clear').addEventListener('click', clearAll);
$('#btn-png').addEventListener('click', exportPNG);
$('#btn-geojson').addEventListener('click', exportGeoJSON);
$('#btn-finish').addEventListener('click', () => {
    if (S.tool === 'polyline') finishPolyline();
    if (S.tool === 'measure') finishMeasure();
});
$('#btn-undo').addEventListener('click', undo);
$('#btn-redo').addEventListener('click', redo);

/* =====================================================
   Keyboard Shortcuts
   ===================================================== */
document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    /* Ctrl+Z / Ctrl+Y undo/redo */
    if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return; }
    if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) { e.preventDefault(); redo(); return; }
    switch (e.key) {
        case 'Escape':
            if (!$('#proxy-modal').classList.contains('hidden')) { closeProxyModal(); break; }
            if (!$('#import-modal').classList.contains('hidden')) { closeImportModal(); break; }
            if (!$('#save-modal').classList.contains('hidden')) { $('#save-modal').classList.add('hidden'); break; }
            if (S.tool === 'polyline') cancelPolyline();
            if (S.tool === 'measure') cancelMeasure();
            if (S.tool === 'label') hideLabelInput();
            break;
        case 'Enter':
            if (!$('#import-modal').classList.contains('hidden')) { confirmImport(); break; }
            if (S.tool === 'polyline') finishPolyline();
            if (S.tool === 'measure') finishMeasure();
            break;
        case '1': setTool('pan');      break;
        case '2': setTool('freehand'); break;
        case '3': setTool('polyline'); break;
        case '4': setTool('arrow');    break;
        case '5': setTool('marker');   break;
        case '6': setTool('label');    break;
        case '7': setTool('measure');  break;
        case '8': setTool('eraser');   break;
        case '9': setTool('fire');     break;
        case 'i': case 'I': openImportModal(); break;
        case 'f': case 'F': toggleFullscreen(); break;
    }
});

/* =====================================================
   Fullscreen Mode
   ===================================================== */
const fsBtn = $('#btn-fullscreen');

function toggleFullscreen() {
    const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement);
    if (!isFs) {
        const el = document.documentElement;
        if (el.requestFullscreen) {
            el.requestFullscreen().catch(function (err) { console.warn('[FS] requestFullscreen failed:', err); });
        } else if (el.webkitRequestFullscreen) {
            el.webkitRequestFullscreen();
        } else if (el.msRequestFullscreen) {
            el.msRequestFullscreen();
        } else {
            console.warn('[FS] Fullscreen API not supported');
        }
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen().catch(function (err) { console.warn('[FS] exitFullscreen failed:', err); });
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
    }
}

function updateFullscreenBtn() {
    const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement);
    console.log('[FS] fullscreen changed, isFs =', isFs);
    if (fsBtn) {
        fsBtn.textContent = isFs ? '⛶ Exit Fullscreen' : '⛶ Fullscreen';
        fsBtn.title = isFs ? 'Exit fullscreen (F)' : 'Toggle fullscreen (F)';
    }
}

if (fsBtn) {
    fsBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        toggleFullscreen();
    });
}
document.addEventListener('fullscreenchange', function () { updateFullscreenBtn(); setTimeout(requestMapResize, 350); });
document.addEventListener('webkitfullscreenchange', function () { updateFullscreenBtn(); setTimeout(requestMapResize, 350); });

/* =====================================================
   Draggable Panels
   ===================================================== */
const DRAG_STORAGE_KEY = 'map-panel-layout';
const SETTINGS_STORAGE_KEY = 'map-layout-settings';
const PRESETS_STORAGE_KEY = 'map-layout-presets';
const SNAP_THRESHOLD = 12;

let cfgSnapEdge = true;

function makeDraggable(el, handleEl, id) {
    let isDragging = false, startX, startY, origLeft, origTop;
    el.classList.add('draggable');
    el.dataset.draggableId = id;

    function startDrag(e) {
        if (e.button && e.button !== 0) return;
        e.preventDefault(); e.stopPropagation();
        isDragging = true;
        const rect = el.getBoundingClientRect();
        const cx = e.touches ? e.touches[0].clientX : e.clientX;
        const cy = e.touches ? e.touches[0].clientY : e.clientY;
        startX = cx; startY = cy;
        origLeft = rect.left; origTop = rect.top;
        /* switch from CSS anchoring to absolute top/left */
        el.style.left = origLeft + 'px';
        el.style.top  = origTop + 'px';
        el.style.bottom = 'auto';
        el.style.right  = 'auto';
        el.style.transform = 'none';
        el.classList.add('dragging');
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', endDrag);
        document.addEventListener('touchmove', onMove, { passive: false });
        document.addEventListener('touchend', endDrag);
    }

    function onMove(e) {
        if (!isDragging) return;
        e.preventDefault();
        const cx = e.touches ? e.touches[0].clientX : e.clientX;
        const cy = e.touches ? e.touches[0].clientY : e.clientY;
        let nl = origLeft + (cx - startX);
        let nt = origTop  + (cy - startY);
        /* clamp to viewport */
        nl = Math.max(0, Math.min(window.innerWidth  - el.offsetWidth,  nl));
        nt = Math.max(0, Math.min(window.innerHeight - el.offsetHeight, nt));
        /* snap to edges */
        if (cfgSnapEdge) {
            if (nl < SNAP_THRESHOLD) nl = 0;
            if (nt < SNAP_THRESHOLD) nt = 0;
            if (nl + el.offsetWidth  > window.innerWidth  - SNAP_THRESHOLD) nl = window.innerWidth  - el.offsetWidth;
            if (nt + el.offsetHeight > window.innerHeight - SNAP_THRESHOLD) nt = window.innerHeight - el.offsetHeight;
        }
        el.style.left = nl + 'px';
        el.style.top  = nt + 'px';
    }

    function endDrag() {
        if (!isDragging) return;
        isDragging = false;
        el.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', endDrag);
        document.removeEventListener('touchmove', onMove);
        document.removeEventListener('touchend', endDrag);
        saveLayout();
        /* Prevent the document click handler from immediately closing panels */
        window._justDraggedPanel = true;
        setTimeout(function () { window._justDraggedPanel = false; }, 100);
    }

    handleEl.addEventListener('mousedown', startDrag);
    handleEl.addEventListener('touchstart', startDrag, { passive: false });
}

function saveLayout() {
    const pos = {};
    document.querySelectorAll('[data-draggable]').forEach(el => {
        const id = el.dataset.draggable;
        if (el.style.left || el.style.top) {
            pos[id] = { left: parseInt(el.style.left) || 0, top: parseInt(el.style.top) || 0 };
        }
    });
    localStorage.setItem(DRAG_STORAGE_KEY, JSON.stringify(pos));
}

function loadLayout() {
    const raw = localStorage.getItem(DRAG_STORAGE_KEY);
    if (!raw) return;
    try {
        const pos = JSON.parse(raw);
        document.querySelectorAll('[data-draggable]').forEach(el => {
            const p = pos[el.dataset.draggable];
            if (!p) return;
            el.style.left      = p.left + 'px';
            el.style.top       = p.top  + 'px';
            el.style.bottom    = 'auto';
            el.style.right     = 'auto';
            el.style.transform = 'none';
        });
    } catch (_) { /* ignore corrupt data */ }
}

function resetLayout() {
    document.querySelectorAll('[data-draggable]').forEach(el => {
        el.style.left = el.style.top = el.style.bottom = el.style.right = el.style.transform = '';
    });
    localStorage.removeItem(DRAG_STORAGE_KEY);
    /* restore checkboxes */
    $('#cfg-show-toolbar').checked  = true;
    $('#cfg-show-style').checked    = true;
    $('#cfg-show-bottombar').checked = true;
    $('#cfg-snap-edge').checked     = true;
    cfgSnapEdge = true;
    applyVisibility();
    saveSettings();
}

function clampDraggablePanels() {
    const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
    const viewportHeight = document.documentElement.clientHeight || window.innerHeight;
    let changed = false;

    document.querySelectorAll('[data-draggable]').forEach(el => {
        /* Only clamp panels that have been manually positioned. CSS-anchored
           panels should keep their responsive default positions. */
        if (!el.style.left && !el.style.top) return;
        const rect = el.getBoundingClientRect();
        const left = Math.max(0, Math.min(viewportWidth - el.offsetWidth, rect.left));
        const top = Math.max(0, Math.min(viewportHeight - el.offsetHeight, rect.top));
        if (Math.round(rect.left) !== Math.round(left) || Math.round(rect.top) !== Math.round(top)) {
            el.style.left = left + 'px';
            el.style.top = top + 'px';
            changed = true;
        }
    });

    if (changed) saveLayout();
}

/* ── Panel Visibility ──────────────────────────── */
function applyVisibility() {
    const tb = $('#toolbar'), sp = $('#style-panel'), bb = $('#bottom-bar');
    if (tb)  tb.style.display  = $('#cfg-show-toolbar').checked  ? '' : 'none';
    if (sp)  sp.style.display  = $('#cfg-show-style').checked    ? '' : 'none';
    if (bb)  bb.style.display  = $('#cfg-show-bottombar').checked ? '' : 'none';
}

function saveSettings() {
    const s = {
        showToolbar:   $('#cfg-show-toolbar').checked,
        showStyle:     $('#cfg-show-style').checked,
        showBottombar: $('#cfg-show-bottombar').checked,
        snapEdge:      $('#cfg-snap-edge').checked
    };
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(s));
}

function loadSettings() {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return;
    try {
        const s = JSON.parse(raw);
        if (s.showToolbar   !== undefined) $('#cfg-show-toolbar').checked   = s.showToolbar;
        if (s.showStyle     !== undefined) $('#cfg-show-style').checked     = s.showStyle;
        if (s.showBottombar !== undefined) $('#cfg-show-bottombar').checked = s.showBottombar;
        if (s.snapEdge      !== undefined) { $('#cfg-snap-edge').checked = s.snapEdge; cfgSnapEdge = s.snapEdge; }
        applyVisibility();
    } catch (_) {}
}

/* ── Layout Presets ────────────────────────────── */
function getPresets() {
    try { return JSON.parse(localStorage.getItem(PRESETS_STORAGE_KEY)) || []; }
    catch (_) { return []; }
}
function setPresets(arr) { localStorage.setItem(PRESETS_STORAGE_KEY, JSON.stringify(arr)); }

function captureCurrentLayout() {
    const pos = {};
    document.querySelectorAll('[data-draggable]').forEach(el => {
        const r = el.getBoundingClientRect();
        pos[el.dataset.draggable] = { left: Math.round(r.left), top: Math.round(r.top) };
    });
    return pos;
}

function applyPresetPosition(pos) {
    document.querySelectorAll('[data-draggable]').forEach(el => {
        const p = pos[el.dataset.draggable];
        if (!p) return;
        el.style.left      = p.left + 'px';
        el.style.top       = p.top  + 'px';
        el.style.bottom    = 'auto';
        el.style.right     = 'auto';
        el.style.transform = 'none';
    });
    saveLayout();
}

function renderPresets() {
    const list = $('#preset-list-settings');
    const presets = getPresets();
    if (!list) return;
    if (presets.length === 0) {
        list.innerHTML = '<div class="preset-empty">No saved presets</div>';
        return;
    }
    list.innerHTML = '';
    presets.forEach((pr, i) => {
        const div = document.createElement('div');
        div.className = 'preset-item';
        const name = document.createElement('span');
        name.className = 'preset-item-name';
        name.textContent = pr.name;
        name.title = 'Click to load';
        name.addEventListener('click', () => {
            applyPresetPosition(pr.layout);
            if (pr.visibility) {
                $('#cfg-show-toolbar').checked   = pr.visibility.showToolbar  !== false;
                $('#cfg-show-style').checked     = pr.visibility.showStyle    !== false;
                $('#cfg-show-bottombar').checked = pr.visibility.showBottombar !== false;
                applyVisibility();
            }
        });
        const del = document.createElement('button');
        del.className = 'preset-item-del';
        del.textContent = '✕';
        del.title = 'Delete preset';
        del.addEventListener('click', (e) => {
            e.stopPropagation();
            const all = getPresets(); all.splice(i, 1); setPresets(all); renderPresets();
        });
        div.appendChild(name);
        div.appendChild(del);
        list.appendChild(div);
    });
}

function savePreset() {
    const input = $('#preset-name-input');
    const name = input.value.trim();
    if (!name) { input.focus(); return; }
    const all = getPresets();
    all.push({
        name: name,
        layout: captureCurrentLayout(),
        visibility: {
            showToolbar:   $('#cfg-show-toolbar').checked,
            showStyle:     $('#cfg-show-style').checked,
            showBottombar: $('#cfg-show-bottombar').checked
        }
    });
    setPresets(all);
    input.value = '';
    renderPresets();
}

/* Wire up draggables */
(function initDraggables () {
    const toolbar    = document.getElementById('toolbar');
    const stylePanel = document.getElementById('style-panel');
    const bottomBar  = document.getElementById('bottom-bar');

    if (toolbar)    makeDraggable(toolbar,    toolbar.querySelector('.drag-handle'),    'toolbar');
    if (stylePanel) makeDraggable(stylePanel, stylePanel.querySelector('.drag-grip'),   'style-panel');
    if (bottomBar)  makeDraggable(bottomBar,  bottomBar.querySelector('.bar-grip'),     'bottom-bar');

    loadLayout();
    loadSettings();
    renderPresets();
})();

/* =====================================================
   Proxy Setup Modal — OS Detection & Management
   ===================================================== */
let proxyPendingUrl = null;   /* Google Maps URL to retry after proxy starts */
let proxyPollTimer = null;    /* interval ID for polling proxy status */

function detectProxyOS() {
    const ua = navigator.userAgent || '';
    if (/Win/i.test(ua)) return 'Windows';
    if (/Mac/i.test(ua)) return 'macOS';
    if (/Linux/i.test(ua)) return 'Linux';
    if (/Android/i.test(ua)) return 'Android';
    if (/iPhone|iPad|iPod/i.test(ua)) return 'iOS';
    return 'Unknown';
}

function openProxyModal(pendingUrl) {
    proxyPendingUrl = pendingUrl || null;
    const modal = $('#proxy-modal');
    modal.classList.remove('hidden');

    /* Detect OS and set instructions */
    const os = detectProxyOS();
    $('#proxy-os-name').textContent = os;
    document.body.classList.remove('proxy-show-win', 'proxy-show-unix');
    if (/Windows/i.test(os)) {
        document.body.classList.add('proxy-show-win');
    } else {
        document.body.classList.add('proxy-show-unix');
    }

    /* Reset status area */
    const statusEl = $('#proxy-status-text');
    statusEl.textContent = '⏳ Waiting for proxy…';
    statusEl.className = '';
    $('#proxy-status-detail').textContent = 'Run the setup script, then click "Check Proxy Status".';
    $('#proxy-check-btn').style.display = '';
    $('#proxy-retry-btn').style.display = 'none';
}

function closeProxyModal() {
    $('#proxy-modal').classList.add('hidden');
    if (proxyPollTimer) { clearInterval(proxyPollTimer); proxyPollTimer = null; }
    proxyPendingUrl = null;
}

function checkProxyStatus() {
    const statusEl = $('#proxy-status-text');
    const detailEl = $('#proxy-status-detail');
    statusEl.textContent = '🔄 Checking proxy…';
    statusEl.className = '';
    detailEl.textContent = 'Connecting to localhost:8080…';

    fetch('http://localhost:8080/kml?url=' + encodeURIComponent('https://www.google.com/maps'), { mode: 'no-cors' })
        .then(() => {
            /* With no-cors we can't read the response, but if it doesn't throw, proxy is up.
               Do a second opaque HEAD-like check by fetching a known-bad URL —
               if we get *any* response (even 400), the proxy is alive. */
            return fetch('http://localhost:8080/');
        })
        .then(r => {
            /* Proxy returned anything — it's alive */
            statusEl.textContent = '✅ Proxy is running!';
            statusEl.className = 'proxy-ok';
            detailEl.textContent = 'Connected to localhost:8080';
            $('#proxy-check-btn').style.display = 'none';
            $('#proxy-retry-btn').style.display = '';

            /* Auto-stop polling */
            if (proxyPollTimer) { clearInterval(proxyPollTimer); proxyPollTimer = null; }
        })
        .catch(() => {
            /* Also try direct fetch to the proxy root */
            fetch('http://localhost:8080/', { mode: 'no-cors' })
                .then(() => {
                    statusEl.textContent = '✅ Proxy is running!';
                    statusEl.className = 'proxy-ok';
                    detailEl.textContent = 'Connected to localhost:8080';
                    $('#proxy-check-btn').style.display = 'none';
                    $('#proxy-retry-btn').style.display = '';
                    if (proxyPollTimer) { clearInterval(proxyPollTimer); proxyPollTimer = null; }
                })
                .catch(() => {
                    statusEl.textContent = '❌ Proxy not running yet';
                    statusEl.className = 'proxy-fail';
                    detailEl.textContent = 'Make sure you ran the setup script and it says "Proxy is running". Then try again.';
                });
        });
}

function startProxyPolling() {
    if (proxyPollTimer) clearInterval(proxyPollTimer);
    proxyPollTimer = setInterval(checkProxyStatus, 3000);
    checkProxyStatus();
}

function retryProxyImport() {
    if (!proxyPendingUrl) {
        closeProxyModal();
        return;
    }
    const statusEl = $('#proxy-status-text');
    const detailEl = $('#proxy-status-detail');
    statusEl.textContent = '🔄 Retrying import…';
    statusEl.className = '';
    detailEl.textContent = 'Fetching KML through local proxy…';

    const mid = parseGoogleMapsUrl(proxyPendingUrl);
    if (mid) {
        const kmlUrl = 'https://www.google.com/maps/d/u/0/kml?mid=' + mid + '&forcekml=1';
        fetch('http://localhost:8080/kml?url=' + encodeURIComponent(kmlUrl))
            .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
            .then(kmlText => {
                importPendingFeatures = parseKML(kmlText);
                if (importPendingFeatures.length === 0) { alert('No features found in this Google Map.'); closeProxyModal(); return; }
                renderPreview(importPendingFeatures);
                closeProxyModal();
            })
            .catch(err => {
                statusEl.textContent = '❌ Retry failed';
                statusEl.className = 'proxy-fail';
                detailEl.textContent = 'Error: ' + (err.message || 'unknown') + '. Make sure the proxy is running.';
            });
    } else {
        /* Not a Google Maps URL — just try direct fetch through proxy */
        fetch('http://localhost:8080/kml?url=' + encodeURIComponent(proxyPendingUrl))
            .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
            .then(text => {
                try {
                    importPendingFeatures = parseKML(text);
                } catch (_) {
                    importPendingFeatures = parseGeoJSONForPreview(JSON.parse(text));
                }
                if (importPendingFeatures.length === 0) { alert('No features found.'); closeProxyModal(); return; }
                renderPreview(importPendingFeatures);
                closeProxyModal();
            })
            .catch(err => {
                statusEl.textContent = '❌ Retry failed';
                statusEl.className = 'proxy-fail';
                detailEl.textContent = 'Error: ' + (err.message || 'unknown');
            });
    }
}

/* Wire up proxy modal buttons */
(function wireProxyModal() {
    $('#proxy-modal-close').addEventListener('click', closeProxyModal);
    $('#proxy-modal').addEventListener('click', e => {
        if (e.target === $('#proxy-modal')) closeProxyModal();
    });
    $('#proxy-check-btn').addEventListener('click', checkProxyStatus);
    $('#proxy-retry-btn').addEventListener('click', retryProxyImport);
})();

/* =====================================================
   Init
   ===================================================== */
$('#zoom-display').textContent = 'Zoom: ' + map.getZoom();
updateCount();
updateUndoRedoUI();
wireImportModal();
wireSaveModal();
setTool('pan');

/* Style panel toggle */
$('#style-toggle').addEventListener('click', (e) => {
    if (e.target.closest('.drag-grip')) return; /* don't toggle when dragging */
    const body = $('#style-panel .panel-body');
    const arrow = $('#style-toggle .toggle-arrow');
    body.classList.toggle('hidden');
    arrow.classList.toggle('open');
});

/* ── Flag Selector Buttons ────────────────────── */
$$('.flag-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const flag = btn.dataset.flag || null;
        S.flag = flag;
        $$('.flag-btn').forEach(b => b.classList.toggle('active', (b.dataset.flag || null) === S.flag));
    });
});
/* Initialize: select "no flag" by default */
$$('.flag-btn').forEach(b => b.classList.toggle('active', b.dataset.flag === ''));

/* Reset layout button (inside settings popup) */

/* Settings popup toggle */
$('#btn-settings').addEventListener('click', (e) => {
    e.stopPropagation();
    const panel = $('#settings-panel');
    const isOpening = panel.classList.contains('hidden');
    panel.classList.toggle('hidden');
    document.body.classList.toggle('layout-editing', isOpening);
});
/* Close settings popup when clicking outside (but not right after a panel drag) */
document.addEventListener('click', (e) => {
    if (window._justDraggedPanel) return;
    const panel = $('#settings-panel');
    if (panel.classList.contains('hidden')) return;
    if (e.target.closest('#settings-panel') || e.target.closest('#btn-settings')) return;
    panel.classList.add('hidden');
    document.body.classList.remove('layout-editing');
});

/* Settings: panel visibility checkboxes */
['cfg-show-toolbar', 'cfg-show-style', 'cfg-show-bottombar'].forEach(id => {
    $('#' + id).addEventListener('change', () => { applyVisibility(); saveSettings(); });
});
$('#cfg-snap-edge').addEventListener('change', () => {
    cfgSnapEdge = $('#cfg-snap-edge').checked;
    saveSettings();
});

/* Settings: presets */
$('#btn-preset-save').addEventListener('click', savePreset);
$('#preset-name-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') savePreset(); });
$('#btn-reset-layout2').addEventListener('click', resetLayout);

console.log('Interactive Map loaded.');