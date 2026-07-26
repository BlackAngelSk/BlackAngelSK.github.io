/* =====================================================
   Interactive Detailed Map with Custom Line Annotations
   ===================================================== */
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
    /* touch fallback */
    lastLatLng: null,
    /* undo/redo */
    undoStack: [],
    redoStack: [],
    /* measure */
    msPoints: [],
    msLine: null,
    msLabels: [],
    msPreview: null,
    /* label input */
    labelLatLng: null
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
    marker: 'Click to drop a pin on the map',
    label: 'Click to place a text label',
    measure: 'Click to measure distance · Dbl-click / Enter to finish · Esc to cancel',
    eraser: 'Click an annotation to erase it'
};

/* ── DOM shortcuts ────────────────────────────────── */
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

/* ── Map init ─────────────────────────────────────── */
const map = L.map('map', {
    center: [48.15, 17.11],
    zoom: 6,
    minZoom: 2,
    maxZoom: 19,
    zoomControl: true,
    doubleClickZoom: false
});

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
osm.addTo(map);
L.control.layers({
    '\u{1F5FA} Standard': osm,
    '\u{1F3D4} Topographic': topo,
    '\u{1F6F0} Satellite': sat
}, null, { position: 'topright' }).addTo(map);

/* =====================================================
   Tool Management
   ===================================================== */
function setTool(tool) {
    /* finish any in-progress drawing */
    if (S.tool === 'polyline' && S.plPoints.length > 0) finishPolyline();
    if (S.tool === 'freehand' && S.drawing) finishFreehand();
    if (S.tool === 'measure' && S.msPoints.length > 0) finishMeasure();

    /* hide label input if switching away */
    if (S.tool === 'label') hideLabelInput();

    /* leave previous tool */
    if (S.tool === 'eraser') disableEraser();

    S.tool = tool;
    S.drawing = false;

    /* map dragging */
    if (tool === 'pan' || tool === 'eraser') map.dragging.enable();
    else map.dragging.disable();

    /* enter new tool */
    if (tool === 'eraser') enableEraser();

    /* UI */
    $$('.tool-btn').forEach(b => b.classList.toggle('active', b.dataset.tool === tool));
    $('#status-text').textContent = STATUS[tool];
    const icons = {
        pan: '\u{1F4CD}', freehand: '\u270F\uFE0F', polyline: '\u{1F4D0}',
        marker: '\u{1F4CC}', label: '\u{1F3F7}\uFE0F', measure: '\u{1F4CF}', eraser: '\u{1F9F9}'
    };
    $('#status-icon').textContent = icons[tool] || '\u{1F4CD}';
    document.body.className = 'tool-' + tool;
    if (tool !== 'polyline' && tool !== 'measure') $('#btn-finish').classList.add('hidden');
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
        html: '<div class="label-marker" style="border-color:' + S.color + ';color:' + S.color + '">' + escapeHtml(text) + '</div>',
        iconSize: null, iconAnchor: [0, 0]
    });
    const m = L.marker(S.labelLatLng, { icon, interactive: true }).addTo(map);
    storeAnnotation(m, 'label', [S.labelLatLng.lat, S.labelLatLng.lng], { text });
    hideLabelInput();
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
            color: '#ffe066', weight: 2, dashArray: '8, 6', opacity: 0.9, interactive: false
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
   Eraser
   ===================================================== */
function enableEraser() {
    S.annotations.forEach(ann => {
        const h = {
            over: function () {
                this.setStyle({ color: '#ff4444', weight: ann.weight + 3 });
                this.bringToFront();
            },
            out: function () {
                this.setStyle({ color: ann.color, weight: ann.weight });
            },
            click: function (ev) {
                L.DomEvent.stop(ev);
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
        ann.layer.off('mouseover', h.over);
        ann.layer.off('mouseout', h.out);
        ann.layer.off('click', h.click);
    });
    S.eraserHandlers = [];
}

/* =====================================================
   Annotation Management
   ===================================================== */
function storeAnnotation(layer, type, coords) {
    const ann = {
        id: S.nextId++,
        layer, type, coords,
        color: S.color,
        weight: S.weight,
        dashStyle: S.dashStyle,
        dashArray: DASH[S.dashStyle]
    };
    layer._annId = ann.id;
    S.annotations.push(ann);
    updateCount();
}

function removeAnnotation(id) {
    const i = S.annotations.findIndex(a => a.id === id);
    if (i === -1) return;
    map.removeLayer(S.annotations[i].layer);
    S.annotations.splice(i, 1);
    updateCount();
}

function clearAll() {
    if (S.annotations.length === 0) return;
    if (!confirm('Clear all annotations? This cannot be undone.')) return;
    disableEraser();
    S.annotations.forEach(a => map.removeLayer(a.layer));
    S.annotations = [];
    updateCount();
}

function updateCount() {
    $('#ann-count').textContent = S.annotations.length + ' annotation' + (S.annotations.length !== 1 ? 's' : '');
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
    const features = S.annotations.map(a => ({
        type: 'Feature',
        properties: {
            id: a.id, type: a.type, color: a.color,
            weight: a.weight, dashStyle: a.dashStyle
        },
        geometry: {
            type: 'LineString',
            coordinates: a.coords.map(c => [c[1], c[0]]) /* [lng, lat] */
        }
    }));
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
   Map Events
   ===================================================== */
let mouseDown = false;

map.on('mousedown', e => {
    mouseDown = true;
    if (S.tool === 'freehand') startFreehand(e);
});

map.on('mousemove', e => {
    S.lastLatLng = e.latlng;
    if (S.tool === 'freehand' && S.drawing) moveFreehand(e);
    if (S.tool === 'polyline') movePolyPreview(e);
});

map.on('mouseup', () => {
    mouseDown = false;
    if (S.tool === 'freehand' && S.drawing) finishFreehand();
});

map.on('click', e => {
    if (S.tool === 'polyline') addPolylinePoint(e);
});

map.on('dblclick', e => {
    if (S.tool === 'polyline') {
        L.DomEvent.stop(e);
        /* remove the last point added by the second click */
        if (S.plPoints.length > 0) {
            S.plPoints.pop();
            if (S.plLine) {
                const ll = S.plLine.getLatLngs();
                if (ll.length > 0) ll.pop();
                S.plLine.setLatLngs(ll);
            }
        }
        finishPolyline();
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
$('#btn-finish').addEventListener('click', finishPolyline);

/* =====================================================
   Keyboard Shortcuts
   ===================================================== */
document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    switch (e.key) {
        case 'Escape': if (S.tool === 'polyline') cancelPolyline(); break;
        case 'Enter':  if (S.tool === 'polyline') finishPolyline(); break;
        case '1': setTool('pan');      break;
        case '2': setTool('freehand'); break;
        case '3': setTool('polyline'); break;
        case '4': setTool('eraser');   break;
    }
});

/* =====================================================
   Init
   ===================================================== */
$('#zoom-display').textContent = 'Zoom: ' + map.getZoom();
updateCount();
setTool('pan');
console.log('Interactive Map loaded.');