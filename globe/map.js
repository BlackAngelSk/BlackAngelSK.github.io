/* =====================================================
   Interactive Detailed Map with Custom Line Annotations
   ===================================================== */
// @ts-nocheck
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

/* =====================================================
   Eraser
   ===================================================== */
function enableEraser() {
    S.annotations.forEach(ann => {
        const isPoint = ann.type === 'marker' || ann.type === 'label';
        const origWeight = ann.weight;
        const h = {
            over: function () {
                if (isPoint) {
                    const el = this.getElement();
                    if (el) { const inner = el.querySelector('.marker-pin, .label-marker'); if (inner) inner.classList.add('eraser-hover'); }
                } else {
                    this.setStyle({ color: '#ff4444', weight: origWeight + 3 });
                    this.bringToFront();
                }
            },
            out: function () {
                if (isPoint) {
                    const el = this.getElement();
                    if (el) { const inner = el.querySelector('.marker-pin, .label-marker'); if (inner) inner.classList.remove('eraser-hover'); }
                } else {
                    this.setStyle({ color: ann.color, weight: origWeight });
                }
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
        S.annotations = S.annotations.filter(x => x.id !== a.ann.id);
    } else if (a.type === 'remove') {
        a.ann.layer.addTo(map);
        S.annotations.push(a.ann);
    } else if (a.type === 'clear') {
        a.anns.forEach(x => { x.layer.addTo(map); S.annotations.push(x); });
    }
    S.redoStack.push(a);
    updateCount(); updateUndoRedoUI();
}

function redo() {
    if (S.redoStack.length === 0) return;
    const a = S.redoStack.pop();
    if (a.type === 'add') {
        a.ann.layer.addTo(map);
        S.annotations.push(a.ann);
    } else if (a.type === 'remove') {
        map.removeLayer(a.ann.layer);
        S.annotations = S.annotations.filter(x => x.id !== a.ann.id);
    } else if (a.type === 'clear') {
        a.anns.forEach(x => { map.removeLayer(x.layer); });
        S.annotations = [];
    }
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
    map.removeLayer(ann.layer);
    S.annotations.splice(i, 1);
    pushUndo({ type: 'remove', ann });
    updateCount();
}

function clearAll() {
    if (S.annotations.length === 0) return;
    if (!confirm('Clear all annotations? You can undo this.')) return;
    disableEraser();
    const snapshot = S.annotations.slice();
    S.annotations.forEach(a => map.removeLayer(a.layer));
    S.annotations = [];
    pushUndo({ type: 'clear', anns: snapshot });
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
let importParsedData = null;

/* ── Built-in Presets (Part 1) ─────────────────── */
const BUILT_IN_PRESETS = [
    {
        name: 'WWI Western Front',
        desc: 'Major trench lines, 1914-1918',
        geojson:{type:'FeatureCollection',features:[
            {type:'Feature',properties:{type:'polyline',color:'#ff4444',weight:3,dashStyle:'dashed',text:'Entente Front Line'},geometry:{type:'LineString',coordinates:[[2.59,51.09],[2.7,50.85],[2.95,50.45],[3.15,50.05],[3.5,49.6],[4.2,49.4],[5,49.2],[5.8,49],[6.35,48.95],[7,48.7],[7.2,48.55],[7.5,48.1],[7,47.7],[6.8,47.55],[6.2,47.5],[5.5,47.6],[4.9,47.5],[4.5,47.4],[3.9,47.4],[3.2,47.5],[2.9,47.8],[2.5,48.3],[2.2,48.8],[2.1,49.4],[2.3,49.9],[2.4,50.3],[2.5,50.7],[2.59,51.09]]}},
            {type:'Feature',properties:{type:'polyline',color:'#4488ff',weight:3,dashStyle:'dashed',text:'Central Powers Front'},geometry:{type:'LineString',coordinates:[[2.8,51.1],[2.9,50.9],[3.2,50.5],[3.5,50.1],[3.9,49.7],[4.6,49.45],[5.4,49.25],[6,49.1],[6.6,49],[7.3,48.8],[7.5,48.6],[7.8,48.2],[7.4,47.8],[7.2,47.6],[6.5,47.55],[5.9,47.65],[5.2,47.55],[4.7,47.5],[4.2,47.45],[3.5,47.55],[3.1,47.85],[2.7,48.35],[2.4,48.85],[2.3,49.45],[2.5,49.95],[2.6,50.35],[2.7,50.75],[2.8,51.1]]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Verdun'},geometry:{type:'Point',coordinates:[5.38,49.16]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Somme'},geometry:{type:'Point',coordinates:[2.7,50]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Ypres'},geometry:{type:'Point',coordinates:[2.89,50.85]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Marne'},geometry:{type:'Point',coordinates:[3.55,49.04]}}
        ]}
    },
    {
        name: 'WWII Eastern Front',
        desc: 'Key lines on Eastern Front, 1941-1945',
        geojson:{type:'FeatureCollection',features:[
            {type:'Feature',properties:{type:'polyline',color:'#ff2222',weight:3,dashStyle:'dashed',text:'Barbarossa Line (Jun 1941)'},geometry:{type:'LineString',coordinates:[[21,54.5],[23,54],[24,53.5],[23.5,52],[24,51],[24,50.5],[26,50],[28,49.5],[30,49],[32,48],[35,47.5],[38,47],[40,46.5]]}},
            {type:'Feature',properties:{type:'polyline',color:'#ff8800',weight:3,dashStyle:'solid',text:'Max German Advance (1941)'},geometry:{type:'LineString',coordinates:[[32,56],[36,55.5],[38,55],[37.5,54],[38,53],[40,52],[42,51],[44,50],[46,49],[48,48],[50,47],[48,46],[44,45.5],[42,44.5]]}},
            {type:'Feature',properties:{type:'polyline',color:'#44aaff',weight:3,dashStyle:'dashed',text:'Soviet Advance to Berlin'},geometry:{type:'LineString',coordinates:[[50,55],[45,54.5],[40,54],[35,53.5],[30,53],[28,52],[24,52],[22,52.5],[20,52.5],[18,52],[15,52.5],[14.5,52.3]]}},
            {type:'Feature',properties:{type:'marker',color:'#ff4444',text:'Stalingrad'},geometry:{type:'Point',coordinates:[43.5,48.7]}},
            {type:'Feature',properties:{type:'marker',color:'#44aaff',text:'Berlin'},geometry:{type:'Point',coordinates:[13.4,52.52]}},
            {type:'Feature',properties:{type:'marker',color:'#ff8800',text:'Leningrad'},geometry:{type:'Point',coordinates:[30.3,59.93]}}
        ]}
    },
    {
        name: 'Napoleonic Campaigns',
        desc: 'Key battles and routes, 1805-1815',
        geojson:{type:'FeatureCollection',features:[
            {type:'Feature',properties:{type:'polyline',color:'#ffaa00',weight:3,dashStyle:'dotted',text:'March to Moscow (1812)'},geometry:{type:'LineString',coordinates:[[13.4,52.52],[18,52],[21,52.2],[24,54.7],[28,55.7],[32,56.8],[36,57.5],[37.6,55.75]]}},
            {type:'Feature',properties:{type:'polyline',color:'#ff4444',weight:3,dashStyle:'solid',text:'Retreat from Moscow'},geometry:{type:'LineString',coordinates:[[37.6,55.75],[35,55],[30,54.5],[26,54],[22,53],[18,52],[14.5,52.3]]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Austerlitz (1805)'},geometry:{type:'Point',coordinates:[16.13,49.13]}},
            {type:'Feature',properties:{type:'marker',color:'#ffcc00',text:'Waterloo (1815)'},geometry:{type:'Point',coordinates:[4.4,50.71]}}
        ]}
    },
    {
        name: 'Cold War Iron Curtain',
        desc: 'The division of Europe, 1945-1991',
        geojson:{type:'FeatureCollection',features:[
            {type:'Feature',properties:{type:'polyline',color:'#ff3333',weight:4,dashStyle:'dotted',text:'Iron Curtain'},geometry:{type:'LineString',coordinates:[[-10,71],[-5,62],[8,55],[10,54.5],[14,54],[15,51],[16.5,49],[17,48],[18,47.5],[22,48],[25,44],[28,41],[29,41],[30,42],[40,43],[50,40]]}},
            {type:'Feature',properties:{type:'label',color:'#ff4444',text:'NATO West'},geometry:{type:'Point',coordinates:[10,50]}},
            {type:'Feature',properties:{type:'label',color:'#4488ff',text:'Warsaw Pact East'},geometry:{type:'Point',coordinates:[30,50]}},
            {type:'Feature',properties:{type:'marker',color:'#ff6644',text:'Berlin Wall'},geometry:{type:'Point',coordinates:[13.4,52.52]}}
        ]}
    },
    {
        name: 'Modern Ukraine Conflict',
        desc: 'Key front lines and areas, 2022-present',
        geojson:{type:'FeatureCollection',features:[
            {type:'Feature',properties:{type:'polyline',color:'#ff4444',weight:3,dashStyle:'dashed',text:'Front Line (approx 2023)'},geometry:{type:'LineString',coordinates:[[38,49.5],[37.5,48.5],[37,48],[36.5,47.5],[37,47],[37.5,46.5],[38,46.8],[38.5,47.2],[38.8,47.8],[39.5,48.3],[40,48.8],[40.5,49],[41,49.5],[41.5,49.8],[42,49.8],[42.5,49.5],[43,49],[44,48.5]]}},
            {type:'Feature',properties:{type:'marker',color:'#ff6644',text:'Kyiv'},geometry:{type:'Point',coordinates:[30.52,50.45]}},
            {type:'Feature',properties:{type:'marker',color:'#ff6644',text:'Bakhmut'},geometry:{type:'Point',coordinates:[38,48.6]}},
            {type:'Feature',properties:{type:'marker',color:'#ff6644',text:'Mariupol'},geometry:{type:'Point',coordinates:[37.54,47.1]}}
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
    importParsedData = null;
    $('#import-preview').classList.add('hidden');
    $('#import-actions').classList.add('hidden');
    $('#import-url-input').value = '';
    refreshPresetsTab();
}
function closeImportModal() {
    $('#import-modal').classList.add('hidden');
    importPendingFeatures = [];
    importParsedData = null;
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
            count++;
        } else if (f.geometryType === 'Point') {
            const ll = [f.geometry.coordinates[1], f.geometry.coordinates[0]];
            const latlng = L.latLng(ll);
            const text = f.text || f.name || '';
            const marker = L.circleMarker(latlng, {
                radius: 5, fillColor: f.color, color: '#fff',
                weight: 1.5, fillOpacity: 0.9, renderer: canvasRenderer
            });
            if (text) marker.bindPopup('<b style="color:' + f.color + '">' + escapeHtml(text) + '</b>', { maxWidth: 250 });
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
    const isKML = file.name.toLowerCase().endsWith('.kml') || file.name.toLowerCase().endsWith('.kmz');
    const reader = new FileReader();
    reader.onload = function (ev) {
        const text = ev.target.result;
        if (isKML) {
            console.log('KML file, length:', text.length);
            try {
                importPendingFeatures = parseKML(text);
                if (importPendingFeatures.length === 0) { alert('No compatible features found.'); return; }
                renderPreview(importPendingFeatures);
            } catch (err) {
                if (err && err.isNetworkLink) {
                    /* Google My Maps stub — fetch the real KML from the NetworkLink URL */
                    const netUrl = err.url;
                    const mapName = err.mapName || 'Google Maps';
                    console.log('Fetching real KML from NetworkLink:', netUrl);
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
                        alert('Could not fetch the full map data (CORS blocked).\n\n'
                            + 'To import this map:\n'
                            + '1. Run the proxy: node proxy.js\n'
                            + '2. Re-import this KML file\n\n'
                            + 'Or paste the Google Maps URL in the URL tab.');
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
    const map = {};
    const styles = doc.getElementsByTagName('Style');
    Array.from(styles).forEach(s => {
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
        map[id] = entry;
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
        map[id] = pairs.normal ? (map[pairs.normal] || { normal: pairs.normal }) : pairs;
    });
    return map;
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
    console.log('  extractKmlFeatures: folder=' + folderName + ', children=' + children.length + ', tags=[' + children.slice(0, 5).map(c => c.localName).join(',') + '...]');
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
            console.log('KML is a NetworkLink stub. URL:', linkHref);
            throw { isNetworkLink: true, url: linkHref.trim(), mapName: docName };
        }
    }

    const styleMap = buildKmlStyleMap(doc);
    const features = extractKmlFeatures(root, styleMap, docName);
    features.forEach(f => { f.mapName = docName; });
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

    const mid = parseGoogleMapsUrl(url);
    if (mid) {
        fetchGoogleKML(mid, btn).then(kmlText => {
            importPendingFeatures = parseKML(kmlText);
            if (importPendingFeatures.length === 0) { alert('No features found in this Google Map.'); return; }
            renderPreview(importPendingFeatures);
        }).catch(err => {
            console.error('KML import failed:', err);
            alert('Could not fetch from Google My Maps (CORS blocked).\n\n'
                + 'To import this map:\n'
                + '1. Open the map in Google Maps\n'
                + '2. Click the three dots (⋮) menu\n'
                + '3. Select "Download KML"\n'
                + '4. Use the File tab here to import the .kml file\n\n'
                + 'Or run the local proxy: node proxy.js');
        }).finally(() => { btn.disabled = false; btn.textContent = 'Fetch & Import'; });
        return;
    }

    fetch(url).then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
    }).then(data => {
        importPendingFeatures = parseGeoJSONForPreview(data);
        if (importPendingFeatures.length === 0) { alert('No compatible features found.'); return; }
        renderPreview(importPendingFeatures);
    }).catch(err => {
        console.error('URL import failed:', err);
        alert('Failed to fetch GeoJSON from URL.\n' + err.message);
    }).finally(() => { btn.disabled = false; btn.textContent = 'Fetch & Import'; });
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
});

map.on('mousemove', e => {
    S.lastLatLng = e.latlng;
    if (S.tool === 'freehand' && S.drawing) moveFreehand(e);
    if (S.tool === 'polyline') movePolyPreview(e);
    if (S.tool === 'measure') moveMeasurePreview(e);
});

map.on('mouseup', () => {
    mouseDown = false;
    if (S.tool === 'freehand' && S.drawing) finishFreehand();
});

map.on('click', e => {
    if (S.tool === 'polyline') addPolylinePoint(e);
    if (S.tool === 'marker') placeMarker(e);
    if (S.tool === 'measure') startMeasure(e);
    if (S.tool === 'label') {
        const pt = map.latLngToContainerPoint(e.latlng);
        showLabelInput(e.latlng, { x: pt.x + 100, y: pt.y + document.getElementById('map').offsetTop });
    }
});

map.on('dblclick', e => {
    if (S.tool === 'polyline') {
        L.DomEvent.stop(e);
        if (S.plPoints.length > 0) {
            S.plPoints.pop();
            if (S.plLine) { const ll = S.plLine.getLatLngs(); if (ll.length > 0) ll.pop(); S.plLine.setLatLngs(ll); }
        }
        finishPolyline();
    }
    if (S.tool === 'measure') {
        L.DomEvent.stop(e);
        finishMeasure();
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
        case '4': setTool('eraser');   break;
        case '5': setTool('marker');   break;
        case '6': setTool('label');    break;
        case '7': setTool('measure');  break;
        case '8': setTool('eraser');   break;
        case 'i': case 'I': openImportModal(); break;
    }
});

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
$('#style-toggle').addEventListener('click', () => {
    const body = $('#style-panel .panel-body');
    const arrow = $('#style-toggle .toggle-arrow');
    body.classList.toggle('hidden');
    arrow.classList.toggle('open');
});

console.log('Interactive Map loaded.');