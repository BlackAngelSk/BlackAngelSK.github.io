/**
 * Aleph-0 Interactive 3D Visualization (v2.0)
 *
 * A Three.js application for exploring countable infinity (ℵ₀) through
 * interactive 3D visualizations: Number Spiral, Hilbert's Hotel,
 * One-to-One Correspondence, Menger Sponge, Cantor's Diagonal Argument,
 * and Rational Number Enumeration.
 *
 * @version 2.0.0
 * @license MIT
 */
(function () {
  'use strict';

  try {
    /* ==========================================
     * Configuration Constants
     * ========================================== */
    var MAX_SPIRAL = 100;
    var MAX_HOTEL = 80;
    var MAX_CORR = 40;
    var MENGER_DEPTH = 2;
    var PHASE_DUR = 4;
    var LERP_SPEED = 0.03;

    function lerp(a, b, t) {
      return a + (b - a) * Math.min(1, Math.max(0, t));
    }

    /* ==========================================
     * State
     * ========================================== */
    var animSpeed = 1;
    var playing = true;
    var curMode = 'spiral';
    var elTime = 0;
    var elCount = 0;

    /* Camera transition */
    var camTransition = true;

    /* Clock */
    var clock = new THREE.Clock();

    /* ==========================================
     * Scene
     * ========================================== */
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020208);
    scene.fog = new THREE.FogExp2(0x020208, 0.004);

    var camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
    camera.position.set(0, 15, 30);

    var renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch (e) {
      renderer = new THREE.WebGLRenderer({ antialias: false });
    }
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    var canvasContainer = document.getElementById('canvas-container');
    if (canvasContainer) {
      canvasContainer.appendChild(renderer.domElement);
    }

    /* ==========================================
     * Orbit Controls
     * ========================================== */
    var orbitData = { drag: false, mx: 0, my: 0, rx: 0, ry: 0, zoom: 30, tx: 0, ty: 0, tz: 0, zMin: 3, zMax: 200 };

    (function () {
      var r = renderer.domElement;
      r.addEventListener('mousedown', function (e) { orbitData.drag = true; orbitData.mx = e.clientX; orbitData.my = e.clientY; });
      r.addEventListener('mousemove', function (e) {
        if (!orbitData.drag) return;
        var dx = e.clientX - orbitData.mx, dy = e.clientY - orbitData.my;
        orbitData.ry -= dx * 0.005; orbitData.rx -= dy * 0.005;
        orbitData.rx = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, orbitData.rx));
        orbitData.mx = e.clientX; orbitData.my = e.clientY;
      });
      r.addEventListener('mouseup', function () { orbitData.drag = false; });
      r.addEventListener('mouseleave', function () { orbitData.drag = false; });
      r.addEventListener('wheel', function (e) { orbitData.zoom += e.deltaY * 0.02; orbitData.zoom = Math.max(orbitData.zMin, Math.min(orbitData.zMax, orbitData.zoom)); e.preventDefault(); }, { passive: false });
      r.addEventListener('contextmenu', function (e) { e.preventDefault(); });
      var td = 0;
      r.addEventListener('touchstart', function (e) {
        if (e.touches.length === 1) { orbitData.drag = true; orbitData.mx = e.touches[0].clientX; orbitData.my = e.touches[0].clientY; }
        if (e.touches.length === 2) { var dx = e.touches[0].clientX - e.touches[1].clientX, dy = e.touches[0].clientY - e.touches[1].clientY; td = Math.sqrt(dx * dx + dy * dy); }
      });
      r.addEventListener('touchmove', function (e) {
        e.preventDefault();
        if (e.touches.length === 1 && orbitData.drag) {
          var dx = e.touches[0].clientX - orbitData.mx, dy = e.touches[0].clientY - orbitData.my;
          orbitData.ry -= dx * 0.005; orbitData.rx -= dy * 0.005;
          orbitData.rx = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, orbitData.rx));
          orbitData.mx = e.touches[0].clientX; orbitData.my = e.touches[0].clientY;
        }
        if (e.touches.length === 2) {
          var dx2 = e.touches[0].clientX - e.touches[1].clientX, dy2 = e.touches[0].clientY - e.touches[1].clientY;
          var d = Math.sqrt(dx2 * dx2 + dy2 * dy2); orbitData.zoom -= (d - td) * 0.05;
          orbitData.zoom = Math.max(orbitData.zMin, Math.min(orbitData.zMax, orbitData.zoom)); td = d;
        }
      }, { passive: false });
      r.addEventListener('touchend', function () { orbitData.drag = false; });
    })();

    function updateCamera() {
      var r = orbitData.zoom;
      var sx = Math.sin(orbitData.ry) * Math.cos(orbitData.rx) * r;
      var sy = Math.sin(orbitData.rx) * r;
      var sz = Math.cos(orbitData.ry) * Math.cos(orbitData.rx) * r;
      if (camTransition) {
        camera.position.x = lerp(camera.position.x, sx + orbitData.tx, LERP_SPEED);
        camera.position.y = lerp(camera.position.y, sy + orbitData.ty, LERP_SPEED);
        camera.position.z = lerp(camera.position.z, sz + orbitData.tz, LERP_SPEED);
      } else {
        camera.position.x = sx + orbitData.tx;
        camera.position.y = sy + orbitData.ty;
        camera.position.z = sz + orbitData.tz;
      }
      camera.lookAt(orbitData.tx, orbitData.ty, orbitData.tz);
    }

    /* ==========================================
     * Lighting
     * ========================================== */
    scene.add(new THREE.AmbientLight(0x334466, 0.6));
    var light1 = new THREE.PointLight(0x88bbff, 1.5, 150); light1.position.set(20, 30, 20); scene.add(light1);
    var light2 = new THREE.PointLight(0xbb88ff, 0.8, 100); light2.position.set(-15, 10, -20); scene.add(light2);

    /* ==========================================
     * Starfield
     * ========================================== */
    var starGeo = new THREE.BufferGeometry();
    var starPos = new Float32Array(1000 * 3);
    for (var i = 0; i < 1000; i++) {
      starPos[i * 3] = (Math.random() - 0.5) * 500;
      starPos[i * 3 + 1] = (Math.random() - 0.5) * 500;
      starPos[i * 3 + 2] = (Math.random() - 0.5) * 500;
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    var stars = new THREE.Points(starGeo, new THREE.PointsMaterial({
      color: 0x88bbff, size: 0.25, transparent: true, opacity: 0.5,
      sizeAttenuation: true, blending: THREE.AdditiveBlending
    }));
    scene.add(stars);

    /* ==========================================
     * Mode Groups
     * ========================================== */
    var spiralGrp = new THREE.Group(); scene.add(spiralGrp); spiralGrp.visible = true;
    var hotelGrp = new THREE.Group(); scene.add(hotelGrp); hotelGrp.visible = false;
    var corrGrp = new THREE.Group(); scene.add(corrGrp); corrGrp.visible = false;
    var mengerGrp = new THREE.Group(); scene.add(mengerGrp); mengerGrp.visible = false;
    var cantorGrp = new THREE.Group(); scene.add(cantorGrp); cantorGrp.visible = false;
    var rationGrp = new THREE.Group(); scene.add(rationGrp); rationGrp.visible = false;

    var hotelPhase = 0, hotelTimer = 0, PHASE_D = PHASE_DUR;
    var spiralEls = [], hotelEls = [], corrEls = [], corrLines = [], mengerEls = [], cantorEls = [], rationEls = [];
    var spiralBuilt = false, hotelBuilt = false, corrBuilt = false, mengerBuilt = false, cantorBuilt = false, rationBuilt = false;
    var mengerTotal = 0;

    /* ==========================================
     * Utilities
     * ========================================== */

    /** Check if a number is prime. */
    function isPrime(n) {
      if (n < 2) return false;
      if (n === 2) return true;
      if (n % 2 === 0) return false;
      var i = 3;
      while (i * i <= n) { if (n % i === 0) return false; i += 2; }
      return true;
    }
    var primes = [];
    for (var ci = 0, ni = 2; ci < MAX_CORR; ) { if (isPrime(ni)) { primes.push(ni); ci++; } ni++; }

    /** Create a text label sprite using a canvas element. */
    function mkLabel(text, color, w, h, fs) {
      w = w || 64; h = h || 32; fs = fs || 20;
      var cv = document.createElement('canvas');
      var cx = cv.getContext('2d');
      cv.width = w; cv.height = h;
      cx.font = 'bold ' + fs + 'px sans-serif';
      cx.textAlign = 'center'; cx.textBaseline = 'middle';
      cx.fillStyle = color || '#fff';
      cx.fillText(text, w / 2, h / 2);
      var tx = new THREE.CanvasTexture(cv);
      tx.minFilter = THREE.LinearFilter;
      var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tx, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
      sp.scale.set(1.5, 0.75, 1);
      return sp;
    }

    /** Dispose Three.js object (geometry + materials) to free GPU memory. */
    function disposeObj(obj) {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(function (m) { m.dispose(); });
        else obj.material.dispose();
      }
      if (obj.children) obj.children.forEach(function (c) { disposeObj(c); });
    }

    /** Clear all children from a group and dispose resources. */
    function clearGrp(grp) {
      while (grp.children.length > 0) {
        var ch = grp.children[0];
        disposeObj(ch);
        grp.remove(ch);
      }
    }

    /* ==========================================
     * MODE 1: Number Spiral
     * ========================================== */
    function buildSpiral() {
      if (spiralBuilt) return; spiralBuilt = true;
      var T = Math.PI * 2, tu = 5, up = 8, H = tu * up;
      var lp = [];
      for (var i = 0; i < 500; i++) {
        var t = i / 500, a = t * tu * T, r = 3 + t * 10, y = t * H - H / 2;
        lp.push(Math.cos(a) * r, y, Math.sin(a) * r);
      }
      var lg = new THREE.BufferGeometry();
      lg.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3));
      spiralGrp.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0x223366, transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending })));
      for (var n = 1; n <= MAX_SPIRAL; n++) {
        var t2 = (n - 1) / MAX_SPIRAL, a2 = t2 * tu * T, r2 = 3 + t2 * 10, y2 = t2 * H - H / 2;
        var hu = (n / MAX_SPIRAL) * 0.6 + 0.55;
        var col = new THREE.Color().setHSL(hu, 0.7, 0.55);
        var sz = 0.2 + t2 * 0.4;
        var sp = new THREE.Mesh(new THREE.SphereGeometry(sz, 6, 6), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending }));
        sp.position.set(Math.cos(a2) * r2, y2, Math.sin(a2) * r2); sp.visible = false; spiralGrp.add(sp);
        var lb = mkLabel(String(n), '#' + col.getHexString());
        lb.position.copy(sp.position); lb.position.y += sz + 0.5; lb.visible = false; spiralGrp.add(lb);
        spiralEls.push({ sp: sp, lb: lb, n: n, t0: n * 0.05, sz: sz, hu: hu });
      }
    }

    function tickSpiral(time) {
      var cnt = 0;
      for (var i = 0; i < spiralEls.length; i++) {
        var e = spiralEls[i], show = time >= e.t0;
        if (e.sp.visible !== show) { e.sp.visible = show; e.lb.visible = show; }
        if (show) { cnt++; e.sp.position.y += Math.sin(time * 2 + e.n * 0.5) * 0.003; }
      }
      elCount = Math.min(cnt, MAX_SPIRAL);
    }

    /* ==========================================
     * MODE 2: Hilbert's Hotel
     * ========================================== */
    function buildHotel() {
      if (hotelBuilt) return; hotelBuilt = true;
      var floors = 3, rpf = Math.ceil(MAX_HOTEL / floors), rw = 2.2;
      var fl = new THREE.Mesh(new THREE.PlaneGeometry(rpf * rw + 10, floors * 3.5 + 10), new THREE.MeshBasicMaterial({ color: 0x080818 }));
      fl.rotation.x = -Math.PI / 2; fl.position.y = -0.5; hotelGrp.add(fl);
      var bw = new THREE.Mesh(new THREE.PlaneGeometry(rpf * rw + 10, floors * 3.5 + 2), new THREE.MeshBasicMaterial({ color: 0x0a0a20, transparent: true, opacity: 0.5 }));
      bw.position.set(0, floors * 1.5, -1.5); hotelGrp.add(bw);
      var strip = new THREE.Mesh(new THREE.PlaneGeometry(rpf * rw + 4, 0.3), new THREE.MeshBasicMaterial({ color: 0x223366, transparent: true, opacity: 0.3 }));
      strip.rotation.x = -Math.PI / 2; strip.position.set(0, -0.45, 0.6); hotelGrp.add(strip);

      var idx = 0;
      for (var fi = 0; fi < floors; fi++) {
        for (var i = 0; i < rpf && idx < MAX_HOTEL; i++) {
          var x = (i - rpf / 2) * rw, y = fi * 2.8 + 0.75, z = fi * 0.5;
          var box = new THREE.Mesh(new THREE.BoxGeometry(1.8, 2.4, 1.6), new THREE.MeshBasicMaterial({ color: 0x1a2a4a, transparent: true, opacity: 0.7 }));
          box.position.set(x, y, z); hotelGrp.add(box);
          var win = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.6), new THREE.MeshBasicMaterial({ color: 0x88ccff, transparent: true, opacity: 0.15, blending: THREE.AdditiveBlending }));
          win.position.set(x, y + 0.4, z + 0.81); hotelGrp.add(win);
          var g = new THREE.Mesh(new THREE.SphereGeometry(0.35, 8, 8), new THREE.MeshBasicMaterial({ color: 0x88ccff, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending }));
          g.position.set(x, y, z); g.visible = false; hotelGrp.add(g);
          var nl = mkLabel(String(idx + 1), '#8ab4f0'); nl.position.set(x, y + 1.8, z); nl.scale.set(1.1, 0.55, 1); hotelGrp.add(nl);
          var ol = mkLabel(String(idx + 1), '#64b5f6'); ol.position.set(x, y + 0.5, z); ol.scale.set(1.0, 0.5, 1); ol.visible = false; hotelGrp.add(ol);
          var edgeGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1.85, 2.45, 1.65));
          var edgeLine = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0x3355aa, transparent: true, opacity: 0.25 }));
          edgeLine.position.copy(box.position); hotelGrp.add(edgeLine);
          hotelEls.push({ box: box, win: win, guest: g, nl: nl, ol: ol, ox: x, oy: y, oz: z, floor: fi, rIdx: i, edge: edgeLine });
          idx++;
        }
      }
      for (var fi2 = 1; fi2 < floors; fi2++) {
        var sy = fi2 * 2.8 + 0.1, sw = rpf * rw / 2 + 2;
        var sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.Float32BufferAttribute([-sw, sy, -0.1, sw, sy, -0.1], 3));
        hotelGrp.add(new THREE.Line(sg, new THREE.LineBasicMaterial({ color: 0x3355aa, transparent: true, opacity: 0.3 })));
      }
      var ng = new THREE.Mesh(new THREE.SphereGeometry(0.45, 10, 10), new THREE.MeshBasicMaterial({ color: 0xff7043, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending }));
      ng.visible = false; hotelGrp.add(ng); hotelGrp.userData.ng = ng;
      var sv = mkLabel('NO VACANCY', '#ff5252'); sv.position.set(0, floors * 2.8 + 2.5, 0); sv.scale.set(6, 3, 1); hotelGrp.add(sv);
      var inf = mkLabel('\u221E', '#64b5f6'); inf.position.set(0, floors * 2.8 + 4, 0); inf.scale.set(3, 3, 1); hotelGrp.add(inf); hotelGrp.userData.inf = inf;
      for (var fi3 = 0; fi3 < floors; fi3++) {
        var gy = fi3 * 2.8 - 0.2, gg = new THREE.BufferGeometry(), gp = [];
        for (var gi = 0; gi < 100; gi++) { gp.push((gi / 100 - 0.5) * rpf * rw, gy, 0); }
        gg.setAttribute('position', new THREE.Float32BufferAttribute(gp, 3));
        hotelGrp.add(new THREE.Line(gg, new THREE.LineBasicMaterial({ color: 0x64b5f6, transparent: true, opacity: 0.06 })));
      }
    }

    function tickHotel(dt, time) {
      hotelTimer += dt; if (hotelTimer > PHASE_D) { hotelTimer -= PHASE_D; hotelPhase = (hotelPhase + 1) % 3; }
      var pr = hotelTimer / PHASE_D, floors = 3;
      var inf = hotelGrp.userData.inf;
      if (inf) inf.position.y = floors * 2.8 + 4 + Math.sin(time * 0.8) * 0.5;
      var ng = hotelGrp.userData.ng;
      for (var i = 0; i < hotelEls.length; i++) {
        var e = hotelEls[i];
        if (hotelPhase === 0) {
          e.guest.visible = true; e.guest.position.set(e.ox, e.oy, e.oz);
          e.win.material.opacity = 0.08 + 0.12 * Math.sin(time * 2 + i * 0.5);
          e.ol.visible = true; e.ol.position.set(e.ox, e.oy + 0.75, e.oz);
        } else if (hotelPhase === 1) {
          e.guest.visible = true;
          var tgt = e.ox + 2.2;
          e.guest.position.x = e.ox + (tgt - e.ox) * Math.min(1, pr * 2);
          e.guest.position.y = e.oy + Math.sin(pr * Math.PI) * 0.8;
          e.guest.position.z = e.oz;
          e.win.material.opacity = 0.05;
          e.ol.visible = true; e.ol.position.set(e.guest.position.x, e.guest.position.y + 0.5, e.oz);
        } else {
          e.guest.visible = true; e.guest.position.set(e.ox, e.oy, e.oz);
          e.win.material.opacity = 0.15 + 0.1 * Math.sin(time * 3 + i * 0.3);
          e.ol.visible = true; e.ol.position.set(e.ox, e.oy + 0.75, e.oz);
        }
        e.box.material.opacity = 0.5 + 0.2 * Math.sin(time * 2 + i * 0.3 + e.floor * 0.5);
        e.edge.material.opacity = 0.2 + 0.15 * Math.sin(time * 1.5 + i * 0.2);
        if (i % 2 === 0) e.win.material.color.setHSL(0.55 + 0.05 * Math.sin(time + i), 0.8, 0.6);
      }
      if (hotelPhase === 0) ng.visible = false;
      else {
        ng.visible = true;
        if (hotelPhase === 1) { ng.position.set(hotelEls[0].ox - 10 + 10 * Math.min(1, pr), hotelEls[0].oy + 2 + Math.sin(pr * Math.PI) * 2, hotelEls[0].oz); }
        else { ng.position.set(hotelEls[0].ox, hotelEls[0].oy + Math.sin(time * 4) * 0.15, hotelEls[0].oz); ng.scale.setScalar(1 + Math.sin(time * 4) * 0.15); }
      }
      elCount = hotelPhase === 0 ? MAX_HOTEL : hotelPhase === 1 ? Math.floor(pr * (MAX_HOTEL + 1)) + 1 : MAX_HOTEL + 1;
    }

    /* ==========================================
     * MODE 3: One-to-One Correspondence
     * ========================================== */
    function buildCorr() {
      if (corrBuilt) return; corrBuilt = true;
      var cols = [
        { l: 'N (Naturals)', c: 0x64b5f6, fn: function (n) { return n; } },
        { l: '2N (Evens)', c: 0x81c784, fn: function (n) { return 2 * n; } },
        { l: 'P (Primes)', c: 0xffb74d, fn: function (n) { return primes[n - 1]; } }
      ];
      var spc = 8, rSp = 1.5;
      for (var ci2 = 0; ci2 < cols.length; ci2++) {
        var col = cols[ci2], xP = (ci2 - 1) * spc;
        var hc = '#' + new THREE.Color(col.c).getHexString();
        var hdr = mkLabel(col.l, hc); hdr.position.set(xP, MAX_CORR * rSp / 2 + 2, 0); hdr.scale.set(3, 1.5, 1); corrGrp.add(hdr);
        var vlp = new Float32Array([xP, -MAX_CORR * rSp / 2 - 1, 0, xP, MAX_CORR * rSp / 2 + 1, 0]);
        var vlg = new THREE.BufferGeometry(); vlg.setAttribute('position', new THREE.Float32BufferAttribute(vlp, 3));
        corrGrp.add(new THREE.Line(vlg, new THREE.LineBasicMaterial({ color: col.c, transparent: true, opacity: 0.1 })));
        for (var n = 1; n <= MAX_CORR; n++) {
          var yP = -n * rSp + MAX_CORR * rSp / 2;
          var dot = new THREE.Mesh(new THREE.SphereGeometry(0.25, 6, 6), new THREE.MeshBasicMaterial({ color: col.c, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending }));
          dot.position.set(xP, yP, 0); dot.visible = false; corrGrp.add(dot);
          var lb2 = mkLabel(String(col.fn(n)), hc); lb2.position.set(xP + 0.8, yP, 0); lb2.visible = false; corrGrp.add(lb2);
          corrEls.push({ ci: ci2, n: n, dot: dot, lb: lb2, t0: n * 0.04 });
        }
      }
      for (var n2 = 1; n2 <= MAX_CORR; n2++) {
        var yl = -n2 * rSp + MAX_CORR * rSp / 2;
        var l1p = new Float32Array([-spc, yl, 0, 0, yl, 0]);
        var l1g = new THREE.BufferGeometry(); l1g.setAttribute('position', new THREE.Float32BufferAttribute(l1p, 3));
        var l1m = new THREE.LineBasicMaterial({ color: 0x81c784, transparent: true, opacity: 0, blending: THREE.AdditiveBlending });
        var l1l = new THREE.Line(l1g, l1m); corrGrp.add(l1l);
        var l2p = new Float32Array([-spc, yl, 0, spc, yl, 0]);
        var l2g = new THREE.BufferGeometry(); l2g.setAttribute('position', new THREE.Float32BufferAttribute(l2p, 3));
        var l2m = new THREE.LineBasicMaterial({ color: 0xffb74d, transparent: true, opacity: 0, blending: THREE.AdditiveBlending });
        var l2l = new THREE.Line(l2g, l2m); corrGrp.add(l2l);
        corrLines.push({ l1: l1l, l2: l2l, t0: n2 * 0.04 });
      }
    }

    function tickCorr(time) {
      var cnt = 0;
      for (var i = 0; i < corrEls.length; i++) {
        var d = corrEls[i], show = time >= d.t0;
        if (d.dot.visible !== show) { d.dot.visible = show; d.lb.visible = show; }
        if (show) cnt++;
      }
      for (var i2 = 0; i2 < corrLines.length; i2++) {
        var l = corrLines[i2], show2 = time >= l.t0;
        l.l1.material.opacity = show2 ? Math.min(0.5, 0.3 - (time - l.t0) * 0.01) : 0;
        l.l2.material.opacity = show2 ? Math.min(0.5, 0.3 - (time - l.t0) * 0.01) : 0;
      }
      elCount = Math.min(cnt, MAX_CORR);
    }

    /* ==========================================
     * MODE 4: Menger Sponge
     * ========================================== */
    function buildMenger() {
      if (mengerBuilt) return; mengerBuilt = true;
      mengerTotal = 0;
      var cubes = [];
      function addCubes(x, y, z, size, depth) {
        if (depth === 0) { cubes.push({ x: x, y: y, z: z, size: size, dist: Math.sqrt(x * x + y * y + z * z) }); return; }
        var s = size / 3;
        for (var dx = 0; dx < 3; dx++) {
          for (var dy = 0; dy < 3; dy++) {
            for (var dz = 0; dz < 3; dz++) {
              if ((dx === 1 && dy === 1) || (dx === 1 && dz === 1) || (dy === 1 && dz === 1)) continue;
              addCubes(x + s * (dx - 1), y + s * (dy - 1), z + s * (dz - 1), s, depth - 1);
            }
          }
        }
      }
      addCubes(0, 0, 0, 1, MENGER_DEPTH);
      mengerTotal = cubes.length;
      cubes.sort(function (a, b) { return a.dist - b.dist; });
      var sf = 2.5;
      for (var i = 0; i < cubes.length; i++) {
        var c = cubes[i];
        var nx = c.x + 0.5, ny = c.y + 0.5, nz = c.z + 0.5;
        var angle = Math.atan2(nz, nx) / (Math.PI * 2) + 0.5;
        var hf = ny + 0.5;
        var hu2 = (angle * 0.4 + hf * 0.2 + 0.48) % 1.0;
        var sat = 0.75 + 0.15 * Math.sin(c.dist * 3);
        var lit = 0.4 + 0.2 * (1 - c.dist / 3);
        var col2 = new THREE.Color().setHSL(hu2, sat, lit);
        var edgeColor = new THREE.Color().setHSL(hu2, sat * 0.8, lit * 0.7);
        var gap = 0.95;
        var geo = new THREE.BoxGeometry(c.size * gap, c.size * gap, c.size * gap);
        var mat = new THREE.MeshBasicMaterial({ color: col2, transparent: true, opacity: 0, blending: THREE.AdditiveBlending });
        var mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(c.x * sf, c.y * sf, c.z * sf); mesh.visible = false; mengerGrp.add(mesh);
        var wir = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: edgeColor, transparent: true, opacity: 0, blending: THREE.AdditiveBlending }));
        wir.visible = false; wir.position.copy(mesh.position); mengerGrp.add(wir);
        var glowMat = new THREE.MeshBasicMaterial({ color: col2, transparent: true, opacity: 0, blending: THREE.AdditiveBlending });
        var glowMesh = new THREE.Mesh(new THREE.SphereGeometry(c.size * 0.3, 6, 6), glowMat);
        glowMesh.position.copy(mesh.position); glowMesh.visible = false; mengerGrp.add(glowMesh);
        mengerEls.push({ mesh: mesh, wir: wir, glow: glowMesh, t0: i * 0.012, size: c.size, dist: c.dist, col: col2, edgeCol: edgeColor, bx: c.x * sf, by: c.y * sf, bz: c.z * sf });
      }
    }

    function tickMenger(time) {
      var cnt = 0;
      mengerGrp.rotation.y += 0.002;
      mengerGrp.rotation.x = Math.sin(time * 0.3) * 0.08;
      for (var i = 0; i < mengerEls.length; i++) {
        var e = mengerEls[i], show = time >= e.t0, el2 = time - e.t0;
        if (e.mesh.visible !== show) { e.mesh.visible = show; e.wir.visible = show; e.glow.visible = show; }
        if (show) {
          cnt++;
          var fadeIn = Math.min(1, el2 / 0.8); fadeIn = fadeIn * fadeIn * (3 - 2 * fadeIn);
          e.mesh.material.opacity = fadeIn * 0.75;
          e.wir.material.opacity = fadeIn * 0.4;
          e.glow.material.opacity = fadeIn * 0.2;
          var pulse = 1 + Math.sin(time * 1.8 + e.dist * 4) * 0.025;
          e.mesh.scale.setScalar(pulse);
          var wirPulse = 1 + Math.sin(time * 1.8 + e.dist * 4 + Math.PI) * 0.01;
          e.wir.scale.setScalar(wirPulse);
          var hueShift = Math.sin(time * 0.5 + e.dist * 2) * 0.05;
          var newCol = new THREE.Color().setHSL(
            (e.by * 0.15 + e.bx * 0.1 + 0.5 + hueShift + time * 0.02) % 1.0,
            0.75 + 0.15 * Math.sin(time + e.dist * 3),
            0.45 + 0.15 * Math.sin(time * 0.8 + e.dist * 5)
          );
          e.mesh.material.color.copy(newCol);
          e.wir.material.color.copy(newCol);
          e.glow.material.color.copy(newCol);
        }
      }
      elCount = Math.min(cnt, mengerTotal);
    }

    /* ==========================================
     * MODE 5: Cantor's Diagonal Argument
     * ========================================== */
    function buildCantor() {
      if (cantorBuilt) return; cantorBuilt = true;
      var gs = 12, sp = 1.0;
      var sX = -(gs - 1) * sp / 2, sY = (gs - 1) * sp / 2;
      var matrix = [];
      for (var r = 0; r < gs; r++) {
        var bits = [];
        for (var c = 0; c < gs; c++) bits.push((r + c) % 2);
        matrix.push(bits);
      }
      var diag = [];
      for (var di = 0; di < gs; di++) diag.push(matrix[di][di]);
      var newStr = diag.map(function (b) { return b === 0 ? 1 : 0; });
      var dLabel = mkLabel('Diagonal', '#ff7043'); dLabel.position.set(sX - gs * sp / 2 - 3, sY + 2, 0); dLabel.scale.set(3, 1.5, 1); cantorGrp.add(dLabel);
      var nLabel = mkLabel('New String', '#66bb6a'); nLabel.position.set(sX, sY - (gs + 2) * sp, 0); nLabel.scale.set(4, 1.5, 1); cantorGrp.add(nLabel);
      for (var ri = 0; ri < gs; ri++) {
        var rLabel = mkLabel('S' + (ri + 1), '#556'); rLabel.position.set(sX - 1.8, sY - ri * sp, 0); rLabel.scale.set(1.8, 0.9, 1); cantorGrp.add(rLabel);
      }
      for (var ri2 = 0; ri2 < gs; ri2++) {
        for (var ci2 = 0; ci2 < gs; ci2++) {
          var xPos = sX + ci2 * sp, yPos = sY - ri2 * sp;
          var isDiag = ri2 === ci2, bit = matrix[ri2][ci2];
          var cellGeo = new THREE.PlaneGeometry(0.85, 0.85);
          var cellCol = isDiag ? 0xff7043 : (bit === 1 ? 0x64b5f6 : 0x222233);
          var cellMat = new THREE.MeshBasicMaterial({ color: cellCol, transparent: true, opacity: 0, blending: THREE.AdditiveBlending });
          var cell = new THREE.Mesh(cellGeo, cellMat);
          cell.position.set(xPos, yPos, 0); cell.visible = false; cantorGrp.add(cell);
          var bLabel = mkLabel(String(bit), isDiag ? '#fff' : (bit === 1 ? '#fff' : '#333'));
          bLabel.position.set(xPos, yPos, 0.01); bLabel.scale.set(0.8, 0.4, 1); bLabel.visible = false; cantorGrp.add(bLabel);
          cantorEls.push({ cell: cell, label: bLabel, row: ri2, col: ci2, isDiag: isDiag, t0: (ri2 * gs + ci2) * 0.02 });
        }
      }
      for (var ci3 = 0; ci3 < gs; ci3++) {
        var xPos2 = sX + ci3 * sp, yPos2 = sY - (gs + 1) * sp;
        var bit2 = newStr[ci3];
        var cGeo2 = new THREE.PlaneGeometry(0.85, 0.85);
        var cMat2 = new THREE.MeshBasicMaterial({ color: 0x66bb6a, transparent: true, opacity: 0, blending: THREE.AdditiveBlending });
        var cell2 = new THREE.Mesh(cGeo2, cMat2);
        cell2.position.set(xPos2, yPos2, 0); cell2.visible = false; cantorGrp.add(cell2);
        var bLabel2 = mkLabel(String(bit2), '#fff');
        bLabel2.position.set(xPos2, yPos2, 0.01); bLabel2.scale.set(0.8, 0.4, 1); bLabel2.visible = false; cantorGrp.add(bLabel2);
        cantorEls.push({ cell: cell2, label: bLabel2, row: gs + 1, col: ci3, isDiag: false, t0: (gs * gs + ci3) * 0.02 });
      }
      var expl = mkLabel('The diagonal \u2260 any row!', '#ff7043');
      expl.position.set(sX + gs * sp / 2 + 3, sY - gs * sp / 2, 0); expl.scale.set(4, 1.5, 1); cantorGrp.add(expl);
    }

    function tickCantor(time) {
      var cnt = 0;
      for (var i = 0; i < cantorEls.length; i++) {
        var e = cantorEls[i], show = time >= e.t0;
        if (e.cell.visible !== show) { e.cell.visible = show; e.label.visible = show; }
        if (show) {
          cnt++;
          var fadeIn = Math.min(1, (time - e.t0) / 0.5);
          e.cell.material.opacity = fadeIn * 0.85;
          if (e.isDiag) e.cell.material.opacity = fadeIn * (0.7 + 0.3 * Math.sin(time * 2 + e.row));
        }
      }
      elCount = Math.min(cnt, 12 * 14);
    }

    /* ==========================================
     * MODE 6: Rationals ℚ Enumeration
     * ========================================== */
    function buildRationals() {
      if (rationBuilt) return; rationBuilt = true;
      var gs = 16, sp = 1.2;
      var sX = -(gs - 1) * sp / 2, sY = (gs - 1) * sp / 2;
      function gcd(a, b) { while (b) { var t = a % b; a = b; b = t; } return a; }
      var fractions = [];
      for (var p = 1; p <= gs; p++) {
        for (var q = 1; q <= gs; q++) {
          if (gcd(p, q) === 1) fractions.push({ p: p, q: q, val: p / q });
        }
      }
      fractions.sort(function (a, b) { return a.val - b.val; });
      var disp = fractions.slice(0, gs * gs);
      var topL = mkLabel('Numerators p \u2192', '#64b5f6');
      topL.position.set(sX + (gs - 1) * sp / 2, sY + 1.5, 0); topL.scale.set(5, 1.5, 1); rationGrp.add(topL);
      var leftL = mkLabel('Denominators q \u2193', '#81c784');
      leftL.position.set(sX - 1.5, sY - (gs - 1) * sp / 2, 0); leftL.scale.set(4, 1.5, 1); rationGrp.add(leftL);
      for (var j = 0; j < disp.length; j++) {
        var frac = disp[j];
        var col = (frac.p - 1) % gs, row = (frac.q - 1) % gs;
        var xPos = sX + col * sp, yPos = sY - row * sp;
        var cGeo = new THREE.PlaneGeometry(0.9, 0.9);
        var hu = (frac.val / gs * 0.5 + 0.5) % 1.0;
        var cCol = new THREE.Color().setHSL(hu, 0.7, 0.45);
        var cMat = new THREE.MeshBasicMaterial({ color: cCol, transparent: true, opacity: 0, blending: THREE.AdditiveBlending });
        var cMesh = new THREE.Mesh(cGeo, cMat);
        cMesh.position.set(xPos, yPos, 0); cMesh.visible = false; rationGrp.add(cMesh);
        var fLabel = mkLabel(frac.p + '/' + frac.q, '#fff');
        fLabel.position.set(xPos, yPos, 0.01); fLabel.scale.set(0.9, 0.45, 1); fLabel.visible = false; rationGrp.add(fLabel);
        rationEls.push({ cell: cMesh, label: fLabel, frac: frac, row: row, col: col, t0: j * 0.03 });
      }
    }

    function tickRationals(time) {
      var cnt = 0;
      for (var i = 0; i < rationEls.length; i++) {
        var e = rationEls[i], show = time >= e.t0;
        if (e.cell.visible !== show) { e.cell.visible = show; e.label.visible = show; }
        if (show) {
          cnt++;
          e.cell.material.opacity = Math.min(1, (time - e.t0) / 0.5) * 0.75;
        }
      }
      elCount = Math.min(cnt, 16 * 16);
    }

    /* ==========================================
     * Info Text Content
     * ========================================== */
    var modeInfo = {
      spiral: {
        title: 'Number Spiral',
        text: 'Numbers <strong>1, 2, 3, ...</strong> are arranged on a 3D helix, spiraling outward and upward. Each glowing point represents a natural number, appearing one by one. The spiral never ends \u2014 it goes on forever, just like the natural numbers.<br><br>This is <strong>\u21350</strong> (Aleph Null): the cardinality of the set of all natural numbers. It is the <strong>smallest infinity</strong> \u2014 infinite, but <em>countable</em>: every number can in principle be listed.'
      },
      hotel: {
        title: "Hilbert's Hotel",
        text: '<strong>Hilbert\u2019s Hotel</strong> is a famous paradox of countable infinity.<br><br>An infinite hotel is <strong>fully occupied</strong> (every room 1, 2, 3, ... is taken). A new guest arrives. The manager simply moves every guest from room <em>n</em> to room <em>n+1</em>, freeing room 1 for the new guest.<br><br><strong>Infinity + 1 = infinity.</strong> The hotel never runs out of rooms. This shows that <strong>\u21350 + 1 = \u21350</strong> \u2014 adding a single element to an infinite set doesn\u2019t change its size.'
      },
      corr: {
        title: 'One-to-One Correspondence',
        text: 'Two sets are <strong>equinumerous</strong> (same cardinality) if there is a <strong>bijection</strong> between them \u2014 a one-to-one, onto mapping.<br><br>Below, each natural number (blue) is paired with its <strong>double</strong> (green) and its corresponding <strong>prime</strong> (orange). These are all infinite sets, but they all have the same size: <strong>\u21350</strong>.<br><br><strong>\u2115 \u2192 2\u2115</strong> maps n \u2192 2n<br><strong>\u2115 \u2192 P</strong> maps n \u2192 the n-th prime<br>Every element of the naturals is matched exactly once \u2014 a perfect one-to-one correspondence.'
      },
      menger: {
        title: 'Menger Sponge',
        text: 'The <strong>Menger Sponge</strong> is a fractal with <strong>infinite surface area</strong> but <strong>zero volume</strong>.<br><br>It is constructed by repeatedly removing the center of each face and the center of the cube from smaller subdivisions \u2014 starting from a single cube, then 20, then 400, then 8000, etc.<br><br>In an <strong>infinite</strong> iteration, it has infinitely many holes, creating a boundary of infinite complexity. This is analogous to <strong>\u21350</strong>: an object of fractal complexity with infinitely many subparts, yet still <em>countable</em> when discretized (each cube can be numbered).<br><br><strong>Key insight:</strong> The Menger Sponge demonstrates how infinity can be <em>structural</em> \u2014 its recursive removal process generates an infinite number of holes, each of which is itself a copy of the shape.'
      },
      cantor: {
        title: "Cantor's Diagonal Argument",
        text: '<strong>Cantor\u2019s Diagonal Argument</strong> proves that the set of real numbers is <strong>uncountable</strong> \u2014 it cannot be put into a one-to-one correspondence with the natural numbers.<br><br>The idea: Suppose you have an infinite list of binary strings (like the grid below). Each row is a different infinite binary string. Now, construct a new string by <strong>taking the diagonal</strong> (row 1, col 1; row 2, col 2; ...) and <strong>flipping each bit</strong>.<br><br>This new string <strong>differs from every row</strong> in at least one position (the diagonal index). Therefore, it cannot be on the list \u2014 proving that the list is <em>incomplete</em>.<br><br>This demonstrates that <strong>|R| > \u21350</strong> \u2014 the real numbers are strictly more numerous than the countable infinity of the naturals.'
      },
      rationals: {
        title: 'Rationale Number Enumeration',
        text: 'The <strong>rational numbers (\u211A)</strong> \u2014 all fractions p/q \u2014 are <strong>countable</strong>, meaning their cardinality is \u21350. This is surprising: there are infinitely many rational numbers between any two integers, yet they can be enumerated.<br><br>The method: list all fractions p/q on a <strong>2D grid</strong> and enumerate them along <strong>diagonals</strong> (p + q = 2, then p + q = 3, then p + q = 4, ...). Skip duplicates (e.g., 2/2 = 1/1).<br><br>Each rational number gets a natural number as its index, proving <strong>|\u211A| = \u21350</strong>.<br><br>This is a beautiful example of how the infinity of the rationals is the <em>same</em> infinity as the naturals \u2014 both are \u21350.'
      }
    };

    function setMod(mode) {
      curMode = mode; elTime = 0; elCount = 0;
      spiralGrp.visible = false; hotelGrp.visible = false; corrGrp.visible = false;
      mengerGrp.visible = false; cantorGrp.visible = false; rationGrp.visible = false;
      camTransition = false;
      if (mode === 'spiral') { spiralGrp.visible = true; buildSpiral(); orbitData.tx = 0; orbitData.ty = 0; orbitData.tz = 0; orbitData.zoom = 30; }
      else if (mode === 'hotel') { hotelGrp.visible = true; buildHotel(); hotelPhase = 0; hotelTimer = 0; orbitData.tx = 0; orbitData.ty = 1; orbitData.tz = 0; orbitData.zoom = 30; }
      else if (mode === 'corr') { corrGrp.visible = true; buildCorr(); orbitData.tx = 0; orbitData.ty = 0; orbitData.tz = 0; orbitData.zoom = 40; }
      else if (mode === 'menger') { mengerGrp.visible = true; buildMenger(); orbitData.tx = 0; orbitData.ty = 0; orbitData.tz = 0; orbitData.zoom = 10; }
      else if (mode === 'cantor') { cantorGrp.visible = true; buildCantor(); orbitData.tx = 0; orbitData.ty = 0; orbitData.tz = 0; orbitData.zoom = 25; }
      else if (mode === 'rationals') { rationGrp.visible = true; buildRationals(); orbitData.tx = 0; orbitData.ty = 0; orbitData.tz = 0; orbitData.zoom = 28; }
      document.getElementById('info-title').textContent = modeInfo[mode].title;
      document.getElementById('info-text').innerHTML = modeInfo[mode].text;
      document.querySelectorAll('.mode-btn').forEach(function (b) {
        var isActive = b.getAttribute('data-mode') === mode;
        b.classList.toggle('active', isActive);
        b.setAttribute('aria-selected', isActive ? 'true' : 'false');
      });
      setTimeout(function () { camTransition = true; }, 100);
    }

    function resetAll() {
      elTime = 0; elCount = 0; hotelPhase = 0; hotelTimer = 0;
      spiralEls.forEach(function (e) { e.sp.visible = false; e.lb.visible = false; });
      corrEls.forEach(function (d) { d.dot.visible = false; d.lb.visible = false; });
      corrLines.forEach(function (l) { l.l1.material.opacity = 0; l.l2.material.opacity = 0; });
      hotelEls.forEach(function (e) { e.guest.visible = false; e.ol.visible = false; });
      if (hotelGrp.userData.ng) hotelGrp.userData.ng.visible = false;
      mengerEls.forEach(function (e) { e.mesh.visible = false; e.wir.visible = false; if (e.glow) e.glow.visible = false; });
      cantorEls.forEach(function (e) { e.cell.visible = false; e.label.visible = false; });
      rationEls.forEach(function (e) { e.cell.visible = false; e.label.visible = false; });
    }

    function updateSpeedDisplay() {
      var el = document.getElementById('speed-label');
      if (el) el.textContent = 'Speed: ' + animSpeed.toFixed(1) + 'x';
    }

    /* ==========================================
     * UI Wire-up
     * ========================================== */
    document.querySelectorAll('.mode-btn').forEach(function (btn) {
      btn.addEventListener('click', function () { setMod(this.getAttribute('data-mode')); });
    });
    document.getElementById('btn-play').addEventListener('click', function () {
      playing = true;
      document.getElementById('btn-play').classList.add('active');
      document.getElementById('btn-pause').classList.remove('active');
      document.getElementById('btn-play').setAttribute('aria-pressed', 'true');
      document.getElementById('btn-pause').setAttribute('aria-pressed', 'false');
    });
    document.getElementById('btn-pause').addEventListener('click', function () {
      playing = false;
      document.getElementById('btn-pause').classList.add('active');
      document.getElementById('btn-play').classList.remove('active');
      document.getElementById('btn-pause').setAttribute('aria-pressed', 'true');
      document.getElementById('btn-play').setAttribute('aria-pressed', 'false');
    });
    document.getElementById('btn-reset').addEventListener('click', function () { resetAll(); });
    document.getElementById('btn-slower').addEventListener('click', function () { animSpeed = Math.max(0.1, animSpeed - 0.2); updateSpeedDisplay(); });
    document.getElementById('btn-faster').addEventListener('click', function () { animSpeed = Math.min(5, animSpeed + 0.2); updateSpeedDisplay(); });
    document.getElementById('fullscreen-btn').addEventListener('click', function () {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(function () {});
      } else {
        document.exitFullscreen().catch(function () {});
      }
    });
    window.addEventListener('keydown', function (e) {
      if (e.key === '1') setMod('spiral');
      if (e.key === '2') setMod('hotel');
      if (e.key === '3') setMod('corr');
      if (e.key === '4') setMod('menger');
      if (e.key === '5') setMod('cantor');
      if (e.key === '6') setMod('rationals');
      if (e.key === ' ') { e.preventDefault(); if (playing) { playing = false; document.getElementById('btn-pause').classList.add('active'); document.getElementById('btn-play').classList.remove('active'); } else { playing = true; document.getElementById('btn-play').classList.add('active'); document.getElementById('btn-pause').classList.remove('active'); } }
      if (e.key === 'f' || e.key === 'F') { if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(function () {}); else document.exitFullscreen().catch(function () {}); }
      if (e.key === 'r' || e.key === 'R') resetAll();
    });
    window.addEventListener('resize', function () {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    setMod('spiral');
    updateSpeedDisplay();

    /* ==========================================
     * Animation Loop
     * ========================================== */
    function animate() {
      requestAnimationFrame(animate);
      var dt = clock.getDelta(), raw = clock.getElapsedTime();
      if (playing) elTime += dt * animSpeed;
      var time = elTime;
      if (curMode === 'spiral') tickSpiral(time);
      else if (curMode === 'hotel') tickHotel(dt * animSpeed, time);
      else if (curMode === 'corr') tickCorr(time);
      else if (curMode === 'menger') tickMenger(time);
      else if (curMode === 'cantor') tickCantor(time);
      else if (curMode === 'rationals') tickRationals(time);
      document.getElementById('counter-number').textContent = elCount;
      light1.position.x = Math.cos(raw * 0.3) * 25;
      light1.position.z = Math.sin(raw * 0.3) * 25;
      light2.position.x = Math.sin(raw * 0.2) * 20;
      light2.position.z = Math.cos(raw * 0.2) * 20;
      stars.rotation.y += 0.0001;
      updateCamera();
      renderer.render(scene, camera);
    }

    animate();

    /* ==========================================
     * Loading Screen
     * ========================================== */
    setTimeout(function () {
      var ld = document.getElementById('loading-screen');
      if (ld) { ld.style.opacity = '0'; setTimeout(function () { ld.style.display = 'none'; }, 800); }
    }, 1500);

  } catch (err) {
    console.error('Aleph-0 initialization error:', err);
    var ld = document.getElementById('loading-screen');
    if (ld) { ld.style.opacity = '0'; ld.style.display = 'none'; }
    var errDiv = document.createElement('div');
    errDiv.id = 'error-display';
    errDiv.innerHTML = '<h2>Error initializing 3D</h2><p>' + err.message + '</p><p style="font-size:12px;margin-top:10px;color:#888">' + (err.stack ? err.stack.replace(/</g, '<').replace(/>/g, '>') : '') + '</p>';
    document.body.appendChild(errDiv);
  }

})();