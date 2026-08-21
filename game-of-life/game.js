(function () {
    /* ═══ Config ═══ */
    var CELL = 12, COLS = 80, ROWS = 50;
    var ALIVE = '#00ff00', DEAD = '#0a0a0a', GRID = '#001a00', BG = '#000';

    /* ═══ DOM ═══ */
    var canvas    = document.getElementById('gol-canvas');
    var ctx       = canvas.getContext('2d');
    var btnPlay   = document.getElementById('btn-play');
    var btnStep   = document.getElementById('btn-step');
    var btnClear  = document.getElementById('btn-clear');
    var btnRandom = document.getElementById('btn-random');
    var speedIn   = document.getElementById('speed');
    var presetSel = document.getElementById('preset');
    var gridTog   = document.getElementById('grid-toggle');
    var statGen   = document.getElementById('stat-gen');
    var statAlive = document.getElementById('stat-alive');
    var statFps   = document.getElementById('stat-fps');

    canvas.width  = COLS * CELL;
    canvas.height = ROWS * CELL;

    /* ═══ State ═══ */
    var grid = makeGrid(), running = false, generation = 0;
    var showGrid = true, painting = false, paintVal = 1;
    var lastFrame = 0, frames = 0, fpsTime = 0;

    function makeGrid() {
        var g = [];
        for (var r = 0; r < ROWS; r++) { g[r] = []; for (var c = 0; c < COLS; c++) g[r][c] = 0; }
        return g;
    }

    /* ═══ Conway Step ═══ */
    function step() {
        var next = makeGrid();
        for (var r = 0; r < ROWS; r++) {
            for (var c = 0; c < COLS; c++) {
                var n = 0;
                for (var dr = -1; dr <= 1; dr++) {
                    for (var dc = -1; dc <= 1; dc++) {
                        if (!dr && !dc) continue;
                        var nr = r + dr, nc = c + dc;
                        if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) n += grid[nr][nc];
                    }
                }
                next[r][c] = grid[r][c] ? (n === 2 || n === 3 ? 1 : 0) : (n === 3 ? 1 : 0);
            }
        }
        grid = next;
        generation++;
    }

    /* ═══ Rendering ═══ */
    function draw() {
        ctx.fillStyle = BG;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = ALIVE;
        for (var r = 0; r < ROWS; r++)
            for (var c = 0; c < COLS; c++)
                if (grid[r][c]) ctx.fillRect(c * CELL, r * CELL, CELL - 1, CELL - 1);
        if (showGrid) {
            ctx.strokeStyle = GRID;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            for (var x = 0; x <= COLS; x++) { ctx.moveTo(x * CELL, 0); ctx.lineTo(x * CELL, ROWS * CELL); }
            for (var y = 0; y <= ROWS; y++) { ctx.moveTo(0, y * CELL); ctx.lineTo(COLS * CELL, y * CELL); }
            ctx.stroke();
        }
        var alive = 0;
        for (var r2 = 0; r2 < ROWS; r2++) for (var c2 = 0; c2 < COLS; c2++) if (grid[r2][c2]) alive++;
        statGen.textContent   = generation;
        statAlive.textContent = alive;
    }

    /* ═══ Game Loop ═══ */
    function getInterval() { return 550 - speedIn.value * 25; }

    function loop(ts) {
        frames++;
        if (ts - fpsTime >= 1000) { statFps.textContent = frames; frames = 0; fpsTime = ts; }
        if (running && ts - lastFrame >= getInterval()) { step(); lastFrame = ts; }
        draw();
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);

    /* ═══ Drawing on Canvas ═══ */
    function cellFromEvent(e) {
        var rect = canvas.getBoundingClientRect();
        var sx = canvas.width / rect.width, sy = canvas.height / rect.height;
        var cx, cy;
        if (e.touches) { cx = e.touches[0].clientX; cy = e.touches[0].clientY; }
        else { cx = e.clientX; cy = e.clientY; }
        var col = Math.floor((cx - rect.left) * sx / CELL);
        var row = Math.floor((cy - rect.top) * sy / CELL);
        return (row >= 0 && row < ROWS && col >= 0 && col < COLS) ? { r: row, c: col } : null;
    }

    function paint(e) { var p = cellFromEvent(e); if (p) grid[p.r][p.c] = paintVal; }

    canvas.addEventListener('mousedown', function (e) {
        e.preventDefault(); painting = true;
        paintVal = e.button === 2 ? 0 : 1; paint(e);
    });
    canvas.addEventListener('mousemove', function (e) { if (painting) paint(e); });
    window.addEventListener('mouseup', function () { painting = false; });
    canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    canvas.addEventListener('touchstart', function (e) { e.preventDefault(); painting = true; paintVal = 1; paint(e); }, { passive: false });
    canvas.addEventListener('touchmove', function (e) { e.preventDefault(); if (painting) paint(e); }, { passive: false });
    canvas.addEventListener('touchend', function () { painting = false; });

    /* ═══ Presets ═══ */
    var PRESETS = {
        'glider':        { cells: [[0,1],[1,2],[2,0],[2,1],[2,2]] },
        'lwss':          { cells: [[0,1],[0,4],[1,0],[2,0],[2,4],[3,0],[3,1],[3,2],[3,3]] },
        'blinker':       { cells: [[0,0],[0,1],[0,2]] },
        'toad':          { cells: [[0,1],[0,2],[0,3],[1,0],[1,1],[1,2]] },
        'beacon':        { cells: [[0,0],[0,1],[1,0],[2,3],[3,2],[3,3]] },
        'pulsar':        { cells: [
            [0,2],[0,3],[0,4],[0,8],[0,9],[0,10],
            [2,0],[2,5],[2,7],[2,12],
            [3,0],[3,5],[3,7],[3,12],
            [4,0],[4,5],[4,7],[4,12],
            [5,2],[5,3],[5,4],[5,8],[5,9],[5,10],
            [7,2],[7,3],[7,4],[7,8],[7,9],[7,10],
            [8,0],[8,5],[8,7],[8,12],
            [9,0],[9,5],[9,7],[9,12],
            [10,0],[10,5],[10,7],[10,12],
            [12,2],[12,3],[12,4],[12,8],[12,9],[12,10]
        ]},
        'pentadecathlon': { cells: [[0,1],[1,1],[2,0],[2,2],[3,1],[4,1],[5,1],[6,1],[7,0],[7,2],[8,1],[9,1]] },
        'glider-gun':    { cells: [
            [0,24],[1,22],[1,24],
            [2,12],[2,13],[2,20],[2,21],[2,34],[2,35],
            [3,11],[3,15],[3,20],[3,21],[3,34],[3,35],
            [4,0],[4,1],[4,10],[4,16],[4,20],[4,21],
            [5,0],[5,1],[5,10],[5,14],[5,16],[5,17],[5,22],[5,24],
            [6,10],[6,16],[6,24],
            [7,11],[7,15],
            [8,12],[8,13]
        ]},
        'r-pentomino':   { cells: [[0,1],[0,2],[1,0],[1,1],[2,1]] },
        'diehard':       { cells: [[0,6],[1,0],[1,1],[2,1],[2,5],[2,6],[2,7]] },
        'acorn':         { cells: [[0,1],[1,3],[2,0],[2,1],[2,4],[2,5],[2,6]] }
    };

    function loadPreset(name) {
        var p = PRESETS[name]; if (!p) return;
        grid = makeGrid(); generation = 0;
        var offR = Math.floor((ROWS - 14) / 2), offC = Math.floor((COLS - 14) / 2);
        p.cells.forEach(function (c) {
            var r = c[0] + offR, col = c[1] + offC;
            if (r >= 0 && r < ROWS && col >= 0 && col < COLS) grid[r][col] = 1;
        });
    }

    /* ═══ Controls ═══ */
    function togglePlay() {
        running = !running;
        btnPlay.textContent = running ? '\u23F8 Pause' : '\u25B6 Play';
        btnPlay.classList.toggle('active', running);
    }

    function doRandom() {
        grid = makeGrid(); generation = 0;
        for (var r = 0; r < ROWS; r++)
            for (var c = 0; c < COLS; c++)
                grid[r][c] = Math.random() < 0.3 ? 1 : 0;
    }

    btnPlay.addEventListener('click', togglePlay);
    btnStep.addEventListener('click', function () { step(); draw(); });
    btnClear.addEventListener('click', function () {
        grid = makeGrid(); generation = 0; running = false;
        btnPlay.textContent = '\u25B6 Play'; btnPlay.classList.remove('active');
    });
    btnRandom.addEventListener('click', doRandom);
    gridTog.addEventListener('change', function () { showGrid = gridTog.checked; });
    presetSel.addEventListener('change', function () {
        if (presetSel.value) { loadPreset(presetSel.value); presetSel.value = ''; }
    });

    /* ═══ Keyboard Shortcuts ═══ */
    document.addEventListener('keydown', function (e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        switch (e.key.toLowerCase()) {
            case ' ': e.preventDefault(); togglePlay(); break;
            case 's': step(); draw(); break;
            case 'c': grid = makeGrid(); generation = 0; break;
            case 'r': doRandom(); break;
        }
    });
})();