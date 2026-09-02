/* Space Invaders — BlackAngelSK (delta-time) */
(function () {
    'use strict';

    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    const scoreEl = document.getElementById('score');
    const livesEl = document.getElementById('lives');
    const waveEl = document.getElementById('wave');
    const overlay = document.getElementById('overlay');
    const overlayTitle = document.getElementById('overlay-title');
    const overlayScore = document.getElementById('overlay-score');
    const overlayMsg = document.getElementById('overlay-msg');

    const W = canvas.width;
    const H = canvas.height;

    /* ── State ───────────────────────────────────────── */
    let state = 'title'; // title | playing | paused | dead | gameover | win
    let score = 0;
    let lives = 3;
    let wave = 1;
    let lastTime = 0;

    const keys = {};
    document.addEventListener('keydown', e => {
        keys[e.code] = true;
        if (e.code === 'Space') e.preventDefault();
    });
    document.addEventListener('keyup', e => { keys[e.code] = false; });

    /* ── Player ──────────────────────────────────────── */
    const PLAYER = { w: 36, h: 20, speed: 220, x: 0, y: 0 }; // px/sec
    let playerBullets = [];
    let playerCooldown = 0;

    function resetPlayer() {
        PLAYER.x = W / 2 - PLAYER.w / 2;
        PLAYER.y = H - 40;
        playerBullets = [];
        playerCooldown = 0;
    }

    /* ── Aliens ──────────────────────────────────────── */
    const ALIEN_ROWS = 5;
    const ALIEN_COLS = 11;
    const ALIEN_PAD = 12;
    const ALIEN_W = 28;
    const ALIEN_H = 22;
    let aliens = [];
    let alienDir = 1;       // 1 = right, -1 = left
    let alienSpeed = 30;    // px/sec horizontal
    let alienMoveTimer = 0;
    let alienMoveInterval = 0.7; // seconds between moves
    let alienBullets = [];
    let alienShootTimer = 0;
    let alienShootInterval = 1.5; // seconds between shots

    const ROW_POINTS = [10, 20, 20, 30, 30];
    const ROW_COLORS = ['#33ff33', '#33ff33', '#00cc00', '#00ffaa', '#00ffaa'];

    function spawnAliens() {
        aliens = [];
        const totalW = ALIEN_COLS * (ALIEN_W + ALIEN_PAD) - ALIEN_PAD;
        const startX = (W - totalW) / 2;
        for (let r = 0; r < ALIEN_ROWS; r++) {
            for (let c = 0; c < ALIEN_COLS; c++) {
                aliens.push({
                    x: startX + c * (ALIEN_W + ALIEN_PAD),
                    y: 60 + r * (ALIEN_H + ALIEN_PAD),
                    w: ALIEN_W,
                    h: ALIEN_H,
                    alive: true,
                    row: r,
                    points: ROW_POINTS[r],
                    color: ROW_COLORS[r]
                });
            }
        }
        alienDir = 1;
        alienSpeed = 30 + wave * 10;
        alienMoveInterval = Math.max(0.15, 0.7 - wave * 0.07);
        alienMoveTimer = 0;
        alienShootInterval = Math.max(0.4, 1.5 - wave * 0.1);
        alienShootTimer = 0;
        alienBullets = [];
    }

    /* ── Shields ─────────────────────────────────────── */
    let shields = [];
    const SHIELD_COUNT = 4;
    const SHIELD_W = 60;
    const SHIELD_H = 40;
    const SHIELD_BLOCK = 4;

    function spawnShields() {
        shields = [];
        const totalW = SHIELD_COUNT * SHIELD_W + (SHIELD_COUNT - 1) * 40;
        const startX = (W - totalW) / 2;
        for (let s = 0; s < SHIELD_COUNT; s++) {
            const sx = startX + s * (SHIELD_W + 40);
            const sy = H - 90;
            for (let bx = 0; bx < SHIELD_W; bx += SHIELD_BLOCK) {
                for (let by = 0; by < SHIELD_H; by += SHIELD_BLOCK) {
                    const cx = bx - SHIELD_W / 2 + SHIELD_BLOCK / 2;
                    const cy = by - SHIELD_H;
                    if (cy > -SHIELD_BLOCK * 4 && Math.abs(cx) < SHIELD_BLOCK * 4) continue;
                    shields.push({
                        x: sx + bx,
                        y: sy + by,
                        w: SHIELD_BLOCK,
                        h: SHIELD_BLOCK,
                        hp: 3
                    });
                }
            }
        }
    }

    /* ── Particles ───────────────────────────────────── */
    let particles = [];

    function explode(x, y, color, count) {
        for (let i = 0; i < count; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.random() * 80 + 30; // px/sec
            particles.push({
                x, y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                life: 0.5 + Math.random() * 0.4,
                age: 0,
                color,
                size: Math.random() * 3 + 1
            });
        }
    }

    /* ── Collision ───────────────────────────────────── */
    function rectHit(a, b) {
        return a.x < b.x + b.w && a.x + a.w > b.x &&
               a.y < b.y + b.h && a.y + a.h > b.y;
    }

    /* ── Init / Reset ────────────────────────────────── */
    function startGame() {
        score = 0;
        lives = 3;
        wave = 1;
        scoreEl.textContent = score;
        livesEl.textContent = lives;
        waveEl.textContent = wave;
        resetPlayer();
        spawnAliens();
        spawnShields();
        particles = [];
        state = 'playing';
        overlay.classList.add('hidden');
    }

    function nextWave() {
        wave++;
        waveEl.textContent = wave;
        resetPlayer();
        spawnAliens();
    }

    /* ── Update (dt in seconds) ──────────────────────── */
    function update(dt) {
        if (state !== 'playing') return;

        /* player movement */
        if (keys['ArrowLeft'] || keys['KeyA'])  PLAYER.x -= PLAYER.speed * dt;
        if (keys['ArrowRight'] || keys['KeyD']) PLAYER.x += PLAYER.speed * dt;
        PLAYER.x = Math.max(0, Math.min(W - PLAYER.w, PLAYER.x));

        /* player shoot */
        if (playerCooldown > 0) playerCooldown -= dt;
        if (keys['Space'] && playerCooldown <= 0) {
            playerBullets.push({
                x: PLAYER.x + PLAYER.w / 2 - 1.5,
                y: PLAYER.y - 6,
                w: 3, h: 8,
                speed: 400 // px/sec
            });
            playerCooldown = 0.2; // seconds
        }

        /* player bullets */
        for (let i = playerBullets.length - 1; i >= 0; i--) {
            const b = playerBullets[i];
            b.y -= b.speed * dt;
            if (b.y + b.h < 0) { playerBullets.splice(i, 1); continue; }

            let hit = false;
            for (const a of aliens) {
                if (!a.alive) continue;
                if (rectHit(b, a)) {
                    a.alive = false;
                    score += a.points;
                    scoreEl.textContent = score;
                    explode(a.x + a.w / 2, a.y + a.h / 2, a.color, 12);
                    hit = true;
                    break;
                }
            }
            if (hit) { playerBullets.splice(i, 1); continue; }

            for (let s = shields.length - 1; s >= 0; s--) {
                if (rectHit(b, shields[s])) {
                    shields[s].hp--;
                    if (shields[s].hp <= 0) shields.splice(s, 1);
                    playerBullets.splice(i, 1);
                    hit = true;
                    break;
                }
            }
        }

        /* alien movement */
        alienMoveTimer += dt;
        if (alienMoveTimer >= alienMoveInterval) {
            alienMoveTimer -= alienMoveInterval;
            let hitEdge = false;
            for (const a of aliens) {
                if (!a.alive) continue;
                if ((alienDir > 0 && a.x + a.w + alienSpeed * alienMoveInterval > W) ||
                    (alienDir < 0 && a.x - alienSpeed * alienMoveInterval < 0)) {
                    hitEdge = true;
                    break;
                }
            }
            if (hitEdge) {
                alienDir *= -1;
                for (const a of aliens) {
                    if (a.alive) a.y += ALIEN_H;
                }
            } else {
                for (const a of aliens) {
                    if (a.alive) a.x += alienSpeed * alienMoveInterval * alienDir;
                }
            }
        }

        /* alien shoot */
        alienShootTimer += dt;
        if (alienShootTimer >= alienShootInterval) {
            alienShootTimer -= alienShootInterval;
            const aliveAliens = aliens.filter(a => a.alive);
            if (aliveAliens.length > 0) {
                const cols = {};
                for (const a of aliveAliens) {
                    const col = Math.round(a.x);
                    if (!cols[col] || a.y > cols[col].y) cols[col] = a;
                }
                const shooters = Object.values(cols);
                const shooter = shooters[Math.floor(Math.random() * shooters.length)];
                alienBullets.push({
                    x: shooter.x + shooter.w / 2 - 1.5,
                    y: shooter.y + shooter.h,
                    w: 3, h: 10,
                    speed: 180 + wave * 18 // px/sec
                });
            }
        }

        /* alien bullets */
        for (let i = alienBullets.length - 1; i >= 0; i--) {
            const b = alienBullets[i];
            b.y += b.speed * dt;
            if (b.y > H) { alienBullets.splice(i, 1); continue; }

            if (rectHit(b, { x: PLAYER.x, y: PLAYER.y, w: PLAYER.w, h: PLAYER.h })) {
                alienBullets.splice(i, 1);
                lives--;
                livesEl.textContent = lives;
                explode(PLAYER.x + PLAYER.w / 2, PLAYER.y + PLAYER.h / 2, '#00ff00', 20);
                if (lives <= 0) {
                    state = 'gameover';
                    showOverlay('GAME OVER', 'Press SPACE to restart', score);
                } else {
                    resetPlayer();
                }
                continue;
            }

            for (let s = shields.length - 1; s >= 0; s--) {
                if (rectHit(b, shields[s])) {
                    shields[s].hp--;
                    if (shields[s].hp <= 0) shields.splice(s, 1);
                    alienBullets.splice(i, 1);
                    break;
                }
            }
        }

        /* aliens reached bottom */
        const aliveAliens = aliens.filter(a => a.alive);
        for (const a of aliveAliens) {
            if (a.y + a.h >= PLAYER.y) {
                state = 'gameover';
                showOverlay('GAME OVER', 'Press SPACE to restart', score);
                return;
            }
        }

        /* all aliens dead → next wave */
        if (aliveAliens.length === 0) {
            nextWave();
        }

        /* particles */
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.age += dt;
            if (p.age >= p.life) particles.splice(i, 1);
        }
    }

    /* ── Draw ────────────────────────────────────────── */
    function drawPlayer() {
        ctx.fillStyle = '#00ff00';
        ctx.fillRect(PLAYER.x + 4, PLAYER.y + 6, PLAYER.w - 8, PLAYER.h - 6);
        ctx.fillRect(PLAYER.x + PLAYER.w / 2 - 2, PLAYER.y, 4, 8);
        ctx.fillRect(PLAYER.x, PLAYER.y + PLAYER.h - 6, 6, 6);
        ctx.fillRect(PLAYER.x + PLAYER.w - 6, PLAYER.y + PLAYER.h - 6, 6, 6);
    }

    function drawAlien(a) {
        ctx.fillStyle = a.color;
        const x = Math.round(a.x);
        const y = Math.round(a.y);

        if (a.row <= 1) {
            ctx.fillRect(x + 8, y, 12, 4);
            ctx.fillRect(x + 4, y + 4, 20, 4);
            ctx.fillRect(x, y + 8, 28, 6);
            ctx.fillRect(x + 4, y + 14, 8, 4);
            ctx.fillRect(x + 16, y + 14, 8, 4);
            ctx.fillRect(x + 8, y + 18, 4, 4);
            ctx.fillRect(x + 16, y + 18, 4, 4);
        } else if (a.row <= 3) {
            ctx.fillRect(x + 4, y, 20, 4);
            ctx.fillRect(x, y + 4, 28, 8);
            ctx.fillRect(x + 4, y + 12, 8, 4);
            ctx.fillRect(x + 16, y + 12, 8, 4);
            ctx.fillRect(x, y + 16, 4, 6);
            ctx.fillRect(x + 24, y + 16, 4, 6);
        } else {
            ctx.fillRect(x + 8, y, 12, 4);
            ctx.fillRect(x + 4, y + 4, 20, 12);
            ctx.fillRect(x, y + 8, 4, 4);
            ctx.fillRect(x + 24, y + 8, 4, 4);
            ctx.fillRect(x + 4, y + 16, 4, 6);
            ctx.fillRect(x + 12, y + 16, 4, 4);
            ctx.fillRect(x + 20, y + 16, 4, 6);
        }
    }

    function drawShields() {
        for (const s of shields) {
            const alpha = s.hp / 3;
            ctx.fillStyle = `rgba(0, 255, 0, ${0.3 + alpha * 0.5})`;
            ctx.fillRect(s.x, s.y, s.w, s.h);
        }
    }

    function drawBullets() {
        ctx.fillStyle = '#33ff33';
        for (const b of playerBullets) ctx.fillRect(b.x, b.y, b.w, b.h);
        ctx.fillStyle = '#ff3333';
        for (const b of alienBullets) ctx.fillRect(b.x, b.y, b.w, b.h);
    }

    function drawParticles() {
        for (const p of particles) {
            const alpha = 1 - p.age / p.life;
            ctx.globalAlpha = alpha;
            ctx.fillStyle = p.color;
            ctx.fillRect(p.x, p.y, p.size, p.size);
        }
        ctx.globalAlpha = 1;
    }

    function drawGrid() {
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.03)';
        ctx.lineWidth = 1;
        for (let x = 0; x < W; x += 32) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
        }
        for (let y = 0; y < H; y += 32) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        }
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        drawGrid();

        if (state === 'title' || state === 'gameover') {
            ctx.globalAlpha = 0.3;
            for (const a of aliens) { if (a.alive) drawAlien(a); }
            ctx.globalAlpha = 1;
            return;
        }

        drawShields();
        drawPlayer();
        for (const a of aliens) { if (a.alive) drawAlien(a); }
        drawBullets();
        drawParticles();
    }

    /* ── Overlay ─────────────────────────────────────── */
    function showOverlay(title, msg, finalScore) {
        overlayTitle.textContent = title;
        overlayMsg.textContent = msg;
        if (finalScore !== undefined) {
            overlayScore.textContent = 'SCORE: ' + finalScore;
            overlayScore.style.display = '';
        } else {
            overlayScore.style.display = 'none';
        }
        overlay.classList.remove('hidden');
    }

    /* ── Loop ────────────────────────────────────────── */
    function loop(timestamp) {
        if (!lastTime) lastTime = timestamp;
        let dt = (timestamp - lastTime) / 1000;
        lastTime = timestamp;

        // clamp dt to avoid spiral of death (e.g. tab was backgrounded)
        if (dt > 0.1) dt = 0.1;

        update(dt);
        draw();
        requestAnimationFrame(loop);
    }

    /* ── Input handling for states ───────────────────── */
    document.addEventListener('keydown', function (e) {
        if (e.code === 'KeyP' && state === 'playing') {
            state = 'paused';
            showOverlay('PAUSED', 'Press P to resume');
        } else if (e.code === 'KeyP' && state === 'paused') {
            state = 'playing';
            overlay.classList.add('hidden');
        } else if (e.code === 'Space') {
            if (state === 'title' || state === 'gameover') {
                startGame();
            }
        }
    });

    /* ── Boot ────────────────────────────────────────── */
    resetPlayer();
    spawnAliens();
    showOverlay('SPACE INVADERS', 'Press SPACE to start');
    requestAnimationFrame(loop);
})();
