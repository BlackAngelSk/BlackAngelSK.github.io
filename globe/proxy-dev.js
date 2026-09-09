#!/usr/bin/env node
/**
 * proxy-dev.js — Auto-restarting proxy wrapper
 * Watches proxy.js and restarts it on changes.
 *
 * Usage:  node proxy-dev.js
 *   or:   chmod +x proxy-dev.js && ./proxy-dev.js
 */
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const PROXY_SCRIPT = path.join(__dirname, 'proxy.js');
let child = null;
let restarting = false;

function start() {
    child = spawn('node', [PROXY_SCRIPT], {
        stdio: 'inherit',
        cwd: __dirname
    });
    child.on('exit', (code) => {
        if (!restarting) {
            console.log(`[proxy-dev] proxy.js exited with code ${code}`);
        }
    });
}

function restart() {
    if (restarting) return;
    restarting = true;
    console.log('\n[proxy-dev] proxy.js changed — restarting…');
    if (child && !child.killed) {
        child.kill('SIGTERM');
        // Force kill after 2s if still alive
        setTimeout(() => { if (child && !child.killed) child.kill('SIGKILL'); }, 2000);
    }
    // Wait a moment for the port to free, then restart
    setTimeout(() => {
        restarting = false;
        start();
    }, 1500);
}

// Watch proxy.js for changes (debounce 500ms)
let debounceTimer = null;
fs.watch(PROXY_SCRIPT, { persistent: false }, () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(restart, 500);
});

console.log('[proxy-dev] Watching proxy.js for changes…');
start();
