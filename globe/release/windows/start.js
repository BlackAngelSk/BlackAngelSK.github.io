#!/usr/bin/env node
/**
 * start.js — thin launcher, kept so `node start.js` still works.
 *
 * Everything now lives in globe-server.js: plain HTTP, no certificate,
 * nothing to accept. The arguments are passed straight through:
 *
 *   node start.js                 proxy + static server
 *   node start.js --proxy         proxy only
 *   node start.js --no-open       do not open the browser
 *   node start.js --https         also listen on https://localhost:8443
 *   node start.js --port 3000     custom static-server port
 */
require('./globe-server.js');
