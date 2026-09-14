#!/usr/bin/env node
/*
 * wasm_repl_runner.js — drive the browser REPL surface (`repl_eval`) of the
 * bytecode VM compiled to WebAssembly, and report what a LINE-ORIENTED host
 * actually received.
 *
 * Why this exists as its own runner
 * ---------------------------------
 * The execute-and-diff lane (wasm_diff_runner.js) drives `run_program`, the
 * BATCH entry point.  That covers program output but never touches
 * `repl_eval`, the entry point the website's REPL and every runnable code
 * block on the docs pages actually call — so a REPL-only regression passes
 * the whole of CI unseen.  One did: the REPL's auto-print of the last
 * expression rode on OP_PRINT, OP_PRINT was corrected to be `display` exactly
 * (no trailing newline), and every REPL answer became an unterminated
 * fragment.  `run_program` was unaffected, so nothing failed.
 *
 * The property this runner gates is the one that broke: Emscripten delivers
 * stdout to the embedder's `print` callback ONE COMPLETE LINE AT A TIME.  An
 * answer that is not newline-terminated is buffered indefinitely and the page
 * sees nothing at all.  So this runner installs exactly the site's callback
 * shape (site/static/index.html accumulates `text + '\n'` per call) and
 * reports, per expression, only what arrived as finished lines.  A missing
 * terminator therefore shows up as EMPTY output — the browser symptom —
 * rather than as text that a terminal would have shown anyway.
 *
 * Usage:  node wasm_repl_runner.js <module.js> <cases.tsv>
 *
 *   cases.tsv   one case per line: <expression><TAB><ignored…>
 *               (blank lines and #-comments skipped; the caller keeps the
 *               expected column and does the comparing)
 *
 * Output: one line per case on stdout
 *
 *   GOT<TAB><index><TAB><lines joined by the two characters \ and n>
 *
 * so a multi-line answer stays on one record and an EMPTY answer is visibly
 * empty.  Stderr the VM produced is reported separately as
 *
 *   ERR<TAB><index><TAB><escaped stderr>
 *
 * A single module instance serves every case on purpose: REPL sessions are
 * persistent, and `(define …)` in one case must still be in scope in the
 * next — that persistence is itself part of the surface under test.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */
'use strict';

const path = require('path');
const fs = require('fs');

if (process.argv.length < 4) {
  process.stderr.write('WASM-RUNNER-USAGE: node wasm_repl_runner.js <module.js> <cases.tsv>\n');
  process.exit(2);
}

const modPath = path.resolve(process.argv[2]);
const casesPath = path.resolve(process.argv[3]);

let cases;
try {
  cases = fs.readFileSync(casesPath, 'utf8')
    .split('\n')
    .filter((l) => l.trim() !== '' && !l.startsWith('#'))
    .map((l) => l.split('\t')[0]);
} catch (e) {
  process.stderr.write('WASM-RUNNER-READ-ERROR: ' + (e && e.message ? e.message : String(e)) + '\n');
  process.exit(2);
}

const escape = (s) => s.replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/\r/g, '\\r').replace(/\t/g, '\\t');

let outLines = [];
let errLines = [];

const Factory = require(modPath);

Factory({
  // Exactly the site's shape: `print` fires once per COMPLETE line.
  print: (text) => { outLines.push(text); },
  printErr: (text) => { errLines.push(text); },
}).then((mod) => {
  const replInit = mod.cwrap('repl_init', null, []);
  const replEval = mod.cwrap('repl_eval', 'string', ['string']);

  replInit();
  if (errLines.length) {
    process.stderr.write('WASM-RUNNER-FATAL: repl_init wrote to stderr: ' + escape(errLines.join('\n')) + '\n');
    process.exit(1);
  }

  for (let i = 0; i < cases.length; i++) {
    outLines = [];
    errLines = [];
    let ret;
    try {
      ret = replEval(cases[i]);
    } catch (e) {
      process.stderr.write('WASM-RUNNER-EXCEPTION: case ' + i + ': ' + (e && e.message ? e.message : String(e)) + '\n');
      process.exit(1);
    }
    if (ret) {
      process.stderr.write('WASM-RUNNER-FATAL: case ' + i + ': repl_eval returned "' + ret + '"\n');
      process.exit(1);
    }
    process.stdout.write('GOT\t' + i + '\t' + escape(outLines.join('\n')) + '\n');
    if (errLines.length) process.stdout.write('ERR\t' + i + '\t' + escape(errLines.join('\n')) + '\n');
  }
  process.exit(0);
}).catch((e) => {
  process.stderr.write('WASM-RUNNER-EXCEPTION: module load: ' + (e && e.message ? e.message : String(e)) + '\n');
  process.exit(1);
});
