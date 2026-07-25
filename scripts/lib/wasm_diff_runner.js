#!/usr/bin/env node
/*
 * wasm_diff_runner.js — execute one Eshkol program under the bytecode VM
 * compiled to WebAssembly, and print exactly what the program wrote to
 * stdout.
 *
 * This is the "execute" half of the WASM execute-and-diff lane
 * (scripts/run_wasm_differential.sh).  It loads the Emscripten module built
 * from lib/backend/vm_wasm_repl.c (EXPORT_NAME='EshkolVMDiff') and drives the
 * `run_program` export, which runs the source in BATCH mode — i.e. the same
 * surface as `eshkol-vm-standalone <file>` and `eshkol-run -r <file>`, with no
 * REPL auto-print of the last expression.  Program output is produced by the
 * VM's own C `display`/`write` code compiled to WASM, so the bytes captured
 * here are a genuine product of WASM execution, not a JS re-implementation of
 * Eshkol's formatting.
 *
 * Usage:  node wasm_diff_runner.js <eshkol-vm-diff.js> <program.esk>
 *   - program stdout  -> this process's stdout
 *   - diagnostics/err -> this process's stderr (prefixed markers the shell
 *                        harness greps for: WASM-RUNNER-EXCEPTION / abort)
 *   - exit code       -> 0 on a clean run, 1 if the WASM run trapped/aborted
 *
 * A fresh module instance is created per invocation, so global VM state never
 * leaks between programs (the shell runs one node process per corpus file,
 * under a timeout guard, so a runaway program cannot wedge the whole lane).
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */
'use strict';

const path = require('path');
const fs = require('fs');

if (process.argv.length < 4) {
  process.stderr.write('WASM-RUNNER-USAGE: node wasm_diff_runner.js <module.js> <program.esk>\n');
  process.exit(2);
}

const modPath = path.resolve(process.argv[2]);
const srcPath = path.resolve(process.argv[3]);

let source;
try {
  source = fs.readFileSync(srcPath, 'utf8');
} catch (e) {
  process.stderr.write('WASM-RUNNER-READ-ERROR: ' + (e && e.message ? e.message : String(e)) + '\n');
  process.exit(2);
}

let factory;
try {
  factory = require(modPath);
} catch (e) {
  process.stderr.write('WASM-RUNNER-LOAD-ERROR: ' + (e && e.message ? e.message : String(e)) + '\n');
  process.exit(2);
}

const dir = path.dirname(modPath);
const out = [];   // program stdout, one entry per emscripten line (newline-stripped)
const err = [];   // program stderr + runner diagnostics
let aborted = false;

// Emscripten line-buffers stdout: `print` is called once per complete line
// (trailing '\n' removed).  We rejoin with '\n' and add a terminating '\n';
// the shell harness strips ALL newlines before comparison (the VM's documented
// display-per-call-newline quirk), so newline placement is not load-bearing —
// only the non-newline content bytes (digits, spaces, text) are compared.
const moduleArgs = {
  print: (s) => out.push(s),
  printErr: (s) => err.push(s),
  locateFile: (p) => path.join(dir, p),
  // Keep the runtime alive after run_program returns so we can fflush; and
  // trap abort() instead of letting it call process.exit and lose captured
  // output.
  noExitRuntime: true,
  onAbort: (what) => { aborted = true; err.push('WASM-RUNNER-ABORT: ' + String(what)); },
  quit: (code, toThrow) => { aborted = aborted || code !== 0; if (toThrow) throw toThrow; },
};

factory(moduleArgs).then((mod) => {
  try {
    mod.ccall('run_program', null, ['string'], [source]);
    // Force any partial (non-newline-terminated) trailing line out of the
    // TTY buffer so it reaches `print`.
    try { mod.ccall('fflush', 'number', ['number'], [0]); } catch (_) { /* fflush optional */ }
  } catch (e) {
    aborted = true;
    err.push('WASM-RUNNER-EXCEPTION: ' + (e && e.message ? e.message : String(e)));
  }
  if (out.length) process.stdout.write(out.join('\n') + '\n');
  if (err.length) process.stderr.write(err.join('\n') + '\n');
  process.exit(aborted ? 1 : 0);
}).catch((e) => {
  process.stderr.write('WASM-RUNNER-FATAL: ' + (e && e.message ? e.message : String(e)) + '\n');
  if (out.length) process.stdout.write(out.join('\n') + '\n');
  process.exit(1);
});
