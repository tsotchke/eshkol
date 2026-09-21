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
 *   - exit code       -> the program's own exit status when it calls
 *                        `(exit N)` (0 included — a clean explicit exit is
 *                        NOT a failure), 0 on a normal fall-off-the-end run,
 *                        or 1 if the WASM run genuinely trapped/aborted.
 *
 * `(exit N)` inside the VM compiles to libc `exit()`, which Emscripten
 * implements by unwinding the C stack back into JS via a thrown `ExitStatus`
 * exception (`e.name === 'ExitStatus'`, `e.status === N`) — this is
 * Emscripten's normal, documented mechanism for a clean exit, not a crash.
 * `run_program`'s ccall therefore throws on every explicit `(exit N)`, exit
 * 0 included; that exception is caught below and its `status` becomes this
 * process's exit code, exactly as native `exit(N)` sets the OS return code.
 * Anything else thrown (a real trap/abort) still exits 1 and is reported as
 * WASM-RUNNER-EXCEPTION, matching the pre-existing failure contract.
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
const repoRoot = path.resolve(path.dirname(srcPath), '../../..');
const out = [];   // Emscripten's complete-line fallback
const stdoutBytes = []; // Exact byte stream, including non-newline display output
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
  // The default Emscripten print bridge is line buffered. Eshkol's display is
  // allowed to leave a line unterminated, so collect FS stdout bytes directly
  // and use them in preference to the line callback below.
  stdout: (byte) => { if (byte !== undefined && byte !== null) stdoutBytes.push(byte); },
  locateFile: (p) => path.join(dir, p),
  // Keep the runtime alive after run_program returns so we can fflush; and
  // trap abort() instead of letting it call process.exit and lose captured
  // output.
  noExitRuntime: true,
  onAbort: (what) => { aborted = true; err.push('WASM-RUNNER-ABORT: ' + String(what)); },
  quit: (code, toThrow) => { aborted = aborted || code !== 0; if (toThrow) throw toThrow; },
};

factory(moduleArgs).then((mod) => {
  let exitCode = 0;
  try {
    // The differential lane exercises R7RS module imports. The product WASM
    // image remains filesystem-free; this test-only module gets the small
    // fixture library through MEMFS before compiling the test program.
    const libraryDir = path.join(repoRoot, 'lib');
    if (fs.existsSync(libraryDir) && mod.FS) {
      const stageLibrary = (hostDir, wasmDir) => {
        mod.FS.mkdirTree(wasmDir);
        for (const entry of fs.readdirSync(hostDir, { withFileTypes: true })) {
          const hostPath = path.join(hostDir, entry.name);
          const wasmPath = `${wasmDir}/${entry.name}`;
          if (entry.isDirectory()) stageLibrary(hostPath, wasmPath);
          else if (entry.isFile() && entry.name.endsWith('.esk')) {
            mod.FS.writeFile(wasmPath, fs.readFileSync(hostPath));
          }
        }
      };
      stageLibrary(libraryDir, '/lib');
    }
    mod.ccall('run_program', null, ['string'], [source]);
  } catch (e) {
    if (e && e.name === 'ExitStatus' && typeof e.status === 'number') {
      // A clean `(exit N)` from the Eshkol program, N == 0 included — this
      // is Emscripten's normal unwind for libc exit(), not a crash. Mirror
      // the status as our own process exit code so it is comparable to
      // native's return code, exactly like `eshkol-run -r`'s $?.
      exitCode = e.status;
    } else {
      aborted = true;
      err.push('WASM-RUNNER-EXCEPTION: ' + (e && e.message ? e.message : String(e)));
    }
  }
  // Force any partial (non-newline-terminated) trailing line out of the TTY
  // buffer so it reaches `print` — attempted even after a caught exit(), in
  // case the runtime still holds a buffered line (libc's own exit() already
  // flushes stdio via atexit, so this is normally a no-op belt-and-braces).
  try { mod.ccall('fflush', 'number', ['number'], [0]); } catch (_) { /* fflush optional */ }
  if (stdoutBytes.length) process.stdout.write(Buffer.from(stdoutBytes));
  else if (out.length) process.stdout.write(out.join('\n') + '\n');
  if (err.length) process.stderr.write(err.join('\n') + '\n');
  process.exit(aborted ? 1 : exitCode);
}).catch((e) => {
  process.stderr.write('WASM-RUNNER-FATAL: ' + (e && e.message ? e.message : String(e)) + '\n');
  if (stdoutBytes.length) process.stdout.write(Buffer.from(stdoutBytes));
  else if (out.length) process.stdout.write(out.join('\n') + '\n');
  process.exit(1);
});
