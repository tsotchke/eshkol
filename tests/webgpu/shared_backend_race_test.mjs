#!/usr/bin/env node

/* Two independent browser VMs dispatch through one backend concurrently in
 * the 1x1 and 8x8 cases. Distinct memories expose state leaking between
 * suspended calls; every result cell and per-case telemetry are checked. */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const WEBGPU = fs.readFileSync(path.join(ROOT, 'web', 'eshkol-webgpu.js'), 'utf8');
const VM_JS = fs.readFileSync(path.join(ROOT, 'site', 'static', 'eshkol-vm.js'));
const VM_WASM = fs.readFileSync(path.join(ROOT, 'site', 'static', 'eshkol-vm.wasm'));
let chromium;
try { ({ chromium } = await import('playwright')); }
catch {
    try { ({ chromium } = await import(path.join(ROOT, '.scratch', 'node_modules', 'playwright', 'index.js'))); }
    catch (e) { console.error('shared_backend_race_test: Playwright unavailable: ' + e); process.exit(2); }
}

const server = http.createServer((req, res) => {
    if (req.url === '/eshkol-webgpu.js') { res.writeHead(200, { 'Content-Type': 'text/javascript' }); res.end(WEBGPU); return; }
    if (req.url === '/eshkol-vm.js') { res.writeHead(200, { 'Content-Type': 'text/javascript' }); res.end(VM_JS); return; }
    if (req.url === '/eshkol-vm.wasm') { res.writeHead(200, { 'Content-Type': 'application/wasm' }); res.end(VM_WASM); return; }
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<!doctype html><script src="/eshkol-webgpu.js"></script><script src="/eshkol-vm.js"></script>');
});
await new Promise((resolve) => server.listen(0, 'localhost', resolve));
let browser;
try {
    browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
    const page = await browser.newPage();
    await page.goto(`http://localhost:${server.address().port}/`);
    const result = await page.evaluate(async () => {
        const G = globalThis.EshkolWebGPU;
        const created = await G.create({ threshold: 1 });
        if (!created.ok) throw new Error(created.reason || 'WebGPU unavailable');
        const backend = created.backend;
        const makeVm = async () => {
            let out = '';
            const arg = {
                stdout: (b) => { if (b != null) out += new TextDecoder().decode(Uint8Array.of(b)); },
                stderr: () => {}, print: (s) => { out += String(s) + '\n'; }, printErr: () => {},
                wasmBinary: undefined
            };
            G.attachVm(arg, backend, { wasmUrl: '/eshkol-vm.wasm' });
            const vm = await EshkolVM(arg);
            vm.ccall('repl_init', null, [], []);
            return { vm, get out() { return out; } };
        };
        const one = (a, b) => `(display (tensor-get (matmul (reshape (list->vector (list ${a}.0)) 1 1) (reshape (list->vector (list ${b}.0)) 1 1)) 0 0)) (newline)`;
        const eight = (a, b) => `(define A (reshape (make-vector 64 ${a}.0) 8 8))
(define B (reshape (make-vector 64 ${b}.0) 8 8))
(define C (matmul A B))
(define (show t r c)
  (let loop ((i 0))
    (if (< i r)
        (begin
          (let lj ((j 0))
            (if (< j c) (begin (display (tensor-get t i j)) (display " ") (lj (+ j 1)))))
          (newline) (loop (+ i 1))))))
(show C 8 8)`;
        const runPair = async (size, programs, expected) => {
            const vms = await Promise.all([makeVm(), makeVm()]);
            const before = { dispatch: backend.dispatchCount, history: backend.dispatchHistory.length,
                             fallback: backend.fallbackCount, diagnostics: backend.diagnostics.length };
            await Promise.all(vms.map((v, i) => G.vmCall(v.vm, 'repl_eval', programs[i])));
            const outputs = vms.map((v) => v.out.trim());
            const values = outputs.map((out) => out.split(/\s+/).filter(Boolean));
            const allExpected = values.every((xs, i) => xs.length === expected.count &&
                xs.every((x) => Number(x) === expected.values[i]));
            return { size, outputs: values, allExpected,
                dispatchDelta: backend.dispatchCount - before.dispatch,
                historyDelta: backend.dispatchHistory.length - before.history,
                fallbackDelta: backend.fallbackCount - before.fallback,
                diagnosticsDelta: backend.diagnostics.length - before.diagnostics };
        };
        const small = await runPair('1x1', [one(2, 3), one(4, 5)], { values: [6, 20], count: 1 });
        const large = await runPair('8x8', [eight(2, 3), eight(4, 5)], { values: [48, 160], count: 64 });
        return { small, large, dispatchCount: backend.dispatchCount,
                 history: backend.dispatchHistory.length, fallbackCount: backend.fallbackCount,
                 diagnostics: backend.diagnostics };
    });
    const pass = result.small.allExpected && result.large.allExpected &&
        result.small.outputs[0][0] === '6' && result.small.outputs[1][0] === '20' &&
        result.large.outputs[0].every((x) => Number(x) === 48) &&
        result.large.outputs[1].every((x) => Number(x) === 160) &&
        result.small.dispatchDelta === 2 && result.large.dispatchDelta === 2 &&
        result.small.historyDelta === 2 && result.large.historyDelta === 2 &&
        result.small.fallbackDelta === 0 && result.large.fallbackDelta === 0 &&
        result.small.diagnosticsDelta === 0 && result.large.diagnosticsDelta === 0 &&
        result.dispatchCount === 4 && result.history === 4 && result.fallbackCount === 0 &&
        result.diagnostics.length === 0;
    console.log((pass ? 'PASS' : 'FAIL') + ' shared backend concurrent VM race ' + JSON.stringify(result));
    if (!pass) process.exitCode = 1;
} finally {
    if (browser) await browser.close();
    server.close();
}
