#!/usr/bin/env node
/* Cross-check IEEE tensor division in an unattached browser VM and in the
 * same bundle with WebGPU eligible or excluded by its dispatch threshold. */
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const staticDir = path.join(root, 'site/static');
const program = fs.readFileSync(path.join(root, 'tests/numeric/tensor_division_ieee_test.esk'), 'utf8');
let chromium;
try { ({ chromium } = await import('playwright')); }
catch { ({ chromium } = await import(path.join(root, '.scratch/node_modules/playwright/index.js'))); }

const server = http.createServer((req, res) => {
    const name = path.basename(req.url.split('?')[0]);
    if (['eshkol-vm.js', 'eshkol-vm.wasm', 'eshkol-webgpu.js'].includes(name)) {
        res.writeHead(200, { 'Content-Type': name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript' });
        res.end(fs.readFileSync(path.join(staticDir, name)));
    } else {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end('<script src="/eshkol-webgpu.js"></script><script src="/eshkol-vm.js"></script>');
    }
});
await new Promise(resolve => server.listen(0, 'localhost', resolve));
let browser;
try {
    browser = await chromium.launch({ channel: 'chrome', headless: true,
                                      args: ['--enable-unsafe-webgpu'] });
    async function run(threshold) {
        const page = await browser.newPage();
        try {
            await page.goto(`http://localhost:${server.address().port}/`);
            return await page.evaluate(async ({ threshold, program }) => {
                const G = globalThis.EshkolWebGPU;
                const made = threshold === null ? null : await G.create({ threshold });
                const backend = made?.ok ? made.backend : null;
                let out = '', errors = '';
                const decoder = new TextDecoder();
                const arg = {
                    stdout: b => { if (b != null) out += decoder.decode(Uint8Array.of(b), { stream: true }); },
                    stderr: b => { if (b != null) errors += String.fromCharCode(b); },
                    print: s => { out += s + '\n'; },
                    printErr: s => { errors += s + '\n'; }
                };
                if (threshold !== null) G.attachVm(arg, backend, { wasmUrl: '/eshkol-vm.wasm' });
                const vm = await EshkolVM(arg);
                vm.ccall('repl_init', null, [], []);
                await G.vmCall(vm, 'repl_eval', program);
                return { out, errors, status: arg.eshkolWebGPUStatus,
                         dispatches: backend?.dispatchCount || 0 };
            }, { threshold, program });
        } finally { await page.close(); }
    }
    const cpu = await run(null);
    const gpu = await run(1);
    const belowThreshold = await run(1000000);
    assert.equal(cpu.errors, '');
    assert.equal(gpu.errors, '');
    assert.equal(belowThreshold.errors, '');
    assert.ok(gpu.status?.ok, JSON.stringify(gpu.status));
    assert.ok(belowThreshold.status?.ok, JSON.stringify(belowThreshold.status));
    assert.ok(gpu.dispatches >= 2, `dispatches=${gpu.dispatches}`);
    assert.equal(belowThreshold.dispatches, 0);
    assert.equal(gpu.out, cpu.out);
    assert.equal(belowThreshold.out, cpu.out);
    assert.match(cpu.out, /\+inf\.0/);
    assert.match(cpu.out, /-inf\.0/);
    assert.match(cpu.out, /\+nan\.0/);
    console.log(`PASS division parity: CPU/GPU/high-threshold, ${gpu.dispatches} dispatches`);
} finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
}
