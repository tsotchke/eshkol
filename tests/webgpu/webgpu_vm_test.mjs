#!/usr/bin/env node

/* Browser-VM WebGPU differential gate (ADR-0029).
 *
 * Loads the shipped browser VM bundle (site/static/eshkol-vm.{js,wasm}) in
 * Chrome and runs tests/webgpu/vm_tensor_ops.esk four ways:
 *
 *   gpu    : WebGPU + JSPI, attached with EshkolWebGPU.attachVm (threshold 1,
 *            so every tensor op is offered to the GPU). The VM's tensor
 *            natives must reach the GPU seam: two matmuls, the four elementwise
 *            ops and the four full reductions are GPU dispatches; softmax has
 *            no kernel and is counted as an explicit CPU fallback.
 *   cpu    : the same bundle, not attached -- the CPU reference.
 *   nogpu  : navigator.gpu hidden -- attachVm reports why, VM runs on CPU.
 *   nojspi : WebAssembly.Suspending hidden -- attachVm reports why.
 *
 * Every matmul/elementwise line must be byte-identical to the CPU run
 * (display prints the shortest round-trip decimal, so equal text is equal
 * bits). Reductions must agree within GPU_GATE_TOL (block reassociation).
 * A missing Chrome/WebGPU device is a failure, not a skip.
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const STATIC = path.join(ROOT, 'site', 'static');
const PROGRAM = fs.readFileSync(path.join(ROOT, 'tests', 'webgpu', 'vm_tensor_ops.esk'), 'utf8');
const GPU_GATE_TOL = Number(process.env.GPU_GATE_TOL || '1e-9');
/* --corrupt: serve a WebGPU module whose sf64 ADD computes a - b, to prove
 * this gate goes red when a kernel the VM reaches is wrong. */
const CORRUPT = process.argv.includes('--corrupt');
let WEBGPU_SRC = fs.readFileSync(path.join(STATIC, 'eshkol-webgpu.js'), 'utf8');
if (CORRUPT) {
    const from = 'case 0u:  { r = f64_add(a, b); }';
    if (!WEBGPU_SRC.includes(from)) { console.error('corruption pattern not found'); process.exit(2); }
    WEBGPU_SRC = WEBGPU_SRC.replace(from, 'case 0u:  { r = f64_add(a, f64_neg(b)); }');
}

let chromium;
try {
    ({ chromium } = await import('playwright'));
} catch {
    try {
        ({ chromium } = await import(path.join(ROOT, '.scratch', 'node_modules',
                                               'playwright', 'index.js')));
    } catch (error) {
        console.error('webgpu_vm_test: Playwright is unavailable: ' + error);
        process.exit(2);
    }
}

const TYPES = { '.js': 'text/javascript', '.wasm': 'application/wasm' };
const server = http.createServer((req, res) => {
    const name = path.basename(req.url.split('?')[0]);
    if (['eshkol-vm.js', 'eshkol-vm.wasm', 'eshkol-webgpu.js'].includes(name)) {
        res.writeHead(200, { 'Content-Type': TYPES[path.extname(name)] });
        res.end(name === 'eshkol-webgpu.js' ? WEBGPU_SRC : fs.readFileSync(path.join(STATIC, name)));
        return;
    }
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<!doctype html><body><script src="/eshkol-webgpu.js"></script>' +
            '<script src="/eshkol-vm.js"></script></body>');
});
await new Promise((resolve) => server.listen(0, 'localhost', resolve));
const port = server.address().port;

async function run(page, mode) {
    return page.evaluate(async ({ mode, program }) => {
        const G = globalThis.EshkolWebGPU;
        let backend = null, created = null;
        if (mode !== 'cpu') {
            created = await G.create({ threshold: 1 });
            backend = created.ok ? created.backend : null;
        }
        let out = '';
        const decoder = new TextDecoder();
        const arg = {
            stdout: (b) => { if (b != null) out += decoder.decode(Uint8Array.of(b), { stream: true }); },
            stderr: () => {},
            print: (t) => { out += t + '\n'; },
            printErr: () => {}
        };
        if (mode !== 'cpu') G.attachVm(arg, backend, { wasmUrl: '/eshkol-vm.wasm' });
        const vm = await EshkolVM(arg);
        vm.ccall('repl_init', null, [], []);
        await G.vmCall(vm, 'repl_eval', program);
        return {
            out,
            created: created ? { ok: created.ok, reason: created.reason || '' } : null,
            status: arg.eshkolWebGPUStatus || null,
            dispatchCount: backend ? backend.dispatchCount : 0,
            fallbackCount: backend ? backend.fallbackCount : 0,
            paths: backend ? backend.dispatchHistory.length : 0,
            diagnostics: backend ? backend.diagnostics.slice() : []
        };
    }, { mode, program: PROGRAM });
}

let failed = 0;
const check = (ok, label, detail) => {
    console.log((ok ? 'PASSED ' : 'FAILED ') + label + (detail ? ' - ' + detail : ''));
    if (!ok) failed++;
};

let browser;
try {
    browser = await chromium.launch({ channel: 'chrome', headless: true,
                                      args: ['--enable-unsafe-webgpu'] });
    const page = async (init) => {
        const p = await browser.newPage();
        if (init) await p.addInitScript(init);
        await p.goto(`http://localhost:${port}/`);
        return p;
    };

    const cpu = await run(await page(), 'cpu');
    const gpu = await run(await page(), 'gpu');

    check(gpu.status && gpu.status.ok, 'webgpu_vm/gpu/attached',
          gpu.status ? gpu.status.reason : 'no status');

    const cpuLines = cpu.out.trimEnd().split('\n');
    const gpuLines = gpu.out.trimEnd().split('\n');
    check(cpuLines.length === 24 + 4 * 24 + 6 && gpuLines.length === cpuLines.length,
          'webgpu_vm/output_shape', `cpu=${cpuLines.length} gpu=${gpuLines.length} lines`);

    /* matmul (24 rows) + add/sub/mul/div (4 x 24 rows): bit-identical. */
    const exactRows = 24 + 4 * 24;
    let firstDiff = -1;
    for (let i = 0; i < exactRows; i++) if (cpuLines[i] !== gpuLines[i]) { firstDiff = i; break; }
    check(firstDiff < 0, 'webgpu_vm/matmul_elementwise_bit_identical',
          firstDiff < 0 ? `${exactRows} rows` : `row ${firstDiff}: ${gpuLines[firstDiff]?.slice(0, 80)} vs ${cpuLines[firstDiff]?.slice(0, 80)}`);

    /* sum, mean, max, min: within GPU_GATE_TOL. softmax (CPU both ways): identical. */
    const num = (s) => Number(String(s).replace(/[#()]/g, ''));
    ['sum', 'mean', 'max', 'min'].forEach((name, k) => {
        const c = num(cpuLines[exactRows + k]), g = num(gpuLines[exactRows + k]);
        const rel = Math.abs(g - c) / Math.max(Math.abs(c), 1e-300);
        check(Number.isFinite(c) && rel <= GPU_GATE_TOL, `webgpu_vm/reduce_${name}`,
              `gpu=${g} cpu=${c} rel=${rel.toExponential(2)}`);
    });
    check(gpuLines[exactRows + 4] === cpuLines[exactRows + 4], 'webgpu_vm/softmax_cpu_identical');
    check(/^\(caught /.test(gpuLines[exactRows + 5]) && gpuLines[exactRows + 5] === cpuLines[exactRows + 5],
          'webgpu_vm/guard_after_suspension', gpuLines[exactRows + 5]);

    /* Non-vacuity: 2 matmuls + 4 elementwise + 4 reductions ran on the GPU;
     * softmax was refused explicitly and says so. */
    check(gpu.dispatchCount === 10, 'webgpu_vm/non_vacuity',
          `dispatchCount=${gpu.dispatchCount} fallbackCount=${gpu.fallbackCount}`);
    check(gpu.diagnostics.some((d) => /softmax has no WebGPU kernel/.test(d)) && gpu.fallbackCount >= 1,
          'webgpu_vm/softmax_explicit_fallback', gpu.diagnostics.join(' | ').slice(0, 200));

    const nogpu = await run(await page(() => {
        Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true });
    }), 'gpu');
    check(nogpu.status && !nogpu.status.ok && /no WebGPU backend/.test(nogpu.status.reason) &&
          nogpu.created && /navigator\.gpu unavailable/.test(nogpu.created.reason),
          'webgpu_vm/no_webgpu/explicit_status',
          `create="${nogpu.created?.reason}" attach="${nogpu.status?.reason}"`);
    check(nogpu.out === cpu.out, 'webgpu_vm/no_webgpu/cpu_identical');

    const nojspi = await run(await page(() => { delete WebAssembly.Suspending; }), 'gpu');
    check(nojspi.status && !nojspi.status.ok && /JSPI unavailable/.test(nojspi.status.reason),
          'webgpu_vm/no_jspi/explicit_status', nojspi.status?.reason);
    check(nojspi.out === cpu.out && nojspi.dispatchCount === 0, 'webgpu_vm/no_jspi/cpu_identical',
          `dispatchCount=${nojspi.dispatchCount}`);
} finally {
    if (browser) await browser.close();
    server.close();
}
console.log(failed === 0 ? 'PASS webgpu browser-VM dispatch' : `FAIL ${failed} check(s)`);
process.exit(failed === 0 ? 0 : 1);
