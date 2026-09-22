#!/usr/bin/env node

/* Live browser contract for the WebGPU bridge. This deliberately uses the
 * installed Chrome channel and a real WebGPU device; a missing device is a
 * test failure, not a green CPU-only result. */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const SOURCE = fs.readFileSync(path.join(ROOT, 'web', 'eshkol-webgpu.js'), 'utf8');
const RUNTIME = fs.readFileSync(path.join(ROOT, 'site', 'static', 'eshkol-runtime.js'), 'utf8');
let chromium;
try {
    ({ chromium } = await import('playwright'));
} catch {
    try {
        ({ chromium } = await import(path.join(ROOT, '.scratch', 'node_modules',
                                               'playwright', 'index.js')));
    } catch (error) {
        console.error('webgpu_live_test: Playwright is unavailable: ' + error);
        process.exit(2);
    }
}

const server = http.createServer((req, res) => {
    if (req.url === '/eshkol-webgpu.js' || req.url === '/eshkol-runtime.js') {
        res.writeHead(200, { 'Content-Type': 'text/javascript' });
        res.end(req.url === '/eshkol-webgpu.js' ? SOURCE : RUNTIME);
        return;
    }
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<!doctype html><body><script src="/eshkol-webgpu.js"></script>' +
            '<script src="/eshkol-runtime.js"></script></body>');
});

await new Promise((resolve) => server.listen(0, 'localhost', resolve));
const port = server.address().port;
let browser;
try {
    browser = await chromium.launch({ channel: 'chrome', headless: true,
                                      args: ['--enable-unsafe-webgpu'] });
    const page = await browser.newPage();
    await page.goto(`http://localhost:${port}/`);
    const result = await page.evaluate(async () => {
        const assert = (condition, message) => {
            if (!condition) throw new Error(message);
        };
        assert.equal = (actual, expected, message) => assert(
            Object.is(actual, expected), message || `${actual} !== ${expected}`);
        const G = globalThis.EshkolWebGPU;
        assert(G, 'EshkolWebGPU did not load');
        assert(navigator.gpu, 'navigator.gpu is unavailable');

        /* Default tier: exact (sf64), the same default as native backends. */
        const created = await G.create({ threshold: 1 });
        assert(created.ok, created.reason || 'WebGPU initialization failed');
        const backend = created.backend;
        assert.equal(backend.precision, 'exact');
        assert(backend.hasFp64(), 'exact tier must report an f64 path');
        const limit = backend.maxComputeWorkgroupsPerDimension;
        assert(Number.isSafeInteger(limit) && limit > 0,
               'device workgroup limit was not captured');

        const memory = new WebAssembly.Memory({ initial: 256 });
        backend.setMemory(memory);
        const fakeMem = { buffer: memory.buffer };
        let bump = 64;
        const alloc = (count) => {
            const ptr = bump;
            bump = (bump + count * 8 + 15) & ~15;
            return ptr;
        };

        async function gemm(M, K, N, aValue, bValue) {
            bump = 64;
            const aPtr = alloc(M * K), bPtr = alloc(K * N);
            const gpuPtr = alloc(M * N), cpuPtr = alloc(M * N);
            const A = new Float64Array(memory.buffer, aPtr, M * K);
            const B = new Float64Array(memory.buffer, bPtr, K * N);
            A.set(Array.from({ length: M * K }, (_, i) => aValue(i)));
            B.set(Array.from({ length: K * N }, (_, i) => bValue(i)));
            const before = backend.dispatchHistory.length;
            await backend.matmulF64(aPtr, bPtr, gpuPtr, M, K, N);
            G.cpu.matmul(fakeMem, aPtr, bPtr, cpuPtr, M, K, N);
            const gpu = new Float64Array(memory.buffer, gpuPtr, M * N);
            const cpu = new Float64Array(memory.buffer, cpuPtr, M * N);
            for (let i = 0; i < gpu.length; i++) {
                assert(Object.is(gpu[i], cpu[i]),
                       `GEMM mismatch at ${i}: ${gpu[i]} !== ${cpu[i]}`);
            }
            return backend.dispatchHistory.slice(before);
        }

        const small = await gemm(8, 8, 8, () => 1, () => 1);
        /* Non-integer operands: only a true f64 path is bit-identical. */
        const nonsquare = await gemm(3, 5, 7, (i) => ((i % 5) - 2) / 3,
                                     (i) => ((i % 7) - 3) / 7 + 1e-9);

        async function boundary(N) {
            bump = 64;
            const aPtr = alloc(1), bPtr = alloc(N), gpuPtr = alloc(N);
            new Float64Array(memory.buffer, aPtr, 1)[0] = 1;
            new Float64Array(memory.buffer, bPtr, N).fill(1);
            const before = backend.dispatchHistory.length;
            await backend.matmulF64(aPtr, bPtr, gpuPtr, 1, 1, N);
            const output = new Float64Array(memory.buffer, gpuPtr, N);
            for (const value of output) assert.equal(value, 1);
            return backend.dispatchHistory.slice(before);
        }

        const atLimit = await boundary((limit * 8));
        const overLimit = await boundary((limit * 8) + 1);
        assert.equal(atLimit.length, 1);
        assert.equal(atLimit[0].x, limit);
        assert.equal(overLimit.length, 2);
        assert.equal(overLimit[0].x, limit);
        assert.equal(overLimit[1].x, 1);
        for (const dispatch of [...small, ...nonsquare, ...atLimit, ...overLimit]) {
            assert(dispatch.x <= limit && dispatch.y <= limit && dispatch.z <= limit,
                   `oversized dispatch: ${JSON.stringify(dispatch)}`);
        }

        /* A real WASM table entry reaches a suspending import and is then
         * wrapped at its JavaScript callback boundary. */
        const wasm = new Uint8Array([
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
            0x02, 0x0f, 0x01, 0x03, 0x65, 0x6e, 0x76, 0x07, 0x73,
            0x75, 0x73, 0x70, 0x65, 0x6e, 0x64, 0x00, 0x00,
            0x03, 0x02, 0x01, 0x00,
            0x04, 0x04, 0x01, 0x70, 0x00, 0x01,
            0x07, 0x14, 0x02, 0x05, 0x74, 0x61, 0x62, 0x6c, 0x65,
            0x01, 0x00, 0x08, 0x63, 0x61, 0x6c, 0x6c, 0x62, 0x61,
            0x63, 0x6b, 0x00, 0x01,
            0x09, 0x07, 0x01, 0x00, 0x41, 0x00, 0x0b, 0x01, 0x01,
            0x0a, 0x06, 0x01, 0x04, 0x00, 0x10, 0x00, 0x0b
        ]);
        const imported = new WebAssembly.Suspending(async () => 7);
        const instance = await WebAssembly.instantiate(wasm, { env: { suspend: imported } });
        const callback = G.promisingEntry(instance.instance.exports.table.get(0));
        assert.equal(await callback(), 7);

        return {
            limit,
            tier: backend.precision,
            smallDispatches: small.length,
            nonsquareDispatches: nonsquare.length,
            boundaryDispatches: `${atLimit.length}/${overLimit.length}`,
            callback: 7,
            dispatchCount: backend.dispatchCount,
            diagnostics: backend.diagnostics
        };
    });
    console.log('LIVE WebGPU initialization complete tier=' + result.tier +
                ' maxComputeWorkgroupsPerDimension=' + result.limit);
    console.log('LIVE GEMM 8x8 exact CPU reference dispatches=' + result.smallDispatches);
    console.log('LIVE GEMM 3x5*5x7 fractional, bit-identical to CPU reference dispatches=' + result.nonsquareDispatches);
    console.log('LIVE dispatch boundary 65535/65536 workgroups=' + result.boundaryDispatches);
    console.log('LIVE JSPI table callback suspension result=' + result.callback);

    /* The same Chrome with WebGPU hidden: the runtime must say why it is on
     * the CPU, report ESHKOL_GPU_NONE, and still compute the right answer. */
    const cpuPage = await browser.newPage();
    await cpuPage.addInitScript(() => {
        Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true });
    });
    await cpuPage.goto(`http://localhost:${port}/`);
    const cpu = await cpuPage.evaluate(async () => {
        const rt = new EshkolRuntime();
        const status = await rt.initWebGPU({ threshold: 1 });
        const env = rt.createImports().env;
        const memory = new WebAssembly.Memory({ initial: 1 });
        rt.memory = memory;
        const A = new Float64Array(memory.buffer, 0, 4);
        const B = new Float64Array(memory.buffer, 32, 4);
        A.set([1.5, -2, 0.25, 3]);
        B.set([2, 0.5, -1, 4]);
        await env.eshkol_matmul_dispatch(0, 32, 64, 2, 2, 2, 0);
        return { ok: status.ok, reason: status.reason,
                 backend: env.eshkol_gpu_get_backend(),
                 shouldUse: env.eshkol_gpu_should_use(1000000),
                 C: Array.from(new Float64Array(memory.buffer, 64, 4)),
                 backendObject: rt.webgpuBackend };
    });
    if (cpu.ok !== false || !/navigator\.gpu unavailable/.test(cpu.reason)) {
        throw new Error('no-WebGPU status is not explicit: ' + JSON.stringify(cpu));
    }
    if (cpu.backend !== 0 || cpu.shouldUse !== 0 || cpu.backendObject !== null) {
        throw new Error('no-WebGPU run still reports a GPU backend: ' + JSON.stringify(cpu));
    }
    const want = [1.5 * 2 + -2 * -1, 1.5 * 0.5 + -2 * 4, 0.25 * 2 + 3 * -1, 0.25 * 0.5 + 3 * 4];
    if (JSON.stringify(cpu.C) !== JSON.stringify(want)) {
        throw new Error('no-WebGPU CPU matmul is wrong: ' + JSON.stringify(cpu.C));
    }
    console.log('LIVE no-WebGPU fallback status="' + cpu.reason + '" backend=' + cpu.backend +
                ' cpu matmul=' + JSON.stringify(cpu.C));
    console.log('PASS WebGPU live Chrome contracts dispatchCount=' + result.dispatchCount);
} finally {
    if (browser) await browser.close();
    server.close();
}
