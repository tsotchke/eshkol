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
    if (req.url === '/eshkol-webgpu.js') {
        res.writeHead(200, { 'Content-Type': 'text/javascript' });
        res.end(SOURCE);
        return;
    }
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<!doctype html><script src="/eshkol-webgpu.js"></script>');
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

        const created = await G.create({ precision: 'fast', gateTolerance: 1e-4,
                                         threshold: 1 });
        assert(created.ok, created.reason || 'WebGPU initialization failed');
        const backend = created.backend;
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
        const nonsquare = await gemm(3, 5, 7, (i) => (i % 5) - 2,
                                     (i) => (i % 7) - 3);

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
            fmaFused: backend.fmaFused,
            smallDispatches: small.length,
            nonsquareDispatches: nonsquare.length,
            boundaryDispatches: `${atLimit.length}/${overLimit.length}`,
            callback: 7,
            dispatchCount: backend.dispatchCount,
            diagnostics: backend.diagnostics
        };
    });
    console.log('LIVE WebGPU initialization complete fmaFused=' + result.fmaFused +
                ' maxComputeWorkgroupsPerDimension=' + result.limit);
    console.log('LIVE GEMM 8x8 exact CPU reference dispatches=' + result.smallDispatches);
    console.log('LIVE GEMM 3x5*5x7 exact CPU reference dispatches=' + result.nonsquareDispatches);
    console.log('LIVE dispatch boundary 65535/65536 workgroups=' + result.boundaryDispatches);
    console.log('LIVE JSPI table callback suspension result=' + result.callback);
    console.log('PASS WebGPU live Chrome contracts dispatchCount=' + result.dispatchCount);
} finally {
    if (browser) await browser.close();
    server.close();
}
