#!/usr/bin/env node

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const module = await import(path.join(ROOT, 'web', 'eshkol-webgpu.js'));
const G = module.default || module;

function installMockJSPI() {
    const oldSuspending = WebAssembly.Suspending;
    const oldPromising = WebAssembly.promising;
    WebAssembly.Suspending = function (fn) { return fn; };
    WebAssembly.promising = function (fn) {
        return (...args) => Promise.resolve(fn(...args));
    };
    return () => {
        if (oldSuspending === undefined) delete WebAssembly.Suspending;
        else WebAssembly.Suspending = oldSuspending;
        if (oldPromising === undefined) delete WebAssembly.promising;
        else WebAssembly.promising = oldPromising;
    };
}

function testPrecisionContracts() {
    const device = {};
    const logs = [];
    const high = new G.EshkolWebGPU(device, { threshold: 1, log: (msg) => logs.push(msg) });
    assert.equal(high.shouldUse(1), false);
    assert.equal(high.supportsOperation('matmul'), false);
    assert.equal(high.supportsOperation('elementwise', G.ELEM.ADD), false);
    assert.equal(high.supportsOperation('elementwise', G.ELEM.EXP), false);
    assert.equal(high.supportsOperation('reduce', G.REDUCE.SUM), false);
    assert.equal(high.supportsOperation('reduce', G.REDUCE.PROD), false);

    const exact = new G.EshkolWebGPU(device, { precision: 'exact', threshold: 1 });
    assert.equal(exact.shouldUse(1), false);
    assert.equal(exact.supportsOperation('matmul'), false);

    const fast = new G.EshkolWebGPU(device, { precision: 'fast', threshold: 1 });
    assert.equal(fast.shouldUse(1), false);
    assert.equal(fast.supportsOperation('matmul'), false);
    const almostGate = new G.EshkolWebGPU(device, {
        precision: 'fast', threshold: 1, gateTolerance: 1.000001e-9
    });
    assert.equal(almostGate.shouldUse(1), false);
    assert.equal(almostGate.supportsOperation('matmul'), false);
    const explicitlyLoose = new G.EshkolWebGPU(device, {
        precision: 'fast', threshold: 1, gateTolerance: 1e-4,
        log: (msg) => logs.push(msg)
    });
    assert.equal(explicitlyLoose.shouldUse(1), true);
    assert.equal(explicitlyLoose.supportsOperation('matmul'), true);
    assert.equal(explicitlyLoose.supportsOperation('reduce', G.REDUCE.SUM), false);
    assert.match(explicitlyLoose.diagnostics.join('\n'), /explicit reduced-precision opt-in/);
    assert.match(logs.find((msg) => msg.includes('explicit reduced-precision opt-in')) || '',
                 /fast tier/);

    for (const precision of ['default', 'auto', 'bogus']) {
        const unknown = new G.EshkolWebGPU(device, { precision, threshold: 1 });
        assert.equal(unknown.shouldUse(1), false);
        assert.equal(unknown.supportsOperation('matmul'), false);
        assert.equal(unknown.supportsOperation('elementwise', G.ELEM.ADD), false);
    }
}

function testCpuReferenceShape() {
    const memory = new WebAssembly.Memory({ initial: 1 });
    const M = 3, K = 5, N = 7;
    const aPtr = 0, bPtr = 128, cPtr = 512;
    const A = new Float64Array(memory.buffer, aPtr, M * K);
    const B = new Float64Array(memory.buffer, bPtr, K * N);
    A.set(Array.from({ length: M * K }, (_, i) => (i - 7) / 3));
    B.set(Array.from({ length: K * N }, (_, i) => (11 - i) / 5));
    G.cpu.matmul({ buffer: memory.buffer }, aPtr, bPtr, cPtr, M, K, N);

    const expected = [];
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            let sum = 0;
            for (let k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
            expected.push(sum);
        }
    }
    assert.deepEqual(Array.from(new Float64Array(memory.buffer, cPtr, M * N)), expected);
}

function testHeadlessCpuPathFailsClosed() {
    const memory = new WebAssembly.Memory({ initial: 1 });
    const imports = G.makeImports(null, () => memory);
    assert.equal(imports.eshkol_gpu_init(), 0);
    assert.equal(imports.eshkol_gpu_should_use(1), 0);
    assert.equal(imports.eshkol_gpu_backend_available(G.ESHKOL_GPU_WEBGPU), 0);
}

async function testExecutionMarkerAndCPUFallback() {
    const restore = installMockJSPI();
    try {
        const memory = new WebAssembly.Memory({ initial: 1 });
        const backend = {
            device: {},
            executionMarker: 0,
            lastExecutionMarker: 0,
            threshold: 1,
            diagnostics: [],
            fallbackCount: 0,
            precision: 'high',
            shouldUse: () => true,
            supportsOperation: () => true,
            setMemory: () => {},
            async elementwiseF64() {
                const marker = ++this.executionMarker;
                this.lastExecutionMarker = marker;
                return marker;
            }
        };
        const imports = G.makeImports(backend, () => memory);
        const markerStatus = await imports.eshkol_gpu_elementwise_f64(0, 16, 32, 2, G.ELEM.ADD);
        assert.equal(markerStatus, 0);
        assert.equal(backend.executionMarker, 1);

        const A = new Float64Array(memory.buffer, 0, 2);
        const B = new Float64Array(memory.buffer, 16, 2);
        A.set([1, 2]);
        B.set([3, 4]);
        backend.elementwiseF64 = async () => 0;
        const fallbackStatus = await imports.eshkol_gpu_elementwise_f64(0, 16, 32, 2, G.ELEM.ADD);
        assert.equal(fallbackStatus, 0);
        assert.deepEqual(Array.from(new Float64Array(memory.buffer, 32, 2)), [4, 6]);
        assert.equal(backend.executionMarker, 1);
        assert.equal(backend.lastPath, 'cpu:elem');
        assert.equal(backend.fallbackCount, 1);

        backend.elementwiseF64 = async () => {
            const error = new Error('dispatch exceeded device limit');
            error.webgpuValidation = true;
            throw error;
        };
        await assert.rejects(
            imports.eshkol_gpu_elementwise_f64(0, 16, 32, 2, G.ELEM.ADD),
            /dispatch exceeded device limit/);
        assert.equal(backend.fallbackCount, 1);
        assert.equal(backend.lastPath, 'webgpu:error');
    } finally {
        restore();
    }
}

async function testPromisingExports() {
    const restore = installMockJSPI();
    try {
        const wrapped = G.promisingExports({ main: () => 7, run_program: () => 9, memory: {} });
        assert.equal(await wrapped.main(), 7);
        assert.equal(await wrapped.run_program(), 9);
        assert.equal(wrapped.memory.constructor, Object);
    } finally {
        restore();
    }
}

function testIntegrationContracts() {
    const webgpu = fs.readFileSync(path.join(ROOT, 'web', 'eshkol-webgpu.js'), 'utf8');
    const siteWebgpu = fs.readFileSync(path.join(ROOT, 'site', 'static', 'eshkol-webgpu.js'), 'utf8');
    const repl = fs.readFileSync(path.join(ROOT, 'web', 'eshkol-repl.js'), 'utf8');
    const runtime = fs.readFileSync(path.join(ROOT, 'site', 'static', 'eshkol-runtime.js'), 'utf8');
    const workflow = fs.readFileSync(path.join(ROOT, '.github', 'workflows', 'gpu-execution-gate.yml'), 'utf8');

    assert.equal(siteWebgpu, webgpu);
    assert.match(webgpu, /@workgroup_size\(\$\{GEMM_TILE\}, \$\{GEMM_TILE\}, 1\)/);
    assert.match(webgpu, /maxComputeWorkgroupsPerDimension/);
    assert.match(webgpu, /baseX \* GEMM_TILE, baseY \* GEMM_TILE/);
    assert.match(webgpu, /_submitDispatch\(enc, pass, x, y, 1/);
    assert.match(webgpu, /pushErrorScope\('validation'\)/);
    assert.match(webgpu, /webgpuValidation/);
    assert.match(webgpu, /@workgroup_size\(\$\{ELEM_WORKGROUP\}, 1, 1\)/);
    assert.match(webgpu, /await this\._submitDispatch\(enc, pass, Math\.ceil\(n \/ ELEM_WORKGROUP\), 1, 1/);
    assert.match(webgpu, /executionMarker/);
    assert.match(webgpu, /lastExecutionMarker/);
    assert.match(repl, /eshkol_batch_matmul_dispatch: gpu\.eshkol_batch_matmul_dispatch/);
    assert.match(runtime, /eshkol_batch_matmul_dispatch: gpu\.eshkol_batch_matmul_dispatch/);
    assert.match(repl, /G\.promisingExports\(instance\.exports\)/);
    assert.match(runtime, /G\.promisingExports\(instance\.exports\)/);
    assert.match(repl, /promisingEntry\(fn\)/);
    assert.match(runtime, /promisingEntry\(fn\)/);
    assert.doesNotMatch(repl, /__indirect_function_table\.get\(callbackFuncPtr\)\(/);
    assert.doesNotMatch(runtime, /__indirect_function_table\.get\(callbackFuncPtr\)\(/);
    assert.match(workflow, /GPU_GATE_TOL: '1e-9'/);
    assert.match(workflow, /node scripts\/lib\/webgpu_diff_runner\.mjs/);
}

testPrecisionContracts();
testCpuReferenceShape();
testHeadlessCpuPathFailsClosed();
await testExecutionMarkerAndCPUFallback();
await testPromisingExports();
testIntegrationContracts();
console.log('PASS webgpu regression contracts');
