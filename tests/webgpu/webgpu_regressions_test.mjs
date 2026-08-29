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
    const high = new G.EshkolWebGPU(device, { threshold: 1 });
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
    const explicitlyLoose = new G.EshkolWebGPU(device, {
        precision: 'fast', threshold: 1, gateTolerance: 1e-4
    });
    assert.equal(explicitlyLoose.shouldUse(1), true);
    assert.equal(explicitlyLoose.supportsOperation('matmul'), true);
    assert.equal(explicitlyLoose.supportsOperation('reduce', G.REDUCE.SUM), false);
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
    assert.match(webgpu, /@workgroup_size\(\$\{ELEM_WORKGROUP\}, 1, 1\)/);
    assert.match(webgpu, /dispatchWorkgroups\(Math\.ceil\(n \/ ELEM_WORKGROUP\), 1, 1\)/);
    assert.match(webgpu, /executionMarker/);
    assert.match(webgpu, /lastExecutionMarker/);
    assert.match(repl, /eshkol_batch_matmul_dispatch: gpu\.eshkol_batch_matmul_dispatch/);
    assert.match(runtime, /eshkol_batch_matmul_dispatch: gpu\.eshkol_batch_matmul_dispatch/);
    assert.match(repl, /G\.promisingExports\(instance\.exports\)/);
    assert.match(runtime, /G\.promisingExports\(instance\.exports\)/);
    assert.match(workflow, /GPU_GATE_TOL: '1e-9'/);
    assert.match(workflow, /node scripts\/lib\/webgpu_diff_runner\.mjs/);
}

testPrecisionContracts();
await testExecutionMarkerAndCPUFallback();
await testPromisingExports();
testIntegrationContracts();
console.log('PASS webgpu regression contracts');
