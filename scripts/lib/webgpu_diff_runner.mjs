/*
 * WebGPU differential runner.
 *
 * Drives headless Chrome (Playwright, channel=chrome -> Metal-backed WebGPU on
 * macOS) and diffs every WGSL kernel in web/eshkol-webgpu.js against the CPU
 * reference in the same module, over data laid out in a real WebAssembly.Memory
 * so the pointer path under test is the one generated code actually uses.
 *
 * Two properties make this a gate rather than a demo:
 *
 *   1. NON-VACUITY. The backend counts GPU dispatches. A run where the GPU
 *      served zero cases is reported FAIL, not PASS -- the failure mode where a
 *      "GPU gate" silently measured the CPU against itself.
 *
 *   2. RED-PROOF. --corrupt=<kernel> rewrites one WGSL kernel before the module
 *      is injected, so the gate can be shown to go red on a deliberately broken
 *      kernel. The corruption lives here in the harness, never in the shipped
 *      module.
 *
 * The page is served over http://localhost because WebGPU requires a secure
 * context; a data: or file: URL reports navigator.gpu === undefined.
 *
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const MODULE_PATH = path.join(REPO, 'web', 'eshkol-webgpu.js');

/* ---- args ---- */
const argv = process.argv.slice(2);
const opt = (name, dflt) => {
    const hit = argv.find((a) => a.startsWith('--' + name + '='));
    return hit ? hit.slice(name.length + 3) : dflt;
};
const CORRUPT = opt('corrupt', null);
const JSON_OUT = argv.includes('--json');
const HEADED = argv.includes('--headed');
const REGRESSIONS = argv.includes('--regressions');
const FAST_REGRESSION_TOL = 1e-4;

/* ---- kernel corruptions for the red-proof ----
 * Each entry is a [pattern, replacement] applied to the module source. They are
 * deliberately small and plausible -- a wrong index, a dropped term -- rather
 * than syntax errors, so the gate is shown to catch WRONG RESULTS and not
 * merely a shader that fails to compile. */
const CORRUPTIONS = {
    /* GEMM f32/df32 share this indexing line; swapping B's stride transposes
     * the second operand. */
    gemm: [
        /acc = acc \+ A\[row \* d\.K \+ k\] \* B\[k \* d\.N \+ col\];/,
        'acc = acc + A[row * d.K + k] * B[col * d.K + k];',
    ],
    gemm_df32: [
        /acc = df_add\(acc, df_mul\(A\[row \* d\.K \+ k\], B\[k \* d\.N \+ col\], slot\), slot\);/,
        'acc = df_add(acc, df_mul(A[row * d.K + k], B[col * d.K + k], slot), slot);',
    ],
    /* Elementwise: make ADD compute a - b. */
    elem: [/case 0u:  \{ r = a \+ b; \}/, 'case 0u:  { r = a - b; }'],
    elem_df32: [/case 0u: \{ r = df_add\(a, b, i\); \}/, 'case 0u: { r = df_add(a, df_neg(b), i); }'],
    /* Reduction: make every partial block empty, producing a wrong identity. */
    reduce: [/var hi = block_start \+ p\.per_group;/, 'var hi = block_start;'],
    /* Double-float core: break two_prod's error term, which silently degrades
     * df32 to f32 -- the subtlest realistic corruption of the set. */
    two_prod: [/let e = opaque_f32\(fma\(aa, bb, -p\), slot\);/, 'let e = opaque_f32(0.0, slot);'],
};

function loadModuleSource() {
    let src = fs.readFileSync(MODULE_PATH, 'utf8');
    if (!CORRUPT) return { src, applied: null };
    const c = CORRUPTIONS[CORRUPT];
    if (!c) {
        console.error(`unknown corruption "${CORRUPT}"; known: ${Object.keys(CORRUPTIONS).join(', ')}`);
        process.exit(2);
    }
    if (!c[0].test(src)) {
        console.error(`corruption "${CORRUPT}" did not match the module source -- ` +
                      'the kernel it targets has changed. Fix the pattern in ' +
                      'scripts/lib/webgpu_diff_runner.mjs rather than skipping the red-proof.');
        process.exit(2);
    }
    src = src.replace(c[0], c[1]);
    return { src, applied: CORRUPT };
}

/* ---- the in-page differential ----
 * Runs entirely inside the browser: builds a WebAssembly.Memory, writes the
 * operands at real byte offsets, runs the WGSL kernel and the module's own CPU
 * reference over the same bytes, and returns both results. */
async function runInPage(page, cases) {
    return page.evaluate(async ({ cases, regression, gateTolerance }) => {
        const G = globalThis.EshkolWebGPU;
        if (!G) return { fatal: 'eshkol-webgpu.js did not load' };
        if (!navigator.gpu) return { skip: 'navigator.gpu unavailable' };

        const created = await G.create({
            precision: regression ? 'fast' : 'high',
            gateTolerance,
            threshold: 1
        });
        if (!created.ok) {
            return created.unsupported
                ? { unsupported: created.reason }
                : { skip: created.reason };
        }
        const be = created.backend;
        if (!be.supportsOperation('matmul')) {
            return { unsupported: regression
                ? 'UNSUPPORTED: WebGPU fast matmul opt-in is not available at gateTolerance=' + gateTolerance
                : 'UNSUPPORTED: WebGPU df32 matmul is not certified at GPU_GATE_TOL=1e-9' };
        }

        /* 16 MB of linear memory: enough for every small shape below, and the
         * same object type generated wasm hands the runtime. */
        const memory = new WebAssembly.Memory({ initial: 256 });
        be.setMemory(memory);
        const fakeMem = { buffer: memory.buffer };

        const results = [];
        let bump = 64;
        const alloc = (n) => { const p = bump; bump += n * 8; bump = (bump + 15) & ~15; return p; };
        const rng = (seed) => { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; };

        for (const c of cases) {
            bump = 64;
            const r = rng(c.seed || 12345);
            be.precision = c.tier;
            const before = be.dispatchCount;
            let entry = { name: c.name, tier: c.tier, kind: c.kind };

            try {
                if (c.kind === 'gemm') {
                    const { M, K, N } = c;
                    const aP = alloc(M * K), bP = alloc(K * N);
                    const gP = alloc(M * N), cP = alloc(M * N);
                    const A = new Float64Array(memory.buffer, aP, M * K);
                    const B = new Float64Array(memory.buffer, bP, K * N);
                    for (let i = 0; i < M * K; i++) A[i] = c.values === 'ones' ? 1 : r() * 2 - 1;
                    for (let i = 0; i < K * N; i++) B[i] = c.values === 'ones' ? 1 : r() * 2 - 1;
                    await be.matmulF64(aP, bP, gP, M, K, N);
                    G.cpu.matmul(fakeMem, aP, bP, cP, M, K, N);
                    entry.gpu = Array.from(new Float64Array(memory.buffer, gP, M * N));
                    entry.cpu = Array.from(new Float64Array(memory.buffer, cP, M * N));
                } else if (c.kind === 'elementwise') {
                    const n = c.n;
                    const aP = alloc(n), bP = alloc(n), gP = alloc(n), cP = alloc(n);
                    const A = new Float64Array(memory.buffer, aP, n);
                    const B = new Float64Array(memory.buffer, bP, n);
                    for (let i = 0; i < n; i++) A[i] = r() * 4 - 2;
                    /* keep the divisor away from zero so DIV is well conditioned */
                    for (let i = 0; i < n; i++) B[i] = (r() * 2 - 1) + (r() > 0.5 ? 1.5 : -1.5);
                    await be.elementwiseF64(aP, bP, gP, n, c.op);
                    G.cpu.elementwise(fakeMem, aP, bP, cP, n, c.op);
                    entry.gpu = Array.from(new Float64Array(memory.buffer, gP, n));
                    entry.cpu = Array.from(new Float64Array(memory.buffer, cP, n));
                } else if (c.kind === 'reduce') {
                    const n = c.n;
                    const iP = alloc(n), gP = alloc(1), cP = alloc(1);
                    const I = new Float64Array(memory.buffer, iP, n);
                    for (let i = 0; i < n; i++) {
                        I[i] = c.op === 1 ? (0.98 + r() * 0.04) : (r() * 2 - 1);
                    }
                    await be.reduceF64(iP, gP, n, c.op);
                    G.cpu.reduce(fakeMem, iP, cP, n, c.op);
                    entry.gpu = Array.from(new Float64Array(memory.buffer, gP, 1));
                    entry.cpu = Array.from(new Float64Array(memory.buffer, cP, 1));
                }
                entry.dispatched = be.dispatchCount - before;
            } catch (e) {
                entry.error = String(e);
            }
            results.push(entry);
        }

        return {
            results,
            dispatchCount: be.dispatchCount,
            fallbackCount: be.fallbackCount,
            fmaFused: be.fmaFused,
            diagnostics: be.diagnostics,
            adapter: created.adapter && created.adapter.info
                ? `${created.adapter.info.vendor}/${created.adapter.info.architecture}`
                : 'unknown',
        };
    }, { cases, regression: REGRESSIONS, gateTolerance: REGRESSIONS ? FAST_REGRESSION_TOL : GPU_GATE_TOL });
}

/* ---- gate tolerance ----
 * This runner is the browser counterpart of tests/gpu/gpu_correctness_gate.sh.
 * It has one contract: a GPU result is acceptable only at GPU_GATE_TOL. The
 * The normal matrix excludes f32 because it cannot promise 1e-9; the
 * --regressions matrix separately exercises its explicit reduced-precision
 * contract. */
const GPU_GATE_TOL = Number(process.env.GPU_GATE_TOL || '1e-9');
if (!Number.isFinite(GPU_GATE_TOL) || GPU_GATE_TOL <= 0) {
    console.error('webgpu_diff_runner: invalid GPU_GATE_TOL=' + process.env.GPU_GATE_TOL);
    process.exit(2);
}
const TOL = { high: GPU_GATE_TOL, fast: FAST_REGRESSION_TOL };

function compare(entry) {
    if (entry.error) return { ok: false, why: 'threw: ' + entry.error };
    if (!entry.gpu || !entry.cpu) return { ok: false, why: 'no result produced' };
    if (entry.dispatched !== 1) {
        return { ok: false, why: `expected exactly 1 GPU dispatch, saw ${entry.dispatched} ` +
                                 '(a case served by the CPU proves nothing)' };
    }
    let worst = 0, at = -1;
    let scale = 0;
    for (let i = 0; i < entry.cpu.length; i++) scale = Math.max(scale, Math.abs(entry.cpu[i]));
    const denom = Math.max(scale, 1e-12);
    for (let i = 0; i < entry.cpu.length; i++) {
        const g = entry.gpu[i], c = entry.cpu[i];
        if (!Number.isFinite(g) || !Number.isFinite(c)) {
            if (g !== c) return { ok: false, why: `non-finite mismatch at ${i}: ${g} vs ${c}` };
            continue;
        }
        const e = Math.abs(g - c) / denom;
        if (e > worst) { worst = e; at = i; }
    }
    const tol = TOL[entry.tier];
    return {
        ok: worst <= tol,
        worst, at, tol,
        why: worst <= tol ? '' :
            `rel err ${worst.toExponential(3)} at [${at}] exceeds ${entry.tier} tolerance ${tol.toExponential(1)}`,
    };
}

/* ---- case matrix (small shapes only: this runs in a browser under the RSS
 * discipline, and correctness of the kernel does not need large N) ---- */
const CASES = [];
for (const tier of REGRESSIONS ? ['fast'] : ['high']) {
    /* Shapes deliberately include non-multiples of the 8x8 workgroup tile so
     * the bounds guards in the kernel are exercised. */
    for (const [M, K, N] of [[8, 8, 8], [16, 32, 16], [33, 17, 9], [1, 64, 1]]) {
        CASES.push({ name: `gemm_${M}x${K}x${N}`, kind: 'gemm', tier, M, K, N, seed: M * 131 + K * 17 + N });
    }
    if (REGRESSIONS) {
        CASES.unshift({ name: 'gemm_fast_8x8_8_ones', kind: 'gemm', tier,
                        M: 8, K: 8, N: 8, values: 'ones', seed: 1 });
    }
    if (!REGRESSIONS) {
        for (const [op, nm] of [[0, 'add'], [1, 'sub'], [2, 'mul'], [3, 'div'], [4, 'neg'], [5, 'abs']]) {
            CASES.push({ name: `elem_${nm}`, kind: 'elementwise', tier, n: 1000, op, seed: 900 + op });
        }
        for (const [op, nm] of [[0, 'sum'], [1, 'prod'], [2, 'min'], [3, 'max'], [4, 'mean']]) {
            CASES.push({ name: `reduce_${nm}`, kind: 'reduce', tier, n: 4096, op, seed: 700 + op });
        }
    }
}

/* ---- main ---- */
let playwright;
try {
    playwright = await import('playwright');
} catch {
    try {
        playwright = await import(path.join(REPO, '.scratch', 'node_modules', 'playwright', 'index.js'));
    } catch {
        console.error('webgpu_diff_runner: SKIP - playwright is not installed.');
        console.error('  npm install playwright   (channel=chrome uses the system Chrome; no browser download)');
        process.exit(77);
    }
}

const { src, applied } = loadModuleSource();

const server = http.createServer((req, res) => {
    if (req.url.startsWith('/eshkol-webgpu.js')) {
        res.writeHead(200, { 'Content-Type': 'text/javascript' });
        res.end(src);
        return;
    }
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<!doctype html><html><head><meta charset="utf-8">' +
            '<script src="/eshkol-webgpu.js"></script></head><body></body></html>');
});
await new Promise((r) => server.listen(0, 'localhost', r));
const port = server.address().port;

let browser, out;
try {
    browser = await playwright.chromium.launch({
        channel: 'chrome',
        headless: !HEADED,
        args: ['--enable-unsafe-webgpu'],
    });
    const page = await browser.newPage();
    const consoleErrors = [];
    page.on('pageerror', (e) => consoleErrors.push(String(e)));
    /* localhost is a secure context; WebGPU is unavailable on data:/file:. */
    await page.goto(`http://localhost:${port}/`);
    out = await runInPage(page, CASES);
    if (consoleErrors.length) out.pageErrors = consoleErrors;
} catch (e) {
    console.error('webgpu_diff_runner: SKIP - could not drive Chrome: ' + e);
    process.exit(77);
} finally {
    if (browser) await browser.close();
    server.close();
}

if (out.skip) {
    console.error('webgpu_diff_runner: SKIP - ' + out.skip);
    process.exit(77);
}
if (out.unsupported) {
    console.log('UNSUPPORTED webgpu_diff - ' + out.unsupported);
    process.exit(77);
}
if (out.fatal) {
    console.error('webgpu_diff_runner: ERROR - ' + out.fatal);
    process.exit(2);
}

const report = { adapter: out.adapter, corrupted: applied, cases: [], dispatchCount: out.dispatchCount };
let failed = 0;
for (const e of out.results) {
    const v = compare(e);
    report.cases.push({ name: e.name, tier: e.tier, ok: v.ok, worst: v.worst, tol: v.tol,
                        at: v.at, gpu: v.at >= 0 ? e.gpu?.[v.at] : undefined,
                        cpu: v.at >= 0 ? e.cpu?.[v.at] : undefined,
                        nonzero: e.gpu ? e.gpu.filter((value) => value !== 0).length : undefined,
                        first: e.gpu ? e.gpu.slice(0, 8) : undefined, why: v.why });
    if (!v.ok) failed++;
    const id = `webgpu_diff/${e.tier}/${e.name}`;
    if (v.ok) console.log(`PASSED ${id} (rel err ${Number(v.worst).toExponential(2)} <= ${v.tol.toExponential(1)})`);
    else console.log(`FAILED ${id} - ${v.why}`);
    if (REGRESSIONS && e.name === 'gemm_fast_8x8_8_ones') {
        console.log(`DETAIL ${id} nonzero=${e.gpu?.filter((value) => value !== 0).length}/${e.gpu?.length} ` +
                    `first=[${e.gpu?.slice(0, 8).join(',')}]`);
    }
}

/* Non-vacuity: a run in which the GPU served nothing is a failed gate. */
if (out.dispatchCount === 0) {
    console.log('FAILED webgpu_diff/non_vacuity - zero GPU dispatches; ' +
                'the CPU was compared against itself');
    failed++;
} else {
    console.log(`PASSED webgpu_diff/non_vacuity (${out.dispatchCount} GPU dispatches, ` +
                `${out.fallbackCount} CPU fallbacks)`);
}
if (!out.fmaFused) {
    console.log('NOTE webgpu_diff/fma - fma() is not fused on this adapter; df32 tier was downgraded');
}
for (const d of out.diagnostics || []) console.log('NOTE ' + d);
if (out.pageErrors) for (const e of out.pageErrors) console.log('NOTE page error: ' + e);

report.failed = failed;
if (JSON_OUT) console.log('JSON ' + JSON.stringify(report));

console.log(`\nadapter=${out.adapter} cases=${out.results.length} failed=${failed}` +
            (applied ? ` corrupted=${applied}` : ''));
process.exit(failed === 0 ? 0 : 1);
