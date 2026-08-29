/*
 * Eshkol WebGPU compute backend.
 *
 * This is the WASM-target sibling of lib/backend/gpu/gpu_memory.mm (Metal) and
 * lib/backend/gpu/gpu_memory_cuda.cpp (CUDA). It implements the same dispatch
 * seam that generated code calls -- eshkol_matmul_dispatch and the
 * eshkol_gpu_*_f64 compute entry points -- so a WASM program reaches the GPU
 * through the ORDINARY dispatch predicate (eshkol_gpu_should_use: active
 * backend + element-count threshold), not through a browser-special path.
 *
 * PRECISION. WGSL has no f64 of any kind, so the f64 entry points cannot be
 * served natively. This backend slots into the existing ESHKOL_GPU_PRECISION
 * tier vocabulary (see docs/breakdown/RUNTIME_CONFIGURATION.md):
 *
 *   exact : IEEE f64, correct to ULP. NOT AVAILABLE on WebGPU -- falls back to
 *           the CPU path. (An sf64/Ozaki-II WGSL port is a named follow-up.)
 *   high  : df32 -- each f64 carried as an unevaluated (hi, lo) pair of f32,
 *           Dekker/Knuth double-float arithmetic. ~48 bits of mantissa against
 *           f64's 53. THIS IS THE WebGPU DEFAULT and is admitted only when the
 *           fused-fma probe succeeds.
 *   fast  : plain f32, ~24 bits.
 *
 * Defaulting to `high` rather than `exact` is a deliberate, documented
 * departure from the native backends, which default to exact. It is reported
 * by eshkol_gpu_has_fp64() returning 0 (no correct-to-ULP path) and logged at
 * init. Callers that require exactness set precision to "exact" and get the
 * CPU path.
 *
 * ASYNC. WebGPU readback is unavoidably asynchronous (mapAsync). Eshkol's
 * runtime is synchronous C compiled to wasm32. The bridge is JSPI
 * (WebAssembly.Suspending / WebAssembly.promising): the GPU imports are marked
 * suspending and the entry export is marked promising, so wasm blocks on the
 * GPU without any rewriting of the module. See attachTo() below. Where JSPI is
 * absent the GPU imports are installed as synchronous CPU implementations and
 * dispatchCount stays 0 -- callers detect that and report SKIP, never a silent
 * pass.
 *
 */

(function (root, factory) {
    'use strict';
    const mod = factory();
    if (typeof module === 'object' && module.exports) module.exports = mod;
    root.EshkolWebGPU = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    /* Must match EshkolGPUBackend in inc/eshkol/backend/gpu/gpu_memory.h */
    const ESHKOL_GPU_NONE = 0;
    const ESHKOL_GPU_WEBGPU = 4;

    /* Must match EshkolElementwiseOp in gpu_memory.h */
    const ELEM = {
        ADD: 0, SUB: 1, MUL: 2, DIV: 3, NEG: 4, ABS: 5, EXP: 6, LOG: 7,
        SIN: 8, COS: 9, TANH: 10, RELU: 11, SIGMOID: 12, SQRT: 13, RECIPROCAL: 14
    };
    /* Must match EshkolReduceOp in gpu_memory.h */
    const REDUCE = { SUM: 0, PROD: 1, MIN: 2, MAX: 3, MEAN: 4 };

    /* Same default as g_gpu_threshold in all three native backends. */
    const DEFAULT_THRESHOLD = 100000;

    const REDUCE_WORKGROUP = 256;
    const GEMM_TILE = 8;
    const ELEM_WORKGROUP = 64;
    const GPU_GATE_TOL = 1e-9;
    /* Plain f32 has about seven decimal digits of relative precision. A
     * tolerance just above the df32 gate is not an honest f32 contract. */
    const FAST_GATE_TOL = 1e-6;
    const PRECISION_TIERS = new Set(['exact', 'high', 'fast']);

    /* ===================== WGSL ===================== */

    /* Double-float (df32) helpers. Each logical f64 is an (hi, lo) f32 pair
     * with |lo| <= ulp(hi)/2. WGSL optimizers are allowed to reassociate
     * ordinary arithmetic, which would fold the TwoSum residual to zero. The
     * storage write/read below is a deliberate opaque barrier: each invocation
     * owns one scratch slot, so the shader compiler cannot fold the
     * error-free transforms into ordinary reassociated f32 arithmetic. */
    const WGSL_DF32 = `
@group(0) @binding(4) var<storage, read_write> OPAQUE: array<f32>;
fn opaque_f32(x: f32, slot: u32) -> f32 {
    OPAQUE[slot] = x;
    return OPAQUE[slot];
}
fn two_sum(a: f32, b: f32, slot: u32) -> vec2<f32> {
    let aa = opaque_f32(a, slot);
    let bb0 = opaque_f32(b, slot);
    let s = opaque_f32(aa + bb0, slot);
    let bb = s - aa;
    let err = opaque_f32((aa - (s - bb)) + (bb0 - bb), slot);
    return vec2<f32>(s, err);
}
fn two_prod(a: f32, b: f32, slot: u32) -> vec2<f32> {
    let aa = opaque_f32(a, slot);
    let bb = opaque_f32(b, slot);
    let p = opaque_f32(aa * bb, slot);
    let e = opaque_f32(fma(aa, bb, -p), slot);
    return vec2<f32>(p, e);
}
fn df_add(a: vec2<f32>, b: vec2<f32>, slot: u32) -> vec2<f32> {
    let s = two_sum(a.x, b.x, slot);
    let e = s.y + (a.y + b.y);
    return two_sum(s.x, e, slot);
}
fn df_mul(a: vec2<f32>, b: vec2<f32>, slot: u32) -> vec2<f32> {
    let p = two_prod(a.x, b.x, slot);
    let e = p.y + fma(a.x, b.y, a.y * b.x);
    return two_sum(p.x, e, slot);
}
`;

    /* GEMM. Accumulation is in the same k order as the CPU triple loop
     * (lib/backend/gpu/gpu_memory_stub.cpp:171-179), so the only divergence
     * from CPU is the working precision, not the summation order. */
    const WGSL_GEMM_F32 = `
struct Dims { M: u32, K: u32, N: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

@compute @workgroup_size(${GEMM_TILE}, ${GEMM_TILE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= d.M || col >= d.N) { return; }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < d.K; k = k + 1u) {
        acc = acc + A[row * d.K + k] * B[k * d.N + col];
    }
    C[row * d.N + col] = acc;
}
`;

    const WGSL_GEMM_DF32 = `
struct Dims { M: u32, K: u32, N: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> d: Dims;
${WGSL_DF32}
@compute @workgroup_size(${GEMM_TILE}, ${GEMM_TILE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= d.M || col >= d.N) { return; }
    var acc = vec2<f32>(0.0, 0.0);
    for (var k: u32 = 0u; k < d.K; k = k + 1u) {
        let slot = row * d.N + col;
        acc = df_add(acc, df_mul(A[row * d.K + k], B[k * d.N + col], slot), slot);
    }
    C[row * d.N + col] = acc;
}
`;

    /* Elementwise. Op numbering is EshkolElementwiseOp verbatim. Unary ops
     * ignore B. Missing-B identity handling matches the stub backend:
     * 0 for add/sub, 1 for mul/div -- the host side supplies that operand. */
    const WGSL_ELEM_F32 = `
struct Params { n: u32, op: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(${ELEM_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let a = A[i];
    let b = B[i];
    var r: f32 = 0.0;
    switch (p.op) {
        case 0u:  { r = a + b; }
        case 1u:  { r = a - b; }
        case 2u:  { r = a * b; }
        case 3u:  { r = a / b; }
        case 4u:  { r = -a; }
        case 5u:  { r = abs(a); }
        case 6u:  { r = exp(a); }
        case 7u:  { r = log(a); }
        case 8u:  { r = sin(a); }
        case 9u:  { r = cos(a); }
        case 10u: { r = tanh(a); }
        case 11u: { r = max(a, 0.0); }
        case 12u: { r = 1.0 / (1.0 + exp(-a)); }
        case 13u: { r = sqrt(a); }
        case 14u: { r = 1.0 / a; }
        default:  { r = 0.0; }
    }
    OUT[i] = r;
}
`;

    /* df32 elementwise covers only the ops that are exactly representable in
     * double-float arithmetic (add/sub/mul/div/neg/abs). The transcendentals
     * have no df32 closed form here, so the high tier refuses them and uses
     * the CPU fallback. The lower-precision fast tier is an explicit opt-in,
     * outside the 1e-9 correctness gate. */
    const WGSL_ELEM_DF32 = `
struct Params { n: u32, op: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> A: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> OUT: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> p: Params;
${WGSL_DF32}
fn df_neg(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(-a.x, -a.y); }
fn df_div(a: vec2<f32>, b: vec2<f32>, slot: u32) -> vec2<f32> {
    let q1 = a.x / b.x;
    let r = df_add(a, df_neg(df_mul(vec2<f32>(q1, 0.0), b, slot)), slot);
    let q2 = r.x / b.x;
    return two_sum(q1, q2, slot);
}
@compute @workgroup_size(${ELEM_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let a = A[i];
    let b = B[i];
    var r = vec2<f32>(0.0, 0.0);
    switch (p.op) {
        case 0u: { r = df_add(a, b, i); }
        case 1u: { r = df_add(a, df_neg(b), i); }
        case 2u: { r = df_mul(a, b, i); }
        case 3u: { r = df_div(a, b, i); }
        case 4u: { r = df_neg(a); }
        case 5u: { if (a.x < 0.0) { r = df_neg(a); } else { r = a; } }
        default: { r = vec2<f32>(0.0, 0.0); }
    }
    OUT[i] = r;
}
`;

    /* Reduction. One workgroup (one invocation) produces one partial block.
     * Keeping the partial loop sequential makes the f64-vs-df32 comparison
     * deterministic and avoids a workgroup tree whose identity/bounds logic
     * can hide an empty block as a successful zero. */
    const WGSL_REDUCE_DF32 = `
struct Params { n: u32, op: u32, per_group: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> IN: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> OUT: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> p: Params;
${WGSL_DF32}
fn ident(op: u32) -> vec2<f32> {
    switch (op) {
        case 1u: { return vec2<f32>(1.0, 0.0); }
        case 2u: { return vec2<f32>(3.4028235e38, 0.0); }
        case 3u: { return vec2<f32>(-3.4028235e38, 0.0); }
        default: { return vec2<f32>(0.0, 0.0); }
    }
}
fn combine(op: u32, a: vec2<f32>, b: vec2<f32>, slot: u32) -> vec2<f32> {
    switch (op) {
        case 1u: { return df_mul(a, b, slot); }
        case 2u: { if (b.x < a.x) { return b; } else { return a; } }
        case 3u: { if (b.x > a.x) { return b; } else { return a; } }
        default: { return df_add(a, b, slot); }
    }
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let op = p.op;
    let group = gid.x;
    let block_start = group * p.per_group;
    var acc = ident(op);
    var hi = block_start + p.per_group;
    if (hi > p.n) { hi = p.n; }
    var i = block_start;
    while (i < hi) {
        acc = combine(op, acc, IN[i], group);
        i = i + 1u;
    }
    OUT[group] = acc;
}
`;

    /* Probe: confirm fma() is fused. If a * b + c is computed with two
     * roundings, two_prod's error term is garbage and df32 silently degrades
     * to f32. Rather than let that pass unnoticed the backend downgrades the
     * advertised tier and says so. */
    const WGSL_FMA_PROBE = `
@group(0) @binding(0) var<storage, read_write> OUT: array<f32>;
@compute @workgroup_size(1, 1, 1)
fn main() {
    // a * b is not representable in f32; the fused residual must be nonzero.
    let a: f32 = 1.0000001;
    let b: f32 = 1.0000002;
    let p = a * b;
    OUT[0] = fma(a, b, -p);
}
`;

    /* ===================== helpers ===================== */

    function splitDf32(v) {
        const hi = Math.fround(v);
        return [hi, Math.fround(v - hi)];
    }

    function encodeDf32(src, count) {
        const out = new Float32Array(count * 2);
        for (let i = 0; i < count; i++) {
            const hi = Math.fround(src[i]);
            out[i * 2] = hi;
            out[i * 2 + 1] = Math.fround(src[i] - hi);
        }
        return out;
    }

    function decodeDf32(buf, dst, count, offset) {
        const o = offset || 0;
        for (let i = 0; i < count; i++) {
            dst[o + i] = buf[i * 2] + buf[i * 2 + 1];
        }
    }

    function encodeF32(src, count) {
        const out = new Float32Array(count);
        for (let i = 0; i < count; i++) out[i] = src[i];
        return out;
    }

    /* ===================== backend ===================== */

    class EshkolWebGPU {
        constructor(device, opts) {
            const o = opts || {};
            this.device = device;
            this.threshold = (typeof o.threshold === 'number' && o.threshold > 0)
                ? o.threshold : DEFAULT_THRESHOLD;
            const requestedPrecision = o.precision === undefined ? 'high' : o.precision;
            this.precision = requestedPrecision;
            this.precisionKnown = PRECISION_TIERS.has(requestedPrecision);
            this.gateTolerance = (typeof o.gateTolerance === 'number' &&
                                  Number.isFinite(o.gateTolerance) &&
                                  o.gateTolerance > 0) ? o.gateTolerance : GPU_GATE_TOL;
            this.pipelines = new Map();
            this.uniformPool = [];
            /* Non-vacuity telemetry: a differential gate asserts these move. */
            this.dispatchCount = 0;
            this.fallbackCount = 0;
            this.executionMarker = 0;
            this.lastExecutionMarker = 0;
            this.lastPath = 'none';
            this.fmaFused = true;
            this.memory = null;
            this.log = o.log || function () {};
            this.diagnostics = [];
            if (!this.precisionKnown) {
                this.diagnostics.push('UNSUPPORTED: unknown WebGPU precision tier ' +
                                       String(requestedPrecision));
                this.log('[WebGPU] ' + this.diagnostics[this.diagnostics.length - 1]);
            } else if (this.precision === 'fast') {
                const optIn = 'explicit reduced-precision opt-in: fast tier, ' +
                    'gate tolerance=' + this.gateTolerance;
                this.diagnostics.push(optIn);
                this.log('[WebGPU] ' + optIn);
            }
        }

        /* Async device acquisition. Done once, before the wasm module is
         * instantiated, so eshkol_gpu_init() on the C side is a synchronous
         * query of an already-resolved device -- no suspension at init. */
        static async create(opts) {
            const o = opts || {};
            if (typeof navigator === 'undefined' || !navigator.gpu) {
                return { ok: false, reason: 'navigator.gpu unavailable (no WebGPU in this browser)' };
            }
            let adapter;
            try {
                adapter = await navigator.gpu.requestAdapter(
                    o.adapterOptions || { powerPreference: 'high-performance' });
            } catch (e) {
                return { ok: false, reason: 'requestAdapter threw: ' + e };
            }
            if (!adapter) return { ok: false, reason: 'no WebGPU adapter (headless without a GPU?)' };
            let device;
            try {
                device = await adapter.requestDevice();
            } catch (e) {
                return { ok: false, reason: 'requestDevice threw: ' + e };
            }
            if (!device) return { ok: false, reason: 'requestDevice returned null' };

            const be = new EshkolWebGPU(device, o);
            device.lost.then((info) => {
                be.diagnostics.push('device lost: ' + info.message);
                be.device = null;
            });
            await be._probeFma();
            if (!be.fmaFused && be.precision === 'high') {
                return { ok: false, unsupported: true,
                    reason: 'UNSUPPORTED: WebGPU adapter does not provide fused fma; df32 cannot meet GPU_GATE_TOL' };
            }
            be.log('[WebGPU] backend active, precision tier=' + be.precision +
                   ', threshold=' + be.threshold +
                   ', fma fused=' + be.fmaFused);
            return { ok: true, backend: be, adapter: adapter };
        }

        async _probeFma() {
            try {
                let out = null;
                let read = null;
                let mapped = false;
                out = this.device.createBuffer({
                    size: 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
                });
                const pipe = this._pipeline('fma_probe', WGSL_FMA_PROBE, [
                    { binding: 0, resource: { buffer: out } }
                ]);
                const bg = this.device.createBindGroup({
                    layout: pipe.getBindGroupLayout(0),
                    entries: [{ binding: 0, resource: { buffer: out } }]
                });
                const enc = this.device.createCommandEncoder();
                const pass = enc.beginComputePass();
                pass.setPipeline(pipe);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(1);
                pass.end();
                read = this.device.createBuffer({
                    size: 4,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                });
                enc.copyBufferToBuffer(out, 0, read, 0, 4);
                this.device.queue.submit([enc.finish()]);
                await read.mapAsync(GPUMapMode.READ);
                mapped = true;
                const v = new Float32Array(read.getMappedRange().slice(0))[0];
                read.unmap();
                mapped = false;
                read.destroy();
                read = null;
                out.destroy();
                out = null;
                this.fmaFused = (v !== 0);
                if (!this.fmaFused) {
                    this.diagnostics.push(
                        'UNSUPPORTED: fma() is not fused; refusing df32 dispatch because it cannot meet GPU_GATE_TOL');
                    this.log('[WebGPU] ' + this.diagnostics[this.diagnostics.length - 1]);
                }
            } catch (e) {
                this.fmaFused = false;
                this.diagnostics.push('fma probe failed: ' + e);
            } finally {
                if (read) {
                    if (mapped) read.unmap();
                    read.destroy();
                }
                if (out) out.destroy();
            }
        }

        /* ---- the dispatch predicate, same shape as the native backends ---- */

        getBackend() { return this.device ? ESHKOL_GPU_WEBGPU : ESHKOL_GPU_NONE; }
        backendName() { return this.device ? 'WebGPU (browser compute)' : 'CPU only'; }
        setThreshold(t) { if (t > 0) this.threshold = t; }
        getThreshold() { return this.threshold; }

        fastAdmitted() {
            return this.precision === 'fast' && this.precisionKnown &&
                this.gateTolerance >= FAST_GATE_TOL;
        }

        /* Mirrors eshkol_gpu_should_use(): active backend AND at or above the
         * element-count threshold. The `exact` tier has no WGSL implementation,
         * so it reports false and the caller takes the CPU path. */
        shouldUse(numElements) {
            if (!this.device || !this.precisionKnown || this.precision === 'exact') return false;
            /* The checked-in df32 shader is intentionally fail-closed until
             * the browser differential gate certifies its compensation path.
             * Selection and operation support must agree: neither may claim
             * that the unverified high tier is GPU-capable. */
            if (this.precision === 'high') return false;
            if (this.precision === 'fast' && !this.fastAdmitted()) return false;
            return numElements >= this.threshold;
        }

        supportsOperation(kind, op) {
            if (!this.device || !this.precisionKnown || this.precision === 'exact') return false;
            if (this.precision === 'high') return false;
            if (this.precision === 'fast') {
                /* f32 is never admitted to the 1e-9 gate. It is available only
                 * when the caller explicitly opts into a contract no tighter
                 * than the f32 floor, and
                 * reductions remain unsupported because their kernel is df32. */
                return this.fastAdmitted() &&
                    ['matmul', 'elementwise'].includes(kind);
            }
            if (kind === 'elementwise') return Number(op) <= ELEM.ABS;
            if (kind === 'reduce') {
                return [REDUCE.SUM, REDUCE.MIN, REDUCE.MAX, REDUCE.MEAN].includes(Number(op));
            }
            return kind === 'matmul';
        }

        supportsF64() { return false; }   /* no native hardware f64 in WGSL */
        hasFp64() { return false; }       /* and no correct-to-ULP emulation yet */

        setMemory(mem) { this.memory = mem; }

        _f64View(ptr, count) {
            return new Float64Array(this.memory.buffer, ptr, count);
        }

        _pipeline(key, wgsl) {
            let p = this.pipelines.get(key);
            if (!p) {
                const mod = this.device.createShaderModule({ code: wgsl });
                p = this.device.createComputePipeline({
                    layout: 'auto',
                    compute: { module: mod, entryPoint: 'main' }
                });
                this.pipelines.set(key, p);
            }
            return p;
        }

        _storage(data) {
            const buf = this.device.createBuffer({
                size: Math.max(data.byteLength, 4),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });
            this.device.queue.writeBuffer(buf, 0, data);
            return buf;
        }

        _outStorage(bytes) {
            return this.device.createBuffer({
                size: Math.max(bytes, 4),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            });
        }

        _opaque(count) {
            return this._storage(new Float32Array(Math.max(1, count || 1)));
        }

        _uniform(u32s) {
            const buf = this.device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(buf, 0, new Uint32Array(u32s));
            return buf;
        }

        _recordExecution(path) {
            const marker = ++this.executionMarker;
            this.lastExecutionMarker = marker;
            this.dispatchCount++;
            this.lastPath = path;
            return marker;
        }

        _destroyBuffers(...buffers) {
            for (const buffer of buffers) if (buffer) buffer.destroy();
        }

        /* The single async boundary. Everything above is synchronous JS;
         * only the readback suspends, and JSPI carries the wasm stack across
         * exactly this await. */
        async _readback(gpuBuf, bytes) {
            let read = null;
            let mapped = false;
            try {
                read = this.device.createBuffer({
                    size: bytes,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                });
                const enc = this.device.createCommandEncoder();
                enc.copyBufferToBuffer(gpuBuf, 0, read, 0, bytes);
                this.device.queue.submit([enc.finish()]);
                await read.mapAsync(GPUMapMode.READ);
                mapped = true;
                return read.getMappedRange().slice(0);
            } finally {
                if (read) {
                    if (mapped) read.unmap();
                    read.destroy();
                }
            }
        }

        /* ---------------- GEMM ---------------- */

        /* C = A * B, all row-major, pointers are byte offsets into wasm memory
         * holding f64. Mirrors eshkol_gpu_matmul_f64 / eshkol_matmul_dispatch. */
        async matmulF64(aPtr, bPtr, cPtr, M, K, N) {
            const df = (this.precision === 'high');
            const A = this._f64View(aPtr, M * K);
            const B = this._f64View(bPtr, K * N);

            const encA = df ? encodeDf32(A, M * K) : encodeF32(A, M * K);
            const encB = df ? encodeDf32(B, K * N) : encodeF32(B, K * N);
            const outBytes = M * N * (df ? 8 : 4);

            let bufA = null, bufB = null, bufC = null, dims = null, opaque = null;
            try {
                bufA = this._storage(encA);
                bufB = this._storage(encB);
                bufC = this._outStorage(outBytes);
                dims = this._uniform([M, K, N, 0]);
                opaque = df ? this._opaque(M * N) : null;

                const pipe = this._pipeline(df ? 'gemm_df32' : 'gemm_f32',
                                            df ? WGSL_GEMM_DF32 : WGSL_GEMM_F32);
                const bg = this.device.createBindGroup({
                    layout: pipe.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: bufA } },
                        { binding: 1, resource: { buffer: bufB } },
                        { binding: 2, resource: { buffer: bufC } },
                        { binding: 3, resource: { buffer: dims } },
                        ...(opaque ? [{ binding: 4, resource: { buffer: opaque } }] : [])
                    ]
                });
                const enc = this.device.createCommandEncoder();
                const pass = enc.beginComputePass();
                pass.setPipeline(pipe);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(Math.ceil(N / GEMM_TILE), Math.ceil(M / GEMM_TILE), 1);
                pass.end();
                this.device.queue.submit([enc.finish()]);

                const raw = await this._readback(bufC, outBytes);
                const C = this._f64View(cPtr, M * N);
                if (df) decodeDf32(new Float32Array(raw), C, M * N);
                else { const f = new Float32Array(raw); for (let i = 0; i < M * N; i++) C[i] = f[i]; }
                return this._recordExecution(df ? 'webgpu:gemm_df32' : 'webgpu:gemm_f32');
            } finally {
                this._destroyBuffers(bufA, bufB, bufC, dims, opaque);
            }
        }

        /* ---------------- elementwise ---------------- */

        async elementwiseF64(aPtr, bPtr, outPtr, n, op) {
            if (!this.supportsOperation('elementwise', op)) {
                throw new Error('UNSUPPORTED: WebGPU elementwise op cannot meet GPU_GATE_TOL');
            }
            /* df32 has closed forms only for the algebraic ops; the
             * transcendentals fall to the f32 kernel (tier `fast` semantics)
             * because a df32 exp/log/sin is not implemented. That is a
             * precision cliff, so it is recorded rather than hidden. */
            const algebraic = (op <= ELEM.ABS);
            const df = (this.precision === 'high') && algebraic;
            if (this.precision === 'high' && !algebraic) {
                this.diagnostics.push(
                    'elementwise op ' + op + ' has no df32 kernel; ran at f32 precision');
            }

            const A = this._f64View(aPtr, n);
            const encA = df ? encodeDf32(A, n) : encodeF32(A, n);

            /* Binary ops read B; unary ops get the identity operand the stub
             * backend uses so the kernel needs no separate unary variant. */
            let encB;
            if (bPtr !== 0 && op <= ELEM.DIV) {
                const B = this._f64View(bPtr, n);
                encB = df ? encodeDf32(B, n) : encodeF32(B, n);
            } else {
                const ident = (op === ELEM.MUL || op === ELEM.DIV) ? 1 : 0;
                encB = df ? new Float32Array(n * 2) : new Float32Array(n);
                if (ident === 1) {
                    for (let i = 0; i < n; i++) encB[df ? i * 2 : i] = 1;
                }
            }

            const bytes = n * (df ? 8 : 4);
            let bufA = null, bufB = null, bufO = null, params = null, opaque = null;
            try {
                bufA = this._storage(encA);
                bufB = this._storage(encB);
                bufO = this._outStorage(bytes);
                params = this._uniform([n, op, 0, 0]);
                opaque = df ? this._opaque(n) : null;

                const pipe = this._pipeline(df ? 'elem_df32' : 'elem_f32',
                                            df ? WGSL_ELEM_DF32 : WGSL_ELEM_F32);
                const bg = this.device.createBindGroup({
                    layout: pipe.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: bufA } },
                        { binding: 1, resource: { buffer: bufB } },
                        { binding: 2, resource: { buffer: bufO } },
                        { binding: 3, resource: { buffer: params } },
                        ...(opaque ? [{ binding: 4, resource: { buffer: opaque } }] : [])
                    ]
                });
                const enc = this.device.createCommandEncoder();
                const pass = enc.beginComputePass();
                pass.setPipeline(pipe);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(Math.ceil(n / ELEM_WORKGROUP), 1, 1);
                pass.end();
                this.device.queue.submit([enc.finish()]);

                const raw = await this._readback(bufO, bytes);
                const O = this._f64View(outPtr, n);
                if (df) decodeDf32(new Float32Array(raw), O, n);
                else { const f = new Float32Array(raw); for (let i = 0; i < n; i++) O[i] = f[i]; }
                return this._recordExecution(df ? 'webgpu:elem_df32' : 'webgpu:elem_f32');
            } finally {
                this._destroyBuffers(bufA, bufB, bufO, params, opaque);
            }
        }

        /* ---------------- reduction ---------------- */

        async reduceF64(inPtr, outPtr, n, op) {
            if (!this.supportsOperation('reduce', op)) {
                throw new Error('UNSUPPORTED: WebGPU reduction cannot meet GPU_GATE_TOL');
            }
            /* Reductions always run df32: a f32 reduction over a large array
             * loses far more than a f32 elementwise op, and the extra cost is
             * one f32 lane. MEAN reduces as SUM then divides on the host,
             * matching the stub backend. */
            const kernelOp = (op === REDUCE.MEAN) ? REDUCE.SUM : op;
            const IN = this._f64View(inPtr, n);
            const enc0 = encodeDf32(IN, n);

            const groups = Math.min(64, Math.max(1, Math.ceil(n / REDUCE_WORKGROUP)));
            const perGroup = Math.ceil(n / groups);

            let bufIn = null, bufOut = null, params = null, opaque = null;
            try {
                bufIn = this._storage(enc0);
                bufOut = this._outStorage(groups * 8);
                params = this._uniform([n, kernelOp, perGroup, 0]);
                opaque = this._opaque(groups);

                const pipe = this._pipeline('reduce_df32', WGSL_REDUCE_DF32);
                const bg = this.device.createBindGroup({
                    layout: pipe.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: bufIn } },
                        { binding: 1, resource: { buffer: bufOut } },
                        { binding: 2, resource: { buffer: params } },
                        { binding: 4, resource: { buffer: opaque } }
                    ]
                });
                const enc = this.device.createCommandEncoder();
                const pass = enc.beginComputePass();
                pass.setPipeline(pipe);
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(groups, 1, 1);
                pass.end();
                this.device.queue.submit([enc.finish()]);

                const raw = await this._readback(bufOut, groups * 8);
                const partials = new Float32Array(raw);

                /* Final cross-group combine on the host in full f64. */
                let acc;
                switch (kernelOp) {
                    case REDUCE.PROD: acc = 1; break;
                    case REDUCE.MIN: acc = Infinity; break;
                    case REDUCE.MAX: acc = -Infinity; break;
                    default: acc = 0; break;
                }
                for (let g = 0; g < groups; g++) {
                    const v = partials[g * 2] + partials[g * 2 + 1];
                    switch (kernelOp) {
                        case REDUCE.PROD: acc *= v; break;
                        case REDUCE.MIN: acc = v < acc ? v : acc; break;
                        case REDUCE.MAX: acc = v > acc ? v : acc; break;
                        default: acc += v; break;
                    }
                }
                if (op === REDUCE.MEAN) acc /= n;
                this._f64View(outPtr, 1)[0] = acc;
                return this._recordExecution('webgpu:reduce_df32');
            } finally {
                this._destroyBuffers(bufIn, bufOut, params, opaque);
            }
        }
    }

    /* ===================== CPU reference =====================
     * Byte-for-byte the arithmetic of lib/backend/gpu/gpu_memory_stub.cpp.
     * This is the fallback below threshold / without WebGPU, AND the
     * reference side of the differential gate. */

    const cpu = {
        matmul(mem, aPtr, bPtr, cPtr, M, K, N) {
            const A = new Float64Array(mem.buffer, aPtr, M * K);
            const B = new Float64Array(mem.buffer, bPtr, K * N);
            const C = new Float64Array(mem.buffer, cPtr, M * N);
            for (let i = 0; i < M; i++) {
                for (let j = 0; j < N; j++) {
                    let s = 0;
                    for (let k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
                    C[i * N + j] = s;
                }
            }
        },
        elementwise(mem, aPtr, bPtr, outPtr, n, op) {
            const A = new Float64Array(mem.buffer, aPtr, n);
            const B = bPtr ? new Float64Array(mem.buffer, bPtr, n) : null;
            const O = new Float64Array(mem.buffer, outPtr, n);
            for (let i = 0; i < n; i++) {
                const a = A[i];
                const b = B ? B[i] : ((op === ELEM.MUL || op === ELEM.DIV) ? 1 : 0);
                switch (op) {
                    case ELEM.ADD: O[i] = a + b; break;
                    case ELEM.SUB: O[i] = a - b; break;
                    case ELEM.MUL: O[i] = a * b; break;
                    case ELEM.DIV: O[i] = a / b; break;
                    case ELEM.NEG: O[i] = -a; break;
                    case ELEM.ABS: O[i] = Math.abs(a); break;
                    case ELEM.EXP: O[i] = Math.exp(a); break;
                    case ELEM.LOG: O[i] = Math.log(a); break;
                    case ELEM.SIN: O[i] = Math.sin(a); break;
                    case ELEM.COS: O[i] = Math.cos(a); break;
                    case ELEM.TANH: O[i] = Math.tanh(a); break;
                    case ELEM.RELU: O[i] = a > 0 ? a : 0; break;
                    case ELEM.SIGMOID: O[i] = 1 / (1 + Math.exp(-a)); break;
                    case ELEM.SQRT: O[i] = Math.sqrt(a); break;
                    case ELEM.RECIPROCAL: O[i] = 1 / a; break;
                }
            }
        },
        reduce(mem, inPtr, outPtr, n, op) {
            const I = new Float64Array(mem.buffer, inPtr, n);
            let r;
            switch (op) {
                case REDUCE.PROD: r = 1; break;
                case REDUCE.MIN: r = Infinity; break;
                case REDUCE.MAX: r = -Infinity; break;
                default: r = 0; break;
            }
            for (let i = 0; i < n; i++) {
                switch (op) {
                    case REDUCE.PROD: r *= I[i]; break;
                    case REDUCE.MIN: r = I[i] < r ? I[i] : r; break;
                    case REDUCE.MAX: r = I[i] > r ? I[i] : r; break;
                    default: r += I[i]; break;
                }
            }
            if (op === REDUCE.MEAN) r /= n;
            new Float64Array(mem.buffer, outPtr, 1)[0] = r;
        },
        batchMatmul(mem, aPtr, bPtr, cPtr, batch, M, K, N) {
            const aStride = M * K, bStride = K * N, cStride = M * N;
            for (let q = 0; q < batch; q++) {
                cpu.matmul(mem, aPtr + q * aStride * 8,
                           bPtr + q * bStride * 8,
                           cPtr + q * cStride * 8, M, K, N);
            }
        }
    };

    /* ===================== import installation =====================
     *
     * Produces the `env` entries the generated wasm imports. Call this from
     * BOTH loaders (web/eshkol-repl.js and site/static/eshkol-runtime.js):
     * scripts/check_wasm_imports.py fails the build if either one lacks an
     * import the codegen emits.
     *
     * `backend` may be null -- then every entry is the synchronous CPU
     * implementation and nothing suspends, so a JSPI-less browser still runs
     * the program correctly, just on the CPU.
     */
    function makeImports(backend, memoryRef) {
        const jspi = jspiAvailable();
        const mem = () => (backend && backend.memory) || memoryRef();

        function sync(fn) { return fn; }
        function suspending(fn) {
            return jspi ? new WebAssembly.Suspending(fn) : null;
        }

        function verified(marker, before) {
            return Number.isSafeInteger(marker) && marker > before &&
                   backend.executionMarker === marker &&
                   backend.lastExecutionMarker === marker;
        }

        /* Each entry: if the GPU can serve this call, suspend into the async
         * kernel; otherwise run the CPU version synchronously. When JSPI is
         * unavailable we cannot suspend at all, so we install the CPU version
         * outright. */
        const useGpu = backend && backend.device && jspi;

        const entries = {};

        if (useGpu) {
            entries.eshkol_matmul_dispatch = suspending(
                async (aPtr, bPtr, cPtr, M, K, N, dtype) => {
                    M = Number(M); K = Number(K); N = Number(N);
                    backend.setMemory(mem());
                    if (backend.shouldUse(M * N) && backend.supportsOperation('matmul')) {
                        const before = backend.executionMarker;
                        try {
                            const marker = await backend.matmulF64(aPtr, bPtr, cPtr, M, K, N);
                            if (verified(marker, before)) return 0;
                            throw new Error('missing WebGPU execution marker');
                        }
                        catch (e) { backend.diagnostics.push('gemm failed, CPU fallback: ' + e); }
                    }
                    if (backend.shouldUse(M * N)) backend.diagnostics.push('UNSUPPORTED: matmul refused before GPU_GATE_TOL certification');
                    backend.fallbackCount++;
                    backend.lastPath = 'cpu:gemm';
                    cpu.matmul(mem(), aPtr, bPtr, cPtr, M, K, N);
                });

            entries.eshkol_gpu_elementwise_f64 = suspending(
                async (aPtr, bPtr, outPtr, n, op) => {
                    n = Number(n); op = Number(op);
                    backend.setMemory(mem());
                    if (backend.shouldUse(n) && backend.supportsOperation('elementwise', op)) {
                        const before = backend.executionMarker;
                        try {
                            const marker = await backend.elementwiseF64(aPtr, bPtr, outPtr, n, op);
                            if (verified(marker, before)) return 0;
                            throw new Error('missing WebGPU execution marker');
                        }
                        catch (e) { backend.diagnostics.push('elementwise failed, CPU fallback: ' + e); }
                    }
                    if (backend.shouldUse(n)) backend.diagnostics.push('UNSUPPORTED: elementwise operation refused before GPU_GATE_TOL certification');
                    backend.fallbackCount++;
                    backend.lastPath = 'cpu:elem';
                    cpu.elementwise(mem(), aPtr, bPtr, outPtr, n, op);
                    return 0;
                });

            entries.eshkol_gpu_reduce_f64 = suspending(
                async (inPtr, outPtr, n, op) => {
                    n = Number(n); op = Number(op);
                    backend.setMemory(mem());
                    if (backend.shouldUse(n) && backend.supportsOperation('reduce', op)) {
                        const before = backend.executionMarker;
                        try {
                            const marker = await backend.reduceF64(inPtr, outPtr, n, op);
                            if (verified(marker, before)) return 0;
                            throw new Error('missing WebGPU execution marker');
                        }
                        catch (e) { backend.diagnostics.push('reduce failed, CPU fallback: ' + e); }
                    }
                    if (backend.shouldUse(n)) backend.diagnostics.push('UNSUPPORTED: reduction refused before GPU_GATE_TOL certification');
                    backend.fallbackCount++;
                    backend.lastPath = 'cpu:reduce';
                    cpu.reduce(mem(), inPtr, outPtr, n, op);
                    return 0;
                });
        } else {
            entries.eshkol_matmul_dispatch = sync(
                (aPtr, bPtr, cPtr, M, K, N, dtype) => {
                    cpu.matmul(mem(), aPtr, bPtr, cPtr, Number(M), Number(K), Number(N));
                });
            entries.eshkol_gpu_elementwise_f64 = sync(
                (aPtr, bPtr, outPtr, n, op) => {
                    cpu.elementwise(mem(), aPtr, bPtr, outPtr, Number(n), Number(op));
                    return 0;
                });
            entries.eshkol_gpu_reduce_f64 = sync(
                (inPtr, outPtr, n, op) => {
                    cpu.reduce(mem(), inPtr, outPtr, Number(n), Number(op));
                    return 0;
                });
        }

        /* Batched matmul has no browser WGSL kernel yet. Keep the import
         * callable and route it through the CPU reference; it must never be
         * mistaken for a GPU dispatch. */
        entries.eshkol_batch_matmul_dispatch = (aPtr, bPtr, cPtr, batch, M, K, N, dtype) =>
            cpu.batchMatmul(mem(), aPtr, bPtr, cPtr,
                            Number(batch), Number(M), Number(K), Number(N));

        /* Query surface -- the C-side predicate mirrored for generated code
         * and for host tooling. Always synchronous. */
        entries.eshkol_gpu_init = () => (backend && backend.device) ? 1 : 0;
        entries.eshkol_gpu_shutdown = () => {};
        entries.eshkol_gpu_get_backend = () => backend ? backend.getBackend() : ESHKOL_GPU_NONE;
        entries.eshkol_gpu_backend_available = (b) =>
            (Number(b) === ESHKOL_GPU_WEBGPU && backend && backend.device) ? 1 : 0;
        entries.eshkol_gpu_supports_f64 = () => 0;
        entries.eshkol_gpu_has_fp64 = () => 0;
        entries.eshkol_gpu_should_use = (n) =>
            (backend && backend.shouldUse(Number(n))) ? 1 : 0;
        entries.eshkol_gpu_set_threshold = (t) => { if (backend) backend.setThreshold(Number(t)); };
        entries.eshkol_gpu_get_threshold = () => backend ? backend.threshold : DEFAULT_THRESHOLD;

        return entries;
    }

    /* Wrap an instantiated module's entry export so JSPI can suspend inside
     * it. Without this the suspending imports throw on first call. */
    function promisingEntry(fn) {
        if (typeof WebAssembly.promising === 'function') {
            return WebAssembly.promising(fn);
        }
        return fn;
    }

    /* A WebAssembly.Instance exports object is not replaceable in place. Build
     * a public export facade so every synchronous wasm entry that can reach a
     * suspending GPU import is paired with WebAssembly.promising. Keeping the
     * non-function exports (memory, tables, globals) intact preserves the
     * loader ABI. */
    function promisingExports(exports) {
        if (!jspiAvailable()) return exports;
        const wrapped = {};
        for (const [name, value] of Object.entries(exports)) {
            wrapped[name] = typeof value === 'function' ? promisingEntry(value) : value;
        }
        return wrapped;
    }

    function jspiAvailable() {
        return typeof WebAssembly.Suspending === 'function' &&
               typeof WebAssembly.promising === 'function';
    }

    return {
        EshkolWebGPU,
        create: EshkolWebGPU.create,
        makeImports,
        promisingEntry,
        promisingExports,
        jspiAvailable,
        cpu,
        ELEM,
        REDUCE,
        DEFAULT_THRESHOLD,
        GPU_GATE_TOL,
        FAST_GATE_TOL,
        ESHKOL_GPU_WEBGPU
    };
});
