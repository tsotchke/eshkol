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
 * PRECISION. WGSL has no f64 type. The f64 entry points are served by sf64:
 * IEEE 754 binary64 arithmetic carried out on the integer bit pattern, the
 * WGSL sibling of the Metal backend's metal_softfloat.h. It uses the same
 * ESHKOL_GPU_PRECISION tier vocabulary as the native backends (see
 * docs/breakdown/RUNTIME_CONFIGURATION.md):
 *
 *   exact : sf64, correctly rounded add/sub/mul/div. THE DEFAULT, as on the
 *           native backends. GEMM and elementwise results are bit-identical
 *           to the CPU path; reductions differ only by block reassociation.
 *   high  : served by the same sf64 kernels, which meet the ~48-bit contract.
 *   fast  : plain f32, ~24 bits, admitted only with an explicit gate
 *           tolerance >= 1e-6.
 *
 * Operations without a kernel for the active tier (transcendentals on sf64)
 * are refused by supportsOperation(); the dispatch then runs the CPU path and
 * records the fallback in diagnostics/fallbackCount.
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

    const REDUCE_BLOCK = 256;
    const REDUCE_WORKGROUP = 64;
    const REDUCE_MAX_BLOCKS = 4096;
    const GEMM_TILE = 8;
    const ELEM_WORKGROUP = 64;
    const GPU_GATE_TOL = 1e-9;
    /* Plain f32 has about seven decimal digits of relative precision. A
     * tolerance just above the f64 gate is not an honest f32 contract. */
    const FAST_GATE_TOL = 1e-6;
    const PRECISION_TIERS = new Set(['exact', 'high', 'fast']);

    /* ===================== WGSL ===================== */

    /* sf64: IEEE 754 binary64 arithmetic on integer words. WGSL has no f64,
     * and float tricks (double-float pairs) are defeated by shader compilers
     * that reassociate f32 arithmetic. Integer arithmetic cannot be
     * reassociated, so this is the WGSL sibling of the Metal backend's
     * metal_softfloat.h: each f64 is its raw bit pattern as vec2<u32>
     * (x = low word, y = high word, which is the little-endian layout of a
     * Float64Array, so operands are uploaded without any conversion).
     * add/sub/mul/div are correctly rounded (round-to-nearest-even) including
     * subnormals, signed zeros, infinities and NaN, following the Berkeley
     * SoftFloat algorithms. */
    const WGSL_SF64 = `
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn u64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}
fn u64_zero(a: vec2<u32>) -> bool { return (a.x | a.y) == 0u; }
fn u64_shl(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n >= 64u) { return vec2<u32>(0u, 0u); }
    if (n >= 32u) { return vec2<u32>(0u, a.x << (n - 32u)); }
    return vec2<u32>(a.x << n, (a.y << n) | (a.x >> (32u - n)));
}
fn u64_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n >= 64u) { return vec2<u32>(0u, 0u); }
    if (n >= 32u) { return vec2<u32>(a.y >> (n - 32u), 0u); }
    return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}
/* Right shift that ORs every shifted-out bit into bit 0 (sticky). */
fn u64_shr_jam(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n >= 64u) { return vec2<u32>(select(0u, 1u, !u64_zero(a)), 0u); }
    let r = u64_shr(a, n);
    let back = u64_shl(r, n);
    let lost = back.x != a.x || back.y != a.y;
    return vec2<u32>(r.x | select(0u, 1u, lost), r.y);
}
fn u64_clz(a: vec2<u32>) -> u32 {
    if (a.y != 0u) { return countLeadingZeros(a.y); }
    return 32u + countLeadingZeros(a.x);
}
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu; let a1 = a >> 16u;
    let b0 = b & 0xFFFFu; let b1 = b >> 16u;
    let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    return vec2<u32>((p00 & 0xFFFFu) | (mid << 16u),
                     p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}
struct U128 { hi: vec2<u32>, lo: vec2<u32> };
fn mul64(a: vec2<u32>, b: vec2<u32>) -> U128 {
    let p0 = mul32(a.x, b.x);
    let p1 = mul32(a.x, b.y);
    let p2 = mul32(a.y, b.x);
    let p3 = mul32(a.y, b.y);
    let t1 = p0.y + p1.x;
    let w1 = t1 + p2.x;
    let c1 = select(0u, 1u, t1 < p0.y) + select(0u, 1u, w1 < t1);
    let t2 = p1.y + p2.y;
    let t3 = t2 + p3.x;
    let w2 = t3 + c1;
    let c2 = select(0u, 1u, t2 < p1.y) + select(0u, 1u, t3 < t2) + select(0u, 1u, w2 < t3);
    return U128(vec2<u32>(w2, p3.y + c2), vec2<u32>(p0.x, w1));
}

fn f64_exp(a: vec2<u32>) -> u32 { return (a.y >> 20u) & 0x7FFu; }
fn f64_frac(a: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.x, a.y & 0xFFFFFu); }
fn f64_neg_bit(a: vec2<u32>) -> bool { return (a.y >> 31u) != 0u; }
fn f64_is_nan(a: vec2<u32>) -> bool { return f64_exp(a) == 0x7FFu && !u64_zero(f64_frac(a)); }
fn f64_is_zero(a: vec2<u32>) -> bool { return a.x == 0u && (a.y & 0x7FFFFFFFu) == 0u; }
fn f64_nan(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    if (f64_is_nan(a)) { return vec2<u32>(a.x, a.y | 0x80000u); }
    return vec2<u32>(b.x, b.y | 0x80000u);
}
fn f64_signed(neg: bool, bits_hi: u32) -> vec2<u32> {
    return vec2<u32>(0u, bits_hi | select(0u, 0x80000000u, neg));
}
fn f64_qnan() -> vec2<u32> { return vec2<u32>(0u, 0x7FF80000u); }
fn f64_neg(a: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.x, a.y ^ 0x80000000u); }
fn f64_abs(a: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.x, a.y & 0x7FFFFFFFu); }

/* Berkeley roundPackToF64: sig has its leading bit at bit 62 and exp is the
 * biased exponent minus one; the pack ADDS the significand so a carry out of
 * the implicit bit (or a subnormal rounding up to normal) bumps the exponent. */
fn f64_round_pack(neg: bool, exp_in: i32, sig_in: vec2<u32>) -> vec2<u32> {
    var e = exp_in;
    var sig = sig_in;
    if (e < 0) {
        sig = u64_shr_jam(sig, u32(min(-e, 64)));
        e = 0;
    } else if (e >= 0x7FD) {
        if (e > 0x7FD || !u64_lt(u64_add(sig, vec2<u32>(0x200u, 0u)), vec2<u32>(0u, 0x80000000u))) {
            return f64_signed(neg, 0x7FF00000u);
        }
    }
    let round_bits = sig.x & 0x3FFu;
    sig = u64_shr(u64_add(sig, vec2<u32>(0x200u, 0u)), 10u);
    if (round_bits == 0x200u) { sig.x = sig.x & 0xFFFFFFFEu; }
    if (u64_zero(sig)) { e = 0; }
    return u64_add(vec2<u32>(0u, (select(0u, 0x80000000u, neg)) | (u32(e) << 20u)), sig);
}

fn f64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let na = f64_neg_bit(a); let nb = f64_neg_bit(b);
    let ea = f64_exp(a); let eb = f64_exp(b);
    if (f64_is_nan(a) || f64_is_nan(b)) { return f64_nan(a, b); }
    if (ea == 0x7FFu) {
        if (eb == 0x7FFu && na != nb) { return f64_qnan(); }
        return a;
    }
    if (eb == 0x7FFu) { return b; }
    let za = f64_is_zero(a); let zb = f64_is_zero(b);
    if (za && zb) { return f64_signed(na && nb, 0u); }
    if (za) { return b; }
    if (zb) { return a; }
    var fa = f64_frac(a); var fb = f64_frac(b);
    var xa = i32(ea); var xb = i32(eb);
    if (ea == 0u) { xa = 1; } else { fa.y = fa.y | 0x100000u; }
    if (eb == 0u) { xb = 1; } else { fb.y = fb.y | 0x100000u; }
    fa = u64_shl(fa, 10u);
    fb = u64_shl(fb, 10u);
    var ez = xa;
    if (xa > xb) { fb = u64_shr_jam(fb, u32(xa - xb)); }
    else if (xb > xa) { fa = u64_shr_jam(fa, u32(xb - xa)); ez = xb; }
    var nz = na;
    var fz: vec2<u32>;
    if (na == nb) {
        fz = u64_add(fa, fb);
        if (fz.y >= 0x80000000u) { fz = u64_shr_jam(fz, 1u); ez = ez + 1; }
    } else {
        if (u64_lt(fa, fb)) { nz = nb; fz = u64_sub(fb, fa); }
        else if (u64_lt(fb, fa)) { fz = u64_sub(fa, fb); }
        else { return vec2<u32>(0u, 0u); }
        let sh = i32(u64_clz(fz)) - 1;
        if (sh > 0) { fz = u64_shl(fz, u32(sh)); ez = ez - sh; }
    }
    return f64_round_pack(nz, ez - 1, fz);
}

/* Normalise a nonzero finite operand: significand with its leading bit at
 * bit 52, and the matching (possibly < 1) biased exponent. */
struct Norm { sig: vec2<u32>, e: i32 };
fn f64_norm(a: vec2<u32>) -> Norm {
    let ea = f64_exp(a);
    var f = f64_frac(a);
    if (ea == 0u) {
        let s = u64_clz(f) - 11u;
        return Norm(u64_shl(f, s), 1 - i32(s));
    }
    f.y = f.y | 0x100000u;
    return Norm(f, i32(ea));
}

fn f64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let nz = f64_neg_bit(a) != f64_neg_bit(b);
    if (f64_is_nan(a) || f64_is_nan(b)) { return f64_nan(a, b); }
    if (f64_exp(a) == 0x7FFu) {
        if (f64_is_zero(b)) { return f64_qnan(); }
        return f64_signed(nz, 0x7FF00000u);
    }
    if (f64_exp(b) == 0x7FFu) {
        if (f64_is_zero(a)) { return f64_qnan(); }
        return f64_signed(nz, 0x7FF00000u);
    }
    if (f64_is_zero(a) || f64_is_zero(b)) { return f64_signed(nz, 0u); }
    let ma = f64_norm(a); let mb = f64_norm(b);
    var ez = ma.e + mb.e - 0x3FF;
    let prod = mul64(u64_shl(ma.sig, 10u), u64_shl(mb.sig, 11u));
    var fz = prod.hi;
    if (!u64_zero(prod.lo)) { fz.x = fz.x | 1u; }
    if (fz.y < 0x40000000u) { ez = ez - 1; fz = u64_shl(fz, 1u); }
    return f64_round_pack(nz, ez, fz);
}

fn f64_div(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let nz = f64_neg_bit(a) != f64_neg_bit(b);
    if (f64_is_nan(a) || f64_is_nan(b)) { return f64_nan(a, b); }
    if (f64_exp(a) == 0x7FFu) {
        if (f64_exp(b) == 0x7FFu) { return f64_qnan(); }
        return f64_signed(nz, 0x7FF00000u);
    }
    if (f64_exp(b) == 0x7FFu) { return f64_signed(nz, 0u); }
    if (f64_is_zero(b)) {
        if (f64_is_zero(a)) { return f64_qnan(); }
        return f64_signed(nz, 0x7FF00000u);
    }
    if (f64_is_zero(a)) { return f64_signed(nz, 0u); }
    let ma = f64_norm(a); let mb = f64_norm(b);
    var ez = ma.e - mb.e + 0x3FE;
    var rem = ma.sig;
    if (u64_lt(rem, mb.sig)) { ez = ez - 1; rem = u64_shl(rem, 1u); }
    var q = vec2<u32>(0u, 0u);
    for (var i = 0u; i < 63u; i = i + 1u) {
        q = u64_shl(q, 1u);
        if (!u64_lt(rem, mb.sig)) { rem = u64_sub(rem, mb.sig); q.x = q.x | 1u; }
        rem = u64_shl(rem, 1u);
    }
    if (!u64_zero(rem)) { q.x = q.x | 1u; }
    return f64_round_pack(nz, ez, q);
}

/* IEEE a < b: false when either is NaN; -0 and +0 compare equal. */
fn f64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (f64_is_nan(a) || f64_is_nan(b)) { return false; }
    if (f64_is_zero(a) && f64_is_zero(b)) { return false; }
    let na = f64_neg_bit(a); let nb = f64_neg_bit(b);
    if (na != nb) { return na; }
    if (na) { return u64_lt(f64_abs(b), f64_abs(a)); }
    return u64_lt(f64_abs(a), f64_abs(b));
}
`;

    /* GEMM. Accumulation is in the same k order as the CPU triple loop
     * (lib/backend/gpu/gpu_memory_stub.cpp), with one rounding per multiply
     * and per add, so the sf64 result is bit-identical to the CPU path. */
    const WGSL_GEMM_F32 = `
struct Dims {
    M: u32, K: u32, N: u32, pad: u32,
    base_x: u32, base_y: u32, pad2: u32, pad3: u32
};
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

@compute @workgroup_size(${GEMM_TILE}, ${GEMM_TILE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y + d.base_y;
    let col = gid.x + d.base_x;
    if (row >= d.M || col >= d.N) { return; }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < d.K; k = k + 1u) {
        acc = acc + A[row * d.K + k] * B[k * d.N + col];
    }
    C[row * d.N + col] = acc;
}
`;

    const WGSL_GEMM_SF64 = `
struct Dims {
    M: u32, K: u32, N: u32, pad: u32,
    base_x: u32, base_y: u32, pad2: u32, pad3: u32
};
@group(0) @binding(0) var<storage, read> A: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> B: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> C: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> d: Dims;
${WGSL_SF64}
@compute @workgroup_size(${GEMM_TILE}, ${GEMM_TILE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y + d.base_y;
    let col = gid.x + d.base_x;
    if (row >= d.M || col >= d.N) { return; }
    var acc = vec2<u32>(0u, 0u);
    for (var k: u32 = 0u; k < d.K; k = k + 1u) {
        acc = f64_add(acc, f64_mul(A[row * d.K + k], B[k * d.N + col]));
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

    /* sf64 elementwise covers the correctly-rounded IEEE operations:
     * add/sub/mul/div/neg/abs/relu/reciprocal. The transcendentals have no
     * sf64 kernel, so supportsOperation() refuses them and the dispatch runs
     * the CPU path and records that it did. */
    const SF64_ELEM_OPS = [0, 1, 2, 3, 4, 5, 11, 14];
    const WGSL_ELEM_SF64 = `
struct Params { n: u32, op: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> A: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> B: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> OUT: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> p: Params;
${WGSL_SF64}
@compute @workgroup_size(${ELEM_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let a = A[i];
    let b = B[i];
    var r = vec2<u32>(0u, 0u);
    switch (p.op) {
        case 0u:  { r = f64_add(a, b); }
        case 1u:  { r = f64_add(a, f64_neg(b)); }
        case 2u:  { r = f64_mul(a, b); }
        case 3u:  { r = f64_div(a, b); }
        case 4u:  { r = f64_neg(a); }
        case 5u:  { r = f64_abs(a); }
        case 11u: { if (f64_lt(vec2<u32>(0u, 0u), a)) { r = a; } else { r = vec2<u32>(0u, 0u); } }
        case 14u: { r = f64_div(vec2<u32>(0u, 0x3FF00000u), a); }
        default:  { r = f64_qnan(); }
    }
    OUT[i] = r;
}
`;

    /* Reduction. Each invocation folds one contiguous block in index order;
     * the host folds the block partials in block order in f64. MIN/MAX use
     * the same strict comparison as the CPU loop, so ties and NaN behave
     * identically; SUM/PROD/MEAN are reassociated at block boundaries only. */
    const WGSL_REDUCE_SF64 = `
struct Params { n: u32, op: u32, per_block: u32, blocks: u32 };
@group(0) @binding(0) var<storage, read> IN: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> OUT: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> p: Params;
${WGSL_SF64}
@compute @workgroup_size(${REDUCE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block = gid.x;
    if (block >= p.blocks) { return; }
    let start = block * p.per_block;
    let end = min(start + p.per_block, p.n);
    var acc = vec2<u32>(0u, 0u);
    if (p.op == 1u) { acc = vec2<u32>(0u, 0x3FF00000u); }
    if (p.op == 2u) { acc = vec2<u32>(0u, 0x7FF00000u); }
    if (p.op == 3u) { acc = vec2<u32>(0u, 0xFFF00000u); }
    for (var i = start; i < end; i = i + 1u) {
        let v = IN[i];
        switch (p.op) {
            case 1u: { acc = f64_mul(acc, v); }
            case 2u: { if (f64_lt(v, acc)) { acc = v; } }
            case 3u: { if (f64_lt(acc, v)) { acc = v; } }
            default: { acc = f64_add(acc, v); }
        }
    }
    OUT[block] = acc;
}
`;

    /* ===================== helpers ===================== */

    function encodeF32(src, count) {
        const out = new Float32Array(count);
        for (let i = 0; i < count; i++) out[i] = src[i];
        return out;
    }

    /* The sf64 kernels consume f64 bit patterns directly: copy the operand
     * bytes out of wasm memory (writeBuffer needs a stable source). */
    function f64Bytes(src) {
        return new Float64Array(src);
    }

    /* ===================== backend ===================== */

    class EshkolWebGPU {
        constructor(device, opts) {
            const o = opts || {};
            this.device = device;
            this.threshold = (typeof o.threshold === 'number' && o.threshold > 0)
                ? o.threshold : DEFAULT_THRESHOLD;
            const deviceLimit = device && device.limits &&
                Number(device.limits.maxComputeWorkgroupsPerDimension);
            this.maxComputeWorkgroupsPerDimension = Number.isSafeInteger(deviceLimit) &&
                deviceLimit > 0 ? deviceLimit : 65535;
            /* Same default as the native backends (ESHKOL_GPU_PRECISION). */
            const requestedPrecision = o.precision === undefined ? 'exact' : o.precision;
            this.precision = requestedPrecision;
            this.precisionKnown = PRECISION_TIERS.has(requestedPrecision);
            this.gateTolerance = (typeof o.gateTolerance === 'number' &&
                                  Number.isFinite(o.gateTolerance) &&
                                  o.gateTolerance > 0) ? o.gateTolerance : GPU_GATE_TOL;
            this.pipelines = new Map();
            /* Non-vacuity telemetry: a differential gate asserts these move. */
            this.dispatchCount = 0;
            this.fallbackCount = 0;
            this.executionMarker = 0;
            this.lastExecutionMarker = 0;
            this.dispatchHistory = [];
            this.lastPath = 'none';
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
            if (!be.precisionKnown) {
                return { ok: false, unsupported: true, reason: be.diagnostics[0] };
            }
            device.lost.then((info) => {
                be.diagnostics.push('device lost: ' + info.message);
                be.device = null;
            });
            be.log('[WebGPU] backend active, precision tier=' + be.precision +
                   (be.precision === 'fast' ? ' (f32)' : ' (sf64, IEEE f64)') +
                   ', threshold=' + be.threshold);
            return { ok: true, backend: be, adapter: adapter };
        }

        /* ---- the dispatch predicate, same shape as the native backends ---- */

        getBackend() { return this.device ? ESHKOL_GPU_WEBGPU : ESHKOL_GPU_NONE; }
        backendName() { return this.device ? 'WebGPU (browser compute)' : 'CPU only'; }
        setThreshold(t) { if (t > 0) this.threshold = t; }
        getThreshold() { return this.threshold; }

        /* `exact` and `high` are both served by the sf64 kernels (IEEE f64,
         * correctly rounded), which meets either contract. `fast` is f32 and
         * is admitted only under an explicit tolerance no tighter than the
         * f32 floor. */
        _f64Tier() { return this.precision === 'exact' || this.precision === 'high'; }

        fastAdmitted() {
            return this.precision === 'fast' && this.precisionKnown &&
                this.gateTolerance >= FAST_GATE_TOL;
        }

        _tierAdmitted() {
            if (!this.device || !this.precisionKnown) return false;
            return this._f64Tier() || this.fastAdmitted();
        }

        /* Mirrors eshkol_gpu_should_use(): active backend AND at or above the
         * element-count threshold. */
        shouldUse(numElements) {
            return this._tierAdmitted() && numElements >= this.threshold;
        }

        supportsOperation(kind, op) {
            if (!this._tierAdmitted()) return false;
            if (kind === 'matmul') return true;
            if (kind === 'elementwise') {
                return this._f64Tier() ? SF64_ELEM_OPS.includes(Number(op))
                                       : (Number(op) >= 0 && Number(op) <= ELEM.RECIPROCAL);
            }
            if (kind === 'reduce') {
                /* The reduction kernel is sf64 only; an f32 reduction would
                 * lose far more than an f32 elementwise op. */
                return this._f64Tier() &&
                    [REDUCE.SUM, REDUCE.PROD, REDUCE.MIN, REDUCE.MAX, REDUCE.MEAN]
                        .includes(Number(op));
            }
            return false;
        }

        supportsF64() { return false; }   /* no native hardware f64 in WGSL */
        /* Any correct f64 path, native or emulated -- same meaning as
         * eshkol_gpu_has_fp64() on Metal, whose f64 is also soft-float. */
        hasFp64() { return !!this.device && this.precisionKnown && this._f64Tier(); }

        setMemory(mem) { this.memory = mem; }

        _f64View(ptr, count, memory) {
            const mem = memory === undefined ? this.memory : memory;
            return new Float64Array(mem.buffer, ptr, count);
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

        _uniform(u32s) {
            const buf = this.device.createBuffer({
                size: Math.max(16, u32s.length * 4),
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
            /* A result token belongs to this dispatch. Backend-wide counters
             * remain telemetry only and must not be used to identify an
             * overlapping call's completion. */
            return { marker, path };
        }

        _destroyBuffers(...buffers) {
            for (const buffer of buffers) if (buffer) buffer.destroy();
        }

        async _submitDispatch(encoder, pass, x, y, z, label, afterPass) {
            let scopeOpen = true;
            this.device.pushErrorScope('validation');
            try {
                pass.dispatchWorkgroups(x, y, z);
                pass.end();
                if (afterPass) afterPass();
                this.device.queue.submit([encoder.finish()]);
                const error = await this.device.popErrorScope();
                scopeOpen = false;
                if (error) {
                    const detail = error.message || String(error);
                    const failure = new Error('WebGPU validation error during ' + label + ': ' + detail);
                    failure.webgpuValidation = true;
                    throw failure;
                }
                this.dispatchHistory.push({ x, y, z, label });
            } catch (e) {
                if (scopeOpen) {
                    try { await this.device.popErrorScope(); } catch (_) {}
                }
                throw e;
            }
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

        /* Operand upload: sf64 kernels take the f64 bit patterns verbatim;
         * the fast tier converts to f32. */
        _encode(ptr, count, memory) {
            const v = this._f64View(ptr, count, memory);
            return this._f64Tier() ? f64Bytes(v) : encodeF32(v, count);
        }

        _decode(raw, ptr, count, memory) {
            const dst = this._f64View(ptr, count, memory);
            if (this._f64Tier()) dst.set(new Float64Array(raw, 0, count));
            else { const f = new Float32Array(raw); for (let i = 0; i < count; i++) dst[i] = f[i]; }
        }

        /* ---------------- GEMM ---------------- */

        /* C = A * B, all row-major, pointers are byte offsets into wasm memory
         * holding f64. Mirrors eshkol_gpu_matmul_f64 / eshkol_matmul_dispatch. */
        async matmulF64(aPtr, bPtr, cPtr, M, K, N, memory = this.memory) {
            if (!this.supportsOperation('matmul')) {
                throw new Error('UNSUPPORTED: WebGPU matmul is not admitted for precision tier ' + this.precision);
            }
            const sf = this._f64Tier();
            const encA = this._encode(aPtr, M * K, memory);
            const encB = this._encode(bPtr, K * N, memory);
            const outBytes = M * N * (sf ? 8 : 4);

            let bufA = null, bufB = null, bufC = null;
            const dims = [];
            try {
                bufA = this._storage(encA);
                bufB = this._storage(encB);
                bufC = this._outStorage(outBytes);

                const pipe = this._pipeline(sf ? 'gemm_sf64' : 'gemm_f32',
                                            sf ? WGSL_GEMM_SF64 : WGSL_GEMM_F32);
                const layout = pipe.getBindGroupLayout(0);
                const groupsX = Math.ceil(N / GEMM_TILE);
                const groupsY = Math.ceil(M / GEMM_TILE);
                const limit = this.maxComputeWorkgroupsPerDimension;
                for (let baseY = 0; baseY < groupsY; baseY += limit) {
                    const y = Math.min(limit, groupsY - baseY);
                    for (let baseX = 0; baseX < groupsX; baseX += limit) {
                        const x = Math.min(limit, groupsX - baseX);
                        const tileDims = this._uniform([
                            M, K, N, 0, baseX * GEMM_TILE, baseY * GEMM_TILE, 0, 0
                        ]);
                        dims.push(tileDims);
                        const bg = this.device.createBindGroup({
                            layout,
                            entries: [
                                { binding: 0, resource: { buffer: bufA } },
                                { binding: 1, resource: { buffer: bufB } },
                                { binding: 2, resource: { buffer: bufC } },
                                { binding: 3, resource: { buffer: tileDims } }
                            ]
                        });
                        const enc = this.device.createCommandEncoder();
                        const pass = enc.beginComputePass();
                        pass.setPipeline(pipe);
                        pass.setBindGroup(0, bg);
                        await this._submitDispatch(enc, pass, x, y, 1,
                            'gemm ' + x + 'x' + y + ' workgroups');
                    }
                }

                const raw = await this._readback(bufC, outBytes);
                this._decode(raw, cPtr, M * N, memory);
                return this._recordExecution(sf ? 'webgpu:gemm_sf64' : 'webgpu:gemm_f32');
            } finally {
                this._destroyBuffers(bufA, bufB, bufC, ...dims);
            }
        }

        /* ---------------- elementwise ---------------- */

        async elementwiseF64(aPtr, bPtr, outPtr, n, op, memory = this.memory) {
            if (!this.supportsOperation('elementwise', op)) {
                throw new Error('UNSUPPORTED: WebGPU elementwise op ' + op +
                                ' has no kernel for precision tier ' + this.precision);
            }
            const sf = this._f64Tier();
            const encA = this._encode(aPtr, n, memory);

            /* Binary ops read B; unary ops get the identity operand the stub
             * backend uses so the kernel needs no separate unary variant. */
            let encB;
            if (bPtr !== 0 && op <= ELEM.DIV) {
                encB = this._encode(bPtr, n, memory);
            } else {
                const ident = (op === ELEM.MUL || op === ELEM.DIV) ? 1 : 0;
                encB = sf ? new Float64Array(n) : new Float32Array(n);
                if (ident === 1) encB.fill(1);
            }

            const bytes = n * (sf ? 8 : 4);
            let bufA = null, bufB = null, bufO = null, params = null;
            try {
                bufA = this._storage(encA);
                bufB = this._storage(encB);
                bufO = this._outStorage(bytes);
                params = this._uniform([n, op, 0, 0]);

                const pipe = this._pipeline(sf ? 'elem_sf64' : 'elem_f32',
                                            sf ? WGSL_ELEM_SF64 : WGSL_ELEM_F32);
                const bg = this.device.createBindGroup({
                    layout: pipe.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: bufA } },
                        { binding: 1, resource: { buffer: bufB } },
                        { binding: 2, resource: { buffer: bufO } },
                        { binding: 3, resource: { buffer: params } }
                    ]
                });
                const enc = this.device.createCommandEncoder();
                const pass = enc.beginComputePass();
                pass.setPipeline(pipe);
                pass.setBindGroup(0, bg);
                const groups = Math.ceil(n / ELEM_WORKGROUP);
                if (groups > this.maxComputeWorkgroupsPerDimension) {
                    throw new Error('UNSUPPORTED: elementwise size ' + n +
                                    ' exceeds the device workgroup limit');
                }
                await this._submitDispatch(enc, pass, groups, 1, 1,
                    'elementwise ' + n + ' elements');

                const raw = await this._readback(bufO, bytes);
                this._decode(raw, outPtr, n, memory);
                return this._recordExecution(sf ? 'webgpu:elem_sf64' : 'webgpu:elem_f32');
            } finally {
                this._destroyBuffers(bufA, bufB, bufO, params);
            }
        }

        /* ---------------- reduction ---------------- */

        async reduceF64(inPtr, outPtr, n, op, memory = this.memory) {
            if (!this.supportsOperation('reduce', op)) {
                throw new Error('UNSUPPORTED: WebGPU reduction op ' + op +
                                ' has no kernel for precision tier ' + this.precision);
            }
            /* MEAN reduces as SUM then divides on the host, matching the stub
             * backend. */
            const kernelOp = (op === REDUCE.MEAN) ? REDUCE.SUM : op;
            const encIn = this._encode(inPtr, n, memory);
            const blocks = Math.min(REDUCE_MAX_BLOCKS, Math.max(1, Math.ceil(n / REDUCE_BLOCK)));
            const perBlock = Math.ceil(n / blocks);

            let bufIn = null, bufOut = null, params = null;
            try {
                bufIn = this._storage(encIn);
                bufOut = this._outStorage(blocks * 8);
                params = this._uniform([n, kernelOp, perBlock, blocks]);

                const pipe = this._pipeline('reduce_sf64', WGSL_REDUCE_SF64);
                const bg = this.device.createBindGroup({
                    layout: pipe.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: bufIn } },
                        { binding: 1, resource: { buffer: bufOut } },
                        { binding: 2, resource: { buffer: params } }
                    ]
                });
                const enc = this.device.createCommandEncoder();
                const pass = enc.beginComputePass();
                pass.setPipeline(pipe);
                pass.setBindGroup(0, bg);
                await this._submitDispatch(enc, pass, Math.ceil(blocks / REDUCE_WORKGROUP), 1, 1,
                    'reduction ' + blocks + ' blocks');

                const partials = new Float64Array(await this._readback(bufOut, blocks * 8));

                /* Final cross-block fold on the host in f64, in block order,
                 * with the CPU loop's own comparisons. */
                let acc;
                switch (kernelOp) {
                    case REDUCE.PROD: acc = 1; break;
                    case REDUCE.MIN: acc = Infinity; break;
                    case REDUCE.MAX: acc = -Infinity; break;
                    default: acc = 0; break;
                }
                for (let g = 0; g < blocks; g++) {
                    const v = partials[g];
                    switch (kernelOp) {
                        case REDUCE.PROD: acc *= v; break;
                        case REDUCE.MIN: acc = v < acc ? v : acc; break;
                        case REDUCE.MAX: acc = v > acc ? v : acc; break;
                        default: acc += v; break;
                    }
                }
                if (op === REDUCE.MEAN) acc /= n;
                this._f64View(outPtr, 1, memory)[0] = acc;
                return this._recordExecution('webgpu:reduce_sf64');
            } finally {
                this._destroyBuffers(bufIn, bufOut, params);
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

    /* ===================== dispatch policy =====================
     *
     * The one policy every browser caller of the GPU seam uses -- the
     * compiled-WASM imports below and the Emscripten VM bridge (attachVm).
     * Returns 'gpu' when the kernel ran and its execution marker verifies,
     * 'refused' when the active tier has no kernel for the operation (the
     * caller runs the CPU path; counted and explained here), 'failed' when a
     * kernel or readback raised. A WebGPU validation error is rethrown: it is
     * a defect in a kernel launch, never a reason to quietly use the CPU. */
    async function gpuServe(backend, memory, kind, op, fn) {
        if (!backend.supportsOperation(kind, op)) {
            backend.fallbackCount++;
            backend.lastPath = 'cpu:' + kind;
            backend.diagnostics.push('CPU fallback: ' + kind +
                (op === undefined ? '' : ' op ' + op) +
                ' has no WebGPU kernel for precision tier ' + backend.precision);
            return 'refused';
        }
        try {
            const marker = await fn();
            if (marker && typeof marker === 'object' && marker.path) return 'gpu';
            throw new Error('missing WebGPU execution token');
        } catch (e) {
            if (e && e.webgpuValidation) {
                backend.diagnostics.push(e.message);
                backend.lastPath = 'webgpu:error';
                throw e;
            }
            backend.fallbackCount++;
            backend.lastPath = 'cpu:' + kind;
            backend.diagnostics.push(kind + ' failed, CPU fallback: ' + e);
            return 'failed';
        }
    }

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
        /* The loader's live memory first: one backend can serve several
         * modules on a page (the site runtime and the VM share a device). */
        const mem = () => memoryRef() || (backend && backend.memory);

        function sync(fn) { return fn; }
        function suspending(fn) {
            return jspi ? new WebAssembly.Suspending(fn) : null;
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
                    if (backend.shouldUse(M * N) &&
                        await gpuServe(backend, mem(), 'matmul', undefined,
                            () => backend.matmulF64(aPtr, bPtr, cPtr, M, K, N, mem())) === 'gpu') return;
                    if (!backend.shouldUse(M * N)) { backend.fallbackCount++; backend.lastPath = 'cpu:matmul'; }
                    cpu.matmul(mem(), aPtr, bPtr, cPtr, M, K, N);
                });

            entries.eshkol_gpu_elementwise_f64 = suspending(
                async (aPtr, bPtr, outPtr, n, op) => {
                    n = Number(n); op = Number(op);
                    if (backend.shouldUse(n) &&
                        await gpuServe(backend, mem(), 'elementwise', op,
                            () => backend.elementwiseF64(aPtr, bPtr, outPtr, n, op, mem())) === 'gpu') return 0;
                    if (!backend.shouldUse(n)) { backend.fallbackCount++; backend.lastPath = 'cpu:elementwise'; }
                    cpu.elementwise(mem(), aPtr, bPtr, outPtr, n, op);
                    return 0;
                });

            entries.eshkol_gpu_reduce_f64 = suspending(
                async (inPtr, outPtr, n, op) => {
                    n = Number(n); op = Number(op);
                    if (backend.shouldUse(n) &&
                        await gpuServe(backend, mem(), 'reduce', op,
                            () => backend.reduceF64(inPtr, outPtr, n, op, mem())) === 'gpu') return 0;
                    if (!backend.shouldUse(n)) { backend.fallbackCount++; backend.lastPath = 'cpu:reduce'; }
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
        entries.eshkol_gpu_has_fp64 = () => (backend && backend.hasFp64()) ? 1 : 0;
        entries.eshkol_gpu_should_use = (n) =>
            (backend && backend.shouldUse(Number(n))) ? 1 : 0;
        entries.eshkol_gpu_set_threshold = (t) => { if (backend) backend.setThreshold(Number(t)); };
        entries.eshkol_gpu_get_threshold = () => backend ? backend.threshold : DEFAULT_THRESHOLD;

        return entries;
    }


    /* ===================== Emscripten-built VM (ADR-0029) =====================
     *
     * The bytecode VM's tensor natives call the ordinary GPU seam
     * (lib/backend/vm_gpu_dispatch.h -> eshkol_gpu_*), which the wasm build
     * implements in lib/backend/gpu/gpu_memory_webgpu.cpp. That file's compute
     * imports are synchronous "no device" stubs; attachVm() installs this
     * bridge in their place, as WebAssembly.Suspending imports, and wraps the
     * VM's entry exports with WebAssembly.promising -- the same JSPI boundary
     * the compiled-WASM loaders use. Without JSPI or a device nothing is
     * replaced, the status says why, and the VM runs its CPU path unchanged. */

    /* C import name -> bridge method. */
    const VM_COMPUTE_IMPORTS = {
        eshkol_webgpu_js_matmul: 'matmul',
        eshkol_webgpu_js_elementwise: 'elementwise',
        eshkol_webgpu_js_reduce: 'reduce'
    };
    /* Exports that run Eshkol code and so may reach a suspending import. */
    const VM_ENTRY_EXPORTS = ['repl_eval', 'run_program'];
    /* Bridge status codes, gpu_memory_webgpu.cpp. */
    const VM_OK = 0, VM_DECLINED = 2, VM_THREW = 3;

    function makeVmBridge(backend, memoryRef) {
        const code = async (kind, op, fn) => {
            const r = await gpuServe(backend, memoryRef(), kind, op, fn);
            return r === 'gpu' ? VM_OK : r === 'refused' ? VM_DECLINED : VM_THREW;
        };
        return {
            backend,
            deviceReady: () => (backend.device ? 1 : 0),
            threshold: () => backend.threshold,
            setThreshold: (t) => backend.setThreshold(Number(t)),
            shouldUse: (n) => (backend.shouldUse(Number(n)) ? 1 : 0),
            hasFp64: () => (backend.hasFp64() ? 1 : 0),
            noteFallback: (what) => {
                backend.fallbackCount++;
                backend.lastPath = 'cpu:' + what;
                backend.diagnostics.push('CPU fallback: ' + what + ' has no WebGPU kernel');
            },
            matmul: (a, b, c, M, K, N) => code('matmul', undefined,
                () => backend.matmulF64(a, b, c, Number(M), Number(K), Number(N), memoryRef())),
            elementwise: (a, b, o, n, op) => code('elementwise', Number(op),
                () => backend.elementwiseF64(a, b, o, Number(n), Number(op), memoryRef())),
            reduce: (i, o, n, op) => code('reduce', Number(op),
                () => backend.reduceF64(i, o, Number(n), Number(op), memoryRef()))
        };
    }

    /* Prepare an Emscripten module argument (the object passed to the
     * EshkolVM factory) so the VM dispatches to `backend`. Returns the same
     * object; `moduleArg.eshkolWebGPUStatus` is { ok, reason } either way.
     * opts.wasmUrl overrides where the .wasm is fetched from. */
    function attachVm(moduleArg, backend, opts) {
        const o = opts || {};
        const m = moduleArg || {};
        if (!backend || !backend.device) {
            m.eshkolWebGPUStatus = { ok: false, reason: backend
                ? 'WebGPU device lost' : 'no WebGPU backend (navigator.gpu, adapter or device unavailable)' };
            return m;
        }
        if (!jspiAvailable()) {
            m.eshkolWebGPUStatus = { ok: false,
                reason: 'JSPI unavailable (WebAssembly.Suspending missing); the VM runs on the CPU' };
            return m;
        }
        let memory = null;
        const bridge = makeVmBridge(backend, () => memory);
        const wasmUrl = o.wasmUrl ||
            (typeof m.locateFile === 'function' ? m.locateFile('eshkol-vm.wasm', '') : 'eshkol-vm.wasm');
        m.eshkolWebGPUBridge = bridge;
        m.eshkolWebGPUStatus = { ok: true, reason: '' };
        m.instantiateWasm = (imports, receive) => {
            const env = imports.env || {};
            for (const [name, method] of Object.entries(VM_COMPUTE_IMPORTS)) {
                if (!(name in env)) {
                    throw new Error('eshkol-vm.wasm has no ' + name +
                                    ' import; rebuild it with scripts/build-wasm-repl.sh');
                }
                env[name] = new WebAssembly.Suspending(bridge[method]);
            }
            const bytes = m.wasmBinary ? Promise.resolve(m.wasmBinary)
                : fetch(wasmUrl).then((r) => {
                    if (!r.ok) throw new Error('fetch ' + wasmUrl + ': HTTP ' + r.status);
                    return r.arrayBuffer();
                });
            bytes.then((b) => WebAssembly.instantiate(b, imports)).then(({ instance, module }) => {
                memory = instance.exports.memory;
                const exports = Object.assign({}, instance.exports);
                for (const name of VM_ENTRY_EXPORTS) {
                    if (typeof exports[name] === 'function') {
                        exports[name] = WebAssembly.promising(exports[name]);
                    }
                }
                receive({ exports }, module);
            }).catch((e) => {
                if (typeof m.onAbort === 'function') m.onAbort(e);
                else console.error('eshkol-vm instantiation failed:', e);
            });
            return {};
        };
        return m;
    }

    /* Call a VM export with one string argument (or none). Returns a Promise
     * of its string result (repl_eval) or undefined. Unqueued: use it inside
     * vmSerial, or use vmCall. */
    async function vmInvoke(vm, name, source) {
        if (source === undefined) return vm['_' + name]();
        const ptr = vm.stringToNewUTF8(String(source));
        try {
            const result = await vm['_' + name](ptr);
            return (typeof result === 'number' && result) ? vm.UTF8ToString(result) :
                   (name === 'repl_eval' ? '' : undefined);
        } finally {
            vm._free(ptr);
        }
    }

    /* Run `fn` with exclusive use of the VM. A suspended evaluation owns the
     * VM's shadow stack until it resumes, so every call into a VM that may
     * suspend is serialised through this per-module queue. */
    function vmSerial(vm, fn) {
        const next = (vm.__eshkolVmQueue || Promise.resolve()).then(fn, fn);
        vm.__eshkolVmQueue = next.catch(() => {});
        return next;
    }

    function vmCall(vm, name, source) {
        return vmSerial(vm, () => vmInvoke(vm, name, source));
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
        makeVmBridge,
        attachVm,
        vmCall,
        vmInvoke,
        vmSerial,
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
