/* Canonical WGSL double-float helpers used by the WebGPU f64 emulation tier.
 * OPAQUE is a per-invocation storage scratch array initialized by the host.
 * The read-back is intentional: WGSL has no precise-math attribute, and a
 * compiler may otherwise fold TwoSum's compensation term to zero. */
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
