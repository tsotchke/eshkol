struct Params { n: u32, op: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= p.n) { return; }
    let a = A[i]; let b = B[i]; var r: f32 = 0.0;
    switch (p.op) {
        case 0u: { r = a + b; } case 1u: { r = a - b; }
        case 2u: { r = a * b; } case 3u: { r = a / b; }
        case 4u: { r = -a; } case 5u: { r = abs(a); }
        case 6u: { r = exp(a); } case 7u: { r = log(a); }
        case 8u: { r = sin(a); } case 9u: { r = cos(a); }
        case 10u: { r = tanh(a); } case 11u: { r = max(a, 0.0); }
        case 12u: { r = 1.0 / (1.0 + exp(-a)); }
        case 13u: { r = sqrt(a); } case 14u: { r = 1.0 / a; }
        default: { r = 0.0; }
    }
    OUT[i] = r;
}
