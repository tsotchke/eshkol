struct Dims { M: u32, K: u32, N: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

@compute @workgroup_size(8, 8, 1)
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
