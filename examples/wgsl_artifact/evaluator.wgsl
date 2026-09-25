struct Params { count: u32, generation: u32, pad0: u32, pad1: u32 };
struct Sample { strain: f32 };
struct Result { value: f32, first: f32, second: f32, in_envelope: u32, status: u32, generation: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> samples: array<Sample>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;
fn law_value(t: f32) -> f32 { return 0.0 + t * (0.0 + t * (3.0 + t * (3.0 + t * (2.25 + t * (1.3499999999999999 + t * (0.6749999999999999 + t * (0.28928571428571426 + t * (0.10848214285714283 + t * (0.03616071428571428 + t * (0.010848214285714284)))))))))); }
fn law_first(t: f32) -> f32 { return 0.0 + t * (6.0 + t * (9.0 + t * (9.0 + t * (6.749999999999999 + t * (4.05 + t * (2.025 + t * (0.8678571428571427 + t * (0.3254464285714285 + t * (0.10848214285714283))))))))); }
fn law_second(t: f32) -> f32 { return 6.0 + t * (18.0 + t * (27.0 + t * (26.999999999999996 + t * (20.25 + t * (12.149999999999999 + t * (6.074999999999998 + t * (2.603571428571428 + t * (0.9763392857142855)))))))); }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= 4096u || i >= params.count || i >= arrayLength(&samples) || i >= arrayLength(&results)) { return; }
  let t = samples[i].strain; let inside = abs(t) <= 0.2;
  if (!inside) { results[i] = Result(0.0, 0.0, 0.0, 0u, 1u, params.generation, 0u, 0u); return; }
  let v = law_value(t); let d = law_first(t); let s = law_second(t);
  let finite = abs(v) < 1e30 && abs(d) < 1e30 && abs(s) < 1e30;
  results[i] = Result(select(0.0, v, finite), select(0.0, d, finite), select(0.0, s, finite), 1u, select(2u, 0u, finite), params.generation, 0u, 0u);
}
