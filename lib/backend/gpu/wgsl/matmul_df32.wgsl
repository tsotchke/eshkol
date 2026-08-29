struct Dims { M: u32, K: u32, N: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> d: Dims;
@group(0) @binding(4) var<storage, read_write> OPAQUE: array<f32>;
fn opaque_f32(x: f32, slot: u32) -> f32 { OPAQUE[slot] = x; return OPAQUE[slot]; }
fn two_sum(a:f32,b:f32,s:u32)->vec2<f32>{let aa=opaque_f32(a,s);let bb0=opaque_f32(b,s);let z=opaque_f32(aa+bb0,s);let q=z-aa;return vec2<f32>(z,opaque_f32((aa-(z-q))+(bb0-q),s));}
fn two_prod(a:f32,b:f32,s:u32)->vec2<f32>{let aa=opaque_f32(a,s);let bb=opaque_f32(b,s);let p=opaque_f32(aa*bb,s);return vec2<f32>(p,opaque_f32(fma(aa,bb,-p),s));}
fn df_add(a:vec2<f32>,b:vec2<f32>,s:u32)->vec2<f32>{let z=two_sum(a.x,b.x,s);return two_sum(z.x,z.y+a.y+b.y,s);}
fn df_mul(a:vec2<f32>,b:vec2<f32>,s:u32)->vec2<f32>{let p=two_prod(a.x,b.x,s);return two_sum(p.x,p.y+fma(a.x,b.y,a.y*b.x),s);}
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){let row=gid.y;let col=gid.x;if(row>=d.M||col>=d.N){return;}let s=row*d.N+col;var acc=vec2<f32>(0.0,0.0);for(var k=0u;k<d.K;k=k+1u){acc=df_add(acc,df_mul(A[row*d.K+k],B[k*d.N+col],s),s);}C[row*d.N+col]=acc;}
