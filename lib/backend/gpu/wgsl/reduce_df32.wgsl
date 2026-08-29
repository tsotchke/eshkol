/* Deterministic one-invocation partial reduction. The host combines the
 * partial df32 values in f64 and refuses any operation outside its gate
 * tolerance contract. */
@group(0) @binding(0) var<storage,read> IN:array<vec2<f32>>;
@group(0) @binding(1) var<storage,read_write> OUT:array<vec2<f32>>;
struct Params{n:u32,op:u32,per_group:u32,pad:u32};
@group(0) @binding(2) var<uniform> p:Params;
@group(0) @binding(4) var<storage,read_write> OPAQUE:array<f32>;
fn opaque(x:f32,s:u32)->f32{OPAQUE[s]=x;return OPAQUE[s];}
fn ts(a:f32,b:f32,s:u32)->vec2<f32>{let aa=opaque(a,s);let bb=opaque(b,s);let z=opaque(aa+bb,s);let q=z-aa;return vec2<f32>(z,opaque((aa-(z-q))+(bb-q),s));}
fn add(a:vec2<f32>,b:vec2<f32>,s:u32)->vec2<f32>{let q=ts(a.x,b.x,s);return ts(q.x,q.y+a.y+b.y,s);}
fn mul(a:vec2<f32>,b:vec2<f32>,s:u32)->vec2<f32>{let aa=opaque(a.x,s);let bb=opaque(b.x,s);let p0=opaque(aa*bb,s);let e=opaque(fma(aa,bb,-p0),s);return ts(p0,e+fma(a.x,b.y,a.y*b.x),s);}
fn ident(op:u32)->vec2<f32>{switch(op){case 1u:{return vec2<f32>(1.0,0.0);}case 2u:{return vec2<f32>(3.4028235e38,0.0);}case 3u:{return vec2<f32>(-3.4028235e38,0.0);}default:{return vec2<f32>(0.0,0.0);}}}
fn combine(op:u32,a:vec2<f32>,b:vec2<f32>,s:u32)->vec2<f32>{switch(op){case 1u:{return mul(a,b,s);}case 2u:{if(b.x<a.x){return b;}return a;}case 3u:{if(b.x>a.x){return b;}return a;}default:{return add(a,b,s);}}}
@compute @workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){let g=gid.x;let start=g*p.per_group;var end=start+p.per_group;if(end>p.n){end=p.n;}var acc=ident(p.op);var i=start;while(i<end){acc=combine(p.op,acc,IN[i],g);i=i+1u;}OUT[g]=acc;}
