struct Params { n:u32, op:u32, pad0:u32, pad1:u32 };
@group(0) @binding(0) var<storage,read> A:array<vec2<f32>>;
@group(0) @binding(1) var<storage,read> B:array<vec2<f32>>;
@group(0) @binding(2) var<storage,read_write> OUT:array<vec2<f32>>;
@group(0) @binding(3) var<uniform> p:Params;
@group(0) @binding(4) var<storage,read_write> OPAQUE:array<f32>;
fn opaque_f32(x:f32,s:u32)->f32{OPAQUE[s]=x;return OPAQUE[s];}
fn two_sum(a:f32,b:f32,s:u32)->vec2<f32>{let aa=opaque_f32(a,s);let bb=opaque_f32(b,s);let z=opaque_f32(aa+bb,s);let q=z-aa;return vec2<f32>(z,opaque_f32((aa-(z-q))+(bb-q),s));}
fn two_prod(a:f32,b:f32,s:u32)->vec2<f32>{let aa=opaque_f32(a,s);let bb=opaque_f32(b,s);let q=opaque_f32(aa*bb,s);return vec2<f32>(q,opaque_f32(fma(aa,bb,-q),s));}
fn add(a:vec2<f32>,b:vec2<f32>,s:u32)->vec2<f32>{let q=two_sum(a.x,b.x,s);return two_sum(q.x,q.y+a.y+b.y,s);}
fn mul(a:vec2<f32>,b:vec2<f32>,s:u32)->vec2<f32>{let q=two_prod(a.x,b.x,s);return two_sum(q.x,q.y+fma(a.x,b.y,a.y*b.x),s);}
fn neg(a:vec2<f32>)->vec2<f32>{return vec2<f32>(-a.x,-a.y);}
fn div(a:vec2<f32>,b:vec2<f32>,s:u32)->vec2<f32>{let q=a.x/b.x;let r=add(a,neg(mul(vec2<f32>(q,0.0),b,s)),s);return two_sum(q,r.x/b.x,s);}
@compute @workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=p.n){return;}let a=A[i];let b=B[i];var r=vec2<f32>(0.0,0.0);switch(p.op){case 0u:{r=add(a,b,i);}case 1u:{r=add(a,neg(b),i);}case 2u:{r=mul(a,b,i);}case 3u:{r=div(a,b,i);}case 4u:{r=neg(a);}case 5u:{if(a.x<0.0){r=neg(a);}else{r=a;}}default:{r=vec2<f32>(0.0,0.0);}}OUT[i]=r;}
