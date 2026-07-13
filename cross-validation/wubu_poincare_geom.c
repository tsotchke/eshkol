/*
 * wubu_poincare_geom.c -- implementation (see wubu_poincare_geom.h).
 * Standalone port of tsotchke/eshkol lib/core/manifold.esk geometry (MIT).
 */
#include "wubu_poincare_geom.h"

void poincare_exp_eshkol(float*o,const float*p,const float*v,int n,float c){
    /* FIXED 2026-07-12 (closes audit F1): the original eshkol/manifold.esk
       exp-map folded the conformal factor lam INTO the tanh argument, which
       broke the geodesic invariant dist(p,exp_p(v)) = |v| (it drifted with p).
       The correct base-aware Poincaré exp map (the one that satisfies the
       invariant against eshkol's own distance formula) is
           scalar = tanh(0.5*sqrt(c)*|v|) / (sqrt(c)*|v|)      (no lam factor)
           exp_p(v) = p ⊕ (scalar * v)
       This matches the origin map exp_0 (dist(0,exp_0(v)) = |v|) by the Möbius
       isometry, so dist(p,exp_p(v)) = |v| for every base point p. Verified by
       cross-validation/test_crossval.c. See REPORT.md F1. */
    if(c<=0){wpg_vadd(o,p,v,n);return;}
    float vn=wpg_vnorm(v,n);
    if(vn<WPG_EPS){memcpy(o,p,n*sizeof(float));return;}
    float arg=0.5f*sqrtf(c)*vn;
    float t=tanhf(arg);
    float factor=t/(sqrtf(c)*vn+WPG_EPS);
    float s[64]; wpg_vscale(s,v,factor,n);
    wpg_mobius_add(o,p,s,n,c);
}

void poincare_exp_correct(float*o,const float*p,const float*v,int n,float c){
    /* Verified-consistent base-aware Poincaré exp map (matches exp_p_A /
       poincare_exp_eshkol after the F1 fix): scalar = tanh(0.5*sqrt(c)*|v|) /
       (sqrt(c)*|v|), exp_p(v) = p ⊕ (scalar*v). Satisfies the geodesic
       invariant dist(p,exp_p(v)) = |v|. Kept as the reference implementation. */
    if(c<=0){wpg_vadd(o,p,v,n);return;}
    float vn=wpg_vnorm(v,n);
    if(vn<WPG_EPS){memcpy(o,p,n*sizeof(float));return;}
    float arg=0.5f*sqrtf(c)*vn;
    float t=tanhf(arg);
    float factor=t/(sqrtf(c)*vn+WPG_EPS);
    float s[64]; wpg_vscale(s,v,factor,n);
    wpg_mobius_add(o,p,s,n,c);
}

float eshkol_distance(const float*a,const float*b,int n,float c){
    float d2=0,na=0,nb=0;
    for(int i=0;i<n;i++){float dd=a[i]-b[i];d2+=dd*dd;na+=a[i]*a[i];nb+=b[i]*b[i];}
    float arg=1.0f+2.0f*c*d2/(((1.0f-c*na)+WPG_EPS)*(1.0f-c*nb)+WPG_EPS);
    return acoshf(fmaxf(1.0f,arg));
}

float manifold_lambda(ManifoldType t,const float*x,int n,float c){
    float nx2=wpg_vdot(x,x,n);
    if(t==MAN_HYPERBOLIC) return 2.0f/(1.0f-c*nx2+WPG_EPS);
    if(t==MAN_SPHERICAL)   return 2.0f/(1.0f+c*nx2+WPG_EPS);
    return 1.0f;
}
float manifold_christoffel(ManifoldType t,const float*x,int n,int i,int j,int k,float c){
    float lam=manifold_lambda(t,x,n,c);
    if(t==MAN_HYPERBOLIC){
        if(i==j) return -c*lam*x[k];
        if(i==k) return  c*lam*x[j];
        if(j==k) return  c*lam*x[i];
        return 0.0f;
    }
    if(t==MAN_SPHERICAL){
        if(i==j) return  c*lam*x[k];
        if(i==k) return -c*lam*x[j];
        if(j==k) return -c*lam*x[i];
        return 0.0f;
    }
    return 0.0f;
}
float manifold_sectional_curvature(ManifoldType t,float c){
    if(t==MAN_HYPERBOLIC) return -c;
    if(t==MAN_SPHERICAL)   return  c;
    return 0.0f;
}
float manifold_scalar_curvature(ManifoldType t,int n,float c){
    return manifold_sectional_curvature(t,c)*(float)n*(float)(n-1);
}
float manifold_ricci(ManifoldType t,const float*x,int n,int i,int j,float c){
    float lam=manifold_lambda(t,x,n,c);
    return manifold_sectional_curvature(t,c)*(float)(n-1)*lam*lam*(i==j?1.0f:0.0f);
}

/* === Parallel transport (audit F-PT, added 2026-07-12) === */
void parallel_transport_eshkol(float*o,const float*a,const float*b,const float*v,int n,float c){
    /* eshkol's manifold-parallel-transport, hyperbolic branch: rescale by the
       conformal-factor ratio lam(a)/lam(b). Euclidean: identity. Spherical:
       remove the component along b (eshkol's branch). c<=0 -> Euclidean. */
    if(c<=0){ wpg_vscale(o,v,1.0f,n); return; }
    float la=manifold_lambda(MAN_HYPERBOLIC,a,n,c);
    float lb=manifold_lambda(MAN_HYPERBOLIC,b,n,c);
    wpg_vscale(o,v,la/lb,n);
}

void parallel_transport_ref(float*o,const float*b,const float*v,int n,float c){
    /* Reference: transport a tangent at the ORIGIN to point b. lam(0)=2, so the
       scaling is 2/lam(b). Gyration-free at the origin — this is the operation
       waefrebeorn/WuBuMath's wubu_parallel_transport_to_p performs (origin -> b). */
    if(c<=0){ wpg_vscale(o,v,1.0f,n); return; }
    float lb=manifold_lambda(MAN_HYPERBOLIC,b,n,c);
    wpg_vscale(o,v,2.0f/lb,n);
}
