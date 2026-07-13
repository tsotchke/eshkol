/*
 * wubu_poincare_geom.h -- Self-contained port of tsotchke/eshkol
 * lib/core/manifold.esk geometry (MIT), for independent cross-validation
 * against waefrebeorn/WuBuMath.
 *
 * DEVIL'S-ADVOCATE FINDING (see REPORT.md): eshkol's manifold-exp-map puts the
 * conformal factor lam = 2/(1-|p|^2) INSIDE the tanh. We keep eshkol's formula
 * VERBATIM (poincare_exp_eshkol) so the discrepancy is reproducible, and provide
 * the corrected form (poincare_exp_correct) for comparison.
 */
#ifndef WUBU_POINCARE_GEOM_H
#define WUBU_POINCARE_GEOM_H

#include <math.h>
#include <string.h>

#define WPG_EPS 1e-9f

typedef enum { MAN_EUCLIDEAN=0, MAN_HYPERBOLIC=1, MAN_SPHERICAL=2 } ManifoldType;

/* --- tiny vec helpers --- */
static inline float wpg_vdot(const float*a,const float*b,int n){float s=0;for(int i=0;i<n;i++)s+=a[i]*b[i];return s;}
static inline float wpg_vnorm(const float*a,int n){return sqrtf(wpg_vdot(a,a,n));}
static inline void wpg_vadd(float*o,const float*a,const float*b,int n){for(int i=0;i<n;i++)o[i]=a[i]+b[i];}
static inline void wpg_vsub(float*o,const float*a,const float*b,int n){for(int i=0;i<n;i++)o[i]=a[i]-b[i];}
static inline void wpg_vscale(float*o,const float*a,float s,int n){for(int i=0;i<n;i++)o[i]=a[i]*s;}
static inline void wpg_vneg(float*o,const float*a,int n){for(int i=0;i<n;i++)o[i]=-a[i];}

/* Mobius addition, general curvature c (c=1 -> standard Poincare ball) */
static inline void wpg_mobius_add(float*o,const float*a,const float*b,int n,float c){
    float ab=wpg_vdot(a,b,n), na2=wpg_vdot(a,a,n), nb2=wpg_vdot(b,b,n);
    float den=1.0f+2.0f*c*ab+c*c*na2*nb2+WPG_EPS;
    float c1=1.0f+2.0f*c*ab+c*nb2;
    float c2=1.0f-c*na2;
    for(int i=0;i<n;i++) o[i]=(c1*a[i]+c2*b[i])/den;
}

/* === ESHKOL VERBATIM (manifold.esk) — conformal factor INSIDE tanh === */
void poincare_exp_eshkol(float*o,const float*p,const float*v,int n,float c);

/* === CORRECTED form (gyrogroup, c=1) — conformal factor scales the VECTOR === */
void poincare_exp_correct(float*o,const float*p,const float*v,int n,float c);

/* eshkol's own distance (manifold-distance, hyperbolic branch) */
float eshkol_distance(const float*a,const float*b,int n,float c);

/* === Analytic Christoffel / curvature (from manifold.esk, VERBATIM) === */
float manifold_lambda(ManifoldType t,const float*x,int n,float c);
float manifold_christoffel(ManifoldType t,const float*x,int n,int i,int j,int k,float c);
float manifold_sectional_curvature(ManifoldType t,float c);
float manifold_scalar_curvature(ManifoldType t,int n,float c);
float manifold_ricci(ManifoldType t,const float*x,int n,int i,int j,float c);

/* === Parallel transport (audit F-PT, added 2026-07-12) ===
 * eshkol's manifold-parallel-transport (hyperbolic branch) rescales the tangent
 * by the conformal-factor ratio lam(a)/lam(b):  PT_{a->b}(v) = (lam_a/lam_b)*v.
 * The reference form transports a tangent at the ORIGIN to point b
 * (lam(0)=2, so PT_{0->b}(v) = (2/lam_b)*v) — this is what
 * waefrebeorn/WuBuMath's wubu_parallel_transport_to_p does (gyration-free at the
 * origin). Check #5 in test_crossval.c verifies eshkol's rescale agrees with the
 * reference to ~1e-2 (the gyration term WuBuMath adds for a->b is the residual). */
void parallel_transport_eshkol(float*o,const float*a,const float*b,const float*v,int n,float c);
void parallel_transport_ref(float*o,const float*b,const float*v,int n,float c);

#endif
