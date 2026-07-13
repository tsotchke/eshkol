/* Independent re-derivation of eshkol exp-map geodesic invariant.
 * Faithful port of manifold.esk lines 107-109 ONLY.
 * No dependency on wubu_poincare_geom.c. */
#include <stdio.h>
#include <math.h>
#include <string.h>

static float rnd(void){static unsigned s=99173;s=s*1664525u+1013904223u;return ((s>>8)&0xffff)/65535.0f*2.0f-1.0f;}

static float vdot(const float*a,const float*b,int n){float s=0;for(int i=0;i<n;i++)s+=a[i]*b[i];return s;}
static float vnorm(const float*a,int n){return sqrtf(vdot(a,a,n));}
static void vscale(float*o,const float*a,float k,int n){for(int i=0;i<n;i++)o[i]=k*a[i];}
/* Mobius addition in B_c^n */
static void mobius_add(float*o,const float*a,const float*b,int n,float c){
    float ab=vdot(a,b,n), na2=vdot(a,a,n), nb2=vdot(b,b,n);
    float den=1.0f-2.0f*c*ab+c*c*na2*nb2;
    float t1[4],t2[4];
    for(int i=0;i<n;i++) t1[i]=(1.0f-2.0f*c*ab+c*c*nb2)*a[i];
    for(int i=0;i<n;i++) t2[i]=(1.0f+c*na2)*b[i];
    for(int i=0;i<n;i++) o[i]=(t1[i]+t2[i])/(den+1e-12f);
}
/* eshkol distance (manifold.esk arccosh form) */
static float dist(const float*a,const float*b,int n,float c){
    float d2=0,na=0,nb=0;
    for(int i=0;i<n;i++){float dd=a[i]-b[i];d2+=dd*dd;na+=a[i]*a[i];nb+=b[i]*b[i];}
    float arg=1.0f+2.0f*c*d2/(((1.0f-c*na)+1e-12f)*(1.0f-c*nb)+1e-12f);
    return acoshf(arg>1.0f?arg:1.0f);
}
/* eshkol exp-map EXACT from manifold.esk 107-109 */
static void exp_eshkol(float*o,const float*p,const float*v,int n,float c){
    float vn=vnorm(v,n);
    if(vn<1e-8f){memcpy(o,p,n*sizeof(float));return;}
    float lam=2.0f/(1.0f-c*vdot(p,p,n));
    float factor=tanhf(0.5f*lam*vn)/vn;
    float s[4]; vscale(s,v,factor,n);
    mobius_add(o,p,s,n,c);
}

int main(void){
    int N=4; float c=1.0f;
    /* check at a FIXED base, varying |v|, see if dist/|v| is constant */
    float p[4]; float p2=2.0f; while(p2>0.4f){p2=0;for(int i=0;i<N;i++){p[i]=0.35f*rnd();p2+=p[i]*p[i];}}
    printf("base p = (%.3f,%.3f,%.3f,%.3f)  |p|=%.3f\n",p[0],p[1],p[2],p[3],sqrtf(p2));
    float vdir[4]; for(int i=0;i<N;i++) vdir[i]=rnd();
    float vn0=vnorm(vdir,N); for(int i=0;i<N;i++) vdir[i]/=vn0;
    printf("\n=== eshkol exp-map: dist(p,exp_p(t*vdir)) / (t) for t in [0.05,0.6] ===\n");
    float prev_ratio=-1;
    for(int t=1;t<=12;t++){
        float tt=0.05f*t;
        float v[4]; for(int i=0;i<N;i++) v[i]=tt*vdir[i];
        float q[4]; exp_eshkol(q,p,v,N,c);
        float d=dist(p,q,N,c);
        float ratio=d/tt;
        printf("  t=%.3f  |v|=%.3f  dist=%.5f  dist/|v|=%.5f  %s\n", tt, tt, d, ratio,
               (prev_ratio>0 && fabsf(ratio-prev_ratio)>0.15f)?"  <-- DRIFT>0.15":"");
        prev_ratio=ratio;
    }
    /* analytic expectation: for Poincaré, dist(p, exp_p(v)) = 2*asinh(|v|_hyp)/?; 
       known identity: dist(0,exp_0(v)) = 2*atanh(|v|) for c=1? check origin */
    printf("\n=== origin check: dist(0,exp_0(v)) vs 2*atanh(|v|) ===\n");
    float z[4]={0,0,0,0};
    for(int t=1;t<=6;t++){
        float tt=0.1f*t; float v[4]; for(int i=0;i<N;i++) v[i]=(i==0)?tt:0.0f;
        float q[4]; exp_eshkol(q,z,v,N,c);
        float d=dist(z,q,N,c);
        printf("  |v|=%.3f  dist=%.5f  2*atanh(|v|)=%.5f  2*|v|=%.5f\n", tt, d, 2*atanhf(tt), 2*tt);
    }
    return 0;
}
