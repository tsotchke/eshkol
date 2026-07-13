#include <stdio.h>
#include <math.h>
#include "wubu_poincare_geom.h"
static float rnd(void){static unsigned s=4242;s=s*1664525u+1013904223u;return ((s>>8)&0xffff)/65535.0f*2.0f-1.0f;}

/* exp_0: move v from origin */
void exp0(float*o,const float*v,int n,float c){
  float vn=wpg_vnorm(v,n); if(vn<WPG_EPS){memset(o,0,n*sizeof(float));return;}
  float arg=0.5f*sqrtf(c)*vn; float t=tanhf(arg);
  float factor=t/(sqrtf(c)*vn+WPG_EPS);
  wpg_vscale(o,v,factor,n);
}
/* exp_p candidates */
void exp_p_A(float*o,const float*p,const float*v,int n,float c){ /* standard textbook: no lam */
  float vn=wpg_vnorm(v,n); if(vn<WPG_EPS){memcpy(o,p,n*sizeof(float));return;}
  float arg=0.5f*sqrtf(c)*vn; float t=tanhf(arg);
  float factor=t/(sqrtf(c)*vn+WPG_EPS);
  float s[64]; wpg_vscale(s,v,factor,n); wpg_mobius_add(o,p,s,n,c);
}
void exp_p_B(float*o,const float*p,const float*v,int n,float c){ /* lam/2 */
  float vn=wpg_vnorm(v,n); if(vn<WPG_EPS){memcpy(o,p,n*sizeof(float));return;}
  float p2=wpg_vdot(p,p,n); float lam=2.0f/(1.0f-c*p2+WPG_EPS);
  float arg=0.5f*sqrtf(c)*vn; float t=tanhf(arg);
  float factor=(lam/2.0f)*t/(vn+WPG_EPS);
  float s[64]; wpg_vscale(s,v,factor,n); wpg_mobius_add(o,p,s,n,c);
}
int main(void){
  const int N=4; float c=1.0f;
  /* 1. True geodesic length from origin */
  float d0ref=-1;
  for(int k=1;k<=6;k++){
    float v[4]; for(int i=0;i<N;i++)v[i]=k*0.15f*rnd();
    float q[4]; exp0(q,v,N,c);
    float d=eshkol_distance(q,q+0,N,c); /* distance from 0: use eshkol_distance(0,q) */
    float d0=eshkol_distance((float[4]){0,0,0,0},q,N,c);
    float vn=wpg_vnorm(v,N);
    if(d0ref<0)d0ref=d0/vn;
    printf("exp0: |v|=%.3f d(0,exp0)=%.4f d/|v|=%.4f\n",vn,d0,d0/vn);
  }
  printf("  => invariant d(0,exp0(v))/|v| = %.4f (expect ~sqrt(c)=1.0)\n\n",d0ref);
  /* 2. test exp_p_A (no lam) */
  float p[4]; float p2=2.0f; while(p2>0.4f){p2=0;for(int i=0;i<N;i++){p[i]=0.35f*rnd();p2+=p[i]*p[i];}}
  printf("base |p|=%.4f  exp_p_A (textbook no-lam):\n",wpg_vnorm(p,N));
  for(int k=1;k<=5;k++){float v[4];for(int i=0;i<N;i++)v[i]=k*0.15f*rnd();float q[4];exp_p_A(q,p,v,N,c);float d=eshkol_distance(p,q,N,c);float vn=wpg_vnorm(v,N);printf("  |v|=%.3f d=%.4f d/|v|=%.4f\n",vn,d,d/vn);}
  printf("exp_p_B (lam/2):\n");
  for(int k=1;k<=5;k++){float v[4];for(int i=0;i<N;i++)v[i]=k*0.15f*rnd();float q[4];exp_p_B(q,p,v,N,c);float d=eshkol_distance(p,q,N,c);float vn=wpg_vnorm(v,N);printf("  |v|=%.3f d=%.4f d/|v|=%.4f\n",vn,d,d/vn);}
  return 0;
}
