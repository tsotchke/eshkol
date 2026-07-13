#include <stdio.h>
#include <math.h>
#include "wubu_poincare_geom.h"
static float rnd(void){static unsigned s=777;s=s*1664525u+1013904223u;return ((s>>8)&0xffff)/65535.0f*2.0f-1.0f;}
/* candidate exp_p using lam(p)=2/(1-c|p|^2) and the standard paired formula */
void exp_p_cand(float*o,const float*p,const float*v,int n,float c){
  if(c<=0){wpg_vadd(o,p,v,n);return;}
  float vn=wpg_vnorm(v,n); if(vn<WPG_EPS){memcpy(o,p,n*sizeof(float));return;}
  float p2=wpg_vdot(p,p,n);
  float lam=2.0f/(1.0f-c*p2+WPG_EPS);
  /* standard: scalar = (lam/2)*tanh(0.5*sqrt(c)*vn) / vn ; then p (+) scalar*v */
  float scalar=(lam/2.0f)*tanhf(0.5f*sqrtf(c)*vn)/(vn+WPG_EPS);
  float s[64]; wpg_vscale(s,v,scalar,n);
  wpg_mobius_add(o,p,s,n,c);
}
int main(void){
  const int N=4; float c=1.0f;
  /* fixed base p, sweep |v|, expect dist(p,exp_p(v))=2|v| exactly (WuBuMath convention) */
  float p[4]; float p2=2.0f;
  while(p2>0.4f){p2=0;for(int i=0;i<N;i++){p[i]=0.35f*rnd();p2+=p[i]*p[i];}}
  printf("base |p|=%.4f\n", wpg_vnorm(p,N));
  int ok=1; float ref=-1;
  for(int k=1;k<=8;k++){
    float v[4]; for(int i=0;i<N;i++)v[i]=k*0.12f*rnd();
    float q[4]; exp_p_cand(q,p,v,N,c);
    float d=eshkol_distance(p,q,N,c);
    float vn=wpg_vnorm(v,N);
    float ratio=d/(2.0f*vn+1e-9f);   /* expect ~1.0 for WuBuMath convention */
    if(ref<0)ref=ratio;
    if(fabsf(ratio-ref)>0.02f)ok=0;
    printf("  |v|=%.3f dist=%.4f dist/(2|v|)=%.4f\n", vn,d,ratio);
  }
  printf("invariant dist/(2|v|) constant? %s\n", ok?"YES -> F1 fixed":"NO");
  return 0;
}
