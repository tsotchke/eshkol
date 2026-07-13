#include <stdio.h>
#include <math.h>
#include "wubu_poincare_geom.h"
static float rnd(void){static unsigned s=12345;s=s*1664525u+1013904223u;return ((s>>8)&0xffff)/65535.0f*2.0f-1.0f;}
int main(void){
  const int N=4; float c=1.0f;
  /* FIXED base point p; vary tangent magnitude by k, check dist(p,exp_p(k v)) == k*dist(p,exp_p(v)) */
  float p[4]; float p2=2.0f;
  while(p2>0.4f){p2=0;for(int i=0;i<N;i++){p[i]=0.35f*rnd();p2+=p[i]*p[i];}}
  float v0[4]; for(int i=0;i<N;i++)v0[i]=0.3f*rnd();
  float q1[4]; poincare_exp_eshkol(q1,p,v0,N,c); float d1=eshkol_distance(p,q1,N,c);
  printf("base |p|=%.3f\n", wpg_vnorm(p,N));
  int ok=1;
  for(int k=1;k<=6;k++){
    float vk[4]; for(int i=0;i<N;i++)vk[i]=k*0.15f*rnd();
    float qk[4]; poincare_exp_eshkol(qk,p,vk,N,c);
    float dk=eshkol_distance(p,qk,N,c);
    float vnk=wpg_vnorm(vk,N), vn0=wpg_vnorm(v0,N);
    float ratio = dk/vnk;
    if(fabsf(ratio - d1/vn0) > 0.02f){ ok=0; }
    printf("  k ~ |v| norm %.3f  dist=%.4f  dist/|v|=%.4f\n", vnk, dk, ratio);
  }
  printf("exp-map linearity (dist proportional to |v| at fixed base): %s\n", ok?"OK (geodesic invariant holds)":"FAIL (bug)");
  return 0;
}
