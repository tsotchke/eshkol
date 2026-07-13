#include <stdio.h>
#include <math.h>
#include "wubu_poincare_geom.h"
static float rnd(void){static unsigned s=12345;s=s*1664525u+1013904223u;return ((s>>8)&0xffff)/65535.0f*2.0f-1.0f;}
int main(void){
  const int N=4; float c=1.0f;
  for(int trial=0;trial<3;trial++){
    float p[4],v[4],q[4]; float p2=2.0f;
    while(p2>0.4f){p2=0;for(int i=0;i<N;i++){p[i]=0.35f*rnd();p2+=p[i]*p[i];}}
    for(int i=0;i<N;i++)v[i]=0.3f*rnd();
    float pn=wpg_vnorm(p,N), vn=wpg_vnorm(v,N);
    poincare_exp_eshkol(q,p,v,N,c);
    float de=eshkol_distance(p,q,N,c);
    printf("trial %d: |p|=%.3f |v|=%.3f dist(p,exp_p(v))=%.4f  dist/|v|=%.4f  dist/(2|v|)=%.4f\n",
           trial,pn,vn,de,de/vn,de/(2*vn));
  }
  return 0;
}
