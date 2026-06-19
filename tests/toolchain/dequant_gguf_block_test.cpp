#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
// ---- copied verbatim from runtime_tensor_quant.cpp ----
static float half_bits_to_float(uint16_t h){uint32_t sign=(uint32_t)(h&0x8000u)<<16,exp=(h>>10)&0x1Fu,mant=h&0x3FFu,f;
 if(exp==0){if(mant==0)f=sign;else{uint32_t e=127-15+1;while((mant&0x400u)==0){mant<<=1;e--;}mant&=0x3FFu;f=sign|(e<<23)|(mant<<13);}}
 else if(exp==0x1Fu)f=sign|0x7F800000u|(mant<<13);else f=sign|((exp+(127-15))<<23)|(mant<<13);
 float o;std::memcpy(&o,&f,4);return o;}
static inline uint16_t rh(const uint8_t*p){return (uint16_t)((uint16_t)p[0]|((uint16_t)p[1]<<8));}
void eshkol_dequant_q8_0(const uint8_t*blocks,int64_t*out,int64_t n){if(!blocks||!out||n<=0)return;double*o=(double*)out;const int QK=32,BS=34;int64_t nb=(n+QK-1)/QK;
 for(int64_t b=0;b<nb;b++){const uint8_t*blk=blocks+b*BS;float d=half_bits_to_float(rh(blk));const int8_t*qs=(const int8_t*)(blk+2);for(int j=0;j<QK;j++){int64_t i=b*QK+j;if(i>=n)return;o[i]=(double)(d*(float)qs[j]);}}}
void eshkol_dequant_q4_0(const uint8_t*blocks,int64_t*out,int64_t n){if(!blocks||!out||n<=0)return;double*o=(double*)out;const int QK=32,BS=18;int64_t nb=(n+QK-1)/QK;
 for(int64_t b=0;b<nb;b++){const uint8_t*blk=blocks+b*BS;float d=half_bits_to_float(rh(blk));const uint8_t*qs=blk+2;for(int j=0;j<16;j++){int x0=(int)(qs[j]&0x0F)-8,x1=(int)(qs[j]>>4)-8;int64_t i0=b*QK+j,i1=b*QK+j+16;if(i0<n)o[i0]=(double)(d*(float)x0);if(i1<n)o[i1]=(double)(d*(float)x1);}}}

int main(){
  int fails=0;
  // q8_0: d=1.0 (0x3C00), qs[j]=j-16
  uint8_t blk8[34]; blk8[0]=0x00; blk8[1]=0x3C;
  for(int j=0;j<32;j++) blk8[2+j]=(uint8_t)(int8_t)(j-16);
  double out8[32]; eshkol_dequant_q8_0(blk8,(int64_t*)out8,32);
  for(int j=0;j<32;j++){ double exp=(double)(j-16); if(out8[j]!=exp){printf("q8 FAIL j=%d got %g exp %g\n",j,out8[j],exp);fails++;} }
  // q8_0 with d=2.0 (0x4000)
  blk8[1]=0x40; double out8b[32]; eshkol_dequant_q8_0(blk8,(int64_t*)out8b,32);
  for(int j=0;j<32;j++){ double exp=2.0*(j-16); if(out8b[j]!=exp){printf("q8d2 FAIL j=%d got %g exp %g\n",j,out8b[j],exp);fails++;} }
  // q4_0: d=1.0, low nibble[j]=j (0..15) -> x[j]=j-8 ; high nibble[j]=15-j -> x[j+16]=(15-j)-8=7-j
  uint8_t blk4[18]; blk4[0]=0x00; blk4[1]=0x3C;
  for(int j=0;j<16;j++){ uint8_t lo=(uint8_t)j, hi=(uint8_t)(15-j); blk4[2+j]=(uint8_t)((hi<<4)|lo); }
  double out4[32]; eshkol_dequant_q4_0(blk4,(int64_t*)out4,32);
  for(int j=0;j<16;j++){ double e0=(double)(j-8); if(out4[j]!=e0){printf("q4lo FAIL j=%d got %g exp %g\n",j,out4[j],e0);fails++;} }
  for(int j=0;j<16;j++){ double e1=(double)(7-j); if(out4[16+j]!=e1){printf("q4hi FAIL j=%d got %g exp %g\n",j,out4[16+j],e1);fails++;} }
  // partial n (not block-aligned): n=20 q8 should fill exactly 20, not overrun
  double out20[20]; for(int i=0;i<20;i++)out20[i]=-999; blk8[1]=0x3C; eshkol_dequant_q8_0(blk8,(int64_t*)out20,20);
  for(int j=0;j<20;j++){ double e=(double)(j-16); if(out20[j]!=e){printf("q8n20 FAIL j=%d\n",j);fails++;} }
  printf(fails?"FAILURES=%d\n":"ALL-DEQUANT-OK\n",fails); return fails;
}
