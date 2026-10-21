#include <eshkol/core/tensor_inverse_checked.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);return 1; } } while(0)
static double residual(const double* a,const double* x,int n,int reverse) {
    double worst=0;
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) {
        double sum=0;
        for(int k=0;k<n;k++) sum+=reverse?x[i*n+k]*a[k*n+j]:a[i*n+k]*x[k*n+j];
        double error=fabs(sum-(i==j)); if(error>worst)worst=error;
    }
    return worst;
}
int main(void) {
    double a[]={0,2,1,1,0,3,4,1,0},copy[9],work[9],out[9];
    memcpy(copy,a,sizeof(a));
    CHECK(eshkol_tensor_inverse_checked(a,work,out,3)==ESHKOL_TENSOR_INVERSE_OK);
    CHECK(memcmp(a,copy,sizeof(a))==0);
    CHECK(residual(a,out,3,0)<1e-12 && residual(a,out,3,1)<1e-12);
    double tiny[]={1e-200,0,0,2e-200};
    CHECK(eshkol_tensor_inverse_checked(tiny,work,out,2)==ESHKOL_TENSOR_INVERSE_OK);
    CHECK(fabs(out[0]/1e200-1)<1e-14 && fabs(out[3]/5e199-1)<1e-14);
    double singular[]={1,2,2,4},nan_input[]={NAN},inf_input[]={INFINITY};
    double overflow[]={1e-308,1e308,0,1},scalar[]={4};
    CHECK(eshkol_tensor_inverse_checked(scalar,work,out,1)==ESHKOL_TENSOR_INVERSE_OK && out[0]==0.25);
    CHECK(eshkol_tensor_inverse_checked(singular,work,out,2)==ESHKOL_TENSOR_INVERSE_SINGULAR);
    CHECK(eshkol_tensor_inverse_checked(nan_input,work,out,1)==ESHKOL_TENSOR_INVERSE_NONFINITE);
    CHECK(eshkol_tensor_inverse_checked(inf_input,work,out,1)==ESHKOL_TENSOR_INVERSE_NONFINITE);
    CHECK(eshkol_tensor_inverse_checked(overflow,work,out,2)==ESHKOL_TENSOR_INVERSE_NONFINITE);
    CHECK(eshkol_tensor_inverse_checked(a,work,out,INT64_MAX)==ESHKOL_TENSOR_INVERSE_SIZE);
    CHECK(eshkol_tensor_inverse_checked(NULL,work,out,1)==ESHKOL_TENSOR_INVERSE_INVALID);
    CHECK(eshkol_tensor_inverse_checked(a,NULL,out,1)==ESHKOL_TENSOR_INVERSE_INVALID);
    CHECK(eshkol_tensor_inverse_checked(a,work,NULL,1)==ESHKOL_TENSOR_INVERSE_INVALID);
    CHECK(eshkol_tensor_inverse_checked(a,work,out,0)==ESHKOL_TENSOR_INVERSE_INVALID);
    puts("RESULT: PASS"); return 0;
}
