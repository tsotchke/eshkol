// Device-independent exercise of the real Emscripten WebGPU C backend.
// No mock reports GPU success; with no attached page bridge, GPU entry points
// refuse and ordinary callers retain their CPU fallback.
#include <eshkol/backend/gpu/gpu_memory.h>
#include <eshkol/logger.h>
#include <emscripten.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#ifdef NDEBUG
#error "Assertions must remain enabled in this contract test"
#endif
extern "C" void eshkol_printf(eshkol_logger_t, const char*, ...) {}
static int batch_calls = 0;
extern "C" void eshkol_batch_matmul_f64(const double *a, const double *b, double *c,
                                        int64_t batch, int64_t m, int64_t k, int64_t n) {
    batch_calls++;
    for (int64_t q=0; q<batch; ++q) for (int64_t i=0; i<m; ++i) for (int64_t j=0; j<n; ++j) {
        double sum=0;
        for (int64_t t=0; t<k; ++t) sum += a[q*m*k+i*k+t]*b[q*k*n+t*n+j];
        c[q*m*n+i*n+j]=sum;
    }
}
static void assert_product(const double *c) {
    assert(c[0]==19 && c[1]==22 && c[2]==43 && c[3]==50);
}
int main() {
    double a[]={1,2,3,4}, b[]={5,6,7,8}, c[]={-9,-9,-9,-9};
    EshkolGPUBuffer aa{},bb{},cc{};
    assert(eshkol_gpu_wrap_host(a,sizeof(a),&aa)==0);
    assert(eshkol_gpu_wrap_host(b,sizeof(b),&bb)==0);
    assert(eshkol_gpu_wrap_host(c,sizeof(c),&cc)==0);
    eshkol_gpu_init();
    assert(eshkol_gpu_get_backend()==ESHKOL_GPU_NONE);
    assert(eshkol_gpu_should_use(1)==0);
    assert(eshkol_gpu_matmul_f64(&aa,&bb,&cc,2,2,2)==-1);
    for (double x:c) assert(x==-9);
    eshkol_matmul_dispatch(a,b,c,2,2,2,0);
    assert_product(c);
    eshkol_batch_matmul_dispatch(a,b,c,1,2,2,2,0);
    assert(batch_calls==1); assert_product(c);
    eshkol_gpu_free(&aa); eshkol_gpu_free(&bb); eshkol_gpu_free(&cc);
    assert(a[0]==1 && b[0]==5); // wrapping never transferred ownership
    puts("PASS WebGPU WASM no-device CPU fallback");
}
