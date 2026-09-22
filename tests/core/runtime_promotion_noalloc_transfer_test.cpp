#include "../../lib/core/arena_memory.h"
#include "../../lib/core/runtime_region_promotion_internal.h"
#include "runtime_promotion_allocation_probe.h"
#include <cstdio>
#include <cstdlib>
#include <setjmp.h>
static void check(bool ok,const char* label){if(!ok){std::fprintf(stderr,"FAIL: %s\n",label);std::abort();}}
static eshkol_tagged_value_t region_vector(){
    void* p=arena_allocate_vector_with_header(region_current()->arena,1);check(p,"fixture allocation");
    *static_cast<uint64_t*>(p)=1;
    auto* slot=reinterpret_cast<eshkol_tagged_value_t*>(static_cast<char*>(p)+8);
    *slot={};slot->type=ESHKOL_VALUE_INT64;slot->data.int_val=19;
    eshkol_tagged_value_t v{};v.type=ESHKOL_VALUE_HEAP_PTR;v.data.ptr_val=reinterpret_cast<uint64_t>(p);return v;
}
static eshkol_promotion_test_stats before;
int main(){
    check(get_global_arena_shared(),"global arena");
    jmp_buf handler;eshkol_push_exception_handler(&handler);
    eshkol_promotion_test_emergency_reset();
    if(setjmp(handler)==0){
        auto* outer=region_create("noalloc-outer",4096);check(outer,"outer region");region_push(outer);
        auto* inner=region_create("noalloc-inner",4096);check(inner,"inner region");region_push(inner);
        auto input=region_vector();eshkol_tagged_value_t output{};
        eshkol_promotion_test_arm(-1,0);
        int32_t status=eshkol_region_write_barrier_checked_v1(&output,nullptr,&input);
        check(status==1,"checked failure returned normally");
        before=eshkol_promotion_test_snapshot();
        for(int i=0;i<4;++i)check(before.live_bytes[i]==0,"transaction cleanup precedes probe interval");
        // Only now start denying wrapper-covered allocations. C++ transaction
        // failure and __cxa throw handling completed before this interval.
        promotion_allocation_probe_arm();
        eshkol_runtime_emergency_raise_v1(status);
    }
    const auto allocation=promotion_allocation_probe_disarm();
    check(allocation.malloc_calls==0&&allocation.calloc_calls==0&&allocation.realloc_calls==0&&
          allocation.aligned_alloc_calls==0&&allocation.posix_memalign_calls==0&&
          allocation.new_calls==0&&allocation.new_array_calls==0,"transfer made no wrapper-covered allocation attempts");
    check(allocation.exception_allocations==0,"transfer did not allocate a C++ exception");
    const auto after=eshkol_promotion_test_snapshot();
    for(int i=0;i<5;++i)check(before.attempts[i]==after.attempts[i],"root-owned emergency unwind allocated no promotion bookkeeping or target");
    check(eshkol_region_mark()==0,"nested regions unwound");
    check(eshkol_promotion_test_emergency_transfers()==1,"one emergency transfer");
    check(g_current_exception!=nullptr,"handler received emergency");
    eshkol_pop_exception_handler();eshkol_promotion_test_reset();
    std::puts("PASS checked-return-to-nearest-handler interval: direct linked malloc/calloc/realloc/aligned_alloc/posix_memalign/new/new[] attempts=0; __cxa_allocate_exception=0; shared-library internal allocations not intercepted");
}
