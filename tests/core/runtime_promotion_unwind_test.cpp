// Native longjmp tests deliberately keep only POD state across setjmp. All
// handlers exist before injection. No user dynamic-wind callback is installed.
#include "../../lib/core/arena_memory.h"
#include "../../lib/core/runtime_region_promotion_internal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <setjmp.h>
using Value=eshkol_tagged_value_t;
static void check(bool ok,const char* text){if(!ok){std::fprintf(stderr,"FAIL: %s\n",text);std::abort();}}
static Value integer(int64_t n){Value v{};v.type=ESHKOL_VALUE_INT64;v.data.int_val=n;return v;}
static Value vector(arena_t* arena){
    void* p=arena_allocate_vector_with_header(arena,1);check(p,"fixture vector");
    *static_cast<uint64_t*>(p)=1;
    *reinterpret_cast<Value*>(static_cast<char*>(p)+8)=integer(19);
    Value v{};v.type=ESHKOL_VALUE_HEAP_PTR;v.data.ptr_val=reinterpret_cast<uint64_t>(p);return v;
}
static Value kept[2],original[2];
static void no_temporary_live(){auto s=eshkol_promotion_test_snapshot();for(int i=0;i<4;++i)check(s.live_bytes[i]==0,"native transaction scratch released before longjmp");}
static void local_failure(bool recycle){
    int status=-1;
    const uint64_t mark=eshkol_region_mark();
    const int64_t handle=eshkol_region_handle_open("failure-stays-live",4096,1,&status);
    check(status==0&&handle,"open reclaiming handle");
    eshkol_region_t* region=region_current();
    kept[0]=vector(region->arena);kept[1]=vector(region->arena);std::memcpy(original,kept,sizeof kept);
    arena_block_t* block=region->arena->current_block;const size_t used=block->used;
    jmp_buf handler;eshkol_push_exception_handler(&handler);
    eshkol_promotion_test_emergency_reset();
    if(setjmp(handler)==0){
        // Allow one target copy, then fail the second kept root. This detects
        // per-root commit/publication and early source reset/handle retirement.
        eshkol_promotion_test_arm(4,1);
        if(recycle)eshkol_iter_nursery_recycle(region,kept,2);
        else eshkol_region_unwind_to(mark,kept,2);
        check(false,"failed operation must transfer");
    }
    check(region_current()==region&&eshkol_region_mark()==mark+1,"failed current level remains live");
    check(eshkol_region_handle_live(handle),"failed level handle not retired");
    check(region->arena->current_block==block&&block->used==used,"failed source not reset");
    check(!std::memcmp(kept,original,sizeof kept),"all kept outputs unchanged");
    check(region->fwd_map==nullptr&&region->escape_count==0,"failed batch not committed");
    check(eshkol_promotion_test_emergency_transfers()==1,"exactly one fixed emergency transfer");
    no_temporary_live();
    eshkol_pop_exception_handler();eshkol_promotion_test_reset();
    check(eshkol_region_handle_close(handle,kept,2)==0,"retry closes handle");
    check(!eshkol_region_handle_live(handle)&&eshkol_region_mark()==mark,"successful retry retires handle");
    check(kept[0].data.ptr_val!=original[0].data.ptr_val&&kept[1].data.ptr_val!=original[1].data.ptr_val,"retry copied both roots");
    for(auto v:kept)check(*reinterpret_cast<uint64_t*>(v.data.ptr_val)==1,"committed kept values survive poisoned close");
}
static void emergency_unwind(){
    const uint64_t mark=eshkol_region_mark();
    jmp_buf handler;eshkol_push_exception_handler(&handler);
    eshkol_promotion_test_emergency_reset();
    if(setjmp(handler)==0){
        int status=-1;
        check(eshkol_region_handle_open("outer-emergency",4096,1,&status)&&status==0,"outer handle");
        check(eshkol_region_handle_open("inner-emergency",4096,1,&status)&&status==0,"inner handle");
        kept[0]=vector(region_current()->arena);kept[1]=vector(region_current()->arena);
        // Continue denying every promotion bookkeeping allocation through the
        // emergency transfer and both ordinary region-unwind levels.
        eshkol_promotion_test_arm(-1,0);
        eshkol_region_unwind_to(mark,kept,2);
        check(false,"unwind promotion must transfer");
    }
    check(eshkol_region_mark()==mark,"emergency reached outer handler after nested unwind");
    auto s=eshkol_promotion_test_snapshot();uint64_t attempts=0;
    for(int i=0;i<4;++i)attempts+=s.attempts[i];
    check(attempts==1&&s.attempts[4]==0,"emergency unwind attempted no further promotion allocation");
    check(g_current_exception!=nullptr,"fixed exception delivered");check(eshkol_promotion_test_emergency_transfers()==1,"exactly one fixed emergency transfer");
    no_temporary_live();
    eshkol_pop_exception_handler();eshkol_promotion_test_reset();
}
int main(){check(get_global_arena_shared(),"global arena");local_failure(false);local_failure(true);emergency_unwind();std::puts("PASS native promotion failure/recycle/unwind");}
