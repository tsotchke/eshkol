// Native transaction contract tests; build runtime and this test with
// ESHKOL_PROMOTION_TESTING. These are synthetic allocation failures, not host OOM.
#include "../../lib/core/arena_memory.h"
#include "../../lib/core/runtime_region_promotion_internal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <initializer_list>

using Value = eshkol_tagged_value_t;
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::abort(); }
}
static Value integer(int64_t n) { Value v{}; v.type=ESHKOL_VALUE_INT64; v.data.int_val=n; return v; }
static Value pointer(void* p) { Value v{}; v.type=ESHKOL_VALUE_HEAP_PTR; v.data.ptr_val=reinterpret_cast<uint64_t>(p); return v; }
static void* address(Value v) { return reinterpret_cast<void*>(v.data.ptr_val); }
static Value* slots(void* p) { return reinterpret_cast<Value*>(static_cast<char*>(p)+8); }
static bool equal(Value a, Value b) { return std::memcmp(&a,&b,sizeof a)==0; }
static void* vector(arena_t* a, size_t n) {
    void* p=arena_allocate_vector_with_header(a,n); check(p,"fixture allocation");
    *static_cast<uint64_t*>(p)=n;
    for(size_t i=0;i<n;++i) slots(p)[i]=integer(100+i);
    return p;
}
struct Graph {
    void* objects[6];
    unsigned char bytes[6][128];
    size_t sizes[6];
    explicit Graph(arena_t* a) {
        objects[0]=vector(a,4);
        objects[1]=arena_allocate_cons_with_header(a);
        check(objects[1],"cons fixture");
        objects[2]=vector(a,2); objects[3]=vector(a,1);
        objects[4]=vector(a,2); objects[5]=vector(a,1);
        slots(objects[0])[0]=pointer(objects[1]); slots(objects[0])[1]=pointer(objects[2]);
        slots(objects[0])[2]=pointer(objects[4]);
        auto* cell=static_cast<arena_tagged_cons_cell_t*>(objects[1]);
        cell->car=pointer(objects[2]); cell->cdr=pointer(objects[0]);
        slots(objects[2])[0]=pointer(objects[3]);
        slots(objects[4])[0]=pointer(objects[5]); slots(objects[5])[0]=pointer(objects[0]);
        for(size_t i=0;i<6;++i) {
            sizes[i]=sizeof(eshkol_object_header_t)+ESHKOL_GET_HEADER(objects[i])->size;
            check(sizes[i]<=sizeof bytes[i],"snapshot capacity");
            std::memcpy(bytes[i],ESHKOL_GET_HEADER(objects[i]),sizes[i]);
        }
    }
    void unchanged() const {
        for(size_t i=0;i<6;++i) check(!std::memcmp(bytes[i],ESHKOL_GET_HEADER(objects[i]),sizes[i]),"source bytes unchanged");
    }
    void verify(Value promoted, Value tail) const {
        void* p=address(promoted); check(p!=objects[0],"root copied");
        auto* cell=static_cast<arena_tagged_cons_cell_t*>(address(slots(p)[0]));
        check(cell!=objects[1],"cons copied");
        check(address(cell->cdr)==p,"cons cycle closes");
        check(equal(cell->car,slots(p)[1]),"shared tail preserved");
        check(equal(tail,cell->car),"multiroot alias preserved");
        void* t=address(cell->car); check(t!=objects[2],"tail copied");
        void* token=address(slots(t)[0]); check(token!=objects[3],"token copied");
        check(slots(token)[0].data.int_val==100,"leaf payload preserved");
        void* n=address(slots(p)[2]); check(n!=objects[4],"branch copied");
        void* h=address(slots(n)[0]); check(h!=objects[5],"cycle node copied");
        check(address(slots(h)[0])==p,"vector cycle closes");
    }
};
struct Fixture {
    eshkol_region_t *outer,*source;
    void* holder;
    explicit Fixture(size_t capacity=65536) {
        outer=region_create("promotion-target",capacity); check(outer,"outer create"); region_push(outer);
        holder=vector(outer->arena,2);
        source=region_create("promotion-source",4096); check(source,"source create"); region_push(source);
    }
    ~Fixture(){ eshkol_promotion_test_reset(); region_pop(); region_pop(); }
};
static void same_live(const decltype(eshkol_promotion_test_snapshot())& before,
                      const decltype(eshkol_promotion_test_snapshot())& after) {
    for(int i=0;i<4;++i) check(before.live_bytes[i]==after.live_bytes[i],"temporary allocations returned to baseline");
}
static size_t matrix(int site,bool seeded,bool batch) {
    size_t failures=0, retained=0;
    uint64_t copy_bytes=0, peak[4]={};
    for(int64_t nth=0;nth<256;++nth) {
        Fixture f;
        Graph g(f.source->arena);
        Value seeded_value=pointer(vector(f.source->arena,1)), committed=integer(0);
        if(seeded) check(eshkol_region_write_barrier_checked_v1(&committed,f.holder,&seeded_value)==0,"seed map");
        void* map=f.source->fwd_map; arena_t* target=f.source->fwd_target;
        size_t escapes=f.source->escape_count;
        Value input[2]={pointer(g.objects[0]),pointer(g.objects[2])};
        Value output[2]={integer(73),integer(79)}, sentinel[2]; std::memcpy(sentinel,output,sizeof output);
        Value holder[2]; std::memcpy(holder,slots(f.holder),sizeof holder);
        arena_block_t* block=f.outer->arena->current_block;
        const size_t used=block->used, reserved=f.outer->arena->total_allocated;
        // Zero alignment gaps before the transaction; any copied source pointer
        // left in the retained allocation interval then makes this check fail.
        std::memset(block->memory+used,0,block->size-used);
        eshkol_promotion_test_arm(site,nth);
        auto before=eshkol_promotion_test_snapshot();
        int32_t status=batch?eshkol_region_copy_tagged_checked(output,f.holder,input,2):
                             eshkol_region_write_barrier_checked_v1(output,f.holder,input);
        auto after=eshkol_promotion_test_snapshot();
        eshkol_promotion_test_reset();
        g.unchanged();
        check(!std::memcmp(holder,slots(f.holder),sizeof holder),"destination unchanged by staging");
        if(status==0) {
            check(after.attempts[site]<=static_cast<uint64_t>(nth),"matrix ended at first uninjected run");
            Value tail=batch?output[1]:slots(address(output[0]))[1]; g.verify(output[0],tail);
            break;
        }
        check(status==1,"injected failure status"); ++failures;
        check(!std::memcmp(output,sentinel,sizeof output),"failed outputs byte-for-byte unchanged");
        check(f.source->fwd_map==map&&f.source->fwd_target==target,"failed persistent map identity unchanged");
        check(f.source->escape_count==escapes,"failed committed copy count unchanged"); same_live(before,after);
        check(f.outer->arena->current_block==block,"bounded fixture did not grow target");
        check(f.outer->arena->total_allocated==reserved,"target reserved bytes unchanged");
        const size_t delta=block->used-used; retained+=delta; copy_bytes+=after.target_bytes;
        for(int i=0;i<4;++i) if(after.peak_bytes[i]>peak[i]) peak[i]=after.peak_bytes[i];
        check(delta>=after.target_bytes,"target byte accounting includes all copied bytes");
        for(size_t i=used;i<block->used;++i) check(block->memory[i]==0,"failed speculative bytes scrubbed");
        if(seeded) {
            Value again=integer(0); eshkol_promotion_test_arm(-1,0);
            check(eshkol_region_write_barrier_checked_v1(&again,f.holder,&seeded_value)==0,"old forwarding lookup bypasses allocation");
            check(equal(again,committed),"old canonical entry preserved"); eshkol_promotion_test_reset();
        }
        check(eshkol_region_copy_tagged_checked(output,f.holder,input,2)==0,"retry succeeds");
        g.verify(output[0],output[1]); g.unchanged();
        // Retire the source before reading only the known committed graph.
        region_pop(); f.source=region_create("empty-replacement",4096); region_push(f.source);
        g.verify(output[0],output[1]);
        check(nth<255,"failpoint matrix terminated");
    }
    std::printf("site=%d seeded=%d batch=%d failed_prefixes=%zu retained_sum=%zu copy_bytes=%llu peak_native=%llu,%llu,%llu,%llu reserved_delta=0 block_delta=0\n",site,seeded,batch,failures,retained,static_cast<unsigned long long>(copy_bytes),static_cast<unsigned long long>(peak[0]),static_cast<unsigned long long>(peak[1]),static_cast<unsigned long long>(peak[2]),static_cast<unsigned long long>(peak[3]));
    return failures;
}
static void bounded_target_failure() {
    for(size_t budget: {size_t(0),size_t(80)}) {
        Fixture f; Graph g(f.source->arena);
        Value input=pointer(g.objects[0]), output=integer(73);
        auto* block=f.outer->arena->current_block;
        const size_t size=block->size, used=block->used;
        const bool bounded=f.outer->arena->bounded;
        check(size-used>=budget,"bounded test capacity");
        std::memset(block->memory+used,0,budget);
        // Restrict existing capacity; do not change used bytes or rewind a target.
        block->size=used+budget;f.outer->arena->bounded=true;
        eshkol_promotion_test_reset();
        const int32_t status=eshkol_region_write_barrier_checked_v1(&output,f.holder,&input);
        block->size=size;f.outer->arena->bounded=bounded;
        check(status==1&&equal(output,integer(73)),"real bounded arena null becomes checked failure");
        check(f.source->fwd_map==nullptr&&f.source->escape_count==0,"real target failure preserves map/count");
        for(size_t i=used;i<block->used;++i)check(block->memory[i]==0,"real failed target spans scrubbed");
        g.unchanged();
        std::printf("real_bounded_budget=%zu retained=%zu\n",budget,block->used-used);
    }
}
static void target_switch() {
    Fixture f; Graph g(f.source->arena);
    Value input=pointer(g.objects[0]), first=integer(0), output=integer(81);
    void* global_holder=vector(get_global_arena_shared(),1);
    check(eshkol_region_write_barrier_checked_v1(&first,f.holder,&input)==0,"first target promotion");
    void* map=f.source->fwd_map; auto* target=f.source->fwd_target;
    eshkol_promotion_test_arm(4,1);
    check(eshkol_region_write_barrier_checked_v1(&output,global_holder,&input)==1,"failed target switch");
    check(equal(output,integer(81))&&f.source->fwd_map==map&&f.source->fwd_target==target,"failed switch preserves map target and output");
    eshkol_promotion_test_reset();
    check(eshkol_region_write_barrier_checked_v1(&output,f.holder,&input)==0&&equal(output,first),"former target lookup survives failure");
    check(eshkol_region_write_barrier_checked_v1(&output,global_holder,&input)==0,"target switch retry");
    check(!equal(output,first)&&f.source->fwd_target==get_global_arena_shared(),"successful switch changes canonical target");
    g.verify(output,slots(address(output))[1]);
    // Legacy and checked callers share the committed map in either order.
    Value legacy=integer(0); eshkol_region_write_barrier_into(&legacy,global_holder,&input);
    check(equal(legacy,output),"legacy reuses checked canonical mapping");
    Value other=pointer(vector(f.source->arena,1)); eshkol_region_write_barrier_into(&legacy,global_holder,&other);
    check(eshkol_region_write_barrier_checked_v1(&output,global_holder,&other)==0&&equal(output,legacy),"checked reuses legacy canonical mapping");
}
static void repeated_retention() {
    Fixture f(1<<20); Graph g(f.source->arena);
    Value input=pointer(g.objects[0]), output=integer(73);
    arena_block_t* block=f.outer->arena->current_block;
    const size_t start=block->used, reserved=f.outer->arena->total_allocated;
    size_t step=0;
    std::memset(block->memory+start,0,block->size-start);
    for(size_t i=0;i<1024;++i) {
        const size_t used=block->used;
        eshkol_promotion_test_arm(4,1);
        check(eshkol_region_write_barrier_checked_v1(&output,f.holder,&input)==1,"repeated failure status");
        auto stats=eshkol_promotion_test_snapshot(); eshkol_promotion_test_reset();
        check(equal(output,integer(73))&&f.source->fwd_map==nullptr,"repeated failure leaves no map/output");
        for(int j=0;j<4;++j) check(stats.live_bytes[j]==0,"repeated failure scratch released");
        check(f.outer->arena->current_block==block,"retention fixture fits one target block");
        const size_t delta=block->used-used;
        if(i==0)step=delta;
        check(delta==step&&step>0,"retained failed bytes have measured constant slope");
        for(size_t j=used;j<block->used;++j)check(block->memory[j]==0,"repeated failed copies scrubbed");
    }
    check(f.outer->arena->total_allocated==reserved,"repeated retention reserved bytes unchanged");
    std::printf("repeated_failures=1024 used_delta=%zu retained_bytes_per_failure=%zu reserved_delta=0 block_delta=0\n",block->used-start,step);
    g.unchanged();
}
static void primitive_entry() {}
static void leaf_contracts() {
    Fixture f;
    auto rejected=[&](void* object, uint8_t type, const char* label) {
        Value input=pointer(object), output=integer(73); input.type=type;
        const size_t escapes=f.source->escape_count;
        check(eshkol_region_write_barrier_checked_v1(&output,f.holder,&input)==2,label);
        check(equal(output,integer(73))&&f.source->fwd_map==nullptr&&f.source->escape_count==escapes,"malformed leaf failure atomic");
    };
    void* small=arena_allocate_with_header_zeroed(f.source->arena,16,HEAP_SUBTYPE_I128,0);
    check(small,"i128 fixture"); ESHKOL_GET_HEADER(small)->size=1;
    rejected(small,ESHKOL_VALUE_HEAP_PTR,"undersized known I128 rejected");
    auto* node=static_cast<ad_node_t*>(arena_allocate_with_header_zeroed(f.source->arena,sizeof(ad_node_t),CALLABLE_SUBTYPE_AD_NODE,0));
    check(node,"AD fixture"); ESHKOL_GET_HEADER(node)->size=1;
    rejected(node,ESHKOL_VALUE_CALLABLE,"undersized known AD node rejected");
    ESHKOL_GET_HEADER(node)->size=sizeof(ad_node_t);
    node->shape=static_cast<int64_t*>(arena_allocate(f.source->arena,8));check(node->shape,"shape fixture");
    *node->shape=1;node->ndim=1;
    rejected(node,ESHKOL_VALUE_CALLABLE,"AD node with regional shape is not a leaf");
    node->shape=nullptr;node->ndim=0;
    node->tensor_gradient=small;
    rejected(node,ESHKOL_VALUE_CALLABLE,"AD node with tensor gradient is not a leaf");
    void* text=arena_allocate_with_header(f.source->arena,8,HEAP_SUBTYPE_STRING,0);check(text,"string fixture");
    std::memset(text,'x',8);
    rejected(text,ESHKOL_VALUE_HEAP_PTR,"unterminated string header span rejected");
    auto* primitive=static_cast<eshkol_primitive_t*>(arena_allocate_with_header_zeroed(f.source->arena,sizeof(eshkol_primitive_t),CALLABLE_SUBTYPE_PRIMITIVE,0));
    check(primitive,"primitive fixture");primitive->func_ptr=reinterpret_cast<uint64_t>(&primitive_entry);primitive->name="stable-primitive";
    Value input=pointer(primitive), output=integer(0);input.type=ESHKOL_VALUE_CALLABLE;
    check(eshkol_region_write_barrier_checked_v1(&output,f.holder,&input)==0,"stable primitive valid coverage preserved");
    auto* copy=static_cast<eshkol_primitive_t*>(address(output));
    check(copy!=primitive&&copy->func_ptr==primitive->func_ptr&&copy->name==primitive->name,"primitive metadata preserved");
}
static void invalid_and_fast() {
    Fixture f; Value out=integer(73), input=integer(11);
    check(eshkol_region_write_barrier_checked_v1(nullptr,f.holder,&input)==4,"null output rejected");
    check(eshkol_region_write_barrier_checked_v1(&out,f.holder,nullptr)==4&&equal(out,integer(73)),"null input rejected atomically");
    void* p=vector(f.source->arena,1); input=pointer(p);
    auto header=*ESHKOL_GET_HEADER(p);
    ESHKOL_GET_HEADER(p)->subtype=255;
    check(eshkol_region_write_barrier_checked_v1(&out,f.holder,&input)==2&&equal(out,integer(73)),"unknown layout rejected");
    *ESHKOL_GET_HEADER(p)=header;
    ESHKOL_GET_HEADER(p)->size=0;
    check(eshkol_region_write_barrier_checked_v1(&out,f.holder,&input)==2&&equal(out,integer(73)),"zero header size rejected");
    ESHKOL_GET_HEADER(p)->size=std::numeric_limits<uint32_t>::max();
    check(eshkol_region_write_barrier_checked_v1(&out,f.holder,&input)==2&&equal(out,integer(73)),"span beyond source arena rejected");
    *ESHKOL_GET_HEADER(p)=header;
    *static_cast<uint64_t*>(p)=2;
    check(eshkol_region_write_barrier_checked_v1(&out,f.holder,&input)==2&&equal(out,integer(73)),"vector count beyond payload rejected");
    *static_cast<uint64_t*>(p)=static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    check(eshkol_region_write_barrier_checked_v1(&out,f.holder,&input)==3&&equal(out,integer(73)),"vector span overflow rejected");
    *static_cast<uint64_t*>(p)=1;
    f.source->escape_count=std::numeric_limits<size_t>::max();
    check(eshkol_region_write_barrier_checked_v1(&out,f.holder,&input)==3&&equal(out,integer(73)),"committed counter overflow rejected atomically");
    check(f.source->escape_count==std::numeric_limits<size_t>::max()&&f.source->fwd_map==nullptr,"counter overflow preserves persistent state");
    f.source->escape_count=0;
    check(eshkol_region_copy_tagged_checked(&out,f.holder,&input,std::numeric_limits<uint64_t>::max())==3&&equal(out,integer(73)),"batch count overflow rejected before reading roots");
    check(eshkol_region_copy_tagged_checked(nullptr,nullptr,nullptr,0)==0,"empty batch needs no mandatory array");
    Value fast[4]={integer(12),pointer(nullptr),pointer(f.holder),input};
    for(int site=0;site<5;++site) {
        eshkol_promotion_test_arm(site,0);
        for(size_t n=0;n<8192;++n) for(int i=0;i<4;++i) {
            const void* owner=i==3?p:f.holder;
            check(eshkol_region_write_barrier_checked_v1(&out,owner,&fast[i])==0&&equal(out,fast[i]),"zero-allocation safe fast path");
        }
        auto stats=eshkol_promotion_test_snapshot();
        for(int i=0;i<5;++i) check(stats.attempts[i]==0,"fast paths perform no instrumented allocation");
        check(f.source->fwd_map==nullptr&&f.source->escape_count==0,"fast paths create no map/control");
        eshkol_promotion_test_reset();
    }
}
int main() {
    check(get_global_arena_shared(),"global arena");
    for(int site=0;site<5;++site) {
        // Scratch is a batch-only requirement; scalar engines may stack-stage.
        if(site!=3) check(matrix(site,false,false)>0,"scalar allocation site exercised");
        check(matrix(site,true,true)>0,"seeded batch allocation site exercised");
    }
    bounded_target_failure(); target_switch(); repeated_retention(); leaf_contracts(); invalid_and_fast();
    auto stats=eshkol_promotion_test_snapshot();
    for(int i=0;i<4;++i) check(stats.live_bytes[i]==0,"all test maps and scratch released");
    std::puts("PASS native promotion transaction synthetic failure matrix");
}
