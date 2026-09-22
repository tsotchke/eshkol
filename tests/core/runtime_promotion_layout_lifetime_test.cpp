#include "../../lib/core/arena_memory.h"
#include "../../lib/core/runtime_region_promotion_internal.h"
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
using Value=eshkol_tagged_value_t;
static void check(bool ok,const char* text){if(!ok){std::fprintf(stderr,"FAIL: %s\n",text);std::abort();}}
static Value integer(int64_t n){Value v{};v.type=ESHKOL_VALUE_INT64;v.data.int_val=n;return v;}
static Value pointer(void* p,uint8_t type=ESHKOL_VALUE_HEAP_PTR){Value v{};v.type=type;v.data.ptr_val=reinterpret_cast<uint64_t>(p);return v;}
static void closure_sexpr(size_t captures){
    auto* source=region_create("closure-sexpr",4096);check(source,"source region");region_push(source);
    auto* sexpr=arena_allocate_cons_with_header(source->arena);check(sexpr,"sexpr cell");
    sexpr->car=integer(99);sexpr->cdr={};
    auto* closure=arena_allocate_closure_with_header(source->arena,0,captures,reinterpret_cast<uint64_t>(sexpr),0,"lifetime-probe");
    check(closure,"closure fixture");
    if(captures){check(closure->env,"closure environment");closure->env->captures[0]=pointer(sexpr);}
    const uint8_t expected=captures?CALLABLE_SUBTYPE_CLOSURE:CALLABLE_SUBTYPE_LAMBDA_SEXPR;
    check(ESHKOL_GET_HEADER(closure)->subtype==expected,"actual producer callable subtype");
    Value input=pointer(closure,ESHKOL_VALUE_CALLABLE),output=integer(73);
    check(eshkol_region_write_barrier_checked_v1(&output,nullptr,&input)==0,"valid produced closure promotes");
    auto* copy=reinterpret_cast<eshkol_closure_t*>(output.data.ptr_val);
    check(copy!=closure&&copy->sexpr_ptr!=reinterpret_cast<uint64_t>(sexpr),"closure sexpr graph promoted");
    if(captures)check(copy->env->captures[0].data.ptr_val==copy->sexpr_ptr,"sexpr/capture alias preserved");
    check(closure->sexpr_ptr==reinterpret_cast<uint64_t>(sexpr)&&sexpr->car.data.int_val==99,"original closure graph untouched");
    region_pop();
    auto* live=reinterpret_cast<arena_tagged_cons_cell_t*>(copy->sexpr_ptr);
    check(live->car.data.int_val==99,"closure sexpr survives poisoned source exit");
    std::printf("PASS closure captures=%zu subtype=%u sexpr lifetime\n",captures,expected);
}
static void tensor_alias_case(bool interior,bool short_first,bool prior_commit,bool zero){
    auto* source=region_create("tensor-view-alias",4096);check(source,"source region");region_push(source);
    auto* elements=static_cast<int64_t*>(arena_allocate(source->arena,3*sizeof(int64_t)));check(elements,"tensor elements");
    elements[0]=17;elements[1]=29;elements[2]=41;
    auto* full=arena_allocate_tensor_with_header(source->arena);
    auto* slice=arena_allocate_tensor_with_header(source->arena);check(full&&slice,"tensor fixtures");
    auto* full_dims=static_cast<uint64_t*>(arena_allocate(source->arena,sizeof(uint64_t)));
    auto* slice_dims=static_cast<uint64_t*>(arena_allocate(source->arena,sizeof(uint64_t)));check(full_dims&&slice_dims,"dimensions");
    *full_dims=3;*slice_dims=zero?0:1;
    full->dimensions=full_dims;full->num_dimensions=1;full->elements=elements;full->total_elements=3;full->dtype=ESHKOL_TENSOR_DTYPE_F64;
    slice->dimensions=slice_dims;slice->num_dimensions=1;slice->elements=elements+(interior?1:0);slice->total_elements=zero?0:1;slice->dtype=ESHKOL_TENSOR_DTYPE_F64;
    unsigned char original_full[sizeof(*full)],original_slice[sizeof(*slice)];
    std::memcpy(original_full,full,sizeof(*full));std::memcpy(original_slice,slice,sizeof(*slice));
    Value first=pointer(short_first?slice:full),second=pointer(short_first?full:slice),committed{};
    Value input,output=integer(73);
    if(prior_commit){
        check(eshkol_region_write_barrier_checked_v1(&committed,nullptr,&first)==0,"first tensor promotion commits");
        input=second;
    }else{
        void* root=arena_allocate_vector_with_header(source->arena,2);check(root,"tensor root");
        *static_cast<uint64_t*>(root)=2;auto* values=reinterpret_cast<Value*>(static_cast<char*>(root)+8);
        // Worklist is LIFO, so the second child supplies the first raw request.
        values[0]=second;values[1]=first;input=pointer(root);
    }
    void* old_map=source->fwd_map;auto* old_target=source->fwd_target;const size_t escapes=source->escape_count;
    auto* target_block=get_global_arena_shared()->current_block;
    std::memset(target_block->memory+target_block->used,0,target_block->size-target_block->used);
    const int32_t status=eshkol_region_write_barrier_checked_v1(&output,nullptr,&input);
    const bool supported=!interior&&!short_first;
    check(status==(supported?0:2),"raw view span admission matches explicit bounded contract");
    check(!std::memcmp(full,original_full,sizeof(*full))&&!std::memcmp(slice,original_slice,sizeof(*slice))&&elements[0]==17&&elements[1]==29&&elements[2]==41,"raw overlap source graph unchanged");
    eshkol_tensor_t *copied_full=nullptr,*copied_slice=nullptr;
    if(!supported){
        check(output.type==ESHKOL_VALUE_INT64&&output.data.int_val==73,"conflicting raw alias output unchanged");
        check(source->fwd_map==old_map&&source->fwd_target==old_target&&source->escape_count==escapes,"conflicting raw alias preserves committed map/target/count");
        if(prior_commit){
            Value again{};check(eshkol_region_write_barrier_checked_v1(&again,nullptr,&first)==0&&again.data.ptr_val==committed.data.ptr_val,"conflicting raw alias preserves old canonical identity");
        }
    }else{
        if(prior_commit){copied_full=reinterpret_cast<eshkol_tensor_t*>(committed.data.ptr_val);copied_slice=reinterpret_cast<eshkol_tensor_t*>(output.data.ptr_val);}
        else{auto* values=reinterpret_cast<Value*>(output.data.ptr_val+8);copied_slice=reinterpret_cast<eshkol_tensor_t*>(values[0].data.ptr_val);copied_full=reinterpret_cast<eshkol_tensor_t*>(values[1].data.ptr_val);}
        check(copied_full->elements==copied_slice->elements&&copied_full->elements!=elements,"covered same-base prefix preserves buffer identity");
        check(copied_full->elements[2]==41,"covered same-base prefix preserves full span");
    }
    region_pop();
    if(supported)check(copied_full->elements[2]==41,"covered prefix survives poisoned exit");
    if(prior_commit){auto* live=reinterpret_cast<eshkol_tensor_t*>(committed.data.ptr_val);if(short_first&&zero)check(live->total_elements==0&&live->elements!=elements+(interior?1:0),"empty committed tensor carries stable pointer");else check(live->elements[0]==(short_first&&interior?29:17),"old committed tensor survives failed later promotion and source exit");}
    std::printf("%s tensor view interior=%d short_first=%d prior_commit=%d zero=%d status=%d\n",supported?"PASS":"UNSUPPORTED",interior,short_first,prior_commit,zero,status);
}
static void tensor_shared_prefix(){for(bool zero:{false,true})for(bool interior:{false,true})for(bool short_first:{false,true})for(bool prior:{false,true})tensor_alias_case(interior,short_first,prior,zero);}

static void zero_endpoint(){
    auto* source=region_create("zero-span-endpoint",4096);check(source,"endpoint source");region_push(source);
    auto* elements=static_cast<int64_t*>(arena_allocate(source->arena,24));check(elements,"endpoint buffer");
    elements[0]=1;elements[1]=2;elements[2]=3;
    auto* full=arena_allocate_tensor_with_header(source->arena);auto* empty=arena_allocate_tensor_with_header(source->arena);check(full&&empty,"endpoint tensors");
    full->num_dimensions=0;full->dimensions=nullptr;full->elements=elements;full->total_elements=3;full->dtype=0;
    empty->num_dimensions=0;empty->dimensions=nullptr;empty->elements=elements+3;empty->total_elements=0;empty->dtype=0;
    Value input=pointer(full),output{};check(eshkol_region_write_barrier_checked_v1(&output,nullptr,&input)==0,"endpoint first span");
    input=pointer(empty);check(eshkol_region_write_barrier_checked_v1(&output,nullptr,&input)==0,"zero-span endpoint is outside half-open copied extent");
    auto* copy=reinterpret_cast<eshkol_tensor_t*>(output.data.ptr_val);check(copy->elements!=elements+3&&copy->total_elements==0,"endpoint pointer canonicalized");
    region_pop();std::puts("PASS zero-span endpoint is disjoint from copied half-open extent");
}
static void admission_edges(){
    auto* source=region_create("layout-admission",4096);check(source,"admission source");region_push(source);
    void* raw=arena_allocate(source->arena,16);check(raw,"linear-resource fixture");
    for(uint8_t type=ESHKOL_VALUE_HANDLE;type<=ESHKOL_VALUE_EVENT;++type){
        Value input=pointer(raw,type),output=integer(73);
        check(eshkol_region_write_barrier_checked_v1(&output,nullptr,&input)==2&&output.data.int_val==73,"regional linear resource rejected explicitly");
    }
    auto* tensor=arena_allocate_tensor_with_header(source->arena);check(tensor,"tensor admission fixture");
    tensor->num_dimensions=0;tensor->dimensions=nullptr;tensor->total_elements=0;tensor->elements=nullptr;tensor->dtype=99;
    Value input=pointer(tensor),output=integer(73);
    check(eshkol_region_write_barrier_checked_v1(&output,nullptr,&input)==2&&output.data.int_val==73,"undeclared tensor dtype rejected");
    tensor->dtype=ESHKOL_TENSOR_DTYPE_F64;ESHKOL_GET_HEADER(tensor)->size=33;
    check(eshkol_region_write_barrier_checked_v1(&output,nullptr,&input)==2&&output.data.int_val==73,"partial tensor dtype field rejected");
    ESHKOL_GET_HEADER(tensor)->size=32;
    check(eshkol_region_write_barrier_checked_v1(&output,nullptr,&input)==0,"legacy32 numeric tensor remains supported");
    auto* dual=arena_allocate_tensor_with_header(source->arena);check(dual,"dual tensor fixture");
    dual->num_dimensions=0;dual->dimensions=nullptr;dual->total_elements=1;dual->dtype=ESHKOL_TENSOR_DTYPE_DUAL;
    auto* payload=arena_allocate_dual_number(source->arena);check(payload,"dual payload");
    auto* numbers=reinterpret_cast<double*>(payload);numbers[0]=3.0;numbers[1]=7.0;
    auto* tagged=static_cast<Value*>(arena_allocate(source->arena,sizeof(Value)));check(tagged,"dual tagged elements");
    *tagged=pointer(payload,ESHKOL_VALUE_DUAL_NUMBER);dual->elements=reinterpret_cast<int64_t*>(tagged);
    input=pointer(dual);check(eshkol_region_write_barrier_checked_v1(&output,nullptr,&input)==0,"dual tensor deep promotion");
    auto* copied_dual=reinterpret_cast<eshkol_tensor_t*>(output.data.ptr_val);
    auto* copied_payload=reinterpret_cast<double*>(reinterpret_cast<Value*>(copied_dual->elements)->data.ptr_val);
    check(copied_dual->elements!=dual->elements&&copied_payload!=numbers,"dual tagged array and payload both copied");
    auto* exception=static_cast<eshkol_exception_t*>(arena_allocate_with_header_zeroed(source->arena,sizeof(eshkol_exception_t),HEAP_SUBTYPE_EXCEPTION,0));check(exception,"empty exception fixture");
    exception->type=ESHKOL_EXCEPTION_ERROR;exception->message=const_cast<char*>("empty-irritants");
    exception->irritants=tagged;exception->num_irritants=0;
    // Use an independent nonnull zero-span pointer, so the preceding dual's
    // committed map does not intentionally trigger a mixed-span rejection.
    exception->irritants=static_cast<Value*>(arena_allocate(source->arena,sizeof(Value)));check(exception->irritants,"empty irritant pointer");
    input=pointer(exception);check(eshkol_region_write_barrier_checked_v1(&output,nullptr,&input)==0,"zero-count nonnull irritant pointer canonicalizes");
    auto* copied_exception=reinterpret_cast<eshkol_exception_t*>(output.data.ptr_val);
    check(copied_exception->irritants!=exception->irritants&&copied_exception->num_irritants==0,"empty irritant pointer is stable");
    region_pop();check(copied_payload[0]==3.0&&copied_payload[1]==7.0,"dual payload survives poisoned exit");
    check(copied_exception->num_irritants==0,"empty exception survives poisoned exit");
    std::puts("PASS linear-resource rejection, tensor legacy/dtype admission, dual payload and empty exception lifetime");
}

// Ownership assertions cover the complete used span, not just its first byte.
static bool contains_span(const arena_t* arena,const void* pointer,size_t size){
    const auto address=reinterpret_cast<uintptr_t>(pointer);
    for(auto* block=arena->current_block;block;block=block->next){
        const auto begin=reinterpret_cast<uintptr_t>(block->memory);
        if(address>=begin&&address-begin<=block->used&&size<=block->used-(address-begin))return true;
    }
    return false;
}
static void root_span(arena_t* root,arena_t* outer,arena_t* inner,const void* pointer,size_t size){
    check(size&&contains_span(root,pointer,size),"complete continuation allocation belongs to stable root");
    check(!arena_contains(outer,pointer)&&!arena_contains(inner,pointer),"continuation allocation is outside both young arenas");
}
static uint64_t bytes_hash(const void* pointer,size_t size){
    auto* bytes=static_cast<const unsigned char*>(pointer);uint64_t hash=14695981039346656037ULL;
    for(size_t i=0;i<size;++i){hash^=bytes[i];hash*=1099511628211ULL;}
    return hash;
}
static void continuation_producer(bool snapshot){
    // Raw C-stack memcpy crosses sanitizer redzones. Exercise that existing
    // hosted operation only via the explicit release-only CLI case below.
    if(snapshot)eshkol_init_stack_size();
    auto* root=eshkol_root_arena_v1();check(root==get_global_arena_shared(),"initial immutable root matches unrouted shared slot");
    const uint64_t initial_mark=eshkol_region_mark();
    auto* outer=region_create("continuation-outer",4096);check(outer,"outer continuation region");region_push(outer);
    auto* outer_saved=eshkol_region_enter(outer);
    check(eshkol_current_arena()==outer->arena&&get_global_arena_shared()==outer->arena&&eshkol_root_arena_v1()==root,"outer routing leaves immutable root unchanged");
    auto* inner=region_create("continuation-inner",4096);check(inner,"inner continuation region");region_push(inner);
    auto* inner_saved=eshkol_region_enter(inner);
    check(eshkol_current_arena()==inner->arena&&get_global_arena_shared()==inner->arena&&eshkol_root_arena_v1()==root,"nested routing leaves immutable root unchanged");
    auto* retained_outer=outer->arena;auto* retained_inner=inner->arena;
    auto* pinned_value=arena_allocate_cons_with_header(inner->arena);check(pinned_value,"pinned regional value");pinned_value->car=integer(113);pinned_value->cdr={};
    std::jmp_buf jump;
    if(setjmp(jump)!=0)check(false,"native fixture never resumes its continuation");
    auto* state=eshkol_make_continuation_state(eshkol_root_arena_v1(),&jump);check(state,"root continuation state");
    auto* closure=static_cast<eshkol_closure_t*>(eshkol_make_continuation_closure(eshkol_root_arena_v1(),state));check(closure&&closure->env,"root continuation closure and environment");
    if(snapshot){
        eshkol_continuation_capture_stack(eshkol_root_arena_v1(),state);
        check(state->saved_stack&&state->saved_len,"hosted capture produced an actual saved stack");
    }else check(!state->saved_stack&&!state->saved_len,"state starts without a captured stack");
    check(state->region_mark==initial_mark+2&&outer->pinned&&inner->pinned,"continuation producer preserves pinning of both active regions");
    check(state->jmp_buf_ptr==&jump&&state->value.type==ESHKOL_VALUE_NULL,"continuation raw state producer fields");
    const auto jump_address=reinterpret_cast<uintptr_t>(&jump),stack_lo=reinterpret_cast<uintptr_t>(state->stack_lo),stack_hi=reinterpret_cast<uintptr_t>(state->stack_hi);
    if(snapshot)check(stack_hi>stack_lo&&state->saved_len==stack_hi-stack_lo&&jump_address>=stack_lo&&jump_address+sizeof(jump)<=stack_hi,"saved stack spans actual setjmp storage");
    auto* header=ESHKOL_GET_HEADER(closure);
    check(header->subtype==CALLABLE_SUBTYPE_CONTINUATION&&header->size==sizeof(*closure),"actual continuation header layout");
    check(closure->env->num_captures==(1|(1ULL<<16))&&closure->env->captures[0].type==ESHKOL_VALUE_HEAP_PTR&&closure->env->captures[0].data.ptr_val==reinterpret_cast<uint64_t>(state),"continuation environment stores raw state identity");
    root_span(root,retained_outer,retained_inner,header,sizeof(*header)+sizeof(*closure));
    root_span(root,retained_outer,retained_inner,closure->env,sizeof(*closure->env)+sizeof(Value));
    root_span(root,retained_outer,retained_inner,state,sizeof(*state));
    if(snapshot)root_span(root,retained_outer,retained_inner,state->saved_stack,state->saved_len);
    const uint64_t saved_hash=bytes_hash(state->saved_stack,state->saved_len);
    const size_t root_used=arena_get_used_memory(root),escapes=inner->escape_count;
    void* old_map=inner->fwd_map;auto* old_target=inner->fwd_target;
    Value input=pointer(closure,ESHKOL_VALUE_CALLABLE),output=integer(73);
    eshkol_promotion_test_arm(-1,0);
    check(eshkol_region_write_barrier_checked_v1(&output,nullptr,&input)==0&&!std::memcmp(&output,&input,sizeof(input)),"root continuation barrier preserves exact tagged identity under allocation denial");
    const auto stats=eshkol_promotion_test_snapshot();
    for(auto attempts:stats.attempts)check(attempts==0,"root continuation barrier attempts no promotion allocation");
    eshkol_promotion_test_reset();
    check(arena_get_used_memory(root)==root_used&&inner->escape_count==escapes&&inner->fwd_map==old_map&&inner->fwd_target==old_target,"root continuation barrier changes no promotion state");

    // The regional producer remains unsupported even after pinning: there is
    // no generic continuation-as-closure or pinned-region admission bypass.
    auto* local_state=eshkol_make_continuation_state(eshkol_current_arena(),&jump);check(local_state,"regional continuation state control");
    auto* local=static_cast<eshkol_closure_t*>(eshkol_make_continuation_closure(eshkol_current_arena(),local_state));check(local&&local->env,"regional continuation closure control");
    check(contains_span(inner->arena,local_state,sizeof(*local_state))&&contains_span(inner->arena,ESHKOL_GET_HEADER(local),sizeof(eshkol_object_header_t)+sizeof(*local)),"regional control still uses current allocation route");
    const auto local_state_hash=bytes_hash(local_state,sizeof(*local_state));
    const auto local_closure_hash=bytes_hash(ESHKOL_GET_HEADER(local),sizeof(eshkol_object_header_t)+sizeof(*local));
    const auto local_env_hash=bytes_hash(local->env,sizeof(*local->env)+sizeof(Value));
    input=pointer(local,ESHKOL_VALUE_CALLABLE);output=integer(73);
    check(eshkol_region_write_barrier_checked_v1(&output,nullptr,&input)==2&&output.type==ESHKOL_VALUE_INT64&&output.data.int_val==73,"regional continuation is explicitly unsupported with unchanged output");
    check(inner->escape_count==escapes&&inner->fwd_map==old_map&&inner->fwd_target==old_target&&arena_get_used_memory(root)==root_used,"regional continuation rejection preserves old map, counter and target used bytes");
    check(bytes_hash(local_state,sizeof(*local_state))==local_state_hash&&bytes_hash(ESHKOL_GET_HEADER(local),sizeof(eshkol_object_header_t)+sizeof(*local))==local_closure_hash&&bytes_hash(local->env,sizeof(*local->env)+sizeof(Value))==local_env_hash,"regional continuation rejection leaves complete source layout untouched");
    region_pop();eshkol_region_leave(inner_saved);
    check(get_global_arena_shared()==retained_outer&&eshkol_current_arena()==retained_outer&&eshkol_root_arena_v1()==root,"nested exit restores outer allocation route");
    region_pop();eshkol_region_leave(outer_saved);
    check(get_global_arena_shared()==root&&eshkol_current_arena()==root&&eshkol_region_mark()==initial_mark,"outer exit restores root allocation route and region mark");
    check(pinned_value->car.data.int_val==113,"captured regions retain their contents by existing pinning contract");
    // Pinned exits above deliberately retain their arenas. This later region
    // is unpinned and really is reclaimed/poisoned under ESHKOL_ARENA_POISON.
    auto* churn=region_create("continuation-unpinned-churn",4096);check(churn,"unpinned churn region");region_push(churn);auto* churn_saved=eshkol_region_enter(churn);
    check(arena_allocate(churn->arena,1024)&&!churn->pinned,"later region has no continuation pin");region_pop();eshkol_region_leave(churn_saved);
    check(ESHKOL_GET_HEADER(closure)->subtype==CALLABLE_SUBTYPE_CONTINUATION&&closure->env->captures[0].data.ptr_val==reinterpret_cast<uint64_t>(state)&&bytes_hash(state->saved_stack,state->saved_len)==saved_hash,"root continuation and saved stack survive region exits and poisoned churn");
    // This fixture never invokes either continuation again. After proving
    // pin retention, release its two deliberately retained arenas explicitly
    // so the native ownership test remains useful with leak detection enabled.
    arena_destroy(retained_inner);arena_destroy(retained_outer);
    std::printf("PASS continuation producer: nested root spans, pinning, zero-allocation identity, regional rejection and poisoned churn; saved-stack capture=%d (no native resume)\n",snapshot);
}

int main(int argc,char** argv){
    check(get_global_arena_shared(),"global arena");
    if(argc==1||!std::strcmp(argv[1],"closure")){closure_sexpr(0);closure_sexpr(1);}
    if(argc==1||!std::strcmp(argv[1],"tensor"))tensor_shared_prefix();
    if(argc==1||!std::strcmp(argv[1],"admission")){admission_edges();zero_endpoint();}
    if(argc==1||!std::strcmp(argv[1],"continuation"))continuation_producer(false);
    if(argc==2&&!std::strcmp(argv[1],"continuation-snapshot"))continuation_producer(true);
}
