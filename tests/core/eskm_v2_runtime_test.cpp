/* Exercise the private staging boundary, including a late allocation failure. */
#include "../../lib/core/model_io.cpp"
#include <cassert>

#ifdef ESKM_V2_MALLOC_INJECTION
static int fail_at, allocation, injected;
extern "C" void* __real_malloc(size_t);
extern "C" void* __wrap_malloc(size_t bytes) {
    if (fail_at && ++allocation == fail_at) { injected = 1; return nullptr; }
    return __real_malloc(bytes);
}
#endif

static int new_fail_at, new_allocation, new_injected;
void* operator new(std::size_t bytes) {
    if (new_fail_at && ++new_allocation == new_fail_at) {
        new_injected = 1;
        throw std::bad_alloc();
    }
    void* result = std::malloc(bytes ? bytes : 1);
    if (!result) throw std::bad_alloc();
    return result;
}
void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }

int main(int argc, char** argv) {
    assert(argc == 2);
    setenv("ESHKOL_EXPERIMENTAL_ESKM_V2", "read", 1);
    arena_t* arena = arena_create(8192);
    assert(arena);
    auto path = make_heap_ptr(arena_allocate_string_with_header(arena, std::strlen(argv[1])));
    std::strcpy(reinterpret_cast<char*>(path.data.ptr_val), argv[1]);
    const auto allocated = arena->total_allocated;
    const auto block = arena->current_block;
    const auto used = block->used;
    {
        CheckpointArena transaction(arena, 1024);
        ParsedTensorRecord record;
        record.ndims = 1;
        record.dims = {2048};
        record.element_bits.resize(2048);
        eshkol_tensor_t* tensor = nullptr;
        assert(!tensor_from_record(transaction.get(), record, &tensor));
    }
    assert(arena->total_allocated == allocated && arena->current_block == block && block->used == used);
    arena_push_scope(arena);
    eshkol_tagged_value_t result;
    eshkol_tensor_load_tagged(arena, &path, &result);
    assert(tagged_is_tensor(&result));
    assert(arena_top_scope_contains(arena, reinterpret_cast<void*>(result.data.ptr_val)));
    arena_pop_scope(arena);
    assert(arena->total_allocated == allocated && arena->current_block == block && block->used == used);
#ifdef ESKM_V2_MALLOC_INJECTION
    int failures = 0;
    for (int nth = 1; nth < 16; ++nth) {
        arena_push_scope(arena);
        allocation = injected = 0; fail_at = nth;
        eshkol_tensor_load_tagged(arena, &path, &result);
        fail_at = 0;
        if (injected) {
            assert(result.type == ESHKOL_VALUE_NULL);
            assert(arena->current_block == block && block->used == used);
            assert(arena->total_allocated == allocated);
            ++failures;
        } else assert(tagged_is_tensor(&result));
        arena_pop_scope(arena);
        if (!injected) break;
    }
    assert(failures >= 2);
    printf("PASS: native public loader rollback at %d malloc sites\n", failures);
#endif
    int parser_failures = 0;
    for (int nth = 1; nth < 16; ++nth) {
        arena_push_scope(arena);
        new_allocation = new_injected = 0; new_fail_at = nth;
        eshkol_model_load_tagged(arena, &path, &result);
        new_fail_at = 0;
        if (new_injected) {
            assert(result.type == ESHKOL_VALUE_NULL);
            assert(arena->current_block == block && block->used == used);
            assert(arena->total_allocated == allocated);
            ++parser_failures;
        } else assert(is_pair(result));
        arena_pop_scope(arena);
        if (!new_injected) break;
    }
    assert(parser_failures >= 3);
    printf("PASS: native model parser rollback at %d new sites\n", parser_failures);
    setenv("ESHKOL_EXPERIMENTAL_ESKM_V2", "write", 1);
    eshkol_tagged_value_t entry, cycle;
    eshkol_tensor_load_tagged(arena, &path, &result);
    assert(prepend_list_node(arena, path, result, &entry));
    assert(prepend_list_node(arena, entry, make_null(), &cycle));
    auto* node = reinterpret_cast<arena_tagged_cons_cell_t*>(cycle.data.ptr_val);
    arena_tagged_cons_set_tagged_value(node, true, &cycle);
    std::vector<TensorRecordView> records;
    assert(!extract_model_entries(&cycle, &records));
    assert(records.size() == ESKM_V2_BACKEND_RECORDS);
    setenv("ESHKOL_EXPERIMENTAL_ESKM_V2", "read", 1);
    arena_t* bounded = arena_create_bounded(8192);
    const auto bounded_used = bounded->current_block->used;
    eshkol_tensor_load_tagged(bounded, &path, &result);
    assert(result.type == ESHKOL_VALUE_NULL && bounded->current_block->used == bounded_used);
    arena_destroy(bounded);
    setenv("ESHKOL_EXPERIMENTAL_ESKM_V2", "invalid", 1);
    eshkol_tensor_load_tagged(arena, &path, &result);
    assert(result.type == ESHKOL_VALUE_NULL);
    arena_destroy(arena);
    puts("PASS: native v2 staged failure, scope reclamation, bounded refusal and invalid mode");
}
