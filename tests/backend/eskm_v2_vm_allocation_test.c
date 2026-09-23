#include <stdlib.h>
#include <assert.h>
static int fail_at, allocation, injected;
static int fail_allocation(void) {
    if (fail_at && ++allocation == fail_at) { injected = 1; return 1; }
    return 0;
}
static void* test_malloc(size_t n) { return fail_allocation() ? NULL : malloc(n); }
static void* test_calloc(size_t n, size_t size) { return fail_allocation() ? NULL : calloc(n, size); }
static void* test_realloc(void* p, size_t n) { return fail_allocation() ? NULL : realloc(p, n); }
#define malloc test_malloc
#define calloc test_calloc
#define realloc test_realloc
#define ESHKOL_VM_LIBRARY_MODE 1
#include "../../lib/backend/eshkol_vm.c"
#undef malloc
#undef calloc
#undef realloc

int main(void) {
    const char* file = ".eskm-v2-vm-allocation.eskm";
    VmModelWriter writer = {0};
    writer.ok = eshkol_atomic_checkpoint_begin(&writer.file, file);
    assert(writer.ok);
    assert(vm_model_write_bytes(&writer, VM_MODEL_MAGIC, 4, 1));
    assert(vm_model_write_u32(&writer, 2, 1));
    assert(vm_model_write_u32(&writer, 2, 1));
    assert(vm_model_write_u32(&writer, 0, 1));
    assert(vm_model_write_u64(&writer, 0, 1));
    for (int r = 0; r < 2; ++r) {
        assert(vm_model_write_u32(&writer, 1, 1));
        assert(vm_model_write_bytes(&writer, "w", 1, 1));
        assert(vm_model_write_u32(&writer, 1, 1));
        assert(vm_model_write_u64(&writer, 2048, 1));
        assert(vm_model_write_u8(&writer, 0, 1));
        for (int i = 0; i < 2048; ++i) assert(vm_model_write_u64(&writer, UINT64_C(0x8000000000000000), 1));
    }
    assert(vm_model_write_u32(&writer, writer.crc, 0));
    assert(eshkol_atomic_checkpoint_commit(&writer.file));
    setenv("ESHKOL_EXPERIMENTAL_ESKM_V2", "read", 1);
    int failures = 0, successes = 0;
    for (int nth = 1; nth < 64; ++nth) {
        VM* vm = vm_create();
        assert(vm);
        int32_t recycled = heap_alloc(&vm->heap);
        assert(recycled >= 0);
        vm->heap.objects[recycled] = NULL;
        vm->heap.free_slots = (int32_t*)malloc(sizeof(int32_t));
        assert(vm->heap.free_slots);
        vm->heap.free_slots[0] = recycled;
        vm->heap.cap_free_slots = 1;
        assert(heap_region_push(&vm->heap, "v2-test", 8192));
        Value path;
        assert(vm_model_make_string_value(vm, file, (int)strlen(file), &path));
        vm->heap.n_free_slots = 1;
        VmArena before = *vm_active_arena(&vm->heap.regions);
        size_t used = before.current->used;
        int32_t next = vm->heap.next_free;
        int32_t slots = vm->heap.region_slots[vm->heap.regions.depth - 1].n_slots;
        vm_push(vm, path);
        allocation = injected = 0; fail_at = nth;
        vm_model_model_load(vm);
        fail_at = 0;
        Value result = vm_pop(vm);
        if (injected) {
            assert(result.type == VAL_NIL);
            VmArena* after = vm_active_arena(&vm->heap.regions);
            assert(after->current == before.current && after->current->used == used);
            assert(after->total_allocated == before.total_allocated && after->total_used == before.total_used);
            assert(vm->heap.next_free == next && vm->heap.n_free_slots == 1 && vm->heap.objects[recycled] == NULL);
            assert(vm->heap.region_slots[vm->heap.regions.depth - 1].n_slots == slots);
            ++failures;
        } else {
            assert(result.type == VAL_PAIR);
            ++successes;
        }
        if (!injected) {
            setenv("ESHKOL_EXPERIMENTAL_ESKM_V2", "write", 1);
            vm->heap.objects[result.as.ptr]->cons.cdr = result;
            assert(vm_model_list_length(result, vm) == -1);
            setenv("ESHKOL_EXPERIMENTAL_ESKM_V2", "read", 1);
        }
        vm_free(vm);
        if (successes) break;
    }
    remove(file);
    assert(failures >= 8 && successes == 1);
    printf("PASS: VM v2 allocation rollback at %d allocation sites\n", failures);
}
