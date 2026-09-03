#define ESHKOL_VM_LIBRARY_MODE 1
#include "../../lib/backend/eshkol_vm.c"

enum FixtureKind {
    FIXTURE_VALID_DUPLICATES,
    FIXTURE_TRAILING_BYTE,
    FIXTURE_TRUNCATED_SECOND_RECORD
};

static int write_record(VmModelWriter* writer, const char* name, double value) {
    union { uint64_t u; double d; } bits;
    bits.d = value;
    return vm_model_write_u32(writer, (unsigned int)strlen(name), 1) &&
           vm_model_write_bytes(writer, name, strlen(name), 1) &&
           vm_model_write_u32(writer, 1u, 1) &&
           vm_model_write_u64(writer, 1u, 1) &&
           vm_model_write_u8(writer, 0u, 1) &&
           vm_model_write_u64(writer, bits.u, 1);
}

static int write_fixture(const char* path, enum FixtureKind kind) {
    VmModelWriter writer = {fopen(path, "wb"), 0u, 1};
    if (!writer.file) return 0;

    const unsigned int count = kind == FIXTURE_VALID_DUPLICATES ? 2u :
                               kind == FIXTURE_TRUNCATED_SECOND_RECORD ? 2u : 1u;
    int ok = vm_model_write_bytes(&writer, VM_MODEL_MAGIC, sizeof(VM_MODEL_MAGIC), 1) &&
             vm_model_write_u32(&writer, VM_MODEL_VERSION, 1) &&
             vm_model_write_u32(&writer, count, 1) &&
             vm_model_write_u32(&writer, 0u, 1) &&
             write_record(&writer, kind == FIXTURE_VALID_DUPLICATES ? "dup" : "w", 1.0);

    if (kind == FIXTURE_VALID_DUPLICATES) {
        ok = ok && write_record(&writer, "dup", 2.0);
    } else if (kind == FIXTURE_TRAILING_BYTE) {
        ok = ok && vm_model_write_u8(&writer, 0xA5u, 1);
    } else {
        ok = ok && vm_model_write_u32(&writer, 8u, 1) &&
             vm_model_write_u8(&writer, 'x', 1);
    }
    ok = ok && vm_model_write_u32(&writer, writer.crc, 0);
    fclose(writer.file);
    return ok && writer.ok;
}

static int reject_without_heap_growth(VM* vm, const char* path, int model_load) {
    Value path_value;
    if (!vm_model_make_string_value(vm, path, (int)strlen(path), &path_value)) return 0;
    VmArena* arena = vm_active_arena(&vm->heap.regions);
    const int32_t before = vm->heap.next_free;
    const int32_t free_slots_before = vm->heap.n_free_slots;
    const size_t arena_used_before = arena->total_used;
    vm_push(vm, path_value);
    if (model_load) vm_model_model_load(vm);
    else vm_model_tensor_load(vm);
    Value result = vm_pop(vm);
    return result.type == VAL_NIL && vm->heap.next_free == before &&
           vm->heap.n_free_slots == free_slots_before &&
           arena->total_used == arena_used_before;
}

static int duplicate_model_preserves_order(VM* vm, const char* path) {
    Value path_value;
    if (!vm_model_make_string_value(vm, path, (int)strlen(path), &path_value)) return 0;
    vm_push(vm, path_value);
    vm_model_model_load(vm);
    Value list = vm_pop(vm);

    const double expected[] = {1.0, 2.0};
    for (int i = 0; i < 2; i++) {
        if (list.type != VAL_PAIR) return 0;
        HeapObject* node = vm->heap.objects[list.as.ptr];
        Value entry = node->cons.car;
        if (entry.type != VAL_PAIR) return 0;
        HeapObject* pair = vm->heap.objects[entry.as.ptr];
        int name_len = 0;
        const char* name = vm_model_string_ptr(vm, pair->cons.car, &name_len);
        VmTensor* tensor = vm_model_value_tensor(vm, pair->cons.cdr);
        if (!name || name_len != 3 || memcmp(name, "dup", 3) != 0 ||
            !tensor || tensor->total != 1 || tensor->data[0] != expected[i]) {
            return 0;
        }
        list = node->cons.cdr;
    }
    return list.type == VAL_NIL;
}

static int expect_case(int condition, const char* label) {
    if (condition) return 1;
    fprintf(stderr, "FAIL: %s\n", label);
    return 0;
}

int main(void) {
    VM* vm = vm_create();
    if (!vm) return 1;

    char valid_path[128];
    char trailing_path[128];
    char truncated_path[128];
    snprintf(valid_path, sizeof(valid_path), ".vm-model-valid-%p.eskm", (void*)vm);
    snprintf(trailing_path, sizeof(trailing_path), ".vm-model-trailing-%p.eskm", (void*)vm);
    snprintf(truncated_path, sizeof(truncated_path), ".vm-model-truncated-%p.eskm", (void*)vm);

    int ok = expect_case(write_fixture(valid_path, FIXTURE_VALID_DUPLICATES),
                         "could not write duplicate-name fixture");
    ok &= expect_case(write_fixture(trailing_path, FIXTURE_TRAILING_BYTE),
                      "could not write trailing-byte fixture");
    ok &= expect_case(write_fixture(truncated_path, FIXTURE_TRUNCATED_SECOND_RECORD),
                      "could not write truncated-record fixture");
    ok &= expect_case(reject_without_heap_growth(vm, trailing_path, 0),
                      "tensor-load materialized a trailing-byte checkpoint");
    ok &= expect_case(reject_without_heap_growth(vm, trailing_path, 1),
                      "model-load materialized a trailing-byte checkpoint");
    ok &= expect_case(reject_without_heap_growth(vm, truncated_path, 1),
                      "model-load materialized before a malformed later record");
    ok &= expect_case(duplicate_model_preserves_order(vm, valid_path),
                      "duplicate-name records lost order or payloads");

    remove(valid_path);
    remove(trailing_path);
    remove(truncated_path);
    vm_free(vm);
    if (!ok) {
        return 1;
    }
    printf("PASS: VM ESKM fail-closed preflight\n");
    return 0;
}
