#define ESHKOL_VM_LIBRARY_MODE 1
#include "../../lib/backend/eshkol_vm.c"

static int rational_limits(void) {
    VmArena arena;
    vm_arena_init(&arena, VM_ARENA_DEFAULT_BLOCK_SIZE);
    VmRational* small = vm_rational_make(&arena, 2, INT64_MIN);
    VmRational* unit = vm_rational_make(&arena, INT64_MIN, INT64_MIN);
    VmRational* positive = vm_rational_make(&arena, INT64_MIN, -1);
    VmRational* reciprocal = vm_rational_make(&arena, 1, INT64_MIN);
    int ok = small && !small->is_big && small->num == -1 && small->denom == (INT64_C(1) << 62)
          && unit && !unit->is_big && unit->num == 1 && unit->denom == 1
          && positive && positive->is_big && bignum_to_double(positive->big_num) == 0x1p63
          && vm_bn_is_i64(positive->big_den, 1)
          && reciprocal && reciprocal->is_big && vm_bn_is_i64(reciprocal->big_num, -1)
          && bignum_to_double(reciprocal->big_den) == 0x1p63;
    vm_arena_destroy(&arena);
    return ok;
}

static int bignum_zero_condition(void) {
    VM* vm = vm_create();
    if (!vm) return 0;
    VmRegionStack* rs = &vm->heap.regions;
    VmBignum* n = bignum_from_int64(rs, 3);
    VmBignum* z = bignum_from_int64(rs, 0);
    int ok = n && z && !bignum_div(rs, n, z) && !bignum_mod(rs, n, z);
    for (int fid = 358; fid <= 359; ++fid) {
        vm->error = 0;
        vm_push(vm, INT_VAL(3));
        vm_push(vm, INT_VAL(0));
        vm_dispatch_native(vm, fid);
        ok = ok && vm->error;
    }
    vm_free(vm);
    return ok;
}

struct diagnostic_call {
    VmRegionStack* regions;
    int angle;
};

static void* evaluate_at_origin(void* opaque) {
    struct diagnostic_call* call = (struct diagnostic_call*)opaque;
    VmComplex origin = {0};
    origin.dreal = 1.0;
    if (call->angle) vm_complex_angle_tangent(call->regions, &origin);
    else vm_complex_magnitude_tangent(call->regions, &origin);
    return NULL;
}

static int diagnostic_contexts(void) {
    VM* first = vm_create();
    VM* second = vm_create();
    if (!first || !second) return 0;
    VmRegionStack* a = &first->heap.regions;
    VmRegionStack* b = &second->heap.regions;
    VmRational* zero = vm_rational_from_int(vm_active_arena(a), 0);
    VmDual* exact_zero = vm_dual_constant(a, 0.0, zero);
    VmComplex* origin = vm_complex_new(a, 0.0, 0.0);
    origin->dreal = 1.0;
    vm_complex_log(a, origin);
    vm_dual_div(a, exact_zero, exact_zero);

    /* Run a successful operation in another context on the same thread
     * while the first context still owns both pending conditions. */
    vm_push_complex_result(second, vm_complex_new(b, 1.0, 0.0));
    int ok = !second->error && !b->dual_error && !b->complex_error &&
             a->dual_error && a->complex_error;
    struct diagnostic_call calls[] = {{a, 0}, {b, 1}};
    pthread_t workers[2];
    int started = 0;
    for (; started < 2; ++started) {
        if (pthread_create(&workers[started], NULL, evaluate_at_origin, &calls[started]) != 0)
            break;
    }
    for (int i = 0; i < started; ++i) pthread_join(workers[i], NULL);
    ok = ok && started == 2 && a->complex_error && b->complex_error &&
         strncmp(a->complex_error, "magnitude:", 10) == 0 &&
         strncmp(b->complex_error, "angle:", 6) == 0;
    vm_free(first);
    vm_free(second);
    return ok;
}

static int symbolic_variable_power(void) {
    VmArena arena;
    vm_arena_init(&arena, sizeof(SymNode) * 4);
    SymNode* x = sym_var(&arena, 0);
    SymNode* power = sym_binop(&arena, SYM_POW, x, x);
    SymNode* slope = sym_differentiate(&arena, power, 0);
    VM* vm = vm_create();
    FuncChunk chunk;
    if (!slope || !vm || chunk_init_arrays(&chunk) != 0) return 0;
    sym_compile(slope, &chunk);
    vm->code_len = 0;
    vm_push(vm, FLOAT_VAL(2.0));
    for (int i = 0; i < chunk.code_len; ++i) {
        Instr instr = chunk.code[i];
        if (instr.op == OP_CONST)
            instr.operand = add_constant(vm, chunk.constants[instr.operand]);
        emit(vm, instr.op, instr.operand);
    }
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    int ok = !vm->error && vm->sp == 2 &&
             fabs(as_number_vm(vm, vm_pop(vm)) - 4.0 * (1.0 + log(2.0))) < 1e-12;
    chunk_free_arrays(&chunk);
    vm_free(vm);
    vm_arena_destroy(&arena);
    return ok;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    if (!strcmp(argv[1], "rational_limits")) return !rational_limits();
    if (!strcmp(argv[1], "bignum_zero_condition")) return !bignum_zero_condition();
    if (!strcmp(argv[1], "diagnostic_contexts")) return !diagnostic_contexts();
    if (!strcmp(argv[1], "symbolic_variable_power")) return !symbolic_variable_power();
    return 2;
}
