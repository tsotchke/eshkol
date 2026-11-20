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
    if (!strcmp(argv[1], "symbolic_variable_power")) return !symbolic_variable_power();
    return 2;
}
