/* Exercise runtime bytecode guards independently of serialized-module
 * admission. The internal VM is assembled directly, and every verdict uses
 * explicit conditions so Release/NDEBUG builds execute the same checks. */
#define ESHKOL_VM_LIBRARY_MODE 1
#include <limits.h>
#include "../../lib/backend/eshkol_vm.c"

static VM* vm_with_instruction(uint8_t opcode, int32_t operand) {
    VM* vm = vm_create();
    if (!vm) return NULL;
    vm->code_len = 0;
    emit(vm, opcode, operand);
    emit(vm, OP_HALT, 0);
    return vm;
}

static int local_refusal(const char* name, uint8_t opcode,
                         int32_t fp, int32_t operand) {
    VM* vm = vm_with_instruction(opcode, operand);
    if (!vm) return 0;
    vm->fp = fp;
    const int initial_sp = opcode == OP_SET_LOCAL ? 1 : 0;
    if (opcode == OP_SET_LOCAL) vm_push(vm, INT_VAL(17));
    vm_run(vm);
    int refused = vm->error && vm->pc == 1 && vm->sp == initial_sp;
    if (!refused)
        fprintf(stderr, "FAIL: %s did not refuse the invalid local index\n", name);
    vm_free(vm);
    return refused;
}

static int wind_refusal(void) {
    VM* vm = vm_with_instruction(OP_WIND_PUSH, 0);
    if (!vm) return 0;
    vm->n_winds = 32;
    vm_push(vm, INT_VAL(1));
    vm_push(vm, INT_VAL(2));
    vm_run(vm);
    int refused = vm->error && vm->pc == 1 && vm->sp == 0 && vm->n_winds == 32;
    if (!refused) fprintf(stderr, "FAIL: OP_WIND_PUSH overflow was not refused\n");
    vm_free(vm);
    return refused;
}

static int callcc_refusal(void) {
    VM* vm = vm_with_instruction(OP_CALLCC, 0);
    if (!vm) return 0;
    int32_t proc_ptr = heap_alloc(&vm->heap);
    if (proc_ptr < 0) { vm_free(vm); return 0; }
    vm->heap.objects[proc_ptr]->type = HEAP_CLOSURE;
    vm->heap.objects[proc_ptr]->closure.func_pc = 2;
    vm->heap.objects[proc_ptr]->closure.arity = 1;
    /* vm_create allocated MAX_FRAMES entries; this deliberately inconsistent
     * capacity metadata is safe to inspect but must be refused before capture. */
    vm->frame_count = 1;
    vm->frame_cap = 0;
    vm_push(vm, (Value){.type = VAL_CLOSURE, .as.ptr = proc_ptr});
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    int refused = vm->error && vm->pc == 1 && vm->sp == 0;
    if (!refused) fprintf(stderr, "FAIL: OP_CALLCC invalid capture state was not refused\n");
    vm_free(vm);
    return refused;
}

static int local_valid(uint8_t opcode) {
    VM* vm = vm_with_instruction(opcode, 0);
    if (!vm) return 0;
    vm->fp = 1; /* positive-frame local access is the valid boundary control */
    if (opcode == OP_GET_LOCAL) {
        vm->stack[1] = INT_VAL(42);
    } else {
        vm_push(vm, INT_VAL(23));
    }
    vm_run(vm);
    int valid = !vm->error && vm->halted &&
        (opcode == OP_GET_LOCAL ? vm->stack[0].as.i == 42 :
                                  vm->stack[1].as.i == 23);
    if (!valid) fprintf(stderr, "FAIL: valid %s at fp > 0\n",
                        opcode == OP_GET_LOCAL ? "OP_GET_LOCAL" : "OP_SET_LOCAL");
    vm_free(vm);
    return valid;
}

static int wind_valid(void) {
    VM* vm = vm_with_instruction(OP_WIND_PUSH, 0);
    if (!vm) return 0;
    vm_push(vm, INT_VAL(1));
    vm_push(vm, INT_VAL(2));
    vm_run(vm);
    int valid = !vm->error && vm->halted && vm->n_winds == 1 &&
        vm->wind_stack[0].before.as.i == 1 && vm->wind_stack[0].after.as.i == 2;
    if (!valid) fprintf(stderr, "FAIL: valid OP_WIND_PUSH\n");
    vm_free(vm);
    return valid;
}

static int callcc_valid(void) {
    VM* vm = vm_with_instruction(OP_CALLCC, 0);
    if (!vm) return 0;
    int32_t proc_ptr = heap_alloc(&vm->heap);
    if (proc_ptr < 0) { vm_free(vm); return 0; }
    vm->heap.objects[proc_ptr]->type = HEAP_CLOSURE;
    vm->heap.objects[proc_ptr]->closure.func_pc = 2;
    vm->heap.objects[proc_ptr]->closure.arity = 1;
    vm_push(vm, (Value){.type = VAL_CLOSURE, .as.ptr = proc_ptr});
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    int valid = !vm->error && vm->halted;
    if (!valid) fprintf(stderr, "FAIL: valid OP_CALLCC\n");
    vm_free(vm);
    return valid;
}

static int behavior_case(const char* name) {
    if (!strcmp(name, "local_get_oob"))
        return local_refusal(name, OP_GET_LOCAL, STACK_SIZE - 1, 1);
    if (!strcmp(name, "local_set_oob"))
        return local_refusal(name, OP_SET_LOCAL, STACK_SIZE - 1, 1);
    if (!strcmp(name, "local_get_negative"))
        return local_refusal(name, OP_GET_LOCAL, 0, -1);
    if (!strcmp(name, "local_get_int32_max"))
        return local_refusal(name, OP_GET_LOCAL, 1, INT32_MAX);
    if (!strcmp(name, "wind_overflow")) return wind_refusal();
    if (!strcmp(name, "callcc_bad_state")) return callcc_refusal();
    if (!strcmp(name, "local_get_valid")) return local_valid(OP_GET_LOCAL);
    if (!strcmp(name, "local_set_valid")) return local_valid(OP_SET_LOCAL);
    if (!strcmp(name, "wind_valid")) return wind_valid();
    if (!strcmp(name, "callcc_valid")) return callcc_valid();
    fprintf(stderr, "FAIL: unknown dispatch behavior case %s\n", name);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2) {
        if (!behavior_case(argv[1])) return 1;
        printf("PASS: %s\n", argv[1]);
        return 0;
    }
    if (argc != 1) return 2;

    VM* vm = vm_create();
    if (!vm) return 1;
    vm->code_len = 0;
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    if (vm->error || !vm->halted) { vm_free(vm); return 1; }
    vm_free(vm);

    unsigned tested = 0;
    for (unsigned op = OP_COUNT; op <= UINT8_MAX; ++op) {
        vm = vm_create();
        if (!vm) return 1;
        vm->code_len = 0;
        emit(vm, (uint8_t)op, 0);
        emit(vm, OP_HALT, 0);
        vm_run(vm);
        if (!vm->error || vm->pc != 1 || vm->n_outputs != 0) {
            fprintf(stderr, "FAIL: unsafe opcode %u dispatch\n", op);
            vm_free(vm);
            return 1;
        }
        vm_free(vm);
        ++tested;
    }
    if (!tested) return 1;
    printf("PASS: rejected %u invalid runtime opcodes; valid HALT accepted\n", tested);
    return 0;
}
