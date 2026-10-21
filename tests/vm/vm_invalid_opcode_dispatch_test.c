/* Exercise the runtime guard independently of serialized-module admission.
 * The internal VM is intentionally assembled directly, as other VM self-tests
 * do; production loading remains unchanged. Explicit verdicts survive NDEBUG. */
#define ESHKOL_VM_LIBRARY_MODE 1
#include "../../lib/backend/eshkol_vm.c"

int main(void) {
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
