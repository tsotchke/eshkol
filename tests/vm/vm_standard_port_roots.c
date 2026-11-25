#define ESHKOL_VM_LIBRARY_MODE 1
#include "../../lib/backend/eshkol_vm.c"

int main(void) {
    VM* vm = vm_create();
    if (!vm || !heap_region_push(&vm->heap, "ports", 4096)) return 1;
    Value parameters[3];
    for (int i = 0; i < 3; ++i) {
        parameters[i] = vm_std_port_parameter(vm, i);
        if (parameters[i].type != VAL_PARAMETER_OBJ) return 1;
    }
    if (!vm_evac_markbits_ensure(&vm->heap) || !vm_evac_mark_roots(vm)) return 1;
    for (int i = 0; i < 3; ++i)
        if (!vm_evac_marked(&vm->heap, parameters[i].as.ptr)) return 1;
    vm_free(vm);
    return 0;
}
