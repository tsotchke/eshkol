#define ESHKOL_VM_LIBRARY_MODE 1
#include "../../lib/backend/eshkol_vm.c"

int main(void) {
    VM* vm = vm_create();
    if (!vm) return 1;
    uint64_t hashes[8];
    for (int i = 0; i < 8; ++i) {
        vm_push_i128(vm, i);
        Value first = vm_pop(vm);
        hashes[i] = vm_hash_key_hash(vm, &first);
        vm_push_i128(vm, i);
        Value second = vm_pop(vm);
        if (vm_hash_key_hash(vm, &second) != hashes[i]) return 1;
        for (int j = 0; j < i; ++j)
            if (hashes[i] == hashes[j]) return 1;
    }
    printf("%016llx\n", (unsigned long long)hashes[0]);
    vm_free(vm);
    return 0;
}
