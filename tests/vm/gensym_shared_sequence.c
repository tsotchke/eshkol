#define ESHKOL_VM_LIBRARY_MODE 1
#include "../../lib/backend/eshkol_vm.c"

struct arena;
extern struct arena* arena_create(size_t);
extern void arena_destroy(struct arena*);
extern void* eshkol_gensym_ptr(void*);

int main(void) {
    VM* vm = vm_create();
    struct arena* arena = arena_create(4096);
    if (!vm || !arena) return 1;
    const char* first = eshkol_gensym_ptr(arena);
    vm_dispatch_native(vm, 2227);
    Value middle = vm_pop(vm);
    VmString* text = vm_value_as_string(vm, middle);
    const char* last = eshkol_gensym_ptr(arena);
    int ok = first && text && last && middle.type == VAL_SYMBOL &&
             strcmp(first, "G1") == 0 && strcmp(text->data, "G2") == 0 &&
             strcmp(last, "G3") == 0;
    vm_free(vm);
    arena_destroy(arena);
    return ok ? 0 : 1;
}
