#include <stddef.h>
#include <stdint.h>

/* An independently compiled wasm32 caller of the production import ABI. */
typedef struct { uint64_t *dimensions; uint64_t rank; int64_t *elements;
                 uint64_t total; uint64_t dtype; } tensor;
typedef struct { uint8_t type, flags; uint16_t reserved; uint64_t data; } tagged;
_Static_assert(sizeof(tensor) == 40 && offsetof(tensor, rank) == 8 &&
               offsetof(tensor, elements) == 16 && offsetof(tensor, total) == 24 &&
               offsetof(tensor, dtype) == 32 && sizeof(tagged) == 16, "wasm32 tensor ABI");
extern tensor *eshkol_jet_tensor_binary(void *, const tensor *, const tensor *, int, int, const char *);
extern tensor *eshkol_jet_tensor_matmul(void *, const tensor *, const tensor *, int, const char *);
__attribute__((export_name("binary")))
uintptr_t binary(uintptr_t a, uintptr_t b, int op, int reverse) {
    return (uintptr_t)eshkol_jet_tensor_binary((void *)1, (tensor *)a, (tensor *)b, op, reverse, 0);
}
__attribute__((export_name("matmul")))
uintptr_t matmul(uintptr_t a, uintptr_t b, int reverse) {
    return (uintptr_t)eshkol_jet_tensor_matmul((void *)1, (tensor *)a, (tensor *)b, reverse, 0);
}
