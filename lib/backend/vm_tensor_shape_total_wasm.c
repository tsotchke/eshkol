#include <stdint.h>
#include <limits.h>

/* The Emscripten VM is built from C amalgamation units and does not link the
 * C++ tensor-validation TU. Keep this ABI helper identical to the hosted
 * implementation so generated tensor code resolves on the WASM path too. */
int64_t eshkol_tensor_shape_total(const int64_t *dims, int64_t ndim) {
    if (!dims || ndim <= 0) return -1;
    int64_t total = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        if (dims[i] < 0) return -1;
        if (dims[i] == 0) { total = 0; continue; }
        if (total > INT64_MAX / dims[i]) return -1;
        total *= dims[i];
    }
    if (total > INT64_MAX / 8) return -1;
    return total;
}
