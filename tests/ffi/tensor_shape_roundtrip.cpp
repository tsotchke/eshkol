#include <eshkol/eshkol_ffi.h>
#include <eshkol/core/runtime.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>

static bool ffi_shape(eshkol_ffi_value_t value, int64_t rows, int64_t cols) {
    int64_t observed[2] = {-1, -1};
    const int copied = eshkol_ffi_tensor_shape(value, observed, 2);
    if (copied != 2 || observed[0] != rows || observed[1] != cols) {
        std::fprintf(stderr, "expected %" PRId64 " x %" PRId64
                     ", copied %d axes: %" PRId64 " x %" PRId64 "\n",
                     rows, cols, copied, observed[0], observed[1]);
        return false;
    }
    return eshkol_ffi_tensor_ndims(value) == 2 &&
           eshkol_ffi_tensor_size(value) == rows * cols;
}

static bool public_boundary() {
    int64_t dims[] = {3, 2};
    double payload[] = {1, 2, 3, 4, 5, 6};
    auto v = eshkol_ffi_tensor_from_data(nullptr, payload, dims, 2);
    auto alias = v;
    if (!ffi_shape(v, 3, 2)) return false;
    std::swap(dims[0], dims[1]);
    payload[0] = 17;
    auto other = eshkol_ffi_tensor_zeros(nullptr, dims, 2);
    if (!ffi_shape(other, 2, 3) || !ffi_shape(v, 3, 2)) return false;
    int64_t copy[2] = {};
    if (eshkol_ffi_tensor_shape(alias, copy, 2) != 2) return false;
    std::swap(copy[0], copy[1]);
    double* data = eshkol_ffi_tensor_data(alias);
    if (!data || data[0] != 1) return false;
    data[0] = 11;
    if (eshkol_ffi_tensor_data(v)[0] != 11 || !ffi_shape(v, 3, 2)) return false;

    // Exercise every supported rank, including a zero-length axis.
    for (int rank = 1; rank <= 8; ++rank) {
        std::array<int64_t, 8> input{}, observed{};
        input.fill(2);
        input[rank - 1] = 0;
        auto empty = eshkol_ffi_tensor_zeros(nullptr, input.data(), rank);
        if (eshkol_ffi_tensor_shape(empty, observed.data(), rank) != rank ||
            eshkol_ffi_tensor_size(empty) != 0 ||
            !std::equal(input.begin(), input.begin() + rank, observed.begin())) return false;
    }
    const int64_t negative[] = {2, -1};
    const int64_t overflow[] = {INT64_MAX, 2};
    return eshkol_ffi_is_null(eshkol_ffi_tensor_zeros(nullptr, negative, 2)) &&
           eshkol_ffi_is_null(eshkol_ffi_tensor_from_data(nullptr, payload, overflow, 2));
}

int main() {
    eshkol_runtime_init();
    if (!public_boundary()) {
        std::fputs("FFI tensor shape or ownership mismatch\n", stderr);
        return 1;
    }
    std::puts("PASS: C FFI tensor shape round-trip");
}
