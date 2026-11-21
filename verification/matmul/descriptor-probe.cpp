// Exercise the public constructors and a separate, fixture-owned native descriptor.
#include <eshkol/core/runtime.h>
#include <eshkol/eshkol_ffi.h>
#include <eshkol/tensor_validation.h>
#include "arena_memory.h"

#include <algorithm>
#include <array>
#include <cstdio>

static bool ffi_shape(eshkol_ffi_value_t value, int64_t rows, int64_t cols) {
    int64_t observed[2] = {-1, -1};
    return eshkol_ffi_tensor_ndims(value) == 2 &&
           eshkol_ffi_tensor_shape(value, observed, 2) == 2 &&
           observed[0] == rows && observed[1] == cols &&
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
    return true;
}

static bool valid_shape(const eshkol_tensor_t& tensor, int64_t rows, int64_t cols) {
    if (tensor.num_dimensions != 2 || tensor.dimensions[0] != static_cast<uint64_t>(rows) ||
        tensor.dimensions[1] != static_cast<uint64_t>(cols)) return false;
    const int64_t dims[] = {rows, cols};
    return eshkol_tensor_metadata_valid(dims, tensor.num_dimensions,
                                        tensor.elements, tensor.total_elements);
}

static bool owned_descriptor() {
    uint64_t dimensions[] = {3, 2};
    int64_t elements[6] = {};
    eshkol_tensor_t object{dimensions, 2, elements, 6, 0};
    const eshkol_tensor_t* const reader = &object;
    eshkol_tensor_t& writer = object;
    if (!valid_shape(*reader, 3, 2)) return false;
    std::swap(writer.dimensions[0], writer.dimensions[1]);
    return reader == &writer && reader->elements == elements &&
           reader->dimensions == dimensions && valid_shape(*reader, 2, 3);
}

int main() {
    eshkol_runtime_init();
    if (!public_boundary()) {
        std::fputs("FFI tensor shape or ownership mismatch\n", stderr);
        return 1;
    }
    if (!owned_descriptor()) {
        std::fputs("Native descriptor identity/shape control failed\n", stderr);
        return 1;
    }
    std::puts("public-ffi: dimensions copied, element mutation observed, shape preserved");
    std::puts("owned-native-descriptor: same object and storage, 3x2 -> 2x3, both shapes valid");
}
