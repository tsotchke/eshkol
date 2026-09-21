#include <eshkol/tensor_validation.h>

#include <limits>

extern "C" int64_t eshkol_tensor_shape_total(const int64_t* dims,
                                               int64_t ndim) {
    if (!dims || ndim <= 0) return -1;
    int64_t total = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        if (dims[i] < 0) {
            return -1;
        }
        if (dims[i] == 0) {
            total = 0;
            continue;
        }
        if (total > std::numeric_limits<int64_t>::max() / dims[i]) return -1;
        total *= dims[i];
    }
    if (total > std::numeric_limits<int64_t>::max() / 8) return -1;
    return total;
}

extern "C" int eshkol_tensor_metadata_valid(const int64_t* dims, int64_t ndim,
                                              const void* elements,
                                              int64_t total) {
    if (total < 0) return 0;
    if (eshkol_tensor_shape_total(dims, ndim) != total) return 0;
    return total == 0 || elements != nullptr;
}

extern "C" int eshkol_tensor_broadcast_shape(const int64_t* a_dims,
                                                int64_t a_ndim,
                                                const int64_t* b_dims,
                                                int64_t b_ndim,
                                                int64_t* out_dims,
                                                int64_t* out_ndim,
                                                int64_t* out_total) {
    if (!a_dims || !b_dims || !out_dims || !out_ndim || !out_total ||
        a_ndim <= 0 || b_ndim <= 0) return 0;
    const int64_t rank = a_ndim > b_ndim ? a_ndim : b_ndim;
    if (rank > 16) return 0;
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t a = i < a_ndim ? a_dims[a_ndim - 1 - i] : 1;
        const int64_t b = i < b_ndim ? b_dims[b_ndim - 1 - i] : 1;
        if (a < 0 || b < 0 || (a != b && a != 1 && b != 1)) return 0;
        out_dims[rank - 1 - i] = a == 1 ? b : a;
    }
    const int64_t total = eshkol_tensor_shape_total(out_dims, rank);
    if (total < 0) return 0;
    *out_ndim = rank;
    *out_total = total;
    return 1;
}

extern "C" int eshkol_tensor_index_offset(const int64_t* dims,
                                             int64_t ndim,
                                             const int64_t* indices,
                                             int64_t n_indices,
                                             int64_t* offset,
                                             int64_t* slice_total) {
    if (!dims || !indices || !offset || !slice_total || ndim <= 0 ||
        n_indices <= 0 || n_indices > ndim) return 0;
    const int64_t total = eshkol_tensor_shape_total(dims, ndim);
    if (total < 0) return 0;

    int64_t linear = 0;
    for (int64_t i = 0; i < n_indices; ++i) {
        if (indices[i] < 0 || indices[i] >= dims[i]) return 0;
        if (linear > (std::numeric_limits<int64_t>::max() - indices[i]) /
                         dims[i]) return 0;
        linear = linear * dims[i] + indices[i];
    }
    int64_t remaining = 1;
    for (int64_t i = n_indices; i < ndim; ++i) {
        if (remaining > std::numeric_limits<int64_t>::max() / dims[i]) return 0;
        remaining *= dims[i];
    }
    if (linear > std::numeric_limits<int64_t>::max() / remaining) return 0;
    linear *= remaining;
    if (linear < 0 || linear >= total) return 0;
    *offset = linear;
    *slice_total = remaining;
    return 1;
}
