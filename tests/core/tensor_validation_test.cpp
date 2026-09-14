#include <eshkol/tensor_validation.h>

#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

int failures = 0;

void expect(bool condition, const char* name) {
    if (!condition) {
        std::printf("FAIL: %s\n", name);
        ++failures;
    }
}

uint64_t next(uint64_t& state) {
    state ^= state << 7;
    state ^= state >> 9;
    return state;
}

}  // namespace

int main() {
    const int64_t good[] = {2, 3, 4};
    expect(eshkol_tensor_shape_total(good, 3) == 24, "checked shape product");
    expect(eshkol_tensor_shape_total(good, 0) < 0, "rank zero rejected");

    const int64_t negative[] = {2, -3};
    const int64_t empty[] = {0, 4};
    const int64_t overflowing[] = {std::numeric_limits<int64_t>::max(), 2};
    expect(eshkol_tensor_shape_total(negative, 2) < 0, "negative dimension rejected");
    expect(eshkol_tensor_shape_total(empty, 2) == 0, "zero extent produces empty tensor");
    expect(eshkol_tensor_metadata_valid(empty, 2, nullptr, 0), "empty metadata accepted");
    expect(eshkol_tensor_shape_total(overflowing, 2) < 0, "overflowing product rejected");
    expect(eshkol_tensor_metadata_valid(good, 3, good, 24), "metadata accepted");
    expect(!eshkol_tensor_metadata_valid(good, 3, good, 23), "metadata count mismatch rejected");

    const int64_t a[] = {2, 1};
    const int64_t b[] = {1, 3};
    int64_t out[16] = {};
    int64_t rank = 0;
    int64_t total = 0;
    expect(eshkol_tensor_broadcast_shape(a, 2, b, 2, out, &rank, &total) &&
           rank == 2 && out[0] == 2 && out[1] == 3 && total == 6,
           "broadcast plan validated");
    const int64_t incompatible[] = {3, 2};
    expect(!eshkol_tensor_broadcast_shape(a, 2, incompatible, 2, out, &rank, &total),
           "incompatible broadcast rejected");
    const int64_t empty_broadcast[] = {0, 3};
    const int64_t singleton_broadcast[] = {1, 3};
    expect(eshkol_tensor_broadcast_shape(empty_broadcast, 2, singleton_broadcast, 2,
                                         out, &rank, &total) && out[0] == 0 && total == 0,
           "zero-extent broadcast preserved");

    const int64_t matrix[] = {2, 3};
    const int64_t full[] = {1, 2};
    const int64_t aliasing_bad[] = {1, -1};
    int64_t offset = 0;
    int64_t slice_total = 0;
    expect(eshkol_tensor_index_offset(matrix, 2, full, 2, &offset, &slice_total) &&
           offset == 5 && slice_total == 1, "full index validated");
    expect(!eshkol_tensor_index_offset(matrix, 2, aliasing_bad, 2, &offset, &slice_total),
           "per-axis index bounds reject alias");

    uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 10000; ++i) {
        int64_t shape[4];
        for (int d = 0; d < 4; ++d) shape[d] = (int64_t)(next(state) % 9) + 1;
        int64_t product = eshkol_tensor_shape_total(shape, 4);
        expect(product > 0 && product <= 6561, "bounded fuzz shape remains valid");
        int64_t index[4] = {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1};
        expect(eshkol_tensor_index_offset(shape, 4, index, 4, &offset, &slice_total) &&
               offset == product - 1 && slice_total == 1,
               "bounded fuzz index remains in range");
    }

    if (failures == 0) std::printf("PASS: tensor validation contract and bounded fuzz\n");
    return failures == 0 ? 0 : 1;
}
