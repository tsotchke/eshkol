#include "../../lib/core/arena_memory.h"

#include <cmath>
#include <cstdio>
#include <limits>

extern "C" void eshkol_ad_node_gradient_tagged(
    void* arena, void* node, eshkol_tagged_value_t* out);

int main() {
    ad_node_t node{};
    eshkol_tagged_value_t exact = eshkol_make_int64(37, true);
    node.gradient = std::numeric_limits<double>::quiet_NaN();
    node.exact_gradient = &exact;

    eshkol_tagged_value_t out{};
    eshkol_ad_node_gradient_tagged(get_global_arena(), &node, &out);
    if ((out.type & 0x0f) != ESHKOL_VALUE_INT64 || out.data.int_val != 37) {
        std::fprintf(stderr,
            "ad_tagged_readback_canary_test: FAIL: exact sidecar was bypassed\n");
        return 1;
    }

    node.exact_gradient = nullptr;
    eshkol_ad_node_gradient_tagged(get_global_arena(), &node, &out);
    if ((out.type & 0x0f) != ESHKOL_VALUE_DOUBLE ||
        !std::isnan(out.data.double_val)) {
        std::fprintf(stderr,
            "ad_tagged_readback_canary_test: FAIL: double fallback changed\n");
        return 1;
    }

    std::puts("ad_tagged_readback_canary_test: PASS");
    return 0;
}
