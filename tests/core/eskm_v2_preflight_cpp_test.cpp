// Link a C++ consumer against the C object, exercising C linkage and ownership.
#include "eskm_v2_preflight.h"

#include <array>
#include <cstdio>
#include <type_traits>

static_assert(std::is_standard_layout<eskm_v2_result>::value, "C-compatible result");
static_assert(sizeof(eskm_v2_workspace) == 16384, "fixed caller workspace");

int main() {
    constexpr std::array<uint8_t, 28> empty = {
        0x45,0x53,0x4b,0x4d,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0x70,0xe8,0xc3
    };
    const eskm_v2_limits limits = ESKM_V2_LIMITS_DEFAULT;
    eskm_v2_workspace workspace{};
    eskm_v2_result result{};
    const auto success = eskm_v2_preflight(empty.data(), empty.size(), &limits, &workspace, &result);
    if (success.status != ESKM_V2_OK || success.offset != 0 ||
        result.extension_offset != 24 || result.extension_bytes != 0 ||
        result.records_offset != 24 || result.records_bytes != 0 ||
        result.record_count != 0 || result.tlv_count != 0 ||
        result.annotation_count != 0 || result.has_annotations != 0) return 1;
    const auto error = eskm_v2_preflight(nullptr, 0, &limits, &workspace, &result);
    if (error.status != ESKM_V2_INVALID_ARGUMENT || error.offset != 0 ||
        result.extension_offset != 0 || result.records_offset != 0) return 1;
    std::puts("ESKM v2 C++ consumer: C linkage and result contract passed");
    return 0;
}
