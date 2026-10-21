extern "C" {
#include "pjrt_c_api.h"
}

namespace {

struct ShortApiTable {
    size_t struct_size;
};

ShortApiTable api = {sizeof(ShortApiTable)};

}  // namespace

extern "C" const PJRT_Api* GetPjrtApi() {
    // The loader must reject this after reading only the leading struct_size.
    return reinterpret_cast<const PJRT_Api*>(&api);
}
