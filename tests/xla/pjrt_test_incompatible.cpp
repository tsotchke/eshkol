extern "C" {
#include "pjrt_c_api.h"
}

namespace {

PJRT_Api api = {};

struct InitializeVersion {
    InitializeVersion() {
        api.struct_size = sizeof(api);
        api.pjrt_api_version.major_version = PJRT_API_MAJOR + 1;
        api.pjrt_api_version.minor_version = PJRT_API_MINOR;
    }
} initialize_version;

}  // namespace

extern "C" const PJRT_Api* GetPjrtApi() { return &api; }
