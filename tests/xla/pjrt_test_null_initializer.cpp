extern "C" {
#include "pjrt_c_api.h"
}

namespace {

PJRT_Api api = {};

struct InitializeVersion {
    InitializeVersion() {
        api.struct_size = sizeof(api);
        api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
        api.pjrt_api_version.major_version = PJRT_API_MAJOR;
        api.pjrt_api_version.minor_version = PJRT_API_MINOR;
        // The intentionally null PJRT_Plugin_Initialize pointer is the fault
        // under test; the loader must reject before calling it.
    }
} initialize_version;

}  // namespace

extern "C" const PJRT_Api* GetPjrtApi() { return &api; }
