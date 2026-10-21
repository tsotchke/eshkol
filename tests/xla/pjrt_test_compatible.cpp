extern "C" {
#include "pjrt_c_api.h"
}

namespace {

void unusedFunction() {}

PJRT_Api api = {};
PJRT_Extension_Base initialized_extension = {};

PJRT_Error* initialize(PJRT_Plugin_Initialize_Args*) {
    api.extension_start = &initialized_extension;
    return nullptr;
}

struct InitializeApi {
    InitializeApi() {
        api.struct_size = sizeof(api);
        api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
        api.pjrt_api_version.major_version = PJRT_API_MAJOR;
        api.pjrt_api_version.minor_version = PJRT_API_MINOR;
        api.PJRT_Plugin_Initialize = &initialize;

#define SET_UNUSED_PJRT_FUNCTION(field) \
        api.field = reinterpret_cast<decltype(api.field)>(&unusedFunction)
        SET_UNUSED_PJRT_FUNCTION(PJRT_Error_Message);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Error_Destroy);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Client_Create);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Client_Destroy);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Client_PlatformName);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Client_Devices);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Client_Compile);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Client_BufferFromHostBuffer);
        SET_UNUSED_PJRT_FUNCTION(PJRT_DeviceDescription_Id);
        SET_UNUSED_PJRT_FUNCTION(PJRT_DeviceDescription_Kind);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Device_GetDescription);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Device_IsAddressable);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Executable_Destroy);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Executable_NumOutputs);
        SET_UNUSED_PJRT_FUNCTION(PJRT_LoadedExecutable_Destroy);
        SET_UNUSED_PJRT_FUNCTION(PJRT_LoadedExecutable_AddressableDevices);
        SET_UNUSED_PJRT_FUNCTION(PJRT_LoadedExecutable_GetExecutable);
        SET_UNUSED_PJRT_FUNCTION(PJRT_LoadedExecutable_Execute);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Buffer_Destroy);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Buffer_Dimensions);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Buffer_ToHostBuffer);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Event_Await);
        SET_UNUSED_PJRT_FUNCTION(PJRT_Event_Destroy);
#undef SET_UNUSED_PJRT_FUNCTION
    }
} initialize_api;

}  // namespace

extern "C" const PJRT_Api* GetPjrtApi() { return &api; }
