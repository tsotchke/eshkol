extern "C" {
#include "pjrt_c_api.h"
}

#include <cstdlib>
#include <cstring>

namespace {

struct FakeHandle { int value = 0; };
FakeHandle client_handle, device_handle, description_handle, executable_handle,
    loaded_executable_handle, input_buffer_handle, output_buffer_handle,
    second_output_buffer_handle,
    transfer_event_handle, execution_event_handle, initialized_extension;
PJRT_Api api = {};
PJRT_Error compile_error_handle;
PJRT_Device* devices[] = {reinterpret_cast<PJRT_Device*>(&device_handle)};
int destroyed_buffers = 0;
int destroyed_executables = 0;
int host_copies = 0;
int successful_host_copies = 0;
int host_transfers = 0;
int compile_calls = 0;
int execute_calls = 0;
int destroyed_clients = 0;
int last_input_element_type = 0;
const int64_t input_dims[] = {1};
const int64_t dtype_output_dims[] = {1};
const int64_t shape_output_dims[] = {3, 2};

PJRT_Error* initialize(PJRT_Plugin_Initialize_Args*) {
    api.extension_start = reinterpret_cast<PJRT_Extension_Base*>(&initialized_extension);
    return nullptr;
}
PJRT_Error* clientCreate(PJRT_Client_Create_Args* a) {
    a->client = reinterpret_cast<PJRT_Client*>(&client_handle);
    return nullptr;
}
PJRT_Error* clientDestroy(PJRT_Client_Destroy_Args*) {
    ++destroyed_clients;
    return nullptr;
}
PJRT_Error* platformName(PJRT_Client_PlatformName_Args* a) {
    a->platform_name = "cpu"; a->platform_name_size = 3; return nullptr;
}
PJRT_Error* clientDevices(PJRT_Client_Devices_Args* a) {
    a->devices = devices; a->num_devices = 1; return nullptr;
}
PJRT_Error* deviceDescription(PJRT_Device_GetDescription_Args* a) {
    a->device_description = reinterpret_cast<PJRT_DeviceDescription*>(&description_handle);
    return nullptr;
}
PJRT_Error* deviceId(PJRT_DeviceDescription_Id_Args* a) { a->id = 0; return nullptr; }
PJRT_Error* deviceKind(PJRT_DeviceDescription_Kind_Args* a) {
    a->device_kind = "offline test device"; a->device_kind_size = 19; return nullptr;
}
PJRT_Error* deviceAddressable(PJRT_Device_IsAddressable_Args* a) {
    a->is_addressable = true; return nullptr;
}
PJRT_Error* compile(PJRT_Client_Compile_Args* a) {
    ++compile_calls;
    const char* fail = std::getenv("PJRT_TEST_COMPILE_FAIL");
    if (fail && std::strcmp(fail, "1") == 0) return &compile_error_handle;
    a->executable = reinterpret_cast<PJRT_LoadedExecutable*>(&loaded_executable_handle);
    return nullptr;
}
PJRT_Error* getExecutable(PJRT_LoadedExecutable_GetExecutable_Args* a) {
    a->executable = reinterpret_cast<PJRT_Executable*>(&executable_handle);
    return nullptr;
}
PJRT_Error* numOutputs(PJRT_Executable_NumOutputs_Args* a) {
    const char* two = std::getenv("PJRT_TEST_TWO_OUTPUTS");
    a->num_outputs = two && std::strcmp(two, "1") == 0 ? 2 : 1; return nullptr;
}
PJRT_Error* destroyExecutable(PJRT_Executable_Destroy_Args*) { return nullptr; }
PJRT_Error* destroyLoadedExecutable(PJRT_LoadedExecutable_Destroy_Args*) {
    ++destroyed_executables; return nullptr;
}
PJRT_Error* execute(PJRT_LoadedExecutable_Execute_Args* a) {
    ++execute_calls;
    a->output_lists[0][0] = reinterpret_cast<PJRT_Buffer*>(&output_buffer_handle);
    const char* two = std::getenv("PJRT_TEST_TWO_OUTPUTS");
    if (two && std::strcmp(two, "1") == 0)
        a->output_lists[0][1] = reinterpret_cast<PJRT_Buffer*>(&second_output_buffer_handle);
    a->device_complete_events[0] = reinterpret_cast<PJRT_Event*>(&execution_event_handle);
    return nullptr;
}
PJRT_Error* bufferFromHost(PJRT_Client_BufferFromHostBuffer_Args* a) {
    ++host_transfers;
    last_input_element_type = static_cast<int>(a->type);
    a->buffer = reinterpret_cast<PJRT_Buffer*>(&input_buffer_handle);
    a->done_with_host_buffer = reinterpret_cast<PJRT_Event*>(&transfer_event_handle);
    return nullptr;
}
PJRT_Error* bufferType(PJRT_Buffer_ElementType_Args* a) {
    const char* mode = std::getenv("PJRT_TEST_OUTPUT_MISMATCH");
    const char* configured = std::getenv("PJRT_TEST_DEVICE_OUTPUT_TYPE");
    const bool output = a->buffer == reinterpret_cast<PJRT_Buffer*>(&output_buffer_handle) ||
        a->buffer == reinterpret_cast<PJRT_Buffer*>(&second_output_buffer_handle);
    if (output && mode && std::strcmp(mode, "dtype") == 0) a->type = PJRT_Buffer_Type_F32;
    else if (output && configured && std::strcmp(configured, "f32") == 0) a->type = PJRT_Buffer_Type_F32;
    else if (output && configured && std::strcmp(configured, "bf16") == 0) a->type = PJRT_Buffer_Type_BF16;
    else if (output && configured && std::strcmp(configured, "s32") == 0) a->type = PJRT_Buffer_Type_S32;
    else if (output && configured && std::strcmp(configured, "u16") == 0) a->type = PJRT_Buffer_Type_U16;
    else a->type = PJRT_Buffer_Type_F64;
    return nullptr;
}
PJRT_Error* bufferDimensions(PJRT_Buffer_Dimensions_Args* a) {
    const char* mode = std::getenv("PJRT_TEST_OUTPUT_MISMATCH");
    if (a->buffer == reinterpret_cast<PJRT_Buffer*>(&output_buffer_handle) &&
        mode && std::strcmp(mode, "shape") == 0) {
        a->dims = shape_output_dims; a->num_dims = 2;
    } else {
        a->dims = input_dims; a->num_dims = 1;
    }
    return nullptr;
}
PJRT_Error* bufferToHost(PJRT_Buffer_ToHostBuffer_Args* a) {
    ++host_copies;
    const char* fail_second = std::getenv("PJRT_TEST_FAIL_SECOND_COPY");
    if (fail_second && std::strcmp(fail_second, "1") == 0 && host_copies == 2)
        return &compile_error_handle;
    if (a->dst && a->dst_size == sizeof(double)) {
        const double value = 8.25;
        std::memcpy(a->dst, &value, sizeof(value));
    } else if (a->dst && a->dst_size == sizeof(uint32_t)) {
        const uint32_t bits = 0x3f800001u;
        std::memcpy(a->dst, &bits, sizeof(bits));
    } else if (a->dst && a->dst_size == sizeof(uint16_t)) {
        const uint16_t bits = 0x3f81u;
        std::memcpy(a->dst, &bits, sizeof(bits));
    } else if (a->dst && a->dst_size) std::memset(a->dst, 0xA5, a->dst_size);
    ++successful_host_copies;
    return nullptr;
}
PJRT_Error* destroyBuffer(PJRT_Buffer_Destroy_Args*) {
    ++destroyed_buffers; return nullptr;
}
PJRT_Error* eventAwait(PJRT_Event_Await_Args*) { return nullptr; }
PJRT_Error* eventDestroy(PJRT_Event_Destroy_Args*) { return nullptr; }
PJRT_Error* unused(PJRT_Error_Message_Args*) { return nullptr; }
void unusedDestroy(PJRT_Error_Destroy_Args*) {}
void compileErrorMessage(PJRT_Error_Message_Args* a) {
    a->message = "offline compile failure"; a->message_size = 23;
}

struct InitializeApi {
    InitializeApi() {
        api.struct_size = sizeof(api);
        api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
        api.pjrt_api_version.major_version = PJRT_API_MAJOR;
        api.pjrt_api_version.minor_version = PJRT_API_MINOR;
        api.PJRT_Plugin_Initialize = &initialize;
#define SET_UNUSED(field) api.field = reinterpret_cast<decltype(api.field)>(&unused)
        SET_UNUSED(PJRT_Error_Message); SET_UNUSED(PJRT_Error_Destroy);
        SET_UNUSED(PJRT_Client_Create); SET_UNUSED(PJRT_Client_Destroy);
        SET_UNUSED(PJRT_Client_PlatformName); SET_UNUSED(PJRT_Client_Devices);
        SET_UNUSED(PJRT_Client_Compile); SET_UNUSED(PJRT_Client_BufferFromHostBuffer);
        SET_UNUSED(PJRT_DeviceDescription_Id); SET_UNUSED(PJRT_DeviceDescription_Kind);
        SET_UNUSED(PJRT_Device_GetDescription); SET_UNUSED(PJRT_Device_IsAddressable);
        SET_UNUSED(PJRT_Executable_Destroy); SET_UNUSED(PJRT_Executable_NumOutputs);
        SET_UNUSED(PJRT_LoadedExecutable_Destroy);
        SET_UNUSED(PJRT_LoadedExecutable_AddressableDevices);
        SET_UNUSED(PJRT_LoadedExecutable_GetExecutable);
        SET_UNUSED(PJRT_LoadedExecutable_Execute); SET_UNUSED(PJRT_Buffer_Destroy);
        SET_UNUSED(PJRT_Buffer_ElementType); SET_UNUSED(PJRT_Buffer_Dimensions);
        SET_UNUSED(PJRT_Buffer_ToHostBuffer); SET_UNUSED(PJRT_Event_Await);
        SET_UNUSED(PJRT_Event_Destroy);
#undef SET_UNUSED
        api.PJRT_Client_Create = &clientCreate;
        api.PJRT_Error_Message = &compileErrorMessage;
        api.PJRT_Error_Destroy = &unusedDestroy;
        api.PJRT_Client_Destroy = &clientDestroy;
        api.PJRT_Client_PlatformName = &platformName;
        api.PJRT_Client_Devices = &clientDevices;
        api.PJRT_Client_Compile = &compile;
        api.PJRT_Client_BufferFromHostBuffer = &bufferFromHost;
        api.PJRT_DeviceDescription_Id = &deviceId;
        api.PJRT_DeviceDescription_Kind = &deviceKind;
        api.PJRT_Device_GetDescription = &deviceDescription;
        api.PJRT_Device_IsAddressable = &deviceAddressable;
        api.PJRT_Executable_Destroy = &destroyExecutable;
        api.PJRT_Executable_NumOutputs = &numOutputs;
        api.PJRT_LoadedExecutable_Destroy = &destroyLoadedExecutable;
        api.PJRT_LoadedExecutable_GetExecutable = &getExecutable;
        api.PJRT_LoadedExecutable_Execute = &execute;
        api.PJRT_Buffer_Destroy = &destroyBuffer;
        api.PJRT_Buffer_ElementType = &bufferType;
        api.PJRT_Buffer_Dimensions = &bufferDimensions;
        api.PJRT_Buffer_ToHostBuffer = &bufferToHost;
        api.PJRT_Event_Await = &eventAwait;
        api.PJRT_Event_Destroy = &eventDestroy;
    }
} initialize_api;

}  // namespace

extern "C" const PJRT_Api* GetPjrtApi() { return &api; }
extern "C" int PjrtTestDestroyedBuffers() { return destroyed_buffers; }
extern "C" int PjrtTestDestroyedExecutables() { return destroyed_executables; }
extern "C" int PjrtTestHostCopies() { return host_copies; }
extern "C" int PjrtTestSuccessfulHostCopies() { return successful_host_copies; }
extern "C" int PjrtTestHostTransfers() { return host_transfers; }
extern "C" int PjrtTestLastInputElementType() { return last_input_element_type; }
extern "C" int PjrtTestCompileCalls() { return compile_calls; }
extern "C" int PjrtTestExecuteCalls() { return execute_calls; }
extern "C" int PjrtTestDestroyedClients() { return destroyed_clients; }
