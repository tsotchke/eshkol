/**
 * @file pjrt_client.cpp
 * @brief PJRT plugin loading and device execution. See pjrt_client.h for why.
 *
 * ABI RULES THIS FILE OBEYS, because getting any of them wrong corrupts memory
 * in ways that surface far from the cause:
 *
 *  1. Every Args struct is zero-initialised with `= {}` and then has
 *     struct_size set from the header's PJRT_<name>_Args_STRUCT_SIZE macro.
 *     struct_size is the ABI's forward-compatibility mechanism: it tells a
 *     newer plugin where this caller's knowledge of the struct stopped. Setting
 *     it to sizeof(Args) would be wrong on a plugin older than us.
 *
 *  2. `extension_start` is always nullptr. We use no extensions.
 *
 *  3. Every PJRT_Error* returned is consumed exactly once by consumeError(),
 *     which extracts the message and destroys it. A dropped error leaks; a
 *     doubly-destroyed one is a use-after-free.
 *
 *  4. Out-params are only read after the call returned no error.
 *
 *  5. Strings from the API are (pointer, size) pairs and are NOT
 *     null-terminated. They must be constructed with the explicit length.
 */

#include "eshkol/backend/xla/pjrt_client.h"

#include <dlfcn.h>

#include <cstring>
#include <cstdlib>

extern "C" {
#include "pjrt_c_api.h"
}

namespace eshkol {
namespace xla {

namespace {

/**
 * @brief Read a PJRT (pointer, size) string, which is not null-terminated.
 */
std::string pjrtString(const char* data, size_t size) {
    if (!data || size == 0) return std::string();
    return std::string(data, size);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────
// PjrtPlugin
// ─────────────────────────────────────────────────────────────────────────

PjrtPlugin::~PjrtPlugin() {
    // The API table and every handle derived from it belong to the shared
    // object, so the dlclose must come last and must not happen while any
    // client is alive. PjrtClient holds a non-owning pointer to us precisely
    // so that ownership order is the caller's explicit responsibility.
    if (handle_) {
        dlclose(handle_);
        handle_ = nullptr;
    }
    api_ = nullptr;
}

std::unique_ptr<PjrtPlugin> PjrtPlugin::load(const std::string& library_path,
                                             std::string* error) {
    auto fail = [&](const std::string& msg) -> std::unique_ptr<PjrtPlugin> {
        if (error) *error = msg;
        return nullptr;
    };

    if (library_path.empty()) {
        return fail("PJRT plugin path is empty; nothing to load");
    }

    // RTLD_LOCAL so a plugin's symbols do not leak into the global namespace
    // and collide with the LLVM/MLIR we are already linked against — libtpu is
    // 359 MB and statically contains much of XLA.
    dlerror();  // clear any stale error
    void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* dl = dlerror();
        return fail("dlopen(" + library_path + ") failed: " +
                    (dl ? dl : "unknown error"));
    }

    // A PJRT plugin is defined by exporting exactly this one symbol. Anything
    // that does not is simply not a plugin, which is worth saying plainly
    // because the alternative diagnostic is a null-pointer crash later.
    dlerror();
    auto get_api = reinterpret_cast<const PJRT_Api* (*)()>(
        dlsym(handle, "GetPjrtApi"));
    if (!get_api) {
        const char* dl = dlerror();
        dlclose(handle);
        return fail(library_path + " does not export GetPjrtApi, so it is not "
                    "a PJRT plugin" + (dl ? std::string(": ") + dl : ""));
    }

    const PJRT_Api* api = get_api();
    if (!api) {
        dlclose(handle);
        return fail(library_path + " GetPjrtApi() returned null");
    }

    // A major-version mismatch means the Args struct layouts differ. Refusing
    // here is deliberate: continuing would write past the end of structs the
    // plugin allocated differently, and the resulting corruption would present
    // as an unrelated crash much later, in code that is not at fault.
    const int plugin_major = api->pjrt_api_version.major_version;
    const int plugin_minor = api->pjrt_api_version.minor_version;
    if (plugin_major != PJRT_API_MAJOR) {
        dlclose(handle);
        return fail(library_path + " reports PJRT API major version " +
                    std::to_string(plugin_major) + " but this build expects " +
                    std::to_string(PJRT_API_MAJOR) +
                    "; struct layouts differ across a major version and "
                    "proceeding would corrupt memory");
    }

    // ONE-TIME PLUGIN SETUP, and the ABI is explicit that it "must be called
    // before any other functions are called."
    //
    // Skipping it does not produce a clean error. libtpu aborts the process
    // from inside PJRT_Client_Create with
    //
    //     Path: /dev_borg/accel[0-9]*: InitGoogle() has not finished yet.
    //
    // which names an internal path and reads like a broken machine or a driver
    // problem rather than a missing call. It cost real time to trace back to
    // here, so it is worth stating plainly: the crash is the plugin's own
    // initialisation framework noticing it was never started.
    PJRT_Plugin_Initialize_Args init_args = {};
    init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    init_args.extension_start = nullptr;
    if (PJRT_Error* init_error = api->PJRT_Plugin_Initialize(&init_args)) {
        std::string message;
        PJRT_Error_Message_Args msg_args = {};
        msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
        msg_args.extension_start = nullptr;
        msg_args.error = init_error;
        api->PJRT_Error_Message(&msg_args);
        if (msg_args.message != nullptr) {
            message.assign(msg_args.message, msg_args.message_size);
        }
        PJRT_Error_Destroy_Args destroy_args = {};
        destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        destroy_args.extension_start = nullptr;
        destroy_args.error = init_error;
        api->PJRT_Error_Destroy(&destroy_args);
        dlclose(handle);
        return fail(library_path + " PJRT_Plugin_Initialize failed: " + message);
    }

    std::unique_ptr<PjrtPlugin> plugin(new PjrtPlugin());
    plugin->handle_ = handle;
    plugin->api_ = api;
    plugin->api_major_ = plugin_major;
    plugin->api_minor_ = plugin_minor;
    plugin->path_ = library_path;
    return plugin;
}

// ─────────────────────────────────────────────────────────────────────────
// PjrtClient
// ─────────────────────────────────────────────────────────────────────────

PjrtStatus PjrtClient::consumeError(PJRT_Error* error) const {
    if (!error) return PjrtStatus();

    const PJRT_Api* api = plugin_->api();

    PJRT_Error_Message_Args msg = {};
    msg.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    msg.extension_start = nullptr;
    msg.error = error;
    api->PJRT_Error_Message(&msg);
    std::string text = pjrtString(msg.message, msg.message_size);

    // The error is owned by us from the moment it is returned; destroying it is
    // not optional and there is no other opportunity to do so.
    PJRT_Error_Destroy_Args destroy = {};
    destroy.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy.extension_start = nullptr;
    destroy.error = error;
    api->PJRT_Error_Destroy(&destroy);

    if (text.empty()) text = "PJRT call failed without a message";
    return PjrtStatus(std::move(text));
}

std::unique_ptr<PjrtClient> PjrtClient::create(PjrtPlugin* plugin,
                                               std::string* error) {
    if (!plugin || !plugin->api()) {
        if (error) *error = "PjrtClient::create called with no loaded plugin";
        return nullptr;
    }

    std::unique_ptr<PjrtClient> client(new PjrtClient());
    client->plugin_ = plugin;

    PJRT_Client_Create_Args args = {};
    args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.create_options = nullptr;
    args.num_options = 0;

    PJRT_Error* err = plugin->api()->PJRT_Client_Create(&args);
    if (err) {
        PjrtStatus status = client->consumeError(err);
        if (error) *error = "PJRT_Client_Create: " + status.message();
        return nullptr;
    }

    client->client_ = args.client;
    client->enumerateDevices();
    return client;
}

PjrtClient::~PjrtClient() {
    if (client_ && plugin_ && plugin_->api()) {
        PJRT_Client_Destroy_Args args = {};
        args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        args.extension_start = nullptr;
        args.client = client_;
        // Nothing useful can be done with a failure in a destructor, but the
        // error must still be consumed rather than leaked.
        PJRT_Error* err = plugin_->api()->PJRT_Client_Destroy(&args);
        if (err) (void)consumeError(err);
    }
    client_ = nullptr;
}

void PjrtClient::enumerateDevices() {
    devices_.clear();
    device_handles_.clear();
    if (!client_) return;

    const PJRT_Api* api = plugin_->api();

    PJRT_Client_Devices_Args args = {};
    args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.client = client_;

    PJRT_Error* err = api->PJRT_Client_Devices(&args);
    if (err) {
        (void)consumeError(err);
        return;
    }

    device_handles_.reserve(args.num_devices);
    devices_.reserve(args.num_devices);
    for (size_t i = 0; i < args.num_devices; ++i) {
        PJRT_Device* dev = args.devices[i];
        device_handles_.push_back(dev);

        PjrtDeviceInfo info;

        PJRT_Device_GetDescription_Args desc_args = {};
        desc_args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
        desc_args.extension_start = nullptr;
        desc_args.device = dev;
        PJRT_Error* derr = api->PJRT_Device_GetDescription(&desc_args);
        if (!derr && desc_args.device_description) {
            PJRT_DeviceDescription_Id_Args id_args = {};
            id_args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
            id_args.extension_start = nullptr;
            id_args.device_description = desc_args.device_description;
            PJRT_Error* ierr = api->PJRT_DeviceDescription_Id(&id_args);
            if (!ierr) info.id = id_args.id;
            else (void)consumeError(ierr);

            PJRT_DeviceDescription_Kind_Args kind_args = {};
            kind_args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
            kind_args.extension_start = nullptr;
            kind_args.device_description = desc_args.device_description;
            PJRT_Error* kerr = api->PJRT_DeviceDescription_Kind(&kind_args);
            if (!kerr) {
                info.kind = pjrtString(kind_args.device_kind,
                                       kind_args.device_kind_size);
            } else {
                (void)consumeError(kerr);
            }
        } else if (derr) {
            (void)consumeError(derr);
        }

        PJRT_Device_IsAddressable_Args addr_args = {};
        addr_args.struct_size = PJRT_Device_IsAddressable_Args_STRUCT_SIZE;
        addr_args.extension_start = nullptr;
        addr_args.device = dev;
        PJRT_Error* aerr = api->PJRT_Device_IsAddressable(&addr_args);
        if (!aerr) info.is_addressable = addr_args.is_addressable;
        else (void)consumeError(aerr);

        devices_.push_back(std::move(info));
    }
}

std::string PjrtClient::platformName() const {
    if (!client_) return std::string();

    PJRT_Client_PlatformName_Args args = {};
    args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.client = client_;

    PJRT_Error* err = plugin_->api()->PJRT_Client_PlatformName(&args);
    if (err) {
        (void)consumeError(err);
        return std::string();
    }
    return pjrtString(args.platform_name, args.platform_name_size);
}

std::vector<PjrtDeviceInfo> PjrtClient::addressableDevices() const {
    std::vector<PjrtDeviceInfo> out;
    for (const auto& d : devices_) {
        if (d.is_addressable) out.push_back(d);
    }
    return out;
}

PJRT_LoadedExecutable* PjrtClient::compile(const std::string& mlir_module,
                                           const std::string& format,
                                           std::string* error) {
    if (!client_) {
        if (error) *error = "compile called on an uninitialised client";
        return nullptr;
    }

    // PJRT_Program's `code` is char*, not const char*, because the same struct
    // is used for output elsewhere in the API. We are only passing it in, so a
    // const_cast is correct here rather than a copy.
    PJRT_Program program = {};
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.extension_start = nullptr;
    program.code = const_cast<char*>(mlir_module.data());
    program.code_size = mlir_module.size();
    program.format = format.data();
    program.format_size = format.size();

    PJRT_Client_Compile_Args args = {};
    args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.client = client_;
    args.program = &program;
    // No compile options: the defaults compile for all addressable devices,
    // which is what a single-host slice wants. Sharding arrives with S7 and
    // will set compile options here rather than anywhere else.
    args.compile_options = nullptr;
    args.compile_options_size = 0;

    PJRT_Error* err = plugin_->api()->PJRT_Client_Compile(&args);
    if (err) {
        PjrtStatus status = consumeError(err);
        if (error) *error = "PJRT_Client_Compile: " + status.message();
        return nullptr;
    }
    return args.executable;
}

void PjrtClient::destroyExecutable(PJRT_LoadedExecutable* executable) {
    if (!executable || !plugin_ || !plugin_->api()) return;
    PJRT_LoadedExecutable_Destroy_Args args = {};
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.executable = executable;
    PJRT_Error* err = plugin_->api()->PJRT_LoadedExecutable_Destroy(&args);
    if (err) (void)consumeError(err);
}

void PjrtClient::destroyBuffer(PJRT_Buffer* buffer) {
    if (!buffer || !plugin_ || !plugin_->api()) return;
    PJRT_Buffer_Destroy_Args args = {};
    args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = buffer;
    PJRT_Error* err = plugin_->api()->PJRT_Buffer_Destroy(&args);
    if (err) (void)consumeError(err);
}

// ─────────────────────────────────────────────────────────────────────────
// Plugin discovery
// ─────────────────────────────────────────────────────────────────────────

namespace {

// The header mirrors PJRT_Buffer_Type's values so that consumers of the XLA
// backend need not see the ABI header. That duplication is only safe if it is
// checked, so check it here, where both definitions are visible. A plugin
// reinterprets bytes according to this number; a wrong one is silent.
static_assert(static_cast<int>(PjrtElementType::kPred) == PJRT_Buffer_Type_PRED, "PJRT_Buffer_Type drift: PRED");
static_assert(static_cast<int>(PjrtElementType::kS8)   == PJRT_Buffer_Type_S8,   "PJRT_Buffer_Type drift: S8");
static_assert(static_cast<int>(PjrtElementType::kS32)  == PJRT_Buffer_Type_S32,  "PJRT_Buffer_Type drift: S32");
static_assert(static_cast<int>(PjrtElementType::kS64)  == PJRT_Buffer_Type_S64,  "PJRT_Buffer_Type drift: S64");
static_assert(static_cast<int>(PjrtElementType::kU8)   == PJRT_Buffer_Type_U8,   "PJRT_Buffer_Type drift: U8");
static_assert(static_cast<int>(PjrtElementType::kU64)  == PJRT_Buffer_Type_U64,  "PJRT_Buffer_Type drift: U64");
static_assert(static_cast<int>(PjrtElementType::kF16)  == PJRT_Buffer_Type_F16,  "PJRT_Buffer_Type drift: F16");
static_assert(static_cast<int>(PjrtElementType::kF32)  == PJRT_Buffer_Type_F32,  "PJRT_Buffer_Type drift: F32");
static_assert(static_cast<int>(PjrtElementType::kF64)  == PJRT_Buffer_Type_F64,  "PJRT_Buffer_Type drift: F64");
static_assert(static_cast<int>(PjrtElementType::kBf16) == PJRT_Buffer_Type_BF16, "PJRT_Buffer_Type drift: BF16");
static_assert(static_cast<int>(PjrtElementType::kC64)  == PJRT_Buffer_Type_C64,  "PJRT_Buffer_Type drift: C64");
static_assert(static_cast<int>(PjrtElementType::kC128) == PJRT_Buffer_Type_C128, "PJRT_Buffer_Type drift: C128");

/**
 * @brief Destroy a PJRT_Error without reading it.
 *
 * Used only where an error is genuinely redundant, i.e. a second failure while
 * already reporting a first. Dropping it instead would leak.
 */
void discardError(const PJRT_Api* api, PJRT_Error* error) {
    if (error == nullptr) return;
    PJRT_Error_Destroy_Args args = {};
    args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.error = error;
    api->PJRT_Error_Destroy(&args);
}

/**
 * @brief Block on @p event, then destroy it. Returns any error for the caller
 *        to consume; the event is destroyed on every path.
 *
 * PJRT events are the ABI's only completion signal, and every one it hands out
 * is owned by us. Awaiting without destroying leaks a event per transfer,
 * which on a training loop is a leak per step.
 */
PJRT_Error* awaitAndDestroyEvent(const PJRT_Api* api, PJRT_Event* event) {
    if (event == nullptr) return nullptr;

    PJRT_Event_Await_Args await_args = {};
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.extension_start = nullptr;
    await_args.event = event;
    PJRT_Error* await_error = api->PJRT_Event_Await(&await_args);

    PJRT_Event_Destroy_Args destroy_args = {};
    destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.event = event;
    PJRT_Error* destroy_error = api->PJRT_Event_Destroy(&destroy_args);

    if (await_error != nullptr) {
        // The await failure is the one worth reporting; the destroy failure
        // still has to be consumed or it leaks.
        discardError(api, destroy_error);
        return await_error;
    }
    return destroy_error;
}

}  // namespace

PJRT_Buffer* PjrtClient::bufferFromHost(const void* data,
                                        PjrtElementType element_type,
                                        const std::vector<int64_t>& dims,
                                        int device_index,
                                        std::string* error) {
    if (device_index < 0 ||
        static_cast<size_t>(device_index) >= device_handles_.size()) {
        if (error) {
            *error = "device index " + std::to_string(device_index) +
                     " out of range; client has " +
                     std::to_string(device_handles_.size()) + " device(s)";
        }
        return nullptr;
    }
    // Placing work on a non-addressable device is not a recoverable runtime
    // condition, it is a programming error, and PJRT's own diagnostic for it is
    // obscure. Reject it here where the cause is still visible.
    if (!devices_[static_cast<size_t>(device_index)].is_addressable) {
        if (error) {
            *error = "device index " + std::to_string(device_index) +
                     " (id " +
                     std::to_string(devices_[static_cast<size_t>(device_index)].id) +
                     ") is not addressable from this process";
        }
        return nullptr;
    }

    PJRT_Client_BufferFromHostBuffer_Args args = {};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.client = client_;
    args.data = data;
    args.type = static_cast<PJRT_Buffer_Type>(element_type);
    // An empty shape is a scalar, which is a real and common case, so dims may
    // legitimately be null here with num_dims 0.
    args.dims = dims.empty() ? nullptr : dims.data();
    args.num_dims = dims.size();
    // Null byte_strides requests the dense major-to-minor layout, which is what
    // every caller in this backend produces.
    args.byte_strides = nullptr;
    args.num_byte_strides = 0;
    // kImmutableUntilTransferCompletes lets the plugin avoid a staging copy,
    // at the price of requiring `data` to stay valid until the returned event
    // fires. We await that event below, so the caller sees a simple
    // "returns when done" contract and cannot get this wrong.
    args.host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    args.device = device_handles_[static_cast<size_t>(device_index)];
    args.memory = nullptr;
    args.device_layout = nullptr;

    PJRT_Error* err = plugin_->api()->PJRT_Client_BufferFromHostBuffer(&args);
    if (err != nullptr) {
        PjrtStatus status = consumeError(err);
        if (error) *error = status.message();
        return nullptr;
    }

    PJRT_Error* wait_err =
        awaitAndDestroyEvent(plugin_->api(), args.done_with_host_buffer);
    if (wait_err != nullptr) {
        PjrtStatus status = consumeError(wait_err);
        if (error) {
            *error = "host buffer transfer did not complete: " + status.message();
        }
        // The device buffer was created, so it must not be leaked even though
        // we are failing.
        destroyBuffer(args.buffer);
        return nullptr;
    }

    return args.buffer;
}

PjrtStatus PjrtClient::bufferToHost(PJRT_Buffer* buffer,
                                    void* dst,
                                    size_t dst_bytes) {
    if (buffer == nullptr) {
        return PjrtStatus("bufferToHost: null device buffer");
    }
    if (dst == nullptr) {
        // A null dst is how the ABI is asked for a size, which is a different
        // operation than the one this method offers. Refuse rather than
        // silently perform it and return success having copied nothing.
        return PjrtStatus("bufferToHost: null destination");
    }

    PJRT_Buffer_ToHostBuffer_Args args = {};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.src = buffer;
    // Null host_layout takes the source buffer's own layout, which is what a
    // straight read-back wants.
    args.host_layout = nullptr;
    args.dst = dst;
    args.dst_size = dst_bytes;

    PJRT_Error* err = plugin_->api()->PJRT_Buffer_ToHostBuffer(&args);
    if (err != nullptr) {
        return consumeError(err);
    }

    // The copy is asynchronous: without this await, the caller reads dst before
    // the device has written it, and gets stale memory that looks like a
    // numerical bug rather than a synchronisation one.
    PJRT_Error* wait_err = awaitAndDestroyEvent(plugin_->api(), args.event);
    if (wait_err != nullptr) {
        PjrtStatus status = consumeError(wait_err);
        return PjrtStatus("device to host copy did not complete: " +
                          status.message());
    }
    return PjrtStatus();
}

PjrtStatus PjrtClient::execute(PJRT_LoadedExecutable* executable,
                               const std::vector<PJRT_Buffer*>& inputs,
                               std::vector<PJRT_Buffer*>& outputs) {
    if (executable == nullptr) {
        return PjrtStatus("execute: null executable");
    }

    // How many outputs to allocate room for is a property of the compiled
    // program, and the only way to learn it is through the unloaded executable
    // handle. That handle is a separate owned object from the loaded one.
    PJRT_LoadedExecutable_GetExecutable_Args get_args = {};
    get_args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
    get_args.extension_start = nullptr;
    get_args.loaded_executable = executable;
    if (PJRT_Error* err =
            plugin_->api()->PJRT_LoadedExecutable_GetExecutable(&get_args)) {
        return consumeError(err);
    }

    PJRT_Executable_NumOutputs_Args count_args = {};
    count_args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
    count_args.extension_start = nullptr;
    count_args.executable = get_args.executable;
    PJRT_Error* count_err =
        plugin_->api()->PJRT_Executable_NumOutputs(&count_args);

    PJRT_Executable_Destroy_Args exec_destroy = {};
    exec_destroy.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
    exec_destroy.extension_start = nullptr;
    exec_destroy.executable = get_args.executable;
    discardError(plugin_->api(),
                 plugin_->api()->PJRT_Executable_Destroy(&exec_destroy));

    if (count_err != nullptr) {
        return consumeError(count_err);
    }
    const size_t num_outputs = count_args.num_outputs;

    // Single-device execution. A multi-device launch is a different shape of
    // call (num_devices > 1, one argument list per device) and belongs to the
    // sharding work, not here; pretending to support it by passing device 0
    // would produce wrong results rather than an error.
    std::vector<PJRT_Buffer*> input_row(inputs.begin(), inputs.end());
    PJRT_Buffer* const* argument_row = input_row.data();
    PJRT_Buffer* const* const* argument_lists = &argument_row;

    std::vector<PJRT_Buffer*> output_row(num_outputs, nullptr);
    PJRT_Buffer** output_row_ptr = output_row.data();
    PJRT_Buffer** const* output_lists = &output_row_ptr;

    PJRT_Event* completion_event = nullptr;

    PJRT_ExecuteOptions options = {};
    options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    options.extension_start = nullptr;

    PJRT_LoadedExecutable_Execute_Args args = {};
    args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.executable = executable;
    args.options = &options;
    args.argument_lists = argument_lists;
    args.num_devices = 1;
    args.num_args = input_row.size();
    args.output_lists = output_lists;
    args.device_complete_events = &completion_event;
    // Null execute_device means "the device(s) chosen at compile time", which
    // is correct for a program compiled for one device.
    args.execute_device = nullptr;

    if (PJRT_Error* err =
            plugin_->api()->PJRT_LoadedExecutable_Execute(&args)) {
        // On error the ABI guarantees device_complete_events is not populated,
        // so there is no event to destroy here.
        return consumeError(err);
    }

    PJRT_Error* wait_err =
        awaitAndDestroyEvent(plugin_->api(), completion_event);
    if (wait_err != nullptr) {
        PjrtStatus status = consumeError(wait_err);
        // Execution was launched, so output buffers may exist and would leak.
        for (PJRT_Buffer* b : output_row) destroyBuffer(b);
        return PjrtStatus("execution did not complete: " + status.message());
    }

    // Documented as appending, so an existing outputs vector is preserved.
    outputs.insert(outputs.end(), output_row.begin(), output_row.end());
    return PjrtStatus();
}

std::string findPjrtPlugin(const std::string& backend) {
    // 1. An explicit override always wins. This is the only mechanism that
    //    works for a plugin in a location nobody anticipated, which on a
    //    research machine is most of them.
    if (const char* explicit_path = std::getenv("ESHKOL_PJRT_PLUGIN_PATH")) {
        if (explicit_path[0] != '\0') return std::string(explicit_path);
    }

    // 2. The vendor wheels. This is how a TPU VM is actually provisioned:
    //    `pip install libtpu` drops libtpu.so into site-packages, and nothing
    //    puts it on a system library path. Checking here rather than making
    //    the operator set a variable is the difference between the backend
    //    working out of the box on a TPU VM and not.
    const char* home = std::getenv("HOME");
    if (home) {
        const std::string candidates[] = {
            std::string(home) + "/.local/lib/python3.10/site-packages/libtpu/libtpu.so",
            std::string(home) + "/.local/lib/python3.11/site-packages/libtpu/libtpu.so",
            std::string(home) + "/.local/lib/python3.12/site-packages/libtpu/libtpu.so",
        };
        for (const auto& c : candidates) {
            if (backend == "tpu" || backend.empty()) {
                if (::access(c.c_str(), R_OK) == 0) return c;
            }
        }
    }

    // 3. System paths, checked last because a system-wide plugin is the least
    //    likely to be the one the operator meant on a machine with several.
    if (backend == "tpu" || backend.empty()) {
        const char* system_paths[] = {
            "/usr/lib/libtpu.so",
            "/usr/local/lib/libtpu.so",
        };
        for (const char* p : system_paths) {
            if (::access(p, R_OK) == 0) return std::string(p);
        }
    }

    return std::string();
}

}  // namespace xla
}  // namespace eshkol
