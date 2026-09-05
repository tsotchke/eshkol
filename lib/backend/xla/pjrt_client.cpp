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
