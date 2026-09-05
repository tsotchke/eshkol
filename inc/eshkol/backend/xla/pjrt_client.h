/**
 * @file pjrt_client.h
 * @brief PJRT plugin loading and device execution for the XLA backend.
 *
 * WHY THIS EXISTS.
 *
 * Until this file, Eshkol's "XLA backend" never reached an XLA device runtime.
 * `XLARuntime::execute()` cast its executable to a plain C function pointer and
 * called it:
 *
 *     using ExecFn = void(*)(const void* const*, void* const*);
 *     auto fn = reinterpret_cast<ExecFn>(executable);
 *     fn(input_ptrs.data(), output_ptrs.data());
 *
 * That is a perfectly good CPU path — StableHLO is lowered through Linalg to
 * LLVM IR and JITted — but it is not XLA execution, and it is why there was no
 * TPU story at all. `ROADMAP.md` marks the XLA backend Complete under
 * v1.1-accelerate; every individual claim there is defensible (the type system,
 * the fusion, CPU/GPU-from-one-source) but none of them is a device runtime,
 * and "Complete" reads as one.
 *
 * PJRT is the device runtime interface. A PJRT plugin is a shared object
 * exporting exactly ONE symbol:
 *
 *     const PJRT_Api* GetPjrtApi();
 *
 * which returns a struct of function pointers — 136 of them at PJRT API
 * version 0.114 — covering client creation, device enumeration, buffer
 * transfer, compilation and execution. Verified present on live hardware:
 * `nm -D libtpu.so` on a v5litepod-8 shows `T GetPjrtApi@@VERS_1.0`.
 *
 * The same symbol is exported by the CPU and GPU plugins, so this one loader
 * covers every backend. Choosing a device is choosing which .so to dlopen.
 *
 * DESIGN NOTES
 *
 * The PJRT C API is a C ABI, which shapes everything here:
 *
 *  - Every call takes an Args struct whose first field is `struct_size`. That
 *    is the ABI's forward-compatibility mechanism: a newer plugin reading an
 *    older caller's struct knows where the caller's knowledge stopped. Every
 *    Args struct MUST therefore be zero-initialised and have struct_size set,
 *    or the plugin reads garbage past the end of what we wrote.
 *
 *  - Errors come back as a `PJRT_Error*` which is non-null on failure and must
 *    be explicitly destroyed, or it leaks. There is no exception path across a
 *    C ABI. `PjrtStatus` below exists so no call site can forget.
 *
 *  - Handles are opaque pointers owned by the plugin. We never free them with
 *    delete; each has its own Destroy entry point.
 *
 * This wrapper deliberately owns as little as possible. It does not abstract
 * over PJRT or hide it behind a "portable" layer, because the last thing this
 * backend needs is a second indirection standing between it and the hardware.
 */

#ifndef ESHKOL_BACKEND_XLA_PJRT_CLIENT_H
#define ESHKOL_BACKEND_XLA_PJRT_CLIENT_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// The PJRT C API header is vendored rather than included from an XLA checkout:
// it is a stable C ABI and pulling all of XLA in to get one header would make
// this backend depend on a build we do not otherwise need.
struct PJRT_Api;
struct PJRT_Client;
struct PJRT_Device;
struct PJRT_Buffer;
struct PJRT_LoadedExecutable;
struct PJRT_Error;

namespace eshkol {
namespace xla {

/**
 * @brief Result of a PJRT call: success, or a message recovered from the
 *        plugin's PJRT_Error before destroying it.
 *
 * The C ABI hands back an owned PJRT_Error* that must be destroyed by calling
 * back into the plugin. Returning this type by value from every wrapped call
 * makes it impossible to check an error without also having consumed it.
 */
class PjrtStatus {
public:
    PjrtStatus() = default;
    explicit PjrtStatus(std::string message) : message_(std::move(message)) {}

    bool ok() const { return message_.empty(); }
    const std::string& message() const { return message_; }

private:
    std::string message_;
};

/**
 * @brief A loaded PJRT plugin: the dlopen handle plus its PJRT_Api table.
 *
 * One plugin per backend. `libtpu.so` for TPU, the CPU plugin for CPU, and so
 * on. The plugin is kept loaded for the lifetime of this object because every
 * handle it hands out is invalidated when it unloads.
 */
class PjrtPlugin {
public:
    ~PjrtPlugin();

    PjrtPlugin(const PjrtPlugin&) = delete;
    PjrtPlugin& operator=(const PjrtPlugin&) = delete;

    /**
     * @brief dlopen @p library_path and resolve `GetPjrtApi`.
     *
     * Returns nullptr and sets @p error when the library cannot be opened, when
     * it does not export GetPjrtApi (i.e. it is not a PJRT plugin), or when the
     * returned API's major version does not match what this code was compiled
     * against. The version check is deliberately fatal rather than a warning:
     * a mismatched major version means the Args struct layouts differ, and
     * proceeding would corrupt memory in a way that presents as an unrelated
     * crash much later.
     */
    static std::unique_ptr<PjrtPlugin> load(const std::string& library_path,
                                            std::string* error);

    /** @brief The plugin's function table. Never null for a loaded plugin. */
    const PJRT_Api* api() const { return api_; }

    /** @brief Reported API version, useful in diagnostics. */
    int apiMajorVersion() const { return api_major_; }
    int apiMinorVersion() const { return api_minor_; }

    /** @brief Path this plugin was loaded from, for error messages. */
    const std::string& path() const { return path_; }

private:
    PjrtPlugin() = default;

    void* handle_ = nullptr;     ///< dlopen handle
    const PJRT_Api* api_ = nullptr;
    int api_major_ = 0;
    int api_minor_ = 0;
    std::string path_;
};

/** @brief One device the plugin exposes. */
struct PjrtDeviceInfo {
    int id = -1;
    int process_index = 0;
    bool is_addressable = false;
    std::string kind;            ///< e.g. "TPU v5 lite", "cpu"
};

/**
 * @brief Element types, mirroring the values of the ABI's PJRT_Buffer_Type.
 *
 * These values are duplicated rather than included because this header
 * deliberately does not pull in the PJRT C ABI header, so that consumers of
 * the XLA backend need not see it. Duplication of an ABI constant is normally
 * a latent correctness bug, so pjrt_client.cpp static_asserts every enumerator
 * here against the real enum. Drift is therefore a compile error rather than a
 * silent misreading of every buffer on the device.
 *
 * This is a typed enum rather than a bare int because the value is otherwise a
 * hand-counted ordinal, and getting it wrong does not fail loudly: the plugin
 * accepts the transfer and reinterprets the bytes.
 */
enum class PjrtElementType : int {
    kInvalid = 0,
    kPred    = 1,
    kS8      = 2,
    kS16     = 3,
    kS32     = 4,
    kS64     = 5,
    kU8      = 6,
    kU16     = 7,
    kU32     = 8,
    kU64     = 9,
    kF16     = 10,
    kF32     = 11,
    kF64     = 12,
    kBf16    = 13,
    kC64     = 14,
    kC128    = 15,
};

/**
 * @brief A PJRT client: the connection to a set of devices, and the thing that
 *        compiles and runs programs on them.
 */
class PjrtClient {
public:
    ~PjrtClient();

    PjrtClient(const PjrtClient&) = delete;
    PjrtClient& operator=(const PjrtClient&) = delete;

    /**
     * @brief Create a client over @p plugin.
     *
     * The plugin must outlive the client: every handle below belongs to it.
     */
    static std::unique_ptr<PjrtClient> create(PjrtPlugin* plugin,
                                              std::string* error);

    /** @brief Platform name as the plugin reports it ("tpu", "cpu", "cuda"). */
    std::string platformName() const;

    /** @brief All devices, addressable or not. */
    const std::vector<PjrtDeviceInfo>& devices() const { return devices_; }

    /** @brief Devices this process can actually place work on. */
    std::vector<PjrtDeviceInfo> addressableDevices() const;

    /**
     * @brief Compile a StableHLO module to a loaded executable.
     *
     * @param mlir_module  StableHLO in its textual or bytecode form, whichever
     *                     @p format names.
     * @param format       PJRT program format, e.g. "mlir" for StableHLO.
     *
     * Returns nullptr on failure with @p error set. The returned executable is
     * owned by the caller and must be released with destroyExecutable().
     */
    PJRT_LoadedExecutable* compile(const std::string& mlir_module,
                                   const std::string& format,
                                   std::string* error);

    void destroyExecutable(PJRT_LoadedExecutable* executable);

    /**
     * @brief Copy a host buffer to a device, returning the device buffer.
     *
     * @param element_type Element type of both the host data and the device
     *                     buffer. No conversion is performed.
     * @param dims         Shape. An empty vector is a scalar, not an error.
     * @param device_index Index into devices(). Must name an addressable
     *                     device; a non-addressable one is rejected rather
     *                     than silently retargeted.
     *
     * Blocks until the plugin has finished reading @p data, so the caller may
     * free or reuse it as soon as this returns.
     */
    PJRT_Buffer* bufferFromHost(const void* data,
                                PjrtElementType element_type,
                                const std::vector<int64_t>& dims,
                                int device_index,
                                std::string* error);

    /** @brief Copy a device buffer back to host memory. Blocks until ready. */
    PjrtStatus bufferToHost(PJRT_Buffer* buffer, void* dst, size_t dst_bytes);

    void destroyBuffer(PJRT_Buffer* buffer);

    /**
     * @brief Execute @p executable over @p inputs, producing @p outputs.
     *
     * Outputs are appended; the caller owns them and must destroyBuffer() each.
     */
    PjrtStatus execute(PJRT_LoadedExecutable* executable,
                       const std::vector<PJRT_Buffer*>& inputs,
                       std::vector<PJRT_Buffer*>& outputs);

private:
    PjrtClient() = default;

    /** @brief Consume a PJRT_Error*, extracting its message and destroying it. */
    PjrtStatus consumeError(PJRT_Error* error) const;

    /** @brief Populate devices_ from the client. */
    void enumerateDevices();

    PjrtPlugin* plugin_ = nullptr;   ///< not owned
    PJRT_Client* client_ = nullptr;
    std::vector<PjrtDeviceInfo> devices_;
    std::vector<PJRT_Device*> device_handles_;
};

/**
 * @brief Locate a PJRT plugin for @p backend, or an empty string.
 *
 * Search order, most explicit first:
 *   1. ESHKOL_PJRT_PLUGIN_PATH — an exact .so path, wins unconditionally.
 *   2. The Python site-packages location the vendor wheels install to
 *      (libtpu ships libtpu.so there), because that is how a TPU VM is
 *      actually provisioned in practice.
 *   3. Well-known system paths.
 *
 * Returning a path does not mean the plugin loads; PjrtPlugin::load() is what
 * establishes that.
 */
std::string findPjrtPlugin(const std::string& backend);

}  // namespace xla
}  // namespace eshkol

#endif  // ESHKOL_BACKEND_XLA_PJRT_CLIENT_H
