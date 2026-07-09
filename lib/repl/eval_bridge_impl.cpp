/*
 * eval bridge — strong REPL-side implementation + auto-registration.
 *
 * See lib/core/eval_bridge.cpp for the rationale. This TU lives in
 * eshkol-repl-lib; force-loading that archive causes the static
 * constructor below to run before main() and install the bridge
 * function pointers. Binaries that don't link eshkol-repl-lib skip
 * this TU entirely and keep the nullptr defaults (eval fails at
 * runtime with a clear message).
 */

#include "repl_jit.h"
#include <eshkol/core/eval_bridge.h>
#include <eshkol/logger.h>

#include <exception>
#include <memory>
#include <mutex>

namespace {

std::mutex g_jit_mutex;
std::unique_ptr<eshkol::ReplJITContext> g_jit_context;

/**
 * @brief Lazily creates and returns the process-wide REPL JIT context.
 *
 * Thread-safe via @c g_jit_mutex: the first caller constructs the singleton
 * eshkol::ReplJITContext and logs success; construction failures are caught,
 * logged, and reported as a null return rather than propagated as an
 * exception. Subsequent calls return the already-constructed instance.
 *
 * @return Pointer to the singleton JIT context, or nullptr if construction
 *         failed.
 */
eshkol::ReplJITContext* acquire_repl_jit(void) {
    std::lock_guard<std::mutex> lock(g_jit_mutex);
    if (!g_jit_context) {
        try {
            g_jit_context = std::make_unique<eshkol::ReplJITContext>();
            eshkol_info("Eval JIT context initialized (via eval_bridge)");
        } catch (const std::exception& e) {
            eshkol_error("Failed to initialize eval JIT: %s", e.what());
            return nullptr;
        }
    }
    return g_jit_context.get();
}

/**
 * @brief Bridge-function-pointer adapter that acquires the REPL JIT context.
 *
 * Matches the `void* (*)(void)` signature expected by
 * eshkol_eval_jit_register(); wraps acquire_repl_jit() so it can be
 * registered as the eval bridge's "acquire" callback.
 *
 * @return Opaque pointer to the eshkol::ReplJITContext singleton (nullptr on
 *         failure).
 */
void* bridge_acquire(void) {
    return static_cast<void*>(acquire_repl_jit());
}

/**
 * @brief Bridge-function-pointer adapter that executes an AST via the REPL JIT.
 *
 * Matches the eval bridge's "execute" callback signature. If @p jit is null
 * (acquisition failed earlier), returns a zeroed ESHKOL_VALUE_NULL tagged
 * value instead of dereferencing it; otherwise forwards to
 * eshkol::ReplJITContext::executeTagged().
 *
 * @param jit Opaque pointer previously returned by bridge_acquire(), or null.
 * @param ast The AST node to compile and execute.
 * @return The tagged result value, or a null tagged value if @p jit is null.
 */
eshkol_tagged_value_t bridge_execute(void* jit, eshkol_ast_t* ast) {
    if (!jit) {
        eshkol_tagged_value_t out;
        out.type = ESHKOL_VALUE_NULL;
        out.flags = 0;
        out.reserved = 0;
        out.data.int_val = 0;
        return out;
    }
    return static_cast<eshkol::ReplJITContext*>(jit)->executeTagged(ast);
}

/**
 * @brief Bridge-function-pointer adapter that looks up a symbol's JIT address.
 *
 * Matches the eval bridge's "lookup" callback signature. Returns 0 if either
 * @p jit or @p name is null, otherwise forwards to
 * eshkol::ReplJITContext::lookupSymbol().
 *
 * @param jit Opaque pointer previously returned by bridge_acquire(), or null.
 * @param name Symbol name to resolve.
 * @return JIT address of the symbol, or 0 if not found or on null input.
 */
uint64_t bridge_lookup(void* jit, const char* name) {
    if (!jit || !name) return 0;
    return static_cast<eshkol::ReplJITContext*>(jit)->lookupSymbol(name);
}

/* Static constructor: runs once before main() and populates the
 * bridge pointers in eshkol-static with our concrete implementations.
 * Wrapped in a struct-with-constructor so the registration is
 * guaranteed to happen exactly once and survives -force_load linking. */
struct BridgeRegistrar {
    /**
     * @brief Registers the bridge_acquire/bridge_execute/bridge_lookup
     * callbacks as the process-wide eval bridge implementation.
     *
     * Runs as a static-initialization side effect (see the enclosing
     * comment) so the bridge pointers are installed before main() when this
     * translation unit is linked in.
     */
    BridgeRegistrar() {
        eshkol_eval_jit_register(bridge_acquire,
                                 bridge_execute,
                                 bridge_lookup);
    }
};

/* Have to name it — ld64 on macOS elides unnamed statics even with
 * force_load unless there's an external symbol tethering them. */
BridgeRegistrar g_eshkol_eval_bridge_registrar;

} /* anonymous namespace */

/* External anchor — gives the TU one non-static symbol so ld64 always
 * keeps it regardless of dead-stripping heuristics. force_load alone
 * isn't enough on some ld64 versions if everything here is static. */
extern "C" int eshkol_eval_bridge_linked = 1;
