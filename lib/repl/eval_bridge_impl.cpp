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

void* bridge_acquire(void) {
    return static_cast<void*>(acquire_repl_jit());
}

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

uint64_t bridge_lookup(void* jit, const char* name) {
    if (!jit || !name) return 0;
    return static_cast<eshkol::ReplJITContext*>(jit)->lookupSymbol(name);
}

/* Static constructor: runs once before main() and populates the
 * bridge pointers in eshkol-static with our concrete implementations.
 * Wrapped in a struct-with-constructor so the registration is
 * guaranteed to happen exactly once and survives -force_load linking. */
struct BridgeRegistrar {
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
