/**
 * @file vm_wasm_repl.c
 * @brief WASM entry point for the Eshkol browser REPL.
 *
 * Uses ReplSession for persistent state — definitions carry across evals.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#define ESHKOL_VM_LIBRARY_MODE 1
#define ESHKOL_VM_NO_DISASM 1

#include "eshkol_vm.c"

#include <emscripten/emscripten.h>

static ReplSession* g_session = NULL;

/** @brief WASM-exported: lazily create the persistent global REPL session. */
EMSCRIPTEN_KEEPALIVE
void repl_init(void) {
    if (!g_session) {
        g_session = repl_session_create();
    }
}

/** @brief WASM-exported: destroy and recreate the global REPL session,
 *         discarding all prior definitions. */
EMSCRIPTEN_KEEPALIVE
void repl_reset(void) {
    if (g_session) {
        repl_session_destroy(g_session);
        g_session = NULL;
    }
    repl_init();
}

/** @brief WASM-exported: evaluate @p source in the persistent global REPL
 *         session (initializing it on first use), printing output via the
 *         session's own output plumbing.
 * @return Always an empty string on success, or an error message if VM
 *         initialization failed.
 */
EMSCRIPTEN_KEEPALIVE
const char* repl_eval(const char* source) {
    if (!g_session) repl_init();
    if (!g_session) return "ERROR: VM init failed";
    repl_session_eval(g_session, source, 1);
    return "";
}
