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

/** @brief WASM-exported: run @p source as a whole program in BATCH mode,
 *         i.e. exactly as `eshkol-vm-standalone <file.esk>` / `eshkol-run -r`
 *         run it — compile the prelude + the program into a fresh chunk and
 *         execute, WITHOUT the REPL's trailing auto-print of the last
 *         expression's value.  Output is emitted only by explicit
 *         display/write/newline, matching the native `-r` batch surface.
 *
 *         This is the entry point the WASM execute-and-diff lane drives
 *         (scripts/run_wasm_differential.sh): the VM's C display code is what
 *         gets compiled to WASM here, so the captured stdout is a genuine
 *         product of WASM execution — not a JS re-implementation of Eshkol
 *         formatting.  Each program should run in a freshly instantiated
 *         module so global VM state does not leak between programs.
 */
EMSCRIPTEN_KEEPALIVE
void run_program(const char* source) {
    if (!source) return;
    g_source_file_path = "<wasm-diff>";
    compile_and_run(source);
}
