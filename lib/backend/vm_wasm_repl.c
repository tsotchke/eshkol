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

EMSCRIPTEN_KEEPALIVE
void repl_init(void) {
    if (!g_session) {
        g_session = repl_session_create();
    }
}

EMSCRIPTEN_KEEPALIVE
const char* repl_eval(const char* source) {
    if (!g_session) repl_init();
    if (!g_session) return "ERROR: VM init failed";
    repl_session_eval(g_session, source, 1);
    return "";
}
