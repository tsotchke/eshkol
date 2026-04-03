/**
 * @file vm_wasm_repl.c
 * @brief WASM entry point for the Eshkol browser REPL.
 *
 * Simple stateless wrapper — each eval recompiles from scratch.
 * Persistent REPL state requires Phase 4 (CompilerState encapsulation)
 * and Phase 12 (ReplState) from the VM refactor plan.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#define ESHKOL_VM_LIBRARY_MODE 1
#define ESHKOL_VM_NO_DISASM 1

#include "eshkol_vm.c"

#include <emscripten/emscripten.h>

static int initialized = 0;

EMSCRIPTEN_KEEPALIVE
void repl_init(void) {
    initialized = 1;
}

EMSCRIPTEN_KEEPALIVE
const char* repl_eval(const char* source) {
    if (!initialized) repl_init();
    compile_and_run(source);
    return "";
}
