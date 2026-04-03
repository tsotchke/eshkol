/**
 * @file vm_wasm_repl.c
 * @brief WASM entry point for the Eshkol browser REPL.
 *
 * Wraps eshkol_vm.c to provide a C API callable from JavaScript via Emscripten.
 * Uses Emscripten's stdout capture for collecting output.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#define ESHKOL_VM_LIBRARY_MODE 1  /* Suppress eshkol_vm.c's main() */
#define ESHKOL_VM_NO_DISASM 1     /* Suppress bytecode dump in WASM builds */

#include "eshkol_vm.c"

#include <emscripten/emscripten.h>

static int initialized = 0;

/* Initialize the VM (call once on page load) */
EMSCRIPTEN_KEEPALIVE
void repl_init(void) {
    initialized = 1;
}

/* Evaluate an Eshkol expression. Output goes through Emscripten's print handler. */
EMSCRIPTEN_KEEPALIVE
const char* repl_eval(const char* source) {
    if (!initialized) repl_init();
    compile_and_run(source);
    return "";
}
