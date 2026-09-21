#ifndef ESHKOL_BUILTIN_LIBRARIES_H
#define ESHKOL_BUILTIN_LIBRARIES_H

#include <stddef.h>
#include <string.h>

/*
 * The ONE table of R7RS library names Eshkol satisfies from a built-in module
 * instead of from a source file on the load path.
 *
 * `(import (scheme base) …)` names a library that has no `scheme/base.esk` to
 * find, so an engine that hands the joined name straight to the source-file
 * resolver reports `module source not found: scheme.base` and refuses the
 * program.  Both engines therefore map the library name through this table
 * FIRST: the native front end in join_r7rs_library_name() (lib/frontend/
 * parser.cpp) and the bytecode VM in vm_library_name_from_datum()
 * (lib/backend/vm_compiler.c).  Adding a built-in library means adding one row
 * here — there is deliberately no second list for either engine to drift from.
 *
 * Header-only and free of allocation so the VM's C unity build and the WASM
 * VM (which does not link the C++ platform runtime) can consult it too.
 */

typedef struct {
    const char* library; /* dotted R7RS library name, e.g. "scheme.base" */
    const char* module;  /* internal Eshkol module that provides it       */
} EshkolBuiltinLibrary;

static const EshkolBuiltinLibrary eshkol_builtin_libraries[] = {
    { "scheme.base", "stdlib" }
};

#define ESHKOL_N_BUILTIN_LIBRARIES \
    ((int)(sizeof(eshkol_builtin_libraries) / sizeof(eshkol_builtin_libraries[0])))

/**
 * @brief The built-in module that provides dotted library name @p library, or
 *        NULL when the name is not built in and must be resolved as a source
 *        file like any other module.
 */
static inline const char* eshkol_builtin_library_module(const char* library) {
    if (!library || !*library) return NULL;
    for (int i = 0; i < ESHKOL_N_BUILTIN_LIBRARIES; i++) {
        if (strcmp(eshkol_builtin_libraries[i].library, library) == 0) {
            return eshkol_builtin_libraries[i].module;
        }
    }
    return NULL;
}

#endif /* ESHKOL_BUILTIN_LIBRARIES_H */
