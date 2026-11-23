#ifndef ESHKOL_MODULE_RESOLVER_H
#define ESHKOL_MODULE_RESOLVER_H

#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESHKOL_MODULE_REFUSAL_PREFIX "Refused module reference "

/* Module references are portable across native and browser filesystems.
 * Check components before normalization, including on hosts where a
 * backslash or drive prefix has no special meaning to std::filesystem. */
static inline int eshkol_module_name_allowed(const char* name) {
    if (!name || !*name || *name == '/' || *name == '\\' ||
        (name[1] == ':' && ((*name >= 'A' && *name <= 'Z') ||
                            (*name >= 'a' && *name <= 'z')))) return 0;
    const char* part = name;
    while (*part) {
        size_t length = strcspn(part, "/\\");
        if (length == 2 && part[0] == '.' && part[1] == '.') return 0;
        part += length;
        if (*part) ++part;
    }
    return 1;
}

/* The installed relative path has static storage. NULL denotes a project
 * name or a refused spelling. */
const char* eshkol_stdlib_module_path(const char* name);
int eshkol_module_is_precompiled(const char* name);

/* Canonicalize a driver-supplied entry file for cycle detection. */
int eshkol_canonical_source_file_c(const char* path, char* output, size_t output_size);

/*
 * Resolve one module reference using the same path policy as the native
 * compiler, REPL, workspace tools, and bytecode VM. The result is copied into
 * the caller-owned buffer, so C consumers do not depend on C++ allocation or
 * string ownership rules.
 *
 * Returns 1 when a regular source file was found and 0 otherwise. A missing
 * or too-small output buffer is a failed resolution, never a truncated path.
 */
int eshkol_resolve_module_source_path_c(const char* module_name,
                                        const char* base_dir,
                                        const char* lib_dir,
                                        char* output,
                                        size_t output_size);

/*
 * Whether a source file lies inside the Eshkol module tree — the directory
 * carrying stdlib.esk — as opposed to being a program's own source.
 *
 * The capability policy asks this about an `(extern …)` declaration: a module
 * shipped with the language declares the runtime entry points it is built on
 * and carries its own gates, while a program's own foreign declaration is the
 * program reaching outside the language, which the `ffi` capability governs.
 *
 * Compares canonical paths. A NULL, empty or unplaceable path answers 0, so a
 * file the runtime cannot place is treated as the program's own.
 */
int eshkol_path_is_stdlib_module(const char* host_path);

#ifdef __cplusplus
}
#endif

#endif
