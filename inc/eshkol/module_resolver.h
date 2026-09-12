#ifndef ESHKOL_MODULE_RESOLVER_H
#define ESHKOL_MODULE_RESOLVER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif
