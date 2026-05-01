/*
 * eval bridge — public accessor API for ReplJITContext.
 *
 * See lib/core/eval_bridge.cpp for the rationale. JIT-dependent
 * functions in introspection.cpp route through these accessor
 * functions so eshkol-static doesn't carry an unresolved
 * ReplJITContext reference. eshkol-repl-lib provides strong override
 * implementations at link time.
 *
 * We use accessor functions rather than plain globals because weak
 * data symbols on macOS ld64 do not reliably get replaced by strong
 * definitions in the same link; weak function symbols do. Callers
 * should always go through eshkol_eval_jit_available() +
 * the getters — never cache the result across link boundaries.
 */
#ifndef ESHKOL_EVAL_BRIDGE_H
#define ESHKOL_EVAL_BRIDGE_H

#include <eshkol/eshkol.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque JIT-context handle. Never dereferenced outside the strong
 * (eshkol-repl-lib) implementation of this API. */
typedef void* (*eshkol_eval_jit_acquire_fn_t)(void);
typedef eshkol_tagged_value_t (*eshkol_eval_jit_execute_fn_t)(void* jit,
                                                              eshkol_ast_t* ast);
typedef uint64_t (*eshkol_eval_jit_lookup_fn_t)(void* jit, const char* name);

/* Return non-zero if the JIT runtime is linked into this binary. */
int eshkol_eval_jit_available(void);

/* Get the concrete function pointers. Default implementations (in
 * eshkol-static) return nullptr; the strong override (in
 * eshkol-repl-lib) returns real functions. */
eshkol_eval_jit_acquire_fn_t eshkol_eval_jit_get_acquire(void);
eshkol_eval_jit_execute_fn_t eshkol_eval_jit_get_execute(void);
eshkol_eval_jit_lookup_fn_t  eshkol_eval_jit_get_lookup(void);

/* Runtime registration — called by eshkol-repl-lib's constructor to
 * install real implementations. Pass nullptr to unregister. */
void eshkol_eval_jit_register(eshkol_eval_jit_acquire_fn_t acquire,
                              eshkol_eval_jit_execute_fn_t execute,
                              eshkol_eval_jit_lookup_fn_t  lookup);

/* Warn once per process that the JIT runtime isn't linked. */
void eshkol_eval_jit_warn_missing(const char* caller);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_EVAL_BRIDGE_H */
