/*
 * Global Workspace for Eshkol Consciousness Engine
 *
 * Implements Global Workspace Theory (Baars, 1988; Bengio, 2017):
 * - Workspace buffer (shared tensor region)
 * - Module registration (closures as cognitive specialists)
 * - Attention-based competition (softmax over salience scores)
 * - Broadcast mechanism (winner's output -> workspace content)
 *
 * ws-step! is implemented in LLVM codegen (not C runtime) because
 * it must invoke Eshkol closures via closure_call_callback_.
 *
 * All objects are arena-allocated with object headers.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_WORKSPACE_H
#define ESHKOL_CORE_WORKSPACE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <eshkol/eshkol.h>

/* Forward declarations */
typedef struct arena arena_t;

/*
 * Workspace Module: a cognitive specialist that processes workspace content.
 *
 * process_fn is an Eshkol closure tagged value:
 *   (lambda (content) -> (cons salience proposal))
 *   Input: current workspace content tensor
 *   Output: cons pair of (salience_score . proposal_tensor)
 *
 * The closure is called from LLVM codegen using closure_call_callback_.
 */
typedef struct eshkol_workspace_module {
    char* name;                        /* Module name (arena-allocated string) */
    eshkol_tagged_value_t process_fn;  /* Closure tagged value */
    double salience;                   /* Last computed salience score */
} eshkol_workspace_module_t;

/*
 * Global Workspace: shared attention bottleneck.
 *
 * Cognitive cycle (ws-step!):
 * 1. Each module processes current workspace content
 * 2. Modules return (salience, proposal) pairs
 * 3. Softmax competition over salience scores
 * 4. Winner's proposal becomes new workspace content
 * 5. All modules see updated content next cycle
 *
 * Layout: [eshkol_object_header_t][eshkol_workspace_t][modules...]
 */
typedef struct eshkol_workspace {
    uint32_t num_modules;
    uint32_t max_modules;
    uint32_t dim;                      /* Workspace vector dimension */
    uint32_t step_count;               /* Cognitive cycle counter */
    double* content;                   /* Current workspace content (arena tensor data) */
    /* Followed by: eshkol_workspace_module_t modules[max_modules] */
} eshkol_workspace_t;

/* Access the modules array (immediately after the struct) */
#define WS_MODULES(ws) ((eshkol_workspace_module_t*)((uint8_t*)(ws) + sizeof(eshkol_workspace_t)))

#ifdef __cplusplus
extern "C" {
#endif

/* ===== Construction ===== */

/*
 * Create an empty workspace with given dimension and max module count.
 * Content is initialized to zeros.
 */
eshkol_workspace_t* eshkol_make_workspace(arena_t* arena,
    uint32_t dim, uint32_t max_modules);

/*
 * Register a cognitive module.
 * name is copied to arena storage.
 * process_fn must be a closure tagged value: (tensor -> (cons double tensor))
 */
void eshkol_ws_register(arena_t* arena, eshkol_workspace_t* ws,
    const char* name, eshkol_tagged_value_t process_fn);

/* ===== Content Access ===== */

/* Get current workspace content as a double array */
const double* eshkol_ws_get_content(const eshkol_workspace_t* ws);

/* Set workspace content (for initialization or broadcast) */
void eshkol_ws_set_content(eshkol_workspace_t* ws,
    const double* data, uint32_t dim);

/* Get workspace dimension */
uint32_t eshkol_ws_get_dim(const eshkol_workspace_t* ws);

/* Get step count */
uint32_t eshkol_ws_get_step_count(const eshkol_workspace_t* ws);

/* ===== Tagged Value Dispatch ===== */
/* Called from LLVM codegen. Same alloca/store/call/load pattern as bignum. */

void eshkol_make_workspace_tagged(arena_t* arena,
    const eshkol_tagged_value_t* dim,
    const eshkol_tagged_value_t* max_modules,
    eshkol_tagged_value_t* result);

void eshkol_ws_register_tagged(arena_t* arena,
    const eshkol_tagged_value_t* ws,
    const eshkol_tagged_value_t* name,
    const eshkol_tagged_value_t* process_fn);

/* ===== ws-step! Helpers ===== */
/* Called from LLVM codegen. Closure invocation happens in codegen;
   these helpers handle tensor wrapping and softmax broadcast. */

/*
 * Wrap workspace content doubles into a tensor tagged value.
 * Used to pass workspace content to module closures.
 */
void eshkol_ws_make_content_tensor(arena_t* arena, const double* content,
    uint32_t dim, eshkol_tagged_value_t* result);

/*
 * Process module closure results after ws-step! loop.
 * results[i] = cons pair (salience . proposal_tensor) from each module.
 * Performs softmax over salience scores, copies winner's proposal to ws->content.
 */
void eshkol_ws_step_finalize(eshkol_workspace_t* ws,
    const eshkol_tagged_value_t* results, uint32_t num_modules);

/* ===== Display ===== */

void eshkol_display_workspace(const eshkol_workspace_t* ws, void* file);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_WORKSPACE_H */
