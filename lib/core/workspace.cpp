/*
 * Global Workspace Implementation for Eshkol Consciousness Engine
 *
 * Implements: workspace construction, module registration, content access.
 * ws-step! is implemented in LLVM codegen (uses closure_call_callback_).
 * All allocations use arena_allocate_with_header (bignum pattern).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/workspace.h>
#include <eshkol/eshkol.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ===== Arena Forward Declarations ===== */

extern "C" void* arena_allocate_with_header(arena_t* arena, size_t data_size,
                                             uint8_t subtype, uint8_t flags);
extern "C" void* arena_allocate_aligned(arena_t* arena, size_t size, size_t alignment);

/* ===== Construction ===== */

eshkol_workspace_t* eshkol_make_workspace(arena_t* arena,
    uint32_t dim, uint32_t max_modules) {
    if (!arena || dim == 0 || max_modules == 0) return NULL;

    size_t data_size = sizeof(eshkol_workspace_t)
                     + max_modules * sizeof(eshkol_workspace_module_t);

    eshkol_workspace_t* ws = (eshkol_workspace_t*)arena_allocate_with_header(
        arena, data_size, HEAP_SUBTYPE_WORKSPACE, 0);
    if (!ws) return NULL;

    memset(ws, 0, data_size);
    ws->num_modules = 0;
    ws->max_modules = max_modules;
    ws->dim = dim;
    ws->step_count = 0;

    /* Allocate content buffer (initialized to zero) */
    ws->content = (double*)arena_allocate_aligned(arena, dim * sizeof(double), 8);
    if (!ws->content) return NULL;
    memset(ws->content, 0, dim * sizeof(double));

    return ws;
}

void eshkol_ws_register(arena_t* arena, eshkol_workspace_t* ws,
    const char* name, eshkol_tagged_value_t process_fn) {
    if (!arena || !ws || !name) return;
    if (ws->num_modules >= ws->max_modules) return;

    eshkol_workspace_module_t* modules = WS_MODULES(ws);
    eshkol_workspace_module_t* mod = &modules[ws->num_modules];

    /* Copy name to arena storage */
    size_t name_len = strlen(name);
    mod->name = (char*)arena_allocate_aligned(arena, name_len + 1, 1);
    if (mod->name) {
        memcpy(mod->name, name, name_len + 1);
    }

    mod->process_fn = process_fn;
    mod->salience = 0.0;

    ws->num_modules++;
}

/* ===== Content Access ===== */

const double* eshkol_ws_get_content(const eshkol_workspace_t* ws) {
    if (!ws) return NULL;
    return ws->content;
}

void eshkol_ws_set_content(eshkol_workspace_t* ws,
    const double* data, uint32_t dim) {
    if (!ws || !data) return;
    uint32_t copy_dim = (dim < ws->dim) ? dim : ws->dim;
    memcpy(ws->content, data, copy_dim * sizeof(double));
}

uint32_t eshkol_ws_get_dim(const eshkol_workspace_t* ws) {
    return ws ? ws->dim : 0;
}

uint32_t eshkol_ws_get_step_count(const eshkol_workspace_t* ws) {
    return ws ? ws->step_count : 0;
}

/* ===== Tagged Value Dispatch ===== */

void eshkol_make_workspace_tagged(arena_t* arena,
    const eshkol_tagged_value_t* dim_tv,
    const eshkol_tagged_value_t* max_modules_tv,
    eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    if (!arena || !dim_tv || !max_modules_tv) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    uint32_t dim = 0;
    uint32_t max_modules = 0;

    if (dim_tv->type == ESHKOL_VALUE_INT64) {
        dim = (uint32_t)dim_tv->data.int_val;
    } else {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    if (max_modules_tv->type == ESHKOL_VALUE_INT64) {
        max_modules = (uint32_t)max_modules_tv->data.int_val;
    } else {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    eshkol_workspace_t* ws = eshkol_make_workspace(arena, dim, max_modules);
    if (ws) {
        result->type = ESHKOL_VALUE_HEAP_PTR;
        result->data.ptr_val = (uint64_t)ws;
    } else {
        result->type = ESHKOL_VALUE_NULL;
    }
}

void eshkol_ws_register_tagged(arena_t* arena,
    const eshkol_tagged_value_t* ws_tv,
    const eshkol_tagged_value_t* name_tv,
    const eshkol_tagged_value_t* process_fn_tv) {
    if (!arena || !ws_tv || !name_tv || !process_fn_tv) return;

    /* Extract workspace pointer */
    if (ws_tv->type != ESHKOL_VALUE_HEAP_PTR || !ws_tv->data.ptr_val) return;
    eshkol_workspace_t* ws = (eshkol_workspace_t*)ws_tv->data.ptr_val;

    /* Extract name string */
    const char* name = NULL;
    if (name_tv->type == ESHKOL_VALUE_HEAP_PTR && name_tv->data.ptr_val) {
        /* Check if it's a string (HEAP_SUBTYPE_STRING) or symbol (HEAP_SUBTYPE_SYMBOL) */
        eshkol_object_header_t* header = ESHKOL_GET_HEADER((void*)name_tv->data.ptr_val);
        if (header->subtype == HEAP_SUBTYPE_STRING || header->subtype == HEAP_SUBTYPE_SYMBOL) {
            name = (const char*)name_tv->data.ptr_val;
        }
    }
    if (!name) name = "unnamed";

    eshkol_ws_register(arena, ws, name, *process_fn_tv);
}

/* ===== ws-step! Helpers ===== */

/*
 * Tensor layout (mirrors inference.cpp private struct).
 * Must match the struct that LLVM codegen emits for #(...) tensor literals.
 */
typedef struct ws_tensor_layout {
    uint64_t* dimensions;
    uint64_t  num_dimensions;
    int64_t*  elements;       /* double bit patterns as int64 */
    uint64_t  total_elements;
} ws_tensor_layout_t;

void eshkol_ws_make_content_tensor(arena_t* arena, const double* content,
    uint32_t dim, eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    if (!arena || !content || dim == 0) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Allocate tensor struct with object header */
    ws_tensor_layout_t* tensor = (ws_tensor_layout_t*)arena_allocate_with_header(
        arena, sizeof(ws_tensor_layout_t), HEAP_SUBTYPE_TENSOR, 0);
    if (!tensor) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    tensor->num_dimensions = 1;
    tensor->total_elements = dim;
    tensor->dimensions = (uint64_t*)arena_allocate_aligned(arena, sizeof(uint64_t), 8);
    if (tensor->dimensions) {
        tensor->dimensions[0] = dim;
    }
    tensor->elements = (int64_t*)arena_allocate_aligned(
        arena, dim * sizeof(int64_t), 8);
    if (!tensor->elements) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Convert doubles to int64 bit patterns */
    for (uint32_t i = 0; i < dim; i++) {
        union { double d; int64_t i; } u;
        u.d = content[i];
        tensor->elements[i] = u.i;
    }

    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->data.ptr_val = (uint64_t)tensor;
}

void eshkol_ws_step_finalize(eshkol_workspace_t* ws,
    const eshkol_tagged_value_t* results, uint32_t num_modules) {
    if (!ws || !results || num_modules == 0) return;

    /* Extract salience scores and proposal tensors from cons pairs */
    double salience[16];   /* max 16 modules */
    const ws_tensor_layout_t* proposals[16];
    uint32_t valid = 0;

    for (uint32_t i = 0; i < num_modules && i < 16; i++) {
        const eshkol_tagged_value_t* r = &results[i];

        /* Result should be a HEAP_PTR to a cons cell */
        if (r->type != ESHKOL_VALUE_HEAP_PTR || !r->data.ptr_val) {
            salience[i] = -1e30;  /* effectively zero after softmax */
            proposals[i] = NULL;
            continue;
        }

        /* Cons cell layout: {car: tagged_value_t, cdr: tagged_value_t} = 32 bytes */
        const eshkol_tagged_value_t* car =
            (const eshkol_tagged_value_t*)r->data.ptr_val;
        const eshkol_tagged_value_t* cdr = car + 1;  /* next 16 bytes */

        /* Car = salience (double) */
        if (car->type == ESHKOL_VALUE_DOUBLE) {
            salience[i] = car->data.double_val;
        } else if (car->type == ESHKOL_VALUE_INT64) {
            salience[i] = (double)car->data.int_val;
        } else {
            salience[i] = 0.0;
        }

        /* Cdr = proposal tensor (HEAP_PTR to tensor) */
        if (cdr->type == ESHKOL_VALUE_HEAP_PTR && cdr->data.ptr_val) {
            proposals[i] = (const ws_tensor_layout_t*)cdr->data.ptr_val;
        } else {
            proposals[i] = NULL;
        }

        valid++;
    }

    if (valid == 0) return;

    /* Softmax over salience scores */
    double max_sal = salience[0];
    for (uint32_t i = 1; i < num_modules && i < 16; i++) {
        if (salience[i] > max_sal) max_sal = salience[i];
    }

    double sum_exp = 0.0;
    double exp_sal[16];
    for (uint32_t i = 0; i < num_modules && i < 16; i++) {
        exp_sal[i] = exp(salience[i] - max_sal);
        sum_exp += exp_sal[i];
    }

    /* Normalize and find winner */
    uint32_t winner = 0;
    double best_score = 0.0;
    eshkol_workspace_module_t* modules = WS_MODULES(ws);
    for (uint32_t i = 0; i < num_modules && i < 16; i++) {
        double normalized = exp_sal[i] / sum_exp;
        modules[i].salience = normalized;
        if (normalized > best_score) {
            best_score = normalized;
            winner = i;
        }
    }

    /* Copy winner's proposal to workspace content */
    if (proposals[winner] && proposals[winner]->elements) {
        uint32_t copy_dim = (uint32_t)proposals[winner]->total_elements;
        if (copy_dim > ws->dim) copy_dim = ws->dim;
        for (uint32_t i = 0; i < copy_dim; i++) {
            union { int64_t ii; double d; } u;
            u.ii = proposals[winner]->elements[i];
            ws->content[i] = u.d;
        }
    }

    ws->step_count++;
}

/* ===== Display ===== */

void eshkol_display_workspace(const eshkol_workspace_t* ws, void* file) {
    FILE* f = file ? (FILE*)file : stdout;
    if (!ws) {
        fprintf(f, "#<workspace: empty>");
        return;
    }
    fprintf(f, "#<workspace: dim=%u, %u modules, step=%u>",
            ws->dim, ws->num_modules, ws->step_count);
}
