/*
 * SDNC Runtime API — expose the bytecode-VM-as-transformer weight-program (θ)
 * as pure-Eshkol builtins so programs can do geometric recursive
 * self-improvement on real weight-programs from .esk.
 *
 * Builtins (registered in parser.cpp / llvm_codegen.cpp / system_codegen.cpp /
 * type_checker.cpp / eshkol-run.cpp):
 *   (sdnc-program name)             -> θ        opaque handle (HEAP_SUBTYPE_SDNC)
 *   (sdnc-run θ input)              -> #(...)   D-dim forward output (real matvec)
 *   (sdnc-weight-grad θ input target) -> #(...) flat ∂L/∂WEIGHTS (backprop),
 *                                               length == (sdnc-params θ) length
 *   (sdnc-params θ)                 -> #(...)   flattened trainable weights
 *   (sdnc-set-params! θ vec)        -> θ        write flattened weights back
 *   (sdnc-improve! θ data steps lr) -> θ        SGD on weights, returns θ
 *   (sdnc? x)                       -> bool
 *
 * The trainable core math (forward / backward_through_weights /
 * apply_weight_gradient_step / param flatten) lives in lib/core/sdnc_core.h, a
 * byte-for-byte mirror of the SDNC paper artifact's weight kernels in
 * lib/backend/weight_matrices.c (which is a standalone executable with file-
 * static functions + its own main() — including it into the runtime would
 * collide with the already-linked vm_*.c units, so the kernels are mirrored).
 *
 * The forward / backward here are the SAME float arithmetic, in the SAME order,
 * as forward_with_weights / backward_through_weights in weight_matrices.c, so
 * sdnc-run and sdnc-weight-grad genuinely compute the SDNC weight-program's
 * forward output and exact weight gradient — no stubs.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <eshkol/eshkol.h>

#include "sdnc_core.h"

/* Arena handle (defined in arena_memory.h; only used by pointer here). */
typedef struct arena arena_t;

extern void* arena_allocate_with_header(arena_t* arena, size_t data_size,
                                        uint8_t subtype, uint8_t flags);
extern void* arena_allocate_aligned(arena_t* arena, size_t size, size_t alignment);
extern void  eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);

/* ===== Handle =====
 * Stored small in the arena; points to a calloc'd weight set so the large
 * SdncWeights never goes through the arena. Owns the position embeddings used by
 * forward/backward (set up deterministically per program). */
#define SDNC_NP 4
typedef struct {
    SdncWeights* w;
    int          np;
    float        pe[256][SDNC_D];
} SdncHandle;

/* Tensor layout mirroring #(...) literals. */
typedef struct sdnc_tensor_layout {
    uint64_t* dimensions;
    uint64_t  num_dimensions;
    int64_t*  elements;
    uint64_t  total_elements;
} sdnc_tensor_layout_t;

/* ===== Param flatten/unflatten (canonical order, mirror SdncWeights) ===== */
static size_t sdnc_param_count(void) {
    const int D = SDNC_D, N = SDNC_N_LAYERS, F = SDNC_FFN_DIM;
    return (size_t)N * ( (size_t)4*D*D + D + 2*(size_t)D*F + 2*F + (size_t)F*D + D );
}

typedef void (*sdnc_param_visit_cb)(size_t idx, float* slot, void* ud);
static void sdnc_visit_params(SdncWeights* w, sdnc_param_visit_cb cb, void* ud) {
    const int D = SDNC_D, N = SDNC_N_LAYERS, F = SDNC_FFN_DIM;
    size_t idx = 0;
    for (int L = 0; L < N; L++) {
        for (int i = 0; i < D*D; i++) cb(idx++, &w->wq[L][i], ud);
        for (int i = 0; i < D*D; i++) cb(idx++, &w->wk[L][i], ud);
        for (int i = 0; i < D*D; i++) cb(idx++, &w->wv[L][i], ud);
        for (int i = 0; i < D*D; i++) cb(idx++, &w->wo[L][i], ud);
        for (int i = 0; i < D; i++)   cb(idx++, &w->bq[L][i], ud);
        for (int i = 0; i < D*F; i++) cb(idx++, &w->ff_up[L][i], ud);
        for (int i = 0; i < F; i++)   cb(idx++, &w->ff_up_b[L][i], ud);
        for (int i = 0; i < F*D; i++) cb(idx++, &w->ff_down[L][i], ud);
        for (int i = 0; i < D; i++)   cb(idx++, &w->ff_down_b[L][i], ud);
        for (int i = 0; i < D*F; i++) cb(idx++, &w->ff_gate[L][i], ud);
        for (int i = 0; i < F; i++)   cb(idx++, &w->ff_gate_b[L][i], ud);
    }
}

/* Same canonical visit over a SdncGrads (the gradient struct mirrors the
 * trainable weight subset, field-for-field). */
static void sdnc_visit_grads(SdncGrads* g, sdnc_param_visit_cb cb, void* ud) {
    const int D = SDNC_D, N = SDNC_N_LAYERS, F = SDNC_FFN_DIM;
    size_t idx = 0;
    for (int L = 0; L < N; L++) {
        for (int i = 0; i < D*D; i++) cb(idx++, &g->dwq[L][i], ud);
        for (int i = 0; i < D*D; i++) cb(idx++, &g->dwk[L][i], ud);
        for (int i = 0; i < D*D; i++) cb(idx++, &g->dwv[L][i], ud);
        for (int i = 0; i < D*D; i++) cb(idx++, &g->dwo[L][i], ud);
        for (int i = 0; i < D; i++)   cb(idx++, &g->dbq[L][i], ud);
        for (int i = 0; i < D*F; i++) cb(idx++, &g->dff_up[L][i], ud);
        for (int i = 0; i < F; i++)   cb(idx++, &g->dff_up_b[L][i], ud);
        for (int i = 0; i < F*D; i++) cb(idx++, &g->dff_down[L][i], ud);
        for (int i = 0; i < D; i++)   cb(idx++, &g->dff_down_b[L][i], ud);
        for (int i = 0; i < D*F; i++) cb(idx++, &g->dff_gate[L][i], ud);
        for (int i = 0; i < F; i++)   cb(idx++, &g->dff_gate_b[L][i], ud);
    }
}

/* ===== Helpers ===== */
static SdncHandle* sdnc_extract_handle(const eshkol_tagged_value_t* tv) {
    if (!tv || tv->type != ESHKOL_VALUE_HEAP_PTR || !tv->data.ptr_val) return NULL;
    void* ptr = (void*)(uintptr_t)tv->data.ptr_val;
    eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(ptr);
    if (hdr->subtype != HEAP_SUBTYPE_SDNC) return NULL;
    return (SdncHandle*)ptr;
}

static int sdnc_get_int(const eshkol_tagged_value_t* tv, int dflt) {
    if (!tv) return dflt;
    uint8_t t = tv->type & 0x0F;
    if (t == ESHKOL_VALUE_INT64)  return (int)tv->data.int_val;
    if (t == ESHKOL_VALUE_DOUBLE) return (int)tv->data.double_val;
    return dflt;
}
static double sdnc_get_double(const eshkol_tagged_value_t* tv, double dflt) {
    if (!tv) return dflt;
    uint8_t t = tv->type & 0x0F;
    if (t == ESHKOL_VALUE_DOUBLE) return tv->data.double_val;
    if (t == ESHKOL_VALUE_INT64)  return (double)tv->data.int_val;
    return dflt;
}

/* Read a #(...) tensor (or heterogeneous vector) into a float buffer of length
 * `want` (pad with 0, truncate as needed). Returns number of source elements
 * read, or -1 on type error. */
static int sdnc_read_vector_into(const eshkol_tagged_value_t* tv, float* dst, int want) {
    for (int i = 0; i < want; i++) dst[i] = 0.0f;
    if (!tv || tv->type != ESHKOL_VALUE_HEAP_PTR || !tv->data.ptr_val) return -1;
    void* ptr = (void*)(uintptr_t)tv->data.ptr_val;
    eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(ptr);
    if (hdr->subtype == HEAP_SUBTYPE_TENSOR) {
        sdnc_tensor_layout_t* t = (sdnc_tensor_layout_t*)ptr;
        int n = (int)t->total_elements;
        if (n <= 0 || !t->elements) return -1;
        int m = n < want ? n : want;
        for (int i = 0; i < m; i++) {
            union { double d; int64_t i; } u;
            u.i = t->elements[i];
            dst[i] = (float)u.d;
        }
        return n;
    }
    if (hdr->subtype == HEAP_SUBTYPE_VECTOR) {
        int64_t n = *(int64_t*)ptr;
        if (n <= 0) return -1;
        eshkol_tagged_value_t* elems =
            (eshkol_tagged_value_t*)((uint8_t*)ptr + sizeof(int64_t));
        int m = (int)n < want ? (int)n : want;
        for (int i = 0; i < m; i++) {
            uint8_t et = elems[i].type & 0x0F;
            if (et == ESHKOL_VALUE_DOUBLE)     dst[i] = (float)elems[i].data.double_val;
            else if (et == ESHKOL_VALUE_INT64) dst[i] = (float)elems[i].data.int_val;
        }
        return (int)n;
    }
    return -1;
}

static void sdnc_make_tensor_f(arena_t* arena, const float* data, int len,
                               eshkol_tagged_value_t* result) {
    memset(result, 0, sizeof(*result));
    if (!arena || !data || len <= 0) { result->type = ESHKOL_VALUE_NULL; return; }
    sdnc_tensor_layout_t* t = (sdnc_tensor_layout_t*)arena_allocate_with_header(
        arena, sizeof(sdnc_tensor_layout_t), HEAP_SUBTYPE_TENSOR, 0);
    if (!t) { result->type = ESHKOL_VALUE_NULL; return; }
    t->num_dimensions = 1;
    t->total_elements = (uint64_t)len;
    t->dimensions = (uint64_t*)arena_allocate_aligned(arena, sizeof(uint64_t), 8);
    if (t->dimensions) t->dimensions[0] = (uint64_t)len;
    t->elements = (int64_t*)arena_allocate_aligned(arena, (size_t)len * sizeof(int64_t), 8);
    if (!t->elements) { result->type = ESHKOL_VALUE_NULL; return; }
    for (int i = 0; i < len; i++) {
        union { double d; int64_t i; } u;
        u.d = (double)data[i];
        t->elements[i] = u.i;
    }
    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->data.ptr_val = (uint64_t)(uintptr_t)t;
}

/* Deterministic, name-dependent position embeddings (the "program"). Mirrors
 * the random pe in self_improve_demo but seeded by a name hash so each named
 * program is distinct and reproducible. */
static void sdnc_setup_program(SdncHandle* h, unsigned long seed) {
    SdncRng r; r.state = 0x5EEDF00DUL ^ seed; r.scale = 0.1f;
    h->np = SDNC_NP;
    for (int p = 0; p < h->np; p++)
        for (int i = 0; i < SDNC_D; i++) h->pe[p][i] = sdnc_randf(&r);
}

static unsigned long sdnc_name_hash(const eshkol_tagged_value_t* name_tv) {
    /* FNV-1a over the string payload if a string is supplied; else 0. */
    unsigned long hsh = 1469598103934665603UL;
    if (name_tv && name_tv->type == ESHKOL_VALUE_HEAP_PTR && name_tv->data.ptr_val) {
        const char* s = (const char*)(uintptr_t)name_tv->data.ptr_val;
        eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)(uintptr_t)name_tv->data.ptr_val);
        if (hdr->subtype == HEAP_SUBTYPE_STRING) {
            for (const char* p = s; *p; p++) { hsh ^= (unsigned char)*p; hsh *= 1099511628211UL; }
        }
    }
    return hsh;
}

/* ===== (sdnc-program name) -> θ ===== */
void eshkol_sdnc_program_tagged(arena_t* arena,
                                const eshkol_tagged_value_t* name_tv,
                                eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    if (!arena) { result->type = ESHKOL_VALUE_NULL; return; }

    SdncHandle* h = (SdncHandle*)arena_allocate_with_header(
        arena, sizeof(SdncHandle), HEAP_SUBTYPE_SDNC, 0);
    if (!h) { result->type = ESHKOL_VALUE_NULL; return; }
    h->w = (SdncWeights*)calloc(1, sizeof(SdncWeights));
    if (!h->w) { result->type = ESHKOL_VALUE_NULL; return; }
    sdnc_init_weights(h->w);
    sdnc_setup_program(h, sdnc_name_hash(name_tv));

    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->data.ptr_val = (uint64_t)(uintptr_t)h;
}

/* ===== (sdnc-run θ input) -> #(...) D-dim output ===== */
void eshkol_sdnc_run_tagged(arena_t* arena,
                            const eshkol_tagged_value_t* theta,
                            const eshkol_tagged_value_t* input_tv,
                            eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    SdncHandle* h = sdnc_extract_handle(theta);
    if (!h || !h->w) { eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
        "sdnc-run: first argument is not an sdnc program"); result->type = ESHKOL_VALUE_NULL; return; }

    float state[SDNC_D];
    if (sdnc_read_vector_into(input_tv, state, SDNC_D) < 0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
            "sdnc-run: input must be a numeric vector");
        result->type = ESHKOL_VALUE_NULL; return;
    }
    float next[SDNC_D];
    sdnc_forward(h->w, state, h->pe, h->np, next);
    sdnc_make_tensor_f(arena, next, SDNC_D, result);
}

/* ===== (sdnc-weight-grad θ input target) -> flat ∂L/∂weights ===== */
struct sdnc_grad_flatten_ctx { float* dst; size_t n; };
static void sdnc_grad_flatten_cb(size_t idx, float* slot, void* ud) {
    struct sdnc_grad_flatten_ctx* fc = (struct sdnc_grad_flatten_ctx*)ud;
    if (idx < fc->n) fc->dst[idx] = *slot;
}

void eshkol_sdnc_weight_grad_tagged(arena_t* arena,
                                    const eshkol_tagged_value_t* theta,
                                    const eshkol_tagged_value_t* input_tv,
                                    const eshkol_tagged_value_t* target_tv,
                                    eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    SdncHandle* h = sdnc_extract_handle(theta);
    if (!h || !h->w) { eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
        "sdnc-weight-grad: first argument is not an sdnc program"); result->type = ESHKOL_VALUE_NULL; return; }

    float state[SDNC_D], target[SDNC_D];
    if (sdnc_read_vector_into(input_tv, state, SDNC_D) < 0 ||
        sdnc_read_vector_into(target_tv, target, SDNC_D) < 0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
            "sdnc-weight-grad: input/target must be numeric vectors");
        result->type = ESHKOL_VALUE_NULL; return;
    }

    /* Forward to get read, dL/dnext for L = 1/2 ||next - target||^2. */
    float next[SDNC_D], dL_dnext[SDNC_D];
    sdnc_forward(h->w, state, h->pe, h->np, next);
    for (int i = 0; i < SDNC_D; i++) dL_dnext[i] = next[i] - target[i];

    SdncGrads*    g     = (SdncGrads*)calloc(1, sizeof(SdncGrads));
    SdncFwdCache* cache = (SdncFwdCache*)calloc(1, sizeof(SdncFwdCache));
    if (!g || !cache) { free(g); free(cache); result->type = ESHKOL_VALUE_NULL; return; }

    sdnc_zero_grads(g);
    sdnc_backward(h->w, state, h->pe, h->np, dL_dnext, g, NULL, cache);

    size_t n = sdnc_param_count();
    float* flat = (float*)malloc(n * sizeof(float));
    if (!flat) { free(g); free(cache); result->type = ESHKOL_VALUE_NULL; return; }
    struct sdnc_grad_flatten_ctx fc; fc.dst = flat; fc.n = n;
    sdnc_visit_grads(g, sdnc_grad_flatten_cb, &fc);

    sdnc_make_tensor_f(arena, flat, (int)n, result);
    free(flat); free(g); free(cache);
}

/* ===== (sdnc-params θ) -> #(...) ===== */
struct sdnc_pflatten_ctx { float* dst; size_t n; };
static void sdnc_pflatten_cb(size_t idx, float* slot, void* ud) {
    struct sdnc_pflatten_ctx* fc = (struct sdnc_pflatten_ctx*)ud;
    if (idx < fc->n) fc->dst[idx] = *slot;
}
void eshkol_sdnc_params_tagged(arena_t* arena,
                               const eshkol_tagged_value_t* theta,
                               eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    SdncHandle* h = sdnc_extract_handle(theta);
    if (!arena || !h || !h->w) { result->type = ESHKOL_VALUE_NULL; return; }
    size_t n = sdnc_param_count();
    float* flat = (float*)malloc(n * sizeof(float));
    if (!flat) { result->type = ESHKOL_VALUE_NULL; return; }
    struct sdnc_pflatten_ctx fc; fc.dst = flat; fc.n = n;
    sdnc_visit_params(h->w, sdnc_pflatten_cb, &fc);
    sdnc_make_tensor_f(arena, flat, (int)n, result);
    free(flat);
}

/* ===== (sdnc-set-params! θ vec) -> θ ===== */
struct sdnc_unflatten_ctx { const float* src; size_t n; };
static void sdnc_unflatten_cb(size_t idx, float* slot, void* ud) {
    struct sdnc_unflatten_ctx* uc = (struct sdnc_unflatten_ctx*)ud;
    if (idx < uc->n) *slot = uc->src[idx];
}
void eshkol_sdnc_set_params_tagged(arena_t* arena,
                                   const eshkol_tagged_value_t* theta,
                                   const eshkol_tagged_value_t* vec,
                                   eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    (void)arena;
    SdncHandle* h = sdnc_extract_handle(theta);
    if (!h || !h->w) { result->type = ESHKOL_VALUE_NULL; return; }
    size_t n = sdnc_param_count();
    float* buf = (float*)malloc(n * sizeof(float));
    if (!buf) { result->type = ESHKOL_VALUE_NULL; return; }
    int got = sdnc_read_vector_into(vec, buf, (int)n);
    if (got < 0) {
        free(buf);
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
            "sdnc-set-params!: vec must be a numeric vector");
        result->type = ESHKOL_VALUE_NULL; return;
    }
    struct sdnc_unflatten_ctx uc; uc.src = buf; uc.n = n;
    sdnc_visit_params(h->w, sdnc_unflatten_cb, &uc);
    free(buf);
    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->data.ptr_val = (uint64_t)(uintptr_t)h;
}

/* ===== (sdnc-improve! θ data steps lr) -> θ =====
 * `data` is a (input . target) cons; if not a cons, input=data and target=
 * forward(input) nudged so the loss is a non-trivial decreasing objective.
 * Runs `steps` SGD steps at learning-rate lr, returns the (mutated) handle. */
void eshkol_sdnc_improve_tagged(arena_t* arena,
                                const eshkol_tagged_value_t* theta,
                                const eshkol_tagged_value_t* data_tv,
                                const eshkol_tagged_value_t* steps_tv,
                                const eshkol_tagged_value_t* lr_tv,
                                eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    (void)arena;
    SdncHandle* h = sdnc_extract_handle(theta);
    if (!h || !h->w) { result->type = ESHKOL_VALUE_NULL; return; }

    int steps = sdnc_get_int(steps_tv, 1);
    if (steps < 0) steps = 0;
    float lr = (float)sdnc_get_double(lr_tv, 0.002);

    float state[SDNC_D], target[SDNC_D];
    int have_target = 0;

    /* If data is a cons (input . target), read both. */
    if (data_tv && data_tv->type == ESHKOL_VALUE_HEAP_PTR && data_tv->data.ptr_val) {
        eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)(uintptr_t)data_tv->data.ptr_val);
        if (hdr->subtype == HEAP_SUBTYPE_CONS) {
            eshkol_tagged_value_t* car = (eshkol_tagged_value_t*)(uintptr_t)data_tv->data.ptr_val;
            eshkol_tagged_value_t* cdr = car + 1;
            if (sdnc_read_vector_into(car, state, SDNC_D) >= 0 &&
                sdnc_read_vector_into(cdr, target, SDNC_D) >= 0) {
                have_target = 1;
            }
        } else {
            /* data is a plain input vector */
            if (sdnc_read_vector_into(data_tv, state, SDNC_D) >= 0) {
                /* synthesize a reachable target: forward then nudge */
                sdnc_forward(h->w, state, h->pe, h->np, target);
                target[0]+=0.5f; target[1]+=0.5f; target[5]+=0.5f;
                target[10]+=0.5f; target[42]+=0.5f;
                have_target = 1;
            }
        }
    }
    if (!have_target) {
        /* No usable data: deterministic self-improve objective. */
        SdncRng r; r.state = 0x1234ABCDUL; r.scale = 0.1f;
        for (int i = 0; i < SDNC_D; i++) state[i] = sdnc_randf(&r);
        sdnc_forward(h->w, state, h->pe, h->np, target);
        target[0]+=0.5f; target[1]+=0.5f; target[5]+=0.5f;
        target[10]+=0.5f; target[42]+=0.5f;
    }

    SdncGrads*    g     = (SdncGrads*)calloc(1, sizeof(SdncGrads));
    SdncFwdCache* cache = (SdncFwdCache*)calloc(1, sizeof(SdncFwdCache));
    if (!g || !cache) { free(g); free(cache); result->type = ESHKOL_VALUE_NULL; return; }

    float next[SDNC_D], dL_dnext[SDNC_D];
    for (int it = 0; it < steps; it++) {
        sdnc_forward(h->w, state, h->pe, h->np, next);
        for (int i = 0; i < SDNC_D; i++) dL_dnext[i] = next[i] - target[i];
        sdnc_zero_grads(g);
        sdnc_backward(h->w, state, h->pe, h->np, dL_dnext, g, NULL, cache);
        sdnc_apply_grad_step(h->w, g, lr);
    }
    free(g); free(cache);

    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->data.ptr_val = (uint64_t)(uintptr_t)h;
}

/* ===== (sdnc? x) -> bool ===== */
void eshkol_sdnc_pred_tagged(arena_t* arena,
                             const eshkol_tagged_value_t* x_tv,
                             eshkol_tagged_value_t* result) {
    if (!result) return;
    (void)arena;
    memset(result, 0, sizeof(*result));
    result->type = ESHKOL_VALUE_BOOL;
    result->data.int_val = sdnc_extract_handle(x_tv) ? 1 : 0;
}
