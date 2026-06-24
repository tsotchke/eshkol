/*
 * DNC Runtime API — expose a differentiable external memory (NTM/DNC head) as
 * pure-Eshkol builtins so programs can build a learnable Mneme/dKB read & write
 * head addressed by content (cosine + softmax) and location, with a temperature
 * knob (bit-exact at high beta, smooth/differentiable at low beta).
 *
 * Builtins (registered in parser.cpp / llvm_codegen.cpp / system_codegen.cpp /
 * type_checker.cpp / eshkol-run.cpp):
 *   (make-dnc-memory N W)              -> mem    opaque handle (HEAP_SUBTYPE_DNC)
 *   (dnc-content-address mem key beta) -> #(...) length-N weight vector
 *   (dnc-loc-address addr beta N)      -> #(...) length-N weight vector
 *   (dnc-read mem wvec)                -> #(...) length-W read row
 *   (dnc-write! mem wvec erase add)    -> mem    (NTM erase/add, returns handle)
 *   (dnc-alloc-weights mem beta)       -> #(...) length-N least-used weights
 *   (dnc-read-grad mem key target beta)-> (dkey . dmem)  exact backprop gradient
 *   (dnc-memory? x)                    -> bool
 *
 * The addressing/read/write/backprop math lives in lib/core/dnc_core.h, a
 * byte-for-byte mirror of the standalone artifact lib/backend/diff_memory_prototype.c.
 * Vectors crossing the .esk boundary are Eshkol #(...) tensors (homogeneous
 * doubles, HEAP_SUBTYPE_TENSOR), matching the workspace/inference convention.
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

#include "dnc_core.h"

/* Arena handle (defined in arena_memory.h; only used by pointer here). */
typedef struct arena arena_t;

extern void* arena_allocate_with_header(arena_t* arena, size_t data_size,
                                        uint8_t subtype, uint8_t flags);
extern void* arena_allocate_aligned(arena_t* arena, size_t size, size_t alignment);
extern void  eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);

/* ===== Handle =====
 * Stored small in the arena; points to calloc'd mem/usage so a large bank never
 * goes through the arena. Owns its own dimensions so .esk can size the bank. */
typedef struct {
    int     N;
    int     W;
    double* mem;     /* N*W row-major */
    double* usage;   /* N */
} DncHandle;

/* Tensor layout mirroring #(...) literals (see workspace.cpp / inference.cpp). */
typedef struct dnc_tensor_layout {
    uint64_t* dimensions;
    uint64_t  num_dimensions;
    int64_t*  elements;       /* double bit patterns as int64 */
    uint64_t  total_elements;
} dnc_tensor_layout_t;

/* ===== Helpers ===== */
static DncHandle* dnc_extract_handle(const eshkol_tagged_value_t* tv) {
    if (!tv || tv->type != ESHKOL_VALUE_HEAP_PTR || !tv->data.ptr_val) return NULL;
    void* ptr = (void*)(uintptr_t)tv->data.ptr_val;
    eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(ptr);
    if (hdr->subtype != HEAP_SUBTYPE_DNC) return NULL;
    return (DncHandle*)ptr;
}

static int dnc_get_int(const eshkol_tagged_value_t* tv, int dflt) {
    if (!tv) return dflt;
    uint8_t t = tv->type & 0x0F;
    if (t == ESHKOL_VALUE_INT64)  return (int)tv->data.int_val;
    if (t == ESHKOL_VALUE_DOUBLE) return (int)tv->data.double_val;
    return dflt;
}

static double dnc_get_double(const eshkol_tagged_value_t* tv, double dflt) {
    if (!tv) return dflt;
    uint8_t t = tv->type & 0x0F;
    if (t == ESHKOL_VALUE_DOUBLE) return tv->data.double_val;
    if (t == ESHKOL_VALUE_INT64)  return (double)tv->data.int_val;
    return dflt;
}

/* Read a #(...) tensor (or heterogeneous vector) into a freshly malloc'd double
 * buffer; sets *out_len. Returns NULL on type error. Caller frees. */
static double* dnc_read_vector(const eshkol_tagged_value_t* tv, int64_t* out_len) {
    *out_len = 0;
    if (!tv || tv->type != ESHKOL_VALUE_HEAP_PTR || !tv->data.ptr_val) return NULL;
    void* ptr = (void*)(uintptr_t)tv->data.ptr_val;
    eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(ptr);

    if (hdr->subtype == HEAP_SUBTYPE_TENSOR) {
        dnc_tensor_layout_t* t = (dnc_tensor_layout_t*)ptr;
        int64_t n = (int64_t)t->total_elements;
        if (n <= 0 || !t->elements) return NULL;
        double* buf = (double*)malloc((size_t)n * sizeof(double));
        if (!buf) return NULL;
        for (int64_t i = 0; i < n; i++) {
            union { double d; int64_t i; } u;
            u.i = t->elements[i];
            buf[i] = u.d;
        }
        *out_len = n;
        return buf;
    }

    if (hdr->subtype == HEAP_SUBTYPE_VECTOR) {
        int64_t n = *(int64_t*)ptr;
        if (n <= 0) return NULL;
        eshkol_tagged_value_t* elems =
            (eshkol_tagged_value_t*)((uint8_t*)ptr + sizeof(int64_t));
        double* buf = (double*)malloc((size_t)n * sizeof(double));
        if (!buf) return NULL;
        for (int64_t i = 0; i < n; i++) {
            uint8_t et = elems[i].type & 0x0F;
            if (et == ESHKOL_VALUE_DOUBLE)      buf[i] = elems[i].data.double_val;
            else if (et == ESHKOL_VALUE_INT64)  buf[i] = (double)elems[i].data.int_val;
            else                                buf[i] = 0.0;
        }
        *out_len = n;
        return buf;
    }
    return NULL;
}

/* Allocate a #(...) tensor of `len` doubles into the arena and store in result. */
static void dnc_make_tensor(arena_t* arena, const double* data, int len,
                            eshkol_tagged_value_t* result) {
    memset(result, 0, sizeof(*result));
    if (!arena || !data || len <= 0) { result->type = ESHKOL_VALUE_NULL; return; }

    dnc_tensor_layout_t* t = (dnc_tensor_layout_t*)arena_allocate_with_header(
        arena, sizeof(dnc_tensor_layout_t), HEAP_SUBTYPE_TENSOR, 0);
    if (!t) { result->type = ESHKOL_VALUE_NULL; return; }
    t->num_dimensions = 1;
    t->total_elements = (uint64_t)len;
    t->dimensions = (uint64_t*)arena_allocate_aligned(arena, sizeof(uint64_t), 8);
    if (t->dimensions) t->dimensions[0] = (uint64_t)len;
    t->elements = (int64_t*)arena_allocate_aligned(arena, (size_t)len * sizeof(int64_t), 8);
    if (!t->elements) { result->type = ESHKOL_VALUE_NULL; return; }
    for (int i = 0; i < len; i++) {
        union { double d; int64_t i; } u;
        u.d = data[i];
        t->elements[i] = u.i;
    }
    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->data.ptr_val = (uint64_t)(uintptr_t)t;
}

/* ===== (make-dnc-memory N W) -> mem ===== */
void eshkol_dnc_make_tagged(arena_t* arena,
                            const eshkol_tagged_value_t* n_tv,
                            const eshkol_tagged_value_t* w_tv,
                            eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    if (!arena) { result->type = ESHKOL_VALUE_NULL; return; }

    int N = dnc_get_int(n_tv, 0);
    int W = dnc_get_int(w_tv, 0);
    if (N <= 0 || W <= 0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
            "make-dnc-memory: N and W must be positive (got N=%d W=%d)", N, W);
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    DncHandle* h = (DncHandle*)arena_allocate_with_header(
        arena, sizeof(DncHandle), HEAP_SUBTYPE_DNC, 0);
    if (!h) { result->type = ESHKOL_VALUE_NULL; return; }
    h->N = N; h->W = W;
    h->mem = (double*)calloc((size_t)N * W, sizeof(double));   /* zeroed bank */
    h->usage = (double*)calloc((size_t)N, sizeof(double));     /* usage = 0 */
    if (!h->mem || !h->usage) {
        free(h->mem); free(h->usage);
        result->type = ESHKOL_VALUE_NULL; return;
    }

    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->data.ptr_val = (uint64_t)(uintptr_t)h;
}

/* ===== (dnc-content-address mem key beta) -> length-N wvec ===== */
void eshkol_dnc_content_address_tagged(arena_t* arena,
                                       const eshkol_tagged_value_t* mem_tv,
                                       const eshkol_tagged_value_t* key_tv,
                                       const eshkol_tagged_value_t* beta_tv,
                                       eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    DncHandle* h = dnc_extract_handle(mem_tv);
    if (!h) { eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
        "dnc-content-address: first argument is not a dnc-memory"); result->type = ESHKOL_VALUE_NULL; return; }

    int64_t klen = 0;
    double* key = dnc_read_vector(key_tv, &klen);
    if (!key) { eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
        "dnc-content-address: key must be a numeric vector"); result->type = ESHKOL_VALUE_NULL; return; }
    if (klen != h->W) {
        free(key);
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
            "dnc-content-address: key length %lld != memory width W=%d",
            (long long)klen, h->W);
        result->type = ESHKOL_VALUE_NULL; return;
    }
    double beta = dnc_get_double(beta_tv, 1.0);

    double* w = (double*)malloc((size_t)h->N * sizeof(double));
    if (!w) { free(key); result->type = ESHKOL_VALUE_NULL; return; }
    dnc_content_weights(h->mem, h->N, h->W, key, beta, w);
    dnc_make_tensor(arena, w, h->N, result);
    free(key); free(w);
}

/* ===== (dnc-loc-address addr beta N) -> length-N wvec ===== */
void eshkol_dnc_loc_address_tagged(arena_t* arena,
                                   const eshkol_tagged_value_t* addr_tv,
                                   const eshkol_tagged_value_t* beta_tv,
                                   const eshkol_tagged_value_t* n_tv,
                                   eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    double addr = dnc_get_double(addr_tv, 0.0);
    double beta = dnc_get_double(beta_tv, 1.0);
    int N = dnc_get_int(n_tv, 0);
    if (N <= 0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
            "dnc-loc-address: N must be positive (got %d)", N);
        result->type = ESHKOL_VALUE_NULL; return;
    }
    double* w = (double*)malloc((size_t)N * sizeof(double));
    if (!w) { result->type = ESHKOL_VALUE_NULL; return; }
    dnc_loc_weights(addr, beta, N, w);
    dnc_make_tensor(arena, w, N, result);
    free(w);
}

/* ===== (dnc-read mem wvec) -> length-W row ===== */
void eshkol_dnc_read_tagged(arena_t* arena,
                            const eshkol_tagged_value_t* mem_tv,
                            const eshkol_tagged_value_t* w_tv,
                            eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    DncHandle* h = dnc_extract_handle(mem_tv);
    if (!h) { eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
        "dnc-read: first argument is not a dnc-memory"); result->type = ESHKOL_VALUE_NULL; return; }

    int64_t wlen = 0;
    double* w = dnc_read_vector(w_tv, &wlen);
    if (!w) { eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
        "dnc-read: wvec must be a numeric vector"); result->type = ESHKOL_VALUE_NULL; return; }
    if (wlen != h->N) {
        free(w);
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
            "dnc-read: wvec length %lld != memory rows N=%d", (long long)wlen, h->N);
        result->type = ESHKOL_VALUE_NULL; return;
    }
    double* rd = (double*)malloc((size_t)h->W * sizeof(double));
    if (!rd) { free(w); result->type = ESHKOL_VALUE_NULL; return; }
    dnc_read_mem(h->mem, h->N, h->W, w, rd);
    dnc_make_tensor(arena, rd, h->W, result);
    free(w); free(rd);
}

/* ===== (dnc-write! mem wvec erase add) -> mem ===== */
void eshkol_dnc_write_tagged(arena_t* arena,
                             const eshkol_tagged_value_t* mem_tv,
                             const eshkol_tagged_value_t* w_tv,
                             const eshkol_tagged_value_t* erase_tv,
                             const eshkol_tagged_value_t* add_tv,
                             eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    (void)arena;
    DncHandle* h = dnc_extract_handle(mem_tv);
    if (!h) { eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
        "dnc-write!: first argument is not a dnc-memory"); result->type = ESHKOL_VALUE_NULL; return; }

    int64_t wlen = 0, elen = 0, alen = 0;
    double* w = dnc_read_vector(w_tv, &wlen);
    double* erase = dnc_read_vector(erase_tv, &elen);
    double* add = dnc_read_vector(add_tv, &alen);
    if (!w || !erase || !add) {
        free(w); free(erase); free(add);
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
            "dnc-write!: wvec/erase/add must be numeric vectors");
        result->type = ESHKOL_VALUE_NULL; return;
    }
    if (wlen != h->N) {
        free(w); free(erase); free(add);
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
            "dnc-write!: wvec length %lld != memory rows N=%d", (long long)wlen, h->N);
        result->type = ESHKOL_VALUE_NULL; return;
    }
    if (elen != h->W || alen != h->W) {
        free(w); free(erase); free(add);
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
            "dnc-write!: erase/add length must equal memory width W=%d (got %lld/%lld)",
            h->W, (long long)elen, (long long)alen);
        result->type = ESHKOL_VALUE_NULL; return;
    }
    dnc_write_mem(h->mem, h->usage, h->N, h->W, w, erase, add);
    free(w); free(erase); free(add);

    /* Return the handle for chaining. */
    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->data.ptr_val = (uint64_t)(uintptr_t)h;
}

/* ===== (dnc-alloc-weights mem beta) -> length-N wvec ===== */
void eshkol_dnc_alloc_weights_tagged(arena_t* arena,
                                     const eshkol_tagged_value_t* mem_tv,
                                     const eshkol_tagged_value_t* beta_tv,
                                     eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    DncHandle* h = dnc_extract_handle(mem_tv);
    if (!h) { eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
        "dnc-alloc-weights: first argument is not a dnc-memory"); result->type = ESHKOL_VALUE_NULL; return; }
    double beta = dnc_get_double(beta_tv, 1.0);
    double* w = (double*)malloc((size_t)h->N * sizeof(double));
    if (!w) { result->type = ESHKOL_VALUE_NULL; return; }
    dnc_alloc_weights(h->usage, h->N, beta, w);
    dnc_make_tensor(arena, w, h->N, result);
    free(w);
}

/* ===== (dnc-read-grad mem key target beta) -> (dkey . dmem) ===== */
extern void* arena_allocate_cons_with_header(arena_t* arena);

void eshkol_dnc_read_grad_tagged(arena_t* arena,
                                 const eshkol_tagged_value_t* mem_tv,
                                 const eshkol_tagged_value_t* key_tv,
                                 const eshkol_tagged_value_t* target_tv,
                                 const eshkol_tagged_value_t* beta_tv,
                                 eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    DncHandle* h = dnc_extract_handle(mem_tv);
    if (!h) { eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
        "dnc-read-grad: first argument is not a dnc-memory"); result->type = ESHKOL_VALUE_NULL; return; }

    int64_t klen = 0, tlen = 0;
    double* key = dnc_read_vector(key_tv, &klen);
    double* target = dnc_read_vector(target_tv, &tlen);
    if (!key || !target) {
        free(key); free(target);
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
            "dnc-read-grad: key/target must be numeric vectors");
        result->type = ESHKOL_VALUE_NULL; return;
    }
    if (klen != h->W || tlen != h->W) {
        free(key); free(target);
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
            "dnc-read-grad: key/target length must equal memory width W=%d (got %lld/%lld)",
            h->W, (long long)klen, (long long)tlen);
        result->type = ESHKOL_VALUE_NULL; return;
    }
    double beta = dnc_get_double(beta_tv, 1.0);

    int N = h->N, W = h->W;
    double* grad_key = (double*)malloc((size_t)W * sizeof(double));
    double* grad_mem = (double*)malloc((size_t)N * W * sizeof(double));
    double* s     = (double*)malloc((size_t)N * sizeof(double));
    double* wv    = (double*)malloc((size_t)N * sizeof(double));
    double* rd    = (double*)malloc((size_t)W * sizeof(double));
    double* dread = (double*)malloc((size_t)W * sizeof(double));
    double* dw    = (double*)malloc((size_t)N * sizeof(double));
    double* ds    = (double*)malloc((size_t)N * sizeof(double));
    if (!grad_key||!grad_mem||!s||!wv||!rd||!dread||!dw||!ds) {
        free(key);free(target);free(grad_key);free(grad_mem);
        free(s);free(wv);free(rd);free(dread);free(dw);free(ds);
        result->type = ESHKOL_VALUE_NULL; return;
    }

    dnc_loss_and_grads(h->mem, N, W, key, target, beta,
                       grad_key, grad_mem, s, wv, rd, dread, dw, ds);

    /* Build (dkey . dmem): dkey = length-W tensor, dmem = length-(N*W) tensor. */
    eshkol_tagged_value_t dkey_tv, dmem_tv;
    dnc_make_tensor(arena, grad_key, W, &dkey_tv);
    dnc_make_tensor(arena, grad_mem, N * W, &dmem_tv);

    void* cellp = arena_allocate_cons_with_header(arena);
    if (!cellp) {
        free(key);free(target);free(grad_key);free(grad_mem);
        free(s);free(wv);free(rd);free(dread);free(dw);free(ds);
        result->type = ESHKOL_VALUE_NULL; return;
    }
    /* arena_tagged_cons_cell_t layout: { eshkol_tagged_value_t car; eshkol_tagged_value_t cdr; }
     * after the object header. We access fields via the known offsets used by
     * the runtime list helpers: car first, cdr second. */
    eshkol_tagged_value_t* car = (eshkol_tagged_value_t*)cellp;
    eshkol_tagged_value_t* cdr = car + 1;
    *car = dkey_tv;
    *cdr = dmem_tv;

    free(key);free(target);free(grad_key);free(grad_mem);
    free(s);free(wv);free(rd);free(dread);free(dw);free(ds);

    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->data.ptr_val = (uint64_t)(uintptr_t)cellp;
}

/* ===== (dnc-memory? x) -> bool ===== */
void eshkol_dnc_pred_tagged(arena_t* arena,
                            const eshkol_tagged_value_t* x_tv,
                            eshkol_tagged_value_t* result) {
    if (!result) return;
    (void)arena;
    memset(result, 0, sizeof(*result));
    result->type = ESHKOL_VALUE_BOOL;
    result->data.int_val = dnc_extract_handle(x_tv) ? 1 : 0;
}
