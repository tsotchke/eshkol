/**
 * @file vm_tensor.c
 * @brief Core tensor type and basic operations for the Eshkol bytecode VM.
 *
 * Implements N-dimensional tensors (up to 8 dims) with stride-based indexing,
 * views (reshape/transpose/flatten without copying), and creation helpers.
 * All data is arena-allocated via vm_arena.h. No GC.
 *
 * Native call IDs: 410-439
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef VM_TENSOR_C_INCLUDED
#define VM_TENSOR_C_INCLUDED

#include "vm_numeric.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ── Max dimensions ── */
#define VM_TENSOR_MAX_DIMS 8

/* ── Tensor dtypes ── */
#define VM_TENSOR_DTYPE_F64  0
#define VM_TENSOR_DTYPE_F32  1
#define VM_TENSOR_DTYPE_F16  2
#define VM_TENSOR_DTYPE_BF16 3
#define VM_TENSOR_DTYPE_I8   4

/* ── Tensor ── */
typedef struct {
    int      n_dims;
    int64_t  shape[VM_TENSOR_MAX_DIMS];
    int64_t  strides[VM_TENSOR_MAX_DIMS];
    double*  data;         /* arena-allocated */
    int64_t  total;        /* total number of elements */
    int      owns_data;    /* 1 = owns data (allocated), 0 = view (shared) */
    int      dtype;        /* VM_TENSOR_DTYPE_* metadata; data stays double-backed */
} VmTensor;

/* ── Internal helpers ── */

/** @brief Clamp an arbitrary dtype tag to a known VM_TENSOR_DTYPE_*
 *         value, defaulting to F64 for anything unrecognized. */
static int vm_tensor_normalize_dtype(int dtype) {
    switch (dtype) {
        case VM_TENSOR_DTYPE_F64:
        case VM_TENSOR_DTYPE_F32:
        case VM_TENSOR_DTYPE_F16:
        case VM_TENSOR_DTYPE_BF16:
        case VM_TENSOR_DTYPE_I8:
            return dtype;
        default:
            return VM_TENSOR_DTYPE_F64;
    }
}

/** @brief Human-readable name ("f64"/"f32"/"f16"/"bf16"/"i8") for a dtype
 *         tag. */
static const char* vm_tensor_dtype_name(int dtype) {
    switch (vm_tensor_normalize_dtype(dtype)) {
        case VM_TENSOR_DTYPE_F32:  return "f32";
        case VM_TENSOR_DTYPE_F16:  return "f16";
        case VM_TENSOR_DTYPE_BF16: return "bf16";
        case VM_TENSOR_DTYPE_I8:   return "i8";
        case VM_TENSOR_DTYPE_F64:
        default:                   return "f64";
    }
}

/** @brief Parse a dtype name (optionally prefixed with `'`/`:`/`#` as in
 *         Scheme symbol/keyword syntax) back to its VM_TENSOR_DTYPE_*
 *         tag, defaulting to F64 if unrecognized. */
static int vm_tensor_dtype_from_name(const char* name) {
    if (!name) return VM_TENSOR_DTYPE_F64;
    while (*name == '\'' || *name == ':' || *name == '#') name++;
    if (strcmp(name, "f32") == 0) return VM_TENSOR_DTYPE_F32;
    if (strcmp(name, "f16") == 0) return VM_TENSOR_DTYPE_F16;
    if (strcmp(name, "bf16") == 0) return VM_TENSOR_DTYPE_BF16;
    if (strcmp(name, "i8") == 0) return VM_TENSOR_DTYPE_I8;
    return VM_TENSOR_DTYPE_F64;
}

/** @brief Result dtype for a binary op between two tensors, per the
 *         precision hierarchy f64 > f32 > bf16 > f16 > i8 (wider/more
 *         precise type wins; null operands are treated as f64). */
static int vm_tensor_promote_dtype(const VmTensor* a, const VmTensor* b) {
    int ad = a ? vm_tensor_normalize_dtype(a->dtype) : VM_TENSOR_DTYPE_F64;
    int bd = b ? vm_tensor_normalize_dtype(b->dtype) : VM_TENSOR_DTYPE_F64;
    if (ad == bd) return ad;
    if (ad == VM_TENSOR_DTYPE_F64 || bd == VM_TENSOR_DTYPE_F64) return VM_TENSOR_DTYPE_F64;
    if (ad == VM_TENSOR_DTYPE_F32 || bd == VM_TENSOR_DTYPE_F32) return VM_TENSOR_DTYPE_F32;
    if (ad == VM_TENSOR_DTYPE_BF16 || bd == VM_TENSOR_DTYPE_BF16) return VM_TENSOR_DTYPE_BF16;
    if (ad == VM_TENSOR_DTYPE_F16 || bd == VM_TENSOR_DTYPE_F16) return VM_TENSOR_DTYPE_F16;
    return VM_TENSOR_DTYPE_I8;
}

/** @brief Convert an IEEE-754 float32 to its IEEE-754 binary16 (half
 *         precision) bit pattern, with round-to-nearest and
 *         subnormal/overflow handling. */
static uint16_t vm_tensor_float_to_half_bits(float value) {
    union { float f; uint32_t u; } in;
    in.f = value;
    uint32_t sign = (in.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((in.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = in.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (uint32_t)(1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000u) >> 13));
}

/** @brief Convert an IEEE-754 binary16 bit pattern back to a float32
 *         (handling subnormals, zero, and infinity/NaN). */
static float vm_tensor_half_bits_to_float(uint16_t bits) {
    uint32_t sign = ((uint32_t)bits & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)bits >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)bits & 0x03ffu;
    uint32_t out;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    union { uint32_t u; float f; } result;
    result.u = out;
    return result.f;
}

/** @brief Round-trip @p value through the storage precision implied by
 *         @p dtype (float32, float16, bfloat16, or int8 clamped to
 *         [-128,127]), while the tensor itself always stays
 *         double-backed. F64 is a no-op passthrough. */
static double vm_tensor_quantize_value(double value, int dtype) {
    switch (vm_tensor_normalize_dtype(dtype)) {
        case VM_TENSOR_DTYPE_F32:
            return (double)(float)value;
        case VM_TENSOR_DTYPE_F16:
            return (double)vm_tensor_half_bits_to_float(vm_tensor_float_to_half_bits((float)value));
        case VM_TENSOR_DTYPE_BF16: {
            union { float f; uint32_t u; } conv;
            conv.f = (float)value;
            conv.u &= 0xffff0000u;
            return (double)conv.f;
        }
        case VM_TENSOR_DTYPE_I8:
            if (value > 127.0) return 127.0;
            if (value < -128.0) return -128.0;
            return nearbyint(value);
        case VM_TENSOR_DTYPE_F64:
        default:
            return value;
    }
}

/** @brief Compute row-major strides from @p shape into @p strides.
 * @return The total element count, 0 on a non-positive dimension, or -1
 *         on overflow.
 */
static int64_t vm_tensor_compute_strides(const int64_t* shape, int n_dims, int64_t* strides) {
    if (n_dims <= 0) return 0;
    strides[n_dims - 1] = 1;
    for (int i = n_dims - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t total = 1;
    for (int i = 0; i < n_dims; i++) {
        if (shape[i] <= 0) return 0;
        if (total > INT64_MAX / shape[i]) {
            fprintf(stderr, "ERROR: tensor shape overflow (dim %d = %lld)\n", i, (long long)shape[i]);
            return -1;
        }
        total *= shape[i];
    }
    return total;
}

/** @brief Convert multi-dimensional @p indices to a flat data offset using
 *         @p t's strides. */
static int64_t vm_tensor_flat_offset(const VmTensor* t, const int64_t* indices, int n_indices) {
    int64_t off = 0;
    for (int i = 0; i < n_indices && i < t->n_dims; i++) {
        off += indices[i] * t->strides[i];
    }
    return off;
}

/** @brief Convert a flat index to multi-dimensional indices for @p shape
 *         (row-major unravel), writing them to @p out. */
static void vm_tensor_unravel(int64_t flat, const int64_t* shape, int n_dims, int64_t* out) {
    for (int i = n_dims - 1; i >= 0; i--) {
        out[i] = flat % shape[i];
        flat /= shape[i];
    }
}

/* ── Allocation ── */

/** @brief Native call 410: allocate a new zero-initialized (f64) tensor
 *         with the given @p shape. */
static VmTensor* vm_tensor_new(VmRegionStack* rs, const int64_t* shape, int n_dims) {
    if (n_dims <= 0 || n_dims > VM_TENSOR_MAX_DIMS) return NULL;

    VmTensor* t = (VmTensor*)vm_alloc_object(rs, VM_SUBTYPE_TENSOR, sizeof(VmTensor));
    if (!t) return NULL;

    t->n_dims = n_dims;
    memcpy(t->shape, shape, (size_t)n_dims * sizeof(int64_t));
    t->total = vm_tensor_compute_strides(shape, n_dims, t->strides);

    if (t->total <= 0) return NULL;

    t->data = (double*)vm_alloc(rs, (size_t)t->total * sizeof(double));
    if (!t->data) return NULL;
    memset(t->data, 0, (size_t)t->total * sizeof(double));
    t->owns_data = 1;
    t->dtype = VM_TENSOR_DTYPE_F64;

    return t;
}

/** @brief Native call 411: allocate a tensor of @p shape filled with
 *         @p fill_val. */
static VmTensor* vm_tensor_fill(VmRegionStack* rs, const int64_t* shape, int n_dims, double fill_val) {
    VmTensor* t = vm_tensor_new(rs, shape, n_dims);
    if (!t) return NULL;
    for (int64_t i = 0; i < t->total; i++) {
        t->data[i] = fill_val;
    }
    return t;
}

/** @brief Native call 412: allocate a tensor of @p shape, copying
 *         @p data's contents into the arena. */
static VmTensor* vm_tensor_from_data(VmRegionStack* rs, const double* data,
                                     const int64_t* shape, int n_dims) {
    if (!data) return NULL;
    VmTensor* t = vm_tensor_new(rs, shape, n_dims);
    if (!t) return NULL;
    memcpy(t->data, data, (size_t)t->total * sizeof(double));
    return t;
}

/* ── Element Access ── */

/** @brief Native call 413: `(tensor-ref t indices...)` — read an element
 *         by multi-dimensional indices (0.0 if out of range). */
static double vm_tensor_ref(const VmTensor* t, const int64_t* indices, int n_indices) {
    if (!t || !t->data || n_indices != t->n_dims) return 0.0;
    int64_t off = vm_tensor_flat_offset(t, indices, n_indices);
    if (off < 0 || off >= t->total) return 0.0;
    return t->data[off];
}

/** @brief Native call 414: `(tensor-set! t indices... val)` — write an
 *         element by multi-dimensional indices (no-op if out of range). */
static void vm_tensor_set(VmTensor* t, const int64_t* indices, int n_indices, double val) {
    if (!t || !t->data || n_indices != t->n_dims) return;
    int64_t off = vm_tensor_flat_offset(t, indices, n_indices);
    if (off < 0 || off >= t->total) return;
    t->data[off] = val;
}

/* ── Views (shared data, no copy) ── */

/** @brief Native call 415: `(tensor-reshape t new-shape)` — create a view
 *         with a new shape over the same underlying data (total element
 *         count must match); strides are recomputed as contiguous
 *         row-major. */
static VmTensor* vm_tensor_reshape(VmRegionStack* rs, const VmTensor* t,
                                   const int64_t* new_shape, int new_dims) {
    if (!t || new_dims <= 0 || new_dims > VM_TENSOR_MAX_DIMS) return NULL;

    /* Compute new total and verify it matches */
    int64_t new_total = 1;
    for (int i = 0; i < new_dims; i++) {
        if (new_shape[i] <= 0) return NULL;
        if (new_total > INT64_MAX / new_shape[i]) return NULL;
        new_total *= new_shape[i];
    }
    if (new_total != t->total) return NULL;

    VmTensor* v = (VmTensor*)vm_alloc_object(rs, VM_SUBTYPE_TENSOR, sizeof(VmTensor));
    if (!v) return NULL;

    v->n_dims = new_dims;
    memcpy(v->shape, new_shape, (size_t)new_dims * sizeof(int64_t));
    v->total = vm_tensor_compute_strides(new_shape, new_dims, v->strides);
    v->data = t->data;   /* shared data */
    v->owns_data = 0;
    v->dtype = t->dtype;

    return v;
}

/** @brief Native call 416: `(tensor-transpose t)` — view swapping
 *         dimensions 0 and 1 (shares data via swapped strides). */
static VmTensor* vm_tensor_transpose(VmRegionStack* rs, const VmTensor* t) {
    if (!t || t->n_dims < 2) return NULL;

    VmTensor* v = (VmTensor*)vm_alloc_object(rs, VM_SUBTYPE_TENSOR, sizeof(VmTensor));
    if (!v) return NULL;

    v->n_dims = t->n_dims;
    memcpy(v->shape, t->shape, (size_t)t->n_dims * sizeof(int64_t));
    memcpy(v->strides, t->strides, (size_t)t->n_dims * sizeof(int64_t));

    /* Swap dim 0 and dim 1 */
    v->shape[0] = t->shape[1];
    v->shape[1] = t->shape[0];
    v->strides[0] = t->strides[1];
    v->strides[1] = t->strides[0];

    v->data = t->data;
    v->total = t->total;
    v->owns_data = 0;
    v->dtype = t->dtype;

    return v;
}

/** @brief Native call 417: `(tensor-flatten t)` — 1D view of all elements
 *         (via vm_tensor_reshape() to shape [total]). */
static VmTensor* vm_tensor_flatten(VmRegionStack* rs, const VmTensor* t) {
    if (!t) return NULL;
    int64_t flat_shape[1] = { t->total };
    return vm_tensor_reshape(rs, t, flat_shape, 1);
}

/* ── Creation Helpers ── */

/** @brief Native call 418: `(tensor-zeros shape)`. */
static VmTensor* vm_tensor_zeros(VmRegionStack* rs, const int64_t* shape, int n_dims) {
    return vm_tensor_new(rs, shape, n_dims);
}

/** @brief Native call 419: `(tensor-ones shape)`. */
static VmTensor* vm_tensor_ones(VmRegionStack* rs, const int64_t* shape, int n_dims) {
    return vm_tensor_fill(rs, shape, n_dims, 1.0);
}

/** @brief Native call 420: `(tensor-arange start stop step)` — 1D tensor
 *         [start, start+step, ..., stop). */
static VmTensor* vm_tensor_arange(VmRegionStack* rs, double start, double stop, double step) {
    if (step == 0.0) return NULL;
    if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) return NULL;

    int64_t n = (int64_t)ceil((stop - start) / step);
    if (n <= 0) return NULL;

    int64_t shape[1] = { n };
    VmTensor* t = vm_tensor_new(rs, shape, 1);
    if (!t) return NULL;

    for (int64_t i = 0; i < n; i++) {
        t->data[i] = start + (double)i * step;
    }
    return t;
}

/** @brief Native call 421: `(tensor-copy t)` — deep copy with freshly
 *         allocated, contiguous data (element-wise, unraveling indices,
 *         for a non-contiguous source such as a transposed view). */
static VmTensor* vm_tensor_copy(VmRegionStack* rs, const VmTensor* t) {
    if (!t) return NULL;

    VmTensor* c = (VmTensor*)vm_alloc_object(rs, VM_SUBTYPE_TENSOR, sizeof(VmTensor));
    if (!c) return NULL;

    c->n_dims = t->n_dims;
    memcpy(c->shape, t->shape, (size_t)t->n_dims * sizeof(int64_t));
    c->total = vm_tensor_compute_strides(t->shape, t->n_dims, c->strides);
    c->owns_data = 1;
    c->dtype = t->dtype;

    c->data = (double*)vm_alloc(rs, (size_t)c->total * sizeof(double));
    if (!c->data) return NULL;

    /* Handle non-contiguous source (e.g. transposed view) by element-wise copy */
    int is_contiguous = 1;
    {
        int64_t expected_strides[VM_TENSOR_MAX_DIMS];
        vm_tensor_compute_strides(t->shape, t->n_dims, expected_strides);
        for (int i = 0; i < t->n_dims; i++) {
            if (t->strides[i] != expected_strides[i]) {
                is_contiguous = 0;
                break;
            }
        }
    }

    if (is_contiguous) {
        memcpy(c->data, t->data, (size_t)c->total * sizeof(double));
    } else {
        /* Element-wise copy for non-contiguous sources */
        int64_t indices[VM_TENSOR_MAX_DIMS];
        for (int64_t i = 0; i < c->total; i++) {
            vm_tensor_unravel(i, t->shape, t->n_dims, indices);
            int64_t src_off = vm_tensor_flat_offset(t, indices, t->n_dims);
            c->data[i] = t->data[src_off];
        }
    }

    return c;
}

/** @brief Native call 422: `(tensor-linspace start stop n)` — 1D tensor of
 *         @p n evenly spaced values from @p start to @p stop (inclusive). */
static VmTensor* vm_tensor_linspace(VmRegionStack* rs, double start, double stop, int64_t n) {
    if (n <= 0) return NULL;

    int64_t shape[1] = { n };
    VmTensor* t = vm_tensor_new(rs, shape, 1);
    if (!t) return NULL;

    if (n == 1) {
        t->data[0] = start;
    } else {
        double step = (stop - start) / (double)(n - 1);
        for (int64_t i = 0; i < n; i++) {
            t->data[i] = start + (double)i * step;
        }
    }
    return t;
}

/** @brief Native call 423: `(tensor-eye n)` — n x n identity matrix. */
static VmTensor* vm_tensor_eye(VmRegionStack* rs, int64_t n) {
    if (n <= 0) return NULL;
    int64_t shape[2] = { n, n };
    VmTensor* t = vm_tensor_zeros(rs, shape, 2);
    if (!t) return NULL;
    for (int64_t i = 0; i < n; i++) {
        t->data[i * n + i] = 1.0;
    }
    return t;
}

/** @brief Native call 424: wrap a scalar as a 0D-like 1-element tensor
 *         (shape [1]). */
static VmTensor* vm_tensor_scalar(VmRegionStack* rs, double val) {
    int64_t shape[1] = { 1 };
    return vm_tensor_fill(rs, shape, 1, val);
}

/** @brief Native call 425: `(tensor-shape t)` — return @p t's shape as a
 *         new 1D tensor. */
static VmTensor* vm_tensor_get_shape(VmRegionStack* rs, const VmTensor* t) {
    if (!t) return NULL;
    int64_t shape[1] = { t->n_dims };
    VmTensor* s = vm_tensor_new(rs, shape, 1);
    if (!s) return NULL;
    for (int i = 0; i < t->n_dims; i++) {
        s->data[i] = (double)t->shape[i];
    }
    return s;
}

/** @brief Native call 426: `(tensor-size t)` — total element count. */
static int64_t vm_tensor_size(const VmTensor* t) {
    return t ? t->total : 0;
}

/** @brief Native call 427: `(tensor-ndim t)` — number of dimensions. */
static int vm_tensor_ndim(const VmTensor* t) {
    return t ? t->n_dims : 0;
}

/** @brief Native call 428: `(tensor-slice t index)` — view of
 *         t[index, ...] along axis 0 (result has n_dims - 1 dimensions,
 *         sharing data). */
static VmTensor* vm_tensor_slice(VmRegionStack* rs, const VmTensor* t, int64_t index) {
    if (!t || t->n_dims < 2) return NULL;
    if (index < 0 || index >= t->shape[0]) return NULL;

    VmTensor* v = (VmTensor*)vm_alloc_object(rs, VM_SUBTYPE_TENSOR, sizeof(VmTensor));
    if (!v) return NULL;

    v->n_dims = t->n_dims - 1;
    for (int i = 0; i < v->n_dims; i++) {
        v->shape[i] = t->shape[i + 1];
        v->strides[i] = t->strides[i + 1];
    }
    v->data = t->data + index * t->strides[0];
    v->total = t->total / t->shape[0];
    v->owns_data = 0;
    v->dtype = t->dtype;

    return v;
}

/** @brief Cast @p src to a new tensor of dtype @p dtype, quantizing each
 *         element via vm_tensor_quantize_value() (element-wise, unraveling
 *         indices, for a non-contiguous source). */
static VmTensor* vm_tensor_cast_dtype(VmRegionStack* rs, const VmTensor* src, int dtype) {
    if (!src) return NULL;
    VmTensor* out = vm_tensor_new(rs, src->shape, src->n_dims);
    if (!out) return NULL;
    out->dtype = vm_tensor_normalize_dtype(dtype);
    int64_t expected_strides[VM_TENSOR_MAX_DIMS];
    vm_tensor_compute_strides(src->shape, src->n_dims, expected_strides);
    int is_contiguous = 1;
    for (int i = 0; i < src->n_dims; i++) {
        if (src->strides[i] != expected_strides[i]) {
            is_contiguous = 0;
            break;
        }
    }
    if (is_contiguous) {
        for (int64_t i = 0; i < out->total; i++) {
            out->data[i] = vm_tensor_quantize_value(src->data[i], out->dtype);
        }
    } else {
        int64_t indices[VM_TENSOR_MAX_DIMS];
        for (int64_t i = 0; i < out->total; i++) {
            vm_tensor_unravel(i, src->shape, src->n_dims, indices);
            int64_t src_off = vm_tensor_flat_offset(src, indices, src->n_dims);
            out->data[i] = vm_tensor_quantize_value(src->data[src_off], out->dtype);
        }
    }
    return out;
}

/* ── Self-Test ── */

#ifdef VM_TENSOR_TEST
#include <assert.h>

/** @brief Standalone self-test (built when VM_TENSOR_TEST is defined):
 *         exercises creation helpers, element access, views
 *         (reshape/transpose/flatten/slice), copy, and dtype casting. */
int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    /* --- zeros --- */
    {
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_zeros(&rs, shape, 2);
        assert(t != NULL);
        assert(t->n_dims == 2);
        assert(t->shape[0] == 2 && t->shape[1] == 3);
        assert(t->total == 6);
        for (int64_t i = 0; i < t->total; i++) assert(t->data[i] == 0.0);
        printf("  zeros(2,3):  OK\n");
    }

    /* --- ones --- */
    {
        int64_t shape[] = { 3, 4 };
        VmTensor* t = vm_tensor_ones(&rs, shape, 2);
        assert(t != NULL);
        assert(t->total == 12);
        for (int64_t i = 0; i < t->total; i++) assert(t->data[i] == 1.0);
        printf("  ones(3,4):   OK\n");
    }

    /* --- fill --- */
    {
        int64_t shape[] = { 5 };
        VmTensor* t = vm_tensor_fill(&rs, shape, 1, 3.14);
        assert(t != NULL);
        for (int64_t i = 0; i < 5; i++) assert(fabs(t->data[i] - 3.14) < 1e-12);
        printf("  fill(5,3.14):OK\n");
    }

    /* --- arange --- */
    {
        VmTensor* t = vm_tensor_arange(&rs, 0.0, 5.0, 1.0);
        assert(t != NULL);
        assert(t->total == 5);
        for (int64_t i = 0; i < 5; i++) assert(t->data[i] == (double)i);
        printf("  arange(0,5,1):OK\n");
    }

    /* --- arange fractional step --- */
    {
        VmTensor* t = vm_tensor_arange(&rs, 0.0, 1.0, 0.25);
        assert(t != NULL);
        assert(t->total == 4);
        assert(fabs(t->data[0] - 0.0) < 1e-12);
        assert(fabs(t->data[3] - 0.75) < 1e-12);
        printf("  arange(0,1,0.25):OK\n");
    }

    /* --- from_data --- */
    {
        double data[] = { 1, 2, 3, 4, 5, 6 };
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        assert(t != NULL);
        assert(t->data[0] == 1.0 && t->data[5] == 6.0);
        printf("  from_data:   OK\n");
    }

    /* --- ref/set --- */
    {
        double data[] = { 1, 2, 3, 4, 5, 6 };
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);

        /* ref: t[1,2] = 6.0 (flat offset = 1*3 + 2 = 5) */
        int64_t idx1[] = { 1, 2 };
        assert(vm_tensor_ref(t, idx1, 2) == 6.0);

        /* ref: t[0,0] = 1.0 */
        int64_t idx0[] = { 0, 0 };
        assert(vm_tensor_ref(t, idx0, 2) == 1.0);

        /* set: t[0,1] = 99.0 */
        int64_t idx01[] = { 0, 1 };
        vm_tensor_set(t, idx01, 2, 99.0);
        assert(vm_tensor_ref(t, idx01, 2) == 99.0);

        printf("  ref/set:     OK\n");
    }

    /* --- strides are row-major --- */
    {
        int64_t shape[] = { 2, 3, 4 };
        VmTensor* t = vm_tensor_new(&rs, shape, 3);
        assert(t->strides[0] == 12); /* 3*4 */
        assert(t->strides[1] == 4);  /* 4 */
        assert(t->strides[2] == 1);
        printf("  strides:     OK\n");
    }

    /* --- reshape --- */
    {
        double data[] = { 1, 2, 3, 4, 5, 6 };
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);

        int64_t new_shape[] = { 3, 2 };
        VmTensor* v = vm_tensor_reshape(&rs, t, new_shape, 2);
        assert(v != NULL);
        assert(v->shape[0] == 3 && v->shape[1] == 2);
        assert(v->data == t->data); /* shared data */
        assert(v->owns_data == 0);
        printf("  reshape:     OK\n");
    }

    /* --- reshape mismatch fails --- */
    {
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_new(&rs, shape, 2);
        int64_t bad_shape[] = { 4, 2 };
        VmTensor* v = vm_tensor_reshape(&rs, t, bad_shape, 2);
        assert(v == NULL);
        printf("  reshape-bad: OK\n");
    }

    /* --- transpose --- */
    {
        double data[] = { 1, 2, 3, 4, 5, 6 };
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* tr = vm_tensor_transpose(&rs, t);
        assert(tr != NULL);
        assert(tr->shape[0] == 3 && tr->shape[1] == 2);
        assert(tr->strides[0] == 1 && tr->strides[1] == 3);
        assert(tr->data == t->data);
        assert(tr->owns_data == 0);

        /* tr[0,0]=1, tr[0,1]=4, tr[1,0]=2, tr[2,1]=6 */
        int64_t i00[] = { 0, 0 }; assert(vm_tensor_ref(tr, i00, 2) == 1.0);
        int64_t i01[] = { 0, 1 }; assert(vm_tensor_ref(tr, i01, 2) == 4.0);
        int64_t i10[] = { 1, 0 }; assert(vm_tensor_ref(tr, i10, 2) == 2.0);
        int64_t i21[] = { 2, 1 }; assert(vm_tensor_ref(tr, i21, 2) == 6.0);

        printf("  transpose:   OK\n");
    }

    /* --- flatten --- */
    {
        int64_t shape[] = { 2, 3 };
        double data[] = { 1, 2, 3, 4, 5, 6 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* f = vm_tensor_flatten(&rs, t);
        assert(f != NULL);
        assert(f->n_dims == 1 && f->shape[0] == 6);
        assert(f->data == t->data);
        printf("  flatten:     OK\n");
    }

    /* --- copy (contiguous) --- */
    {
        double data[] = { 1, 2, 3, 4 };
        int64_t shape[] = { 2, 2 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* c = vm_tensor_copy(&rs, t);
        assert(c != NULL);
        assert(c->data != t->data); /* different buffer */
        assert(c->owns_data == 1);
        for (int64_t i = 0; i < 4; i++) assert(c->data[i] == t->data[i]);

        /* Modifying copy does not affect original */
        c->data[0] = 999.0;
        assert(t->data[0] == 1.0);
        printf("  copy:        OK\n");
    }

    /* --- copy (non-contiguous / transposed) --- */
    {
        double data[] = { 1, 2, 3, 4, 5, 6 };
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* tr = vm_tensor_transpose(&rs, t);
        VmTensor* c = vm_tensor_copy(&rs, tr);
        assert(c != NULL);
        assert(c->data != tr->data);
        /* c is now contiguous 3x2: [[1,4],[2,5],[3,6]] */
        assert(c->data[0] == 1.0);
        assert(c->data[1] == 4.0);
        assert(c->data[2] == 2.0);
        assert(c->data[3] == 5.0);
        printf("  copy-trans:  OK\n");
    }

    /* --- linspace --- */
    {
        VmTensor* t = vm_tensor_linspace(&rs, 0.0, 1.0, 5);
        assert(t != NULL);
        assert(t->total == 5);
        assert(fabs(t->data[0] - 0.0) < 1e-12);
        assert(fabs(t->data[2] - 0.5) < 1e-12);
        assert(fabs(t->data[4] - 1.0) < 1e-12);
        printf("  linspace:    OK\n");
    }

    /* --- eye --- */
    {
        VmTensor* t = vm_tensor_eye(&rs, 3);
        assert(t != NULL);
        assert(t->shape[0] == 3 && t->shape[1] == 3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                assert(t->data[i * 3 + j] == (i == j ? 1.0 : 0.0));
        printf("  eye(3):      OK\n");
    }

    /* --- scalar --- */
    {
        VmTensor* t = vm_tensor_scalar(&rs, 42.0);
        assert(t != NULL);
        assert(t->total == 1 && t->data[0] == 42.0);
        printf("  scalar(42):  OK\n");
    }

    /* --- shape/size/ndim --- */
    {
        int64_t shape[] = { 2, 3, 4 };
        VmTensor* t = vm_tensor_new(&rs, shape, 3);
        assert(vm_tensor_size(t) == 24);
        assert(vm_tensor_ndim(t) == 3);
        VmTensor* s = vm_tensor_get_shape(&rs, t);
        assert(s && s->total == 3);
        assert(s->data[0] == 2.0 && s->data[1] == 3.0 && s->data[2] == 4.0);
        printf("  shape/size:  OK\n");
    }

    /* --- slice --- */
    {
        double data[] = { 1, 2, 3, 4, 5, 6 };
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* row0 = vm_tensor_slice(&rs, t, 0);
        assert(row0 != NULL);
        assert(row0->n_dims == 1 && row0->shape[0] == 3);
        assert(row0->data[0] == 1.0 && row0->data[2] == 3.0);
        VmTensor* row1 = vm_tensor_slice(&rs, t, 1);
        assert(row1->data[0] == 4.0 && row1->data[2] == 6.0);
        printf("  slice:       OK\n");
    }

    /* --- edge: 1D tensor --- */
    {
        int64_t shape[] = { 5 };
        VmTensor* t = vm_tensor_new(&rs, shape, 1);
        assert(t && t->total == 5 && t->strides[0] == 1);
        printf("  1D edge:     OK\n");
    }

    /* --- edge: high-dimensional --- */
    {
        int64_t shape[] = { 2, 3, 4, 5, 6, 7, 8, 1 };
        VmTensor* t = vm_tensor_new(&rs, shape, 8);
        assert(t && t->total == 2*3*4*5*6*7*8*1);
        assert(t->n_dims == 8);
        printf("  8D tensor:   OK\n");
    }

    /* --- edge: invalid dims --- */
    {
        int64_t shape[] = { 2 };
        assert(vm_tensor_new(&rs, shape, 0) == NULL);
        assert(vm_tensor_new(&rs, shape, 9) == NULL);
        printf("  bad dims:    OK\n");
    }

    vm_region_stack_destroy(&rs);
    printf("vm_tensor: ALL TESTS PASSED\n");
    return 0;
}
#endif

#endif /* VM_TENSOR_C_INCLUDED */
