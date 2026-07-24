/**
 * @file ad_tape_builtins.c
 * @brief Reverse-mode AD tape operations for LLVM-compiled code.
 *
 * Wraps the VM's AdTape functions (vm_autodiff.c) with the sret
 * calling convention for LLVM codegen compatibility.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* Tagged value struct matching LLVM IR layout */
typedef struct {
    uint8_t type;
    uint8_t flags;
    uint16_t reserved;
    uint32_t padding;
    uint64_t data;
} ad_tagged_t;

#define AD_TYPE_NULL  0
#define AD_TYPE_INT64 1
#define AD_TYPE_DOUBLE 2
#define AD_TYPE_HEAP_PTR 8

/* Forward declarations from vm_autodiff.c */
typedef struct AdTape AdTape;
extern void* get_global_arena(void);

/* These functions are defined in vm_autodiff.c but are static in the VM unity build.
 * For LLVM access, we need extern versions. They're re-declared here and the
 * linker resolves them from the static library. */
AdTape* ad_tape_new_from_arena(void* arena);
/* Bug I (2026-04-20): per-tape owned sub-arena. */
AdTape* ad_tape_new_owned(void);
void ad_tape_release_owned(AdTape* tape);
int ad_const(AdTape* tape, double value);
int ad_var(AdTape* tape, double value);
int ad_add(AdTape* tape, int left, int right);
int ad_sub(AdTape* tape, int left, int right);
int ad_mul(AdTape* tape, int left, int right);
int ad_div(AdTape* tape, int left, int right);
int ad_pow(AdTape* tape, int base, int exponent);
int ad_tape_length(const AdTape* tape);
int ad_sin(AdTape* tape, int node);
int ad_cos(AdTape* tape, int node);
int ad_exp(AdTape* tape, int node);
int ad_log(AdTape* tape, int node);
int ad_sqrt(AdTape* tape, int node);
int ad_neg(AdTape* tape, int node);
int ad_abs(AdTape* tape, int node);
int ad_relu(AdTape* tape, int node);
int ad_sigmoid(AdTape* tape, int node);
int ad_tanh(AdTape* tape, int node);
void ad_backward(AdTape* tape, int output);
double ad_get_gradient(AdTape* tape, int node);
double ad_get_value(AdTape* tape, int node);

/* ── Sret wrappers ── */

/** Wrap an int64 in a tagged value. */
static ad_tagged_t make_int(int64_t v) {
    ad_tagged_t r = {AD_TYPE_INT64, 0, 0, 0, 0};
    memcpy(&r.data, &v, sizeof(int64_t));
    return r;
}

/** Wrap a double in a tagged value. */
static ad_tagged_t make_double(double v) {
    ad_tagged_t r = {AD_TYPE_DOUBLE, 0, 0, 0, 0};
    memcpy(&r.data, &v, sizeof(double));
    return r;
}

/** Wrap a raw pointer in a tagged heap-pointer value. */
static ad_tagged_t make_heap(void* p) {
    ad_tagged_t r = {AD_TYPE_HEAP_PTR, 0, 0, 0, 0};
    r.data = (uint64_t)p;
    return r;
}

/** Construct a tagged null value. */
static ad_tagged_t make_null(void) {
    ad_tagged_t r = {AD_TYPE_NULL, 0, 0, 0, 0};
    return r;
}

/** Unwrap a tagged heap-pointer value into an AdTape*, or NULL if `v`
 *  is not a (non-zero) heap pointer. */
static AdTape* extract_tape(const ad_tagged_t* v) {
    if (v->type == AD_TYPE_HEAP_PTR && v->data != 0)
        return (AdTape*)(uintptr_t)v->data;
    return NULL;
}

/* All-pointer sret wrappers for LLVM codegen */

/** Sret wrapper for (ad-tape-new): allocate a new owned AD tape and
 *  return it as a tagged heap pointer (tagged null on allocation failure). */
void eshkol_ad_tape_new_sret(ad_tagged_t* out) {
    /* Bug I (2026-04-20): route Scheme ad-tape-new to the owned-arena
     * factory so iterative fit loops can reclaim memory via
     * ad-tape-release. The legacy ad_tape_new_from_arena remains for
     * VM-internal / test callers where arena lifecycle is external. */
    AdTape* tape = ad_tape_new_owned();
    *out = tape ? make_heap(tape) : make_null();
}

/* Release an owned tape — destroys its sub-arena. After release, the
 * tape pointer is invalid; any subsequent ad-* op on it is UB from
 * the Scheme side. Guarded by a magic sentinel so (a) double-release
 * is safe and (b) passing a legacy non-owned tape silently no-ops. */
void eshkol_ad_tape_release_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv) {
    AdTape* tape = extract_tape(tape_tv);
    ad_tape_release_owned(tape);
    *out = make_null();
}

/** Sret wrapper for (ad-const tape value): record a constant leaf node
 *  on the tape and return its node index (-1 if `tape_tv` is not a
 *  valid tape). Accepts either an int or double tagged value. */
void eshkol_ad_const_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* val_tv) {
    AdTape* tape = extract_tape(tape_tv);
    if (!tape) { *out = make_int(-1); return; }
    double val;
    if (val_tv->type == AD_TYPE_DOUBLE) memcpy(&val, &val_tv->data, sizeof(double));
    else val = (double)(int64_t)val_tv->data;
    *out = make_int(ad_const(tape, val));
}

/** Sret wrapper for (ad-var tape value): record a differentiable
 *  variable leaf node on the tape and return its node index (-1 if
 *  `tape_tv` is not a valid tape). Accepts either an int or double
 *  tagged value. */
void eshkol_ad_var_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* val_tv) {
    AdTape* tape = extract_tape(tape_tv);
    if (!tape) { *out = make_int(-1); return; }
    double val;
    if (val_tv->type == AD_TYPE_DOUBLE) memcpy(&val, &val_tv->data, sizeof(double));
    else val = (double)(int64_t)val_tv->data;
    *out = make_int(ad_var(tape, val));
}

/* Binary ops: (tape, left_node, right_node) → node_index */
/** Defines an sret wrapper function `name` that records a binary AD
 *  op (`func`) on the tape linking `left`/`right` node indices, and
 *  returns the new node's index (-1 if `tape_tv` is not a valid tape). */
#define AD_BINARY_SRET(name, func) \
void name(ad_tagged_t* out, const ad_tagged_t* tape_tv, \
          const ad_tagged_t* left_tv, const ad_tagged_t* right_tv) { \
    AdTape* tape = extract_tape(tape_tv); \
    if (!tape) { *out = make_int(-1); return; } \
    *out = make_int(func(tape, (int)(int64_t)left_tv->data, (int)(int64_t)right_tv->data)); \
}

/** Sret wrapper for (ad-add tape left right). */
AD_BINARY_SRET(eshkol_ad_add_sret, ad_add)
/** Sret wrapper for (ad-sub tape left right). */
AD_BINARY_SRET(eshkol_ad_sub_sret, ad_sub)
/** Sret wrapper for (ad-mul tape left right). */
AD_BINARY_SRET(eshkol_ad_mul_sret, ad_mul)
/** Sret wrapper for (ad-div tape left right). */
AD_BINARY_SRET(eshkol_ad_div_sret, ad_div)
/** Sret wrapper for (ad-pow tape base-node exponent-node): records
 *  base^exponent with ordinary pow() forward value and the reverse
 *  derivatives d/dbase = exp*base^(exp-1), d/dexp = value*ln(base). */
AD_BINARY_SRET(eshkol_ad_pow_sret, ad_pow)

/* Unary ops: (tape, node) → node_index */
/** Defines an sret wrapper function `name` that records a unary AD op
 *  (`func`) on the tape for the given node index, and returns the new
 *  node's index (-1 if `tape_tv` is not a valid tape). */
#define AD_UNARY_SRET(name, func) \
void name(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* node_tv) { \
    AdTape* tape = extract_tape(tape_tv); \
    if (!tape) { *out = make_int(-1); return; } \
    *out = make_int(func(tape, (int)(int64_t)node_tv->data)); \
}

/** Sret wrapper for (ad-sin tape node). */
AD_UNARY_SRET(eshkol_ad_sin_sret, ad_sin)
/** Sret wrapper for (ad-cos tape node). */
AD_UNARY_SRET(eshkol_ad_cos_sret, ad_cos)
/** Sret wrapper for (ad-exp tape node). */
AD_UNARY_SRET(eshkol_ad_exp_sret, ad_exp)
/** Sret wrapper for (ad-log tape node). */
AD_UNARY_SRET(eshkol_ad_log_sret, ad_log)
/** Sret wrapper for (ad-sqrt tape node). */
AD_UNARY_SRET(eshkol_ad_sqrt_sret, ad_sqrt)
/** Sret wrapper for (ad-neg tape node). */
AD_UNARY_SRET(eshkol_ad_neg_sret, ad_neg)
/** Sret wrapper for (ad-abs tape node). */
AD_UNARY_SRET(eshkol_ad_abs_sret, ad_abs)
/** Sret wrapper for (ad-relu tape node). */
AD_UNARY_SRET(eshkol_ad_relu_sret, ad_relu)
/** Sret wrapper for (ad-sigmoid tape node). */
AD_UNARY_SRET(eshkol_ad_sigmoid_sret, ad_sigmoid)
/** Sret wrapper for (ad-tanh tape node). */
AD_UNARY_SRET(eshkol_ad_tanh_sret, ad_tanh)

/** Sret wrapper for (ad-backward tape output): run reverse-mode
 *  backpropagation from `output` node to populate gradients on the
 *  tape. Returns tagged null; no-op if `tape_tv` is not a valid tape. */
void eshkol_ad_backward_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* node_tv) {
    AdTape* tape = extract_tape(tape_tv);
    if (!tape) { *out = make_null(); return; }
    ad_backward(tape, (int)(int64_t)node_tv->data);
    *out = make_null();
}

/** Sret wrapper for (ad-gradient tape node): return the accumulated
 *  gradient at `node` as a tagged double (0.0 if `tape_tv` is not a
 *  valid tape). Requires ad-backward to have been run first. */
void eshkol_ad_gradient_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* node_tv) {
    AdTape* tape = extract_tape(tape_tv);
    if (!tape) { *out = make_double(0.0); return; }
    *out = make_double(ad_get_gradient(tape, (int)(int64_t)node_tv->data));
}

/** Sret wrapper for (ad-node-value tape node): return the forward
 *  (primal) value stored at `node` as a tagged double (0.0 if
 *  `tape_tv` is not a valid tape). */
void eshkol_ad_tape_length_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv) {
    AdTape* tape = extract_tape(tape_tv);
    *out = make_int(tape ? ad_tape_length(tape) : 0);
}

void eshkol_ad_node_value_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* node_tv) {
    AdTape* tape = extract_tape(tape_tv);
    if (!tape) { *out = make_double(0.0); return; }
    *out = make_double(ad_get_value(tape, (int)(int64_t)node_tv->data));
}
