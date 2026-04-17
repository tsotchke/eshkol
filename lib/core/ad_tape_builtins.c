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
int ad_const(AdTape* tape, double value);
int ad_var(AdTape* tape, double value);
int ad_add(AdTape* tape, int left, int right);
int ad_sub(AdTape* tape, int left, int right);
int ad_mul(AdTape* tape, int left, int right);
int ad_div(AdTape* tape, int left, int right);
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

static ad_tagged_t make_int(int64_t v) {
    ad_tagged_t r = {AD_TYPE_INT64, 0, 0, 0, 0};
    memcpy(&r.data, &v, sizeof(int64_t));
    return r;
}

static ad_tagged_t make_double(double v) {
    ad_tagged_t r = {AD_TYPE_DOUBLE, 0, 0, 0, 0};
    memcpy(&r.data, &v, sizeof(double));
    return r;
}

static ad_tagged_t make_heap(void* p) {
    ad_tagged_t r = {AD_TYPE_HEAP_PTR, 0, 0, 0, 0};
    r.data = (uint64_t)p;
    return r;
}

static ad_tagged_t make_null(void) {
    ad_tagged_t r = {AD_TYPE_NULL, 0, 0, 0, 0};
    return r;
}

static AdTape* extract_tape(const ad_tagged_t* v) {
    if (v->type == AD_TYPE_HEAP_PTR && v->data != 0)
        return (AdTape*)(uintptr_t)v->data;
    return NULL;
}

/* All-pointer sret wrappers for LLVM codegen */

void eshkol_ad_tape_new_sret(ad_tagged_t* out) {
    void* arena = get_global_arena();
    AdTape* tape = ad_tape_new_from_arena(arena);
    *out = tape ? make_heap(tape) : make_null();
}

void eshkol_ad_const_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* val_tv) {
    AdTape* tape = extract_tape(tape_tv);
    if (!tape) { *out = make_int(-1); return; }
    double val;
    if (val_tv->type == AD_TYPE_DOUBLE) memcpy(&val, &val_tv->data, sizeof(double));
    else val = (double)(int64_t)val_tv->data;
    *out = make_int(ad_const(tape, val));
}

void eshkol_ad_var_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* val_tv) {
    AdTape* tape = extract_tape(tape_tv);
    if (!tape) { *out = make_int(-1); return; }
    double val;
    if (val_tv->type == AD_TYPE_DOUBLE) memcpy(&val, &val_tv->data, sizeof(double));
    else val = (double)(int64_t)val_tv->data;
    *out = make_int(ad_var(tape, val));
}

/* Binary ops: (tape, left_node, right_node) → node_index */
#define AD_BINARY_SRET(name, func) \
void name(ad_tagged_t* out, const ad_tagged_t* tape_tv, \
          const ad_tagged_t* left_tv, const ad_tagged_t* right_tv) { \
    AdTape* tape = extract_tape(tape_tv); \
    if (!tape) { *out = make_int(-1); return; } \
    *out = make_int(func(tape, (int)(int64_t)left_tv->data, (int)(int64_t)right_tv->data)); \
}

AD_BINARY_SRET(eshkol_ad_add_sret, ad_add)
AD_BINARY_SRET(eshkol_ad_sub_sret, ad_sub)
AD_BINARY_SRET(eshkol_ad_mul_sret, ad_mul)
AD_BINARY_SRET(eshkol_ad_div_sret, ad_div)

/* Unary ops: (tape, node) → node_index */
#define AD_UNARY_SRET(name, func) \
void name(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* node_tv) { \
    AdTape* tape = extract_tape(tape_tv); \
    if (!tape) { *out = make_int(-1); return; } \
    *out = make_int(func(tape, (int)(int64_t)node_tv->data)); \
}

AD_UNARY_SRET(eshkol_ad_sin_sret, ad_sin)
AD_UNARY_SRET(eshkol_ad_cos_sret, ad_cos)
AD_UNARY_SRET(eshkol_ad_exp_sret, ad_exp)
AD_UNARY_SRET(eshkol_ad_log_sret, ad_log)
AD_UNARY_SRET(eshkol_ad_sqrt_sret, ad_sqrt)
AD_UNARY_SRET(eshkol_ad_neg_sret, ad_neg)
AD_UNARY_SRET(eshkol_ad_abs_sret, ad_abs)
AD_UNARY_SRET(eshkol_ad_relu_sret, ad_relu)
AD_UNARY_SRET(eshkol_ad_sigmoid_sret, ad_sigmoid)
AD_UNARY_SRET(eshkol_ad_tanh_sret, ad_tanh)

void eshkol_ad_backward_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* node_tv) {
    AdTape* tape = extract_tape(tape_tv);
    if (!tape) { *out = make_null(); return; }
    ad_backward(tape, (int)(int64_t)node_tv->data);
    *out = make_null();
}

void eshkol_ad_gradient_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* node_tv) {
    AdTape* tape = extract_tape(tape_tv);
    if (!tape) { *out = make_double(0.0); return; }
    *out = make_double(ad_get_gradient(tape, (int)(int64_t)node_tv->data));
}

void eshkol_ad_node_value_sret(ad_tagged_t* out, const ad_tagged_t* tape_tv, const ad_tagged_t* node_tv) {
    AdTape* tape = extract_tape(tape_tv);
    if (!tape) { *out = make_double(0.0); return; }
    *out = make_double(ad_get_value(tape, (int)(int64_t)node_tv->data));
}
