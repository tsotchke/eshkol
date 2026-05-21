/**
 * @file eshkol_ffi.h
 * @brief Stable C FFI for embedding Eshkol in other languages.
 *
 * This header provides a clean, stable C API for:
 *   - Initializing and shutting down the Eshkol runtime
 *   - Evaluating Eshkol expressions via in-process JIT
 *   - Constructing and inspecting Eshkol values (numbers, strings, lists, tensors)
 *   - Error handling
 *
 * All functions use `extern "C"` linkage for cross-language compatibility.
 * All pointer-returning functions return NULL on failure.
 * All int-returning functions return 0 on success, non-zero on error.
 *
 * Thread safety: the runtime is single-threaded. Call all FFI functions
 * from the same thread that called eshkol_ffi_init().
 *
 * Memory: values allocated through the FFI use the global arena.
 * Call eshkol_ffi_shutdown() to release all memory.
 *
 * Prerequisites: link against libeshkol-static.a and LLVM libraries.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef ESHKOL_FFI_H
#define ESHKOL_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Opaque Types
 * ============================================================================ */

/** Opaque handle to an Eshkol JIT context (manages compilation + execution). */
typedef struct eshkol_ffi_context eshkol_ffi_context_t;

/** Tagged value — the universal Eshkol value type (16 bytes).
 *  Layout: {type:8, flags:8, reserved:16, data:64}
 *  Use the accessor functions below to construct and inspect values.
 *
 *  This is ABI-compatible with the runtime's eshkol_tagged_value_t. */
typedef struct {
    uint8_t  type;
    uint8_t  flags;
    uint16_t reserved;
    union {
        int64_t  int_val;
        double   double_val;
        uint64_t ptr_val;
        uint64_t raw_val;
    } data;
} eshkol_ffi_value_t;

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/** Initialize the Eshkol runtime with JIT compilation support.
 *  Creates an in-process JIT context for evaluating expressions.
 *  @return Context handle, or NULL on failure. */
eshkol_ffi_context_t* eshkol_ffi_init(void);

/** Shut down the Eshkol runtime and release all resources.
 *  @param ctx Context from eshkol_ffi_init(). Safe to pass NULL. */
void eshkol_ffi_shutdown(eshkol_ffi_context_t* ctx);

/* ============================================================================
 * Evaluation
 * ============================================================================ */

/** Parse and evaluate an Eshkol expression string via in-process JIT.
 *  State (definitions, variables) persists across calls within the same context.
 *
 *  @param ctx     Context from eshkol_ffi_init()
 *  @param source  Eshkol source code (null-terminated)
 *  @param result  Output: the result value (may be NULL if not needed)
 *  @return 0 on success, non-zero on error. */
int eshkol_ffi_eval(eshkol_ffi_context_t* ctx,
                    const char* source,
                    eshkol_ffi_value_t* result);

/** Parse and evaluate, returning the result as a double.
 *  Convenience for numeric computations.
 *
 *  @param ctx     Context from eshkol_ffi_init()
 *  @param source  Eshkol source code (null-terminated)
 *  @param result  Output: the numeric result
 *  @return 0 on success, non-zero on error. */
int eshkol_ffi_eval_double(eshkol_ffi_context_t* ctx,
                           const char* source,
                           double* result);

/** Load and evaluate a file.
 *  @param ctx   Context
 *  @param path  Path to .esk file
 *  @return 0 on success, non-zero on error. */
int eshkol_ffi_eval_file(eshkol_ffi_context_t* ctx, const char* path);

/* ============================================================================
 * Value Construction
 * ============================================================================ */

/** Construct an integer value. */
eshkol_ffi_value_t eshkol_ffi_int64(int64_t value);

/** Construct a floating-point value. */
eshkol_ffi_value_t eshkol_ffi_double(double value);

/** Construct a boolean value. */
eshkol_ffi_value_t eshkol_ffi_bool(int value);

/** Construct the null/nil value. */
eshkol_ffi_value_t eshkol_ffi_null(void);

/** Construct a string value (copies the string into the arena).
 *  @param ctx  Context
 *  @param str  Null-terminated string
 *  @return String value, or null value on failure. */
eshkol_ffi_value_t eshkol_ffi_string(eshkol_ffi_context_t* ctx, const char* str);

/** Construct a cons pair.
 *  @param ctx  Context
 *  @param car  First element
 *  @param cdr  Second element (or null for end of list)
 *  @return Pair value, or null on failure. */
eshkol_ffi_value_t eshkol_ffi_cons(eshkol_ffi_context_t* ctx,
                                    eshkol_ffi_value_t car,
                                    eshkol_ffi_value_t cdr);

/** Construct a list from an array of values.
 *  @param ctx     Context
 *  @param values  Array of values
 *  @param count   Number of values
 *  @return List value (proper list ending with null). */
eshkol_ffi_value_t eshkol_ffi_list(eshkol_ffi_context_t* ctx,
                                    const eshkol_ffi_value_t* values,
                                    size_t count);

/* ============================================================================
 * Value Inspection
 * ============================================================================ */

/** Get the type tag of a value.
 *  @return Type constant (ESHKOL_FFI_TYPE_*). */
int eshkol_ffi_type(eshkol_ffi_value_t value);

/** Extract an int64 from a value. Returns 0 if not an integer. */
int64_t eshkol_ffi_to_int64(eshkol_ffi_value_t value);

/** Extract a double from a value. Converts int64 to double if needed. */
double eshkol_ffi_to_double(eshkol_ffi_value_t value);

/** Extract a boolean from a value. */
int eshkol_ffi_to_bool(eshkol_ffi_value_t value);

/** Extract the C string from a string value.
 *  @return Pointer to string data (owned by arena, do not free), or NULL. */
const char* eshkol_ffi_to_string(eshkol_ffi_value_t value);

/** Check if a value is null/nil. */
int eshkol_ffi_is_null(eshkol_ffi_value_t value);

/** Check if a value is a pair/list. */
int eshkol_ffi_is_pair(eshkol_ffi_value_t value);

/** Get the car (first element) of a pair. */
eshkol_ffi_value_t eshkol_ffi_car(eshkol_ffi_value_t pair);

/** Get the cdr (rest) of a pair. */
eshkol_ffi_value_t eshkol_ffi_cdr(eshkol_ffi_value_t pair);

/** Display a value to stdout (for debugging). */
void eshkol_ffi_display(eshkol_ffi_value_t value);

/* ============================================================================
 * Tensor Operations
 * ============================================================================ */

/** Create a tensor with the given shape, initialised to zero.
 *  The tensor's backing storage lives in the Eshkol global arena —
 *  callers must NOT free the pointer returned by eshkol_ffi_tensor_data()
 *  on it. Lifetime is bounded by the arena (process-wide unless the host
 *  explicitly resets/destroys it). */
eshkol_ffi_value_t eshkol_ffi_tensor_zeros(eshkol_ffi_context_t* ctx,
                                            const int64_t* shape,
                                            int ndims);

/** Create a tensor from existing data (copies data).
 *
 *  @param ctx    FFI context (unused for allocation but kept for symmetry).
 *  @param data   Source pointer. Only READ during this call — the caller
 *                retains ownership and may free/reuse @p data immediately
 *                after the call returns. The returned tensor owns its own
 *                freshly-allocated copy of the bytes.
 *  @param shape  Dimension sizes, length @p ndims. Read-only.
 *  @param ndims  Number of dimensions (1–8).
 *  @return       Tensor value whose storage lives in the global arena
 *                (lifetime managed by the arena, NOT the caller). Returns
 *                the null value on allocation failure or invalid shape. */
eshkol_ffi_value_t eshkol_ffi_tensor_from_data(eshkol_ffi_context_t* ctx,
                                                const double* data,
                                                const int64_t* shape,
                                                int ndims);

/** Get a pointer to the tensor's data (read/write access).
 *  The pointer aliases into arena memory and is valid until the arena is
 *  reset or destroyed. It must NOT be free()'d by the caller. Writes made
 *  through this pointer are visible to subsequent Eshkol code that reads
 *  the same tensor value.
 *  @return Pointer to flat double array, or NULL if not a tensor. */
double* eshkol_ffi_tensor_data(eshkol_ffi_value_t tensor);

/** Get the total number of elements in a tensor. */
int64_t eshkol_ffi_tensor_size(eshkol_ffi_value_t tensor);

/** Get the number of dimensions of a tensor. */
int eshkol_ffi_tensor_ndims(eshkol_ffi_value_t tensor);

/** Copy the tensor's per-dimension sizes into @p out_shape.
 *  Writes at most @p max_ndims entries and returns the number actually
 *  written (== tensor's ndims, or 0 if @p tensor is not a tensor or
 *  @p out_shape is NULL). Use together with eshkol_ffi_tensor_ndims
 *  to size the output buffer. */
int eshkol_ffi_tensor_shape(eshkol_ffi_value_t tensor,
                             int64_t* out_shape,
                             int max_ndims);

/* ============================================================================
 * Error Handling
 * ============================================================================ */

/** Get the last error message (thread-local).
 *  @return Error message string, or NULL if no error. Do not free. */
const char* eshkol_ffi_last_error(void);

/** Clear the last error. */
void eshkol_ffi_clear_error(void);

/* ============================================================================
 * Type Constants — match runtime eshkol_value_type_t exactly
 * ============================================================================ */

#define ESHKOL_FFI_TYPE_NULL     0   /* Empty/null value */
#define ESHKOL_FFI_TYPE_INT64    1   /* 64-bit signed integer */
#define ESHKOL_FFI_TYPE_DOUBLE   2   /* Double-precision float */
#define ESHKOL_FFI_TYPE_BOOL     3   /* Boolean (#t/#f) */
#define ESHKOL_FFI_TYPE_CHAR     4   /* Unicode character */
#define ESHKOL_FFI_TYPE_SYMBOL   5   /* Interned symbol */
#define ESHKOL_FFI_TYPE_HEAP_PTR 8   /* Heap data: cons, string, vector, tensor */
#define ESHKOL_FFI_TYPE_CALLABLE 9   /* Callables: closure, lambda, primitive */

/*
 * FFI-side subtype tags.
 *
 * IMPORTANT: these constants are FFI-only markers stored in the `flags`
 * byte of `eshkol_ffi_value_t`. They are NOT the canonical runtime heap
 * subtypes (defined as `heap_subtype_t` in `eshkol.h`, where
 * HEAP_SUBTYPE_CONS=0, _STRING=1, _VECTOR=2, _TENSOR=3, and the byte lives
 * in the object header at `ptr - sizeof(eshkol_object_header_t)`).
 *
 * Cross-FFI consumers should classify heap values via these flag markers;
 * code inside the Eshkol runtime should read the canonical subtype from
 * the object header using `ESHKOL_GET_SUBTYPE`. Do not assume the numeric
 * values below agree with `heap_subtype_t` — they were chosen
 * independently and are stable for the FFI ABI.
 */
#define ESHKOL_FFI_SUBTYPE_STRING  0x01
#define ESHKOL_FFI_SUBTYPE_PAIR    0x02
#define ESHKOL_FFI_SUBTYPE_TENSOR  0x03
#define ESHKOL_FFI_SUBTYPE_VECTOR  0x04

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_FFI_H */
