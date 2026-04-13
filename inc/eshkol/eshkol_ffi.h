/**
 * @file eshkol_ffi.h
 * @brief Stable C FFI for embedding Eshkol in other languages.
 *
 * This header provides a clean, stable C API for:
 *   - Initializing and shutting down the Eshkol runtime
 *   - Parsing Eshkol source code from strings
 *   - Compiling and JIT-executing Eshkol expressions
 *   - Constructing and inspecting Eshkol values (numbers, strings, lists, tensors)
 *   - Calling Eshkol closures from C
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
 *  Layout: {type:8, flags:8, reserved:16, padding:32, data:64}
 *  Use the accessor functions below to construct and inspect values. */
typedef struct {
    uint8_t  type;
    uint8_t  flags;
    uint16_t reserved;
    uint32_t padding;
    uint64_t data;
} eshkol_ffi_value_t;

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/** Initialize the Eshkol runtime. Must be called before any other FFI function.
 *  @return Context handle, or NULL on failure. */
eshkol_ffi_context_t* eshkol_ffi_init(void);

/** Shut down the Eshkol runtime and release all resources.
 *  @param ctx Context from eshkol_ffi_init(). Safe to pass NULL. */
void eshkol_ffi_shutdown(eshkol_ffi_context_t* ctx);

/* ============================================================================
 * Evaluation
 * ============================================================================ */

/** Parse and evaluate an Eshkol expression string.
 *  This is the simplest way to execute Eshkol code from C.
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
 *  @return Type constant: 0=null, 1=int64, 2=double, 3=bool, 4=heap_ptr, etc. */
int eshkol_ffi_type(eshkol_ffi_value_t value);

/** Extract an int64 from a value. Returns 0 if not an integer. */
int64_t eshkol_ffi_to_int64(eshkol_ffi_value_t value);

/** Extract a double from a value. Converts int64 to double if needed. */
double eshkol_ffi_to_double(eshkol_ffi_value_t value);

/** Extract a boolean from a value. */
int eshkol_ffi_to_bool(eshkol_ffi_value_t value);

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

/** Create a tensor with the given shape, initialized to zero.
 *  @param ctx    Context
 *  @param shape  Array of dimension sizes
 *  @param ndims  Number of dimensions
 *  @return Tensor value, or null on failure. */
eshkol_ffi_value_t eshkol_ffi_tensor_zeros(eshkol_ffi_context_t* ctx,
                                            const int64_t* shape,
                                            int ndims);

/** Create a tensor from existing data (copies data).
 *  @param ctx    Context
 *  @param data   Flat array of doubles (row-major)
 *  @param shape  Array of dimension sizes
 *  @param ndims  Number of dimensions
 *  @return Tensor value, or null on failure. */
eshkol_ffi_value_t eshkol_ffi_tensor_from_data(eshkol_ffi_context_t* ctx,
                                                const double* data,
                                                const int64_t* shape,
                                                int ndims);

/** Get a pointer to the tensor's data (read/write access).
 *  @param tensor  Tensor value
 *  @return Pointer to flat double array, or NULL if not a tensor. */
double* eshkol_ffi_tensor_data(eshkol_ffi_value_t tensor);

/** Get the total number of elements in a tensor.
 *  @return Element count, or 0 if not a tensor. */
int64_t eshkol_ffi_tensor_size(eshkol_ffi_value_t tensor);

/** Get the number of dimensions of a tensor.
 *  @return Number of dimensions, or 0 if not a tensor. */
int eshkol_ffi_tensor_ndims(eshkol_ffi_value_t tensor);

/* ============================================================================
 * Error Handling
 * ============================================================================ */

/** Get the last error message (thread-local).
 *  @return Error message string, or NULL if no error. Do not free. */
const char* eshkol_ffi_last_error(void);

/** Clear the last error. */
void eshkol_ffi_clear_error(void);

/* ============================================================================
 * Type Constants
 * ============================================================================ */

#define ESHKOL_FFI_TYPE_NULL     0
#define ESHKOL_FFI_TYPE_INT64    1
#define ESHKOL_FFI_TYPE_DOUBLE   2
#define ESHKOL_FFI_TYPE_BOOL     3
#define ESHKOL_FFI_TYPE_HEAP_PTR 4
#define ESHKOL_FFI_TYPE_CALLABLE 5

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_FFI_H */
