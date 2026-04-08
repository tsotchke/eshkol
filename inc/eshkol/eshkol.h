/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#ifndef ESHKOL_ESHKOL_H
#define ESHKOL_ESHKOL_H

// Version information
#define ESHKOL_VERSION_MAJOR 1
#define ESHKOL_VERSION_MINOR 1
#define ESHKOL_VERSION_PATCH 0
#define ESHKOL_VERSION_STRING "1.1.0-accelerate"

#include <stdint.h>
#include <stdbool.h>

// ═══════════════════════════════════════════════════════════════════════════
// CROSS-PLATFORM STATIC ASSERTION MACRO
// _Static_assert is C11, static_assert is C++11. GCC in C++ mode doesn't
// recognize _Static_assert. This macro provides a unified interface.
// ═══════════════════════════════════════════════════════════════════════════
#ifdef __cplusplus
#define ESHKOL_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ESHKOL_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

#ifdef __cplusplus

#include <fstream>

extern "C" {
#endif

typedef enum {
    ESHKOL_INVALID,
    ESHKOL_UNTYPED,
    ESHKOL_UINT8,
    ESHKOL_UINT16,
    ESHKOL_UINT32,
    ESHKOL_UINT64,
    ESHKOL_INT8,
    ESHKOL_INT16,
    ESHKOL_INT32,
    ESHKOL_INT64,
    ESHKOL_DOUBLE,
    ESHKOL_STRING,
    ESHKOL_FUNC,
    ESHKOL_VAR,
    ESHKOL_OP,
    ESHKOL_CONS,
    ESHKOL_NULL,
    ESHKOL_TENSOR,
    ESHKOL_CHAR,
    ESHKOL_BOOL,
    ESHKOL_BIGNUM_LITERAL,  // Integer literal too large for int64 (stored as string)
    ESHKOL_SYMBOL           // Symbol literal (stored as string)
} eshkol_type_t;

// ═══════════════════════════════════════════════════════════════════════════
// VALUE TYPE TAGS - 8-bit type field with subtype headers for pointers
// ═══════════════════════════════════════════════════════════════════════════
typedef enum {
    // ═══════════════════════════════════════════════════════════════════════
    // IMMEDIATE VALUES (0-7) - data stored directly in tagged value
    // No heap allocation, no header needed
    // ═══════════════════════════════════════════════════════════════════════
    ESHKOL_VALUE_NULL        = 0,   // Empty/null value
    ESHKOL_VALUE_INT64       = 1,   // 64-bit signed integer
    ESHKOL_VALUE_DOUBLE      = 2,   // Double-precision float
    ESHKOL_VALUE_BOOL        = 3,   // Boolean (#t/#f)
    ESHKOL_VALUE_CHAR        = 4,   // Unicode character
    ESHKOL_VALUE_SYMBOL      = 5,   // Interned symbol
    ESHKOL_VALUE_DUAL_NUMBER = 6,   // Forward-mode AD dual number
    ESHKOL_VALUE_COMPLEX     = 7,   // Complex number (real + imaginary)

    // ═══════════════════════════════════════════════════════════════════════
    // CONSOLIDATED POINTER TYPES (8-9) - subtype in object header
    // All heap-allocated objects have eshkol_object_header_t prefix
    // ═══════════════════════════════════════════════════════════════════════
    ESHKOL_VALUE_HEAP_PTR    = 8,   // Heap data: cons, string, vector, tensor, hash, exception
    ESHKOL_VALUE_CALLABLE    = 9,   // Callables: closure, lambda-sexpr, ad-node

    // Neuro-symbolic consciousness engine types
    ESHKOL_VALUE_LOGIC_VAR   = 10,  // Logic variable ?x (data = var_id : int64)

    // ═══════════════════════════════════════════════════════════════════════
    // MULTIMEDIA TYPES (16-19) - linear resources with lifecycle management
    // ═══════════════════════════════════════════════════════════════════════
    ESHKOL_VALUE_HANDLE      = 16,  // Managed resource handles
    ESHKOL_VALUE_BUFFER      = 17,  // Typed data buffers
    ESHKOL_VALUE_STREAM      = 18,  // Lazy data streams
    ESHKOL_VALUE_EVENT       = 19,  // Input/system events

    // ═══════════════════════════════════════════════════════════════════════
    // LEGACY TYPES (DEPRECATED) - For display system backward compatibility only
    // M1 Migration is COMPLETE. New code MUST use consolidated types:
    //   - HEAP_PTR (8) + subtype for: cons, string, vector, tensor, hash, exception
    //   - CALLABLE (9) + subtype for: closure, lambda-sexpr, ad-node
    // These constants are retained only for the display system to render old data.
    // DO NOT use these in new code - they will be removed in a future version.
    // ═══════════════════════════════════════════════════════════════════════
    ESHKOL_VALUE_CONS_PTR    = 32,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_CONS
    ESHKOL_VALUE_STRING_PTR  = 33,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_STRING
    ESHKOL_VALUE_VECTOR_PTR  = 34,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_VECTOR
    ESHKOL_VALUE_TENSOR_PTR  = 35,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_TENSOR
    ESHKOL_VALUE_HASH_PTR    = 36,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_HASH
    ESHKOL_VALUE_EXCEPTION   = 37,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_EXCEPTION
    ESHKOL_VALUE_CLOSURE_PTR = 38,  // DEPRECATED: use CALLABLE + CALLABLE_SUBTYPE_CLOSURE
    ESHKOL_VALUE_LAMBDA_SEXPR = 39, // DEPRECATED: use CALLABLE + CALLABLE_SUBTYPE_LAMBDA_SEXPR
    ESHKOL_VALUE_AD_NODE_PTR = 40,  // DEPRECATED: use CALLABLE + CALLABLE_SUBTYPE_AD_NODE
} eshkol_value_type_t;

// Type flags for Scheme exactness tracking
#define ESHKOL_VALUE_EXACT_FLAG   0x10
#define ESHKOL_VALUE_INEXACT_FLAG 0x20

// Port type flags (OR'd into type byte with ESHKOL_VALUE_HEAP_PTR)
#define ESHKOL_PORT_INPUT_FLAG    0x10   // Input port
#define ESHKOL_PORT_OUTPUT_FLAG   0x40   // Output port
#define ESHKOL_PORT_BINARY_FLAG   0x04   // Binary port (vs textual)
#define ESHKOL_PORT_ANY_FLAG      0x50   // Mask: input (0x10) | output (0x40)

// Combined type constants for common cases
#define ESHKOL_VALUE_EXACT_INT64     (ESHKOL_VALUE_INT64 | ESHKOL_VALUE_EXACT_FLAG)
#define ESHKOL_VALUE_INEXACT_DOUBLE  (ESHKOL_VALUE_DOUBLE | ESHKOL_VALUE_INEXACT_FLAG)

// Tagged data union for cons cell values
typedef union eshkol_tagged_data {
    int64_t int_val;     // Integer value
    double double_val;   // Double-precision floating point value
    uint64_t ptr_val;    // Pointer value (for cons cell pointers)
    uint64_t raw_val;    // Raw 64-bit value for manipulation
} eshkol_tagged_data_t;

// Runtime tagged value representation for ALL Eshkol values
// This struct is used throughout the system to preserve type information
typedef struct eshkol_tagged_value {
    uint8_t type;        // Value type (eshkol_value_type_t)
    uint8_t flags;       // Exactness and other flags
    uint16_t reserved;   // Reserved for future use
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
        uint64_t raw_val;   // For raw manipulation and zero-initialization
    } data;
} eshkol_tagged_value_t;

// Compile-time size validation for tagged values
ESHKOL_STATIC_ASSERT(sizeof(eshkol_tagged_value_t) <= 16,
                     "Tagged value must fit in 16 bytes for efficiency");

// Dual number for forward-mode automatic differentiation
// Stores value and derivative simultaneously for efficient chain rule computation
typedef struct eshkol_dual_number {
    double value;       // f(x) - the function value
    double derivative;  // f'(x) - the derivative value
} eshkol_dual_number_t;

// Compile-time size validation for dual numbers
ESHKOL_STATIC_ASSERT(sizeof(eshkol_dual_number_t) == 16,
                     "Dual number must be 16 bytes for cache efficiency");

// Complex number for signal processing, FFT, and general complex analysis
// Stores real and imaginary components as IEEE 754 double-precision floats
typedef struct eshkol_complex_number {
    double real;        // Real component (ℜ)
    double imag;        // Imaginary component (𝕴)
} eshkol_complex_number_t;

// Compile-time size validation for complex numbers
ESHKOL_STATIC_ASSERT(sizeof(eshkol_complex_number_t) == 16,
                     "Complex number must be 16 bytes for cache efficiency");

// Helper functions for tagged value manipulation
static inline eshkol_tagged_value_t eshkol_make_int64(int64_t val, bool exact) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_INT64;
    result.flags = exact ? ESHKOL_VALUE_EXACT_FLAG : 0;
    result.reserved = 0;
    result.data.int_val = val;
    return result;
}

static inline eshkol_tagged_value_t eshkol_make_double(double val) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_DOUBLE;
    result.flags = ESHKOL_VALUE_INEXACT_FLAG;
    result.reserved = 0;
    result.data.double_val = val;
    return result;
}

static inline eshkol_tagged_value_t eshkol_make_ptr(uint64_t ptr, uint8_t type) {
    eshkol_tagged_value_t result;
    result.type = type;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = ptr;
    return result;
}

static inline eshkol_tagged_value_t eshkol_make_complex(uint64_t ptr) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_COMPLEX;
    result.flags = ESHKOL_VALUE_INEXACT_FLAG;  // Complex numbers are always inexact
    result.reserved = 0;
    result.data.ptr_val = ptr;
    return result;
}

static inline int64_t eshkol_unpack_int64(const eshkol_tagged_value_t* val) {
    return val->data.int_val;
}

static inline double eshkol_unpack_double(const eshkol_tagged_value_t* val) {
    return val->data.double_val;
}

static inline uint64_t eshkol_unpack_ptr(const eshkol_tagged_value_t* val) {
    return val->data.ptr_val;
}

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CHECKING MACROS - Handle both new consolidated and legacy types
// ═══════════════════════════════════════════════════════════════════════════

// Immediate type checks (no masking needed for new types)
#define ESHKOL_IS_NULL_TYPE(type)        ((type) == ESHKOL_VALUE_NULL)
#define ESHKOL_IS_INT64_TYPE(type)       ((type) == ESHKOL_VALUE_INT64)
#define ESHKOL_IS_DOUBLE_TYPE(type)      ((type) == ESHKOL_VALUE_DOUBLE)
#define ESHKOL_IS_BOOL_TYPE(type)        ((type) == ESHKOL_VALUE_BOOL)
#define ESHKOL_IS_CHAR_TYPE(type)        ((type) == ESHKOL_VALUE_CHAR)
#define ESHKOL_IS_SYMBOL_TYPE(type)      ((type) == ESHKOL_VALUE_SYMBOL)
#define ESHKOL_IS_DUAL_NUMBER_TYPE(type) ((type) == ESHKOL_VALUE_DUAL_NUMBER)
#define ESHKOL_IS_COMPLEX_TYPE(type)     ((type) == ESHKOL_VALUE_COMPLEX)

// Storage type checks - for cons cell setters that take int64 or double storage
// INT64 storage: INT64, BOOL, CHAR, SYMBOL (types that use int_val in the union)
// Also includes legacy pointer types (32+) and consolidated types (HEAP_PTR, CALLABLE)
// which store pointer addresses as int64
#define ESHKOL_IS_INT_STORAGE_TYPE(type) ((type) == ESHKOL_VALUE_INT64 || \
                                          (type) == ESHKOL_VALUE_BOOL || \
                                          (type) == ESHKOL_VALUE_CHAR || \
                                          (type) == ESHKOL_VALUE_SYMBOL || \
                                          (type) == ESHKOL_VALUE_HEAP_PTR || \
                                          (type) == ESHKOL_VALUE_CALLABLE || \
                                          (type) >= 32)

// Consolidated type checks
#define ESHKOL_IS_HEAP_PTR_TYPE(type)    ((type) == ESHKOL_VALUE_HEAP_PTR)
#define ESHKOL_IS_CALLABLE_TYPE(type)    ((type) == ESHKOL_VALUE_CALLABLE)
#define ESHKOL_IS_LOGIC_VAR_TYPE(type)   ((type) == ESHKOL_VALUE_LOGIC_VAR)

// Neuro-symbolic consciousness engine macros
#define ESHKOL_IS_LOGIC_VAR(tv)  ((tv).type == ESHKOL_VALUE_LOGIC_VAR)
#define ESHKOL_LOGIC_VAR_ID(tv)  ((tv).data.int_val)

// ───────────────────────────────────────────────────────────────────────────
// DEPRECATED Legacy type checks - for display system backward compatibility
// New code should check HEAP_PTR/CALLABLE type and read subtype from header
// ───────────────────────────────────────────────────────────────────────────
#define ESHKOL_IS_CONS_PTR_TYPE(type)    ((type) == ESHKOL_VALUE_CONS_PTR)     // DEPRECATED
#define ESHKOL_IS_STRING_PTR_TYPE(type)  ((type) == ESHKOL_VALUE_STRING_PTR)   // DEPRECATED
#define ESHKOL_IS_VECTOR_PTR_TYPE(type)  ((type) == ESHKOL_VALUE_VECTOR_PTR)   // DEPRECATED
#define ESHKOL_IS_TENSOR_PTR_TYPE(type)  ((type) == ESHKOL_VALUE_TENSOR_PTR)   // DEPRECATED
#define ESHKOL_IS_HASH_PTR_TYPE(type)    ((type) == ESHKOL_VALUE_HASH_PTR)     // DEPRECATED
#define ESHKOL_IS_EXCEPTION_TYPE(type)   ((type) == ESHKOL_VALUE_EXCEPTION)    // DEPRECATED
#define ESHKOL_IS_CLOSURE_PTR_TYPE(type) ((type) == ESHKOL_VALUE_CLOSURE_PTR)  // DEPRECATED
#define ESHKOL_IS_LAMBDA_SEXPR_TYPE(type) ((type) == ESHKOL_VALUE_LAMBDA_SEXPR) // DEPRECATED
#define ESHKOL_IS_AD_NODE_PTR_TYPE(type) ((type) == ESHKOL_VALUE_AD_NODE_PTR)  // DEPRECATED

// Combined type checks (consolidated + legacy for display compatibility)
#define ESHKOL_IS_ANY_HEAP_TYPE(type)    (ESHKOL_IS_HEAP_PTR_TYPE(type) || \
                                          ESHKOL_IS_CONS_PTR_TYPE(type) || \
                                          ESHKOL_IS_STRING_PTR_TYPE(type) || \
                                          ESHKOL_IS_VECTOR_PTR_TYPE(type) || \
                                          ESHKOL_IS_TENSOR_PTR_TYPE(type) || \
                                          ESHKOL_IS_HASH_PTR_TYPE(type) || \
                                          ESHKOL_IS_EXCEPTION_TYPE(type))

#define ESHKOL_IS_ANY_CALLABLE_TYPE(type) (ESHKOL_IS_CALLABLE_TYPE(type) || \
                                           ESHKOL_IS_CLOSURE_PTR_TYPE(type) || \
                                           ESHKOL_IS_LAMBDA_SEXPR_TYPE(type) || \
                                           ESHKOL_IS_AD_NODE_PTR_TYPE(type))

// General pointer type check: any type that stores a pointer value
#define ESHKOL_IS_ANY_PTR_TYPE(type)     (ESHKOL_IS_ANY_HEAP_TYPE(type) || \
                                          ESHKOL_IS_ANY_CALLABLE_TYPE(type))

// Exactness checking macros
#define ESHKOL_IS_EXACT(type)         (((type) & ESHKOL_VALUE_EXACT_FLAG) != 0)
#define ESHKOL_IS_INEXACT(type)       (((type) & ESHKOL_VALUE_INEXACT_FLAG) != 0)

// Type manipulation macros
#define ESHKOL_MAKE_EXACT(type)       ((type) | ESHKOL_VALUE_EXACT_FLAG)
#define ESHKOL_MAKE_INEXACT(type)     ((type) | ESHKOL_VALUE_INEXACT_FLAG)
// For new types, no masking needed. For types with exactness flags, mask them off.
#define ESHKOL_GET_BASE_TYPE(type)    ((type) & 0x3F)  // 6 bits for base type (0-63)

// ═══════════════════════════════════════════════════════════════════════════
// OBJECT HEADER SYSTEM (Foundation for Pointer Consolidation)
// All heap-allocated objects will be prefixed with this header.
// The header enables type consolidation: multiple specific types (cons, string,
// vector, etc.) are unified under HEAP_PTR with subtypes stored in the header.
// ═══════════════════════════════════════════════════════════════════════════

// Object flags for GC, linear types, and resource management
#define ESHKOL_OBJ_FLAG_MARKED    0x01  // GC mark bit
#define ESHKOL_OBJ_FLAG_LINEAR    0x02  // Linear type (must be consumed exactly once)
#define ESHKOL_OBJ_FLAG_BORROWED  0x04  // Currently borrowed (temporary access)
#define ESHKOL_OBJ_FLAG_CONSUMED  0x08  // Linear value has been consumed
#define ESHKOL_OBJ_FLAG_SHARED    0x10  // Reference-counted shared object
#define ESHKOL_OBJ_FLAG_WEAK      0x20  // Weak reference (doesn't prevent collection)
#define ESHKOL_OBJ_FLAG_PINNED    0x40  // Pinned in memory (no relocation)
#define ESHKOL_OBJ_FLAG_EXTERNAL  0x80  // External resource (needs explicit cleanup)

// Object header structure (8 bytes, prepended to all heap objects)
typedef struct eshkol_object_header {
    uint8_t  subtype;      // Distinguishes types within HEAP_PTR/CALLABLE/HANDLE
    uint8_t  flags;        // Object flags (GC marks, linear status, etc.)
    uint16_t ref_count;    // Reference count for shared objects (0 = not ref-counted)
    uint32_t size;         // Object size in bytes (excluding header)
} eshkol_object_header_t;

// Compile-time validation
ESHKOL_STATIC_ASSERT(sizeof(eshkol_object_header_t) == 8,
                     "Object header must be 8 bytes for alignment");

// ───────────────────────────────────────────────────────────────────────────
// HEAP_PTR SUBTYPES (type = ESHKOL_VALUE_HEAP_PTR = 8)
// Data structures allocated on the arena
// ───────────────────────────────────────────────────────────────────────────
typedef enum {
    HEAP_SUBTYPE_CONS        = 0,   // Cons cell (pair/list node)
    HEAP_SUBTYPE_STRING      = 1,   // String (UTF-8 with length)
    HEAP_SUBTYPE_VECTOR      = 2,   // Heterogeneous vector
    HEAP_SUBTYPE_TENSOR      = 3,   // N-dimensional numeric tensor
    HEAP_SUBTYPE_MULTI_VALUE = 4,   // Multiple return values container
    HEAP_SUBTYPE_HASH        = 5,   // Hash table / dictionary
    HEAP_SUBTYPE_EXCEPTION   = 6,   // Exception object
    HEAP_SUBTYPE_RECORD      = 7,   // User-defined record type
    HEAP_SUBTYPE_BYTEVECTOR  = 8,   // Raw byte vector (R7RS)
    HEAP_SUBTYPE_PORT        = 9,   // I/O port
    HEAP_SUBTYPE_SYMBOL      = 10,  // Interned symbol (distinct from string)
    HEAP_SUBTYPE_BIGNUM      = 11,  // Arbitrary-precision integer (R7RS exact)
    // Neuro-symbolic consciousness engine types
    HEAP_SUBTYPE_SUBSTITUTION    = 12,  // Immutable binding map {var_id -> tagged_value}
    HEAP_SUBTYPE_FACT            = 13,  // Predicate + arguments: (pred arg1 arg2 ...)
    // Reserved: 14 for RULE (v1.2 backward chaining)
    HEAP_SUBTYPE_KNOWLEDGE_BASE  = 15,  // Collection of facts with query support
    HEAP_SUBTYPE_FACTOR_GRAPH    = 16,  // Factor graph for probabilistic inference
    HEAP_SUBTYPE_WORKSPACE       = 17,  // Global workspace for cognitive competition
    HEAP_SUBTYPE_PROMISE         = 18,  // Lazy promise (delay/force with memoization)
    HEAP_SUBTYPE_RATIONAL        = 19,  // Exact rational number (numerator/denominator)
    // Reserved: 20-255 for future heap types
} heap_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// CALLABLE SUBTYPES (type = ESHKOL_VALUE_CALLABLE = 9)
// Function-like objects that can be invoked
// ───────────────────────────────────────────────────────────────────────────
typedef enum {
    CALLABLE_SUBTYPE_CLOSURE      = 0,  // Compiled closure (func_ptr + env)
    CALLABLE_SUBTYPE_LAMBDA_SEXPR = 1,  // Lambda as data (homoiconicity)
    CALLABLE_SUBTYPE_AD_NODE      = 2,  // Autodiff computation node
    CALLABLE_SUBTYPE_PRIMITIVE    = 3,  // Built-in primitive function
    CALLABLE_SUBTYPE_CONTINUATION = 4,  // First-class continuation (future)
    // Reserved: 5-255 for future callable types
} callable_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// HANDLE SUBTYPES (type = ESHKOL_VALUE_HANDLE = 16, future multimedia)
// External resources requiring explicit lifecycle management
// ───────────────────────────────────────────────────────────────────────────
typedef enum {
    HANDLE_SUBTYPE_WINDOW       = 0,   // Window/surface handle
    HANDLE_SUBTYPE_GL_CONTEXT   = 1,   // OpenGL/Vulkan/Metal context
    HANDLE_SUBTYPE_AUDIO_DEVICE = 2,   // Audio output device
    HANDLE_SUBTYPE_MIDI_PORT    = 3,   // MIDI I/O port
    HANDLE_SUBTYPE_CAMERA       = 4,   // Camera capture device
    HANDLE_SUBTYPE_SOCKET       = 5,   // Network socket
    HANDLE_SUBTYPE_FRAMEBUFFER  = 6,   // Offscreen render target
    HANDLE_SUBTYPE_FILE         = 7,   // File handle (distinct from PORT)
    HANDLE_SUBTYPE_THREAD       = 8,   // Thread handle
    HANDLE_SUBTYPE_MUTEX        = 9,   // Mutex/lock handle
    // Reserved: 10-255 for future handle types
} handle_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// BUFFER SUBTYPES (type = ESHKOL_VALUE_BUFFER = 17, future multimedia)
// Memory regions for zero-copy data transfer
// ───────────────────────────────────────────────────────────────────────────
typedef enum {
    BUFFER_SUBTYPE_RAW         = 0,   // Raw byte buffer
    BUFFER_SUBTYPE_IMAGE       = 1,   // Image pixel data
    BUFFER_SUBTYPE_AUDIO       = 2,   // Audio sample data
    BUFFER_SUBTYPE_VERTEX      = 3,   // GPU vertex buffer
    BUFFER_SUBTYPE_INDEX       = 4,   // GPU index buffer
    BUFFER_SUBTYPE_UNIFORM     = 5,   // GPU uniform buffer
    BUFFER_SUBTYPE_TEXTURE     = 6,   // Texture data
    BUFFER_SUBTYPE_MAPPED      = 7,   // Memory-mapped file
    // Reserved: 8-255 for future buffer types
} buffer_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// STREAM SUBTYPES (type = ESHKOL_VALUE_STREAM = 18, future multimedia)
// Async I/O and data flow pipelines
// ───────────────────────────────────────────────────────────────────────────
typedef enum {
    STREAM_SUBTYPE_BYTE        = 0,   // Raw byte stream
    STREAM_SUBTYPE_AUDIO       = 1,   // Audio sample stream
    STREAM_SUBTYPE_VIDEO       = 2,   // Video frame stream
    STREAM_SUBTYPE_MIDI        = 3,   // MIDI event stream
    STREAM_SUBTYPE_NETWORK     = 4,   // Network data stream
    STREAM_SUBTYPE_TRANSFORM   = 5,   // Transform/filter pipeline
    // Reserved: 6-255 for future stream types
} stream_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// EVENT SUBTYPES (type = ESHKOL_VALUE_EVENT = 19, future multimedia)
// Real-time event handling
// ───────────────────────────────────────────────────────────────────────────
typedef enum {
    EVENT_SUBTYPE_INPUT        = 0,   // Keyboard/mouse/touch input
    EVENT_SUBTYPE_WINDOW       = 1,   // Window resize/focus/close
    EVENT_SUBTYPE_AUDIO        = 2,   // Audio buffer ready/underrun
    EVENT_SUBTYPE_MIDI         = 3,   // MIDI note/CC/sysex
    EVENT_SUBTYPE_TIMER        = 4,   // Timer tick
    EVENT_SUBTYPE_NETWORK      = 5,   // Network packet/connection
    EVENT_SUBTYPE_CUSTOM       = 6,   // User-defined event
    // Reserved: 7-255 for future event types
} event_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// HEADER ACCESS MACROS
// These macros allow accessing the header from a data pointer
// Memory layout: [header (8 bytes)][object data (variable)]
// ───────────────────────────────────────────────────────────────────────────

// Get header pointer from data pointer (subtracts header size)
#define ESHKOL_GET_HEADER(data_ptr) \
    ((eshkol_object_header_t*)((uint8_t*)(data_ptr) - sizeof(eshkol_object_header_t)))

// Get data pointer from header pointer (adds header size)
#define ESHKOL_GET_DATA_PTR(header_ptr) \
    ((void*)((uint8_t*)(header_ptr) + sizeof(eshkol_object_header_t)))

// Get subtype from data pointer
#define ESHKOL_GET_SUBTYPE(data_ptr) \
    (ESHKOL_GET_HEADER(data_ptr)->subtype)

// Get flags from data pointer
#define ESHKOL_GET_FLAGS(data_ptr) \
    (ESHKOL_GET_HEADER(data_ptr)->flags)

// Set subtype on data pointer
#define ESHKOL_SET_SUBTYPE(data_ptr, st) \
    (ESHKOL_GET_HEADER(data_ptr)->subtype = (st))

// Set flags on data pointer
#define ESHKOL_SET_FLAGS(data_ptr, fl) \
    (ESHKOL_GET_HEADER(data_ptr)->flags = (fl))

// Check if object has specific flag
#define ESHKOL_HAS_FLAG(data_ptr, flag) \
    ((ESHKOL_GET_FLAGS(data_ptr) & (flag)) != 0)

// Set a specific flag
#define ESHKOL_ADD_FLAG(data_ptr, flag) \
    (ESHKOL_GET_HEADER(data_ptr)->flags |= (flag))

// Clear a specific flag
#define ESHKOL_CLEAR_FLAG(data_ptr, flag) \
    (ESHKOL_GET_HEADER(data_ptr)->flags &= ~(flag))

// Get object size from data pointer
#define ESHKOL_GET_OBJ_SIZE(data_ptr) \
    (ESHKOL_GET_HEADER(data_ptr)->size)

// Get reference count from data pointer
#define ESHKOL_GET_REF_COUNT(data_ptr) \
    (ESHKOL_GET_HEADER(data_ptr)->ref_count)

// Increment reference count (returns new count)
#define ESHKOL_INC_REF(data_ptr) \
    (++(ESHKOL_GET_HEADER(data_ptr)->ref_count))

// Decrement reference count (returns new count)
#define ESHKOL_DEC_REF(data_ptr) \
    (--(ESHKOL_GET_HEADER(data_ptr)->ref_count))

// ───────────────────────────────────────────────────────────────────────────
// COMPATIBILITY MACROS FOR TYPE CHECKING (Display System Use)
// ───────────────────────────────────────────────────────────────────────────
// These macros support BOTH legacy and consolidated formats for the display
// system to render old data. M1 Migration is COMPLETE - new code should use:
//   - HEAP_PTR type + ESHKOL_GET_SUBTYPE() to check heap subtypes
//   - CALLABLE type + ESHKOL_GET_SUBTYPE() to check callable subtypes
// ───────────────────────────────────────────────────────────────────────────

// Check if value is a cons cell (legacy CONS_PTR or new HEAP_PTR with CONS subtype)
#define ESHKOL_IS_CONS_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_CONS_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_CONS))

// Check if value is a string (legacy STRING_PTR or new HEAP_PTR with STRING subtype)
#define ESHKOL_IS_STRING_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_STRING_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_STRING))

// Check if value is a vector (legacy VECTOR_PTR or new HEAP_PTR with VECTOR subtype)
#define ESHKOL_IS_VECTOR_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_VECTOR_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_VECTOR))

// Check if value is a tensor (legacy TENSOR_PTR or new HEAP_PTR with TENSOR subtype)
#define ESHKOL_IS_TENSOR_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_TENSOR_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_TENSOR))

// Check if value is a hash table (legacy HASH_PTR or new HEAP_PTR with HASH subtype)
#define ESHKOL_IS_HASH_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_HASH_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_HASH))

// Check if value is an exception (legacy EXCEPTION or new HEAP_PTR with EXCEPTION subtype)
#define ESHKOL_IS_EXCEPTION_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_EXCEPTION || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_EXCEPTION))

// Check if value is a multi-value (ONLY new HEAP_PTR with MULTI_VALUE subtype)
#define ESHKOL_IS_MULTI_VALUE(val) \
    ((val).type == ESHKOL_VALUE_HEAP_PTR && \
     (val).data.ptr_val != 0 && \
     ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_MULTI_VALUE)

// Check if value is a bignum
#define ESHKOL_IS_BIGNUM(val) \
    ((val).type == ESHKOL_VALUE_HEAP_PTR && \
     (val).data.ptr_val != 0 && \
     ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_BIGNUM)

// Check if value is a closure (legacy CLOSURE_PTR or new CALLABLE with CLOSURE subtype)
#define ESHKOL_IS_CLOSURE_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_CLOSURE_PTR || \
     ((val).type == ESHKOL_VALUE_CALLABLE && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == CALLABLE_SUBTYPE_CLOSURE))

// Check if value is a lambda sexpr (legacy LAMBDA_SEXPR or new CALLABLE with LAMBDA_SEXPR subtype)
#define ESHKOL_IS_LAMBDA_SEXPR_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_LAMBDA_SEXPR || \
     ((val).type == ESHKOL_VALUE_CALLABLE && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == CALLABLE_SUBTYPE_LAMBDA_SEXPR))

// Check if value is an AD node (legacy AD_NODE_PTR or new CALLABLE with AD_NODE subtype)
#define ESHKOL_IS_AD_NODE_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_AD_NODE_PTR || \
     ((val).type == ESHKOL_VALUE_CALLABLE && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == CALLABLE_SUBTYPE_AD_NODE))

// Check if value is any heap pointer (legacy individual types OR new consolidated HEAP_PTR)
#define ESHKOL_IS_HEAP_PTR_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_HEAP_PTR || \
     (val).type == ESHKOL_VALUE_CONS_PTR || \
     (val).type == ESHKOL_VALUE_STRING_PTR || \
     (val).type == ESHKOL_VALUE_VECTOR_PTR || \
     (val).type == ESHKOL_VALUE_TENSOR_PTR || \
     (val).type == ESHKOL_VALUE_HASH_PTR || \
     (val).type == ESHKOL_VALUE_EXCEPTION)

// Check if value is any callable (legacy individual types OR new consolidated CALLABLE)
#define ESHKOL_IS_CALLABLE_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_CALLABLE || \
     (val).type == ESHKOL_VALUE_CLOSURE_PTR || \
     (val).type == ESHKOL_VALUE_LAMBDA_SEXPR || \
     (val).type == ESHKOL_VALUE_AD_NODE_PTR)

// Check if value is a multimedia handle
#define ESHKOL_IS_HANDLE(val) ((val).type == ESHKOL_VALUE_HANDLE)

// Check if value is a multimedia buffer
#define ESHKOL_IS_BUFFER(val) ((val).type == ESHKOL_VALUE_BUFFER)

// Check if value is a multimedia stream
#define ESHKOL_IS_STREAM(val) ((val).type == ESHKOL_VALUE_STREAM)

// Check if value is a multimedia event
#define ESHKOL_IS_EVENT(val)  ((val).type == ESHKOL_VALUE_EVENT)

// ═══════════════════════════════════════════════════════════════════════════
// END OBJECT HEADER SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

// Dual number helper functions for forward-mode automatic differentiation
static inline eshkol_dual_number_t eshkol_make_dual(double value, double derivative) {
    eshkol_dual_number_t result;
    result.value = value;
    result.derivative = derivative;
    return result;
}

static inline double eshkol_dual_value(const eshkol_dual_number_t* d) {
    return d->value;
}

static inline double eshkol_dual_derivative(const eshkol_dual_number_t* d) {
    return d->derivative;
}

// ===== COMPUTATIONAL GRAPH NODE TYPES =====
// AD node types for reverse-mode automatic differentiation

typedef enum {
    // Core operations (0-11)
    AD_NODE_CONSTANT,
    AD_NODE_VARIABLE,
    AD_NODE_ADD,
    AD_NODE_SUB,
    AD_NODE_MUL,
    AD_NODE_DIV,
    AD_NODE_SIN,
    AD_NODE_COS,
    AD_NODE_EXP,
    AD_NODE_LOG,
    AD_NODE_POW,
    AD_NODE_NEG,

    // Activation gradients (12-18)
    AD_NODE_RELU,
    AD_NODE_SIGMOID,
    AD_NODE_SOFTMAX,
    AD_NODE_TANH,
    AD_NODE_GELU,
    AD_NODE_LEAKY_RELU,
    AD_NODE_SILU,

    // Tensor operation gradients (19-28)
    AD_NODE_CONV2D,
    AD_NODE_MAXPOOL2D,
    AD_NODE_AVGPOOL2D,
    AD_NODE_BATCHNORM,
    AD_NODE_LAYERNORM,
    AD_NODE_MATMUL,
    AD_NODE_TRANSPOSE,
    AD_NODE_RESHAPE,
    AD_NODE_SUM,
    AD_NODE_MEAN,

    // Transformer gradients (29-32)
    AD_NODE_ATTENTION,
    AD_NODE_MULTIHEAD_ATTENTION,
    AD_NODE_POSITIONAL_ENCODING,
    AD_NODE_EMBEDDING,

    // qLLM Geometric gradients (33-40)
    AD_NODE_HYPERBOLIC_DISTANCE,
    AD_NODE_POINCARE_EXP_MAP,
    AD_NODE_POINCARE_LOG_MAP,
    AD_NODE_TANGENT_PROJECT,
    AD_NODE_GEODESIC_ATTENTION,
    AD_NODE_MOBIUS_ADD,
    AD_NODE_MOBIUS_MATMUL,
    AD_NODE_GYROVECTOR_SPACE,

    // Additional math operations (41-44)
    AD_NODE_SQRT,
    AD_NODE_ABS,
    AD_NODE_SQUARE,
    AD_NODE_MAX,
    AD_NODE_MIN,

    // Phase 4 activation gradients (46-53)
    AD_NODE_ELU = 46,
    AD_NODE_SELU,
    AD_NODE_MISH,
    AD_NODE_HARDSWISH,
    AD_NODE_HARDSIGMOID,
    AD_NODE_SOFTPLUS,
    AD_NODE_DROPOUT,
    AD_NODE_CELU,

    // Complete math function gradients (54-66)
    AD_NODE_TAN = 54,
    AD_NODE_ASIN,
    AD_NODE_ACOS,
    AD_NODE_ATAN,
    AD_NODE_SINH,
    AD_NODE_COSH,
    AD_NODE_ASINH,
    AD_NODE_ACOSH,
    AD_NODE_ATANH,
    AD_NODE_LOG10,
    AD_NODE_LOG2,
    AD_NODE_EXP2,
    AD_NODE_CBRT,

    // Tensor AD nodes for qLLM bridge (67-79)
    AD_NODE_TENSOR_MATMUL = 67,       // dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    AD_NODE_TENSOR_SOFTMAX,            // Jacobian-vector product
    AD_NODE_TENSOR_LAYERNORM,          // Chain rule through mean/variance
    AD_NODE_TENSOR_RMSNORM,            // Chain rule through RMS
    AD_NODE_TENSOR_ATTENTION,          // 5-step chain rule through scaled dot-product
    AD_NODE_TENSOR_GELU,               // GELU backward
    AD_NODE_TENSOR_SILU,               // SiLU/Swish backward
    AD_NODE_TENSOR_TRANSPOSE,          // Permutation backward
    AD_NODE_TENSOR_SUM,                // Broadcast backward
    AD_NODE_TENSOR_BROADCAST_ADD,      // Sum-reduce backward
    AD_NODE_TENSOR_BROADCAST_MUL,      // Product-rule backward
    AD_NODE_TENSOR_EMBEDDING,          // Sparse update backward
    AD_NODE_TENSOR_CROSS_ENTROPY,      // Numerically stable backward
    AD_NODE_FRECHET_MEAN,              // Riemannian center of mass backward

    // Sentinel for bounds checking
    AD_NODE_TYPE_COUNT
} ad_node_type_t;

// Computational graph node for reverse-mode AD
// Stores the computational graph for backpropagation
typedef struct ad_node {
    ad_node_type_t type;     // Type of operation
    double value;            // Computed value during forward pass (scalar)
    double gradient;         // Accumulated gradient during backward pass (scalar)
    struct ad_node* input1;  // First parent node (null for constants/variables)
    struct ad_node* input2;  // Second parent node (null for unary ops)
    size_t id;              // Unique node ID for topological sorting

    // === Extended fields for tensor operations ===
    // These are optional and only used for tensor-based AD nodes

    // Tensor data (when operating on tensors instead of scalars)
    void* tensor_value;      // Pointer to tensor value (null for scalar nodes)
    void* tensor_gradient;   // Pointer to tensor gradient (null for scalar nodes)

    // Additional inputs for multi-input operations (attention, etc.)
    struct ad_node* input3;  // Third input (e.g., V in attention)
    struct ad_node* input4;  // Fourth input (e.g., mask in attention)

    // Saved tensors for backward pass (e.g., softmax output, conv indices)
    void** saved_tensors;    // Array of saved tensors for backward
    size_t num_saved;        // Number of saved tensors

    // Operation parameters
    union {
        double alpha;        // For leaky_relu, dropout rate, etc.
        double curvature;    // For hyperbolic/Poincare operations
        int64_t axis;        // For reduction operations (sum, mean)
        struct {
            int64_t kernel_h, kernel_w;   // For conv2d, pooling
            int64_t stride_h, stride_w;
            int64_t pad_h, pad_w;
        } conv_params;
        struct {
            int64_t num_heads;            // For multi-head attention
            int64_t head_dim;
        } attention_params;
    } params;

    // Shape information for tensor operations
    int64_t* shape;          // Output shape
    size_t ndim;             // Number of dimensions
} ad_node_t;

// Computational graph tape for recording operations
// Maintains all nodes in evaluation order for backpropagation
typedef struct ad_tape {
    ad_node_t** nodes;       // Array of nodes in evaluation order
    size_t num_nodes;        // Current number of nodes
    size_t capacity;         // Allocated capacity
    ad_node_t** variables;   // Input variable nodes
    size_t num_variables;    // Number of input variables
} ad_tape_t;

// ===== CLOSURE ENVIRONMENT STRUCTURES =====
// Support for lexical closures - capturing parent scope variables

// Closure environment structure (arena-allocated)
// Holds captured variables from parent scope for nested functions
//
// VARIADIC ENCODING: The num_captures field encodes both capture count and variadic info:
//   - Bits 0-15:  num_captures (up to 65535 captures)
//   - Bits 16-31: fixed_param_count (up to 65535 fixed params)
//   - Bit 63:     is_variadic flag (1 = variadic, 0 = not variadic)
//
// Use the macros below to extract/encode these values:
#define CLOSURE_ENV_GET_NUM_CAPTURES(packed) ((packed) & 0xFFFF)
#define CLOSURE_ENV_GET_FIXED_PARAMS(packed) (((packed) >> 16) & 0xFFFF)
#define CLOSURE_ENV_IS_VARIADIC(packed) (((packed) >> 63) & 1)
#define CLOSURE_ENV_PACK(num_caps, fixed_params, is_var) \
    (((size_t)(num_caps) & 0xFFFF) | \
     (((size_t)(fixed_params) & 0xFFFF) << 16) | \
     ((size_t)(is_var) << 63))

typedef struct eshkol_closure_env {
    size_t num_captures;                  // Packed: num_captures | (fixed_params << 16) | (is_variadic << 63)
    eshkol_tagged_value_t captures[];     // Flexible array of captured values
} eshkol_closure_env_t;

// Compile-time size validation
ESHKOL_STATIC_ASSERT(sizeof(eshkol_closure_env_t) == sizeof(size_t),
                     "Closure environment header must be minimal");

// Closure return type constants (matches eshkol_value_type_t but with additional info)
// These are used to determine function behavior without calling it
#define CLOSURE_RETURN_UNKNOWN     0x00  // Return type not known (untyped lambda)
#define CLOSURE_RETURN_SCALAR      0x01  // Returns a scalar (int64 or double)
#define CLOSURE_RETURN_VECTOR      0x02  // Returns a vector/tensor
#define CLOSURE_RETURN_LIST        0x03  // Returns a list (cons cells)
#define CLOSURE_RETURN_FUNCTION    0x04  // Returns another function (higher-order)
#define CLOSURE_RETURN_BOOL        0x05  // Returns a boolean
#define CLOSURE_RETURN_STRING      0x06  // Returns a string
#define CLOSURE_RETURN_VOID        0x07  // Returns null/void

// Closure flags (stored in closure->flags field)
#define CLOSURE_FLAG_VARIADIC              0x01  // Closure accepts variadic arguments
#define CLOSURE_FLAG_NAMED                 0x02  // Closure has a bound name
#define ESHKOL_CLOSURE_FLAG_VARIADIC       CLOSURE_FLAG_VARIADIC  // Alias for consistency
#define ESHKOL_CLOSURE_FLAG_NAMED          CLOSURE_FLAG_NAMED     // Alias for consistency

// Full closure structure combining function pointer and environment
// This is what gets allocated when a closure-returning function is called
typedef struct eshkol_closure {
    uint64_t func_ptr;                    // Pointer to the lambda function
    eshkol_closure_env_t* env;            // Pointer to captured environment (may be NULL for no captures)
    uint64_t sexpr_ptr;                   // Pointer to S-expression representation for homoiconicity
    const char* name;                     // Bound name from (define name ...) or NULL for anonymous lambdas
    uint8_t return_type;                  // Return type category (CLOSURE_RETURN_*)
    uint8_t input_arity;                  // Number of expected input arguments (0-255)
    uint8_t flags;                        // Additional flags (CLOSURE_FLAG_VARIADIC, etc.)
    uint8_t reserved;                     // Padding for alignment
    uint32_t hott_type_id;                // HoTT TypeId for the return type (0 = unknown)
} eshkol_closure_t;

// Compile-time size validation for closure structure
ESHKOL_STATIC_ASSERT(sizeof(eshkol_closure_t) == 40,
                     "Closure structure must be 40 bytes for alignment");

// Helper macros for closure type queries
#define CLOSURE_RETURNS_VECTOR(closure) ((closure)->return_type == CLOSURE_RETURN_VECTOR)
#define CLOSURE_RETURNS_SCALAR(closure) ((closure)->return_type == CLOSURE_RETURN_SCALAR)
#define CLOSURE_RETURNS_FUNCTION(closure) ((closure)->return_type == CLOSURE_RETURN_FUNCTION)
#define CLOSURE_TYPE_KNOWN(closure) ((closure)->return_type != CLOSURE_RETURN_UNKNOWN)

// Get closure return type from a tagged value
// Supports both consolidated CALLABLE type and legacy CLOSURE_PTR
static inline uint8_t eshkol_closure_get_return_type(eshkol_tagged_value_t tagged) {
    uint8_t base_type = tagged.type & 0x3F;  // Mask off flags

    // Check for consolidated CALLABLE type
    if (base_type == ESHKOL_VALUE_CALLABLE) {
        uint64_t ptr = tagged.data.ptr_val;
        if (!ptr) return CLOSURE_RETURN_UNKNOWN;
        // For CALLABLE, verify subtype is closure before reading
        eshkol_object_header_t* header = ESHKOL_GET_HEADER((void*)ptr);
        if (header->subtype != CALLABLE_SUBTYPE_CLOSURE) {
            return CLOSURE_RETURN_UNKNOWN;
        }
        eshkol_closure_t* closure = (eshkol_closure_t*)ptr;
        return closure->return_type;
    }

    // Legacy support: CLOSURE_PTR (deprecated)
    if (base_type == ESHKOL_VALUE_CLOSURE_PTR) {
        uint64_t ptr = tagged.data.ptr_val;
        eshkol_closure_t* closure = (eshkol_closure_t*)ptr;
        return closure ? closure->return_type : CLOSURE_RETURN_UNKNOWN;
    }

    return CLOSURE_RETURN_UNKNOWN;
}

// Check if a closure returns a vector (for autodiff)
static inline bool eshkol_closure_returns_vector(eshkol_tagged_value_t tagged) {
    return eshkol_closure_get_return_type(tagged) == CLOSURE_RETURN_VECTOR;
}

// Check if a closure returns a scalar (for autodiff)
static inline bool eshkol_closure_returns_scalar(eshkol_tagged_value_t tagged) {
    return eshkol_closure_get_return_type(tagged) == CLOSURE_RETURN_SCALAR;
}

// ===== PRIMITIVE FUNCTION STRUCTURE =====
// Primitive functions are built-in operations that are not closures.
// Unlike closures, they don't have captured environments or S-expression representations.
// The object header's subtype field indicates CALLABLE_SUBTYPE_PRIMITIVE.

// Primitive flags
#define PRIMITIVE_FLAG_VARIADIC    0x01  // Primitive accepts variadic arguments
#define PRIMITIVE_FLAG_PURE        0x02  // Primitive has no side effects

/**
 * Runtime representation of a primitive/builtin function.
 *
 * Primitives are similar to closures but without captured environments.
 * They store metadata needed for introspection (arity, name, type).
 */
typedef struct eshkol_primitive {
    uint64_t func_ptr;                    // Pointer to the native function implementation
    const char* name;                     // Name of the primitive (e.g., "car", "cdr", "+")
    uint8_t input_arity;                  // Number of expected input arguments (0-255)
    uint8_t flags;                        // Primitive flags (variadic, pure, etc.)
    uint16_t reserved;                    // Padding for alignment
    uint32_t hott_type_id;                // HoTT TypeId for function signature (0 = unknown)
} eshkol_primitive_t;

// Compile-time size validation for primitive structure
ESHKOL_STATIC_ASSERT(sizeof(eshkol_primitive_t) == 24,
                     "Primitive structure must be 24 bytes for alignment");

// Helper to check if primitive is variadic
#define PRIMITIVE_IS_VARIADIC(prim) (((prim)->flags & PRIMITIVE_FLAG_VARIADIC) != 0)

// Helper to check if primitive is pure (no side effects)
#define PRIMITIVE_IS_PURE(prim) (((prim)->flags & PRIMITIVE_FLAG_PURE) != 0)

// ===== END CLOSURE ENVIRONMENT STRUCTURES =====

// ===== EXCEPTION HANDLING STRUCTURES =====
// Support for R7RS-compatible exception handling (guard, raise, error)

// Exception type codes for built-in exception types
typedef enum {
    ESHKOL_EXCEPTION_ERROR = 0,        // Generic error
    ESHKOL_EXCEPTION_TYPE_ERROR,       // Type mismatch
    ESHKOL_EXCEPTION_FILE_ERROR,       // File operation failed
    ESHKOL_EXCEPTION_READ_ERROR,       // Read/parse error
    ESHKOL_EXCEPTION_SYNTAX_ERROR,     // Syntax error
    ESHKOL_EXCEPTION_RANGE_ERROR,      // Index out of bounds
    ESHKOL_EXCEPTION_ARITY_ERROR,      // Wrong number of arguments
    ESHKOL_EXCEPTION_DIVIDE_BY_ZERO,   // Division by zero
    ESHKOL_EXCEPTION_USER_DEFINED      // User-defined exception
} eshkol_exception_type_t;

// Exception object structure (arena-allocated)
typedef struct eshkol_exception {
    eshkol_exception_type_t type;      // Exception type code
    char* message;                      // Error message
    eshkol_tagged_value_t* irritants;   // Array of irritant values
    uint32_t num_irritants;             // Number of irritants
    uint32_t line;                      // Source line (0 = unknown)
    uint32_t column;                    // Source column (0 = unknown)
    char* filename;                     // Source filename (NULL = unknown)
} eshkol_exception_t;

// Exception handler entry for the handler stack
typedef struct eshkol_exception_handler {
    void* jmp_buf_ptr;                  // Pointer to setjmp buffer
    struct eshkol_exception_handler* prev;  // Previous handler in stack
} eshkol_exception_handler_t;

// Global exception state (thread-local in multi-threaded context)
// Current exception being handled (NULL if none)
extern eshkol_exception_t* g_current_exception;
// Top of exception handler stack (NULL if no handlers)
extern eshkol_exception_handler_t* g_exception_handler_stack;

// Exception API functions (implemented in arena_memory.cpp)
eshkol_exception_t* eshkol_make_exception(eshkol_exception_type_t type, const char* message);
eshkol_exception_t* eshkol_make_exception_with_header(eshkol_exception_type_t type, const char* message);
void eshkol_exception_add_irritant(eshkol_exception_t* exc, eshkol_tagged_value_t irritant);
void eshkol_exception_set_location(eshkol_exception_t* exc, uint32_t line, uint32_t column, const char* filename);
void eshkol_raise(eshkol_exception_t* exception);
void eshkol_push_exception_handler(void* jmp_buf_ptr);
void eshkol_pop_exception_handler(void);
int eshkol_exception_type_matches(eshkol_exception_t* exc, eshkol_exception_type_t type);

// Helper to create tagged exception value
static inline eshkol_tagged_value_t eshkol_make_exception_value(eshkol_exception_t* exc) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_EXCEPTION;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = (uint64_t)exc;
    return result;
}

// ===== END EXCEPTION HANDLING STRUCTURES =====

// ===== FIRST-CLASS CONTINUATIONS =====

// Continuation state: captures the setjmp point and holds the value being passed
typedef struct eshkol_continuation_state {
    void* jmp_buf_ptr;                  // Points to jmp_buf on the call/cc caller's stack
    eshkol_tagged_value_t value;        // Value passed when continuation is invoked
    void* wind_mark;                    // Dynamic-wind stack marker at capture time
} eshkol_continuation_state_t;

// Dynamic-wind handler entry for the handler stack
typedef struct eshkol_dynamic_wind_entry {
    eshkol_tagged_value_t before;       // Before thunk (callable)
    eshkol_tagged_value_t after;        // After thunk (callable)
    struct eshkol_dynamic_wind_entry* prev;  // Previous entry in stack
} eshkol_dynamic_wind_entry_t;

// Global dynamic-wind stack
extern eshkol_dynamic_wind_entry_t* g_dynamic_wind_stack;

// Continuation runtime functions
eshkol_continuation_state_t* eshkol_make_continuation_state(void* arena, void* jmp_buf_ptr);
void* eshkol_make_continuation_closure(void* arena, void* state_ptr);
void eshkol_push_dynamic_wind(void* arena, const eshkol_tagged_value_t* before, const eshkol_tagged_value_t* after);
void eshkol_pop_dynamic_wind(void);
void eshkol_unwind_dynamic_wind(void* saved_wind_mark);

// ===== END FIRST-CLASS CONTINUATIONS =====

// Initialize process/thread stack sizing for deep recursion support.
void eshkol_init_stack_size(void);

// ===== LAMBDA REGISTRY FOR HOMOICONICITY =====
// Runtime table mapping function pointers to their S-expression representations
// This enables full homoiconicity: (display (list double)) shows the lambda source

typedef struct eshkol_lambda_entry {
    uint64_t func_ptr;      // Function pointer as uint64
    uint64_t sexpr_ptr;     // Pointer to S-expression cons cell (0 if none)
    const char* name;       // Function name for debugging (may be NULL)
} eshkol_lambda_entry_t;

typedef struct eshkol_lambda_registry {
    eshkol_lambda_entry_t* entries;
    size_t count;
    size_t capacity;
} eshkol_lambda_registry_t;

// Global lambda registry (defined in arena_memory.cpp)
extern eshkol_lambda_registry_t* g_lambda_registry;

// Lambda registry API
void eshkol_lambda_registry_init(void);
void eshkol_lambda_registry_destroy(void);
void eshkol_lambda_registry_add(uint64_t func_ptr, uint64_t sexpr_ptr, const char* name);
uint64_t eshkol_lambda_registry_lookup(uint64_t func_ptr);

// ===== END LAMBDA REGISTRY =====

// ===== UNIFIED DISPLAY SYSTEM =====
// Single source of truth for displaying all Eshkol values

// Forward declaration for tagged cons cell
struct arena_tagged_cons_cell;

// Display options for customizing output
typedef struct eshkol_display_opts {
    int max_depth;          // Maximum recursion depth (default: 100)
    int current_depth;      // Current depth (internal use)
    uint8_t quote_strings;  // Quote strings with "" (true for 'write', false for 'display')
    uint8_t show_types;     // Debug: show type tags
    void* output;           // Output stream (FILE*, default: stdout)
} eshkol_display_opts_t;

// Default display options
static inline eshkol_display_opts_t eshkol_display_default_opts(void) {
    eshkol_display_opts_t opts;
    opts.max_depth = 100;
    opts.current_depth = 0;
    opts.quote_strings = 0;
    opts.show_types = 0;
    opts.output = 0;  // NULL means stdout
    return opts;
}

// Main display functions (implemented in arena_memory.cpp)
void eshkol_display_value(const eshkol_tagged_value_t* value);
void eshkol_display_value_opts(const eshkol_tagged_value_t* value, eshkol_display_opts_t* opts);
void eshkol_write_value(const eshkol_tagged_value_t* value);  // Scheme 'write' semantics
void eshkol_write_value_to_port(const eshkol_tagged_value_t* value, void* port);
void eshkol_display_value_to_port(const eshkol_tagged_value_t* value, void* port);

// Display a list (cons cell chain)
void eshkol_display_list(uint64_t cons_ptr, eshkol_display_opts_t* opts);

// Display a lambda by looking up its S-expression in the registry
void eshkol_display_lambda(uint64_t func_ptr, eshkol_display_opts_t* opts);

// Display a closure by extracting its embedded S-expression
void eshkol_display_closure(uint64_t closure_ptr, eshkol_display_opts_t* opts);

// ===== END UNIFIED DISPLAY SYSTEM =====

// ===== END COMPUTATIONAL GRAPH TYPES =====

// ===== HoTT TYPE SYSTEM =====
// Homotopy Type Theory inspired type expressions for static type checking

// Type expression kinds
typedef enum {
    HOTT_TYPE_INVALID,
    // Primitive types
    HOTT_TYPE_INTEGER,      // integer
    HOTT_TYPE_REAL,         // real (double)
    HOTT_TYPE_NUMBER,       // number (supertype of integer and real)
    HOTT_TYPE_BOOLEAN,      // boolean
    HOTT_TYPE_STRING,       // string
    HOTT_TYPE_CHAR,         // char
    HOTT_TYPE_SYMBOL,       // symbol
    HOTT_TYPE_NULL,         // null (empty list)
    HOTT_TYPE_ANY,          // any (top type)
    HOTT_TYPE_NOTHING,      // nothing (bottom type)
    // Compound types
    HOTT_TYPE_ARROW,        // (-> a b) function type
    HOTT_TYPE_PRODUCT,      // (* a b) product type
    HOTT_TYPE_SUM,          // (+ a b) sum type (either)
    HOTT_TYPE_LIST,         // (list a) list type
    HOTT_TYPE_VECTOR,       // (vector a) vector type
    HOTT_TYPE_TENSOR,       // (tensor a) tensor type (multi-dimensional array)
    HOTT_TYPE_PAIR,         // (pair a b) cons pair type
    // Polymorphic types
    HOTT_TYPE_VAR,          // type variable (e.g., 'a in forall)
    HOTT_TYPE_FORALL,       // (forall (a b ...) type) universal quantification
    // Advanced types (future)
    HOTT_TYPE_DEPENDENT,    // dependent type (Pi type)
    HOTT_TYPE_PATH,         // path type (identity type)
    HOTT_TYPE_UNIVERSE      // universe level
} hott_type_kind_t;

// Forward declaration
struct hott_type_expr;

// Type expression structure
typedef struct hott_type_expr {
    hott_type_kind_t kind;
    union {
        // For type variables: the variable name
        char* var_name;

        // For arrow types: domain -> codomain
        struct {
            struct hott_type_expr** param_types;  // Array of parameter types
            uint64_t num_params;
            struct hott_type_expr* return_type;
        } arrow;

        // For forall types: quantified variables and body
        struct {
            char** type_vars;           // Array of type variable names
            uint64_t num_vars;
            struct hott_type_expr* body;
        } forall;

        // For compound types: list, vector, etc.
        struct {
            struct hott_type_expr* element_type;
        } container;

        // For pair/product types
        struct {
            struct hott_type_expr* left;
            struct hott_type_expr* right;
        } pair;

        // For sum types
        struct {
            struct hott_type_expr* left;
            struct hott_type_expr* right;
        } sum;

        // For universe types
        struct {
            uint32_t level;  // Universe level (U0, U1, U2, ...)
        } universe;
    };
} hott_type_expr_t;

// Helper macros for type construction
#define HOTT_MAKE_PRIMITIVE(kind) ((hott_type_expr_t){.kind = (kind)})

// ===== END HoTT TYPE SYSTEM =====

// ===== PATTERN MATCHING SYSTEM =====

// Pattern types for match expressions
typedef enum {
    PATTERN_INVALID,
    PATTERN_LITERAL,      // Literal value: 42, "hello", #t
    PATTERN_VARIABLE,     // Variable binding: x, y, z
    PATTERN_WILDCARD,     // Wildcard: _
    PATTERN_CONS,         // Cons pattern: (cons car cdr)
    PATTERN_LIST,         // List pattern: (list x y z ...)
    PATTERN_PREDICATE,    // Predicate: (? pred)
    PATTERN_OR            // Or pattern: (or p1 p2)
} pattern_type_t;

// Forward declarations
struct eshkol_pattern;
struct eshkol_ast;

// Pattern structure for match expressions
typedef struct eshkol_pattern {
    pattern_type_t type;
    union {
        // PATTERN_LITERAL: stores the literal value
        struct {
            struct eshkol_ast *value;
        } literal;

        // PATTERN_VARIABLE: variable name to bind
        struct {
            char *name;
        } variable;

        // PATTERN_CONS: (cons car_pat cdr_pat)
        struct {
            struct eshkol_pattern *car_pattern;
            struct eshkol_pattern *cdr_pattern;
        } cons;

        // PATTERN_LIST: (list pat1 pat2 ...)
        struct {
            struct eshkol_pattern **patterns;
            uint64_t num_patterns;
        } list;

        // PATTERN_PREDICATE: (? pred-expr)
        struct {
            struct eshkol_ast *predicate;
        } predicate;

        // PATTERN_OR: (or pat1 pat2 ...)
        struct {
            struct eshkol_pattern **patterns;
            uint64_t num_patterns;
        } or_pat;
    };
} eshkol_pattern_t;

// Match clause structure
typedef struct eshkol_match_clause {
    eshkol_pattern_t *pattern;      // Pattern to match
    struct eshkol_ast *guard;       // Optional (when ...) guard expression
    struct eshkol_ast *body;        // Body expression
} eshkol_match_clause_t;

// ===== END PATTERN MATCHING SYSTEM =====

// ═══════════════════════════════════════════════════════════════════════════
// MACRO SYSTEM (syntax-rules)
// Implements hygienic macros via pattern matching and template substitution.
// Pattern: ((macro-name args...) template)
// Supports: literals, pattern variables, ellipsis (...), nested patterns
// ═══════════════════════════════════════════════════════════════════════════

// Macro pattern element types
typedef enum {
    MACRO_PAT_LITERAL,      // Literal identifier that must match exactly
    MACRO_PAT_VARIABLE,     // Pattern variable to capture value
    MACRO_PAT_ELLIPSIS,     // Ellipsis marker for repetition (...)
    MACRO_PAT_LIST,         // Nested list pattern
    MACRO_PAT_IMPROPER      // Improper list pattern (x . rest)
} macro_pattern_type_t;

// Forward declaration
struct eshkol_macro_pattern;

// Macro pattern element
typedef struct eshkol_macro_pattern {
    macro_pattern_type_t type;
    union {
        // MACRO_PAT_LITERAL / MACRO_PAT_VARIABLE
        char *identifier;

        // MACRO_PAT_LIST / MACRO_PAT_IMPROPER
        struct {
            struct eshkol_macro_pattern **elements;
            uint64_t num_elements;
            struct eshkol_macro_pattern *rest;  // For improper lists only
        } list;
    };
    uint8_t followed_by_ellipsis;  // True if this element is followed by ...
} eshkol_macro_pattern_t;

// Template element types (for template construction)
typedef enum {
    MACRO_TPL_LITERAL,      // Literal value (copied as-is)
    MACRO_TPL_VARIABLE,     // Pattern variable reference
    MACRO_TPL_LIST,         // List constructor
    MACRO_TPL_ELLIPSIS      // Repetition of preceding element
} macro_template_type_t;

// Forward declaration
struct eshkol_macro_template;

// Macro template element
typedef struct eshkol_macro_template {
    macro_template_type_t type;
    union {
        // MACRO_TPL_LITERAL
        struct eshkol_ast *literal;

        // MACRO_TPL_VARIABLE
        char *variable_name;

        // MACRO_TPL_LIST
        struct {
            struct eshkol_macro_template **elements;
            uint64_t num_elements;
        } list;
    };
    uint8_t followed_by_ellipsis;  // True if template element followed by ...
} eshkol_macro_template_t;

// Macro rule: (pattern template)
typedef struct eshkol_macro_rule {
    eshkol_macro_pattern_t *pattern;    // Pattern to match (including macro name)
    eshkol_macro_template_t *template_; // Template for expansion
} eshkol_macro_rule_t;

// Macro definition: (define-syntax name (syntax-rules (literals...) rules...))
typedef struct eshkol_macro_def {
    char *name;                         // Macro name
    char **literals;                    // Literal identifiers that must match exactly
    uint64_t num_literals;
    eshkol_macro_rule_t *rules;         // Array of rules
    uint64_t num_rules;
} eshkol_macro_def_t;

// ===== END MACRO SYSTEM =====

typedef enum {
    ESHKOL_INVALID_OP,
    ESHKOL_COMPOSE_OP,
    ESHKOL_IF_OP,
    ESHKOL_ADD_OP,
    ESHKOL_SUB_OP,
    ESHKOL_MUL_OP,
    ESHKOL_DIV_OP,
    ESHKOL_CALL_OP,
    ESHKOL_DEFINE_OP,
    ESHKOL_SEQUENCE_OP,
    ESHKOL_EXTERN_OP,
    ESHKOL_EXTERN_VAR_OP,
    ESHKOL_LAMBDA_OP,
    ESHKOL_LET_OP,
    ESHKOL_LET_STAR_OP,  // let* - sequential bindings
    ESHKOL_LETREC_OP,    // letrec - recursive bindings (all bindings visible to all values)
    ESHKOL_LETREC_STAR_OP, // letrec* - sequential recursive bindings (R7RS: left-to-right evaluation)
    ESHKOL_AND_OP,       // short-circuit and
    ESHKOL_OR_OP,        // short-circuit or
    ESHKOL_COND_OP,      // multi-branch conditional
    ESHKOL_CASE_OP,      // case expression (switch on value)
    ESHKOL_MATCH_OP,     // pattern matching (match expr (pattern body) ...)
    ESHKOL_DO_OP,        // do loop (iteration construct)
    ESHKOL_WHEN_OP,      // when - one-armed if (execute when true)
    ESHKOL_UNLESS_OP,    // unless - negated when (execute when false)
    ESHKOL_QUOTE_OP,     // quote - literal data
    ESHKOL_QUASIQUOTE_OP,        // quasiquote (`) - template with unquotes
    ESHKOL_UNQUOTE_OP,           // unquote (,) - escape from quasiquote
    ESHKOL_UNQUOTE_SPLICING_OP,  // unquote-splicing (,@) - splice list into quasiquote
    ESHKOL_SET_OP,       // set! - variable mutation
    ESHKOL_DEFINE_TYPE_OP, // define-type - type alias definition
    ESHKOL_IMPORT_OP,    // import - load another Eshkol file (legacy string path)
    ESHKOL_REQUIRE_OP,   // require - import module by symbolic name (new module system)
    ESHKOL_PROVIDE_OP,   // provide - export symbols from module
    // Memory management operators (OALR - Ownership-Aware Lexical Regions)
    ESHKOL_WITH_REGION_OP,  // with-region - lexical region for batch allocation/free
    ESHKOL_OWNED_OP,        // owned - linear type for resources
    ESHKOL_MOVE_OP,         // move - transfer ownership
    ESHKOL_BORROW_OP,       // borrow - temporary read-only access
    ESHKOL_SHARED_OP,       // shared - reference-counted allocation
    ESHKOL_WEAK_REF_OP,     // weak-ref - weak reference (doesn't prevent cleanup)
    ESHKOL_TENSOR_OP,
    ESHKOL_DIFF_OP,
    // Automatic differentiation operators
    ESHKOL_DERIVATIVE_OP,
    ESHKOL_GRADIENT_OP,
    ESHKOL_JACOBIAN_OP,
    ESHKOL_HESSIAN_OP,
    ESHKOL_DIVERGENCE_OP,
    ESHKOL_CURL_OP,
    ESHKOL_LAPLACIAN_OP,
    ESHKOL_DIRECTIONAL_DERIV_OP,
    // HoTT Type System operators
    ESHKOL_TYPE_ANNOTATION_OP,  // (: name type) - standalone type declaration
    ESHKOL_FORALL_OP,           // (forall (a b) type) - polymorphic type
    // Exception handling operators
    ESHKOL_GUARD_OP,            // (guard (var clause ...) body ...) - exception handler
    ESHKOL_RAISE_OP,            // (raise exception) - raise exception
    // Multiple return values operators
    ESHKOL_LET_VALUES_OP,       // (let-values (((vars ...) producer) ...) body)
    ESHKOL_LET_STAR_VALUES_OP,  // (let*-values (((vars ...) producer) ...) body) - sequential
    ESHKOL_VALUES_OP,           // (values v1 v2 ...) - return multiple values
    ESHKOL_CALL_WITH_VALUES_OP, // (call-with-values producer consumer)
    // Macro system operators
    ESHKOL_DEFINE_SYNTAX_OP,    // (define-syntax name (syntax-rules ...))
    ESHKOL_LET_SYNTAX_OP,      // (let-syntax ((name (syntax-rules ...)) ...) body ...)
    ESHKOL_LETREC_SYNTAX_OP,   // (letrec-syntax ((name (syntax-rules ...)) ...) body ...)
    // First-class continuations
    ESHKOL_CALL_CC_OP,         // (call/cc proc) or (call-with-current-continuation proc)
    ESHKOL_DYNAMIC_WIND_OP,    // (dynamic-wind before thunk after)
    // Neuro-symbolic consciousness engine operations
    ESHKOL_LOGIC_VAR_OP,              // ?x - create/reference logic variable
    ESHKOL_UNIFY_OP,                  // (unify t1 t2 subst) -> subst|#f
    ESHKOL_MAKE_SUBST_OP,             // (make-substitution) -> empty subst
    ESHKOL_WALK_OP,                   // (walk term subst) -> resolved term
    ESHKOL_MAKE_FACT_OP,              // (make-fact 'pred arg...) -> fact
    ESHKOL_MAKE_KB_OP,                // (make-kb) -> empty KB
    ESHKOL_KB_ASSERT_OP,              // (kb-assert! kb fact) -> void
    ESHKOL_KB_QUERY_OP,               // (kb-query kb pattern) -> list of substs
    ESHKOL_MAKE_FACTOR_GRAPH_OP,      // (make-factor-graph n-vars dims) -> fg
    ESHKOL_FG_ADD_FACTOR_OP,          // (fg-add-factor! fg vars cpt) -> void
    ESHKOL_FG_INFER_OP,               // (fg-infer! fg iterations) -> beliefs
    ESHKOL_FREE_ENERGY_OP,            // (free-energy beliefs log-joint) -> scalar
    ESHKOL_EXPECTED_FREE_ENERGY_OP,   // (efe model action-var action-state) -> scalar
    ESHKOL_MAKE_WORKSPACE_OP,         // (make-workspace dim max-modules) -> ws
    ESHKOL_WS_REGISTER_OP,            // (ws-register! ws name module) -> void
    ESHKOL_WS_STEP_OP,                // (ws-step! ws) -> broadcast-content
    ESHKOL_FG_UPDATE_CPT_OP,          // (fg-update-cpt! fg factor-idx new-cpt) -> fg
    ESHKOL_FG_OBSERVE_OP,             // (fg-observe! fg var-id observed-state) -> bool
    ESHKOL_LOGIC_VAR_PRED_OP,         // (logic-var? x) -> bool
    ESHKOL_SUBSTITUTION_PRED_OP,      // (substitution? x) -> bool
    ESHKOL_KB_PRED_OP,                // (kb? x) -> bool
    ESHKOL_FACT_PRED_OP,              // (fact? x) -> bool
    ESHKOL_FACTOR_GRAPH_PRED_OP,      // (factor-graph? x) -> bool
    ESHKOL_WORKSPACE_PRED_OP,         // (workspace? x) -> bool
    // ===== R7RS WAVE 3 SPECIAL FORMS =====
    ESHKOL_CASE_LAMBDA_OP,           // (case-lambda ((formals) body ...) ...) - transformed at parse time
    ESHKOL_DEFINE_RECORD_TYPE_OP,    // (define-record-type name ctor pred field ...) - transformed at parse time
    ESHKOL_PARAMETERIZE_OP,          // (parameterize ((param val) ...) body ...) - transformed at parse time
    ESHKOL_MAKE_PARAMETER_OP,        // (make-parameter init) - transformed at parse time
    ESHKOL_COND_EXPAND_OP,           // (cond-expand (feature body ...) ...) - transformed at parse time
    ESHKOL_INCLUDE_OP,               // (include "file" ...) - transformed at parse time
    ESHKOL_SYNTAX_ERROR_OP,          // (syntax-error "msg" datum ...) - handled at parse time
} eshkol_op_t;

struct eshkol_ast;
struct eshkol_operation;

typedef struct eshkol_operation {
    eshkol_op_t op;
    union {
        struct {
            struct eshkol_ast *base;
            struct eshkol_ast *ptr;
        } assign_op;
        struct {
            struct eshkol_ast *func_a;
            struct eshkol_ast *func_b;
        } compose_op;
        struct {
            struct eshkol_operation *if_true;
            struct eshkol_operation *if_false;
        } if_op;
        struct {
            struct eshkol_ast *func;
            struct eshkol_ast *variables;
            uint64_t num_vars;
        } call_op;
        struct {
            char *name;
            struct eshkol_ast *value;
            uint8_t is_function;
            struct eshkol_ast *parameters;
            uint64_t num_params;
            uint8_t is_variadic;      // True if function accepts variable arguments
            char *rest_param;         // Name of rest parameter (for variadic functions)
            uint8_t is_external;      // True if function is external (body from linked .o)
            // HoTT type annotations
            hott_type_expr_t *return_type;    // Return type annotation (NULL if not annotated)
            hott_type_expr_t **param_types;   // Array of parameter type annotations (NULL entries for unannotated)
        } define_op;
        struct {
            struct eshkol_ast *expressions;
            uint64_t num_expressions;
        } sequence_op;
        struct {
            char *name;
            char *real_name;
            char *return_type;
            struct eshkol_ast *parameters;
            uint64_t num_params;
        } extern_op;
        struct {
            char *name;
            char *type;
        } extern_var_op;
	struct {
	           struct eshkol_ast *parameters;
	           uint64_t num_params;
	           struct eshkol_ast *body;
	           struct eshkol_ast *captured_vars;
	           uint64_t num_captured;
	           uint8_t is_variadic;       // True if lambda accepts variable arguments
	           char *rest_param;          // Name of rest parameter (for variadic lambdas)
	           // HoTT type annotations
	           hott_type_expr_t *return_type;    // Return type annotation (NULL if not annotated)
	           hott_type_expr_t **param_types;   // Array of parameter type annotations (NULL entries for unannotated)
	       } lambda_op;
	       struct {
	           struct eshkol_ast *bindings;      // Array of (variable value) pairs
	           uint64_t num_bindings;
	           struct eshkol_ast *body;
	           char *name;                       // Named let: loop name (NULL for regular let)
	           // HoTT type annotations for bindings
	           hott_type_expr_t **binding_types; // Array of type annotations (NULL entries for unannotated)
	       } let_op;
	       struct {
	           char *name;                       // Variable name to mutate
	           struct eshkol_ast *value;         // New value
	       } set_op;
	       struct {
	           char *name;                       // Type alias name
	           hott_type_expr_t *type_expr;      // The type expression this aliases
	           char **type_params;               // Type parameter names (for parameterized types)
	           uint64_t num_type_params;         // Number of type parameters
	       } define_type_op;
	       struct {
	           char *path;                       // Path to file to import
	       } import_op;
	       struct {
	           char **module_names;              // Array of symbolic module names (e.g., "data.json")
	           uint64_t num_modules;             // Number of modules to require
	       } require_op;
	       struct {
	           char **export_names;              // Array of exported symbol names
	           uint64_t num_exports;             // Number of symbols to export
	       } provide_op;
	       // ===== MEMORY MANAGEMENT OPERATIONS (OALR) =====
	       struct {
	           char *name;                       // Optional region name (NULL for anonymous)
	           uint64_t size_hint;               // Optional size hint in bytes (0 for default)
	           struct eshkol_ast *body;          // Body expressions to execute in region
	           uint64_t num_body_exprs;          // Number of body expressions
	       } with_region_op;
	       struct {
	           struct eshkol_ast *value;         // Value to mark as owned
	       } owned_op;
	       struct {
	           struct eshkol_ast *value;         // Value to transfer ownership of
	       } move_op;
	       struct {
	           struct eshkol_ast *value;         // Value to borrow
	           struct eshkol_ast *body;          // Body expressions during borrow
	           uint64_t num_body_exprs;          // Number of body expressions
	       } borrow_op;
	       struct {
	           struct eshkol_ast *value;         // Value to make shared (ref-counted)
	       } shared_op;
	       struct {
	           struct eshkol_ast *value;         // Shared value to create weak ref from
	       } weak_ref_op;
	       // ===== END MEMORY MANAGEMENT OPERATIONS =====
	       struct {
            struct eshkol_ast *elements;
            uint64_t *dimensions;
            uint64_t num_dimensions;
            uint64_t total_elements;
        } tensor_op;
        struct {
            struct eshkol_ast *expression;  // Expression to differentiate
            char *variable;                 // Variable to differentiate with respect to
        } diff_op;
        struct {
            struct eshkol_ast *function;    // Function to differentiate (lambda or function reference)
            struct eshkol_ast *point;       // Point to evaluate derivative at
            uint8_t mode;                   // 0=forward, 1=reverse, 2=auto (for future use)
        } derivative_op;
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate gradient at
        } gradient_op;
        struct {
            struct eshkol_ast *function;    // Vector field function: ℝⁿ → ℝᵐ
            struct eshkol_ast *point;       // Point to evaluate jacobian at
        } jacobian_op;
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate hessian at
        } hessian_op;
        struct {
            struct eshkol_ast *function;    // Vector field function: ℝⁿ → ℝⁿ
            struct eshkol_ast *point;       // Point to evaluate divergence at
        } divergence_op;
        struct {
            struct eshkol_ast *function;    // Vector field function: ℝ³ → ℝ³
            struct eshkol_ast *point;       // Point to evaluate curl at
        } curl_op;
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate laplacian at
        } laplacian_op;
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate directional derivative at
            struct eshkol_ast *direction;   // Direction vector
        } directional_deriv_op;
        // ===== HoTT Type System Operations =====
        struct {
            char *name;                     // Name being annotated
            hott_type_expr_t *type_expr;    // The type expression
        } type_annotation_op;
        struct {
            char **type_vars;               // Quantified type variable names
            uint64_t num_vars;
            hott_type_expr_t *body;         // Body type expression
        } forall_op;
        // ===== EXCEPTION HANDLING OPERATIONS =====
        struct {
            char *var_name;                 // Exception variable name (bound in clauses)
            struct eshkol_ast *clauses;     // Array of guard clauses: ((test expr ...) ...)
            uint64_t num_clauses;           // Number of clauses
            struct eshkol_ast *body;        // Body expressions to evaluate
            uint64_t num_body_exprs;        // Number of body expressions
        } guard_op;
        struct {
            struct eshkol_ast *exception;   // Exception value to raise
        } raise_op;
        // ===== END EXCEPTION HANDLING OPERATIONS =====
        // ===== MULTIPLE RETURN VALUES OPERATIONS =====
        struct {
            struct eshkol_ast *expressions; // Array of expressions to return
            uint64_t num_values;            // Number of values to return
        } values_op;
        struct {
            struct eshkol_ast *producer;    // Thunk that produces multiple values
            struct eshkol_ast *consumer;    // Function that consumes multiple values
        } call_with_values_op;
        struct {
            // Each binding is: ((var1 var2 ...) producer)
            // Stored as array of structs containing var names and producer
            char ***binding_vars;           // Array of arrays of variable names
            uint64_t *binding_var_counts;   // Count of vars per binding
            struct eshkol_ast *producers;   // Array of producer expressions
            uint64_t num_bindings;          // Number of bindings
            struct eshkol_ast *body;        // Body expression
        } let_values_op;
        // ===== END MULTIPLE RETURN VALUES OPERATIONS =====
        // ===== PATTERN MATCHING OPERATIONS =====
        struct {
            struct eshkol_ast *expr;             // Expression to match against
            eshkol_match_clause_t *clauses;      // Array of match clauses
            uint64_t num_clauses;                // Number of clauses
        } match_op;
        // ===== END PATTERN MATCHING OPERATIONS =====

        // ===== MACRO OPERATIONS =====
        struct {
            eshkol_macro_def_t *macro;           // Macro definition
        } define_syntax_op;
        struct {
            eshkol_macro_def_t **macros;         // Array of macro definitions
            uint64_t num_macros;                 // Number of macro bindings
            struct eshkol_ast *body;             // Body expression
        } let_syntax_op;
        // ===== END MACRO OPERATIONS =====
        // ===== CONTINUATION OPERATIONS =====
        struct {
            struct eshkol_ast *proc;             // Procedure to call with continuation
        } call_cc_op;
        struct {
            struct eshkol_ast *before;           // Before thunk
            struct eshkol_ast *thunk;            // Body thunk
            struct eshkol_ast *after;            // After thunk
        } dynamic_wind_op;
        // ===== END CONTINUATION OPERATIONS =====
        // ===== NEURO-SYMBOLIC CONSCIOUSNESS ENGINE OPERATIONS =====
        struct {
            uint64_t var_id;                     // Logic variable ID (global registry)
            const char *name;                    // Logic variable name (e.g., "?x")
        } logic_var_op;
        // ===== R7RS WAVE 3 OPERATIONS =====
        struct {
            // Each clause: (formals body ...)
            // formals stored as lambda_op sub-ASTs
            struct eshkol_ast *clauses;          // Array of lambda ASTs (one per clause)
            uint64_t num_clauses;                // Number of arity clauses
        } case_lambda_op;
        struct {
            // (parameterize ((param1 val1) (param2 val2) ...) body ...)
            struct eshkol_ast *params;           // Array of parameter expressions
            struct eshkol_ast *values;           // Array of value expressions
            uint64_t num_bindings;               // Number of parameter bindings
            struct eshkol_ast *body;             // Body expression
        } parameterize_op;
        // ===== END R7RS WAVE 3 OPERATIONS =====
    };
} eshkol_operations_t;

typedef struct eshkol_ast {
    eshkol_type_t type;
    union {
        void *untyped_data;
        uint8_t uint8_val;
        uint16_t uint16_val;
        uint32_t uint32_val;
        uint64_t uint64_val;
        int8_t int8_val;
        int16_t int16_val;
        int32_t int32_val;
        int64_t int64_val;
        double double_val;
        struct {
            char *ptr;
            uint64_t size;
        } str_val;
        struct {
            char *id;
            uint8_t is_lambda;
            eshkol_operations_t *func_commands;
            struct eshkol_ast *variables;
            uint64_t num_variables;
            uint64_t size;
            uint8_t is_variadic;      // True if function accepts variable arguments
            char *rest_param;         // Name of rest parameter (for variadic functions)
            // HoTT type annotations (for inline annotations like (x : int))
            hott_type_expr_t **param_types;   // Array of parameter type annotations
            hott_type_expr_t *return_type;    // Return type annotation (optional)
        } eshkol_func;
        struct {
            char *id;
            struct eshkol_ast *data;
        } variable;
        struct {
            struct eshkol_ast *car;
            struct eshkol_ast *cdr;
        } cons_cell;
        struct {
            struct eshkol_ast *elements;
            uint64_t *dimensions;
            uint64_t num_dimensions;
            uint64_t total_elements;
        } tensor_val;
        eshkol_operations_t operation;
    };
    // HoTT type system: inferred type from type checker
    // Packed format: bits 0-15 = TypeId.id, bits 16-23 = universe level, bits 24-31 = flags
    // Value 0 means "not yet type-checked"
    uint32_t inferred_hott_type;

    // Source location for error reporting
    uint32_t line;      // 1-based line number (0 = unknown)
    uint32_t column;    // 1-based column number (0 = unknown)
} eshkol_ast_t;

// ===== Unified AST Literal Builders =====
// These set both the AST type fields AND inferred_hott_type for consistent type tracking.
// Packed HoTT TypeId format: bits 0-15 = id, bits 16-23 = universe, bits 24-31 = flags

static inline void eshkol_ast_make_int64(eshkol_ast_t* node, int64_t val) {
    node->type = ESHKOL_INT64;
    node->int64_val = val;
    node->inferred_hott_type = 13; // BuiltinTypes::Int64
}

static inline void eshkol_ast_make_double(eshkol_ast_t* node, double val) {
    node->type = ESHKOL_DOUBLE;
    node->double_val = val;
    node->inferred_hott_type = 16; // BuiltinTypes::Float64
}

static inline void eshkol_ast_make_bool(eshkol_ast_t* node, bool val) {
    node->type = ESHKOL_BOOL;
    node->int64_val = val ? 1 : 0;
    node->inferred_hott_type = 25; // BuiltinTypes::Boolean
}

static inline void eshkol_ast_make_char(eshkol_ast_t* node, int64_t val) {
    node->type = ESHKOL_CHAR;
    node->int64_val = val;
    node->inferred_hott_type = 22; // BuiltinTypes::Char
}

static inline void eshkol_ast_make_null(eshkol_ast_t* node) {
    node->type = ESHKOL_NULL;
    node->int64_val = 0;
    node->inferred_hott_type = 26; // BuiltinTypes::Null
}

static inline void eshkol_ast_make_string(eshkol_ast_t* node, const char* ptr, uint64_t size) {
    node->type = ESHKOL_STRING;
    node->str_val.ptr = (char*)ptr;
    node->str_val.size = size;
    node->inferred_hott_type = 21; // BuiltinTypes::String
}

static inline void eshkol_ast_make_symbol(eshkol_ast_t* node, const char* ptr, uint64_t size) {
    node->type = ESHKOL_SYMBOL;
    node->str_val.ptr = (char*)ptr;
    node->str_val.size = size;
    node->inferred_hott_type = 27; // BuiltinTypes::Symbol
}

void eshkol_ast_clean(eshkol_ast_t *ast);
void eshkol_ast_pretty_print(const eshkol_ast_t *ast, int indent);

// Symbolic differentiation AST helpers
eshkol_ast_t* eshkol_alloc_symbolic_ast(void);
eshkol_ast_t* eshkol_make_var_ast(const char* name);
eshkol_ast_t* eshkol_make_int_ast(int64_t value);
eshkol_ast_t* eshkol_make_double_ast(double value);
eshkol_ast_t* eshkol_make_binary_op_ast(const char* op, eshkol_ast_t* left, eshkol_ast_t* right);
eshkol_ast_t* eshkol_make_unary_call_ast(const char* func, eshkol_ast_t* arg);
eshkol_ast_t* eshkol_copy_ast(const eshkol_ast_t* ast);

// REPL display helper
eshkol_ast_t* eshkol_wrap_with_display(eshkol_ast_t* expr);

// ===== HoTT Type Expression Helpers =====
// Create primitive type expressions
hott_type_expr_t* hott_make_integer_type(void);
hott_type_expr_t* hott_make_real_type(void);
hott_type_expr_t* hott_make_boolean_type(void);
hott_type_expr_t* hott_make_string_type(void);
hott_type_expr_t* hott_make_char_type(void);
hott_type_expr_t* hott_make_symbol_type(void);
hott_type_expr_t* hott_make_null_type(void);
hott_type_expr_t* hott_make_any_type(void);
hott_type_expr_t* hott_make_nothing_type(void);

// Create type variables
hott_type_expr_t* hott_make_type_var(const char* name);

// Create compound types
hott_type_expr_t* hott_make_arrow_type(hott_type_expr_t** param_types, uint64_t num_params, hott_type_expr_t* return_type);
hott_type_expr_t* hott_make_list_type(hott_type_expr_t* element_type);
hott_type_expr_t* hott_make_vector_type(hott_type_expr_t* element_type);
hott_type_expr_t* hott_make_tensor_type(hott_type_expr_t* element_type);
hott_type_expr_t* hott_make_pair_type(hott_type_expr_t* left, hott_type_expr_t* right);
hott_type_expr_t* hott_make_product_type(hott_type_expr_t* left, hott_type_expr_t* right);
hott_type_expr_t* hott_make_sum_type(hott_type_expr_t* left, hott_type_expr_t* right);
hott_type_expr_t* hott_make_forall_type(char** type_vars, uint64_t num_vars, hott_type_expr_t* body);

// Copy and free type expressions
hott_type_expr_t* hott_copy_type_expr(const hott_type_expr_t* type);
void hott_free_type_expr(hott_type_expr_t* type);

// Type expression to string (for display/error messages)
char* hott_type_to_string(const hott_type_expr_t* type);

// ===== Inferred HoTT Type Helpers =====
// Pack/unpack TypeId to/from uint32_t for AST storage
// Format: bits 0-15 = id, bits 16-23 = universe, bits 24-31 = flags

static inline uint32_t hott_pack_type_id(uint16_t id, uint8_t universe, uint8_t flags) {
    return (uint32_t)id | ((uint32_t)universe << 16) | ((uint32_t)flags << 24);
}

static inline uint16_t hott_unpack_type_id(uint32_t packed) {
    return (uint16_t)(packed & 0xFFFF);
}

static inline uint8_t hott_unpack_universe(uint32_t packed) {
    return (uint8_t)((packed >> 16) & 0xFF);
}

static inline uint8_t hott_unpack_flags(uint32_t packed) {
    return (uint8_t)((packed >> 24) & 0xFF);
}

static inline int hott_type_is_set(uint32_t packed) {
    return packed != 0;
}

#ifdef __cplusplus
};

// Parse next AST from file stream
eshkol_ast_t eshkol_parse_next_ast(std::ifstream &in_file);

// Parse next AST from any input stream (including string streams for stdlib)
eshkol_ast_t eshkol_parse_next_ast_from_stream(std::istream &in_stream);

#endif

#endif
