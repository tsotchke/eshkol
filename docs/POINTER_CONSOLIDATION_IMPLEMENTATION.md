# Pointer Consolidation Implementation Strategy

## Technical Specification for Eshkol Type System Refactoring

**Version:** 1.0
**Date:** 2025-12-09
**Status:** Implementation Guide
**Scope:** Consolidation of pointer types and expansion to 8-bit type field

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current System Analysis](#2-current-system-analysis)
3. [Target Architecture](#3-target-architecture)
4. [Implementation Phases](#4-implementation-phases)
   - 4.1 [Phase 1: Header Infrastructure](#41-phase-1-header-infrastructure)
   - 4.2 [Phase 2: Type Enum Refactoring](#42-phase-2-type-enum-refactoring)
   - 4.3 [Phase 3: Arena Memory Updates](#43-phase-3-arena-memory-updates)
   - 4.4 [Phase 4: Codegen Migration](#44-phase-4-codegen-migration)
   - 4.5 [Phase 5: Display System Updates](#45-phase-5-display-system-updates)
   - 4.6 [Phase 6: Testing and Validation](#46-phase-6-testing-and-validation)
5. [File-by-File Migration Guide](#5-file-by-file-migration-guide)
6. [Compatibility Layer](#6-compatibility-layer)
7. [Testing Strategy](#7-testing-strategy)
8. [Rollback Plan](#8-rollback-plan)
9. [Success Criteria](#9-success-criteria)

---

## 1. Executive Summary

This document provides a complete implementation strategy for consolidating Eshkol's pointer-based types from the current 4-bit type system (16 types, all slots exhausted) to an 8-bit type system with subtype headers. This refactoring:

1. **Consolidates 8 pointer types** into 2 supertype categories (HEAP_PTR, CALLABLE)
2. **Frees type slots 10-15** for future use
3. **Enables multimedia types** (HANDLE, BUFFER, STREAM, EVENT) at slots 16-19
4. **Introduces object headers** with subtype, flags, and metadata
5. **Maintains backward compatibility** during transition via compatibility macros

The refactoring is designed to be completed incrementally across multiple sessions, with each phase producing a working system that passes all existing tests.

---

## 2. Current System Analysis

### 2.1 Current Type Enum (eshkol.h:49-67)

```c
typedef enum {
    ESHKOL_VALUE_NULL        = 0,   // Empty/null value
    ESHKOL_VALUE_INT64       = 1,   // 64-bit signed integer
    ESHKOL_VALUE_DOUBLE      = 2,   // Double-precision floating point
    ESHKOL_VALUE_CONS_PTR    = 3,   // Pointer to another cons cell
    ESHKOL_VALUE_DUAL_NUMBER = 4,   // Dual number for forward-mode AD
    ESHKOL_VALUE_AD_NODE_PTR = 5,   // Pointer to AD computation graph node
    ESHKOL_VALUE_TENSOR_PTR  = 6,   // Pointer to tensor structure
    ESHKOL_VALUE_LAMBDA_SEXPR = 7,  // Lambda S-expression metadata
    ESHKOL_VALUE_STRING_PTR  = 8,   // Pointer to string
    ESHKOL_VALUE_CHAR        = 9,   // Character
    ESHKOL_VALUE_VECTOR_PTR  = 10,  // Pointer to Scheme vector
    ESHKOL_VALUE_SYMBOL      = 11,  // Symbol
    ESHKOL_VALUE_CLOSURE_PTR = 12,  // Pointer to closure
    ESHKOL_VALUE_BOOL        = 13,  // Boolean
    ESHKOL_VALUE_HASH_PTR    = 14,  // Pointer to hash table
    ESHKOL_VALUE_EXCEPTION   = 15,  // Pointer to exception object
    // Note: 4-bit type field limit reached (0x0F mask)
} eshkol_value_type_t;
```

### 2.2 Current Type Checking Pattern

All type checks currently use a 4-bit mask:

```c
#define ESHKOL_IS_CONS_PTR_TYPE(type)    (((type) & 0x0F) == ESHKOL_VALUE_CONS_PTR)
#define ESHKOL_IS_STRING_PTR_TYPE(type)  (((type) & 0x0F) == ESHKOL_VALUE_STRING_PTR)
// ... etc
```

### 2.3 Current Tagged Value Structure

```c
typedef struct eshkol_tagged_value {
    uint8_t type;        // Value type (4-bit used, 8-bit available)
    uint8_t flags;       // Exactness and other flags
    uint16_t reserved;   // Reserved for future use
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
        uint64_t raw_val;
    } data;
} eshkol_tagged_value_t;
```

### 2.4 Usage Statistics (from codebase analysis)

| Type Constant | Occurrences in .cpp files |
|---------------|---------------------------|
| ESHKOL_VALUE_CONS_PTR | ~95 |
| ESHKOL_VALUE_CLOSURE_PTR | ~75 |
| ESHKOL_VALUE_VECTOR_PTR | ~60 |
| ESHKOL_VALUE_STRING_PTR | ~55 |
| ESHKOL_VALUE_TENSOR_PTR | ~45 |
| ESHKOL_VALUE_EXCEPTION | ~25 |
| ESHKOL_VALUE_HASH_PTR | ~20 |
| ESHKOL_VALUE_AD_NODE_PTR | ~15 |
| ESHKOL_VALUE_LAMBDA_SEXPR | ~15 |
| **Total pointer type refs** | **~405** |

### 2.5 Files Requiring Modification

| File | Type Refs | Priority |
|------|-----------|----------|
| lib/backend/llvm_codegen.cpp | 261 | HIGH |
| lib/backend/homoiconic_codegen.cpp | 47 | HIGH |
| lib/backend/collection_codegen.cpp | 28 | HIGH |
| lib/backend/tensor_codegen.cpp | 25 | MEDIUM |
| lib/backend/arithmetic_codegen.cpp | 18 | MEDIUM |
| lib/backend/string_io_codegen.cpp | 17 | MEDIUM |
| lib/core/arena_memory.cpp | 16 | HIGH |
| lib/types/hott_types.cpp | 12 | LOW |
| lib/backend/system_codegen.cpp | 7 | LOW |
| lib/backend/map_codegen.cpp | 6 | LOW |
| lib/backend/hash_codegen.cpp | 4 | LOW |
| lib/backend/call_apply_codegen.cpp | 3 | LOW |
| lib/backend/function_codegen.cpp | 2 | LOW |
| lib/backend/binding_codegen.cpp | 1 | LOW |

---

## 3. Target Architecture

### 3.1 New Type Enum

```c
typedef enum {
    // ═══════════════════════════════════════════════════════════
    // IMMEDIATE VALUES (0-7) - data stored directly in tagged value
    // No heap allocation, no header needed
    // ═══════════════════════════════════════════════════════════
    ESHKOL_VALUE_NULL        = 0,   // Empty/null value
    ESHKOL_VALUE_INT64       = 1,   // 64-bit signed integer
    ESHKOL_VALUE_DOUBLE      = 2,   // Double-precision float
    ESHKOL_VALUE_BOOL        = 3,   // Boolean (#t/#f)
    ESHKOL_VALUE_CHAR        = 4,   // Unicode character (codepoint in data)
    ESHKOL_VALUE_SYMBOL      = 5,   // Interned symbol ID
    ESHKOL_VALUE_DUAL_NUMBER = 6,   // Forward-mode AD dual number
    // Reserved: 7

    // ═══════════════════════════════════════════════════════════
    // HEAP POINTERS (8-9) - consolidated, subtype in object header
    // All heap-allocated objects with eshkol_object_header_t prefix
    // ═══════════════════════════════════════════════════════════
    ESHKOL_VALUE_HEAP_PTR    = 8,   // All heap-allocated data objects
    ESHKOL_VALUE_CALLABLE    = 9,   // All callable objects

    // Reserved: 10-15 (for future core types)

    // ═══════════════════════════════════════════════════════════
    // MULTIMEDIA TYPES (16-19) - new category, subtype in header
    // Linear resources with explicit lifecycle management
    // ═══════════════════════════════════════════════════════════
    ESHKOL_VALUE_HANDLE      = 16,  // Managed resource handles
    ESHKOL_VALUE_BUFFER      = 17,  // Typed data buffers
    ESHKOL_VALUE_STREAM      = 18,  // Lazy data streams
    ESHKOL_VALUE_EVENT       = 19,  // Input/system events

    // Reserved: 20-31 (for future multimedia expansion)

    // ═══════════════════════════════════════════════════════════
    // ADVANCED TYPES (32+) - HoTT and type theory constructs
    // ═══════════════════════════════════════════════════════════
    ESHKOL_VALUE_PATH        = 32,  // Identity/path types
    ESHKOL_VALUE_FIBER       = 33,  // Fiber types
    ESHKOL_VALUE_UNIVERSE    = 34,  // Universe levels

} eshkol_value_type_t;
```

### 3.2 Subtype Enumerations

```c
// ═══════════════════════════════════════════════════════════
// HEAP_PTR SUBTYPES (type = ESHKOL_VALUE_HEAP_PTR = 8)
// Data structures allocated on the arena
// ═══════════════════════════════════════════════════════════
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
    // Reserved: 10-255
} heap_subtype_t;

// ═══════════════════════════════════════════════════════════
// CALLABLE SUBTYPES (type = ESHKOL_VALUE_CALLABLE = 9)
// Objects that can be invoked as functions
// ═══════════════════════════════════════════════════════════
typedef enum {
    CALLABLE_SUBTYPE_CLOSURE      = 0,  // Lambda + captured environment
    CALLABLE_SUBTYPE_LAMBDA_SEXPR = 1,  // S-expression representation
    CALLABLE_SUBTYPE_AD_NODE      = 2,  // Autodiff computation node
    CALLABLE_SUBTYPE_CONTINUATION = 3,  // First-class continuation
    CALLABLE_SUBTYPE_BUILTIN      = 4,  // Built-in primitive function
    CALLABLE_SUBTYPE_GENERIC      = 5,  // Generic function (multimethod)
    // Reserved: 6-255
} callable_subtype_t;

// ═══════════════════════════════════════════════════════════
// HANDLE SUBTYPES (type = ESHKOL_VALUE_HANDLE = 16)
// Linear resources requiring explicit lifecycle management
// ═══════════════════════════════════════════════════════════
typedef enum {
    HANDLE_SUBTYPE_WINDOW       = 0,   // GUI window
    HANDLE_SUBTYPE_SOCKET_TCP   = 1,   // TCP socket
    HANDLE_SUBTYPE_SOCKET_UDP   = 2,   // UDP socket
    HANDLE_SUBTYPE_AUDIO_OUT    = 3,   // Audio output device
    HANDLE_SUBTYPE_AUDIO_IN     = 4,   // Audio input device
    HANDLE_SUBTYPE_FILE         = 5,   // File handle
    HANDLE_SUBTYPE_MOTOR        = 6,   // Robotics: motor actuator
    HANDLE_SUBTYPE_SENSOR       = 7,   // Robotics: sensor
    HANDLE_SUBTYPE_GPIO         = 8,   // Embedded: GPIO pin
    HANDLE_SUBTYPE_I2C          = 9,   // Embedded: I2C bus
    HANDLE_SUBTYPE_SPI          = 10,  // Embedded: SPI bus
    HANDLE_SUBTYPE_UART         = 11,  // Embedded: UART/serial
    HANDLE_SUBTYPE_CAMERA       = 12,  // Camera device
    HANDLE_SUBTYPE_DISPLAY      = 13,  // Display/framebuffer
    // Reserved: 14-255
} handle_subtype_t;

// ═══════════════════════════════════════════════════════════
// BUFFER SUBTYPES (type = ESHKOL_VALUE_BUFFER = 17)
// Typed contiguous data arrays with known element types
// ═══════════════════════════════════════════════════════════
typedef enum {
    BUFFER_SUBTYPE_BYTE         = 0,   // Raw bytes (uint8)
    BUFFER_SUBTYPE_INT8         = 1,   // Signed 8-bit integers
    BUFFER_SUBTYPE_INT16        = 2,   // Signed 16-bit integers
    BUFFER_SUBTYPE_INT32        = 3,   // Signed 32-bit integers
    BUFFER_SUBTYPE_INT64        = 4,   // Signed 64-bit integers
    BUFFER_SUBTYPE_UINT8        = 5,   // Unsigned 8-bit integers
    BUFFER_SUBTYPE_UINT16       = 6,   // Unsigned 16-bit integers
    BUFFER_SUBTYPE_UINT32       = 7,   // Unsigned 32-bit integers
    BUFFER_SUBTYPE_UINT64       = 8,   // Unsigned 64-bit integers
    BUFFER_SUBTYPE_FLOAT32      = 9,   // 32-bit floats
    BUFFER_SUBTYPE_FLOAT64      = 10,  // 64-bit floats
    BUFFER_SUBTYPE_PIXEL_RGBA8  = 11,  // RGBA 8-bit per channel
    BUFFER_SUBTYPE_PIXEL_RGB8   = 12,  // RGB 8-bit per channel
    BUFFER_SUBTYPE_PIXEL_GRAY8  = 13,  // Grayscale 8-bit
    BUFFER_SUBTYPE_PIXEL_RGBA16 = 14,  // RGBA 16-bit per channel
    BUFFER_SUBTYPE_PIXEL_FLOAT  = 15,  // Float per channel (HDR)
    BUFFER_SUBTYPE_SAMPLE_I16   = 16,  // Audio: 16-bit PCM
    BUFFER_SUBTYPE_SAMPLE_I24   = 17,  // Audio: 24-bit PCM
    BUFFER_SUBTYPE_SAMPLE_F32   = 18,  // Audio: 32-bit float
    BUFFER_SUBTYPE_COMPLEX64    = 19,  // Complex float (FFT)
    BUFFER_SUBTYPE_COMPLEX128   = 20,  // Complex double
    BUFFER_SUBTYPE_BOOL         = 21,  // Packed booleans
    // Reserved: 22-255
} buffer_subtype_t;

// ═══════════════════════════════════════════════════════════
// STREAM SUBTYPES (type = ESHKOL_VALUE_STREAM = 18)
// Lazy, potentially infinite data sequences
// ═══════════════════════════════════════════════════════════
typedef enum {
    STREAM_SUBTYPE_BYTE         = 0,   // Byte stream (file, network)
    STREAM_SUBTYPE_CHAR         = 1,   // Character stream (text)
    STREAM_SUBTYPE_LINE         = 2,   // Line-buffered text stream
    STREAM_SUBTYPE_TOKEN        = 3,   // Tokenized stream
    STREAM_SUBTYPE_AUDIO        = 4,   // Audio sample stream
    STREAM_SUBTYPE_VIDEO        = 5,   // Video frame stream
    STREAM_SUBTYPE_SENSOR       = 6,   // Sensor data stream
    STREAM_SUBTYPE_EVENT        = 7,   // Event stream
    STREAM_SUBTYPE_VALUE        = 8,   // Generic tagged value stream
    // Reserved: 9-255
} stream_subtype_t;

// ═══════════════════════════════════════════════════════════
// EVENT SUBTYPES (type = ESHKOL_VALUE_EVENT = 19)
// Discrete input and system notifications
// ═══════════════════════════════════════════════════════════
typedef enum {
    EVENT_SUBTYPE_KEY           = 0,   // Keyboard event
    EVENT_SUBTYPE_MOUSE_MOVE    = 1,   // Mouse motion
    EVENT_SUBTYPE_MOUSE_BUTTON  = 2,   // Mouse click
    EVENT_SUBTYPE_MOUSE_SCROLL  = 3,   // Mouse scroll wheel
    EVENT_SUBTYPE_TOUCH_BEGIN   = 4,   // Touch start
    EVENT_SUBTYPE_TOUCH_MOVE    = 5,   // Touch drag
    EVENT_SUBTYPE_TOUCH_END     = 6,   // Touch end
    EVENT_SUBTYPE_WINDOW_RESIZE = 7,   // Window resized
    EVENT_SUBTYPE_WINDOW_CLOSE  = 8,   // Window close requested
    EVENT_SUBTYPE_WINDOW_FOCUS  = 9,   // Window focus changed
    EVENT_SUBTYPE_NETWORK_READY = 10,  // Socket ready for I/O
    EVENT_SUBTYPE_NETWORK_CLOSE = 11,  // Socket closed
    EVENT_SUBTYPE_TIMER         = 12,  // Timer expired
    EVENT_SUBTYPE_SENSOR        = 13,  // Sensor trigger
    EVENT_SUBTYPE_SIGNAL        = 14,  // OS signal
    EVENT_SUBTYPE_CUSTOM        = 15,  // User-defined event
    // Reserved: 16-255
} event_subtype_t;
```

### 3.3 Universal Object Header

```c
// ═══════════════════════════════════════════════════════════
// UNIVERSAL OBJECT HEADER
// All heap-allocated objects (HEAP_PTR, CALLABLE, HANDLE,
// BUFFER, STREAM, EVENT) begin with this 8-byte header
// ═══════════════════════════════════════════════════════════
typedef struct eshkol_object_header {
    uint8_t  subtype;     // Object-specific subtype (heap_subtype_t, etc.)
    uint8_t  flags;       // Object flags (see below)
    uint16_t aux;         // Type-specific auxiliary data
    uint32_t size;        // Total object size in bytes (for GC/debugging)
} eshkol_object_header_t;

// Verify header is 8 bytes for alignment
_Static_assert(sizeof(eshkol_object_header_t) == 8,
               "Object header must be 8 bytes for alignment");

// ═══════════════════════════════════════════════════════════
// OBJECT FLAGS
// Applicable to all heap-allocated objects
// ═══════════════════════════════════════════════════════════
#define OBJECT_FLAG_NONE       0x00  // No special flags
#define OBJECT_FLAG_LINEAR     0x01  // Linear resource (must consume exactly once)
#define OBJECT_FLAG_BORROWED   0x02  // Currently borrowed (temporary reference)
#define OBJECT_FLAG_SHARED     0x04  // Reference counted (shared ownership)
#define OBJECT_FLAG_WEAK       0x08  // Weak reference (doesn't prevent collection)
#define OBJECT_FLAG_IMMUTABLE  0x10  // Cannot be modified after creation
#define OBJECT_FLAG_PINNED     0x20  // Don't move during GC compaction
#define OBJECT_FLAG_MARKED     0x40  // GC mark bit (internal use)
#define OBJECT_FLAG_FINALIZED  0x80  // Destructor has been called

// ═══════════════════════════════════════════════════════════
// AUXILIARY FIELD USAGE BY TYPE
// ═══════════════════════════════════════════════════════════
// HEAP_PTR:
//   - CONS:        unused (0)
//   - STRING:      encoding (0=UTF-8, 1=ASCII, 2=Latin-1)
//   - VECTOR:      unused (0)
//   - MULTI_VALUE: number of values
//   - TENSOR:      number of dimensions (1-4)
//   - HASH:        load factor threshold (as percentage)
//   - EXCEPTION:   exception type code
//
// CALLABLE:
//   - CLOSURE:     arity (number of parameters)
//   - AD_NODE:     node type (add, mul, sin, etc.)
//
// HANDLE:
//   - All:         platform-specific flags
//
// BUFFER:
//   - All:         element size in bytes
//
// STREAM:
//   - All:         buffer size hint
//
// EVENT:
//   - All:         timestamp (milliseconds, low 16 bits)
```

### 3.4 Type Checking Macros (New System)

```c
// ═══════════════════════════════════════════════════════════
// FAST TYPE CHECKS (no dereference needed)
// These check only the type field in the tagged value
// ═══════════════════════════════════════════════════════════
#define ESHKOL_IS_NULL(t)        ((t).type == ESHKOL_VALUE_NULL)
#define ESHKOL_IS_INT64(t)       ((t).type == ESHKOL_VALUE_INT64)
#define ESHKOL_IS_DOUBLE(t)      ((t).type == ESHKOL_VALUE_DOUBLE)
#define ESHKOL_IS_BOOL(t)        ((t).type == ESHKOL_VALUE_BOOL)
#define ESHKOL_IS_CHAR(t)        ((t).type == ESHKOL_VALUE_CHAR)
#define ESHKOL_IS_SYMBOL(t)      ((t).type == ESHKOL_VALUE_SYMBOL)
#define ESHKOL_IS_DUAL_NUMBER(t) ((t).type == ESHKOL_VALUE_DUAL_NUMBER)
#define ESHKOL_IS_HEAP_PTR(t)    ((t).type == ESHKOL_VALUE_HEAP_PTR)
#define ESHKOL_IS_CALLABLE(t)    ((t).type == ESHKOL_VALUE_CALLABLE)
#define ESHKOL_IS_HANDLE(t)      ((t).type == ESHKOL_VALUE_HANDLE)
#define ESHKOL_IS_BUFFER(t)      ((t).type == ESHKOL_VALUE_BUFFER)
#define ESHKOL_IS_STREAM(t)      ((t).type == ESHKOL_VALUE_STREAM)
#define ESHKOL_IS_EVENT(t)       ((t).type == ESHKOL_VALUE_EVENT)

// Check for any pointer type (requires heap access)
#define ESHKOL_IS_ANY_PTR(t)     ((t).type >= ESHKOL_VALUE_HEAP_PTR)

// ═══════════════════════════════════════════════════════════
// SUBTYPE ACCESS HELPERS
// ═══════════════════════════════════════════════════════════
static inline eshkol_object_header_t* eshkol_get_header(eshkol_tagged_value_t t) {
    return (eshkol_object_header_t*)t.data.ptr_val;
}

static inline uint8_t eshkol_get_subtype(eshkol_tagged_value_t t) {
    eshkol_object_header_t* hdr = eshkol_get_header(t);
    return hdr ? hdr->subtype : 0;
}

static inline uint8_t eshkol_get_object_flags(eshkol_tagged_value_t t) {
    eshkol_object_header_t* hdr = eshkol_get_header(t);
    return hdr ? hdr->flags : 0;
}

static inline uint16_t eshkol_get_aux(eshkol_tagged_value_t t) {
    eshkol_object_header_t* hdr = eshkol_get_header(t);
    return hdr ? hdr->aux : 0;
}

static inline uint32_t eshkol_get_object_size(eshkol_tagged_value_t t) {
    eshkol_object_header_t* hdr = eshkol_get_header(t);
    return hdr ? hdr->size : 0;
}

// ═══════════════════════════════════════════════════════════
// HEAP_PTR SUBTYPE CHECKS (require header read)
// ═══════════════════════════════════════════════════════════
static inline bool eshkol_is_cons(eshkol_tagged_value_t t) {
    return ESHKOL_IS_HEAP_PTR(t) && eshkol_get_subtype(t) == HEAP_SUBTYPE_CONS;
}

static inline bool eshkol_is_string(eshkol_tagged_value_t t) {
    return ESHKOL_IS_HEAP_PTR(t) && eshkol_get_subtype(t) == HEAP_SUBTYPE_STRING;
}

static inline bool eshkol_is_vector(eshkol_tagged_value_t t) {
    return ESHKOL_IS_HEAP_PTR(t) && eshkol_get_subtype(t) == HEAP_SUBTYPE_VECTOR;
}

static inline bool eshkol_is_multi_value(eshkol_tagged_value_t t) {
    return ESHKOL_IS_HEAP_PTR(t) && eshkol_get_subtype(t) == HEAP_SUBTYPE_MULTI_VALUE;
}

static inline bool eshkol_is_tensor(eshkol_tagged_value_t t) {
    return ESHKOL_IS_HEAP_PTR(t) && eshkol_get_subtype(t) == HEAP_SUBTYPE_TENSOR;
}

static inline bool eshkol_is_hash(eshkol_tagged_value_t t) {
    return ESHKOL_IS_HEAP_PTR(t) && eshkol_get_subtype(t) == HEAP_SUBTYPE_HASH;
}

static inline bool eshkol_is_exception(eshkol_tagged_value_t t) {
    return ESHKOL_IS_HEAP_PTR(t) && eshkol_get_subtype(t) == HEAP_SUBTYPE_EXCEPTION;
}

// ═══════════════════════════════════════════════════════════
// CALLABLE SUBTYPE CHECKS
// ═══════════════════════════════════════════════════════════
static inline bool eshkol_is_closure(eshkol_tagged_value_t t) {
    return ESHKOL_IS_CALLABLE(t) && eshkol_get_subtype(t) == CALLABLE_SUBTYPE_CLOSURE;
}

static inline bool eshkol_is_lambda_sexpr(eshkol_tagged_value_t t) {
    return ESHKOL_IS_CALLABLE(t) && eshkol_get_subtype(t) == CALLABLE_SUBTYPE_LAMBDA_SEXPR;
}

static inline bool eshkol_is_ad_node(eshkol_tagged_value_t t) {
    return ESHKOL_IS_CALLABLE(t) && eshkol_get_subtype(t) == CALLABLE_SUBTYPE_AD_NODE;
}

// ═══════════════════════════════════════════════════════════
// OBJECT FLAG CHECKS
// ═══════════════════════════════════════════════════════════
static inline bool eshkol_is_linear(eshkol_tagged_value_t t) {
    return ESHKOL_IS_ANY_PTR(t) && (eshkol_get_object_flags(t) & OBJECT_FLAG_LINEAR);
}

static inline bool eshkol_is_borrowed(eshkol_tagged_value_t t) {
    return ESHKOL_IS_ANY_PTR(t) && (eshkol_get_object_flags(t) & OBJECT_FLAG_BORROWED);
}

static inline bool eshkol_is_immutable(eshkol_tagged_value_t t) {
    return ESHKOL_IS_ANY_PTR(t) && (eshkol_get_object_flags(t) & OBJECT_FLAG_IMMUTABLE);
}

// ═══════════════════════════════════════════════════════════
// LIST/SEQUENCE TYPE CHECKS (combines null and cons)
// ═══════════════════════════════════════════════════════════
static inline bool eshkol_is_list(eshkol_tagged_value_t t) {
    return ESHKOL_IS_NULL(t) || eshkol_is_cons(t);
}

static inline bool eshkol_is_proper_list(eshkol_tagged_value_t t) {
    // Traverse to check for proper nil termination
    while (eshkol_is_cons(t)) {
        eshkol_cons_t* cons = (eshkol_cons_t*)t.data.ptr_val;
        t = cons->cdr;
    }
    return ESHKOL_IS_NULL(t);
}

// ═══════════════════════════════════════════════════════════
// NUMERIC TYPE CHECKS
// ═══════════════════════════════════════════════════════════
static inline bool eshkol_is_number(eshkol_tagged_value_t t) {
    return ESHKOL_IS_INT64(t) || ESHKOL_IS_DOUBLE(t);
}

static inline bool eshkol_is_exact(eshkol_tagged_value_t t) {
    return ESHKOL_IS_INT64(t);
}

static inline bool eshkol_is_inexact(eshkol_tagged_value_t t) {
    return ESHKOL_IS_DOUBLE(t);
}
```

---

## 4. Implementation Phases

### 4.1 Phase 1: Header Infrastructure

**Goal:** Add object header structure and helper functions without breaking existing code.

**Duration:** ~1 hour

**Files to modify:**
- `inc/eshkol/eshkol.h`

**Changes:**

1. Add `eshkol_object_header_t` structure definition:

```c
// Add after line 97 (after eshkol_tagged_value_t)

// ═══════════════════════════════════════════════════════════
// UNIVERSAL OBJECT HEADER (Phase 1 of consolidation)
// Will be prepended to all heap-allocated objects
// ═══════════════════════════════════════════════════════════
typedef struct eshkol_object_header {
    uint8_t  subtype;     // Object-specific subtype
    uint8_t  flags;       // Object flags
    uint16_t aux;         // Type-specific auxiliary data
    uint32_t size;        // Total object size in bytes
} eshkol_object_header_t;

_Static_assert(sizeof(eshkol_object_header_t) == 8,
               "Object header must be 8 bytes for alignment");

// Object flags
#define OBJECT_FLAG_NONE       0x00
#define OBJECT_FLAG_LINEAR     0x01
#define OBJECT_FLAG_BORROWED   0x02
#define OBJECT_FLAG_SHARED     0x04
#define OBJECT_FLAG_WEAK       0x08
#define OBJECT_FLAG_IMMUTABLE  0x10
#define OBJECT_FLAG_PINNED     0x20
#define OBJECT_FLAG_MARKED     0x40
#define OBJECT_FLAG_FINALIZED  0x80
```

2. Add subtype enumerations (heap_subtype_t, callable_subtype_t):

```c
// Add after object header definition

// Heap pointer subtypes (for future ESHKOL_VALUE_HEAP_PTR)
typedef enum {
    HEAP_SUBTYPE_CONS        = 0,
    HEAP_SUBTYPE_STRING      = 1,
    HEAP_SUBTYPE_VECTOR      = 2,
    HEAP_SUBTYPE_MULTI_VALUE = 3,
    HEAP_SUBTYPE_TENSOR      = 4,
    HEAP_SUBTYPE_HASH        = 5,
    HEAP_SUBTYPE_EXCEPTION   = 6,
    HEAP_SUBTYPE_RECORD      = 7,
    HEAP_SUBTYPE_BYTEVECTOR  = 8,
    HEAP_SUBTYPE_PORT        = 9,
} heap_subtype_t;

// Callable subtypes (for future ESHKOL_VALUE_CALLABLE)
typedef enum {
    CALLABLE_SUBTYPE_CLOSURE      = 0,
    CALLABLE_SUBTYPE_LAMBDA_SEXPR = 1,
    CALLABLE_SUBTYPE_AD_NODE      = 2,
    CALLABLE_SUBTYPE_CONTINUATION = 3,
    CALLABLE_SUBTYPE_BUILTIN      = 4,
    CALLABLE_SUBTYPE_GENERIC      = 5,
} callable_subtype_t;
```

3. Add header access helper functions:

```c
// Add helper functions for header access
static inline eshkol_object_header_t* eshkol_get_header_ptr(uint64_t ptr) {
    return (eshkol_object_header_t*)ptr;
}

static inline uint8_t eshkol_header_get_subtype(eshkol_object_header_t* hdr) {
    return hdr ? hdr->subtype : 0;
}

static inline uint8_t eshkol_header_get_flags(eshkol_object_header_t* hdr) {
    return hdr ? hdr->flags : 0;
}
```

**Validation:**
- Build project: `cmake --build build`
- Run all tests: `./scripts/run_features_tests.sh`
- Verify no regressions

**Commit message:**
```
Add object header infrastructure for pointer consolidation

- Add eshkol_object_header_t structure (8 bytes)
- Add OBJECT_FLAG_* constants for linear types, borrowing, etc.
- Add heap_subtype_t and callable_subtype_t enumerations
- Add header access helper functions

This is Phase 1 of pointer consolidation. The header structure will
be prepended to all heap-allocated objects in subsequent phases.
```

---

### 4.2 Phase 2: Type Enum Refactoring

**Goal:** Add new consolidated type values while keeping old values as aliases.

**Duration:** ~1 hour

**Files to modify:**
- `inc/eshkol/eshkol.h`

**Changes:**

1. Add new type constants (without removing old ones yet):

```c
// Modify eshkol_value_type_t enum to add new values

typedef enum {
    // Existing values (keep for now)
    ESHKOL_VALUE_NULL        = 0,
    ESHKOL_VALUE_INT64       = 1,
    ESHKOL_VALUE_DOUBLE      = 2,
    ESHKOL_VALUE_CONS_PTR    = 3,   // DEPRECATED: use HEAP_PTR + CONS subtype
    ESHKOL_VALUE_DUAL_NUMBER = 4,
    ESHKOL_VALUE_AD_NODE_PTR = 5,   // DEPRECATED: use CALLABLE + AD_NODE subtype
    ESHKOL_VALUE_TENSOR_PTR  = 6,   // DEPRECATED: use HEAP_PTR + TENSOR subtype
    ESHKOL_VALUE_LAMBDA_SEXPR = 7,  // DEPRECATED: use CALLABLE + LAMBDA_SEXPR subtype
    ESHKOL_VALUE_STRING_PTR  = 8,   // DEPRECATED: use HEAP_PTR + STRING subtype
    ESHKOL_VALUE_CHAR        = 9,
    ESHKOL_VALUE_VECTOR_PTR  = 10,  // DEPRECATED: use HEAP_PTR + VECTOR subtype
    ESHKOL_VALUE_SYMBOL      = 11,
    ESHKOL_VALUE_CLOSURE_PTR = 12,  // DEPRECATED: use CALLABLE + CLOSURE subtype
    ESHKOL_VALUE_BOOL        = 13,
    ESHKOL_VALUE_HASH_PTR    = 14,  // DEPRECATED: use HEAP_PTR + HASH subtype
    ESHKOL_VALUE_EXCEPTION   = 15,  // DEPRECATED: use HEAP_PTR + EXCEPTION subtype

    // === NEW CONSOLIDATED TYPES ===
    // These will replace the deprecated types above

    // Immediate values (final positions)
    // ESHKOL_VALUE_NULL     = 0   (already correct)
    // ESHKOL_VALUE_INT64    = 1   (already correct)
    // ESHKOL_VALUE_DOUBLE   = 2   (already correct)
    ESHKOL_VALUE_BOOL_NEW    = 3,  // Will become ESHKOL_VALUE_BOOL
    ESHKOL_VALUE_CHAR_NEW    = 4,  // Will become ESHKOL_VALUE_CHAR
    ESHKOL_VALUE_SYMBOL_NEW  = 5,  // Will become ESHKOL_VALUE_SYMBOL
    ESHKOL_VALUE_DUAL_NUMBER_NEW = 6, // Will become ESHKOL_VALUE_DUAL_NUMBER
    // Reserved: 7

    // Consolidated heap pointers
    ESHKOL_VALUE_HEAP_PTR    = 8,   // All heap data objects (cons, string, vector, etc.)
    ESHKOL_VALUE_CALLABLE    = 9,   // All callable objects (closure, lambda, ad_node)

    // Reserved: 10-15

    // Multimedia types
    ESHKOL_VALUE_HANDLE      = 16,  // Managed resource handles
    ESHKOL_VALUE_BUFFER      = 17,  // Typed data buffers
    ESHKOL_VALUE_STREAM      = 18,  // Lazy data streams
    ESHKOL_VALUE_EVENT       = 19,  // Input/system events

    // Advanced types
    ESHKOL_VALUE_PATH        = 32,
    ESHKOL_VALUE_FIBER       = 33,
    ESHKOL_VALUE_UNIVERSE    = 34,

} eshkol_value_type_t;
```

2. Add compatibility checking macros that work with BOTH old and new types:

```c
// ═══════════════════════════════════════════════════════════
// COMPATIBILITY MACROS (Phase 2)
// These work with both old individual types AND new consolidated types
// ═══════════════════════════════════════════════════════════

// Check for cons: old CONS_PTR OR new HEAP_PTR with CONS subtype
#define ESHKOL_IS_CONS_COMPAT(tagged) \
    (((tagged).type == ESHKOL_VALUE_CONS_PTR) || \
     ((tagged).type == ESHKOL_VALUE_HEAP_PTR && \
      eshkol_header_get_subtype(eshkol_get_header_ptr((tagged).data.ptr_val)) == HEAP_SUBTYPE_CONS))

// Check for string: old STRING_PTR OR new HEAP_PTR with STRING subtype
#define ESHKOL_IS_STRING_COMPAT(tagged) \
    (((tagged).type == ESHKOL_VALUE_STRING_PTR) || \
     ((tagged).type == ESHKOL_VALUE_HEAP_PTR && \
      eshkol_header_get_subtype(eshkol_get_header_ptr((tagged).data.ptr_val)) == HEAP_SUBTYPE_STRING))

// Check for vector: old VECTOR_PTR OR new HEAP_PTR with VECTOR subtype
#define ESHKOL_IS_VECTOR_COMPAT(tagged) \
    (((tagged).type == ESHKOL_VALUE_VECTOR_PTR) || \
     ((tagged).type == ESHKOL_VALUE_HEAP_PTR && \
      eshkol_header_get_subtype(eshkol_get_header_ptr((tagged).data.ptr_val)) == HEAP_SUBTYPE_VECTOR))

// Check for tensor: old TENSOR_PTR OR new HEAP_PTR with TENSOR subtype
#define ESHKOL_IS_TENSOR_COMPAT(tagged) \
    (((tagged).type == ESHKOL_VALUE_TENSOR_PTR) || \
     ((tagged).type == ESHKOL_VALUE_HEAP_PTR && \
      eshkol_header_get_subtype(eshkol_get_header_ptr((tagged).data.ptr_val)) == HEAP_SUBTYPE_TENSOR))

// Check for hash: old HASH_PTR OR new HEAP_PTR with HASH subtype
#define ESHKOL_IS_HASH_COMPAT(tagged) \
    (((tagged).type == ESHKOL_VALUE_HASH_PTR) || \
     ((tagged).type == ESHKOL_VALUE_HEAP_PTR && \
      eshkol_header_get_subtype(eshkol_get_header_ptr((tagged).data.ptr_val)) == HEAP_SUBTYPE_HASH))

// Check for exception: old EXCEPTION OR new HEAP_PTR with EXCEPTION subtype
#define ESHKOL_IS_EXCEPTION_COMPAT(tagged) \
    (((tagged).type == ESHKOL_VALUE_EXCEPTION) || \
     ((tagged).type == ESHKOL_VALUE_HEAP_PTR && \
      eshkol_header_get_subtype(eshkol_get_header_ptr((tagged).data.ptr_val)) == HEAP_SUBTYPE_EXCEPTION))

// Check for closure: old CLOSURE_PTR OR new CALLABLE with CLOSURE subtype
#define ESHKOL_IS_CLOSURE_COMPAT(tagged) \
    (((tagged).type == ESHKOL_VALUE_CLOSURE_PTR) || \
     ((tagged).type == ESHKOL_VALUE_CALLABLE && \
      eshkol_header_get_subtype(eshkol_get_header_ptr((tagged).data.ptr_val)) == CALLABLE_SUBTYPE_CLOSURE))

// Check for lambda sexpr: old LAMBDA_SEXPR OR new CALLABLE with LAMBDA_SEXPR subtype
#define ESHKOL_IS_LAMBDA_SEXPR_COMPAT(tagged) \
    (((tagged).type == ESHKOL_VALUE_LAMBDA_SEXPR) || \
     ((tagged).type == ESHKOL_VALUE_CALLABLE && \
      eshkol_header_get_subtype(eshkol_get_header_ptr((tagged).data.ptr_val)) == CALLABLE_SUBTYPE_LAMBDA_SEXPR))

// Check for AD node: old AD_NODE_PTR OR new CALLABLE with AD_NODE subtype
#define ESHKOL_IS_AD_NODE_COMPAT(tagged) \
    (((tagged).type == ESHKOL_VALUE_AD_NODE_PTR) || \
     ((tagged).type == ESHKOL_VALUE_CALLABLE && \
      eshkol_header_get_subtype(eshkol_get_header_ptr((tagged).data.ptr_val)) == CALLABLE_SUBTYPE_AD_NODE))

// Check for multi-value (new type only)
#define ESHKOL_IS_MULTI_VALUE(tagged) \
    ((tagged).type == ESHKOL_VALUE_HEAP_PTR && \
     eshkol_header_get_subtype(eshkol_get_header_ptr((tagged).data.ptr_val)) == HEAP_SUBTYPE_MULTI_VALUE)
```

**Validation:**
- Build project
- Run all tests
- Verify no regressions (old code still works)

**Commit message:**
```
Add consolidated type constants and compatibility macros

- Add ESHKOL_VALUE_HEAP_PTR (8) for all heap data objects
- Add ESHKOL_VALUE_CALLABLE (9) for all callable objects
- Add ESHKOL_VALUE_HANDLE/BUFFER/STREAM/EVENT (16-19) for multimedia
- Add _COMPAT macros that work with both old and new type systems
- Mark old pointer types as DEPRECATED in comments

This is Phase 2 of pointer consolidation. Old code continues to work
while new code can use the consolidated types.
```

---

### 4.3 Phase 3: Arena Memory Updates

**Goal:** Update allocation functions to prepend object headers.

**Duration:** ~2 hours

**Files to modify:**
- `lib/core/arena_memory.cpp`
- `inc/eshkol/eshkol.h` (add new allocation helpers)

**Changes:**

1. Add new object structure definitions with headers:

```c
// In eshkol.h - add after object header definition

// ═══════════════════════════════════════════════════════════
// OBJECT STRUCTURES WITH HEADERS
// These structures have the header prepended
// ═══════════════════════════════════════════════════════════

// Cons cell with header
typedef struct eshkol_cons {
    eshkol_object_header_t header;  // subtype = HEAP_SUBTYPE_CONS
    eshkol_tagged_value_t car;
    eshkol_tagged_value_t cdr;
} eshkol_cons_t;

// String with header
typedef struct eshkol_string {
    eshkol_object_header_t header;  // subtype = HEAP_SUBTYPE_STRING
    uint64_t length;                // Number of characters
    uint64_t capacity;              // Allocated capacity
    char data[];                    // Flexible array for string data
} eshkol_string_t;

// Vector with header (also used for multi-values)
typedef struct eshkol_vector {
    eshkol_object_header_t header;  // subtype = HEAP_SUBTYPE_VECTOR or MULTI_VALUE
    uint64_t length;                // Number of elements
    uint64_t capacity;              // Allocated capacity (0 if fixed)
    eshkol_tagged_value_t elements[];  // Flexible array of tagged values
} eshkol_vector_t;

// Hash table with header
typedef struct eshkol_hash {
    eshkol_object_header_t header;  // subtype = HEAP_SUBTYPE_HASH
    uint64_t size;                  // Number of entries
    uint64_t capacity;              // Number of buckets
    // ... bucket array follows
} eshkol_hash_t;

// Exception with header
typedef struct eshkol_exception_obj {
    eshkol_object_header_t header;  // subtype = HEAP_SUBTYPE_EXCEPTION
    eshkol_exception_type_t type;   // Exception type code
    char* message;                  // Error message
    eshkol_tagged_value_t* irritants;
    uint32_t num_irritants;
    uint32_t line;
    uint32_t column;
    char* filename;
} eshkol_exception_obj_t;

// Closure with header
typedef struct eshkol_closure_obj {
    eshkol_object_header_t header;  // subtype = CALLABLE_SUBTYPE_CLOSURE
    uint64_t func_ptr;
    eshkol_closure_env_t* env;
    uint64_t sexpr_ptr;
    uint8_t return_type;
    uint8_t input_arity;
    uint8_t flags;
    uint8_t reserved;
    uint32_t hott_type_id;
} eshkol_closure_obj_t;
```

2. Update arena allocation functions in arena_memory.cpp:

```cpp
// New allocation function that sets up header
void* arena_alloc_with_header(size_t data_size, uint8_t subtype, uint8_t flags) {
    size_t total_size = sizeof(eshkol_object_header_t) + data_size;
    void* ptr = arena_alloc(total_size);
    if (ptr) {
        eshkol_object_header_t* header = (eshkol_object_header_t*)ptr;
        header->subtype = subtype;
        header->flags = flags;
        header->aux = 0;
        header->size = (uint32_t)total_size;
    }
    return ptr;
}

// Specific allocators for each type
eshkol_cons_t* arena_alloc_cons(void) {
    eshkol_cons_t* cons = (eshkol_cons_t*)arena_alloc(sizeof(eshkol_cons_t));
    if (cons) {
        cons->header.subtype = HEAP_SUBTYPE_CONS;
        cons->header.flags = 0;
        cons->header.aux = 0;
        cons->header.size = sizeof(eshkol_cons_t);
    }
    return cons;
}

eshkol_vector_t* arena_alloc_vector(uint64_t capacity) {
    size_t size = sizeof(eshkol_vector_t) + capacity * sizeof(eshkol_tagged_value_t);
    eshkol_vector_t* vec = (eshkol_vector_t*)arena_alloc(size);
    if (vec) {
        vec->header.subtype = HEAP_SUBTYPE_VECTOR;
        vec->header.flags = 0;
        vec->header.aux = 0;
        vec->header.size = (uint32_t)size;
        vec->length = 0;
        vec->capacity = capacity;
    }
    return vec;
}

eshkol_vector_t* arena_alloc_multi_value(uint64_t num_values) {
    size_t size = sizeof(eshkol_vector_t) + num_values * sizeof(eshkol_tagged_value_t);
    eshkol_vector_t* mv = (eshkol_vector_t*)arena_alloc(size);
    if (mv) {
        mv->header.subtype = HEAP_SUBTYPE_MULTI_VALUE;
        mv->header.flags = 0;
        mv->header.aux = (uint16_t)num_values;
        mv->header.size = (uint32_t)size;
        mv->length = num_values;
        mv->capacity = num_values;
    }
    return mv;
}

// ... similar for string, hash, exception, closure
```

3. Add helper functions to create tagged values with new types:

```c
// In eshkol.h

static inline eshkol_tagged_value_t eshkol_make_heap_ptr(void* ptr) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_HEAP_PTR;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = (uint64_t)ptr;
    return result;
}

static inline eshkol_tagged_value_t eshkol_make_callable(void* ptr) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_CALLABLE;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = (uint64_t)ptr;
    return result;
}

// Get data pointer (skip header)
static inline void* eshkol_get_data_ptr(eshkol_tagged_value_t t) {
    return (void*)t.data.ptr_val;
}
```

**Validation:**
- Build project
- Run all tests
- Verify allocations work correctly
- Check memory alignment

**Commit message:**
```
Update arena allocations to use object headers

- Add eshkol_cons_t, eshkol_vector_t, etc. structures with headers
- Add arena_alloc_cons(), arena_alloc_vector(), etc. functions
- Add arena_alloc_multi_value() for multiple return values
- Add eshkol_make_heap_ptr() and eshkol_make_callable() helpers

This is Phase 3 of pointer consolidation. New allocations use headers
while existing code continues to work via compatibility macros.
```

---

### 4.4 Phase 4: Codegen Migration

**Goal:** Update LLVM codegen to use new type system.

**Duration:** ~4-6 hours (largest phase)

**Files to modify:**
- `lib/backend/llvm_codegen.cpp`
- `lib/backend/collection_codegen.cpp`
- `lib/backend/homoiconic_codegen.cpp`
- `lib/backend/tensor_codegen.cpp`
- `lib/backend/arithmetic_codegen.cpp`
- `lib/backend/string_io_codegen.cpp`
- `lib/backend/hash_codegen.cpp`
- `lib/backend/map_codegen.cpp`
- `lib/backend/function_codegen.cpp`
- `lib/backend/call_apply_codegen.cpp`
- `lib/backend/system_codegen.cpp`
- `lib/backend/binding_codegen.cpp`
- `inc/eshkol/backend/tagged_value_codegen.h`

**Strategy:**

The codegen migration follows a pattern for each type check. Here's the transformation:

**Old pattern (checking type directly):**
```cpp
// Check if value is a cons
llvm::Value* type = extractTypeFromTaggedValue(val);
llvm::Value* is_cons = builder->CreateICmpEQ(
    type,
    ConstantInt::get(int8_type, ESHKOL_VALUE_CONS_PTR));
```

**New pattern (check type, then subtype if needed):**
```cpp
// Check if value is a cons (new system)
llvm::Value* type = extractTypeFromTaggedValue(val);
llvm::Value* is_heap_ptr = builder->CreateICmpEQ(
    type,
    ConstantInt::get(int8_type, ESHKOL_VALUE_HEAP_PTR));

// Only check subtype if it's a heap pointer
llvm::Value* ptr = extractPtrFromTaggedValue(val);
llvm::Value* subtype_ptr = builder->CreateGEP(
    int8_type, ptr, builder->getInt64(0));  // Subtype is first byte of header
llvm::Value* subtype = builder->CreateLoad(int8_type, subtype_ptr);
llvm::Value* is_cons_subtype = builder->CreateICmpEQ(
    subtype,
    ConstantInt::get(int8_type, HEAP_SUBTYPE_CONS));

llvm::Value* is_cons = builder->CreateAnd(is_heap_ptr, is_cons_subtype);
```

**Create helper functions in tagged_value_codegen.h:**

```cpp
// Helper to check for specific heap subtype
llvm::Value* isHeapSubtype(llvm::Value* tagged_val, uint8_t subtype) {
    llvm::Value* type = extractTypeFromTaggedValue(tagged_val);
    llvm::Value* is_heap = builder->CreateICmpEQ(
        type, ConstantInt::get(int8_type, ESHKOL_VALUE_HEAP_PTR));

    // Get subtype from header
    llvm::Value* ptr = extractPtrFromTaggedValue(tagged_val);
    llvm::Value* subtype_val = builder->CreateLoad(
        int8_type,
        builder->CreateGEP(int8_type, ptr, builder->getInt64(0)));
    llvm::Value* subtype_match = builder->CreateICmpEQ(
        subtype_val,
        ConstantInt::get(int8_type, subtype));

    return builder->CreateAnd(is_heap, subtype_match);
}

// Helper to check for specific callable subtype
llvm::Value* isCallableSubtype(llvm::Value* tagged_val, uint8_t subtype) {
    llvm::Value* type = extractTypeFromTaggedValue(tagged_val);
    llvm::Value* is_callable = builder->CreateICmpEQ(
        type, ConstantInt::get(int8_type, ESHKOL_VALUE_CALLABLE));

    llvm::Value* ptr = extractPtrFromTaggedValue(tagged_val);
    llvm::Value* subtype_val = builder->CreateLoad(
        int8_type,
        builder->CreateGEP(int8_type, ptr, builder->getInt64(0)));
    llvm::Value* subtype_match = builder->CreateICmpEQ(
        subtype_val,
        ConstantInt::get(int8_type, subtype));

    return builder->CreateAnd(is_callable, subtype_match);
}

// Convenience wrappers
llvm::Value* isCons(llvm::Value* val) {
    return isHeapSubtype(val, HEAP_SUBTYPE_CONS);
}

llvm::Value* isString(llvm::Value* val) {
    return isHeapSubtype(val, HEAP_SUBTYPE_STRING);
}

llvm::Value* isVector(llvm::Value* val) {
    return isHeapSubtype(val, HEAP_SUBTYPE_VECTOR);
}

llvm::Value* isMultiValue(llvm::Value* val) {
    return isHeapSubtype(val, HEAP_SUBTYPE_MULTI_VALUE);
}

llvm::Value* isTensor(llvm::Value* val) {
    return isHeapSubtype(val, HEAP_SUBTYPE_TENSOR);
}

llvm::Value* isHash(llvm::Value* val) {
    return isHeapSubtype(val, HEAP_SUBTYPE_HASH);
}

llvm::Value* isException(llvm::Value* val) {
    return isHeapSubtype(val, HEAP_SUBTYPE_EXCEPTION);
}

llvm::Value* isClosure(llvm::Value* val) {
    return isCallableSubtype(val, CALLABLE_SUBTYPE_CLOSURE);
}

llvm::Value* isLambdaSexpr(llvm::Value* val) {
    return isCallableSubtype(val, CALLABLE_SUBTYPE_LAMBDA_SEXPR);
}

llvm::Value* isAdNode(llvm::Value* val) {
    return isCallableSubtype(val, CALLABLE_SUBTYPE_AD_NODE);
}
```

**File-by-file migration order:**

1. **tagged_value_codegen.h** - Add helper functions first
2. **llvm_codegen.cpp** - Main file, migrate incrementally
3. **collection_codegen.cpp** - List/vector operations
4. **homoiconic_codegen.cpp** - S-expression handling
5. **string_io_codegen.cpp** - String operations
6. **tensor_codegen.cpp** - Tensor operations
7. **hash_codegen.cpp** - Hash table operations
8. **arithmetic_codegen.cpp** - Numeric operations
9. **function_codegen.cpp** - Function calls
10. **call_apply_codegen.cpp** - Apply/call operations
11. **map_codegen.cpp** - Map operations
12. **system_codegen.cpp** - System primitives
13. **binding_codegen.cpp** - Variable bindings

**Migration process for each file:**

1. Find all uses of `ESHKOL_VALUE_*_PTR` constants
2. Replace type comparisons with helper function calls
3. Update value creation to use new types
4. Run tests after each file

**Commit message (per file or group):**
```
Migrate [filename] to consolidated type system

- Replace ESHKOL_VALUE_*_PTR checks with helper functions
- Update type creation to use HEAP_PTR/CALLABLE
- Verify all tests pass
```

---

### 4.5 Phase 5: Display System Updates

**Goal:** Update display/print functions to work with new type system.

**Duration:** ~1 hour

**Files to modify:**
- `lib/core/arena_memory.cpp` (display functions)

**Changes:**

Update `eshkol_display_value_opts()` to handle new type system:

```cpp
void eshkol_display_value_opts(const eshkol_tagged_value_t* value,
                                eshkol_display_opts_t* opts) {
    if (!value) {
        fprintf(out, "#<null-ptr>");
        return;
    }

    uint8_t type = value->type;

    switch (type) {
        case ESHKOL_VALUE_NULL:
            fprintf(out, "()");
            break;

        case ESHKOL_VALUE_INT64:
            fprintf(out, "%lld", (long long)value->data.int_val);
            break;

        case ESHKOL_VALUE_DOUBLE:
            fprintf(out, "%g", value->data.double_val);
            break;

        case ESHKOL_VALUE_BOOL:
            fprintf(out, value->data.int_val ? "#t" : "#f");
            break;

        case ESHKOL_VALUE_CHAR:
            display_char(value->data.int_val, out);
            break;

        case ESHKOL_VALUE_SYMBOL:
            display_symbol(value->data.ptr_val, out);
            break;

        case ESHKOL_VALUE_DUAL_NUMBER:
            display_dual_number(value, out);
            break;

        case ESHKOL_VALUE_HEAP_PTR: {
            // Dispatch based on subtype
            eshkol_object_header_t* hdr = (eshkol_object_header_t*)value->data.ptr_val;
            switch (hdr->subtype) {
                case HEAP_SUBTYPE_CONS:
                    display_list(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_STRING:
                    display_string((eshkol_string_t*)value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_VECTOR:
                    display_vector((eshkol_vector_t*)value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_MULTI_VALUE:
                    display_multi_value((eshkol_vector_t*)value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_TENSOR:
                    display_tensor(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_HASH:
                    display_hash(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_EXCEPTION:
                    display_exception((eshkol_exception_obj_t*)value->data.ptr_val, opts);
                    break;
                default:
                    fprintf(out, "#<heap-object subtype=%d>", hdr->subtype);
            }
            break;
        }

        case ESHKOL_VALUE_CALLABLE: {
            eshkol_object_header_t* hdr = (eshkol_object_header_t*)value->data.ptr_val;
            switch (hdr->subtype) {
                case CALLABLE_SUBTYPE_CLOSURE:
                    display_closure((eshkol_closure_obj_t*)value->data.ptr_val, opts);
                    break;
                case CALLABLE_SUBTYPE_LAMBDA_SEXPR:
                    display_lambda_sexpr(value->data.ptr_val, opts);
                    break;
                case CALLABLE_SUBTYPE_AD_NODE:
                    fprintf(out, "#<ad-node>");
                    break;
                default:
                    fprintf(out, "#<callable subtype=%d>", hdr->subtype);
            }
            break;
        }

        // Legacy types (for backward compatibility during migration)
        case ESHKOL_VALUE_CONS_PTR:
            display_list(value->data.ptr_val, opts);
            break;
        // ... other legacy cases

        default:
            fprintf(out, "#<unknown type=%d>", type);
    }
}

// New helper for multi-value display
void display_multi_value(eshkol_vector_t* mv, eshkol_display_opts_t* opts) {
    FILE* out = opts->output ? (FILE*)opts->output : stdout;
    fprintf(out, "#<values");
    for (uint64_t i = 0; i < mv->length; i++) {
        fprintf(out, " ");
        eshkol_display_value_opts(&mv->elements[i], opts);
    }
    fprintf(out, ">");
}
```

**Validation:**
- Build project
- Run all tests
- Test display output for various types

**Commit message:**
```
Update display system for consolidated types

- Add HEAP_PTR and CALLABLE handling with subtype dispatch
- Add display_multi_value() for multiple return values
- Maintain backward compatibility with legacy type constants
```

---

### 4.6 Phase 6: Testing and Validation

**Goal:** Comprehensive testing to ensure all functionality works.

**Duration:** ~2 hours

**Test suite execution:**

```bash
# Run all test suites
./scripts/run_features_tests.sh
./scripts/run_memory_tests.sh
./scripts/run_stdlib_tests.sh
./scripts/run_list_tests.sh
./scripts/run_modules_tests.sh
./scripts/run_autodiff_tests.sh

# If all pass, run extended tests
./scripts/run_ml_tests.sh
./scripts/run_neural_tests.sh
```

**Create new test file for multi-values:**

```scheme
;;; tests/features/multi_value_test.esk
;;; Tests for multiple return values

;;; Test 1: Basic values
(define result1
  (call-with-values
    (lambda () (values 1 2 3))
    (lambda (a b c) (+ a (+ b c)))))

(display "Test 1 - values sum: ")
(display result1)
(newline)

;;; Test 2: Single value passes through
(define result2
  (call-with-values
    (lambda () 42)
    (lambda (x) (* x 2))))

(display "Test 2 - single value: ")
(display result2)
(newline)

;;; Test 3: let-values
(define result3
  (let-values (((a b) (values 10 20)))
    (- b a)))

(display "Test 3 - let-values: ")
(display result3)
(newline)

;;; Test 4: Nested values
(define result4
  (call-with-values
    (lambda ()
      (call-with-values
        (lambda () (values 2 3))
        (lambda (a b) (values (* a b) (+ a b)))))
    (lambda (prod sum) (- prod sum))))

(display "Test 4 - nested: ")
(display result4)
(newline)

(display "All multi-value tests completed!")
(newline)
```

**Create regression test for type system:**

```scheme
;;; tests/features/type_system_test.esk
;;; Tests to verify type system consolidation

;;; Test cons cells
(define pair1 (cons 1 2))
(display "cons: ")
(display pair1)
(newline)

;;; Test vectors
(define vec1 (vector 1 2 3))
(display "vector: ")
(display vec1)
(newline)

;;; Test strings
(define str1 "hello")
(display "string: ")
(display str1)
(newline)

;;; Test closures
(define add5 (lambda (x) (+ x 5)))
(display "closure result: ")
(display (add5 10))
(newline)

;;; Test exceptions
(define exc-result
  (guard (e (else "caught"))
    (error "test")))
(display "exception: ")
(display exc-result)
(newline)

(display "All type system tests passed!")
(newline)
```

**Validation checklist:**

- [ ] All existing tests pass
- [ ] Multi-value tests pass
- [ ] Type system regression tests pass
- [ ] No memory leaks (valgrind if available)
- [ ] Display output is correct for all types
- [ ] Exception handling still works
- [ ] Autodiff still works
- [ ] Module system still works

**Commit message:**
```
Complete pointer consolidation with tests

- Add multi_value_test.esk for multiple return values
- Add type_system_test.esk for type system regression
- Verify all test suites pass
- Document any known issues or limitations
```

---

## 5. File-by-File Migration Guide

This section provides specific guidance for each file that needs modification.

### 5.1 inc/eshkol/eshkol.h

**Location of changes:**

| Line Range | Change Description |
|------------|-------------------|
| 49-67 | Expand type enum with new values |
| 68-76 | Add subtype enumerations |
| 87-97 | Add object header structure |
| 98-140 | Add new object structures with headers |
| 155-180 | Update/add type checking macros |
| After 180 | Add compatibility macros |

### 5.2 lib/core/arena_memory.cpp

**Location of changes:**

| Function | Change Description |
|----------|-------------------|
| `arena_alloc_cons()` | New function - allocate cons with header |
| `arena_alloc_vector()` | New function - allocate vector with header |
| `arena_alloc_multi_value()` | New function - allocate multi-value |
| `arena_alloc_string()` | Update to set header |
| `eshkol_display_value_opts()` | Add HEAP_PTR/CALLABLE dispatch |
| `display_multi_value()` | New function |

### 5.3 lib/backend/llvm_codegen.cpp

**Search patterns to find and replace:**

```
ESHKOL_VALUE_CONS_PTR     -> isCons() helper
ESHKOL_VALUE_STRING_PTR   -> isString() helper
ESHKOL_VALUE_VECTOR_PTR   -> isVector() helper
ESHKOL_VALUE_TENSOR_PTR   -> isTensor() helper
ESHKOL_VALUE_HASH_PTR     -> isHash() helper
ESHKOL_VALUE_CLOSURE_PTR  -> isClosure() helper
ESHKOL_VALUE_LAMBDA_SEXPR -> isLambdaSexpr() helper
ESHKOL_VALUE_AD_NODE_PTR  -> isAdNode() helper
ESHKOL_VALUE_EXCEPTION    -> isException() helper
```

**Critical sections:**

| Line Range | Function | Priority |
|------------|----------|----------|
| 171-172 | TypedValue::isList/isVector | HIGH |
| 3191-3345 | Type inference | HIGH |
| 3814-3911 | Type dispatch | HIGH |
| 4139 | Rest list creation | MEDIUM |
| 4326-4470 | More type dispatch | HIGH |

### 5.4 Other Codegen Files

Each file follows the same pattern - search for `ESHKOL_VALUE_*_PTR` and replace with helper calls.

---

## 6. Compatibility Layer

During the migration, both old and new type constants coexist. The compatibility layer ensures:

1. **Old allocations** continue to work (they don't have headers, but checks handle this)
2. **New allocations** use headers and new types
3. **Type checks** work with both systems via `_COMPAT` macros

### 6.1 Gradual Migration Strategy

```
Phase 1-2: Add infrastructure, keep old code working
           ↓
Phase 3:   New allocations use headers, old checks still work
           ↓
Phase 4:   Migrate checks to use helper functions
           ↓
Phase 5-6: Test everything, verify compatibility
           ↓
Future:    Remove deprecated type constants
```

### 6.2 Removing Deprecated Types (Future)

After all code is migrated and tested, the deprecated type constants can be removed:

```c
// These will be removed in a future version:
// ESHKOL_VALUE_CONS_PTR    -> HEAP_PTR + CONS
// ESHKOL_VALUE_STRING_PTR  -> HEAP_PTR + STRING
// ESHKOL_VALUE_VECTOR_PTR  -> HEAP_PTR + VECTOR
// ESHKOL_VALUE_TENSOR_PTR  -> HEAP_PTR + TENSOR
// ESHKOL_VALUE_HASH_PTR    -> HEAP_PTR + HASH
// ESHKOL_VALUE_CLOSURE_PTR -> CALLABLE + CLOSURE
// ESHKOL_VALUE_LAMBDA_SEXPR -> CALLABLE + LAMBDA_SEXPR
// ESHKOL_VALUE_AD_NODE_PTR -> CALLABLE + AD_NODE
// ESHKOL_VALUE_EXCEPTION   -> HEAP_PTR + EXCEPTION
```

---

## 7. Testing Strategy

### 7.1 Unit Testing

Each phase should be validated with:

```bash
# Quick validation
cmake --build build && ./build/eshkol-run tests/features/exception_test.esk

# Full validation
./scripts/run_features_tests.sh
```

### 7.2 Integration Testing

After Phase 4 completion:

```bash
# Run all test suites
for script in scripts/run_*_tests.sh; do
    echo "Running $script..."
    $script || echo "FAILED: $script"
done
```

### 7.3 Regression Testing

Create a comprehensive regression test:

```scheme
;;; tests/regression/type_consolidation_test.esk

;; Test each type individually
(define test-results '())

;; Cons/list
(define list-test (list 1 2 3))
(set! test-results (cons (pair? (car (list (cons 1 2)))) test-results))

;; Vector
(define vec-test (vector 1 2 3))
(set! test-results (cons (vector? vec-test) test-results))

;; String
(define str-test "hello")
(set! test-results (cons (string? str-test) test-results))

;; Closure
(define closure-test (lambda (x) x))
(set! test-results (cons (procedure? closure-test) test-results))

;; ... more tests

(display "Test results: ")
(display test-results)
(newline)
```

---

## 8. Rollback Plan

If critical issues are discovered:

### 8.1 Git-based Rollback

```bash
# Find the last working commit
git log --oneline

# Revert to that commit
git revert HEAD~N..HEAD  # Where N is number of commits to revert

# Or hard reset (destructive)
git reset --hard <commit-hash>
```

### 8.2 Compatibility Fallback

If partial migration causes issues:

1. Keep both old and new type constants
2. Use `_COMPAT` macros everywhere
3. Investigate specific failures
4. Fix and continue migration

### 8.3 Emergency Restore Points

Create tagged commits at each phase:

```bash
git tag -a consolidation-phase1 -m "Phase 1: Header infrastructure"
git tag -a consolidation-phase2 -m "Phase 2: Type enum refactoring"
# etc.
```

---

## 9. Success Criteria

The pointer consolidation is complete when:

### 9.1 Functional Requirements

- [ ] All existing tests pass (100% pass rate)
- [ ] New type constants (HEAP_PTR, CALLABLE) are used throughout
- [ ] Object headers are correctly set on all heap allocations
- [ ] Type checking works via helper functions
- [ ] Display system correctly identifies all types
- [ ] Multi-value support is functional

### 9.2 Non-Functional Requirements

- [ ] No performance regression (< 5% slowdown acceptable)
- [ ] No increase in binary size (< 10% acceptable)
- [ ] Memory usage unchanged or improved
- [ ] Clean build with no warnings

### 9.3 Documentation Requirements

- [ ] All deprecated types marked in code
- [ ] Migration guide updated with actual changes
- [ ] Test coverage documented
- [ ] Known issues documented

---

## Appendix A: Quick Reference

### A.1 Type Value Quick Reference

| Old Type | New Type | Subtype |
|----------|----------|---------|
| ESHKOL_VALUE_CONS_PTR (3) | ESHKOL_VALUE_HEAP_PTR (8) | HEAP_SUBTYPE_CONS (0) |
| ESHKOL_VALUE_STRING_PTR (8) | ESHKOL_VALUE_HEAP_PTR (8) | HEAP_SUBTYPE_STRING (1) |
| ESHKOL_VALUE_VECTOR_PTR (10) | ESHKOL_VALUE_HEAP_PTR (8) | HEAP_SUBTYPE_VECTOR (2) |
| ESHKOL_VALUE_TENSOR_PTR (6) | ESHKOL_VALUE_HEAP_PTR (8) | HEAP_SUBTYPE_TENSOR (3) |
| (new) | ESHKOL_VALUE_HEAP_PTR (8) | HEAP_SUBTYPE_MULTI_VALUE (4) |
| ESHKOL_VALUE_HASH_PTR (14) | ESHKOL_VALUE_HEAP_PTR (8) | HEAP_SUBTYPE_HASH (5) |
| ESHKOL_VALUE_EXCEPTION (15) | ESHKOL_VALUE_HEAP_PTR (8) | HEAP_SUBTYPE_EXCEPTION (6) |
| ESHKOL_VALUE_CLOSURE_PTR (12) | ESHKOL_VALUE_CALLABLE (9) | CALLABLE_SUBTYPE_CLOSURE (0) |
| ESHKOL_VALUE_LAMBDA_SEXPR (7) | ESHKOL_VALUE_CALLABLE (9) | CALLABLE_SUBTYPE_LAMBDA_SEXPR (1) |
| ESHKOL_VALUE_AD_NODE_PTR (5) | ESHKOL_VALUE_CALLABLE (9) | CALLABLE_SUBTYPE_AD_NODE (2) |

### A.2 Macro Quick Reference

```c
// Old macro          -> New function
ESHKOL_IS_CONS_PTR_TYPE(type)   -> eshkol_is_cons(tagged)
ESHKOL_IS_STRING_PTR_TYPE(type) -> eshkol_is_string(tagged)
ESHKOL_IS_VECTOR_PTR_TYPE(type) -> eshkol_is_vector(tagged)
ESHKOL_IS_TENSOR_PTR_TYPE(type) -> eshkol_is_tensor(tagged)
ESHKOL_IS_HASH_PTR_TYPE(type)   -> eshkol_is_hash(tagged)
ESHKOL_IS_CLOSURE_PTR_TYPE(type) -> eshkol_is_closure(tagged)
ESHKOL_IS_LAMBDA_SEXPR_TYPE(type) -> eshkol_is_lambda_sexpr(tagged)
ESHKOL_IS_AD_NODE_PTR_TYPE(type) -> eshkol_is_ad_node(tagged)
ESHKOL_IS_EXCEPTION_TYPE(type)  -> eshkol_is_exception(tagged)
```

---

*End of Implementation Strategy Document*
