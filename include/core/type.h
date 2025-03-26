/**
 * @file type.h
 * @brief Type system for Eshkol
 * 
 * This file contains the core type definitions for the Eshkol type system.
 * The actual implementations are split into separate files:
 * - type_creation.h: Type creation functions
 * - type_comparison.h: Type comparison functions
 * - type_conversion.h: Type conversion functions
 */

#ifndef ESHKOL_TYPE_H
#define ESHKOL_TYPE_H

#include "core/memory.h"
#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Type kinds
 */
typedef enum {
    TYPE_VOID,       // Void type
    TYPE_BOOLEAN,    // Boolean type
    TYPE_INTEGER,    // Integer type
    TYPE_FLOAT,      // Floating-point type
    TYPE_CHAR,       // Character type
    TYPE_STRING,     // String type
    TYPE_SYMBOL,     // Symbol type
    TYPE_PAIR,       // Pair type (cons cell)
    TYPE_VECTOR,     // Vector type
    TYPE_FUNCTION,   // Function type
    TYPE_STRUCT,     // Structure type
    TYPE_UNION,      // Union type
    TYPE_ANY,        // Any type (dynamic)
    TYPE_UNKNOWN     // Unknown type (for type inference)
} TypeKind;

/**
 * @brief Integer type sizes
 */
typedef enum {
    INT_SIZE_8,      // 8-bit integer
    INT_SIZE_16,     // 16-bit integer
    INT_SIZE_32,     // 32-bit integer
    INT_SIZE_64      // 64-bit integer
} IntSize;

/**
 * @brief Float type sizes
 */
typedef enum {
    FLOAT_SIZE_32,   // 32-bit float
    FLOAT_SIZE_64    // 64-bit float
} FloatSize;

/**
 * @brief Type structure
 */
typedef struct Type Type;

/**
 * @brief Function type parameters
 */
typedef struct {
    size_t param_count;   // Number of parameters
    Type** params;        // Parameter types
    Type* return_type;    // Return type
    bool variadic;        // Whether the function is variadic
} FunctionTypeInfo;

/**
 * @brief Vector type parameters
 */
typedef struct {
    Type* element_type;   // Element type
    size_t size;          // Size of the vector (0 for dynamic size)
} VectorTypeInfo;

/**
 * @brief Structure field
 */
typedef struct {
    const char* name;     // Field name
    Type* type;           // Field type
} StructField;

/**
 * @brief Structure type parameters
 */
typedef struct {
    size_t field_count;   // Number of fields
    StructField* fields;  // Fields
} StructTypeInfo;

/**
 * @brief Union type parameters
 */
typedef struct {
    size_t variant_count; // Number of variants
    Type** variants;      // Variant types
} UnionTypeInfo;

/**
 * @brief Type structure
 */
struct Type {
    TypeKind kind;        // Type kind
    union {
        IntSize int_size;                 // Integer size
        FloatSize float_size;             // Float size
        FunctionTypeInfo function;        // Function type info
        VectorTypeInfo vector;            // Vector type info
        StructTypeInfo structure;         // Structure type info
        UnionTypeInfo union_type;         // Union type info
    };
};

#endif // ESHKOL_TYPE_H
