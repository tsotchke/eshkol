/**
 * @file type.h
 * @brief Type system for Eshkol
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

/**
 * @brief Create a void type
 * @param arena Memory arena
 * @return Void type
 */
Type* type_void_create(Arena* arena);

/**
 * @brief Create a boolean type
 * @param arena Memory arena
 * @return Boolean type
 */
Type* type_boolean_create(Arena* arena);

/**
 * @brief Create an integer type
 * @param arena Memory arena
 * @param size Integer size
 * @return Integer type
 */
Type* type_integer_create(Arena* arena, IntSize size);

/**
 * @brief Create a float type
 * @param arena Memory arena
 * @param size Float size
 * @return Float type
 */
Type* type_float_create(Arena* arena, FloatSize size);

/**
 * @brief Create a character type
 * @param arena Memory arena
 * @return Character type
 */
Type* type_char_create(Arena* arena);

/**
 * @brief Create a string type
 * @param arena Memory arena
 * @return String type
 */
Type* type_string_create(Arena* arena);

/**
 * @brief Create a symbol type
 * @param arena Memory arena
 * @return Symbol type
 */
Type* type_symbol_create(Arena* arena);

/**
 * @brief Create a pair type
 * @param arena Memory arena
 * @param car_type Car type
 * @param cdr_type Cdr type
 * @return Pair type
 */
Type* type_pair_create(Arena* arena, Type* car_type, Type* cdr_type);

/**
 * @brief Create a vector type
 * @param arena Memory arena
 * @param element_type Element type
 * @param size Size of the vector (0 for dynamic size)
 * @return Vector type
 */
Type* type_vector_create(Arena* arena, Type* element_type, size_t size);

/**
 * @brief Create a function type
 * @param arena Memory arena
 * @param param_count Number of parameters
 * @param params Parameter types
 * @param return_type Return type
 * @param variadic Whether the function is variadic
 * @return Function type
 */
Type* type_function_create(Arena* arena, size_t param_count, Type** params, Type* return_type, bool variadic);

/**
 * @brief Create a structure type
 * @param arena Memory arena
 * @param field_count Number of fields
 * @param fields Fields
 * @return Structure type
 */
Type* type_struct_create(Arena* arena, size_t field_count, StructField* fields);

/**
 * @brief Create a union type
 * @param arena Memory arena
 * @param variant_count Number of variants
 * @param variants Variant types
 * @return Union type
 */
Type* type_union_create(Arena* arena, size_t variant_count, Type** variants);

/**
 * @brief Create an any type
 * @param arena Memory arena
 * @return Any type
 */
Type* type_any_create(Arena* arena);

/**
 * @brief Create an unknown type
 * @param arena Memory arena
 * @return Unknown type
 */
Type* type_unknown_create(Arena* arena);

/**
 * @brief Check if two types are equal
 * @param a First type
 * @param b Second type
 * @return Whether the types are equal
 */
bool type_equals(Type* a, Type* b);

/**
 * @brief Check if a type is a subtype of another
 * @param sub Subtype
 * @param super Supertype
 * @return Whether sub is a subtype of super
 */
bool type_is_subtype(Type* sub, Type* super);

/**
 * @brief Get the common supertype of two types
 * @param arena Memory arena
 * @param a First type
 * @param b Second type
 * @return Common supertype
 */
Type* type_common_supertype(Arena* arena, Type* a, Type* b);

/**
 * @brief Convert a type to a string
 * @param arena Memory arena
 * @param type Type
 * @return String representation of the type
 */
char* type_to_string(Arena* arena, Type* type);

/**
 * @brief Parse a type from a string
 * @param arena Memory arena
 * @param str String
 * @return Parsed type
 */
Type* type_from_string(Arena* arena, const char* str);

#endif // ESHKOL_TYPE_H
