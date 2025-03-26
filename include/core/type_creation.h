/**
 * @file type_creation.h
 * @brief Type creation functions for Eshkol
 */

#ifndef ESHKOL_TYPE_CREATION_H
#define ESHKOL_TYPE_CREATION_H

#include "core/type.h"
#include <stdbool.h>
#include <stddef.h>

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

#endif // ESHKOL_TYPE_CREATION_H
