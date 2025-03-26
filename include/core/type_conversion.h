/**
 * @file type_conversion.h
 * @brief Type conversion functions for Eshkol
 */

#ifndef ESHKOL_TYPE_CONVERSION_H
#define ESHKOL_TYPE_CONVERSION_H

#include "core/type.h"
#include <stdbool.h>
#include <stddef.h>

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

/**
 * @brief Check if a type can be converted to another type
 * @param from Source type
 * @param to Target type
 * @return Whether the conversion is possible
 */
bool type_can_convert(Type* from, Type* to);

/**
 * @brief Apply type conversion to a value
 * @param arena Memory arena
 * @param expr Expression to convert
 * @param from Source type
 * @param to Target type
 * @return Converted expression
 */
char* type_apply_conversion(Arena* arena, const char* expr, Type* from, Type* to);

#endif // ESHKOL_TYPE_CONVERSION_H
