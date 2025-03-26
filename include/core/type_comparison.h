/**
 * @file type_comparison.h
 * @brief Type comparison functions for Eshkol
 */

#ifndef ESHKOL_TYPE_COMPARISON_H
#define ESHKOL_TYPE_COMPARISON_H

#include "core/type.h"
#include <stdbool.h>
#include <stddef.h>

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

#endif // ESHKOL_TYPE_COMPARISON_H
