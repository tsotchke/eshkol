/**
 * @file parameter.h
 * @brief Parameter structure for Eshkol
 * 
 * This file defines the parameter structure for the Eshkol language,
 * which includes optional type annotations.
 */

#ifndef ESHKOL_PARAMETER_H
#define ESHKOL_PARAMETER_H

#include "core/memory.h"
#include "core/string_table.h"
#include "core/type.h"
#include <stddef.h>

/**
 * @brief String identifier type
 * 
 * This is a pointer to an interned string in the string table.
 */
typedef const char* StringId;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parameter structure
 * 
 * This structure represents a function parameter, which can have an optional
 * type annotation.
 */
typedef struct {
    StringId name;     /**< Parameter name */
    Type* type;        /**< Parameter type (can be NULL for untyped) */
    size_t line;       /**< Line number */
    size_t column;     /**< Column number */
} Parameter;

/**
 * @brief Create a new parameter
 * 
 * @param arena Arena allocator
 * @param name Parameter name
 * @param type Parameter type (can be NULL for untyped)
 * @param line Line number
 * @param column Column number
 * @return A new parameter, or NULL on failure
 */
Parameter* parameter_create(Arena* arena, StringId name, Type* type, size_t line, size_t column);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PARAMETER_H */
