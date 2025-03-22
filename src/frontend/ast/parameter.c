/**
 * @file parameter.c
 * @brief Implementation of the parameter structure for Eshkol
 */

#include "frontend/ast/parameter.h"
#include <assert.h>

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
Parameter* parameter_create(Arena* arena, StringId name, Type* type, size_t line, size_t column) {
    assert(arena != NULL);
    assert(name != NULL);
    
    Parameter* param = arena_alloc(arena, sizeof(Parameter));
    if (!param) {
        return NULL;
    }
    
    param->name = name;
    param->type = type;  // Can be NULL for untyped parameters
    param->line = line;
    param->column = column;
    
    return param;
}
