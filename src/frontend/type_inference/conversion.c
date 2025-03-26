/**
 * @file conversion.c
 * @brief Implementation of type conversion functions
 */

#include "frontend/type_inference/conversion.h"
#include "core/memory.h"
#include "core/type.h"
#include "core/type_creation.h"
#include "core/type_comparison.h"
#include "core/type_conversion.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

/**
 * @brief Check if a type can be implicitly converted to another type
 */
bool type_can_convert(Type* from, Type* to) {
    assert(from != NULL);
    assert(to != NULL);
    
    // Same type, no conversion needed
    if (type_equals(from, to)) {
        return true;
    }
    
    // Any type can be converted to any
    if (to->kind == TYPE_ANY) {
        return true;
    }
    
    // Unknown type can be converted to any concrete type
    if (from->kind == TYPE_UNKNOWN) {
        return true;
    }
    
    // Integer to float conversion
    if (from->kind == TYPE_INTEGER && to->kind == TYPE_FLOAT) {
        return true;
    }
    
    // Vector conversions - allow if element types are convertible
    if (from->kind == TYPE_VECTOR && to->kind == TYPE_VECTOR) {
        // If both have the same size or the target is dynamic size
        if (from->vector.size == to->vector.size || to->vector.size == 0) {
            return type_can_convert(from->vector.element_type, to->vector.element_type);
        }
    }
    
    // Function conversions - allow if return type and parameter types are convertible
    if (from->kind == TYPE_FUNCTION && to->kind == TYPE_FUNCTION) {
        // Check return type
        if (!type_can_convert(from->function.return_type, to->function.return_type)) {
            return false;
        }
        
        // Check parameter count
        if (from->function.param_count != to->function.param_count && !to->function.variadic) {
            return false;
        }
        
        // Check parameter types
        for (size_t i = 0; i < from->function.param_count && i < to->function.param_count; i++) {
            if (!type_can_convert(from->function.params[i], to->function.params[i])) {
                return false;
            }
        }
        
        return true;
    }
    
    // Pair conversions - allow if car and cdr types are convertible
    if (from->kind == TYPE_PAIR && to->kind == TYPE_PAIR) {
        return type_can_convert(from->function.params[0], to->function.params[0]) &&
               type_can_convert(from->function.params[1], to->function.params[1]);
    }
    
    // Special case for autodiff and vector calculus
    // VectorF* can be implicitly converted to float for scalar operations
    if (from->kind == TYPE_VECTOR && 
        (to->kind == TYPE_FLOAT || to->kind == TYPE_INTEGER)) {
        // Allow vector to scalar conversion for autodiff operations
        return true;
    }
    
    // Float can be implicitly converted to VectorF* for vector operations
    if ((from->kind == TYPE_FLOAT || from->kind == TYPE_INTEGER) && 
        to->kind == TYPE_VECTOR) {
        // Allow scalar to vector conversion for autodiff operations
        return true;
    }
    
    return false;
}

/**
 * @brief Generate code for type conversion
 */
const char* type_conversion_code(Arena* arena, Type* from, Type* to) {
    assert(arena != NULL);
    assert(from != NULL);
    assert(to != NULL);
    
    // Same type, no conversion needed
    if (type_equals(from, to)) {
        return "";
    }
    
    // Integer to float conversion
    if (from->kind == TYPE_INTEGER && to->kind == TYPE_FLOAT) {
        return "(float)";
    }
    
    // Float to integer conversion
    if (from->kind == TYPE_FLOAT && to->kind == TYPE_INTEGER) {
        return "(int)";
    }
    
    // Vector to scalar conversion (for autodiff)
    if (from->kind == TYPE_VECTOR && to->kind == TYPE_FLOAT) {
        // Extract the first element of the vector
        char* code = arena_alloc(arena, 100);
        if (!code) return "";
        
        snprintf(code, 100, "vector_f_get(%%s, 0)");
        return code;
    }
    
    // Scalar to vector conversion (for autodiff)
    if ((from->kind == TYPE_FLOAT || from->kind == TYPE_INTEGER) && 
        to->kind == TYPE_VECTOR) {
        // Create a vector with a single element
        char* code = arena_alloc(arena, 100);
        if (!code) return "";
        
        snprintf(code, 100, "vector_f_create_from_array(arena, (float[]){%%s}, 1)");
        return code;
    }
    
    // Default: no conversion
    return "";
}

/**
 * @brief Apply type conversion to an expression
 */
char* type_apply_conversion(Arena* arena, const char* expr, Type* from, Type* to) {
    assert(arena != NULL);
    assert(expr != NULL);
    assert(from != NULL);
    assert(to != NULL);
    
    // Same type, no conversion needed
    if (type_equals(from, to)) {
        return (char*)expr;
    }
    
    // Get conversion code
    const char* conversion = type_conversion_code(arena, from, to);
    
    // If no conversion needed
    if (conversion[0] == '\0') {
        return (char*)expr;
    }
    
    // If the conversion is a simple cast
    if (conversion[0] == '(' && conversion[strlen(conversion) - 1] == ')') {
        char* result = arena_alloc(arena, strlen(conversion) + strlen(expr) + 1);
        if (!result) return (char*)expr;
        
        sprintf(result, "%s%s", conversion, expr);
        return result;
    }
    
    // If the conversion is a function call
    if (strstr(conversion, "%s") != NULL) {
        char* result = arena_alloc(arena, strlen(conversion) + strlen(expr) + 1);
        if (!result) return (char*)expr;
        
        sprintf(result, conversion, expr);
        return result;
    }
    
    // Default: no conversion
    return (char*)expr;
}
