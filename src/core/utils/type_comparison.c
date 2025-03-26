/**
 * @file type_comparison.c
 * @brief Implementation of type comparison functions
 */

#include "core/type_comparison.h"
#include "core/type_creation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Check if two types are equal
 */
bool type_equals(Type* a, Type* b) {
    if (a == b) return true;
    if (a == NULL || b == NULL) return false;
    if (a->kind != b->kind) return false;
    
    switch (a->kind) {
        case TYPE_VOID:
        case TYPE_BOOLEAN:
        case TYPE_CHAR:
        case TYPE_STRING:
        case TYPE_SYMBOL:
        case TYPE_ANY:
        case TYPE_UNKNOWN:
            return true;
            
        case TYPE_INTEGER:
            return a->int_size == b->int_size;
            
        case TYPE_FLOAT:
            return a->float_size == b->float_size;
            
        case TYPE_PAIR:
            return type_equals(a->function.params[0], b->function.params[0]) &&
                   type_equals(a->function.params[1], b->function.params[1]);
            
        case TYPE_VECTOR:
            return type_equals(a->vector.element_type, b->vector.element_type) &&
                   a->vector.size == b->vector.size;
            
        case TYPE_FUNCTION:
            if (a->function.param_count != b->function.param_count ||
                a->function.variadic != b->function.variadic ||
                !type_equals(a->function.return_type, b->function.return_type)) {
                return false;
            }
            
            for (size_t i = 0; i < a->function.param_count; i++) {
                if (!type_equals(a->function.params[i], b->function.params[i])) {
                    return false;
                }
            }
            
            return true;
            
        case TYPE_STRUCT:
            if (a->structure.field_count != b->structure.field_count) {
                return false;
            }
            
            for (size_t i = 0; i < a->structure.field_count; i++) {
                if (strcmp(a->structure.fields[i].name, b->structure.fields[i].name) != 0 ||
                    !type_equals(a->structure.fields[i].type, b->structure.fields[i].type)) {
                    return false;
                }
            }
            
            return true;
            
        case TYPE_UNION:
            if (a->union_type.variant_count != b->union_type.variant_count) {
                return false;
            }
            
            // Check if all variants in a are in b
            for (size_t i = 0; i < a->union_type.variant_count; i++) {
                bool found = false;
                
                for (size_t j = 0; j < b->union_type.variant_count; j++) {
                    if (type_equals(a->union_type.variants[i], b->union_type.variants[j])) {
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    return false;
                }
            }
            
            return true;
    }
    
    return false;
}

/**
 * @brief Check if a type is a subtype of another
 */
bool type_is_subtype(Type* sub, Type* super) {
    if (sub == NULL || super == NULL) return false;
    
    // Any type is a subtype of itself
    if (type_equals(sub, super)) return true;
    
    // Any type is a subtype of TYPE_ANY
    if (super->kind == TYPE_ANY) return true;
    
    // TYPE_UNKNOWN is a subtype of any type
    if (sub->kind == TYPE_UNKNOWN) return true;
    
    switch (sub->kind) {
        case TYPE_INTEGER:
            // Integer widening
            if (super->kind == TYPE_INTEGER) {
                return sub->int_size <= super->int_size;
            }
            
            // Integer to float conversion
            if (super->kind == TYPE_FLOAT) {
                return true;
            }
            
            return false;
            
        case TYPE_FLOAT:
            // Float widening
            if (super->kind == TYPE_FLOAT) {
                return sub->float_size <= super->float_size;
            }
            
            return false;
            
        case TYPE_FUNCTION:
            if (super->kind != TYPE_FUNCTION) {
                return false;
            }
            
            // Contravariant parameter types
            if (sub->function.param_count < super->function.param_count) {
                return false;
            }
            
            for (size_t i = 0; i < super->function.param_count; i++) {
                // For contravariance, the parameter types of the supertype must be subtypes of the parameter types of the subtype
                if (!type_is_subtype(super->function.params[i], sub->function.params[i])) {
                    return false;
                }
            }
            
            // Covariant return type
            // For covariance, the return type of the subtype must be a subtype of the return type of the supertype
            if (!type_is_subtype(sub->function.return_type, super->function.return_type)) {
                return false;
            }
            
            // Variadic functions
            if (super->function.variadic && !sub->function.variadic) {
                return false;
            }
            
            return true;
            
        case TYPE_VECTOR:
            if (super->kind != TYPE_VECTOR) {
                return false;
            }
            
            // Vector element type
            if (!type_is_subtype(sub->vector.element_type, super->vector.element_type)) {
                return false;
            }
            
            // Vector size
            if (super->vector.size != 0 && sub->vector.size != super->vector.size) {
                return false;
            }
            
            return true;
            
        case TYPE_STRUCT:
            if (super->kind != TYPE_STRUCT) {
                return false;
            }
            
            // Structural subtyping: all fields in super must be in sub
            for (size_t i = 0; i < super->structure.field_count; i++) {
                bool found = false;
                
                for (size_t j = 0; j < sub->structure.field_count; j++) {
                    if (strcmp(super->structure.fields[i].name, sub->structure.fields[j].name) == 0) {
                        if (!type_is_subtype(sub->structure.fields[j].type, super->structure.fields[i].type)) {
                            return false;
                        }
                        
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    return false;
                }
            }
            
            return true;
            
        case TYPE_UNION:
            if (super->kind == TYPE_UNION) {
                // All variants in sub must be subtypes of some variant in super
                for (size_t i = 0; i < sub->union_type.variant_count; i++) {
                    bool found = false;
                    
                    for (size_t j = 0; j < super->union_type.variant_count; j++) {
                        if (type_is_subtype(sub->union_type.variants[i], super->union_type.variants[j])) {
                            found = true;
                            break;
                        }
                    }
                    
                    if (!found) {
                        return false;
                    }
                }
                
                return true;
            } else {
                // A union type is a subtype of a non-union type if all its variants are subtypes of that type
                for (size_t i = 0; i < sub->union_type.variant_count; i++) {
                    if (!type_is_subtype(sub->union_type.variants[i], super)) {
                        return false;
                    }
                }
                
                return true;
            }
            
        default:
            return false;
    }
}

/**
 * @brief Get the common supertype of two types
 */
Type* type_common_supertype(Arena* arena, Type* a, Type* b) {
    assert(arena != NULL);
    assert(a != NULL);
    assert(b != NULL);
    
    // Same type
    if (type_equals(a, b)) {
        return a;
    }
    
    // Numeric types
    if ((a->kind == TYPE_INTEGER || a->kind == TYPE_FLOAT) &&
        (b->kind == TYPE_INTEGER || b->kind == TYPE_FLOAT)) {
        // If either is float, the result is float
        if (a->kind == TYPE_FLOAT || b->kind == TYPE_FLOAT) {
            // Use the larger float size
            FloatSize size = FLOAT_SIZE_32;
            if (a->kind == TYPE_FLOAT && b->kind == TYPE_FLOAT) {
                size = a->float_size > b->float_size ? a->float_size : b->float_size;
            } else if (a->kind == TYPE_FLOAT) {
                size = a->float_size;
            } else {
                size = b->float_size;
            }
            return type_float_create(arena, size);
        }
        
        // Both are integers, use the larger size
        IntSize size = a->int_size > b->int_size ? a->int_size : b->int_size;
        return type_integer_create(arena, size);
    }
    
    // String and numeric types
    if ((a->kind == TYPE_STRING && (b->kind == TYPE_INTEGER || b->kind == TYPE_FLOAT)) ||
        (b->kind == TYPE_STRING && (a->kind == TYPE_INTEGER || a->kind == TYPE_FLOAT))) {
        // For scientific computing, we need to be careful with mixed types
        // In this case, we'll use void* as a generic return type
        return type_any_create(arena);
    }
    
    // Vector types
    if (a->kind == TYPE_VECTOR && b->kind == TYPE_VECTOR) {
        // Get the common element type
        Type* element_type = type_common_supertype(arena, a->vector.element_type, b->vector.element_type);
        
        // Use the larger size (or 0 for dynamic size)
        size_t size = 0;
        if (a->vector.size == b->vector.size) {
            size = a->vector.size;
        } else if (a->vector.size == 0 || b->vector.size == 0) {
            size = 0;
        } else {
            // Different fixed sizes, use the larger one
            size = a->vector.size > b->vector.size ? a->vector.size : b->vector.size;
        }
        
        return type_vector_create(arena, element_type, size);
    }
    
    // Function types
    if (a->kind == TYPE_FUNCTION && b->kind == TYPE_FUNCTION) {
        // For functions, we need to check parameter and return types
        // For now, we'll just use the first function's type
        // TODO: Implement proper function type unification
        return a;
    }
    
    // Pair types
    if (a->kind == TYPE_PAIR && b->kind == TYPE_PAIR) {
        // Get the common car and cdr types
        Type* car_type = type_common_supertype(arena, a->function.params[0], b->function.params[0]);
        Type* cdr_type = type_common_supertype(arena, a->function.params[1], b->function.params[1]);
        
        return type_pair_create(arena, car_type, cdr_type);
    }
    
    // If one is void, use the other
    if (a->kind == TYPE_VOID) {
        return b;
    }
    if (b->kind == TYPE_VOID) {
        return a;
    }
    
    // If one is any, use any
    if (a->kind == TYPE_ANY || b->kind == TYPE_ANY) {
        return type_any_create(arena);
    }
    
    // Default to any
    return type_any_create(arena);
}
