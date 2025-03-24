/**
 * @file type_inference.c
 * @brief Implementation of the type inference system
 */

#include "frontend/type_inference.h"
#include "core/memory.h"
#include "core/type.h"
#include "frontend/ast/ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

/**
 * @brief Type inference context structure
 */
struct TypeInferenceContext {
    Arena* arena;                /**< Arena for allocations */
    AstNode** nodes;             /**< Array of AST nodes */
    Type** types;                /**< Array of inferred types */
    size_t capacity;             /**< Capacity of the arrays */
    size_t count;                /**< Number of nodes */
};

/**
 * @brief Create a type inference context
 */
TypeInferenceContext* type_inference_context_create(Arena* arena) {
    assert(arena != NULL);
    
    // Allocate context
    TypeInferenceContext* context = arena_alloc(arena, sizeof(TypeInferenceContext));
    if (!context) return NULL;
    
    // Initialize context
    context->arena = arena;
    context->capacity = 1024;
    context->count = 0;
    
    // Allocate arrays
    context->nodes = arena_alloc(arena, context->capacity * sizeof(AstNode*));
    if (!context->nodes) return NULL;
    
    context->types = arena_alloc(arena, context->capacity * sizeof(Type*));
    if (!context->types) return NULL;
    
    return context;
}

/**
 * @brief Add a node to the context
 */
static bool add_node(TypeInferenceContext* context, AstNode* node, Type* type) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Check if the node is already in the context
    for (size_t i = 0; i < context->count; i++) {
        if (context->nodes[i] == node) {
            // Update the type
            context->types[i] = type;
            return true;
        }
    }
    
    // Check if we need to resize the arrays
    if (context->count >= context->capacity) {
        size_t new_capacity = context->capacity * 2;
        
        // Allocate new arrays
        AstNode** new_nodes = arena_alloc(context->arena, new_capacity * sizeof(AstNode*));
        if (!new_nodes) return false;
        
        Type** new_types = arena_alloc(context->arena, new_capacity * sizeof(Type*));
        if (!new_types) return false;
        
        // Copy old arrays to new arrays
        for (size_t i = 0; i < context->count; i++) {
            new_nodes[i] = context->nodes[i];
            new_types[i] = context->types[i];
        }
        
        // Update context
        context->nodes = new_nodes;
        context->types = new_types;
        context->capacity = new_capacity;
    }
    
    // Add the node
    context->nodes[context->count] = node;
    context->types[context->count] = type;
    context->count++;
    
    return true;
}

// Forward declaration
static Type* type_inference_infer_node(TypeInferenceContext* context, AstNode* node);

/**
 * @brief Infer the type of a number literal
 */
static Type* infer_number_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_NUMBER);
    
    // Check if the number is an integer
    if (node->as.number.value == (int)node->as.number.value) {
        return type_integer_create(context->arena, INT_SIZE_32);
    } else {
        return type_float_create(context->arena, FLOAT_SIZE_32);
    }
}

/**
 * @brief Infer the type of a boolean literal
 */
static Type* infer_boolean_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_BOOLEAN);
    
    return type_boolean_create(context->arena);
}

/**
 * @brief Infer the type of a character literal
 */
static Type* infer_character_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_CHARACTER);
    
    return type_char_create(context->arena);
}

/**
 * @brief Infer the type of a string literal
 */
static Type* infer_string_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_STRING);
    
    return type_string_create(context->arena);
}

/**
 * @brief Infer the type of a vector literal
 */
static Type* infer_vector_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_VECTOR);
    
    // Infer the type of the first element
    Type* element_type = NULL;
    if (node->as.vector.count > 0) {
        element_type = type_inference_infer_node(context, node->as.vector.elements[0]);
    } else {
        // Default to float for empty vectors
        element_type = type_float_create(context->arena, FLOAT_SIZE_32);
    }
    
    return type_vector_create(context->arena, element_type, node->as.vector.count);
}

/**
 * @brief Infer the type of a nil literal
 */
static Type* infer_nil_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_NIL);
    
    return type_void_create(context->arena);
}

/**
 * @brief Infer the type of an identifier
 */
static Type* infer_identifier(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_IDENTIFIER);
    
    // Check if the identifier is already in the context
    for (size_t i = 0; i < context->count; i++) {
        if (context->nodes[i]->type == AST_DEFINE &&
            context->nodes[i]->as.define.name->type == AST_IDENTIFIER &&
            strcmp(context->nodes[i]->as.define.name->as.identifier.name, node->as.identifier.name) == 0) {
            return context->types[i];
        }
    }
    
    // If not found, infer from name
    const char* name = node->as.identifier.name;
    
    // Vector operations
    if (strstr(name, "vector") || 
        strstr(name, "gradient") || 
        strstr(name, "curl") ||
        strstr(name, "divergence") ||
        strstr(name, "cross") ||
        strstr(name, "grad") || 
        strstr(name, "v_")) {
        return type_vector_create(context->arena, type_float_create(context->arena, FLOAT_SIZE_32), 0);
    }
    
    // Matrix operations
    if (strstr(name, "jacobian") || 
        strstr(name, "hessian") ||
        strstr(name, "matrix") ||
        strstr(name, "jac") || 
        strstr(name, "hess") ||
        strstr(name, "mat")) {
        // Create a matrix type (represented as a vector of vectors)
        Type* element_type = type_vector_create(context->arena, type_float_create(context->arena, FLOAT_SIZE_32), 0);
        return type_vector_create(context->arena, element_type, 0);
    }
    
    // Scalar operations
    if (strstr(name, "derivative") ||
        strstr(name, "dot") ||
        strstr(name, "norm") ||
        strstr(name, "magnitude") ||
        strstr(name, "sin") ||
        strstr(name, "cos") ||
        strstr(name, "exp") ||
        strstr(name, "log")) {
        return type_float_create(context->arena, FLOAT_SIZE_32);
    }
    
    // Integer operations
    if (strcmp(name, "int") == 0 ||  // Exact match for "int"
        strstr(name, "count_") ||    // Prefix match
        strstr(name, "index_") ||
        strstr(name, "size_") ||
        strstr(name, "length_") ||
        strstr(name, "position_") ||
        strstr(name, "idx_")) {
        return type_integer_create(context->arena, INT_SIZE_32);
    }
    
    // Complex number operations
    if (strstr(name, "complex") ||
        strstr(name, "cmplx")) {
        // For now, we'll represent complex numbers as a vector of two floats
        return type_vector_create(context->arena, type_float_create(context->arena, FLOAT_SIZE_32), 2);
    }
    
    // Rational number operations
    if (strstr(name, "rational") ||
        strstr(name, "ratio") ||
        strstr(name, "frac")) {
        // For now, we'll represent rational numbers as a vector of two integers
        return type_vector_create(context->arena, type_integer_create(context->arena, INT_SIZE_32), 2);
    }
    
    // Boolean operations
    if (strstr(name, "bool") ||
        strstr(name, "flag") ||
        strstr(name, "predicate") ||
        strstr(name, "is_") ||
        strstr(name, "has_")) {
        return type_boolean_create(context->arena);
    }
    
    // String operations
    if (strstr(name, "str") ||
        strstr(name, "name") ||
        strstr(name, "text") ||
        strstr(name, "message") ||
        strstr(name, "label")) {
        return type_string_create(context->arena);
    }
    
    // Character operations
    if (strstr(name, "ch") ||
        strstr(name, "char")) {
        return type_char_create(context->arena);
    }
    
    // Symbol operations
    if (strstr(name, "symbol") ||
        strstr(name, "sym")) {
        return type_symbol_create(context->arena);
    }
    
    // Pair and list operations
    if (strstr(name, "pair") ||
        strstr(name, "cons")) {
        // Create a pair type with unknown element types
        return type_pair_create(context->arena, 
                               type_any_create(context->arena), 
                               type_any_create(context->arena));
    }
    
    if (strstr(name, "list") ||
        strstr(name, "lst")) {
        // For lists, we'll use a pair type with the cdr being another pair or nil
        Type* element_type = type_any_create(context->arena);
        Type* rest_type = type_pair_create(context->arena, element_type, 
                                          type_any_create(context->arena));
        return type_pair_create(context->arena, element_type, rest_type);
    }
    
    // Port operations (I/O)
    if (strstr(name, "port") ||
        strstr(name, "input") ||
        strstr(name, "output") ||
        strstr(name, "io") ||
        strstr(name, "file")) {
        // For now, we'll represent ports as a special kind of pointer
        return type_any_create(context->arena); // TODO: Create a proper port type
    }
    
    // Procedure operations
    if (strstr(name, "proc") ||
        strstr(name, "func") ||
        strstr(name, "callback") ||
        strstr(name, "handler")) {
        // Create a function type with unknown parameters and return type
        return type_function_create(context->arena, 0, NULL, 
                                   type_any_create(context->arena), true);
    }
    
    // Default to float for numeric variables, as that's most common in scientific computing
    return type_float_create(context->arena, FLOAT_SIZE_32);
}

/**
 * @brief Infer the type of a function call
 */
static Type* infer_call(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_CALL);
    
    // Check if it's an operator call
    if (node->as.call.callee->type == AST_IDENTIFIER) {
        const char* op_name = node->as.call.callee->as.identifier.name;
        
        // Handle arithmetic operators
        if (strcmp(op_name, "+") == 0 || 
            strcmp(op_name, "-") == 0 || 
            strcmp(op_name, "*") == 0 || 
            strcmp(op_name, "/") == 0) {
            // Infer the type of the operands
            Type* left_type = type_inference_infer_node(context, node->as.call.args[0]);
            Type* right_type = NULL;
            
            if (node->as.call.arg_count > 1) {
                right_type = type_inference_infer_node(context, node->as.call.args[1]);
            } else {
                // Unary operator
                return left_type;
            }
            
            // If both operands are integers, the result is an integer
            if (left_type->kind == TYPE_INTEGER && right_type->kind == TYPE_INTEGER) {
                return type_integer_create(context->arena, INT_SIZE_32);
            }
            
            // If either operand is a float, the result is a float
            if (left_type->kind == TYPE_FLOAT || right_type->kind == TYPE_FLOAT) {
                return type_float_create(context->arena, FLOAT_SIZE_32);
            }
            
            // If either operand is a vector, the result is a vector
            if (left_type->kind == TYPE_VECTOR || right_type->kind == TYPE_VECTOR) {
                // Determine the element type
                Type* element_type = NULL;
                
                if (left_type->kind == TYPE_VECTOR && right_type->kind == TYPE_VECTOR) {
                    // Both operands are vectors, use the common supertype of the element types
                    element_type = type_common_supertype(context->arena, left_type->vector.element_type, right_type->vector.element_type);
                } else if (left_type->kind == TYPE_VECTOR) {
                    // Left operand is a vector, use its element type
                    element_type = left_type->vector.element_type;
                } else {
                    // Right operand is a vector, use its element type
                    element_type = right_type->vector.element_type;
                }
                
                return type_vector_create(context->arena, element_type, 0);
            }
            
            // Default to float
            return type_float_create(context->arena, FLOAT_SIZE_32);
        }
        
        // Handle comparison operators
        else if (strcmp(op_name, "<") == 0 || 
                 strcmp(op_name, ">") == 0 || 
                 strcmp(op_name, "<=") == 0 || 
                 strcmp(op_name, ">=") == 0 || 
                 strcmp(op_name, "=") == 0) {
            return type_boolean_create(context->arena);
        }
        
        // Handle vector operations
        else if (strcmp(op_name, "vector") == 0) {
            // Vector literal
            Type* element_type = NULL;
            
            if (node->as.call.arg_count > 0) {
                element_type = type_inference_infer_node(context, node->as.call.args[0]);
            } else {
                // Default to float for empty vectors
                element_type = type_float_create(context->arena, FLOAT_SIZE_32);
            }
            
            return type_vector_create(context->arena, element_type, node->as.call.arg_count);
        } else if (strcmp(op_name, "v+") == 0 || 
                   strcmp(op_name, "v-") == 0) {
            // Vector addition/subtraction
            Type* left_type = type_inference_infer_node(context, node->as.call.args[0]);
            Type* right_type = type_inference_infer_node(context, node->as.call.args[1]);
            
            // Both operands should be vectors
            if (left_type->kind == TYPE_VECTOR && right_type->kind == TYPE_VECTOR) {
                // Use the common supertype of the element types
                Type* element_type = type_common_supertype(context->arena, left_type->vector.element_type, right_type->vector.element_type);
                return type_vector_create(context->arena, element_type, 0);
            }
            
            // Default to float vector
            return type_vector_create(context->arena, type_float_create(context->arena, FLOAT_SIZE_32), 0);
        } else if (strcmp(op_name, "v*") == 0) {
            // Vector scalar multiplication
            Type* left_type = type_inference_infer_node(context, node->as.call.args[0]);
            Type* right_type = type_inference_infer_node(context, node->as.call.args[1]);
            
            // One operand should be a vector and the other a scalar
            if (left_type->kind == TYPE_VECTOR && (right_type->kind == TYPE_INTEGER || right_type->kind == TYPE_FLOAT)) {
                return type_vector_create(context->arena, left_type->vector.element_type, 0);
            } else if ((left_type->kind == TYPE_INTEGER || left_type->kind == TYPE_FLOAT) && right_type->kind == TYPE_VECTOR) {
                return type_vector_create(context->arena, right_type->vector.element_type, 0);
            }
            
            // Default to float vector
            return type_vector_create(context->arena, type_float_create(context->arena, FLOAT_SIZE_32), 0);
        } else if (strcmp(op_name, "dot") == 0) {
            // Dot product
            return type_float_create(context->arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "cross") == 0) {
            // Cross product
            return type_vector_create(context->arena, type_float_create(context->arena, FLOAT_SIZE_32), 0);
        } else if (strcmp(op_name, "norm") == 0) {
            // Vector magnitude
            return type_float_create(context->arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "gradient") == 0) {
            // Gradient of a scalar field
            return type_vector_create(context->arena, type_float_create(context->arena, FLOAT_SIZE_32), 0);
        } else if (strcmp(op_name, "divergence") == 0) {
            // Divergence of a vector field
            return type_float_create(context->arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "curl") == 0) {
            // Curl of a vector field
            return type_vector_create(context->arena, type_float_create(context->arena, FLOAT_SIZE_32), 0);
        } else if (strcmp(op_name, "laplacian") == 0) {
            // Laplacian of a scalar field
            return type_float_create(context->arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "display") == 0) {
            // Display function (Scheme compatibility)
            return type_void_create(context->arena);
        }
        
        // Handle autodiff functions
        else if (strcmp(op_name, "autodiff-forward") == 0 || 
                 strcmp(op_name, "autodiff-reverse") == 0) {
            // Forward/reverse-mode autodiff
            return type_float_create(context->arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "autodiff-forward-gradient") == 0 || 
                   strcmp(op_name, "autodiff-reverse-gradient") == 0) {
            // Forward/reverse-mode gradient
            return type_vector_create(context->arena, type_float_create(context->arena, FLOAT_SIZE_32), 0);
        } else if (strcmp(op_name, "autodiff-jacobian") == 0) {
            // Jacobian matrix
            // Create a matrix type (represented as a vector of vectors)
            Type* element_type = type_vector_create(context->arena, type_float_create(context->arena, FLOAT_SIZE_32), 0);
            return type_vector_create(context->arena, element_type, 0);
        } else if (strcmp(op_name, "autodiff-hessian") == 0) {
            // Hessian matrix
            // Create a matrix type (represented as a vector of vectors)
            Type* element_type = type_vector_create(context->arena, type_float_create(context->arena, FLOAT_SIZE_32), 0);
            return type_vector_create(context->arena, element_type, 0);
        } else if (strcmp(op_name, "derivative") == 0) {
            // Derivative of a function at a point
            return type_float_create(context->arena, FLOAT_SIZE_32);
        }
        
        // Handle vector and matrix operations
        else if (strcmp(op_name, "vector-ref") == 0) {
            // Vector element access
            Type* vec_type = type_inference_infer_node(context, node->as.call.args[0]);
            
            if (vec_type->kind == TYPE_VECTOR) {
                return vec_type->vector.element_type;
            }
            
            // Default to float
            return type_float_create(context->arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "matrix-ref") == 0) {
            // Matrix element access
            Type* mat_type = type_inference_infer_node(context, node->as.call.args[0]);
            
            if (mat_type->kind == TYPE_VECTOR && mat_type->vector.element_type->kind == TYPE_VECTOR) {
                return mat_type->vector.element_type->vector.element_type;
            }
            
            // Default to float
            return type_float_create(context->arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "printf") == 0) {
            // Printf function
            return type_void_create(context->arena);
        } else if (strcmp(op_name, "string-append") == 0) {
            // String append function
            return type_string_create(context->arena);
        } else if (strcmp(op_name, "number->string") == 0) {
            // Number to string conversion
            return type_string_create(context->arena);
        } else if (strcmp(op_name, "string->number") == 0) {
            // String to number conversion
            return type_float_create(context->arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "string->symbol") == 0) {
            // String to symbol conversion
            return type_symbol_create(context->arena);
        } else if (strcmp(op_name, "symbol->string") == 0) {
            // Symbol to string conversion
            return type_string_create(context->arena);
        } else if (strcmp(op_name, "char->integer") == 0) {
            // Character to integer conversion
            return type_integer_create(context->arena, INT_SIZE_32);
        } else if (strcmp(op_name, "integer->char") == 0) {
            // Integer to character conversion
            return type_char_create(context->arena);
        } else if (strcmp(op_name, "cons") == 0) {
            // Cons operation (create a pair)
            Type* car_type = type_inference_infer_node(context, node->as.call.args[0]);
            Type* cdr_type = type_inference_infer_node(context, node->as.call.args[1]);
            return type_pair_create(context->arena, car_type, cdr_type);
        } else if (strcmp(op_name, "car") == 0) {
            // Car operation (get first element of pair)
            Type* pair_type = type_inference_infer_node(context, node->as.call.args[0]);
            if (pair_type->kind == TYPE_PAIR) {
                return pair_type->function.params[0]; // Car type
            }
            return type_any_create(context->arena);
        } else if (strcmp(op_name, "cdr") == 0) {
            // Cdr operation (get rest of pair)
            Type* pair_type = type_inference_infer_node(context, node->as.call.args[0]);
            if (pair_type->kind == TYPE_PAIR) {
                return pair_type->function.params[1]; // Cdr type
            }
            return type_any_create(context->arena);
        } else if (strcmp(op_name, "list") == 0) {
            // List creation
            // Infer the type of the first element
            Type* element_type = NULL;
            if (node->as.call.arg_count > 0) {
                element_type = type_inference_infer_node(context, node->as.call.args[0]);
            } else {
                // Empty list
                return type_void_create(context->arena);
            }
            
            // Create a list type (a pair with the cdr being another pair or nil)
            Type* rest_type = type_void_create(context->arena); // Empty list for the last cdr
            
            // Build the list type from the end to the beginning
            for (int i = node->as.call.arg_count - 1; i >= 0; i--) {
                Type* item_type = type_inference_infer_node(context, node->as.call.args[i]);
                rest_type = type_pair_create(context->arena, item_type, rest_type);
            }
            
            return rest_type;
        } else if (strcmp(op_name, "make-vector") == 0) {
            // Vector creation
            // The second argument (if present) gives the element type
            Type* element_type = NULL;
            if (node->as.call.arg_count > 1) {
                element_type = type_inference_infer_node(context, node->as.call.args[1]);
            } else {
                // Default to float for vector elements
                element_type = type_float_create(context->arena, FLOAT_SIZE_32);
            }
            
            // Get the size if specified
            size_t size = 0; // Dynamic size
            if (node->as.call.arg_count > 0 && 
                node->as.call.args[0]->type == AST_LITERAL_NUMBER) {
                size = (size_t)node->as.call.args[0]->as.number.value;
            }
            
            return type_vector_create(context->arena, element_type, size);
        } else if (strcmp(op_name, "vector-length") == 0) {
            // Vector length
            return type_integer_create(context->arena, INT_SIZE_32);
        } else if (strcmp(op_name, "vector-set!") == 0) {
            // Vector element setting
            return type_void_create(context->arena);
        } else if (strcmp(op_name, "make-string") == 0) {
            // String creation
            return type_string_create(context->arena);
        } else if (strcmp(op_name, "string-length") == 0) {
            // String length
            return type_integer_create(context->arena, INT_SIZE_32);
        } else if (strcmp(op_name, "string-ref") == 0) {
            // String element access
            return type_char_create(context->arena);
        } else if (strcmp(op_name, "string-set!") == 0) {
            // String element setting
            return type_void_create(context->arena);
        } else if (strcmp(op_name, "open-input-file") == 0 ||
                  strcmp(op_name, "open-output-file") == 0) {
            // File opening
            return type_any_create(context->arena); // Port type
        } else if (strcmp(op_name, "close-port") == 0 ||
                  strcmp(op_name, "close-input-port") == 0 ||
                  strcmp(op_name, "close-output-port") == 0) {
            // Port closing
            return type_void_create(context->arena);
        } else if (strcmp(op_name, "read") == 0 ||
                  strcmp(op_name, "read-char") == 0 ||
                  strcmp(op_name, "read-line") == 0) {
            // Reading from port
            return type_any_create(context->arena); // Could be any type
        } else if (strcmp(op_name, "write") == 0 ||
                  strcmp(op_name, "write-char") == 0 ||
                  strcmp(op_name, "write-string") == 0 ||
                  strcmp(op_name, "newline") == 0) {
            // Writing to port
            return type_void_create(context->arena);
        } else if (strcmp(op_name, "apply") == 0) {
            // Apply function
            // The first argument is the function to apply
            Type* func_type = type_inference_infer_node(context, node->as.call.args[0]);
            if (func_type->kind == TYPE_FUNCTION) {
                return func_type->function.return_type;
            }
            return type_any_create(context->arena);
        } else if (strcmp(op_name, "map") == 0 ||
                  strcmp(op_name, "for-each") == 0) {
            // Map/for-each function
            // The first argument is the function to apply
            Type* func_type = type_inference_infer_node(context, node->as.call.args[0]);
            if (func_type->kind == TYPE_FUNCTION) {
                if (strcmp(op_name, "map") == 0) {
                    // Map returns a list of results
                    return type_pair_create(context->arena, 
                                          func_type->function.return_type, 
                                          type_void_create(context->arena));
                } else {
                    // for-each returns void
                    return type_void_create(context->arena);
                }
            }
            return type_any_create(context->arena);
        }
    }
    
    // Regular function call
    // Infer the type of the callee
    Type* callee_type = type_inference_infer_node(context, node->as.call.callee);
    
    // If the callee is a function, return its return type
    if (callee_type->kind == TYPE_FUNCTION) {
        return callee_type->function.return_type;
    }
    
    // Default to float
    return type_float_create(context->arena, FLOAT_SIZE_32);
}

/**
 * @brief Infer the type of an if expression
 */
static Type* infer_if(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_IF);
    
    // Infer the type of the condition
    Type* condition_type = type_inference_infer_node(context, node->as.if_expr.condition);
    
    // Infer the type of the then branch
    Type* then_type = type_inference_infer_node(context, node->as.if_expr.then_branch);
    
    // Infer the type of the else branch
    Type* else_type = type_inference_infer_node(context, node->as.if_expr.else_branch);
    
    // Return the common supertype of the then and else branches
    return type_common_supertype(context->arena, then_type, else_type);
}

/**
 * @brief Infer the type of a begin expression
 */
static Type* infer_begin(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_BEGIN);
    
    // Infer the type of each expression
    Type* type = NULL;
    
    for (size_t i = 0; i < node->as.begin.expr_count; i++) {
        type = type_inference_infer_node(context, node->as.begin.exprs[i]);
    }
    
    // Return the type of the last expression
    return type;
}

/**
 * @brief Infer the type of a lambda expression
 */
static Type* infer_lambda(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LAMBDA);
    
    // Infer the type of the body
    Type* body_type = type_inference_infer_node(context, node->as.lambda.body);
    
    // Create parameter types
    Type** param_types = arena_alloc(context->arena, node->as.lambda.param_count * sizeof(Type*));
    if (!param_types) return NULL;
    
    for (size_t i = 0; i < node->as.lambda.param_count; i++) {
        // Default to float for parameters
        param_types[i] = type_float_create(context->arena, FLOAT_SIZE_32);
    }
    
    // Create function type
    return type_function_create(context->arena, node->as.lambda.param_count, param_types, body_type, false);
}

/**
 * @brief Infer the type of a define expression
 */
static Type* infer_define(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_DEFINE);
    
    // Infer the type of the value
    Type* value_type = type_inference_infer_node(context, node->as.define.value);
    
    // Add the definition to the context
    add_node(context, node, value_type);
    
    return value_type;
}

/**
 * @brief Infer the type of a function definition
 */
static Type* infer_function_def(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_FUNCTION_DEF);
    
    // Infer the type of the body
    Type* body_type = type_inference_infer_node(context, node->as.function_def.body);
    
    // Create parameter types
    Type** param_types = arena_alloc(context->arena, node->as.function_def.param_count * sizeof(Type*));
    if (!param_types) return NULL;
    
    for (size_t i = 0; i < node->as.function_def.param_count; i++) {
        // Use the parameter type if specified, otherwise default to float
        if (node->as.function_def.params[i]->type != NULL) {
            param_types[i] = node->as.function_def.params[i]->type;
        } else {
            param_types[i] = type_float_create(context->arena, FLOAT_SIZE_32);
        }
    }
    
    // Create function type
    Type* function_type = type_function_create(context->arena, node->as.function_def.param_count, param_types, body_type, false);
    
    // Add the function to the context
    add_node(context, node, function_type);
    
    return function_type;
}

/**
 * @brief Infer the type of a variable definition
 */
static Type* infer_variable_def(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_VARIABLE_DEF);
    
    // Infer the type of the value
    Type* value_type = type_inference_infer_node(context, node->as.variable_def.value);
    
    // Add the variable to the context
    add_node(context, node, value_type);
    
    return value_type;
}

/**
 * @brief Infer the type of a let expression
 */
static Type* infer_let(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LET);
    
    // Infer the type of each binding
    for (size_t i = 0; i < node->as.let.binding_count; i++) {
        type_inference_infer_node(context, node->as.let.bindings[i]);
    }
    
    // Infer the type of the body
    return type_inference_infer_node(context, node->as.let.body);
}

/**
 * @brief Infer the type of a sequence of expressions
 */
static Type* infer_sequence(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_SEQUENCE);
    
    // Infer the type of each expression
    Type* type = NULL;
    
    for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
        type = type_inference_infer_node(context, node->as.sequence.exprs[i]);
    }
    
    // Return the type of the last expression
    return type;
}

/**
 * @brief Infer the type of an AST node
 */
static Type* type_inference_infer_node(TypeInferenceContext* context, AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Check if the node is already in the context
    for (size_t i = 0; i < context->count; i++) {
        if (context->nodes[i] == node) {
            return context->types[i];
        }
    }
    
    // Infer the type based on the node type
    Type* type = NULL;
    
    switch (node->type) {
        case AST_LITERAL_NUMBER:
            type = infer_number_literal(context, node);
            break;
        case AST_LITERAL_BOOLEAN:
            type = infer_boolean_literal(context, node);
            break;
        case AST_LITERAL_CHARACTER:
            type = infer_character_literal(context, node);
            break;
        case AST_LITERAL_STRING:
            type = infer_string_literal(context, node);
            break;
        case AST_LITERAL_VECTOR:
            type = infer_vector_literal(context, node);
            break;
        case AST_LITERAL_NIL:
            type = infer_nil_literal(context, node);
            break;
        case AST_IDENTIFIER:
            type = infer_identifier(context, node);
            break;
        case AST_CALL:
            type = infer_call(context, node);
            break;
        case AST_IF:
            type = infer_if(context, node);
            break;
        case AST_BEGIN:
            type = infer_begin(context, node);
            break;
        case AST_LAMBDA:
            type = infer_lambda(context, node);
            break;
        case AST_DEFINE:
            type = infer_define(context, node);
            break;
        case AST_FUNCTION_DEF:
            type = infer_function_def(context, node);
            break;
        case AST_VARIABLE_DEF:
            type = infer_variable_def(context, node);
            break;
        case AST_LET:
            type = infer_let(context, node);
            break;
        case AST_SEQUENCE:
            type = infer_sequence(context, node);
            break;
        default:
            // Default to float
            type = type_float_create(context->arena, FLOAT_SIZE_32);
            break;
    }
    
    // Add the node to the context
    add_node(context, node, type);
    
    return type;
}

/**
 * @brief Infer types for a program
 */
static bool infer_program(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_PROGRAM);
    
    // Infer the type of each expression
    for (size_t i = 0; i < node->as.program.expr_count; i++) {
        if (!type_inference_infer_node(context, node->as.program.exprs[i])) {
            return false;
        }
    }
    
    return true;
}

/**
 * @brief Infer types for an AST
 */
bool type_inference_infer(TypeInferenceContext* context, AstNode* ast) {
    assert(context != NULL);
    assert(ast != NULL);
    
    if (ast->type == AST_PROGRAM) {
        return infer_program(context, ast);
    } else {
        return type_inference_infer_node(context, ast) != NULL;
    }
}

/**
 * @brief Get the inferred type for an AST node
 */
Type* type_inference_get_type(TypeInferenceContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Check if the node is in the context
    for (size_t i = 0; i < context->count; i++) {
        if (context->nodes[i] == node) {
            return context->types[i];
        }
    }
    
    return NULL;
}

/**
 * @brief Convert a Type to a C type string
 */
static const char* type_to_c_type(Type* type) {
    if (type == NULL) {
        return "int"; // Default to int for untyped parameters
    }
    
    switch (type->kind) {
        case TYPE_VOID:
            return "void";
        case TYPE_BOOLEAN:
            return "bool";
        case TYPE_INTEGER:
            switch (type->int_size) {
                case INT_SIZE_8:
                    return "int8_t";
                case INT_SIZE_16:
                    return "int16_t";
                case INT_SIZE_32:
                    return "int32_t";
                case INT_SIZE_64:
                    return "int64_t";
                default:
                    return "int";
            }
        case TYPE_FLOAT:
            switch (type->float_size) {
                case FLOAT_SIZE_32:
                    return "float";
                case FLOAT_SIZE_64:
                    return "double";
                default:
                    return "float";
            }
        case TYPE_CHAR:
            return "char";
        case TYPE_STRING:
            return "char*";
        case TYPE_SYMBOL:
            return "const char*";
        case TYPE_PAIR:
            return "void*"; // Pairs are represented as pointers
        case TYPE_VECTOR:
            return "VectorF*"; // Vectors are represented as VectorF*
        case TYPE_FUNCTION:
            return "void*"; // Functions are represented as function pointers
        case TYPE_STRUCT:
            return "void*"; // Structs are represented as pointers
        case TYPE_UNION:
            return "void*"; // Unions are represented as pointers
        case TYPE_ANY:
            return "eshkol_value_t"; // Any type is represented as eshkol_value_t
        case TYPE_UNKNOWN:
            return "void*"; // Unknown type is represented as void*
        default:
            return "int"; // Default to int
    }
}

/**
 * @brief Get the C type string for an AST node
 */
const char* type_inference_get_c_type(TypeInferenceContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Get the inferred type
    Type* type = type_inference_get_type(context, node);
    
    // Convert to C type string
    return type_to_c_type(type);
}
