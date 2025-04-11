/**
 * @file inference.c
 * @brief Implementation of the type inference system
 */

#include "frontend/type_inference/inference.h"
#include "backend/codegen/context.h"
#include "frontend/type_inference/context.h"
#include "core/memory.h"
#include "core/type.h"
#include "frontend/binding/binding.h"
#include "core/type_creation.h"
#include "core/type_comparison.h"
#include "core/type_conversion.h"
#include "frontend/ast/ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <ctype.h>

// Forward declaration
Type* type_inference_infer_node(TypeInferenceContext* context, AstNode* node);

/**
 * @brief Infer the type of a number literal
 */
static Type* infer_number_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_NUMBER);
    
    // Check if the number is an integer
    if (node->as.number.value == (int)node->as.number.value) {
        return type_integer_create(type_inference_get_arena(context), INT_SIZE_32);
    } else {
        return type_float_create(type_inference_get_arena(context), FLOAT_SIZE_32);
    }
}

/**
 * @brief Infer the type of a boolean literal
 */
static Type* infer_boolean_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_BOOLEAN);
    
    return type_boolean_create(type_inference_get_arena(context));
}

/**
 * @brief Infer the type of a character literal
 */
static Type* infer_character_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_CHARACTER);
    
    return type_char_create(type_inference_get_arena(context));
}

/**
 * @brief Infer the type of a string literal
 */
static Type* infer_string_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_STRING);
    
    return type_string_create(type_inference_get_arena(context));
}

/**
 * @brief Infer the type of a vector literal
 */
static Type* infer_vector_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_VECTOR);
    Arena* arena = type_inference_get_arena(context);
    
    // Infer the type of the first element
    Type* element_type = NULL;
    if (node->as.vector.count > 0) {
        element_type = type_inference_infer_node(context, node->as.vector.elements[0]);
    } else {
        // Default to float for empty vectors
        element_type = type_float_create(arena, FLOAT_SIZE_32);
    }
    
    return type_vector_create(arena, element_type, node->as.vector.count);
}

/**
 * @brief Infer the type of a nil literal
 */
static Type* infer_nil_literal(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_LITERAL_NIL);
    
    return type_void_create(type_inference_get_arena(context));
}

/**
 * @brief Infer the type of an identifier
 */
static Type* infer_identifier(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_IDENTIFIER);

    Arena* arena = type_inference_get_arena(context);

    // Get binding system
    BindingSystem* binding_system = type_inference_get_binding_system(context);

    // Check if the identifier name contains special characters
    const char* name = node->as.identifier.name;
    for (size_t i = 0; name[i] != '\0'; i++) {
        char c = name[i];
        if (!isalnum(c) && c != '_') {
            // Special identifiers like operators are not variables
            return type_any_create(arena);
        }
    } 

    void* function_signature = type_inference_get_function_signature(context, name);
    if (function_signature) {
      return type_integer_create(arena, INT_SIZE_32);
    }

    // Vector operations
    if (strstr(name, "vector") || 
        strstr(name, "gradient") || 
        strstr(name, "curl") ||
        strstr(name, "divergence") ||
        strstr(name, "cross") ||
        strstr(name, "grad") || 
        strstr(name, "v_")) {
        return type_vector_create(arena, type_float_create(arena, FLOAT_SIZE_32), 0);
    }
    
    // Matrix operations
    if (strstr(name, "jacobian") || 
        strstr(name, "hessian") ||
        strstr(name, "matrix") ||
        strstr(name, "jac") || 
        strstr(name, "hess") ||
        strstr(name, "mat")) {
        // Create a matrix type (represented as a vector of vectors)
        Type* element_type = type_vector_create(arena, type_float_create(arena, FLOAT_SIZE_32), 0);
        return type_vector_create(arena, element_type, 0);
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
        return type_float_create(arena, FLOAT_SIZE_32);
    }
    
    // Integer operations
    if (strcmp(name, "int") == 0 ||  // Exact match for "int"
        strstr(name, "count_") ||    // Prefix match
        strstr(name, "index_") ||
        strstr(name, "size_") ||
        strstr(name, "length_") ||
        strstr(name, "position_") ||
        strstr(name, "idx_")) {
        return type_integer_create(arena, INT_SIZE_32);
    }
    
    // Complex number operations
    if (strstr(name, "complex") ||
        strstr(name, "cmplx")) {
        // For now, we'll represent complex numbers as a vector of two floats
        return type_vector_create(arena, type_float_create(arena, FLOAT_SIZE_32), 2);
    }
    
    // Rational number operations
    if (strstr(name, "rational") ||
        strstr(name, "ratio") ||
        strstr(name, "frac")) {
        // For now, we'll represent rational numbers as a vector of two integers
        return type_vector_create(arena, type_integer_create(arena, INT_SIZE_32), 2);
    }
    
    // Boolean operations
    if (strstr(name, "bool") ||
        strstr(name, "flag") ||
        strstr(name, "predicate") ||
        strstr(name, "is_") ||
        strstr(name, "has_")) {
        return type_boolean_create(arena);
    }
    
    // String operations
    if (strstr(name, "str") ||
        strstr(name, "name") ||
        strstr(name, "text") ||
        strstr(name, "message") ||
        strstr(name, "label")) {
        return type_string_create(arena);
    }
    
    // Character operations
    if (strstr(name, "ch") ||
        strstr(name, "char")) {
        return type_char_create(arena);
    }
    
    // Symbol operations
    if (strstr(name, "symbol") ||
        strstr(name, "sym")) {
        return type_symbol_create(arena);
    }
    
    // Pair and list operations
    if (strstr(name, "pair") ||
        strstr(name, "cons")) {
        // Create a pair type with unknown element types
        return type_pair_create(arena, 
                               type_any_create(arena), 
                               type_any_create(arena));
    }
    
    if (strstr(name, "list") ||
        strstr(name, "lst")) {
        // For lists, we'll use a pair type with the cdr being another pair or nil
        Type* element_type = type_any_create(arena);
        Type* rest_type = type_pair_create(arena, element_type, 
                                          type_any_create(arena));
        return type_pair_create(arena, element_type, rest_type);
    }
    
    // Port operations (I/O)
    if (strstr(name, "port") ||
        strstr(name, "input") ||
        strstr(name, "output") ||
        strstr(name, "io") ||
        strstr(name, "file")) {
        // For now, we'll represent ports as a special kind of pointer
        return type_any_create(arena); // TODO: Create a proper port type
    }
    
    // Procedure operations
    if (strstr(name, "proc") ||
        strstr(name, "func") ||
        strstr(name, "callback") ||
        strstr(name, "handler")) {
        // Create a function type with unknown parameters and return type
        return type_function_create(arena, 0, NULL, 
                                   type_any_create(arena), true);
    }
    
    // Default to float for numeric variables, as that's most common in scientific computing
    return type_float_create(arena, FLOAT_SIZE_32);
}

/**
 * @brief Infer the type of a call expression
 */
static Type* infer_call(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_CALL);
    Arena* arena = type_inference_get_arena(context);
    
    // Check if it's an operator call
    if (node->as.call.callee->type == AST_IDENTIFIER) {
        const char* op_name = node->as.call.callee->as.identifier.name;

        BindingSystem* binding_system = type_inference_get_binding_system(context);
        
        // Resolve the identifier in the binding system
        uint64_t binding_id = binding_system_resolve_binding(binding_system, op_name);
        printf("Resolved binding ID %lu for %s\n", binding_id, op_name);
        if (binding_id != 0) {
          AstNode* binded_node = binding_system_get_definition(binding_system, binding_id);
          if(binded_node) {
            return type_inference_infer_node(context, binded_node);
          }
        }
        void* function_signature = type_inference_get_function_signature(context, op_name);
        if (function_signature) {
          return type_integer_create(arena, INT_SIZE_32);
        }
        // Handle string-append and number->string operations
        if (strcmp(op_name, "string-append") == 0) {
            return type_string_create(arena);
        } else if (strcmp(op_name, "number->string") == 0) {
            return type_string_create(arena);
        }
        
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
                return type_integer_create(arena, INT_SIZE_32);
            }
            
            // If either operand is a float, the result is a float
            if (left_type->kind == TYPE_FLOAT || right_type->kind == TYPE_FLOAT) {
                return type_float_create(arena, FLOAT_SIZE_32);
            }
            
            // If either operand is a vector, the result is a vector
            if (left_type->kind == TYPE_VECTOR || right_type->kind == TYPE_VECTOR) {
                // Determine the element type
                Type* element_type = NULL;
                
                if (left_type->kind == TYPE_VECTOR && right_type->kind == TYPE_VECTOR) {
                    // Both operands are vectors, use the common supertype of the element types
                    element_type = type_common_supertype(arena, left_type->vector.element_type, right_type->vector.element_type);
                } else if (left_type->kind == TYPE_VECTOR) {
                    // Left operand is a vector, use its element type
                    element_type = left_type->vector.element_type;
                } else {
                    // Right operand is a vector, use its element type
                    element_type = right_type->vector.element_type;
                }
                
                return type_vector_create(arena, element_type, 0);
            }
            
            // Default to float
            return type_float_create(arena, FLOAT_SIZE_32);
        }
        
        // Handle comparison operators
        else if (strcmp(op_name, "<") == 0 || 
                 strcmp(op_name, ">") == 0 || 
                 strcmp(op_name, "<=") == 0 || 
                 strcmp(op_name, ">=") == 0 || 
                 strcmp(op_name, "=") == 0) {
            return type_boolean_create(arena);
        }
        
        // Handle vector operations
        else if (strcmp(op_name, "vector") == 0) {
            // Vector literal
            Type* element_type = NULL;
            
            if (node->as.call.arg_count > 0) {
                element_type = type_inference_infer_node(context, node->as.call.args[0]);
            } else {
                // Default to float for empty vectors
                element_type = type_float_create(arena, FLOAT_SIZE_32);
            }
            
            return type_vector_create(arena, element_type, node->as.call.arg_count);
        } else if (strcmp(op_name, "v+") == 0 || 
                   strcmp(op_name, "v-") == 0) {
            // Vector addition/subtraction
            Type* left_type = type_inference_infer_node(context, node->as.call.args[0]);
            Type* right_type = type_inference_infer_node(context, node->as.call.args[1]);
            
            // Both operands should be vectors
            if (left_type->kind == TYPE_VECTOR && right_type->kind == TYPE_VECTOR) {
                // Use the common supertype of the element types
                Type* element_type = type_common_supertype(arena, left_type->vector.element_type, right_type->vector.element_type);
                return type_vector_create(arena, element_type, 0);
            }
            
            // Default to float vector
            return type_vector_create(arena, type_float_create(arena, FLOAT_SIZE_32), 0);
        } else if (strcmp(op_name, "v*") == 0) {
            // Vector scalar multiplication
            Type* left_type = type_inference_infer_node(context, node->as.call.args[0]);
            Type* right_type = type_inference_infer_node(context, node->as.call.args[1]);
            
            // One operand should be a vector and the other a scalar
            if (left_type->kind == TYPE_VECTOR && (right_type->kind == TYPE_INTEGER || right_type->kind == TYPE_FLOAT)) {
                return type_vector_create(arena, left_type->vector.element_type, 0);
            } else if ((left_type->kind == TYPE_INTEGER || left_type->kind == TYPE_FLOAT) && right_type->kind == TYPE_VECTOR) {
                return type_vector_create(arena, right_type->vector.element_type, 0);
            }
            
            // Default to float vector
            return type_vector_create(arena, type_float_create(arena, FLOAT_SIZE_32), 0);
        } else if (strcmp(op_name, "dot") == 0) {
            // Dot product
            return type_float_create(arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "cross") == 0) {
            // Cross product
            return type_vector_create(arena, type_float_create(arena, FLOAT_SIZE_32), 0);
        } else if (strcmp(op_name, "norm") == 0) {
            // Vector magnitude
            return type_float_create(arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "gradient") == 0) {
            // Gradient of a scalar field
            return type_vector_create(arena, type_float_create(arena, FLOAT_SIZE_32), 0);
        } else if (strcmp(op_name, "divergence") == 0) {
            // Divergence of a vector field
            return type_float_create(arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "curl") == 0) {
            // Curl of a vector field
            return type_vector_create(arena, type_float_create(arena, FLOAT_SIZE_32), 0);
        } else if (strcmp(op_name, "laplacian") == 0) {
            // Laplacian of a scalar field
            return type_float_create(arena, FLOAT_SIZE_32);
        } else if (strcmp(op_name, "display") == 0) {
            // Display function (Scheme compatibility)
            return type_void_create(arena);
        }
        
        // Handle autodiff functions
        else if (strcmp(op_name, "autodiff-forward") == 0 || 
                 strcmp(op_name, "autodiff-reverse") == 0) {
            // Forward/reverse-mode autodiff
            return type_float_create(arena, FLOAT_SIZE_32);
        } 
        // Handle gradient functions - these return vectors
        else if (strstr(op_name, "gradient") || 
                 strstr(op_name, "grad") ||
                 strcmp(op_name, "autodiff-forward-gradient") == 0 || 
                 strcmp(op_name, "autodiff-reverse-gradient") == 0 ||
                 strcmp(op_name, "sum-of-squares-gradient-forward") == 0 ||
                 strcmp(op_name, "sum-of-squares-gradient-reverse") == 0 ||
                 strcmp(op_name, "sin-cos-gradient-forward") == 0 ||
                 strcmp(op_name, "sin-cos-gradient-reverse") == 0) {
            // Forward/reverse-mode gradient
            return type_vector_create(arena, type_float_create(arena, FLOAT_SIZE_32), 0);
        } 
        // Handle Jacobian and Hessian functions - these return matrices (vector of vectors)
        else if (strstr(op_name, "jacobian") ||
                 strstr(op_name, "hessian") ||
                 strcmp(op_name, "autodiff-jacobian") == 0 ||
                 strcmp(op_name, "vector-function-jacobian") == 0 ||
                 strcmp(op_name, "autodiff-hessian") == 0 ||
                 strcmp(op_name, "sum-of-squares-hessian") == 0) {
            // Jacobian/Hessian matrix
            // Create a matrix type (represented as a vector of vectors)
            Type* element_type = type_vector_create(arena, type_float_create(arena, FLOAT_SIZE_32), 0);
            return type_vector_create(arena, element_type, 0);
        } else if (strcmp(op_name, "derivative") == 0) {
            // Derivative of a function at a point
            return type_float_create(arena, FLOAT_SIZE_32);
        }
    }
    
    // Regular function call
    // Infer the type of the callee
    Type* callee_type = type_inference_infer_node(context, node->as.call.callee);

    // If the callee is a function, return its return type
    if (callee_type->kind == TYPE_FUNCTION) {
        if (callee_type->function.param_count == node->as.call.arg_count) {
          return callee_type->function.return_type;
        }
        
        //TODO: Partial function application could be implemented here
        return callee_type;
    }
    
    // Default to float
    return type_float_create(arena, FLOAT_SIZE_32);
}

/**
 * @brief Infer the type of an if expression
 */
static Type* infer_if(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_IF);
    Arena* arena = type_inference_get_arena(context);
    DiagnosticContext* diagnostics = type_inference_get_diagnostics(context);
    
    // Infer the type of the condition (we don't use the result, but we need to infer it)
    type_inference_infer_node(context, node->as.if_expr.condition);
    
    // Infer the type of the then branch
    Type* then_type = type_inference_infer_node(context, node->as.if_expr.then_branch);
    
    // Infer the type of the else branch
    Type* else_type = NULL;
    if (node->as.if_expr.else_branch) {
        else_type = type_inference_infer_node(context, node->as.if_expr.else_branch);
    } else {
        // If there's no else branch, the result is void
        else_type = type_void_create(arena);
    }
    
    // Check if we have mixed types (e.g., number and string)
    if (then_type && else_type && 
        then_type->kind != else_type->kind &&
        then_type->kind != TYPE_VOID && else_type->kind != TYPE_VOID) {
        
        // Special case for mixed numeric types (integer and float)
        if ((then_type->kind == TYPE_INTEGER && else_type->kind == TYPE_FLOAT) ||
            (then_type->kind == TYPE_FLOAT && else_type->kind == TYPE_INTEGER)) {
            // Promote to float
            return type_float_create(arena, FLOAT_SIZE_32);
        }
        
        // Special case for string and numeric types
        if ((then_type->kind == TYPE_STRING && (else_type->kind == TYPE_INTEGER || else_type->kind == TYPE_FLOAT)) ||
            (else_type->kind == TYPE_STRING && (then_type->kind == TYPE_INTEGER || then_type->kind == TYPE_FLOAT))) {
            
            // For scientific computing, we need to be careful with mixed types
            // In this case, we'll use void* as a generic return type
            diagnostic_debug(diagnostics, node->line, node->column, 
                           "Mixed return types in conditional expression (string and numeric)");
            return type_any_create(arena);
        }
    }
    
    // Return the common supertype of the then and else branches
    return type_common_supertype(arena, then_type, else_type);
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
    Arena* arena = type_inference_get_arena(context);
    
    // Infer the type of the body
    Type* body_type = type_inference_infer_node(context, node->as.lambda.body);
    
    // Create parameter types
    Type** param_types = NULL;
    if (node->as.lambda.param_count > 0) {
        param_types = arena_alloc(arena, node->as.lambda.param_count * sizeof(Type*));
        if (!param_types) return NULL;
        
        for (size_t i = 0; i < node->as.lambda.param_count; i++) {
            // Default to float for parameters
            param_types[i] = type_float_create(arena, FLOAT_SIZE_32);
        }
    }
    
    // Create function type
    return type_function_create(arena, node->as.lambda.param_count, param_types, body_type, false);
}

/**
 * @brief Infer the type of a define expression
 */
static Type* infer_define(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_DEFINE);
    
    // Infer the type of the value
    Type* value_type = type_inference_infer_node(context, node->as.define.value);
    
    // Add the definition to the context
    type_inference_add_node(context, node, value_type);
    
    return value_type;
}

/**
 * @brief Infer the type of a function definition
 */
static Type* infer_function_def(TypeInferenceContext* context, AstNode* node) {
    assert(node->type == AST_FUNCTION_DEF);
    Arena* arena = type_inference_get_arena(context);
    
    // Infer the type of the body
    Type* body_type = type_inference_infer_node(context, node->as.function_def.body);
    
    // Create parameter types
    Type** param_types = NULL;
    if (node->as.function_def.param_count > 0) {
        param_types = arena_alloc(arena, node->as.function_def.param_count * sizeof(Type*));
        if (!param_types) return NULL;
        
        for (size_t i = 0; i < node->as.function_def.param_count; i++) {
            // Use the parameter type if specified, otherwise default to float
            if (node->as.function_def.params[i]->type != NULL) {
                param_types[i] = node->as.function_def.params[i]->type;
            } else {
                param_types[i] = type_float_create(arena, FLOAT_SIZE_32);
            }
        }
    }
    
    // Special case for main function - always returns int
    if (node->as.function_def.name->type == AST_IDENTIFIER) {
        const char* name = node->as.function_def.name->as.identifier.name;
        if (strcmp(name, "main") == 0) {
            body_type = type_integer_create(arena, INT_SIZE_32);
        }
    }
    
    // Create function type
    Type* function_type = type_function_create(arena, node->as.function_def.param_count, param_types, body_type, false);
    
    // Add the function to the context
    type_inference_add_node(context, node, function_type);
    
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
    type_inference_add_node(context, node, value_type);
    
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
Type* type_inference_infer_node(TypeInferenceContext* context, AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Check if the node is already in the context
    Type* existing_type = type_inference_get_type(context, node);
    if (existing_type) {
        return existing_type;
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
            type = type_float_create(type_inference_get_arena(context), FLOAT_SIZE_32);
            break;
    }
    
    // Add the node to the context
    type_inference_add_node(context, node, type);
    
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
 * @brief Collect explicit types from the AST
 */
bool type_inference_collect_explicit_types(TypeInferenceContext* context, AstNode* ast) {
    if (!ast) return true;
    
    switch (ast->type) {
        case AST_TYPE_DECLARATION: {
            // Store the function signature
            StringId function_name = ast->as.type_declaration.function_name;
            Type* type = ast->as.type_declaration.type;
            type_inference_set_function_signature(context, function_name, type);
            break;
        }
        
        case AST_FUNCTION_DEF: {
            // Process function parameters with inline types
            for (size_t i = 0; i < ast->as.function_def.param_count; i++) {
                Parameter* param = ast->as.function_def.params[i];
                if (param->type) {
                    // Store parameter type
                    AstNode* param_node = ast->as.function_def.param_nodes[i];
                    type_inference_set_explicit_type(context, param_node, param->type);
                }
            }
            
            // Process function return type
            if (ast->as.function_def.return_type) {
                type_inference_set_explicit_type(context, ast, ast->as.function_def.return_type);
            }
            
            // Recursively process function body
            type_inference_collect_explicit_types(context, ast->as.function_def.body);
            break;
        }
        
        case AST_LET: {
            // Process let bindings with inline types
            for (size_t i = 0; i < ast->as.let.binding_count; i++) {
                AstNode* binding = ast->as.let.bindings[i];
                if (binding->type == AST_DEFINE && binding->as.define.value->type_info) {
                    // Store binding type
                    AstNode* binding_node = ast->as.let.binding_nodes[i];
                    type_inference_set_explicit_type(context, binding_node, binding->as.define.value->type_info);
                }
                
                // Recursively process binding value
                type_inference_collect_explicit_types(context, binding);
            }
            
            // Recursively process let body
            type_inference_collect_explicit_types(context, ast->as.let.body);
            break;
        }
        
        case AST_PROGRAM: {
            // Process top-level expressions
            for (size_t i = 0; i < ast->as.program.expr_count; i++) {
                type_inference_collect_explicit_types(context, ast->as.program.exprs[i]);
            }
            break;
        }
        
        default:
            // Recursively process child nodes based on node type
            switch (ast->type) {
                case AST_CALL:
                    // Process callee
                    type_inference_collect_explicit_types(context, ast->as.call.callee);
                    // Process arguments
                    for (size_t i = 0; i < ast->as.call.arg_count; i++) {
                        type_inference_collect_explicit_types(context, ast->as.call.args[i]);
                    }
                    break;
                
                case AST_IF:
                    // Process condition, then branch, and else branch
                    type_inference_collect_explicit_types(context, ast->as.if_expr.condition);
                    type_inference_collect_explicit_types(context, ast->as.if_expr.then_branch);
                    if (ast->as.if_expr.else_branch) {
                        type_inference_collect_explicit_types(context, ast->as.if_expr.else_branch);
                    }
                    break;
                
                case AST_BEGIN:
                    // Process expressions
                    for (size_t i = 0; i < ast->as.begin.expr_count; i++) {
                        type_inference_collect_explicit_types(context, ast->as.begin.exprs[i]);
                    }
                    break;
                
                case AST_LAMBDA:
                    // Process body
                    type_inference_collect_explicit_types(context, ast->as.lambda.body);
                    break;
                
                case AST_DEFINE:
                    // Process name and value
                    type_inference_collect_explicit_types(context, ast->as.define.name);
                    type_inference_collect_explicit_types(context, ast->as.define.value);
                    break;
                
                case AST_VARIABLE_DEF:
                    // Process name and value
                    type_inference_collect_explicit_types(context, ast->as.variable_def.name);
                    type_inference_collect_explicit_types(context, ast->as.variable_def.value);
                    break;
                
                case AST_SEQUENCE:
                    // Process expressions
                    for (size_t i = 0; i < ast->as.sequence.expr_count; i++) {
                        type_inference_collect_explicit_types(context, ast->as.sequence.exprs[i]);
                    }
                    break;
                
                default:
                    // No child nodes to process
                    break;
            }
            break;
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
 * @brief Resolve the final type for an AST node
 */
Type* type_inference_resolve_type(TypeInferenceContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // First check for explicit type
    Type* explicit_type = type_inference_get_explicit_type(context, node);
    if (explicit_type) {
        return explicit_type;
    }
    
    // If it's a function, check for separate declaration
    if (node->type == AST_FUNCTION_DEF && node->as.function_def.name->type == AST_IDENTIFIER) {
        StringId name = node->as.function_def.name->as.identifier.name;
        Type* signature = type_inference_get_function_signature(context, name);
        if (signature) {
            return signature;
        }
    }
    
    // Fall back to inferred type
    return type_inference_get_type(context, node);
}
