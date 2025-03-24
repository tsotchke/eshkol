/**
 * @file codegen.c
 * @brief Implementation of the code generation system
 */

#include "backend/codegen.h"
#include "core/memory.h"
#include "core/diagnostics.h"
#include "core/file_io.h"
#include "frontend/ast/ast.h"
#include "frontend/type_inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <unistd.h>

/**
 * @brief Code generator context structure
 */
struct CodegenContext {
    Arena* arena;                // Arena for allocations
    DiagnosticContext* diagnostics; // Diagnostic context for error reporting
    FILE* output;                // Output file
    int indent_level;            // Current indentation level
    bool in_function;            // Whether we're currently in a function
    char* temp_dir;              // Temporary directory for compilation
    TypeInferenceContext* type_inference; // Type inference context
};

/**
 * @brief Create a code generator context
 */
CodegenContext* codegen_context_create(Arena* arena, DiagnosticContext* diagnostics) {
    assert(arena != NULL);
    assert(diagnostics != NULL);
    
    // Allocate context
    CodegenContext* context = arena_alloc(arena, sizeof(CodegenContext));
    if (!context) return NULL;
    
    // Initialize context
    context->arena = arena;
    context->diagnostics = diagnostics;
    context->output = NULL;
    context->indent_level = 0;
    context->in_function = false;
    context->temp_dir = NULL;
    
    // Create type inference context
    context->type_inference = type_inference_context_create(arena);
    if (!context->type_inference) return NULL;
    
    return context;
}

/**
 * @brief Write indentation to the output
 */
static void write_indent(CodegenContext* context) {
    for (int i = 0; i < context->indent_level; i++) {
        fprintf(context->output, "    ");
    }
}

// Forward declaration
static bool generate_expression(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a number literal
 */
static bool generate_number_literal(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_LITERAL_NUMBER);
    
    fprintf(context->output, "%g", node->as.number.value);
    
    return true;
}

/**
 * @brief Generate C code for a boolean literal
 */
static bool generate_boolean_literal(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_LITERAL_BOOLEAN);
    
    fprintf(context->output, "%s", node->as.boolean.value ? "true" : "false");
    
    return true;
}

/**
 * @brief Generate C code for a character literal
 */
static bool generate_character_literal(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_LITERAL_CHARACTER);
    
    fprintf(context->output, "'%c'", node->as.character.value);
    
    return true;
}

/**
 * @brief Generate C code for a string literal
 */
static bool generate_string_literal(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_LITERAL_STRING);
    
    fprintf(context->output, "\"%s\"", node->as.string.value);
    
    return true;
}

/**
 * @brief Generate C code for a vector literal
 */
static bool generate_vector_literal(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_LITERAL_VECTOR);
    
    fprintf(context->output, "(VectorF*)vector_f_create_from_array(arena, (float[]){");
    
    // Generate vector elements
    for (size_t i = 0; i < node->as.vector.count; i++) {
        if (i > 0) {
            fprintf(context->output, ", ");
        }
        
        if (!generate_expression(context, node->as.vector.elements[i])) {
            return false;
        }
    }
    
    fprintf(context->output, "}, %zu)", node->as.vector.count);
    
    return true;
}

/**
 * @brief Generate C code for a nil literal
 */
static bool generate_nil_literal(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_LITERAL_NIL);
    
    fprintf(context->output, "NULL");
    
    return true;
}

/**
 * @brief Generate C code for an identifier
 */
static bool generate_identifier(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_IDENTIFIER);
    
    // Convert Scheme identifiers to valid C identifiers
    // Replace hyphens with underscores
    char* c_name = strdup(node->as.identifier.name);
    if (!c_name) {
        diagnostic_error(context->diagnostics, node->line, node->column, 
                        "Failed to allocate memory for identifier");
        return false;
    }
    
    for (char* p = c_name; *p; p++) {
        if (*p == '-') {
            *p = '_';
        }
    }
    
    fprintf(context->output, "%s", c_name);
    
    free(c_name);
    
    return true;
}

/**
 * @brief Generate C code for a function call
 */
static bool generate_call(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_CALL);
    
    // Check if it's an operator call
    if (node->as.call.callee->type == AST_IDENTIFIER) {
        const char* op_name = node->as.call.callee->as.identifier.name;
        
        // Handle arithmetic operators
        if (strcmp(op_name, "+") == 0 && node->as.call.arg_count == 2) {
            fprintf(context->output, "(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, " + ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "-") == 0) {
            if (node->as.call.arg_count == 1) {
                // Unary minus
                fprintf(context->output, "(-");
                if (!generate_expression(context, node->as.call.args[0])) {
                    return false;
                }
                fprintf(context->output, ")");
                return true;
            } else if (node->as.call.arg_count == 2) {
                // Binary minus
                fprintf(context->output, "(");
                if (!generate_expression(context, node->as.call.args[0])) {
                    return false;
                }
                fprintf(context->output, " - ");
                if (!generate_expression(context, node->as.call.args[1])) {
                    return false;
                }
                fprintf(context->output, ")");
                return true;
            }
        } else if (strcmp(op_name, "*") == 0 && node->as.call.arg_count == 2) {
            fprintf(context->output, "(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, " * ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "/") == 0 && node->as.call.arg_count == 2) {
            fprintf(context->output, "(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, " / ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        }
        
        // Handle comparison operators
        else if (strcmp(op_name, "<") == 0 && node->as.call.arg_count == 2) {
            fprintf(context->output, "(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, " < ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, ">") == 0 && node->as.call.arg_count == 2) {
            fprintf(context->output, "(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, " > ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "<=") == 0 && node->as.call.arg_count == 2) {
            fprintf(context->output, "(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, " <= ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, ">=") == 0 && node->as.call.arg_count == 2) {
            fprintf(context->output, "(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, " >= ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "=") == 0 && node->as.call.arg_count == 2) {
            fprintf(context->output, "(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, " == ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        }
        
        // Handle vector operations
        else if (strcmp(op_name, "vector") == 0) {
            // Vector literal
            fprintf(context->output, "vector_f_create_from_array(arena, (float[]){");
            
            // Generate vector elements
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                if (i > 0) {
                    fprintf(context->output, ", ");
                }
                
                if (!generate_expression(context, node->as.call.args[i])) {
                    return false;
                }
            }
            
            fprintf(context->output, "}, %zu)", node->as.call.arg_count);
            return true;
        } else if (strcmp(op_name, "v+") == 0 && node->as.call.arg_count == 2) {
            // Vector addition
            fprintf(context->output, "vector_f_add(arena, ");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, ", ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "v-") == 0 && node->as.call.arg_count == 2) {
            // Vector subtraction
            fprintf(context->output, "vector_f_sub(arena, ");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, ", ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "v*") == 0 && node->as.call.arg_count == 2) {
            // Vector scalar multiplication
            fprintf(context->output, "vector_f_mul_scalar(arena, ");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, ", ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "dot") == 0 && node->as.call.arg_count == 2) {
            // Dot product
            fprintf(context->output, "vector_f_dot(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, ", ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "cross") == 0 && node->as.call.arg_count == 2) {
            // Cross product
            fprintf(context->output, "vector_f_cross(arena, ");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, ", ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "norm") == 0 && node->as.call.arg_count == 1) {
            // Vector magnitude
            fprintf(context->output, "vector_f_magnitude(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "gradient") == 0 && node->as.call.arg_count == 2) {
            // Gradient of a scalar field
            fprintf(context->output, "compute_gradient(arena, ");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, ", ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "divergence") == 0 && node->as.call.arg_count == 2) {
            // Divergence of a vector field
            fprintf(context->output, "compute_divergence(arena, ");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, ", ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "curl") == 0 && node->as.call.arg_count == 2) {
            // Curl of a vector field
            fprintf(context->output, "compute_curl(arena, ");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, ", ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "laplacian") == 0 && node->as.call.arg_count == 2) {
            // Laplacian of a scalar field
            fprintf(context->output, "compute_laplacian(arena, ");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, ", ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "display") == 0 && node->as.call.arg_count == 1) {
            // Display function (Scheme compatibility)
            fprintf(context->output, "printf(\"%%s\\n\", ");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, ")");
            return true;
        } else if (strcmp(op_name, "string-append") == 0) {
            // String append function
            fprintf(context->output, "({ char buffer[1024] = \"\"; ");
            
            // Concatenate all arguments
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                fprintf(context->output, "strcat(buffer, ");
                if (!generate_expression(context, node->as.call.args[i])) {
                    return false;
                }
                fprintf(context->output, "); ");
            }
            
            fprintf(context->output, "strdup(buffer); })");
            return true;
        } else if (strcmp(op_name, "number->string") == 0 && node->as.call.arg_count == 1) {
            // Number to string conversion
            fprintf(context->output, "({ char buffer[64]; snprintf(buffer, sizeof(buffer), \"%%g\", ");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, "); strdup(buffer); })");
            return true;
        }
        
        // Handle autodiff functions
        else if (strcmp(op_name, "autodiff-forward") == 0 && node->as.call.arg_count == 2) {
            // Forward-mode autodiff
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(context->output, "({ float (*wrapper_func)(VectorF*) = (float (*)(VectorF*))");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, "; ");
            
            // Create a vector from the scalar input
            fprintf(context->output, "VectorF* vec_input = vector_f_create_from_array(arena, (float[]){");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, "}, 1); ");
            
            // Call the autodiff function
            fprintf(context->output, "vector_f_get(compute_gradient_autodiff(arena, wrapper_func, vec_input), 0); })");
            return true;
        } else if (strcmp(op_name, "autodiff-reverse") == 0 && node->as.call.arg_count == 2) {
            // Reverse-mode autodiff
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(context->output, "({ float (*wrapper_func)(VectorF*) = (float (*)(VectorF*))");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, "; ");
            
            // Create a vector from the scalar input
            fprintf(context->output, "VectorF* vec_input = vector_f_create_from_array(arena, (float[]){");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, "}, 1); ");
            
            // Call the autodiff function
            fprintf(context->output, "vector_f_get(compute_gradient_reverse_mode(arena, wrapper_func, vec_input), 0); })");
            return true;
        } else if (strcmp(op_name, "autodiff-forward-gradient") == 0 && node->as.call.arg_count == 2) {
            // Forward-mode gradient
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(context->output, "({ float (*wrapper_func)(VectorF*) = (float (*)(VectorF*))");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, "; ");
            
            // The input is already a vector
            fprintf(context->output, "compute_gradient_autodiff(arena, wrapper_func, ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, "); })");
            return true;
        } else if (strcmp(op_name, "autodiff-reverse-gradient") == 0 && node->as.call.arg_count == 2) {
            // Reverse-mode gradient
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(context->output, "({ float (*wrapper_func)(VectorF*) = (float (*)(VectorF*))");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, "; ");
            
            // The input is already a vector
            fprintf(context->output, "compute_gradient_reverse_mode(arena, wrapper_func, ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, "); })");
            return true;
        } else if (strcmp(op_name, "autodiff-jacobian") == 0 && node->as.call.arg_count == 2) {
            // Jacobian matrix
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(context->output, "({ VectorF* (*wrapper_func)(Arena*, VectorF*) = (VectorF* (*)(Arena*, VectorF*))");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, "; ");
            
            // Call the jacobian function
            fprintf(context->output, "compute_jacobian(arena, wrapper_func, ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, "); })");
            return true;
        } else if (strcmp(op_name, "autodiff-hessian") == 0 && node->as.call.arg_count == 2) {
            // Hessian matrix
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(context->output, "({ float (*wrapper_func)(VectorF*) = (float (*)(VectorF*))");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, "; ");
            
            // Call the hessian function
            fprintf(context->output, "compute_hessian(arena, wrapper_func, ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, "); })");
            return true;
        } else if (strcmp(op_name, "derivative") == 0 && node->as.call.arg_count == 2) {
            // Derivative of a function at a point
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(context->output, "({ float (*wrapper_func)(float) = (float (*)(float))");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, "; ");
            
            // Call the derivative function
            fprintf(context->output, "compute_nth_derivative(arena, wrapper_func, ");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, ", 1); })");
            return true;
        }
        
        // Handle vector and matrix operations
        else if (strcmp(op_name, "vector-ref") == 0 && node->as.call.arg_count == 2) {
            // Vector element access
            fprintf(context->output, "(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, "->data[");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, "])");
            return true;
        } else if (strcmp(op_name, "matrix-ref") == 0 && node->as.call.arg_count == 3) {
            // Matrix element access
            fprintf(context->output, "(");
            if (!generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(context->output, "[");
            if (!generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(context->output, "]->data[");
            if (!generate_expression(context, node->as.call.args[2])) {
                return false;
            }
            fprintf(context->output, "])");
            return true;
        } else if (strcmp(op_name, "printf") == 0) {
            // Printf function
            fprintf(context->output, "printf(");
            
            // Generate arguments
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                if (i > 0) {
                    fprintf(context->output, ", ");
                }
                
                if (!generate_expression(context, node->as.call.args[i])) {
                    return false;
                }
            }
            
            fprintf(context->output, ")");
            return true;
        }
    }
    
    // Regular function call
    if (!generate_expression(context, node->as.call.callee)) {
        return false;
    }
    
    // Generate arguments
    fprintf(context->output, "(");
    
    for (size_t i = 0; i < node->as.call.arg_count; i++) {
        if (i > 0) {
            fprintf(context->output, ", ");
        }
        
        if (!generate_expression(context, node->as.call.args[i])) {
            return false;
        }
    }
    
    fprintf(context->output, ")");
    
    return true;
}

/**
 * @brief Generate C code for an if expression
 */
static bool generate_if(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_IF);
    
    fprintf(context->output, "(");
    
    // Generate condition
    if (!generate_expression(context, node->as.if_expr.condition)) {
        return false;
    }
    
    fprintf(context->output, " ? ");
    
    // Generate then expression
    if (!generate_expression(context, node->as.if_expr.then_branch)) {
        return false;
    }
    
    fprintf(context->output, " : ");
    
    // Generate else expression
    if (!generate_expression(context, node->as.if_expr.else_branch)) {
        return false;
    }
    
    fprintf(context->output, ")");
    
    return true;
}

/**
 * @brief Generate C code for a begin expression
 */
static bool generate_begin(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_BEGIN);
    
    fprintf(context->output, "({ ");
    
    // Generate expressions
    for (size_t i = 0; i < node->as.begin.expr_count; i++) {
        if (i > 0) {
            fprintf(context->output, "; ");
        }
        
        if (!generate_expression(context, node->as.begin.exprs[i])) {
            return false;
        }
    }
    
    fprintf(context->output, "; })");
    
    return true;
}

/**
 * @brief Generate C code for a lambda expression
 */
static bool generate_lambda(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_LAMBDA);
    
    // TODO: Implement lambda expressions
    diagnostic_error(context->diagnostics, node->line, node->column, 
                    "Lambda expressions not yet implemented");
    return false;
}

/**
 * @brief Generate C code for a define expression
 */
static bool generate_define(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_DEFINE);
    
    // Generate name
    if (!generate_expression(context, node->as.define.name)) {
        return false;
    }
    
    fprintf(context->output, " = ");
    
    // Generate value
    if (!generate_expression(context, node->as.define.value)) {
        return false;
    }
    
    return true;
}


/**
 * @brief Convert a Type to a C type string
 */
static const char* type_to_c_type(Type* type) {
    // We should never have a NULL type, but just in case, default to int32_t
    // if (type == NULL) {
    //     return "int32_t";
    // }
    
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
                    return "double";
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
 * @brief Generate C code for a function definition
 */
static bool generate_function_def(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_FUNCTION_DEF);
    
    // Get return type
    const char* return_type = "int"; // Default to int
    if (node->as.function_def.return_type != NULL) {
        return_type = type_to_c_type(node->as.function_def.return_type);
    } else if (node->as.function_def.body != NULL) {
        // Infer return type using the type inference system
        type_inference_infer(context->type_inference, (AstNode*)node->as.function_def.body);
        return_type = type_inference_get_c_type(context->type_inference, node->as.function_def.body);
    }
    
    // Generate function declaration with return type
    fprintf(context->output, "%s ", return_type);
    
    // Generate function name
    if (!generate_expression(context, node->as.function_def.name)) {
        return false;
    }
    
    // Generate parameters with types
    fprintf(context->output, "(");
    
    for (size_t i = 0; i < node->as.function_def.param_count; i++) {
        if (i > 0) {
            fprintf(context->output, ", ");
        }
        
        // Get parameter type
        const char* param_type = "float"; // Default to float for autodiff functions
        Parameter* param = node->as.function_def.params[i];
        if (param->type != NULL) {
            param_type = type_to_c_type(param->type);
        }
        
        fprintf(context->output, "%s %s", param_type, param->name);
    }
    
    fprintf(context->output, ") {\n");
    
    // Generate function body
    context->indent_level++;
    context->in_function = true;
    
    write_indent(context);
    fprintf(context->output, "return ");
    if (!generate_expression(context, node->as.function_def.body)) {
        return false;
    }
    fprintf(context->output, ";\n");
    
    context->indent_level--;
    context->in_function = false;
    
    fprintf(context->output, "}\n\n");
    
    return true;
}

/**
 * @brief Generate C code for a variable definition
 */
static bool generate_variable_def(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_VARIABLE_DEF);
    
    // Get variable type
    const char* var_type = "int"; // Default to int
    if (node->type_info != NULL) {
        var_type = type_to_c_type(node->type_info);
    }
    
    // Generate variable declaration with type
    fprintf(context->output, "%s ", var_type);
    
    // Generate variable name
    if (!generate_expression(context, node->as.variable_def.name)) {
        return false;
    }
    
    fprintf(context->output, " = ");
    
    // Generate value
    if (!generate_expression(context, node->as.variable_def.value)) {
        return false;
    }
    
    fprintf(context->output, ";\n\n");
    
    return true;
}

/**
 * @brief Generate C code for a let expression
 */
static bool generate_let(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_LET);
    
    fprintf(context->output, "({ ");
    
    // Generate bindings
    for (size_t i = 0; i < node->as.let.binding_count; i++) {
        if (i > 0) {
            fprintf(context->output, "; ");
        }
        
        // Each binding is a define expression
        AstNode* binding = node->as.let.bindings[i];
        if (binding->type == AST_DEFINE) {
            // Use the existing type information from the AST node
            const char* var_type = type_to_c_type(binding->as.define.value->type_info);
            
            // Output the variable declaration
            fprintf(context->output, "%s ", var_type);
            if (!generate_identifier(context, binding->as.define.name)) {
                return false;
            }
            fprintf(context->output, " = ");
            
            // Generate value
            if (!generate_expression(context, binding->as.define.value)) {
                return false;
            }
        } else {
            // For non-define bindings, use the type inference system
            type_inference_infer(context->type_inference, binding);
            Type* type = type_inference_get_type(context->type_inference, binding);
            const char* var_type = type ? type_to_c_type(type) : "float";
            
            fprintf(context->output, "%s ", var_type);
            if (!generate_expression(context, binding)) {
                return false;
            }
        }
    }
    
    fprintf(context->output, "; ");
    
    // Generate body
    if (!generate_expression(context, node->as.let.body)) {
        return false;
    }
    
    fprintf(context->output, "; })");
    
    return true;
}

/**
 * @brief Generate C code for a sequence of expressions
 */
static bool generate_sequence(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_SEQUENCE);
    
    fprintf(context->output, "({ ");
    
    // Generate expressions
    for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
        if (i > 0) {
            fprintf(context->output, "; ");
        }
        
        // Special handling for let expressions within sequences
        if (node->as.sequence.exprs[i]->type == AST_LET) {
            // For let expressions, we'll handle them specially to avoid type conflicts
            AstNode* let_node = node->as.sequence.exprs[i];
            
            // Generate bindings
            for (size_t j = 0; j < let_node->as.let.binding_count; j++) {
                if (j > 0) {
                    fprintf(context->output, "; ");
                }
                
                // Each binding is a define expression
                AstNode* binding = let_node->as.let.bindings[j];
                if (binding->type == AST_DEFINE) {
                    // Use the existing type information from the AST node
                    const char* var_type = type_to_c_type(binding->as.define.value->type_info);
                    
                    // Output the variable declaration
                    fprintf(context->output, "%s ", var_type);
                    if (!generate_identifier(context, binding->as.define.name)) {
                        return false;
                    }
                    fprintf(context->output, " = ");
                    
                    // Generate value
                    if (!generate_expression(context, binding->as.define.value)) {
                        return false;
                    }
                } else {
                    // For non-define bindings, use the type inference system
                    type_inference_infer(context->type_inference, binding);
                    Type* type = type_inference_get_type(context->type_inference, binding);
                    const char* var_type = type ? type_to_c_type(type) : "float";
                    
                    fprintf(context->output, "%s ", var_type);
                    if (!generate_expression(context, binding)) {
                        return false;
                    }
                }
            }
            
            fprintf(context->output, "; ");
            
            // Generate body
            if (!generate_expression(context, let_node->as.let.body)) {
                return false;
            }
        } else {
            // For non-let expressions, generate normally
            if (!generate_expression(context, node->as.sequence.exprs[i])) {
                return false;
            }
        }
    }
    
    fprintf(context->output, "; })");
    
    return true;
}

/**
 * @brief Generate C code for an expression
 */
static bool generate_expression(CodegenContext* context, const AstNode* node) {
    switch (node->type) {
        case AST_LITERAL_NUMBER:
            return generate_number_literal(context, node);
        case AST_LITERAL_BOOLEAN:
            return generate_boolean_literal(context, node);
        case AST_LITERAL_CHARACTER:
            return generate_character_literal(context, node);
        case AST_LITERAL_STRING:
            return generate_string_literal(context, node);
        case AST_LITERAL_VECTOR:
            return generate_vector_literal(context, node);
        case AST_LITERAL_NIL:
            return generate_nil_literal(context, node);
        case AST_IDENTIFIER:
            return generate_identifier(context, node);
        case AST_CALL:
            return generate_call(context, node);
        case AST_IF:
            return generate_if(context, node);
        case AST_BEGIN:
            return generate_begin(context, node);
        case AST_LAMBDA:
            return generate_lambda(context, node);
        case AST_DEFINE:
            return generate_define(context, node);
        case AST_FUNCTION_DEF:
            return generate_function_def(context, node);
        case AST_VARIABLE_DEF:
            return generate_variable_def(context, node);
        case AST_LET:
            return generate_let(context, node);
        case AST_SEQUENCE:
            return generate_sequence(context, node);
        default:
            diagnostic_error(context->diagnostics, node->line, node->column, 
                            "Unknown AST node type");
            return false;
    }
}

/**
 * @brief Generate C code for a program
 */
static bool generate_program(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_PROGRAM);
    
    // Generate header
    fprintf(context->output, "#include <stdio.h>\n");
    fprintf(context->output, "#include <stdlib.h>\n");
    fprintf(context->output, "#include <stdbool.h>\n");
    fprintf(context->output, "#include <math.h>\n");
    fprintf(context->output, "#include \"core/vector.h\"\n");
    fprintf(context->output, "#include \"core/memory.h\"\n");
    fprintf(context->output, "#include \"core/autodiff.h\"\n\n");
    
    // Global arena variable
    fprintf(context->output, "// Global arena for memory allocations\n");
    fprintf(context->output, "Arena* arena = NULL;\n\n");
    
    // Define eshkol_value_t
    fprintf(context->output, "// Eshkol value type\n");
    fprintf(context->output, "typedef union {\n");
    fprintf(context->output, "    long integer;\n");
    fprintf(context->output, "    double floating;\n");
    fprintf(context->output, "    bool boolean;\n");
    fprintf(context->output, "    char character;\n");
    fprintf(context->output, "    char* string;\n");
    fprintf(context->output, "    void* pointer;\n");
    fprintf(context->output, "} eshkol_value_t;\n\n");
    
    // Forward declare functions
    fprintf(context->output, "// Forward declarations\n");
    for (size_t i = 0; i < node->as.program.expr_count; i++) {
        if (node->as.program.exprs[i]->type == AST_FUNCTION_DEF) {
            // Get return type
            const char* return_type = "int"; // Default to int
            if (node->as.program.exprs[i]->as.function_def.return_type != NULL) {
                return_type = type_to_c_type(node->as.program.exprs[i]->as.function_def.return_type);
            } else if (node->as.program.exprs[i]->as.function_def.body != NULL) {
                // Infer return type using the type inference system
                type_inference_infer(context->type_inference, (AstNode*)node->as.program.exprs[i]->as.function_def.body);
                return_type = type_inference_get_c_type(context->type_inference, node->as.program.exprs[i]->as.function_def.body);
            }
            
            fprintf(context->output, "%s ", return_type);
            generate_expression(context, node->as.program.exprs[i]->as.function_def.name);
            fprintf(context->output, "(");
            
            for (size_t j = 0; j < node->as.program.exprs[i]->as.function_def.param_count; j++) {
                if (j > 0) {
                    fprintf(context->output, ", ");
                }
                
                // Get parameter type
                const char* param_type = "float"; // Default to float for autodiff functions
                Parameter* param = node->as.program.exprs[i]->as.function_def.params[j];
                if (param->type != NULL) {
                    param_type = type_to_c_type(param->type);
                }
                
                fprintf(context->output, "%s", param_type);
            }
            
            fprintf(context->output, ");\n");
        }
    }
    fprintf(context->output, "\n");
    
    // Generate expressions
    for (size_t i = 0; i < node->as.program.expr_count; i++) {
        if (!generate_expression(context, node->as.program.exprs[i])) {
            return false;
        }
    }
    
    // Generate main function if there isn't one already
    bool has_main = false;
    for (size_t i = 0; i < node->as.program.expr_count; i++) {
        if (node->as.program.exprs[i]->type == AST_FUNCTION_DEF) {
            AstNode* name = node->as.program.exprs[i]->as.function_def.name;
            if (name->type == AST_IDENTIFIER && strcmp(name->as.identifier.name, "main") == 0) {
                has_main = true;
                break;
            }
        }
    }
    
    if (!has_main) {
        fprintf(context->output, "int main(int argc, char** argv) {\n");
        fprintf(context->output, "    // Initialize arena\n");
        fprintf(context->output, "    arena = arena_create(1024 * 1024);\n");
        fprintf(context->output, "    if (!arena) {\n");
        fprintf(context->output, "        fprintf(stderr, \"Failed to create memory arena\\n\");\n");
        fprintf(context->output, "        return 1;\n");
        fprintf(context->output, "    }\n\n");
        fprintf(context->output, "    printf(\"Hello from Eshkol!\\n\");\n\n");
        fprintf(context->output, "    // Clean up arena\n");
        fprintf(context->output, "    arena_destroy(arena);\n");
        fprintf(context->output, "    return 0;\n");
        fprintf(context->output, "}\n");
    }
    
    return true;
}

/**
 * @brief Generate C code from an AST
 */
bool codegen_generate(CodegenContext* context, const AstNode* ast, const char* output_file) {
    assert(context != NULL);
    assert(ast != NULL);
    
    // Open output file
    if (output_file != NULL) {
        context->output = fopen(output_file, "w");
        if (!context->output) {
            diagnostic_error(context->diagnostics, 0, 0, "Failed to open output file");
            return false;
        }
    } else {
        context->output = stdout;
    }
    
    // Generate code
    bool result = false;
    
    if (ast->type == AST_PROGRAM) {
        result = generate_program(context, ast);
    } else {
        result = generate_expression(context, ast);
    }
    
    // Close output file
    if (output_file != NULL) {
        fclose(context->output);
    }
    
    context->output = NULL;
    
    return result;
}

/**
 * @brief Compile and execute the generated code
 */
int codegen_compile_and_execute(CodegenContext* context, const char* c_file, char** args, int argc) {
    assert(context != NULL);
    assert(c_file != NULL);
    
    // Determine the include path based on the executable location
    char include_path[1024] = {0};
    
    // Get the path to the source file
    char source_path[1024] = {0};
    char *source_dir = NULL;
    
    // Get the absolute path of the source file
    if (c_file[0] == '/') {
        // Absolute path
        strncpy(source_path, c_file, sizeof(source_path) - 1);
    } else {
        // Relative path, prepend current directory
        char cwd[1024] = {0};
        if (getcwd(cwd, sizeof(cwd)) == NULL) {
            diagnostic_error(context->diagnostics, 0, 0, "Failed to get current working directory");
            return -1;
        }
        snprintf(source_path, sizeof(source_path), "%s/%s", cwd, c_file);
    }
    
    // Extract the directory from the source path
    source_dir = strdup(source_path);
    char *last_slash = strrchr(source_dir, '/');
    if (last_slash) {
        *last_slash = '\0';
    }
    
    // Construct the include path relative to the source directory
    snprintf(include_path, sizeof(include_path), "%s/../include", source_dir);
    
    // Create command to compile the C file
    char compile_cmd[1024];
    snprintf(compile_cmd, sizeof(compile_cmd), "gcc -I%s -o %s.out %s", include_path, c_file, c_file);
    
    // Free allocated memory
    free(source_dir);
    
    // Compile the C file
    int result = system(compile_cmd);
    if (result != 0) {
        diagnostic_error(context->diagnostics, 0, 0, "Failed to compile generated C code");
        return -1;
    }
    
    // Create command to execute the compiled program
    char execute_cmd[1024];
    snprintf(execute_cmd, sizeof(execute_cmd), "./%s.out", c_file);
    
    // Add arguments
    for (int i = 0; i < argc; i++) {
        strcat(execute_cmd, " ");
        strcat(execute_cmd, args[i]);
    }
    
    // Execute the compiled program
    result = system(execute_cmd);
    
    return result;
}
