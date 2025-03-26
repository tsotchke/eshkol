/**
 * @file calls.c
 * @brief Function call code generation implementation
 */

#include "backend/codegen/calls.h"
#include "backend/codegen/expressions.h"
#include "backend/codegen/context.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Generate C code for a function call
 */
bool codegen_generate_call(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_CALL);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Check if it's an operator call
    if (node->as.call.callee->type == AST_IDENTIFIER) {
        const char* op_name = node->as.call.callee->as.identifier.name;
        
        // Handle arithmetic operators
        if (strcmp(op_name, "+") == 0 && node->as.call.arg_count == 2) {
            fprintf(output, "(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, " + ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "-") == 0) {
            if (node->as.call.arg_count == 1) {
                // Unary minus
                fprintf(output, "(-");
                if (!codegen_generate_expression(context, node->as.call.args[0])) {
                    return false;
                }
                fprintf(output, ")");
                return true;
            } else if (node->as.call.arg_count == 2) {
                // Binary minus
                fprintf(output, "(");
                if (!codegen_generate_expression(context, node->as.call.args[0])) {
                    return false;
                }
                fprintf(output, " - ");
                if (!codegen_generate_expression(context, node->as.call.args[1])) {
                    return false;
                }
                fprintf(output, ")");
                return true;
            }
        } else if (strcmp(op_name, "*") == 0 && node->as.call.arg_count == 2) {
            fprintf(output, "(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, " * ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "/") == 0 && node->as.call.arg_count == 2) {
            fprintf(output, "(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, " / ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        }
        
        // Handle comparison operators
        else if (strcmp(op_name, "<") == 0 && node->as.call.arg_count == 2) {
            fprintf(output, "(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, " < ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, ">") == 0 && node->as.call.arg_count == 2) {
            fprintf(output, "(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, " > ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "<=") == 0 && node->as.call.arg_count == 2) {
            fprintf(output, "(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, " <= ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, ">=") == 0 && node->as.call.arg_count == 2) {
            fprintf(output, "(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, " >= ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "=") == 0 && node->as.call.arg_count == 2) {
            fprintf(output, "(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, " == ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        }
        
        // Handle vector operations
        else if (strcmp(op_name, "vector") == 0) {
            // Vector literal
            fprintf(output, "vector_f_create_from_array(arena, (float[]){");
            
            // Generate vector elements
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                if (i > 0) {
                    fprintf(output, ", ");
                }
                
                if (!codegen_generate_expression(context, node->as.call.args[i])) {
                    return false;
                }
            }
            
            fprintf(output, "}, %zu)", node->as.call.arg_count);
            return true;
        } else if (strcmp(op_name, "v+") == 0 && node->as.call.arg_count == 2) {
            // Vector addition
            fprintf(output, "vector_f_add(arena, ");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, ", ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "v-") == 0 && node->as.call.arg_count == 2) {
            // Vector subtraction
            fprintf(output, "vector_f_sub(arena, ");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, ", ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "v*") == 0 && node->as.call.arg_count == 2) {
            // Vector scalar multiplication
            fprintf(output, "vector_f_mul_scalar(arena, ");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, ", ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "dot") == 0 && node->as.call.arg_count == 2) {
            // Dot product
            fprintf(output, "vector_f_dot(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, ", ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "cross") == 0 && node->as.call.arg_count == 2) {
            // Cross product
            fprintf(output, "vector_f_cross(arena, ");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, ", ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "norm") == 0 && node->as.call.arg_count == 1) {
            // Vector magnitude
            fprintf(output, "vector_f_magnitude(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "gradient") == 0 && node->as.call.arg_count == 2) {
            // Gradient of a scalar field
            fprintf(output, "compute_gradient(arena, ");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, ", ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "divergence") == 0 && node->as.call.arg_count == 2) {
            // Divergence of a vector field
            fprintf(output, "compute_divergence(arena, ");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, ", ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "curl") == 0 && node->as.call.arg_count == 2) {
            // Curl of a vector field
            fprintf(output, "compute_curl(arena, ");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, ", ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "laplacian") == 0 && node->as.call.arg_count == 2) {
            // Laplacian of a scalar field
            fprintf(output, "compute_laplacian(arena, ");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, ", ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ")");
            return true;
        } else if (strcmp(op_name, "display") == 0 && node->as.call.arg_count == 1) {
            // Display function (Scheme compatibility)
            // Check if the argument is a string literal
            AstNode* arg = node->as.call.args[0];
            if (arg->type == AST_LITERAL_STRING) {
                fprintf(output, "printf(\"%%s\", ");
                if (!codegen_generate_expression(context, arg)) {
                    return false;
                }
                fprintf(output, ")");
            } else {
                // For non-string values, use %d for integers and %f for floats
                fprintf(output, "printf(\"%%d\", (int)(");
                if (!codegen_generate_expression(context, arg)) {
                    return false;
                }
                fprintf(output, "))");
            }
            return true;
        } else if (strcmp(op_name, "newline") == 0 && node->as.call.arg_count == 0) {
            // Newline function (Scheme compatibility)
            fprintf(output, "printf(\"\\n\")");
            return true;
        } else if (strcmp(op_name, "string-append") == 0) {
            // String append function
            fprintf(output, "({ char buffer[1024] = \"\"; ");
            
            // Concatenate all arguments
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                fprintf(output, "strcat(buffer, ");
                if (!codegen_generate_expression(context, node->as.call.args[i])) {
                    return false;
                }
                fprintf(output, "); ");
            }
            
            fprintf(output, "strdup(buffer); })");
            return true;
        } else if (strcmp(op_name, "number->string") == 0 && node->as.call.arg_count == 1) {
            // Number to string conversion
            fprintf(output, "({ char buffer[64]; snprintf(buffer, sizeof(buffer), \"%%g\", ");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, "); strdup(buffer); })");
            return true;
        }
        
        // Handle autodiff functions
        else if (strcmp(op_name, "autodiff-forward") == 0 && node->as.call.arg_count == 2) {
            // Forward-mode autodiff
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(output, "({ float (*wrapper_func)(VectorF*) = (float (*)(VectorF*))");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, "; ");
            
            // Create a vector from the scalar input
            fprintf(output, "VectorF* vec_input = vector_f_create_from_array(arena, (float[]){");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, "}, 1); ");
            
            // Call the autodiff function
            fprintf(output, "vector_f_get(compute_gradient_autodiff(arena, wrapper_func, vec_input), 0); })");
            return true;
        } else if (strcmp(op_name, "autodiff-reverse") == 0 && node->as.call.arg_count == 2) {
            // Reverse-mode autodiff
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(output, "({ float (*wrapper_func)(VectorF*) = (float (*)(VectorF*))");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, "; ");
            
            // Create a vector from the scalar input
            fprintf(output, "VectorF* vec_input = vector_f_create_from_array(arena, (float[]){");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, "}, 1); ");
            
            // Call the autodiff function
            fprintf(output, "vector_f_get(compute_gradient_reverse_mode(arena, wrapper_func, vec_input), 0); })");
            return true;
        } else if (strcmp(op_name, "autodiff-forward-gradient") == 0 && node->as.call.arg_count == 2) {
            // Forward-mode gradient
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(output, "({ float (*wrapper_func)(VectorF*) = (float (*)(VectorF*))");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, "; ");
            
            // The input is already a vector
            fprintf(output, "compute_gradient_autodiff(arena, wrapper_func, ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, "); })");
            return true;
        } else if (strcmp(op_name, "autodiff-reverse-gradient") == 0 && node->as.call.arg_count == 2) {
            // Reverse-mode gradient
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(output, "({ float (*wrapper_func)(VectorF*) = (float (*)(VectorF*))");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, "; ");
            
            // The input is already a vector
            fprintf(output, "compute_gradient_reverse_mode(arena, wrapper_func, ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, "); })");
            return true;
        } else if (strcmp(op_name, "autodiff-jacobian") == 0 && node->as.call.arg_count == 2) {
            // Jacobian matrix
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(output, "({ VectorF* (*wrapper_func)(Arena*, VectorF*) = (VectorF* (*)(Arena*, VectorF*))");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, "; ");
            
            // Call the jacobian function
            fprintf(output, "compute_jacobian(arena, wrapper_func, ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, "); })");
            return true;
        } else if (strcmp(op_name, "autodiff-hessian") == 0 && node->as.call.arg_count == 2) {
            // Hessian matrix
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(output, "({ float (*wrapper_func)(VectorF*) = (float (*)(VectorF*))");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, "; ");
            
            // Call the hessian function
            fprintf(output, "compute_hessian(arena, wrapper_func, ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, "); })");
            return true;
        } else if (strcmp(op_name, "derivative") == 0 && node->as.call.arg_count == 2) {
            // Derivative of a function at a point
            // Create a wrapper function that adapts the user function to the expected signature
            fprintf(output, "({ float (*wrapper_func)(float) = (float (*)(float))");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, "; ");
            
            // Call the derivative function
            fprintf(output, "compute_nth_derivative(arena, wrapper_func, ");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, ", 1); })");
            return true;
        }
        
        // Handle vector and matrix operations
        else if (strcmp(op_name, "vector-ref") == 0 && node->as.call.arg_count == 2) {
            // Vector element access
            fprintf(output, "(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, "->data[");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, "])");
            return true;
        } else if (strcmp(op_name, "matrix-ref") == 0 && node->as.call.arg_count == 3) {
            // Matrix element access
            fprintf(output, "(");
            if (!codegen_generate_expression(context, node->as.call.args[0])) {
                return false;
            }
            fprintf(output, "[");
            if (!codegen_generate_expression(context, node->as.call.args[1])) {
                return false;
            }
            fprintf(output, "]->data[");
            if (!codegen_generate_expression(context, node->as.call.args[2])) {
                return false;
            }
            fprintf(output, "])");
            return true;
        } else if (strcmp(op_name, "printf") == 0) {
            // Printf function
            if (node->as.call.arg_count > 0) {
                // Get the format string
                const AstNode* format_node = node->as.call.args[0];
                
                // Check if it's a string literal
                if (format_node->type == AST_LITERAL_STRING) {
                    // Get the format string
                    const char* format = format_node->as.string.value;
                    
                    // Generate the printf call with the original format string
                    fprintf(output, "printf(\"%s\"", format);
                    
                    // Generate the rest of the arguments
                    for (size_t i = 1; i < node->as.call.arg_count; i++) {
                        fprintf(output, ", ");
                        
                        // Cast the argument to the appropriate type based on the format specifier
                        // Find the i-th format specifier in the format string
                        const char* p = format;
                        size_t count = 0;
                        while ((p = strchr(p, '%')) != NULL) {
                            if (p[1] != '%') {  // Skip %% (literal %)
                                count++;
                                if (count == i) {
                                    break;
                                }
                            }
                            p++;
                        }
                        
                        // If we found a format specifier, check its type
                        if (p != NULL) {
                            // Skip the %
                            p++;
                            
                            // Check the format specifier
                            if (*p == 'd' || *p == 'i') {
                                // Integer format specifier
                                fprintf(output, "(int)(");
                                if (!codegen_generate_expression(context, node->as.call.args[i])) {
                                    return false;
                                }
                                fprintf(output, ")");
                            } else {
                                // Other format specifier, no cast needed
                                if (!codegen_generate_expression(context, node->as.call.args[i])) {
                                    return false;
                                }
                            }
                        } else {
                            // No format specifier found, just generate the expression
                            if (!codegen_generate_expression(context, node->as.call.args[i])) {
                                return false;
                            }
                        }
                    }
                    
                    fprintf(output, ")");
                    return true;
                }
            }
            
            // Default printf handling
            fprintf(output, "printf(");
            
            // Generate arguments
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                if (i > 0) {
                    fprintf(output, ", ");
                }
                
                if (!codegen_generate_expression(context, node->as.call.args[i])) {
                    return false;
                }
            }
            
            fprintf(output, ")");
            return true;
        }
    }
    
    // Regular function call
    if (node->as.call.callee->type == AST_IDENTIFIER) {
        // Replace hyphens with underscores in function names
        char* function_name = strdup(node->as.call.callee->as.identifier.name);
        if (function_name) {
            for (char* p = function_name; *p; p++) {
                if (*p == '-') {
                    *p = '_';
                }
            }
            fprintf(output, "%s", function_name);
            free(function_name);
        } else {
            if (!codegen_generate_expression(context, node->as.call.callee)) {
                return false;
            }
        }
    } else {
        if (!codegen_generate_expression(context, node->as.call.callee)) {
            return false;
        }
    }
    
    // Generate arguments
    fprintf(output, "(");
    
    for (size_t i = 0; i < node->as.call.arg_count; i++) {
        if (i > 0) {
            fprintf(output, ", ");
        }
        
        if (!codegen_generate_expression(context, node->as.call.args[i])) {
            return false;
        }
    }
    
    fprintf(output, ")");
    
    return true;
}
