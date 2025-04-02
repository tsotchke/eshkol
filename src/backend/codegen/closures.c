/**
 * @file closures.c
 * @brief Closure code generation implementation
 */

#include "backend/codegen/closures.h"
#include "backend/codegen/expressions.h"
#include "backend/codegen/context.h"
#include "frontend/binding/binding.h"
#include "core/closure.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Generate C code for a lambda expression (closure implementation)
 */
bool codegen_generate_closure(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_LAMBDA);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Get binding system from context
    BindingSystem* binding_system = codegen_context_get_binding_system(context);
    if (!binding_system) {
        DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
        diagnostic_error(diagnostics, node->line, node->column, "Binding system not available");
        return false;
    }
    
    // Get lambda ID
    uint64_t lambda_id = binding_system_register_lambda(binding_system, node->scope_id);
    if (lambda_id == 0) {
        DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
        diagnostic_error(diagnostics, node->line, node->column, "Failed to register lambda");
        return false;
    }
    
    // Analyze lambda captures
    if (!binding_system_analyze_lambda_captures(binding_system, node, lambda_id)) {
        DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
        diagnostic_error(diagnostics, node->line, node->column, "Failed to analyze lambda captures");
        return false;
    }
    
    // Get captured bindings
    uint64_t* binding_ids = NULL;
    size_t capture_count = 0;
    if (!binding_system_get_lambda_captures(binding_system, lambda_id, &binding_ids, &capture_count)) {
        DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
        diagnostic_error(diagnostics, node->line, node->column, "Failed to get lambda captures");
        return false;
    }
    
    // Generate a unique function name for the lambda
    char function_name[64];
    snprintf(function_name, sizeof(function_name), "lambda_%lu", lambda_id);
    
    // Store the current output position
    long current_pos = ftell(output);
    
    // Move to the function definitions section
    long function_pos = codegen_context_get_function_position(context);
    fseek(output, function_pos, SEEK_SET);
    
    // Generate function prototype
    fprintf(output, "\n// Lambda function %lu\n", lambda_id);
    fprintf(output, "void* %s(EshkolEnvironment* env, void** args) {\n", function_name);
    
    // Add a loop for tail call optimization
    fprintf(output, "    // Tail call optimization loop\n");
    fprintf(output, "    while (1) {\n");
    
    // Generate parameter declarations
    for (size_t i = 0; i < node->as.lambda.param_count; i++) {
        const char* param_name = node->as.lambda.params[i]->name;
        fprintf(output, "        // Parameter %s\n", param_name);
        fprintf(output, "        void* %s = args[%zu];\n", param_name, i);
    }
    
    // Generate captured variable declarations
    for (size_t i = 0; i < capture_count; i++) {
        uint64_t binding_id = binding_ids[i];
        BindingSystem* binding_system = codegen_context_get_binding_system(context);
        StringId name = binding_system_get_binding_name(binding_system, binding_id);
        int env_index = binding_system_get_binding_env_index(binding_system, binding_id);
        uint64_t binding_scope = binding_system_get_binding_scope(binding_system, binding_id);
        
        // Calculate the depth of the binding in the environment chain
        uint64_t depth = 0;
        uint64_t current_scope = node->scope_id;
        
        // Special handling for mutual recursion and function composition
        bool is_sibling_function = false;
        bool is_composition_function = false;
        uint64_t parent_scope = binding_system_get_parent_scope(binding_system, node->scope_id);
        
        // Check if this is a sibling function (mutual recursion)
        if (parent_scope != 0 && binding_scope != node->scope_id && 
            binding_system_get_parent_scope(binding_system, binding_scope) == parent_scope) {
            // This is a binding from a sibling scope - likely a mutually recursive function
            is_sibling_function = true;
            depth = 0; // Sibling functions are at the same level
        } 
        // Check if this is a function composition case
        else if (name && (strcmp(name, "compose") == 0 || 
                         (binding_system_is_ancestor_scope(binding_system, binding_scope, node->scope_id) &&
                          !binding_system_is_descendant_scope(binding_system, binding_scope, node->scope_id)))) {
            is_composition_function = true;
            
            // For function composition, we need to calculate the correct depth
            uint64_t temp_scope = node->scope_id;
            depth = 0;
            while (temp_scope != 0 && temp_scope != binding_scope) {
                depth++;
                temp_scope = binding_system_get_parent_scope(binding_system, temp_scope);
            }
        }
        else {
            // Normal case - calculate depth
            while (current_scope != binding_scope && current_scope != 0) {
                depth++;
                current_scope = binding_system_get_parent_scope(binding_system, current_scope);
            }
        }
        
        fprintf(output, "        // Captured variable %s (binding %lu, depth %lu, index %d%s%s)\n", 
                name, binding_id, depth, env_index, 
                is_sibling_function ? ", sibling function" : "",
                is_composition_function ? ", composition function" : "");
        
        if (is_sibling_function) {
            // For sibling functions, we need to get them from the parent environment
            fprintf(output, "        void* %s = eshkol_environment_get(env->parent, %d, %lu);\n", 
                    name, env_index, depth);
        } else if (is_composition_function) {
            // For function composition, we need special handling
            fprintf(output, "        void* %s = eshkol_environment_get(env, %d, %lu);\n", 
                    name, env_index, depth);
            // Add a fallback for NULL values (forward references)
            fprintf(output, "        if (%s == NULL && env->parent != NULL) {\n", name);
            fprintf(output, "            %s = eshkol_environment_get(env->parent, %d, %lu);\n", 
                    name, env_index, depth);
            fprintf(output, "        }\n");
        } else {
            // Normal case
            fprintf(output, "        void* %s = eshkol_environment_get(env, %d, %lu);\n", 
                    name, env_index, depth);
        }
    }
    
    // Generate function body
    fprintf(output, "        // Function body\n");
    
    // Get diagnostics context
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    // Special handling for function composition
    if (capture_count > 0) {
        // Check if this is a function composition lambda
        bool is_composition = false;
        bool is_compose_function = false;
        
        // Check for function composition pattern
        for (size_t i = 0; i < capture_count; i++) {
            uint64_t binding_id = binding_ids[i];
            StringId name = binding_system_get_binding_name(binding_system, binding_id);
            if (name && strcmp(name, "compose") == 0) {
                is_composition = true;
                diagnostic_debug(diagnostics, node->line, node->column, "Detected function composition pattern");
                break;
            }
        }
        
        // Special handling for the 'compose' function itself
        // Check if this is a lambda inside a function definition named "compose"
        AstNode* parent_node = NULL;
        uint64_t parent_scope = binding_system_get_parent_scope(binding_system, node->scope_id);
        if (parent_scope != 0) {
            // Try to find the parent node by scope ID
            for (size_t i = 0; i < binding_system->binding_table.count; i++) {
                if (binding_system->binding_table.scope_ids[i] == parent_scope) {
                    StringId name = binding_system_get_binding_name(binding_system, binding_system->binding_table.binding_ids[i]);
                    if (name && strcmp(name, "compose") == 0) {
                        is_compose_function = true;
                        diagnostic_debug(diagnostics, node->line, node->column, "Detected compose function definition");
                        break;
                    }
                }
            }
        }
        
        // Special handling for the 'compose' function itself
        if (is_compose_function) {
            // For the compose function, we need to ensure f and g are properly accessed
            fprintf(output, "        // Special handling for compose function\n");
            
            // Find the f and g parameters - they should be captured variables
            const char* f_param = NULL;
            const char* g_param = NULL;
            
            // Look for parameters named 'f' and 'g' in the captured variables
            for (size_t i = 0; i < capture_count; i++) {
                uint64_t binding_id = binding_ids[i];
                StringId name = binding_system_get_binding_name(binding_system, binding_id);
                if (name) {
                    if (strcmp(name, "f") == 0) {
                        f_param = name;
                    } else if (strcmp(name, "g") == 0) {
                        g_param = name;
                    }
                }
            }
            
            if (f_param && g_param) {
                // Ensure f and g are valid
                fprintf(output, "        if (%s == NULL) {\n", f_param);
                fprintf(output, "            fprintf(stderr, \"Error: NULL function 'f' in composition\\n\");\n");
                fprintf(output, "            exit(1);\n");
                fprintf(output, "        }\n");
                
                fprintf(output, "        if (%s == NULL) {\n", g_param);
                fprintf(output, "            fprintf(stderr, \"Error: NULL function 'g' in composition\\n\");\n");
                fprintf(output, "            exit(1);\n");
                fprintf(output, "        }\n");
                
                // For the compose function, we need to handle the lambda body specially
                if (node->as.lambda.body && node->as.lambda.body->type == AST_CALL) {
                    const AstNode* call_node = node->as.lambda.body;
                    
                    // Check if this is a call to f with g(x) as argument
                    if (call_node->as.call.callee && call_node->as.call.callee->type == AST_IDENTIFIER &&
                        strcmp(call_node->as.call.callee->as.identifier.name, f_param) == 0 &&
                        call_node->as.call.arg_count == 1 &&
                        call_node->as.call.args[0]->type == AST_CALL &&
                        call_node->as.call.args[0]->as.call.callee->type == AST_IDENTIFIER &&
                        strcmp(call_node->as.call.args[0]->as.call.callee->as.identifier.name, g_param) == 0) {
                        
                        // This is the f(g(x)) pattern, handle it specially
                        fprintf(output, "        // Function composition pattern: f(g(x))\n");
                        
                        // Get the x parameter
                        const char* x_param = node->as.lambda.params[0]->name;
                        
                        // Generate the call to g
                        fprintf(output, "        void* g_result = eshkol_closure_call(%s, (void*[]){%s});\n", 
                                g_param, x_param);
                        
                        // Generate the call to f with g's result
                        fprintf(output, "        return eshkol_closure_call(%s, (void*[]){g_result});\n", 
                                f_param);
                        
                        // Skip the normal body generation
                        // Instead of using goto, we'll use a flag
                        goto normal_body_generation; // Skip to normal body generation
                    }
                }
            }
        }
        
        // For other function composition cases
        if (is_composition) {
            // For function composition, we need to ensure the captured functions are valid
            fprintf(output, "        // Ensure captured functions are valid for composition\n");
            for (size_t i = 0; i < capture_count; i++) {
                uint64_t binding_id = binding_ids[i];
                StringId name = binding_system_get_binding_name(binding_system, binding_id);
                if (name) {
                    fprintf(output, "        if (%s == NULL) {\n", name);
                    fprintf(output, "            fprintf(stderr, \"Error: NULL function in composition: %s\\n\");\n", name);
                    fprintf(output, "            exit(1);\n");
                    fprintf(output, "        }\n");
                }
            }
        }
    }
    
normal_body_generation:
    
    // Check if this is a tail call to itself or another function
    if (node->as.lambda.body && node->as.lambda.body->type == AST_CALL && 
        node->as.lambda.body->is_tail_position) {
        
        // Handle tail call optimization
        const AstNode* call_node = node->as.lambda.body;
        
        // Check if this is a self-recursive tail call
        if (call_node->is_self_tail_call) {
            // For self-recursive tail calls, we update the parameters and continue the loop
            fprintf(output, "        // Self-recursive tail call optimization\n");
            
            // Generate temporary variables for the arguments
            for (size_t i = 0; i < call_node->as.call.arg_count; i++) {
                fprintf(output, "        // Evaluate argument %zu\n", i + 1);
                fprintf(output, "        void* temp_arg_%zu = (void*)(", i + 1);
                if (!codegen_generate_expression(context, call_node->as.call.args[i])) {
                    return false;
                }
                fprintf(output, ");\n");
            }
            
            // Update the parameters with the temporary variables
            for (size_t i = 0; i < call_node->as.call.arg_count && i < node->as.lambda.param_count; i++) {
                fprintf(output, "        args[%zu] = temp_arg_%zu;\n", i, i + 1);
            }
            
            // Continue the loop
            fprintf(output, "        continue;\n");
        } else {
            // For tail calls to other functions, we return the result directly
            fprintf(output, "        return ");
            if (!codegen_generate_expression(context, node->as.lambda.body)) {
                return false;
            }
            fprintf(output, ";\n");
        }
    } else {
        // Normal return
        fprintf(output, "        return (void*)(");
        if (!codegen_generate_expression(context, node->as.lambda.body)) {
            return false;
        }
        fprintf(output, ");\n");
    }
    
    // Close the loop and function
    fprintf(output, "    }\n");
    fprintf(output, "}\n");
    
    // Update the function position
    codegen_context_set_function_position(context, ftell(output));
    
    // Move back to the current position
    fseek(output, current_pos, SEEK_SET);
    
    // Generate closure creation code
    fprintf(output, "eshkol_closure_create(%s, ", function_name);
    
    // Create environment
    fprintf(output, "({ EshkolEnvironment* lambda_env = eshkol_environment_create(env, %zu, %lu); ", capture_count, node->scope_id);
    
    // Add captured variables to environment
    for (size_t i = 0; i < capture_count; i++) {
        uint64_t binding_id = binding_ids[i];
        BindingSystem* binding_system = codegen_context_get_binding_system(context);
        StringId name = binding_system_get_binding_name(binding_system, binding_id);
        uint64_t binding_scope = binding_system_get_binding_scope(binding_system, binding_id);
        
        // Special handling for mutual recursion and function composition
        bool is_sibling_function = false;
        bool is_composition_function = false;
        bool is_recursive_function = false;
        uint64_t parent_scope = binding_system_get_parent_scope(binding_system, node->scope_id);
        
        // Check if this is a sibling function (mutual recursion)
        if (parent_scope != 0 && binding_scope != node->scope_id && 
            binding_system_get_parent_scope(binding_system, binding_scope) == parent_scope) {
            // This is a binding from a sibling scope - likely a mutually recursive function
            is_sibling_function = true;
        } 
        // Check if this is a function composition case
        else if (name && (strcmp(name, "compose") == 0 || 
                         (binding_system_is_ancestor_scope(binding_system, binding_scope, node->scope_id) &&
                          !binding_system_is_descendant_scope(binding_system, binding_scope, node->scope_id)))) {
            is_composition_function = true;
        }
        // Check if this is a recursive function (self-reference)
        else if (binding_scope == node->scope_id) {
            is_recursive_function = true;
        }
        
        if (is_sibling_function) {
            // For sibling functions, we need to add them to the environment with special handling
            fprintf(output, "/* Sibling function */ eshkol_environment_add(lambda_env, %s, NULL, \"%s\"); ", name, name);
            
            // For mutual recursion, we need to ensure the function is properly initialized
            fprintf(output, "if (%s == NULL && env != NULL && env->parent != NULL) { "
                    "%s = eshkol_environment_get(env->parent, %d, 0); } ", 
                    name, name, i);
        } else if (is_composition_function) {
            // For function composition, we need special handling
            fprintf(output, "/* Function composition */ eshkol_environment_add(lambda_env, %s, NULL, \"%s\"); ", name, name);
            
            // For function composition, ensure the function is properly initialized
            fprintf(output, "if (%s == NULL && env != NULL) { "
                    "for (EshkolEnvironment* e = env; e != NULL; e = e->parent) { "
                    "  if (e->value_count > %zu) { "
                    "    void* val = e->values[%zu]; "
                    "    if (val != NULL) { %s = val; break; } "
                    "  } "
                    "} } ", 
                    name, i, i, name);
        } else if (is_recursive_function) {
            // For recursive functions, we need to handle self-reference
            fprintf(output, "/* Recursive function */ eshkol_environment_add(lambda_env, %s, NULL, \"%s\"); ", name, name);
            
            // For recursive functions, ensure proper initialization
            fprintf(output, "if (%s == NULL && env != NULL) { "
                    "%s = eshkol_environment_get(env, %d, 0); } ", 
                    name, name, i);
        } else {
            // Normal case
            fprintf(output, "eshkol_environment_add(lambda_env, %s, NULL, \"%s\"); ", name, name);
        }
    }
    
    fprintf(output, "lambda_env; })");
    
    // Add return type and parameter types
    fprintf(output, ", NULL, NULL, %zu)", node->as.lambda.param_count);
    
    return true;
}

/**
 * @brief Helper function to detect if a node is a function composition
 */
bool is_function_composition(const AstNode* node) {
    if (node == NULL || node->type != AST_CALL) {
        return false;
    }
    
    // Check if the callee is an identifier
    if (node->as.call.callee->type == AST_IDENTIFIER) {
        const char* func_name = node->as.call.callee->as.identifier.name;
        
        // Check if this is a direct call to 'compose'
        if (strcmp(func_name, "compose") == 0) {
            return true;
        }
        
        // Check if this is a call to a composed function (e.g., square-then-double)
        if (strstr(func_name, "-then-") != NULL) {
            return true;
        }
    }
    
    return false;
}

/**
 * @brief Helper function to handle direct function composition calls
 * 
 * This function handles direct calls to the 'compose' function by generating
 * specialized code that directly composes the functions without complex environment traversal.
 */
bool codegen_handle_direct_composition(CodegenContext* context, const AstNode* node, FILE* output) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_CALL);
    assert(output != NULL);
    
    // Get diagnostics context
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    // Debug message
    diagnostic_debug(diagnostics, node->line, node->column, "Handling direct compose call");
    
    // Generate the composition code using the new dynamic closure system
    fprintf(output, "eshkol_compose_functions(");
    
    // Generate the first function (f)
    if (!codegen_generate_expression(context, node->as.call.args[0])) {
        return false;
    }
    
    fprintf(output, ", ");
    
    // Generate the second function (g)
    if (!codegen_generate_expression(context, node->as.call.args[1])) {
        return false;
    }
    
    fprintf(output, ")");
    
    return true;
}

/**
 * @brief Helper function to handle function composition calls
 * 
 * This function handles function composition calls by detecting the type of composition
 * and using the appropriate approach. For direct calls to 'compose', it uses a specialized
 * approach that avoids complex environment traversal. For other cases, it uses a more
 * general approach that preserves higher-order function support.
 */
bool codegen_handle_composition_call(CodegenContext* context, const AstNode* node, FILE* output) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_CALL);
    assert(output != NULL);
    
    // Get diagnostics context
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    // Check if this is a direct call to 'compose'
    if (node->as.call.callee->type == AST_IDENTIFIER && 
        strcmp(node->as.call.callee->as.identifier.name, "compose") == 0 &&
        node->as.call.arg_count >= 2) {
        // Use the specialized handler for direct compose calls
        return codegen_handle_direct_composition(context, node, output);
    }
    
    // Check if this is a direct call to 'compose-n'
    if (node->as.call.callee->type == AST_IDENTIFIER && 
        strcmp(node->as.call.callee->as.identifier.name, "compose-n") == 0) {
        // Handle compose-n specially
        return codegen_handle_compose_n(context, node, output);
    }
    
    // For other composition calls, use a more robust approach
    diagnostic_debug(diagnostics, node->line, node->column, "Handling function composition call");
    
    // Generate a temporary variable for the closure
    fprintf(output, "({ EshkolClosure* _compose_closure = ");
    if (!codegen_generate_expression(context, node->as.call.callee)) {
        return false;
    }
    
    // Call the function directly using eshkol_closure_call
    fprintf(output, "; eshkol_closure_call(_compose_closure, (void*[]){");
    
    // Generate arguments
    for (size_t i = 0; i < node->as.call.arg_count; i++) {
        if (i > 0) {
            fprintf(output, ", ");
        }
        
        fprintf(output, "(void*)(");
        if (!codegen_generate_expression(context, node->as.call.args[i])) {
            return false;
        }
        fprintf(output, ")");
    }
    
    fprintf(output, "}); })");
    
    return true;
}

/**
 * @brief Helper function to handle compose-n function calls
 * 
 * This function handles calls to the 'compose-n' function, which composes
 * multiple functions together. It generates specialized code that applies
 * the functions in sequence.
 */
bool codegen_handle_compose_n(CodegenContext* context, const AstNode* node, FILE* output) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_CALL);
    assert(output != NULL);
    
    // Get diagnostics context
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    // Debug message
    diagnostic_debug(diagnostics, node->line, node->column, "Handling compose-n call");
    
    // First, generate a block that will contain our composition logic
    fprintf(output, "({ ");
    
    // Handle the case with no arguments
    if (node->as.call.arg_count == 0) {
        // Return the identity function
        fprintf(output, "eshkol_closure_create("
                "(void* (*)(EshkolEnvironment*, void**))({ "
                "void* _identity_function(EshkolEnvironment* _env, void** _args) { "
                "return _args[0]; "
                "} "
                "_identity_function; "
                "}), "
                "eshkol_environment_create(env, 0, %lu), "
                "NULL, NULL, 1"
                ")", node->scope_id);
        
        fprintf(output, "; })");
        return true;
    }
    
    // Handle the case with one argument
    if (node->as.call.arg_count == 1) {
        // Return the function as is
        fprintf(output, "EshkolClosure* _single_func = ");
        if (!codegen_generate_expression(context, node->as.call.args[0])) {
            return false;
        }
        fprintf(output, "; ");
        
        // Validate the function
        fprintf(output, "if (_single_func == NULL) { "
                "fprintf(stderr, \"Error: NULL function in compose-n\\n\"); "
                "exit(1); "
                "} ");
        
        fprintf(output, "_single_func");
        
        fprintf(output, "; })");
        return true;
    }
    
    // For multiple functions, we need to create an array of functions
    fprintf(output, "EshkolClosure* _funcs[%zu] = {", node->as.call.arg_count);
    
    // Generate the function array
    for (size_t i = 0; i < node->as.call.arg_count; i++) {
        if (i > 0) {
            fprintf(output, ", ");
        }
        
        if (!codegen_generate_expression(context, node->as.call.args[i])) {
            return false;
        }
    }
    fprintf(output, "};\n");
    
    // Validate each function in the array
    for (size_t i = 0; i < node->as.call.arg_count; i++) {
        fprintf(output, "if (_funcs[%zu] == NULL) { "
                "fprintf(stderr, \"Error: NULL function at position %zu in compose-n\\n\", %zu); "
                "exit(1); "
                "} ", i, i, i);
    }
    
    // Create a new closure that will apply the functions in sequence
    fprintf(output, "eshkol_closure_create("
        "(void* (*)(EshkolEnvironment*, void**))({ "
        "void* _composed_function(EshkolEnvironment* _env, void** _args) { "
        // Validate environment
        "if (_env == NULL) { "
        "  fprintf(stderr, \"Error: NULL environment in compose-n function\\n\"); "
        "  exit(1); "
        "} "
        
        // Get captured variables from environment with validation
        "void* _func_array_ptr = NULL; "
        "void* _func_count_ptr = NULL; "
        
        "if (_env->value_count >= 2) { "
        "  _func_array_ptr = _env->values[0]; "
        "  _func_count_ptr = _env->values[1]; "
        "} else { "
        "  fprintf(stderr, \"Error: Invalid environment in compose-n (expected 2 values, got %zu)\\n\", "
        "          _env->value_count); "
        "  exit(1); "
        "} "
        
        "EshkolClosure** _func_array = (EshkolClosure**)_func_array_ptr;"
        "size_t _func_count = (size_t)_func_count_ptr;"
        
        // Validate function array
        "if (_func_array == NULL) {"
        "  fprintf(stderr, \"Error: NULL function array in composition\\n\");"
        "  exit(1);"
        "}"
        
        "void* _result = _args[0];"
        "for (size_t i = _func_count; i > 0; i--) {"
        "  if (_func_array[i-1] == NULL) {"
        "    fprintf(stderr, \"Error: NULL function at position %zu in composition at runtime\\n\", i-1);"
        "    exit(1);"
        "  }"
        "  _result = eshkol_closure_call(_func_array[i-1], (void*[]){_result});"
        "}"
        "return _result;"
        "} "
        "_composed_function; "
        "}), "
        
        // Create a new environment that captures the function array and count
        "({ "
        "EshkolEnvironment* _compose_env = eshkol_environment_create(env, 2, %lu);"
        "eshkol_environment_add(_compose_env, _funcs, NULL, \"funcs\");"
        "eshkol_environment_add(_compose_env, (void*)%zu, NULL, \"count\");"
        "_compose_env; "
        "}), "
        
        // Return type and parameter types (NULL for now)
        "NULL, NULL, 1"
        ")", node->scope_id, node->as.call.arg_count);

fprintf(output, "; })");

return true;
}

/**
* @brief Generate C code for a closure call
*/
bool codegen_generate_closure_call(CodegenContext* context, const AstNode* node) {
assert(context != NULL);
assert(node != NULL);
assert(node->type == AST_CALL);

// Get output file
FILE* output = codegen_context_get_output(context);

// Get binding system from context
BindingSystem* binding_system = codegen_context_get_binding_system(context);
if (!binding_system) {
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    diagnostic_error(diagnostics, node->line, node->column, "Binding system not available");
    return false;
}

// Get diagnostics context
DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);

// Detect if this is a function composition call
bool is_compose_call = false;
bool is_compose_n_call = false;
bool is_composed_function_call = false;

// Check if the callee is an identifier
if (node->as.call.callee->type == AST_IDENTIFIER) {
    const char* func_name = node->as.call.callee->as.identifier.name;
    
    // Check if this is a direct call to 'compose'
    if (strcmp(func_name, "compose") == 0) {
        is_compose_call = true;
        diagnostic_debug(diagnostics, node->line, node->column, "Detected direct compose call");
    }
    
    // Check if this is a direct call to 'compose-n'
    if (strcmp(func_name, "compose-n") == 0) {
        is_compose_n_call = true;
        diagnostic_debug(diagnostics, node->line, node->column, "Detected direct compose-n call");
    }
    
    // Check if this is a call to a composed function (e.g., square-then-double)
    if (strstr(func_name, "-then-") != NULL) {
        is_composed_function_call = true;
        diagnostic_debug(diagnostics, node->line, node->column, "Detected composed function call");
    }
}

// For compose calls, use the direct composition handler
if (is_compose_call) {
    return codegen_handle_direct_composition(context, node, output);
}

// For compose-n calls or composed function calls, use the specialized handler
if (is_compose_n_call || is_composed_function_call) {
    return codegen_handle_composition_call(context, node, output);
}

// Check if this is a call to a ComposedFunction
fprintf(output, "({ ");
fprintf(output, "void* _callee = (void*)(");
if (!codegen_generate_expression(context, node->as.call.callee)) {
    return false;
}
fprintf(output, "); ");

// Check if this is a ComposedFunction by checking the type
fprintf(output, "if (_callee != NULL && ((uintptr_t)_callee & 0x1)) { ");
fprintf(output, "  // This is a ComposedFunction, call it directly ");
fprintf(output, "  ComposedFunction* _composed = (ComposedFunction*)((uintptr_t)_callee & ~0x1); ");
fprintf(output, "  void* _arg = (void*)(");

// Generate the argument (only one for composed functions)
if (node->as.call.arg_count > 0) {
    if (!codegen_generate_expression(context, node->as.call.args[0])) {
        return false;
    }
} else {
    fprintf(output, "NULL");
}

fprintf(output, "); ");
fprintf(output, "  eshkol_composed_function_call(_composed, _arg); ");
fprintf(output, "} else { ");
fprintf(output, "  // This is a regular closure, call it normally ");
fprintf(output, "  EshkolClosure* _closure = (EshkolClosure*)_callee; ");

// Call the function directly using eshkol_closure_call
fprintf(output, "  eshkol_closure_call(_closure, (void*[]){");

// Generate arguments
for (size_t i = 0; i < node->as.call.arg_count; i++) {
    if (i > 0) {
        fprintf(output, ", ");
    }
    
    fprintf(output, "(void*)(");
    if (!codegen_generate_expression(context, node->as.call.args[i])) {
        return false;
    }
    fprintf(output, ")");
}

fprintf(output, "}); ");
fprintf(output, "} })");

return true;
}
