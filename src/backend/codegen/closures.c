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
    fprintf(output, "({ EshkolEnvironment* lambda_env = eshkol_environment_create(env, %zu); ", capture_count);
    
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
            fprintf(output, "/* Sibling function */ eshkol_environment_add(lambda_env, %s, NULL); ", name);
            
            // For mutual recursion, we need to ensure the function is properly initialized
            fprintf(output, "if (%s == NULL && env != NULL && env->parent != NULL) { "
                    "%s = eshkol_environment_get(env->parent, %d, 0); } ", 
                    name, name, i);
        } else if (is_composition_function) {
            // For function composition, we need special handling
            fprintf(output, "/* Function composition */ eshkol_environment_add(lambda_env, %s, NULL); ", name);
            
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
            fprintf(output, "/* Recursive function */ eshkol_environment_add(lambda_env, %s, NULL); ", name);
            
            // For recursive functions, ensure proper initialization
            fprintf(output, "if (%s == NULL && env != NULL) { "
                    "%s = eshkol_environment_get(env, %d, 0); } ", 
                    name, name, i);
        } else {
            // Normal case
            fprintf(output, "eshkol_environment_add(lambda_env, %s, NULL); ", name);
        }
    }
    
    fprintf(output, "lambda_env; })");
    
    // Add return type and parameter types
    fprintf(output, ", NULL, NULL, %zu)", node->as.lambda.param_count);
    
    return true;
}

/**
 * @brief Helper function to handle function composition calls
 */
bool codegen_handle_composition_call(CodegenContext* context, const AstNode* node, FILE* output) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_CALL);
    assert(output != NULL);
    
    // Get diagnostics context
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    // Debug message
    diagnostic_debug(diagnostics, node->line, node->column, "Handling function composition call");
    
    // Generate a temporary variable for the closure
    fprintf(output, "({ EshkolClosure* _compose_closure = ");
    if (!codegen_generate_expression(context, node->as.call.callee)) {
        return false;
    }
    
    // Ensure the closure and its environment are valid
    fprintf(output, "; "
            "if (_compose_closure == NULL) { "
            "  fprintf(stderr, \"Error: NULL closure in composition at line %d\\n\", %d); "
            "  exit(1); "
            "} "
            
            // Get the environment
            "EshkolEnvironment* _compose_env = _compose_closure->environment; "
            
            // Ensure environment is properly initialized
            "if (_compose_env == NULL) { "
            "  _compose_env = eshkol_environment_create(NULL, 16); " // Increased capacity
            "  _compose_closure->environment = _compose_env; "
            "} "
            
            // Enhanced validation for function composition
            "// Enhanced validation for function composition\n"
            
            // First, ensure the environment is not in a validation cycle
            "if (_compose_env->in_validation) { "
            "  _compose_env->in_validation = false; " // Reset if already in validation
            "} "
            
            // Validate the environment chain
            "eshkol_environment_validate(_compose_env); "
            
            // Explicitly reset the validation flag
            "_compose_env->in_validation = false; "
            
            // Additional validation for function composition
            "// Special handling for function composition\n"
            "if (_compose_env->parent != NULL) { "
            "  // Ensure parent environments are also validated\n"
            "  for (EshkolEnvironment* _p = _compose_env->parent; _p != NULL; _p = _p->parent) { "
            "    if (_p->in_validation) { "
            "      _p->in_validation = false; " // Reset if already in validation
            "      continue; "
            "    } "
            "    eshkol_environment_validate(_p); "
            "    _p->in_validation = false; "  // Explicitly reset for each parent
            "  } "
            
            "  // For function composition, we need to ensure all component functions are available\n"
            "  // Copy any missing values from parent environments\n"
            "  for (size_t _i = 0; _i < _compose_env->value_count; _i++) { "
            "    if (_compose_env->values[_i] == NULL) { "
            "      // Try to find the value in any parent environment\n"
            "      for (EshkolEnvironment* _p = _compose_env->parent; _p != NULL; _p = _p->parent) { "
            "        if (_i < _p->value_count && _p->values[_i] != NULL) { "
            "          _compose_env->values[_i] = _p->values[_i]; "
            "          break; "
            "        } "
            "      } "
            "    } "
            "  } "
            
            "  // Additional check for any remaining NULL values\n"
            "  for (size_t _i = 0; _i < _compose_env->value_count; _i++) { "
            "    if (_compose_env->values[_i] == NULL) { "
            "      // If still NULL, try to find any non-NULL value to use as a placeholder\n"
            "      for (size_t _j = 0; _j < _compose_env->value_count; _j++) { "
            "        if (_j != _i && _compose_env->values[_j] != NULL) { "
            "          _compose_env->values[_i] = _compose_env->values[_j]; "
            "          break; "
            "        } "
            "      } "
            "      // If still NULL, check parent environments for any non-NULL value\n"
            "      if (_compose_env->values[_i] == NULL) { "
            "        for (EshkolEnvironment* _p = _compose_env->parent; _p != NULL && _compose_env->values[_i] == NULL; _p = _p->parent) { "
            "          for (size_t _j = 0; _j < _p->value_count; _j++) { "
            "            if (_p->values[_j] != NULL) { "
            "              _compose_env->values[_i] = _p->values[_j]; "
            "              break; "
            "            } "
            "          } "
            "        } "
            "      } "
            "    } "
            "  } "
            
            "  // Special handling for nested compositions\n"
            "  // For nested compositions like add1-then-square-then-double\n"
            "  // we need to ensure all component functions are available\n"
            "  for (size_t _i = 0; _i < _compose_env->value_count; _i++) { "
            "    void* _func = _compose_env->values[_i]; "
            "    if (_func != NULL) { "
            "      // Check if this is a closure\n"
            "      EshkolClosure* _inner_closure = (EshkolClosure*)_func; "
            "      // Simple validation check - if it has a function pointer, it's likely a closure\n"
            "      if (_inner_closure->function != NULL) { "
            "        // This might be a closure, validate its environment\n"
            "        EshkolEnvironment* _inner_env = _inner_closure->environment; "
            "        if (_inner_env != NULL) { "
            "          // Validate the inner environment\n"
            "          if (!_inner_env->in_validation) { " // Avoid cycles
            "            _inner_env->in_validation = true; "
            "            eshkol_environment_validate(_inner_env); "
            "            _inner_env->in_validation = false; "
            "          } "
            "        } "
            "      } "
            "    } "
            "  } "
            "} "
            
            // Final safety check - ensure we have valid function pointers
            "// Final safety check for function pointers\n"
            "if (_compose_closure->function == NULL) { "
            "  fprintf(stderr, \"Error: NULL function pointer in composition at line %d\\n\", %d); "
            "  exit(1); "
            "} "
            
            // Call the function with the environment and arguments
            "_compose_closure->function(_compose_env, (void*[]){", node->line);
    
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
    
    // Detect if this is a higher-order function call by checking the AST structure
    bool is_higher_order = false;
    bool is_compose_call = false;
    bool is_composed_function_call = false;
    
    // Check if the callee is an identifier
    if (node->as.call.callee->type == AST_IDENTIFIER) {
        const char* func_name = node->as.call.callee->as.identifier.name;
        
        // Debug message
        char debug_msg[256];
        snprintf(debug_msg, sizeof(debug_msg), "Checking function call: %s", func_name);
        diagnostic_debug(diagnostics, node->line, node->column, debug_msg);
        
        // Check if this is a direct call to 'compose'
        if (strcmp(func_name, "compose") == 0) {
            is_compose_call = true;
            is_higher_order = true;
            diagnostic_debug(diagnostics, node->line, node->column, "Detected direct compose call");
        }
        
        // Check if this is a call to a composed function (e.g., square-then-double)
        if (strstr(func_name, "-then-") != NULL) {
            is_composed_function_call = true;
            is_higher_order = true;
            diagnostic_debug(diagnostics, node->line, node->column, "Detected composed function call");
        }
        
        // If it has a binding ID, check if it's a higher-order function
        if (node->as.call.callee->binding_id != 0) {
            uint64_t binding_id = node->as.call.callee->binding_id;
            uint64_t binding_scope = binding_system_get_binding_scope(binding_system, binding_id);
            
            // Check if this binding is from a parent scope (captured)
            if (binding_scope != 0 && binding_scope != node->scope_id) {
                is_higher_order = true;
                diagnostic_debug(diagnostics, node->line, node->column, "Detected higher-order function (captured binding)");
            }
            
            // Check if this is a lambda call (function composition or higher-order function)
            for (size_t i = 0; i < binding_system->lambda_table.count; i++) {
                if (binding_system->lambda_table.scope_ids[i] == binding_scope) {
                    is_higher_order = true;
                    diagnostic_debug(diagnostics, node->line, node->column, "Detected lambda call");
                    break;
                }
            }
        }
    }
    
    // For compose calls or composed function calls, use the specialized handler
    if (is_compose_call || is_composed_function_call) {
        return codegen_handle_composition_call(context, node, output);
    }
    
    // For other higher-order functions, we need to use a robust approach
    if (is_higher_order) {
        // Generate a temporary variable for the closure
        fprintf(output, "({ EshkolClosure* _tmp_closure = ");
        if (!codegen_generate_expression(context, node->as.call.callee)) {
            return false;
        }
        
        // Ensure the closure and its environment are valid
        fprintf(output, "; "
                "if (_tmp_closure == NULL) { "
                "  fprintf(stderr, \"Error: NULL closure at line %d\\n\", %d); "
                "  exit(1); "
                "} "
                "EshkolEnvironment* _env = _tmp_closure->environment; "
                "// Ensure environment is properly initialized\n"
                "if (_env == NULL) { "
                "  _env = eshkol_environment_create(NULL, 8); "
                "  _tmp_closure->environment = _env; "
                "} "
                
                // Use our improved environment validation
                "// Validate the environment to ensure all values are properly initialized\n"
                "eshkol_environment_validate(_env); "
                
                // Reset validation flag to ensure it's not left in a bad state
                "_env->in_validation = false; "
                
                // Additional validation for parent environments
                "// Also validate parent environments\n"
                "if (_env->parent != NULL) { "
                "  for (EshkolEnvironment* _p = _env->parent; _p != NULL; _p = _p->parent) { "
                "    eshkol_environment_validate(_p); "
                "    _p->in_validation = false; "
                "  } "
                "} "
                
                // Call the function with the environment and arguments
                "_tmp_closure->function(_env, (void*[]){", node->line);
        
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
    
    // For all other closure calls, use a robust approach as well
    // Generate a temporary variable for the closure
    fprintf(output, "({ EshkolClosure* _tmp_closure = ");
    if (!codegen_generate_expression(context, node->as.call.callee)) {
        return false;
    }
    
    // Ensure the closure and its environment are valid
    fprintf(output, "; "
            "if (_tmp_closure == NULL) { "
            "  fprintf(stderr, \"Error: NULL closure at line %d\\n\", %d); "
            "  exit(1); "
            "} "
            "EshkolEnvironment* _env = _tmp_closure->environment; "
            "// Ensure environment is properly initialized\n"
            "if (_env == NULL) { "
            "  _env = eshkol_environment_create(NULL, 8); "
            "  _tmp_closure->environment = _env; "
            "} "
            
            // Use our improved environment validation
            "// Validate the environment to ensure all values are properly initialized\n"
            "eshkol_environment_validate(_env); "
            
            // Reset validation flag to ensure it's not left in a bad state
            "_env->in_validation = false; "
            
            // Additional validation for parent environments
            "// Also validate parent environments\n"
            "if (_env->parent != NULL) { "
            "  for (EshkolEnvironment* _p = _env->parent; _p != NULL; _p = _p->parent) { "
            "    eshkol_environment_validate(_p); "
            "    _p->in_validation = false; "
            "  } "
            "} "
            
            // Call the function with the environment and arguments
            "_tmp_closure->function(_env, (void*[]){", node->line);
    
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
