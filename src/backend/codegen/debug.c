/**
 * @file debug.c
 * @brief Debug utilities for code generation
 */

#include "backend/codegen/debug.h"
#include "backend/codegen/context.h"
#include "backend/codegen/compiler.h"
#include "frontend/ast/ast.h"
#include "frontend/binding/binding.h"
#include "core/memory_tracking.h"
#include "core/closure.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h> /* Added for malloc and free */
#include <stdint.h>

// Forward declarations for IR generation functions
static void generate_ir_for_node(FILE* output, const AstNode* node, int indent);
static void generate_ir_for_lambda(FILE* output, const AstNode* node, int indent);
static void generate_ir_for_call(FILE* output, const AstNode* node, int indent);
static void generate_ir_for_function_def(FILE* output, const AstNode* node, int indent);
static void generate_ir_for_variable_def(FILE* output, const AstNode* node, int indent);
static void generate_ir_for_type_declaration(FILE* output, const AstNode* node, int indent);

// Forward declarations for C code generation functions
static void generate_c_code_for_node(FILE* output, const AstNode* node, int indent);
static void generate_c_code_for_lambda(FILE* output, const AstNode* node, int indent);
static void generate_c_code_for_call(FILE* output, const AstNode* node, int indent);
static void generate_c_code_for_function_def(FILE* output, const AstNode* node, int indent);
static void generate_c_code_for_variable_def(FILE* output, const AstNode* node, int indent);
static void generate_c_code_for_type_declaration(FILE* output, const AstNode* node, int indent);

// Forward declaration for binding system function
uint64_t binding_system_get_lambda_scope(BindingSystem* system, uint64_t lambda_id);

// Helper function to print indentation
static void print_indent(FILE* output, int indent) {
    for (int i = 0; i < indent; i++) {
        fprintf(output, "  ");
    }
}

bool codegen_debug(CodegenContext* context, AstNode* ast, const char* stage) {
    assert(context != NULL);
    assert(ast != NULL);
    assert(stage != NULL);
    
    printf("Code Generation Debug Output\n");
    printf("===========================\n\n");

    if (strcmp(stage, "ast") == 0 || strcmp(stage, "all") == 0) {
        printf("AST Structure:\n");
        printf("-------------\n");
        ast_print(ast, 0);
        printf("\n");
    }

    if (strcmp(stage, "ir") == 0 || strcmp(stage, "all") == 0) {
        printf("Intermediate Representation:\n");
        printf("--------------------------\n");
        
        // Generate IR for the AST
        generate_ir_for_node(stdout, ast, 0);
        printf("\n");
    }

    if (strcmp(stage, "c-code") == 0 || strcmp(stage, "all") == 0) {
        printf("Generated C Code:\n");
        printf("----------------\n");
        
        // Generate C code for the AST
        generate_c_code_for_node(stdout, ast, 0);
        printf("\n");
    }

    return true;
}

/**
 * @brief Structure to hold profiling data for a phase
 */
typedef struct {
    const char* name;
    clock_t start_time;
    clock_t end_time;
    MemoryStats start_memory;
    MemoryStats end_memory;
} ProfilingPhase;

/**
 * @brief Start profiling a phase
 * 
 * @param phase The phase to profile
 * @param name The name of the phase
 */
static void start_profiling_phase(ProfilingPhase* phase, const char* name) {
    assert(phase != NULL);
    assert(name != NULL);
    
    phase->name = name;
    phase->start_time = clock();
    phase->start_memory = memory_tracking_get_stats();
}

/**
 * @brief End profiling a phase
 * 
 * @param phase The phase to profile
 */
static void end_profiling_phase(ProfilingPhase* phase) {
    assert(phase != NULL);
    
    phase->end_time = clock();
    phase->end_memory = memory_tracking_get_stats();
}

/**
 * @brief Print profiling results for a phase
 * 
 * @param phase The phase to print results for
 */
static void print_profiling_results(const ProfilingPhase* phase) {
    assert(phase != NULL);
    
    double time_seconds = (double)(phase->end_time - phase->start_time) / CLOCKS_PER_SEC;
    size_t memory_used = phase->end_memory.active_bytes_allocated - phase->start_memory.active_bytes_allocated;
    size_t allocations = phase->end_memory.active_allocations - phase->start_memory.active_allocations;
    
    printf("Phase: %s\n", phase->name);
    printf("Time: %.6f seconds\n", time_seconds);
    printf("Memory: %zu bytes (%zu allocations)\n", memory_used, allocations);
    printf("Peak Memory: %zu bytes\n\n", phase->end_memory.peak_bytes_allocated);
}

/**
 * @brief Analyze AST for closures and tail recursion
 * 
 * @param ast The AST to analyze
 * @param binding_system The binding system
 * @return Analysis results as a string
 */
static char* analyze_ast_for_closures_and_tail_recursion(const AstNode* ast, BindingSystem* binding_system) {
    assert(ast != NULL);
    assert(binding_system != NULL);
    
    // Allocate a buffer for the analysis results
    char* buffer = malloc(4096);
    if (!buffer) {
        return NULL;
    }
    buffer[0] = '\0';
    
    // Count lambdas and tail recursive calls
    int lambda_count = 0;
    int tail_recursive_count = 0;
    int closure_count = 0;
    int mutual_recursion_count = 0;
    
    // Analyze the AST
    if (ast->type == AST_PROGRAM) {
        // Analyze each expression in the program
        for (size_t i = 0; i < ast->as.program.expr_count; i++) {
            const AstNode* expr = ast->as.program.exprs[i];
            
            // Count lambdas
            if (expr->type == AST_LAMBDA) {
                lambda_count++;
                
                // Check if this lambda captures any variables
                uint64_t lambda_id = binding_system_register_lambda(binding_system, expr->scope_id);
                if (lambda_id != 0) {
                    uint64_t* binding_ids = NULL;
                    size_t capture_count = 0;
                    if (binding_system_get_lambda_captures(binding_system, lambda_id, &binding_ids, &capture_count)) {
                        if (capture_count > 0) {
                            closure_count++;
                        }
                    }
                }
            }
            
            // Check for tail recursive calls
            if (expr->type == AST_DEFINE && expr->as.define.value->type == AST_LAMBDA) {
                const AstNode* lambda = expr->as.define.value;
                const AstNode* body = lambda->as.lambda.body;
                
                // Check if the body is a call
                if (body->type == AST_CALL) {
                    const AstNode* callee = body->as.call.callee;
                    
                    // Check if the callee is the same as the defined function
                    if (callee->type == AST_IDENTIFIER && 
                        expr->as.define.name->type == AST_IDENTIFIER &&
                        strcmp(callee->as.identifier.name, expr->as.define.name->as.identifier.name) == 0) {
                        tail_recursive_count++;
                    }
                }
            }
            
            // Check for mutual recursion
            if (expr->type == AST_DEFINE && expr->as.define.name->type == AST_IDENTIFIER) {
                const char* func_name = expr->as.define.name->as.identifier.name;
                
                // Check if this function calls another function that calls this function
                for (size_t j = 0; j < ast->as.program.expr_count; j++) {
                    if (i != j && ast->as.program.exprs[j]->type == AST_DEFINE && 
                        ast->as.program.exprs[j]->as.define.name->type == AST_IDENTIFIER) {
                        const char* other_func_name = ast->as.program.exprs[j]->as.define.name->as.identifier.name;
                        
                        // Check if this function calls the other function
                        bool calls_other = false;
                        if (expr->as.define.value->type == AST_LAMBDA) {
                            const AstNode* body = expr->as.define.value->as.lambda.body;
                            if (body->type == AST_CALL && body->as.call.callee->type == AST_IDENTIFIER &&
                                strcmp(body->as.call.callee->as.identifier.name, other_func_name) == 0) {
                                calls_other = true;
                            }
                        }
                        
                        // Check if the other function calls this function
                        bool other_calls_this = false;
                        if (ast->as.program.exprs[j]->as.define.value->type == AST_LAMBDA) {
                            const AstNode* other_body = ast->as.program.exprs[j]->as.define.value->as.lambda.body;
                            if (other_body->type == AST_CALL && other_body->as.call.callee->type == AST_IDENTIFIER &&
                                strcmp(other_body->as.call.callee->as.identifier.name, func_name) == 0) {
                                other_calls_this = true;
                            }
                        }
                        
                        if (calls_other && other_calls_this) {
                            mutual_recursion_count++;
                        }
                    }
                }
            }
        }
    }
    
    // Format the analysis results
    snprintf(buffer, 4096, 
             "AST Analysis Results:\n"
             "--------------------\n"
             "Lambda expressions: %d\n"
             "Closures (lambdas with captures): %d\n"
             "Tail recursive functions: %d\n"
             "Mutual recursive functions: %d\n",
             lambda_count, closure_count, tail_recursive_count, mutual_recursion_count);
    
    return buffer;
}

/**
 * @brief Get the scope ID for a lambda
 * 
 * @param binding_system The binding system
 * @param lambda_id The lambda ID
 * @return The scope ID
 */
uint64_t binding_system_get_lambda_scope(BindingSystem* system, uint64_t lambda_id) {
    // This is a temporary implementation
    // In a real implementation, this would look up the scope ID from the binding system
    return binding_system_enter_lambda_scope(system);
}

/**
 * @brief Analyze the environment chain for a closure
 * 
 * @param binding_system The binding system
 * @param lambda_id The lambda ID
 * @return Analysis results as a string
 */
static char* analyze_environment_chain(BindingSystem* binding_system, uint64_t lambda_id) {
    assert(binding_system != NULL);
    assert(lambda_id != 0);
    
    // Allocate a buffer for the analysis results
    char* buffer = malloc(4096);
    if (!buffer) {
        return NULL;
    }
    buffer[0] = '\0';
    
    // Get the lambda's scope ID
    uint64_t scope_id = binding_system_get_lambda_scope(binding_system, lambda_id);
    if (scope_id == 0) {
        snprintf(buffer, 4096, "Error: Lambda %llu not found in binding system\n", (unsigned long long)lambda_id);
        return buffer;
    }
    
    // Get the captured bindings
    uint64_t* binding_ids = NULL;
    size_t capture_count = 0;
    if (!binding_system_get_lambda_captures(binding_system, lambda_id, &binding_ids, &capture_count)) {
        snprintf(buffer, 4096, "Error: Failed to get captures for lambda %llu\n", (unsigned long long)lambda_id);
        return buffer;
    }
    
    // Format the environment chain
    int offset = 0;
    offset += snprintf(buffer + offset, 4096 - offset, 
                      "Environment Chain for Lambda %llu (Scope %llu):\n"
                      "---------------------------------------------\n"
                      "Captured bindings: %zu\n\n",
                      (unsigned long long)lambda_id, (unsigned long long)scope_id, capture_count);
    
    // Print each captured binding
    for (size_t i = 0; i < capture_count; i++) {
        uint64_t binding_id = binding_ids[i];
        StringId name = binding_system_get_binding_name(binding_system, binding_id);
        uint64_t binding_scope = binding_system_get_binding_scope(binding_system, binding_id);
        int env_index = binding_system_get_binding_env_index(binding_system, binding_id);
        
        // Calculate the depth of the binding in the environment chain
        uint64_t depth = 0;
        uint64_t current_scope = scope_id;
        while (current_scope != binding_scope && current_scope != 0) {
            depth++;
            current_scope = binding_system_get_parent_scope(binding_system, current_scope);
        }
        
        offset += snprintf(buffer + offset, 4096 - offset, 
                          "  Binding %llu: %s (Scope %llu, Depth %llu, Index %d)\n",
                          (unsigned long long)binding_id, name, 
                          (unsigned long long)binding_scope, 
                          (unsigned long long)depth, env_index);
    }
    
    return buffer;
}

/**
 * @brief Analyze tail call optimization opportunities
 * 
 * @param ast The AST to analyze
 * @return Analysis results as a string
 */
static char* analyze_tail_call_optimization(const AstNode* ast) {
    assert(ast != NULL);
    
    // Allocate a buffer for the analysis results
    char* buffer = malloc(4096);
    if (!buffer) {
        return NULL;
    }
    buffer[0] = '\0';
    
    // Count tail call opportunities
    int tail_call_count = 0;
    int optimizable_count = 0;
    
    // Analyze the AST
    if (ast->type == AST_PROGRAM) {
        // Analyze each expression in the program
        for (size_t i = 0; i < ast->as.program.expr_count; i++) {
            const AstNode* expr = ast->as.program.exprs[i];
            
            // Check for tail calls in function definitions
            if (expr->type == AST_DEFINE && expr->as.define.value->type == AST_LAMBDA) {
                const AstNode* lambda = expr->as.define.value;
                const AstNode* body = lambda->as.lambda.body;
                
                // Check if the body is a call
                if (body->type == AST_CALL) {
                    tail_call_count++;
                    
                    // Check if the call is to the same function (direct recursion)
                    if (body->as.call.callee->type == AST_IDENTIFIER && 
                        expr->as.define.name->type == AST_IDENTIFIER &&
                        strcmp(body->as.call.callee->as.identifier.name, expr->as.define.name->as.identifier.name) == 0) {
                        optimizable_count++;
                    }
                    // Check for mutual recursion
                    else if (body->as.call.callee->type == AST_IDENTIFIER) {
                        const char* callee_name = body->as.call.callee->as.identifier.name;
                        
                        // Check if there's another function that calls this function
                        for (size_t j = 0; j < ast->as.program.expr_count; j++) {
                            if (i != j && ast->as.program.exprs[j]->type == AST_DEFINE && 
                                ast->as.program.exprs[j]->as.define.name->type == AST_IDENTIFIER &&
                                strcmp(ast->as.program.exprs[j]->as.define.name->as.identifier.name, callee_name) == 0) {
                                
                                // Check if that function calls this function in tail position
                                const AstNode* other_lambda = ast->as.program.exprs[j]->as.define.value;
                                if (other_lambda->type == AST_LAMBDA) {
                                    const AstNode* other_body = other_lambda->as.lambda.body;
                                    if (other_body->type == AST_CALL && other_body->as.call.callee->type == AST_IDENTIFIER &&
                                        expr->as.define.name->type == AST_IDENTIFIER &&
                                        strcmp(other_body->as.call.callee->as.identifier.name, expr->as.define.name->as.identifier.name) == 0) {
                                        optimizable_count++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Format the analysis results
    snprintf(buffer, 4096, 
             "Tail Call Optimization Analysis:\n"
             "-------------------------------\n"
             "Total tail calls: %d\n"
             "Optimizable tail calls: %d\n",
             tail_call_count, optimizable_count);
    
    return buffer;
}

bool codegen_profile(CodegenContext* context, AstNode* ast) {
    assert(context != NULL);
    assert(ast != NULL);
    
    printf("Code Generation Profiling\n");
    printf("========================\n\n");

    // Make sure memory tracking is enabled
    bool was_enabled = memory_tracking_is_enabled();
    memory_tracking_set_enabled(true);
    
    // Get binding system from context
    BindingSystem* binding_system = codegen_context_get_binding_system(context);
    if (!binding_system) {
        DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
        diagnostic_error(diagnostics, 0, 0, "Binding system not available");
        return false;
    }
    
    // Profile AST analysis phase
    ProfilingPhase ast_phase;
    start_profiling_phase(&ast_phase, "AST Analysis");
    
    // Analyze the AST for closures and tail recursion
    char* ast_analysis = analyze_ast_for_closures_and_tail_recursion(ast, binding_system);
    if (ast_analysis) {
        printf("%s\n", ast_analysis);
        free(ast_analysis);
    }
    
    end_profiling_phase(&ast_phase);
    print_profiling_results(&ast_phase);
    
    // Profile IR generation phase
    ProfilingPhase ir_phase;
    start_profiling_phase(&ir_phase, "IR Generation");
    
    // Generate IR for the AST (to a temporary buffer)
    FILE* ir_buffer = tmpfile();
    if (ir_buffer) {
        generate_ir_for_node(ir_buffer, ast, 0);
        fclose(ir_buffer);
    }
    
    end_profiling_phase(&ir_phase);
    print_profiling_results(&ir_phase);
    
    // Profile C code generation phase
    ProfilingPhase c_phase;
    start_profiling_phase(&c_phase, "C Code Generation");
    
    // Generate C code for the AST (to a temporary buffer)
    FILE* c_buffer = tmpfile();
    if (c_buffer) {
        generate_c_code_for_node(c_buffer, ast, 0);
        fclose(c_buffer);
    }
    
    end_profiling_phase(&c_phase);
    print_profiling_results(&c_phase);
    
    // Profile closure analysis phase
    ProfilingPhase closure_phase;
    start_profiling_phase(&closure_phase, "Closure Analysis");
    
    // Analyze closures in the AST
    if (ast->type == AST_PROGRAM) {
        for (size_t i = 0; i < ast->as.program.expr_count; i++) {
            const AstNode* expr = ast->as.program.exprs[i];
            
            // Check for lambdas
            if (expr->type == AST_LAMBDA) {
                uint64_t lambda_id = binding_system_register_lambda(binding_system, expr->scope_id);
                if (lambda_id != 0) {
                    // Analyze the environment chain for this lambda
                    char* env_analysis = analyze_environment_chain(binding_system, lambda_id);
                    if (env_analysis) {
                        printf("%s\n", env_analysis);
                        free(env_analysis);
                    }
                }
            }
            
            // Check for lambda in define
            if (expr->type == AST_DEFINE && expr->as.define.value->type == AST_LAMBDA) {
                const AstNode* lambda = expr->as.define.value;
                uint64_t lambda_id = binding_system_register_lambda(binding_system, lambda->scope_id);
                if (lambda_id != 0) {
                    // Analyze the environment chain for this lambda
                    char* env_analysis = analyze_environment_chain(binding_system, lambda_id);
                    if (env_analysis) {
                        printf("%s\n", env_analysis);
                        free(env_analysis);
                    }
                }
            }
        }
    }
    
    end_profiling_phase(&closure_phase);
    print_profiling_results(&closure_phase);
    
    // Profile tail call optimization phase
    ProfilingPhase tail_call_phase;
    start_profiling_phase(&tail_call_phase, "Tail Call Optimization");
    
    // Analyze tail call optimization opportunities
    char* tail_call_analysis = analyze_tail_call_optimization(ast);
    if (tail_call_analysis) {
        printf("%s\n", tail_call_analysis);
        free(tail_call_analysis);
    }
    
    end_profiling_phase(&tail_call_phase);
    print_profiling_results(&tail_call_phase);
    
    // Overall statistics
    printf("Overall Statistics\n");
    printf("-----------------\n");
    printf("Total Time: %.6f seconds\n", 
           (double)((ast_phase.end_time - ast_phase.start_time) + 
                   (ir_phase.end_time - ir_phase.start_time) + 
                   (c_phase.end_time - c_phase.start_time) +
                   (closure_phase.end_time - closure_phase.start_time) +
                   (tail_call_phase.end_time - tail_call_phase.start_time)) / CLOCKS_PER_SEC);
    printf("Peak Memory Usage: %zu bytes\n", memory_tracking_get_stats().peak_bytes_allocated);
    printf("Total Allocations: %zu\n", memory_tracking_get_stats().total_allocations);
    
    // Restore memory tracking state
    memory_tracking_set_enabled(was_enabled);
    
    return true;
}

/**
 * @brief Generate IR for an AST node
 * 
 * @param output The output file
 * @param node The AST node
 * @param indent The indentation level
 */
static void generate_ir_for_node(FILE* output, const AstNode* node, int indent) {
    assert(output != NULL);
    assert(node != NULL);
    
    print_indent(output, indent);
    
    switch (node->type) {
        case AST_LITERAL_NUMBER:
            fprintf(output, "NUMBER %g\n", node->as.number.value);
            break;
        case AST_LITERAL_BOOLEAN:
            fprintf(output, "BOOLEAN %s\n", node->as.boolean.value ? "true" : "false");
            break;
        case AST_LITERAL_CHARACTER:
            fprintf(output, "CHARACTER '%c'\n", node->as.character.value);
            break;
        case AST_LITERAL_STRING:
            fprintf(output, "STRING \"%s\"\n", node->as.string.value);
            break;
        case AST_LITERAL_VECTOR:
            fprintf(output, "VECTOR [%zu]\n", node->as.vector.count);
            for (size_t i = 0; i < node->as.vector.count; i++) {
                generate_ir_for_node(output, node->as.vector.elements[i], indent + 1);
            }
            break;
        case AST_LITERAL_NIL:
            fprintf(output, "NIL\n");
            break;
        case AST_IDENTIFIER:
            fprintf(output, "IDENTIFIER %s (binding_id: %lu)\n", node->as.identifier.name, node->binding_id);
            break;
        case AST_DEFINE:
            fprintf(output, "DEFINE\n");
            print_indent(output, indent + 1);
            fprintf(output, "NAME:\n");
            generate_ir_for_node(output, node->as.define.name, indent + 2);
            print_indent(output, indent + 1);
            fprintf(output, "VALUE:\n");
            generate_ir_for_node(output, node->as.define.value, indent + 2);
            break;
        case AST_LAMBDA:
            generate_ir_for_lambda(output, node, indent);
            break;
        case AST_IF:
            fprintf(output, "IF\n");
            print_indent(output, indent + 1);
            fprintf(output, "CONDITION:\n");
            generate_ir_for_node(output, node->as.if_expr.condition, indent + 2);
            print_indent(output, indent + 1);
            fprintf(output, "THEN:\n");
            generate_ir_for_node(output, node->as.if_expr.then_branch, indent + 2);
            if (node->as.if_expr.else_branch) {
                print_indent(output, indent + 1);
                fprintf(output, "ELSE:\n");
                generate_ir_for_node(output, node->as.if_expr.else_branch, indent + 2);
            }
            break;
        case AST_BEGIN:
            fprintf(output, "BEGIN [%zu]\n", node->as.begin.expr_count);
            for (size_t i = 0; i < node->as.begin.expr_count; i++) {
                generate_ir_for_node(output, node->as.begin.exprs[i], indent + 1);
            }
            break;
        case AST_QUOTE:
            fprintf(output, "QUOTE\n");
            generate_ir_for_node(output, node->as.quote.expr, indent + 1);
            break;
        case AST_SET:
            fprintf(output, "SET!\n");
            print_indent(output, indent + 1);
            fprintf(output, "NAME:\n");
            generate_ir_for_node(output, node->as.set.name, indent + 2);
            print_indent(output, indent + 1);
            fprintf(output, "VALUE:\n");
            generate_ir_for_node(output, node->as.set.value, indent + 2);
            break;
        case AST_LET:
        case AST_LETREC:
        case AST_LETSTAR:
            fprintf(output, "%s [%zu]\n", 
                    node->type == AST_LET ? "LET" : 
                    node->type == AST_LETREC ? "LETREC" : "LET*", 
                    node->as.let.binding_count);
            print_indent(output, indent + 1);
            fprintf(output, "BINDINGS:\n");
            for (size_t i = 0; i < node->as.let.binding_count; i++) {
                generate_ir_for_node(output, node->as.let.bindings[i], indent + 2);
            }
            print_indent(output, indent + 1);
            fprintf(output, "BODY:\n");
            generate_ir_for_node(output, node->as.let.body, indent + 2);
            break;
        case AST_AND:
        case AST_OR:
            fprintf(output, "%s [%zu]\n", 
                    node->type == AST_AND ? "AND" : "OR", 
                    node->as.logical.expr_count);
            for (size_t i = 0; i < node->as.logical.expr_count; i++) {
                generate_ir_for_node(output, node->as.logical.exprs[i], indent + 1);
            }
            break;
        case AST_COND:
            fprintf(output, "COND [%zu]\n", node->as.cond.clause_count);
            for (size_t i = 0; i < node->as.cond.clause_count; i++) {
                generate_ir_for_node(output, node->as.cond.clauses[i], indent + 1);
            }
            break;
        case AST_CASE:
            fprintf(output, "CASE [%zu]\n", node->as.case_expr.clause_count);
            print_indent(output, indent + 1);
            fprintf(output, "KEY:\n");
            generate_ir_for_node(output, node->as.case_expr.key, indent + 2);
            print_indent(output, indent + 1);
            fprintf(output, "CLAUSES:\n");
            for (size_t i = 0; i < node->as.case_expr.clause_count; i++) {
                generate_ir_for_node(output, node->as.case_expr.clauses[i], indent + 2);
            }
            break;
        case AST_DO:
            fprintf(output, "DO [%zu]\n", node->as.do_expr.binding_count);
            print_indent(output, indent + 1);
            fprintf(output, "BINDINGS:\n");
            for (size_t i = 0; i < node->as.do_expr.binding_count; i++) {
                generate_ir_for_node(output, node->as.do_expr.bindings[i], indent + 2);
            }
            print_indent(output, indent + 1);
            fprintf(output, "TEST:\n");
            generate_ir_for_node(output, node->as.do_expr.test, indent + 2);
            print_indent(output, indent + 1);
            fprintf(output, "RESULT:\n");
            for (size_t i = 0; i < node->as.do_expr.result_count; i++) {
                generate_ir_for_node(output, node->as.do_expr.result[i], indent + 2);
            }
            print_indent(output, indent + 1);
            fprintf(output, "BODY:\n");
            for (size_t i = 0; i < node->as.do_expr.body_count; i++) {
                generate_ir_for_node(output, node->as.do_expr.body[i], indent + 2);
            }
            break;
        case AST_DELAY:
            fprintf(output, "DELAY\n");
            generate_ir_for_node(output, node->as.delay.expr, indent + 1);
            break;
        case AST_QUASIQUOTE:
            fprintf(output, "QUASIQUOTE\n");
            generate_ir_for_node(output, node->as.quasiquote.expr, indent + 1);
            break;
        case AST_UNQUOTE:
            fprintf(output, "UNQUOTE\n");
            generate_ir_for_node(output, node->as.unquote.expr, indent + 1);
            break;
        case AST_UNQUOTE_SPLICING:
            fprintf(output, "UNQUOTE-SPLICING\n");
            generate_ir_for_node(output, node->as.unquote_splicing.expr, indent + 1);
            break;
        case AST_CALL:
            generate_ir_for_call(output, node, indent);
            break;
        case AST_PROGRAM:
            fprintf(output, "PROGRAM [%zu]\n", node->as.program.expr_count);
            for (size_t i = 0; i < node->as.program.expr_count; i++) {
                generate_ir_for_node(output, node->as.program.exprs[i], indent + 1);
            }
            break;
        case AST_BINDING_PAIR:
            fprintf(output, "BINDING-PAIR\n");
            print_indent(output, indent + 1);
            fprintf(output, "NAME:\n");
            generate_ir_for_node(output, node->as.binding_pair.name, indent + 2);
            print_indent(output, indent + 1);
            fprintf(output, "VALUE:\n");
            generate_ir_for_node(output, node->as.binding_pair.value, indent + 2);
            break;
        case AST_COND_CLAUSE:
            fprintf(output, "COND-CLAUSE\n");
            print_indent(output, indent + 1);
            fprintf(output, "TEST:\n");
            generate_ir_for_node(output, node->as.cond_clause.test, indent + 2);
            print_indent(output, indent + 1);
            fprintf(output, "RESULT:\n");
            generate_ir_for_node(output, node->as.cond_clause.result, indent + 2);
            break;
        case AST_CASE_CLAUSE:
            fprintf(output, "CASE-CLAUSE\n");
            print_indent(output, indent + 1);
            fprintf(output, "DATUM:\n");
            generate_ir_for_node(output, node->as.case_clause.datum, indent + 2);
            print_indent(output, indent + 1);
            fprintf(output, "EXPR:\n");
            generate_ir_for_node(output, node->as.case_clause.expr, indent + 2);
            break;
        default:
            fprintf(output, "UNKNOWN NODE TYPE: %d\n", node->type);
            break;
    }
}

static void generate_ir_for_call(FILE* output, const AstNode* node, int indent) {
    assert(output != NULL);
    assert(node != NULL);
    assert(node->type == AST_CALL);
    
    fprintf(output, "CALL\n");
    print_indent(output, indent + 1);
    fprintf(output, "CALLEE:\n");
    generate_ir_for_node(output, node->as.call.callee, indent + 2);
    
    print_indent(output, indent + 1);
    fprintf(output, "ARGUMENTS [%zu]:\n", node->as.call.arg_count);
    for (size_t i = 0; i < node->as.call.arg_count; i++) {
        generate_ir_for_node(output, node->as.call.args[i], indent + 2);
    }
}

static void generate_ir_for_lambda(FILE* output, const AstNode* node, int indent) {
    assert(output != NULL);
    assert(node != NULL);
    assert(node->type == AST_LAMBDA);
    
    fprintf(output, "LAMBDA (scope_id: %lu)\n", node->scope_id);
    
    print_indent(output, indent + 1);
    fprintf(output, "PARAMETERS [%zu]:\n", node->as.lambda.param_count);
    for (size_t i = 0; i < node->as.lambda.param_count; i++) {
        print_indent(output, indent + 2);
        fprintf(output, "%s\n", node->as.lambda.params[i]);
    }
    
    print_indent(output, indent + 1);
    fprintf(output, "BODY:\n");
    generate_ir_for_node(output, node->as.lambda.body, indent + 2);
}

static void generate_c_code_for_node(FILE* output, const AstNode* node, int indent) {
    assert(output != NULL);
    assert(node != NULL);
    
    print_indent(output, indent);
    
    switch (node->type) {
        case AST_LITERAL_NUMBER:
            fprintf(output, "%g", node->as.number.value);
            break;
        case AST_LITERAL_BOOLEAN:
            fprintf(output, "%s", node->as.boolean.value ? "true" : "false");
            break;
        case AST_LITERAL_CHARACTER:
            fprintf(output, "'%c'", node->as.character.value);
            break;
        case AST_LITERAL_STRING:
            fprintf(output, "\"%s\"", node->as.string.value);
            break;
        case AST_LITERAL_VECTOR:
            fprintf(output, "/* Vector literal */\n");
            print_indent(output, indent);
            fprintf(output, "eshkol_vector_create(%zu, (void*[]){", node->as.vector.count);
            for (size_t i = 0; i < node->as.vector.count; i++) {
                if (i > 0) {
                    fprintf(output, ", ");
                }
                fprintf(output, "(void*)(");
                generate_c_code_for_node(output, node->as.vector.elements[i], 0);
                fprintf(output, ")");
            }
            fprintf(output, "})");
            break;
        case AST_LITERAL_NIL:
            fprintf(output, "NULL");
            break;
        case AST_IDENTIFIER:
            fprintf(output, "%s", node->as.identifier.name);
            break;
        case AST_DEFINE:
            if (node->as.define.name->type == AST_IDENTIFIER) {
                if (node->as.define.value->type == AST_LAMBDA) {
                    // Function definition
                    fprintf(output, "/* Function definition: %s */\n", node->as.define.name->as.identifier.name);
                    print_indent(output, indent);
                    fprintf(output, "void* %s(EshkolEnvironment* env, void** args) {\n", 
                            node->as.define.name->as.identifier.name);
                    
                    // Parameters
                    for (size_t i = 0; i < node->as.define.value->as.lambda.param_count; i++) {
                        print_indent(output, indent + 1);
                        fprintf(output, "void* %s = args[%zu];\n", 
                                node->as.define.value->as.lambda.params[i], i);
                    }
                    
                    // Body
                    print_indent(output, indent + 1);
                    fprintf(output, "return (void*)(");
                    generate_c_code_for_node(output, node->as.define.value->as.lambda.body, 0);
                    fprintf(output, ");\n");
                    
                    print_indent(output, indent);
                    fprintf(output, "}");
                } else {
                    // Variable definition
                    fprintf(output, "/* Variable definition: %s */\n", node->as.define.name->as.identifier.name);
                    print_indent(output, indent);
                    fprintf(output, "void* %s = (void*)(", node->as.define.name->as.identifier.name);
                    generate_c_code_for_node(output, node->as.define.value, 0);
                    fprintf(output, ")");
                }
            } else {
                fprintf(output, "/* Complex define */");
            }
            break;
        case AST_LAMBDA:
            generate_c_code_for_lambda(output, node, indent);
            break;
        case AST_IF:
            fprintf(output, "((");
            generate_c_code_for_node(output, node->as.if_expr.condition, 0);
            fprintf(output, ") ? (");
            generate_c_code_for_node(output, node->as.if_expr.then_branch, 0);
            fprintf(output, ") : (");
            if (node->as.if_expr.else_branch) {
                generate_c_code_for_node(output, node->as.if_expr.else_branch, 0);
            } else {
                fprintf(output, "NULL");
            }
            fprintf(output, "))");
            break;
        case AST_BEGIN:
            fprintf(output, "({ /* begin */\n");
            for (size_t i = 0; i < node->as.begin.expr_count - 1; i++) {
                print_indent(output, indent + 1);
                generate_c_code_for_node(output, node->as.begin.exprs[i], 0);
                fprintf(output, ";\n");
            }
            if (node->as.begin.expr_count > 0) {
                print_indent(output, indent + 1);
                fprintf(output, "(void*)(");
                generate_c_code_for_node(output, node->as.begin.exprs[node->as.begin.expr_count - 1], 0);
                fprintf(output, ");\n");
            }
            print_indent(output, indent);
            fprintf(output, "})");
            break;
        case AST_CALL:
            generate_c_code_for_call(output, node, indent);
            break;
        default:
            fprintf(output, "/* Unsupported node type: %d */", node->type);
            break;
    }
}

static void generate_c_code_for_lambda(FILE* output, const AstNode* node, int indent) {
    assert(output != NULL);
    assert(node != NULL);
    assert(node->type == AST_LAMBDA);
    
    fprintf(output, "/* Lambda (scope_id: %lu) */\n", node->scope_id);
    print_indent(output, indent);
    fprintf(output, "eshkol_closure_create(\n");
    
    // Function
    print_indent(output, indent + 1);
    fprintf(output, "/* Function */\n");
    print_indent(output, indent + 1);
    fprintf(output, "({ void* lambda_%lu(EshkolEnvironment* env, void** args) {\n", node->scope_id);
    
    // Parameters
    for (size_t i = 0; i < node->as.lambda.param_count; i++) {
        print_indent(output, indent + 2);
        fprintf(output, "void* %s = args[%zu];\n", node->as.lambda.params[i], i);
    }
    
    // Body
    print_indent(output, indent + 2);
    fprintf(output, "return (void*)(");
    generate_c_code_for_node(output, node->as.lambda.body, 0);
    fprintf(output, ");\n");
    
    print_indent(output, indent + 1);
    fprintf(output, "} lambda_%lu; }),\n", node->scope_id);
    
    // Environment
    print_indent(output, indent + 1);
    fprintf(output, "/* Environment */\n");
    print_indent(output, indent + 1);
    fprintf(output, "eshkol_environment_create(env, 0),\n");
    
    // Return type and parameter types
    print_indent(output, indent + 1);
    fprintf(output, "/* Types */\n");
    print_indent(output, indent + 1);
    fprintf(output, "NULL, NULL, %zu\n", node->as.lambda.param_count);
    
    print_indent(output, indent);
    fprintf(output, ")");
}

static void generate_c_code_for_call(FILE* output, const AstNode* node, int indent) {
    assert(output != NULL);
    assert(node != NULL);
    assert(node->type == AST_CALL);
    
    // Check if this is a direct function call or a closure call
    if (node->as.call.callee->type == AST_IDENTIFIER) {
        // Direct function call
        fprintf(output, "%s(env, (void*[]){", node->as.call.callee->as.identifier.name);
        
        // Arguments
        for (size_t i = 0; i < node->as.call.arg_count; i++) {
            if (i > 0) {
                fprintf(output, ", ");
            }
            fprintf(output, "(void*)(");
            generate_c_code_for_node(output, node->as.call.args[i], 0);
            fprintf(output, ")");
        }
        
        fprintf(output, "})");
    } else {
        // Closure call
        fprintf(output, "eshkol_closure_call(\n");
        
        // Callee
        print_indent(output, indent + 1);
        fprintf(output, "/* Callee */\n");
        print_indent(output, indent + 1);
        fprintf(output, "(EshkolClosure*)(");
        generate_c_code_for_node(output, node->as.call.callee, 0);
        fprintf(output, "),\n");
        
        // Arguments
        print_indent(output, indent + 1);
        fprintf(output, "/* Arguments */\n");
        print_indent(output, indent + 1);
        fprintf(output, "(void*[]){");
        
        for (size_t i = 0; i < node->as.call.arg_count; i++) {
            if (i > 0) {
                fprintf(output, ", ");
            }
            fprintf(output, "(void*)(");
            generate_c_code_for_node(output, node->as.call.args[i], 0);
            fprintf(output, ")");
        }
        
        fprintf(output, "}\n");
        
        print_indent(output, indent);
        fprintf(output, ")");
    }
}