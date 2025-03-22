/**
 * @file codegen.c
 * @brief Implementation of the code generation system
 */

#include "backend/codegen.h"
#include "core/memory.h"
#include "core/diagnostics.h"
#include "core/file_io.h"
#include "frontend/ast/ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

/**
 * @brief Code generator context structure
 */
struct CodegenContext {
    Arena* arena;                /**< Arena for allocations */
    DiagnosticContext* diagnostics; /**< Diagnostic context for error reporting */
    FILE* output;                /**< Output file */
    int indent_level;            /**< Current indentation level */
    bool in_function;            /**< Whether we're currently in a function */
    char* temp_dir;              /**< Temporary directory for compilation */
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
    
    // TODO: Implement vector literals
    diagnostic_error(context->diagnostics, node->line, node->column, 
                    "Vector literals not yet implemented");
    return false;
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
    
    fprintf(context->output, "%s", node->as.identifier.name);
    
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
 * @brief Generate C code for a function definition
 */
static bool generate_function_def(CodegenContext* context, const AstNode* node) {
    assert(node->type == AST_FUNCTION_DEF);
    
    // Generate function declaration with return type
    fprintf(context->output, "int ");
    
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
        
        fprintf(context->output, "int ");
        if (!generate_expression(context, node->as.function_def.params[i])) {
            return false;
        }
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
        
        fprintf(context->output, "eshkol_value_t ");
        if (!generate_expression(context, node->as.let.bindings[i])) {
            return false;
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
        
        if (!generate_expression(context, node->as.sequence.exprs[i])) {
            return false;
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
    fprintf(context->output, "#include <stdbool.h>\n\n");
    
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
            fprintf(context->output, "int ");
            generate_expression(context, node->as.program.exprs[i]->as.function_def.name);
            fprintf(context->output, "(");
            
            for (size_t j = 0; j < node->as.program.exprs[i]->as.function_def.param_count; j++) {
                if (j > 0) {
                    fprintf(context->output, ", ");
                }
                fprintf(context->output, "int");
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
        fprintf(context->output, "    printf(\"Hello from Eshkol!\\n\");\n");
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
    
    // Create command to compile the C file
    char compile_cmd[1024];
    snprintf(compile_cmd, sizeof(compile_cmd), "gcc -o %s.out %s", c_file, c_file);
    
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
