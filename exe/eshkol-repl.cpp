//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//

#include <eshkol/eshkol.h>
#include "../lib/repl/repl_jit.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <unistd.h>  // For close(), unlink(), mkstemp()

#ifdef HAVE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#else
// Fallback if readline not available
char* readline(const char* prompt) {
    std::cout << prompt;
    std::string line;
    if (!std::getline(std::cin, line)) {
        return nullptr;  // EOF
    }
    char* result = (char*)malloc(line.length() + 1);
    strcpy(result, line.c_str());
    return result;
}
void add_history(const char*) {}  // No-op if no readline
#endif

// Parser function from eshkol.h (declared in C++)
extern "C++" {
    eshkol_ast_t eshkol_parse_next_ast(std::ifstream &in_file);
}

// Helper: Parse a string to AST by writing to temp file
// This is a temporary solution - we should modify the parser to accept std::istream
eshkol_ast_t parse_string(const std::string& input) {
    // Create a temporary file
    std::string temp_file = "/tmp/eshkol_repl_input_XXXXXX";
    char temp_name[256];
    strcpy(temp_name, temp_file.c_str());

    int fd = mkstemp(temp_name);
    if (fd == -1) {
        eshkol_ast_t invalid_ast;
        invalid_ast.type = ESHKOL_INVALID;
        return invalid_ast;
    }

    // Write input to temp file (add newline for parser)
    std::ofstream temp_out(temp_name);
    temp_out << input << "\n";
    temp_out.close();
    close(fd);

    // Parse from temp file
    std::ifstream temp_in(temp_name);
    eshkol_ast_t ast = eshkol_parse_next_ast(temp_in);
    temp_in.close();

    // Clean up temp file
    unlink(temp_name);

    return ast;
}

// Helper: Check if AST is a definition statement (returns false for expressions)
bool is_definition_statement(const eshkol_ast_t& ast) {
    if (ast.type != ESHKOL_OP) {
        return false;  // Atoms are expressions
    }

    // Check if it's a define operation
    if (ast.operation.op == ESHKOL_DEFINE_OP) {
        return true;
    }

    // All other operations are expressions
    return false;
}

// Helper: Check if AST defines a lambda and return the variable name if so
const char* get_lambda_var_name(const eshkol_ast_t& ast) {
    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_DEFINE_OP) {
        return nullptr;
    }

    // Check if the value is a lambda
    if (ast.operation.define_op.value &&
        ast.operation.define_op.value->type == ESHKOL_OP &&
        ast.operation.define_op.value->operation.op == ESHKOL_LAMBDA_OP) {
        return ast.operation.define_op.name;
    }

    return nullptr;
}

int main(int argc, char** argv) {
    std::cout << "Eshkol REPL v0.1.1 (Compiler-First Architecture)\n";
    std::cout << "Type (exit) or Ctrl+D to quit\n";
    std::cout << "Type :help for commands\n\n";

    // Initialize REPL JIT context
    eshkol::ReplJITContext repl_ctx;

    std::cout << "\n";  // After JIT initialization message

    while (true) {
        char* input = readline("eshkol> ");

        // EOF (Ctrl+D)
        if (!input) {
            std::cout << "\nGoodbye!" << std::endl;  // Use endl to flush
            break;
        }

        // Empty input
        if (strlen(input) == 0) {
            free(input);
            continue;
        }

        // Add to history
        add_history(input);

        std::string input_str(input);
        free(input);

        // Check for exit command
        if (input_str == "(exit)" || input_str == "exit" || input_str == "quit") {
            std::cout << "Goodbye!" << std::endl;  // Use endl to flush
            break;
        }

        // Check for special commands
        if (input_str == ":help") {
            std::cout << "Available commands:\n";
            std::cout << "  :help     - Show this help\n";
            std::cout << "  :quit     - Exit REPL\n";
            std::cout << "  (exit)    - Exit REPL\n";
            std::cout << "\nAll compiler features available:\n";
            std::cout << "  - Arithmetic: (+ 1 2), (- 10 3), (* 4 5), (/ 10 2)\n";
            std::cout << "  - Variables: (define x 10)\n";
            std::cout << "  - Lambdas: (define square (lambda (n) (* n n)))\n";
            std::cout << "  - Autodiff: (derivative (lambda (x) (* x x)) 5.0)\n";
            std::cout << "  - Lists: (list 1 2 3), (map + (list 1 2) (list 3 4))\n";
            continue;
        }

        if (input_str == ":quit") {
            std::cout << "Goodbye!" << std::endl;  // Use endl to flush
            break;
        }

        // Parse and evaluate
        try {
            eshkol_ast_t ast = parse_string(input_str);

            if (ast.type == ESHKOL_INVALID) {
                std::cerr << "Error: Failed to parse input\n";
                continue;
            }

            // Check if this defines a lambda variable - if so, pre-register it
            const char* lambda_var = get_lambda_var_name(ast);
            if (lambda_var) {
                repl_ctx.registerLambdaVar(lambda_var);
            }

            // Wrap expressions (not definitions) with display
            eshkol_ast_t* ast_to_execute = &ast;
            bool should_display = !is_definition_statement(ast);

            if (should_display) {
                // Wrap: (begin (display <expr>) (newline))
                ast_to_execute = eshkol_wrap_with_display(&ast);
            }

            // Execute using JIT
            void* result = repl_ctx.execute(ast_to_execute);

            // Clean up wrapper if created
            if (should_display && ast_to_execute != &ast) {
                // Only clean the wrapper, not the original AST
                // (display call contains pointer to original AST)
                free(ast_to_execute->operation.call_op.func);
                free(ast_to_execute->operation.call_op.variables[0].operation.call_op.func);
                free(ast_to_execute->operation.call_op.variables[0].operation.call_op.variables);
                free(ast_to_execute->operation.call_op.variables[1].operation.call_op.func);
                free(ast_to_execute->operation.call_op.variables);
                free(ast_to_execute);
            }

            // Clean up original AST
            eshkol_ast_clean(&ast);

            // Clean up result
            if (result) {
                delete static_cast<int64_t*>(result);
            }

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
    }

    // Use _exit() instead of return to avoid static destructor issues with LLVM JIT
    _exit(0);
}
