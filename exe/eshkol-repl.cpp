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
#include <csignal>
#include <unistd.h>  // For close(), unlink(), mkstemp()

// Global flag for Ctrl+C handling
volatile sig_atomic_t g_interrupted = 0;

// Signal handler for Ctrl+C
void sigint_handler(int sig) {
    (void)sig;  // Unused
    g_interrupted = 1;
    // Don't call non-async-signal-safe functions here
    // Just set the flag and let the main loop handle it
}

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

// Helper: Check if input has balanced parentheses
bool is_balanced(const std::string& input) {
    int depth = 0;
    bool in_string = false;
    bool in_comment = false;

    for (size_t i = 0; i < input.length(); ++i) {
        char c = input[i];

        // Handle line comments
        if (c == ';' && !in_string) {
            in_comment = true;
        }
        if (c == '\n' && in_comment) {
            in_comment = false;
            continue;
        }
        if (in_comment) {
            continue;
        }

        // Handle string literals
        if (c == '"' && (i == 0 || input[i-1] != '\\')) {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }

        // Count parentheses
        if (c == '(') {
            depth++;
        } else if (c == ')') {
            depth--;
            if (depth < 0) {
                return false;  // More closing than opening
            }
        }
    }

    return depth == 0;
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
    // Install Ctrl+C handler
    signal(SIGINT, sigint_handler);

    std::cout << "Eshkol REPL v0.1.1 (Compiler-First Architecture)\n";
    std::cout << "Type (exit) or Ctrl+D to quit\n";
    std::cout << "Type :help for commands\n";
    std::cout << "Press Ctrl+C to cancel multi-line input\n\n";

    // Initialize REPL JIT context
    eshkol::ReplJITContext repl_ctx;

    std::cout << "\n";  // After JIT initialization message

    while (true) {
        // Reset interrupt flag
        g_interrupted = 0;

        // Accumulate multi-line input
        std::string input_str;
        bool first_line = true;
        int line_count = 0;

        while (true) {
            // Check for interrupt before reading
            if (g_interrupted) {
                std::cout << "\n^C (cancelled)\n";
                g_interrupted = 0;
                input_str.clear();
                break;
            }

            // Calculate current paren depth for better UX
            int current_depth = 0;
            if (!first_line && !input_str.empty()) {
                bool in_str = false, in_com = false;
                for (char c : input_str) {
                    if (c == ';' && !in_str) in_com = true;
                    if (c == '\n' && in_com) { in_com = false; continue; }
                    if (in_com) continue;
                    if (c == '"') { in_str = !in_str; continue; }
                    if (in_str) continue;
                    if (c == '(') current_depth++;
                    else if (c == ')') current_depth--;
                }
            }

            // Create prompt showing line number and paren depth for continuation
            std::string prompt_str;
            if (first_line) {
                prompt_str = "eshkol> ";
            } else {
                prompt_str = "  [" + std::to_string(line_count) +
                             "," + std::to_string(current_depth) + "]> ";
            }

            char* input = readline(prompt_str.c_str());

            // EOF (Ctrl+D)
            if (!input) {
                if (!first_line && !input_str.empty()) {
                    // Treat EOF in continuation as force-completing the input
                    std::cout << " (force completing with unbalanced parentheses)\n";
                    break;
                }
                std::cout << "\nGoodbye!" << std::endl;
                _exit(0);
            }

            // Check if continuation line is empty - treat as "remove last line" or cancel
            if (!first_line && strlen(input) == 0 && !input_str.empty()) {
                // Find the last newline and remove everything after it
                size_t last_newline = input_str.rfind('\n');
                if (last_newline != std::string::npos) {
                    // Multiple lines - remove the last one
                    input_str = input_str.substr(0, last_newline);
                    line_count--;
                    std::cout << "(removed last line, " << line_count << " line"
                              << (line_count != 1 ? "s" : "") << " remaining)\n";
                    free(input);
                    continue;
                } else {
                    // Only one line - cancel input instead of removing
                    std::cout << "(cancelled)\n";
                    input_str.clear();
                    line_count = 0;
                    first_line = true;
                    free(input);
                    continue;
                }
            }

            // Add to accumulated input
            if (!first_line && !input_str.empty()) {
                input_str += "\n";
            }
            input_str += input;
            line_count++;

            // Check for exit on first line
            if (first_line && (strcmp(input, "(exit)") == 0 ||
                              strcmp(input, "exit") == 0 ||
                              strcmp(input, "quit") == 0)) {
                free(input);
                std::cout << "Goodbye!" << std::endl;
                _exit(0);
            }

            // Check for special commands (work on any line)
            if (strcmp(input, ":cancel") == 0) {
                free(input);
                std::cout << "(cancelled multi-line input)\n";
                input_str.clear();
                break;
            }

            // First-line-only commands
            if (first_line && strcmp(input, ":help") == 0) {
                free(input);
                std::cout << "Available commands:\n";
                std::cout << "  :help     - Show this help\n";
                std::cout << "  :quit     - Exit REPL\n";
                std::cout << "  :cancel   - Cancel multi-line input\n";
                std::cout << "  (exit)    - Exit REPL\n";
                std::cout << "\nMulti-line editing:\n";
                std::cout << "  - Incomplete expressions auto-continue to next line\n";
                std::cout << "  - Continuation prompt: [line,depth]>\n";
                std::cout << "    * line  = continuation line number (starts at 1)\n";
                std::cout << "    * depth = unclosed '(' count (evaluates when 0)\n";
                std::cout << "  - Empty line on continuation removes last line\n";
                std::cout << "  - Ctrl+C cancels multi-line input\n";
                std::cout << "  - Ctrl+D on continuation force-completes input\n";
                std::cout << "\nAll compiler features available:\n";
                std::cout << "  - Arithmetic: (+ 1 2), (- 10 3), (* 4 5), (/ 10 2)\n";
                std::cout << "  - Variables: (define x 10)\n";
                std::cout << "  - Lambdas: (define square (lambda (n) (* n n)))\n";
                std::cout << "  - Autodiff: (derivative (lambda (x) (* x x)) 5.0)\n";
                std::cout << "  - Lists: (list 1 2 3), (map + (list 1 2) (list 3 4))\n";
                input_str.clear();
                break;
            }

            if (first_line && strcmp(input, ":quit") == 0) {
                free(input);
                std::cout << "Goodbye!" << std::endl;
                _exit(0);
            }

            free(input);

            // Empty first line - just continue
            if (first_line && input_str.empty()) {
                input_str.clear();
                break;
            }

            // Check parenthesis balance
            int depth = 0;
            bool in_string = false;
            bool in_comment = false;
            for (size_t i = 0; i < input_str.length(); ++i) {
                char c = input_str[i];
                if (c == ';' && !in_string) in_comment = true;
                if (c == '\n' && in_comment) { in_comment = false; continue; }
                if (in_comment) continue;
                if (c == '"' && (i == 0 || input_str[i-1] != '\\')) {
                    in_string = !in_string;
                    continue;
                }
                if (in_string) continue;
                if (c == '(') depth++;
                else if (c == ')') {
                    depth--;
                    if (depth < 0) {
                        // More closing than opening - syntax error
                        std::cerr << "Error: Unmatched closing parenthesis\n";
                        input_str.clear();
                        break;
                    }
                }
            }

            // Check if we had an error (depth < 0)
            if (input_str.empty()) {
                break;  // Error was reported, start over
            }

            // Check if expression is complete (balanced)
            if (depth == 0) {
                // Complete expression - add to history and process
                add_history(input_str.c_str());
                break;
            }

            // Not balanced - continue reading more lines
            first_line = false;
        }

        // Skip if empty (from :help or empty line)
        if (input_str.empty()) {
            continue;
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
