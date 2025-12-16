//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//
// Eshkol Interactive REPL - A visual live coding experience
//

#include <eshkol/eshkol.h>
#include "../lib/repl/repl_jit.h"
#include "../lib/repl/repl_utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <setjmp.h>

using namespace eshkol::repl;

// Jump buffer for exception handling during JIT execution
static jmp_buf g_repl_exception_jmp_buf;

// Jump buffer for signal-based crash recovery (segfaults, etc.)
static sigjmp_buf g_crash_jmp_buf;
static volatile sig_atomic_t g_in_jit = 0;
static volatile sig_atomic_t g_crash_signal = 0;

// Global flag for Ctrl+C handling
volatile sig_atomic_t g_interrupted = 0;

// Global state for REPL
static std::string g_last_loaded_file;
static std::vector<std::string> g_defined_symbols;

// Signal handler for Ctrl+C
void sigint_handler(int sig) {
    (void)sig;
    g_interrupted = 1;
}

// Signal handler for crashes during JIT execution (SIGSEGV, SIGFPE, SIGBUS)
void crash_handler(int sig) {
    if (g_in_jit) {
        g_crash_signal = sig;
        siglongjmp(g_crash_jmp_buf, 1);
    } else {
        // Not in JIT - reraise with default handler
        signal(sig, SIG_DFL);
        raise(sig);
    }
}

// Get human-readable message for crash signal
const char* crash_signal_message(int sig) {
    switch (sig) {
        case SIGSEGV: return "Segmentation fault - likely a type error (e.g., arithmetic on non-numeric value)";
        case SIGFPE:  return "Floating point exception - likely division by zero";
        case SIGBUS:  return "Bus error - memory access issue";
        default:      return "Unknown runtime error";
    }
}

// Display an exception with nice formatting
void display_exception(eshkol_exception_t* exc) {
    using namespace color;

    const char* type_name = "error";
    switch (exc->type) {
        case ESHKOL_EXCEPTION_ERROR: type_name = "error"; break;
        case ESHKOL_EXCEPTION_TYPE_ERROR: type_name = "type-error"; break;
        case ESHKOL_EXCEPTION_FILE_ERROR: type_name = "file-error"; break;
        case ESHKOL_EXCEPTION_READ_ERROR: type_name = "read-error"; break;
        case ESHKOL_EXCEPTION_SYNTAX_ERROR: type_name = "syntax-error"; break;
        case ESHKOL_EXCEPTION_RANGE_ERROR: type_name = "range-error"; break;
        case ESHKOL_EXCEPTION_ARITY_ERROR: type_name = "arity-error"; break;
        case ESHKOL_EXCEPTION_DIVIDE_BY_ZERO: type_name = "divide-by-zero"; break;
        case ESHKOL_EXCEPTION_USER_DEFINED: type_name = "user-exception"; break;
    }

    std::cerr << error() << type_name << reset() << ": ";
    if (exc->message) {
        std::cerr << exc->message;
    }

    if (exc->line > 0) {
        std::cerr << dim() << " at line " << exc->line;
        if (exc->column > 0) {
            std::cerr << ", column " << exc->column;
        }
        std::cerr << reset();
    }
    std::cerr << "\n";
}

// Check if running interactively
static bool g_interactive = false;

#ifdef HAVE_READLINE
#include <readline/readline.h>
#include <readline/history.h>

// Compatibility: rl_basic_word_break_characters type
#if defined(__APPLE__) || defined(__MACH__)
// macOS readline uses char* (not const char*)
static char word_break_chars[] = " \t\n\"\\'`@$><=;|&{(";
#endif

// Tab completion generator
static char* symbol_generator(const char* text, int state) {
    static size_t list_index, len;
    static std::vector<std::string> matches;

    if (state == 0) {
        matches.clear();
        list_index = 0;
        len = strlen(text);

        // Add builtin matches
        for (const auto& sym : get_builtin_symbols()) {
            if (sym.compare(0, len, text) == 0) {
                matches.push_back(sym);
            }
        }

        // Add user-defined symbol matches
        for (const auto& sym : g_defined_symbols) {
            if (sym.compare(0, len, text) == 0) {
                matches.push_back(sym);
            }
        }
    }

    if (list_index < matches.size()) {
        return strdup(matches[list_index++].c_str());
    }

    return nullptr;
}

static char** eshkol_completion(const char* text, int start, int end) {
    (void)start;
    (void)end;
    rl_attempted_completion_over = 1;
    return rl_completion_matches(text, symbol_generator);
}

void init_readline() {
    // Set up tab completion
    rl_attempted_completion_function = eshkol_completion;
#if defined(__APPLE__) || defined(__MACH__)
    rl_basic_word_break_characters = word_break_chars;
#else
    rl_basic_word_break_characters = (char*)" \t\n\"\\'`@$><=;|&{(";
#endif

    // Load history
    std::string history_file = get_history_file_path();
    read_history(history_file.c_str());
}

void save_readline_history() {
    std::string history_file = get_history_file_path();
    write_history(history_file.c_str());
}

// Simple readline for non-interactive mode (pipes)
// In non-interactive mode, don't print prompts (cleaner output)
char* simple_readline(const char* /* prompt */) {
    std::string line;
    if (!std::getline(std::cin, line)) {
        return nullptr;
    }
    char* result = (char*)malloc(line.length() + 1);
    strcpy(result, line.c_str());
    return result;
}

// Wrapper that uses real readline only in interactive mode
char* eshkol_readline(const char* prompt) {
    if (g_interactive) {
        return readline(prompt);
    } else {
        return simple_readline(prompt);
    }
}

#else
// Fallback if readline not available
char* eshkol_readline(const char* prompt) {
    std::cout << prompt << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) {
        return nullptr;
    }
    char* result = (char*)malloc(line.length() + 1);
    strcpy(result, line.c_str());
    return result;
}
void add_history(const char*) {}
void init_readline() {}
void save_readline_history() {}
#endif

// Parser function from eshkol.h
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

        if (c == '"' && (i == 0 || input[i-1] != '\\')) {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }

        if (c == '(') {
            depth++;
        } else if (c == ')') {
            depth--;
            if (depth < 0) {
                return false;
            }
        }
    }

    return depth == 0;
}

// Helper: Calculate paren depth
int get_paren_depth(const std::string& input) {
    int depth = 0;
    bool in_string = false;
    bool in_comment = false;

    for (size_t i = 0; i < input.length(); ++i) {
        char c = input[i];

        if (c == ';' && !in_string) in_comment = true;
        if (c == '\n' && in_comment) { in_comment = false; continue; }
        if (in_comment) continue;
        if (c == '"' && (i == 0 || input[i-1] != '\\')) {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;
        if (c == '(') depth++;
        else if (c == ')') depth--;
    }

    return depth;
}

// Helper: Parse a string to AST by writing to temp file
eshkol_ast_t parse_string(const std::string& input) {
    std::string temp_file = "/tmp/eshkol_repl_input_XXXXXX";
    char temp_name[256];
    strcpy(temp_name, temp_file.c_str());

    int fd = mkstemp(temp_name);
    if (fd == -1) {
        eshkol_ast_t invalid_ast;
        invalid_ast.type = ESHKOL_INVALID;
        return invalid_ast;
    }

    std::ofstream temp_out(temp_name);
    temp_out << input << "\n";
    temp_out.close();
    close(fd);

    std::ifstream temp_in(temp_name);
    eshkol_ast_t ast = eshkol_parse_next_ast(temp_in);
    temp_in.close();

    unlink(temp_name);

    return ast;
}

// Helper: Check if AST is a statement that shouldn't be wrapped with display
bool is_definition_statement(const eshkol_ast_t& ast) {
    if (ast.type != ESHKOL_OP) {
        return false;
    }
    // Definition statements and module system statements don't produce displayable values
    if (ast.operation.op == ESHKOL_DEFINE_OP ||
        ast.operation.op == ESHKOL_IMPORT_OP ||
        ast.operation.op == ESHKOL_REQUIRE_OP ||
        ast.operation.op == ESHKOL_PROVIDE_OP) {
        return true;
    }
    // Check for display/print/newline calls - they already produce output
    if (ast.operation.op == ESHKOL_CALL_OP && ast.operation.call_op.func) {
        if (ast.operation.call_op.func->type == ESHKOL_VAR) {
            const char* name = ast.operation.call_op.func->variable.id;
            if (name && (strcmp(name, "display") == 0 ||
                         strcmp(name, "newline") == 0 ||
                         strcmp(name, "print") == 0 ||
                         strcmp(name, "write") == 0 ||
                         strcmp(name, "displayln") == 0)) {
                return true;
            }
        }
    }
    return false;
}

// Helper: Get defined name from AST
const char* get_defined_name(const eshkol_ast_t& ast) {
    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_DEFINE_OP) {
        return nullptr;
    }
    return ast.operation.define_op.name;
}

// Helper: Check if AST defines a lambda/function
const char* get_lambda_var_name(const eshkol_ast_t& ast) {
    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_DEFINE_OP) {
        return nullptr;
    }

    if (ast.operation.define_op.is_function) {
        return ast.operation.define_op.name;
    }

    if (ast.operation.define_op.value &&
        ast.operation.define_op.value->type == ESHKOL_OP &&
        ast.operation.define_op.value->operation.op == ESHKOL_LAMBDA_OP) {
        return ast.operation.define_op.name;
    }

    return nullptr;
}

// Print help message
void print_help() {
    using namespace color;

    std::cout << "\n" << bold() << bright_cyan() << "Eshkol REPL Commands" << reset() << "\n";
    std::cout << dim() << "───────────────────────────────────────────────────────────" << reset() << "\n\n";

    for (const auto& cmd : get_repl_commands()) {
        std::cout << "  " << bright_blue() << std::left << std::setw(12) << cmd.name << reset();
        if (!cmd.alias.empty()) {
            std::cout << dim() << "(" << cmd.alias << ")" << reset() << " ";
        } else {
            std::cout << "     ";
        }
        std::cout << cmd.description << "\n";
    }

    std::cout << "\n" << bold() << "Multi-line Editing:" << reset() << "\n";
    std::cout << "  " << dim() << "Incomplete expressions auto-continue to next line" << reset() << "\n";
    std::cout << "  " << dim() << "Continuation prompt shows: " << reset() << "[line,depth]>\n";
    std::cout << "  " << dim() << "Empty line on continuation removes last line" << reset() << "\n";
    std::cout << "  " << dim() << "Ctrl+C cancels, Ctrl+D force-completes" << reset() << "\n";
    std::cout << "\n";
}

// Print environment (defined symbols)
void print_environment() {
    using namespace color;

    std::cout << "\n" << bold() << bright_cyan() << "Defined Symbols" << reset() << "\n";
    std::cout << dim() << "───────────────────────────────────────────────────────────" << reset() << "\n";

    if (g_defined_symbols.empty()) {
        std::cout << dim() << "  (no user-defined symbols)" << reset() << "\n";
    } else {
        for (const auto& sym : g_defined_symbols) {
            std::cout << "  " << symbol() << sym << reset() << "\n";
        }
    }
    std::cout << "\n";
}

// Clear screen
void clear_screen() {
    std::cout << "\033[2J\033[H" << std::flush;
}

// Load and execute a file
bool load_file(const std::string& filename, eshkol::ReplJITContext& repl_ctx) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        print_error("Could not open file", filename);
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();

    // Parse and execute each expression in the file
    std::string remaining = content;
    int expr_count = 0;

    while (!remaining.empty()) {
        // Skip whitespace and comments
        size_t start = 0;
        while (start < remaining.length()) {
            char c = remaining[start];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                start++;
            } else if (c == ';') {
                // Skip comment to end of line
                while (start < remaining.length() && remaining[start] != '\n') {
                    start++;
                }
            } else {
                break;
            }
        }
        remaining = remaining.substr(start);

        if (remaining.empty()) break;

        // Find the end of this expression
        int depth = 0;
        bool in_string = false;
        size_t end = 0;

        for (size_t i = 0; i < remaining.length(); ++i) {
            char c = remaining[i];
            if (c == '"' && (i == 0 || remaining[i-1] != '\\')) {
                in_string = !in_string;
            }
            if (!in_string) {
                if (c == '(') depth++;
                else if (c == ')') {
                    depth--;
                    if (depth == 0) {
                        end = i + 1;
                        break;
                    }
                }
            }
        }

        if (end == 0) break;

        std::string expr = remaining.substr(0, end);
        remaining = remaining.substr(end);

        try {
            eshkol_ast_t ast = parse_string(expr);
            if (ast.type == ESHKOL_INVALID) {
                print_error("Failed to parse expression in file");
                continue;
            }

            // Check for definitions that should be skipped (reload scenario)
            const char* defined_name = get_defined_name(ast);
            if (defined_name && repl_ctx.isSymbolDefined(defined_name)) {
                // Symbol already exists - skip to avoid duplicate definition error
                eshkol_ast_clean(&ast);
                expr_count++;
                continue;
            }

            // Register lambda variables for cross-reference support
            const char* lambda_var = get_lambda_var_name(ast);
            if (lambda_var) {
                repl_ctx.registerLambdaVar(lambda_var);
            }

            // Track defined symbols for :env display
            if (defined_name) {
                g_defined_symbols.push_back(defined_name);
            }

            eshkol_ast_t* ast_to_execute = &ast;
            bool should_display = !is_definition_statement(ast);

            if (should_display) {
                ast_to_execute = eshkol_wrap_with_display(&ast);
            }

            void* result = repl_ctx.execute(ast_to_execute);

            if (should_display && ast_to_execute != &ast) {
                free(ast_to_execute->operation.call_op.func);
                free(ast_to_execute->operation.call_op.variables[0].operation.call_op.func);
                free(ast_to_execute->operation.call_op.variables[0].operation.call_op.variables);
                free(ast_to_execute->operation.call_op.variables[1].operation.call_op.func);
                free(ast_to_execute->operation.call_op.variables);
                free(ast_to_execute);
            }

            eshkol_ast_clean(&ast);
            if (result) {
                delete static_cast<int64_t*>(result);
            }

            expr_count++;

        } catch (const std::exception& e) {
            print_error("Execution error", e.what());
        }
    }

    g_last_loaded_file = filename;
    print_success("Loaded " + std::to_string(expr_count) + " expressions from " + filename);
    return true;
}

// Handle REPL commands (return true if command was handled)
bool handle_command(const std::string& input, eshkol::ReplJITContext& repl_ctx) {
    // Trim input
    std::string cmd = input;
    while (!cmd.empty() && (cmd.back() == ' ' || cmd.back() == '\t')) {
        cmd.pop_back();
    }

    if (cmd == ":help" || cmd == ":h") {
        print_help();
        return true;
    }

    if (cmd == ":quit" || cmd == ":q" || cmd == "(exit)" || cmd == "exit") {
        std::cout << color::dim() << "Goodbye!" << color::reset() << "\n";
        save_readline_history();
        _exit(0);
    }

    if (cmd == ":clear") {
        clear_screen();
        return true;
    }

    if (cmd == ":env" || cmd == ":e") {
        print_environment();
        return true;
    }

    if (cmd == ":examples") {
        print_examples();
        return true;
    }

    if (cmd == ":version" || cmd == ":v") {
        std::cout << color::bold() << "Eshkol" << color::reset() << " version "
                  << color::bright_cyan() << "0.1.1" << color::reset() << "\n";
        std::cout << color::dim() << "A Scheme dialect with automatic differentiation" << color::reset() << "\n";
        return true;
    }

    if (cmd == ":reload" || cmd == ":r") {
        if (g_last_loaded_file.empty()) {
            print_error("No file has been loaded yet");
        } else {
            load_file(g_last_loaded_file, repl_ctx);
        }
        return true;
    }

    if (cmd == ":stdlib") {
        std::cout << color::dim() << "Loading standard library..." << color::reset() << std::flush;
        if (repl_ctx.loadStdlib()) {
            std::cout << color::dim() << " done" << color::reset() << "\n";
            print_success("Standard library loaded. Functions available: length, filter, fold, map, etc.");
        } else {
            std::cout << color::error() << " failed" << color::reset() << "\n";
        }
        return true;
    }

    if (cmd == ":reset") {
        g_defined_symbols.clear();
        print_info("REPL state cleared (note: JIT symbols persist until restart)");
        return true;
    }

    // :type command - show type info
    if (cmd == ":type" || cmd == ":t") {
        print_info("Usage: :type <expression>");
        return true;
    }
    if (cmd.substr(0, 6) == ":type " || cmd.substr(0, 3) == ":t ") {
        size_t space_pos = cmd.find(' ');
        if (space_pos != std::string::npos) {
            std::string expr = cmd.substr(space_pos + 1);
            while (!expr.empty() && expr.front() == ' ') expr.erase(0, 1);

            if (!expr.empty()) {
                try {
                    eshkol_ast_t ast = parse_string(expr);
                    if (ast.type != ESHKOL_INVALID) {
                        std::string type_str = get_ast_type_string(&ast);
                        std::cout << color::type() << "Type: " << color::reset();
                        std::cout << color::bright_cyan() << type_str << color::reset() << "\n";
                        eshkol_ast_clean(&ast);
                    } else {
                        print_error("Could not parse expression");
                    }
                } catch (const std::exception& e) {
                    print_error("Parse error", e.what());
                }
            }
        }
        return true;
    }

    // :doc command - show documentation
    if (cmd == ":doc" || cmd == ":d") {
        print_doc_topics();
        return true;
    }
    if (cmd.substr(0, 5) == ":doc " || cmd.substr(0, 3) == ":d ") {
        size_t space_pos = cmd.find(' ');
        if (space_pos != std::string::npos) {
            std::string name = cmd.substr(space_pos + 1);
            while (!name.empty() && name.front() == ' ') name.erase(0, 1);
            while (!name.empty() && name.back() == ' ') name.pop_back();

            if (!name.empty()) {
                print_doc(name);
            } else {
                print_doc_topics();
            }
        }
        return true;
    }

    // :ast command - show AST structure
    if (cmd == ":ast") {
        print_info("Usage: :ast <expression>");
        return true;
    }
    if (cmd.substr(0, 5) == ":ast ") {
        std::string expr = cmd.substr(5);
        while (!expr.empty() && expr.front() == ' ') expr.erase(0, 1);

        if (!expr.empty()) {
            try {
                eshkol_ast_t ast = parse_string(expr);
                if (ast.type != ESHKOL_INVALID) {
                    std::cout << color::dim() << "AST Structure:" << color::reset() << "\n";
                    eshkol_ast_pretty_print(&ast, 0);
                    eshkol_ast_clean(&ast);
                } else {
                    print_error("Could not parse expression");
                }
            } catch (const std::exception& e) {
                print_error("Parse error", e.what());
            }
        }
        return true;
    }

    if (cmd == ":load" || cmd == ":l") {
        print_info("Usage: :load <filename>");
        return true;
    }
    if (cmd.substr(0, 6) == ":load " || cmd.substr(0, 3) == ":l ") {
        size_t space_pos = cmd.find(' ');
        if (space_pos != std::string::npos) {
            std::string filename = cmd.substr(space_pos + 1);
            // Trim filename
            while (!filename.empty() && filename.front() == ' ') filename.erase(0, 1);
            while (!filename.empty() && filename.back() == ' ') filename.pop_back();
            load_file(filename, repl_ctx);
        }
        return true;
    }

    if (cmd == ":time") {
        print_info("Usage: :time <expression>");
        return true;
    }
    if (cmd.substr(0, 6) == ":time ") {
        std::string expr = cmd.substr(6);
        while (!expr.empty() && expr.front() == ' ') expr.erase(0, 1);

        if (!expr.empty()) {
            auto total_start = std::chrono::high_resolution_clock::now();

            try {
                // Time parsing
                auto parse_start = std::chrono::high_resolution_clock::now();
                eshkol_ast_t ast = parse_string(expr);
                auto parse_end = std::chrono::high_resolution_clock::now();
                auto parse_time = std::chrono::duration_cast<std::chrono::microseconds>(parse_end - parse_start);

                if (ast.type != ESHKOL_INVALID) {
                    eshkol_ast_t* ast_to_execute = &ast;
                    bool should_display = !is_definition_statement(ast);

                    if (should_display) {
                        ast_to_execute = eshkol_wrap_with_display(&ast);
                    }

                    // Time JIT compilation + execution
                    auto exec_start = std::chrono::high_resolution_clock::now();
                    void* result = repl_ctx.execute(ast_to_execute);
                    auto exec_end = std::chrono::high_resolution_clock::now();

                    auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(exec_end - exec_start);
                    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(exec_end - total_start);

                    // Display timing breakdown
                    std::cout << color::dim() << "─── Timing ───" << color::reset() << "\n";
                    std::cout << color::dim() << "  Parse:   " << color::reset()
                              << color::bright_cyan() << std::setw(8) << parse_time.count() << color::reset()
                              << color::dim() << " μs" << color::reset() << "\n";
                    std::cout << color::dim() << "  JIT+Run: " << color::reset()
                              << color::bright_cyan() << std::setw(8) << exec_time.count() << color::reset()
                              << color::dim() << " μs" << color::reset() << "\n";
                    std::cout << color::dim() << "  Total:   " << color::reset()
                              << color::bright_yellow() << std::setw(8) << total_time.count() << color::reset()
                              << color::dim() << " μs" << color::reset() << "\n";
                    std::cout << color::dim() << "Note: JIT compilation dominates for simple expressions" << color::reset() << "\n";

                    if (should_display && ast_to_execute != &ast) {
                        free(ast_to_execute->operation.call_op.func);
                        free(ast_to_execute->operation.call_op.variables[0].operation.call_op.func);
                        free(ast_to_execute->operation.call_op.variables[0].operation.call_op.variables);
                        free(ast_to_execute->operation.call_op.variables[1].operation.call_op.func);
                        free(ast_to_execute->operation.call_op.variables);
                        free(ast_to_execute);
                    }

                    eshkol_ast_clean(&ast);
                    if (result) delete static_cast<int64_t*>(result);
                }
            } catch (const std::exception& e) {
                print_error("Execution error", e.what());
            }
        }
        return true;
    }

    if (cmd == ":history") {
        #ifdef HAVE_READLINE
        // Display history using history_get
        int hist_len = history_length;
        if (hist_len > 0) {
            std::cout << color::dim() << "Command History:" << color::reset() << "\n";
            int start = (hist_len > 20) ? hist_len - 20 : 1;
            for (int i = start; i <= hist_len; i++) {
                HIST_ENTRY *entry = history_get(i);
                if (entry && entry->line) {
                    std::cout << "  " << color::dim() << std::setw(4) << i << color::reset()
                              << "  " << entry->line << "\n";
                }
            }
        } else {
            print_info("No history yet");
        }
        #else
        print_info("History not available (readline not installed)");
        #endif
        return true;
    }

    // Catch-all for unknown commands starting with :
    if (!cmd.empty() && cmd[0] == ':') {
        print_error("Unknown command: " + cmd, "Type :help for available commands");
        return true;
    }

    return false;
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    bool load_stdlib = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--stdlib" || arg == "-s") {
            load_stdlib = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: eshkol-repl [OPTIONS]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --stdlib, -s    Load standard library on startup\n";
            std::cout << "  --help, -h      Show this help message\n";
            return 0;
        }
    }

    // Check if running interactively
    g_interactive = isatty(STDIN_FILENO);

    // Install signal handlers
    signal(SIGINT, sigint_handler);
    signal(SIGSEGV, crash_handler);
    signal(SIGFPE, crash_handler);
    signal(SIGBUS, crash_handler);

    // Initialize readline with completion and history (only if interactive)
    if (g_interactive) {
        init_readline();
        // Print welcome banner only in interactive mode
        print_welcome_banner();
    }

    // Initialize REPL JIT context
    eshkol::ReplJITContext repl_ctx;

    // Load stdlib if requested
    if (load_stdlib) {
        if (g_interactive) {
            std::cout << color::dim() << "Loading standard library..." << color::reset() << std::flush;
        }
        if (repl_ctx.loadStdlib()) {
            if (g_interactive) {
                std::cout << color::dim() << " done" << color::reset() << "\n";
            }
        } else {
            if (g_interactive) {
                std::cout << color::error() << " failed" << color::reset() << "\n";
            }
        }
    }

    std::cout << "\n";

    while (true) {
        g_interrupted = 0;

        std::string input_str;
        bool first_line = true;
        int line_count = 0;

        while (true) {
            if (g_interrupted) {
                std::cout << "\n" << color::dim() << "^C (cancelled)" << color::reset() << "\n";
                g_interrupted = 0;
                input_str.clear();
                break;
            }

            int current_depth = first_line ? 0 : get_paren_depth(input_str);

            std::string prompt_str = make_prompt(!first_line, line_count, current_depth);

            char* input = eshkol_readline(prompt_str.c_str());

            // EOF (Ctrl+D)
            if (!input) {
                if (!first_line && !input_str.empty()) {
                    std::cout << color::dim() << " (force completing)" << color::reset() << "\n";
                    break;
                }
                std::cout << "\n" << color::dim() << "Goodbye!" << color::reset() << "\n";
                save_readline_history();
                _exit(0);
            }

            // Empty continuation line - remove last line or cancel
            if (!first_line && strlen(input) == 0 && !input_str.empty()) {
                size_t last_newline = input_str.rfind('\n');
                if (last_newline != std::string::npos) {
                    input_str = input_str.substr(0, last_newline);
                    line_count--;
                    std::cout << color::dim() << "(removed last line, "
                              << line_count << " line" << (line_count != 1 ? "s" : "")
                              << " remaining)" << color::reset() << "\n";
                    free(input);
                    continue;
                } else {
                    std::cout << color::dim() << "(cancelled)" << color::reset() << "\n";
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

            // Check for REPL commands on first line
            if (first_line && input[0] == ':') {
                add_history(input);
                free(input);
                if (handle_command(input_str, repl_ctx)) {
                    input_str.clear();
                    break;
                }
            }

            // Check for exit
            if (first_line && (strcmp(input, "(exit)") == 0 ||
                              strcmp(input, "exit") == 0 ||
                              strcmp(input, "quit") == 0)) {
                free(input);
                std::cout << color::dim() << "Goodbye!" << color::reset() << "\n";
                save_readline_history();
                _exit(0);
            }

            // Check for :cancel on any line
            if (strcmp(input, ":cancel") == 0 || strcmp(input, ":c") == 0) {
                free(input);
                std::cout << color::dim() << "(cancelled)" << color::reset() << "\n";
                input_str.clear();
                break;
            }

            free(input);

            // Empty first line - just continue
            if (first_line && input_str.empty()) {
                input_str.clear();
                break;
            }

            // Check parenthesis balance
            int depth = get_paren_depth(input_str);

            if (depth < 0) {
                print_error("Unmatched closing parenthesis");
                input_str.clear();
                break;
            }

            if (depth == 0) {
                add_history(input_str.c_str());
                break;
            }

            first_line = false;
        }

        if (input_str.empty()) {
            continue;
        }

        // Check if this is a command
        if (input_str[0] == ':') {
            handle_command(input_str, repl_ctx);
            continue;
        }

        // Parse and evaluate
        try {
            eshkol_ast_t ast = parse_string(input_str);

            if (ast.type == ESHKOL_INVALID) {
                print_error("Failed to parse input");
                continue;
            }

            // Track defined symbols
            const char* defined_name = get_defined_name(ast);
            if (defined_name) {
                // Check if already defined, if so, don't add again
                bool found = false;
                for (const auto& sym : g_defined_symbols) {
                    if (sym == defined_name) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    g_defined_symbols.push_back(defined_name);
                }
            }

            // Check if this defines a lambda variable
            const char* lambda_var = get_lambda_var_name(ast);
            if (lambda_var) {
                repl_ctx.registerLambdaVar(lambda_var);
            }

            // Wrap expressions with display
            eshkol_ast_t* ast_to_execute = &ast;
            bool should_display = !is_definition_statement(ast);

            if (should_display) {
                ast_to_execute = eshkol_wrap_with_display(&ast);
            }

            // Execute using JIT with crash recovery and exception handling
            void* result = nullptr;
            bool had_error = false;

            // Outer layer: catch crashes (SIGSEGV, SIGFPE, etc.)
            g_in_jit = 1;
            if (sigsetjmp(g_crash_jmp_buf, 1) == 0) {
                // Inner layer: catch Eshkol exceptions
                eshkol_push_exception_handler(&g_repl_exception_jmp_buf);

                if (setjmp(g_repl_exception_jmp_buf) == 0) {
                    // Normal execution path
                    result = repl_ctx.execute(ast_to_execute);
                } else {
                    // Exception was raised - handle it
                    had_error = true;
                    eshkol_exception_t* exc = g_current_exception;
                    if (exc) {
                        display_exception(exc);
                    } else {
                        print_error("Unknown runtime error");
                    }
                }

                eshkol_pop_exception_handler();
            } else {
                // Crash occurred - display error and continue
                had_error = true;
                print_error("Runtime error", crash_signal_message(g_crash_signal));
                // Re-install signal handler (it was reset to default)
                signal(g_crash_signal, crash_handler);
            }
            g_in_jit = 0;

            // Clean up wrapper
            if (should_display && ast_to_execute != &ast) {
                free(ast_to_execute->operation.call_op.func);
                free(ast_to_execute->operation.call_op.variables[0].operation.call_op.func);
                free(ast_to_execute->operation.call_op.variables[0].operation.call_op.variables);
                free(ast_to_execute->operation.call_op.variables[1].operation.call_op.func);
                free(ast_to_execute->operation.call_op.variables);
                free(ast_to_execute);
            }

            // Clean up
            eshkol_ast_clean(&ast);
            if (result && !had_error) {
                delete static_cast<int64_t*>(result);
            }

        } catch (const std::exception& e) {
            print_error("Execution error", e.what());
        }
    }

    save_readline_history();
    _exit(0);
}
