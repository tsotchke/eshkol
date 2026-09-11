//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//
// Eshkol Interactive REPL - A visual live coding experience
//

#include <eshkol/eshkol.h>
#include <eshkol/platform_runtime.h>
#include <eshkol/core/runtime.h>
#include <eshkol/backend/thread_pool.h>
#include "../lib/repl/repl_jit.h"
#include "../lib/repl/repl_utils.h"

#include <eshkol/core/introspection.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <csignal>
#include <chrono>
#include <iomanip>
#include <setjmp.h>

#ifdef _WIN32
#include <process.h>
#include <io.h>
#define ESHKOL_GETPID() _getpid()
#define ESHKOL_FILENO(f) _fileno(f)
#define ESHKOL_DUP(fd) _dup(fd)
#define ESHKOL_DUP2(oldfd, newfd) _dup2(oldfd, newfd)
#define ESHKOL_CLOSE_FD(fd) _close(fd)
#else
#include <unistd.h>
#define ESHKOL_GETPID() getpid()
#define ESHKOL_FILENO(f) fileno(f)
#define ESHKOL_DUP(fd) dup(fd)
#define ESHKOL_DUP2(oldfd, newfd) dup2(oldfd, newfd)
#define ESHKOL_CLOSE_FD(fd) close(fd)
#endif

using namespace eshkol::repl;

/* ---------------------------------------------------------------------------
 * LeakSanitizer visibility for the REPL (ADR-0010 gap A12).
 *
 * repl_clean_exit() ends the process with std::_Exit(), deliberately: it must
 * not run atexit handlers or static destructors while JIT worker threads may
 * still hold libsystem locks. But LeakSanitizer installs its whole-process
 * leak check AS an atexit handler, so _Exit() skips it — and the REPL, the one
 * genuinely long-lived process this project ships, was therefore structurally
 * invisible to leak detection. A REPL that leaked a megabyte per input line
 * would have produced exactly the output an entirely clean one does: none.
 *
 * Under ASan/LSan we therefore run the leak check EXPLICITLY, at the last
 * point where the process state is still the one we want to audit — after the
 * ordered teardown above, before _Exit tears the process down. Outside a
 * sanitizer build this compiles to nothing.
 * ------------------------------------------------------------------------- */
#ifndef ESHKOL_HAS_ASAN
# if defined(__SANITIZE_ADDRESS__)
#  define ESHKOL_HAS_ASAN 1
# elif defined(__clang__)
#  if defined(__has_feature)
#   if __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
#    define ESHKOL_HAS_ASAN 1
#   endif
#  endif
# endif
#endif

#ifdef ESHKOL_HAS_ASAN
#include <sanitizer/lsan_interface.h>
#endif

// Jump buffer for exception handling during JIT execution
static jmp_buf g_repl_exception_jmp_buf;

// Jump buffer for signal-based crash recovery (segfaults, etc.)
#ifdef _WIN32
using crash_jmp_buf_t = jmp_buf;
#define ESHKOL_SIGSETJMP(env) setjmp(env)
#define ESHKOL_SIGLONGJMP(env, value) longjmp(env, value)
#else
using crash_jmp_buf_t = sigjmp_buf;
#define ESHKOL_SIGSETJMP(env) sigsetjmp(env, 1)
#define ESHKOL_SIGLONGJMP(env, value) siglongjmp(env, value)
#endif

static crash_jmp_buf_t g_crash_jmp_buf;
static volatile sig_atomic_t g_in_jit = 0;
static volatile sig_atomic_t g_crash_signal = 0;

// Global flag for Ctrl+C handling
volatile sig_atomic_t g_interrupted = 0;

// Global state for REPL
static std::string g_last_loaded_file;
static std::vector<std::string> g_defined_symbols;

// Signal handler for Ctrl+C.
//
// Outside a JIT-executing form this is the original behavior: set a flag the
// read loop notices between lines. While a form IS executing (g_in_jit, the
// same flag crash_handler below already uses), Ctrl+C -- or, in --machine
// mode, an EREPL client sending SIGINT to abort a hung evaluation -- instead
// aborts the evaluation right away via the same longjmp path crash_handler
// uses, so a runaway `(let loop () (loop))` can be interrupted without
// killing the process. g_crash_signal is left as SIGINT so both the legacy
// crash-recovery message and the machine-mode structured error can tell an
// interrupt apart from an actual crash.
void sigint_handler(int sig) {
    (void)sig;
    if (g_in_jit) {
        g_crash_signal = SIGINT;
        ESHKOL_SIGLONGJMP(g_crash_jmp_buf, 1);
    }
    g_interrupted = 1;
}

// Forward decl — defined later, but repl_clean_exit references it.
void save_readline_history();

// Ordered teardown before std::_Exit. Joins JIT thread-pool workers and runs
// runtime shutdown hooks so we don't terminate the process while worker
// threads still hold libsystem locks (which surfaces as a spurious
// "Abort trap: 6" on macOS even though the user's program completed
// normally).  Called from each :quit / EOF / (exit) path.
[[noreturn]] static void repl_clean_exit(int code) {
    save_readline_history();
    std::fflush(stdout);
    std::fflush(stderr);
    thread_pool_global_shutdown();
    eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_NONE);
#ifdef ESHKOL_HAS_ASAN
    /* _Exit() below skips LSan's atexit leak check; run it here so the REPL
     * is auditable at all. __lsan_do_leak_check() honours ASAN_OPTIONS'
     * detect_leaks (it is a no-op when leak detection is off) and terminates
     * the process itself with the configured exitcode if it finds leaks, so
     * the _Exit(code) below is reached only on a clean check. */
    __lsan_do_leak_check();
#endif
    std::_Exit(code);
}

// Signal handler for crashes during JIT execution (SIGSEGV, SIGFPE, SIGBUS)
void crash_handler(int sig) {
    if (g_in_jit) {
        g_crash_signal = sig;
        ESHKOL_SIGLONGJMP(g_crash_jmp_buf, 1);
    } else {
        signal(sig, SIG_DFL);
        raise(sig);
    }
}

// Get human-readable message for crash signal
const char* crash_signal_message(int sig) {
    switch (sig) {
        case SIGSEGV: return "Segmentation fault - likely a type error (e.g., arithmetic on non-numeric value)";
        case SIGFPE:  return "Floating point exception - likely division by zero";
#ifdef SIGBUS
        case SIGBUS:  return "Bus error - memory access issue";
#endif
        case SIGINT:  return "Evaluation interrupted";
        default:      return "Unknown runtime error";
    }
}

// Stable, wording-independent name for an exception's category. Shared by
// the human-readable display_exception() below and the machine-mode
// structured error payload (EREPL_JSON_HELPERS), so both agree on the exact
// same closed set of `kind` strings -- classification never depends on the
// prose in `message`.
static const char* exception_type_name(eshkol_exception_type_t type) {
    switch (type) {
        case ESHKOL_EXCEPTION_ERROR: return "error";
        case ESHKOL_EXCEPTION_TYPE_ERROR: return "type-error";
        case ESHKOL_EXCEPTION_FILE_ERROR: return "file-error";
        case ESHKOL_EXCEPTION_READ_ERROR: return "read-error";
        case ESHKOL_EXCEPTION_SYNTAX_ERROR: return "syntax-error";
        case ESHKOL_EXCEPTION_RANGE_ERROR: return "range-error";
        case ESHKOL_EXCEPTION_ARITY_ERROR: return "arity-error";
        case ESHKOL_EXCEPTION_DIVIDE_BY_ZERO: return "divide-by-zero";
        case ESHKOL_EXCEPTION_USER_DEFINED: return "user-exception";
    }
    return "error";
}

// Display an exception with nice formatting
void display_exception(eshkol_exception_t* exc) {
    using namespace color;

    const char* type_name = exception_type_name(exc->type);

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

// Helper: Parse a string to AST using istringstream (no temp file needed).
// Resets the parser's cumulative line counter so each REPL form starts at
// line 1 — otherwise diagnostics would creep forward across commands.
eshkol_ast_t parse_string(const std::string& input) {
    std::string parse_input = input + "\n";
    std::istringstream stream(parse_input);
    eshkol_reset_parse_line_counter();
    return eshkol_parse_next_ast_from_stream(stream);
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
        repl_clean_exit(0);
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
                  << color::bright_cyan() << ESHKOL_VERSION_STRING << color::reset() << "\n";
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

// --machine mode framing markers (Noesis warm-worker support, 2026-05-07;
// versioned as EREPL protocol v1, 2026-09-10 -- see the block just below
// and docs/reference/runtime/eshkol-repl.md for the full contract).
// Sentinels go to STDERR so user program output on stdout stays clean.
// Clients drive eshkol-repl as a long-running JIT-warm worker:
//   1. Spawn `eshkol-repl --machine`
//   2. Read stderr until "EREPL READY" — JIT + stdlib are warm. v1 clients
//      then read one more stderr line, `EREPL/1 {"type":"ready",...}`,
//      which announces protocol_version/pid/eshkol_version.
//   3a. Legacy: send a bare form on stdin (newline + balanced parens).
//       Read stdout until "EREPL DONE" / "EREPL FAIL" appears on stderr.
//   3b. v1: send one JSON line on stdin, e.g.
//       {"id":"1","op":"eval","code":"(+ 1 2)"}. Read stdout for whatever
//       the form itself wrote, and stderr for the matching
//       `EREPL/1 {"type":"result","id":"1",...}` frame (preceded, for
//       op=eval, by the same bare DONE/FAIL line legacy clients watch).
//   4. Repeat for each new form/request, never paying the cold-start cost
//      again. op=shutdown ends the session cleanly.
// EREPL_READY emits exactly once after init; EREPL_DONE / EREPL_FAIL emits
// once per evaluated form (legacy or op=eval). Both end with \n and an
// explicit fflush, as does every EREPL/1 frame.
static constexpr const char* EREPL_READY = "EREPL READY\n";
static constexpr const char* EREPL_DONE  = "EREPL DONE\n";
static constexpr const char* EREPL_FAIL  = "EREPL FAIL\n";

// =============================================================================
// EREPL v1 -- machine-mode JSON request/response protocol.
//
// See docs/reference/runtime/eshkol-repl.md ("Machine mode (EREPL protocol)")
// for the full contract. Summary: a --machine session still emits the
// original bare `EREPL READY` / `EREPL DONE` / `EREPL FAIL` lines on stderr
// unchanged (nothing that watched only those breaks), but now additionally
// accepts JSON request lines on stdin (any line whose first non-whitespace
// character is '{' -- no Eshkol source form starts with '{', so this can
// never collide with a bare Scheme form) and answers with one JSON line per
// response, tagged `EREPL/1 ` on stderr. stdout carries only bytes the
// evaluated program itself wrote (explicit display/write/print calls) --
// never protocol framing and never an auto-echoed result -- so a driver
// never has to guess which stdout bytes are "the answer" versus program
// output; the answer is always the structured frame's `value` field.
// =============================================================================

static constexpr int EREPL_PROTOCOL_VERSION = 1;

// ---- JSON string escaping/parsing -----------------------------------------
//
// Hand-rolled rather than pulling in a JSON dependency: every EREPL v1 frame
// this file emits or accepts is a flat object whose values are plain JSON
// strings (or, for a handful of response fields, numbers/booleans/null the
// code below writes directly) -- there is no nesting deep enough, and no
// non-string request field, to justify a general parser.

// Escapes `s` for use as a JSON string body (the caller supplies the quotes).
// Raw UTF-8 bytes above 0x7F are passed through unchanged -- valid inside a
// JSON string and cheaper than re-encoding through \u escapes.
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// Parses one JSON string literal starting at s[i] == '"' (decoding escapes,
// including \uXXXX and UTF-16 surrogate pairs into UTF-8). On success leaves
// i just past the closing quote. Used only by json_parse_flat_object below.
static bool json_parse_string(const std::string& s, size_t& i, std::string& out) {
    if (i >= s.size() || s[i] != '"') return false;
    ++i;
    out.clear();
    auto hex4 = [&](size_t pos, unsigned& v) -> bool {
        if (pos + 4 > s.size()) return false;
        v = 0;
        for (int k = 0; k < 4; ++k) {
            char h = s[pos + k];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') v |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= static_cast<unsigned>(h - 'A' + 10);
            else return false;
        }
        return true;
    };
    auto append_utf8 = [&](unsigned cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    };
    while (i < s.size()) {
        char c = s[i];
        if (c == '"') { ++i; return true; }
        if (c == '\\') {
            ++i;
            if (i >= s.size()) return false;
            char e = s[i];
            switch (e) {
                case '"':  out += '"';  ++i; break;
                case '\\': out += '\\'; ++i; break;
                case '/':  out += '/';  ++i; break;
                case 'b':  out += '\b'; ++i; break;
                case 'f':  out += '\f'; ++i; break;
                case 'n':  out += '\n'; ++i; break;
                case 'r':  out += '\r'; ++i; break;
                case 't':  out += '\t'; ++i; break;
                case 'u': {
                    unsigned cp = 0;
                    if (!hex4(i + 1, cp)) return false;
                    i += 5;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < s.size() &&
                        s[i] == '\\' && s[i + 1] == 'u') {
                        unsigned lo = 0;
                        if (hex4(i + 2, lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            i += 6;
                        }
                    }
                    append_utf8(cp);
                    break;
                }
                default:
                    return false;
            }
        } else {
            out += c;
            ++i;
        }
    }
    return false; // unterminated string
}

// Parses a flat JSON object `{"k":"v", ...}` -- every value must itself be a
// JSON string, which covers every field EREPL v1 requests use (id, op,
// code, prefix). A non-string value is a malformed request, not something
// to silently coerce.
static bool json_parse_flat_object(const std::string& s,
                                    std::unordered_map<std::string, std::string>& out,
                                    std::string& err) {
    size_t i = 0;
    auto skip_ws = [&]() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; };
    skip_ws();
    if (i >= s.size() || s[i] != '{') { err = "expected '{'"; return false; }
    ++i;
    skip_ws();
    if (i < s.size() && s[i] == '}') { ++i; return true; }
    while (true) {
        skip_ws();
        std::string key;
        if (!json_parse_string(s, i, key)) { err = "expected a string key"; return false; }
        skip_ws();
        if (i >= s.size() || s[i] != ':') { err = "expected ':' after \"" + key + "\""; return false; }
        ++i;
        skip_ws();
        if (i >= s.size() || s[i] != '"') { err = "expected a string value for \"" + key + "\""; return false; }
        std::string value;
        if (!json_parse_string(s, i, value)) { err = "malformed string value for \"" + key + "\""; return false; }
        out[key] = value;
        skip_ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == '}') { ++i; return true; }
        err = "expected ',' or '}'";
        return false;
    }
}

// ---- Result-value capture ---------------------------------------------------

// Portable in-memory capture of a tagged value's R7RS `write` representation
// (strings quoted, #t/#f, etc.) -- the machine-mode `value` field always
// uses `write` form so a driver never has to guess whether a bare word came
// back as a string or a symbol.
static std::string capture_written_value(const eshkol_tagged_value_t& tv) {
#ifdef _WIN32
    FILE* fp = std::tmpfile();
    if (!fp) return std::string();
    eshkol_write_value_to_port(&tv, fp);
    std::fflush(fp);
    long len = std::ftell(fp);
    if (len <= 0) { std::fclose(fp); return std::string(); }
    std::rewind(fp);
    std::string out(static_cast<size_t>(len), '\0');
    size_t got = std::fread(&out[0], 1, static_cast<size_t>(len), fp);
    out.resize(got);
    std::fclose(fp);
    return out;
#else
    char* buf = nullptr;
    size_t size = 0;
    FILE* fp = open_memstream(&buf, &size);
    if (!fp) return std::string();
    eshkol_write_value_to_port(&tv, fp);
    std::fflush(fp);
    std::string out(buf, size);
    std::fclose(fp);
    std::free(buf);
    return out;
#endif
}

// Coarse type name for the machine-mode `value_type` field, via the same
// `type-of` classification exposed to user code -- so the field is never a
// wording guess, it is a name the runtime already commits to elsewhere.
static std::string describe_value_type(const eshkol_tagged_value_t& tv) {
    eshkol_tagged_value_t name = eshkol_type_of(tv);
    std::string s = capture_written_value(name);
    return s.empty() ? std::string("unknown") : s;
}

// ---- stdout capture for a JSON eval request ---------------------------------
//
// The legacy bare-form path leaves stdout exactly as it always was: the
// program's own output streams to the real pipe in real time, and a client
// has to read it and separately watch stderr for DONE/FAIL. That works for
// a human but not for a driver -- two independent OS pipes (this process's
// stdout and its stderr) give a reader no ordering guarantee between "the
// stdout bytes are readable" and "the stderr response frame is readable",
// even though this process always writes stdout before flushing the
// response frame. Rather than ask every driver to race two pipes (and get
// it right on every platform), a JSON eval's own stdout is captured here
// and embedded verbatim in its response frame as the single source of
// truth; the same bytes are then replayed to the real stdout once capture
// ends, so a plain pipe-tailing consumer still sees them -- just after the
// form finishes rather than incrementally while it runs.
class StdoutCapture {
public:
    StdoutCapture() {
        std::fflush(stdout);
        capture_file_ = std::tmpfile();
        if (!capture_file_) return;
        int cap_fd = ESHKOL_FILENO(capture_file_);
        saved_fd_ = ESHKOL_DUP(ESHKOL_FILENO(stdout));
        if (saved_fd_ == -1) return;
        if (ESHKOL_DUP2(cap_fd, ESHKOL_FILENO(stdout)) == -1) {
            ESHKOL_CLOSE_FD(saved_fd_);
            saved_fd_ = -1;
            return;
        }
        active_ = true;
    }

    ~StdoutCapture() {
        if (active_) finish();
        if (capture_file_) std::fclose(capture_file_);
    }

    StdoutCapture(const StdoutCapture&) = delete;
    StdoutCapture& operator=(const StdoutCapture&) = delete;

    // Ends capture, restores the real stdout, replays the captured bytes to
    // it, and returns them. Idempotent: a second call returns "".
    std::string finish() {
        if (!active_) return std::string();
        active_ = false;
        std::fflush(stdout);
        ESHKOL_DUP2(saved_fd_, ESHKOL_FILENO(stdout));
        ESHKOL_CLOSE_FD(saved_fd_);
        saved_fd_ = -1;

        std::string out;
        std::fflush(capture_file_);
        long len = std::ftell(capture_file_);
        if (len > 0) {
            std::rewind(capture_file_);
            out.resize(static_cast<size_t>(len));
            size_t got = std::fread(&out[0], 1, static_cast<size_t>(len), capture_file_);
            out.resize(got);
        }
        if (!out.empty()) {
            std::fwrite(out.data(), 1, out.size(), stdout);
            std::fflush(stdout);
        }
        return out;
    }

private:
    FILE* capture_file_ = nullptr;
    int saved_fd_ = -1;
    bool active_ = false;
};

// ---- Frame emission ---------------------------------------------------------

static void emit_frame(const std::string& body) {
    std::fputs("EREPL/1 ", stderr);
    std::fputs(body.c_str(), stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

// Bare requests, or a request whose "id" field is missing/empty, echo back
// JSON null -- a driver that cares about pairing responses always sends a
// non-empty id, so null unambiguously means "none was supplied".
static std::string json_id_field(const std::string& id) {
    if (id.empty()) return "null";
    return "\"" + json_escape(id) + "\"";
}

static std::string build_error_object(const std::string& kind,
                                       const std::string& message,
                                       long line, long column,
                                       const std::string* filename,
                                       const std::string& printed,
                                       const std::vector<std::string>& irritants) {
    std::ostringstream o;
    o << "{\"kind\":\"" << json_escape(kind) << "\""
      << ",\"message\":\"" << json_escape(message) << "\""
      << ",\"line\":" << (line > 0 ? std::to_string(line) : std::string("null"))
      << ",\"column\":" << (column > 0 ? std::to_string(column) : std::string("null"))
      << ",\"filename\":" << (filename ? ("\"" + json_escape(*filename) + "\"") : std::string("null"))
      << ",\"printed\":\"" << json_escape(printed) << "\""
      << ",\"irritants\":[";
    for (size_t k = 0; k < irritants.size(); ++k) {
        if (k) o << ",";
        o << "\"" << json_escape(irritants[k]) << "\"";
    }
    o << "]}";
    return o.str();
}

// `stdout_text` is this call's captured output (see StdoutCapture above) --
// embedded directly in the frame so a driver never has to correlate it
// against the raw stdout pipe. Non-eval callers (protocol errors, requests
// that never reached execution) always pass "".
static void emit_result_ok(const std::string& id, const eshkol_tagged_value_t& tv,
                            const std::string& stdout_text) {
    std::string value = capture_written_value(tv);
    std::string vtype = describe_value_type(tv);
    std::ostringstream body;
    body << "{\"type\":\"result\",\"id\":" << json_id_field(id)
         << ",\"ok\":true,\"stdout\":\"" << json_escape(stdout_text)
         << "\",\"value\":\"" << json_escape(value)
         << "\",\"value_type\":\"" << json_escape(vtype) << "\"}";
    emit_frame(body.str());
}

static void emit_result_error(const std::string& id, const std::string& error_obj,
                               const std::string& stdout_text) {
    std::ostringstream body;
    body << "{\"type\":\"result\",\"id\":" << json_id_field(id)
         << ",\"ok\":false,\"stdout\":\"" << json_escape(stdout_text)
         << "\",\"error\":" << error_obj << "}";
    emit_frame(body.str());
}

static void emit_result_error_simple(const std::string& id, const std::string& kind,
                                      const std::string& message,
                                      const std::string& stdout_text = std::string()) {
    emit_result_error(id, build_error_object(kind, message, -1, -1, nullptr, message, {}),
                       stdout_text);
}

// Builds the structured error object for a real Eshkol exception (as opposed
// to a parse failure, an interrupt, or a native crash, which have no
// eshkol_exception_t and go through emit_result_error_simple instead).
static void emit_result_error_from_exception(const std::string& id, eshkol_exception_t* exc,
                                              const std::string& stdout_text) {
    if (!exc) {
        emit_result_error_simple(id, "error", "unknown runtime error", stdout_text);
        return;
    }
    std::string kind = exception_type_name(exc->type);
    std::string message = exc->message ? exc->message : "";
    std::vector<std::string> irritants;
    irritants.reserve(exc->num_irritants);
    for (uint32_t k = 0; k < exc->num_irritants; ++k) {
        irritants.push_back(capture_written_value(exc->irritants[k]));
    }
    std::ostringstream printed;
    printed << kind << ": " << message;
    if (exc->line > 0) {
        printed << " at line " << exc->line;
        if (exc->column > 0) printed << ", column " << exc->column;
    }
    std::string filename_storage;
    const std::string* filename_ptr = nullptr;
    if (exc->filename) {
        filename_storage = exc->filename;
        filename_ptr = &filename_storage;
    }
    emit_result_error(id, build_error_object(kind, message, exc->line, exc->column,
                                              filename_ptr, printed.str(), irritants),
                       stdout_text);
}

// A frame for a request that never reached evaluation at all (malformed
// JSON, or an unrecognized "op") -- distinct from `result` so a driver never
// has to infer "no evaluation happened" from context. Never carries a
// "stdout" field: by construction nothing was ever evaluated.
static void emit_protocol_error(const std::string& id, const std::string& message) {
    std::ostringstream body;
    body << "{\"type\":\"error\",\"id\":" << json_id_field(id)
         << ",\"error\":" << build_error_object("protocol-error", message, -1, -1, nullptr, message, {})
         << "}";
    emit_frame(body.str());
}

// ---- Request handling --------------------------------------------------------

static void track_defined_symbol(const eshkol_ast_t& ast) {
    const char* defined_name = get_defined_name(ast);
    if (!defined_name) return;
    for (const auto& sym : g_defined_symbols) {
        if (sym == defined_name) return;
    }
    g_defined_symbols.push_back(defined_name);
}

// Candidates for a "complete" request: builtins plus every symbol this
// session has defined, sorted and deduplicated -- the same universe
// interactive tab completion (symbol_generator, above) draws from.
static std::vector<std::string> complete_prefix(const std::string& prefix) {
    std::vector<std::string> matches;
    for (const auto& sym : get_builtin_symbols()) {
        if (sym.compare(0, prefix.size(), prefix) == 0) matches.push_back(sym);
    }
    for (const auto& sym : g_defined_symbols) {
        if (sym.compare(0, prefix.size(), prefix) == 0) matches.push_back(sym);
    }
    std::sort(matches.begin(), matches.end());
    matches.erase(std::unique(matches.begin(), matches.end()), matches.end());
    return matches;
}

// "is_complete": does `code` form a complete top-level expression? Uses the
// exact same paren/string/comment scan the interactive multi-line editor
// uses (get_paren_depth, above) -- "complete" here means precisely what it
// means to the accumulation loop, never a second, drifting notion of it.
static const char* is_complete_status(const std::string& code) {
    int depth = get_paren_depth(code);
    if (depth < 0) return "invalid";
    if (depth > 0) return "incomplete";
    return "complete";
}

// Handles a JSON "eval" request: parses `code` as exactly one top-level
// form and evaluates it via executeTagged() (NOT the display-wrapping path
// the legacy bare-form protocol uses below) so stdout receives only bytes
// the form's own code explicitly writes; the form's own value is reported
// separately in the structured response. Mirrors the legacy path's crash/
// exception/signal handling (see the near-identical block in main()) so an
// error, a crash, or an interrupt during a JSON eval behaves the same as
// during a legacy one -- just reported structurally instead of by text.
static void handle_eval_request(const std::string& id, const std::string& code,
                                 eshkol::ReplJITContext& repl_ctx) {
    if (code.empty()) {
        emit_result_error_simple(id, "parse-error", "empty code");
        return;
    }

    try {
        eshkol_ast_t ast = parse_string(code);
        if (ast.type == ESHKOL_INVALID) {
            emit_result_error_simple(id, "parse-error", "failed to parse input");
            return;
        }

        track_defined_symbol(ast);

        eshkol_tagged_value_t tv{};
        bool had_error = false;
        const char* synthetic_kind = nullptr;

        // Captures exactly the bytes this evaluation writes to stdout (see
        // StdoutCapture above); constructed before the setjmp span so it is
        // never itself skipped by a longjmp out of a crash or an interrupt.
        StdoutCapture stdout_capture;

        g_in_jit = 1;
        if (ESHKOL_SIGSETJMP(g_crash_jmp_buf) == 0) {
            eshkol_push_exception_handler(&g_repl_exception_jmp_buf);
            if (setjmp(g_repl_exception_jmp_buf) == 0) {
                tv = repl_ctx.executeTagged(&ast);
            } else {
                had_error = true; // g_current_exception is set
            }
            eshkol_pop_exception_handler();
        } else {
            had_error = true;
            synthetic_kind = (g_crash_signal == SIGINT) ? "interrupted" : "crash";
            // Re-install the correct handler for the signal that fired --
            // the legacy path below needs the identical fix (see main()).
            signal(g_crash_signal, (g_crash_signal == SIGINT) ? sigint_handler : crash_handler);
        }
        g_in_jit = 0;

        std::string captured_stdout = stdout_capture.finish();

        if (had_error) {
            std::fputs(EREPL_FAIL, stderr);
            std::fflush(stderr);
            if (synthetic_kind) {
                emit_result_error_simple(id, synthetic_kind,
                    std::string(synthetic_kind) == "interrupted"
                        ? "evaluation interrupted"
                        : crash_signal_message(g_crash_signal),
                    captured_stdout);
            } else {
                emit_result_error_from_exception(id, g_current_exception, captured_stdout);
            }
        } else {
            std::fputs(EREPL_DONE, stderr);
            std::fflush(stderr);
            emit_result_ok(id, tv, captured_stdout);
        }
        eshkol_ast_clean(&ast);
    } catch (const std::exception& e) {
        g_in_jit = 0;
        std::fflush(stdout);
        std::fputs(EREPL_FAIL, stderr);
        std::fflush(stderr);
        emit_result_error_simple(id, "internal-error", e.what());
    }
}

// Top-level dispatch for one machine-mode JSON request line.
static void handle_json_request(const std::string& line, eshkol::ReplJITContext& repl_ctx) {
    std::unordered_map<std::string, std::string> fields;
    std::string err;
    if (!json_parse_flat_object(line, fields, err)) {
        emit_protocol_error(std::string(), "malformed JSON request: " + err);
        return;
    }

    std::string id = fields.count("id") ? fields["id"] : std::string();
    std::string op = fields.count("op") ? fields["op"] : std::string();

    if (op == "eval") {
        handle_eval_request(id, fields.count("code") ? fields["code"] : std::string(), repl_ctx);
    } else if (op == "complete") {
        std::string prefix = fields.count("prefix") ? fields["prefix"] : std::string();
        auto matches = complete_prefix(prefix);
        std::ostringstream body;
        body << "{\"type\":\"completion\",\"id\":" << json_id_field(id) << ",\"matches\":[";
        for (size_t k = 0; k < matches.size(); ++k) {
            if (k) body << ",";
            body << "\"" << json_escape(matches[k]) << "\"";
        }
        body << "]}";
        emit_frame(body.str());
    } else if (op == "is_complete") {
        std::string code = fields.count("code") ? fields["code"] : std::string();
        std::ostringstream body;
        body << "{\"type\":\"is_complete\",\"id\":" << json_id_field(id)
             << ",\"status\":\"" << is_complete_status(code) << "\"}";
        emit_frame(body.str());
    } else if (op == "reset") {
        g_defined_symbols.clear();
        std::ostringstream body;
        body << "{\"type\":\"reset\",\"id\":" << json_id_field(id) << ",\"ok\":true}";
        emit_frame(body.str());
    } else if (op == "shutdown") {
        std::ostringstream body;
        body << "{\"type\":\"shutdown\",\"id\":" << json_id_field(id) << ",\"ok\":true}";
        emit_frame(body.str());
        repl_clean_exit(0);
    } else {
        emit_protocol_error(id, "unknown op: " + op);
    }
}

// --machine mode framing markers: see the EREPL_READY/EREPL_DONE/EREPL_FAIL
// constants and their doc comment above, right before the "EREPL v1" block
// (moved there so the request handlers in that block, which reference them,
// don't need a forward declaration).

int main(int argc, char** argv) {
    // Parse command-line arguments
    bool load_stdlib = false;
    bool machine_mode = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--stdlib" || arg == "-s") {
            load_stdlib = true;
        } else if (arg == "--machine" || arg == "-m") {
            machine_mode = true;
            load_stdlib = true; // Whole point of the mode is to be JIT-warm
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: eshkol-repl [OPTIONS]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --stdlib, -s    Load standard library on startup\n";
            std::cout << "  --machine, -m   Machine-driven mode: emits EREPL READY/DONE/FAIL\n";
            std::cout << "                  framing on stderr plus a versioned JSON protocol\n";
            std::cout << "                  (EREPL v1: eval/complete/is_complete/reset/shutdown\n";
            std::cout << "                  requests on stdin, EREPL/1 {...} responses on\n";
            std::cout << "                  stderr; see docs/reference/runtime/eshkol-repl.md).\n";
            std::cout << "                  Suppresses banner / prompts; implies --stdlib\n";
            std::cout << "                  (warm-worker for sister projects).\n";
            std::cout << "  --help, -h      Show this help message\n";
            return 0;
        }
    }

    // Check if running interactively
    g_interactive = eshkol::platform::stdin_isatty();
    if (machine_mode) g_interactive = false; // Force machine framing path

    // Install signal handlers.
    signal(SIGINT, sigint_handler);
    signal(SIGSEGV, crash_handler);
    signal(SIGFPE, crash_handler);
#ifdef SIGBUS
    signal(SIGBUS, crash_handler);
#endif

    // Initialize readline with completion and history (only if interactive)
    if (g_interactive) {
        eshkol::platform::initialize_interactive_console();
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

    if (g_interactive) {
        std::cout << "\n";
    }

    // Machine-mode handshake: emit READY on stderr once init is done
    // so the controlling process knows it can start sending forms.
    // Done AFTER stdlib loads (when load_stdlib is set, which --machine
    // forces) so the sentinel really does mean "no more JIT cold-start".
    if (machine_mode) {
        std::fputs(EREPL_READY, stderr);
        std::fflush(stderr);
        std::ostringstream ready_body;
        ready_body << "{\"type\":\"ready\",\"protocol_version\":" << EREPL_PROTOCOL_VERSION
                   << ",\"pid\":" << static_cast<long long>(ESHKOL_GETPID())
                   << ",\"eshkol_version\":\"" << json_escape(ESHKOL_VERSION_STRING) << "\"}";
        emit_frame(ready_body.str());
    }

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
                    if (g_interactive) {
                        std::cout << color::dim() << " (force completing)" << color::reset() << "\n";
                    }
                    break;
                }
                if (g_interactive) {
                    std::cout << "\n" << color::dim() << "Goodbye!" << color::reset() << "\n";
                }
                repl_clean_exit(0);
            }

            // EREPL v1: a machine-mode line beginning with '{' is a JSON
            // request, dispatched immediately as its own unit. No Eshkol
            // source form starts with '{', so this can never collide with
            // the legacy bare-form protocol, and it deliberately bypasses
            // the paren-depth multi-line accumulator below -- a JSON
            // request is always exactly one stdin line; embedded newlines
            // in a "code" field travel JSON-escaped, not raw.
            if (machine_mode && first_line && input[0] == '{') {
                std::string json_line(input);
                free(input);
                handle_json_request(json_line, repl_ctx);
                input_str.clear();
                break;
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
                if (g_interactive) {
                    std::cout << color::dim() << "Goodbye!" << color::reset() << "\n";
                }
                repl_clean_exit(0);
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

        // Skip comment-only or whitespace-only input
        {
            bool is_comment_or_whitespace = true;
            for (size_t i = 0; i < input_str.size(); i++) {
                char c = input_str[i];
                if (std::isspace(c)) continue;
                if (c == ';') break;  // Rest is comment
                is_comment_or_whitespace = false;
                break;
            }
            if (is_comment_or_whitespace) {
                continue;
            }
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
            if (ESHKOL_SIGSETJMP(g_crash_jmp_buf) == 0) {
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
                // Crash, OR an interrupt delivered mid-evaluation (see
                // sigint_handler above -- both longjmp here and are told
                // apart by g_crash_signal) - display error and continue.
                had_error = true;
                print_error("Runtime error", crash_signal_message(g_crash_signal));
                // Re-install the handler for whichever signal fired so the
                // REPL can continue: SIGINT must get sigint_handler back
                // (not crash_handler), or a second Ctrl+C / interrupt would
                // be treated as a crash instead of aborting cleanly again.
                signal(g_crash_signal, (g_crash_signal == SIGINT) ? sigint_handler : crash_handler);
            }
            g_in_jit = 0;

            // Clean up
            eshkol_ast_clean(&ast);
            if (result && !had_error) {
                delete static_cast<int64_t*>(result);
            }

            // Machine-mode per-form sentinel — emit AFTER cleanup so any
            // exception printing has already flushed. Flush stdout first so
            // the client receives all program output before the marker.
            if (machine_mode) {
                std::fflush(stdout);
                std::fputs(had_error ? EREPL_FAIL : EREPL_DONE, stderr);
                std::fflush(stderr);
            }

        } catch (const std::exception& e) {
            print_error("Execution error", e.what());
            if (machine_mode) {
                std::fflush(stdout);
                std::fputs(EREPL_FAIL, stderr);
                std::fflush(stderr);
            }
        }
    }

    save_readline_history();
    std::_Exit(0);
}
