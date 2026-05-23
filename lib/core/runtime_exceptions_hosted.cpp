/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted exception handling and forward-reference diagnostics.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"
#include "../../inc/eshkol/eshkol.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <setjmp.h>
#include <string>
#include <string.h>

// ===== EXCEPTION HANDLING IMPLEMENTATION =====
// Runtime support for R7RS-compatible exception handling

// Global exception state
eshkol_exception_t* g_current_exception = nullptr;
eshkol_exception_handler_t* g_exception_handler_stack = nullptr;

// R7RS: stores the original raised tagged_value for with-exception-handler
eshkol_tagged_value_t g_raised_tagged_value = {0, 0, 0, {0}};
static bool g_raised_value_set_by_user = false;

// Store a tagged value before raising (called from codegen for user `raise`)
extern "C" void eshkol_set_raised_value(const eshkol_tagged_value_t* value) {
    g_raised_tagged_value = *value;
    g_raised_value_set_by_user = true;
}

// Retrieve the raised tagged value (called from with-exception-handler and guard)
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t* out) {
    *out = g_raised_tagged_value;
}

// Create a new exception object with object header (for consolidated HEAP_PTR type)
extern "C" eshkol_exception_t* eshkol_make_exception_with_header(eshkol_exception_type_t type, const char* message) {
    arena_t* arena = __repl_shared_arena.load();
    if (!arena) {
        eshkol_error("No arena available for exception allocation");
        return nullptr;
    }

    size_t data_size = sizeof(eshkol_exception_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~7;

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate exception with header");
        return nullptr;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_EXCEPTION;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    eshkol_exception_t* exc = (eshkol_exception_t*)(mem + sizeof(eshkol_object_header_t));

    exc->type = type;
    if (message) {
        size_t len = strlen(message) + 1;
        exc->message = (char*)arena_allocate(arena, len);
        if (exc->message) {
            memcpy(exc->message, message, len - 1);
            exc->message[len - 1] = '\0';
        }
    } else {
        exc->message = nullptr;
    }
    exc->irritants = nullptr;
    exc->num_irritants = 0;
    exc->line = 0;
    exc->column = 0;
    exc->filename = nullptr;

    return exc;
}

// Create a new exception object (legacy - no header)
extern "C" eshkol_exception_t* eshkol_make_exception(eshkol_exception_type_t type, const char* message) {
    arena_t* arena = __repl_shared_arena.load();
    if (!arena) {
        // Allocate from heap if no arena available
        eshkol_exception_t* exc = (eshkol_exception_t*)malloc(sizeof(eshkol_exception_t));
        if (!exc) return nullptr;

        exc->type = type;
        exc->message = message ? strdup(message) : nullptr;
        exc->irritants = nullptr;
        exc->num_irritants = 0;
        exc->line = 0;
        exc->column = 0;
        exc->filename = nullptr;
        return exc;
    }

    // Allocate from arena
    eshkol_exception_t* exc = (eshkol_exception_t*)arena_allocate(arena, sizeof(eshkol_exception_t));
    if (!exc) return nullptr;

    exc->type = type;
    if (message) {
        size_t len = strlen(message) + 1;
        exc->message = (char*)arena_allocate(arena, len);
        if (exc->message) {
            memcpy(exc->message, message, len - 1);
            exc->message[len - 1] = '\0';
        }
    } else {
        exc->message = nullptr;
    }
    exc->irritants = nullptr;
    exc->num_irritants = 0;
    exc->line = 0;
    exc->column = 0;
    exc->filename = nullptr;

    return exc;
}

// Add an irritant to an exception
extern "C" void eshkol_exception_add_irritant(eshkol_exception_t* exc, eshkol_tagged_value_t irritant) {
    if (!exc) return;

    // Grow irritants array
    uint32_t new_count = exc->num_irritants + 1;
    eshkol_tagged_value_t* new_irritants;

    arena_t* arena = __repl_shared_arena.load();
    if (arena) {
        new_irritants = (eshkol_tagged_value_t*)arena_allocate(arena, new_count * sizeof(eshkol_tagged_value_t));
    } else {
        new_irritants = (eshkol_tagged_value_t*)malloc(new_count * sizeof(eshkol_tagged_value_t));
    }

    if (!new_irritants) return;

    // Copy existing irritants
    if (exc->irritants && exc->num_irritants > 0) {
        memcpy(new_irritants, exc->irritants, exc->num_irritants * sizeof(eshkol_tagged_value_t));
    }

    // Add new irritant
    new_irritants[exc->num_irritants] = irritant;
    exc->irritants = new_irritants;
    exc->num_irritants = new_count;
}

// Set source location on exception
extern "C" void eshkol_exception_set_location(eshkol_exception_t* exc, uint32_t line, uint32_t column, const char* filename) {
    if (!exc) return;

    exc->line = line;
    exc->column = column;

    if (filename) {
        arena_t* arena = __repl_shared_arena.load();
        if (arena) {
            size_t len = strlen(filename) + 1;
            exc->filename = (char*)arena_allocate(arena, len);
            if (exc->filename) {
                memcpy(exc->filename, filename, len - 1);
                exc->filename[len - 1] = '\0';
            }
        } else {
            exc->filename = strdup(filename);
        }
    }
}

// Raise an exception - jumps to nearest handler
/* Bug W: REPL forward-ref call where the named function was never
 * defined. The codegen call site emits this BEFORE the indirect call,
 * passing the function name as a literal. The codegen then calls this
 * helper with the loaded fn_ptr and the stub-detection sentinel. If
 * the slot still points at the unresolved stub, raise with a clear
 * message that names the function. Otherwise return the resolved ptr.
 *
 * The sentinel comparison is necessary because every unresolved slot
 * shares the same stub address — codegen can't tell at emit time
 * whether the slot will be resolved by JIT load order.
 */
/* Bug W ask 2: scan project .esk files for `(provide …)` or `(define …)`
 * lists that mention `name` and suggest the most likely file to (load …).
 *
 * Heuristics — text-based, not a full parse — so this never blocks the
 * error from being raised.  We bound the work: max 4 dirs deep, max 800
 * files, max 256 KB read per file.  Result is an arena-allocated
 * malloc'd buffer (caller-owned), or NULL if nothing found.
 *
 * Search order:
 *   1. CWD recursively
 *   2. ./src recursively
 *   3. ./lib recursively
 *   4. $ESHKOL_PROJECT_ROOT recursively (if set)
 *
 * Match scoring (highest wins):
 *   - 100  exact `(provide name)` form found
 *   -  50  exact `(define name …)` or `(define (name …) …)` found
 *   -  10  name appears in some other position (weak hint)
 */
static char* eshkol_find_provider_file(const char* name);

// Derive an Eshkol module name from a file path under a `lib/` (or
// equivalent) directory. e.g.
//   /Users/.../lib/agent/http.esk  → "agent.http"
//   /Users/.../lib/core/json.esk   → "core.json"
//   /tmp/myfile.esk                → ""  (no module-style location)
// Returns malloc'd string the caller frees, or NULL if no module name
// can be derived.
static char* derive_module_name_from_path(const char* file_path) {
    if (!file_path || !file_path[0]) return nullptr;
    std::string path(file_path);

    // Look for "/lib/" anywhere in the path; the module path starts
    // after that. Falls back to "src/" since some projects use that.
    size_t lib_pos = path.find("/lib/");
    size_t skip = 5;  // "/lib/" length
    if (lib_pos == std::string::npos) {
        lib_pos = path.find("/src/");
        skip = 5;
    }
    if (lib_pos == std::string::npos) return nullptr;

    std::string mod = path.substr(lib_pos + skip);
    // Drop ".esk" suffix
    if (mod.size() > 4 && mod.compare(mod.size() - 4, 4, ".esk") == 0) {
        mod.resize(mod.size() - 4);
    }
    // Replace "/" with "." for module syntax. Reject paths containing
    // any character that wouldn't be valid in a Scheme symbol —
    // unusual filenames mean we can't safely synthesise a module name.
    for (size_t i = 0; i < mod.size(); ++i) {
        char c = mod[i];
        if (c == '/') { mod[i] = '.'; continue; }
        if (!(std::isalnum(static_cast<unsigned char>(c)) ||
              c == '-' || c == '_' || c == '.' || c == '!' || c == '?')) {
            return nullptr;
        }
    }
    if (mod.empty()) return nullptr;
    return strdup(mod.c_str());
}

extern "C" void* eshkol_check_forward_ref(void* loaded_fn_ptr,
                                          void* stub_sentinel,
                                          const char* func_name) {
    if (loaded_fn_ptr == stub_sentinel) {
        char buf[1024];
        char* hint = nullptr;
        if (func_name && func_name[0]) {
            hint = eshkol_find_provider_file(func_name);
            if (hint) {
                // If the matched file looks like a module (has a `lib/`
                // ancestor), suggest `(require module-name)` since that's
                // the modern entry point. Falls back to `(load …)` for
                // ad-hoc files that aren't part of the module tree.
                // Bug W "ask 2" (eshkol-agent 2026-05-06): the previous
                // hint always suggested (load …), but agents normally
                // wire imports through (require …) and provide forms.
                char* module_name = derive_module_name_from_path(hint);
                if (module_name) {
                    snprintf(buf, sizeof(buf),
                             "called undefined function '%s' "
                             "(forward-referenced but never defined). "
                             "Likely missing: (require %s) — that module's "
                             "file %s appears to define '%s'.",
                             func_name, module_name, hint, func_name);
                    std::free(module_name);
                } else {
                    snprintf(buf, sizeof(buf),
                             "called undefined function '%s' "
                             "(forward-referenced but never defined). "
                             "Likely missing: (load \"%s\") — that file "
                             "appears to define '%s'.",
                             func_name, hint, func_name);
                }
                std::free(hint);
            } else {
                snprintf(buf, sizeof(buf),
                         "called undefined function '%s' "
                         "(forward-referenced but never defined; "
                         "check that the file containing its `define` is `(load …)`ed "
                         "or `(require …)`d before the call site)",
                         func_name);
            }
        } else {
            snprintf(buf, sizeof(buf),
                     "called a forward-referenced function that was never defined");
        }
        eshkol_exception_t* exc = eshkol_make_exception(
            ESHKOL_EXCEPTION_ERROR, buf);
        if (exc) {
            eshkol_raise(exc);
        }
        // eshkol_raise exits if no handler. If it returns (handler was
        // installed), we still must not proceed with a stub call —
        // return null and let the caller's null-handling raise an
        // exception of its own. This keeps the "fatal under no handler"
        // invariant while preserving guard-clause behaviour.
        return nullptr;
    }
    return loaded_fn_ptr;
}

// Bug W ask 2 — implementation.
// Walk a few project-relative directories looking for `.esk` files that
// either `(provide … name …)` or `(define name …)` / `(define (name …) …)`.
// Returns a malloc'd string the caller frees, or NULL.  Bounded work:
// max ~800 files, max 256 KB per file, max 4 dirs deep.
namespace {
constexpr size_t kMaxFilesScanned   = 800;
constexpr size_t kMaxFileBytes      = 256 * 1024;
constexpr int    kMaxDepth          = 4;
constexpr int    kScoreProvide      = 100;
constexpr int    kScoreDefineHead   =  50;
constexpr int    kScoreDefineParen  =  50;
constexpr int    kScoreWeakMention  =  10;

struct ScanResult {
    std::string best_path;
    int         best_score = 0;
    size_t      files_scanned = 0;
};

bool starts_word(const std::string& s, size_t pos, const std::string& word) {
    if (pos + word.size() > s.size()) return false;
    return s.compare(pos, word.size(), word) == 0;
}

bool name_at(const std::string& haystack, size_t pos, const std::string& name) {
    if (pos + name.size() > haystack.size()) return false;
    if (haystack.compare(pos, name.size(), name) != 0) return false;
    // Word boundary on both sides — Eshkol identifiers can contain alnum,
    // -, _, !, ?, *, +, /, <, >, =, .  Anything else is a separator.
    auto is_id = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) ||
               c == '-' || c == '_' || c == '!' || c == '?' ||
               c == '*' || c == '+' || c == '/' || c == '<' ||
               c == '>' || c == '=' || c == '.';
    };
    if (pos > 0 && is_id(haystack[pos - 1])) return false;
    size_t end = pos + name.size();
    if (end < haystack.size() && is_id(haystack[end])) return false;
    return true;
}

int score_file_text(const std::string& text, const std::string& name) {
    int best = 0;
    size_t pos = 0;
    while ((pos = text.find(name, pos)) != std::string::npos) {
        if (!name_at(text, pos, name)) { pos += name.size(); continue; }

        // Look back to find the most recent `(`
        size_t back = pos;
        while (back > 0 && text[back - 1] != '(' && text[back - 1] != '\n') --back;
        if (back == 0 || text[back - 1] != '(') {
            // No `(` on this line before the name → weak mention only
            if (best < kScoreWeakMention) best = kScoreWeakMention;
            pos += name.size();
            continue;
        }
        // Identify the head form just after `(`
        size_t head_start = back;
        while (head_start < text.size() &&
               std::isspace(static_cast<unsigned char>(text[head_start]))) ++head_start;
        if (starts_word(text, head_start, "provide")) {
            return kScoreProvide;  // can't do better
        }
        if (starts_word(text, head_start, "define")) {
            // (define name …)  OR  (define (name …) …)
            // Distinguish: head is "define", then whitespace, then either
            //   - the name itself
            //   - `(` then the name
            size_t after = head_start + 6;  // past "define"
            while (after < text.size() &&
                   std::isspace(static_cast<unsigned char>(text[after]))) ++after;
            if (after < text.size() && text[after] == '(') {
                ++after;
                while (after < text.size() &&
                       std::isspace(static_cast<unsigned char>(text[after]))) ++after;
                if (after == pos) {
                    if (best < kScoreDefineParen) best = kScoreDefineParen;
                }
            } else if (after == pos) {
                if (best < kScoreDefineHead) best = kScoreDefineHead;
            }
        }
        pos += name.size();
    }
    return best;
}

void scan_dir(const std::filesystem::path& dir, const std::string& name,
              int depth, ScanResult& out) {
    if (depth > kMaxDepth || out.files_scanned >= kMaxFilesScanned) return;
    if (out.best_score >= kScoreProvide) return;  // can't beat that

    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) {
        return;
    }

    for (auto it = std::filesystem::directory_iterator(dir, std::filesystem::directory_options::skip_permission_denied, ec);
         !ec && it != std::filesystem::directory_iterator{} && out.files_scanned < kMaxFilesScanned;
         it.increment(ec)) {
        if (ec) break;
        const auto& entry = *it;
        std::error_code ec2;
        if (entry.is_directory(ec2)) {
            // Skip well-known build / vendor / hidden dirs
            std::string dn = entry.path().filename().string();
            if (dn.empty() || dn[0] == '.' ||
                dn == "build" || dn == "build-x64" || dn == "build-asan" ||
                dn == "build-xla" || dn == "build-cuda" || dn == "build-debug" ||
                dn == "node_modules" || dn == "deps" || dn == "dist" ||
                dn == "artifacts" || dn == "Testing") continue;
            scan_dir(entry.path(), name, depth + 1, out);
            continue;
        }
        if (!entry.is_regular_file(ec2)) continue;
        if (entry.path().extension() != ".esk") continue;

        ++out.files_scanned;
        std::ifstream f(entry.path(), std::ios::binary);
        if (!f) continue;
        // Read up to kMaxFileBytes
        std::string text;
        text.resize(kMaxFileBytes);
        f.read(text.data(), kMaxFileBytes);
        text.resize(f.gcount());

        int sc = score_file_text(text, name);
        if (sc > out.best_score) {
            out.best_score = sc;
            out.best_path  = entry.path().string();
            if (out.best_score >= kScoreProvide) return;
        }
    }
}
}  // anonymous namespace

static char* eshkol_find_provider_file(const char* name) {
    if (!name || !name[0]) return nullptr;
    std::string sname(name);

    ScanResult res;
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path cwd = fs::current_path(ec);
    if (!ec) {
        scan_dir(cwd, sname, 0, res);
    }
    if (res.best_score < kScoreProvide) {
        if (const char* root = std::getenv("ESHKOL_PROJECT_ROOT")) {
            scan_dir(fs::path(root), sname, 0, res);
        }
    }
    if (res.best_score == 0) return nullptr;
    return strdup(res.best_path.c_str());
}

extern "C" void eshkol_raise(eshkol_exception_t* exception) {
    g_current_exception = exception;

    // If user didn't set a raised value via eshkol_set_raised_value,
    // use the exception pointer as a fallback HEAP_PTR tagged value
    if (!g_raised_value_set_by_user) {
        g_raised_tagged_value.type = ESHKOL_VALUE_HEAP_PTR;
        g_raised_tagged_value.flags = 0;
        g_raised_tagged_value.reserved = 0;
        g_raised_tagged_value.data.ptr_val = (uint64_t)exception;
    }
    g_raised_value_set_by_user = false;  // Reset for next raise

    if (g_exception_handler_stack && g_exception_handler_stack->jmp_buf_ptr) {
        // Jump to the handler
        longjmp(*(jmp_buf*)g_exception_handler_stack->jmp_buf_ptr, 1);
    } else {
        // No handler - print error and exit gracefully (not abort)
        fprintf(stderr, "Unhandled exception: ");
        if (exception && exception->message) {
            fprintf(stderr, "%s", exception->message);
        } else {
            fprintf(stderr, "(unknown error)");
        }
        if (exception && exception->line > 0) {
            fprintf(stderr, " at line %u", exception->line);
            if (exception->column > 0) {
                fprintf(stderr, ", column %u", exception->column);
            }
            if (exception->filename) {
                fprintf(stderr, " in %s", exception->filename);
            }
        }
        fprintf(stderr, "\n");
        exit(1);
    }
}

// Push exception handler onto stack
extern "C" void eshkol_push_exception_handler(void* jmp_buf_ptr) {
    eshkol_exception_handler_t* handler;

    arena_t* arena = __repl_shared_arena.load();
    if (arena) {
        handler = (eshkol_exception_handler_t*)arena_allocate(arena, sizeof(eshkol_exception_handler_t));
    } else {
        handler = (eshkol_exception_handler_t*)malloc(sizeof(eshkol_exception_handler_t));
    }

    if (!handler) {
        eshkol_error("Failed to allocate exception handler");
        return;
    }

    handler->jmp_buf_ptr = jmp_buf_ptr;
    handler->prev = g_exception_handler_stack;
    g_exception_handler_stack = handler;
}

// Pop exception handler from stack
extern "C" void eshkol_pop_exception_handler(void) {
    if (g_exception_handler_stack) {
        eshkol_exception_handler_t* popped = g_exception_handler_stack;
        g_exception_handler_stack = popped->prev;
        // Note: If allocated from arena, memory is automatically freed with arena
        // If from heap, we leak here - but exception handlers should be short-lived
    }
}

// Check if exception matches a specific type
extern "C" int eshkol_exception_type_matches(eshkol_exception_t* exc, eshkol_exception_type_t type) {
    if (!exc) return 0;
    return exc->type == type;
}

// Get current exception (for handlers)
extern "C" eshkol_exception_t* eshkol_get_current_exception(void) {
    return g_current_exception;
}

// Clear current exception
extern "C" void eshkol_clear_current_exception(void) {
    g_current_exception = nullptr;
}

// Display exception for debugging
extern "C" void eshkol_display_exception(eshkol_exception_t* exc) {
    if (!exc) {
        printf("#<exception:null>");
        return;
    }

    const char* type_name;
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
        default: type_name = "unknown"; break;
    }

    printf("#<%s: %s>", type_name, exc->message ? exc->message : "");
}

// ===== END EXCEPTION HANDLING IMPLEMENTATION =====
