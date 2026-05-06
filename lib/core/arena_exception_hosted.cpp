/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted arena exception and reporting support.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/eshkol.h"
#include "../../inc/eshkol/logger.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <setjmp.h>
#include <string>

eshkol_exception_t* g_current_exception = nullptr;
eshkol_exception_handler_t* g_exception_handler_stack = nullptr;

namespace {

eshkol_tagged_value_t g_raised_tagged_value = {0, 0, 0, {0}};
bool g_raised_value_set_by_user = false;

thread_local int64_t g_recursion_depth = 0;
constexpr int64_t kEshkolMaxRecursionDepth = 100000;

void print_unhandled_exception_and_exit(const eshkol_exception_t* exception) {
    std::fprintf(stderr, "Unhandled exception: ");
    if (exception && exception->message) {
        std::fprintf(stderr, "%s", exception->message);
    } else {
        std::fprintf(stderr, "(unknown error)");
    }
    if (exception && exception->line > 0) {
        std::fprintf(stderr, " at line %u", exception->line);
        if (exception->column > 0) {
            std::fprintf(stderr, ", column %u", exception->column);
        }
        if (exception->filename) {
            std::fprintf(stderr, " in %s", exception->filename);
        }
    }
    std::fprintf(stderr, "\n");
    std::exit(1);
}

constexpr size_t kMaxProviderFilesScanned = 800;
constexpr size_t kMaxProviderFileBytes = 256 * 1024;
constexpr int kMaxProviderScanDepth = 4;
constexpr int kScoreProvide = 100;
constexpr int kScoreDefine = 50;
constexpr int kScoreWeakMention = 10;

struct ProviderScanResult {
    std::string best_path;
    int best_score = 0;
    size_t files_scanned = 0;
};

bool starts_word(const std::string& text, size_t pos, const std::string& word) {
    return pos + word.size() <= text.size() &&
           text.compare(pos, word.size(), word) == 0;
}

bool is_identifier_char(char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) ||
           ch == '-' || ch == '_' || ch == '!' || ch == '?' ||
           ch == '*' || ch == '+' || ch == '/' || ch == '<' ||
           ch == '>' || ch == '=' || ch == '.';
}

bool name_at(const std::string& text, size_t pos, const std::string& name) {
    if (pos + name.size() > text.size()) {
        return false;
    }
    if (text.compare(pos, name.size(), name) != 0) {
        return false;
    }
    if (pos > 0 && is_identifier_char(text[pos - 1])) {
        return false;
    }
    const size_t end = pos + name.size();
    return end >= text.size() || !is_identifier_char(text[end]);
}

int score_provider_text(const std::string& text, const std::string& name) {
    int best = 0;
    size_t pos = 0;
    while ((pos = text.find(name, pos)) != std::string::npos) {
        if (!name_at(text, pos, name)) {
            pos += name.size();
            continue;
        }

        size_t back = pos;
        while (back > 0 && text[back - 1] != '(' && text[back - 1] != '\n') {
            --back;
        }
        if (back == 0 || text[back - 1] != '(') {
            if (best < kScoreWeakMention) {
                best = kScoreWeakMention;
            }
            pos += name.size();
            continue;
        }

        size_t head_start = back;
        while (head_start < text.size() &&
               std::isspace(static_cast<unsigned char>(text[head_start]))) {
            ++head_start;
        }
        if (starts_word(text, head_start, "provide")) {
            return kScoreProvide;
        }
        if (starts_word(text, head_start, "define")) {
            size_t after = head_start + 6;
            while (after < text.size() &&
                   std::isspace(static_cast<unsigned char>(text[after]))) {
                ++after;
            }
            if (after < text.size() && text[after] == '(') {
                ++after;
                while (after < text.size() &&
                       std::isspace(static_cast<unsigned char>(text[after]))) {
                    ++after;
                }
                if (after == pos && best < kScoreDefine) {
                    best = kScoreDefine;
                }
            } else if (after == pos && best < kScoreDefine) {
                best = kScoreDefine;
            }
        }
        pos += name.size();
    }
    return best;
}

void scan_provider_dir(const std::filesystem::path& dir,
                       const std::string& name,
                       int depth,
                       ProviderScanResult& result) {
    if (depth > kMaxProviderScanDepth ||
        result.files_scanned >= kMaxProviderFilesScanned ||
        result.best_score >= kScoreProvide) {
        return;
    }

    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) ||
        !std::filesystem::is_directory(dir, ec)) {
        return;
    }

    const auto options =
        std::filesystem::directory_options::skip_permission_denied;
    for (std::filesystem::directory_iterator it(dir, options, ec), end;
         !ec && it != end && result.files_scanned < kMaxProviderFilesScanned;
         it.increment(ec)) {
        if (ec) {
            break;
        }

        const auto& entry = *it;
        std::error_code entry_ec;
        if (entry.is_directory(entry_ec)) {
            const std::string dirname = entry.path().filename().string();
            if (dirname.empty() || dirname[0] == '.' ||
                dirname == "build" || dirname == "build-x64" ||
                dirname == "build-asan" || dirname == "build-xla" ||
                dirname == "build-cuda" || dirname == "build-debug" ||
                dirname == "node_modules" || dirname == "deps" ||
                dirname == "dist" || dirname == "artifacts" ||
                dirname == "Testing") {
                continue;
            }
            scan_provider_dir(entry.path(), name, depth + 1, result);
            continue;
        }

        if (!entry.is_regular_file(entry_ec) || entry.path().extension() != ".esk") {
            continue;
        }

        ++result.files_scanned;
        std::ifstream file(entry.path(), std::ios::binary);
        if (!file) {
            continue;
        }
        std::string text(kMaxProviderFileBytes, '\0');
        file.read(text.data(), static_cast<std::streamsize>(kMaxProviderFileBytes));
        text.resize(static_cast<size_t>(file.gcount()));

        const int score = score_provider_text(text, name);
        if (score > result.best_score) {
            result.best_score = score;
            result.best_path = entry.path().string();
            if (result.best_score >= kScoreProvide) {
                return;
            }
        }
    }
}

char* find_provider_file(const char* name) {
    if (!name || !name[0]) {
        return nullptr;
    }

    ProviderScanResult result;
    const std::string symbol_name(name);
    std::error_code ec;
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec) {
        scan_provider_dir(cwd, symbol_name, 0, result);
    }

    if (result.best_score < kScoreProvide) {
        if (const char* root = std::getenv("ESHKOL_PROJECT_ROOT")) {
            scan_provider_dir(std::filesystem::path(root), symbol_name, 0, result);
        }
    }

    if (result.best_score == 0) {
        return nullptr;
    }
    return strdup(result.best_path.c_str());
}

} // namespace

extern "C" void eshkol_set_raised_value(const eshkol_tagged_value_t* value) {
    g_raised_tagged_value = *value;
    g_raised_value_set_by_user = true;
}

extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t* out) {
    *out = g_raised_tagged_value;
}

extern "C" eshkol_exception_t* eshkol_make_exception_with_header(eshkol_exception_type_t type,
                                                                  const char* message) {
    arena_t* arena = __repl_shared_arena.load();
    if (!arena) {
        eshkol_error("No arena available for exception allocation");
        return nullptr;
    }

    size_t data_size = sizeof(eshkol_exception_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~7;

    auto* mem = static_cast<uint8_t*>(arena_allocate_aligned(arena, total, 8));
    if (!mem) {
        eshkol_error("Failed to allocate exception with header");
        return nullptr;
    }

    auto* hdr = reinterpret_cast<eshkol_object_header_t*>(mem);
    hdr->subtype = HEAP_SUBTYPE_EXCEPTION;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = static_cast<uint32_t>(data_size);

    auto* exc = reinterpret_cast<eshkol_exception_t*>(mem + sizeof(eshkol_object_header_t));
    exc->type = type;
    if (message) {
        size_t len = std::strlen(message) + 1;
        exc->message = static_cast<char*>(arena_allocate(arena, len));
        if (exc->message) {
            std::memcpy(exc->message, message, len - 1);
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

extern "C" eshkol_exception_t* eshkol_make_exception(eshkol_exception_type_t type,
                                                       const char* message) {
    arena_t* arena = __repl_shared_arena.load();
    if (!arena) {
        auto* exc = static_cast<eshkol_exception_t*>(std::malloc(sizeof(eshkol_exception_t)));
        if (!exc) {
            return nullptr;
        }

        exc->type = type;
        exc->message = message ? strdup(message) : nullptr;
        exc->irritants = nullptr;
        exc->num_irritants = 0;
        exc->line = 0;
        exc->column = 0;
        exc->filename = nullptr;
        return exc;
    }

    auto* exc = static_cast<eshkol_exception_t*>(arena_allocate(arena, sizeof(eshkol_exception_t)));
    if (!exc) {
        return nullptr;
    }

    exc->type = type;
    if (message) {
        size_t len = std::strlen(message) + 1;
        exc->message = static_cast<char*>(arena_allocate(arena, len));
        if (exc->message) {
            std::memcpy(exc->message, message, len - 1);
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

extern "C" void eshkol_exception_add_irritant(eshkol_exception_t* exc,
                                               eshkol_tagged_value_t irritant) {
    if (!exc) {
        return;
    }

    uint32_t new_count = exc->num_irritants + 1;
    eshkol_tagged_value_t* new_irritants;

    arena_t* arena = __repl_shared_arena.load();
    if (arena) {
        new_irritants = static_cast<eshkol_tagged_value_t*>(
            arena_allocate(arena, new_count * sizeof(eshkol_tagged_value_t)));
    } else {
        new_irritants = static_cast<eshkol_tagged_value_t*>(
            std::malloc(new_count * sizeof(eshkol_tagged_value_t)));
    }

    if (!new_irritants) {
        return;
    }

    if (exc->irritants && exc->num_irritants > 0) {
        std::memcpy(new_irritants,
                    exc->irritants,
                    exc->num_irritants * sizeof(eshkol_tagged_value_t));
    }

    new_irritants[exc->num_irritants] = irritant;
    exc->irritants = new_irritants;
    exc->num_irritants = new_count;
}

extern "C" void eshkol_exception_set_location(eshkol_exception_t* exc,
                                               uint32_t line,
                                               uint32_t column,
                                               const char* filename) {
    if (!exc) {
        return;
    }

    exc->line = line;
    exc->column = column;

    if (filename) {
        arena_t* arena = __repl_shared_arena.load();
        if (arena) {
            size_t len = std::strlen(filename) + 1;
            exc->filename = static_cast<char*>(arena_allocate(arena, len));
            if (exc->filename) {
                std::memcpy(exc->filename, filename, len - 1);
                exc->filename[len - 1] = '\0';
            }
        } else {
            exc->filename = strdup(filename);
        }
    }
}

extern "C" void eshkol_raise(eshkol_exception_t* exception) {
    g_current_exception = exception;

    if (!g_raised_value_set_by_user) {
        g_raised_tagged_value.type = ESHKOL_VALUE_HEAP_PTR;
        g_raised_tagged_value.flags = 0;
        g_raised_tagged_value.reserved = 0;
        g_raised_tagged_value.data.ptr_val = reinterpret_cast<uint64_t>(exception);
    }
    g_raised_value_set_by_user = false;

    if (g_exception_handler_stack && g_exception_handler_stack->jmp_buf_ptr) {
        longjmp(*reinterpret_cast<jmp_buf*>(g_exception_handler_stack->jmp_buf_ptr), 1);
    }

    print_unhandled_exception_and_exit(exception);
}

extern "C" void* eshkol_check_forward_ref(void* loaded_fn_ptr,
                                          void* stub_sentinel,
                                          const char* func_name) {
    if (loaded_fn_ptr != stub_sentinel) {
        return loaded_fn_ptr;
    }

    char buffer[1024];
    char* hint = nullptr;
    if (func_name && func_name[0]) {
        hint = find_provider_file(func_name);
        if (hint) {
            std::snprintf(buffer,
                          sizeof(buffer),
                          "called undefined function '%s' "
                          "(forward-referenced but never defined). "
                          "Likely missing: (load \"%s\") - that file appears "
                          "to define '%s'.",
                          func_name,
                          hint,
                          func_name);
            std::free(hint);
        } else {
            std::snprintf(buffer,
                          sizeof(buffer),
                          "called undefined function '%s' "
                          "(forward-referenced but never defined; check that "
                          "the file containing its `define` is `(load ...)`ed "
                          "or `(require ...)`d before the call site)",
                          func_name);
        }
    } else {
        std::snprintf(buffer,
                      sizeof(buffer),
                      "called a forward-referenced function that was never defined");
    }

    eshkol_exception_t* exc = eshkol_make_exception(ESHKOL_EXCEPTION_ERROR, buffer);
    if (exc) {
        eshkol_raise(exc);
    }
    return nullptr;
}

extern "C" void eshkol_raise_not_pair(const char* op_name) {
    const char* message = op_name ? op_name : "car/cdr: argument is not a pair";
    eshkol_exception_t* exc =
        eshkol_make_exception(ESHKOL_EXCEPTION_TYPE_ERROR, message);
    if (exc) {
        eshkol_raise(exc);
    }
    std::fprintf(stderr, "Eshkol: %s\n", message);
    std::exit(1);
}

extern "C" void eshkol_raise_index_oob(const char* op_name, int64_t idx,
                                        int64_t length) {
    char buffer[160];
    std::snprintf(buffer, sizeof(buffer),
                  "%s: index %lld out of bounds (length=%lld)",
                  op_name ? op_name : "list-ref/vector-ref",
                  static_cast<long long>(idx),
                  static_cast<long long>(length));
    eshkol_exception_t* exc =
        eshkol_make_exception(ESHKOL_EXCEPTION_ERROR, buffer);
    if (exc) {
        eshkol_raise(exc);
    }
    std::fprintf(stderr, "Eshkol: %s\n", buffer);
    std::exit(1);
}

extern "C" void eshkol_raise_improper_list(const char* msg) {
    const char* message = msg ? msg : "improper list";
    eshkol_exception_t* exc =
        eshkol_make_exception(ESHKOL_EXCEPTION_ERROR, message);
    if (exc) {
        eshkol_raise(exc);
    }
    std::fprintf(stderr, "Eshkol: %s\n", message);
    std::exit(1);
}

extern "C" void eshkol_push_exception_handler(void* jmp_buf_ptr) {
    eshkol_exception_handler_t* handler;

    arena_t* arena = __repl_shared_arena.load();
    if (arena) {
        handler = static_cast<eshkol_exception_handler_t*>(
            arena_allocate(arena, sizeof(eshkol_exception_handler_t)));
    } else {
        handler = static_cast<eshkol_exception_handler_t*>(
            std::malloc(sizeof(eshkol_exception_handler_t)));
    }

    if (!handler) {
        eshkol_error("Failed to allocate exception handler");
        return;
    }

    handler->jmp_buf_ptr = jmp_buf_ptr;
    handler->prev = g_exception_handler_stack;
    g_exception_handler_stack = handler;
}

extern "C" void eshkol_pop_exception_handler(void) {
    if (g_exception_handler_stack) {
        g_exception_handler_stack = g_exception_handler_stack->prev;
    }
}

extern "C" int eshkol_exception_type_matches(eshkol_exception_t* exc,
                                              eshkol_exception_type_t type) {
    if (!exc) {
        return 0;
    }
    return exc->type == type;
}

extern "C" eshkol_exception_t* eshkol_get_current_exception(void) {
    return g_current_exception;
}

extern "C" void eshkol_clear_current_exception(void) {
    g_current_exception = nullptr;
    g_raised_tagged_value.type = ESHKOL_VALUE_NULL;
    g_raised_tagged_value.flags = 0;
    g_raised_tagged_value.reserved = 0;
    g_raised_tagged_value.data.raw_val = 0;
    g_raised_value_set_by_user = false;
}

extern "C" void eshkol_display_exception(eshkol_exception_t* exc) {
    if (!exc) {
        std::printf("#<exception:null>");
        return;
    }

    const char* type_name;
    switch (exc->type) {
        case ESHKOL_EXCEPTION_ERROR:
            type_name = "error";
            break;
        case ESHKOL_EXCEPTION_TYPE_ERROR:
            type_name = "type-error";
            break;
        case ESHKOL_EXCEPTION_FILE_ERROR:
            type_name = "file-error";
            break;
        case ESHKOL_EXCEPTION_READ_ERROR:
            type_name = "read-error";
            break;
        case ESHKOL_EXCEPTION_SYNTAX_ERROR:
            type_name = "syntax-error";
            break;
        case ESHKOL_EXCEPTION_RANGE_ERROR:
            type_name = "range-error";
            break;
        case ESHKOL_EXCEPTION_ARITY_ERROR:
            type_name = "arity-error";
            break;
        case ESHKOL_EXCEPTION_DIVIDE_BY_ZERO:
            type_name = "divide-by-zero";
            break;
        case ESHKOL_EXCEPTION_USER_DEFINED:
            type_name = "user-exception";
            break;
        default:
            type_name = "unknown";
            break;
    }

    std::printf("#<%s: %s>", type_name, exc->message ? exc->message : "");
}

extern "C" int64_t eshkol_check_recursion_depth(void) {
    g_recursion_depth++;
    if (g_recursion_depth > kEshkolMaxRecursionDepth) {
        g_recursion_depth = 0;
        eshkol_exception_t* exc = eshkol_make_exception(
            ESHKOL_EXCEPTION_ERROR,
            "maximum recursion depth exceeded");
        if (exc) {
            eshkol_raise(exc);
        }
        std::fprintf(stderr,
                     "Error: maximum recursion depth (%lld) exceeded\n",
                     static_cast<long long>(kEshkolMaxRecursionDepth));
        std::exit(1);
    }
    return g_recursion_depth;
}

extern "C" void eshkol_decrement_recursion_depth(void) {
    if (g_recursion_depth > 0) {
        g_recursion_depth--;
    }
}

extern "C" void eshkol_reset_recursion_depth(void) {
    g_recursion_depth = 0;
}
