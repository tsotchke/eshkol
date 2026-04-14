/**
 * @file eshkol_ffi.cpp
 * @brief Implementation of the stable C FFI for Eshkol embedding.
 *
 * Uses ReplJITContext for in-process JIT evaluation — no subprocess needed.
 * State persists across eval calls within the same context.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <eshkol/eshkol_ffi.h>
#include <eshkol/eshkol.h>
#include <eshkol/core/runtime.h>
#include "../../lib/core/arena_memory.h"
#include "../../lib/repl/repl_jit.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#ifndef _WIN32
#include <unistd.h>
#endif

/* ── Thread-local error message ── */
static thread_local char g_ffi_error[4096] = {0};

static void ffi_set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_ffi_error, sizeof(g_ffi_error), fmt, ap);
    va_end(ap);
}

/* ── Context — holds the JIT engine ── */
struct eshkol_ffi_context {
    bool initialized;
    eshkol::ReplJITContext* jit;
};

/* ── Convert between FFI and runtime value types ── */
/* These are layout-compatible (both 16 bytes, same field order). */
static inline eshkol_ffi_value_t to_ffi(eshkol_tagged_value_t v) {
    eshkol_ffi_value_t result;
    static_assert(sizeof(result) == sizeof(v), "FFI and tagged value must be same size");
    memcpy(&result, &v, sizeof(v));
    return result;
}

static inline eshkol_tagged_value_t from_ffi(eshkol_ffi_value_t v) {
    eshkol_tagged_value_t result;
    memcpy(&result, &v, sizeof(v));
    return result;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

extern "C" eshkol_ffi_context_t* eshkol_ffi_init(void) {
    auto* ctx = (eshkol_ffi_context_t*)calloc(1, sizeof(eshkol_ffi_context_t));
    if (!ctx) {
        ffi_set_error("Failed to allocate FFI context");
        return NULL;
    }

    eshkol_runtime_init();

    /* Create JIT context — same engine used by the REPL */
    try {
        ctx->jit = new eshkol::ReplJITContext();
    } catch (const std::exception& e) {
        ffi_set_error("Failed to initialize JIT: %s", e.what());
        free(ctx);
        return NULL;
    } catch (...) {
        ffi_set_error("Failed to initialize JIT (unknown error)");
        free(ctx);
        return NULL;
    }

    /* Load stdlib for the JIT context */
    ctx->jit->loadStdlib();

    ctx->initialized = true;
    g_ffi_error[0] = 0;
    return ctx;
}

extern "C" void eshkol_ffi_shutdown(eshkol_ffi_context_t* ctx) {
    if (!ctx) return;
    if (ctx->jit) {
        delete ctx->jit;
        ctx->jit = nullptr;
    }
    if (ctx->initialized) {
        eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_REQUESTED);
    }
    free(ctx);
}

/* ============================================================================
 * Evaluation — in-process JIT via ReplJITContext
 * ============================================================================ */

extern "C" int eshkol_ffi_eval(eshkol_ffi_context_t* ctx,
                                const char* source,
                                eshkol_ffi_value_t* result) {
    if (!ctx || !ctx->initialized || !ctx->jit || !source) {
        ffi_set_error("Invalid context or source");
        return -1;
    }

    g_ffi_error[0] = 0;

    /* Wrap the last expression in (display ...) so we can capture the result.
     * This is necessary because the LLVM codegen generates main() functions
     * that return 0, not the expression value. The display output is captured
     * via pipe redirection and parsed back to a tagged value. */
    std::string wrapped_source = source;
    if (result) {
        /* Only wrap if caller wants a result */
        wrapped_source = std::string("(display ") + source + ")";
    }

    /* Parse source into ASTs */
    std::istringstream stream(wrapped_source);
    std::vector<eshkol_ast_t> asts;

    while (stream.good() && !stream.eof()) {
        /* Skip whitespace and comments */
        while (stream.good()) {
            int c = stream.peek();
            if (c == EOF) break;
            if (std::isspace(c)) {
                stream.get();
            } else if (c == ';') {
                std::string dummy;
                std::getline(stream, dummy);
            } else {
                break;
            }
        }
        if (stream.eof() || stream.peek() == EOF) break;

        eshkol_ast_t ast = eshkol_parse_next_ast_from_stream(stream);
        if (ast.type == ESHKOL_INVALID) break;
        asts.push_back(ast);
    }

    if (asts.empty()) {
        ffi_set_error("No valid expressions in source");
        return -1;
    }

    /* Execute each AST via JIT.
     * The JIT's execute() generates a main() that DISPLAYS the result
     * via eshkol_display_value. For the FFI, we capture the output
     * and parse it back to a tagged value. This is the correct approach
     * because the LLVM codegen wraps expressions in display calls. */

    /* Capture stdout */
    fflush(stdout);
    int stdout_fd = dup(STDOUT_FILENO);
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        ffi_set_error("Failed to create capture pipe");
        for (auto& a : asts) eshkol_ast_clean(&a);
        return -1;
    }
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[1]);

    /* Execute */
    for (size_t i = 0; i < asts.size(); i++) {
        try {
            ctx->jit->execute(&asts[i]);
        } catch (const std::exception& e) {
            /* Restore stdout before error */
            fflush(stdout);
            dup2(stdout_fd, STDOUT_FILENO);
            close(stdout_fd);
            close(pipefd[0]);
            ffi_set_error("JIT execution failed: %s", e.what());
            for (auto& a : asts) eshkol_ast_clean(&a);
            return -1;
        } catch (...) {
            fflush(stdout);
            dup2(stdout_fd, STDOUT_FILENO);
            close(stdout_fd);
            close(pipefd[0]);
            ffi_set_error("JIT execution failed (unknown error)");
            for (auto& a : asts) eshkol_ast_clean(&a);
            return -1;
        }
    }

    /* Read captured output */
    fflush(stdout);
    dup2(stdout_fd, STDOUT_FILENO);
    close(stdout_fd);

    char output[4096] = {0};
    ssize_t nread = read(pipefd[0], output, sizeof(output) - 1);
    close(pipefd[0]);
    if (nread > 0) output[nread] = '\0';
    else output[0] = '\0';

    /* Clean up ASTs */
    for (auto& a : asts) eshkol_ast_clean(&a);

    /* Parse output to tagged value */
    if (result) {
        /* Strip trailing newline */
        size_t len = strlen(output);
        while (len > 0 && (output[len-1] == '\n' || output[len-1] == '\r')) {
            output[--len] = '\0';
        }

        if (len == 0) {
            *result = eshkol_ffi_null();
        } else {
            /* Try to parse as number */
            char* endptr;
            double dval = strtod(output, &endptr);
            if (endptr != output && *endptr == '\0') {
                /* Numeric result */
                if (dval == (double)(int64_t)dval && strchr(output, '.') == NULL
                    && dval >= -9e18 && dval <= 9e18) {
                    *result = eshkol_ffi_int64((int64_t)dval);
                } else {
                    *result = eshkol_ffi_double(dval);
                }
            } else if (strcmp(output, "#t") == 0) {
                *result = eshkol_ffi_bool(1);
            } else if (strcmp(output, "#f") == 0) {
                *result = eshkol_ffi_bool(0);
            } else if (strcmp(output, "()") == 0) {
                *result = eshkol_ffi_null();
            } else {
                *result = eshkol_ffi_string(ctx, output);
            }
        }
    }

    return 0;
}

extern "C" int eshkol_ffi_eval_double(eshkol_ffi_context_t* ctx,
                                       const char* source,
                                       double* result) {
    eshkol_ffi_value_t val;
    int rc = eshkol_ffi_eval(ctx, source, &val);
    if (rc != 0) return rc;
    if (result) *result = eshkol_ffi_to_double(val);
    return 0;
}

extern "C" int eshkol_ffi_eval_file(eshkol_ffi_context_t* ctx, const char* path) {
    if (!ctx || !path) {
        ffi_set_error("Invalid context or path");
        return -1;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        ffi_set_error("Cannot open file: %s", path);
        return -1;
    }

    std::string source((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    return eshkol_ffi_eval(ctx, source.c_str(), NULL);
}

/* ============================================================================
 * Value Construction
 * ============================================================================ */

extern "C" eshkol_ffi_value_t eshkol_ffi_int64(int64_t value) {
    eshkol_ffi_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_FFI_TYPE_INT64;
    v.data.int_val = value;
    return v;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_double(double value) {
    eshkol_ffi_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_FFI_TYPE_DOUBLE;
    v.data.double_val = value;
    return v;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_bool(int value) {
    eshkol_ffi_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_FFI_TYPE_BOOL;
    v.data.raw_val = value ? 1 : 0;
    return v;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_null(void) {
    eshkol_ffi_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_FFI_TYPE_NULL;
    return v;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_string(eshkol_ffi_context_t* ctx,
                                                  const char* str) {
    (void)ctx;
    if (!str) return eshkol_ffi_null();

    arena_t* arena = get_global_arena();
    if (!arena) return eshkol_ffi_null();

    size_t len = strlen(str);
    char* copy = (char*)arena_allocate(arena, len + 1);
    if (!copy) return eshkol_ffi_null();
    memcpy(copy, str, len + 1);

    eshkol_ffi_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_FFI_TYPE_HEAP_PTR;
    v.flags = ESHKOL_FFI_SUBTYPE_STRING;
    v.data.ptr_val = (uint64_t)copy;
    return v;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_cons(eshkol_ffi_context_t* ctx,
                                                eshkol_ffi_value_t car,
                                                eshkol_ffi_value_t cdr) {
    (void)ctx;
    arena_t* arena = get_global_arena();
    if (!arena) return eshkol_ffi_null();

    void* cell = arena_allocate_tagged_cons_cell(arena);
    if (!cell) return eshkol_ffi_null();

    /* Copy car and cdr into the cons cell (each 16 bytes) */
    memcpy(cell, &car, sizeof(eshkol_ffi_value_t));
    memcpy((char*)cell + sizeof(eshkol_ffi_value_t), &cdr, sizeof(eshkol_ffi_value_t));

    eshkol_ffi_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_FFI_TYPE_HEAP_PTR;
    v.flags = ESHKOL_FFI_SUBTYPE_PAIR;
    v.data.ptr_val = (uint64_t)cell;
    return v;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_list(eshkol_ffi_context_t* ctx,
                                                const eshkol_ffi_value_t* values,
                                                size_t count) {
    eshkol_ffi_value_t result = eshkol_ffi_null();
    for (size_t i = count; i > 0; i--) {
        result = eshkol_ffi_cons(ctx, values[i - 1], result);
    }
    return result;
}

/* ============================================================================
 * Value Inspection
 * ============================================================================ */

extern "C" int eshkol_ffi_type(eshkol_ffi_value_t value) {
    return value.type;
}

extern "C" int64_t eshkol_ffi_to_int64(eshkol_ffi_value_t value) {
    return value.data.int_val;
}

extern "C" double eshkol_ffi_to_double(eshkol_ffi_value_t value) {
    if (value.type == ESHKOL_FFI_TYPE_INT64) {
        return (double)value.data.int_val;
    }
    return value.data.double_val;
}

extern "C" int eshkol_ffi_to_bool(eshkol_ffi_value_t value) {
    return value.data.raw_val != 0;
}

extern "C" const char* eshkol_ffi_to_string(eshkol_ffi_value_t value) {
    if (value.type != ESHKOL_FFI_TYPE_HEAP_PTR ||
        value.flags != ESHKOL_FFI_SUBTYPE_STRING) {
        return NULL;
    }
    return (const char*)value.data.ptr_val;
}

extern "C" int eshkol_ffi_is_null(eshkol_ffi_value_t value) {
    return value.type == ESHKOL_FFI_TYPE_NULL;
}

extern "C" int eshkol_ffi_is_pair(eshkol_ffi_value_t value) {
    return value.type == ESHKOL_FFI_TYPE_HEAP_PTR &&
           value.flags == ESHKOL_FFI_SUBTYPE_PAIR;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_car(eshkol_ffi_value_t pair) {
    if (!eshkol_ffi_is_pair(pair)) return eshkol_ffi_null();
    void* cell = (void*)pair.data.ptr_val;
    eshkol_ffi_value_t result;
    memcpy(&result, cell, sizeof(eshkol_ffi_value_t));
    return result;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_cdr(eshkol_ffi_value_t pair) {
    if (!eshkol_ffi_is_pair(pair)) return eshkol_ffi_null();
    void* cell = (void*)pair.data.ptr_val;
    eshkol_ffi_value_t result;
    memcpy(&result, (char*)cell + sizeof(eshkol_ffi_value_t), sizeof(eshkol_ffi_value_t));
    return result;
}

extern "C" void eshkol_ffi_display(eshkol_ffi_value_t value) {
    eshkol_tagged_value_t tv = from_ffi(value);
    eshkol_display_value(&tv);
}

/* ============================================================================
 * Tensor Operations
 * ============================================================================ */

extern "C" eshkol_ffi_value_t eshkol_ffi_tensor_zeros(eshkol_ffi_context_t* ctx,
                                                        const int64_t* shape,
                                                        int ndims) {
    (void)ctx;
    if (!shape || ndims <= 0 || ndims > 8) return eshkol_ffi_null();

    arena_t* arena = get_global_arena();
    if (!arena) return eshkol_ffi_null();

    int64_t total = 1;
    for (int i = 0; i < ndims; i++) total *= shape[i];

    eshkol_tensor_t* t = arena_allocate_tensor_full(arena, (uint64_t)ndims, (uint64_t)total);
    if (!t) return eshkol_ffi_null();

    eshkol_ffi_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_FFI_TYPE_HEAP_PTR;
    v.flags = ESHKOL_FFI_SUBTYPE_TENSOR;
    v.data.ptr_val = (uint64_t)t;
    return v;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_tensor_from_data(eshkol_ffi_context_t* ctx,
                                                            const double* data,
                                                            const int64_t* shape,
                                                            int ndims) {
    eshkol_ffi_value_t v = eshkol_ffi_tensor_zeros(ctx, shape, ndims);
    if (eshkol_ffi_is_null(v) || !data) return v;

    double* tdata = eshkol_ffi_tensor_data(v);
    int64_t total = eshkol_ffi_tensor_size(v);
    if (tdata && total > 0) {
        memcpy(tdata, data, (size_t)total * sizeof(double));
    }
    return v;
}

extern "C" double* eshkol_ffi_tensor_data(eshkol_ffi_value_t tensor) {
    if (tensor.type != ESHKOL_FFI_TYPE_HEAP_PTR ||
        tensor.flags != ESHKOL_FFI_SUBTYPE_TENSOR) return NULL;
    eshkol_tensor_t* t = (eshkol_tensor_t*)tensor.data.ptr_val;
    if (!t || !t->elements) return NULL;
    return (double*)t->elements;
}

extern "C" int64_t eshkol_ffi_tensor_size(eshkol_ffi_value_t tensor) {
    if (tensor.type != ESHKOL_FFI_TYPE_HEAP_PTR ||
        tensor.flags != ESHKOL_FFI_SUBTYPE_TENSOR) return 0;
    eshkol_tensor_t* t = (eshkol_tensor_t*)tensor.data.ptr_val;
    if (!t) return 0;
    return (int64_t)t->total_elements;
}

extern "C" int eshkol_ffi_tensor_ndims(eshkol_ffi_value_t tensor) {
    if (tensor.type != ESHKOL_FFI_TYPE_HEAP_PTR ||
        tensor.flags != ESHKOL_FFI_SUBTYPE_TENSOR) return 0;
    eshkol_tensor_t* t = (eshkol_tensor_t*)tensor.data.ptr_val;
    return t ? (int)t->num_dimensions : 0;
}

/* ============================================================================
 * Error Handling
 * ============================================================================ */

extern "C" const char* eshkol_ffi_last_error(void) {
    return g_ffi_error[0] ? g_ffi_error : NULL;
}

extern "C" void eshkol_ffi_clear_error(void) {
    g_ffi_error[0] = 0;
}
