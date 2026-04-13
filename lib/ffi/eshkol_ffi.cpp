/**
 * @file eshkol_ffi.cpp
 * @brief Implementation of the stable C FFI for Eshkol embedding.
 *
 * Wraps the REPL JIT pipeline to provide eval-from-string functionality.
 * Uses the global arena for all allocations.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <eshkol/eshkol_ffi.h>
#include <eshkol/eshkol.h>
#include <eshkol/llvm_backend.h>
#include <eshkol/core/runtime.h>
#include <eshkol/logger.h>
#include "../../lib/core/arena_memory.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

/* ── Thread-local error message ── */
static thread_local char g_ffi_error[4096] = {0};

static void ffi_set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_ffi_error, sizeof(g_ffi_error), fmt, ap);
    va_end(ap);
}

/* ── Context ── */
struct eshkol_ffi_context {
    bool initialized;
    bool stdlib_loaded;
};

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
    ctx->initialized = true;
    ctx->stdlib_loaded = false;

    g_ffi_error[0] = 0;
    return ctx;
}

extern "C" void eshkol_ffi_shutdown(eshkol_ffi_context_t* ctx) {
    if (!ctx) return;
    if (ctx->initialized) {
        eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_REQUESTED);
    }
    free(ctx);
}

/* ============================================================================
 * Evaluation
 * ============================================================================ */

extern "C" int eshkol_ffi_eval(eshkol_ffi_context_t* ctx,
                                const char* source,
                                eshkol_ffi_value_t* result) {
    if (!ctx || !ctx->initialized || !source) {
        ffi_set_error("Invalid context or source");
        return -1;
    }

    g_ffi_error[0] = 0;

    /* Parse source into ASTs */
    std::istringstream stream(source);
    std::vector<eshkol_ast_t> asts;

    eshkol_ast_t ast = eshkol_parse_next_ast_from_stream(stream);
    while (ast.type != ESHKOL_INVALID) {
        asts.push_back(ast);
        ast = eshkol_parse_next_ast_from_stream(stream);
    }

    if (asts.empty()) {
        ffi_set_error("No valid expressions in source");
        return -1;
    }

    /* Compile to LLVM IR */
    eshkol_set_source_context("<ffi>", source);
    LLVMModuleRef module = eshkol_generate_llvm_ir(
        asts.data(), asts.size(), "ffi_module");

    if (!module) {
        ffi_set_error("Failed to compile source to LLVM IR");
        return -1;
    }

    /* Compile to object and execute via temporary file */
    char tmppath[256];
    snprintf(tmppath, sizeof(tmppath), "/tmp/eshkol_ffi_%d", (int)getpid());

    int rc = eshkol_compile_llvm_ir_to_executable(module, tmppath, NULL, 0, NULL, 0);
    eshkol_dispose_llvm_module(module);

    if (rc != 0) {
        ffi_set_error("Failed to compile to executable");
        return -1;
    }

    /* Execute and capture output */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s 2>&1", tmppath);
    FILE* proc = popen(cmd, "r");
    if (!proc) {
        unlink(tmppath);
        ffi_set_error("Failed to execute compiled program");
        return -1;
    }

    char output[4096] = {0};
    size_t total = 0;
    size_t n;
    while ((n = fread(output + total, 1, sizeof(output) - total - 1, proc)) > 0)
        total += n;
    output[total] = 0;
    int status = pclose(proc);
    unlink(tmppath);

    if (status != 0) {
        ffi_set_error("Program exited with error: %s", output);
        return -1;
    }

    /* Parse output as a value */
    if (result) {
        /* Try to parse as number */
        char* endptr;
        double dval = strtod(output, &endptr);
        if (endptr != output && (*endptr == '\0' || *endptr == '\n')) {
            /* Check if it's an integer */
            if (dval == (double)(int64_t)dval && strchr(output, '.') == NULL) {
                *result = eshkol_ffi_int64((int64_t)dval);
            } else {
                *result = eshkol_ffi_double(dval);
            }
        } else if (strncmp(output, "#t", 2) == 0) {
            *result = eshkol_ffi_bool(1);
        } else if (strncmp(output, "#f", 2) == 0) {
            *result = eshkol_ffi_bool(0);
        } else {
            /* Return as string */
            *result = eshkol_ffi_string(ctx, output);
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
    memcpy(&v.data, &value, sizeof(int64_t));
    return v;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_double(double value) {
    eshkol_ffi_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_FFI_TYPE_DOUBLE;
    memcpy(&v.data, &value, sizeof(double));
    return v;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_bool(int value) {
    eshkol_ffi_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_FFI_TYPE_BOOL;
    v.data = value ? 1 : 0;
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

    /* Allocate string in global arena */
    arena_t* arena = get_global_arena();
    if (!arena) return eshkol_ffi_null();

    size_t len = strlen(str);
    char* copy = (char*)arena_allocate(arena, len + 1);
    if (!copy) return eshkol_ffi_null();
    memcpy(copy, str, len + 1);

    eshkol_ffi_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_FFI_TYPE_HEAP_PTR;
    v.flags = 0x01; /* string subtype marker */
    memcpy(&v.data, &copy, sizeof(char*));
    return v;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_cons(eshkol_ffi_context_t* ctx,
                                                eshkol_ffi_value_t car,
                                                eshkol_ffi_value_t cdr) {
    (void)ctx;
    arena_t* arena = get_global_arena();
    if (!arena) return eshkol_ffi_null();

    /* Allocate tagged cons cell */
    void* cell = arena_allocate_tagged_cons_cell(arena);
    if (!cell) return eshkol_ffi_null();

    /* Copy car and cdr into the cons cell */
    memcpy(cell, &car, sizeof(eshkol_ffi_value_t));
    memcpy((char*)cell + sizeof(eshkol_ffi_value_t), &cdr, sizeof(eshkol_ffi_value_t));

    eshkol_ffi_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_FFI_TYPE_HEAP_PTR;
    v.flags = 0x02; /* pair subtype marker */
    memcpy(&v.data, &cell, sizeof(void*));
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
    int64_t result;
    memcpy(&result, &value.data, sizeof(int64_t));
    return result;
}

extern "C" double eshkol_ffi_to_double(eshkol_ffi_value_t value) {
    if (value.type == ESHKOL_FFI_TYPE_INT64) {
        return (double)eshkol_ffi_to_int64(value);
    }
    double result;
    memcpy(&result, &value.data, sizeof(double));
    return result;
}

extern "C" int eshkol_ffi_to_bool(eshkol_ffi_value_t value) {
    return value.data != 0;
}

extern "C" int eshkol_ffi_is_null(eshkol_ffi_value_t value) {
    return value.type == ESHKOL_FFI_TYPE_NULL;
}

extern "C" int eshkol_ffi_is_pair(eshkol_ffi_value_t value) {
    return value.type == ESHKOL_FFI_TYPE_HEAP_PTR && value.flags == 0x02;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_car(eshkol_ffi_value_t pair) {
    if (!eshkol_ffi_is_pair(pair)) return eshkol_ffi_null();
    void* cell;
    memcpy(&cell, &pair.data, sizeof(void*));
    eshkol_ffi_value_t result;
    memcpy(&result, cell, sizeof(eshkol_ffi_value_t));
    return result;
}

extern "C" eshkol_ffi_value_t eshkol_ffi_cdr(eshkol_ffi_value_t pair) {
    if (!eshkol_ffi_is_pair(pair)) return eshkol_ffi_null();
    void* cell;
    memcpy(&cell, &pair.data, sizeof(void*));
    eshkol_ffi_value_t result;
    memcpy(&result, (char*)cell + sizeof(eshkol_ffi_value_t), sizeof(eshkol_ffi_value_t));
    return result;
}

extern "C" void eshkol_ffi_display(eshkol_ffi_value_t value) {
    switch (value.type) {
        case ESHKOL_FFI_TYPE_NULL: printf("()"); break;
        case ESHKOL_FFI_TYPE_INT64: printf("%lld", (long long)eshkol_ffi_to_int64(value)); break;
        case ESHKOL_FFI_TYPE_DOUBLE: printf("%g", eshkol_ffi_to_double(value)); break;
        case ESHKOL_FFI_TYPE_BOOL: printf("%s", value.data ? "#t" : "#f"); break;
        default: printf("<value type=%d>", value.type); break;
    }
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
    v.flags = 0x03; /* tensor subtype marker */
    memcpy(&v.data, &t, sizeof(void*));
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
    if (tensor.type != ESHKOL_FFI_TYPE_HEAP_PTR || tensor.flags != 0x03) return NULL;
    eshkol_tensor_t* t;
    memcpy(&t, &tensor.data, sizeof(void*));
    if (!t || !t->elements) return NULL;
    /* Elements stored as int64 bit patterns of doubles */
    return (double*)t->elements;
}

extern "C" int64_t eshkol_ffi_tensor_size(eshkol_ffi_value_t tensor) {
    if (tensor.type != ESHKOL_FFI_TYPE_HEAP_PTR || tensor.flags != 0x03) return 0;
    eshkol_tensor_t* t;
    memcpy(&t, &tensor.data, sizeof(void*));
    if (!t) return 0;
    return (int64_t)t->total_elements;
}

extern "C" int eshkol_ffi_tensor_ndims(eshkol_ffi_value_t tensor) {
    if (tensor.type != ESHKOL_FFI_TYPE_HEAP_PTR || tensor.flags != 0x03) return 0;
    eshkol_tensor_t* t;
    memcpy(&t, &tensor.data, sizeof(void*));
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
