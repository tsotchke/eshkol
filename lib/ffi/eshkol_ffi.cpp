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
#include <eshkol/core/sexp_to_ast.h>
#include "../../lib/core/arena_memory.h"
#include "../../lib/repl/repl_jit.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cerrno>
#include <csetjmp>
#include <fstream>
#include <sstream>
#include <string>
#ifndef _WIN32
#include <fcntl.h>    /* O_RDONLY, O_NOFOLLOW, O_CLOEXEC (audit H8) */
#endif
#include <vector>
#include <memory>
#ifndef _WIN32
#include <unistd.h>
#endif

/* ── Forward declarations for runtime helpers not in the public header ── */
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t* out);

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

/* Unique-name counter for FFI result globals. The JIT entry function returns
 * an i32 exit code, not the expression value, so to capture the value we wrap
 * the last AST in (define __eshkol_ffi_result_N__ <expr>) and read the stored
 * tagged_value out of that named global after execution. Thread-safety: atomic
 * increment — concurrent FFI calls do not collide on counter values. */
static std::atomic<uint64_t> g_ffi_result_counter{0};

extern "C" int eshkol_ffi_eval(eshkol_ffi_context_t* ctx,
                                const char* source,
                                eshkol_ffi_value_t* result) {
    /* Zero-initialise *result up front so every early-return error path
     * leaves the caller's slot with a defined null tagged value. */
    if (result) {
        *result = eshkol_ffi_null();
    }
    if (!ctx || !ctx->initialized || !ctx->jit || !source) {
        ffi_set_error("Invalid context or source");
        return -1;
    }

    g_ffi_error[0] = 0;

    /* Parse source into ASTs. All but the last are executed for side
     * effects; the last is what the caller wants the value of (if a
     * result slot was provided). */
    std::istringstream stream(source);
    std::vector<eshkol_ast_t> asts;

    while (stream.good() && !stream.eof()) {
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

    /* Determine which AST carries the value we want to return. If there is
     * no result slot, every AST is executed purely for effect. Otherwise the
     * LAST AST is wrapped in a unique (define __eshkol_ffi_result_N__ ...)
     * so the JIT-emitted main stores its value into a named global that we
     * can read back via lookupSymbol. This is the same pattern used by
     * eshkol_eval (introspection.cpp:727); copying it here keeps the FFI
     * aligned with the Scheme-level eval and avoids the old display+reparse
     * round-trip that lost type information (lists → "(1 2 3)" strings). */
    const size_t last = asts.size() - 1;
    eshkol_ast_t* wrapper = nullptr;
    std::string result_name;

    /* Some top-level forms (define / import / require / provide) have no
     * expression value — don't wrap them, the wrapper would be semantically
     * invalid. */
    bool last_is_valueless = false;
    if (asts[last].type == ESHKOL_OP) {
        uint64_t op = asts[last].operation.op;
        if (op == ESHKOL_DEFINE_OP || op == ESHKOL_IMPORT_OP ||
            op == ESHKOL_REQUIRE_OP || op == ESHKOL_PROVIDE_OP) {
            last_is_valueless = true;
        }
    }

    if (result && !last_is_valueless) {
        uint64_t id = g_ffi_result_counter.fetch_add(1, std::memory_order_relaxed);
        char name_buf[64];
        snprintf(name_buf, sizeof(name_buf), "__eshkol_ffi_result_%llu__",
                 (unsigned long long)id);
        result_name = name_buf;

        wrapper = eshkol_alloc_symbolic_ast();
        if (!wrapper) {
            ffi_set_error("Failed to allocate FFI result wrapper AST");
            for (auto& a : asts) eshkol_ast_clean(&a);
            return -1;
        }
        /* The wrapper owns a heap-allocated copy of the original AST so the
         * caller's vector-cleanup path stays intact. */
        eshkol_ast_t* inner = eshkol_alloc_symbolic_ast();
        if (!inner) {
            eshkol_free_sexp_ast(wrapper);
            ffi_set_error("Failed to allocate FFI inner AST");
            for (auto& a : asts) eshkol_ast_clean(&a);
            return -1;
        }
        *inner = asts[last];
        /* Transfer ownership: clear the slot in the vector so
         * eshkol_ast_clean below doesn't double-free the subtree. */
        std::memset(&asts[last], 0, sizeof(eshkol_ast_t));

        wrapper->type = ESHKOL_OP;
        wrapper->operation.op = ESHKOL_DEFINE_OP;
        wrapper->operation.define_op.name = strdup(name_buf);
        wrapper->operation.define_op.value = inner;
        wrapper->operation.define_op.is_function = 0;
        wrapper->operation.define_op.parameters = nullptr;
        wrapper->operation.define_op.num_params = 0;
        wrapper->operation.define_op.is_variadic = 0;
        wrapper->operation.define_op.rest_param = nullptr;
        wrapper->operation.define_op.is_external = 0;
        wrapper->operation.define_op.return_type = nullptr;
        wrapper->operation.define_op.param_types = nullptr;
    }

    /* Execute all preceding ASTs for side effects, plus the last one (either
     * as-is if no result is wanted, or through the DEFINE wrapper).
     *
     * Install a top-level exception handler so any unhandled Eshkol raise
     * longjmps back here instead of calling exit(1) and killing the Python
     * host process. On setjmp(...)==1 path, pull the raised value via
     * eshkol_get_raised_value and translate to a C FFI error + -1 return. */
    jmp_buf ffi_jmp;
    bool unwound = false;
    if (setjmp(ffi_jmp) != 0) {
        /* Control returned from longjmp inside eshkol_raise. The Eshkol
         * runtime pushed our handler and is about to exit(1); instead it
         * jumped here. We do NOT pop here — longjmp bypasses normal
         * unwinding, and the handler stack above ours may be in an
         * inconsistent state. Clear the handler stack back to our saved
         * top so subsequent FFI calls start fresh. */
        unwound = true;
    }

    if (!unwound) {
        eshkol_push_exception_handler(&ffi_jmp);
    }

    if (!unwound) {
        for (size_t i = 0; i < asts.size(); i++) {
            eshkol_ast_t* to_run = (i == last && wrapper) ? wrapper : &asts[i];
            if (to_run->type == ESHKOL_INVALID) continue;
            try {
                ctx->jit->execute(to_run);
            } catch (const std::exception& e) {
                eshkol_pop_exception_handler();
                ffi_set_error("JIT execution failed: %s", e.what());
                for (auto& a : asts) eshkol_ast_clean(&a);
                if (wrapper) eshkol_free_sexp_ast(wrapper);
                return -1;
            } catch (...) {
                eshkol_pop_exception_handler();
                ffi_set_error("JIT execution failed (unknown error)");
                for (auto& a : asts) eshkol_ast_clean(&a);
                if (wrapper) eshkol_free_sexp_ast(wrapper);
                return -1;
            }
        }
        eshkol_pop_exception_handler();
    } else {
        /* After longjmp the handler was consumed by the raise path; the
         * current top-of-stack is whatever was below us before the push.
         * Extract the raised value to build a descriptive error. */
        eshkol_tagged_value_t raised;
        std::memset(&raised, 0, sizeof(raised));
        eshkol_get_raised_value(&raised);
        const char* msg = "Eshkol runtime exception";
        if (raised.type == ESHKOL_VALUE_HEAP_PTR && raised.data.ptr_val) {
            /* If the raised value is an exception struct, dig out the message. */
            eshkol_exception_t* exc =
                reinterpret_cast<eshkol_exception_t*>(raised.data.ptr_val);
            if (exc && exc->message) msg = exc->message;
        }
        ffi_set_error("Eshkol exception: %s", msg);
        for (auto& a : asts) eshkol_ast_clean(&a);
        if (wrapper) eshkol_free_sexp_ast(wrapper);
        return -1;
    }

    /* If we wrapped the last AST, read back the stored tagged_value. */
    if (wrapper) {
        uint64_t addr = ctx->jit->lookupSymbol(result_name);
        if (addr != 0 && result) {
            eshkol_tagged_value_t tv;
            std::memcpy(&tv, reinterpret_cast<void*>(addr), sizeof(tv));
            *result = to_ffi(tv);
        }
        eshkol_free_sexp_ast(wrapper);
    }

    /* Clean up the non-wrapped ASTs. */
    for (auto& a : asts) eshkol_ast_clean(&a);

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

    /* Audit H8: open with O_NOFOLLOW so a symlink can't redirect the
     * read at `path` to an unrelated file (e.g. /etc/shadow) between
     * the caller's permission check and the actual open. std::ifstream
     * doesn't expose open flags, so route through fopen(fd) with a
     * pre-opened fd that has O_NOFOLLOW applied. On Windows there's
     * no symlink-follow by default in the same way; fall through to
     * the plain ifstream path there. */
#ifndef _WIN32
    int fd = ::open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        /* errno == ELOOP on Linux and EMLINK on macOS when NOFOLLOW
         * blocks a symlink; report generically so we don't leak which
         * branch was hit. */
        ffi_set_error("Cannot open file: %s", path);
        return -1;
    }
    FILE* fp = ::fdopen(fd, "rb");
    if (!fp) {
        ::close(fd);
        ffi_set_error("Cannot open file: %s", path);
        return -1;
    }
    std::string source;
    char buf[8192];
    size_t n;
    while ((n = ::fread(buf, 1, sizeof(buf), fp)) > 0) {
        source.append(buf, n);
    }
    ::fclose(fp);
#else
    std::ifstream file(path);
    if (!file.is_open()) {
        ffi_set_error("Cannot open file: %s", path);
        return -1;
    }
    std::string source((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
#endif
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

/* Inspect the HEAP_SUBTYPE byte from an object's header (at ptr-8), returning
 * 0xFF if the pointer is null so callers can distinguish "no header" cleanly.
 * All consolidated runtime values (JIT-returned cons, strings, tensors, ...)
 * carry their subtype in this header. FFI-constructed values from
 * eshkol_ffi_cons / eshkol_ffi_tensor_zeros additionally encode the subtype
 * into the legacy `flags` byte so old callers that only checked flags still
 * work. Detection below prefers the header when a non-null pointer is
 * present; this keeps the runtime and FFI views of the world in sync. */
static inline uint8_t ffi_header_subtype(eshkol_ffi_value_t value) {
    if (value.type != ESHKOL_FFI_TYPE_HEAP_PTR) return 0xFF;
    if (value.data.ptr_val == 0) return 0xFF;
    uint8_t* hdr = (uint8_t*)(uintptr_t)value.data.ptr_val - 8;
    return hdr[0];  // subtype is the first byte of eshkol_object_header_t
}

extern "C" const char* eshkol_ffi_to_string(eshkol_ffi_value_t value) {
    if (value.type != ESHKOL_FFI_TYPE_HEAP_PTR) return NULL;
    uint8_t st = ffi_header_subtype(value);
    bool is_str = (value.flags == ESHKOL_FFI_SUBTYPE_STRING) ||
                  (st == 1 /* HEAP_SUBTYPE_STRING */);
    if (!is_str) return NULL;
    return (const char*)value.data.ptr_val;
}

extern "C" int eshkol_ffi_is_null(eshkol_ffi_value_t value) {
    return value.type == ESHKOL_FFI_TYPE_NULL;
}

extern "C" int eshkol_ffi_is_pair(eshkol_ffi_value_t value) {
    if (value.type != ESHKOL_FFI_TYPE_HEAP_PTR) return 0;
    uint8_t st = ffi_header_subtype(value);
    return (value.flags == ESHKOL_FFI_SUBTYPE_PAIR) ||
           (st == 0 /* HEAP_SUBTYPE_CONS */);
}

extern "C" eshkol_ffi_value_t eshkol_ffi_car(eshkol_ffi_value_t pair) {
    if (!eshkol_ffi_is_pair(pair)) return eshkol_ffi_null();
    /* Both FFI-side and runtime-side cons cells use 32-byte
     * arena_tagged_cons_cell_t layout: [car:16][cdr:16]. Safe to read
     * directly as a pair of eshkol_ffi_value_t (same 16-byte layout). */
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

/* Accept either the legacy flags byte (=ESHKOL_FFI_SUBTYPE_TENSOR) or the
 * header-subtype byte (=HEAP_SUBTYPE_TENSOR=3), the same dual-detection we
 * use for pairs and strings. JIT-returned tensors only have the header;
 * FFI-constructed tensors have both. */
static inline bool ffi_is_tensor(eshkol_ffi_value_t v) {
    if (v.type != ESHKOL_FFI_TYPE_HEAP_PTR) return false;
    uint8_t st = ffi_header_subtype(v);
    return (v.flags == ESHKOL_FFI_SUBTYPE_TENSOR) ||
           (st == 3 /* HEAP_SUBTYPE_TENSOR */);
}

extern "C" double* eshkol_ffi_tensor_data(eshkol_ffi_value_t tensor) {
    if (!ffi_is_tensor(tensor)) return NULL;
    eshkol_tensor_t* t = (eshkol_tensor_t*)tensor.data.ptr_val;
    if (!t || !t->elements) return NULL;
    return (double*)t->elements;
}

extern "C" int64_t eshkol_ffi_tensor_size(eshkol_ffi_value_t tensor) {
    if (!ffi_is_tensor(tensor)) return 0;
    eshkol_tensor_t* t = (eshkol_tensor_t*)tensor.data.ptr_val;
    if (!t) return 0;
    return (int64_t)t->total_elements;
}

extern "C" int eshkol_ffi_tensor_ndims(eshkol_ffi_value_t tensor) {
    if (!ffi_is_tensor(tensor)) return 0;
    eshkol_tensor_t* t = (eshkol_tensor_t*)tensor.data.ptr_val;
    return t ? (int)t->num_dimensions : 0;
}

extern "C" int eshkol_ffi_tensor_shape(eshkol_ffi_value_t tensor,
                                        int64_t* out_shape,
                                        int max_ndims) {
    if (!ffi_is_tensor(tensor) || !out_shape || max_ndims <= 0) return 0;
    eshkol_tensor_t* t = (eshkol_tensor_t*)tensor.data.ptr_val;
    if (!t || !t->dimensions) return 0;
    int n = (int)t->num_dimensions;
    if (n > max_ndims) n = max_ndims;
    for (int i = 0; i < n; i++) {
        out_shape[i] = (int64_t)t->dimensions[i];
    }
    return n;
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
