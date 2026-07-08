/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted display and current port runtime support.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/core/bignum.h"
#include "../../inc/eshkol/core/logic.h"
#include "../../inc/eshkol/core/inference.h"
#include "../../inc/eshkol/core/workspace.h"
#include "../../inc/eshkol/core/rational.h"

#include <cstdio>
#include <cstring>
#include <cmath>

// ===== R7RS FLONUM EXTERNAL REPRESENTATION =====
// Single source of truth for rendering an IEEE double as text. R7RS (6.2.6 /
// 7.1.1) mandates the special external representations +inf.0 / -inf.0 /
// +nan.0 for the non-finite flonums; plain "%g" (which prints "inf"/"nan")
// is NOT readable back by the reader. Every double-formatting site — display,
// write, number->string, logic-term printing — routes through this so the
// representation is consistent everywhere it matters.
//
// Finite values keep the existing "%g" shortest-round-trip-ish convention to
// avoid perturbing the large body of existing output.
extern "C" int eshkol_format_double(char* buf, size_t n, double v) {
    if (std::isnan(v)) {
        return snprintf(buf, n, "+nan.0");
    }
    if (std::isinf(v)) {
        return snprintf(buf, n, v < 0.0 ? "-inf.0" : "+inf.0");
    }
    return snprintf(buf, n, "%g", v);
}

// Convenience wrapper: print a double to a FILE* in R7RS external form.
// Typed as void* in the public header so eshkol.h need not pull in <cstdio>.
extern "C" void eshkol_fprint_double(void* file, double v) {
    char tmp[64];
    eshkol_format_double(tmp, sizeof(tmp), v);
    fputs(tmp, file ? (FILE*)file : stdout);
}

// ===== UNIFIED DISPLAY IMPLEMENTATION =====
// Single source of truth for displaying all Eshkol values

// Forward declarations for internal helpers
static void display_tensor(uint64_t tensor_ptr, eshkol_display_opts_t* opts);
static void display_vector(uint64_t vector_ptr, eshkol_display_opts_t* opts);
static void display_char(uint32_t codepoint, eshkol_display_opts_t* opts);

// R7RS `write` string external representation: wrap in double quotes and
// escape the characters the reader would otherwise mis-parse. `\\` and `\"`
// are mandatory; `\a \b \t \n \r` are the named mnemonic escapes; every
// other control byte is emitted as an inline hex escape `\xNN;`. Bytes >=
// 0x80 (UTF-8 continuation/lead bytes) pass through untouched so multi-byte
// characters round-trip. `display` never calls this — only `write`.
static void write_escaped_string(FILE* out, const char* data, size_t len) {
    fputc('"', out);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)data[i];
        switch (c) {
            case '"':  fputs("\\\"", out); break;
            case '\\': fputs("\\\\", out); break;
            case '\a': fputs("\\a", out); break;
            case '\b': fputs("\\b", out); break;
            case '\t': fputs("\\t", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            default:
                if (c < 0x20 || c == 0x7f) {
                    fprintf(out, "\\x%x;", (unsigned)c);
                } else {
                    fputc((int)c, out);
                }
                break;
        }
    }
    fputc('"', out);
}

// ─── R7RS current-output-port / current-input-port / current-error-port ───
// The cells back the Scheme-level `current-output-port` / etc. parameter
// objects. `parameterize` mutates them via the setter; runtime helpers
// (display/write/newline with no explicit port arg) read them.
//
// Default to NULL → fall back to stdio in get_output(). Lazy default so
// constructors run before any FILE* is portable to capture (Windows DLLs).
static FILE* g_current_output_fp = nullptr;
static FILE* g_current_input_fp  = nullptr;
static FILE* g_current_error_fp  = nullptr;

/** @brief Return the current-output-port FILE*, defaulting to stdout when unset. */
extern "C" void* eshkol_runtime_current_output_fp(void) {
    return (void*)(g_current_output_fp ? g_current_output_fp : stdout);
}
/** @brief Return the current-input-port FILE*, defaulting to stdin when unset. */
extern "C" void* eshkol_runtime_current_input_fp(void) {
    return (void*)(g_current_input_fp ? g_current_input_fp : stdin);
}
/** @brief Return the current-error-port FILE*, defaulting to stderr when unset. */
extern "C" void* eshkol_runtime_current_error_fp(void) {
    return (void*)(g_current_error_fp ? g_current_error_fp : stderr);
}
/**
 * @brief Set the FILE* backing `current-output-port`.
 *
 * Called by the `parameterize` machinery when rebinding the port parameter;
 * a null `fp` reverts subsequent reads to the stdio default.
 * @param fp  New output FILE* (may be null).
 */
extern "C" void eshkol_runtime_set_current_output_fp(void* fp) {
    g_current_output_fp = (FILE*)fp;
}

// Helper used by `(string ch ch ...)` and similar: encode a single codepoint
// as UTF-8 into `out`, returning the number of bytes written (1..4). The
// pre-fix codegen truncated each codepoint to one byte, mangling every
// non-ASCII char in `(string ch …)` and `(string-append (string ch) …)` —
// causing Quirk 15 (Unicode block chars hung the bench output).
static int eshkol_utf8_encode_one(uint32_t cp, char* out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 |  (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 |  (cp & 0x3F));
        return 3;
    } else if (cp < 0x110000) {
        out[0] = (char)(0xF0 |  (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 |  (cp & 0x3F));
        return 4;
    }
    // Out-of-range codepoint — encode replacement char U+FFFD.
    out[0] = (char)0xEF; out[1] = (char)0xBF; out[2] = (char)0xBD;
    return 3;
}

// Build a HEAP_SUBTYPE_STRING from N codepoints, encoded as UTF-8.
// Caller passes `n` codepoints; we compute the total byte length, allocate
// (n bytes plus a few for multi-byte expansion plus null terminator), and
// fill. Returns the arena-allocated, null-terminated UTF-8 char* — same
// shape as any other Eshkol string, so display / fputs work unchanged.
extern "C" void* eshkol_string_from_codepoints(void* arena_void,
                                               const int64_t* codepoints,
                                               uint64_t n) {
    arena_t* arena = (arena_t*)arena_void;
    if (!arena || (!codepoints && n > 0)) return nullptr;

    // Compute exact byte length first so we don't waste arena space.
    size_t bytes = 0;
    for (uint64_t i = 0; i < n; i++) {
        uint32_t cp = (uint32_t)codepoints[i];
        if      (cp < 0x80)     bytes += 1;
        else if (cp < 0x800)    bytes += 2;
        else if (cp < 0x10000)  bytes += 3;
        else if (cp < 0x110000) bytes += 4;
        else                    bytes += 3;  // U+FFFD replacement
    }

    char* str = (char*)arena_allocate_with_header(
        arena, bytes + 1, HEAP_SUBTYPE_STRING, 0);
    if (!str) return nullptr;

    char* p = str;
    for (uint64_t i = 0; i < n; i++) {
        p += eshkol_utf8_encode_one((uint32_t)codepoints[i], p);
    }
    *p = '\0';
    return (void*)str;
}
/**
 * @brief Set the FILE* backing `current-input-port`.
 * @param fp  New input FILE* (may be null to revert to the stdio default).
 */
extern "C" void eshkol_runtime_set_current_input_fp(void* fp) {
    g_current_input_fp = (FILE*)fp;
}
/**
 * @brief Set the FILE* backing `current-error-port`.
 * @param fp  New error FILE* (may be null to revert to the stdio default).
 */
extern "C" void eshkol_runtime_set_current_error_fp(void* fp) {
    g_current_error_fp = (FILE*)fp;
}

// Get output stream (defaults to stdout via current-output-port parameter)
static FILE* get_output(eshkol_display_opts_t* opts) {
    if (opts && opts->output) {
        return (FILE*)opts->output;
    }
    return (FILE*)eshkol_runtime_current_output_fp();
}

// Display a single tagged value
void eshkol_display_value(const eshkol_tagged_value_t* value) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    eshkol_display_value_opts(value, &opts);
}

/**
 * @brief Core dispatcher that renders one tagged value to `opts->output` (or the
 * current-output-port default).
 *
 * Switches on the value's base type (masking off exactness flags for immediate
 * types, using the raw tag for consolidated HEAP_PTR/CALLABLE/legacy types) and
 * recurses into the appropriate helper: lists, vectors, tensors, closures,
 * lambdas, bignums, rationals, complex numbers, logic-engine objects, etc.
 * Honors `opts->quote_strings` (write vs. display semantics), `opts->max_depth`
 * / `opts->current_depth` (truncates with "..." past the depth limit), and
 * `opts->show_types` for the unknown-type fallback. This is the single
 * recursion point every other display helper in this file calls back into.
 *
 * @param value  Tagged value to render; a null pointer prints "()".
 * @param opts   Display options (output port, depth state, quoting); must be non-null.
 */
void eshkol_display_value_opts(const eshkol_tagged_value_t* value, eshkol_display_opts_t* opts) {
    if (!value) {
        fprintf(get_output(opts), "()");
        return;
    }

    // Check depth limit
    if (opts->current_depth > opts->max_depth) {
        fprintf(get_output(opts), "...");
        return;
    }

    uint8_t full_type = value->type;
    // Compute base type correctly:
    // - Legacy types (>= 32): use full_type directly, no masking
    // - Consolidated (8-15) and multimedia (16-31): use full_type directly
    // - Immediate types (< 8): might have exactness flags, mask with 0x0F
    uint8_t base_type;
    if (full_type >= 32) {
        base_type = full_type;  // Legacy types: CONS_PTR=32, STRING_PTR=33, etc.
    } else if (full_type >= 8) {
        base_type = full_type;  // Consolidated/multimedia: HEAP_PTR=8, HANDLE=16, etc.
    } else {
        base_type = full_type & 0x0F;  // Immediate types: strip exactness flags
    }

    // Check for port types BEFORE the switch (they use special flag encoding)
    // Input port: CONS_PTR | 0x10 = 48
    // Output port: CONS_PTR | 0x40 = 96 (NOT 0x20! CONS_PTR=32=0x20)
    if (full_type == (ESHKOL_VALUE_CONS_PTR | 0x10)) {
        FILE* fp = (FILE*)value->data.ptr_val;
        int fd = fp ? fileno(fp) : -1;
        fprintf(get_output(opts), "#<input-port fd:%d>", fd);
        return;
    }
    if (full_type == (ESHKOL_VALUE_CONS_PTR | 0x40)) {
        FILE* fp = (FILE*)value->data.ptr_val;
        int fd = fp ? fileno(fp) : -1;
        fprintf(get_output(opts), "#<output-port fd:%d>", fd);
        return;
    }

    switch (base_type) {
        case ESHKOL_VALUE_NULL:
            fprintf(get_output(opts), "()");
            break;

        case ESHKOL_VALUE_HEAP_PTR: {
            // Consolidated heap pointer - read subtype from object header
            void* data_ptr = (void*)value->data.ptr_val;
            if (!data_ptr) {
                fprintf(get_output(opts), "()");
                break;
            }
            eshkol_object_header_t* header = ESHKOL_GET_HEADER(data_ptr);
            switch (header->subtype) {
                case HEAP_SUBTYPE_CONS:
                    eshkol_display_list(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_STRING: {
                    // Use the header's recorded byte length, NOT strlen, so
                    // that strings containing embedded NUL bytes are emitted
                    // in full. `arena_allocate_string_with_header` stores
                    // `length + 1` (including the trailing NUL terminator)
                    // in `header->size`, so subtract one to get the payload
                    // byte count.  Without this, `(display rgb-bytes
                    // binary-port)` truncates at the first 0x00 byte —
                    // blocking P6 PPM raw-output and any other binary
                    // protocol that pipes through `display`.
                    size_t payload = (header->size > 0) ? (size_t)header->size - 1 : 0;
                    FILE* out = get_output(opts);
                    if (opts->quote_strings) {
                        write_escaped_string(out, (const char*)data_ptr, payload);
                    } else {
                        if (payload > 0) fwrite(data_ptr, 1, payload, out);
                    }
                    break;
                }
                case HEAP_SUBTYPE_SYMBOL:
                    // Display symbol name without quotes (homoiconic representation)
                    fprintf(get_output(opts), "%s", (const char*)data_ptr);
                    break;
                case HEAP_SUBTYPE_VECTOR:
                    display_vector(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_TENSOR:
                    display_tensor(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_HASH:
                    fprintf(get_output(opts), "#<hash>");
                    break;
                case HEAP_SUBTYPE_EXCEPTION:
                    fprintf(get_output(opts), "#<exception>");
                    break;
                case HEAP_SUBTYPE_MULTI_VALUE:
                    fprintf(get_output(opts), "#<values>");
                    break;
                case HEAP_SUBTYPE_BIGNUM:
                    eshkol_bignum_display((const eshkol_bignum_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_SUBSTITUTION:
                    eshkol_display_substitution((const eshkol_substitution_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_FACT:
                    eshkol_display_fact((const eshkol_fact_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_KNOWLEDGE_BASE:
                    eshkol_display_kb((const eshkol_knowledge_base_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_FACTOR_GRAPH:
                    eshkol_display_factor_graph((const eshkol_factor_graph_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_WORKSPACE:
                    eshkol_display_workspace((const eshkol_workspace_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_PROMISE:
                    fprintf(get_output(opts), "#<promise>");
                    break;
                case HEAP_SUBTYPE_RATIONAL: {
                    eshkol_rational_t* rat = (eshkol_rational_t*)data_ptr;
                    fprintf(get_output(opts), "%lld/%lld",
                        (long long)rat->numerator, (long long)rat->denominator);
                    break;
                }
                default:
                    fprintf(get_output(opts), "#<heap:%d>", header->subtype);
                    break;
            }
            break;
        }

        case ESHKOL_VALUE_CALLABLE: {
            // Consolidated callable - read subtype from object header
            void* data_ptr = (void*)value->data.ptr_val;
            if (!data_ptr) {
                fprintf(get_output(opts), "#<procedure>");
                break;
            }
            eshkol_object_header_t* header = ESHKOL_GET_HEADER(data_ptr);
            switch (header->subtype) {
                case CALLABLE_SUBTYPE_CLOSURE:
                    eshkol_display_closure(value->data.ptr_val, opts);
                    break;
                case CALLABLE_SUBTYPE_LAMBDA_SEXPR:
                    eshkol_display_lambda(value->data.ptr_val, opts);
                    break;
                case CALLABLE_SUBTYPE_AD_NODE:
                    fprintf(get_output(opts), "#<ad-node>");
                    break;
                case CALLABLE_SUBTYPE_PRIMITIVE:
                    fprintf(get_output(opts), "#<primitive>");
                    break;
                case CALLABLE_SUBTYPE_CONTINUATION:
                    fprintf(get_output(opts), "#<continuation>");
                    break;
                default:
                    fprintf(get_output(opts), "#<callable:%d>", header->subtype);
                    break;
            }
            break;
        }

        case ESHKOL_VALUE_LOGIC_VAR:
            eshkol_display_logic_var(value->data.int_val, get_output(opts));
            break;

        case ESHKOL_VALUE_INT64:
            fprintf(get_output(opts), "%lld", (long long)value->data.int_val);
            break;

        case ESHKOL_VALUE_DOUBLE:
            eshkol_fprint_double(get_output(opts), value->data.double_val);
            break;

        case ESHKOL_VALUE_BOOL:
            fprintf(get_output(opts), "%s", value->data.int_val ? "#t" : "#f");
            break;

        case ESHKOL_VALUE_CHAR:
            display_char((uint32_t)value->data.int_val, opts);
            break;

        case ESHKOL_VALUE_STRING_PTR:
            if (opts->quote_strings) {
                const char* s = (const char*)value->data.ptr_val;
                write_escaped_string(get_output(opts), s, s ? strlen(s) : 0);
            } else {
                fprintf(get_output(opts), "%s", (const char*)value->data.ptr_val);
            }
            break;

        case ESHKOL_VALUE_SYMBOL:
            fprintf(get_output(opts), "%s", (const char*)value->data.ptr_val);
            break;

        case ESHKOL_VALUE_CONS_PTR:
            eshkol_display_list(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_LAMBDA_SEXPR:
            eshkol_display_lambda(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_CLOSURE_PTR:
            eshkol_display_closure(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_TENSOR_PTR:
            display_tensor(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_VECTOR_PTR:
            display_vector(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_DUAL_NUMBER: {
            eshkol_dual_number_t* dual = (eshkol_dual_number_t*)value->data.ptr_val;
            if (dual) {
                fprintf(get_output(opts), "(dual %g %g)", dual->value, dual->derivative);
            } else {
                fprintf(get_output(opts), "(dual 0 0)");
            }
            break;
        }

        case ESHKOL_VALUE_COMPLEX: {
            // Complex number: data.ptr_val points to {double real, double imag}
            double* parts = (double*)value->data.ptr_val;
            if (parts) {
                double re = parts[0];
                double im = parts[1];
                FILE* cf = get_output(opts);
                char rbuf[64];
                if (im == 0.0) {
                    // Purely real complex: print just the real part.
                    eshkol_format_double(rbuf, sizeof(rbuf), re);
                    fputs(rbuf, cf);
                } else {
                    // R7RS external repr: <real>±<imag>i, with the real part
                    // omitted when it is zero and the ±i shorthand for ±1.
                    if (re != 0.0) {
                        eshkol_format_double(rbuf, sizeof(rbuf), re);
                        fputs(rbuf, cf);
                    }
                    if (im == 1.0) {
                        fputs("+i", cf);
                    } else if (im == -1.0) {
                        fputs("-i", cf);
                    } else {
                        char ibuf[64];
                        eshkol_format_double(ibuf, sizeof(ibuf), im);
                        // Guarantee an explicit sign on the imaginary part.
                        if (ibuf[0] != '+' && ibuf[0] != '-') {
                            fputc('+', cf);
                        }
                        fputs(ibuf, cf);
                        fputc('i', cf);
                    }
                }
            } else {
                fprintf(get_output(opts), "0");
            }
            break;
        }

        case ESHKOL_VALUE_AD_NODE_PTR:
            fprintf(get_output(opts), "#<ad-node>");
            break;

        case ESHKOL_VALUE_HASH_PTR: {
            eshkol_hash_table_t* ht = (eshkol_hash_table_t*)value->data.ptr_val;
            if (!ht) {
                fprintf(get_output(opts), "#<hash:0>");
            } else {
                fprintf(get_output(opts), "#<hash:%zu>", ht->size);
            }
            break;
        }

        default:
            if (opts->show_types) {
                fprintf(get_output(opts), "#<unknown-type-%d:0x%llx>",
                       base_type, (unsigned long long)value->data.ptr_val);
            } else {
                fprintf(get_output(opts), "#<unknown>");
            }
            break;
    }
}

// Scheme 'write' semantics - quotes strings
void eshkol_write_value(const eshkol_tagged_value_t* value) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    opts.quote_strings = 1;
    eshkol_display_value_opts(value, &opts);
}

/** @brief Like `eshkol_write_value`, but renders to an explicit port (FILE*) instead of current-output-port. */
void eshkol_write_value_to_port(const eshkol_tagged_value_t* value, void* port) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    opts.quote_strings = 1;
    opts.output = port;
    eshkol_display_value_opts(value, &opts);
}

/** @brief Display a value (unquoted strings) to an explicit port (FILE*) instead of current-output-port. */
void eshkol_display_value_to_port(const eshkol_tagged_value_t* value, void* port) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    opts.output = port;
    eshkol_display_value_opts(value, &opts);
}

// Display a list (cons cell chain)
void eshkol_display_list(uint64_t cons_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (cons_ptr == 0) {
        fprintf(out, "()");
        return;
    }

    // Check depth limit
    if (opts->current_depth > opts->max_depth) {
        fprintf(out, "(...)");
        return;
    }

    fprintf(out, "(");
    opts->current_depth++;

    uint64_t current = cons_ptr;
    bool first = true;

    while (current != 0) {
        arena_tagged_cons_cell_t* cell = (arena_tagged_cons_cell_t*)current;

        if (!first) {
            fprintf(out, " ");
        }
        first = false;

        // Display car
        eshkol_display_value_opts(&cell->car, opts);

        // Check cdr type - handle both legacy and consolidated formats
        uint8_t cdr_full = cell->cdr.type;
        uint8_t cdr_type = (cdr_full >= 32) ? cdr_full : (cdr_full >= 8) ? cdr_full : (cdr_full & 0x0F);

        if (cdr_type == ESHKOL_VALUE_NULL) {
            // Proper list end
            break;
        } else if (cdr_type == ESHKOL_VALUE_CONS_PTR) {
            // Legacy cons - continue to next cell
            current = cell->cdr.data.ptr_val;
        } else if (cdr_type == ESHKOL_VALUE_HEAP_PTR) {
            // Consolidated heap pointer - check if it's a cons cell
            void* cdr_ptr = (void*)cell->cdr.data.ptr_val;
            if (cdr_ptr) {
                eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(cdr_ptr);
                if (hdr->subtype == HEAP_SUBTYPE_CONS) {
                    // Continue to next cons cell
                    current = cell->cdr.data.ptr_val;
                } else {
                    // Not a cons - dotted pair
                    fprintf(out, " . ");
                    eshkol_display_value_opts(&cell->cdr, opts);
                    break;
                }
            } else {
                break;
            }
        } else {
            // Dotted pair - display cdr and break
            fprintf(out, " . ");
            eshkol_display_value_opts(&cell->cdr, opts);
            break;
        }
    }

    opts->current_depth--;
    fprintf(out, ")");
}

// Display a lambda by extracting embedded S-expression or looking up in registry
void eshkol_display_lambda(uint64_t closure_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (closure_ptr == 0) {
        fprintf(out, "#<procedure>");
        return;
    }

    // For LAMBDA_SEXPR subtype, the closure_ptr IS a closure struct with sexpr_ptr
    // (same structure as CLOSURE subtype, just no environment)
    eshkol_closure_t* closure = (eshkol_closure_t*)closure_ptr;
    uint64_t sexpr = closure->sexpr_ptr;

    if (sexpr != 0) {
        // Display the embedded S-expression
        eshkol_display_list(sexpr, opts);
    } else {
        // No embedded S-expression - try the registry as fallback
        uint64_t registry_sexpr = eshkol_lambda_registry_lookup(closure->func_ptr);
        if (registry_sexpr != 0) {
            eshkol_display_list(registry_sexpr, opts);
        } else {
            fprintf(out, "#<procedure>");
        }
    }
}

// Display a closure by extracting its embedded S-expression
void eshkol_display_closure(uint64_t closure_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (closure_ptr == 0) {
        fprintf(out, "#<closure>");
        return;
    }

    // Closure struct: { func_ptr (8), env (8), sexpr_ptr (8) }
    eshkol_closure_t* closure = (eshkol_closure_t*)closure_ptr;
    uint64_t sexpr = closure->sexpr_ptr;

    if (sexpr != 0) {
        // Display the embedded S-expression
        eshkol_display_list(sexpr, opts);
    } else {
        // No S-expression - try the registry as fallback
        uint64_t registry_sexpr = eshkol_lambda_registry_lookup(closure->func_ptr);
        if (registry_sexpr != 0) {
            eshkol_display_list(registry_sexpr, opts);
        } else {
            fprintf(out, "#<closure>");
        }
    }
}

// Display a character
static void display_char(uint32_t codepoint, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (!opts->quote_strings) {
        // display mode: output raw character
        if (codepoint < 128) {
            fputc((char)codepoint, out);
        } else {
            // UTF-8 encode
            if (codepoint < 0x80) {
                fputc(codepoint, out);
            } else if (codepoint < 0x800) {
                fputc(0xC0 | (codepoint >> 6), out);
                fputc(0x80 | (codepoint & 0x3F), out);
            } else if (codepoint < 0x10000) {
                fputc(0xE0 | (codepoint >> 12), out);
                fputc(0x80 | ((codepoint >> 6) & 0x3F), out);
                fputc(0x80 | (codepoint & 0x3F), out);
            } else {
                fputc(0xF0 | (codepoint >> 18), out);
                fputc(0x80 | ((codepoint >> 12) & 0x3F), out);
                fputc(0x80 | ((codepoint >> 6) & 0x3F), out);
                fputc(0x80 | (codepoint & 0x3F), out);
            }
        }
        return;
    }

    // write mode: output #\ notation
    switch (codepoint) {
        case ' ':  fprintf(out, "#\\space"); break;
        case '\n': fprintf(out, "#\\newline"); break;
        case '\t': fprintf(out, "#\\tab"); break;
        case '\r': fprintf(out, "#\\return"); break;
        case 0:    fprintf(out, "#\\null"); break;
        default:
            if (codepoint < 128 && codepoint >= 32) {
                fprintf(out, "#\\%c", (char)codepoint);
            } else {
                fprintf(out, "#\\x%X", codepoint);
            }
            break;
    }
}

// Tensor struct layout is defined in arena_memory.h (eshkol_tensor_t)
// Must match LLVM TypeSystem tensor_type:
// struct Tensor {
//     uint64_t* dimensions;      // idx 0: array of dimension sizes
//     uint64_t  num_dimensions;  // idx 1: number of dimensions
//     int64_t*  elements;        // idx 2: element data (doubles stored as int64 bits)
//     uint64_t  total_elements;  // idx 3: total number of elements
// };

// Recursive helper for displaying N-dimensional tensors
static void display_tensor_recursive(FILE* out, const eshkol_tensor_t* tensor,
                                      uint64_t current_dim, uint64_t offset) {
    if (tensor->num_dimensions == 0) {
        fprintf(out, "#()");
        return;
    }

    uint64_t dim_size = tensor->dimensions[current_dim];

    // Base case: innermost dimension - print actual elements
    if (current_dim == tensor->num_dimensions - 1) {
        fprintf(out, "(");
        for (uint64_t i = 0; i < dim_size; i++) {
            if (i > 0) fprintf(out, " ");
            // Elements stored as int64 bit pattern of double
            int64_t bits = tensor->elements[offset + i];
            double value;
            memcpy(&value, &bits, sizeof(double));
            eshkol_fprint_double(out, value);
        }
        fprintf(out, ")");
        return;
    }

    // Recursive case: compute stride and iterate over slices
    uint64_t stride = 1;
    for (uint64_t k = current_dim + 1; k < tensor->num_dimensions; k++) {
        stride *= tensor->dimensions[k];
    }

    fprintf(out, "(");
    for (uint64_t i = 0; i < dim_size; i++) {
        if (i > 0) fprintf(out, " ");
        display_tensor_recursive(out, tensor, current_dim + 1, offset + i * stride);
    }
    fprintf(out, ")");
}

// Display a tensor with proper N-dimensional structure
static void display_tensor(uint64_t tensor_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (tensor_ptr == 0) {
        fprintf(out, "#()");
        return;
    }

    const eshkol_tensor_t* tensor = (const eshkol_tensor_t*)tensor_ptr;

    // Validate tensor structure
    if (tensor->num_dimensions == 0 || tensor->total_elements == 0) {
        fprintf(out, "#()");
        return;
    }

    if (tensor->dimensions == NULL || tensor->elements == NULL) {
        fprintf(out, "#<invalid-tensor>");
        return;
    }

    // Print tensor prefix then contents
    fprintf(out, "#");
    display_tensor_recursive(out, tensor, 0, 0);
}

// Scheme vector structure:
// [length: i64] at offset 0 (8 bytes)
// [elem0: tagged_value] at offset 8 (16 bytes each)
// [elem1: tagged_value] at offset 24
// etc.

// Display a Scheme vector with proper element formatting
static void display_vector(uint64_t vector_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (vector_ptr == 0) {
        fprintf(out, "#()");
        return;
    }

    // Read length from start of vector
    uint64_t* len_ptr = (uint64_t*)vector_ptr;
    uint64_t length = *len_ptr;

    // Validate length (sanity check)
    if (length > 10000) {
        fprintf(out, "#<invalid-vector>");
        return;
    }

    fprintf(out, "#(");

    // Elements start after the 8-byte length field
    // Each element is a tagged_value (16 bytes)
    uint8_t* elem_base = (uint8_t*)vector_ptr + 8;

    for (uint64_t i = 0; i < length; i++) {
        if (i > 0) fprintf(out, " ");

        // Get pointer to i-th tagged_value element
        eshkol_tagged_value_t* elem = (eshkol_tagged_value_t*)(elem_base + i * sizeof(eshkol_tagged_value_t));

        // Recursively display the element
        eshkol_display_value_opts(elem, opts);
    }

    fprintf(out, ")");
}

// ===== END UNIFIED DISPLAY IMPLEMENTATION =====
