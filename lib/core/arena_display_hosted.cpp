/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted tagged-value display implementation.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/core/bignum.h"
#include "../../inc/eshkol/core/inference.h"
#include "../../inc/eshkol/core/logic.h"
#include "../../inc/eshkol/core/rational.h"
#include "../../inc/eshkol/core/workspace.h"

#include <cstdio>
#include <cstdint>
#include <cstring>

namespace {

void display_tensor(std::uint64_t tensor_ptr, eshkol_display_opts_t* opts);
void display_vector(std::uint64_t vector_ptr, eshkol_display_opts_t* opts);
void display_char(std::uint32_t codepoint, eshkol_display_opts_t* opts);

FILE* g_current_output_fp = nullptr;
FILE* g_current_input_fp = nullptr;
FILE* g_current_error_fp = nullptr;

FILE* get_output(eshkol_display_opts_t* opts) {
    if (opts && opts->output) {
        return static_cast<FILE*>(opts->output);
    }
    return g_current_output_fp ? g_current_output_fp : stdout;
}

int utf8_encode_one(std::uint32_t codepoint, char* out) {
    if (codepoint < 0x80) {
        out[0] = static_cast<char>(codepoint);
        return 1;
    }
    if (codepoint < 0x800) {
        out[0] = static_cast<char>(0xC0 | (codepoint >> 6));
        out[1] = static_cast<char>(0x80 | (codepoint & 0x3F));
        return 2;
    }
    if (codepoint < 0x10000) {
        out[0] = static_cast<char>(0xE0 | (codepoint >> 12));
        out[1] = static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out[2] = static_cast<char>(0x80 | (codepoint & 0x3F));
        return 3;
    }
    if (codepoint < 0x110000) {
        out[0] = static_cast<char>(0xF0 | (codepoint >> 18));
        out[1] = static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
        out[2] = static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out[3] = static_cast<char>(0x80 | (codepoint & 0x3F));
        return 4;
    }

    out[0] = static_cast<char>(0xEF);
    out[1] = static_cast<char>(0xBF);
    out[2] = static_cast<char>(0xBD);
    return 3;
}

void display_tensor_recursive(FILE* out,
                              const eshkol_tensor_t* tensor,
                              std::uint64_t current_dim,
                              std::uint64_t offset) {
    if (tensor->num_dimensions == 0) {
        std::fprintf(out, "#()");
        return;
    }

    const std::uint64_t dim_size = tensor->dimensions[current_dim];

    if (current_dim == tensor->num_dimensions - 1) {
        std::fprintf(out, "(");
        for (std::uint64_t i = 0; i < dim_size; i++) {
            if (i > 0) {
                std::fprintf(out, " ");
            }
            const std::int64_t bits = tensor->elements[offset + i];
            double value;
            std::memcpy(&value, &bits, sizeof(double));
            std::fprintf(out, "%g", value);
        }
        std::fprintf(out, ")");
        return;
    }

    std::uint64_t stride = 1;
    for (std::uint64_t k = current_dim + 1; k < tensor->num_dimensions; k++) {
        stride *= tensor->dimensions[k];
    }

    std::fprintf(out, "(");
    for (std::uint64_t i = 0; i < dim_size; i++) {
        if (i > 0) {
            std::fprintf(out, " ");
        }
        display_tensor_recursive(out, tensor, current_dim + 1, offset + i * stride);
    }
    std::fprintf(out, ")");
}

void display_tensor(std::uint64_t tensor_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (tensor_ptr == 0) {
        std::fprintf(out, "#()");
        return;
    }

    const auto* tensor = reinterpret_cast<const eshkol_tensor_t*>(tensor_ptr);
    if (tensor->num_dimensions == 0 || tensor->total_elements == 0) {
        std::fprintf(out, "#()");
        return;
    }

    if (tensor->dimensions == nullptr || tensor->elements == nullptr) {
        std::fprintf(out, "#<invalid-tensor>");
        return;
    }

    std::fprintf(out, "#");
    display_tensor_recursive(out, tensor, 0, 0);
}

void display_vector(std::uint64_t vector_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (vector_ptr == 0) {
        std::fprintf(out, "#()");
        return;
    }

    auto* len_ptr = reinterpret_cast<std::uint64_t*>(vector_ptr);
    const std::uint64_t length = *len_ptr;
    if (length > 10000) {
        std::fprintf(out, "#<invalid-vector>");
        return;
    }

    std::fprintf(out, "#(");
    auto* elem_base = reinterpret_cast<std::uint8_t*>(vector_ptr) + 8;

    for (std::uint64_t i = 0; i < length; i++) {
        if (i > 0) {
            std::fprintf(out, " ");
        }

        auto* elem = reinterpret_cast<eshkol_tagged_value_t*>(
            elem_base + i * sizeof(eshkol_tagged_value_t));
        eshkol_display_value_opts(elem, opts);
    }

    std::fprintf(out, ")");
}

void display_char(std::uint32_t codepoint, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (!opts->quote_strings) {
        if (codepoint < 128) {
            std::fputc(static_cast<char>(codepoint), out);
        } else if (codepoint < 0x800) {
            std::fputc(0xC0 | (codepoint >> 6), out);
            std::fputc(0x80 | (codepoint & 0x3F), out);
        } else if (codepoint < 0x10000) {
            std::fputc(0xE0 | (codepoint >> 12), out);
            std::fputc(0x80 | ((codepoint >> 6) & 0x3F), out);
            std::fputc(0x80 | (codepoint & 0x3F), out);
        } else {
            std::fputc(0xF0 | (codepoint >> 18), out);
            std::fputc(0x80 | ((codepoint >> 12) & 0x3F), out);
            std::fputc(0x80 | ((codepoint >> 6) & 0x3F), out);
            std::fputc(0x80 | (codepoint & 0x3F), out);
        }
        return;
    }

    switch (codepoint) {
        case ' ':
            std::fprintf(out, "#\\space");
            break;
        case '\n':
            std::fprintf(out, "#\\newline");
            break;
        case '\t':
            std::fprintf(out, "#\\tab");
            break;
        case '\r':
            std::fprintf(out, "#\\return");
            break;
        case 0:
            std::fprintf(out, "#\\null");
            break;
        default:
            if (codepoint < 128 && codepoint >= 32) {
                std::fprintf(out, "#\\%c", static_cast<char>(codepoint));
            } else {
                std::fprintf(out, "#\\x%X", codepoint);
            }
            break;
    }
}

} // namespace

extern "C" void* eshkol_runtime_current_output_fp(void) {
    return g_current_output_fp ? g_current_output_fp : stdout;
}

extern "C" void* eshkol_runtime_current_input_fp(void) {
    return g_current_input_fp ? g_current_input_fp : stdin;
}

extern "C" void* eshkol_runtime_current_error_fp(void) {
    return g_current_error_fp ? g_current_error_fp : stderr;
}

extern "C" void eshkol_runtime_set_current_output_fp(void* fp) {
    g_current_output_fp = static_cast<FILE*>(fp);
}

extern "C" void eshkol_runtime_set_current_input_fp(void* fp) {
    g_current_input_fp = static_cast<FILE*>(fp);
}

extern "C" void eshkol_runtime_set_current_error_fp(void* fp) {
    g_current_error_fp = static_cast<FILE*>(fp);
}

extern "C" void* eshkol_string_from_codepoints(void* arena_void,
                                               const std::int64_t* codepoints,
                                               std::uint64_t count) {
    auto* arena = static_cast<arena_t*>(arena_void);
    if (!arena || (!codepoints && count > 0)) {
        return nullptr;
    }

    std::uint64_t bytes = 0;
    for (std::uint64_t i = 0; i < count; i++) {
        const auto codepoint = static_cast<std::uint32_t>(codepoints[i]);
        if (codepoint < 0x80) {
            bytes += 1;
        } else if (codepoint < 0x800) {
            bytes += 2;
        } else if (codepoint < 0x10000) {
            bytes += 3;
        } else if (codepoint < 0x110000) {
            bytes += 4;
        } else {
            bytes += 3;
        }
    }

    auto* str = static_cast<char*>(
        arena_allocate_with_header(arena, bytes + 1, HEAP_SUBTYPE_STRING, 0));
    if (!str) {
        return nullptr;
    }

    char* out = str;
    for (std::uint64_t i = 0; i < count; i++) {
        out += utf8_encode_one(static_cast<std::uint32_t>(codepoints[i]), out);
    }
    *out = '\0';
    return str;
}

extern "C" void eshkol_display_value(const eshkol_tagged_value_t* value) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    eshkol_display_value_opts(value, &opts);
}

extern "C" void eshkol_display_value_opts(const eshkol_tagged_value_t* value,
                                          eshkol_display_opts_t* opts) {
    if (!value) {
        std::fprintf(get_output(opts), "()");
        return;
    }

    if (opts->current_depth > opts->max_depth) {
        std::fprintf(get_output(opts), "...");
        return;
    }

    const std::uint8_t full_type = value->type;
    std::uint8_t base_type;
    if (full_type >= 32) {
        base_type = full_type;
    } else if (full_type >= 8) {
        base_type = full_type;
    } else {
        base_type = full_type & 0x0F;
    }

    if (full_type == (ESHKOL_VALUE_CONS_PTR | 0x10)) {
        FILE* fp = reinterpret_cast<FILE*>(value->data.ptr_val);
        const int fd = fp ? fileno(fp) : -1;
        std::fprintf(get_output(opts), "#<input-port fd:%d>", fd);
        return;
    }
    if (full_type == (ESHKOL_VALUE_CONS_PTR | 0x40)) {
        FILE* fp = reinterpret_cast<FILE*>(value->data.ptr_val);
        const int fd = fp ? fileno(fp) : -1;
        std::fprintf(get_output(opts), "#<output-port fd:%d>", fd);
        return;
    }

    switch (base_type) {
        case ESHKOL_VALUE_NULL:
            std::fprintf(get_output(opts), "()");
            break;

        case ESHKOL_VALUE_HEAP_PTR: {
            void* data_ptr = reinterpret_cast<void*>(value->data.ptr_val);
            if (!data_ptr) {
                std::fprintf(get_output(opts), "()");
                break;
            }
            eshkol_object_header_t* header = ESHKOL_GET_HEADER(data_ptr);
            switch (header->subtype) {
                case HEAP_SUBTYPE_CONS:
                    eshkol_display_list(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_STRING:
                    if (opts->quote_strings) {
                        std::fprintf(get_output(opts), "\"%s\"", static_cast<const char*>(data_ptr));
                    } else {
                        std::fprintf(get_output(opts), "%s", static_cast<const char*>(data_ptr));
                    }
                    break;
                case HEAP_SUBTYPE_SYMBOL:
                    std::fprintf(get_output(opts), "%s", static_cast<const char*>(data_ptr));
                    break;
                case HEAP_SUBTYPE_VECTOR:
                    display_vector(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_TENSOR:
                    display_tensor(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_HASH:
                    std::fprintf(get_output(opts), "#<hash>");
                    break;
                case HEAP_SUBTYPE_EXCEPTION:
                    std::fprintf(get_output(opts), "#<exception>");
                    break;
                case HEAP_SUBTYPE_MULTI_VALUE:
                    std::fprintf(get_output(opts), "#<values>");
                    break;
                case HEAP_SUBTYPE_BIGNUM:
                    eshkol_bignum_display(reinterpret_cast<const eshkol_bignum_t*>(data_ptr),
                                          get_output(opts));
                    break;
                case HEAP_SUBTYPE_SUBSTITUTION:
                    eshkol_display_substitution(
                        reinterpret_cast<const eshkol_substitution_t*>(data_ptr),
                        get_output(opts));
                    break;
                case HEAP_SUBTYPE_FACT:
                    eshkol_display_fact(reinterpret_cast<const eshkol_fact_t*>(data_ptr),
                                        get_output(opts));
                    break;
                case HEAP_SUBTYPE_KNOWLEDGE_BASE:
                    eshkol_display_kb(reinterpret_cast<const eshkol_knowledge_base_t*>(data_ptr),
                                      get_output(opts));
                    break;
                case HEAP_SUBTYPE_FACTOR_GRAPH:
                    eshkol_display_factor_graph(
                        reinterpret_cast<const eshkol_factor_graph_t*>(data_ptr),
                        get_output(opts));
                    break;
                case HEAP_SUBTYPE_WORKSPACE:
                    eshkol_display_workspace(reinterpret_cast<const eshkol_workspace_t*>(data_ptr),
                                             get_output(opts));
                    break;
                case HEAP_SUBTYPE_PROMISE:
                    std::fprintf(get_output(opts), "#<promise>");
                    break;
                case HEAP_SUBTYPE_RATIONAL: {
                    auto* rat = reinterpret_cast<eshkol_rational_t*>(data_ptr);
                    std::fprintf(get_output(opts),
                                 "%lld/%lld",
                                 static_cast<long long>(rat->numerator),
                                 static_cast<long long>(rat->denominator));
                    break;
                }
                default:
                    std::fprintf(get_output(opts), "#<heap:%d>", header->subtype);
                    break;
            }
            break;
        }

        case ESHKOL_VALUE_CALLABLE: {
            void* data_ptr = reinterpret_cast<void*>(value->data.ptr_val);
            if (!data_ptr) {
                std::fprintf(get_output(opts), "#<procedure>");
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
                    std::fprintf(get_output(opts), "#<ad-node>");
                    break;
                case CALLABLE_SUBTYPE_PRIMITIVE:
                    std::fprintf(get_output(opts), "#<primitive>");
                    break;
                case CALLABLE_SUBTYPE_CONTINUATION:
                    std::fprintf(get_output(opts), "#<continuation>");
                    break;
                default:
                    std::fprintf(get_output(opts), "#<callable:%d>", header->subtype);
                    break;
            }
            break;
        }

        case ESHKOL_VALUE_LOGIC_VAR:
            eshkol_display_logic_var(value->data.int_val, get_output(opts));
            break;

        case ESHKOL_VALUE_INT64:
            std::fprintf(get_output(opts), "%lld", static_cast<long long>(value->data.int_val));
            break;

        case ESHKOL_VALUE_DOUBLE:
            std::fprintf(get_output(opts), "%g", value->data.double_val);
            break;

        case ESHKOL_VALUE_BOOL:
            std::fprintf(get_output(opts), "%s", value->data.int_val ? "#t" : "#f");
            break;

        case ESHKOL_VALUE_CHAR:
            display_char(static_cast<std::uint32_t>(value->data.int_val), opts);
            break;

        case ESHKOL_VALUE_STRING_PTR:
            if (opts->quote_strings) {
                std::fprintf(get_output(opts), "\"%s\"",
                             reinterpret_cast<const char*>(value->data.ptr_val));
            } else {
                std::fprintf(get_output(opts), "%s",
                             reinterpret_cast<const char*>(value->data.ptr_val));
            }
            break;

        case ESHKOL_VALUE_SYMBOL:
            std::fprintf(get_output(opts), "%s",
                         reinterpret_cast<const char*>(value->data.ptr_val));
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
            auto* dual = reinterpret_cast<eshkol_dual_number_t*>(value->data.ptr_val);
            if (dual) {
                std::fprintf(get_output(opts), "(dual %g %g)", dual->value, dual->derivative);
            } else {
                std::fprintf(get_output(opts), "(dual 0 0)");
            }
            break;
        }

        case ESHKOL_VALUE_COMPLEX: {
            auto* parts = reinterpret_cast<double*>(value->data.ptr_val);
            if (parts) {
                const double re = parts[0];
                const double im = parts[1];
                if (im == 0.0) {
                    std::fprintf(get_output(opts), "%g", re);
                } else if (re == 0.0) {
                    std::fprintf(get_output(opts), "%gi", im);
                } else if (im < 0.0) {
                    std::fprintf(get_output(opts), "%g%gi", re, im);
                } else {
                    std::fprintf(get_output(opts), "%g+%gi", re, im);
                }
            } else {
                std::fprintf(get_output(opts), "0");
            }
            break;
        }

        case ESHKOL_VALUE_AD_NODE_PTR:
            std::fprintf(get_output(opts), "#<ad-node>");
            break;

        case ESHKOL_VALUE_HASH_PTR: {
            auto* ht = reinterpret_cast<eshkol_hash_table_t*>(value->data.ptr_val);
            if (!ht) {
                std::fprintf(get_output(opts), "#<hash:0>");
            } else {
                std::fprintf(get_output(opts), "#<hash:%zu>", ht->size);
            }
            break;
        }

        default:
            if (opts->show_types) {
                std::fprintf(get_output(opts),
                             "#<unknown-type-%d:0x%llx>",
                             base_type,
                             static_cast<unsigned long long>(value->data.ptr_val));
            } else {
                std::fprintf(get_output(opts), "#<unknown>");
            }
            break;
    }
}

extern "C" void eshkol_write_value(const eshkol_tagged_value_t* value) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    opts.quote_strings = 1;
    eshkol_display_value_opts(value, &opts);
}

extern "C" void eshkol_write_value_to_port(const eshkol_tagged_value_t* value, void* port) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    opts.quote_strings = 1;
    opts.output = port;
    eshkol_display_value_opts(value, &opts);
}

extern "C" void eshkol_display_value_to_port(const eshkol_tagged_value_t* value, void* port) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    opts.output = port;
    eshkol_display_value_opts(value, &opts);
}

extern "C" void eshkol_display_list(std::uint64_t cons_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (cons_ptr == 0) {
        std::fprintf(out, "()");
        return;
    }

    if (opts->current_depth > opts->max_depth) {
        std::fprintf(out, "(...)");
        return;
    }

    std::fprintf(out, "(");
    opts->current_depth++;

    std::uint64_t current = cons_ptr;
    bool first = true;

    while (current != 0) {
        auto* cell = reinterpret_cast<arena_tagged_cons_cell_t*>(current);

        if (!first) {
            std::fprintf(out, " ");
        }
        first = false;

        eshkol_display_value_opts(&cell->car, opts);

        const std::uint8_t cdr_full = cell->cdr.type;
        const std::uint8_t cdr_type =
            (cdr_full >= 32) ? cdr_full : (cdr_full >= 8) ? cdr_full : (cdr_full & 0x0F);

        if (cdr_type == ESHKOL_VALUE_NULL) {
            break;
        } else if (cdr_type == ESHKOL_VALUE_CONS_PTR) {
            current = cell->cdr.data.ptr_val;
        } else if (cdr_type == ESHKOL_VALUE_HEAP_PTR) {
            void* cdr_ptr = reinterpret_cast<void*>(cell->cdr.data.ptr_val);
            if (cdr_ptr) {
                eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(cdr_ptr);
                if (hdr->subtype == HEAP_SUBTYPE_CONS) {
                    current = cell->cdr.data.ptr_val;
                } else {
                    std::fprintf(out, " . ");
                    eshkol_display_value_opts(&cell->cdr, opts);
                    break;
                }
            } else {
                break;
            }
        } else {
            std::fprintf(out, " . ");
            eshkol_display_value_opts(&cell->cdr, opts);
            break;
        }
    }

    opts->current_depth--;
    std::fprintf(out, ")");
}

extern "C" void eshkol_display_lambda(std::uint64_t closure_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (closure_ptr == 0) {
        std::fprintf(out, "#<procedure>");
        return;
    }

    auto* closure = reinterpret_cast<eshkol_closure_t*>(closure_ptr);
    const std::uint64_t sexpr = closure->sexpr_ptr;

    if (sexpr != 0) {
        eshkol_display_list(sexpr, opts);
        return;
    }

    const std::uint64_t registry_sexpr = eshkol_lambda_registry_lookup(closure->func_ptr);
    if (registry_sexpr != 0) {
        eshkol_display_list(registry_sexpr, opts);
    } else {
        std::fprintf(out, "#<procedure>");
    }
}

extern "C" void eshkol_display_closure(std::uint64_t closure_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (closure_ptr == 0) {
        std::fprintf(out, "#<closure>");
        return;
    }

    auto* closure = reinterpret_cast<eshkol_closure_t*>(closure_ptr);
    const std::uint64_t sexpr = closure->sexpr_ptr;

    if (sexpr != 0) {
        eshkol_display_list(sexpr, opts);
        return;
    }

    const std::uint64_t registry_sexpr = eshkol_lambda_registry_lookup(closure->func_ptr);
    if (registry_sexpr != 0) {
        eshkol_display_list(registry_sexpr, opts);
    } else {
        std::fprintf(out, "#<closure>");
    }
}
