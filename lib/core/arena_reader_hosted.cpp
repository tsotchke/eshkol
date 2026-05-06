/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted S-expression reader runtime.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/core/rational.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

static int read_skip_whitespace(FILE* fp) {
    int ch;
    for (;;) {
        ch = std::fgetc(fp);
        if (ch == EOF) return EOF;
        if (ch == ';') {
            while ((ch = std::fgetc(fp)) != EOF && ch != '\n') {}
            if (ch == EOF) return EOF;
            continue;
        }
        if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') return ch;
    }
}

static eshkol_tagged_value_t read_datum(arena_t* arena, FILE* fp, int first_char);

constexpr int kEshkolReadMaxDepth = 4096;
thread_local int g_reader_depth = 0;

static eshkol_tagged_value_t make_eof_tagged(void) {
    eshkol_tagged_value_t val;
    std::memset(&val, 0, sizeof(val));
    val.type = 0xFF;
    return val;
}

static eshkol_tagged_value_t make_null_tagged(void) {
    eshkol_tagged_value_t val;
    std::memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_NULL;
    return val;
}

static eshkol_tagged_value_t make_int_tagged(int64_t n) {
    eshkol_tagged_value_t val;
    std::memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_INT64;
    val.data.int_val = n;
    return val;
}

static eshkol_tagged_value_t make_double_tagged(double d) {
    eshkol_tagged_value_t val;
    std::memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_DOUBLE;
    union { double d; int64_t i; } conv;
    conv.d = d;
    val.data.int_val = conv.i;
    return val;
}

static eshkol_tagged_value_t make_bool_tagged(int b) {
    eshkol_tagged_value_t val;
    std::memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_BOOL;
    val.data.int_val = b ? 1 : 0;
    return val;
}

static eshkol_tagged_value_t make_char_tagged(int32_t ch) {
    eshkol_tagged_value_t val;
    std::memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_CHAR;
    val.data.int_val = ch;
    return val;
}

static eshkol_tagged_value_t make_string_tagged(arena_t* arena, const char* str, size_t len) {
    char* s = (char*)arena_allocate_with_header(arena, len + 1, HEAP_SUBTYPE_STRING, 0);
    std::memcpy(s, str, len);
    s[len] = '\0';
    eshkol_tagged_value_t val;
    std::memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)s;
    return val;
}

extern "C" void* eshkol_intern_symbol_lookup(const char* name);

static eshkol_tagged_value_t make_symbol_tagged(arena_t* arena, const char* name, size_t len) {
    (void)arena;
    for (size_t i = 0; i < len; i++) {
        if (name[i] == '\0') {
            static const char kRejectedSymbol[] = "|invalid-symbol|";
            void* interned = eshkol_intern_symbol_lookup(kRejectedSymbol);
            eshkol_tagged_value_t rejected;
            std::memset(&rejected, 0, sizeof(rejected));
            rejected.type = ESHKOL_VALUE_HEAP_PTR;
            rejected.data.int_val = (int64_t)(uintptr_t)interned;
            return rejected;
        }
    }

    char stack_buf[256];
    char* heap_buf = nullptr;
    char* name_cstr = stack_buf;
    if (len + 1 > sizeof(stack_buf)) {
        heap_buf = static_cast<char*>(std::malloc(len + 1));
        if (!heap_buf) {
            return make_null_tagged();
        }
        name_cstr = heap_buf;
    }
    std::memcpy(name_cstr, name, len);
    name_cstr[len] = '\0';
    void* interned = eshkol_intern_symbol_lookup(name_cstr);
    if (heap_buf) std::free(heap_buf);

    eshkol_tagged_value_t val;
    std::memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)interned;
    return val;
}

static eshkol_tagged_value_t make_cons_tagged(arena_t* arena,
                                              eshkol_tagged_value_t car,
                                              eshkol_tagged_value_t cdr) {
    arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
    if (!cell) {
        return make_null_tagged();
    }
    cell->car = car;
    cell->cdr = cdr;

    eshkol_tagged_value_t val;
    std::memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)cell;
    return val;
}

static eshkol_tagged_value_t read_list(arena_t* arena, FILE* fp) {
    int ch = read_skip_whitespace(fp);
    if (ch == EOF) return make_eof_tagged();
    if (ch == ')') return make_null_tagged();

    eshkol_tagged_value_t car = read_datum(arena, fp, ch);
    if (car.type == 0xFF) return car;

    ch = read_skip_whitespace(fp);
    if (ch == '.') {
        int next = std::fgetc(fp);
        if (next == ' ' || next == '\t' || next == '\n' || next == '\r') {
            int ch2 = read_skip_whitespace(fp);
            eshkol_tagged_value_t cdr = read_datum(arena, fp, ch2);
            read_skip_whitespace(fp);
            return make_cons_tagged(arena, car, cdr);
        }
        std::ungetc(next, fp);
        std::ungetc('.', fp);
    } else {
        std::ungetc(ch, fp);
    }

    eshkol_tagged_value_t cdr = read_list(arena, fp);
    return make_cons_tagged(arena, car, cdr);
}

static eshkol_tagged_value_t read_vector(arena_t* arena, FILE* fp) {
    eshkol_tagged_value_t elems[1024];
    int count = 0;

    for (;;) {
        int ch = read_skip_whitespace(fp);
        if (ch == EOF) return make_eof_tagged();
        if (ch == ')') break;
        if (count >= 1024) break;
        elems[count++] = read_datum(arena, fp, ch);
    }

    size_t data_size = 8 + count * sizeof(eshkol_tagged_value_t);
    char* vec = (char*)arena_allocate_with_header(arena, data_size, HEAP_SUBTYPE_VECTOR, 0);
    *(int64_t*)vec = count;
    eshkol_tagged_value_t* vec_elems = (eshkol_tagged_value_t*)(vec + 8);
    for (int i = 0; i < count; ++i) {
        vec_elems[i] = elems[i];
    }

    eshkol_tagged_value_t val;
    std::memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)vec;
    return val;
}

static eshkol_tagged_value_t read_atom(arena_t* arena, FILE* fp, int first_char) {
    if (first_char == '"') {
        char buf[4096];
        int len = 0;
        int ch;
        while ((ch = std::fgetc(fp)) != EOF && ch != '"') {
            if (ch == '\\') {
                ch = std::fgetc(fp);
                if (ch == EOF) break;
                switch (ch) {
                    case 'n': buf[len++] = '\n'; break;
                    case 't': buf[len++] = '\t'; break;
                    case 'r': buf[len++] = '\r'; break;
                    case '\\': buf[len++] = '\\'; break;
                    case '"': buf[len++] = '"'; break;
                    default: buf[len++] = ch; break;
                }
            } else if (len < 4095) {
                buf[len++] = ch;
            }
        }
        return make_string_tagged(arena, buf, len);
    }

    if (first_char == '#') {
        int ch = std::fgetc(fp);
        if (ch == 't') {
            int next = std::fgetc(fp);
            if (next == EOF || next == ' ' || next == '\n' || next == '\r' ||
                next == '\t' || next == ')' || next == '(') {
                if (next != EOF) std::ungetc(next, fp);
                return make_bool_tagged(1);
            }
            std::ungetc(next, fp);
            char rest[16];
            int rlen = 0;
            rest[rlen++] = 'r';
            while (rlen < 15) {
                int c = std::fgetc(fp);
                if (c == EOF || c == ' ' || c == '\n' || c == ')' || c == '(') {
                    if (c != EOF) std::ungetc(c, fp);
                    break;
                }
                rest[rlen++] = c;
            }
            rest[rlen] = '\0';
            if (std::strcmp(rest, "rue") == 0) return make_bool_tagged(1);
            return make_bool_tagged(1);
        }
        if (ch == 'f') {
            int next = std::fgetc(fp);
            if (next == EOF || next == ' ' || next == '\n' || next == '\r' ||
                next == '\t' || next == ')' || next == '(') {
                if (next != EOF) std::ungetc(next, fp);
                return make_bool_tagged(0);
            }
            std::ungetc(next, fp);
            for (;;) {
                int c = std::fgetc(fp);
                if (c == EOF || c == ' ' || c == '\n' || c == ')' || c == '(') {
                    if (c != EOF) std::ungetc(c, fp);
                    break;
                }
            }
            return make_bool_tagged(0);
        }
        if (ch == '\\') {
            int c1 = std::fgetc(fp);
            if (c1 == EOF) return make_eof_tagged();
            int c2 = std::fgetc(fp);
            if (c2 == EOF || c2 == ' ' || c2 == '\n' || c2 == '\r' ||
                c2 == '\t' || c2 == ')' || c2 == '(') {
                if (c2 != EOF) std::ungetc(c2, fp);
                return make_char_tagged(c1);
            }

            char name[32];
            name[0] = c1;
            name[1] = c2;
            int nlen = 2;
            while (nlen < 31) {
                int c = std::fgetc(fp);
                if (c == EOF || c == ' ' || c == '\n' || c == ')' || c == '(') {
                    if (c != EOF) std::ungetc(c, fp);
                    break;
                }
                name[nlen++] = c;
            }
            name[nlen] = '\0';
            if (std::strcmp(name, "space") == 0) return make_char_tagged(' ');
            if (std::strcmp(name, "newline") == 0) return make_char_tagged('\n');
            if (std::strcmp(name, "tab") == 0) return make_char_tagged('\t');
            if (std::strcmp(name, "return") == 0) return make_char_tagged('\r');
            if (std::strcmp(name, "null") == 0) return make_char_tagged(0);
            if (name[0] == 'x') {
                int codepoint = (int)std::strtol(name + 1, nullptr, 16);
                return make_char_tagged(codepoint);
            }
            return make_char_tagged(c1);
        }
        if (ch == '(') {
            return read_vector(arena, fp);
        }

        char buf[256];
        buf[0] = '#';
        buf[1] = ch;
        int blen = 2;
        while (blen < 255) {
            int c = std::fgetc(fp);
            if (c == EOF || c == ' ' || c == '\n' || c == '\r' ||
                c == '\t' || c == ')' || c == '(' || c == '"') {
                if (c != EOF) std::ungetc(c, fp);
                break;
            }
            buf[blen++] = c;
        }
        return make_symbol_tagged(arena, buf, blen);
    }

    char buf[256];
    buf[0] = first_char;
    int blen = 1;
    while (blen < 255) {
        int ch = std::fgetc(fp);
        if (ch == EOF || ch == ' ' || ch == '\n' || ch == '\r' ||
            ch == '\t' || ch == ')' || ch == '(' || ch == '"' || ch == ';') {
            if (ch != EOF) std::ungetc(ch, fp);
            break;
        }
        buf[blen++] = ch;
    }
    buf[blen] = '\0';

    char* endp;
    long long ival = std::strtoll(buf, &endp, 10);
    if (endp == buf + blen && blen > 0) {
        return make_int_tagged(ival);
    }

    double dval = std::strtod(buf, &endp);
    if (endp == buf + blen && blen > 0) {
        return make_double_tagged(dval);
    }

    char* slash = std::strchr(buf, '/');
    if (slash && slash != buf && slash != buf + blen - 1) {
        *slash = '\0';
        char* ep1;
        char* ep2;
        long long num = std::strtoll(buf, &ep1, 10);
        long long den = std::strtoll(slash + 1, &ep2, 10);
        if (*ep1 == '\0' && *ep2 == '\0' && den != 0) {
            void* rat = eshkol_rational_create(arena, num, den);
            eshkol_tagged_value_t val;
            std::memset(&val, 0, sizeof(val));
            if (((eshkol_rational_t*)rat)->denominator == 1) {
                val.type = ESHKOL_VALUE_INT64;
                val.data.int_val = ((eshkol_rational_t*)rat)->numerator;
            } else {
                val.type = ESHKOL_VALUE_HEAP_PTR;
                val.data.int_val = (int64_t)(uintptr_t)rat;
            }
            return val;
        }
        *slash = '/';
    }

    return make_symbol_tagged(arena, buf, blen);
}

static eshkol_tagged_value_t read_datum(arena_t* arena, FILE* fp, int first_char) {
    if (first_char == EOF) return make_eof_tagged();

    if (g_reader_depth >= kEshkolReadMaxDepth) {
        return make_eof_tagged();
    }
    g_reader_depth++;

    eshkol_tagged_value_t result;
    if (first_char == '\'') {
        int ch = read_skip_whitespace(fp);
        eshkol_tagged_value_t quoted = read_datum(arena, fp, ch);
        eshkol_tagged_value_t quote_sym = make_symbol_tagged(arena, "quote", 5);
        eshkol_tagged_value_t inner = make_cons_tagged(arena, quoted, make_null_tagged());
        result = make_cons_tagged(arena, quote_sym, inner);
    } else if (first_char == '(') {
        result = read_list(arena, fp);
    } else {
        result = read_atom(arena, fp, first_char);
    }

    g_reader_depth--;
    return result;
}

} // namespace

extern "C" void eshkol_read_sexpr(void* arena_void, void* fp_void,
                                   eshkol_tagged_value_t* result) {
    arena_t* arena = (arena_t*)arena_void;
    FILE* fp = (FILE*)fp_void;
    if (!fp) fp = stdin;

    g_reader_depth = 0;
    const int ch = read_skip_whitespace(fp);
    *result = read_datum(arena, fp, ch);
}
