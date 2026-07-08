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
#include <cstdint>
#include <cstdlib>
#include <cstring>

// ===== S-EXPRESSION READER (R7RS `read`) =====
// Lightweight runtime S-expression reader: tokenizer + recursive descent parser
// Produces tagged values directly (not compiler AST nodes)

static int read_skip_whitespace(FILE* fp) {
    int ch;
    for (;;) {
        ch = fgetc(fp);
        if (ch == EOF) return EOF;
        if (ch == ';') {
            // Line comment: skip to end of line
            while ((ch = fgetc(fp)) != EOF && ch != '\n') {}
            if (ch == EOF) return EOF;
            continue;
        }
        if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') return ch;
    }
}

// Forward declaration
static eshkol_tagged_value_t read_datum(arena_t* arena, FILE* fp, int first_char);

/* Reader recursion-depth guard (audit C3).
 *
 * (read port) walks `(((((...)))))` via recursive C calls. We ship a
 * 512 MB linker-requested stack so 1M-deep forms happen to work in
 * practice, but that's a platform quirk, not a design. A malicious
 * input file can still exhaust stack deterministically on any build
 * with default limits (8 MB on Linux, ≈ 500 k depth). Counter is
 * thread-local so concurrent `(read)` calls don't share state. */
#define ESHKOL_READ_MAX_DEPTH 4096
static thread_local int g_reader_depth = 0;

static inline eshkol_tagged_value_t make_reader_depth_error(void) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = 0xFF;  /* reuse EOF sentinel — caller treats as malformed */
    return val;
}

static eshkol_tagged_value_t make_eof_tagged(void) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = 0xFF; // EOF object type
    return val;
}

static eshkol_tagged_value_t make_null_tagged(void) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_NULL;
    return val;
}

static eshkol_tagged_value_t make_int_tagged(int64_t n) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_INT64;
    val.data.int_val = n;
    return val;
}

static eshkol_tagged_value_t make_double_tagged(double d) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_DOUBLE;
    union { double d; int64_t i; } conv;
    conv.d = d;
    val.data.int_val = conv.i;
    return val;
}

static eshkol_tagged_value_t make_bool_tagged(int b) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_BOOL;
    val.data.int_val = b ? 1 : 0;
    return val;
}

static eshkol_tagged_value_t make_char_tagged(int32_t ch) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_CHAR;
    val.data.int_val = ch;
    return val;
}

static eshkol_tagged_value_t make_string_tagged(arena_t* arena, const char* str, size_t len) {
    char* s = (char*)arena_allocate_with_header(arena, len + 1, HEAP_SUBTYPE_STRING, 0);
    memcpy(s, str, len);
    s[len] = '\0';
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)s;
    return val;
}

/* Runtime symbol interning (Noesis Bug F, 2026-04-19).
 * Previously this allocated a fresh arena block for every (read) symbol,
 * so `(eq? (read port) 'foo)` always returned #f even when the spelled
 * name matched — violating R7RS §6.5 which mandates canonical symbols.
 * Route through the process-global intern pool used by parser-path
 * symbol literals (lib/core/symbol_intern.cpp). Every distinct spelling
 * maps to exactly one pointer so `eq?` across source-quoted symbols
 * and reader-produced symbols matches. The arena arg is now unused
 * (kept for ABI continuity with existing call sites). */
extern "C" void* eshkol_intern_symbol_lookup(const char* name);

static eshkol_tagged_value_t make_symbol_tagged(arena_t* arena, const char* name, size_t len) {
    (void)arena;  /* symbols live in the process-lifetime intern pool now */

    /* Security (audit C1 — symbol NUL-truncation spoofing).
     * Previous code memcpy'd `len` bytes then NUL-terminated. The
     * intern table keys with std::string(cstr) which truncates at the
     * first NUL, so `"admin\0guest"` collapsed to `'admin` and was
     * eq? to the legitimate `'admin` symbol. Any tag-dispatch
     * `(cond ((eq? role 'admin) ...))` on attacker-controlled
     * deserialised data was spoofable. Reject embedded NULs up-front. */
    for (size_t i = 0; i < len; i++) {
        if (name[i] == '\0') {
            /* Return a distinct empty symbol rather than silently
             * truncating. Callers treat NULL-ish returns uniformly;
             * the invalid-symbol surface is auditable. */
            static const char kRejectedSym[] = "|invalid-symbol|";
            void* interned_rej = eshkol_intern_symbol_lookup(kRejectedSym);
            eshkol_tagged_value_t rej;
            memset(&rej, 0, sizeof(rej));
            rej.type = ESHKOL_VALUE_HEAP_PTR;
            rej.data.int_val = (int64_t)(uintptr_t)interned_rej;
            return rej;
        }
    }

    char stack_buf[256];
    char* name_cstr = stack_buf;
    char* heap_buf = NULL;
    if (len + 1 > sizeof(stack_buf)) {
        heap_buf = (char*)malloc(len + 1);
        if (!heap_buf) {
            eshkol_tagged_value_t null_val;
            memset(&null_val, 0, sizeof(null_val));
            null_val.type = ESHKOL_VALUE_NULL;
            return null_val;
        }
        name_cstr = heap_buf;
    }
    memcpy(name_cstr, name, len);
    name_cstr[len] = '\0';
    void* interned = eshkol_intern_symbol_lookup(name_cstr);
    if (heap_buf) free(heap_buf);

    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)interned;
    return val;
}

static eshkol_tagged_value_t make_cons_tagged(arena_t* arena,
    eshkol_tagged_value_t car, eshkol_tagged_value_t cdr) {
    // Use the header-carrying cons allocator so every cons cell has a proper
    // eshkol_object_header_t with HEAP_SUBTYPE_CONS. This lets pair?,
    // equal?, kb-query, eval, and every consolidated-HEAP_PTR dispatch path
    // treat read's output identically to code-constructed cons cells.
    arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
    if (!cell) {
        eshkol_tagged_value_t null_val;
        memset(&null_val, 0, sizeof(null_val));
        null_val.type = ESHKOL_VALUE_NULL;
        return null_val;
    }
    cell->car = car;
    cell->cdr = cdr;
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)cell;
    return val;
}

// Read a list: ( datum ... ) or ( datum ... . datum )
//
// Bug 2 (reader stack overflow on long lists): this used to recurse via
// read_list() for the cdr of every element — one native stack frame per
// list element. A flat list of ~46k elements overflowed the native stack
// during `read`, independent of ESHKOL_STACK_SIZE. Rewritten to be
// iterative: elements are collected into a growable heap buffer (realloc,
// like read_vector's approach) with a plain loop, then the proper/
// improper list is consed together right-to-left over the buffer — no
// native recursion per element. Nested lists still legitimately recurse
// through read_datum, bounded by g_reader_depth / ESHKOL_READ_MAX_DEPTH;
// that path is untouched.
static eshkol_tagged_value_t read_list(arena_t* arena, FILE* fp) {
    size_t capacity = 64;
    size_t count = 0;
    eshkol_tagged_value_t* elems =
        (eshkol_tagged_value_t*)malloc(capacity * sizeof(eshkol_tagged_value_t));
    if (!elems) return make_null_tagged(); // OOM: fail safe, mirrors make_cons_tagged's OOM path

    eshkol_tagged_value_t tail = make_null_tagged();

    for (;;) {
        int ch = read_skip_whitespace(fp);
        if (ch == EOF) {
            // Matches original semantics: EOF before the closing ')'
            // becomes the terminating tail rather than a hard error —
            // e.g. an unterminated "(1 2 3" reads as (1 2 3 . #<eof>).
            tail = make_eof_tagged();
            break;
        }
        if (ch == ')') {
            tail = make_null_tagged(); // proper list end
            break;
        }

        eshkol_tagged_value_t car = read_datum(arena, fp, ch);
        if (car.type == 0xFF) {
            // Propagate EOF / reader-depth error as the terminating tail
            // (original code returned it bare from the recursive call,
            // which became the enclosing cons's cdr — same shape here).
            tail = car;
            break;
        }

        if (count == capacity) {
            capacity *= 2;
            eshkol_tagged_value_t* grown = (eshkol_tagged_value_t*)realloc(
                elems, capacity * sizeof(eshkol_tagged_value_t));
            if (!grown) {
                free(elems);
                return make_null_tagged(); // OOM: fail safe
            }
            elems = grown;
        }
        elems[count++] = car;

        // Check for dot notation
        ch = read_skip_whitespace(fp);
        if (ch == '.') {
            // Improper list: (a . b)
            int next = fgetc(fp);
            if (next == ' ' || next == '\t' || next == '\n' || next == '\r') {
                int ch2 = read_skip_whitespace(fp);
                eshkol_tagged_value_t cdr = read_datum(arena, fp, ch2);
                // Consume closing paren
                read_skip_whitespace(fp); // should be ')'
                tail = cdr;
                break;
            }
            // Not a dot pair, it's a symbol starting with .
            ungetc(next, fp);
            ungetc('.', fp);
            // Falls through to the next loop iteration, which re-reads
            // '.' via read_skip_whitespace as the start of the next
            // element — same as the original's recursive re-entry.
        } else {
            ungetc(ch, fp);
        }
    }

    // Cons the collected elements onto the tail, right-to-left, so the
    // whole list is built with a plain loop instead of native recursion.
    eshkol_tagged_value_t result = tail;
    for (size_t i = count; i > 0; i--) {
        result = make_cons_tagged(arena, elems[i - 1], result);
    }
    free(elems);
    return result;
}

// Read a vector: #( datum ... )
static eshkol_tagged_value_t read_vector(arena_t* arena, FILE* fp) {
    // Collect elements into a temporary list first, then convert to vector
    // Use a bounded stack to avoid recursion issues
    eshkol_tagged_value_t elems[1024];
    int count = 0;

    for (;;) {
        int ch = read_skip_whitespace(fp);
        if (ch == EOF) return make_eof_tagged();
        if (ch == ')') break;
        if (count >= 1024) break; // safety limit
        elems[count++] = read_datum(arena, fp, ch);
    }

    // Allocate vector: [header | length(i64) | elements(tagged_value * count)]
    size_t data_size = 8 + count * sizeof(eshkol_tagged_value_t);
    char* vec = (char*)arena_allocate_with_header(arena, data_size, HEAP_SUBTYPE_VECTOR, 0);
    *(int64_t*)vec = count; // length
    eshkol_tagged_value_t* vec_elems = (eshkol_tagged_value_t*)(vec + 8);
    for (int i = 0; i < count; i++) {
        vec_elems[i] = elems[i];
    }

    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)vec;
    return val;
}

// Read an atom (number, symbol, string, #t, #f, #\char)
static eshkol_tagged_value_t read_atom(arena_t* arena, FILE* fp, int first_char) {
    // String literal (audit C4: bound *every* write into buf).
    // Previously only the non-escape branch checked `len < 4095`; the
    // escape branch wrote buf[len++] unconditionally. A crafted source
    // with 4100+ `\\` escapes overflowed the stack buffer — stack smash
    // / return-address overwrite surface reachable via (read port) on
    // untrusted input. Apply the bound uniformly and consume-without-
    // storing past the cap so the delimiter search still terminates.
    if (first_char == '"') {
        char buf[4096];
        int len = 0;
        int ch;
        while ((ch = fgetc(fp)) != EOF && ch != '"') {
            char decoded;
            if (ch == '\\') {
                ch = fgetc(fp);
                if (ch == EOF) break;
                switch (ch) {
                    case 'n':  decoded = '\n'; break;
                    case 't':  decoded = '\t'; break;
                    case 'r':  decoded = '\r'; break;
                    case '\\': decoded = '\\'; break;
                    case '"':  decoded = '"';  break;
                    default:   decoded = (char)ch; break;
                }
            } else {
                decoded = (char)ch;
            }
            if (len < (int)sizeof(buf) - 1) {
                buf[len++] = decoded;
            }
            /* else: silently drop past cap. An error-raising variant
             * would be more correct but would change a 1-byte overflow
             * from an RCE primitive into a hard-fail DoS; the cap is
             * documented as a reader-level string-length ceiling. */
        }
        return make_string_tagged(arena, buf, len);
    }

    // Hash prefix: #t, #f, #\char, #(vector
    if (first_char == '#') {
        int ch = fgetc(fp);
        if (ch == 't') {
            int next = fgetc(fp);
            if (next == EOF || next == ' ' || next == '\n' || next == '\r' ||
                next == '\t' || next == ')' || next == '(') {
                if (next != EOF) ungetc(next, fp);
                return make_bool_tagged(1);
            }
            ungetc(next, fp);
            // Could be #true
            char rest[16];
            int rlen = 0;
            rest[rlen++] = 'r';
            while (rlen < 15) {
                int c = fgetc(fp);
                if (c == EOF || c == ' ' || c == '\n' || c == ')' || c == '(') {
                    if (c != EOF) ungetc(c, fp);
                    break;
                }
                rest[rlen++] = c;
            }
            rest[rlen] = '\0';
            if (strcmp(rest, "rue") == 0) return make_bool_tagged(1);
            return make_bool_tagged(1); // fallback
        }
        if (ch == 'f') {
            int next = fgetc(fp);
            if (next == EOF || next == ' ' || next == '\n' || next == '\r' ||
                next == '\t' || next == ')' || next == '(') {
                if (next != EOF) ungetc(next, fp);
                return make_bool_tagged(0);
            }
            ungetc(next, fp);
            // Could be #false — consume rest and return false
            while (1) {
                int c = fgetc(fp);
                if (c == EOF || c == ' ' || c == '\n' || c == ')' || c == '(') {
                    if (c != EOF) ungetc(c, fp);
                    break;
                }
            }
            return make_bool_tagged(0);
        }
        if (ch == '\\') {
            // Character literal
            int c1 = fgetc(fp);
            if (c1 == EOF) return make_eof_tagged();
            int c2 = fgetc(fp);
            if (c2 == EOF || c2 == ' ' || c2 == '\n' || c2 == '\r' ||
                c2 == '\t' || c2 == ')' || c2 == '(') {
                if (c2 != EOF) ungetc(c2, fp);
                return make_char_tagged(c1);
            }
            // Multi-char name: space, newline, tab, etc.
            char name[32];
            name[0] = c1;
            name[1] = c2;
            int nlen = 2;
            while (nlen < 31) {
                int c = fgetc(fp);
                if (c == EOF || c == ' ' || c == '\n' || c == ')' || c == '(') {
                    if (c != EOF) ungetc(c, fp);
                    break;
                }
                name[nlen++] = c;
            }
            name[nlen] = '\0';
            if (strcmp(name, "space") == 0) return make_char_tagged(' ');
            if (strcmp(name, "newline") == 0) return make_char_tagged('\n');
            if (strcmp(name, "tab") == 0) return make_char_tagged('\t');
            if (strcmp(name, "return") == 0) return make_char_tagged('\r');
            if (strcmp(name, "null") == 0) return make_char_tagged(0);
            if (name[0] == 'x') {
                // Hex character: #\x41
                int codepoint = (int)strtol(name + 1, NULL, 16);
                return make_char_tagged(codepoint);
            }
            return make_char_tagged(c1); // fallback: first char
        }
        if (ch == '(') {
            return read_vector(arena, fp);
        }
        // Unknown # form — treat as symbol
        char buf[256];
        buf[0] = '#';
        buf[1] = ch;
        int blen = 2;
        while (blen < 255) {
            int c = fgetc(fp);
            if (c == EOF || c == ' ' || c == '\n' || c == '\r' ||
                c == '\t' || c == ')' || c == '(' || c == '"') {
                if (c != EOF) ungetc(c, fp);
                break;
            }
            buf[blen++] = c;
        }
        return make_symbol_tagged(arena, buf, blen);
    }

    // Number or symbol
    char buf[256];
    buf[0] = first_char;
    int blen = 1;
    while (blen < 255) {
        int ch = fgetc(fp);
        if (ch == EOF || ch == ' ' || ch == '\n' || ch == '\r' ||
            ch == '\t' || ch == ')' || ch == '(' || ch == '"' || ch == ';') {
            if (ch != EOF) ungetc(ch, fp);
            break;
        }
        buf[blen++] = ch;
    }
    buf[blen] = '\0';

    // Try to parse as number
    char* endp;
    long long ival = strtoll(buf, &endp, 10);
    if (endp == buf + blen && blen > 0) {
        return make_int_tagged(ival);
    }
    // Try as double
    double dval = strtod(buf, &endp);
    if (endp == buf + blen && blen > 0) {
        return make_double_tagged(dval);
    }
    // Try as rational: num/denom
    char* slash = strchr(buf, '/');
    if (slash && slash != buf && slash != buf + blen - 1) {
        *slash = '\0';
        char *ep1, *ep2;
        long long num = strtoll(buf, &ep1, 10);
        long long den = strtoll(slash + 1, &ep2, 10);
        if (*ep1 == '\0' && *ep2 == '\0' && den != 0) {
            void* rat = eshkol_rational_create(arena, num, den);
            eshkol_tagged_value_t val;
            memset(&val, 0, sizeof(val));
            if (((eshkol_rational_t*)rat)->denominator == 1) {
                val.type = ESHKOL_VALUE_INT64;
                val.data.int_val = ((eshkol_rational_t*)rat)->numerator;
            } else {
                val.type = ESHKOL_VALUE_HEAP_PTR;
                val.data.int_val = (int64_t)(uintptr_t)rat;
            }
            return val;
        }
        *slash = '/'; // restore
    }

    // Symbol
    return make_symbol_tagged(arena, buf, blen);
}

// Read a single S-expression datum from a FILE*
static eshkol_tagged_value_t read_datum(arena_t* arena, FILE* fp, int first_char) {
    if (first_char == EOF) return make_eof_tagged();

    if (g_reader_depth >= ESHKOL_READ_MAX_DEPTH) {
        return make_reader_depth_error();
    }
    g_reader_depth++;
    eshkol_tagged_value_t result;

    // Quote shorthand: 'x -> (quote x)
    if (first_char == '\'') {
        int ch = read_skip_whitespace(fp);
        eshkol_tagged_value_t quoted = read_datum(arena, fp, ch);
        eshkol_tagged_value_t quote_sym = make_symbol_tagged(arena, "quote", 5);
        eshkol_tagged_value_t inner = make_cons_tagged(arena, quoted, make_null_tagged());
        result = make_cons_tagged(arena, quote_sym, inner);
    }
    else if (first_char == '(') {
        result = read_list(arena, fp);
    }
    else {
        result = read_atom(arena, fp, first_char);
    }

    g_reader_depth--;
    return result;
}

// Main entry point: read one S-expression from a FILE*
extern "C" void eshkol_read_sexpr(void* arena_void, void* fp_void,
                                   eshkol_tagged_value_t* result) {
    arena_t* arena = (arena_t*)arena_void;
    FILE* fp = (FILE*)fp_void;
    if (!fp) fp = stdin;

    /* Reset depth counter at every top-level entry so a prior
     * exit-via-sentinel doesn't leave a stale count. */
    g_reader_depth = 0;
    int ch = read_skip_whitespace(fp);
    *result = read_datum(arena, fp, ch);
}
