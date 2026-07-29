/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * symbol_syntax.h — R7RS 7.1.1 identifier syntax for symbols: the
 * "does this name need vertical bars?" predicate and the `|...|` body escaper.
 *
 * Single source of truth for the WRITE half of R7RS 7.1.1, shared by the native
 * runtime writer (lib/core/runtime_display_hosted.cpp) and the bytecode VM
 * writer (lib/backend/vm_native.c). One implementation is what guarantees the
 * two substrates emit byte-identical external representations for symbols
 * (ADR-0003 parity), and that `write` -> `read` is an identity on both.
 *
 * ── The needs-bars predicate ─────────────────────────────────────────────
 *
 * A symbol prints BARE exactly when its name is spelled by the R7RS 7.1.1
 * <identifier> grammar WITHOUT bars, i.e. by production 1 or 3:
 *
 *   <identifier>   -> <initial> <subsequent>*                        (1)
 *                   | <vertical line> <symbol element>* <vertical line>
 *                   | <peculiar identifier>                          (3)
 *   <initial>      -> <letter> | <special initial>
 *   <letter>       -> a..z | A..Z
 *   <special initial> -> ! $ % & * / : < = > ? ^ _ ~
 *   <subsequent>   -> <initial> | <digit> | <special subsequent>
 *   <special subsequent> -> <explicit sign> | . | @
 *   <explicit sign>      -> + | -
 *   <peculiar identifier> -> <explicit sign>
 *                          | <explicit sign> <sign subsequent> <subsequent>*
 *                          | <explicit sign> . <dot subsequent> <subsequent>*
 *                          | . <dot subsequent> <subsequent>*
 *   <dot subsequent>  -> <sign subsequent> | .
 *   <sign subsequent> -> <initial> | <explicit sign> | @
 *
 * Anything else gets bars. That single rule already covers the cases that
 * matter: the empty name (`||`), embedded whitespace (`|weird sym|`), a name
 * that would read back as a NUMBER (`|1|`, `|+1|`, `|.5|` -- a digit is not an
 * <initial>, and a digit is neither a <sign subsequent> nor a <dot
 * subsequent>), and the bare dot (`|.|`, which is the dotted-pair delimiter,
 * not an identifier). `foo`, `with->arrow`, `...`, `+`, `-` and `+soup+` all
 * stay bare, so ordinary output is unchanged.
 *
 * Two adjustments make the predicate agree with the reader Eshkol actually
 * has, which is the property that matters -- `write` must emit something
 * `read` turns back into the same symbol:
 *
 *   - RESERVED SPELLINGS.  A handful of names satisfy the grammar above but
 *     are claimed by another token in Eshkol's lexer, so writing them bare
 *     would not read back as a symbol: `->` (the function-type arrow) and the
 *     four R7RS special reals `+inf.0` `-inf.0` `+nan.0` `-nan.0` (numbers,
 *     which R7RS also resolves in favour of the number). These force bars.
 *
 *   - COLON.  R7RS lists `:` as a <special initial>, but Eshkol's tokenizer
 *     gives `:` a second job as the type-annotation separator: it ends a
 *     symbol everywhere except at the very start of one, where `:key` is read
 *     as a whole keyword symbol. So a LEADING colon stays bare (the `:key`
 *     alist idiom keeps printing as `:key`) and a colon anywhere else forces
 *     bars, because `a:b` written bare would read back as three tokens.
 *
 * Bytes >= 0x80 count as <letter>. R7RS permits implementations to extend
 * <letter> this way, and Eshkol's readers already accept UTF-8 identifiers
 * bare, so leaving them unbarred round-trips.
 *
 * ── The escaper ──────────────────────────────────────────────────────────
 *
 *   <symbol element> -> <any character other than <vertical line> or \>
 *                     | <inline hex escape> | <mnemonic escape> | \|
 *   <mnemonic escape> -> \a | \b | \t | \n | \r
 *   <inline hex escape> -> \x <hex scalar value> ;
 *
 * Note that `\\` is NOT a <symbol element> -- unlike <string element>, R7RS
 * gives a symbol no two-character spelling for a backslash. So the escaper
 * emits `\x5c;` for a backslash, which every conforming reader accepts. (The
 * Eshkol readers additionally ACCEPT `\\` on input, a strict superset of the
 * grammar; they just never produce it.) Control characters use their mnemonic
 * escape where R7RS defines one and `\x<hex>;` otherwise, so no symbol name
 * can put a raw control byte into the output stream.
 *
 * Header-only `static inline` so the VM unity build and the C++ runtime each
 * get a private copy with no link dependency between them.
 */
#ifndef ESHKOL_CORE_SYMBOL_SYNTAX_H
#define ESHKOL_CORE_SYMBOL_SYNTAX_H

#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* <letter>, extended with the non-ASCII bytes Eshkol's readers accept bare. */
static inline int eshkol_symbol_is_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}

/* <special initial>, MINUS ':' -- see the COLON note in the file comment. */
static inline int eshkol_symbol_is_special_initial(unsigned char c) {
    switch (c) {
    case '!': case '$': case '%': case '&': case '*': case '/':
    case '<': case '=': case '>': case '?': case '^': case '_': case '~':
        return 1;
    default:
        return 0;
    }
}

static inline int eshkol_symbol_is_initial(unsigned char c) {
    return eshkol_symbol_is_letter(c) || eshkol_symbol_is_special_initial(c);
}

static inline int eshkol_symbol_is_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}

static inline int eshkol_symbol_is_subsequent(unsigned char c) {
    return eshkol_symbol_is_initial(c) || eshkol_symbol_is_digit(c) ||
           c == '+' || c == '-' || c == '.' || c == '@';
}

static inline int eshkol_symbol_is_sign_subsequent(unsigned char c) {
    return eshkol_symbol_is_initial(c) || c == '+' || c == '-' || c == '@';
}

static inline int eshkol_symbol_is_dot_subsequent(unsigned char c) {
    return eshkol_symbol_is_sign_subsequent(c) || c == '.';
}

/* Every byte from `from` on is a <subsequent>. */
static inline int eshkol_symbol_all_subsequent(const char* name, size_t len,
                                               size_t from) {
    size_t i;
    for (i = from; i < len; ++i) {
        if (!eshkol_symbol_is_subsequent((unsigned char)name[i])) return 0;
    }
    return 1;
}

/**
 * @brief Does `write` have to wrap this symbol name in vertical bars?
 *
 * Returns 1 when the name is not spelled by the bar-free R7RS 7.1.1
 * <identifier> productions, or when it is but some other Eshkol token would
 * claim the bare spelling. See the file comment for the full rule.
 *
 * @param name Symbol name bytes; need not be NUL-terminated.
 * @param len  Length of the name in bytes.
 */
static inline int eshkol_symbol_needs_bars(const char* name, size_t len) {
    size_t start = 0;
    size_t i;
    size_t n;
    unsigned char c0;

    if (!name || len == 0) return 1;  /* the empty symbol is only writable as || */

    /* Spellings another Eshkol token would claim. */
    if (len == 2 && name[0] == '-' && name[1] == '>') return 1;
    if (len == 6 && (name[0] == '+' || name[0] == '-') &&
        (memcmp(name + 1, "inf.0", 5) == 0 || memcmp(name + 1, "nan.0", 5) == 0))
        return 1;

    /* A leading ':' is read back whole (`:key`); ':' elsewhere splits. */
    if (name[0] == ':') {
        if (len == 1) return 1;  /* bare ':' is the type-annotation token */
        start = 1;
    }
    for (i = start; i < len; ++i) {
        if (name[i] == ':') return 1;
    }

    n = len - start;
    name += start;
    c0 = (unsigned char)name[0];

    /* (1) <initial> <subsequent>* */
    if (eshkol_symbol_is_initial(c0)) {
        return !eshkol_symbol_all_subsequent(name, n, 1);
    }

    /* (3) <peculiar identifier> */
    if (c0 == '+' || c0 == '-') {
        if (n == 1) return 0;
        if (name[1] == '.') {
            if (n < 3 || !eshkol_symbol_is_dot_subsequent((unsigned char)name[2]))
                return 1;
            return !eshkol_symbol_all_subsequent(name, n, 3);
        }
        if (!eshkol_symbol_is_sign_subsequent((unsigned char)name[1])) return 1;
        return !eshkol_symbol_all_subsequent(name, n, 2);
    }
    if (c0 == '.') {
        if (n < 2 || !eshkol_symbol_is_dot_subsequent((unsigned char)name[1]))
            return 1;
        return !eshkol_symbol_all_subsequent(name, n, 2);
    }

    return 1;
}

/* The <mnemonic escape> letter for `c`, or 0 if R7RS defines none. */
static inline char eshkol_symbol_mnemonic_escape(unsigned char c) {
    switch (c) {
    case '\a': return 'a';
    case '\b': return 'b';
    case '\t': return 't';
    case '\n': return 'n';
    case '\r': return 'r';
    default:   return 0;
    }
}

/* A byte needs escaping inside |...| if it is a bar, a backslash, or a
 * control character (which would otherwise go out raw). */
static inline int eshkol_symbol_element_needs_escape(unsigned char c) {
    return c == '|' || c == '\\' || c < 0x20 || c == 0x7f;
}

/* ── Read direction ───────────────────────────────────────────────────────
 *
 * The four readers (native tokenizer, VM tokenizer, native runtime `read`, VM
 * runtime `read`) each scan their own input medium -- a std::string cursor, a
 * char* cursor, a FILE*, a VmPort -- so they each own their scan loop. What
 * they must NOT own separately is the escape ALPHABET, which is what these
 * helpers pin: the same set of mnemonics the escaper above emits, plus the
 * `\\` spelling that R7RS does not define but that every reader accepts.
 */

/* Inverse of eshkol_symbol_mnemonic_escape: the byte a <mnemonic escape>
 * letter denotes, or -1 if `c` does not spell one. `\|` and `\\` are handled
 * here too, so a reader can route every non-hex escape through this. */
static inline int eshkol_symbol_escape_value(unsigned char c) {
    switch (c) {
    case 'a':  return '\a';
    case 'b':  return '\b';
    case 't':  return '\t';
    case 'n':  return '\n';
    case 'r':  return '\r';
    case '|':  return '|';
    case '\\': return '\\';   /* accepted, never emitted -- see file comment */
    default:   return -1;
    }
}

/* Value of a hex digit, or -1. */
static inline int eshkol_symbol_hex_value(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

/**
 * @brief UTF-8 encode one codepoint from an <inline hex escape>.
 * @return Number of bytes written to `out` (1..4), or 0 if `cp` is out of
 *         Unicode range.
 */
static inline int eshkol_symbol_utf8_encode(unsigned long cp, char* out) {
    if (cp < 0x80UL) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800UL) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000UL) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cp <= 0x10FFFFUL) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

/**
 * @brief Byte length of the escaped <symbol element>* body, excluding the
 *        surrounding bars and the NUL terminator.
 */
static inline size_t eshkol_symbol_escaped_body_len(const char* name,
                                                    size_t len) {
    size_t total = 0;
    size_t i;
    for (i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)name[i];
        if (c == '|') {
            total += 2;                       /* \| */
        } else if (eshkol_symbol_element_needs_escape(c)) {
            total += eshkol_symbol_mnemonic_escape(c) ? 2 : 5;  /* \n or \xNN; */
        } else {
            total += 1;
        }
    }
    return total;
}

/**
 * @brief Write the escaped <symbol element>* body of `name` into `out`.
 *
 * `out` must have room for eshkol_symbol_escaped_body_len() + 1 bytes; the
 * result is NUL-terminated. The bars themselves are the caller's business.
 */
static inline void eshkol_symbol_escape_body(const char* name, size_t len,
                                             char* out) {
    static const char kHex[] = "0123456789abcdef";
    size_t w = 0;
    size_t i;
    for (i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)name[i];
        char mnemonic;
        if (c == '|') {
            out[w++] = '\\';
            out[w++] = '|';
            continue;
        }
        if (!eshkol_symbol_element_needs_escape(c)) {
            out[w++] = (char)c;
            continue;
        }
        mnemonic = eshkol_symbol_mnemonic_escape(c);
        if (mnemonic) {
            out[w++] = '\\';
            out[w++] = mnemonic;
            continue;
        }
        /* Backslash and the remaining control bytes: \xNN; -- R7RS gives a
         * <symbol element> no `\\` spelling, so a backslash goes out as
         * \x5c; rather than the string-literal form. */
        out[w++] = '\\';
        out[w++] = 'x';
        out[w++] = kHex[(c >> 4) & 0xf];
        out[w++] = kHex[c & 0xf];
        out[w++] = ';';
    }
    out[w] = '\0';
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ESHKOL_CORE_SYMBOL_SYNTAX_H */
