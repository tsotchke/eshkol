/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_FRONTEND_SYNTAX_COLOR_H
#define ESHKOL_FRONTEND_SYNTAX_COLOR_H

/**
 * @file syntax_color.h
 * @brief The one identifier-renaming rule for `syntax-rules` hygiene,
 *        shared by the native macro expander and the bytecode VM
 *        (ADR-0026).
 *
 * Every expansion of a `syntax-rules` template allocates a fresh *color*
 * (a positive integer). Each identifier the template itself introduces --
 * every template symbol that is not a pattern variable, outside quoted data
 * -- is emitted with that color appended. Substituted pattern variables are
 * caller syntax and keep their own spelling. A colored identifier is
 * therefore a different name from every identifier the caller wrote:
 *
 *   - a binder the template introduces (a `let`, `lambda`, `do`,
 *     `let-values`, `guard` or named-`let` variable, an internal `define`)
 *     binds the colored name, so it can neither capture caller code nor be
 *     captured by it -- without the expander knowing which forms bind;
 *   - a colored reference that no colored binder of the same expansion
 *     binds is *free in the template*. It is resolved by removing its last
 *     color and looking the remaining name up in the macro's definition
 *     environment: a definition-site local if one was visible where the
 *     macro was defined, otherwise the top-level binding. A caller's local
 *     binding of the same spelling is never consulted.
 *
 * Colors nest: a macro-defining macro produces templates whose symbols
 * already carry a color, and each expansion appends one more. Resolution
 * peels one color per definition environment, innermost first.
 *
 * A top-level `define` of a colored name defines the uncolored name, as it
 * did before this rule existed. Quoted data never contains a colored symbol:
 * the colorer skips quoted data, and each engine strips colors when a symbol
 * becomes a datum (a quoted operand may be substituted caller syntax that an
 * outer template colored).
 *
 * The pattern-language markers `...` and `_` are never colored: they belong
 * to the transformer and to `match` patterns, not to the program's bindings.
 * Every other identifier is colored, keywords included; each engine compares
 * keyword spellings through eshkol_syntax_base_is(), so a template's `if`,
 * `else` or `=>` still means the keyword while a caller's local binding of the
 * same spelling cannot capture it.
 *
 * The mark byte is ASCII GS (0x1d). No reader of either engine produces it
 * inside a symbol, so a colored name cannot collide with a source name.
 *
 * Header-only and C-compatible: included by the C++ frontend and by the
 * single-translation-unit VM (`vm_macro.c`).
 */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/** The byte that introduces one color suffix. */
#define ESHKOL_SYNTAX_COLOR_MARK '\x1d'

/** @brief Length of the uncolored base spelling of @p name. */
static inline size_t eshkol_syntax_base_length(const char* name) {
    if (!name) return 0;
    const char* mark = strchr(name, ESHKOL_SYNTAX_COLOR_MARK);
    return mark ? (size_t)(mark - name) : strlen(name);
}

/** @brief True if @p name carries at least one color. */
static inline int eshkol_syntax_is_colored(const char* name) {
    return name && strchr(name, ESHKOL_SYNTAX_COLOR_MARK) != NULL;
}

/**
 * @brief Split off the LAST color of @p name.
 * @param prefix_len receives the length of @p name without its last color
 *        (the name as the macro's definition environment spelled it).
 * @return the last color, or 0 if @p name is uncolored.
 */
static inline unsigned eshkol_syntax_last_color(const char* name, size_t* prefix_len) {
    const char* mark = name ? strrchr(name, ESHKOL_SYNTAX_COLOR_MARK) : NULL;
    if (!mark) {
        if (prefix_len) *prefix_len = name ? strlen(name) : 0;
        return 0;
    }
    if (prefix_len) *prefix_len = (size_t)(mark - name);
    unsigned color = 0;
    for (const char* p = mark + 1; *p >= '0' && *p <= '9'; ++p)
        color = color * 10u + (unsigned)(*p - '0');
    return color;
}

/**
 * @brief Write @p name with @p color appended into @p out (capacity @p cap).
 * @return 1 on success, 0 if the colored name does not fit.
 */
static inline int eshkol_syntax_color_name(char* out, size_t cap,
                                           const char* name, unsigned color) {
    int n = snprintf(out, cap, "%s%c%u", name ? name : "",
                     ESHKOL_SYNTAX_COLOR_MARK, color);
    return n > 0 && (size_t)n < cap;
}

/** @brief True if the base spellings of @p a and @p b are equal. */
static inline int eshkol_syntax_same_base(const char* a, const char* b) {
    size_t la = eshkol_syntax_base_length(a), lb = eshkol_syntax_base_length(b);
    return la == lb && (la == 0 || memcmp(a, b, la) == 0);
}

/** @brief True if base spelling of @p name equals the C string @p keyword. */
static inline int eshkol_syntax_base_is(const char* name, const char* keyword) {
    size_t len = eshkol_syntax_base_length(name);
    return keyword && strlen(keyword) == len && memcmp(name, keyword, len) == 0;
}

/**
 * @brief Pattern markers that a template emits uncolored (see file comment).
 */
static inline int eshkol_syntax_is_auxiliary(const char* name) {
    static const char* const aux[] = { "...", "_" };
    if (!name) return 0;
    for (size_t i = 0; i < sizeof(aux) / sizeof(aux[0]); ++i)
        if (strcmp(name, aux[i]) == 0) return 1;
    return 0;
}

/*
 * Binder renaming and the source spelling.
 *
 * The native expander gives every lexical binder a fresh spelling,
 * `_v<n>.<base>`, so a colored free identifier can reach a definition-site
 * local that a caller's binding of the same spelling would otherwise shadow
 * (ADR-0026, referential transparency). That spelling is an implementation
 * name. Whatever a user reads -- a diagnostic, the source form of a
 * procedure -- names the identifier as it was written. These functions are
 * the one place that knows the format in both directions.
 */

/** The prefix of a fresh binder spelling: `_v<n>.` */
#define ESHKOL_SYNTAX_FRESH_PREFIX "_v"
#define ESHKOL_SYNTAX_FRESH_SEPARATOR '.'

/** @brief Length of the `_v<n>.` prefix at @p name, or 0 if there is none. */
static inline size_t eshkol_syntax_fresh_prefix_length(const char* name) {
    if (!name || name[0] != '_' || name[1] != 'v') return 0;
    const char* p = name + 2;
    if (*p < '0' || *p > '9') return 0;
    while (*p >= '0' && *p <= '9') ++p;
    return *p == ESHKOL_SYNTAX_FRESH_SEPARATOR ? (size_t)(p + 1 - name) : 0;
}

/**
 * @brief The spelling the user wrote for @p name: every fresh-binder prefix
 *        removed, then every color.
 * @param len receives the length of the source spelling.
 * @return a pointer into @p name where the source spelling starts.
 */
static inline const char* eshkol_syntax_source_spelling(const char* name, size_t* len) {
    if (!name) {
        if (len) *len = 0;
        return name;
    }
    size_t skip;
    while ((skip = eshkol_syntax_fresh_prefix_length(name)) != 0) name += skip;
    if (len) *len = eshkol_syntax_base_length(name);
    return name;
}

/** @brief True for a byte that can continue an identifier in running text. */
static inline int eshkol_syntax_is_identifier_byte(char c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
        return 1;
    return c != '\0' && strchr("!$%&*/:<=>?^_~+-.@", c) != NULL;
}

/**
 * @brief Rewrite running text (a diagnostic) in place so every identifier in
 *        it carries its source spelling: fresh-binder prefixes that start an
 *        identifier are removed, and so is every color suffix.
 *
 * The result is never longer than the input.
 */
static inline void eshkol_syntax_source_text_inplace(char* text) {
    if (!text) return;
    char* out = text;
    const char* in = text;
    int at_boundary = 1;
    while (*in) {
        if (at_boundary) {
            size_t skip;
            while ((skip = eshkol_syntax_fresh_prefix_length(in)) != 0) in += skip;
            if (!*in) break;
        }
        if (*in == ESHKOL_SYNTAX_COLOR_MARK) {
            ++in;
            while (*in >= '0' && *in <= '9') ++in;
            continue;
        }
        at_boundary = !eshkol_syntax_is_identifier_byte(*in);
        *out++ = *in++;
    }
    *out = '\0';
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus
#include <string>
/** @brief C++ convenience over eshkol_syntax_source_text_inplace(). */
static inline std::string eshkol_syntax_source_text(std::string text) {
    eshkol_syntax_source_text_inplace(&text[0]);
    text.resize(strlen(text.c_str()));
    return text;
}
/** @brief C++ convenience over eshkol_syntax_source_spelling(). */
static inline std::string eshkol_syntax_source_name(const char* name) {
    size_t len = 0;
    const char* start = eshkol_syntax_source_spelling(name, &len);
    return start ? std::string(start, len) : std::string();
}
#endif

#endif /* ESHKOL_FRONTEND_SYNTAX_COLOR_H */
