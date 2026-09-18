/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
/**
 * @file ast_strings_test.cpp
 * @brief Contract tests for the AST string owner (ADR-0016).
 *
 * Three claims are under test:
 *
 *   1. The owner itself: copies are exact and NUL-terminated, a large
 *      request and many small ones both work across chunk boundaries, the
 *      counters move, and teardown releases everything and leaves the owner
 *      usable.
 *
 *   2. Every producer uses it. A walk over real parser output, over the
 *      macro expander's hygienic rewrite, over module-private renaming and
 *      over eshkol_copy_ast() finds that every string pointer on every node
 *      is owned. A producer that went back to `new char[]` or `strdup()`
 *      would fail here before it could reach a leak report.
 *
 *   3. Consumers never free: eshkol_ast_clean() detaches a literal's text
 *      without releasing it, and a module rename leaves the old spelling
 *      readable until teardown.
 */

#include <eshkol/eshkol.h>
#include <eshkol/frontend/ast_strings.h>
#include <eshkol/frontend/macro_expander.h>
#include <eshkol/module_visibility.h>

#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("  FAIL: %s\n", what);
        ++g_failures;
    }
}

/* ---- 1. the owner ---------------------------------------------------- */

static void test_owner_contract() {
    eshkol_ast_strings_stats_t before{};
    eshkol_ast_strings_stats(&before);

    char* a = eshkol_ast_strdup("define");
    check(a && std::strcmp(a, "define") == 0, "strdup copies the text");
    check(eshkol_ast_string_is_owned(a), "strdup result is owned");
    check(eshkol_ast_strdup(nullptr) == nullptr, "strdup(NULL) is NULL");

    const char raw[] = {'a', '\0', 'b'};
    char* b = eshkol_ast_strndup(raw, sizeof raw);
    check(b && b[0] == 'a' && b[1] == '\0' && b[2] == 'b' && b[3] == '\0',
          "strndup copies embedded NULs and terminates");

    char* e = eshkol_ast_string_alloc(0);
    check(e && e[0] == '\0', "a zero-byte request is a valid empty string");

    char* z = eshkol_ast_string_alloc(32);
    bool zeroed = z != nullptr;
    for (int i = 0; z && i < 32; ++i) zeroed = zeroed && z[i] == 0;
    check(zeroed, "alloc returns zero-filled storage");

    std::string big(200000, 'x');
    char* g = eshkol_ast_string_copy(big);
    check(g && std::strlen(g) == big.size() && g[0] == 'x' && g[big.size() - 1] == 'x',
          "a request larger than a chunk is served whole");

    std::vector<char*> many;
    for (int i = 0; i < 20000; ++i) {
        many.push_back(eshkol_ast_string_copy("identifier-" + std::to_string(i)));
    }
    bool all_ok = true;
    for (int i = 0; i < 20000; ++i) {
        all_ok = all_ok && many[i] &&
                 std::strcmp(many[i], ("identifier-" + std::to_string(i)).c_str()) == 0;
    }
    check(all_ok, "20000 small copies survive crossing many chunk boundaries");
    check(eshkol_ast_string_is_owned(many.front()) && eshkol_ast_string_is_owned(many.back()),
          "first and last small copies are owned");
    check(!eshkol_ast_string_is_owned(big.c_str()), "foreign storage is not owned");

    eshkol_ast_strings_stats_t after{};
    eshkol_ast_strings_stats(&after);
    check(after.allocations >= before.allocations + 20005, "allocation counter moves");
    check(after.live_chunks > before.live_chunks, "chunks are live");
    check(after.live_bytes_reserved >= after.live_chunks * 1024, "reserved bytes are reported");
}

/* ---- 2. every producer uses it --------------------------------------- */

struct Walk {
    size_t strings = 0;
    size_t foreign = 0;
    std::vector<std::string> foreign_names;

    void name(const char* s) {
        if (!s) return;
        ++strings;
        if (!eshkol_ast_string_is_owned(s)) {
            ++foreign;
            foreign_names.emplace_back(s);
        }
    }

    void nodes(const eshkol_ast_t* v, uint64_t n) {
        for (uint64_t i = 0; v && i < n; ++i) node(&v[i]);
    }

    void node(const eshkol_ast_t* ast) {
        if (!ast) return;
        switch (ast->type) {
            case ESHKOL_VAR: name(ast->variable.id); return;
            case ESHKOL_STRING:
            case ESHKOL_SYMBOL:
            case ESHKOL_BIGNUM_LITERAL: name(ast->str_val.ptr); return;
            case ESHKOL_CONS:
                node(ast->cons_cell.car);
                node(ast->cons_cell.cdr);
                return;
            case ESHKOL_OP: break;
            default: return;
        }
        const eshkol_operations_t& op = ast->operation;
        switch (op.op) {
            case ESHKOL_CALL_OP:
            case ESHKOL_IF_OP:
            case ESHKOL_COND_OP:
                node(op.call_op.func);
                nodes(op.call_op.variables, op.call_op.num_vars);
                return;
            case ESHKOL_DEFINE_OP:
                name(op.define_op.name);
                name(op.define_op.rest_param);
                nodes(op.define_op.parameters, op.define_op.num_params);
                node(op.define_op.value);
                return;
            case ESHKOL_LAMBDA_OP:
                name(op.lambda_op.rest_param);
                nodes(op.lambda_op.parameters, op.lambda_op.num_params);
                node(op.lambda_op.body);
                return;
            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP:
            case ESHKOL_LETREC_OP:
            case ESHKOL_LETREC_STAR_OP:
                name(op.let_op.name);
                nodes(op.let_op.bindings, op.let_op.num_bindings);
                node(op.let_op.body);
                return;
            case ESHKOL_SEQUENCE_OP:
                nodes(op.sequence_op.expressions, op.sequence_op.num_expressions);
                return;
            case ESHKOL_SET_OP:
                name(op.set_op.name);
                node(op.set_op.value);
                return;
            default:
                return;
        }
    }
};

static std::vector<eshkol_ast_t> parse_all(const char* source) {
    eshkol_set_parse_source_context("ast_strings_test.esk");
    eshkol_reset_parse_line_counter();
    std::istringstream in(source);
    std::vector<eshkol_ast_t> forms;
    for (;;) {
        eshkol_ast_t form = eshkol_parse_next_ast_from_stream(in);
        if (form.type == ESHKOL_INVALID) break;
        forms.push_back(form);
    }
    return forms;
}

static void report_foreign(const Walk& w, const char* stage) {
    for (const auto& s : w.foreign_names) {
        std::printf("  foreign string after %s: \"%s\"\n", stage, s.c_str());
    }
}

static void test_producers() {
    static const char kSource[] =
        "(define (f x . rest) (display \"hello\") (set! x 99999999999999999999999) x)\n"
        "(define counter (let loop ((i 0)) (if (< i 3) (loop (+ i 1)) 'done)))\n"
        "(define-syntax swap!\n"
        "  (syntax-rules ()\n"
        "    ((_ a b) (let ((tmp a)) (set! a b) (set! b tmp)))))\n"
        "(define (g p q) (swap! p q) (list p q))\n"
        "(define (hidden y) (* y 2))\n"
        "(define (visible z) (hidden z))\n";

    std::vector<eshkol_ast_t> forms = parse_all(kSource);
    check(forms.size() == 6, "all six forms parse");

    Walk parsed;
    parsed.nodes(forms.data(), forms.size());
    check(parsed.strings >= 20, "the walk sees the parser's strings");
    check(parsed.foreign == 0, "every parser-produced string is owned");
    report_foreign(parsed, "parse");

    eshkol::MacroExpander expander;
    std::vector<eshkol_ast_t> expanded = expander.expandAll(forms);
    Walk hygienic;
    hygienic.nodes(expanded.data(), expanded.size());
    check(hygienic.strings >= parsed.strings / 2, "the walk sees the expander's strings");
    check(hygienic.foreign == 0, "every macro-expanded (hygienic) string is owned");
    report_foreign(hygienic, "expansion");

    /* Module-private renaming: `hidden` is not exported, so it is renamed;
     * the replaced spelling must still be readable (never freed). */
    const char* old_hidden = nullptr;
    for (auto& form : expanded) {
        if (form.type == ESHKOL_OP && form.operation.op == ESHKOL_DEFINE_OP &&
            form.operation.define_op.name &&
            std::strcmp(form.operation.define_op.name, "hidden") == 0) {
            old_hidden = form.operation.define_op.name;
        }
    }
    check(old_hidden != nullptr, "the private definition is present before renaming");
    eshkol::rename_private_symbols(expanded, "test.mod",
                                   std::set<std::string>{"f", "counter", "g", "visible"});
    bool renamed = false;
    for (auto& form : expanded) {
        if (form.type == ESHKOL_OP && form.operation.op == ESHKOL_DEFINE_OP &&
            form.operation.define_op.name &&
            std::strstr(form.operation.define_op.name, "hidden") &&
            std::strcmp(form.operation.define_op.name, "hidden") != 0) {
            renamed = true;
        }
    }
    check(renamed, "the private definition was renamed");
    check(old_hidden && std::strcmp(old_hidden, "hidden") == 0 &&
              eshkol_ast_string_is_owned(old_hidden),
          "the replaced spelling is left to the owner, not freed");
    Walk after_rename;
    after_rename.nodes(expanded.data(), expanded.size());
    check(after_rename.foreign == 0, "every renamed string is owned");
    report_foreign(after_rename, "rename");

    /* eshkol_copy_ast() and the symbolic builders. */
    eshkol_ast_t* call = eshkol_make_binary_op_ast("+", eshkol_make_var_ast("u"),
                                                   eshkol_make_int_ast(1));
    eshkol_ast_t* copy = eshkol_copy_ast(call);
    Walk copied;
    copied.node(call);
    copied.node(copy);
    check(copied.strings >= 4 && copied.foreign == 0,
          "symbolic builders and eshkol_copy_ast() use the owner");
    report_foreign(copied, "copy");
}

/* ---- 3. consumers never free ----------------------------------------- */

static void test_clean_detaches() {
    std::vector<eshkol_ast_t> forms = parse_all("\"a literal\"\n");
    check(forms.size() == 1 && forms[0].type == ESHKOL_STRING, "a string literal parses");
    if (forms.size() != 1 || forms[0].type != ESHKOL_STRING) return;
    char* text = forms[0].str_val.ptr;
    check(eshkol_ast_string_is_owned(text), "the literal text is owned");
    eshkol_ast_clean(&forms[0]);
    check(forms[0].type == ESHKOL_INVALID && forms[0].str_val.ptr == nullptr,
          "eshkol_ast_clean detaches the text");
    /* Readable after clean: under ASan a free here would be a
     * heap-use-after-free on this very line. */
    check(std::strcmp(text, "a literal") == 0, "the text is still owned after clean");
}

static void test_teardown() {
    char* s = eshkol_ast_strdup("short-lived");
    check(eshkol_ast_string_is_owned(s), "owned before teardown");
    eshkol_ast_strings_stats_t before{};
    eshkol_ast_strings_stats(&before);
    eshkol_ast_strings_teardown();
    eshkol_ast_strings_stats_t after{};
    eshkol_ast_strings_stats(&after);
    check(after.live_chunks == 0 && after.live_bytes_reserved == 0,
          "teardown releases every chunk");
    check(after.teardowns == before.teardowns + 1, "teardown is counted");
    check(!eshkol_ast_string_is_owned(s), "nothing is owned after teardown");
    char* again = eshkol_ast_strdup("reborn");
    check(again && std::strcmp(again, "reborn") == 0 && eshkol_ast_string_is_owned(again),
          "the owner is usable after teardown");
    eshkol_ast_strings_teardown();
}

int main() {
    test_owner_contract();
    test_producers();
    test_clean_detaches();
    test_teardown();
    if (g_failures) {
        std::printf("FAIL: AST string owner (%d failures)\n", g_failures);
        return 1;
    }
    std::printf("PASS: AST string owner\n");
    return 0;
}
