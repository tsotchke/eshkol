/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
/**
 * @file ast_nodes_test.cpp
 * @brief Contract tests for the AST node-storage owner (ast_nodes.h).
 *
 * Two claims are under test:
 *
 *   1. The owner itself: allocations are zero-filled and aligned, a zero
 *      count still yields a distinct pointer, eshkol_ast_t construction
 *      applies the AST birth location, the counters move, and teardown
 *      releases everything and leaves the owner usable.
 *
 *   2. Every producer uses it. A walk over real parser output and over the
 *      macro expander's rewrite finds that every child array, parameter and
 *      binding array, type-annotation array, pattern, match clause, macro
 *      definition and tensor dimension array hung off a node is owned. A
 *      producer that went back to a raw `new` would fail here before it could
 *      reach a LeakSanitizer report.
 */

#include <eshkol/eshkol.h>
#include <eshkol/frontend/ast_nodes.h>
#include <eshkol/frontend/macro_expander.h>

#include <cstdint>
#include <cstdio>
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
    eshkol_ast_nodes_stats_t before{};
    eshkol_ast_nodes_stats(&before);

    auto* zero_a = eshkol_ast_new_array<eshkol_ast_t>(0);
    auto* zero_b = eshkol_ast_new_array<eshkol_ast_t>(0);
    check(zero_a && zero_b && zero_a != zero_b, "a zero count yields distinct pointers");

    auto* dims = eshkol_ast_new_array<uint64_t>(7);
    bool zeroed = dims != nullptr;
    for (int i = 0; dims && i < 7; ++i) zeroed = zeroed && dims[i] == 0;
    check(zeroed, "arrays are value-initialised");
    check(reinterpret_cast<uintptr_t>(dims) % 8 == 0, "storage is 8-byte aligned");

    {
        EshkolAstBirthLocationScope scope(41, 7);
        auto* nodes = eshkol_ast_new_array<eshkol_ast_t>(3);
        auto* one = eshkol_ast_new<eshkol_ast_t>();
        check(nodes[0].line == 41 && nodes[2].column == 7 && one->line == 41,
              "eshkol_ast_t construction applies the birth location");
        check(nodes[1].type == ESHKOL_INVALID && nodes[1].node_id == 0,
              "nodes start invalid with no identity");
        auto* copy = eshkol_ast_new<eshkol_ast_t>(nodes[0]);
        check(copy != &nodes[0] && copy->line == 41, "copy construction is supported");
    }

    std::vector<eshkol_ast_t*> many;
    for (int i = 0; i < 5000; ++i) {
        auto* n = eshkol_ast_new_array<eshkol_ast_t>(3);
        n[2].type = ESHKOL_INT64;
        n[2].int64_val = i;
        many.push_back(n);
    }
    bool all_ok = true;
    for (int i = 0; i < 5000; ++i) {
        all_ok = all_ok && many[i][2].type == ESHKOL_INT64 && many[i][2].int64_val == i;
    }
    check(all_ok, "5000 arrays survive crossing many chunk boundaries");

    auto* big = eshkol_ast_new_array<eshkol_ast_t>(2000);
    check(big && eshkol_ast_node_storage_is_owned(&big[1999]),
          "a request larger than a chunk is served whole");
    check(eshkol_ast_node_storage_is_owned(many.front()) &&
              eshkol_ast_node_storage_is_owned(many.back()),
          "first and last small arrays are owned");
    uint64_t foreign = 0;
    check(!eshkol_ast_node_storage_is_owned(&foreign), "foreign storage is not owned");

    eshkol_ast_nodes_stats_t after{};
    eshkol_ast_nodes_stats(&after);
    check(after.allocations >= before.allocations + 5006, "allocation counter moves");
    check(after.bytes_requested >= before.bytes_requested + 5000 * 3 * sizeof(eshkol_ast_t),
          "requested bytes are counted");
    check(after.live_chunks > before.live_chunks, "chunks are live");
}

/* ---- 2. every producer uses it --------------------------------------- */

struct Walk {
    size_t pointers = 0;
    size_t foreign = 0;
    std::set<const void*> seen;
    std::vector<std::string> where;

    bool own(const void* p, const char* field) {
        if (!p) return false;
        ++pointers;
        if (!eshkol_ast_node_storage_is_owned(p)) {
            ++foreign;
            where.emplace_back(field);
        }
        return seen.insert(p).second;
    }

    void nodes(const eshkol_ast_t* v, uint64_t n, const char* field) {
        if (!v) return;
        own(v, field);
        for (uint64_t i = 0; i < n; ++i) body(&v[i]);
    }

    void node(const eshkol_ast_t* ast, const char* field) {
        if (!ast) return;
        if (own(ast, field)) body(ast);
    }

    void types(hott_type_expr_t* const* v, const char* field) { if (v) own(v, field); }

    void names(char* const* v, const char* field) { if (v) own(v, field); }

    void pattern(const eshkol_pattern_t* p) {
        if (!p || !own(p, "pattern")) return;
        switch (p->type) {
            case PATTERN_CONS:
                pattern(p->cons.car_pattern);
                pattern(p->cons.cdr_pattern);
                break;
            case PATTERN_LIST:
                if (p->list.patterns) own(p->list.patterns, "pattern list");
                for (uint64_t i = 0; p->list.patterns && i < p->list.num_patterns; ++i)
                    pattern(p->list.patterns[i]);
                break;
            case PATTERN_OR:
                if (p->or_pat.patterns) own(p->or_pat.patterns, "or-pattern list");
                for (uint64_t i = 0; p->or_pat.patterns && i < p->or_pat.num_patterns; ++i)
                    pattern(p->or_pat.patterns[i]);
                break;
            default:
                break;
        }
    }

    void body(const eshkol_ast_t* ast) {
        switch (ast->type) {
            case ESHKOL_CONS:
                node(ast->cons_cell.car, "cons car");
                node(ast->cons_cell.cdr, "cons cdr");
                return;
            case ESHKOL_OP: break;
            default: return;
        }
        const eshkol_operations_t& op = ast->operation;
        switch (op.op) {
            case ESHKOL_CALL_OP:
            case ESHKOL_IF_OP:
            case ESHKOL_COND_OP:
                node(op.call_op.func, "call func");
                nodes(op.call_op.variables, op.call_op.num_vars, "call arguments");
                return;
            case ESHKOL_DEFINE_OP:
                nodes(op.define_op.parameters, op.define_op.num_params, "define parameters");
                node(op.define_op.value, "define value");
                types(op.define_op.param_types, "define param types");
                return;
            case ESHKOL_LAMBDA_OP:
                nodes(op.lambda_op.parameters, op.lambda_op.num_params, "lambda parameters");
                node(op.lambda_op.body, "lambda body");
                types(op.lambda_op.param_types, "lambda param types");
                return;
            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP:
            case ESHKOL_LETREC_OP:
            case ESHKOL_LETREC_STAR_OP:
                nodes(op.let_op.bindings, op.let_op.num_bindings, "let bindings");
                node(op.let_op.body, "let body");
                types(op.let_op.binding_types, "let binding types");
                return;
            case ESHKOL_SEQUENCE_OP:
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP:
                nodes(op.sequence_op.expressions, op.sequence_op.num_expressions,
                      "sequence expressions");
                return;
            case ESHKOL_SET_OP:
                node(op.set_op.value, "set! value");
                return;
            case ESHKOL_TENSOR_OP:
                nodes(op.tensor_op.elements, op.tensor_op.total_elements, "tensor elements");
                if (op.tensor_op.dimensions) own(op.tensor_op.dimensions, "tensor dimensions");
                return;
            case ESHKOL_MATCH_OP:
                node(op.match_op.expr, "match subject");
                if (op.match_op.clauses) own(op.match_op.clauses, "match clauses");
                for (uint64_t i = 0; op.match_op.clauses && i < op.match_op.num_clauses; ++i) {
                    pattern(op.match_op.clauses[i].pattern);
                    node(op.match_op.clauses[i].guard, "match guard");
                    node(op.match_op.clauses[i].body, "match body");
                }
                return;
            case ESHKOL_LET_VALUES_OP:
            case ESHKOL_LET_STAR_VALUES_OP:
                if (op.let_values_op.binding_vars) {
                    own(op.let_values_op.binding_vars, "let-values variable lists");
                    for (uint64_t i = 0; i < op.let_values_op.num_bindings; ++i)
                        names(op.let_values_op.binding_vars[i], "let-values variables");
                }
                if (op.let_values_op.binding_var_counts)
                    own(op.let_values_op.binding_var_counts, "let-values counts");
                nodes(op.let_values_op.producers, op.let_values_op.num_bindings,
                      "let-values producers");
                node(op.let_values_op.body, "let-values body");
                return;
            case ESHKOL_REQUIRE_OP:
                names(op.require_op.module_names, "require module names");
                names(op.require_op.import_prefixes, "require prefixes");
                return;
            case ESHKOL_PROVIDE_OP:
                names(op.provide_op.export_names, "provide names");
                return;
            case ESHKOL_DEFINE_SYNTAX_OP:
                if (op.define_syntax_op.macro) {
                    own(op.define_syntax_op.macro, "macro definition");
                    names(op.define_syntax_op.macro->literals, "macro literals");
                }
                return;
            default:
                return;
        }
    }

    void report(const char* stage) const {
        for (const auto& w : where) std::printf("  foreign storage after %s: %s\n", stage, w.c_str());
    }
};

static std::vector<eshkol_ast_t> parse_all(const char* source) {
    eshkol_set_parse_source_context("ast_nodes_test.esk");
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

static void test_producers() {
    static const char kSource[] =
        "(require core.list)\n"
        "(provide f g)\n"
        "(define (f (x : integer) . rest) (display \"hello\") (set! x 3) x)\n"
        "(define counter (let loop ((i 0)) (if (< i 3) (loop (+ i 1)) 'done)))\n"
        "(define-syntax swap!\n"
        "  (syntax-rules ()\n"
        "    ((_ a b) (let ((tmp a)) (set! a b) (set! b tmp)))))\n"
        "(define (g p q) (swap! p q) (list p q '(1 (2 3))))\n"
        "(define (h v) (match v ((1 2) 'pair) ((x . y) x) (_ (let-values (((a b) (values 1 2))) a))))\n"
        "(define m #(#(1 2) #(3 4)))\n"
        "(define (k n) (cond ((= n 0) 1) (else (and n (or n (* n (k (- n 1))))))))\n";

    std::vector<eshkol_ast_t> forms = parse_all(kSource);
    check(forms.size() == 9, "all nine forms parse");

    Walk parsed;
    for (const auto& form : forms) parsed.body(&form);
    check(parsed.pointers >= 40, "the walk sees the parser's node storage");
    check(parsed.foreign == 0, "every parser-produced node array is owned");
    parsed.report("parse");

    eshkol::MacroExpander expander;
    std::vector<eshkol_ast_t> expanded = expander.expandAll(forms);
    Walk rewritten;
    for (const auto& form : expanded) rewritten.body(&form);
    check(rewritten.pointers >= parsed.pointers / 2, "the walk sees the expander's node storage");
    check(rewritten.foreign == 0, "every macro-expanded node array is owned");
    rewritten.report("expansion");
}

static void test_teardown() {
    auto* n = eshkol_ast_new<eshkol_ast_t>();
    check(eshkol_ast_node_storage_is_owned(n), "owned before teardown");
    eshkol_ast_nodes_stats_t before{};
    eshkol_ast_nodes_stats(&before);
    eshkol_ast_nodes_teardown();
    eshkol_ast_nodes_stats_t after{};
    eshkol_ast_nodes_stats(&after);
    check(after.live_chunks == 0 && after.live_bytes_reserved == 0,
          "teardown releases every chunk");
    check(after.teardowns == before.teardowns + 1, "teardown is counted");
    check(!eshkol_ast_node_storage_is_owned(n), "nothing is owned after teardown");
    auto* again = eshkol_ast_new_array<eshkol_ast_t>(2);
    check(again && eshkol_ast_node_storage_is_owned(again), "the owner is usable after teardown");
    eshkol_ast_nodes_teardown();
}

int main() {
    test_owner_contract();
    test_producers();
    test_teardown();
    if (g_failures) {
        std::printf("FAIL: AST node owner (%d failures)\n", g_failures);
        return 1;
    }
    std::printf("PASS: AST node owner\n");
    return 0;
}
