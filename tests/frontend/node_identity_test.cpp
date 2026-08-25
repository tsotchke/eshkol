/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
/**
 * @file node_identity_test.cpp
 * @brief Contract tests for the ADR-0000 Stage 1 node-identity substrate.
 *
 * Two things are under test, and they are different claims:
 *
 *   1. The substrate itself: ids are dense and tagged, a span round-trips
 *      unchanged, an unset or garbage id NEVER resolves to a location, and
 *      an id survives the by-value copies the AST is moved around with.
 *      "Never resolves" is the load-bearing one — the whole reason the id
 *      is tagged is that `eshkol_ast_t` is built in places that do not
 *      zero-initialize, so a wrong answer here would be a diagnostic
 *      pointing confidently at unrelated source.
 *
 *   2. The parser really stamps: a form parsed through the public entry
 *      point comes back carrying a NodeId that resolves to that form's own
 *      line, column and file. Without this the substrate is an empty table
 *      with a nice API.
 */

#include <eshkol/eshkol.h>
#include <eshkol/frontend/node_identity.h>

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

static int g_failures = 0;

static void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("  FAIL: %s\n", what);
        ++g_failures;
    }
}

/* ---- 1. substrate contract ------------------------------------------- */

static void test_none_never_resolves() {
    eshkol_source_span_t span;
    std::memset(&span, 0xAB, sizeof(span));
    check(!eshkol_node_span_lookup(ESHKOL_NODE_ID_NONE, &span),
          "ESHKOL_NODE_ID_NONE must not resolve");
}

static void test_garbage_never_resolves() {
    /* Every word whose high byte is not the tag must be rejected outright,
     * whatever the low bits look like. These are the shapes an
     * uninitialised field actually takes: zero-ish, all-ones, ASCII debris,
     * and a small integer that would be a perfectly valid index if the id
     * were untagged. */
    const uint32_t garbage[] = {
        0xFFFFFFFFu, 0x00000001u, 0x0000002Au, 0xDEADBEEFu,
        0x41414141u, 0x7F7F7F7Fu, 0x80000000u, 0xE4000001u, 0xE6000001u,
    };
    for (uint32_t word : garbage) {
        eshkol_source_span_t span;
        char label[80];
        std::snprintf(label, sizeof(label),
                      "garbage word 0x%08X must not resolve", word);
        check(!eshkol_node_span_lookup((eshkol_node_id_t)word, &span), label);
    }
    /* A correctly tagged but not-yet-issued index must also be rejected:
     * the tag alone is not authority, the bound is. */
    const uint32_t beyond =
        (uint32_t)(ESHKOL_NODE_ID_TAG << 24) | ESHKOL_NODE_ID_MAX_INDEX;
    eshkol_source_span_t span;
    check(!eshkol_node_span_lookup((eshkol_node_id_t)beyond, &span),
          "a tagged id past the issued range must not resolve");
}

static void test_roundtrip_and_density() {
    const uint64_t before = eshkol_node_id_count();

    eshkol_node_id_t a = eshkol_node_id_new(7, 12, 34);
    eshkol_node_id_t b = eshkol_node_id_new(7, 12, 40);

    check(a != ESHKOL_NODE_ID_NONE && b != ESHKOL_NODE_ID_NONE,
          "fresh ids must not be NONE");
    check((a >> 24) == ESHKOL_NODE_ID_TAG, "a fresh id carries the tag");
    check((b & ESHKOL_NODE_ID_MAX_INDEX) == (a & ESHKOL_NODE_ID_MAX_INDEX) + 1,
          "ids are dense: consecutive allocations get consecutive indices");
    check(eshkol_node_id_count() == before + 2, "count tracks allocations");

    eshkol_source_span_t span;
    check(eshkol_node_span_lookup(a, &span), "a fresh id resolves");
    check(span.file_id == 7 && span.start_line == 12 && span.start_column == 34,
          "the span round-trips exactly what was stored");
    check(span.end_line == 12 && span.end_column == 34 && !span.has_extent,
          "a point span mirrors its start and reports no extent");
}

static void test_extent_is_monotone() {
    eshkol_node_id_t id = eshkol_node_id_new(3, 20, 5);
    eshkol_source_span_t span;

    eshkol_node_span_set_extent(id, 24, 9);
    check(eshkol_node_span_lookup(id, &span), "extent target resolves");
    check(span.has_extent && span.end_line == 24 && span.end_column == 9,
          "setting an extent records it and flags it as measured");
    check(span.start_line == 20 && span.start_column == 5,
          "setting an extent never moves the start");

    /* An end before the start is not a span; it is a bug upstream, and the
     * substrate must refuse it rather than hand a consumer a backwards
     * range to render a caret from. */
    eshkol_node_span_set_extent(id, 2, 1);
    check(eshkol_node_span_lookup(id, &span), "target still resolves");
    check(span.end_line == 24 && span.end_column == 9,
          "an end before the start is refused");

    eshkol_node_span_set_extent(ESHKOL_NODE_ID_NONE, 1, 1);  /* must not crash */
}

static void test_id_survives_node_copy() {
    /* AST nodes are shuffled between arrays by value all over the frontend
     * and codegen. Identity that did not survive a copy would be identity
     * of a storage location, not of a node. */
    eshkol_ast_t node = {};
    node.type = ESHKOL_INT64;
    node.int64_val = 41;
    node.node_id = eshkol_node_id_new(9, 100, 3);

    eshkol_ast_t assigned = node;
    eshkol_ast_t blitted;
    std::memcpy(&blitted, &node, sizeof(blitted));

    eshkol_source_span_t direct, via_assign, via_memcpy;
    check(eshkol_node_span_lookup(node.node_id, &direct), "origin resolves");
    check(eshkol_node_span_lookup(assigned.node_id, &via_assign),
          "id survives struct assignment");
    check(eshkol_node_span_lookup(blitted.node_id, &via_memcpy),
          "id survives a raw memcpy");
    check(direct.start_line == via_assign.start_line &&
              direct.start_line == via_memcpy.start_line &&
              direct.start_column == via_memcpy.start_column,
          "every copy resolves to the same span");
}

static void test_counters_move() {
    eshkol_node_identity_reset_stats();
    uint64_t queried = 0, resolved = 0, located = 0, extent = 0, allocated = 0;
    eshkol_node_identity_stats(&queried, &resolved, &located, &extent, &allocated);
    check(queried == 0 && resolved == 0, "reset zeroes the coverage counters");

    eshkol_node_identity_record_lookup(true, true, false);
    eshkol_node_identity_record_lookup(false, false, false);
    eshkol_node_identity_stats(&queried, &resolved, &located, &extent, &allocated);
    check(queried == 2, "every lookup is counted");
    check(resolved == 1, "only resolved lookups count as resolved");
    check(located == 1 && extent == 0, "location and extent are counted apart");
    check(allocated >= 3, "allocation count is cumulative for the process");
    eshkol_node_identity_reset_stats();
}

/* ---- 2. the parser really stamps ------------------------------------- */

static void test_parser_stamps_forms() {
    const char* kSource =
        "(define (square x)\n"
        "  (* x x))\n"
        "\n"
        "(display (square 7))\n";

    eshkol_set_parse_source_context("node_identity_test.esk");
    eshkol_reset_parse_line_counter();

    std::istringstream in(kSource);

    eshkol_ast_t first = eshkol_parse_next_ast_from_stream(in);
    check(first.type != ESHKOL_INVALID, "first form parses");

    eshkol_source_span_t span;
    check(eshkol_node_span_lookup(first.node_id, &span),
          "the parser stamped the first form with a resolvable NodeId");
    if (span.start_line != 0) {
        check(span.start_line == 1,
              "the first form's span starts on line 1");
        check(span.start_column == first.column && span.start_line == first.line,
              "the span agrees with the node's own line/column fields");
        const char* name = eshkol_source_file_name(span.file_id);
        check(name != nullptr &&
                  std::string(name).find("node_identity_test.esk") != std::string::npos,
              "the span names the file the form was measured in");
        check(span.has_extent && span.end_line >= span.start_line,
              "a top-level form carries a measured extent, not just a point");
    }

    eshkol_ast_t second = eshkol_parse_next_ast_from_stream(in);
    check(second.type != ESHKOL_INVALID, "second form parses");
    eshkol_source_span_t span2;
    check(eshkol_node_span_lookup(second.node_id, &span2),
          "the parser stamped the second form too");
    if (span.start_line != 0 && span2.start_line != 0) {
        check(span2.start_line == 4,
              "the second form's span starts on line 4, past the blank line");
        check(second.node_id != first.node_id,
              "two distinct forms have distinct identities");
    }
}

int main() {
    std::printf("node_identity_test: ADR-0000 Stage 1 substrate\n");

    test_none_never_resolves();
    test_garbage_never_resolves();
    test_roundtrip_and_density();
    test_extent_is_monotone();
    test_id_survives_node_copy();
    test_counters_move();
    test_parser_stamps_forms();

    if (g_failures == 0) {
        std::printf("PASS: node identity substrate\n");
        return 0;
    }
    std::printf("FAIL: node identity substrate (%d check(s) failed)\n", g_failures);
    return 1;
}
