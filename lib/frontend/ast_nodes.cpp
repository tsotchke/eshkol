/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
/**
 * @file ast_nodes.cpp
 * @brief Storage for AST nodes and their side arrays.
 *
 * One instance of the rooted, chunked owner arena (ast_owner_arena.h); see
 * inc/eshkol/frontend/ast_nodes.h for the ownership model.
 */

#include <eshkol/frontend/ast_nodes.h>

#include "ast_owner_arena.h"

namespace {

/* tests/memory/leak_audit_gate.sh adds the teardown report's `requested` to
 * its per-line retention figure, beside the string owner's. */
eshkol::frontend::detail::OwnerArena g_nodes("eshkol-ast-nodes", "ESHKOL_AST_NODES_STATS");

}  // namespace

extern "C" void* eshkol_ast_node_storage_alloc(size_t bytes) {
    return g_nodes.allocate(bytes);
}

extern "C" bool eshkol_ast_node_storage_is_owned(const void* p) {
    return g_nodes.owns(p);
}

extern "C" void eshkol_ast_nodes_stats(eshkol_ast_nodes_stats_t* out) {
    if (!out) return;
    const auto c = g_nodes.counters();
    out->allocations = c.allocations;
    out->bytes_requested = c.bytes_requested;
    out->live_chunks = c.live_chunks;
    out->live_bytes_reserved = c.live_bytes_reserved;
    out->teardowns = c.teardowns;
}

extern "C" void eshkol_ast_nodes_teardown(void) {
    g_nodes.teardown();
}
