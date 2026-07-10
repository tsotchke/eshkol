/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Automatic differentiation runtime helpers.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"

#include <cstdio>
#include <cstring>

extern "C" {

// Global tape pointer for AD operations.
// NOTE: Not thread_local — LLVM ExternalLinkage globals cannot link against
// C thread_local symbols portably (macOS ARM64 TLS ABI mismatch). Thread
// safety is ensured by the AD tape stack, which is thread_local.
// parallel-map workers don't perform AD internally; the main thread controls
// AD mode. If parallel AD is needed, use per-task tape allocation.
ad_tape_t* __current_ad_tape = nullptr;

// Global AD mode flag. It is intentionally shared so lambdas from one LLVM
// module can see AD mode set by another module in REPL/JIT workflows.
bool __ad_mode_active = false;

// ===== AD Phase A instrumentation counters =====
// Process-global (not thread_local): AD runs on the main thread and the Scheme
// reader builtins that expose these must observe the same object the emitted
// increments write. Kept plain so LLVM ExternalLinkage globals link portably.
static EshkolADCounters __eshkol_ad_counters = {0, 0, 0, 0, 0};

void eshkol_ad_counters_reset(void) {
    __eshkol_ad_counters.primal_calls = 0;
    __eshkol_ad_counters.reverse_passes = 0;
    __eshkol_ad_counters.tape_allocations = 0;
    __eshkol_ad_counters.tape_nodes = 0;
    __eshkol_ad_counters.finite_difference_evals = 0;
}
void eshkol_ad_counters_get(EshkolADCounters* out) {
    if (out) *out = __eshkol_ad_counters;
}
void eshkol_ad_count_primal(void)  { __eshkol_ad_counters.primal_calls++; }
void eshkol_ad_count_reverse(void) { __eshkol_ad_counters.reverse_passes++; }
void eshkol_ad_count_fd(void)      { __eshkol_ad_counters.finite_difference_evals++; }
uint64_t eshkol_ad_counter_primal_calls(void)  { return __eshkol_ad_counters.primal_calls; }
uint64_t eshkol_ad_counter_reverse_passes(void){ return __eshkol_ad_counters.reverse_passes; }
uint64_t eshkol_ad_counter_tape_allocations(void) { return __eshkol_ad_counters.tape_allocations; }
uint64_t eshkol_ad_counter_tape_nodes(void)    { return __eshkol_ad_counters.tape_nodes; }
uint64_t eshkol_ad_counter_finite_difference_evals(void) {
    return __eshkol_ad_counters.finite_difference_evals;
}

// Reverse-over-forward detector for the one-pass gradient guard (see header).
static uint64_t __eshkol_ad_mixed_record_count = 0;
uint64_t eshkol_ad_mixed_record_count(void) { return __eshkol_ad_mixed_record_count; }

/**
 * @brief Debug helper that logs the current global AD-mode flag to stderr.
 *
 * Intended for troubleshooting AD activation issues; has no effect on
 * program behavior beyond the diagnostic print.
 *
 * @param context Label identifying the call site, printed alongside the flag.
 */
void debug_print_ad_mode(const char* context) {
    std::fprintf(stderr, "[AD_MODE_DEBUG] %s: __ad_mode_active = %s\n",
                 context, __ad_mode_active ? "TRUE" : "FALSE");
}

/**
 * @brief Debug helper that logs a labeled pointer value to stderr.
 *
 * @param context Label identifying the call site.
 * @param ptr     Pointer to print (address and hex value).
 */
void debug_print_ptr(const char* context, void* ptr) {
    std::fprintf(stderr, "[PTR_DEBUG] %s: ptr = %p (0x%llx)\n",
                 context, ptr, (unsigned long long)(uintptr_t)ptr);
}

thread_local ad_tape_t* __ad_tape_stack[ESHKOL_ARENA_MAX_TAPE_DEPTH] = {nullptr};
thread_local uint64_t __ad_tape_depth = 0;

// ESH-0070: runtime forward-mode perturbation level (see codegen_context.h).
thread_local uint64_t __ad_pert_level = 0;

// ESH-0190 (P5): Taylor-tower differentiation context. __ad_tower_active is a
// depth counter (> 0 while a `derivative-n`/`taylor` tower pass is running); it
// is the fast gate that tells maybeJetLiftTapeOperand to freeze a reverse-tape
// AD node into a dual-tower (carrying its seed tangent) instead of letting the
// reverse tape swallow the tower. __ad_tower_order is the current innermost
// tower order, passed to eshkol_taylor_lift_ad_node as the lifted constant's
// order (a constant tower zero-extends, so best-effort order is sufficient).
// Codegen-only state (mirrors __ad_pert_level): written at the derivative-n
// call site, read in the lambda body — same module, so internal-linkage AOT
// globals stay consistent. Not thread-local: AD runs on the main thread and
// LLVM external globals link more portably as plain symbols (see note above).
uint64_t __ad_tower_active = 0;
uint64_t __ad_tower_order = 0;

thread_local void* __outer_ad_node_storage = nullptr;
thread_local void* __outer_ad_node_to_inner = nullptr;
thread_local void* __outer_grad_accumulator = nullptr;
thread_local void* __inner_var_node_ptr = nullptr;
thread_local uint64_t __gradient_x_degree = 0;

thread_local void* __outer_ad_node_stack[ESHKOL_ARENA_MAX_TAPE_DEPTH] = {nullptr};
thread_local uint64_t __outer_ad_node_depth = 0;

// ===== ESH-0093: mixed-mode AD (reverse tape over forward jets) =====
//
// A reverse-mode `gradient` computes each partial d/dtheta_i in its own pass:
// it rebuilds the tape with fresh AD variable nodes and only reads the ACTIVE
// node's gradient afterwards. Before calling the target function, the pass
// publishes that active variable node here. While a forward-mode derivative
// is live inside the pass (__ad_pert_level > 0), tape nodes that flow into
// scalar arithmetic are "jet-lifted" to dual numbers: value = node->value,
// and e2 = 1.0 iff the node IS the published seed (eshkol_ad_seed_flag).
// The forward 4-jet then carries the mixed e1e2 coefficient
// d(inner-derivative-result)/d(theta_i) to the derivative's return site,
// which records the result back onto the tape (eshkol_ad_mixed_record) as
//   result = const(a1 - a12*theta_i) + const(a12) * theta_i_node
// — an exact local linearization whose backward edge to theta_i is a12.
// Only existing node types (CONSTANT/MUL/ADD) are used, so backpropagation
// needs no new rules. With k tape-carried captures, no extra jet passes are
// needed: the outer gradient's per-component loop already runs one pass per
// theta_i, and each pass only requires the partial w.r.t. its own seed.
thread_local void* __ad_active_seed_node = nullptr;

// Publish a new active seed node; returns the previous one so callers can
// save/restore around nested gradient passes.
void* eshkol_ad_seed_swap(void* node) {
    void* old = __ad_active_seed_node;
    __ad_active_seed_node = node;
    return old;
}

// 1.0 iff `node` is the active seed for the current gradient pass.
double eshkol_ad_seed_flag(void* node) {
    return (node && node == __ad_active_seed_node) ? 1.0 : 0.0;
}

// Record an inner forward-mode derivative result on the active reverse tape.
//   value : the derivative's value (a1, the e1 coefficient)
//   dseed : d(value)/d(seed) (a12, the mixed e1e2 coefficient)
// Returns the AD node to use as the result, or NULL when there is nothing to
// record (no tape, no seed, or no dependency) — the caller then returns the
// plain scalar as before.
void* eshkol_ad_mixed_record(void* arena_v, void* tape_v, double value, double dseed) {
    if (!arena_v || !tape_v) return nullptr;
    ad_node_t* seed = (ad_node_t*)__ad_active_seed_node;
    if (!seed) return nullptr;
    // A forward-mode derivative returned while a reverse seed is live: this pass
    // is reverse-over-forward. Mark it (before the no-dependency early-out) so the
    // one-pass gradient's snapshot delta detects it even when this component's
    // partial happens to be zero, and falls back to per-component replay.
    __eshkol_ad_mixed_record_count++;
    if (dseed == 0.0) return nullptr;  // no dependency on this pass's variable

    arena_t* arena = (arena_t*)arena_v;
    ad_tape_t* tape = (ad_tape_t*)tape_v;

    ad_node_t* coeff  = arena_allocate_ad_node_with_header(arena);
    ad_node_t* scaled = arena_allocate_ad_node_with_header(arena);
    ad_node_t* offset = arena_allocate_ad_node_with_header(arena);
    ad_node_t* result = arena_allocate_ad_node_with_header(arena);
    if (!coeff || !scaled || !offset || !result) return nullptr;

    coeff->type = AD_NODE_CONSTANT;
    coeff->value = dseed;

    scaled->type = AD_NODE_MUL;          // backward: seed.grad += grad * dseed
    scaled->value = dseed * seed->value;
    scaled->input1 = seed;
    scaled->input2 = coeff;

    offset->type = AD_NODE_CONSTANT;
    offset->value = value - dseed * seed->value;

    result->type = AD_NODE_ADD;          // value = a1 exactly
    result->value = value;
    result->input1 = scaled;
    result->input2 = offset;

    // Topological order for the reverse traversal: result last.
    arena_tape_add_node(tape, coeff);
    arena_tape_add_node(tape, offset);
    arena_tape_add_node(tape, scaled);
    arena_tape_add_node(tape, result);
    return result;
}

/**
 * @brief Allocates and zero-initializes a single forward-mode dual number.
 *
 * A dual number carries a value and its derivative (tangent) for forward-mode
 * automatic differentiation. The result is allocated from `arena` and
 * initialized to value = 0.0, derivative = 0.0.
 *
 * @param arena Arena to allocate from; must be non-null.
 * @return      Newly allocated dual number, or nullptr on failure/null arena.
 */
eshkol_dual_number_t* arena_allocate_dual_number(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate dual number: null arena");
        return nullptr;
    }

    eshkol_dual_number_t* dual = (eshkol_dual_number_t*)
        arena_allocate_aligned(arena, sizeof(eshkol_dual_number_t), 8);

    if (dual) {
        dual->value = 0.0;
        dual->derivative = 0.0;
    }

    return dual;
}

/**
 * @brief Allocates and zero-initializes a contiguous array of dual numbers.
 *
 * Used for forward-mode AD over batches/vectors: each element is initialized
 * to value = 0.0, derivative = 0.0.
 *
 * @param arena Arena to allocate from; must be non-null.
 * @param count Number of dual numbers to allocate; must be non-zero.
 * @return      Pointer to the first element of the array, or nullptr on failure.
 */
eshkol_dual_number_t* arena_allocate_dual_batch(arena_t* arena, size_t count) {
    if (!arena || count == 0) {
        eshkol_error("Invalid parameters for batch dual number allocation");
        return nullptr;
    }

    size_t total_size = sizeof(eshkol_dual_number_t) * count;
    eshkol_dual_number_t* duals = (eshkol_dual_number_t*)
        arena_allocate_aligned(arena, total_size, 8);

    if (duals) {
        for (size_t i = 0; i < count; i++) {
            duals[i].value = 0.0;
            duals[i].derivative = 0.0;
        }
    }

    return duals;
}

/**
 * @brief Allocates and default-initializes a single reverse-mode AD tape node.
 *
 * The node is allocated from `arena` as a plain (headerless) ad_node_t and
 * reset to an AD_NODE_CONSTANT with value/gradient 0.0, no parent/extra
 * inputs, and no saved-tensor/shape data. Unlike
 * arena_allocate_ad_node_with_header(), this allocation carries no
 * eshkol_object_header_t, so it is not usable directly as a heap-tagged
 * callable value.
 *
 * @param arena Arena to allocate from; must be non-null.
 * @return      Newly initialized node, or nullptr on failure.
 */
ad_node_t* arena_allocate_ad_node(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate AD node: null arena");
        return nullptr;
    }

    ad_node_t* node = (ad_node_t*)
        arena_allocate_aligned(arena, sizeof(ad_node_t), 8);

    if (node) {
        node->type = AD_NODE_CONSTANT;
        node->value = 0.0;
        node->gradient = 0.0;
        node->input1 = nullptr;
        node->input2 = nullptr;
        node->id = 0;
        node->tensor_value = nullptr;
        node->tensor_gradient = nullptr;
        node->input3 = nullptr;
        node->input4 = nullptr;
        node->saved_tensors = nullptr;
        node->num_saved = 0;
        std::memset(&node->params, 0, sizeof(node->params));
        node->shape = nullptr;
        node->ndim = 0;
    }

    return node;
}

/**
 * @brief Allocates a reverse-mode AD tape node prefixed with an object header.
 *
 * Reserves space for an eshkol_object_header_t immediately followed by the
 * ad_node_t payload (8-byte aligned total size), tags the header's subtype
 * as CALLABLE_SUBTYPE_AD_NODE, and default-initializes the node fields
 * (type = AD_NODE_CONSTANT, value/gradient = 0.0, no inputs/tensors/shape).
 * The returned pointer addresses the payload, not the header. This is the
 * form needed wherever an AD node must also be a tagged/callable heap value
 * (e.g. the tape entries created by eshkol_ad_mixed_record()).
 *
 * @param arena Arena to allocate from; must be non-null.
 * @return      Pointer to the node payload (past the header), or nullptr on failure.
 */
ad_node_t* arena_allocate_ad_node_with_header(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate AD node with header: null arena");
        return nullptr;
    }

    size_t data_size = sizeof(ad_node_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~7;

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate AD node with header");
        return nullptr;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = CALLABLE_SUBTYPE_AD_NODE;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    ad_node_t* node = (ad_node_t*)(mem + sizeof(eshkol_object_header_t));
    node->type = AD_NODE_CONSTANT;
    node->value = 0.0;
    node->gradient = 0.0;
    node->input1 = nullptr;
    node->input2 = nullptr;
    node->id = 0;
    node->tensor_value = nullptr;
    node->tensor_gradient = nullptr;
    node->input3 = nullptr;
    node->input4 = nullptr;
    node->saved_tensors = nullptr;
    node->num_saved = 0;
    std::memset(&node->params, 0, sizeof(node->params));
    node->shape = nullptr;
    node->ndim = 0;

    return node;
}

/**
 * @brief Allocates and default-initializes a contiguous array of headerless AD nodes.
 *
 * Each element is reset to AD_NODE_CONSTANT with value/gradient 0.0 and no
 * inputs/tensors/shape, mirroring arena_allocate_ad_node() but for a batch.
 *
 * @param arena Arena to allocate from; must be non-null.
 * @param count Number of nodes to allocate; must be non-zero.
 * @return      Pointer to the first element of the array, or nullptr on failure.
 */
ad_node_t* arena_allocate_ad_batch(arena_t* arena, size_t count) {
    if (!arena || count == 0) {
        eshkol_error("Invalid parameters for batch AD node allocation");
        return nullptr;
    }

    size_t total_size = sizeof(ad_node_t) * count;
    ad_node_t* nodes = (ad_node_t*)
        arena_allocate_aligned(arena, total_size, 8);

    if (nodes) {
        for (size_t i = 0; i < count; i++) {
            nodes[i].type = AD_NODE_CONSTANT;
            nodes[i].value = 0.0;
            nodes[i].gradient = 0.0;
            nodes[i].input1 = nullptr;
            nodes[i].input2 = nullptr;
            nodes[i].id = 0;
            nodes[i].tensor_value = nullptr;
            nodes[i].tensor_gradient = nullptr;
            nodes[i].input3 = nullptr;
            nodes[i].input4 = nullptr;
            nodes[i].saved_tensors = nullptr;
            nodes[i].num_saved = 0;
            std::memset(&nodes[i].params, 0, sizeof(nodes[i].params));
            nodes[i].shape = nullptr;
            nodes[i].ndim = 0;
        }
    }

    return nodes;
}

/**
 * @brief Allocates and initializes an empty reverse-mode AD tape.
 *
 * Allocates the ad_tape_t header plus a nodes array sized to
 * `initial_capacity` (0 is treated as 64) from `arena`. The tape starts
 * with zero recorded nodes and no variable list; nodes are appended later
 * via arena_tape_add_node() as operations are recorded.
 *
 * @param arena             Arena to allocate from; must be non-null.
 * @param initial_capacity  Initial nodes-array capacity; 0 is treated as 64.
 * @return                  Newly allocated empty tape, or nullptr on failure.
 */
ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity) {
    if (!arena) {
        eshkol_error("Cannot allocate tape: null arena");
        return nullptr;
    }

    if (initial_capacity == 0) {
        initial_capacity = 64;
    }

    ad_tape_t* tape = (ad_tape_t*)
        arena_allocate_aligned(arena, sizeof(ad_tape_t), 8);

    if (!tape) {
        eshkol_error("Failed to allocate tape structure");
        return nullptr;
    }

    size_t nodes_size = sizeof(ad_node_t*) * initial_capacity;
    tape->nodes = (ad_node_t**)arena_allocate_aligned(arena, nodes_size, 8);

    if (!tape->nodes) {
        eshkol_error("Failed to allocate tape nodes array");
        return nullptr;
    }

    tape->num_nodes = 0;
    tape->capacity = initial_capacity;
    tape->variables = nullptr;
    tape->num_variables = 0;

    __eshkol_ad_counters.tape_allocations++;
    return tape;
}

/**
 * @brief Register a gradient pass's input variable nodes on the tape.
 *
 * Populates ad_tape_t::variables / num_variables (previously declared-but-dead)
 * so a single reverse sweep can hand back every input's gradient without
 * replaying the loss per component. `vars` is borrowed (arena-owned by the
 * caller for the tape's lifetime); no copy is made.
 */
void arena_tape_set_variables(ad_tape_t* tape, ad_node_t** vars, size_t n) {
    if (!tape) {
        eshkol_error("Cannot set tape variables: null tape");
        return;
    }
    tape->variables = vars;
    tape->num_variables = n;
}

/**
 * @brief Appends a node to the tape's evaluation-order node list.
 *
 * Records `node` as the next entry in `tape`, growing the backing array
 * (doubling capacity, minimum 128) from the shared REPL arena
 * (__repl_shared_arena) when the tape is full. If the tape is full and no
 * arena is available, or growth fails, the append is silently dropped after
 * logging an error. Nodes must be added in the order they should be visited
 * during the reverse (backward) pass, since the tape is walked in this
 * recorded order to propagate gradients.
 *
 * @param tape Tape to append to; no-op (with error log) if null.
 * @param node Node to append; no-op (with error log) if null.
 */
void arena_tape_add_node(ad_tape_t* tape, ad_node_t* node) {
    if (!tape || !node) {
        eshkol_error("Cannot add node to tape: null parameter");
        return;
    }

    if (tape->num_nodes >= tape->capacity) {
        arena_t* arena = __repl_shared_arena.load();
        if (!arena) {
            eshkol_error("Tape capacity exceeded and no arena available for growth: %zu/%zu",
                         tape->num_nodes, tape->capacity);
            return;
        }

        size_t new_capacity = tape->capacity * 2;
        if (new_capacity < 128) new_capacity = 128;

        size_t new_size = sizeof(ad_node_t*) * new_capacity;
        ad_node_t** new_nodes = (ad_node_t**)arena_allocate_aligned(arena, new_size, 8);

        if (!new_nodes) {
            eshkol_error("Failed to grow tape from %zu to %zu nodes",
                         tape->capacity, new_capacity);
            return;
        }

        std::memcpy(new_nodes, tape->nodes, sizeof(ad_node_t*) * tape->num_nodes);

        tape->nodes = new_nodes;
        tape->capacity = new_capacity;
    }

    tape->nodes[tape->num_nodes++] = node;
    __eshkol_ad_counters.tape_nodes++;
}

/**
 * @brief Clears a tape's recorded nodes and zeroes their accumulated gradients.
 *
 * Resets every currently-recorded node's `gradient` field to 0.0 and sets
 * the tape's node count back to zero, so the same arena-backed nodes array
 * can be reused for a fresh forward/backward pass without freeing memory.
 * Does not touch node values, types, or parent links.
 *
 * @param tape Tape to reset; no-op (with error log) if null.
 */
void arena_tape_reset(ad_tape_t* tape) {
    if (!tape) {
        eshkol_error("Cannot reset tape: null parameter");
        return;
    }

    size_t node_count = tape->num_nodes;

    for (size_t i = 0; i < node_count; i++) {
        if (tape->nodes[i]) {
            tape->nodes[i]->gradient = 0.0;
        }
    }

    tape->num_nodes = 0;
}

/**
 * @brief Fetches the node recorded at a given position on the tape.
 *
 * @param tape  Tape to read from; must be non-null.
 * @param index Zero-based position in evaluation order; must be less than
 *              the tape's current node count.
 * @return      The node at `index`, or nullptr if `tape` is null or `index`
 *              is out of bounds (both cases log an error).
 */
ad_node_t* arena_tape_get_node(const ad_tape_t* tape, size_t index) {
    if (!tape) {
        eshkol_error("Cannot get node from null tape");
        return nullptr;
    }

    if (index >= tape->num_nodes) {
        eshkol_error("Tape index out of bounds: %zu >= %zu", index, tape->num_nodes);
        return nullptr;
    }

    return tape->nodes[index];
}

/**
 * @brief Returns the number of nodes currently recorded on the tape.
 *
 * @param tape Tape to query; must be non-null.
 * @return     Current node count, or 0 if `tape` is null (logs an error).
 */
size_t arena_tape_get_node_count(const ad_tape_t* tape) {
    if (!tape) {
        eshkol_error("Cannot get node count from null tape");
        return 0;
    }

    return tape->num_nodes;
}

}  // extern "C"
