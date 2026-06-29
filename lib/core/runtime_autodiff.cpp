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

void debug_print_ad_mode(const char* context) {
    std::fprintf(stderr, "[AD_MODE_DEBUG] %s: __ad_mode_active = %s\n",
                 context, __ad_mode_active ? "TRUE" : "FALSE");
}

void debug_print_ptr(const char* context, void* ptr) {
    std::fprintf(stderr, "[PTR_DEBUG] %s: ptr = %p (0x%llx)\n",
                 context, ptr, (unsigned long long)(uintptr_t)ptr);
}

thread_local ad_tape_t* __ad_tape_stack[ESHKOL_ARENA_MAX_TAPE_DEPTH] = {nullptr};
thread_local uint64_t __ad_tape_depth = 0;

// ESH-0070: runtime forward-mode perturbation level (see codegen_context.h).
thread_local uint64_t __ad_pert_level = 0;

thread_local void* __outer_ad_node_storage = nullptr;
thread_local void* __outer_ad_node_to_inner = nullptr;
thread_local void* __outer_grad_accumulator = nullptr;
thread_local void* __inner_var_node_ptr = nullptr;
thread_local uint64_t __gradient_x_degree = 0;

thread_local void* __outer_ad_node_stack[ESHKOL_ARENA_MAX_TAPE_DEPTH] = {nullptr};
thread_local uint64_t __outer_ad_node_depth = 0;

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

    return tape;
}

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
}

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

size_t arena_tape_get_node_count(const ad_tape_t* tape) {
    if (!tape) {
        eshkol_error("Cannot get node count from null tape");
        return 0;
    }

    return tape->num_nodes;
}

}  // extern "C"
