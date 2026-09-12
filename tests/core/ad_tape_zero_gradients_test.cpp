#include <cstdio>

#include "../../lib/core/arena_memory.h"

static int forward_once(void* context, ad_tape_t* tape, ad_node_t** output) {
    arena_t* arena = static_cast<arena_t*>(context);
    ad_node_t* node = arena_allocate_ad_node(arena);
    if (!node) return 0;
    node->value = 7.0;
    arena_tape_add_node(tape, node);
    ad_node_t* variables[1] = {node};
    arena_tape_set_variables(tape, variables, 1);
    *output = node;
    return 1;
}

static int backward_once(void*, ad_tape_t*, ad_node_t* output) {
    output->gradient = 3.0;
    return 1;
}

int main() {
    arena_t* arena = get_global_arena();
    ad_tape_t* tape = arena_allocate_tape(arena, 4);
    if (!tape) return 1;

    ad_node_t* scalar = arena_allocate_ad_node(arena);
    ad_node_t* tensor = arena_allocate_ad_node(arena);
    if (!scalar || !tensor) return 1;
    scalar->gradient = 9.0;
    tensor->tensor_value = tensor;
    tensor->ndim = 1;
    tensor->shape = static_cast<int64_t*>(arena_allocate_aligned(arena, sizeof(int64_t), 8));
    tensor->tensor_gradient = arena_allocate_aligned(arena, 2 * sizeof(double), 8);
    if (!tensor->shape || !tensor->tensor_gradient) return 1;
    tensor->shape[0] = 2;
    static_cast<double*>(tensor->tensor_gradient)[0] = 3.0;
    static_cast<double*>(tensor->tensor_gradient)[1] = 4.0;

    arena_tape_add_node(tape, scalar);
    arena_tape_add_node(tape, tensor);
    arena_tape_zero_gradients(tape);

    const double* gradient = static_cast<const double*>(tensor->tensor_gradient);
    if (scalar->gradient != 0.0 || gradient[0] != 0.0 || gradient[1] != 0.0) return 1;
    eshkol_ad_counters_reset();
    double value = 0.0;
    double gradients[1] = {0.0};
    if (!eshkol_value_and_grad(arena, tape, forward_once, backward_once,
                               &value, gradients, 1)) return 1;
    if (value != 7.0 || gradients[0] != 3.0 ||
        eshkol_ad_counter_primal_calls() != 1 ||
        eshkol_ad_counter_reverse_passes() != 1) return 1;
    std::puts("PASS: tape zeroes scalar and tensor gradients");
    return 0;
}
