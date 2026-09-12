// Copyright (C) tsotchke
// SPDX-License-Identifier: MIT
#ifndef ESHKOL_CORE_AST_ROUTING_H
#define ESHKOL_CORE_AST_ROUTING_H

#include <eshkol/eshkol.h>
#include <cstdlib>
#include <type_traits>

namespace eshkol {

// A consumer declares its closed result domain and partitions the operation
// domain into named policies. There is no fallback group: every operation must
// be named exactly once, including those that intentionally have no effect.
template <auto Route, eshkol_op_t... Operations>
struct AstRouteGroup {
    static constexpr auto route = Route;
    template <eshkol_op_t Operation>
    static constexpr unsigned matches = (0u + ... + unsigned(Operation == Operations));
};

namespace ast_routing_detail {
template <eshkol_op_t Operation, class First, class... Rest>
constexpr auto select() {
    if constexpr (First::template matches<Operation> == 1) {
        return First::route;
    } else {
        static_assert(sizeof...(Rest) != 0, "AST operation has no routing policy");
        return select<Operation, Rest...>();
    }
}

template <eshkol_op_t Operation, class... Groups>
constexpr auto checkedRoute() {
    static_assert((0u + ... + Groups::template matches<Operation>) == 1,
                  "AST operation must have exactly one routing policy");
    return select<Operation, Groups...>();
}
} // namespace ast_routing_detail

// The sole runtime operation dispatch. Instantiation checks every operation,
// even when a particular caller reaches only a subset at runtime. Adding an
// operation requires a deliberate policy in every consumer. Invalid numeric
// tags abort rather than borrowing another operation's union layout.
template <class First, class... Rest>
inline auto routeAstOperation(eshkol_op_t operation, First, Rest...) {
    static_assert((std::is_same_v<decltype(First::route), decltype(Rest::route)> && ...),
                  "AST routing groups must share one result enum");
    ESHKOL_EXHAUSTIVE_SWITCH_BEGIN
    switch (operation) {
        case ESHKOL_INVALID_OP: return ast_routing_detail::checkedRoute<ESHKOL_INVALID_OP, First, Rest...>();
        case ESHKOL_COMPOSE_OP: return ast_routing_detail::checkedRoute<ESHKOL_COMPOSE_OP, First, Rest...>();
        case ESHKOL_IF_OP: return ast_routing_detail::checkedRoute<ESHKOL_IF_OP, First, Rest...>();
        case ESHKOL_ADD_OP: return ast_routing_detail::checkedRoute<ESHKOL_ADD_OP, First, Rest...>();
        case ESHKOL_SUB_OP: return ast_routing_detail::checkedRoute<ESHKOL_SUB_OP, First, Rest...>();
        case ESHKOL_MUL_OP: return ast_routing_detail::checkedRoute<ESHKOL_MUL_OP, First, Rest...>();
        case ESHKOL_DIV_OP: return ast_routing_detail::checkedRoute<ESHKOL_DIV_OP, First, Rest...>();
        case ESHKOL_CALL_OP: return ast_routing_detail::checkedRoute<ESHKOL_CALL_OP, First, Rest...>();
        case ESHKOL_DEFINE_OP: return ast_routing_detail::checkedRoute<ESHKOL_DEFINE_OP, First, Rest...>();
        case ESHKOL_SEQUENCE_OP: return ast_routing_detail::checkedRoute<ESHKOL_SEQUENCE_OP, First, Rest...>();
        case ESHKOL_EXTERN_OP: return ast_routing_detail::checkedRoute<ESHKOL_EXTERN_OP, First, Rest...>();
        case ESHKOL_EXTERN_VAR_OP: return ast_routing_detail::checkedRoute<ESHKOL_EXTERN_VAR_OP, First, Rest...>();
        case ESHKOL_LAMBDA_OP: return ast_routing_detail::checkedRoute<ESHKOL_LAMBDA_OP, First, Rest...>();
        case ESHKOL_LET_OP: return ast_routing_detail::checkedRoute<ESHKOL_LET_OP, First, Rest...>();
        case ESHKOL_LET_STAR_OP: return ast_routing_detail::checkedRoute<ESHKOL_LET_STAR_OP, First, Rest...>();
        case ESHKOL_LETREC_OP: return ast_routing_detail::checkedRoute<ESHKOL_LETREC_OP, First, Rest...>();
        case ESHKOL_LETREC_STAR_OP: return ast_routing_detail::checkedRoute<ESHKOL_LETREC_STAR_OP, First, Rest...>();
        case ESHKOL_AND_OP: return ast_routing_detail::checkedRoute<ESHKOL_AND_OP, First, Rest...>();
        case ESHKOL_OR_OP: return ast_routing_detail::checkedRoute<ESHKOL_OR_OP, First, Rest...>();
        case ESHKOL_COND_OP: return ast_routing_detail::checkedRoute<ESHKOL_COND_OP, First, Rest...>();
        case ESHKOL_CASE_OP: return ast_routing_detail::checkedRoute<ESHKOL_CASE_OP, First, Rest...>();
        case ESHKOL_MATCH_OP: return ast_routing_detail::checkedRoute<ESHKOL_MATCH_OP, First, Rest...>();
        case ESHKOL_DO_OP: return ast_routing_detail::checkedRoute<ESHKOL_DO_OP, First, Rest...>();
        case ESHKOL_WHEN_OP: return ast_routing_detail::checkedRoute<ESHKOL_WHEN_OP, First, Rest...>();
        case ESHKOL_UNLESS_OP: return ast_routing_detail::checkedRoute<ESHKOL_UNLESS_OP, First, Rest...>();
        case ESHKOL_QUOTE_OP: return ast_routing_detail::checkedRoute<ESHKOL_QUOTE_OP, First, Rest...>();
        case ESHKOL_QUASIQUOTE_OP: return ast_routing_detail::checkedRoute<ESHKOL_QUASIQUOTE_OP, First, Rest...>();
        case ESHKOL_UNQUOTE_OP: return ast_routing_detail::checkedRoute<ESHKOL_UNQUOTE_OP, First, Rest...>();
        case ESHKOL_UNQUOTE_SPLICING_OP: return ast_routing_detail::checkedRoute<ESHKOL_UNQUOTE_SPLICING_OP, First, Rest...>();
        case ESHKOL_SET_OP: return ast_routing_detail::checkedRoute<ESHKOL_SET_OP, First, Rest...>();
        case ESHKOL_DEFINE_TYPE_OP: return ast_routing_detail::checkedRoute<ESHKOL_DEFINE_TYPE_OP, First, Rest...>();
        case ESHKOL_IMPORT_OP: return ast_routing_detail::checkedRoute<ESHKOL_IMPORT_OP, First, Rest...>();
        case ESHKOL_REQUIRE_OP: return ast_routing_detail::checkedRoute<ESHKOL_REQUIRE_OP, First, Rest...>();
        case ESHKOL_PROVIDE_OP: return ast_routing_detail::checkedRoute<ESHKOL_PROVIDE_OP, First, Rest...>();
        case ESHKOL_WITH_REGION_OP: return ast_routing_detail::checkedRoute<ESHKOL_WITH_REGION_OP, First, Rest...>();
        case ESHKOL_OWNED_OP: return ast_routing_detail::checkedRoute<ESHKOL_OWNED_OP, First, Rest...>();
        case ESHKOL_MOVE_OP: return ast_routing_detail::checkedRoute<ESHKOL_MOVE_OP, First, Rest...>();
        case ESHKOL_BORROW_OP: return ast_routing_detail::checkedRoute<ESHKOL_BORROW_OP, First, Rest...>();
        case ESHKOL_SHARED_OP: return ast_routing_detail::checkedRoute<ESHKOL_SHARED_OP, First, Rest...>();
        case ESHKOL_WEAK_REF_OP: return ast_routing_detail::checkedRoute<ESHKOL_WEAK_REF_OP, First, Rest...>();
        case ESHKOL_TENSOR_OP: return ast_routing_detail::checkedRoute<ESHKOL_TENSOR_OP, First, Rest...>();
        case ESHKOL_DIFF_OP: return ast_routing_detail::checkedRoute<ESHKOL_DIFF_OP, First, Rest...>();
        case ESHKOL_DERIVATIVE_OP: return ast_routing_detail::checkedRoute<ESHKOL_DERIVATIVE_OP, First, Rest...>();
        case ESHKOL_GRADIENT_OP: return ast_routing_detail::checkedRoute<ESHKOL_GRADIENT_OP, First, Rest...>();
        case ESHKOL_JACOBIAN_OP: return ast_routing_detail::checkedRoute<ESHKOL_JACOBIAN_OP, First, Rest...>();
        case ESHKOL_HESSIAN_OP: return ast_routing_detail::checkedRoute<ESHKOL_HESSIAN_OP, First, Rest...>();
        case ESHKOL_DIVERGENCE_OP: return ast_routing_detail::checkedRoute<ESHKOL_DIVERGENCE_OP, First, Rest...>();
        case ESHKOL_CURL_OP: return ast_routing_detail::checkedRoute<ESHKOL_CURL_OP, First, Rest...>();
        case ESHKOL_LAPLACIAN_OP: return ast_routing_detail::checkedRoute<ESHKOL_LAPLACIAN_OP, First, Rest...>();
        case ESHKOL_DIRECTIONAL_DERIV_OP: return ast_routing_detail::checkedRoute<ESHKOL_DIRECTIONAL_DERIV_OP, First, Rest...>();
        case ESHKOL_TAYLOR_OP: return ast_routing_detail::checkedRoute<ESHKOL_TAYLOR_OP, First, Rest...>();
        case ESHKOL_DERIVATIVE_N_OP: return ast_routing_detail::checkedRoute<ESHKOL_DERIVATIVE_N_OP, First, Rest...>();
        case ESHKOL_TYPE_ANNOTATION_OP: return ast_routing_detail::checkedRoute<ESHKOL_TYPE_ANNOTATION_OP, First, Rest...>();
        case ESHKOL_FORALL_OP: return ast_routing_detail::checkedRoute<ESHKOL_FORALL_OP, First, Rest...>();
        case ESHKOL_GUARD_OP: return ast_routing_detail::checkedRoute<ESHKOL_GUARD_OP, First, Rest...>();
        case ESHKOL_RAISE_OP: return ast_routing_detail::checkedRoute<ESHKOL_RAISE_OP, First, Rest...>();
        case ESHKOL_LET_VALUES_OP: return ast_routing_detail::checkedRoute<ESHKOL_LET_VALUES_OP, First, Rest...>();
        case ESHKOL_LET_STAR_VALUES_OP: return ast_routing_detail::checkedRoute<ESHKOL_LET_STAR_VALUES_OP, First, Rest...>();
        case ESHKOL_VALUES_OP: return ast_routing_detail::checkedRoute<ESHKOL_VALUES_OP, First, Rest...>();
        case ESHKOL_CALL_WITH_VALUES_OP: return ast_routing_detail::checkedRoute<ESHKOL_CALL_WITH_VALUES_OP, First, Rest...>();
        case ESHKOL_DEFINE_SYNTAX_OP: return ast_routing_detail::checkedRoute<ESHKOL_DEFINE_SYNTAX_OP, First, Rest...>();
        case ESHKOL_LET_SYNTAX_OP: return ast_routing_detail::checkedRoute<ESHKOL_LET_SYNTAX_OP, First, Rest...>();
        case ESHKOL_LETREC_SYNTAX_OP: return ast_routing_detail::checkedRoute<ESHKOL_LETREC_SYNTAX_OP, First, Rest...>();
        case ESHKOL_CALL_CC_OP: return ast_routing_detail::checkedRoute<ESHKOL_CALL_CC_OP, First, Rest...>();
        case ESHKOL_DYNAMIC_WIND_OP: return ast_routing_detail::checkedRoute<ESHKOL_DYNAMIC_WIND_OP, First, Rest...>();
        case ESHKOL_LOGIC_VAR_OP: return ast_routing_detail::checkedRoute<ESHKOL_LOGIC_VAR_OP, First, Rest...>();
        case ESHKOL_UNIFY_OP: return ast_routing_detail::checkedRoute<ESHKOL_UNIFY_OP, First, Rest...>();
        case ESHKOL_MAKE_SUBST_OP: return ast_routing_detail::checkedRoute<ESHKOL_MAKE_SUBST_OP, First, Rest...>();
        case ESHKOL_WALK_OP: return ast_routing_detail::checkedRoute<ESHKOL_WALK_OP, First, Rest...>();
        case ESHKOL_MAKE_FACT_OP: return ast_routing_detail::checkedRoute<ESHKOL_MAKE_FACT_OP, First, Rest...>();
        case ESHKOL_MAKE_KB_OP: return ast_routing_detail::checkedRoute<ESHKOL_MAKE_KB_OP, First, Rest...>();
        case ESHKOL_KB_ASSERT_OP: return ast_routing_detail::checkedRoute<ESHKOL_KB_ASSERT_OP, First, Rest...>();
        case ESHKOL_KB_QUERY_OP: return ast_routing_detail::checkedRoute<ESHKOL_KB_QUERY_OP, First, Rest...>();
        case ESHKOL_MAKE_FACTOR_GRAPH_OP: return ast_routing_detail::checkedRoute<ESHKOL_MAKE_FACTOR_GRAPH_OP, First, Rest...>();
        case ESHKOL_FG_ADD_FACTOR_OP: return ast_routing_detail::checkedRoute<ESHKOL_FG_ADD_FACTOR_OP, First, Rest...>();
        case ESHKOL_FG_INFER_OP: return ast_routing_detail::checkedRoute<ESHKOL_FG_INFER_OP, First, Rest...>();
        case ESHKOL_FREE_ENERGY_OP: return ast_routing_detail::checkedRoute<ESHKOL_FREE_ENERGY_OP, First, Rest...>();
        case ESHKOL_EXPECTED_FREE_ENERGY_OP: return ast_routing_detail::checkedRoute<ESHKOL_EXPECTED_FREE_ENERGY_OP, First, Rest...>();
        case ESHKOL_MAKE_WORKSPACE_OP: return ast_routing_detail::checkedRoute<ESHKOL_MAKE_WORKSPACE_OP, First, Rest...>();
        case ESHKOL_WS_REGISTER_OP: return ast_routing_detail::checkedRoute<ESHKOL_WS_REGISTER_OP, First, Rest...>();
        case ESHKOL_WS_STEP_OP: return ast_routing_detail::checkedRoute<ESHKOL_WS_STEP_OP, First, Rest...>();
        case ESHKOL_FG_UPDATE_CPT_OP: return ast_routing_detail::checkedRoute<ESHKOL_FG_UPDATE_CPT_OP, First, Rest...>();
        case ESHKOL_FG_OBSERVE_OP: return ast_routing_detail::checkedRoute<ESHKOL_FG_OBSERVE_OP, First, Rest...>();
        case ESHKOL_LOGIC_VAR_PRED_OP: return ast_routing_detail::checkedRoute<ESHKOL_LOGIC_VAR_PRED_OP, First, Rest...>();
        case ESHKOL_SUBSTITUTION_PRED_OP: return ast_routing_detail::checkedRoute<ESHKOL_SUBSTITUTION_PRED_OP, First, Rest...>();
        case ESHKOL_KB_PRED_OP: return ast_routing_detail::checkedRoute<ESHKOL_KB_PRED_OP, First, Rest...>();
        case ESHKOL_FACT_PRED_OP: return ast_routing_detail::checkedRoute<ESHKOL_FACT_PRED_OP, First, Rest...>();
        case ESHKOL_FACTOR_GRAPH_PRED_OP: return ast_routing_detail::checkedRoute<ESHKOL_FACTOR_GRAPH_PRED_OP, First, Rest...>();
        case ESHKOL_WORKSPACE_PRED_OP: return ast_routing_detail::checkedRoute<ESHKOL_WORKSPACE_PRED_OP, First, Rest...>();
        case ESHKOL_CASE_LAMBDA_OP: return ast_routing_detail::checkedRoute<ESHKOL_CASE_LAMBDA_OP, First, Rest...>();
        case ESHKOL_DEFINE_RECORD_TYPE_OP: return ast_routing_detail::checkedRoute<ESHKOL_DEFINE_RECORD_TYPE_OP, First, Rest...>();
        case ESHKOL_PARAMETERIZE_OP: return ast_routing_detail::checkedRoute<ESHKOL_PARAMETERIZE_OP, First, Rest...>();
        case ESHKOL_MAKE_PARAMETER_OP: return ast_routing_detail::checkedRoute<ESHKOL_MAKE_PARAMETER_OP, First, Rest...>();
        case ESHKOL_COND_EXPAND_OP: return ast_routing_detail::checkedRoute<ESHKOL_COND_EXPAND_OP, First, Rest...>();
        case ESHKOL_INCLUDE_OP: return ast_routing_detail::checkedRoute<ESHKOL_INCLUDE_OP, First, Rest...>();
        case ESHKOL_SYNTAX_ERROR_OP: return ast_routing_detail::checkedRoute<ESHKOL_SYNTAX_ERROR_OP, First, Rest...>();
        case ESHKOL_KB_QUERY_PREFIX_OP: return ast_routing_detail::checkedRoute<ESHKOL_KB_QUERY_PREFIX_OP, First, Rest...>();
        case ESHKOL_DNC_MAKE_OP: return ast_routing_detail::checkedRoute<ESHKOL_DNC_MAKE_OP, First, Rest...>();
        case ESHKOL_DNC_CONTENT_ADDR_OP: return ast_routing_detail::checkedRoute<ESHKOL_DNC_CONTENT_ADDR_OP, First, Rest...>();
        case ESHKOL_DNC_LOC_ADDR_OP: return ast_routing_detail::checkedRoute<ESHKOL_DNC_LOC_ADDR_OP, First, Rest...>();
        case ESHKOL_DNC_READ_OP: return ast_routing_detail::checkedRoute<ESHKOL_DNC_READ_OP, First, Rest...>();
        case ESHKOL_DNC_WRITE_OP: return ast_routing_detail::checkedRoute<ESHKOL_DNC_WRITE_OP, First, Rest...>();
        case ESHKOL_DNC_ALLOC_WEIGHTS_OP: return ast_routing_detail::checkedRoute<ESHKOL_DNC_ALLOC_WEIGHTS_OP, First, Rest...>();
        case ESHKOL_DNC_READ_GRAD_OP: return ast_routing_detail::checkedRoute<ESHKOL_DNC_READ_GRAD_OP, First, Rest...>();
        case ESHKOL_DNC_PRED_OP: return ast_routing_detail::checkedRoute<ESHKOL_DNC_PRED_OP, First, Rest...>();
        case ESHKOL_SDNC_PROGRAM_OP: return ast_routing_detail::checkedRoute<ESHKOL_SDNC_PROGRAM_OP, First, Rest...>();
        case ESHKOL_SDNC_RUN_OP: return ast_routing_detail::checkedRoute<ESHKOL_SDNC_RUN_OP, First, Rest...>();
        case ESHKOL_SDNC_WEIGHT_GRAD_OP: return ast_routing_detail::checkedRoute<ESHKOL_SDNC_WEIGHT_GRAD_OP, First, Rest...>();
        case ESHKOL_SDNC_PARAMS_OP: return ast_routing_detail::checkedRoute<ESHKOL_SDNC_PARAMS_OP, First, Rest...>();
        case ESHKOL_SDNC_SET_PARAMS_OP: return ast_routing_detail::checkedRoute<ESHKOL_SDNC_SET_PARAMS_OP, First, Rest...>();
        case ESHKOL_SDNC_IMPROVE_OP: return ast_routing_detail::checkedRoute<ESHKOL_SDNC_IMPROVE_OP, First, Rest...>();
        case ESHKOL_SDNC_PRED_OP: return ast_routing_detail::checkedRoute<ESHKOL_SDNC_PRED_OP, First, Rest...>();
        case ESHKOL_THE_OP: return ast_routing_detail::checkedRoute<ESHKOL_THE_OP, First, Rest...>();
    }
    ESHKOL_EXHAUSTIVE_SWITCH_END
    std::abort();
}

} // namespace eshkol
#endif
