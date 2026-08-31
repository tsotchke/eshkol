#include <eshkol/module_visibility.h>

#include <cstring>

namespace eshkol {
namespace {

using RenameMap = std::map<std::string, std::string>;
using BoundNames = std::set<std::string>;

static void replace_name(char*& slot, const std::string& name) {
    if (!slot) return;
    char* replacement = new char[name.size() + 1];
    std::memcpy(replacement, name.c_str(), name.size() + 1);
    delete[] slot;
    slot = replacement;
}

static std::string private_name(const std::string& module_name,
                                const std::string& name) {
    std::string result = "__";
    for (char c : module_name) result += c == '.' ? '_' : c;
    result += "__";
    result += name;
    return result;
}

static void erase_bound(RenameMap& names, const BoundNames& bound) {
    for (const auto& name : bound) names.erase(name);
}

static void rename_ast(eshkol_ast_t* ast, const RenameMap& names,
                       const BoundNames& bound);

static void rename_quasiquote(eshkol_ast_t* ast, const RenameMap& names,
                              const BoundNames& bound) {
    if (!ast) return;
    if (ast->type == ESHKOL_CONS) {
        rename_quasiquote(ast->cons_cell.car, names, bound);
        rename_quasiquote(ast->cons_cell.cdr, names, bound);
        return;
    }
    if (ast->type != ESHKOL_OP) return;

    switch (ast->operation.op) {
        case ESHKOL_QUOTE_OP:
            return;
        case ESHKOL_QUASIQUOTE_OP:
            return;
        case ESHKOL_UNQUOTE_OP:
        case ESHKOL_UNQUOTE_SPLICING_OP:
            for (uint64_t i = 0; i < ast->operation.call_op.num_vars; ++i)
                rename_ast(&ast->operation.call_op.variables[i], names, bound);
            return;
        case ESHKOL_TENSOR_OP:
            for (uint64_t i = 0; i < ast->operation.tensor_op.total_elements; ++i)
                rename_quasiquote(&ast->operation.tensor_op.elements[i], names, bound);
            return;
        case ESHKOL_CALL_OP:
        case ESHKOL_COND_OP:
        case ESHKOL_IF_OP:
            rename_quasiquote(ast->operation.call_op.func, names, bound);
            for (uint64_t i = 0; i < ast->operation.call_op.num_vars; ++i)
                rename_quasiquote(&ast->operation.call_op.variables[i], names, bound);
            return;
        default:
            return;
    }
}

static void rename_macro_templates(eshkol_macro_def_t* macro,
                                   const RenameMap& names,
                                   const BoundNames& bound) {
    if (!macro) return;
    for (uint64_t r = 0; r < macro->num_rules; ++r) {
        eshkol_macro_template_t* template_ = macro->rules[r].template_;
        if (template_ && template_->literal)
            rename_ast(template_->literal, names, bound);
    }
}

static std::string ast_name(const eshkol_ast_t* ast) {
    if (!ast || ast->type != ESHKOL_VAR || !ast->variable.id) return {};
    return ast->variable.id;
}

static void rename_lambda(eshkol_ast_t* ast, const RenameMap& names,
                          const BoundNames& bound) {
    BoundNames lambda_bound = bound;
    for (uint64_t i = 0; i < ast->operation.lambda_op.num_params; ++i) {
        std::string name = ast_name(&ast->operation.lambda_op.parameters[i]);
        if (!name.empty()) lambda_bound.insert(name);
    }
    if (ast->operation.lambda_op.rest_param)
        lambda_bound.insert(ast->operation.lambda_op.rest_param);
    RenameMap body_names = names;
    erase_bound(body_names, lambda_bound);
    rename_ast(ast->operation.lambda_op.body, body_names, lambda_bound);
}

static void rename_let(eshkol_ast_t* ast, const RenameMap& names,
                       const BoundNames& bound) {
    const auto kind = ast->operation.op;
    BoundNames let_bound = bound;
    RenameMap value_names = names;

    if (kind == ESHKOL_LETREC_OP || kind == ESHKOL_LETREC_STAR_OP) {
        for (uint64_t i = 0; i < ast->operation.let_op.num_bindings; ++i) {
            const auto& binding = ast->operation.let_op.bindings[i];
            std::string name = binding.type == ESHKOL_CONS
                ? ast_name(binding.cons_cell.car) : std::string();
            if (!name.empty()) let_bound.insert(name);
        }
        erase_bound(value_names, let_bound);
    }

    for (uint64_t i = 0; i < ast->operation.let_op.num_bindings; ++i) {
        auto& binding = ast->operation.let_op.bindings[i];
        if (binding.type != ESHKOL_CONS) {
            rename_ast(&binding, value_names, let_bound);
            continue;
        }
        rename_ast(binding.cons_cell.cdr, value_names, let_bound);
        std::string name = ast_name(binding.cons_cell.car);
        if (!name.empty()) {
            let_bound.insert(name);
            if (kind == ESHKOL_LET_OP || kind == ESHKOL_LET_STAR_OP) {
                value_names = names;
                erase_bound(value_names, let_bound);
            }
        }
    }
    RenameMap body_names = names;
    erase_bound(body_names, let_bound);
    rename_ast(ast->operation.let_op.body, body_names, let_bound);
}

static void rename_let_values(eshkol_ast_t* ast, const RenameMap& names,
                              const BoundNames& bound) {
    const bool sequential = ast->operation.op == ESHKOL_LET_STAR_VALUES_OP;
    BoundNames values_bound = bound;
    RenameMap producer_names = names;

    for (uint64_t i = 0; i < ast->operation.let_values_op.num_bindings; ++i) {
        rename_ast(&ast->operation.let_values_op.producers[i], producer_names,
                   sequential ? values_bound : bound);
        for (uint64_t j = 0;
             j < ast->operation.let_values_op.binding_var_counts[i]; ++j) {
            const char* name = ast->operation.let_values_op.binding_vars[i][j];
            if (name) values_bound.insert(name);
        }
        if (sequential) erase_bound(producer_names, values_bound);
    }

    RenameMap body_names = names;
    erase_bound(body_names, values_bound);
    rename_ast(ast->operation.let_values_op.body, body_names, values_bound);
}

static void rename_call_operands(eshkol_ast_t* ast, const RenameMap& names,
                                 const BoundNames& bound) {
    rename_ast(ast->operation.call_op.func, names, bound);
    for (uint64_t i = 0; i < ast->operation.call_op.num_vars; ++i)
        rename_ast(&ast->operation.call_op.variables[i], names, bound);
}

static void rename_ast(eshkol_ast_t* ast, const RenameMap& names,
                       const BoundNames& bound) {
    if (!ast) return;
    if (ast->type == ESHKOL_VAR) {
        if (ast->variable.id && !bound.count(ast->variable.id)) {
            auto it = names.find(ast->variable.id);
            if (it != names.end()) replace_name(ast->variable.id, it->second);
        }
        return;
    }
    if (ast->type == ESHKOL_CONS) {
        rename_ast(ast->cons_cell.car, names, bound);
        rename_ast(ast->cons_cell.cdr, names, bound);
        return;
    }
    if (ast->type != ESHKOL_OP) return;

    switch (ast->operation.op) {
        case ESHKOL_QUOTE_OP:
            return;
        case ESHKOL_QUASIQUOTE_OP:
            for (uint64_t i = 0; i < ast->operation.call_op.num_vars; ++i)
                rename_quasiquote(&ast->operation.call_op.variables[i], names, bound);
            return;
        case ESHKOL_LAMBDA_OP:
            rename_lambda(ast, names, bound);
            return;
        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP:
        case ESHKOL_LETREC_STAR_OP:
            rename_let(ast, names, bound);
            return;
        case ESHKOL_DEFINE_OP:
            if (ast->operation.define_op.is_function) {
                BoundNames function_bound = bound;
                for (uint64_t i = 0; i < ast->operation.define_op.num_params; ++i) {
                    std::string name = ast_name(&ast->operation.define_op.parameters[i]);
                    if (!name.empty()) function_bound.insert(name);
                }
                if (ast->operation.define_op.rest_param)
                    function_bound.insert(ast->operation.define_op.rest_param);
                RenameMap body_names = names;
                erase_bound(body_names, function_bound);
                rename_ast(ast->operation.define_op.value, body_names,
                           function_bound);
            } else {
                rename_ast(ast->operation.define_op.value, names, bound);
            }
            return;
        case ESHKOL_SET_OP: {
            if (ast->operation.set_op.name && !bound.count(ast->operation.set_op.name)) {
                auto it = names.find(ast->operation.set_op.name);
                if (it != names.end()) replace_name(ast->operation.set_op.name, it->second);
            }
            rename_ast(ast->operation.set_op.value, names, bound);
            return;
        }
        case ESHKOL_DEFINE_SYNTAX_OP:
            rename_macro_templates(ast->operation.define_syntax_op.macro, names, bound);
            return;
        case ESHKOL_LET_SYNTAX_OP:
        case ESHKOL_LETREC_SYNTAX_OP:
            rename_ast(ast->operation.let_syntax_op.body, names, bound);
            return;
        case ESHKOL_SEQUENCE_OP:
        case ESHKOL_AND_OP:
        case ESHKOL_OR_OP:
            for (uint64_t i = 0; i < ast->operation.sequence_op.num_expressions; ++i)
                rename_ast(&ast->operation.sequence_op.expressions[i], names, bound);
            return;
        case ESHKOL_CALL_OP:
        case ESHKOL_COND_OP:
        case ESHKOL_IF_OP:
        case ESHKOL_CASE_OP:
        case ESHKOL_DO_OP:
        case ESHKOL_WHEN_OP:
        case ESHKOL_UNLESS_OP:
            rename_call_operands(ast, names, bound);
            return;
        case ESHKOL_EXTERN_OP:
            for (uint64_t i = 0; i < ast->operation.extern_op.num_params; ++i)
                rename_ast(&ast->operation.extern_op.parameters[i], names, bound);
            return;
        case ESHKOL_WITH_REGION_OP:
            for (uint64_t i = 0; i < ast->operation.with_region_op.num_body_exprs; ++i)
                rename_ast(&ast->operation.with_region_op.body[i], names, bound);
            return;
        case ESHKOL_BORROW_OP:
            rename_ast(ast->operation.borrow_op.value, names, bound);
            for (uint64_t i = 0; i < ast->operation.borrow_op.num_body_exprs; ++i)
                rename_ast(&ast->operation.borrow_op.body[i], names, bound);
            return;
        case ESHKOL_TENSOR_OP:
            for (uint64_t i = 0; i < ast->operation.tensor_op.total_elements; ++i)
                rename_ast(&ast->operation.tensor_op.elements[i], names, bound);
            return;
        case ESHKOL_MATCH_OP:
            rename_ast(ast->operation.match_op.expr, names, bound);
            for (uint64_t i = 0; i < ast->operation.match_op.num_clauses; ++i) {
                rename_ast(ast->operation.match_op.clauses[i].guard, names, bound);
                rename_ast(ast->operation.match_op.clauses[i].body, names, bound);
            }
            return;
        case ESHKOL_CALL_CC_OP:
            rename_ast(ast->operation.call_cc_op.proc, names, bound);
            return;
        case ESHKOL_DYNAMIC_WIND_OP:
            rename_ast(ast->operation.dynamic_wind_op.before, names, bound);
            rename_ast(ast->operation.dynamic_wind_op.thunk, names, bound);
            rename_ast(ast->operation.dynamic_wind_op.after, names, bound);
            return;
        case ESHKOL_GUARD_OP: {
            BoundNames clause_bound = bound;
            if (ast->operation.guard_op.var_name)
                clause_bound.insert(ast->operation.guard_op.var_name);
            RenameMap clause_names = names;
            erase_bound(clause_names, clause_bound);
            for (uint64_t i = 0; i < ast->operation.guard_op.num_clauses; ++i)
                rename_ast(&ast->operation.guard_op.clauses[i], clause_names,
                           clause_bound);
            for (uint64_t i = 0; i < ast->operation.guard_op.num_body_exprs; ++i)
                rename_ast(&ast->operation.guard_op.body[i], names, bound);
            return;
        }
        case ESHKOL_CALL_WITH_VALUES_OP:
            rename_ast(ast->operation.call_with_values_op.producer, names, bound);
            rename_ast(ast->operation.call_with_values_op.consumer, names, bound);
            return;
        case ESHKOL_LET_VALUES_OP:
        case ESHKOL_LET_STAR_VALUES_OP:
            rename_let_values(ast, names, bound);
            return;
        case ESHKOL_VALUES_OP:
            for (uint64_t i = 0; i < ast->operation.values_op.num_values; ++i)
                rename_ast(&ast->operation.values_op.expressions[i], names, bound);
            return;
        case ESHKOL_DIFF_OP:
            rename_ast(ast->operation.diff_op.expression, names, bound);
            return;
        case ESHKOL_GRADIENT_OP:
            rename_ast(ast->operation.gradient_op.function, names, bound);
            rename_ast(ast->operation.gradient_op.point, names, bound);
            return;
        case ESHKOL_DERIVATIVE_OP:
            rename_ast(ast->operation.derivative_op.function, names, bound);
            rename_ast(ast->operation.derivative_op.point, names, bound);
            return;
        case ESHKOL_TAYLOR_OP:
        case ESHKOL_DERIVATIVE_N_OP:
            rename_ast(ast->operation.taylor_op.function, names, bound);
            rename_ast(ast->operation.taylor_op.point, names, bound);
            rename_ast(ast->operation.taylor_op.order, names, bound);
            return;
        case ESHKOL_DIRECTIONAL_DERIV_OP:
            rename_ast(ast->operation.directional_deriv_op.function, names, bound);
            rename_ast(ast->operation.directional_deriv_op.point, names, bound);
            rename_ast(ast->operation.directional_deriv_op.direction, names, bound);
            return;
        case ESHKOL_JACOBIAN_OP:
            rename_ast(ast->operation.jacobian_op.function, names, bound);
            rename_ast(ast->operation.jacobian_op.point, names, bound);
            return;
        case ESHKOL_HESSIAN_OP:
            rename_ast(ast->operation.hessian_op.function, names, bound);
            rename_ast(ast->operation.hessian_op.point, names, bound);
            return;
        case ESHKOL_DIVERGENCE_OP:
            rename_ast(ast->operation.divergence_op.function, names, bound);
            rename_ast(ast->operation.divergence_op.point, names, bound);
            return;
        case ESHKOL_CURL_OP:
            rename_ast(ast->operation.curl_op.function, names, bound);
            rename_ast(ast->operation.curl_op.point, names, bound);
            return;
        case ESHKOL_LAPLACIAN_OP:
            rename_ast(ast->operation.laplacian_op.function, names, bound);
            rename_ast(ast->operation.laplacian_op.point, names, bound);
            return;
        case ESHKOL_OWNED_OP:
            rename_ast(ast->operation.owned_op.value, names, bound);
            return;
        case ESHKOL_MOVE_OP:
            rename_ast(ast->operation.move_op.value, names, bound);
            return;
        case ESHKOL_SHARED_OP:
            rename_ast(ast->operation.shared_op.value, names, bound);
            return;
        case ESHKOL_WEAK_REF_OP:
            rename_ast(ast->operation.weak_ref_op.value, names, bound);
            return;
        case ESHKOL_THE_OP:
            rename_ast(ast->operation.the_op.expr, names, bound);
            return;
        case ESHKOL_MAKE_PARAMETER_OP:
            rename_call_operands(ast, names, bound);
            return;
        case ESHKOL_RAISE_OP:
            rename_ast(ast->operation.raise_op.exception, names, bound);
            return;
        default:
            if ((ast->operation.op >= ESHKOL_UNIFY_OP &&
                 ast->operation.op <= ESHKOL_WORKSPACE_PRED_OP) ||
                ast->operation.op == ESHKOL_KB_QUERY_PREFIX_OP ||
                (ast->operation.op >= ESHKOL_DNC_MAKE_OP &&
                 ast->operation.op <= ESHKOL_SDNC_PRED_OP)) {
                rename_call_operands(ast, names, bound);
            }
            return;
    }
}

}  // namespace

void rename_private_symbols(std::vector<eshkol_ast_t>& asts,
                            const std::string& module_name,
                            const std::set<std::string>& exports) {
    RenameMap names;
    for (auto& ast : asts) {
        if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_DEFINE_OP ||
            !ast.operation.define_op.name) continue;
        const std::string name = ast.operation.define_op.name;
        if (!exports.count(name)) names.emplace(name, private_name(module_name, name));
    }
    if (names.empty()) return;

    for (auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_OP &&
            ast.operation.define_op.name) {
            auto it = names.find(ast.operation.define_op.name);
            if (it != names.end()) replace_name(ast.operation.define_op.name, it->second);
        }
        rename_ast(&ast, names, {});
    }
}

}  // namespace eshkol
