/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * HomoiconicCodegen - Code-as-data (quote/S-expression) code generation
 */

#include <eshkol/backend/homoiconic_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>

using namespace llvm;

namespace eshkol {

HomoiconicCodegen::HomoiconicCodegen(CodegenContext& ctx,
                                      TaggedValueCodegen& tagged,
                                      CollectionCodegen& collection,
                                      StringIOCodegen& string_io)
    : ctx_(ctx), tagged_(tagged), collection_(collection), string_io_(string_io) {}

// === Helper Methods ===

Value* HomoiconicCodegen::packNull() {
    return tagged_.packNull();
}

Value* HomoiconicCodegen::packPtr(Value* ptr, eshkol_value_type_t type) {
    return tagged_.packPtr(ptr, type);
}

Value* HomoiconicCodegen::consFromTagged(Value* car, Value* cdr) {
    return collection_.allocConsCell(car, cdr);
}

// === Quote Operations ===

Value* HomoiconicCodegen::quoteAST(const eshkol_ast_t* ast) {
    if (!ast) return packNull();

    switch (ast->type) {
        case ESHKOL_INT64:
            // Return integer directly (not as list)
            return tagged_.packInt64(
                ConstantInt::get(ctx_.int64Type(), ast->int64_val),
                true);

        case ESHKOL_DOUBLE:
            // Return double directly (not as list)
            return tagged_.packDouble(
                ConstantFP::get(ctx_.doubleType(), ast->double_val));

        case ESHKOL_VAR:
            // Return symbol as string - use STRING_PTR type for symbols
            return tagged_.packPtr(
                string_io_.createString(ast->variable.id),
                ESHKOL_VALUE_STRING_PTR);

        case ESHKOL_BOOL:
            // Return boolean as #t or #f symbol
            return tagged_.packPtr(
                string_io_.createString(ast->int64_val ? "#t" : "#f"),
                ESHKOL_VALUE_STRING_PTR);

        case ESHKOL_STRING:
            // Return string literal
            return tagged_.packPtr(
                string_io_.createString(ast->str_val.ptr),
                ESHKOL_VALUE_STRING_PTR);

        case ESHKOL_CHAR:
            // Return character as #\char symbol
            {
                char char_buf[16];
                char ch = (char)ast->int64_val;
                if (ch == ' ') snprintf(char_buf, sizeof(char_buf), "#\\space");
                else if (ch == '\n') snprintf(char_buf, sizeof(char_buf), "#\\newline");
                else if (ch == '\t') snprintf(char_buf, sizeof(char_buf), "#\\tab");
                else snprintf(char_buf, sizeof(char_buf), "#\\%c", ch);
                return tagged_.packPtr(
                    string_io_.createString(char_buf),
                    ESHKOL_VALUE_STRING_PTR);
            }

        case ESHKOL_OP:
            return quoteOperation(&ast->operation);

        case ESHKOL_NULL:
            // Empty list / null - return properly tagged NULL value
            return packNull();

        case ESHKOL_UINT64:
            // Type 0 often represents empty/null in quoted data
            if (ast->int64_val == 0) {
                return packNull();
            }
            return tagged_.packInt64(
                ConstantInt::get(ctx_.int64Type(), ast->int64_val), true);

        default:
            // Unknown type - return null
            eshkol_debug("quoteAST: unhandled type %d", ast->type);
            return packNull();
    }
}

Value* HomoiconicCodegen::quoteOperation(const eshkol_operations_t* op) {
    if (!op) return packNull();

    switch (op->op) {
        case ESHKOL_CALL_OP: {
            // Build list: (op arg1 arg2 ...) and wrap as tagged_value
            Value* list_ptr = quoteList(op);
            if (list_ptr == ConstantInt::get(ctx_.int64Type(), 0)) {
                return packNull();
            }
            return tagged_.packPtr(
                ctx_.builder().CreateIntToPtr(list_ptr, ctx_.builder().getPtrTy()),
                ESHKOL_VALUE_CONS_PTR);
        }

        case ESHKOL_LAMBDA_OP: {
            // Handle nested lambdas in S-expression generation
            Value* nested_sexpr = lambdaToSExpr(op);
            if (nested_sexpr == ConstantInt::get(ctx_.int64Type(), 0)) {
                return packNull();
            }
            return tagged_.packPtr(
                ctx_.builder().CreateIntToPtr(nested_sexpr, ctx_.builder().getPtrTy()),
                ESHKOL_VALUE_CONS_PTR);
        }

        case ESHKOL_IF_OP: {
            // Build (if test then else)
            // IF_OP uses call_op structure: variables[0]=condition, variables[1]=then, variables[2]=else
            Value* if_sym = tagged_.packPtr(
                string_io_.createString("if"), ESHKOL_VALUE_STRING_PTR);

            // Build from right to left: else -> then -> test -> if
            Value* result = packNull();

            // Add else branch if present (variables[2])
            if (op->call_op.num_vars >= 3) {
                Value* else_branch = quoteAST(&op->call_op.variables[2]);
                Value* cons_int = consFromTagged(else_branch, result);
                result = tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(cons_int, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);
            }

            // Add then branch (variables[1])
            if (op->call_op.num_vars >= 2) {
                Value* then_branch = quoteAST(&op->call_op.variables[1]);
                Value* cons_int = consFromTagged(then_branch, result);
                result = tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(cons_int, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);
            }

            // Add condition (variables[0])
            if (op->call_op.num_vars >= 1) {
                Value* test = quoteAST(&op->call_op.variables[0]);
                Value* cons_int = consFromTagged(test, result);
                result = tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(cons_int, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);
            }

            Value* final_cons = consFromTagged(if_sym, result);
            return tagged_.packPtr(
                ctx_.builder().CreateIntToPtr(final_cons, ctx_.builder().getPtrTy()),
                ESHKOL_VALUE_CONS_PTR);
        }

        case ESHKOL_AND_OP: {
            // Build (and expr1 expr2 ...)
            return quoteNaryOp("and", op->sequence_op.expressions, op->sequence_op.num_expressions);
        }

        case ESHKOL_OR_OP: {
            // Build (or expr1 expr2 ...)
            return quoteNaryOp("or", op->sequence_op.expressions, op->sequence_op.num_expressions);
        }

        case ESHKOL_COND_OP: {
            // Build (cond (test1 expr1) (test2 expr2) ...)
            Value* cond_sym = tagged_.packPtr(
                string_io_.createString("cond"), ESHKOL_VALUE_STRING_PTR);
            Value* result = packNull();

            // Build clauses from right to left
            for (int64_t i = op->call_op.num_vars - 1; i >= 0; i--) {
                const eshkol_ast_t* clause_ast = &op->call_op.variables[i];
                Value* clause;
                if (clause_ast->type == ESHKOL_OP && clause_ast->operation.op == ESHKOL_CALL_OP) {
                    Value* test = quoteAST(clause_ast->operation.call_op.func);
                    // Build (test expr1 expr2 ...) clause
                    clause = packNull();
                    for (int64_t j = clause_ast->operation.call_op.num_vars - 1; j >= 0; j--) {
                        Value* expr = quoteAST(&clause_ast->operation.call_op.variables[j]);
                        Value* cons_int = consFromTagged(expr, clause);
                        clause = tagged_.packPtr(
                            ctx_.builder().CreateIntToPtr(cons_int, ctx_.builder().getPtrTy()),
                            ESHKOL_VALUE_CONS_PTR);
                    }
                    Value* test_cons = consFromTagged(test, clause);
                    clause = tagged_.packPtr(
                        ctx_.builder().CreateIntToPtr(test_cons, ctx_.builder().getPtrTy()),
                        ESHKOL_VALUE_CONS_PTR);
                } else {
                    // Fallback: just quote the clause directly
                    clause = quoteAST(clause_ast);
                }

                Value* cons_int = consFromTagged(clause, result);
                result = tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(cons_int, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);
            }

            Value* final_cons = consFromTagged(cond_sym, result);
            return tagged_.packPtr(
                ctx_.builder().CreateIntToPtr(final_cons, ctx_.builder().getPtrTy()),
                ESHKOL_VALUE_CONS_PTR);
        }

        case ESHKOL_SEQUENCE_OP: {
            // Build (begin expr1 expr2 ...)
            return quoteNaryOp("begin", op->sequence_op.expressions, op->sequence_op.num_expressions);
        }

        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP: {
            // Build (let/let*/letrec ((var1 val1) ...) body)
            const char* let_name = op->op == ESHKOL_LET_OP ? "let" :
                                   op->op == ESHKOL_LET_STAR_OP ? "let*" : "letrec";
            Value* let_sym = tagged_.packPtr(
                string_io_.createString(let_name), ESHKOL_VALUE_STRING_PTR);

            // Build bindings list
            Value* bindings = packNull();
            for (int64_t i = op->let_op.num_bindings - 1; i >= 0; i--) {
                const eshkol_ast_t* binding_cons = &op->let_op.bindings[i];
                Value* binding;
                if (binding_cons->type == ESHKOL_CONS && binding_cons->cons_cell.car) {
                    // Get variable name from car
                    const eshkol_ast_t* var_ast = binding_cons->cons_cell.car;
                    Value* var;
                    if (var_ast->type == ESHKOL_VAR && var_ast->variable.id) {
                        var = tagged_.packPtr(
                            string_io_.createString(var_ast->variable.id),
                            ESHKOL_VALUE_STRING_PTR);
                    } else {
                        var = quoteAST(var_ast);
                    }
                    // Get value from cdr
                    Value* val = quoteAST(binding_cons->cons_cell.cdr);

                    // Build (var val) binding
                    binding = packNull();
                    Value* val_cons = consFromTagged(val, binding);
                    binding = tagged_.packPtr(
                        ctx_.builder().CreateIntToPtr(val_cons, ctx_.builder().getPtrTy()),
                        ESHKOL_VALUE_CONS_PTR);
                    Value* var_cons = consFromTagged(var, binding);
                    binding = tagged_.packPtr(
                        ctx_.builder().CreateIntToPtr(var_cons, ctx_.builder().getPtrTy()),
                        ESHKOL_VALUE_CONS_PTR);
                } else {
                    // Fallback: quote the binding directly
                    binding = quoteAST(binding_cons);
                }

                Value* cons_int = consFromTagged(binding, bindings);
                bindings = tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(cons_int, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);
            }

            // Build body
            Value* body = quoteAST(op->let_op.body);

            // Combine: (let bindings body)
            Value* result = packNull();
            Value* body_cons = consFromTagged(body, result);
            result = tagged_.packPtr(
                ctx_.builder().CreateIntToPtr(body_cons, ctx_.builder().getPtrTy()),
                ESHKOL_VALUE_CONS_PTR);
            Value* bindings_cons = consFromTagged(bindings, result);
            result = tagged_.packPtr(
                ctx_.builder().CreateIntToPtr(bindings_cons, ctx_.builder().getPtrTy()),
                ESHKOL_VALUE_CONS_PTR);
            Value* let_cons = consFromTagged(let_sym, result);
            return tagged_.packPtr(
                ctx_.builder().CreateIntToPtr(let_cons, ctx_.builder().getPtrTy()),
                ESHKOL_VALUE_CONS_PTR);
        }

        case ESHKOL_DEFINE_OP: {
            // Build (define name value) or (define (name params) body)
            Value* define_sym = tagged_.packPtr(
                string_io_.createString("define"), ESHKOL_VALUE_STRING_PTR);
            Value* name = tagged_.packPtr(
                string_io_.createString(op->define_op.name), ESHKOL_VALUE_STRING_PTR);

            if (op->define_op.is_function) {
                // Build (define (name params...) body)
                // First build the name+params list
                Value* name_params = packNull();
                for (int64_t i = op->define_op.num_params - 1; i >= 0; i--) {
                    Value* param = tagged_.packPtr(
                        string_io_.createString(op->define_op.parameters[i].variable.id),
                        ESHKOL_VALUE_STRING_PTR);
                    Value* cons_int = consFromTagged(param, name_params);
                    name_params = tagged_.packPtr(
                        ctx_.builder().CreateIntToPtr(cons_int, ctx_.builder().getPtrTy()),
                        ESHKOL_VALUE_CONS_PTR);
                }
                Value* name_cons = consFromTagged(name, name_params);
                name_params = tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(name_cons, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);

                Value* body = quoteAST(op->define_op.value);

                Value* result = packNull();
                Value* body_cons = consFromTagged(body, result);
                result = tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(body_cons, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);
                Value* np_cons = consFromTagged(name_params, result);
                result = tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(np_cons, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);
                Value* def_cons = consFromTagged(define_sym, result);
                return tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(def_cons, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);
            } else {
                // Build (define name value)
                Value* value = quoteAST(op->define_op.value);
                Value* result = packNull();
                Value* val_cons = consFromTagged(value, result);
                result = tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(val_cons, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);
                Value* name_cons = consFromTagged(name, result);
                result = tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(name_cons, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);
                Value* def_cons = consFromTagged(define_sym, result);
                return tagged_.packPtr(
                    ctx_.builder().CreateIntToPtr(def_cons, ctx_.builder().getPtrTy()),
                    ESHKOL_VALUE_CONS_PTR);
            }
        }

        default:
            eshkol_debug("quoteOperation: unhandled op type %d", op->op);
            return packNull();
    }
}

Value* HomoiconicCodegen::quoteNaryOp(const char* op_name,
                                       const eshkol_ast_t* args,
                                       uint64_t num_args) {
    Value* op_sym = tagged_.packPtr(
        string_io_.createString(op_name), ESHKOL_VALUE_STRING_PTR);
    Value* result = packNull();

    // Build args from right to left
    for (int64_t i = num_args - 1; i >= 0; i--) {
        Value* arg = quoteAST(&args[i]);
        Value* cons_int = consFromTagged(arg, result);
        result = tagged_.packPtr(
            ctx_.builder().CreateIntToPtr(cons_int, ctx_.builder().getPtrTy()),
            ESHKOL_VALUE_CONS_PTR);
    }

    Value* final_cons = consFromTagged(op_sym, result);
    return tagged_.packPtr(
        ctx_.builder().CreateIntToPtr(final_cons, ctx_.builder().getPtrTy()),
        ESHKOL_VALUE_CONS_PTR);
}

Value* HomoiconicCodegen::quoteList(const eshkol_operations_t* op) {
    if (!op || op->op != ESHKOL_CALL_OP) {
        return ConstantInt::get(ctx_.int64Type(), 0);
    }

    // Build list from right to left: (op arg1 arg2 ...)
    // Start with empty list (null)
    Value* result_int = ConstantInt::get(ctx_.int64Type(), 0);

    // Add arguments in reverse
    for (int64_t i = op->call_op.num_vars - 1; i >= 0; i--) {
        Value* elem_tagged = quoteAST(&op->call_op.variables[i]);

        // Convert result_int to tagged value
        Value* result_tagged;
        if (result_int == ConstantInt::get(ctx_.int64Type(), 0)) {
            result_tagged = packNull();
        } else {
            result_tagged = tagged_.packPtr(
                ctx_.builder().CreateIntToPtr(result_int, ctx_.builder().getPtrTy()),
                ESHKOL_VALUE_CONS_PTR);
        }

        // Create cons cell from two tagged values
        Value* cons_cell = consFromTagged(elem_tagged, result_tagged);
        result_int = cons_cell;
    }

    // Add operator symbol at front
    // CRITICAL: Must check func->type is ESHKOL_VAR before accessing variable.id
    if (op->call_op.func && op->call_op.func->type == ESHKOL_VAR && op->call_op.func->variable.id) {
        std::string func_name = op->call_op.func->variable.id;

        // For "list" function (used to represent quoted data), just return the list without the operator
        if (func_name == "list") {
            if (result_int == ConstantInt::get(ctx_.int64Type(), 0)) {
                return ConstantInt::get(ctx_.int64Type(), 0);
            }
            return result_int;
        }

        Value* op_string = string_io_.createString(op->call_op.func->variable.id);
        Value* op_tagged = tagged_.packPtr(op_string, ESHKOL_VALUE_STRING_PTR);

        Value* result_tagged;
        if (result_int == ConstantInt::get(ctx_.int64Type(), 0)) {
            result_tagged = packNull();
        } else {
            result_tagged = tagged_.packPtr(
                ctx_.builder().CreateIntToPtr(result_int, ctx_.builder().getPtrTy()),
                ESHKOL_VALUE_CONS_PTR);
        }

        Value* final_list = consFromTagged(op_tagged, result_tagged);
        // Return int64 pointer, not tagged (for consistency)
        return final_list;
    }

    // Return int64 pointer, not tagged (for consistency)
    return result_int;
}

// === Lambda S-Expression ===

Value* HomoiconicCodegen::buildParameterList(const eshkol_ast_t* params,
                                              uint64_t num_params) {
    if (num_params == 0) {
        return ConstantInt::get(ctx_.int64Type(), 0); // Empty list
    }

    Value* result = ConstantInt::get(ctx_.int64Type(), 0); // Start with null

    // Build list backwards for proper cons chain
    for (int64_t i = num_params - 1; i >= 0; i--) {
        if (params[i].type != ESHKOL_VAR || !params[i].variable.id) continue;

        // Create parameter symbol string
        Value* param_name = string_io_.createString(params[i].variable.id);
        Value* param_tagged = tagged_.packPtr(param_name, ESHKOL_VALUE_STRING_PTR);

        // Get rest of list as tagged value
        Value* rest_tagged;
        if (result == ConstantInt::get(ctx_.int64Type(), 0)) {
            rest_tagged = packNull();
        } else {
            rest_tagged = tagged_.packPtr(
                ctx_.builder().CreateIntToPtr(result, ctx_.builder().getPtrTy()),
                ESHKOL_VALUE_CONS_PTR);
        }

        // Cons parameter onto rest
        result = consFromTagged(param_tagged, rest_tagged);
    }

    return result;
}

Value* HomoiconicCodegen::lambdaToSExpr(const eshkol_operations_t* op) {
    if (!op) {
        eshkol_error("lambdaToSExpr: null operation");
        return ConstantInt::get(ctx_.int64Type(), 0);
    }

    // Extract parameters and body based on operation type
    const eshkol_ast_t* params = nullptr;
    uint64_t num_params = 0;
    const eshkol_ast_t* body = nullptr;

    if (op->op == ESHKOL_LAMBDA_OP) {
        params = op->lambda_op.parameters;
        num_params = op->lambda_op.num_params;
        body = op->lambda_op.body;
    } else if (op->op == ESHKOL_DEFINE_OP && op->define_op.is_function) {
        params = op->define_op.parameters;
        num_params = op->define_op.num_params;
        body = op->define_op.value;
    } else {
        eshkol_error("lambdaToSExpr: not a lambda or function definition");
        return ConstantInt::get(ctx_.int64Type(), 0);
    }

    // Step 1: Build parameter list - (param1 param2 ...)
    Value* param_list = buildParameterList(params, num_params);

    // Step 2: Convert body AST to quoted S-expression
    Value* body_tagged = packNull();
    if (body) {
        body_tagged = quoteAST(body);
        // Ensure it's a tagged value
        if (!body_tagged || body_tagged->getType() != ctx_.taggedValueType()) {
            body_tagged = packNull();
        }
    }

    // Step 3: Build complete structure: (lambda (params) body)

    // 3a: Create "lambda" symbol string
    Value* lambda_symbol = string_io_.createString("lambda");
    Value* lambda_tagged = tagged_.packPtr(lambda_symbol, ESHKOL_VALUE_STRING_PTR);

    // 3b: Pack param_list as tagged value
    Value* param_list_tagged;
    if (param_list == ConstantInt::get(ctx_.int64Type(), 0)) {
        param_list_tagged = packNull();
    } else {
        param_list_tagged = tagged_.packPtr(
            ctx_.builder().CreateIntToPtr(param_list, ctx_.builder().getPtrTy()),
            ESHKOL_VALUE_CONS_PTR);
    }

    // 3c: body_tagged is already a tagged_value from step 2

    // 3d: Build ((params) . (body . null))
    Value* body_null_tagged = packNull();
    Value* body_cons = consFromTagged(body_tagged, body_null_tagged);
    Value* body_cons_tagged = tagged_.packPtr(
        ctx_.builder().CreateIntToPtr(body_cons, ctx_.builder().getPtrTy()),
        ESHKOL_VALUE_CONS_PTR);

    Value* params_body = consFromTagged(param_list_tagged, body_cons_tagged);

    // 3e: Build (lambda . (params body))
    Value* params_body_tagged = tagged_.packPtr(
        ctx_.builder().CreateIntToPtr(params_body, ctx_.builder().getPtrTy()),
        ESHKOL_VALUE_CONS_PTR);

    Value* result = consFromTagged(lambda_tagged, params_body_tagged);

    eshkol_debug("Generated lambda S-expression with %llu parameters",
                (unsigned long long)num_params);

    return result;
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
