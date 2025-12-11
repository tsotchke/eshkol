/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * CallApplyCodegen implementation
 *
 * This module implements Scheme's apply and closure call operations.
 * Key optimizations:
 * - Direct dispatch for known function types (arithmetic, cons, etc.)
 * - Specialized code paths for common argument/capture counts
 * - Efficient list traversal using arena cons cells
 */

#include <eshkol/backend/call_apply_codegen.h>
#include <eshkol/eshkol.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>

using namespace llvm;

namespace eshkol {

CallApplyCodegen::CallApplyCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged,
                                   ArithmeticCodegen& arith)
    : ctx_(ctx)
    , tagged_(tagged)
    , arith_(arith) {
    eshkol_debug("CallApplyCodegen initialized");
}

// === Internal Helpers ===

Value* CallApplyCodegen::extractConsCarAsTaggedValue(Value* cons_ptr) {
    if (!extract_cons_car_callback_) {
        eshkol_error("extractConsCarAsTaggedValue callback not set");
        return tagged_.packNull();
    }
    return extract_cons_car_callback_(cons_ptr, callback_context_);
}

Function* CallApplyCodegen::getTaggedConsGetPtrFunc() {
    if (!get_cons_accessor_callback_) {
        eshkol_error("getTaggedConsGetPtrFunc callback not set");
        return nullptr;
    }
    return get_cons_accessor_callback_(callback_context_);
}

// === Apply Operations ===

Value* CallApplyCodegen::apply(const eshkol_operations_t* op) {
    if (op->call_op.num_vars < 2) {
        eshkol_warn("apply requires at least 2 arguments: function and argument list");
        return nullptr;
    }

    // Get the function argument
    const eshkol_ast_t* func_arg = &op->call_op.variables[0];

    // Get the argument list
    if (!codegen_ast_callback_) {
        eshkol_error("apply: codegen_ast_callback not set");
        return nullptr;
    }
    Value* arg_list = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!arg_list) return nullptr;

    // Safely extract i64 from possibly-tagged value
    Value* list_int = tagged_.safeExtractInt64(arg_list);

    // Check if function is a symbol (built-in or user-defined)
    if (func_arg->type == ESHKOL_VAR) {
        std::string func_name = func_arg->variable.id;

        // Handle variadic built-in arithmetic operations
        if (func_name == "+" || func_name == "-" || func_name == "*" || func_name == "/") {
            return applyArithmetic(func_name, list_int);
        }

        // Handle list operation
        if (func_name == "list") {
            // (apply list '(1 2 3)) returns the list itself
            return arg_list;
        }

        // Handle cons
        if (func_name == "cons") {
            return applyCons(list_int);
        }

        // Try to find function by name in the module
        Function* named_func = ctx_.module().getFunction(func_name);
        if (named_func) {
            return applyUserFunction(named_func, list_int);
        }

        // FIRST-CLASS FUNCTION FIX: Check for function pointer stored with _func suffix
        // MUTABLE CAPTURE FIX: Skip this path if function has capture parameters (ptr types)
        // because applyUserFunction doesn't handle captures - use closure path instead
        if (symbol_table_) {
            auto func_it = symbol_table_->find(func_name + "_func");
            if (func_it != symbol_table_->end()) {
                if (auto* stored_func = dyn_cast<Function>(func_it->second)) {
                    // Check if function has capture parameters (ptr type params after regular params)
                    bool has_captures = false;
                    for (auto& arg : stored_func->args()) {
                        if (arg.getType()->isPointerTy()) {
                            has_captures = true;
                            break;
                        }
                    }
                    if (!has_captures) {
                        return applyUserFunction(stored_func, list_int);
                    }
                    // Has captures - fall through to closure path below
                }
            }
        }

        // Check if the variable is in the symbol table (closure case)
        Value* func_value = nullptr;
        if (symbol_table_) {
            auto sym_it = symbol_table_->find(func_name);
            if (sym_it != symbol_table_->end()) {
                func_value = sym_it->second;
            }
        }
        if (!func_value && global_symbol_table_) {
            auto gsym_it = global_symbol_table_->find(func_name);
            if (gsym_it != global_symbol_table_->end()) {
                func_value = gsym_it->second;
            }
        }

        if (func_value) {
            // Check if it's a Function* directly
            // MUTABLE CAPTURE FIX: Only use applyUserFunction for functions without captures
            if (auto* direct_func = dyn_cast<Function>(func_value)) {
                bool has_captures = false;
                for (auto& arg : direct_func->args()) {
                    if (arg.getType()->isPointerTy()) {
                        has_captures = true;
                        break;
                    }
                }
                if (!has_captures) {
                    return applyUserFunction(direct_func, list_int);
                }
                // Has captures - fall through to closure path
            }

            // Load from storage if needed
            if (isa<GlobalVariable>(func_value)) {
                func_value = ctx_.builder().CreateLoad(
                    cast<GlobalVariable>(func_value)->getValueType(), func_value);
            } else if (isa<AllocaInst>(func_value)) {
                func_value = ctx_.builder().CreateLoad(
                    cast<AllocaInst>(func_value)->getAllocatedType(), func_value);
            }

            // Treat as a closure/tagged value
            return applyClosure(func_value, list_int);
        }

        eshkol_warn("apply: Unknown function: %s", func_name.c_str());
        return tagged_.packNull();
    }

    // Handle lambda expression passed directly
    if (func_arg->type == ESHKOL_OP && func_arg->operation.op == ESHKOL_LAMBDA_OP) {
        Value* lambda_val = codegen_ast_callback_(func_arg, callback_context_);
        if (!lambda_val) {
            eshkol_warn("apply: Could not compile lambda");
            return tagged_.packNull();
        }
        return applyClosure(lambda_val, list_int);
    }

    // Handle any expression that returns a function/closure (e.g., function calls like (factory-fn))
    if (func_arg->type == ESHKOL_OP) {
        Value* func_val = codegen_ast_callback_(func_arg, callback_context_);
        if (!func_val) {
            eshkol_warn("apply: Could not evaluate function expression");
            return tagged_.packNull();
        }
        return applyClosure(func_val, list_int);
    }

    eshkol_warn("apply: First argument must be a function");
    return tagged_.packNull();
}

Value* CallApplyCodegen::applyCons(Value* list_int) {
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    Function* cons_get_ptr = getTaggedConsGetPtrFunc();
    if (!cons_get_ptr) return tagged_.packNull();

    // Create all blocks upfront
    BasicBlock* valid_block = BasicBlock::Create(ctx_.context(), "apply_cons_valid", current_func);
    BasicBlock* error_block1 = BasicBlock::Create(ctx_.context(), "apply_cons_error1", current_func);
    BasicBlock* error_block2 = BasicBlock::Create(ctx_.context(), "apply_cons_error2", current_func);
    BasicBlock* has_second = BasicBlock::Create(ctx_.context(), "apply_cons_has_second", current_func);
    BasicBlock* continue_block = BasicBlock::Create(ctx_.context(), "apply_cons_continue", current_func);

    // Check if list is null
    Value* is_null = ctx_.builder().CreateICmpEQ(list_int, ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(is_null, error_block1, valid_block);

    // Error block 1: list is null
    ctx_.builder().SetInsertPoint(error_block1);
    Value* error_result1 = tagged_.packNull();
    BasicBlock* error_exit1 = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(continue_block);

    // Valid: get first element
    ctx_.builder().SetInsertPoint(valid_block);
    Value* first_cons = ctx_.builder().CreateIntToPtr(list_int, ctx_.ptrType());
    Value* first_elem = extractConsCarAsTaggedValue(first_cons);
    Value* is_cdr = ConstantInt::get(ctx_.int1Type(), 1);
    Value* rest_int = ctx_.builder().CreateCall(cons_get_ptr, {first_cons, is_cdr});

    // Check if rest is null
    Value* rest_null = ctx_.builder().CreateICmpEQ(rest_int, ConstantInt::get(ctx_.int64Type(), 0));
    // Capture block after extractConsCarAsTaggedValue (it may create blocks)
    BasicBlock* valid_exit_block = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateCondBr(rest_null, error_block2, has_second);

    // Error block 2: rest is null
    ctx_.builder().SetInsertPoint(error_block2);
    Value* error_result2 = tagged_.packNull();
    BasicBlock* error_exit2 = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(continue_block);

    // Has second element
    ctx_.builder().SetInsertPoint(has_second);
    Value* second_cons = ctx_.builder().CreateIntToPtr(rest_int, ctx_.ptrType());
    Value* second_elem = extractConsCarAsTaggedValue(second_cons);

    // Create cons cell using callback - returns i64 pointer with header (consolidated format)
    Value* result;
    if (create_cons_callback_) {
        Value* cons_ptr_int = create_cons_callback_(first_elem, second_elem, callback_context_);
        result = tagged_.packHeapPtr(
            ctx_.builder().CreateIntToPtr(cons_ptr_int, ctx_.ptrType()));
    } else {
        eshkol_error("applyCons: create_cons_callback not set");
        result = tagged_.packNull();
    }
    BasicBlock* valid_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(continue_block);

    // Continue: merge all paths
    ctx_.builder().SetInsertPoint(continue_block);
    PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "apply_cons_result");
    phi->addIncoming(error_result1, error_exit1);
    phi->addIncoming(error_result2, error_exit2);
    phi->addIncoming(result, valid_exit);

    return phi;
}

Value* CallApplyCodegen::applyArithmetic(const std::string& op, Value* list_int) {
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    Function* cons_get_ptr = getTaggedConsGetPtrFunc();
    if (!cons_get_ptr) return tagged_.packNull();

    // Determine identity element
    Value* identity;
    if (op == "+" || op == "-") {
        identity = tagged_.packInt64(ConstantInt::get(ctx_.int64Type(), 0), true);
    } else { // "*" or "/"
        identity = tagged_.packInt64(ConstantInt::get(ctx_.int64Type(), 1), true);
    }

    // Check if list is empty
    Value* is_empty = ctx_.builder().CreateICmpEQ(list_int, ConstantInt::get(ctx_.int64Type(), 0));

    BasicBlock* empty_block = BasicBlock::Create(ctx_.context(), "apply_arith_empty", current_func);
    BasicBlock* non_empty = BasicBlock::Create(ctx_.context(), "apply_arith_non_empty", current_func);
    BasicBlock* loop_cond = BasicBlock::Create(ctx_.context(), "apply_arith_cond", current_func);
    BasicBlock* loop_body = BasicBlock::Create(ctx_.context(), "apply_arith_body", current_func);
    BasicBlock* done_block = BasicBlock::Create(ctx_.context(), "apply_arith_done", current_func);
    BasicBlock* loop_exit = BasicBlock::Create(ctx_.context(), "apply_arith_exit", current_func);

    ctx_.builder().CreateCondBr(is_empty, empty_block, non_empty);

    // Empty list: return identity
    ctx_.builder().SetInsertPoint(empty_block);
    ctx_.builder().CreateBr(loop_exit);

    // Non-empty list: get first element as initial accumulator
    ctx_.builder().SetInsertPoint(non_empty);
    Value* first_cons = ctx_.builder().CreateIntToPtr(list_int, ctx_.ptrType());
    Value* first_elem = extractConsCarAsTaggedValue(first_cons);
    Value* is_cdr = ConstantInt::get(ctx_.int1Type(), 1);
    Value* rest_list = ctx_.builder().CreateCall(cons_get_ptr, {first_cons, is_cdr});

    // Allocate accumulator and current pointer
    Value* accum_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "apply_accum");
    ctx_.builder().CreateStore(first_elem, accum_ptr);
    Value* current_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "apply_current");
    ctx_.builder().CreateStore(rest_list, current_ptr);

    ctx_.builder().CreateBr(loop_cond);

    // Loop condition: check if current != null
    ctx_.builder().SetInsertPoint(loop_cond);
    Value* current_val = ctx_.builder().CreateLoad(ctx_.int64Type(), current_ptr);
    Value* is_not_null = ctx_.builder().CreateICmpNE(current_val, ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(is_not_null, loop_body, done_block);

    // Loop body: apply operation
    ctx_.builder().SetInsertPoint(loop_body);
    Value* cons_ptr = ctx_.builder().CreateIntToPtr(current_val, ctx_.ptrType());
    Value* elem = extractConsCarAsTaggedValue(cons_ptr);
    Value* accum = ctx_.builder().CreateLoad(ctx_.taggedValueType(), accum_ptr);

    // Apply the operation using ArithmeticCodegen
    Value* new_accum;
    if (op == "+") {
        new_accum = arith_.add(accum, elem);
    } else if (op == "-") {
        new_accum = arith_.sub(accum, elem);
    } else if (op == "*") {
        new_accum = arith_.mul(accum, elem);
    } else { // "/"
        new_accum = arith_.div(accum, elem);
    }

    ctx_.builder().CreateStore(new_accum, accum_ptr);

    // Move to next element
    Value* next_val = ctx_.builder().CreateCall(cons_get_ptr, {cons_ptr, is_cdr});
    ctx_.builder().CreateStore(next_val, current_ptr);
    ctx_.builder().CreateBr(loop_cond);

    // Done block: load final accumulator value
    ctx_.builder().SetInsertPoint(done_block);
    Value* final_accum = ctx_.builder().CreateLoad(ctx_.taggedValueType(), accum_ptr);
    ctx_.builder().CreateBr(loop_exit);

    // Loop exit: return result
    ctx_.builder().SetInsertPoint(loop_exit);
    PHINode* result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "apply_result");
    result->addIncoming(identity, empty_block);
    result->addIncoming(final_accum, done_block);

    return result;
}

Value* CallApplyCodegen::applyUserFunction(Function* func, Value* list_int) {
    std::string func_name = func->getName().str();

    // VARIADIC ARITHMETIC FIX: Check if this is a builtin arithmetic function wrapper
    // These are created with names like "builtin_+_2arg" when (define my-add +) is used
    // We need to use applyArithmetic to handle all arguments, not just the fixed arity
    if (func_name.find("builtin_") == 0) {
        if (func_name.find("_+_") != std::string::npos) {
            return applyArithmetic("+", list_int);
        } else if (func_name.find("_-_") != std::string::npos) {
            return applyArithmetic("-", list_int);
        } else if (func_name.find("_*_") != std::string::npos) {
            return applyArithmetic("*", list_int);
        } else if (func_name.find("_/_") != std::string::npos) {
            return applyArithmetic("/", list_int);
        }
    }

    Function* cons_get_ptr = getTaggedConsGetPtrFunc();
    if (!cons_get_ptr) return tagged_.packNull();

    // Check if this is a variadic function
    bool is_variadic = false;
    uint64_t fixed_params = func->arg_size();

    if (variadic_function_info_) {
        auto variadic_it = variadic_function_info_->find(func_name);
        if (variadic_it != variadic_function_info_->end()) {
            is_variadic = variadic_it->second.second;
            if (is_variadic) {
                fixed_params = variadic_it->second.first;
            }
        }
    }

    size_t num_params = func->arg_size();

    // If function expects 0 arguments, just call it
    if (num_params == 0) {
        return ctx_.builder().CreateCall(func, {});
    }

    // Extract arguments from list
    std::vector<Value*> args;
    Value* current = list_int;

    size_t extract_count = is_variadic ? fixed_params : num_params;

    for (size_t i = 0; i < extract_count; i++) {
        Value* is_null = ctx_.builder().CreateICmpEQ(current, ConstantInt::get(ctx_.int64Type(), 0));

        Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        BasicBlock* has_elem = BasicBlock::Create(ctx_.context(), "apply_has_elem", current_func);
        BasicBlock* no_elem = BasicBlock::Create(ctx_.context(), "apply_no_elem", current_func);
        BasicBlock* continue_block = BasicBlock::Create(ctx_.context(), "apply_continue", current_func);

        ctx_.builder().CreateCondBr(is_null, no_elem, has_elem);

        // No element: use null as default
        ctx_.builder().SetInsertPoint(no_elem);
        Value* default_val = tagged_.packNull();
        ctx_.builder().CreateBr(continue_block);

        // Has element: extract it
        ctx_.builder().SetInsertPoint(has_elem);
        Value* cons_ptr = ctx_.builder().CreateIntToPtr(current, ctx_.ptrType());
        Value* elem = extractConsCarAsTaggedValue(cons_ptr);
        Value* is_cdr = ConstantInt::get(ctx_.int1Type(), 1);
        Value* next = ctx_.builder().CreateCall(cons_get_ptr, {cons_ptr, is_cdr});
        BasicBlock* has_elem_exit = ctx_.builder().GetInsertBlock();
        ctx_.builder().CreateBr(continue_block);

        // Continue: use PHI to merge
        ctx_.builder().SetInsertPoint(continue_block);
        PHINode* arg_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "apply_arg");
        arg_phi->addIncoming(default_val, no_elem);
        arg_phi->addIncoming(elem, has_elem_exit);

        PHINode* next_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "apply_next");
        next_phi->addIncoming(ConstantInt::get(ctx_.int64Type(), 0), no_elem);
        next_phi->addIncoming(next, has_elem_exit);

        args.push_back(arg_phi);
        current = next_phi;
    }

    // For variadic functions, pass remaining list as rest parameter
    if (is_variadic) {
        // Pack as HEAP_PTR tagged value (or NULL if empty) - consolidated format
        Value* rest_tagged = tagged_.packHeapPtr(current);
        args.push_back(rest_tagged);
    }

    return ctx_.builder().CreateCall(func, args);
}

Value* CallApplyCodegen::applyClosure(Value* func_value, Value* list_int) {
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    Function* cons_get_ptr = getTaggedConsGetPtrFunc();
    if (!cons_get_ptr) return tagged_.packNull();

    // Allocate arrays in function entry for proper SSA
    IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
    BasicBlock& entry = current_func->getEntryBlock();
    ctx_.builder().SetInsertPoint(&entry, entry.begin());

    ArrayType* args_array_type = ArrayType::get(ctx_.taggedValueType(), MAX_APPLY_ARGS);
    Value* args_array = ctx_.builder().CreateAlloca(args_array_type, nullptr, "apply_args");
    ctx_.builder().restoreIP(saved_ip);

    // Extract arguments from list into array and count them
    BasicBlock* extract_loop = BasicBlock::Create(ctx_.context(), "apply_extract_loop", current_func);
    BasicBlock* extract_done = BasicBlock::Create(ctx_.context(), "apply_extract_done", current_func);

    BasicBlock* pre_loop_block = ctx_.builder().GetInsertBlock();

    Value* init_count = ConstantInt::get(ctx_.int64Type(), 0);
    ctx_.builder().CreateBr(extract_loop);

    ctx_.builder().SetInsertPoint(extract_loop);
    PHINode* count_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "arg_count");
    PHINode* list_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "list_ptr");
    count_phi->addIncoming(init_count, pre_loop_block);
    list_phi->addIncoming(list_int, pre_loop_block);

    // Check if list is null or max args reached
    Value* list_null = ctx_.builder().CreateICmpEQ(list_phi, ConstantInt::get(ctx_.int64Type(), 0));
    Value* max_reached = ctx_.builder().CreateICmpUGE(count_phi,
        ConstantInt::get(ctx_.int64Type(), MAX_APPLY_ARGS));
    Value* done_cond = ctx_.builder().CreateOr(list_null, max_reached);

    BasicBlock* extract_body = BasicBlock::Create(ctx_.context(), "apply_extract_body", current_func);
    ctx_.builder().CreateCondBr(done_cond, extract_done, extract_body);

    ctx_.builder().SetInsertPoint(extract_body);
    Value* cons_ptr = ctx_.builder().CreateIntToPtr(list_phi, ctx_.ptrType());
    Value* elem = extractConsCarAsTaggedValue(cons_ptr);

    // Store in array
    Value* idx_gep = ctx_.builder().CreateGEP(args_array_type, args_array,
        {ConstantInt::get(ctx_.int64Type(), 0), count_phi});
    ctx_.builder().CreateStore(elem, idx_gep);

    // Get next list element
    Value* is_cdr = ConstantInt::get(ctx_.int1Type(), 1);
    Value* next_list = ctx_.builder().CreateCall(cons_get_ptr, {cons_ptr, is_cdr});
    Value* next_count = ctx_.builder().CreateAdd(count_phi, ConstantInt::get(ctx_.int64Type(), 1));

    BasicBlock* loop_back_block = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(extract_loop);

    count_phi->addIncoming(next_count, loop_back_block);
    list_phi->addIncoming(next_list, loop_back_block);

    // Done extracting - now call closure
    ctx_.builder().SetInsertPoint(extract_done);
    Value* final_count = count_phi;

    // Get closure info
    // Use getBaseType() to properly handle legacy types (CLOSURE_PTR=38, etc.)
    // DO NOT use 0x0F mask - 38 & 0x0F = 6 which is WRONG!
    // M1 CONSOLIDATION COMPLETE: All callable objects use CALLABLE type (9)
    // Legacy CLOSURE_PTR (38) is no longer used
    Value* type_tag = tagged_.getType(func_value);
    Value* base_type = tagged_.getBaseType(type_tag);
    Value* is_closure = ctx_.builder().CreateICmpEQ(base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));

    BasicBlock* closure_path = BasicBlock::Create(ctx_.context(), "apply_closure", current_func);
    BasicBlock* direct_path = BasicBlock::Create(ctx_.context(), "apply_direct", current_func);
    BasicBlock* merge_bb = BasicBlock::Create(ctx_.context(), "apply_merge", current_func);

    // Declare closure_results early so we can add arithmetic results
    std::vector<std::pair<BasicBlock*, Value*>> closure_results;

    ctx_.builder().CreateCondBr(is_closure, closure_path, direct_path);

    // CLOSURE PATH
    ctx_.builder().SetInsertPoint(closure_path);
    Value* closure_ptr_i64 = tagged_.unpackInt64(func_value);
    Value* closure_ptr = ctx_.builder().CreateIntToPtr(closure_ptr_i64, ctx_.ptrType());

    // Load func_ptr from closure
    Value* func_ptr_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(), closure_ptr);
    Value* actual_func_ptr = ctx_.builder().CreateIntToPtr(func_ptr_i64, ctx_.ptrType());

    // VARIADIC ARITHMETIC FIX: Check if closure wraps a builtin arithmetic function
    // If so, use applyArithmetic to handle all arguments correctly
    if (get_builtin_arithmetic_callback_) {
        Function* builtin_add = get_builtin_arithmetic_callback_("+", callback_context_);
        Function* builtin_sub = get_builtin_arithmetic_callback_("-", callback_context_);
        Function* builtin_mul = get_builtin_arithmetic_callback_("*", callback_context_);
        Function* builtin_div = get_builtin_arithmetic_callback_("/", callback_context_);

        BasicBlock* closure_add_block = BasicBlock::Create(ctx_.context(), "closure_builtin_add", current_func);
        BasicBlock* closure_sub_check = BasicBlock::Create(ctx_.context(), "closure_sub_check", current_func);
        BasicBlock* closure_sub_block = BasicBlock::Create(ctx_.context(), "closure_builtin_sub", current_func);
        BasicBlock* closure_mul_check = BasicBlock::Create(ctx_.context(), "closure_mul_check", current_func);
        BasicBlock* closure_mul_block = BasicBlock::Create(ctx_.context(), "closure_builtin_mul", current_func);
        BasicBlock* closure_div_check = BasicBlock::Create(ctx_.context(), "closure_div_check", current_func);
        BasicBlock* closure_div_block = BasicBlock::Create(ctx_.context(), "closure_builtin_div", current_func);
        BasicBlock* closure_regular = BasicBlock::Create(ctx_.context(), "closure_regular", current_func);

        // Check for builtin +
        Value* is_closure_add = ctx_.builder().CreateICmpEQ(actual_func_ptr, builtin_add);
        ctx_.builder().CreateCondBr(is_closure_add, closure_add_block, closure_sub_check);

        // Add block
        ctx_.builder().SetInsertPoint(closure_add_block);
        Value* closure_add_result = applyArithmetic("+", list_int);
        ctx_.builder().CreateBr(merge_bb);
        closure_results.push_back({ctx_.builder().GetInsertBlock(), closure_add_result});

        // Check for builtin -
        ctx_.builder().SetInsertPoint(closure_sub_check);
        Value* is_closure_sub = ctx_.builder().CreateICmpEQ(actual_func_ptr, builtin_sub);
        ctx_.builder().CreateCondBr(is_closure_sub, closure_sub_block, closure_mul_check);

        ctx_.builder().SetInsertPoint(closure_sub_block);
        Value* closure_sub_result = applyArithmetic("-", list_int);
        ctx_.builder().CreateBr(merge_bb);
        closure_results.push_back({ctx_.builder().GetInsertBlock(), closure_sub_result});

        // Check for builtin *
        ctx_.builder().SetInsertPoint(closure_mul_check);
        Value* is_closure_mul = ctx_.builder().CreateICmpEQ(actual_func_ptr, builtin_mul);
        ctx_.builder().CreateCondBr(is_closure_mul, closure_mul_block, closure_div_check);

        ctx_.builder().SetInsertPoint(closure_mul_block);
        Value* closure_mul_result = applyArithmetic("*", list_int);
        ctx_.builder().CreateBr(merge_bb);
        closure_results.push_back({ctx_.builder().GetInsertBlock(), closure_mul_result});

        // Check for builtin /
        ctx_.builder().SetInsertPoint(closure_div_check);
        Value* is_closure_div = ctx_.builder().CreateICmpEQ(actual_func_ptr, builtin_div);
        ctx_.builder().CreateCondBr(is_closure_div, closure_div_block, closure_regular);

        ctx_.builder().SetInsertPoint(closure_div_block);
        Value* closure_div_result = applyArithmetic("/", list_int);
        ctx_.builder().CreateBr(merge_bb);
        closure_results.push_back({ctx_.builder().GetInsertBlock(), closure_div_result});

        // Continue with regular closure handling
        ctx_.builder().SetInsertPoint(closure_regular);
    }

    // Load env and num_captures
    Value* env_ptr_addr = ctx_.builder().CreateGEP(ctx_.int8Type(), closure_ptr,
        ConstantInt::get(ctx_.int64Type(), 8));
    Value* env_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), env_ptr_addr);

    // Save current block for PHI node - this may be closure_path or closure_regular
    BasicBlock* pre_env_check_bb = ctx_.builder().GetInsertBlock();

    Value* env_null = ctx_.builder().CreateICmpEQ(env_ptr,
        ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())));
    BasicBlock* env_valid_bb = BasicBlock::Create(ctx_.context(), "apply_env_valid", current_func);
    BasicBlock* env_checked_bb = BasicBlock::Create(ctx_.context(), "apply_env_checked", current_func);

    ctx_.builder().CreateCondBr(env_null, env_checked_bb, env_valid_bb);

    ctx_.builder().SetInsertPoint(env_valid_bb);
    Value* loaded_captures = ctx_.builder().CreateLoad(ctx_.int64Type(), env_ptr);
    ctx_.builder().CreateBr(env_checked_bb);

    ctx_.builder().SetInsertPoint(env_checked_bb);
    PHINode* num_captures_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "num_captures");
    num_captures_phi->addIncoming(ConstantInt::get(ctx_.int64Type(), 0), pre_env_check_bb);
    num_captures_phi->addIncoming(loaded_captures, env_valid_bb);

    // Clamp captures to max
    Value* clamped_captures = ctx_.builder().CreateSelect(
        ctx_.builder().CreateICmpUGT(num_captures_phi,
            ConstantInt::get(ctx_.int64Type(), MAX_APPLY_CAPTURES)),
        ConstantInt::get(ctx_.int64Type(), MAX_APPLY_CAPTURES),
        num_captures_phi);

    // Load captures base pointer
    Value* captures_base = ctx_.builder().CreateGEP(ctx_.int8Type(), env_ptr,
        ConstantInt::get(ctx_.int64Type(), 8));
    Value* captures_typed = ctx_.builder().CreateBitCast(captures_base, ctx_.ptrType());

    // Dispatch based on (arg_count * (MAX_CAPTURES+1) + num_captures)
    Value* dispatch_idx = ctx_.builder().CreateAdd(
        ctx_.builder().CreateMul(final_count,
            ConstantInt::get(ctx_.int64Type(), MAX_APPLY_CAPTURES + 1)),
        clamped_captures);

    BasicBlock* dispatch_default = BasicBlock::Create(ctx_.context(), "apply_dispatch_default", current_func);
    SwitchInst* sw = ctx_.builder().CreateSwitch(dispatch_idx, dispatch_default,
        (MAX_APPLY_ARGS + 1) * (MAX_APPLY_CAPTURES + 1));

    // Note: closure_results was declared earlier (before arithmetic handling)

    // Generate cases for each (arg_count, capture_count) pair
    for (int ac = 0; ac <= MAX_APPLY_ARGS; ac++) {
        for (int cc = 0; cc <= MAX_APPLY_CAPTURES; cc++) {
            int case_idx = ac * (MAX_APPLY_CAPTURES + 1) + cc;
            BasicBlock* case_bb = BasicBlock::Create(ctx_.context(),
                "apply_" + std::to_string(ac) + "_" + std::to_string(cc), current_func);
            sw->addCase(ConstantInt::get(ctx_.int64Type(), case_idx), case_bb);

            ctx_.builder().SetInsertPoint(case_bb);

            // Build args: first from array, then captures
            std::vector<Value*> call_args;
            for (int i = 0; i < ac; i++) {
                Value* arg_ptr = ctx_.builder().CreateGEP(args_array_type, args_array,
                    {ConstantInt::get(ctx_.int64Type(), 0),
                     ConstantInt::get(ctx_.int64Type(), i)});
                call_args.push_back(ctx_.builder().CreateLoad(ctx_.taggedValueType(), arg_ptr));
            }
            // MUTABLE CAPTURE FIX: Pass pointers to captures, not loaded values
            // This matches the updated lambda signature that expects ptr types for captures
            for (int i = 0; i < cc; i++) {
                Value* cap_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), captures_typed,
                    ConstantInt::get(ctx_.int64Type(), i));
                call_args.push_back(cap_ptr);  // Pass pointer, not loaded value
            }

            // Create function type and call
            // MUTABLE CAPTURE FIX: Capture params use pointer type, not tagged_value
            std::vector<Type*> param_types;
            for (int i = 0; i < ac; i++) {
                param_types.push_back(ctx_.taggedValueType());
            }
            for (int i = 0; i < cc; i++) {
                param_types.push_back(ctx_.ptrType());  // Pointer type for captures
            }
            FunctionType* func_type = FunctionType::get(ctx_.taggedValueType(), param_types, false);
            Value* result = ctx_.builder().CreateCall(func_type, actual_func_ptr, call_args);
            ctx_.builder().CreateBr(merge_bb);
            closure_results.push_back({ctx_.builder().GetInsertBlock(), result});
        }
    }

    ctx_.builder().SetInsertPoint(dispatch_default);
    Value* default_closure_result = tagged_.packNull();
    ctx_.builder().CreateBr(merge_bb);
    closure_results.push_back({dispatch_default, default_closure_result});

    // DIRECT PATH (lambda without captures)
    ctx_.builder().SetInsertPoint(direct_path);
    Value* direct_func_ptr_i64 = tagged_.unpackInt64(func_value);
    Value* direct_func_ptr = ctx_.builder().CreateIntToPtr(direct_func_ptr_i64, ctx_.ptrType());

    std::vector<std::pair<BasicBlock*, Value*>> direct_results;

    // VARIADIC ARITHMETIC FIX: Check if function is a builtin arithmetic function
    // If so, use applyArithmetic which properly handles variadic arguments
    if (get_builtin_arithmetic_callback_) {
        // Get the builtin arithmetic functions
        Function* builtin_add = get_builtin_arithmetic_callback_("+", callback_context_);
        Function* builtin_sub = get_builtin_arithmetic_callback_("-", callback_context_);
        Function* builtin_mul = get_builtin_arithmetic_callback_("*", callback_context_);
        Function* builtin_div = get_builtin_arithmetic_callback_("/", callback_context_);

        // Create blocks for arithmetic dispatch
        BasicBlock* add_block = BasicBlock::Create(ctx_.context(), "apply_builtin_add", current_func);
        BasicBlock* sub_block = BasicBlock::Create(ctx_.context(), "apply_builtin_sub", current_func);
        BasicBlock* mul_block = BasicBlock::Create(ctx_.context(), "apply_builtin_mul", current_func);
        BasicBlock* div_block = BasicBlock::Create(ctx_.context(), "apply_builtin_div", current_func);
        BasicBlock* regular_dispatch = BasicBlock::Create(ctx_.context(), "apply_regular_dispatch", current_func);

        // Check if function matches builtin +
        Value* is_add = ctx_.builder().CreateICmpEQ(direct_func_ptr, builtin_add);
        ctx_.builder().CreateCondBr(is_add, add_block, sub_block);

        // Add block - call applyArithmetic for +
        ctx_.builder().SetInsertPoint(add_block);
        Value* add_result = applyArithmetic("+", list_int);
        ctx_.builder().CreateBr(merge_bb);
        direct_results.push_back({ctx_.builder().GetInsertBlock(), add_result});

        // Sub block - check for -
        ctx_.builder().SetInsertPoint(sub_block);
        Value* is_sub = ctx_.builder().CreateICmpEQ(direct_func_ptr, builtin_sub);
        BasicBlock* sub_do = BasicBlock::Create(ctx_.context(), "apply_do_sub", current_func);
        ctx_.builder().CreateCondBr(is_sub, sub_do, mul_block);

        ctx_.builder().SetInsertPoint(sub_do);
        Value* sub_result = applyArithmetic("-", list_int);
        ctx_.builder().CreateBr(merge_bb);
        direct_results.push_back({ctx_.builder().GetInsertBlock(), sub_result});

        // Mul block - check for *
        ctx_.builder().SetInsertPoint(mul_block);
        Value* is_mul = ctx_.builder().CreateICmpEQ(direct_func_ptr, builtin_mul);
        BasicBlock* mul_do = BasicBlock::Create(ctx_.context(), "apply_do_mul", current_func);
        ctx_.builder().CreateCondBr(is_mul, mul_do, div_block);

        ctx_.builder().SetInsertPoint(mul_do);
        Value* mul_result = applyArithmetic("*", list_int);
        ctx_.builder().CreateBr(merge_bb);
        direct_results.push_back({ctx_.builder().GetInsertBlock(), mul_result});

        // Div block - check for /
        ctx_.builder().SetInsertPoint(div_block);
        Value* is_div = ctx_.builder().CreateICmpEQ(direct_func_ptr, builtin_div);
        BasicBlock* div_do = BasicBlock::Create(ctx_.context(), "apply_do_div", current_func);
        ctx_.builder().CreateCondBr(is_div, div_do, regular_dispatch);

        ctx_.builder().SetInsertPoint(div_do);
        Value* div_result = applyArithmetic("/", list_int);
        ctx_.builder().CreateBr(merge_bb);
        direct_results.push_back({ctx_.builder().GetInsertBlock(), div_result});

        // Continue with regular dispatch
        ctx_.builder().SetInsertPoint(regular_dispatch);
    }

    BasicBlock* direct_dispatch_default = BasicBlock::Create(ctx_.context(),
        "apply_direct_default", current_func);
    SwitchInst* direct_sw = ctx_.builder().CreateSwitch(final_count, direct_dispatch_default,
        MAX_APPLY_ARGS + 1);

    for (int ac = 0; ac <= MAX_APPLY_ARGS; ac++) {
        BasicBlock* case_bb = BasicBlock::Create(ctx_.context(),
            "apply_direct_" + std::to_string(ac), current_func);
        direct_sw->addCase(ConstantInt::get(ctx_.int64Type(), ac), case_bb);

        ctx_.builder().SetInsertPoint(case_bb);

        std::vector<Value*> call_args;
        for (int i = 0; i < ac; i++) {
            Value* arg_ptr = ctx_.builder().CreateGEP(args_array_type, args_array,
                {ConstantInt::get(ctx_.int64Type(), 0),
                 ConstantInt::get(ctx_.int64Type(), i)});
            call_args.push_back(ctx_.builder().CreateLoad(ctx_.taggedValueType(), arg_ptr));
        }

        std::vector<Type*> param_types(ac, ctx_.taggedValueType());
        FunctionType* func_type = FunctionType::get(ctx_.taggedValueType(), param_types, false);
        Value* result = ctx_.builder().CreateCall(func_type, direct_func_ptr, call_args);
        ctx_.builder().CreateBr(merge_bb);
        direct_results.push_back({ctx_.builder().GetInsertBlock(), result});
    }

    ctx_.builder().SetInsertPoint(direct_dispatch_default);
    Value* default_direct_result = tagged_.packNull();
    ctx_.builder().CreateBr(merge_bb);
    direct_results.push_back({direct_dispatch_default, default_direct_result});

    // Merge all results
    ctx_.builder().SetInsertPoint(merge_bb);
    PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(),
        closure_results.size() + direct_results.size(), "apply_result");
    for (auto& [bb, val] : closure_results) {
        result_phi->addIncoming(val, bb);
    }
    for (auto& [bb, val] : direct_results) {
        result_phi->addIncoming(val, bb);
    }

    return result_phi;
}

Value* CallApplyCodegen::closureCall(Value* closure, const std::vector<Value*>& args) {
    // TODO: Implement direct closure call with pre-evaluated arguments
    // For now, this is a placeholder - the main codegen uses codegenClosureCall
    eshkol_warn("closureCall not yet implemented in CallApplyCodegen");
    return tagged_.packNull();
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
