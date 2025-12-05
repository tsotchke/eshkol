/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * MapCodegen implementation
 */

#include <eshkol/backend/map_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/BasicBlock.h>

using namespace llvm;

namespace eshkol {

MapCodegen::MapCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged)
    : ctx_(ctx), tagged_(tagged) {}

Function* MapCodegen::getConsGetPtrFunc() {
    if (get_cons_get_ptr_callback_) {
        return get_cons_get_ptr_callback_(callback_context_);
    }
    return nullptr;
}

Function* MapCodegen::getConsSetPtrFunc() {
    if (get_cons_set_ptr_callback_) {
        return get_cons_set_ptr_callback_(callback_context_);
    }
    return nullptr;
}

Value* MapCodegen::map(const eshkol_operations_t* op) {
    if (op->call_op.num_vars < 2) {
        eshkol_warn("map requires at least 2 arguments: procedure and list");
        return nullptr;
    }

    // Add function context isolation
    if (push_function_context_) {
        push_function_context_(callback_context_);
    }

    // Calculate required arity for builtin functions (number of lists)
    size_t num_lists = op->call_op.num_vars - 1;

    // Debug output
    printf("[DEBUG] codegenMap: proc arg type=%d, num_lists=%zu\n",
           (int)op->call_op.variables[0].type, num_lists);
    fflush(stdout);
    if (op->call_op.variables[0].type == ESHKOL_VAR) {
        printf("[DEBUG] codegenMap: proc is VAR, id=%s\n", op->call_op.variables[0].variable.id);
        fflush(stdout);
    } else if (op->call_op.variables[0].type == ESHKOL_OP) {
        printf("[DEBUG] codegenMap: proc is OP, op=%d\n", (int)op->call_op.variables[0].operation.op);
        fflush(stdout);
    }

    // CLOSURE FIX: Check if the variable contains a closure (CLOSURE_PTR)
    // If so, we need to use mapWithClosure to properly extract captured values
    if (op->call_op.variables[0].type == ESHKOL_VAR && global_symbol_table_ && nested_function_captures_) {
        std::string var_name = op->call_op.variables[0].variable.id;

        // Check if this variable is known to hold a closure at compile time
        std::string func_key = var_name + "_func";
        auto func_it = global_symbol_table_->find(func_key);
        if (func_it != global_symbol_table_->end() && isa<Function>(func_it->second)) {
            Function* lambda_func = cast<Function>(func_it->second);
            std::string lambda_name = lambda_func->getName().str();
            auto captures_it = nested_function_captures_->find(lambda_name);
            if (captures_it != nested_function_captures_->end() && !captures_it->second.empty()) {
                eshkol_debug("Map: variable '%s' holds closure '%s' with %zu captures, using closure dispatch",
                            var_name.c_str(), lambda_name.c_str(), captures_it->second.size());

                // Load the closure value from the variable
                Value* var_value = nullptr;
                auto git = global_symbol_table_->find(var_name);
                if (git != global_symbol_table_->end()) {
                    var_value = git->second;
                } else if (symbol_table_) {
                    auto lit = symbol_table_->find(var_name);
                    if (lit != symbol_table_->end()) {
                        var_value = lit->second;
                    }
                }

                if (var_value) {
                    Value* loaded_val = nullptr;
                    if (isa<GlobalVariable>(var_value)) {
                        GlobalVariable* gv = cast<GlobalVariable>(var_value);
                        if (gv->getValueType() == ctx_.taggedValueType()) {
                            loaded_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                        }
                    } else if (isa<AllocaInst>(var_value)) {
                        AllocaInst* ai = cast<AllocaInst>(var_value);
                        if (ai->getAllocatedType() == ctx_.taggedValueType()) {
                            loaded_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                        }
                    }

                    if (loaded_val && codegen_ast_callback_) {
                        Value* list = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
                        if (!list) {
                            if (pop_function_context_) pop_function_context_(callback_context_);
                            return nullptr;
                        }
                        Value* result = mapWithClosure(loaded_val, list);
                        if (pop_function_context_) pop_function_context_(callback_context_);
                        return result;
                    }
                }
            }
        }
    }

    // Resolve the procedure function
    Value* proc = nullptr;
    Function* proc_func = nullptr;

    if (resolve_lambda_callback_) {
        proc = resolve_lambda_callback_(&op->call_op.variables[0], num_lists, callback_context_);
    }

    if (!proc) {
        // Try to get procedure as a function parameter (higher-order function support)
        if (op->call_op.variables[0].type == ESHKOL_VAR && current_function_ && *current_function_) {
            std::string func_name = op->call_op.variables[0].variable.id;
            for (auto& arg : (*current_function_)->args()) {
                if (arg.getName() == func_name) {
                    if (indirect_call_callback_) {
                        proc = indirect_call_callback_(&arg, num_lists, callback_context_);
                        if (proc) {
                            proc_func = dyn_cast<Function>(proc);
                        }
                    }
                    break;
                }
            }
        }

        // Handle call expressions that return closures (e.g., (compose f g))
        if (!proc && op->call_op.variables[0].type == ESHKOL_OP &&
            op->call_op.variables[0].operation.op == ESHKOL_CALL_OP) {
            eshkol_debug("Map: evaluating closure expression as procedure");
            if (codegen_ast_callback_) {
                Value* closure_val = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
                if (closure_val) {
                    Value* list = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
                    if (!list) {
                        if (pop_function_context_) pop_function_context_(callback_context_);
                        return nullptr;
                    }
                    Value* result = mapWithClosure(closure_val, list);
                    if (pop_function_context_) pop_function_context_(callback_context_);
                    return result;
                }
            }
        }

        // RUNTIME CLOSURE FIX: If we couldn't resolve at compile-time,
        // try evaluating the variable - it might be a runtime closure
        // (e.g., add10 = (make-adder 10) returns a closure at runtime)
        if (!proc && op->call_op.variables[0].type == ESHKOL_VAR && codegen_ast_callback_) {
            eshkol_debug("Map: trying runtime closure evaluation for variable '%s'",
                        op->call_op.variables[0].variable.id);
            Value* var_val = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
            if (var_val) {
                Value* list = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
                if (!list) {
                    if (pop_function_context_) pop_function_context_(callback_context_);
                    return nullptr;
                }
                Value* result = mapWithClosure(var_val, list);
                if (pop_function_context_) pop_function_context_(callback_context_);
                return result;
            }
        }

        if (!proc) {
            eshkol_error("Failed to resolve procedure for map");
            if (pop_function_context_) pop_function_context_(callback_context_);
            return nullptr;
        }
    } else {
        proc_func = dyn_cast<Function>(proc);
    }

    if (!proc_func) {
        eshkol_error("map procedure must be a function");
        if (pop_function_context_) pop_function_context_(callback_context_);
        return nullptr;
    }

    // Single-list map: (map proc list)
    if (op->call_op.num_vars == 2) {
        if (!codegen_ast_callback_) {
            if (pop_function_context_) pop_function_context_(callback_context_);
            return nullptr;
        }
        Value* list = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
        if (!list) {
            if (pop_function_context_) pop_function_context_(callback_context_);
            return nullptr;
        }
        Value* result = mapSingleList(proc_func, list);
        if (pop_function_context_) pop_function_context_(callback_context_);
        return result;
    }

    // Multi-list map: (map proc list1 list2 ...)
    std::vector<Value*> lists;
    for (uint64_t i = 1; i < op->call_op.num_vars; i++) {
        if (codegen_ast_callback_) {
            Value* list = codegen_ast_callback_(&op->call_op.variables[i], callback_context_);
            if (list) {
                lists.push_back(list);
            }
        }
    }

    if (lists.empty()) {
        eshkol_error("map requires at least one list argument");
        if (pop_function_context_) pop_function_context_(callback_context_);
        return nullptr;
    }

    Value* result = mapMultiList(proc_func, lists);
    if (pop_function_context_) pop_function_context_(callback_context_);
    return result;
}

Value* MapCodegen::mapWithClosure(Value* closure_val, Value* list) {
    if (!closure_val || !list) return nullptr;

    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    Function* cons_get_ptr = getConsGetPtrFunc();
    if (!cons_get_ptr) return tagged_.packNull();

    eshkol_debug("Map with closure dispatch starting");

    // Extract list pointer
    Value* list_ptr;
    if (list->getType() == ctx_.taggedValueType()) {
        list_ptr = tagged_.unpackInt64(list);
    } else if (list->getType()->isIntegerTy(64)) {
        list_ptr = list;
    } else {
        eshkol_error("mapWithClosure: list argument has unexpected type");
        return nullptr;
    }

    // Initialize result list
    Value* result_head = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "map_closure_result_head");
    Value* result_tail = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "map_closure_result_tail");
    Value* current_input = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "map_closure_current");

    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), result_head);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), result_tail);
    ctx_.builder().CreateStore(list_ptr, current_input);

    // Create loop blocks
    BasicBlock* loop_condition = BasicBlock::Create(ctx_.context(), "map_closure_loop_cond", current_func);
    BasicBlock* loop_body = BasicBlock::Create(ctx_.context(), "map_closure_loop_body", current_func);
    BasicBlock* loop_exit = BasicBlock::Create(ctx_.context(), "map_closure_loop_exit", current_func);

    ctx_.builder().CreateBr(loop_condition);

    // Loop condition
    ctx_.builder().SetInsertPoint(loop_condition);
    Value* current_val = ctx_.builder().CreateLoad(ctx_.int64Type(), current_input);
    Value* is_not_null = ctx_.builder().CreateICmpNE(current_val, ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(is_not_null, loop_body, loop_exit);

    // Loop body: apply closure and build result
    ctx_.builder().SetInsertPoint(loop_body);

    // Extract car as tagged_value
    Value* car_tagged = nullptr;
    if (extract_car_callback_) {
        car_tagged = extract_car_callback_(current_val, callback_context_);
    } else {
        eshkol_error("mapWithClosure: extract_car_callback not set");
        return tagged_.packNull();
    }

    // Call closure using runtime dispatch
    std::vector<Value*> call_args = {car_tagged};
    Value* mapped_val = nullptr;
    if (closure_call_callback_) {
        mapped_val = closure_call_callback_(closure_val, call_args, callback_context_);
    } else {
        eshkol_error("mapWithClosure: closure_call_callback not set");
        return tagged_.packNull();
    }

    // Allocate new cons cell
    Value* null_tagged = tagged_.packNull();
    Value* new_cons_int = nullptr;
    if (create_cons_callback_) {
        new_cons_int = create_cons_callback_(mapped_val, null_tagged, callback_context_);
    } else {
        eshkol_error("mapWithClosure: create_cons_callback not set");
        return tagged_.packNull();
    }

    // Add to result list
    Value* curr_head = ctx_.builder().CreateLoad(ctx_.int64Type(), result_head);
    Value* is_first = ctx_.builder().CreateICmpEQ(curr_head, ConstantInt::get(ctx_.int64Type(), 0));

    BasicBlock* set_head_bb = BasicBlock::Create(ctx_.context(), "map_closure_set_head", current_func);
    BasicBlock* append_bb = BasicBlock::Create(ctx_.context(), "map_closure_append", current_func);
    BasicBlock* continue_bb = BasicBlock::Create(ctx_.context(), "map_closure_continue", current_func);

    ctx_.builder().CreateCondBr(is_first, set_head_bb, append_bb);

    ctx_.builder().SetInsertPoint(set_head_bb);
    ctx_.builder().CreateStore(new_cons_int, result_head);
    ctx_.builder().CreateStore(new_cons_int, result_tail);
    ctx_.builder().CreateBr(continue_bb);

    ctx_.builder().SetInsertPoint(append_bb);
    // Update tail's cdr to point to new cons cell
    Value* tail_ptr = ctx_.builder().CreateLoad(ctx_.int64Type(), result_tail);
    Value* tail_cons = ctx_.builder().CreateIntToPtr(tail_ptr, ctx_.ptrType());
    // Tagged cons cell has: car (16 bytes) then cdr (16 bytes for tagged_value)
    Value* tail_cdr_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), tail_cons,
        ConstantInt::get(ctx_.int64Type(), 16));
    // Store new_cons as cdr (pack as CONS_PTR)
    Value* new_cons_as_ptr = ctx_.builder().CreateIntToPtr(new_cons_int, ctx_.ptrType());
    Value* new_cons_tagged = tagged_.packPtr(new_cons_as_ptr, ESHKOL_VALUE_CONS_PTR);
    ctx_.builder().CreateStore(new_cons_tagged, tail_cdr_ptr);
    ctx_.builder().CreateStore(new_cons_int, result_tail);
    ctx_.builder().CreateBr(continue_bb);

    ctx_.builder().SetInsertPoint(continue_bb);

    // Move to next element
    Value* current_val_reload = ctx_.builder().CreateLoad(ctx_.int64Type(), current_input);
    Value* cons_ptr = ctx_.builder().CreateIntToPtr(current_val_reload, ctx_.ptrType());

    // Cdr is at offset 16 (after 16-byte tagged_value car)
    Value* cdr_addr = ctx_.builder().CreateGEP(ctx_.int8Type(), cons_ptr,
        ConstantInt::get(ctx_.int64Type(), 16));
    Value* cdr_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), cdr_addr);

    // Extract pointer from cdr tagged_value
    Value* next_val = tagged_.unpackInt64(cdr_tagged);
    ctx_.builder().CreateStore(next_val, current_input);
    ctx_.builder().CreateBr(loop_condition);

    // Loop exit: return result
    ctx_.builder().SetInsertPoint(loop_exit);
    Value* final_head = ctx_.builder().CreateLoad(ctx_.int64Type(), result_head);
    Value* is_empty = ctx_.builder().CreateICmpEQ(final_head, ConstantInt::get(ctx_.int64Type(), 0));
    Value* null_val = tagged_.packNull();
    Value* list_val = tagged_.packPtr(
        ctx_.builder().CreateIntToPtr(final_head, ctx_.ptrType()),
        ESHKOL_VALUE_CONS_PTR);
    return ctx_.builder().CreateSelect(is_empty, null_val, list_val);
}

void MapCodegen::loadCapturedValues(
    Function* proc_func,
    const std::string& func_name,
    size_t first_capture_idx,
    std::vector<Value*>& args
) {
    FunctionType* proc_type = proc_func->getFunctionType();
    size_t expected_params = proc_type->getNumParams();

    if (expected_params <= first_capture_idx) return;

    std::string lambda_name = func_name;
    size_t num_captures = expected_params - first_capture_idx;

    for (size_t i = 0; i < num_captures; i++) {
        // Get variable name from parameter
        std::string var_name;
        auto arg_it = proc_func->arg_begin();
        std::advance(arg_it, i + first_capture_idx);
        if (arg_it != proc_func->arg_end()) {
            var_name = arg_it->getName().str();
            if (var_name.find("captured_") == 0) {
                var_name = var_name.substr(9);
            }
        }

        // Construct capture storage key
        std::string capture_key = lambda_name + "_capture_" + var_name;

        // Try capture-specific storage
        Value* storage = nullptr;
        bool found = false;

        if (global_symbol_table_) {
            auto it = global_symbol_table_->find(capture_key);
            if (it != global_symbol_table_->end()) {
                storage = it->second;
                found = true;
            }
        }
        if (!found && symbol_table_) {
            auto it = symbol_table_->find(capture_key);
            if (it != symbol_table_->end()) {
                storage = it->second;
                found = true;
            }
        }

        if (found && storage) {
            Value* captured_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), storage);
            args.push_back(captured_val);
        } else {
            // Fall back to looking up the original variable name in current scope
            Value* captured_val = nullptr;
            if (symbol_table_) {
                auto var_it = symbol_table_->find(var_name);
                if (var_it != symbol_table_->end() && var_it->second) {
                    Value* var_storage = var_it->second;
                    if (isa<AllocaInst>(var_storage)) {
                        captured_val = ctx_.builder().CreateLoad(
                            dyn_cast<AllocaInst>(var_storage)->getAllocatedType(), var_storage);
                    } else if (isa<GlobalVariable>(var_storage)) {
                        captured_val = ctx_.builder().CreateLoad(
                            dyn_cast<GlobalVariable>(var_storage)->getValueType(), var_storage);
                    } else {
                        captured_val = var_storage;
                    }
                    // Ensure tagged_value type
                    if (captured_val && captured_val->getType() != ctx_.taggedValueType()) {
                        if (captured_val->getType()->isIntegerTy(64)) {
                            captured_val = tagged_.packInt64(captured_val, true);
                        } else if (captured_val->getType()->isDoubleTy()) {
                            captured_val = tagged_.packDouble(captured_val);
                        }
                    }
                }
            }

            if (captured_val) {
                args.push_back(captured_val);
                eshkol_debug("Map: resolved capture '%s' from current scope for lambda '%s'",
                            var_name.c_str(), lambda_name.c_str());
            } else {
                // Fallback to zero
                args.push_back(tagged_.packInt64(ConstantInt::get(ctx_.int64Type(), 0), true));
                eshkol_warn("Map: capture '%s' not found for lambda '%s', using 0",
                            var_name.c_str(), lambda_name.c_str());
            }
        }
    }
}

Value* MapCodegen::mapSingleList(Function* proc_func, Value* list) {
    if (!proc_func || !list) return nullptr;

    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    Function* cons_get_ptr = getConsGetPtrFunc();
    Function* cons_set_ptr = getConsSetPtrFunc();
    if (!cons_get_ptr || !cons_set_ptr) return tagged_.packNull();

    eshkol_debug("Single-list map starting - no arena scoping to prevent memory corruption");

    // Handle tagged_value input
    Value* list_ptr;
    if (list->getType() == ctx_.taggedValueType()) {
        list_ptr = tagged_.unpackInt64(list);
    } else if (list->getType()->isIntegerTy(64)) {
        list_ptr = list;
    } else {
        eshkol_error("Map: list argument has unexpected type");
        return nullptr;
    }

    // Initialize result list
    Value* result_head = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "map_result_head");
    Value* result_tail = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "map_result_tail");
    Value* current_input = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "map_current");

    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), result_head);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), result_tail);
    ctx_.builder().CreateStore(list_ptr, current_input);

    // Create loop blocks
    BasicBlock* loop_condition = BasicBlock::Create(ctx_.context(), "map_loop_cond", current_func);
    BasicBlock* loop_body = BasicBlock::Create(ctx_.context(), "map_loop_body", current_func);
    BasicBlock* loop_exit = BasicBlock::Create(ctx_.context(), "map_loop_exit", current_func);

    ctx_.builder().CreateBr(loop_condition);

    // Loop condition
    ctx_.builder().SetInsertPoint(loop_condition);
    Value* current_val = ctx_.builder().CreateLoad(ctx_.int64Type(), current_input);
    Value* is_not_null = ctx_.builder().CreateICmpNE(current_val, ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(is_not_null, loop_body, loop_exit);

    // Loop body
    ctx_.builder().SetInsertPoint(loop_body);

    // Extract car as tagged_value
    Value* car_tagged = nullptr;
    if (extract_car_callback_) {
        car_tagged = extract_car_callback_(current_val, callback_context_);
    } else {
        eshkol_error("mapSingleList: extract_car_callback not set");
        return tagged_.packNull();
    }

    // Build procedure arguments
    std::vector<Value*> proc_args;
    std::string func_name = proc_func->getName().str();

    // Check for indirect call wrapper
    bool is_indirect_call = func_name.find("indirect_call_") == 0;
    if (is_indirect_call && symbol_table_) {
        auto indirect_it = symbol_table_->find("__indirect_func_arg__");
        if (indirect_it != symbol_table_->end() && indirect_it->second) {
            proc_args.push_back(indirect_it->second);
        } else {
            eshkol_error("Map: indirect call wrapper missing function argument");
            return nullptr;
        }
    }

    proc_args.push_back(car_tagged);

    // Load captured values
    size_t first_capture_idx = is_indirect_call ? 2 : 1;
    loadCapturedValues(proc_func, func_name, first_capture_idx, proc_args);

    // Apply procedure
    Value* proc_result = ctx_.builder().CreateCall(proc_func, proc_args);

    // Create new cons cell for result
    Value* cdr_null_tagged = tagged_.packNull();
    Value* new_result_cons = nullptr;
    if (create_cons_callback_) {
        new_result_cons = create_cons_callback_(proc_result, cdr_null_tagged, callback_context_);
    } else {
        eshkol_error("mapSingleList: create_cons_callback not set");
        return tagged_.packNull();
    }

    // Update result list
    Value* head_val = ctx_.builder().CreateLoad(ctx_.int64Type(), result_head);
    Value* head_is_empty = ctx_.builder().CreateICmpEQ(head_val, ConstantInt::get(ctx_.int64Type(), 0));

    BasicBlock* set_head = BasicBlock::Create(ctx_.context(), "map_set_head", current_func);
    BasicBlock* update_tail = BasicBlock::Create(ctx_.context(), "map_update_tail", current_func);
    BasicBlock* continue_map = BasicBlock::Create(ctx_.context(), "map_continue", current_func);

    ctx_.builder().CreateCondBr(head_is_empty, set_head, update_tail);

    // Set head
    ctx_.builder().SetInsertPoint(set_head);
    ctx_.builder().CreateStore(new_result_cons, result_head);
    ctx_.builder().CreateStore(new_result_cons, result_tail);
    ctx_.builder().CreateBr(continue_map);

    // Update tail
    ctx_.builder().SetInsertPoint(update_tail);
    Value* tail_val = ctx_.builder().CreateLoad(ctx_.int64Type(), result_tail);
    Value* tail_cons_ptr = ctx_.builder().CreateIntToPtr(tail_val, ctx_.ptrType());

    Value* is_cdr_set = ConstantInt::get(ctx_.int1Type(), 1);
    Value* ptr_type = ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CONS_PTR);
    ctx_.builder().CreateCall(cons_set_ptr, {tail_cons_ptr, is_cdr_set, new_result_cons, ptr_type});
    ctx_.builder().CreateStore(new_result_cons, result_tail);
    ctx_.builder().CreateBr(continue_map);

    // Continue: move to next input element
    ctx_.builder().SetInsertPoint(continue_map);
    Value* input_cons_ptr = ctx_.builder().CreateIntToPtr(current_val, ctx_.ptrType());
    Value* is_cdr_get = ConstantInt::get(ctx_.int1Type(), 1);
    Value* input_cdr = ctx_.builder().CreateCall(cons_get_ptr, {input_cons_ptr, is_cdr_get});
    ctx_.builder().CreateStore(input_cdr, current_input);
    ctx_.builder().CreateBr(loop_condition);

    // Loop exit
    ctx_.builder().SetInsertPoint(loop_exit);
    Value* final_result = ctx_.builder().CreateLoad(ctx_.int64Type(), result_head);

    eshkol_debug("Single-list map completed - cons cells remain persistent in arena memory");

    return tagged_.packPtr(ctx_.builder().CreateIntToPtr(final_result, ctx_.ptrType()),
                          ESHKOL_VALUE_CONS_PTR);
}

Value* MapCodegen::mapMultiList(Function* proc_func, const std::vector<Value*>& lists) {
    if (!proc_func || lists.empty()) return nullptr;

    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    Function* cons_get_ptr = getConsGetPtrFunc();
    Function* cons_set_ptr = getConsSetPtrFunc();
    if (!cons_get_ptr || !cons_set_ptr) return tagged_.packNull();

    eshkol_debug("Map operation starting - no arena scoping to prevent memory corruption");

    // Initialize result list
    Value* result_head = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "multimap_result_head");
    Value* result_tail = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "multimap_result_tail");

    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), result_head);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), result_tail);

    // Initialize current pointers for each input list
    std::vector<Value*> current_ptrs;
    for (size_t i = 0; i < lists.size(); i++) {
        Value* list_ptr;
        if (lists[i]->getType() == ctx_.taggedValueType()) {
            list_ptr = tagged_.unpackInt64(lists[i]);
        } else if (lists[i]->getType()->isIntegerTy(64)) {
            list_ptr = lists[i];
        } else {
            eshkol_error("MultiMap: list %zu has unexpected type", i);
            return nullptr;
        }

        Value* current_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr,
            ("multimap_current_" + std::to_string(i)).c_str());
        ctx_.builder().CreateStore(list_ptr, current_ptr);
        current_ptrs.push_back(current_ptr);
    }

    // Create loop blocks
    BasicBlock* loop_condition = BasicBlock::Create(ctx_.context(), "multimap_loop_cond", current_func);
    BasicBlock* loop_body = BasicBlock::Create(ctx_.context(), "multimap_loop_body", current_func);
    BasicBlock* loop_exit = BasicBlock::Create(ctx_.context(), "multimap_loop_exit", current_func);

    ctx_.builder().CreateBr(loop_condition);

    // Loop condition: check if ALL lists still have elements
    ctx_.builder().SetInsertPoint(loop_condition);
    Value* all_not_null = ConstantInt::get(ctx_.int1Type(), 1);

    for (size_t i = 0; i < current_ptrs.size(); i++) {
        Value* current_val = ctx_.builder().CreateLoad(ctx_.int64Type(), current_ptrs[i]);
        Value* is_not_null = ctx_.builder().CreateICmpNE(current_val, ConstantInt::get(ctx_.int64Type(), 0));
        all_not_null = ctx_.builder().CreateAnd(all_not_null, is_not_null);
    }

    ctx_.builder().CreateCondBr(all_not_null, loop_body, loop_exit);

    // Loop body
    ctx_.builder().SetInsertPoint(loop_body);

    // Extract car from each list
    std::vector<Value*> proc_args;
    for (size_t i = 0; i < current_ptrs.size(); i++) {
        Value* current_val = ctx_.builder().CreateLoad(ctx_.int64Type(), current_ptrs[i]);
        Value* car_tagged = nullptr;
        if (extract_car_callback_) {
            car_tagged = extract_car_callback_(current_val, callback_context_);
        } else {
            eshkol_error("mapMultiList: extract_car_callback not set");
            return tagged_.packNull();
        }
        proc_args.push_back(car_tagged);
    }

    // Load captured values
    std::string func_name = proc_func->getName().str();
    loadCapturedValues(proc_func, func_name, lists.size(), proc_args);

    eshkol_debug("MultiMap: About to call %s function with %zu arguments",
                func_name.c_str(), proc_args.size());

    // Apply procedure
    Value* proc_result = ctx_.builder().CreateCall(proc_func, proc_args);

    // Create new cons cell for result
    Value* cdr_null_tagged = tagged_.packNull();
    Value* new_result_cons = nullptr;
    if (create_cons_callback_) {
        new_result_cons = create_cons_callback_(proc_result, cdr_null_tagged, callback_context_);
    } else {
        eshkol_error("mapMultiList: create_cons_callback not set");
        return tagged_.packNull();
    }

    // Update result list
    Value* head_val = ctx_.builder().CreateLoad(ctx_.int64Type(), result_head);
    Value* head_is_empty = ctx_.builder().CreateICmpEQ(head_val, ConstantInt::get(ctx_.int64Type(), 0));

    BasicBlock* set_head = BasicBlock::Create(ctx_.context(), "multimap_set_head", current_func);
    BasicBlock* update_tail = BasicBlock::Create(ctx_.context(), "multimap_update_tail", current_func);
    BasicBlock* continue_multimap = BasicBlock::Create(ctx_.context(), "multimap_continue", current_func);

    ctx_.builder().CreateCondBr(head_is_empty, set_head, update_tail);

    // Set head
    ctx_.builder().SetInsertPoint(set_head);
    ctx_.builder().CreateStore(new_result_cons, result_head);
    ctx_.builder().CreateStore(new_result_cons, result_tail);
    ctx_.builder().CreateBr(continue_multimap);

    // Update tail
    ctx_.builder().SetInsertPoint(update_tail);
    Value* tail_val = ctx_.builder().CreateLoad(ctx_.int64Type(), result_tail);
    Value* tail_cons_ptr = ctx_.builder().CreateIntToPtr(tail_val, ctx_.ptrType());

    Value* is_cdr_set = ConstantInt::get(ctx_.int1Type(), 1);
    Value* ptr_type = ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CONS_PTR);
    ctx_.builder().CreateCall(cons_set_ptr, {tail_cons_ptr, is_cdr_set, new_result_cons, ptr_type});
    ctx_.builder().CreateStore(new_result_cons, result_tail);
    ctx_.builder().CreateBr(continue_multimap);

    // Continue: advance all list pointers
    ctx_.builder().SetInsertPoint(continue_multimap);
    for (size_t i = 0; i < current_ptrs.size(); i++) {
        Value* current_val = ctx_.builder().CreateLoad(ctx_.int64Type(), current_ptrs[i]);
        Value* cons_ptr = ctx_.builder().CreateIntToPtr(current_val, ctx_.ptrType());

        Value* is_cdr_get = ConstantInt::get(ctx_.int1Type(), 1);
        Value* cdr_val = ctx_.builder().CreateCall(cons_get_ptr, {cons_ptr, is_cdr_get});
        ctx_.builder().CreateStore(cdr_val, current_ptrs[i]);
    }

    ctx_.builder().CreateBr(loop_condition);

    // Loop exit
    ctx_.builder().SetInsertPoint(loop_exit);
    Value* final_result = ctx_.builder().CreateLoad(ctx_.int64Type(), result_head);

    eshkol_debug("Multi-list map completed - cons cells remain persistent in arena memory");

    return tagged_.packPtr(ctx_.builder().CreateIntToPtr(final_result, ctx_.ptrType()),
                          ESHKOL_VALUE_CONS_PTR);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
