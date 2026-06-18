/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Tensor Creation. Extracted from tensor_codegen.cpp
 * during the v1.2 mechanical split.
 *
 * Implements the tensor-creation surface: createTensorWithDims (the
 * shared shape-aware allocator) plus the user-facing factory ops
 * (zeros, ones, eye, arange, linspace, full, etc.).
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-creation-extract baseline.
 */
#include <eshkol/backend/tensor_codegen.h>

#include <cstring>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Config/llvm-config.h>

// LLVM VERSION COMPATIBILITY
#if LLVM_VERSION_MAJOR >= 21
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getOrInsertDeclaration(mod, id, types)
#else
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getDeclaration(mod, id, types)
#endif

namespace eshkol {

// ===== TENSOR CREATION HELPER =====

llvm::Value* TensorCodegen::createTensorWithDims(const std::vector<llvm::Value*>& dims,
                                                   llvm::Value* fill_value,
                                                   bool use_memset_zero) {
    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get arena pointer
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(context, 0), ctx_.globalArena());

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Calculate total elements
    llvm::Value* total_elements = dims[0];
    for (size_t i = 1; i < dims.size(); i++) {
        total_elements = builder.CreateMul(total_elements, dims[i]);
    }

    // Allocate tensor structure with header using arena
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* typed_tensor_ptr = builder.CreateCall(alloc_tensor_func, {arena_ptr}, "new_tensor");

    // Allocate dimensions array using arena
    llvm::Value* dims_size = llvm::ConstantInt::get(ctx_.int64Type(), dims.size() * sizeof(uint64_t));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* dims_ptr = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "new_dims");
    llvm::Value* typed_dims_ptr = builder.CreatePointerCast(dims_ptr, ctx_.ptrType());

    // Store dimensions
    for (size_t i = 0; i < dims.size(); i++) {
        llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), typed_dims_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        builder.CreateStore(dims[i], dim_ptr);
    }

    // Allocate elements array using arena
    llvm::Value* elements_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* elements_ptr = builder.CreateCall(arena_alloc, {arena_ptr, elements_size}, "new_elems");
    llvm::Value* typed_elements_ptr = builder.CreatePointerCast(elements_ptr, ctx_.ptrType());

    // Fill elements if requested
    if (use_memset_zero) {
        // Use memset for efficient zero-fill
        llvm::Function* memset_func = ctx_.module().getFunction("memset");
        if (!memset_func) {
            // Declare memset: void* memset(void*, int, size_t)
            llvm::FunctionType* memset_type = llvm::FunctionType::get(
                ctx_.ptrType(),
                {ctx_.ptrType(), ctx_.int32Type(), ctx_.int64Type()},
                false);
            memset_func = llvm::Function::Create(
                memset_type, llvm::Function::ExternalLinkage, "memset", &ctx_.module());
        }
        ctx_.builder().CreateCall(memset_func, {
            typed_elements_ptr,
            llvm::ConstantInt::get(ctx_.int32Type(), 0),
            elements_size
        });
    } else if (fill_value) {
        // Fill with specified value using loop
        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "fill_cond", current_func);
        llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "fill_body", current_func);
        llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "fill_exit", current_func);

        llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "fill_i");
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
        ctx_.builder().CreateBr(loop_cond);

        ctx_.builder().SetInsertPoint(loop_cond);
        llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
        llvm::Value* cmp = ctx_.builder().CreateICmpULT(i, total_elements);
        ctx_.builder().CreateCondBr(cmp, loop_body, loop_exit);

        ctx_.builder().SetInsertPoint(loop_body);
        llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr, i);
        ctx_.builder().CreateStore(fill_value, elem_ptr);
        llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        ctx_.builder().CreateStore(next_i, counter);
        ctx_.builder().CreateBr(loop_cond);

        ctx_.builder().SetInsertPoint(loop_exit);
    }

    // Store tensor fields
    llvm::Value* dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_tensor_ptr, 0);
    ctx_.builder().CreateStore(typed_dims_ptr, dims_field_ptr);

    llvm::Value* num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_tensor_ptr, 1);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), dims.size()), num_dims_field_ptr);

    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_tensor_ptr, 2);
    ctx_.builder().CreateStore(typed_elements_ptr, elements_field_ptr);

    llvm::Value* total_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_tensor_ptr, 3);
    ctx_.builder().CreateStore(total_elements, total_elements_field_ptr);

    return typed_tensor_ptr;
}

// ===== TENSOR CREATION FUNCTIONS =====

llvm::Value* TensorCodegen::tensor(const eshkol_operations_t* op) {
    // tensor: (tensor e1 e2 e3 ...) - Create 1D tensor from elements
    if (op->call_op.num_vars < 1) {
        eshkol_error("tensor requires at least 1 element");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    uint64_t n = op->call_op.num_vars;

    // Create 1D tensor with n elements
    std::vector<llvm::Value*> dims = {llvm::ConstantInt::get(ctx_.int64Type(), n)};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, false);
    if (!tensor_ptr) return nullptr;

    // Get elements pointer from tensor struct
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // Store each element
    for (uint64_t i = 0; i < n; ++i) {
        llvm::Value* elem = codegenAST(&op->call_op.variables[i]);
        if (!elem) return nullptr;

        // Extract double value (handle tagged or raw)
        llvm::Value* elem_double;
        if (elem->getType() == ctx_.taggedValueType()) {
            elem_double = tagged_.unpackDouble(elem);
        } else if (elem->getType() == ctx_.doubleType()) {
            elem_double = elem;
        } else if (elem->getType()->isIntegerTy(64)) {
            elem_double = builder.CreateBitCast(elem, ctx_.doubleType());
        } else {
            elem_double = builder.CreateSIToFP(elem, ctx_.doubleType());
        }

        // Store as int64 bit pattern
        llvm::Value* elem_bits = builder.CreateBitCast(elem_double, ctx_.int64Type());
        llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        builder.CreateStore(elem_bits, elem_ptr);
    }

    return tagged_.packHeapPtr(tensor_ptr);
}

namespace {

// Read the name of a quoted-symbol argument. `'f16` parses to an
// ESHKOL_QUOTE_OP wrapping a variable node; a bare symbol literal is
// ESHKOL_SYMBOL. Returns nullptr if the node is not symbol-like.
const char* extract_symbol_arg_name(const eshkol_ast_t* node) {
    if (!node) return nullptr;
    const eshkol_ast_t* inner = node;
    if (node->type == ESHKOL_OP &&
        node->operation.op == ESHKOL_QUOTE_OP &&
        node->operation.call_op.num_vars >= 1 &&
        node->operation.call_op.variables) {
        inner = &node->operation.call_op.variables[0];
    }
    if (inner->type == ESHKOL_VAR && inner->variable.id) return inner->variable.id;
    if (inner->type == ESHKOL_SYMBOL && inner->str_val.ptr) return inner->str_val.ptr;
    return nullptr;
}

// Map a dtype symbol name to its eshkol_tensor_dtype_t code, or -1 if unknown.
int64_t dtype_code_for_name(const char* name) {
    if (!name) return -1;
    if (std::strcmp(name, "f64") == 0)  return 0;
    if (std::strcmp(name, "f32") == 0)  return 1;
    if (std::strcmp(name, "f16") == 0)  return 2;
    if (std::strcmp(name, "bf16") == 0) return 3;
    if (std::strcmp(name, "i8") == 0)   return 4;
    return -1;
}

}  // namespace

// Apply a dtype code to an already-built tensor value in place (reduces element
// precision + records the dtype). Returns the same tagged tensor value.
llvm::Value* TensorCodegen::applyDtypeToTensor(llvm::Value* tensor_val,
                                               int64_t dtype_code) {
    auto& builder = ctx_.builder();
    llvm::Value* ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* ptr = builder.CreateIntToPtr(ptr_int, ctx_.ptrType());
    llvm::Function* fn = ctx_.module().getFunction("eshkol_tensor_apply_dtype");
    if (!fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.ptrType(), {ctx_.ptrType(), ctx_.int64Type()}, false);
        fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                    "eshkol_tensor_apply_dtype", &ctx_.module());
    }
    builder.CreateCall(fn, {ptr, llvm::ConstantInt::get(ctx_.int64Type(), dtype_code)});
    return tensor_val;
}

// Set result_ptr's dtype from two input tensor pointers (binary op promotion).
// All three are raw tensor pointers (not tagged values).
void TensorCodegen::emitDtypePropagateBinary(llvm::Value* result_ptr,
                                             llvm::Value* a_ptr,
                                             llvm::Value* b_ptr) {
    auto& builder = ctx_.builder();
    llvm::Function* fn = ctx_.module().getFunction("eshkol_tensor_result_dtype_binary");
    if (!fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.ptrType(), {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
        fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                    "eshkol_tensor_result_dtype_binary", &ctx_.module());
    }
    builder.CreateCall(fn, {result_ptr, a_ptr, b_ptr});
}

// Set result_ptr's dtype from a single input tensor pointer (unary op / view).
void TensorCodegen::emitDtypePropagateUnary(llvm::Value* result_ptr,
                                            llvm::Value* a_ptr) {
    auto& builder = ctx_.builder();
    llvm::Function* fn = ctx_.module().getFunction("eshkol_tensor_result_dtype_unary");
    if (!fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.ptrType(), {ctx_.ptrType(), ctx_.ptrType()}, false);
        fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                    "eshkol_tensor_result_dtype_unary", &ctx_.module());
    }
    builder.CreateCall(fn, {result_ptr, a_ptr});
}

// (tensor-dtype t) -> symbol naming the tensor's element dtype.
llvm::Value* TensorCodegen::tensorDtype(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-dtype requires exactly 1 argument");
        return nullptr;
    }
    auto& builder = ctx_.builder();
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    llvm::Value* ptr_int = tagged_.unpackInt64(t);
    llvm::Value* ptr = builder.CreateIntToPtr(ptr_int, ctx_.ptrType());
    llvm::Function* fn = ctx_.module().getFunction("eshkol_tensor_dtype_symbol");
    if (!fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.ptrType(), {ctx_.ptrType()}, false);
        fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                    "eshkol_tensor_dtype_symbol", &ctx_.module());
    }
    llvm::Value* sym = builder.CreateCall(fn, {ptr});
    return tagged_.packHeapPtr(sym);
}

// (tensor-cast t 'dtype) -> new tensor with elements cast through that dtype.
llvm::Value* TensorCodegen::tensorCast(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) {
        eshkol_error("tensor-cast requires 2 arguments: (tensor-cast t 'dtype)");
        return nullptr;
    }
    int64_t code = dtype_code_for_name(extract_symbol_arg_name(&op->call_op.variables[1]));
    if (code < 0) {
        eshkol_error("tensor-cast dtype must be one of f64/f32/f16/bf16/i8");
        return nullptr;
    }
    auto& builder = ctx_.builder();
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    llvm::Value* arena_ptr = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* ptr_int = tagged_.unpackInt64(t);
    llvm::Value* ptr = builder.CreateIntToPtr(ptr_int, ctx_.ptrType());
    llvm::Function* fn = ctx_.module().getFunction("eshkol_tensor_cast_alloc");
    if (!fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.ptrType(), {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
        fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                    "eshkol_tensor_cast_alloc", &ctx_.module());
    }
    llvm::Value* result = builder.CreateCall(
        fn, {arena_ptr, ptr, llvm::ConstantInt::get(ctx_.int64Type(), code)});
    return tagged_.packHeapPtr(result);
}

// make-tensor wrapper: strips an optional trailing `:dtype 'sym` keyword, builds
// the tensor via makeTensorImpl, then applies the requested dtype.
llvm::Value* TensorCodegen::makeTensor(const eshkol_operations_t* op) {
    int64_t dtype_code = -1;
    uint32_t positional = op->call_op.num_vars;
    for (uint32_t i = 0; i + 1 < op->call_op.num_vars; i++) {
        const eshkol_ast_t* a = &op->call_op.variables[i];
        if (a->type == ESHKOL_VAR && a->variable.id &&
            std::strcmp(a->variable.id, ":dtype") == 0) {
            int64_t code = dtype_code_for_name(
                extract_symbol_arg_name(&op->call_op.variables[i + 1]));
            if (code < 0) {
                eshkol_error("make-tensor :dtype expects one of f64/f32/f16/bf16/i8");
                return nullptr;
            }
            dtype_code = code;
            positional = i;  // keyword args are trailing
            break;
        }
    }
    if (dtype_code < 0) return makeTensorImpl(op);

    eshkol_operations_t eff = *op;
    eff.call_op.num_vars = positional;
    llvm::Value* t = makeTensorImpl(&eff);
    if (!t) return nullptr;
    return applyDtypeToTensor(t, dtype_code);
}

llvm::Value* TensorCodegen::makeTensorImpl(const eshkol_operations_t* op) {
    // make-tensor: Two modes:
    //   (make-tensor shape fill_value) - create tensor with given shape filled with fill_value
    //   (make-tensor e1 e2 e3 ...)    - create 1D tensor from elements (falls back to tensor())
    if (op->call_op.num_vars < 1) {
        eshkol_error("make-tensor requires at least 1 argument");
        return nullptr;
    }

    // If not exactly 2 args, fall back to element-literal mode
    if (op->call_op.num_vars != 2) {
        return tensor(op);
    }

    // 2 args: (make-tensor shape fill_value)
    // First arg is shape (list of dimensions), second is fill value
    auto& builder = ctx_.builder();

    llvm::Value* shape_arg = codegenAST(&op->call_op.variables[0]);
    if (!shape_arg) return nullptr;

    llvm::Value* fill_arg = codegenAST(&op->call_op.variables[1]);
    if (!fill_arg) return nullptr;

    // Extract fill value as double → int64 bit pattern
    llvm::Value* fill_double;
    if (fill_arg->getType() == ctx_.taggedValueType()) {
        fill_double = extractAsDouble(fill_arg);
    } else if (fill_arg->getType() == ctx_.doubleType()) {
        fill_double = fill_arg;
    } else if (fill_arg->getType()->isIntegerTy()) {
        fill_double = builder.CreateSIToFP(fill_arg, ctx_.doubleType());
    } else {
        fill_double = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    }
    llvm::Value* fill_bits = builder.CreateBitCast(fill_double, ctx_.int64Type());

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Check if first arg is a list/cons (shape) or a scalar (element)
    // If it's a list → shape+fill mode; if scalar → element-literal mode
    llvm::BasicBlock* heap_dispatch = llvm::BasicBlock::Create(ctx_.context(), "mt_heap_dispatch", current_func);
    llvm::BasicBlock* list_path = llvm::BasicBlock::Create(ctx_.context(), "mt_list", current_func);
    llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "mt_vector", current_func);
    llvm::BasicBlock* tensor_path = llvm::BasicBlock::Create(ctx_.context(), "mt_tensor_shape", current_func);
    llvm::BasicBlock* scalar_path = llvm::BasicBlock::Create(ctx_.context(), "mt_scalar", current_func);

    if (shape_arg->getType() == ctx_.taggedValueType()) {
        llvm::Value* type_tag = tagged_.getType(shape_arg);
        llvm::Value* base_type = tagged_.getBaseType(type_tag);
        // Lists are HEAP_PTR (cons cells)
        llvm::Value* is_heap = builder.CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        builder.CreateCondBr(is_heap, heap_dispatch, scalar_path);
    } else {
        // Raw value — not a list, use element-literal mode
        builder.CreateBr(scalar_path);
    }

    builder.SetInsertPoint(heap_dispatch);
    llvm::Value* shape_ptr_int = tagged_.unpackInt64(shape_arg);
    llvm::Value* shape_ptr = builder.CreateIntToPtr(shape_ptr_int, ctx_.ptrType());
    llvm::Value* shape_header_ptr = builder.CreateGEP(
        ctx_.int8Type(), shape_ptr, llvm::ConstantInt::get(ctx_.int64Type(), -8));
    llvm::Value* shape_subtype = builder.CreateLoad(ctx_.int8Type(), shape_header_ptr);
    llvm::Value* is_vector_shape = builder.CreateICmpEQ(
        shape_subtype,
        llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    llvm::Value* is_tensor_shape = builder.CreateICmpEQ(
        shape_subtype,
        llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));
    llvm::BasicBlock* heap_subtype_check = llvm::BasicBlock::Create(
        ctx_.context(), "mt_heap_subtype_check", current_func);
    builder.CreateCondBr(is_vector_shape, vector_path, heap_subtype_check);

    builder.SetInsertPoint(heap_subtype_check);
    builder.CreateCondBr(is_tensor_shape, tensor_path, list_path);

    // SCALAR PATH: treat as 2-element tensor (fallback to tensor())
    builder.SetInsertPoint(scalar_path);
    // Create 1D tensor with 2 elements [shape_arg, fill_arg]
    std::vector<llvm::Value*> scalar_dims = {llvm::ConstantInt::get(ctx_.int64Type(), 2)};
    llvm::Value* scalar_tensor = createTensorWithDims(scalar_dims, nullptr, false);
    if (scalar_tensor) {
        llvm::StructType* tensor_type = ctx_.tensorType();
        llvm::Value* s_elems_field = builder.CreateStructGEP(tensor_type, scalar_tensor, 2);
        llvm::Value* s_elems = builder.CreateLoad(ctx_.ptrType(), s_elems_field);
        // Store shape_arg as element 0
        llvm::Value* e0_double;
        if (shape_arg->getType() == ctx_.taggedValueType()) {
            e0_double = extractAsDouble(shape_arg);
        } else if (shape_arg->getType() == ctx_.doubleType()) {
            e0_double = shape_arg;
        } else {
            e0_double = builder.CreateSIToFP(shape_arg, ctx_.doubleType());
        }
        builder.CreateStore(builder.CreateBitCast(e0_double, ctx_.int64Type()),
            builder.CreateGEP(ctx_.int64Type(), s_elems, llvm::ConstantInt::get(ctx_.int64Type(), 0)));
        // Store fill_arg as element 1
        builder.CreateStore(fill_bits,
            builder.CreateGEP(ctx_.int64Type(), s_elems, llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    }
    llvm::BasicBlock* final_merge = llvm::BasicBlock::Create(ctx_.context(), "mt_final", current_func);
    builder.CreateBr(final_merge);
    llvm::BasicBlock* scalar_exit = builder.GetInsertBlock();

    // LIST PATH: Extract dimensions from cons list (up to 4D)
    builder.SetInsertPoint(list_path);
    llvm::Value* list_ptr_int = tagged_.unpackInt64(shape_arg);
    llvm::Value* cons_ptr = builder.CreateIntToPtr(list_ptr_int, ctx_.ptrType());
    llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    llvm::Value* is_cdr_flag = llvm::ConstantInt::get(ctx_.int1Type(), 1);

    // Extract first dimension
    llvm::Value* dim1 = builder.CreateCall(
        mem_.getTaggedConsGetInt64(), {cons_ptr, is_car});

    // Get cdr
    llvm::Value* cdr_int = builder.CreateCall(
        mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr_flag});
    llvm::Value* cdr_null = builder.CreateICmpEQ(cdr_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    llvm::BasicBlock* mt_has_dim2 = llvm::BasicBlock::Create(ctx_.context(), "mt_has_dim2", current_func);
    llvm::BasicBlock* mt_list_1d = llvm::BasicBlock::Create(ctx_.context(), "mt_list_1d", current_func);
    builder.CreateCondBr(cdr_null, mt_list_1d, mt_has_dim2);

    // 1D case
    builder.SetInsertPoint(mt_list_1d);
    builder.CreateBr(final_merge);  // placeholder, will fix
    llvm::BasicBlock* mt_merge = llvm::BasicBlock::Create(ctx_.context(), "mt_dims_merge", current_func);
    // Re-route: go to merge
    mt_list_1d->getTerminator()->eraseFromParent();
    builder.SetInsertPoint(mt_list_1d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* list_1d_exit = builder.GetInsertBlock();

    // 2D+ case
    builder.SetInsertPoint(mt_has_dim2);
    llvm::Value* cdr_cons = builder.CreateIntToPtr(cdr_int, ctx_.ptrType());
    llvm::Value* dim2 = builder.CreateCall(
        mem_.getTaggedConsGetInt64(), {cdr_cons, is_car});

    llvm::Value* cddr_int = builder.CreateCall(
        mem_.getTaggedConsGetPtr(), {cdr_cons, is_cdr_flag});
    llvm::Value* cddr_null = builder.CreateICmpEQ(cddr_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    llvm::BasicBlock* mt_has_dim3 = llvm::BasicBlock::Create(ctx_.context(), "mt_has_dim3", current_func);
    llvm::BasicBlock* mt_list_2d = llvm::BasicBlock::Create(ctx_.context(), "mt_list_2d", current_func);
    builder.CreateCondBr(cddr_null, mt_list_2d, mt_has_dim3);

    // 2D case
    builder.SetInsertPoint(mt_list_2d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* list_2d_exit = builder.GetInsertBlock();

    // 3D case
    builder.SetInsertPoint(mt_has_dim3);
    llvm::Value* cddr_cons = builder.CreateIntToPtr(cddr_int, ctx_.ptrType());
    llvm::Value* dim3 = builder.CreateCall(
        mem_.getTaggedConsGetInt64(), {cddr_cons, is_car});
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* list_3d_exit = builder.GetInsertBlock();

    // MERGE dimensions
    builder.SetInsertPoint(mt_merge);

    llvm::PHINode* ndims_phi = builder.CreatePHI(ctx_.int64Type(), 9, "mt_ndims");
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 2), list_2d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 3), list_3d_exit);

    llvm::PHINode* d1_phi = builder.CreatePHI(ctx_.int64Type(), 9, "mt_d1");
    d1_phi->addIncoming(dim1, list_1d_exit);
    d1_phi->addIncoming(dim1, list_2d_exit);
    d1_phi->addIncoming(dim1, list_3d_exit);

    llvm::PHINode* d2_phi = builder.CreatePHI(ctx_.int64Type(), 9, "mt_d2");
    d2_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
    d2_phi->addIncoming(dim2, list_2d_exit);
    d2_phi->addIncoming(dim2, list_3d_exit);

    llvm::PHINode* d3_phi = builder.CreatePHI(ctx_.int64Type(), 9, "mt_d3");
    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_2d_exit);
    d3_phi->addIncoming(dim3, list_3d_exit);

    // Create tensor based on ndims
    llvm::Value* is_1d = builder.CreateICmpEQ(ndims_phi,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* is_2d = builder.CreateICmpEQ(ndims_phi,
        llvm::ConstantInt::get(ctx_.int64Type(), 2));

    llvm::BasicBlock* mt_create_1d = llvm::BasicBlock::Create(ctx_.context(), "mt_c1d", current_func);
    llvm::BasicBlock* mt_check_2d = llvm::BasicBlock::Create(ctx_.context(), "mt_chk2d", current_func);
    llvm::BasicBlock* mt_create_2d = llvm::BasicBlock::Create(ctx_.context(), "mt_c2d", current_func);
    llvm::BasicBlock* mt_create_3d = llvm::BasicBlock::Create(ctx_.context(), "mt_c3d", current_func);
    llvm::BasicBlock* mt_list_done = llvm::BasicBlock::Create(ctx_.context(), "mt_ldone", current_func);

    builder.CreateCondBr(is_1d, mt_create_1d, mt_check_2d);

    builder.SetInsertPoint(mt_create_1d);
    std::vector<llvm::Value*> dims_1d = {d1_phi};
    llvm::Value* t1d = createTensorWithDims(dims_1d, fill_bits, false);
    builder.CreateBr(mt_list_done);
    llvm::BasicBlock* c1d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_check_2d);
    builder.CreateCondBr(is_2d, mt_create_2d, mt_create_3d);

    builder.SetInsertPoint(mt_create_2d);
    std::vector<llvm::Value*> dims_2d = {d1_phi, d2_phi};
    llvm::Value* t2d = createTensorWithDims(dims_2d, fill_bits, false);
    builder.CreateBr(mt_list_done);
    llvm::BasicBlock* c2d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_create_3d);
    std::vector<llvm::Value*> dims_3d = {d1_phi, d2_phi, d3_phi};
    llvm::Value* t3d = createTensorWithDims(dims_3d, fill_bits, false);
    builder.CreateBr(mt_list_done);
    llvm::BasicBlock* c3d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_list_done);
    llvm::PHINode* list_tensor = builder.CreatePHI(ctx_.ptrType(), 3, "mt_tensor");
    list_tensor->addIncoming(t1d, c1d_exit);
    list_tensor->addIncoming(t2d, c2d_exit);
    list_tensor->addIncoming(t3d, c3d_exit);

    builder.CreateBr(final_merge);
    llvm::BasicBlock* list_done_exit = builder.GetInsertBlock();

    // VECTOR PATH: Extract dimensions from Scheme vector #(d1 d2 d3)
    builder.SetInsertPoint(vector_path);
    llvm::Value* vector_ptr_int = tagged_.unpackInt64(shape_arg);
    llvm::Value* vector_ptr = builder.CreateIntToPtr(vector_ptr_int, ctx_.ptrType());
    llvm::Value* vector_len = builder.CreateLoad(ctx_.int64Type(), vector_ptr);
    llvm::Value* vector_elems = builder.CreateGEP(
        ctx_.int8Type(), vector_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* vector_elems_typed = builder.CreatePointerCast(vector_elems, ctx_.ptrType());

    llvm::Value* vec_dim1_ptr = builder.CreateGEP(
        ctx_.taggedValueType(), vector_elems_typed, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* vec_dim1_tagged = builder.CreateLoad(ctx_.taggedValueType(), vec_dim1_ptr);
    llvm::Value* vec_dim1 = tagged_.safeExtractInt64(vec_dim1_tagged);

    llvm::Value* vec_has_dim2 = builder.CreateICmpUGT(
        vector_len, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::BasicBlock* mt_vec_has_dim2 = llvm::BasicBlock::Create(ctx_.context(), "mt_vec_has_dim2", current_func);
    llvm::BasicBlock* mt_vec_1d = llvm::BasicBlock::Create(ctx_.context(), "mt_vec_1d", current_func);
    builder.CreateCondBr(vec_has_dim2, mt_vec_has_dim2, mt_vec_1d);

    builder.SetInsertPoint(mt_vec_1d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* vec_1d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_vec_has_dim2);
    llvm::Value* vec_dim2_ptr = builder.CreateGEP(
        ctx_.taggedValueType(), vector_elems_typed, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* vec_dim2_tagged = builder.CreateLoad(ctx_.taggedValueType(), vec_dim2_ptr);
    llvm::Value* vec_dim2 = tagged_.safeExtractInt64(vec_dim2_tagged);

    llvm::Value* vec_has_dim3 = builder.CreateICmpUGT(
        vector_len, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::BasicBlock* mt_vec_has_dim3 = llvm::BasicBlock::Create(ctx_.context(), "mt_vec_has_dim3", current_func);
    llvm::BasicBlock* mt_vec_2d = llvm::BasicBlock::Create(ctx_.context(), "mt_vec_2d", current_func);
    builder.CreateCondBr(vec_has_dim3, mt_vec_has_dim3, mt_vec_2d);

    builder.SetInsertPoint(mt_vec_2d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* vec_2d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_vec_has_dim3);
    llvm::Value* vec_dim3_ptr = builder.CreateGEP(
        ctx_.taggedValueType(), vector_elems_typed, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* vec_dim3_tagged = builder.CreateLoad(ctx_.taggedValueType(), vec_dim3_ptr);
    llvm::Value* vec_dim3 = tagged_.safeExtractInt64(vec_dim3_tagged);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* vec_3d_exit = builder.GetInsertBlock();

    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), vec_1d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 2), vec_2d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 3), vec_3d_exit);

    d1_phi->addIncoming(vec_dim1, vec_1d_exit);
    d1_phi->addIncoming(vec_dim1, vec_2d_exit);
    d1_phi->addIncoming(vec_dim1, vec_3d_exit);

    d2_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), vec_1d_exit);
    d2_phi->addIncoming(vec_dim2, vec_2d_exit);
    d2_phi->addIncoming(vec_dim2, vec_3d_exit);

    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), vec_1d_exit);
    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), vec_2d_exit);
    d3_phi->addIncoming(vec_dim3, vec_3d_exit);

    // TENSOR SHAPE PATH: #(...) literals are parsed as numeric tensors.
    // Use their element list as the requested output shape.
    builder.SetInsertPoint(tensor_path);
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(shape_arg);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Value* tensor_total_field = builder.CreateStructGEP(ctx_.tensorType(), tensor_ptr, 3);
    llvm::Value* tensor_shape_len = builder.CreateLoad(ctx_.int64Type(), tensor_total_field);
    llvm::Value* tensor_elems_field = builder.CreateStructGEP(ctx_.tensorType(), tensor_ptr, 2);
    llvm::Value* tensor_elems = builder.CreateLoad(ctx_.ptrType(), tensor_elems_field);

    llvm::Value* tensor_dim1_bits = builder.CreateLoad(
        ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), tensor_elems,
            llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    llvm::Value* tensor_dim1_double = builder.CreateBitCast(tensor_dim1_bits, ctx_.doubleType());
    llvm::Value* tensor_dim1 = builder.CreateFPToSI(tensor_dim1_double, ctx_.int64Type());

    llvm::Value* tensor_has_dim2 = builder.CreateICmpUGT(
        tensor_shape_len, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::BasicBlock* mt_tensor_has_dim2 = llvm::BasicBlock::Create(
        ctx_.context(), "mt_tensor_has_dim2", current_func);
    llvm::BasicBlock* mt_tensor_1d = llvm::BasicBlock::Create(
        ctx_.context(), "mt_tensor_1d", current_func);
    builder.CreateCondBr(tensor_has_dim2, mt_tensor_has_dim2, mt_tensor_1d);

    builder.SetInsertPoint(mt_tensor_1d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* tensor_1d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_tensor_has_dim2);
    llvm::Value* tensor_dim2_bits = builder.CreateLoad(
        ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), tensor_elems,
            llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    llvm::Value* tensor_dim2_double = builder.CreateBitCast(tensor_dim2_bits, ctx_.doubleType());
    llvm::Value* tensor_dim2 = builder.CreateFPToSI(tensor_dim2_double, ctx_.int64Type());

    llvm::Value* tensor_has_dim3 = builder.CreateICmpUGT(
        tensor_shape_len, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::BasicBlock* mt_tensor_has_dim3 = llvm::BasicBlock::Create(
        ctx_.context(), "mt_tensor_has_dim3", current_func);
    llvm::BasicBlock* mt_tensor_2d = llvm::BasicBlock::Create(
        ctx_.context(), "mt_tensor_2d", current_func);
    builder.CreateCondBr(tensor_has_dim3, mt_tensor_has_dim3, mt_tensor_2d);

    builder.SetInsertPoint(mt_tensor_2d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* tensor_2d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_tensor_has_dim3);
    llvm::Value* tensor_dim3_bits = builder.CreateLoad(
        ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), tensor_elems,
            llvm::ConstantInt::get(ctx_.int64Type(), 2)));
    llvm::Value* tensor_dim3_double = builder.CreateBitCast(tensor_dim3_bits, ctx_.doubleType());
    llvm::Value* tensor_dim3 = builder.CreateFPToSI(tensor_dim3_double, ctx_.int64Type());
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* tensor_3d_exit = builder.GetInsertBlock();

    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), tensor_1d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 2), tensor_2d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 3), tensor_3d_exit);

    d1_phi->addIncoming(tensor_dim1, tensor_1d_exit);
    d1_phi->addIncoming(tensor_dim1, tensor_2d_exit);
    d1_phi->addIncoming(tensor_dim1, tensor_3d_exit);

    d2_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), tensor_1d_exit);
    d2_phi->addIncoming(tensor_dim2, tensor_2d_exit);
    d2_phi->addIncoming(tensor_dim2, tensor_3d_exit);

    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), tensor_1d_exit);
    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), tensor_2d_exit);
    d3_phi->addIncoming(tensor_dim3, tensor_3d_exit);

    // FINAL MERGE: scalar path or heap-shape path
    builder.SetInsertPoint(final_merge);
    llvm::PHINode* result = builder.CreatePHI(ctx_.ptrType(), 2, "mt_result");
    result->addIncoming(scalar_tensor, scalar_exit);
    result->addIncoming(list_tensor, list_done_exit);

    return tagged_.packHeapPtr(result);
}

llvm::Value* TensorCodegen::zeros(const eshkol_operations_t* op) {
    // zeros: (zeros dim1 dim2 ...) OR (zeros '(dim1 dim2 ...)) - Create tensor filled with zeros
    // Supports both NumPy-style list syntax and separate dimension arguments
    if (op->call_op.num_vars < 1) {
        eshkol_error("zeros requires at least 1 dimension argument");
        return nullptr;
    }

    std::vector<llvm::Value*> dims;

    if (op->call_op.num_vars == 1) {
        // Could be a single dimension OR a list of dimensions
        llvm::Value* dim_arg = codegenAST(&op->call_op.variables[0]);
        if (!dim_arg) return nullptr;

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* list_path = llvm::BasicBlock::Create(ctx_.context(), "zeros_list", current_func);
        llvm::BasicBlock* single_path = llvm::BasicBlock::Create(ctx_.context(), "zeros_single", current_func);
        llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "zeros_merge", current_func);

        // Check if it's a list (CONS_PTR or HEAP_PTR) at runtime
        // IMPORTANT: Only check type if dim_arg is actually a tagged_value struct
        // If it's a raw i64, it's definitely not a list
        if (dim_arg->getType() == ctx_.taggedValueType()) {
            llvm::Value* type_tag = tagged_.getType(dim_arg);
            // Use getBaseType() to properly handle legacy types (CONS_PTR=32)
            // DO NOT use 0x0F mask - 32 & 0x0F = 0 (NULL) which is WRONG!
            llvm::Value* base_type = tagged_.getBaseType(type_tag);
            llvm::Value* is_cons_legacy = ctx_.builder().CreateICmpEQ(base_type,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
            llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(base_type,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
            llvm::Value* is_list = ctx_.builder().CreateOr(is_cons_legacy, is_heap_ptr);
            ctx_.builder().CreateCondBr(is_list, list_path, single_path);
        } else {
            // Raw integer - definitely not a list, go directly to single_path
            ctx_.builder().CreateBr(single_path);
        }

        // LIST PATH: Extract dimensions from the list (up to 4D supported)
        ctx_.builder().SetInsertPoint(list_path);
        llvm::Value* list_ptr_int = tagged_.unpackInt64(dim_arg);
        llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(list_ptr_int, ctx_.ptrType());
        llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
        llvm::Value* is_cdr_flag = llvm::ConstantInt::get(ctx_.int1Type(), 1);

        // Extract first element (car)
        llvm::Value* dim1_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cons_ptr, is_car});

        // Get cdr (rest of list)
        llvm::Value* cdr_ptr_int = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr_flag});

        // Check if cdr is null (single-element list)
        llvm::Value* cdr_is_null = ctx_.builder().CreateICmpEQ(cdr_ptr_int,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* has_dim2 = llvm::BasicBlock::Create(ctx_.context(), "zeros_has_dim2", current_func);
        llvm::BasicBlock* list_1d = llvm::BasicBlock::Create(ctx_.context(), "zeros_list_1d", current_func);

        ctx_.builder().CreateCondBr(cdr_is_null, list_1d, has_dim2);

        // 1D list case: only one dimension
        ctx_.builder().SetInsertPoint(list_1d);
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_1d_exit = ctx_.builder().GetInsertBlock();

        // 2D+ list case: extract second dimension
        ctx_.builder().SetInsertPoint(has_dim2);
        llvm::Value* cdr_cons_ptr = ctx_.builder().CreateIntToPtr(cdr_ptr_int, ctx_.ptrType());
        llvm::Value* dim2_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cdr_cons_ptr, is_car});

        // Check for third dimension
        llvm::Value* cddr_ptr_int = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetPtr(), {cdr_cons_ptr, is_cdr_flag});
        llvm::Value* cddr_is_null = ctx_.builder().CreateICmpEQ(cddr_ptr_int,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* has_dim3 = llvm::BasicBlock::Create(ctx_.context(), "zeros_has_dim3", current_func);
        llvm::BasicBlock* list_2d = llvm::BasicBlock::Create(ctx_.context(), "zeros_list_2d", current_func);

        ctx_.builder().CreateCondBr(cddr_is_null, list_2d, has_dim3);

        // 2D list case
        ctx_.builder().SetInsertPoint(list_2d);
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_2d_exit = ctx_.builder().GetInsertBlock();

        // 3D+ list case
        ctx_.builder().SetInsertPoint(has_dim3);
        llvm::Value* cddr_cons_ptr = ctx_.builder().CreateIntToPtr(cddr_ptr_int, ctx_.ptrType());
        llvm::Value* dim3_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cddr_cons_ptr, is_car});
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_3d_exit = ctx_.builder().GetInsertBlock();

        // SINGLE PATH: Use the value as a single dimension
        ctx_.builder().SetInsertPoint(single_path);
        llvm::Value* single_dim = dim_arg;
        if (single_dim->getType() == ctx_.taggedValueType()) {
            single_dim = tagged_.unpackInt64(single_dim);
        }
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* single_exit = ctx_.builder().GetInsertBlock();

        // MERGE: Use phi nodes to get final dimensions and count
        ctx_.builder().SetInsertPoint(merge_block);

        // num_dims: 1 for single/1d-list, 2 for 2d-list, 3 for 3d-list
        llvm::PHINode* num_dims_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "num_dims");
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 2), list_2d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 3), list_3d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        llvm::PHINode* final_dim1 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim1");
        final_dim1->addIncoming(dim1_from_list, list_1d_exit);
        final_dim1->addIncoming(dim1_from_list, list_2d_exit);
        final_dim1->addIncoming(dim1_from_list, list_3d_exit);
        final_dim1->addIncoming(single_dim, single_exit);

        llvm::PHINode* final_dim2 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim2");
        final_dim2->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        final_dim2->addIncoming(dim2_from_list, list_2d_exit);
        final_dim2->addIncoming(dim2_from_list, list_3d_exit);
        final_dim2->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        llvm::PHINode* final_dim3 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim3");
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_2d_exit);
        final_dim3->addIncoming(dim3_from_list, list_3d_exit);
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        // Build dims vector based on num_dims at runtime
        // For simplicity, we push all dims and let createTensorWithDims handle it
        // But since createTensorWithDims uses compile-time vector size, we need runtime handling

        // Compute total elements: dim1 * dim2 * dim3 (with dim2=1, dim3=1 for lower dims)
        llvm::Value* total = ctx_.builder().CreateMul(final_dim1, final_dim2);
        total = ctx_.builder().CreateMul(total, final_dim3);

        // Use dynamic tensor creation with runtime dimension count
        // For now, create based on num_dims PHI
        llvm::Value* is_1d = ctx_.builder().CreateICmpEQ(num_dims_phi,
            llvm::ConstantInt::get(ctx_.int64Type(), 1));
        llvm::Value* is_2d = ctx_.builder().CreateICmpEQ(num_dims_phi,
            llvm::ConstantInt::get(ctx_.int64Type(), 2));

        llvm::BasicBlock* create_1d = llvm::BasicBlock::Create(ctx_.context(), "create_1d", current_func);
        llvm::BasicBlock* check_2d = llvm::BasicBlock::Create(ctx_.context(), "check_2d", current_func);
        llvm::BasicBlock* create_2d = llvm::BasicBlock::Create(ctx_.context(), "create_2d", current_func);
        llvm::BasicBlock* create_3d = llvm::BasicBlock::Create(ctx_.context(), "create_3d", current_func);
        llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "zeros_done", current_func);

        ctx_.builder().CreateCondBr(is_1d, create_1d, check_2d);

        ctx_.builder().SetInsertPoint(create_1d);
        std::vector<llvm::Value*> dims_1d = {final_dim1};
        llvm::Value* tensor_1d = createTensorWithDims(dims_1d, nullptr, true);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_1d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(check_2d);
        ctx_.builder().CreateCondBr(is_2d, create_2d, create_3d);

        ctx_.builder().SetInsertPoint(create_2d);
        std::vector<llvm::Value*> dims_2d = {final_dim1, final_dim2};
        llvm::Value* tensor_2d = createTensorWithDims(dims_2d, nullptr, true);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_2d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(create_3d);
        std::vector<llvm::Value*> dims_3d = {final_dim1, final_dim2, final_dim3};
        llvm::Value* tensor_3d = createTensorWithDims(dims_3d, nullptr, true);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_3d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(done_block);
        llvm::PHINode* tensor_phi = ctx_.builder().CreatePHI(ctx_.ptrType(), 3, "tensor_ptr");
        tensor_phi->addIncoming(tensor_1d, create_1d_exit);
        tensor_phi->addIncoming(tensor_2d, create_2d_exit);
        tensor_phi->addIncoming(tensor_3d, create_3d_exit);

        // Pack as consolidated HEAP_PTR (subtype in header)
        return tagged_.packHeapPtr(tensor_phi);
    }

    // Multiple explicit dimension arguments: (zeros 2 3 4)
    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
        llvm::Value* dim = codegenAST(&op->call_op.variables[i]);
        if (!dim) return nullptr;
        // Extract int64 from tagged value if needed
        if (dim->getType() == ctx_.taggedValueType()) {
            dim = tagged_.unpackInt64(dim);
        }
        dims.push_back(dim);
    }

    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, true);  // use_memset_zero = true
    if (!tensor_ptr) return nullptr;

    // Pack as consolidated HEAP_PTR (subtype in header)
    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::ones(const eshkol_operations_t* op) {
    // ones: (ones dim1 dim2 ...) OR (ones '(dim1 dim2 ...)) - Create tensor filled with ones
    // Supports both NumPy-style list syntax and separate dimension arguments
    if (op->call_op.num_vars < 1) {
        eshkol_error("ones requires at least 1 dimension argument");
        return nullptr;
    }

    // Create fill value: 1.0 as double, stored as int64 bit pattern
    llvm::Value* one_double = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_bits = ctx_.builder().CreateBitCast(one_double, ctx_.int64Type());

    std::vector<llvm::Value*> dims;

    if (op->call_op.num_vars == 1) {
        // Could be a single dimension OR a list of dimensions
        llvm::Value* dim_arg = codegenAST(&op->call_op.variables[0]);
        if (!dim_arg) return nullptr;

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* list_path = llvm::BasicBlock::Create(ctx_.context(), "ones_list", current_func);
        llvm::BasicBlock* single_path = llvm::BasicBlock::Create(ctx_.context(), "ones_single", current_func);
        llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "ones_merge", current_func);

        // Check if it's a list (CONS_PTR or HEAP_PTR) at runtime
        // IMPORTANT: Only check type if dim_arg is actually a tagged_value struct
        // If it's a raw i64, it's definitely not a list
        if (dim_arg->getType() == ctx_.taggedValueType()) {
            llvm::Value* type_tag = tagged_.getType(dim_arg);
            // Use getBaseType() to properly handle legacy types (CONS_PTR=32)
            // DO NOT use 0x0F mask - 32 & 0x0F = 0 (NULL) which is WRONG!
            llvm::Value* base_type = tagged_.getBaseType(type_tag);
            llvm::Value* is_cons_legacy = ctx_.builder().CreateICmpEQ(base_type,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
            llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(base_type,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
            llvm::Value* is_list = ctx_.builder().CreateOr(is_cons_legacy, is_heap_ptr);
            ctx_.builder().CreateCondBr(is_list, list_path, single_path);
        } else {
            // Raw integer - definitely not a list, go directly to single_path
            ctx_.builder().CreateBr(single_path);
        }

        // LIST PATH: Extract dimensions from the list (up to 4D supported)
        ctx_.builder().SetInsertPoint(list_path);
        llvm::Value* list_ptr_int = tagged_.unpackInt64(dim_arg);
        llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(list_ptr_int, ctx_.ptrType());
        llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
        llvm::Value* is_cdr_flag = llvm::ConstantInt::get(ctx_.int1Type(), 1);

        // Extract first element (car)
        llvm::Value* dim1_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cons_ptr, is_car});

        // Get cdr (rest of list)
        llvm::Value* cdr_ptr_int = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr_flag});

        // Check if cdr is null (single-element list)
        llvm::Value* cdr_is_null = ctx_.builder().CreateICmpEQ(cdr_ptr_int,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* has_dim2 = llvm::BasicBlock::Create(ctx_.context(), "ones_has_dim2", current_func);
        llvm::BasicBlock* list_1d = llvm::BasicBlock::Create(ctx_.context(), "ones_list_1d", current_func);

        ctx_.builder().CreateCondBr(cdr_is_null, list_1d, has_dim2);

        // 1D list case: only one dimension
        ctx_.builder().SetInsertPoint(list_1d);
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_1d_exit = ctx_.builder().GetInsertBlock();

        // 2D+ list case: extract second dimension
        ctx_.builder().SetInsertPoint(has_dim2);
        llvm::Value* cdr_cons_ptr = ctx_.builder().CreateIntToPtr(cdr_ptr_int, ctx_.ptrType());
        llvm::Value* dim2_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cdr_cons_ptr, is_car});

        // Check for third dimension
        llvm::Value* cddr_ptr_int = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetPtr(), {cdr_cons_ptr, is_cdr_flag});
        llvm::Value* cddr_is_null = ctx_.builder().CreateICmpEQ(cddr_ptr_int,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* has_dim3 = llvm::BasicBlock::Create(ctx_.context(), "ones_has_dim3", current_func);
        llvm::BasicBlock* list_2d = llvm::BasicBlock::Create(ctx_.context(), "ones_list_2d", current_func);

        ctx_.builder().CreateCondBr(cddr_is_null, list_2d, has_dim3);

        // 2D list case
        ctx_.builder().SetInsertPoint(list_2d);
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_2d_exit = ctx_.builder().GetInsertBlock();

        // 3D+ list case
        ctx_.builder().SetInsertPoint(has_dim3);
        llvm::Value* cddr_cons_ptr = ctx_.builder().CreateIntToPtr(cddr_ptr_int, ctx_.ptrType());
        llvm::Value* dim3_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cddr_cons_ptr, is_car});
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_3d_exit = ctx_.builder().GetInsertBlock();

        // SINGLE PATH: Use the value as a single dimension
        ctx_.builder().SetInsertPoint(single_path);
        llvm::Value* single_dim = dim_arg;
        if (single_dim->getType() == ctx_.taggedValueType()) {
            single_dim = tagged_.unpackInt64(single_dim);
        }
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* single_exit = ctx_.builder().GetInsertBlock();

        // MERGE: Use phi nodes to get final dimensions and count
        ctx_.builder().SetInsertPoint(merge_block);

        // num_dims: 1 for single/1d-list, 2 for 2d-list, 3 for 3d-list
        llvm::PHINode* num_dims_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "num_dims");
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 2), list_2d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 3), list_3d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        llvm::PHINode* final_dim1 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim1");
        final_dim1->addIncoming(dim1_from_list, list_1d_exit);
        final_dim1->addIncoming(dim1_from_list, list_2d_exit);
        final_dim1->addIncoming(dim1_from_list, list_3d_exit);
        final_dim1->addIncoming(single_dim, single_exit);

        llvm::PHINode* final_dim2 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim2");
        final_dim2->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        final_dim2->addIncoming(dim2_from_list, list_2d_exit);
        final_dim2->addIncoming(dim2_from_list, list_3d_exit);
        final_dim2->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        llvm::PHINode* final_dim3 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim3");
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_2d_exit);
        final_dim3->addIncoming(dim3_from_list, list_3d_exit);
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        // Create based on num_dims PHI
        llvm::Value* is_1d = ctx_.builder().CreateICmpEQ(num_dims_phi,
            llvm::ConstantInt::get(ctx_.int64Type(), 1));
        llvm::Value* is_2d = ctx_.builder().CreateICmpEQ(num_dims_phi,
            llvm::ConstantInt::get(ctx_.int64Type(), 2));

        llvm::BasicBlock* create_1d = llvm::BasicBlock::Create(ctx_.context(), "ones_create_1d", current_func);
        llvm::BasicBlock* check_2d = llvm::BasicBlock::Create(ctx_.context(), "ones_check_2d", current_func);
        llvm::BasicBlock* create_2d = llvm::BasicBlock::Create(ctx_.context(), "ones_create_2d", current_func);
        llvm::BasicBlock* create_3d = llvm::BasicBlock::Create(ctx_.context(), "ones_create_3d", current_func);
        llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "ones_done", current_func);

        ctx_.builder().CreateCondBr(is_1d, create_1d, check_2d);

        ctx_.builder().SetInsertPoint(create_1d);
        std::vector<llvm::Value*> dims_1d = {final_dim1};
        llvm::Value* tensor_1d = createTensorWithDims(dims_1d, one_bits, false);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_1d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(check_2d);
        ctx_.builder().CreateCondBr(is_2d, create_2d, create_3d);

        ctx_.builder().SetInsertPoint(create_2d);
        std::vector<llvm::Value*> dims_2d = {final_dim1, final_dim2};
        llvm::Value* tensor_2d = createTensorWithDims(dims_2d, one_bits, false);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_2d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(create_3d);
        std::vector<llvm::Value*> dims_3d = {final_dim1, final_dim2, final_dim3};
        llvm::Value* tensor_3d = createTensorWithDims(dims_3d, one_bits, false);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_3d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(done_block);
        llvm::PHINode* tensor_phi = ctx_.builder().CreatePHI(ctx_.ptrType(), 3, "tensor_ptr");
        tensor_phi->addIncoming(tensor_1d, create_1d_exit);
        tensor_phi->addIncoming(tensor_2d, create_2d_exit);
        tensor_phi->addIncoming(tensor_3d, create_3d_exit);

        // Pack as consolidated HEAP_PTR
        return tagged_.packHeapPtr(tensor_phi);
    }

    // Multiple explicit dimension arguments: (ones 2 3 4)
    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
        llvm::Value* dim = codegenAST(&op->call_op.variables[i]);
        if (!dim) return nullptr;
        if (dim->getType() == ctx_.taggedValueType()) {
            dim = tagged_.unpackInt64(dim);
        }
        dims.push_back(dim);
    }

    llvm::Value* tensor_ptr = createTensorWithDims(dims, one_bits, false);
    if (!tensor_ptr) return nullptr;

    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::eye(const eshkol_operations_t* op) {
    // eye: (eye n) or (eye rows cols) - Create identity matrix
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("eye requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* rows = codegenAST(&op->call_op.variables[0]);
    if (!rows) return nullptr;
    if (rows->getType() == ctx_.taggedValueType()) {
        rows = tagged_.unpackInt64(rows);
    }

    llvm::Value* cols = rows;  // Default: square matrix
    if (op->call_op.num_vars == 2) {
        cols = codegenAST(&op->call_op.variables[1]);
        if (!cols) return nullptr;
        if (cols->getType() == ctx_.taggedValueType()) {
            cols = tagged_.unpackInt64(cols);
        }
    }

    // Create zero-filled tensor first
    std::vector<llvm::Value*> dims = {rows, cols};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, true);  // Zero fill
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get elements pointer
    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elements_field_ptr);
    llvm::Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.ptrType());

    // Set diagonal to 1.0
    llvm::Value* one_double = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_bits = ctx_.builder().CreateBitCast(one_double, ctx_.int64Type());

    // Loop to set diagonal
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "eye_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "eye_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "eye_exit", current_func);

    llvm::Value* min_dim = ctx_.builder().CreateSelect(
        ctx_.builder().CreateICmpULT(rows, cols), rows, cols);
    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "eye_i");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = ctx_.builder().CreateICmpULT(i, min_dim);
    ctx_.builder().CreateCondBr(cmp, loop_body, loop_exit);

    ctx_.builder().SetInsertPoint(loop_body);
    // Diagonal index: i * cols + i
    llvm::Value* diag_idx = ctx_.builder().CreateMul(i, cols);
    diag_idx = ctx_.builder().CreateAdd(diag_idx, i);
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr, diag_idx);
    ctx_.builder().CreateStore(one_bits, elem_ptr);
    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_exit);

    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::arange(const eshkol_operations_t* op) {
    // arange: (arange n) or (arange start end) or (arange start end step)
    // Works with both integers and floats
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 3) {
        eshkol_error("arange requires 1-3 arguments");
        return nullptr;
    }

    // Helper lambda to extract double from tagged value
    auto extractDouble = [&](llvm::Value* tagged_val) -> llvm::Value* {
        llvm::Value* val_type = tagged_.getType(tagged_val);
        llvm::Value* is_int = ctx_.builder().CreateICmpEQ(val_type,
            llvm::ConstantInt::get(ctx_.int8Type(), 1)); // INT64 = 1

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* int_bb = llvm::BasicBlock::Create(ctx_.context(), "arange_int", current_func);
        llvm::BasicBlock* dbl_bb = llvm::BasicBlock::Create(ctx_.context(), "arange_dbl", current_func);
        llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "arange_merge", current_func);

        ctx_.builder().CreateCondBr(is_int, int_bb, dbl_bb);

        ctx_.builder().SetInsertPoint(int_bb);
        llvm::Value* int_val = tagged_.unpackInt64(tagged_val);
        llvm::Value* int_as_dbl = ctx_.builder().CreateSIToFP(int_val, ctx_.doubleType());
        ctx_.builder().CreateBr(merge_bb);
        llvm::BasicBlock* int_bb_end = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(dbl_bb);
        llvm::Value* dbl_val = tagged_.unpackDouble(tagged_val);
        ctx_.builder().CreateBr(merge_bb);
        llvm::BasicBlock* dbl_bb_end = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(merge_bb);
        llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.doubleType(), 2);
        result->addIncoming(int_as_dbl, int_bb_end);
        result->addIncoming(dbl_val, dbl_bb_end);
        return result;
    };

    llvm::Value* start_dbl;
    llvm::Value* end_dbl;
    llvm::Value* step_dbl;

    if (op->call_op.num_vars == 1) {
        // (arange n) -> 0 to n-1
        start_dbl = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        step_dbl = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

        llvm::Value* end_tagged = codegenAST(&op->call_op.variables[0]);
        if (!end_tagged) return nullptr;
        if (end_tagged->getType() == ctx_.taggedValueType()) {
            end_dbl = extractDouble(end_tagged);
        } else if (end_tagged->getType() == ctx_.int64Type()) {
            end_dbl = ctx_.builder().CreateSIToFP(end_tagged, ctx_.doubleType());
        } else {
            end_dbl = end_tagged;
        }
    } else {
        // (arange start end) or (arange start end step)
        llvm::Value* start_tagged = codegenAST(&op->call_op.variables[0]);
        llvm::Value* end_tagged = codegenAST(&op->call_op.variables[1]);
        if (!start_tagged || !end_tagged) return nullptr;

        if (start_tagged->getType() == ctx_.taggedValueType()) {
            start_dbl = extractDouble(start_tagged);
        } else if (start_tagged->getType() == ctx_.int64Type()) {
            start_dbl = ctx_.builder().CreateSIToFP(start_tagged, ctx_.doubleType());
        } else {
            start_dbl = start_tagged;
        }

        if (end_tagged->getType() == ctx_.taggedValueType()) {
            end_dbl = extractDouble(end_tagged);
        } else if (end_tagged->getType() == ctx_.int64Type()) {
            end_dbl = ctx_.builder().CreateSIToFP(end_tagged, ctx_.doubleType());
        } else {
            end_dbl = end_tagged;
        }

        if (op->call_op.num_vars == 3) {
            llvm::Value* step_tagged = codegenAST(&op->call_op.variables[2]);
            if (!step_tagged) return nullptr;
            if (step_tagged->getType() == ctx_.taggedValueType()) {
                step_dbl = extractDouble(step_tagged);
            } else if (step_tagged->getType() == ctx_.int64Type()) {
                step_dbl = ctx_.builder().CreateSIToFP(step_tagged, ctx_.doubleType());
            } else {
                step_dbl = step_tagged;
            }
        } else {
            step_dbl = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        }
    }

    // Calculate number of elements: ceil((end - start) / step) to handle FP rounding
    llvm::Value* range_dbl = ctx_.builder().CreateFSub(end_dbl, start_dbl);
    llvm::Value* num_elements_dbl = ctx_.builder().CreateFDiv(range_dbl, step_dbl);
    // Use ceil to handle floating-point rounding errors (e.g., 11.9999... -> 12)
    llvm::Function* ceil_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::ceil, {ctx_.doubleType()});
    llvm::Value* num_elements_ceil = ctx_.builder().CreateCall(ceil_func, {num_elements_dbl});
    llvm::Value* num_elements = ctx_.builder().CreateFPToSI(num_elements_ceil, ctx_.int64Type());

    // Create tensor
    std::vector<llvm::Value*> dims = {num_elements};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, false);
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get elements pointer
    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elements_field_ptr);
    llvm::Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.ptrType());

    // Fill with range values
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "arange_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "arange_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "arange_exit", current_func);

    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "arange_i");
    llvm::Value* current_val = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "arange_val");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    ctx_.builder().CreateStore(start_dbl, current_val);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = ctx_.builder().CreateICmpULT(i, num_elements);
    ctx_.builder().CreateCondBr(cmp, loop_body, loop_exit);

    ctx_.builder().SetInsertPoint(loop_body);
    llvm::Value* val = ctx_.builder().CreateLoad(ctx_.doubleType(), current_val);
    // Store as double bit pattern for consistency
    llvm::Value* val_bits = ctx_.builder().CreateBitCast(val, ctx_.int64Type());
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr, i);
    ctx_.builder().CreateStore(val_bits, elem_ptr);

    llvm::Value* next_val = ctx_.builder().CreateFAdd(val, step_dbl);
    ctx_.builder().CreateStore(next_val, current_val);
    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_exit);

    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::linspace(const eshkol_operations_t* op) {
    // linspace: (linspace start end num) - num evenly spaced values from start to end
    if (op->call_op.num_vars != 3) {
        eshkol_error("linspace requires exactly 3 arguments: start, end, num");
        return nullptr;
    }

    llvm::Value* start = codegenAST(&op->call_op.variables[0]);
    llvm::Value* end = codegenAST(&op->call_op.variables[1]);
    llvm::Value* num = codegenAST(&op->call_op.variables[2]);
    if (!start || !end || !num) return nullptr;

    // Extract values - convert to double for computation
    if (start->getType() == ctx_.taggedValueType()) {
        start = tagged_.unpackDouble(start);
    } else if (start->getType()->isIntegerTy(64)) {
        start = ctx_.builder().CreateSIToFP(start, ctx_.doubleType());
    }
    if (end->getType() == ctx_.taggedValueType()) {
        end = tagged_.unpackDouble(end);
    } else if (end->getType()->isIntegerTy(64)) {
        end = ctx_.builder().CreateSIToFP(end, ctx_.doubleType());
    }
    if (num->getType() == ctx_.taggedValueType()) {
        num = tagged_.unpackInt64(num);
    }

    // Create tensor
    std::vector<llvm::Value*> dims = {num};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, false);
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get elements pointer
    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elements_field_ptr);
    llvm::Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.ptrType());

    // Calculate step: (end - start) / (num - 1)
    llvm::Value* range = ctx_.builder().CreateFSub(end, start);
    llvm::Value* num_minus_1 = ctx_.builder().CreateSub(num, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* num_minus_1_fp = ctx_.builder().CreateSIToFP(num_minus_1, ctx_.doubleType());
    llvm::Value* step = ctx_.builder().CreateFDiv(range, num_minus_1_fp);

    // Fill with linspace values
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "linspace_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "linspace_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "linspace_exit", current_func);

    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "ls_i");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = ctx_.builder().CreateICmpULT(i, num);
    ctx_.builder().CreateCondBr(cmp, loop_body, loop_exit);

    ctx_.builder().SetInsertPoint(loop_body);
    // val = start + i * step
    llvm::Value* i_fp = ctx_.builder().CreateSIToFP(i, ctx_.doubleType());
    llvm::Value* offset = ctx_.builder().CreateFMul(i_fp, step);
    llvm::Value* val = ctx_.builder().CreateFAdd(start, offset);
    llvm::Value* val_bits = ctx_.builder().CreateBitCast(val, ctx_.int64Type());
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr, i);
    ctx_.builder().CreateStore(val_bits, elem_ptr);

    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_exit);

    return tagged_.packHeapPtr(tensor_ptr);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
