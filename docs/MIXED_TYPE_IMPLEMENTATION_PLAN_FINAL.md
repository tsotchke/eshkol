# Mixed Type Lists Implementation Plan - Final Strategy

## Overview

This document outlines the complete, step-by-step implementation plan to fix mixed type lists in Eshkol and migrate from 16-byte untyped cons cells to 24-byte tagged cons cells.

## Problem Summary

1. **CRITICAL (Segfault):** Compound car/cdr operations use wrong struct layout
2. **HIGH (Type Safety):** Missing C helper for NULL type storage  
3. **MEDIUM (Completeness):** Most list operations still use old untyped cells

## Recommended Strategy: Phased Migration

### Phase 1: Critical Fixes (Immediate - Stop Segfault)

**Goal:** Make basic mixed type lists work without segfaults

#### Step 1.1: Add Missing NULL Helper Function

**File: [`lib/core/arena_memory.h`](lib/core/arena_memory.h:120)**

Add function declaration after `arena_tagged_cons_set_ptr`:
```c
void arena_tagged_cons_set_null(arena_tagged_cons_cell_t* cell, bool is_cdr);
```

**File: [`lib/core/arena_memory.cpp`](lib/core/arena_memory.cpp:482)**

Add implementation after `arena_tagged_cons_set_ptr`:
```c
void arena_tagged_cons_set_null(arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot set null on null tagged cons cell");
        return;
    }
    
    if (is_cdr) {
        cell->cdr_type = ESHKOL_VALUE_NULL;
        cell->cdr_data.raw_val = 0;
    } else {
        cell->car_type = ESHKOL_VALUE_NULL;
        cell->car_data.raw_val = 0;
    }
}
```

#### Step 1.2: Add LLVM Declaration for NULL Helper

**File: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:110)**

Add to class member declarations:
```cpp
Function* arena_tagged_cons_set_null_func;
```

**File: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:617)**

Add function declaration after `arena_tagged_cons_get_type_func`:
```cpp
// arena_tagged_cons_set_null function: void arena_tagged_cons_set_null(arena_tagged_cons_cell_t* cell, bool is_cdr)
std::vector<Type*> arena_tagged_cons_set_null_args;
arena_tagged_cons_set_null_args.push_back(PointerType::getUnqual(*context)); // arena_tagged_cons_cell_t* cell
arena_tagged_cons_set_null_args.push_back(Type::getInt1Ty(*context)); // bool is_cdr

FunctionType* arena_tagged_cons_set_null_type = FunctionType::get(
    Type::getVoidTy(*context),
    arena_tagged_cons_set_null_args,
    false
);

arena_tagged_cons_set_null_func = Function::Create(
    arena_tagged_cons_set_null_type,
    Function::ExternalLinkage,
    "arena_tagged_cons_set_null",
    module.get()
);

function_table["arena_tagged_cons_set_null"] = arena_tagged_cons_set_null_func;
```

#### Step 1.3: Fix codegenTaggedArenaConsCell to Use Proper Helpers

**File: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:970)**

Replace the car storage logic (lines 970-987) with:
```cpp
// Set car value using appropriate C helper function
if (car_val.isDouble()) {
    builder->CreateCall(arena_tagged_cons_set_double_func,
        {cons_ptr, is_car, car_val.llvm_value, car_type_tag});
} else if (car_val.isInt64()) {
    builder->CreateCall(arena_tagged_cons_set_int64_func,
        {cons_ptr, is_car, car_val.llvm_value, car_type_tag});
} else if (car_val.isNull()) {
    builder->CreateCall(arena_tagged_cons_set_null_func, {cons_ptr, is_car});
} else {
    // CONS_PTR type - use set_ptr helper
    Value* car_as_uint64 = car_val.llvm_value;
    if (car_val.llvm_value->getType()->isPointerTy()) {
        car_as_uint64 = builder->CreatePtrToInt(car_val.llvm_value, Type::getInt64Ty(*context));
    }
    builder->CreateCall(arena_tagged_cons_set_ptr_func,
        {cons_ptr, is_car, car_as_uint64, car_type_tag});
}
```

**File: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:998)**

Replace the cdr storage logic (lines 998-1015) with:
```cpp
// Set cdr value using appropriate C helper function
if (cdr_val.isDouble()) {
    builder->CreateCall(arena_tagged_cons_set_double_func,
        {cons_ptr, is_cdr, cdr_val.llvm_value, cdr_type_tag});
} else if (cdr_val.isInt64()) {
    builder->CreateCall(arena_tagged_cons_set_int64_func,
        {cons_ptr, is_cdr, cdr_val.llvm_value, cdr_type_tag});
} else if (cdr_val.isNull()) {
    builder->CreateCall(arena_tagged_cons_set_null_func, {cons_ptr, is_cdr});
} else {
    // CONS_PTR type - use set_ptr helper
    Value* cdr_as_uint64 = cdr_val.llvm_value;
    if (cdr_val.llvm_value->getType()->isPointerTy()) {
        cdr_as_uint64 = builder->CreatePtrToInt(cdr_val.llvm_value, Type::getInt64Ty(*context));
    }
    builder->CreateCall(arena_tagged_cons_set_ptr_func,
        {cons_ptr, is_cdr, cdr_as_uint64, cdr_type_tag});
}
```

#### Step 1.4: Fix Compound Car/Cdr to Use Tagged Cells (CRITICAL)

**Approach:** Completely rewrite to delegate to existing `codegenCar()` and `codegenCdr()`

**File: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:4294)**

**Current Implementation:** Lines 4294-4361 use old 16-byte struct

**New Implementation:**
```cpp
Value* codegenCompoundCarCdr(const eshkol_operations_t* op, const std::string& pattern) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("compound car/cdr requires exactly 1 argument");
        return nullptr;
    }
    
    // STRATEGY: Build nested operations and use existing car/cdr implementations
    // Example: cadr(list) = car(cdr(list))
    // We apply operations RIGHT-TO-LEFT: for "ad", apply 'd' then 'a'
    
    Value* current_result = codegenAST(&op->call_op.variables[0]);
    if (!current_result) return nullptr;
    
    // Apply each operation in reverse order (innermost first)
    for (int i = pattern.length() - 1; i >= 0; i--) {
        char c = pattern[i];
        
        // Create a synthetic call_op for car or cdr
        eshkol_operations_t synthetic_op;
        synthetic_op.op = ESHKOL_CALL_OP;
        synthetic_op.call_op.num_vars = 1;
        
        // Create a synthetic AST node to hold current_result
        eshkol_ast_t synthetic_arg;
        
        // Check if current_result is a tagged_value struct
        if (current_result->getType() == tagged_value_type) {
            // It's already a tagged value from previous car/cdr
            // We need to extract the pointer from it
            synthetic_arg.type = ESHKOL_INT64;
            synthetic_arg.int64_val = 0; // Placeholder, won't be used
        } else {
            // It's a raw int64 (pointer)
            synthetic_arg.type = ESHKOL_INT64;
            synthetic_arg.int64_val = 0; // Placeholder
        }
        
        synthetic_op.call_op.variables = &synthetic_arg;
        
        // Save current builder state
        IRBuilderBase::InsertPoint save_point = builder->saveIP();
        
        // Generate the operation using existing implementations
        if (c == 'a') {
            // Call existing codegenCar which handles tagged cells correctly
            // But we need to work with the Value* we have, not generate from AST
            
            // Actually, let's inline the logic from codegenCar/codegenCdr
            // but operating on current_result instead of generating from AST
            
            // This is getting complex. Better approach: generate actual car/cdr calls
        }
        
        // Restore builder state
        builder->restoreIP(save_point);
    }
    
    return current_result;
}
```

**Better Approach - Inline Tagged Cell Access:**

The compound operations need to be completely rewritten to work with tagged cons cells directly:

```cpp
Value* codegenCompoundCarCdr(const eshkol_operations_t* op, const std::string& pattern) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("compound car/cdr requires exactly 1 argument");
        return nullptr;
    }
    
    Value* current = codegenAST(&op->call_op.variables[0]);
    if (!current) return nullptr;
    
    // If current is a tagged_value struct, extract the pointer
    Value* current_ptr_int = current;
    if (current->getType() == tagged_value_type) {
        // Extract pointer from tagged value using unpackPtrFromTaggedValue
        current_ptr_int = unpackPtrFromTaggedValue(current);
        // Convert back to int64
        current_ptr_int = builder->CreatePtrToInt(current_ptr_int, Type::getInt64Ty(*context));
    }
    
    Function* current_func = builder->GetInsertBlock()->getParent();
    
    // Apply operations in REVERSE order (right-to-left)
    for (int i = pattern.length() - 1; i >= 0; i--) {
        char c = pattern[i];
        
        // NULL CHECK for safety
        Value* is_null = builder->CreateICmpEQ(current_ptr_int, 
            ConstantInt::get(Type::getInt64Ty(*context), 0));
        
        BasicBlock* null_block = BasicBlock::Create(*context,
            std::string("compound_") + c + "_null", current_func);
        BasicBlock* valid_block = BasicBlock::Create(*context,
            std::string("compound_") + c + "_valid", current_func);
        BasicBlock* continue_block = BasicBlock::Create(*context,
            std::string("compound_") + c + "_continue", current_func);
        
        builder->CreateCondBr(is_null, null_block, valid_block);
        
        // Null block: return null tagged value
        builder->SetInsertPoint(null_block);
        Value* null_result = packInt64ToTaggedValue(
            ConstantInt::get(Type::getInt64Ty(*context), 0), true);
        builder->CreateBr(continue_block);
        
        // Valid block: perform car or cdr operation
        builder->SetInsertPoint(valid_block);
        
        Value* cons_ptr = builder->CreateIntToPtr(current_ptr_int, builder->getPtrTy());
        Value* operation_result = nullptr;
        
        if (c == 'a') {
            // CAR operation using tagged cons cell helpers
            Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
            Value* car_type = builder->CreateCall(arena_tagged_cons_get_type_func, 
                {cons_ptr, is_car});
            
            // Mask to get base type
            Value* car_base_type = builder->CreateAnd(car_type,
                ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
            
            // Branch on type
            Value* is_double = builder->CreateICmpEQ(car_base_type,
                ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
            
            BasicBlock* car_double_block = BasicBlock::Create(*context, "car_double", current_func);
            BasicBlock* car_int_block = BasicBlock::Create(*context, "car_int", current_func);
            BasicBlock* car_merge = BasicBlock::Create(*context, "car_merge", current_func);
            
            builder->CreateCondBr(is_double, car_double_block, car_int_block);
            
            // Extract double
            builder->SetInsertPoint(car_double_block);
            Value* car_double = builder->CreateCall(arena_tagged_cons_get_double_func, 
                {cons_ptr, is_car});
            Value* tagged_double = packDoubleToTaggedValue(car_double);
            builder->CreateBr(car_merge);
            
            // Extract int64
            builder->SetInsertPoint(car_int_block);
            Value* car_int = builder->CreateCall(arena_tagged_cons_get_int64_func,
                {cons_ptr, is_car});
            Value* tagged_int = packInt64ToTaggedValue(car_int, true);
            builder->CreateBr(car_merge);
            
            // Merge results
            builder->SetInsertPoint(car_merge);
            PHINode* car_phi = builder->CreatePHI(tagged_value_type, 2);
            car_phi->addIncoming(tagged_double, car_double_block);
            car_phi->addIncoming(tagged_int, car_int_block);
            operation_result = car_phi;
            
        } else if (c == 'd') {
            // CDR operation using tagged cons cell helpers
            Value* is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);
            Value* cdr_type = builder->CreateCall(arena_tagged_cons_get_type_func,
                {cons_ptr, is_cdr});
            
            // Mask to get base type
            Value* cdr_base_type = builder->CreateAnd(cdr_type,
                ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
            
            // Branch on type
            Value* is_double = builder->CreateICmpEQ(cdr_base_type,
                ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
            
            BasicBlock* cdr_double_block = BasicBlock::Create(*context, "cdr_double", current_func);
            BasicBlock* cdr_int_block = BasicBlock::Create(*context, "cdr_int", current_func);
            BasicBlock* cdr_merge = BasicBlock::Create(*context, "cdr_merge", current_func);
            
            builder->CreateCondBr(is_double, cdr_double_block, cdr_int_block);
            
            // Extract double
            builder->SetInsertPoint(cdr_double_block);
            Value* cdr_double = builder->CreateCall(arena_tagged_cons_get_double_func,
                {cons_ptr, is_cdr});
            Value* tagged_double_cdr = packDoubleToTaggedValue(cdr_double);
            builder->CreateBr(cdr_merge);
            
            // Extract int64/ptr
            builder->SetInsertPoint(cdr_int_block);
            Value* cdr_int = builder->CreateCall(arena_tagged_cons_get_int64_func,
                {cons_ptr, is_cdr});
            Value* tagged_int_cdr = packInt64ToTaggedValue(cdr_int, true);
            builder->CreateBr(cdr_merge);
            
            // Merge results
            builder->SetInsertPoint(cdr_merge);
            PHINode* cdr_phi = builder->CreatePHI(tagged_value_type, 2);
            cdr_phi->addIncoming(tagged_double_cdr, cdr_double_block);
            cdr_phi->addIncoming(tagged_int_cdr, cdr_int_block);
            operation_result = cdr_phi;
        }
        
        builder->CreateBr(continue_block);
        
        // Continue block: merge null and valid results
        builder->SetInsertPoint(continue_block);
        PHINode* result_phi = builder->CreatePHI(tagged_value_type, 2);
        result_phi->addIncoming(null_result, null_block);
        result_phi->addIncoming(operation_result, car_merge.isValid() ? car_merge : cdr_merge);
        
        // For next iteration, extract the pointer value
        // If we have more operations to apply, we need the pointer
        if (i > 0) {
            current_ptr_int = unpackInt64FromTaggedValue(result_phi);
        } else {
            // Last operation, return the tagged value
            current = result_phi;
        }
    }
    
    return current;
}
```

**SIMPLER APPROACH:** Just call existing car/cdr recursively

```cpp
Value* codegenCompoundCarCdr(const eshkol_operations_t* op, const std::string& pattern) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("compound car/cdr requires exactly 1 argument");
        return nullptr;
    }
    
    // Build nested car/cdr operations
    // For "ad" (cadr): car(cdr(list))
    // Apply in reverse: cdr first, then car
    
    eshkol_ast_t current_ast = op->call_op.variables[0];
    
    for (int i = pattern.length() - 1; i >= 0; i--) {
        char c = pattern[i];
        
        // Create synthetic operation for this car/cdr
        eshkol_operations_t synthetic_op;
        synthetic_op.op = ESHKOL_CALL_OP;
        synthetic_op.call_op.num_vars = 1;
        synthetic_op.call_op.variables = &current_ast;
        
        // Create function AST node for car or cdr
        eshkol_ast_t func_ast;
        func_ast.type = ESHKOL_VAR;
        func_ast.variable.id = (c == 'a') ? "car" : "cdr";
        func_ast.variable.data = nullptr;
        synthetic_op.call_op.func = &func_ast;
        
        // Generate the operation
        Value* result = (c == 'a') ? codegenCar(&synthetic_op) : codegenCdr(&synthetic_op);
        if (!result) return nullptr;
        
        // For next iteration, wrap result back into AST
        // This is getting complex because we need to convert Value* back to AST
    }
}
```

**BEST APPROACH:** Create helper operations that work with Value* directly

Actually, the issue is that we can't easily build AST nodes from LLVM Values. The cleanest solution is to **inline the logic** from `codegenCar` and `codegenCdr` but work with Value* throughout.

Let me provide the complete replacement for `codegenCompoundCarCdr`:

```cpp
Value* codegenCompoundCarCdr(const eshkol_operations_t* op, const std::string& pattern) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("compound car/cdr requires exactly 1 argument");
        return nullptr;
    }
    
    Value* current = codegenAST(&op->call_op.variables[0]);
    if (!current) return nullptr;
    
    Function* current_func = builder->GetInsertBlock()->getParent();
    
    // Apply each operation in reverse order (right-to-left)
    // For cadr: apply 'd' (cdr) first, then 'a' (car)
    for (int i = pattern.length() - 1; i >= 0; i--) {
        char c = pattern[i];
        
        // Extract int64 pointer from tagged value if needed
        Value* ptr_int = current;
        if (current->getType() == tagged_value_type) {
            ptr_int = unpackInt64FromTaggedValue(current);
        }
        
        // NULL CHECK
        Value* is_null = builder->CreateICmpEQ(ptr_int, 
            ConstantInt::get(Type::getInt64Ty(*context), 0));
        
        BasicBlock* null_block = BasicBlock::Create(*context, 
            "compound_null", current_func);
        BasicBlock* valid_block = BasicBlock::Create(*context,
            "compound_valid", current_func);
        BasicBlock* continue_block = BasicBlock::Create(*context,
            "compound_continue", current_func);
        
        builder->CreateCondBr(is_null, null_block, valid_block);
        
        // NULL: return null tagged value
        builder->SetInsertPoint(null_block);
        Value* null_tagged = packInt64ToTaggedValue(
            ConstantInt::get(Type::getInt64Ty(*context), 0), true);
        builder->CreateBr(continue_block);
        
        // VALID: extract car or cdr
        builder->SetInsertPoint(valid_block);
        Value* cons_ptr = builder->CreateIntToPtr(ptr_int, builder->getPtrTy());
        
        Value* is_car_op = ConstantInt::get(Type::getInt1Ty(*context), (c == 'a') ? 0 : 1);
        Value* field_type = builder->CreateCall(arena_tagged_cons_get_type_func,
            {cons_ptr, is_car_op});
        
        // Mask to get base type
        Value* base_type = builder->CreateAnd(field_type,
            ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
        
        // Check if double
        Value* is_double = builder->CreateICmpEQ(base_type,
            ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
        
        BasicBlock* double_block = BasicBlock::Create(*context, "extract_double", current_func);
        BasicBlock* int_block = BasicBlock::Create(*context, "extract_int", current_func);
        BasicBlock* merge_block = BasicBlock::Create(*context, "merge_extract", current_func);
        
        builder->CreateCondBr(is_double, double_block, int_block);
        
        // Extract double
        builder->SetInsertPoint(double_block);
        Value* double_val = builder->CreateCall(
            (c == 'a') ? arena_tagged_cons_get_double_func : arena_tagged_cons_get_double_func,
            {cons_ptr, is_car_op});
        Value* tagged_double = packDoubleToTaggedValue(double_val);
        builder->CreateBr(merge_block);
        
        // Extract int64
        builder->SetInsertPoint(int_block);
        Value* int_val = builder->CreateCall(
            arena_tagged_cons_get_int64_func, {cons_ptr, is_car_op});
        Value* tagged_int = packInt64ToTaggedValue(int_val, true);
        builder->CreateBr(merge_block);
        
        // Merge
        builder->SetInsertPoint(merge_block);
        PHINode* extract_phi = builder->CreatePHI(tagged_value_type, 2);
        extract_phi->addIncoming(tagged_double, double_block);
        extract_phi->addIncoming(tagged_int, int_block);
        builder->CreateBr(continue_block);
        
        // Continue: merge null and valid results
        builder->SetInsertPoint(continue_block);
        PHINode* result_phi = builder->CreatePHI(tagged_value_type, 2);
        result_phi->addIncoming(null_tagged, null_block);
        result_phi->addIncoming(extract_phi, merge_block);
        
        current = result_phi;
    }
    
    return current;
}
```

**Testing:** After this fix, `(cadr mixed-list)` should work correctly.

### Phase 2: Complete Migration (Follow-up)

#### Step 2.1: Update All List Utility Functions

All functions currently using old cons cells need migration:

**Files to update:** [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)

- `codegenLength()` (lines 4364-4414)
- `codegenIterativeAppend()` (lines 4443-4559)
- `codegenReverse()` (lines 4562-4618)
- `codegenListRef()` (lines 4621-4699)
- `codegenListTail()` (lines 4702-4756)
- `codegenSetCar()` (lines 4759-4777)
- `codegenSetCdr()` (lines 4780-4798)
- `codegenMapSingleList()` (lines 4943-5034)
- `codegenMapMultiList()` (lines 5037-5154)
- And all other list operations...

**Migration Pattern for Each:**
```cpp
// OLD:
StructType* arena_cons_type = StructType::get(Type::getInt64Ty(*context), 
                                               Type::getInt64Ty(*context));

// NEW: Remove - don't access struct directly
// Instead use C helper functions:
// - arena_tagged_cons_get_int64/double
// - arena_tagged_cons_set_int64/double/ptr/null
// - arena_tagged_cons_get_type
```

#### Step 2.2: Remove Old Untagged Cons Code

**Remove or deprecate:**
- `codegenArenaConsCell()` function (lines 919-945)
- `arena_allocate_cons_cell_func` declaration
- Old 16-byte struct definitions

#### Step 2.3: Update TypedValue to Handle CONS_PTR

**File:** [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:62)

Add method:
```cpp
struct TypedValue {
    ...
    bool isConsPtr() const { return type == ESHKOL_VALUE_CONS_PTR; }
};
```

### Phase 3: Optimization & Future-Proofing

#### Step 3.1: Performance Optimization

Consider caching struct type definitions:
```cpp
// Cache the 24-byte tagged cons struct type
StructType* tagged_cons_cell_type;  // Member variable

// Initialize once in constructor
std::vector<Type*> tagged_cons_fields;
tagged_cons_fields.push_back(Type::getInt8Ty(*context));   // car_type
tagged_cons_fields.push_back(Type::getInt8Ty(*context));   // cdr_type
tagged_cons_fields.push_back(Type::getInt16Ty(*context));  // flags
tagged_cons_fields.push_back(Type::getInt64Ty(*context));  // car_data union
tagged_cons_fields.push_back(Type::getInt64Ty(*context));  // cdr_data union
tagged_cons_cell_type = StructType::create(*context, tagged_cons_fields, 
                                           "arena_tagged_cons_cell");
```

#### Step 3.2: Add More Type-Safe Helpers

For future scientific types:
```c
void arena_tagged_cons_set_complex(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                   double real, double imag, uint8_t type);
void arena_tagged_cons_set_rational(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                    int64_t numerator, int64_t denominator, uint8_t type);
```

## Detailed Implementation Order

### Session 1: Critical Fixes (Current)

1. ✅ Diagnostic analysis (THIS DOCUMENT)
2. ⏳ Add `arena_tagged_cons_set_null` to C layer
3. ⏳ Add LLVM declaration for new function
4. ⏳ Update `codegenTaggedArenaConsCell` to use all proper helpers
5. ⏳ Completely rewrite `codegenCompoundCarCdr` with tagged cells
6. ✅ Build and test: verify no segfaults

### Session 2: Verification & Migration Planning

7. Run comprehensive tests on mixed type lists
8. Document all functions still using old cells
9. Create migration checklist for each function

### Session 3+: Complete Migration

10. Migrate utility functions (length, append, reverse, etc.)
11. Migrate higher-order functions (map, filter, fold)
12. Remove all old untagged cons cell code
13. Performance testing and optimization

## Testing Strategy

### Test 1: Basic Operations (Current Test)
```scheme
(define mixed-list (list 1 2.5 3 4.75 5))
(display (car mixed-list))   ; Should print: 1
(display (cadr mixed-list))  ; Should print: 2.5 (CURRENTLY SEGFAULTS)
(display (caddr mixed-list)) ; Should print: 3
```

### Test 2: Deep Nesting
```scheme
(define nested (list (list 1 2) (list 3.5 4.5)))
(display (caar nested))      ; Should print: 1
(display (cadar nested))     ; Should print: 3.5
```

### Test 3: Mixed Arithmetic
```scheme
(define int-val 10)
(define double-val 2.5)
(display (+ int-val double-val))  ; Should print: 12.5
```

### Test 4: Higher-Order Functions
```scheme
(define nums (list 1 2 3))
(display (map (lambda (x) (* x 2)) nums))  ; Should print: (2 4 6)
```

## Type System Evolution Path

### Current Implementation
- NULL (0), INT64 (1), DOUBLE (2), CONS_PTR (3)
- 4-bit type field allows 16 types total
- 8-bit flags for exactness tracking

### Future Scientific Types
- COMPLEX (4): Complex numbers
- RATIONAL (5): Exact rational numbers
- BIGINT (6): Arbitrary precision integers
- QUATERNION (7): Quaternions for 3D graphics
- SYMBOLIC (8): Symbolic math expressions
- INTERVAL (9): Interval arithmetic
- Types 10-15: Reserved for HoTT constructs

### HoTT Integration
- Type tags map to universe levels
- Exactness flags track proof obligations
- Reserved field stores coherence data

## Code Review Checklist

Before switching to Code mode, verify:

- [ ] NULL helper function signature correct
- [ ] LLVM function declaration matches C signature
- [ ] Compound car/cdr completely rewritten
- [ ] All type checks use mask (type & 0x0F)
- [ ] No remaining references to old 16-byte struct
- [ ] Error messages are clear and actionable
- [ ] Debug logging shows correct types

## Success Criteria

1. No type validation errors during list creation
2. No segmentation faults on compound car/cdr operations
3. Mixed type lists (integers + doubles) work correctly
4. All tests in `tests/mixed_type_lists_basic_test.esk` pass
5. Type information preserved through all operations

## Risk Assessment

**Low Risk:**
- Adding NULL helper function (isolated change)
- Using existing set_ptr function (already tested)

**Medium Risk:**
- Rewriting compound car/cdr (complex logic but well-understood)

**High Risk:**
- Migrating all list operations at once (should be phased)

## Timeline Estimate

- **Session 1 (Critical Fixes):** 1-2 hours
  - Add NULL helper: 15 minutes
  - Fix codegenTaggedArenaConsCell: 15 minutes
  - Rewrite codegenCompoundCarCdr: 45 minutes
  - Testing and debugging: 30 minutes

- **Session 2 (Verification):** 1 hour
  - Comprehensive testing
  - Documentation updates

- **Session 3+ (Full Migration):** 3-4 hours
  - Migrate ~20 list operations
  - Remove old code
  - Final testing

**Total Estimated Time:** 5-7 hours for complete implementation

## Next Steps

1. **Review this plan with user**
2. **Get approval for recommended approach**
3. **Switch to Code mode**
4. **Implement Phase 1 critical fixes**
5. **Test and verify**
6. **Plan Phase 2 migration**

---

**Status:** Ready for implementation  
**Priority:** Critical (segfault blocking basic functionality)  
**Complexity:** Medium (well-understood problem, clear solution)