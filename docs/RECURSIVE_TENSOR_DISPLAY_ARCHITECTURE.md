# Recursive Tensor Display Architecture

**Date**: 2025-11-27  
**Purpose**: Fix Jacobian/matrix flattening with dimension-agnostic recursive display

## Problem Statement

Jacobian matrices are stored correctly as 2D tensors (`num_dimensions=2`) but displayed as flat vectors:
- **Current**: `#(3 2 4 0)` for 2×2 matrix
- **Required**: `#((3 2) (4 0))` showing nested structure
- **Constraint**: Must work for ANY dimensionality (1D, 2D, 3D, ..., ND)

## Core Principle

**NO hardcoded loops or dimension checks.** Use recursion with stride calculation.

## Recursive Display Algorithm

### Signature
```cpp
void displayTensorRecursive(
    double* elements,     // Flat array of element values
    uint64_t* dims,       // Dimension sizes array [d0, d1, ..., dn-1]
    size_t num_dims,      // Number of dimensions
    size_t current_dim,   // Current dimension being processed (0-indexed)
    size_t offset         // Current offset in elements array
)
```

### Algorithm

```cpp
// BASE CASE: Innermost dimension (current_dim == num_dims - 1)
if (current_dim == num_dims - 1) {
    printf("(");
    for (i = 0; i < dims[current_dim]; i++) {
        if (i > 0) printf(" ");
        printf("%g", elements[offset + i]);
    }
    printf(")");
    return;
}

// RECURSIVE CASE: Outer dimensions
printf("(");
size_t stride = 1;
for (k = current_dim + 1; k < num_dims; k++) {
    stride *= dims[k];  // Product of all inner dimensions
}

for (i = 0; i < dims[current_dim]; i++) {
    if (i > 0) printf(" ");
    // RECURSION: Process next inner dimension
    displayTensorRecursive(
        elements,
        dims,
        num_dims,
        current_dim + 1,      // Go one level deeper
        offset + i * stride    // Advance by stride
    );
}
printf(")");
```

### Initial Call

```cpp
// For any tensor, call with:
displayTensorRecursive(elements, dims, num_dims, 0, 0);
```

## Examples

### 1D Vector: dims=[3]
```
Call: displayTensorRecursive(elems, [3], 1, 0, 0)
- current_dim=0, num_dims=1 → BASE CASE
- Print "(1 2 3)"
Result: #(1 2 3)
```

### 2D Matrix: dims=[2,3]  
```
Call: displayTensorRecursive(elems, [2,3], 2, 0, 0)
- current_dim=0, num_dims=2 → RECURSIVE CASE
- stride = dims[1] = 3
- Loop i=0,1:
  - i=0: recurse(elems, [2,3], 2, 1, 0)
    - current_dim=1, num_dims=2 → BASE CASE
    - Print "(1 2 3)"
  - i=1: recurse(elems, [2,3], 2, 1, 3)
    - current_dim=1, num_dims=2 → BASE CASE
    - Print "(4 5 6)"
Result: #((1 2 3) (4 5 6))
```

### 3D Tensor: dims=[2,2,2]
```
Call: displayTensorRecursive(elems, [2,2,2], 3, 0, 0)
- current_dim=0, num_dims=3 → RECURSIVE
- stride = dims[1] * dims[2] = 2*2 = 4
- Loop i=0,1:
  - i=0: recurse(elems, [2,2,2], 3, 1, 0)
    - current_dim=1, num_dims=3 → RECURSIVE
    - stride = dims[2] = 2
    - Loop j=0,1:
      - j=0: recurse(elems, [2,2,2], 3, 2, 0) → BASE → "(1 2)"
      - j=1: recurse(elems, [2,2,2], 3, 2, 2) → BASE → "(3 4)"
    - Prints "((1 2) (3 4))"
  - i=1: recurse(elems, [2,2,2], 3, 1, 4)
    - Similar → "((5 6) (7 8))"
Result: #(((1 2) (3 4)) ((5 6) (7 8)))
```

## LLVM IR Implementation Strategy

### Create Helper Function

```cpp
// In EshkolLLVMCodeGen class:
Function* display_tensor_recursive_func;  // Declare as member

void createDisplayTensorRecursiveFunction() {
    // Signature: void displayTensorRecursive(
    //     double* elements, 
    //     uint64_t* dims, 
    //     uint64_t num_dims,
    //     uint64_t current_dim,
    //     uint64_t offset
    // )
    
    std::vector<Type*> params = {
        PointerType::getUnqual(*context),  // double* elements
        PointerType::getUnqual(*context),  // uint64_t* dims
        Type::getInt64Ty(*context),        // uint64_t num_dims
        Type::getInt64Ty(*context),        // uint64_t current_dim
        Type::getInt64Ty(*context)         // uint64_t offset
    };
    
    FunctionType* func_type = FunctionType::get(
        Type::getVoidTy(*context),
        params,
        false
    );
    
    display_tensor_recursive_func = Function::Create(
        func_type,
        Function::InternalLinkage,
        "displayTensorRecursive",
        module.get()
    );
    
    // Create function body (see implementation below)
    implementDisplayTensorRecursive();
}
```

### Function Body Implementation

```cpp
void implementDisplayTensorRecursive() {
    BasicBlock* entry = BasicBlock::Create(*context, "entry", display_tensor_recursive_func);
    
    // Save old insertion point
    IRBuilderBase::InsertPoint old_ip = builder->saveIP();
    builder->SetInsertPoint(entry);
    
    // Get parameters
    auto arg_it = display_tensor_recursive_func->arg_begin();
    Value* elements = &*arg_it++; elements->setName("elements");
    Value* dims = &*arg_it++; dims->setName("dims");
    Value* num_dims = &*arg_it++; num_dims->setName("num_dims");
    Value* current_dim = &*arg_it++; current_dim->setName("current_dim");
    Value* offset = &*arg_it; offset->setName("offset");
    
    Function* printf_func = function_table["printf"];
    
    // Check: current_dim == num_dims - 1? (base case)
    Value* num_dims_minus_1 = builder->CreateSub(num_dims, ConstantInt::get(Type::getInt64Ty(*context), 1));
    Value* is_base_case = builder->CreateICmpEQ(current_dim, num_dims_minus_1);
    
    BasicBlock* base_case = BasicBlock::Create(*context, "base_case", display_tensor_recursive_func);
    BasicBlock* recursive_case = BasicBlock::Create(*context, "recursive_case", display_tensor_recursive_func);
    BasicBlock* func_exit = BasicBlock::Create(*context, "exit", display_tensor_recursive_func);
    
    builder->CreateCondBr(is_base_case, base_case, recursive_case);
    
    // === BASE CASE: Print innermost dimension ===
    builder->SetInsertPoint(base_case);
    builder->CreateCall(printf_func, {codegenString("(")});
    
    // Get dimension size at current_dim
    Value* dim_size_ptr = builder->CreateGEP(Type::getInt64Ty(*context), dims, current_dim);
    Value* dim_size = builder->CreateLoad(Type::getInt64Ty(*context), dim_size_ptr);
    
    // Loop: print elements
    BasicBlock* base_loop_cond = BasicBlock::Create(*context, "base_loop_cond", display_tensor_recursive_func);
    BasicBlock* base_loop_body = BasicBlock::Create(*context, "base_loop_body", display_tensor_recursive_func);
    BasicBlock* base_loop_exit = BasicBlock::Create(*context, "base_loop_exit", display_tensor_recursive_func);
    
    Value* base_i = builder->CreateAlloca(Type::getInt64Ty(*context), nullptr, "base_i");
    builder->CreateStore(ConstantInt::get(Type::getInt64Ty(*context), 0), base_i);
    builder->CreateBr(base_loop_cond);
    
    builder->SetInsertPoint(base_loop_cond);
    Value* i_val = builder->CreateLoad(Type::getInt64Ty(*context), base_i);
    Value* i_less = builder->CreateICmpULT(i_val, dim_size);
    builder->CreateCondBr(i_less, base_loop_body, base_loop_exit);
    
    builder->SetInsertPoint(base_loop_body);
    // Add space before non-first elements
    Value* i_is_zero = builder->CreateICmpEQ(i_val, ConstantInt::get(Type::getInt64Ty(*context), 0));
    BasicBlock* skip_space = BasicBlock::Create(*context, "skip_space", display_tensor_recursive_func);
    BasicBlock* print_space = BasicBlock::Create(*context, "print_space", display_tensor_recursive_func);
    BasicBlock* print_elem = BasicBlock::Create(*context, "print_elem", display_tensor_recursive_func);
    
    builder->CreateCondBr(i_is_zero, skip_space, print_space);
    
    builder->SetInsertPoint(print_space);
    builder->CreateCall(printf_func, {codegenString(" ")});
    builder->CreateBr(print_elem);
    
    builder->SetInsertPoint(skip_space);
    builder->CreateBr(print_elem);
    
    builder->SetInsertPoint(print_elem);
    // Print element at offset + i
    Value* elem_idx = builder->CreateAdd(offset, i_val);
    Value* elem_ptr = builder->CreateGEP(Type::getDoubleTy(*context), elements, elem_idx);
    Value* elem_val = builder->CreateLoad(Type::getDoubleTy(*context), elem_ptr);
    builder->CreateCall(printf_func, {codegenString("%g"), elem_val});
    
    // Increment i
    Value* i_next = builder->CreateAdd(i_val, ConstantInt::get(Type::getInt64Ty(*context), 1));
    builder->CreateStore(i_next, base_i);
    builder->CreateBr(base_loop_cond);
    
    builder->SetInsertPoint(base_loop_exit);
    builder->CreateCall(printf_func, {codegenString(")")});
    builder->CreateBr(func_exit);
    
    // === RECURSIVE CASE: Process outer dimension ===
    builder->SetInsertPoint(recursive_case);
    builder->CreateCall(printf_func, {codegenString("(")});
    
    // Calculate stride = product of dims[current_dim+1..num_dims-1]
    Value* stride = builder->CreateAlloca(Type::getInt64Ty(*context), nullptr, "stride");
    builder->CreateStore(ConstantInt::get(Type::getInt64Ty(*context), 1), stride);
    
    // Compute stride using loop
    Value* next_dim = builder->CreateAdd(current_dim, ConstantInt::get(Type::getInt64Ty(*context), 1));
    
    BasicBlock* stride_loop_cond = BasicBlock::Create(*context, "stride_cond", display_tensor_recursive_func);
    BasicBlock* stride_loop_body = BasicBlock::Create(*context, "stride_body", display_tensor_recursive_func);
    BasicBlock* stride_loop_exit = BasicBlock::Create(*context, "stride_exit", display_tensor_recursive_func);
    
    Value* k = builder->CreateAlloca(Type::getInt64Ty(*context), nullptr, "k");
    builder->CreateStore(next_dim, k);
    builder->CreateBr(stride_loop_cond);
    
    builder->SetInsertPoint(stride_loop_cond);
    Value* k_val = builder->CreateLoad(Type::getInt64Ty(*context), k);
    Value* k_less = builder->CreateICmpULT(k_val, num_dims);
    builder->CreateCondBr(k_less, stride_loop_body, stride_loop_exit);
    
    builder->SetInsertPoint(stride_loop_body);
    Value* dim_k_ptr = builder->CreateGEP(Type::getInt64Ty(*context), dims, k_val);
    Value* dim_k = builder->CreateLoad(Type::getInt64Ty(*context), dim_k_ptr);
    Value* current_stride = builder->CreateLoad(Type::getInt64Ty(*context), stride);
    Value* new_stride = builder->CreateMul(current_stride, dim_k);
    builder->CreateStore(new_stride, stride);
    Value* k_next = builder->CreateAdd(k_val, ConstantInt::get(Type::getInt64Ty(*context), 1));
    builder->CreateStore(k_next, k);
    builder->CreateBr(stride_loop_cond);
    
    builder->SetInsertPoint(stride_loop_exit);
    Value* final_stride = builder->CreateLoad(Type::getInt64Ty(*context), stride);
    
    // Get current dimension size
    Value* curr_dim_ptr = builder->CreateGEP(Type::getInt64Ty(*context), dims, current_dim);
    Value* curr_dim_size = builder->CreateLoad(Type::getInt64Ty(*context), curr_dim_ptr);
    
    // Loop over current dimension, recursing for each slice
    BasicBlock* rec_loop_cond = BasicBlock::Create(*context, "rec_loop_cond", display_tensor_recursive_func);
    BasicBlock* rec_loop_body = BasicBlock::Create(*context, "rec_loop_body", display_tensor_recursive_func);
    BasicBlock* rec_loop_exit = BasicBlock::Create(*context, "rec_loop_exit", display_tensor_recursive_func);
    
    Value* rec_i = builder->CreateAlloca(Type::getInt64Ty(*context), nullptr, "rec_i");
    builder->CreateStore(ConstantInt::get(Type::getInt64Ty(*context), 0), rec_i);
    builder->CreateBr(rec_loop_cond);
    
    builder->SetInsertPoint(rec_loop_cond);
    Value* i_val_rec = builder->CreateLoad(Type::getInt64Ty(*context), rec_i);
    Value* i_less_rec = builder->CreateICmpULT(i_val_rec, curr_dim_size);
    builder->CreateCondBr(i_less_rec, rec_loop_body, rec_loop_exit);
    
    builder->SetInsertPoint(rec_loop_body);
    // Add space before non-first slices
    Value* i_is_zero_rec = builder->CreateICmpEQ(i_val_rec, ConstantInt::get(Type::getInt64Ty(*context), 0));
    BasicBlock* skip_space_rec = BasicBlock::Create(*context, "skip_space_rec", display_tensor_recursive_func);
    BasicBlock* print_space_rec = BasicBlock::Create(*context, "print_space_rec", display_tensor_recursive_func);
    BasicBlock* make_recursive_call = BasicBlock::Create(*context, "recursive_call", display_tensor_recursive_func);
    
    builder->CreateCondBr(i_is_zero_rec, skip_space_rec, print_space_rec);
    
    builder->SetInsertPoint(print_space_rec);
    builder->CreateCall(printf_func, {codegenString(" ")});
    builder->CreateBr(make_recursive_call);
    
    builder->SetInsertPoint(skip_space_rec);
    builder->CreateBr(make_recursive_call);
    
    builder->SetInsertPoint(make_recursive_call);
    // RECURSIVE CALL: Process next dimension at offset + i*stride
    Value* new_offset = builder->CreateAdd(offset, builder->CreateMul(i_val_rec, final_stride));
    builder->CreateCall(display_tensor_recursive_func, {
        elements,
        dims,
        num_dims,
        next_dim,
        new_offset
    });
    
    // Increment i
    Value* i_next_rec = builder->CreateAdd(i_val_rec, ConstantInt::get(Type::getInt64Ty(*context), 1));
    builder->CreateStore(i_next_rec, rec_i);
    builder->CreateBr(rec_loop_cond);
    
    builder->SetInsertPoint(rec_loop_exit);
    builder->CreateCall(printf_func, {codegenString(")")});
    builder->CreateBr(func_exit);
    
    // Exit point
    builder->SetInsertPoint(func_exit);
    builder->CreateRetVoid();
    
    // Restore insertion point
    builder->restoreIP(old_ip);
}
```

## Integration with Display

### Modify `codegenDisplay()` Tensor Path

```cpp
// Around line 4450 in llvm_codegen.cpp
// BEFORE:
builder->CreateCall(printf_func, {codegenString("#(")});
// ... loop through elements linearly ...
builder->CreateCall(printf_func, {codegenString(")")});

// AFTER:
// Load tensor metadata
Value* num_dims_field = builder->CreateStructGEP(tensor_type, tensor_ptr, 1);
Value* num_dims_val = builder->CreateLoad(Type::getInt64Ty(*context), num_dims_field);

Value* dims_field = builder->CreateStructGEP(tensor_type, tensor_ptr, 0);
Value* dims_ptr_val = builder->CreateLoad(PointerType::getUnqual(*context), dims_field);

Value* elems_field = builder->CreateStructGEP(tensor_type, tensor_ptr, 2);
Value* elems_ptr_val = builder->CreateLoad(PointerType::getUnqual(*context), elems_field);
Value* typed_elems = builder->CreatePointerCast(elems_ptr_val, 
    PointerType::get(Type::getDoubleTy(*context), 0));

// Print opening "#"
builder->CreateCall(printf_func, {codegenString("#")});

// Call recursive display
builder->CreateCall(display_tensor_recursive_func, {
    typed_elems,
    dims_ptr_val,
    num_dims_val,
    ConstantInt::get(Type::getInt64Ty(*context), 0),  // start at dim 0
    ConstantInt::get(Type::getInt64Ty(*context), 0)   // start at offset 0
});
```

## Benefits

1. **Works for ANY dimensionality** - 1D through ND
2. **No hardcoded indices** - fully general
3. **Proper nesting** - visually clear structure
4. **Efficient** - single pass through elements
5. **Mathematically sound** - respects row-major layout

## Test Cases

### After Fix, Expected Outputs:

**1D Vector**:
```scheme
(vector 1.0 2.0 3.0)
→ #(1 2 3)  ✓ (unchanged)
```

**2D Jacobian**:
```scheme
(jacobian F (vector 2.0 3.0))  ; 2×2 matrix
→ #((3 2) (4 0))  ✓ (was: #(3 2 4 0))
```

**3D Curl** (if returned as tensor):
```scheme
(curl F (vector 1.0 2.0 3.0))
→ #(2 -1 0)  ✓ (3D vector)
```

**4D Tensor** (theoretical):
```scheme
dims=[2,2,2,2], 16 elements
→ #((((1 2) (3 4)) ((5 6) (7 8))) (((9 10) (11 12)) ((13 14) (15 16))))
```

## Implementation Location

**File**: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)

**Add Function**:
- After `createDisplaySExprListFunction()` (line 948)
- Call from `createBuiltinFunctions()` (line 323)

**Modify Display**:
- In `codegenDisplay()` around lines 4080-4165
- Replace flat tensor display with recursive call

## Estimated Implementation Time

- **Helper function creation**: 2-3 hours (LLVM IR is verbose)
- **Integration**: 1 hour
- **Testing**: 1-2 hours
- **Total**: 4-6 hours

## Success Criteria

✅ Vectors display as `#(e1 e2 ...)`  
✅ Matrices display as `#((r1) (r2) ...)`  
✅ 3D+ tensors display with proper nesting  
✅ Works for arbitrary dimensions (no hardcoded limits)  
✅ Jacobians clearly show 2D structure  
✅ All existing tests still pass