# Nested S-Expression Display Implementation Plan

## Executive Summary

**Goal**: Replace `(...)` placeholders in symbolic differentiation results with actual nested S-expression content.

**Current State**: 
- Simple derivatives work: `(* 2 x)`, `(cos x)`
- Nested expressions show placeholders: `(+ (...) (...))`
- String symbols display correctly: `+`, `*`, `sin`, `cos`

**Target State**: 
- Complete nested expansion: `(+ (* 1 (+ x 2)) (* (+ x 1) 1))`
- All 35 tests in [`phase0_diff_fixes.esk`](tests/autodiff/phase0_diff_fixes.esk) show full symbolic derivatives

## Problem Analysis

### Current Implementation

Two locations have placeholder logic:

1. **Location 1**: [`llvm_codegen.cpp:4072-4076`](lib/backend/llvm_codegen.cpp:4072) - `sexpr_display_nested` block
   ```cpp
   builder->SetInsertPoint(sexpr_display_nested);
   // Nested list - display recursively by calling display on the pointer
   // For now show the symbolic form
   builder->CreateCall(printf_func, {codegenString("(...)")});
   builder->CreateBr(sexpr_elem_done);
   ```

2. **Location 2**: [`llvm_codegen.cpp:4427-4429`](lib/backend/llvm_codegen.cpp:4427) - `display_nested_list` block
   ```cpp
   // Display nested list recursively (existing cons_ptr logic)
   builder->SetInsertPoint(display_nested_list);
   // For nested S-expressions, we need to recursively display them
   // For now, just display as "(nested)" - full recursive display would go here
   builder->CreateCall(printf_func, {codegenString("(nested)")});
   builder->CreateBr(element_done);
   ```

### Why Placeholders Exist

The original implementation couldn't use C++ recursion (calling `codegenDisplay` recursively) because:
- LLVM IR generation happens at compile-time, not runtime
- Need runtime loops/branches, not compile-time C++ function calls
- Must use LLVM `BasicBlock` control flow

## Solution Design

### Strategy: Stack-Based Iterative Display

Use **explicit LLVM IR loops** to traverse nested lists without C++ recursion.

#### Key Insight
The existing code already has a working pattern for list traversal:
- Lines 3985-4118: Main S-expression display loop
- Lines 4335-4495: Regular list display loop

**We can reuse this pattern** for nested list display by creating an inner loop structure.

### Implementation Approach

#### Pattern to Follow (from existing code)

```cpp
// Existing pattern (lines 3985-4118):
BasicBlock* sexpr_loop_cond = BasicBlock::Create(...);
BasicBlock* sexpr_loop_body = BasicBlock::Create(...);
BasicBlock* sexpr_loop_exit = BasicBlock::Create(...);

Value* sexpr_current = builder->CreateAlloca(...);
Value* sexpr_first = builder->CreateAlloca(...);

builder->CreateBr(sexpr_loop_cond);

// Condition: check if current != null
builder->SetInsertPoint(sexpr_loop_cond);
Value* sexpr_val = builder->CreateLoad(..., sexpr_current);
Value* sexpr_not_null = builder->CreateICmpNE(...);
builder->CreateCondBr(sexpr_not_null, sexpr_loop_body, sexpr_loop_exit);

// Body: display element, move to cdr
builder->SetInsertPoint(sexpr_loop_body);
// ... element display logic ...
builder->CreateBr(sexpr_loop_cond);

// Exit: done
builder->SetInsertPoint(sexpr_loop_exit);
```

## Detailed Implementation Plan

### Step 1: Replace `sexpr_display_nested` Placeholder (Line 4075)

**Location**: Inside main S-expression display loop, when car is nested CONS_PTR

**Current Context**:
```cpp
builder->SetInsertPoint(sexpr_display_nested);
// Nested list - display recursively by calling display on the pointer
// For now show the symbolic form
builder->CreateCall(printf_func, {codegenString("(...)")});  // ← REPLACE THIS
builder->CreateBr(sexpr_elem_done);
```

**New Implementation**:
```cpp
builder->SetInsertPoint(sexpr_display_nested);
// Nested list - display using iterative traversal (NO recursion!)

// Print opening parenthesis for nested list
builder->CreateCall(printf_func, {codegenString("(")});

// Initialize nested list traversal state
Value* nested_current = builder->CreateAlloca(Type::getInt64Ty(*context), nullptr, "nested_sexpr_current");
Value* nested_first = builder->CreateAlloca(Type::getInt1Ty(*context), nullptr, "nested_sexpr_first");
builder->CreateStore(sexpr_car_ptr_int, nested_current);  // sexpr_car_ptr_int already available
builder->CreateStore(ConstantInt::get(Type::getInt1Ty(*context), 1), nested_first);

// Create nested loop blocks
BasicBlock* nested_sexpr_cond = BasicBlock::Create(*context, "nested_sexpr_loop_cond", current_func);
BasicBlock* nested_sexpr_body = BasicBlock::Create(*context, "nested_sexpr_loop_body", current_func);
BasicBlock* nested_sexpr_exit = BasicBlock::Create(*context, "nested_sexpr_loop_exit", current_func);

builder->CreateBr(nested_sexpr_cond);

// Nested loop condition: check if current != null
builder->SetInsertPoint(nested_sexpr_cond);
Value* nested_sexpr_val = builder->CreateLoad(Type::getInt64Ty(*context), nested_current);
Value* nested_sexpr_not_null = builder->CreateICmpNE(nested_sexpr_val, 
    ConstantInt::get(Type::getInt64Ty(*context), 0));
builder->CreateCondBr(nested_sexpr_not_null, nested_sexpr_body, nested_sexpr_exit);

// Nested loop body: display element with spacing
builder->SetInsertPoint(nested_sexpr_body);

// Add space before non-first elements
Value* nested_first_flag = builder->CreateLoad(Type::getInt1Ty(*context), nested_first);
BasicBlock* nested_skip_space = BasicBlock::Create(*context, "nested_sexpr_skip_space", current_func);
BasicBlock* nested_add_space = BasicBlock::Create(*context, "nested_sexpr_add_space", current_func);
BasicBlock* nested_display_elem = BasicBlock::Create(*context, "nested_sexpr_display_elem", current_func);

builder->CreateCondBr(nested_first_flag, nested_skip_space, nested_add_space);

builder->SetInsertPoint(nested_add_space);
builder->CreateCall(printf_func, {codegenString(" ")});
builder->CreateBr(nested_display_elem);

builder->SetInsertPoint(nested_skip_space);
builder->CreateStore(ConstantInt::get(Type::getInt1Ty(*context), 0), nested_first);
builder->CreateBr(nested_display_elem);

// Display nested element - extract car and check type
builder->SetInsertPoint(nested_display_elem);
Value* nested_car_tagged = extractCarAsTaggedValue(nested_sexpr_val);
Value* nested_car_type = getTaggedValueType(nested_car_tagged);
Value* nested_car_base = builder->CreateAnd(nested_car_type,
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));

// Check element type: CONS_PTR (string/nested), DOUBLE, or INT64
Value* nested_car_is_ptr = builder->CreateICmpEQ(nested_car_base,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_CONS_PTR));
Value* nested_car_is_double = builder->CreateICmpEQ(nested_car_base,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));

// Create display branches for nested element
BasicBlock* nested_check_string = BasicBlock::Create(*context, "nested_check_string", current_func);
BasicBlock* nested_check_double = BasicBlock::Create(*context, "nested_check_double", current_func);
BasicBlock* nested_display_string = BasicBlock::Create(*context, "nested_display_string", current_func);
BasicBlock* nested_display_doublenested = BasicBlock::Create(*context, "nested_display_doublenested", current_func);
BasicBlock* nested_display_double = BasicBlock::Create(*context, "nested_display_double", current_func);
BasicBlock* nested_display_int = BasicBlock::Create(*context, "nested_display_int", current_func);
BasicBlock* nested_elem_done = BasicBlock::Create(*context, "nested_sexpr_elem_done", current_func);

builder->CreateCondBr(nested_car_is_ptr, nested_check_string, nested_check_double);

// Check if CONS_PTR is string or doubly-nested list
builder->SetInsertPoint(nested_check_string);
Value* nested_car_ptr_int = unpackInt64FromTaggedValue(nested_car_tagged);
Value* nested_car_ptr = builder->CreateIntToPtr(nested_car_ptr_int, builder->getPtrTy());

// String detection using ASCII heuristic (reuse existing logic)
Value* nested_ptr_reasonable = builder->CreateICmpUGT(nested_car_ptr_int,
    ConstantInt::get(Type::getInt64Ty(*context), 1000));
Value* nested_ptr_not_huge = builder->CreateICmpULT(nested_car_ptr_int,
    ConstantInt::get(Type::getInt64Ty(*context), 0x7FFFFFFFFFFFFFFFULL));
Value* nested_ptr_in_range = builder->CreateAnd(nested_ptr_reasonable, nested_ptr_not_huge);

BasicBlock* nested_check_ascii = BasicBlock::Create(*context, "nested_check_ascii", current_func);

builder->CreateCondBr(nested_ptr_in_range, nested_check_ascii, nested_display_doublenested);

builder->SetInsertPoint(nested_check_ascii);
Value* nested_first_byte = builder->CreateLoad(Type::getInt8Ty(*context), nested_car_ptr);
Value* nested_is_printable = builder->CreateAnd(
    builder->CreateICmpUGE(nested_first_byte, ConstantInt::get(Type::getInt8Ty(*context), 32)),
    builder->CreateICmpULE(nested_first_byte, ConstantInt::get(Type::getInt8Ty(*context), 126)));

builder->CreateCondBr(nested_is_printable, nested_display_string, nested_display_doublenested);

// Display string symbol
builder->SetInsertPoint(nested_display_string);
builder->CreateCall(printf_func, {codegenString("%s"), nested_car_ptr});
builder->CreateBr(nested_elem_done);

// Display doubly-nested list - RECURSION POINT!
// For now, just print another (...) but mark for future depth-2 implementation
builder->SetInsertPoint(nested_display_doublenested);
builder->CreateCall(printf_func, {codegenString("(...)")});  // Depth 2+ placeholder
builder->CreateBr(nested_elem_done);

// Display double
builder->SetInsertPoint(nested_check_double);
builder->CreateCondBr(nested_car_is_double, nested_display_double, nested_display_int);

builder->SetInsertPoint(nested_display_double);
Value* nested_car_double = unpackDoubleFromTaggedValue(nested_car_tagged);
builder->CreateCall(printf_func, {codegenString("%g"), nested_car_double});
builder->CreateBr(nested_elem_done);

// Display int
builder->SetInsertPoint(nested_display_int);
Value* nested_car_int = unpackInt64FromTaggedValue(nested_car_tagged);
builder->CreateCall(printf_func, {codegenString("%lld"), nested_car_int});
builder->CreateBr(nested_elem_done);

// Move to next element in nested list
builder->SetInsertPoint(nested_elem_done);
Value* nested_cons_ptr = builder->CreateIntToPtr(nested_sexpr_val, builder->getPtrTy());
Value* nested_is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);
Value* nested_cdr_type = builder->CreateCall(arena_tagged_cons_get_type_func, 
    {nested_cons_ptr, nested_is_cdr});
Value* nested_cdr_base = builder->CreateAnd(nested_cdr_type,
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
Value* nested_cdr_is_ptr = builder->CreateICmpEQ(nested_cdr_base,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_CONS_PTR));
Value* nested_cdr_is_null = builder->CreateICmpEQ(nested_cdr_base,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_NULL));

BasicBlock* nested_cdr_ptr = BasicBlock::Create(*context, "nested_sexpr_cdr_ptr", current_func);
BasicBlock* nested_cdr_null = BasicBlock::Create(*context, "nested_sexpr_cdr_null", current_func);

builder->CreateCondBr(nested_cdr_is_ptr, nested_cdr_ptr, nested_cdr_null);

builder->SetInsertPoint(nested_cdr_ptr);
Value* nested_next = builder->CreateCall(arena_tagged_cons_get_ptr_func, 
    {nested_cons_ptr, nested_is_cdr});
builder->CreateStore(nested_next, nested_current);
builder->CreateBr(nested_sexpr_cond);

builder->SetInsertPoint(nested_cdr_null);
builder->CreateCondBr(nested_cdr_is_null, nested_sexpr_exit, nested_sexpr_exit);

// Nested loop exit: close parenthesis
builder->SetInsertPoint(nested_sexpr_exit);
builder->CreateCall(printf_func, {codegenString(")")});
builder->CreateBr(sexpr_elem_done);
```

### Step 2: Replace `display_nested_list` Placeholder (Line 4429)

**Location**: Inside regular list display loop, when car is nested CONS_PTR

**Current Context**:
```cpp
// Display nested list recursively (existing cons_ptr logic)
builder->SetInsertPoint(display_nested_list);
// For nested S-expressions, we need to recursively display them
// For now, just display as "(nested)" - full recursive display would go here
builder->CreateCall(printf_func, {codegenString("(nested)")});  // ← REPLACE THIS
builder->CreateBr(element_done);
```

**New Implementation**: **Same pattern as Location 1**, just different block names to avoid conflicts:
- Use `nested_list_` prefix instead of `nested_sexpr_`
- Reuse exact same logic structure

### Step 3: Handle Depth-2+ Nesting (Future Enhancement)

For expressions like `(+ (* (+ x 1) (+ x 2)) (* 3 x))`, we have:
- Depth 0: Outer `(+  ...)`
- Depth 1: `(* ...)` terms - **will work with Step 1+2**
- Depth 2: `(+ x 1)`, `(+ x 2)` - **needs recursive placeholder replacement**

**Options**:
1. **Iterative with stack**: Track depth explicitly, use LLVM array/stack
2. **Depth limit**: Display up to depth 2-3, then `(...)` for deeper
3. **Full recursion**: Use helper function (complex, requires LLVM function calls)

**Recommendation**: Start with **depth limit of 2** (sufficient for most derivatives), implement full stack-based iteration in future phase if needed.

## Implementation Checklist

### Phase 1: Single-Level Nesting (Lines 4075, 4429)
- [ ] Create nested loop structure in `sexpr_display_nested` block
- [ ] Add spacing logic for nested elements
- [ ] Add element type detection (string/double/int/nested)
- [ ] Add cdr traversal logic
- [ ] Add proper parenthesis wrapping
- [ ] Copy same pattern to `display_nested_list` block with different block names

### Phase 2: Testing
- [ ] Test simple nested case: `(diff (* (+ x 1) (+ x 2)) x)`
- [ ] Expected: `(+ (* 1 (+ x 2)) (* (+ x 1) 1))`
- [ ] Run all 35 tests in [`phase0_diff_fixes.esk`](tests/autodiff/phase0_diff_fixes.esk)

### Phase 3: Depth-2 Support (If Needed)
- [ ] Replace depth-2 `(...)` placeholder with another nested loop
- [ ] Add depth counter to prevent infinite loops
- [ ] Test complex expressions with 3+ depth levels

## Code Reuse Strategy

### Reusable Patterns from Existing Code

1. **Spacing Logic** (lines 4004-4017):
   ```cpp
   Value* sexpr_first_flag = builder->CreateLoad(..., sexpr_first);
   BasicBlock* sexpr_skip_space = BasicBlock::Create(...);
   BasicBlock* sexpr_add_space = BasicBlock::Create(...);
   builder->CreateCondBr(sexpr_first_flag, sexpr_skip_space, sexpr_add_space);
   ```

2. **Element Type Detection** (lines 4021-4029):
   ```cpp
   Value* sexpr_car_tagged = extractCarAsTaggedValue(sexpr_val);
   Value* sexpr_car_type = getTaggedValueType(sexpr_car_tagged);
   Value* sexpr_car_base = builder->CreateAnd(sexpr_car_type, ...);
   Value* sexpr_car_is_ptr = builder->CreateICmpEQ(...);
   ```

3. **String Detection Heuristic** (lines 4045-4066):
   ```cpp
   Value* ptr_is_reasonable = builder->CreateICmpUGT(..., 1000);
   Value* first_byte = builder->CreateLoad(Type::getInt8Ty(*context), ptr);
   Value* is_printable = builder->CreateAnd(
       builder->CreateICmpUGE(first_byte, 32),
       builder->CreateICmpULE(first_byte, 126));
   ```

4. **Cdr Traversal** (lines 4093-4114):
   ```cpp
   Value* sexpr_is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);
   Value* sexpr_cdr_type = builder->CreateCall(arena_tagged_cons_get_type_func, ...);
   Value* sexpr_cdr_is_ptr = builder->CreateICmpEQ(...);
   Value* sexpr_next = builder->CreateCall(arena_tagged_cons_get_ptr_func, ...);
   builder->CreateStore(sexpr_next, sexpr_current);
   ```

### Variable Naming Convention

To avoid conflicts with existing code:
- Depth 0 (main loop): `sexpr_*` (already exists)
- Depth 1 (nested): `nested_sexpr_*` or `nested_list_*`
- Depth 2 (if implemented): `nested2_*`

## Expected Test Results

### Before Implementation
```
Test 4: d/dx((x+1) * (x+2)) = (+ (...) (...))
Test 7: d/dx((2*x) * (3*x)) = (+ (...) (...))
Test 10: d/dx(sin(2*x)) = cos(2x)*2 = (* (...) 2)
Test 11: d/dx(cos(x*x)) = -sin(x²)*2x = (* (...) (...))
```

### After Implementation
```
Test 4: d/dx((x+1) * (x+2)) = (+ (* 1 (+ x 2)) (* (+ x 1) 1))
Test 7: d/dx((2*x) * (3*x)) = (+ (* 2 (* 3 x)) (* (* 2 x) 3))
Test 10: d/dx(sin(2*x)) = cos(2x)*2 = (* (cos (* 2 x)) 2)
Test 11: d/dx(cos(x*x)) = -sin(x²)*2x = (* (- (sin (* x x))) (* 2 x))
```

## Technical Notes

### Why Iterative Over Recursive

**Recursive approach** (calling `codegenDisplay` again):
- ❌ C++ function recursion during IR generation
- ❌ Mixes compile-time and runtime behavior
- ❌ Stack depth issues for complex expressions

**Iterative approach** (LLVM loops):
- ✅ Pure runtime control flow
- ✅ Predictable stack usage
- ✅ Follows existing code patterns
- ✅ Easy to add depth limits

### Performance Considerations

- Nested loop adds O(n*m) complexity where n=outer list length, m=nested list length
- For symbolic derivatives, typically: n=2-4, m=2-4, so total ~16 iterations max
- Negligible performance impact

### Error Handling

Already present in existing code:
- Null pointer checks before dereferencing
- Type validation before extraction
- Safe string detection heuristic

## Risk Assessment

**LOW RISK**:
- Reuses proven patterns from existing code
- Only modifies display logic (no memory management changes)
- Easy to debug with printf statements
- Can add depth limit as safety measure

## Success Metrics

1. **Functional**: All 35 tests show complete S-expressions (no `(...)` placeholders)
2. **Correctness**: Symbolic derivatives match expected mathematical formulas
3. **Robustness**: No crashes or memory errors during display
4. **Readability**: Output is properly formatted with spaces and parentheses

## Future Enhancements

### Depth-N Support (Optional)
Replace depth-2 placeholder with full stack-based DFS:
```cpp
// Pseudo-code for stack-based display
struct DisplayStackFrame {
    uint64_t cons_ptr;
    bool first_element;
    bool opened_paren;
};

Value* stack_array[MAX_DEPTH];
Value* stack_top = 0;

while (stack_top >= 0) {
    // Pop frame
    // Process element
    // Push children if nested
    // Print parentheses at appropriate times
}
```

### Pretty Printing (Optional)
- Indentation based on nesting depth
- Line breaks for long expressions
- Color coding for operators/variables/constants

## Dependencies

**Required Functions** (already implemented):
- [`extractCarAsTaggedValue()`](lib/backend/llvm_codegen.cpp:1646)
- [`getTaggedValueType()`](lib/backend/llvm_codegen.cpp:1603)
- [`unpackInt64FromTaggedValue()`](lib/backend/llvm_codegen.cpp:1610)
- [`unpackDoubleFromTaggedValue()`](lib/backend/llvm_codegen.cpp:1617)
- [`arena_tagged_cons_get_type_func`](lib/backend/llvm_codegen.cpp:758)
- [`arena_tagged_cons_get_ptr_func`](lib/backend/llvm_codegen.cpp:651)

**No new dependencies needed!**

## Validation Strategy

1. **Unit Test**: Simple nested case first
   ```scheme
   (diff (* (+ x 1) 2) x)  ; Should show: (* 1 2) = 2
   ```

2. **Integration Test**: Run [`phase0_diff_fixes.esk`](tests/autodiff/phase0_diff_fixes.esk)

3. **Edge Cases**:
   - Empty nested lists (shouldn't occur in diff results)
   - Deeply nested (3+ levels)
   - Mixed types in nested expressions

## Timeline Estimate

- **Step 1** (Replace placeholders): 30-45 minutes
- **Step 2** (Testing & debugging): 15-30 minutes
- **Total**: 45-75 minutes for complete depth-1 nesting support

## Next Steps

Ready to switch to **Code mode** to implement the solution. The plan is:

1. Modify [`llvm_codegen.cpp:4075`](lib/backend/llvm_codegen.cpp:4075) - replace first placeholder
2. Modify [`llvm_codegen.cpp:4429`](lib/backend/llvm_codegen.cpp:4429) - replace second placeholder  
3. Build and test with [`phase0_diff_fixes.esk`](tests/autodiff/phase0_diff_fixes.esk)
4. Iterate on any display formatting issues