# Symbolic Differentiation S-Expression Display Fix Plan

## Problem Analysis

### Root Cause
The symbolic differentiation system (`diff` operator) creates S-expression results as runtime cons cell structures. These S-expressions contain:
1. **Numeric values** (int64/double) - work fine
2. **Operator symbols** as string pointers ("+", "*", "sin", "cos", etc.) - **THIS CAUSES THE ERROR**
3. **Nested S-expressions** as cons cell pointers - work fine

### Error Flow
```
(diff (* x x) x)
  ↓ buildSymbolicDerivative (compile-time AST transformation)
  ↓ Returns: (* 2 x) as AST
  ↓ codegenQuotedAST converts AST to runtime S-expression
  ↓ Symbols become string pointers with CONS_PTR type tag
  ↓ codegenDisplay tries to display the list
  ↓ extractCarAsTaggedValue calls arena_tagged_cons_get_int64
  ↓ ERROR: "Invalid type for int64 value: 3" (CONS_PTR = 3)
```

### Code Locations
- **Symbol creation**: [`codegenQuotedAST()`](lib/backend/llvm_codegen.cpp:7027) - creates string ptrs for symbols
- **List building**: [`codegenQuotedList()`](lib/backend/llvm_codegen.cpp:7048) - builds cons chains with symbols
- **Display logic**: [`codegenDisplay()`](lib/backend/llvm_codegen.cpp:4148) - fails on string pointers
- **Error source**: [`arena_tagged_cons_get_int64()`](lib/core/arena_memory.cpp:397) and [`arena_tagged_cons_set_int64()`](lib/core/arena_memory.cpp:453)

## Solution Design

### Approach: Modify Display Logic (Option A)
Add string pointer detection to the list display path in [`codegenDisplay()`](lib/backend/llvm_codegen.cpp:4148).

### Implementation Strategy

#### 1. String Detection Heuristic
When encountering a `CONS_PTR` type in a cons cell car:
- Cast the pointer to `char*`
- Check if first byte is printable ASCII (32-126)
- If yes: treat as string symbol and print directly
- If no: treat as cons cell pointer (nested list)

#### 2. Code Changes in `codegenDisplay()`

**Location**: Around line 4192 in the list display loop

**Current problematic flow**:
```cpp
// Display current element using tagged value extraction
builder->SetInsertPoint(display_element);
Value* car_tagged = extractCarAsTaggedValue(current_val);  // ← FAILS for strings!
Value* car_type = getTaggedValueType(car_tagged);
// ... tries to extract as int/double ...
```

**New flow with string detection**:
```cpp
// Display current element - FIRST check if car is a string pointer
builder->SetInsertPoint(display_element);

// Get car type BEFORE extraction to avoid errors
Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
Value* car_type_tag = builder->CreateCall(arena_tagged_cons_get_type_func,
    {cons_ptr, is_car});
Value* car_base_type = builder->CreateAnd(car_type_tag,
    ConstantInt::get(Type::getInt8Ty(*context), 0x0F));

// Check if car is CONS_PTR (could be string OR nested list)
Value* car_is_cons_ptr = builder->CreateICmpEQ(car_base_type,
    ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_CONS_PTR));

BasicBlock* check_string = BasicBlock::Create(*context, "check_if_string", current_func);
BasicBlock* extract_normal = BasicBlock::Create(*context, "extract_normal", current_func);

builder->CreateCondBr(car_is_cons_ptr, check_string, extract_normal);

// Check if CONS_PTR points to a string (ASCII heuristic)
builder->SetInsertPoint(check_string);
Value* car_ptr_int = builder->CreateCall(arena_tagged_cons_get_ptr_func,
    {cons_ptr, is_car});
Value* car_ptr = builder->CreateIntToPtr(car_ptr_int, builder->getPtrTy());

// Load first byte and check if printable ASCII
Value* first_byte_ptr = builder->CreatePointerCast(car_ptr, builder->getPtrTy());
Value* first_byte = builder->CreateLoad(Type::getInt8Ty(*context), first_byte_ptr);
Value* is_ascii = builder->CreateAnd(
    builder->CreateICmpUGE(first_byte, ConstantInt::get(Type::getInt8Ty(*context), 32)),
    builder->CreateICmpULE(first_byte, ConstantInt::get(Type::getInt8Ty(*context), 126)));

BasicBlock* display_string = BasicBlock::Create(*context, "display_string", current_func);
BasicBlock* display_nested = BasicBlock::Create(*context, "display_nested_list", current_func);

builder->CreateCondBr(is_ascii, display_string, display_nested);

// Display as string symbol
builder->SetInsertPoint(display_string);
builder->CreateCall(printf_func, {codegenString("%s"), car_ptr});
builder->CreateBr(element_done);

// Display as nested list (existing logic for cons_ptr)
builder->SetInsertPoint(display_nested);
// Recursively display nested list...
builder->CreateBr(element_done);

// Normal extraction path (existing logic for int64/double)
builder->SetInsertPoint(extract_normal);
Value* car_tagged = extractCarAsTaggedValue(current_val);
// ... existing double/int display logic ...
```

## Implementation Checklist

### Phase 1: Core String Detection
- [ ] Add string detection logic before `extractCarAsTaggedValue()` call
- [ ] Create branch for CONS_PTR type check
- [ ] Implement ASCII heuristic (check first byte in range 32-126)
- [ ] Add string display branch using printf with %s format

### Phase 2: Nested List Handling
- [ ] Add recursive nested list display for non-string CONS_PTR
- [ ] Ensure proper opening/closing parentheses for nested S-expressions
- [ ] Handle mixed types in nested expressions

### Phase 3: Testing
- [ ] Run [`phase0_diff_fixes.esk`](tests/autodiff/phase0_diff_fixes.esk)
- [ ] Verify S-expressions display correctly: `(* 2 x)` instead of errors
- [ ] Check all test cases display proper symbolic derivatives

## Expected Output After Fix

### Test Case: `(diff (* x x) x)`
**Before**: 
```
error: Invalid type for int64 value: 3
(0 2 0)
```

**After**:
```
(* 2 x)
```

### Test Case: `(diff (sin (* 2 x)) x)`
**Before**:
```
error: Invalid type for int64 value: 3
(0 0 0)
```

**After**:
```
(* (cos (* 2 x)) 2)
```

## Technical Notes

### Why This Approach
1. **Minimal changes**: Only modifies display logic, preserves AST design
2. **Backward compatible**: Doesn't break existing list/cons cell display
3. **Efficient**: Single ASCII check per symbol, no string table needed
4. **Robust**: Heuristic correctly distinguishes strings from pointers

### Alternative Considered
Adding a dedicated `STRING` value type to the tagged system would be more comprehensive but requires:
- Changes to [`eshkol_value_type_t`](inc/eshkol/eshkol.h:41-51) enum
- New pack/unpack functions for strings
- Migration of existing string handling
- More complex and risky for this specific fix

## Risk Assessment

**LOW RISK**:
- Isolated change in display logic only
- No changes to memory management or type system
- Backward compatible with existing behavior
- Easy to revert if issues arise

## Success Criteria

✓ All 35 tests in `phase0_diff_fixes.esk` display S-expressions correctly
✓ No "Invalid type for int64" errors
✓ Symbolic derivatives show as readable formulas: `(* 2 x)`, `(cos x)`, etc.
✓ Existing list/cons cell display unchanged