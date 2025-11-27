# N-Depth Recursive S-Expression Display Implementation Plan

## Revised Approach: True LLVM IR Recursion

### Key Insight
Instead of inline loops (limited depth), create a **recursive LLVM IR function** that can call itself for arbitrary nesting depth.

## Solution Architecture

### Helper Function Design

**Function Signature** (LLVM IR):
```cpp
void displaySExprList(int64_t list_ptr, int32_t depth)
```

**Behavior**:
1. Check depth limit (prevent stack overflow)
2. Print opening `(`
3. Loop through list elements:
   - If element is **string**: print directly
   - If element is **int/double**: print value
   - If element is **nested list**: **recursively call self** with depth+1
4. Print closing `)`

**Advantages**:
- ✅ True N-depth recursion (no hardcoded limits)
- ✅ No placeholders at any depth
- ✅ Clean, maintainable code
- ✅ Standard recursive pattern

### Implementation Strategy

#### Step 1: Create Recursive Helper Function

Add to `EshkolLLVMCodeGen` class (after line 157):
```cpp
Function* display_sexpr_list_func;  // Declaration
```

Create function in `createBuiltinFunctions()`:
```cpp
void createDisplaySExprListFunction() {
    // Function type: void displaySExprList(int64_t list_ptr, int32_t depth)
    std::vector<Type*> params;
    params.push_back(Type::getInt64Ty(*context));  // list_ptr
    params.push_back(Type::getInt32Ty(*context));  // depth
    
    FunctionType* func_type = FunctionType::get(
        Type::getVoidTy(*context),  // returns void
        params,
        false  // not varargs
    );
    
    display_sexpr_list_func = Function::Create(
        func_type,
        Function::InternalLinkage,  // Internal - only used within module
        "displaySExprList",
        module.get()
    );
    
    // Create function body
    BasicBlock* entry = BasicBlock::Create(*context, "entry", display_sexpr_list_func);
    IRBuilderBase::InsertPoint old_point = builder->saveIP();
    builder->SetInsertPoint(entry);
    
    // Get parameters
    auto arg_it = display_sexpr_list_func->arg_begin();
    Value* list_ptr_param = &*arg_it++;
    Value* depth_param = &*arg_it;
    
    Function* printf_func = function_table["printf"];
    
    // DEPTH LIMIT CHECK (prevent infinite loops)
    Value* max_depth = ConstantInt::get(Type::getInt32Ty(*context), 20);
    Value* depth_exceeded = builder->CreateICmpUGT(depth_param, max_depth);
    
    BasicBlock* depth_ok = BasicBlock::Create(*context, "depth_ok", display_sexpr_list_func);
    BasicBlock* depth_exceeded_block = BasicBlock::Create(*context, "depth_exceeded", display_sexpr_list_func);
    BasicBlock* func_exit = BasicBlock::Create(*context, "exit", display_sexpr_list_func);
    
    builder->CreateCondBr(depth_exceeded, depth_exceeded_block, depth_ok);
    
    // Depth exceeded: print warning and return
    builder->SetInsertPoint(depth_exceeded_block);
    builder->CreateCall(printf_func, {codegenString("[depth-limit]")});
    builder->CreateBr(func_exit);
    
    // Depth OK: continue with display
    builder->SetInsertPoint(depth_ok);
    
    // Print opening parenthesis
    builder->CreateCall(printf_func, {codegenString("(")});
    
    // Initialize loop state
    Value* current = builder->CreateAlloca(Type::getInt64Ty(*context), nullptr, "current");
    Value* is_first = builder->CreateAlloca(Type::getInt1Ty(*context), nullptr, "is_first");
    builder->CreateStore(list_ptr_param, current);
    builder->CreateStore(ConstantInt::get(Type::getInt1Ty(*context), 1), is_first);
    
    // Create loop blocks
    BasicBlock* loop_cond = BasicBlock::Create(*context, "loop_cond", display_sexpr_list_func);
    BasicBlock* loop_body = BasicBlock::Create(*context, "loop_body", display_sexpr_list_func);
    BasicBlock* loop_exit = BasicBlock::Create(*context, "loop_exit", display_sexpr_list_func);
    
    builder->CreateBr(loop_cond);
    
    // Loop condition: while current != null
    builder->SetInsertPoint(loop_cond);
    Value* current_val = builder->CreateLoad(Type::getInt64Ty(*context), current);
    Value* not_null = builder->CreateICmpNE(current_val, ConstantInt::get(Type::getInt64Ty(*context), 0));
    builder->CreateCondBr(not_null, loop_body, loop_exit);
    
    // Loop body: display element
    builder->SetInsertPoint(loop_body);
    
    // Add spacing
    Value* first_flag = builder->CreateLoad(Type::getInt1Ty(*context), is_first);
    BasicBlock* skip_space = BasicBlock::Create(*context, "skip_space", display_sexpr_list_func);
    BasicBlock* add_space = BasicBlock::Create(*context, "add_space", display_sexpr_list_func);
    BasicBlock* display_elem = BasicBlock::Create(*context, "display_elem", display_sexpr_list_func);
    
    builder->CreateCondBr(first_flag, skip_space, add_space);
    
    builder->SetInsertPoint(add_space);
    builder->CreateCall(printf_func, {codegenString(" ")});
    builder->CreateBr(display_elem);
    
    builder->SetInsertPoint(skip_space);
    builder->CreateStore(ConstantInt::get(Type::getInt1Ty(*context), 0), is_first);
    builder->CreateBr(display_elem);
    
    // Display element - extract car and check type
    builder->SetInsertPoint(display_elem);
    Value* car_tagged = extractCarAsTaggedValue(current_val);
    Value* car_type = getTaggedValueType(car_tagged);
    Value* car_base = builder->CreateAnd(car_type, 
        ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    
    // Type branching
    Value* car_is_ptr = builder->CreateICmpEQ(car_base,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_CONS_PTR));
    Value* car_is_double = builder->CreateICmpEQ(car_base,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
    
    BasicBlock* check_ptr = BasicBlock::Create(*context, "check_ptr", display_sexpr_list_func);
    BasicBlock* check_double = BasicBlock::Create(*context, "check_double", display_sexpr_list_func);
    BasicBlock* display_string = BasicBlock::Create(*context, "display_string", display_sexpr_list_func);
    BasicBlock* display_nested = BasicBlock::Create(*context, "display_nested", display_sexpr_list_func);
    BasicBlock* display_double = BasicBlock::Create(*context, "display_double", display_sexpr_list_func);
    BasicBlock* display_int = BasicBlock::Create(*context, "display_int", display_sexpr_list_func);
    BasicBlock* elem_done = BasicBlock::Create(*context, "elem_done", display_sexpr_list_func);
    
    builder->CreateCondBr(car_is_ptr, check_ptr, check_double);
    
    // Check if CONS_PTR is string or nested list
    builder->SetInsertPoint(check_ptr);
    Value* car_ptr_int = unpackInt64FromTaggedValue(car_tagged);
    Value* car_ptr = builder->CreateIntToPtr(car_ptr_int, builder->getPtrTy());
    
    // String detection heuristic
    Value* ptr_reasonable = builder->CreateICmpUGT(car_ptr_int,
        ConstantInt::get(Type::getInt64Ty(*context), 1000));
    Value* ptr_not_huge = builder->CreateICmpULT(car_ptr_int,
        ConstantInt::get(Type::getInt64Ty(*context), 0x7FFFFFFFFFFFFFFFULL));
    Value* ptr_in_range = builder->CreateAnd(ptr_reasonable, ptr_not_huge);
    
    BasicBlock* check_ascii = BasicBlock::Create(*context, "check_ascii", display_sexpr_list_func);
    builder->CreateCondBr(ptr_in_range, check_ascii, display_nested);
    
    builder->SetInsertPoint(check_ascii);
    Value* first_byte = builder->CreateLoad(Type::getInt8Ty(*context), car_ptr);
    Value* is_printable = builder->CreateAnd(
        builder->CreateICmpUGE(first_byte, ConstantInt::get(Type::getInt8Ty(*context), 32)),
        builder->CreateICmpULE(first_byte, ConstantInt::get(Type::getInt8Ty(*context), 126)));
    builder->CreateCondBr(is_printable, display_string, display_nested);
    
    // Display string
    builder->SetInsertPoint(display_string);
    builder->CreateCall(printf_func, {codegenString("%s"), car_ptr});
    builder->CreateBr(elem_done);
    
    // Display nested list - RECURSIVE CALL!
    builder->SetInsertPoint(display_nested);
    Value* next_depth = builder->CreateAdd(depth_param, ConstantInt::get(Type::getInt32Ty(*context), 1));
    builder->CreateCall(display_sexpr_list_func, {car_ptr_int, next_depth});  // ← RECURSION
    builder->CreateBr(elem_done);
    
    // Display double
    builder->SetInsertPoint(check_double);
    builder->CreateCondBr(car_is_double, display_double, display_int);
    
    builder->SetInsertPoint(display_double);
    Value* car_double_val = unpackDoubleFromTaggedValue(car_tagged);
    builder->CreateCall(printf_func, {codegenString("%g"), car_double_val});
    builder->CreateBr(elem_done);
    
    // Display int
    builder->SetInsertPoint(display_int);
    Value* car_int_val = unpackInt64FromTaggedValue(car_tagged);
    builder->CreateCall(printf_func, {codegenString("%lld"), car_int_val});
    builder->CreateBr(elem_done);
    
    // Move to next element
    builder->SetInsertPoint(elem_done);
    Value* cons_ptr = builder->CreateIntToPtr(current_val, builder->getPtrTy());
    Value* is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);
    Value* cdr_type = builder->CreateCall(arena_tagged_cons_get_type_func, {cons_ptr, is_cdr});
    Value* cdr_base = builder->CreateAnd(cdr_type, ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
    Value* cdr_is_ptr = builder->CreateICmpEQ(cdr_base,
        ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_CONS_PTR));
    
    BasicBlock* cdr_ptr = BasicBlock::Create(*context, "cdr_ptr", display_sexpr_list_func);
    BasicBlock* cdr_end = BasicBlock::Create(*context, "cdr_end", display_sexpr_list_func);
    
    builder->CreateCondBr(cdr_is_ptr, cdr_ptr, cdr_end);
    
    builder->SetInsertPoint(cdr_ptr);
    Value* next_ptr = builder->CreateCall(arena_tagged_cons_get_ptr_func, {cons_ptr, is_cdr});
    builder->CreateStore(next_ptr, current);
    builder->CreateBr(loop_cond);
    
    builder->SetInsertPoint(cdr_end);
    builder->CreateBr(loop_exit);
    
    // Loop exit: close parenthesis and return
    builder->SetInsertPoint(loop_exit);
    builder->CreateCall(printf_func, {codegenString(")")});
    builder->CreateBr(func_exit);
    
    builder->SetInsertPoint(func_exit);
    builder->CreateRetVoid();
    
    // Restore insertion point
    builder->restoreIP(old_point);
    
    function_table["displaySExprList"] = display_sexpr_list_func;
}
```

#### Step 2: Call Helper from Placeholders

**Replace line 4075** (`sexpr_display_nested`):
```cpp
builder->SetInsertPoint(sexpr_display_nested);
// Call recursive helper function for arbitrary depth
Value* initial_depth = ConstantInt::get(Type::getInt32Ty(*context), 1);
builder->CreateCall(display_sexpr_list_func, {sexpr_car_ptr_int, initial_depth});
builder->CreateBr(sexpr_elem_done);
```

**Replace line 4429** (`display_nested_list`):
```cpp
builder->SetInsertPoint(display_nested_list);
// Call recursive helper function for arbitrary depth
Value* initial_depth = ConstantInt::get(Type::getInt32Ty(*context), 1);
builder->CreateCall(display_sexpr_list_func, {car_ptr_int, initial_depth});
builder->CreateBr(element_done);
```

## Technical Details

### Why This Works for N-Depth

**Example**: `(+ (* (- x 1) (+ x 2)) 3)`

```
displaySExprList(outer_list, 0)
  ↓ prints "("
  ↓ element 0: string "+" → prints "+"
  ↓ element 1: nested list → displaySExprList(middle_list, 1)
      ↓ prints "("
      ↓ element 0: string "*" → prints "*"  
      ↓ element 1: nested list → displaySExprList(inner1_list, 2)
          ↓ prints "("
          ↓ element 0: string "-" → prints "-"
          ↓ element 1: variable "x" → prints "x"
          ↓ element 2: int 1 → prints "1"
          ↓ prints ")"
      ↓ element 2: nested list → displaySExprList(inner2_list, 2)
          ↓ prints "("
          ↓ element 0: string "+" → prints "+"
          ↓ element 1: variable "x" → prints "x"
          ↓ element 2: int 2 → prints "2"
          ↓ prints ")"
      ↓ prints ")"
  ↓ element 2: int 3 → prints "3"
  ↓ prints ")"
```

**Output**: `(+ (* (- x 1) (+ x 2)) 3)` ✅

### Depth Limit Safety

**Default**: 20 levels (more than sufficient for any practical derivative)
- Prevents stack overflow from circular references
- Easy to increase if needed
- Prints `[depth-limit]` warning if exceeded

### Performance

**Stack Usage**: O(depth) - each recursive call adds one stack frame
- Typical derivatives: depth 2-4
- Complex expressions: depth 5-10
- Safety limit: depth 20

**Time Complexity**: O(n) where n = total nodes in S-expression tree
- Same as iterative approach
- No performance penalty for recursion

## Implementation Checklist

### Phase 1: Helper Function Creation
- [ ] Add `display_sexpr_list_func` member variable to class
- [ ] Create `createDisplaySExprListFunction()` method
- [ ] Call from `createBuiltinFunctions()` after printf declaration
- [ ] Implement depth limit check
- [ ] Implement loop structure (current != null)
- [ ] Implement element type detection
- [ ] Implement recursive call for nested lists
- [ ] Implement cdr traversal
- [ ] Add proper return

### Phase 2: Replace Placeholders
- [ ] Replace line 4075 with helper call (depth=1)
- [ ] Replace line 4429 with helper call (depth=1)
- [ ] Ensure all required variables are in scope

### Phase 3: Testing
- [ ] Test depth-1: `(* 2 x)`
- [ ] Test depth-2: `(+ (* 1 2) 3)`
- [ ] Test depth-3: `(+ (* (- x 1) 2) 3)`
- [ ] Test depth-4+: Complex product rule derivatives
- [ ] Run full [`phase0_diff_fixes.esk`](tests/autodiff/phase0_diff_fixes.esk) suite

## Code Organization

### Where to Add Function

**Location 1**: Member declaration (line ~157)
```cpp
Function* arena_allocate_cons_cell_func;

// Nested S-expression display helper (N-depth recursive)
Function* display_sexpr_list_func;
```

**Location 2**: Function creation (line ~474, in `createBuiltinFunctions()`)
```cpp
// After printf function declaration
createDisplaySExprListFunction();
```

**Location 3**: Method implementation (new private method, after line ~474)
```cpp
void createDisplaySExprListFunction() {
    // Full implementation here
}
```

## Expected Test Results

### All Tests Should Show Complete Expansion

**Test 4** (Product rule):
```
Before: d/dx((x+1) * (x+2)) = (+ (...) (...))
After:  d/dx((x+1) * (x+2)) = (+ (* 1 (+ x 2)) (* (+ x 1) 1))
```

**Test 10** (Chain rule):
```
Before: d/dx(sin(2*x)) = (* (...) 2)
After:  d/dx(sin(2*x)) = (* (cos (* 2 x)) 2)
```

**Test 11** (Complex chain rule):
```
Before: d/dx(cos(x*x)) = (* (...) (...))
After:  d/dx(cos(x*x)) = (* (- (sin (* x x))) (* 2 x))
```

**Test 27** (Nested chain rule):
```
Before: d/dx(sin(x*x)) = (* (...) (...))
After:  d/dx(sin(x*x)) = (* (cos (* x x)) (* 2 x))
```

**Test 33** (Product of trig):
```
Before: d/dx(sin(x)*cos(x)) = (+ (...) (...))
After:  d/dx(sin(x)*cos(x)) = (+ (* (cos x) (cos x)) (* (sin x) (- (sin x))))
```

## Advantages Over Inline Approach

| Aspect | Inline Loops | Recursive Helper |
|--------|--------------|------------------|
| **Depth support** | Limited (hardcoded) | Unlimited (N-depth) |
| **Code clarity** | Complex nested blocks | Clean, standard pattern |
| **Maintenance** | Difficult | Easy |
| **Reusability** | No | Yes (call from anywhere) |
| **Stack usage** | O(1) | O(depth) - acceptable |
| **No placeholders** | ❌ | ✅ |

## Risk Assessment

**MEDIUM-LOW RISK**:
- ✅ Standard recursive pattern
- ✅ Depth limit prevents stack overflow
- ✅ Isolated helper function (easy to debug)
- ⚠️ Recursive LLVM IR functions need careful block management
- ✅ Can test incrementally (depth 1, then 2, then N)

## Success Criteria

1. ✅ Zero `(...)` placeholders at any depth
2. ✅ All 35 tests show complete S-expressions
3. ✅ Proper parenthesization at all levels
4. ✅ Correct spacing between elements
5. ✅ No crashes or stack overflows
6. ✅ Readable symbolic derivative output

## Implementation Order

1. **Create helper function** - new LLVM IR function
2. **Test helper alone** - call it directly from a simple test
3. **Replace placeholder 1** - line 4075
4. **Replace placeholder 2** - line 4429
5. **Integration test** - run full test suite

## Alternative: Iterative Stack-Based (If Recursion Issues)

If LLVM IR recursion causes issues, fallback to **explicit stack** using LLVM arrays:

```cpp
// Allocate stack array
Value* stack_size = ConstantInt::get(Type::getInt64Ty(*context), 100);
Value* stack = builder->CreateAlloca(Type::getInt64Ty(*context), stack_size, "stack");
Value* stack_top = builder->CreateAlloca(Type::getInt32Ty(*context), nullptr, "stack_top");

// Push/pop operations using array indexing
// Simulate DFS traversal with explicit stack
```

But **recursive helper is preferred** - cleaner and more maintainable.

## Timeline

- **Step 1** (Create helper): 45-60 minutes
- **Step 2** (Replace placeholders): 10-15 minutes
- **Step 3** (Testing): 20-30 minutes
- **Total**: ~90 minutes for complete N-depth support

## Next Steps

Ready to switch to **Code mode** to implement:
1. Create `createDisplaySExprListFunction()` method
2. Add recursive helper function to class
3. Replace both placeholder locations with helper calls
4. Test with full suite