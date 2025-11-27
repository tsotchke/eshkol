# Nested Lambda Closure Root Cause Analysis

## Problem Statement

Test 10 in `test_let_and_lambda.esk` returns pointer (5836398730) instead of calling the nested lambda and returning 15.

```scheme
(let ((make-adder (lambda (n) (lambda (x) (+ x n)))))
  (let ((add5 (make-adder 5)))
    (display (add5 10))))  ; Should display 15, shows pointer instead
```

## What We've Fixed So Far

✅ Lambda resolution works correctly:
- `add5_func` IS found in symbol_table
- Successfully cast to Function* (lambda_4)
- Function call IS being generated

✅ Captured variable detection works:
- System correctly identifies that `add5` (lambda_4) needs `captured_n`
- System searches for `n` in symbol tables
- System reports "Successfully loaded capture 'n'"

## The Actual Problem

❌ **Architecture Flaw**: Loading captured values from CALLING context

When `(add5 10)` is called:
1. System looks for captured variable `n` in CURRENT context (where add5 is being called)
2. But `n` only existed in `make-adder`'s scope when the lambda was created
3. The `n` it "finds" is either:
   - A stale reference from global_symbol_table
   - A coincidental variable with the same name
   - Not the actual captured value (5) from `make-adder`

## Evidence from Diagnostics

```
error: DEBUG: Closure call to add5 needs captured param: captured_n
error: DEBUG: Looking for captured variable: n
error: DEBUG: n not in symbol_table, checking global
error: DEBUG: Found n value, loading...
error: DEBUG: Successfully loaded capture 'n' for add5 from current context
```

Result: Returns pointer (5836398730) instead of 15

This means:
- We load SOME value for `n`
- We call the lambda with it
- But the lambda returns a pointer instead of computing `(+ 10 5)`

## Root Cause

**The current architecture fundamentally misunderstands closures:**

Current approach (WRONG):
```
1. Create lambda with free variables (e.g., lambda_4 needs 'n')
2. When calling lambda, look up 'n' in CURRENT context
3. Pass the found value as a parameter
```

Correct approach (NEEDED):
```
1. Create lambda with free variables
2. CAPTURE the values when lambda is CREATED (not when called)
3. STORE captured values with the closure
4. When calling lambda, LOAD stored values from closure environment
```

## Why It's Returning a Pointer

The lambda is being called, but one of two things is happening:

1. The captured `n` value is wrong (not 5), causing `(+ x n)` to produce garbage
2. The lambda isn't actually executing its body, just returning itself
3. The result of `(+ x n)` is being interpreted as a pointer instead of an integer

## Required Fix

This cannot be fixed with simple parameter passing. We need:

### Phase 1: Closure Environment Structure
- Create a struct to hold captured values
- Allocate this struct when lambda is returned from `make-adder`
- Store captured values (n=5) in this struct

### Phase 2: Modified Lambda Creation
- When `make-adder` returns lambda_4, bundle it with its environment
- Return a closure object (function pointer + environment pointer)

### Phase 3: Modified Lambda Calling
- When calling `add5`, unpack closure to get function + environment
- Extract captured values from environment (not from calling context)
- Pass captured values as parameters to the actual lambda function

##Current Status

The current fix improved lambda resolution but exposed the fundamental architecture limitation.

The "successful" load of `n` is actually loading the wrong value or a stale reference, which is why the result is still wrong.

## Next Steps

1. Add diagnostic to show WHAT value is loaded for `n` (should be 5)
2. Confirm the loaded value is wrong
3. Implement proper closure environment architecture
4. This is a multi-phase refactoring, not a quick fix
