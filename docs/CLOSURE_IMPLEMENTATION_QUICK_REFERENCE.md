# Closure Implementation Quick Reference

## The Fix in 3 Steps

### 1. Store Captures at Creation (30 min)
**File**: `lib/backend/llvm_codegen.cpp` line 5783
**What**: Add loop to store captured values in arena
**Code**: See CLOSURE_PRODUCTION_IMPLEMENTATION_PLAN.md Phase 1.1

### 2. Load Captures from Storage (60 min)
**File**: `lib/backend/llvm_codegen.cpp` lines 4176-4259
**What**: Replace capture loading logic
**Code**: See CLOSURE_PRODUCTION_IMPLEMENTATION_PLAN.md Phase 2.1

### 3. Fix Autodiff (45 min)
**Files**: 
- `lib/backend/llvm_codegen.cpp` lines 9236-9287 (gradient)
- `lib/backend/llvm_codegen.cpp` line 8892 (derivative)
**What**: Use capture storage in gradient/derivative
**Code**: See CLOSURE_PRODUCTION_IMPLEMENTATION_PLAN.md Phase 3

## Root Cause

**Current Bug**: Captures loaded from CALLING context
```scheme
(let ((make-adder (lambda (n) (lambda (x) (+ x n)))))
  (let ((add5 (make-adder 5)))
    (add5 10)))  ; Looks for 'n' HERE (wrong!) instead of where lambda was created
```

**Fix**: Store at creation, load from storage
```
Creation: lambda_1_capture_n ← 5 (store in arena)
Call:     load lambda_1_capture_n → 5 (from arena)
```

## Key Architecture

```
Lambda Creation (in make-adder):
  detect: n is free variable
  store:  arena[lambda_1_capture_n] = 5
  create: lambda_1(x, captured_n)

Lambda Call (calling add5):
  lookup: storage = arena[lambda_1_capture_n]
  load:   value = storage  (value = 5)
  call:   lambda_1(10, 5) → 15
```

## Testing Checklist

- [ ] Simple closure works: `(let ((n 5)) (lambda (x) (+ x n)))`
- [ ] Nested lambda works: `(lambda (n) (lambda (x) (+ x n)))`
- [ ] Multiple captures work: `(lambda (a b) (lambda (x) (+ x a b)))`
- [ ] With gradient works: `(gradient (lambda (v) (* (vref v 0) n)) ...)`
- [ ] All 110 existing tests pass
- [ ] Test 10 in test_let_and_lambda.esk passes

## Debug Commands

```bash
# Compile
cmake --build build

# Test specific file
./build/eshkol-run tests/autodiff/test_let_and_lambda.esk

# Check for capture storage
grep "Stored capture" <output>

# Check for capture loading  
grep "Loaded capture" <output>

# Run all tests
./scripts/run_all_tests.sh
```

## If It Doesn't Work

1. Check capture storage keys in debug output
2. Verify storage is in global_symbol_table
3. Check captured value types (should be tagged_value)
4. Verify arena is initialized before lambda creation
5. Check LLVM verification errors (no cross-function references)

## Success Metrics

**Before**: Test 10 returns pointer (5836398730)
**After**: Test 10 returns 15

**Time**: 6-7 hours actual work + 1-2 hour buffer
**Risk**: Low (targeted changes, well-tested infrastructure)