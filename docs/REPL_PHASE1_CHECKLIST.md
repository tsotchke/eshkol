# Eshkol REPL Phase 1 - Implementation Checklist

**Goal**: Working terminal REPL in 6-8 hours
**Status**: Ready to implement
**Dependencies**: None (completely independent)

---

## Pre-Implementation Checklist

- [ ] Read [REPL_ARCHITECTURE_AND_VISION.md](REPL_ARCHITECTURE_AND_VISION.md) (understand long-term vision)
- [ ] Read [REPL_PHASE1_MINIMAL_IMPLEMENTATION.md](REPL_PHASE1_MINIMAL_IMPLEMENTATION.md) (understand Phase 1 scope)
- [ ] Confirm **NOT** blocking v1.0-foundation work
- [ ] Commit any pending compiler work first

---

## Implementation Checklist

### Day 1: Core Implementation (6-8 hours)

#### Hour 1: Value Representation
- [ ] Create `lib/repl/` directory
- [ ] Create `lib/repl/repl_value.h`
  - [ ] Define `ReplValue` class
  - [ ] Add `Type` enum (NIL, BOOLEAN, INTEGER, DOUBLE, STRING, CONS)
  - [ ] Add static factory methods (makeInt, makeDouble, etc.)
  - [ ] Add accessors (asInt, asDouble, etc.)
  - [ ] Add `toString()` method
- [ ] Create `lib/repl/repl_value.cpp`
  - [ ] Implement constructors
  - [ ] Implement `toString()` with proper formatting
  - [ ] Implement accessors with type checking
- [ ] **Test**: Can create and print values

#### Hour 2: Environment
- [ ] Create `lib/repl/repl_env.h`
  - [ ] Define `Environment` class
  - [ ] Add `define()`, `lookup()`, `set()` methods
  - [ ] Add parent pointer for nested scopes
- [ ] Create `lib/repl/repl_env.cpp`
  - [ ] Implement `define()` (insert into map)
  - [ ] Implement `lookup()` (search parent chain)
  - [ ] Implement `set()` (update existing binding)
- [ ] **Test**: Can store and retrieve bindings

#### Hour 3: Evaluator Structure
- [ ] Create `lib/repl/repl_eval.h`
  - [ ] Define `Evaluator` class
  - [ ] Add `eval()` methods
  - [ ] Add private helper methods (evalDefine, evalIf, etc.)
- [ ] Create `lib/repl/repl_eval.cpp`
  - [ ] Implement constructor (setup global env)
  - [ ] Implement main `eval()` dispatcher
  - [ ] Add basic error handling
- [ ] **Test**: Structure compiles

#### Hour 4: Core Evaluation
- [ ] Implement literal evaluation
  - [ ] ESHKOL_INTEGER → ReplValue::makeInt
  - [ ] ESHKOL_FLOAT → ReplValue::makeDouble
  - [ ] ESHKOL_BOOLEAN → ReplValue::makeBool
  - [ ] ESHKOL_NULL → ReplValue::makeNil
- [ ] Implement variable evaluation
  - [ ] ESHKOL_VARIABLE → env->lookup()
  - [ ] Error if unbound
- [ ] Implement `define`
  - [ ] Extract variable name and value
  - [ ] Evaluate value
  - [ ] Store in environment
- [ ] **Test**: `(define x 42)` and `x` work

#### Hour 5: Operators & Control Flow
- [ ] Implement arithmetic operators
  - [ ] `+` (handle multiple args, int/double coercion)
  - [ ] `-` (handle negation and subtraction)
  - [ ] `*` (multiplication)
  - [ ] `/` (division, always returns double)
- [ ] Implement `if`
  - [ ] Evaluate condition
  - [ ] Check truthiness (only #f and '() are false)
  - [ ] Evaluate consequent or alternate
- [ ] Implement `let`
  - [ ] Create new environment with parent
  - [ ] Evaluate all bindings in outer environment
  - [ ] Evaluate body in new environment
- [ ] **Test**: Arithmetic and control flow work

#### Hour 6: REPL Loop & Main
- [ ] Create `exe/eshkol-repl.cpp`
  - [ ] Add banner/help text
  - [ ] Implement readLine() (with/without readline)
  - [ ] Implement main loop:
    - [ ] Read input
    - [ ] Check for special commands (exit, help)
    - [ ] Parse to AST
    - [ ] Evaluate
    - [ ] Print result
    - [ ] Handle errors
- [ ] **Test**: Interactive loop works

#### Hour 7: CMake & Build
- [ ] Update `CMakeLists.txt`
  - [ ] Add eshkol-repl-lib target
  - [ ] Add eshkol-repl executable target
  - [ ] Link only parser from eshkol-static (NOT whole compiler!)
  - [ ] Add readline detection
  - [ ] Add install target
- [ ] Build
  ```bash
  cmake --build build --target eshkol-repl
  ```
- [ ] **Test**: Builds successfully

#### Hour 8: Testing & Polish
- [ ] Manual testing
  - [ ] Test arithmetic: `(+ 1 2 3)` → `6`
  - [ ] Test variables: `(define x 10)`, `x` → `10`
  - [ ] Test if: `(if (> x 5) "yes" "no")` → `"yes"`
  - [ ] Test let: `(let ((y 5)) (+ x y))` → `15`
  - [ ] Test errors: unbound variable, parse error, etc.
  - [ ] Test history (up/down arrows)
- [ ] Fix any bugs found
- [ ] Clean up error messages
- [ ] Add more helpful output
- [ ] **Test**: All manual tests pass

---

## Verification Checklist

### Functionality Tests
- [ ] REPL starts and shows prompt
- [ ] Can evaluate arithmetic: `(+ 1 2 3)` → `6`
- [ ] Can define variables: `(define x 10)` → `10`
- [ ] Can reference variables: `x` → `10`
- [ ] Can use if: `(if #t 1 2)` → `1`
- [ ] Can use let: `(let ((y 5)) y)` → `5`
- [ ] Can nest let: `(let ((x 1)) (let ((x 2)) x))` → `2`
- [ ] Can use begin: `(begin (define z 1) z)` → `1`
- [ ] History works (up arrow recalls)
- [ ] Can exit cleanly: `(exit)`

### Error Handling Tests
- [ ] Parse error gives message (not crash)
- [ ] Unbound variable gives message
- [ ] Wrong type in operator gives message
- [ ] Division by zero gives message
- [ ] All errors are recoverable (REPL continues)

### Code Quality Checks
- [ ] No warnings when compiling with `-Wall`
- [ ] No memory leaks (run under valgrind if possible)
- [ ] Code is commented
- [ ] File headers are correct
- [ ] CMake target is independent

---

## Post-Implementation Checklist

### Documentation
- [ ] Update [REPL_PHASE1_MINIMAL_IMPLEMENTATION.md](REPL_PHASE1_MINIMAL_IMPLEMENTATION.md) with any discoveries
- [ ] Document any deviations from plan
- [ ] Note any bugs/limitations for Phase 2
- [ ] Write quick usage guide in README

### Git
- [ ] Commit REPL code separately from compiler work
  ```bash
  git add lib/repl/ exe/eshkol-repl.cpp
  git commit -m "Add Phase 1 minimal REPL

  - Tree-walk interpreter for basic expressions
  - Arithmetic, variables, let, if, begin
  - Readline support for history
  - Completely independent of main compiler
  - Foundation for future enhancements

  Part of: REPL development (independent track)
  Phase: 1 (Minimal)
  Time: 6-8 hours
  Status: Working, ready for Phase 2"
  ```

### Testing
- [ ] Create `tests/repl/` directory
- [ ] Add test file with examples
- [ ] Document expected behavior

### Communication
- [ ] Demo REPL to team/users
- [ ] Gather feedback
- [ ] Plan Phase 2 priorities based on feedback

---

## Phase 2 Planning Checklist

**After Phase 1 is done and v1.0-foundation is shipped:**

- [ ] Review Phase 1 limitations
- [ ] Prioritize Phase 2 features:
  - [ ] Lambdas (most requested?)
  - [ ] Lists (needed for data?)
  - [ ] Compiler integration (for performance?)
  - [ ] Better error messages?
  - [ ] Multi-line input?
- [ ] Estimate Phase 2 timeline (2-4 weeks?)
- [ ] Create Phase 2 implementation plan

---

## Emergency Rollback Plan

**If Phase 1 is taking too long or blocking v1.0:**

### Option A: Pause REPL, Focus on v1.0
- [ ] Commit current REPL work (even if incomplete)
- [ ] Return to v1.0-foundation work
- [ ] Resume REPL after v1.0 ships

### Option B: Ship Incomplete REPL
- [ ] Document current state
- [ ] Mark as "experimental"
- [ ] Ship with v1.0 as bonus feature
- [ ] Complete in v1.1

### Option C: Abandon Phase 1
- [ ] Save work in branch
- [ ] Delete from main tree
- [ ] Revisit later
- **This is OK!** REPL is not critical for v1.0.

---

## Quick Reference: What Works in Phase 1

```scheme
;; ✅ Works
(+ 1 2 3)                    ; Arithmetic
(define x 10)                ; Variables
(if (> x 5) "yes" "no")     ; Conditionals
(let ((y 5)) (+ x y))       ; Let bindings
(begin (define z 1) z)      ; Sequencing
(display "Hello")            ; Output
(newline)                    ; Output

;; ❌ Doesn't Work (Phase 2)
(lambda (x) (* x 2))         ; Lambdas
(list 1 2 3)                 ; List construction
(map + '(1 2) '(3 4))       ; Higher-order functions
(derivative f 5.0)           ; Autodiff
(let* ((a 1) (b a)) b)      ; let*
(for i 0 10 ...)            ; Iteration
```

---

## Time Budget (Realistic)

| Task | Planned | Actual | Notes |
|------|---------|--------|-------|
| Value representation | 1.5h | | |
| Environment | 1h | | |
| Evaluator structure | 1h | | |
| Core evaluation | 1h | | |
| Operators & control | 1h | | |
| REPL loop | 1h | | |
| CMake & build | 0.5h | | |
| Testing & polish | 1h | | |
| **Total** | **8h** | | |
| Buffer | +2h | | For unexpected issues |
| **Realistic Total** | **10h** | | |

**Plan for 2 focused days** if working 4-5 hours/day.

---

## Success Definition

**Phase 1 is successful if:**

1. ✅ REPL works for basic expressions
2. ✅ Doesn't interfere with compiler development
3. ✅ Provides value for experimentation
4. ✅ Foundation is solid for Phase 2+
5. ✅ Team is excited to use it

**Phase 1 is NOT successful if:**

1. ❌ Delays v1.0-foundation release
2. ❌ Creates technical debt in compiler
3. ❌ Is too buggy to use
4. ❌ Doesn't actually save time vs file-based workflow

**Be honest about success/failure and adjust accordingly!**

---

## Next Steps

1. **Now**: Review all documentation
2. **Before coding**: Ensure v1.0 compiler work is committed
3. **Start coding**: Follow checklist hour-by-hour
4. **During coding**: Update checklist with actuals
5. **After Phase 1**: Commit, test, gather feedback
6. **Return to v1.0**: Ship foundation release
7. **Later**: Phase 2 (after v1.0 ships)

---

**Ready? Let's build this!** 🚀
