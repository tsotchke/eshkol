# Eshkol REPL Phase 1 - Compiler-First Implementation Checklist

**Architecture**: Thin wrapper around Eshkol compiler using LLVM ORC JIT
**Result**: Automatic feature parity with compiler from day 1
**Time**: 9-14 hours (vs 30-35h for independent interpreter)

---

## Pre-Implementation Checklist

- [ ] Read [REPL_COMPILER_FIRST_ARCHITECTURE.md](REPL_COMPILER_FIRST_ARCHITECTURE.md)
- [ ] Confirm current compiler status (closures working, autodiff working)
- [ ] Verify LLVM ORC JIT is available in current LLVM version
- [ ] Commit any pending compiler work first

---

## Implementation Checklist

### Hour 1-2: JIT Context Setup

#### Create JIT Infrastructure
- [ ] Create `lib/repl/` directory
- [ ] Create `lib/repl/repl_jit.h`
  - [ ] Define `ReplJITContext` class
  - [ ] Add LLVM ORC LLJIT member
  - [ ] Add context and module management
  - [ ] Add symbol table for persistent definitions
- [ ] Create `lib/repl/repl_jit.cpp`
  - [ ] Implement constructor (initialize LLJIT)
  - [ ] Implement `execute()` method skeleton
  - [ ] Add basic error handling
- [ ] **Test**: LLJIT initializes without crashing

#### Verify LLVM Dependencies
- [ ] Check LLVM version supports ORC JIT v2
- [ ] Add LLVM ORC components to CMakeLists.txt
- [ ] Link LLVMOrcJIT, LLVMExecutionEngine
- [ ] **Test**: Builds successfully with JIT libraries

---

### Hour 2-3: Integration with Existing Compiler

#### Connect to Parser and Codegen
- [ ] Include existing `parser.h` in REPL
- [ ] Include existing `llvm_codegen.h` in REPL
- [ ] Create `ReplJITContext::compileAndExecute(const std::string& input)`
  - [ ] Parse input to AST using existing parser
  - [ ] Generate LLVM IR using existing codegen
  - [ ] Get module from codegen
  - [ ] Add module to LLJIT
- [ ] **Test**: Can compile simple expression `(+ 1 2)` to IR

#### First Execution
- [ ] Wrap each input in anonymous function:
  ```llvm
  define i64 @__repl_eval_N() {
      ; user's code here
      ret i64 %result
  }
  ```
- [ ] Look up function symbol in JIT
- [ ] Call function pointer
- [ ] Return result
- [ ] **Test**: Execute `(+ 1 2)` and get `3`

---

### Hour 3-4: Persistent Definitions

#### Handle `define` Across Evaluations
- [ ] Track global definitions in symbol table
- [ ] For `(define x value)`:
  - [ ] Compile as global variable or function
  - [ ] Store symbol in persistent map
  - [ ] Return value to user
- [ ] For subsequent references to `x`:
  - [ ] Codegen looks up in environment
  - [ ] Generates reference to global symbol
- [ ] **Test**:
  ```scheme
  (define x 10)  ; returns 10
  (+ x 5)        ; returns 15
  ```

#### Handle Lambda Definitions
- [ ] `(define square (lambda (n) (* n n)))`
  - [ ] Codegen generates closure struct (already implemented!)
  - [ ] Store closure pointer in symbol table
  - [ ] Return closure representation
- [ ] Calling defined lambda:
  - [ ] Look up closure from symbol table
  - [ ] Generate call through closure
- [ ] **Test**:
  ```scheme
  (define square (lambda (n) (* n n)))
  (square 5)  ; returns 25
  ```

---

### Hour 4-5: Result Extraction and Formatting

#### Create Value Wrapper
- [ ] Create `lib/repl/repl_value.h`
  - [ ] Define `ReplValue` class
  - [ ] Add type enum (NIL, BOOLEAN, INTEGER, DOUBLE, STRING, CLOSURE, LIST)
  - [ ] Add `fromJITResult(void* ptr, Type type)` factory
  - [ ] Add `toString()` for pretty printing
- [ ] Create `lib/repl/repl_value.cpp`
  - [ ] Implement `fromJITResult()` for each type
  - [ ] Implement `toString()` with formatting:
    - Integers: `42`
    - Doubles: `3.14159`
    - Booleans: `true` / `false`
    - Strings: `"hello"`
    - Closures: `<closure at 0x...>`
    - Lists: `(1 2 3 4)`
- [ ] **Test**: All value types display correctly

#### Handle Different Return Types
- [ ] Detect return type from codegen
- [ ] Cast JIT result appropriately:
  - `int64_t` for integers
  - `double` for floats
  - `bool` for booleans
  - `char*` for strings
  - `void*` for closures
- [ ] **Test**: Mixed-type expressions work

---

### Hour 5-6: Session Management

#### Create Session Class
- [ ] Create `lib/repl/repl_session.h`
  - [ ] Define `ReplSession` class
  - [ ] Add `ReplJITContext` member
  - [ ] Add `Parser` member (from existing code)
  - [ ] Add `CodeGenerator` member (from existing code)
  - [ ] Add evaluation history
- [ ] Create `lib/repl/repl_session.cpp`
  - [ ] Implement constructor (initialize compiler components)
  - [ ] Implement `eval(const std::string& input)` → `ReplValue`
    - [ ] Parse input
    - [ ] Compile to IR
    - [ ] Execute via JIT
    - [ ] Extract result
    - [ ] Add to history
  - [ ] Implement error handling
- [ ] **Test**: Full eval pipeline works

#### Handle Special Commands
- [ ] Implement `isSpecialCommand(input)`
- [ ] Add special commands:
  - [ ] `:help` - show available commands
  - [ ] `:type <expr>` - show type of expression
  - [ ] `:ir` - show LLVM IR for last eval
  - [ ] `:symbols` - list defined symbols
  - [ ] `:clear` - clear session
- [ ] **Test**: Special commands work

---

### Hour 6-7: Interactive Shell

#### Create REPL Main
- [ ] Create `exe/eshkol-repl.cpp`
  - [ ] Add banner/help text
  - [ ] Initialize readline
  - [ ] Create `ReplSession` instance
  - [ ] Implement main loop:
    ```cpp
    while (true) {
        char* input = readline("eshkol> ");
        if (!input) break;  // EOF (Ctrl+D)

        if (*input) add_history(input);

        try {
            ReplValue result = session.eval(input);
            std::cout << result.toString() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }

        free(input);
    }
    ```
  - [ ] Handle exit conditions: `(exit)`, Ctrl+D, Ctrl+C
- [ ] **Test**: Interactive loop works

#### Add Readline Features
- [ ] History (up/down arrows)
- [ ] Basic tab completion (optional for Phase 1)
- [ ] Multi-line input detection (optional for Phase 1)
- [ ] **Test**: Can recall previous inputs with up arrow

---

### Hour 7-8: CMake & Build

#### Update Build System
- [ ] Update `CMakeLists.txt`:
  ```cmake
  # REPL library
  add_library(eshkol-repl-lib
      lib/repl/repl_jit.cpp
      lib/repl/repl_value.cpp
      lib/repl/repl_session.cpp
  )

  target_link_libraries(eshkol-repl-lib
      eshkol-static  # Existing compiler!
      LLVMOrcJIT
      LLVMExecutionEngine
      LLVMSupport
  )

  # REPL executable
  add_executable(eshkol-repl exe/eshkol-repl.cpp)
  target_link_libraries(eshkol-repl eshkol-repl-lib readline)

  install(TARGETS eshkol-repl DESTINATION bin)
  ```
- [ ] Add readline detection:
  ```cmake
  find_library(READLINE_LIBRARY readline)
  if(NOT READLINE_LIBRARY)
      message(WARNING "readline not found, using basic input")
  endif()
  ```
- [ ] Add optional build flag: `-DBUILD_REPL=ON/OFF`
- [ ] **Test**: Builds successfully

#### Build and Install
- [ ] Build:
  ```bash
  cmake --build build --target eshkol-repl
  ```
- [ ] Test executable:
  ```bash
  ./build/exe/eshkol-repl
  ```
- [ ] Install (optional):
  ```bash
  cmake --install build
  ```
- [ ] **Test**: Installed binary works

---

### Hour 8-9: Testing & Verification

#### Test All Features (Should Work Automatically!)
- [ ] **Arithmetic**:
  ```scheme
  (+ 1 2 3)           ; 6
  (- 10 3)            ; 7
  (* 4 5)             ; 20
  (/ 10 4)            ; 2.5
  (% 10 3)            ; 1
  ```

- [ ] **Variables**:
  ```scheme
  (define x 42)       ; 42
  x                   ; 42
  (define y (+ x 8))  ; 50
  ```

- [ ] **Lambdas & Closures**:
  ```scheme
  (define square (lambda (n) (* n n)))  ; <closure>
  (square 5)                            ; 25

  (define make-adder (lambda (n)
                        (lambda (x) (+ x n))))
  (define add5 (make-adder 5))         ; <closure>
  (add5 10)                             ; 15
  ```

- [ ] **Control Flow**:
  ```scheme
  (if #t 1 2)         ; 1
  (if #f 1 2)         ; 2
  (if (> 5 3) "yes" "no")  ; "yes"
  ```

- [ ] **Let Bindings**:
  ```scheme
  (let ((x 5) (y 10)) (+ x y))  ; 15
  (let ((x 1))
    (let ((x 2)) x))             ; 2
  ```

- [ ] **Lists** (if implemented in compiler):
  ```scheme
  (list 1 2 3 4)      ; (1 2 3 4)
  (cons 1 (cons 2 '()))  ; (1 2)
  (car (list 1 2 3))  ; 1
  (cdr (list 1 2 3))  ; (2 3)
  ```

- [ ] **Higher-Order Functions**:
  ```scheme
  (map (lambda (x) (* x 2)) (list 1 2 3))  ; (2 4 6)
  (filter (lambda (x) (> x 2)) (list 1 2 3 4))  ; (3 4)
  (fold + 0 (list 1 2 3 4))                ; 10
  ```

- [ ] **Automatic Differentiation** (the big test!):
  ```scheme
  (define f (lambda (x) (* x x x)))
  (derivative f 2.0)   ; 12.0 (3*x^2 at x=2)

  (gradient (lambda (v)
              (+ (* (car v) (car v))
                 (* (cadr v) (cadr v))))
            (list 3.0 4.0))  ; (6.0 8.0)
  ```

- [ ] **Math Functions**:
  ```scheme
  (sin 0.0)     ; 0.0
  (cos 0.0)     ; 1.0
  (exp 1.0)     ; 2.71828...
  (log 2.71828) ; 1.0
  (sqrt 16.0)   ; 4.0
  ```

#### Test Error Handling
- [ ] Parse errors don't crash:
  ```scheme
  (+ 1 2    ; Missing paren
  ```
- [ ] Unbound variables give clear errors:
  ```scheme
  undefined-var
  ```
- [ ] Type errors are caught:
  ```scheme
  (+ 1 "hello")
  ```
- [ ] Division by zero handled:
  ```scheme
  (/ 1 0)
  ```
- [ ] REPL continues after all errors

---

## Verification Checklist

### Compiler Features Work in REPL
- [ ] All features that work in compiled code work in REPL
- [ ] No need to reimplement anything
- [ ] New compiler features automatically available

### Performance
- [ ] JIT compilation is reasonably fast (< 100ms for simple expressions)
- [ ] No memory leaks (check with valgrind if available)
- [ ] Can run thousands of evaluations without issues

### Code Quality
- [ ] No warnings with `-Wall -Wextra`
- [ ] Code is well-commented
- [ ] File headers are correct
- [ ] Follows existing code style

### Independence
- [ ] REPL can be disabled in build (`-DBUILD_REPL=OFF`)
- [ ] REPL code doesn't modify compiler internals
- [ ] Breaking REPL doesn't break compiler

---

## Post-Implementation Checklist

### Documentation
- [ ] Update README with REPL usage
- [ ] Document special commands
- [ ] Add example session
- [ ] Note any limitations

### Git
- [ ] Commit REPL separately from compiler work:
  ```bash
  git add lib/repl/ exe/eshkol-repl.cpp CMakeLists.txt
  git commit -m "Add compiler-first REPL with JIT execution

  - Uses LLVM ORC JIT for interactive evaluation
  - All compiler features work automatically
  - Autodiff, closures, lists all supported day 1
  - Completely independent from compiler internals
  - Foundation for enhanced REPL features

  Part of: REPL development (independent track)
  Architecture: Compiler-first with ORC JIT
  Time: ~10 hours
  Status: Full feature parity with compiler"
  ```

### Testing
- [ ] Create `tests/repl/` directory
- [ ] Add comprehensive test suite
- [ ] Test script that runs all examples

---

## What Works on Day 1 (Zero Extra Implementation!)

Because we're using the compiler directly via JIT:

✅ All arithmetic: `+`, `-`, `*`, `/`, `%`, `^`
✅ All comparisons: `<`, `>`, `<=`, `>=`, `=`, `!=`
✅ All logical ops: `and`, `or`, `not`
✅ Variables: `define`, `set!`
✅ Control flow: `if`, `cond`, `when`, `unless`
✅ Let bindings: `let`, `let*` (once implemented in compiler)
✅ Lambdas: Full closure support
✅ Lists: `list`, `cons`, `car`, `cdr`, `map`, `filter`, `fold`
✅ **Automatic differentiation**: `derivative`, `gradient` 🎉
✅ Math functions: `sin`, `cos`, `exp`, `log`, `sqrt`, etc.
✅ Vectors: Vector operations (if implemented)
✅ Mixed types: Tagged value system

**This is the whole language, available immediately!**

---

## Success Metrics

**Phase 1 is successful if:**

1. ✅ REPL compiles and runs without crashes
2. ✅ All compiler features work in REPL (automatic parity)
3. ✅ Autodiff works (`derivative` and `gradient`)
4. ✅ Closures work (define lambda, call it)
5. ✅ Errors are recoverable (REPL continues)
6. ✅ History works (up/down arrows)
7. ✅ Doesn't interfere with compiler development
8. ✅ Completed in ~10-14 hours

**Phase 1 fails if:**

1. ❌ Takes significantly longer (>20 hours)
2. ❌ Doesn't achieve feature parity
3. ❌ Autodiff doesn't work
4. ❌ Too buggy to use
5. ❌ Delays v1.0-foundation

---

## Time Budget (Compiler-First)

| Task | Planned | Actual | Notes |
|------|---------|--------|-------|
| JIT context setup | 2-3h | | LLVM ORC LLJIT |
| Compiler integration | 1-2h | | Parser + Codegen |
| Persistent definitions | 1-2h | | Symbol management |
| Result extraction | 1-2h | | Value formatting |
| Session management | 1h | | Eval pipeline |
| Interactive shell | 1-2h | | Readline loop |
| CMake & build | 1h | | Link LLVM JIT |
| Testing | 1-2h | | Verify all features |
| **Total** | **9-14h** | | |
| Buffer | +2-3h | | Unexpected issues |
| **Realistic Total** | **12-17h** | | Plan for 2 days |

**Compare to original plan**: 30-35h for full parity → **saves 15-20 hours!**

---

## Next Steps

1. **Now**: Ensure compiler is in good state (commit closure work)
2. **Hour 1**: Start with JIT context setup
3. **Track progress**: Update actual times in table above
4. **Test incrementally**: Each hour has testable milestone
5. **Ship when working**: Don't gold-plate Phase 1

---

## Emergency Rollback (Same as Before)

If taking too long or blocking v1.0:

### Option A: Pause REPL
- Commit current work
- Return to v1.0 priorities
- Resume after release

### Option B: Ship Incomplete
- Document current state
- Mark as experimental
- Complete in v1.1

### Option C: Abandon
- Save in branch
- Delete from main
- Revisit later

**REPL is not critical for v1.0 - don't let it block the release!**

---

## Quick Reference: Example Session

```scheme
$ eshkol-repl
Eshkol v0.1.1 Interactive REPL
Type (exit) or Ctrl+D to quit, :help for commands

eshkol> (+ 1 2 3)
6

eshkol> (define factorial
          (lambda (n)
            (if (= n 0)
                1
                (* n (factorial (- n 1))))))
<closure>

eshkol> (factorial 5)
120

eshkol> (define f (lambda (x) (* x x x)))
<closure>

eshkol> (derivative f 2.0)
12.0

eshkol> (map (lambda (x) (derivative f x))
             (list 1.0 2.0 3.0))
(3.0 12.0 27.0)

eshkol> :type factorial
(Integer -> Integer)

eshkol> :symbols
factorial : (Integer -> Integer)
f : (Double -> Double)

eshkol> (exit)
Goodbye!
$
```

---

**Ready to build the compiler-first REPL!** 🚀

This architecture gives us:
- ✅ Automatic feature parity
- ✅ Less implementation time
- ✅ Maintainability (no porting)
- ✅ Foundation for future enhancements

Let's do this!
