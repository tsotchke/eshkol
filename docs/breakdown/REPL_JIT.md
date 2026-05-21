# Eshkol REPL JIT System

**Status**: Production (v1.2.1-scale)
**Source file**: `lib/repl/repl_jit.cpp` (2063 lines)
**Header**: `lib/repl/repl_jit.h`

---

## 1. Overview

The Eshkol REPL is backed by LLVM's ORC (On-Request Compilation) JIT framework, specifically the `LLJIT` API introduced in LLVM 11. Rather than interpreting Eshkol code, the REPL compiles each expression to native machine code via the same codegen pipeline used by the AOT compiler (`eshkol-run`), then executes the result within the same process. This gives REPL sessions full native performance and byte-for-byte ABI compatibility with precompiled artifacts (stdlib.o, user libraries).

Key design commitments:

- **Shared arena**: a single `arena_t` persists across all REPL evaluations, enabling heap-allocated values (strings, closures, cons cells, tensors) to be referenced across evaluation boundaries.
- **Incremental module accumulation**: each evaluation produces a separate LLVM module. Symbols defined in earlier modules are injected as external declarations into later ones, preserving cross-evaluation visibility.
- **Hot reload**: redefining a function replaces the previous JIT symbol. Forward-reference stubs allow mutually recursive definitions to be entered interactively.
- **ABI parity**: `CodeGenOptLevel::None` is set on the JIT compiler to match stdlib.o, which is compiled at `-O0`. Mismatches cause silent struct-argument corruption on ARM64.

---

## 2. Architecture

```
eshkol-repl binary
│
├── ReplJITContext (lib/repl/repl_jit.cpp)
│   ├── LLJIT instance                 — ORC JIT with 1 compile thread
│   ├── MainJITDylib                   — symbol namespace for all modules
│   │   ├── DynamicLibrarySearchGenerator  — resolves from process image
│   │   └── absoluteSymbols map        — explicit runtime registrations
│   ├── ThreadSafeModule per eval      — each expression gets its own module
│   ├── shared_arena_ (arena_t*)       — persistent heap across evals
│   ├── defined_lambdas_               — var_name → {lambda_N, arity}
│   ├── defined_globals_               — registered global variable names
│   ├── registered_lambdas_            — dedup set for JIT lookups
│   ├── forward_ref_slots_             — __repl_fwd_* pointer-to-pointer stubs
│   └── eval_counter_                  — monotonically increasing eval index
│
├── eshkol-static (linked into binary)
│   ├── Arena runtime          — arena_create, arena_allocate, ...
│   ├── Exception runtime      — eshkol_raise, eshkol_make_exception, ...
│   ├── AD runtime             — arena_allocate_dual_number, tape ops, ...
│   ├── BLAS/matmul            — eshkol_matmul_f64, eshkol_blas_available
│   ├── Parallel runtime       — eshkol_parallel_map/fold/filter/for_each
│   ├── Hash table             — arena_hash_table_create, hash_table_set, ...
│   └── Global state           — __repl_shared_arena, __global_arena,
│                                 __current_ad_tape, __ad_mode_active, ...
│
└── stdlib.o (pre-compiled object file)
    └── loaded via addObjectFile() — instant availability, no JIT recompile
```

Process symbol resolution works because `eshkol-repl` is linked with `-force_load` (macOS) or `--whole-archive` (Linux) on `eshkol-static.a`, combined with `-export_dynamic`. This ensures every symbol in the static library is exported from the process image and visible to `DynamicLibrarySearchGenerator`.

---

## 3. JIT Initialization

`ReplJITContext::initializeJIT()` — `lib/repl/repl_jit.cpp`

### 3.1 LLVM Target Setup

```cpp
InitializeNativeTarget();
InitializeNativeTargetAsmPrinter();
InitializeNativeTargetAsmParser();
sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
```

`LoadLibraryPermanently(nullptr)` loads the current process as a dynamic library, making every exported symbol (arena functions, printf, math builtins, etc.) resolvable by JIT-compiled code. This is the first layer of symbol resolution; explicit registration in `registerRuntimeSymbols()` is the second layer for symbols that may not survive dead-stripping on macOS.

### 3.2 Target Machine Builder

```cpp
auto jtmb = orc::JITTargetMachineBuilder::detectHost();
jtmb->setRelocationModel(Reloc::PIC_);
jtmb->setCodeGenOptLevel(CodeGenOptLevel::None);  // CRITICAL: must match stdlib.o
```

`detectHost()` queries the OS/CPU for the exact feature set (AVX-512, NEON, AMX, etc.) so the JIT uses the same instruction set as the precompiled stdlib.o. The relocation model is set to PIC to match the `-fPIC` flag used when compiling stdlib.o.

### 3.3 LLJIT Construction

```cpp
auto jit_or_err = LLJITBuilder()
    .setJITTargetMachineBuilder(std::move(*jtmb))
    .setNumCompileThreads(1)
    .create();
```

Single compile thread avoids races on the `ReplJITContext` data structures, which are not internally synchronized. Thread safety is the responsibility of callers (the REPL loop is single-threaded).

### 3.4 Symbol Generator

```cpp
auto generator = orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
    jit_->getDataLayout().getGlobalPrefix());
main_dylib.addGenerator(std::move(*generator));
```

This is the fallback resolver: if a symbol is not found in `MainJITDylib` after explicit registration, ORC queries the process image. The global prefix (underscore on macOS) is passed so mangled names are matched correctly.

### 3.5 Shared Arena Initialization

```cpp
__repl_shared_arena.store(arena_create(8192));
shared_arena_ = __repl_shared_arena.load();
```

The arena is created with 8 KB initial block size. Because `__repl_shared_arena` is a global registered as a `ADD_DATA_SYMBOL`, every JIT-compiled module that references it receives the same pointer. Heap-allocated values created in evaluation N remain reachable in evaluation N+1 as long as they are stored in a live global variable.

---

## 4. Symbol Registration

`ReplJITContext::registerRuntimeSymbols()` — `lib/repl/repl_jit.cpp`

Two macros drive registration:

```cpp
#define ADD_SYMBOL(name)      // JITSymbolFlags::Callable | Exported
#define ADD_DATA_SYMBOL(name) // JITSymbolFlags::Exported only
```

All entries go into a single `orc::SymbolMap` which is committed to `MainJITDylib` via `main_dylib.define(orc::absoluteSymbols(symbols))`.

Categories registered (~60+ symbols):

| Category | Examples |
|---|---|
| Arena allocation | `arena_create`, `arena_allocate`, `arena_allocate_cons_cell`, `arena_allocate_vector_with_header`, `arena_allocate_tensor_full` |
| Cons cell accessors | `arena_tagged_cons_get_int64`, `arena_tagged_cons_set_tagged_value`, etc. |
| Exception handling | `eshkol_raise`, `eshkol_make_exception`, `eshkol_push_exception_handler`, `g_current_exception` |
| Automatic differentiation | `arena_allocate_dual_number`, `arena_allocate_tape`, `arena_tape_add_node`, `arena_tape_reset` |
| Closure management | `arena_allocate_closure_env`, `arena_allocate_closure_with_header` |
| Tensor allocation | `arena_allocate_tensor_with_header`, `arena_allocate_tensor_full` |
| BLAS | `eshkol_matmul_f64`, `eshkol_blas_available`, `eshkol_blas_get_threshold` |
| Hash tables | `arena_hash_table_create`, `hash_table_set`, `hash_table_get`, `hash_keys_equal` |
| OALR regions | `region_create`, `region_push`, `region_pop`, `region_allocate` |
| Ref-counted shared | `shared_allocate`, `shared_retain`, `shared_release`, `weak_ref_create` |
| Parallel execution | `eshkol_parallel_map`, `eshkol_parallel_fold`, `eshkol_parallel_filter`, `__eshkol_register_parallel_workers` |
| Math (C stdlib) | `sin`, `cos`, `sqrt`, `pow` (cast to `double(*)(double)` to resolve overloads) |
| Global state | `__repl_shared_arena`, `__global_arena`, `__current_ad_tape`, `__ad_mode_active`, `__eshkol_argc`, `__eshkol_argv` |

Math functions require explicit casts because `std::sin` et al. are overloaded:

```cpp
typedef double (*MathFunc1)(double);
symbols[ES.intern("sin")] = {
    orc::ExecutorAddr::fromPtr((void*)(MathFunc1)&std::sin),
    JITSymbolFlags::Callable | JITSymbolFlags::Exported
};
```

---

## 5. Stdlib Integration

### 5.1 Fast Path: addObjectFile

`ReplJITContext::loadStdlib()` — `lib/repl/repl_jit.cpp`

```cpp
std::string stdlib_obj_path = findStdlibObject();
auto buffer_or_err = MemoryBuffer::getFile(stdlib_obj_path);
auto err = jit_->addObjectFile(main_dylib, std::move(*buffer_or_err));
```

`addObjectFile` relocates and links the pre-compiled `stdlib.o` into `MainJITDylib` directly. This is several orders of magnitude faster than JIT-recompiling the stdlib from source on each REPL startup. The object file is found by searching: current directory, `build/`, `../build/`, `/usr/local/lib/eshkol/`, `/usr/lib/eshkol/`, and paths relative to the executable.

### 5.2 Symbol Discovery via .bc Metadata

`ReplJITContext::registerStdlibSymbols()` — `lib/repl/repl_jit.cpp`

```cpp
auto mod_or_err = parseBitcodeFile(buf->getMemBufferRef(), *ctx);
Module& mod = **mod_or_err;
for (auto& F : mod) {
    if (F.isDeclaration() || F.hasInternalLinkage()) continue;
    size_t arity = F.arg_size();
    eshkol_repl_register_function(name.c_str(), 0, arity);
    defined_lambdas_[name] = {name, arity};
}
for (auto& G : mod.globals()) {
    if (name ends with "_sexpr") {
        eshkol_repl_register_symbol(name.c_str(), 0);
        defined_globals_.insert(name);
    }
}
```

LLVM bitcode preserves the pre-scalarization function signatures, so `F.arg_size()` reports the true arity of each stdlib function — this is not available from the object file alone. Results (237 functions, 305 globals as of v1.1-accelerate) are registered into the REPL compiler's symbol tables so subsequent user code can call stdlib functions without redeclaring them.

Skipped during discovery: `__eshkol_*` internal names, `lambda_*` closures, `main`.

### 5.3 Module Name Tracking

After successful stdlib loading, 22 module names are inserted into `loaded_modules`:

```
stdlib, core.io, core.operators.arithmetic, core.operators.compare,
core.logic.predicates, core.logic.types, core.logic.boolean,
core.functional.compose, core.functional.curry, core.functional.flip,
core.control.trampoline, core.list.{compound,generate,transform,query,
sort,higher_order,search,convert}, core.strings, core.json,
core.data.{csv,base64}
```

This prevents `(require core.list.transform)` from triggering a redundant load.

---

## 6. Evaluation Pipeline

### 6.1 Single Expression: `execute(ast)`

`ReplJITContext::execute()` — `lib/repl/repl_jit.cpp`

Full pipeline for one REPL expression:

```
1. Handle require/import/provide at AST level (load dependencies first)
2. Pre-register lambda variables (for hot reload tracking)
3. eshkol_generate_llvm_ir(ast, 1, "__repl_module_N")
   → calls full Eshkol codegen pipeline (parser, type checker, llvm_codegen.cpp)
   → returns LLVMModuleRef
4. llvm::unwrap(c_module) — C API → C++ API conversion
5. injectPreviousSymbols(module)
   → scans defined_lambdas_, defined_globals_
   → emits external declarations so linker resolves them from earlier modules
6. Scan module functions:
   → fill pending lambda_* slots in defined_lambdas_
   → associate var names with implementation function names
7. Rename entry point: __top_level / main → __repl_eval_N
8. eshkol_extract_module_context_for_jit(c_module)
   → releases module from g_llvm_modules ownership
   → returns unique_ptr<LLVMContext> paired with the module
9. addModule(unique_ptr<Module>, unique_ptr<LLVMContext>)
   → forward ref resolution, hot reload, verifyModule, ThreadSafeModule wrap
   → jit_->addIRModule(tsm)
10. lookupSymbol("__repl_eval_N") → JIT address
11. call eval_func(0, nullptr) as i32(i32, char**)
12. After execution: read _sexpr globals (initialized inside entry function)
    → eshkol_repl_register_sexpr for homoiconic metadata
13. Register lambdas and globals into REPL symbol tables
```

### 6.2 Batch Execution: `executeBatch(asts, silent)`

`ReplJITContext::executeBatch()` — `lib/repl/repl_jit.cpp`

Used when loading a module file (which may define many functions). All ASTs are passed to `eshkol_generate_llvm_ir(asts.data(), asts.size(), ...)` in a single call, enabling forward references between functions within the same file. The entry function is renamed `__repl_batch_eval_N`.

### 6.3 Module Path Resolution

`resolveModulePath()` — `lib/repl/repl_jit.cpp`

Dot-separated module names are converted to filesystem paths:

```
"core.list.transform" → "core/list/transform.esk"
```

Search order:
1. `base_dir/core/list/transform.esk` (relative to caller)
2. `g_lib_dir/core/list/transform.esk` (discovered lib/ directory)
3. Paths from `$ESHKOL_PATH` (colon-separated)
4. Legacy fallbacks: `lib/...`, `../lib/...`

`g_lib_dir` is discovered once via `findLibDir()` by searching upward from the executable location for the `lib/` subdirectory containing `core/`.

### 6.4 Typed Results: `executeTagged(ast)`

`ReplJITContext::executeTagged()` — `lib/repl/repl_jit.cpp`

Returns `eshkol_tagged_value_t` (16 bytes: `{type:u8, flags:u8, reserved:u16, padding:u32, data:u64}`). The raw `int64_t` return from `execute()` is reinterpreted based on the AST's `inferred_hott_type` field:

- `packed_type = ast->inferred_hott_type` (bits 0-15 = TypeId.id, bits 16-23 = universe, bits 24-31 = flags)
- `BuiltinTypes::Int64` / `Integer` / `Number` → `ESHKOL_VALUE_INT64`
- `BuiltinTypes::Float64` / `Real` → `ESHKOL_VALUE_DOUBLE` (raw bits reinterpreted as `double`)
- `BuiltinTypes::Boolean` → `ESHKOL_VALUE_BOOL`
- `BuiltinTypes::Complex` → `ESHKOL_VALUE_COMPLEX` (heap pointer to `{f64, f64}`)
- `BuiltinTypes::Function` / `Closure` → `ESHKOL_VALUE_CALLABLE`
- TypeIds >= 500 (dynamically allocated function types) → `ESHKOL_VALUE_CALLABLE`
- TypeIds >= 1000 (user-defined types) → `ESHKOL_VALUE_HEAP_PTR` (fallback)
- Universe types / proof terms (TypeU0, Eq, LessThan) → `ESHKOL_VALUE_NULL` (runtime-erased)

When `inferred_hott_type == 0` (type inference did not run), the function falls back to examining `ast->type` directly (ESHKOL_INT64, ESHKOL_DOUBLE, ESHKOL_BOOL, etc.).

---

## 7. Hot Reload

`ReplJITContext::addModule()` — `lib/repl/repl_jit.cpp`

### 7.1 Symbol Removal for Redefinition

Before adding a new module, previously defined user symbols that appear in the incoming module are removed from `MainJITDylib`:

```cpp
for (auto& func : *module) {
    if (func.getLinkage() == GlobalValue::LinkOnceODRLinkage) continue;
    if (fname.starts_with("__repl_") || fname.starts_with("lambda_")) continue;
    if (defined_lambdas_.count(fname) == 0) continue;  // only remove known user symbols
    to_remove.insert(jit_->mangleAndIntern(func.getName()));
}
```

Skipped from removal:
- `LinkOnceODRLinkage` — shared builtins like `builtin_+_2arg` that must not be replaced
- `lambda_*` — each lambda gets a fresh unique name, no collision
- `__repl_*` — REPL infrastructure functions

### 7.2 Forward Reference Stubs

When module A calls function `B` that has not yet been defined, the codegen emits a reference to `__repl_fwd_B` (a global holding a function pointer). `addModule` handles both sides:

**First occurrence (B not yet defined):**

```cpp
// Allocate a pointer slot initialized to the stub function
void** ptr_slot = new void*;
*ptr_slot = reinterpret_cast<void*>(&__repl_forward_ref_stub);
forward_ref_slots_["__repl_fwd_B"] = ptr_slot;
// Register the slot address as the symbol
main_dylib.define(absoluteSymbols({{intern("__repl_fwd_B"),
    ExecutorAddr::fromPtr(ptr_slot), Exported}}));
pending_forward_refs_.insert("__repl_fwd_B");
```

**When B is defined (later evaluation):**

```cpp
// Scan for __repl_fwd_* globals with initializers pointing to real functions
if (name.starts_with("__repl_fwd_") && pending_forward_refs_.count(name)) {
    auto* func = dyn_cast<Function>(gv.getInitializer());
    forward_ref_updates.push_back({name, func->getName().str()});
    gv.setInitializer(nullptr);  // convert to external declaration
}
// After addIRModule, look up B's address and patch the pointer slot
*ptr_slot = func_symbol->toPtr<void*>();
pending_forward_refs_.erase("__repl_fwd_B");
```

The stub function (`__repl_forward_ref_stub`, line 445) raises an `ESHKOL_EXCEPTION_ERROR` if called before the real definition is loaded.

---

## 8. CodeGenOptLevel ABI Bug

This is the most critical correctness constraint in the JIT system.

### 8.1 The Problem

`JITTargetMachineBuilder::detectHost()` defaults to `CodeGenOptLevel::Default` (-O2). The pre-compiled `stdlib.o` is built at `-O0` (`CodeGenOptLevel::None`). On ARM64 (Apple Silicon), these two optimization levels produce different calling conventions for struct arguments.

The Eshkol tagged value is a 5-field struct:

```c
typedef struct {
    uint8_t  type;     // field 0
    uint8_t  flags;    // field 1
    uint16_t reserved; // field 2
    uint32_t padding;  // field 3
    union { int64_t int_val; double double_val; uint64_t ptr_val; uint64_t raw_val; } data; // field 4
} eshkol_tagged_value_t;
```

At `-O2`, the ARM64 backend uses LLVM's struct scalarization pass, which decomposes the struct into separate registers. At `-O0`, the struct is passed on the stack as a contiguous 16-byte block. When the JIT generates a call to a stdlib function compiled at `-O0`, and the JIT itself uses `-O2` conventions, the callee reads the third argument from the wrong stack slot — observing all-zero bytes.

### 8.2 The Fix

```cpp
// lib/repl/repl_jit.cpp
jtmb->setCodeGenOptLevel(CodeGenOptLevel::None);
```

This single line ensures the JIT's `ConcurrentIRCompiler` generates the same ABI as `stdlib.o`. The fix was identified by observing that 3+ argument stdlib calls (e.g., `(substring str 2 5)`) returned garbage — the first two arguments arrived correctly but the third was zero.

### 8.3 Generality

The same issue would affect any precompiled object file that passes structs by value. The root constraint is: **the JIT compiler's `CodeGenOptLevel` must exactly match the optimization level used when compiling any `.o` file loaded via `addObjectFile`**.

---

## 9. Platform-Specific Linking

### 9.1 macOS (Darwin)

```cmake
target_link_options(eshkol-repl PRIVATE
    -Wl,-force_load,$<TARGET_FILE:eshkol-static>
    -Wl,-export_dynamic
)
```

`-force_load` forces all archive members of `eshkol-static.a` to be linked, preventing the linker's dead-stripping from removing runtime functions that are only referenced from JIT-generated code (which the linker cannot see at link time). `-export_dynamic` marks all symbols as exported from the process image so `DynamicLibrarySearchGenerator` can resolve them.

### 9.2 Linux (ELF)

```cmake
target_link_options(eshkol-repl PRIVATE
    -Wl,--whole-archive $<TARGET_FILE:eshkol-static> -Wl,--no-whole-archive
    -Wl,-export-dynamic
)
```

`--whole-archive` / `--no-whole-archive` is the ELF equivalent of `-force_load`, wrapping just `eshkol-static.a`. `-export-dynamic` (one hyphen on Linux) adds all symbols to the dynamic symbol table.

### 9.3 XLA Runtime Note

The XLA backend runtime (`lib/backend/xla/xla_runtime.cpp`) is an archive member that would otherwise be dead-stripped on both platforms. The `force_load` / `whole-archive` flags ensure it is retained so `eshkol_xla_*` symbols resolve in REPL sessions that use the XLA backend.

---

## 10. Debug Environment Variables

| Variable | Effect |
|---|---|
| `ESHKOL_DUMP_REPL_IR=1` | Dumps the LLVM IR of each JIT module to stderr before `addIRModule`. Equivalent to `--dump-ir` in the AOT compiler. |
| `ESHKOL_DEBUG_DL=1` | Prints DataLayout string and target triple for both the module and the LLJIT instance. Used to diagnose ABI mismatches. |

Both checks occur at `lib/repl/repl_jit.cpp`:

```cpp
if (getenv("ESHKOL_DUMP_REPL_IR")) {
    module->print(errs(), nullptr);
}
if (getenv("ESHKOL_DEBUG_DL")) {
    std::cerr << "[REPL] Module DataLayout: " << module->getDataLayoutStr() << std::endl;
    std::cerr << "[REPL] LLJIT DataLayout: " << jit_->getDataLayout().getStringRepresentation() << std::endl;
}
```

The DataLayout strings for the module (generated by `eshkol_generate_llvm_ir`) and the LLJIT instance (from `detectHost()`) must be identical. A mismatch indicates that the codegen is targeting a different triple than the JIT.

---

## 11. Code Examples

### 11.1 Basic REPL Session

```scheme
;; Evaluation 1 — defines a function, eval_counter=0 → module __repl_module_0
eshkol> (define (square x) (* x x))
;; ReplJITContext::execute() runs:
;;   registerLambdaVar("square")
;;   eshkol_generate_llvm_ir → module with lambda_0(x) and global square
;;   entry renamed → __repl_eval_0
;;   addModule → JIT
;;   lookupSymbol("lambda_0") → register as REPL function
;;   eval_counter_ = 1

;; Evaluation 2 — references symbol from eval 0
eshkol> (square 7)
;; injectPreviousSymbols emits: declare i64 @square(...), declare %tagged* @square_func(...)
;; codegen sees square as known REPL function, generates call
;; __repl_eval_1(0, null) returns 49
;; => 49
```

### 11.2 Hot Reload

```scheme
eshkol> (define (greet name) (string-append "Hello, " name "!"))
;; lambda_2 registered

eshkol> (greet "world")
;; => "Hello, world!"

eshkol> (define (greet name) (string-append "Hi, " name "."))
;; Hot reload:
;;   registered_lambdas_.erase("lambda_2")  — clear old entry
;;   symbol_table_.erase("greet")           — invalidate cached address
;;   addModule detects greet in defined_lambdas_, removes from JIT
;;   lambda_5 registered at new address

eshkol> (greet "world")
;; => "Hi, world."
```

### 11.3 Forward References

```scheme
;; Entering mutually recursive functions interactively:
eshkol> (define (even? n) (if (= n 0) #t (odd? (- n 1))))
;; odd? not yet defined — codegen emits __repl_fwd_odd? global
;; forward_ref_slots_["__repl_fwd_odd?"] → stub pointer slot

eshkol> (define (odd? n) (if (= n 0) #f (even? (- n 1))))
;; addModule detects __repl_fwd_odd? initializer pointing to lambda_odd?
;; Patches pointer slot: *ptr_slot = jit_->lookup("lambda_odd?")

eshkol> (even? 10)
;; => #t
```

### 11.4 Stdlib Integration

```scheme
eshkol> (require stdlib)
;; loadStdlib():
;;   addObjectFile(main_dylib, stdlib.o)  — instant, no recompile
;;   registerStdlibSymbols():
;;     parseBitcodeFile("stdlib.bc")
;;     discovers 237 functions, 305 globals
;;     loaded_modules += {stdlib, core.io, core.list.*, ...}

eshkol> (map (lambda (x) (* x x)) '(1 2 3 4 5))
;; map resolved from stdlib.o, lambda compiled by JIT, list from stdlib
;; => (1 4 9 16 25)
```

### 11.5 Inspecting JIT IR

```bash
ESHKOL_DUMP_REPL_IR=1 eshkol-repl
eshkol> (define (fib n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))
# LLVM IR for the module is printed to stderr:
# define i64 @__repl_eval_0(i32 %argc, i8** %argv) {
# entry:
#   ...
# }
```

---

## 12. Data Structures

| Field | Type | Purpose |
|---|---|---|
| `jit_` | `std::unique_ptr<orc::LLJIT>` | The ORC JIT instance |
| `eval_counter_` | `size_t` | Monotonic counter, appended to module/function names |
| `shared_arena_` | `arena_t*` | Local copy of `__repl_shared_arena` for cleanup |
| `defined_lambdas_` | `unordered_map<string, pair<string,size_t>>` | var_name → {lambda_N, arity} |
| `registered_lambdas_` | `unordered_set<string>` | Set of lambda function names already registered |
| `defined_globals_` | `unordered_set<string>` | Global variable names tracked in REPL context |
| `symbol_table_` | `unordered_map<string, uint64_t>` | Cached JIT lookup results |
| `forward_ref_slots_` | `unordered_map<string, void**>` | __repl_fwd_* pointer-to-pointer slots |
| `pending_forward_refs_` | `unordered_set<string>` | Forward refs awaiting resolution |
| `module_exports_` | `unordered_map<string, unordered_set<string>>` | Module name → exported symbols |
| `private_symbols_` | `unordered_set<string>` | Symbols defined but not exported in loaded modules |

---

## 13. See Also

- `lib/repl/repl_jit.h` — `ReplJITContext` class declaration and `LambdaInfo` struct
- `exe/eshkol-run.cpp` — AOT compiler entry point; `process_requires()` uses `collect_all_submodules()` for precompiled library discovery
- `lib/backend/llvm_codegen.cpp` — `eshkol_generate_llvm_ir()` entry point (line ~14097), `eshkol_repl_enable()`, `eshkol_repl_register_function()`, `eshkol_extract_module_context_for_jit()`
- `lib/core/arena_memory.h` — Arena allocator API used by all runtime symbol categories
- `docs/breakdown/COMPILER_ARCHITECTURE.md` — Full LLVM backend pipeline overview
- `docs/breakdown/MEMORY_MANAGEMENT.md` — OALR arena model that the shared arena implements
- `CHANGELOG.md` — v1.1-accelerate section covers JIT `CodeGenOptLevel` fix and stdlib duplicate-symbol fix (`LinkOnceODRLinkage`)

---

## 14. What Landed in v1.2 / v1.2.1

The v1.2 release introduced two architectural moves in the REPL JIT path:
a machine-driven framing mode (`--machine` / `-m`) that turns a single
REPL process into a long-lived JIT-warm worker, and a preferred bitcode
loading path (`stdlib.bc` via `addIRModule`) that replaces the legacy
object-file loading path (`stdlib.o` via `addObjectFile`) for ABI parity
with subsequent REPL batches. This section documents both, plus the
variadic-attribute round-trip across the stdlib boundary, the explicit
`__eshkol_lib_init__` invocation that v1.2 added, and the ORC compile-thread
sizing change.

### 14.1 The `--machine` / `-m` Warm-Worker Mode

The motivating use case was a sister project (Noesis) running an
80-frame asteroids benchmark via repeated `eshkol-run -r`. Each
invocation paid a ~30 s JIT-init + stdlib-load tax. The team needed
either a persistent `eshkol-server` daemon or a `(persist-runtime!)`
primitive that would snapshot LLVM ORC state to disk. The latter is
impractical: LLVM ORC holds relocations, RuntimeDyld symbol tables, and
materialized object files in process memory with no clean serialization
contract. A long-running process is the correct architectural answer.

`exe/eshkol-repl.cpp` §764–851 introduces the `--machine` flag. When
supplied, it:

- Forces non-interactive mode (no banner, no readline prompt, no color
  output): `g_interactive = false` (line 803).
- Implies `--stdlib` (`load_stdlib = true` on line 788), because the
  point of warm-worker mode is to be JIT-warm with stdlib pre-loaded.
- Emits `EREPL READY\n` to stderr exactly once, *after* JIT init and
  stdlib load complete (lines 848–851):

  ```cpp
  if (machine_mode) {
      std::fputs(EREPL_READY, stderr);
      std::fflush(stderr);
  }
  ```

- After every top-level form's evaluation, emits `EREPL DONE\n` (success)
  or `EREPL FAIL\n` (exception/crash) to stderr, with `fflush(stderr)`
  so the client gets it without buffering delay (lines 1071–1075,
  1079–1083).

The three framing constants are verbatim from `exe/eshkol-repl.cpp`
§774–776:

```cpp
static constexpr const char* EREPL_READY = "EREPL READY\n";
static constexpr const char* EREPL_DONE  = "EREPL DONE\n";
static constexpr const char* EREPL_FAIL  = "EREPL FAIL\n";
```

All program output stays on stdout. The framing markers go to stderr
only, so binary writes like P6 PPM, JSON-L emission, or any custom
machine-readable output round-trips cleanly through the warm worker.

The companion shell driver `scripts/eshkol-warm-demo.sh` illustrates
the protocol from the client side (named pipes for stdin/stdout/stderr
separation, READY-then-loop pattern). Real clients use language-native
subprocess libraries (Python `subprocess.Popen`, Node.js
`child_process.spawn`, etc.).

**Measured performance.** Five trivial `(display N)(newline)` forms:

- 5× cold `eshkol-run -r` invocations: 3:59.74 wall (~48 s per form,
  dominated by JIT init each time).
- 5 forms via a single `eshkol-repl --machine`: 47.35 s wall (~9.5 s
  amortized; one cold start, five essentially-free forms).

This is ~5× at N=5 and the marginal cost per additional warm form is
sub-millisecond. For an 80-frame benchmark, this is the difference
between 64 minutes and ~30 s of overhead.

**Caveats.**

- `eshkol-repl` evaluates one top-level form per submission (existing
  behavior: `parse_string` reads the first form and drops the rest).
  Clients should send one form per submission unit, or use multi-line
  forms with balanced parens.
- Sentinels are line-prefix matches, not length-prefixed framing.
  Robust enough for line-based protocols. If a form happens to emit
  literal `EREPL DONE` text on stdout, the sentinel still does not
  collide because the sentinel is stderr-only.

### 14.2 stdlib.bc as the Preferred Loading Path

Prior to v1.2, `ReplJITContext::loadStdlib()` loaded the precompiled
`stdlib.o` via `jit_->addObjectFile`. That worked, but it crossed the
native object-file ABI boundary for Eshkol aggregate values, which
made it fragile in the same way the `CodeGenOptLevel::None` fix in §8
addressed (ARM64 struct scalarization at -O2 vs -O0).

The v1.2 preferred path uses `stdlib.bc` (LLVM bitcode), loaded as a
`ThreadSafeModule` via `addIRModule`. Bitcode keeps stdlib's IR in the
same ORC compilation pipeline as subsequent user-code modules, so they
share the JIT's TargetMachine and ABI exactly. The legacy
`addObjectFile` path remains as fallback for installed layouts that
have not yet shipped `stdlib.bc`.

The dispatch is in `lib/repl/repl_jit.cpp` §1804–1918. First branch
(§1846–1892), bitcode fast path:

```cpp
std::string stdlib_bc_path = findStdlibBitcode();
if (!stdlib_bc_path.empty()) {
    auto buffer_or_err = MemoryBuffer::getFile(stdlib_bc_path);
    if (buffer_or_err) {
        auto module_context = std::make_unique<LLVMContext>();
        auto mod_or_err = parseBitcodeFile(
            buffer_or_err->get()->getMemBufferRef(), *module_context);
        if (mod_or_err) {
            std::unique_ptr<Module> stdlib_module = std::move(*mod_or_err);
            if (stdlib_module->getDataLayout().isDefault()) {
                stdlib_module->setDataLayout(jit_->getDataLayout());
            }
            auto ts_ctx = orc::ThreadSafeContext(std::move(module_context));
            auto tsm = ThreadSafeModule(std::move(stdlib_module), ts_ctx);
            if (auto err = jit_->addIRModule(std::move(tsm))) { /* fallback */ }
            else { registerStdlibSymbols();
                   run_stdlib_initializers(stdlib_bc_path);
                   stdlib_loaded_ = true; return true; }
        }
    }
}
```

Second branch (§1894–1918), legacy object-file path:

```cpp
std::string stdlib_obj_path = findStdlibObject();
if (!stdlib_obj_path.empty()) {
    auto buffer_or_err = MemoryBuffer::getFile(stdlib_obj_path);
    auto err = jit_->addObjectFile(main_dylib, std::move(*buffer_or_err));
    if (!err) { registerStdlibSymbols(); run_stdlib_initializers(stdlib_obj_path);
                stdlib_loaded_ = true; return true; }
}
```

`findStdlibBitcode()` (§1629–1660) searches a platform-specific list
ending at `/usr/local/lib/eshkol/stdlib.bc` and `/usr/lib/eshkol/stdlib.bc`
on POSIX. `findStdlibObject()` (§1597–1626) does the same for `stdlib.o`.

### 14.3 `__eshkol_lib_init__` Explicit Invocation

In AOT-compiled binaries, `main()` calls `__eshkol_lib_init__(arena)` to
populate every module-local `define` after stdlib link time — e.g.
`core.data.base64`'s `base64-chars` alphabet, `core.json`'s `JSON_SPACE`,
`core.testing`'s counters. ORC does not call this Eshkol-level library
initializer for REPL preloads (it has no equivalent of an Eshkol-defined
constructor list), so the REPL must invoke it explicitly.

`lib/repl/repl_jit.cpp` §1813–1830:

```cpp
auto run_stdlib_initializers = [this](const std::string& source_path) {
    if (auto lib_init = jit_->lookup("__eshkol_lib_init__")) {
        using LibInitFn = void (*)(void*);
        auto fn = reinterpret_cast<LibInitFn>(lib_init->getValue());
        fn(shared_arena_);
    } else {
        consumeError(lib_init.takeError());
        std::cerr << "[REPL] Warning: stdlib loaded from " << source_path
                  << " but __eshkol_lib_init__ was not found — module "
                     "constants remain zero-initialised." << std::endl;
    }
    // ...
};
```

Without this invocation, REPL users observed `base64-encode` returning
all zeros, JSON parsing returning empty objects, and similar
silent-zero failures: the stdlib functions resolved correctly, but
their module-local data globals were still zero-initialized.

The same lambda also explicitly invokes
`__eshkol_init_parallel_workers` (§1837–1843), which the AOT path runs
through `llvm.global_ctors`. The REPL never relied on ORC ctor
execution, so this dependency was made explicit to keep the parallel
runtime consistent across all stdlib loader paths (bitcode, object
file, or any future loader).

### 14.4 The `eshkol-variadic` Attribute

Scheme-level variadic functions (e.g. `(make-list n . xs)`, `(partial
f . args)`, `(log-info fmt . args)`) are erased to a fixed-arity
single-`tagged_value`-argument signature in LLVM IR. Without an
out-of-band marker, the REPL cannot distinguish a Scheme variadic
function from a 1-argument fixed function at the bitcode level, so a
REPL call like `(make-list 3 'x)` would silently pack the second arg
as the entire args list, and the stdlib body's `(cdr args)` would read
garbage.

The fix introduces an LLVM function attribute that carries the
fixed-parameter count. The codegen emits it in `llvm_codegen.cpp` §3739:

```cpp
function->addFnAttr("eshkol-variadic", std::to_string(num_params));
```

`ReplJITContext::registerStdlibSymbols` (`repl_jit.cpp` §1731–1742)
reads it back when discovering stdlib symbols from `stdlib.bc`:

```cpp
if (F.hasFnAttribute("eshkol-variadic")) {
    auto attr = F.getFnAttribute("eshkol-variadic");
    size_t fixed_params = 0;
    try { fixed_params = std::stoul(attr.getValueAsString().str()); }
    catch (...) { fixed_params = 0; }
    eshkol_repl_register_variadic_function(name.c_str(), fixed_params, true);
}
```

The discovery loop logs the count of variadic entries restored:
`[REPL] Discovered <funcs> functions, <globals> globals, <variadic>
variadic entries from stdlib.bc`.

### 14.5 Module-Level Tagged-Value Globals

The same `registerStdlibSymbols` loop (`repl_jit.cpp` §1756–1786) also
registers stdlib's top-level data globals — `define`s whose right-hand
side is a value (not a lambda) and which lower to a `GlobalVariable` of
type `%eshkol_tagged_value`. Without this registration, references to
such names from inside a `delay`-generated lambda or any
free-var-captured closure body fall through to the "Undefined variable"
path. The check is:

```cpp
llvm::Type* gt = G.getValueType();
if (gt && gt->isStructTy()) {
    auto* st = llvm::cast<llvm::StructType>(gt);
    if (st->hasName() && st->getName() == "eshkol_tagged_value") {
        eshkol_repl_register_symbol(name.c_str(), 0);
        defined_globals_.insert(name);
    }
}
```

This was discovered while implementing SRFI 41 streams (issue #174):
`stream-null` is the empty stream defined as a plain value at module
top level. `(stream-cons head (delay stream-null))` failed because
`stream-null` inside the delay's lambda was not findable. The fix is
general: any stdlib module-level tagged-value global that is referenced
from a downstream user lambda needs to be in the REPL's symbol address
table so that `lib/backend/llvm_codegen.cpp` §7480 emits an external
`GlobalVariable` load, not a free-variable capture.

The capture rules in REPL mode therefore are: data globals from stdlib
are *not* captured as free variables in a user lambda; they are loaded
directly via an external `GlobalVariable` declaration emitted at codegen
time, with the address resolved by the JIT's `DynamicLibrarySearchGenerator`
or by the entry registered via `g_repl_symbol_addresses`.

### 14.6 ORC Compile-Thread Sizing

`lib/repl/repl_jit.cpp` §492–499 sets the LLJIT's compile thread count.
v1.1 hardcoded `setNumCompileThreads(1)`, which serialized every IR
materialization. Profiling on a 24-core M2 Max showed 100% of worker
time inside `MaterializationTask::run → CloneFunctionInto`, with eight
parallel tasks delivering only 1.02× speedup. v1.2 makes the count
dynamic:

```cpp
unsigned compile_threads = std::thread::hardware_concurrency();
if (compile_threads == 0) compile_threads = 4;
compile_threads = std::max(1u, compile_threads / 2);
if (compile_threads > 16) compile_threads = 16;
if (const char* env = std::getenv("ESHKOL_JIT_COMPILE_THREADS")) {
    long v = std::strtol(env, nullptr, 10);
    if (v > 0 && v <= 64) compile_threads = static_cast<unsigned>(v);
}
```

Default is `hardware_concurrency()/2`, clamped to `[1, 16]`. The cap
protects against memory pressure: each compile thread holds its own
`LLVMContext` plus a cloned module, and at 16 threads on a 24-core
M2 Max the working set is already several hundred megabytes.
`ESHKOL_JIT_COMPILE_THREADS` overrides at runtime.

This change reduces ORC compile-thread contention proportionally but
does not eliminate the per-symbol materialization-lock contention. The
real fix for that lives in `parallel_codegen.cpp` §`eshkol_parallel_map`,
which runs `item[0]` synchronously before submitting the rest of the
list (the JIT warmup phase documented in PARALLEL_COMPUTING.md
§13.2).

### 14.7 Loaded Module Set

`registerStdlibSymbols` (§1665–1689) inserts a fixed list of 23 module
names into `loaded_modules` before running discovery. Verbatim from
source:

```
stdlib, core.io, core.operators.arithmetic, core.operators.compare,
core.logic.predicates, core.logic.types, core.logic.boolean,
core.functional.compose, core.functional.curry, core.functional.flip,
core.control.trampoline, core.list.compound, core.list.generate,
core.list.transform, core.list.query, core.list.sort,
core.list.higher_order, core.list.search, core.list.convert,
core.strings, core.json, core.data.csv, core.data.base64
```

A `(require core.list.transform)` after stdlib load is therefore a
no-op (the module name is already in `loaded_modules` so the require
short-circuits). Any module not in this set still triggers normal
file-based loading via `resolveModulePath`. The hardcoded list is
intentional: discovery from bitcode names the *symbols*, not the
*modules*, so the module set has to be known a priori. Future work
could parse `(provide)` declarations from stdlib's .esk sources to
generate the list at build time, but the current arrangement is the
v1.2 form.
