# Continuation operands and synchronous LLVM emission

Candidate `9594cdce` crashes compiling stdlib on macOS ARM with AppleClang
15.0.0.15000309. Diagnostic run
[34305851970](https://github.com/tsotchke/eshkol/actions/runs/34305851970)
locates the fault at `llvm::GetElementPtrInst::Create + 216`, reading address
`0x8`. Upstream Clang 15.0.7 on ARM reproduces the same instruction and fault
when building `llvm_codegen.cpp`; Clang 15 parser/type objects with AppleClang
16 codegen do not reproduce it. The minimal language case is
`(display (string #\A))` with stdlib loading disabled.

LLDB locates the bad operand at the string codepoint GEP in `codegenCallTask`.
The array pointer is valid; the single index in `ArrayRef<Value*>` is null.
Passing `ConstantInt::get(...)` implicitly constructs that view by binding a
temporary converted `Value*` to `ArrayRef`'s singleton-reference constructor.
A reduced two-translation-unit continuation reproduces the failure without
LLVM: a derived-pointer factory passed to a borrowed base-pointer view loses
its value after suspension. Clang 15's generated resume function retains a
view of a native stack slot from before suspension, calls the factory, then
discards its return value rather than filling that slot.

The emission boundary now takes scalar GEP indices and call arguments by
value. `CodegenIRBuilder` constructs LLVM's borrowed views from named pointer
storage inside the synchronous emission method. The compiler's central builder
uses this API for every lowering path; no string-specific lowering rule changes.
Inherited range, initializer-list, empty-argument and operand-bundle APIs remain
available. This adds neither allocation nor another continuation frame.

Parser/type continuations, their scheduler, result carriers, source guards,
evaluation order, insertion-point behavior and stack configuration are unchanged.
There are no compiler-version branches, compiler flags, CI exclusions or
optimization-disabling attributes in the change. All temporary signal-handler
instrumentation was removed; that file matches `9594cdce` byte for byte.

Regression evidence on macOS ARM64 with LLVM 21:

- The new C++ regression checks scalar GEP/inbounds-GEP and both scalar call
  overloads after repeated suspension, checks existing range/list/empty APIs,
  and verifies the resulting LLVM module. It passes with Clang 15. Replacing
  the adapter with the raw LLVM builder fails its GEP operand-identity assertion.
- Clang 15 compiler-pass objects linked into the native compiler now produce
  nonempty stdlib object and bitcode files. All ten stages of the actual 8 MiB
  stdlib/JIT/AOT gate pass, including 16,000 nested expressions and string
 construction yielding `Aλ` followed by evaluation-order marker `123`.
- AppleClang 16 Release: strict type-system suite 56/56, parser suite 31/31,
  architecture/parser/operand/mutation CTests 9/9, and AD/capture/tail-position
  CTests 10/10 pass.
- Upstream Clang 21 Release: operand, parser explicit-stack, parser compilation
  and callable-routing CTests 4/4 pass, including the same ten-stage gate.

ICC was queried before diagnosis and refused context as BLIND because the
registered `eshkol_lang` index and memory stores were missing. The diagnosis
and evidence above use live source, compiler disassembly, LLDB and executed
regressions rather than an ICC readiness claim.
