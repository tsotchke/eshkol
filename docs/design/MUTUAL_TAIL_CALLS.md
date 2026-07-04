# Proper mutual tail calls (ESH-0102 / ESH-0171)

Status: **AArch64 done (ESH-0102, this PR). x86 / arm32 / riscv64 scoped as an ABI
change (ESH-0171).**

## The requirement

R7RS section 3.5 requires *proper tail calls*: a procedure call in tail position
must run in constant stack, **including a tail call from one procedure to
another** (mutual recursion). The canonical Scheme idiom of a state machine
written as a set of mutually tail-calling functions (`even?`/`odd?`, a lexer's
state functions, a ping/pong 2-cycle) depends on this. Self tail recursion has
always been O(1) in Eshkol (the named-let / self-recursion loop transform); the
gap was tail calls to *other* functions.

## What shipped (ESH-0102, this PR)

`codegenCall` already collected non-self tail-position call sites into
`mutual_tail_call_sites_`. Previously it emitted an LLVM `TCK_Tail` marker there
on LLVM >= 18 — but `tail` is only a *hint* the backend may ignore, so mutual
recursion still grew a native frame per hop and overflowed the stack (~200–300k
hops → SIGBUS/SIGILL). The fix emits a real `musttail` call (which the backend is
*required* to lower as a stack-reusing jump), guarded by:

1. **Signature + calling-convention match** — same return type, same param
   count/types, same calling convention, non-varargs, `args.size()` equals the
   callee's param count.
2. **No pointer-into-frame arguments** — `musttail` is illegal if any argument is
   a pointer (it may alias an alloca in the caller, whose frame is torn down
   before the callee runs). This excludes higher-order tail calls that forward a
   freshly-built closure / capture slot (e.g. `none?`→`all?` passing
   `(negate pred)`, or named-let capture-forwarding allocas). Those fall back to
   an ordinary bounded call — correct, just not O(1).
3. **Target arch accepts an aggregate-return `musttail`** — see below.

Result on AArch64 (the primary target): `mutual_tail2` / `mutual_tail3` run to
5,000,000+ hops in O(1) stack under both `-r` and AOT; `even?`/`odd?` to 10^7;
`tests/stress/found/mutual_tail_1e7.esk` prints `OK ping` at ~225 MB (JIT) /
~27 MB (AOT). stdlib builds clean.

## Why x86 / arm32 / riscv64 are deferred (ESH-0171)

Every Eshkol function returns the tagged value **by value** as the 16-byte struct
`{i8, i8, i16, i32, i64}` (`type_system.cpp`). On LLVM 21, the x86, 32-bit arm,
and riscv64 backends **fatally reject `musttail` when the return type is an
aggregate**:

```
LLVM ERROR: failed to perform tail call elimination on a call site marked musttail
```

This was confirmed by lowering a minimal two-function ping/pong module with
`llc -mtriple=<T> -filetype=obj`:

| target triple                    | aggregate-return `musttail` |
|----------------------------------|------------------------------|
| `aarch64-*` / `aarch64_be` / `arm64_32` | **OK** |
| `wasm32-unknown-unknown`         | OK (needs runtime tail-call feature; not enabled) |
| `x86_64-*`, `i386-*`             | **FAILED** |
| `arm-*` (32-bit)                 | **FAILED** |
| `riscv64-*`                      | **FAILED** |

Crucially, the same backends lower `musttail` fine when the return is a **scalar**
(`i64` or `i128`) — it is specifically the by-value struct return they refuse
(even under the `tailcc` calling convention). So the compiler emits the real
`musttail` only on the AArch64 family and keeps the `TCK_Tail` hint (bounded, but
never a build failure) elsewhere. This is what the original code was working
around when it disabled `musttail` on LLVM >= 18 — but it disabled it on *all*
targets, including AArch64 where it works.

## The x86 root fix (ESH-0171, estimated large)

Return the tagged value as an **`i128` scalar** instead of the
`{i8,i8,i16,i32,i64}` struct. i128 occupies the same two registers (RAX:RDX on
x86-64, X0:X1 on arm64) and is bit-compatible with the C struct, so the C runtime
view is unchanged — but scalar-return `musttail` lowers on x86/arm/riscv.

Scope (why it is a separate PR, not this one):

- **Every function signature** built in the backend returns
  `taggedValueType()`; all would move to `i128` (or a thin wrapper that bitcasts
  at the boundary).
- **Every call site** consumes a struct result; all pack/unpack helpers
  (`packInt64ToTaggedValue`, `unpackInt64`, `extractvalue`-based field reads,
  etc.) would need an i128 view or a bitcast at entry/exit.
- **C runtime interop**: functions called from / calling C (the whole runtime,
  FFI, `apply`, closures' function pointers) must agree on the ABI. i128 vs the
  struct must be verified register-identical on each platform; any place that
  passes a tagged value *by pointer* is unaffected, but by-value boundaries all
  change.
- **Verification surface**: SICP 88/88, AD oracle, VM parity, stress, and the
  tagged-value C-struct layout tests all re-run on x86_64 and arm64.

Given that breadth and the risk to the C-runtime boundary, the i128 ABI change is
tracked as **ESH-0171** rather than bundled into the AArch64 fix.

## Alternatives considered

- **Trampoline** (`tail_call_codegen.cpp` already has `eshkol_trampoline` +
  `BOUNCE_TAG`): correct and portable, but requires the mutually-recursive
  functions and their callers to be restructured to return bounce thunks and run
  under a driver loop — a larger, more invasive change than the i128 ABI, and it
  taxes the fast path on targets where `musttail` already works.
- **`tailcc` calling convention**: does not help — the x86 backend still refuses
  the *aggregate* return under `tailcc`, and switching CC would break C interop.
