---
kind: reference
status: current
owner-area: runtime
since: v1.4.0-connection
sources:
  - CMakeLists.txt
  - inc/eshkol/stdlib_identity.h
  - exe/eshkol-run.cpp
  - lib/repl/repl_jit.cpp
  - tests/toolchain/stdlib_identity_test.sh
---

# Precompiled standard-library identity

`stdlib.o` and `stdlib.bc` carry the same externally visible identity symbol.
The symbol is derived at CMake configure time from the Eshkol release string
(including the `-evolve` series suffix), selected object ABI, and a deterministic
SHA-256 over relative paths and contents for `lib/stdlib.esk`, its transitive
`.esk` source closure, the compiler/runtime/JIT ABI sources and configuration,
and the C/C++/LLVM toolchain versions. CMake tracks those files as configure
dependencies, so edits regenerate the expected identity before stdlib rebuild.
The symbol spells out the normalized release and object ABI plus that digest.
The compiler emits it into both formats from the shared LLVM module.

Native AOT lookup, REPL bitcode loading, and the REPL object fallback validate
the embedded symbol before using an artifact. This applies to artifacts found
through `$ESHKOL_LIB_DIR`, explicit `-L` roots, co-located installs, build
trees, and system prefixes. A mismatch or absent symbol rejects that artifact
with regeneration guidance; JIT paths can compile the installed source tree
instead. AOT may also compile from source when its module tree is present.

Packages built before this guard do not have the symbol and must be rebuilt or
reinstalled as a matching release. This check detects accidental mixing of
releases, stdlib sources, and object ABIs. It is an identity consistency check,
not cryptographic authentication against deliberate artifact modification.
