---
kind: explanation
status: current
owner-area: memory
since: v1.3.5
sources:
  - inc/eshkol/frontend/ast_strings.h
  - inc/eshkol/frontend/source_paths.h
  - lib/frontend/ast_strings.cpp
  - lib/frontend/source_paths.cpp
  - .icc/lsan-suppressions.txt
---
# ADR-0021: One owner for AST string payloads, one spelling for recorded paths

**Status:** Accepted
**Amends:** ADR-0000 Stage 1 (the frontend identity substrate), ADR-0010 gap A12
(leak detection)
**Scope:** Frontend (parser, macro expander), module-private renaming, the
`eshkol-run` driver, the REPL, the runtime `eval` bridge, HoTT type terms,
recorded source paths, and the LeakSanitizer policy for sanitizer builds.

## Context

The strings hung off frontend structures had no owner. Identifiers
(`variable.id`, `eshkol_func.id`), literal text (`str_val.ptr`), operation
names (`define_op.name`, `set_op.name`, `let_op.name`, rest parameters, extern,
import, require, provide, guard, region and diff names), HoTT type-variable
names and the names in macro definitions were allocated five different ways
and released three different ways:

- the parser used `new char[]` at about 70 sites, `strdup` at 11, and three
  private copy helpers that did the same thing;
- the macro expander allocated with `strdup` and freed with `free()`;
- `eshkol-run` and the REPL replaced names with `delete[]` and `strdup`;
- module-private renaming (`lib/core/module_visibility.cpp` `replace_name`)
  used `new[]`/`delete[]`;
- `lib/core/ast.cpp` put some names in the runtime's global arena and freed
  literal text with `delete[]` in `eshkol_ast_clean`, including text that the
  runtime `eval` bridge had allocated with `strdup`.

Nothing freed the renamed strings. The release dry run's sanitizer build
compiles the standard library with the ASan+UBSan compiler under
`detect_leaks=1`, and LeakSanitizer stopped it: 13 673 bytes in 380
allocations, every direct leak allocated by `replace_name`. The suppression
file already held one rule per allocation site that fed this retention
(`make_parser_string_ast`, `parse_extern_modifier_tail`, and the string halves
of `parse_atom`, `parse_list`, `parse_quoted_list_internal` and
`parse_function_signature`). Adding `leak:replace_name` would have been the
same local patch again, and the next producer would have needed another.

## Decision

AST string payloads have exactly one owner:
`inc/eshkol/frontend/ast_strings.h`, implemented in
`lib/frontend/ast_strings.cpp`.

1. **One allocator.** Every producer allocates through
   `eshkol_ast_strdup()`, `eshkol_ast_strndup()`, `eshkol_ast_string_alloc()`
   or the C++ `eshkol_ast_string_copy()` overloads. The producers are the
   parser, the macro expander (copies and hygienic fresh names), module-private
   renaming, `eshkol-run`'s module rewriting and its synthesized
   `(require stdlib)`, the REPL's import rewriting, the runtime `eval` bridge
   (`sexp_to_ast.cpp`, `introspection.cpp`), HoTT type-variable names
   (`ast.cpp`, `type_checker.cpp`), the symbolic AST builders and
   `eshkol_copy_ast()`, and codegen's synthesized nodes.
2. **No consumer frees a string.** A rename stores a new pointer and leaves the
   old bytes to the owner. `eshkol_ast_clean()` detaches literal text without
   releasing it. There is no per-string release call, so there is no way to
   release with the wrong deallocator.
3. **Rooted storage.** The owner is an append-only, chunked arena whose
   chunks are linked from one process-global root. LeakSanitizer follows that
   root and classes every AST string as live while the compilation lives. The
   strings are rooted, not suppressed.
4. **A defined teardown point.** `eshkol_ast_strings_teardown()` releases
   every chunk at once. `eshkol-run` calls it when `main()` returns, through
   a scope guard declared first in `main()`. The REPL calls it in its ordered
   exit, after the runtime shuts down and before its explicit leak check,
   because it always leaves through `std::_Exit()`. Embedders that parse
   repeatedly in one process (the C FFI, runtime `eval`) keep the
   process-lifetime default.
5. **Sanitizer precision is kept.** Under AddressSanitizer each allocation is
   followed by a poisoned redzone and unused chunk space stays poisoned, so an
   overread of an AST string is still reported, as it was when each string was
   its own heap block. After teardown, a stale AST pointer is a
   heap-use-after-free.

### Fit with the identity substrate

The owner sits in the `semantic-identity` component beside the
`NodeId -> SourceSpan` table (`node_identity.h`), and follows the same
lifetime rule: frontend data is valid for the whole compilation that produced
it, because a node built during parsing is read by expansion, renaming, type
checking and codegen long after its parse unit is gone. It adds no identity.
`NodeId` remains the only node key. Interning names into the `SymbolId` column
that ADR-0000 Stage 2 owes (ADR-0006 slices 1-2) is a layer on top of this
owner, not a second one: an interner stores each distinct spelling here once
and maps it to an id. Node storage (`new eshkol_ast_t[N]`) is not covered;
that stays epic #182, and its suppression rules now cover node storage only.

### Recorded source paths are normalized at one place

A compiler does two different things with a source path, and they need
different spellings. It READS the file by its host path. It also RECORDS where
code came from: in the interned file table behind `source_file_id`, in the
parse context diagnostics print, and in string constants the backend embeds so
the runtime can name a location at error time. Recording the host path put the
build machine's directory layout inside shipped artifacts — the site
WebAssembly module carried the absolute path of `lib/core/ad/interval.esk`
from the worktree it was built in, user name included — and made those
artifacts differ between two builds of the same source.

`inc/eshkol/frontend/source_paths.h` is the one normalizer. Its result is the
DISPLAY path: a module-relative path when the file is under a directory on
`ESHKOL_PATH`, else a repository-relative path when it is under a project root
(the nearest ancestor holding `.git` or `CMakeLists.txt`), else a path relative
to the working directory, else the file name alone. Never an absolute host
path. `eshkol_intern_source_file()` stores both columns, so
`eshkol_source_file_name()` answers with the display path for every recording
site while `eshkol_source_file_host_path()` still lets the diagnostic printer
open the file for a caret line. The backend's ambient context holds the same
pair.

DWARF debug info (`--debug-info`) is deliberately not normalized: a debugger
needs the directory to find the source, and that output is opt-in and not
shipped.

### Leak detection during builds

There is one policy. A sanitizer build runs the instrumented compiler on the
standard library, and that is a real compiler workload, so it runs under
`detect_leaks=1` with the checked-in suppression file, like every other
LeakSanitizer workload. `scripts/build-sanitizer.sh` applies this by default on
Linux. The CI sanitizer lane's Build step uses the same setting; before this
decision it used `detect_leaks=0`. The release producer
(`scripts/run_v1_3_release_producers.sh`) already did. macOS keeps detection
off because its system toolchain has no working LeakSanitizer.

## Consequences

- The retention in `replace_name` is gone, and so are the two suppression
  rules that covered only string payloads (`make_parser_string_ast`,
  `parse_extern_modifier_tail`). The six remaining front-end rules say they
  cover node storage only. The suppression file tells whoever reads it that a
  front-end string leak means a producer bypassed the owner, and must not
  become a new rule.
- Mixed `new[]`/`malloc`/`free`/`delete[]` ownership of AST strings is gone,
  and with it the alloc-dealloc mismatches it allowed (for example
  `eshkol_ast_clean` calling `delete[]` on `strdup`ed `eval` text).
- Retention is proportional to the source text read plus the names that
  expansion synthesizes, which is the same growth law as the `NodeId` table. A
  REPL session keeps its strings until it exits, as it already kept its nodes.
  `eshkol_ast_clean` used to release literal text per REPL input, so for that
  one case retention now includes that text. `tests/memory/leak_audit_gate.sh`
  section B adds the owner's own report (`ESHKOL_AST_STRINGS_STATS`) to
  LeakSanitizer's figure. The rooted retention stays in the per-line slope it
  pins; it does not disappear from the measurement because it is no longer a
  leak. Re-measured on Linux with gcc-13 ASan, the slope is still 1628 bytes
  per line: 1608 bytes of node storage and 20 bytes of rooted identifier
  text.
- Long-running processes that parse on every request (the language server's
  workspace checks, `eshkol-server`, runtime `eval`, the C FFI and the Python
  binding) keep the process-lifetime default. Before this decision they leaked
  the same identifiers, and they still retain node storage per parse. A
  per-request compilation scope needs owned nodes too, so it belongs with epic
  #182 and not in a strings-only owner. The parser fuzzer treats each input as
  one compilation and tears the owner down after it.
- Allocation is a bump in a 64 KiB chunk under a mutex, not one heap block
  per string. Compiling the standard library allocates 37 843 AST strings,
  295 050 bytes requested, in 7 chunks (459 KB reserved), against a peak RSS
  of about 2.2 GB. On a Release build the standard-library compile took
  33.1 s and 2159 MiB before this change, and 32.2 s and 2141 MiB after. The
  nested-expression scaling gate is within run-to-run noise.
- Under the release producer's sanitizer build, the front-end rule
  `parse_atom` now matches 22 objects (5 280 bytes) while compiling the
  standard library. It matched 63 341 objects in the original audit, and
  nearly all of those were literal and symbol text.

- A recorded source path no longer names the build machine. The shipped site
  WebAssembly module embedded one absolute host path before this change and
  none after; `scripts/check_artifact_paths.py` reads shipped artifacts as
  bytes and fails on any home-directory path, and runs as the third layer of
  `scripts/check_disclosure.py` as well as on its own. A file outside every
  root records as its file name alone, so a diagnostic still names the file
  while no artifact names the machine.

## Verification

- `tests/frontend/ast_strings_test.cpp` (`ast_strings_test`) checks the owner
  contract: exact copies, chunk crossing, large requests, counters, and
  teardown. It then walks real parser output, the macro expander's hygienic
  rewrite, module-private renaming and `eshkol_copy_ast()`, and requires
  every string pointer on every node to be owned. It also proves that
  `eshkol_ast_clean()` and a rename leave the old bytes readable.
- `scripts/check_ast_string_owner.py` (`ast_string_owner_gate`, with a
  `--self-test` that plants each violation) fails on a raw `new char[]`,
  `strdup` or `strndup` in an AST producer, or a `delete[]`/`free()` of an AST
  string field.
- `tests/frontend/source_paths_test.cpp` (`source_paths_test`) checks the
  display rules, memoization, and that the interned table and the parse
  context hand back the display path while the host path stays available.
- `scripts/check_artifact_paths.py` (`artifact_host_path_gate`, with a
  `--self-test` that plants a host path of each shape) scans the shipped
  WebAssembly modules and anything a caller passes with `--artifact`.
- The release producer's sanitizer build (`scripts/build-sanitizer.sh
  asan+ubsan` under the producer's `ASAN_OPTIONS`/`LSAN_OPTIONS`) compiles the
  standard library leak-clean. `tests/memory/leak_audit_gate.sh` and
  `scripts/check_leak_detection_selftest.sh` stay green with the smaller
  suppression file.
