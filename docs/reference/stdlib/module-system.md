# Module system — `require`, `provide`, and `stdlib.o` precompilation

This page documents how the Eshkol standard library is packaged and loaded:
how `(require ...)` resolves a module name to a file, what `(require stdlib)`
does versus requiring an individual module, and how the precompiled
`stdlib.o` / `stdlib.bc` artifacts change loading. Source of record:
[`exe/eshkol-run.cpp`](../../../exe/eshkol-run.cpp) and
[`lib/stdlib.esk`](../../../lib/stdlib.esk).

---

## `provide` and `require`

Every stdlib module declares its public surface with a `(provide sym ...)`
form near the top of the file. Only provided symbols are importable by other
modules; definitions omitted from that list are module-private on native and VM
engines. Modules without a `provide` form retain the legacy rule that all
their definitions are importable.

```scheme
;;; lib/core/list/transform.esk
(provide take drop take-right drop-right append reverse filter
         unzip partition list-copy)
```

A consumer pulls the module in with `(require <module-name>)`:

```scheme
(require core.list.transform)
(display (take (list 1 2 3 4 5) 2)) (newline)   ;; => (1 2)
```

## Module-name → file resolution

`eshkol::platform::resolve_module_source_path` (`lib/core/platform_runtime.cpp`,
declared in `inc/eshkol/platform_runtime.h`) maps a dotted module name to a
file. It is the single authority: the AOT driver, the in-process JIT and the
persistent `-r` run cache all call it, so no execution engine can resolve a
module differently from another.

- **Dotted names** have each `.` rewritten to `/` and `.esk` appended:
  - `core.strings`   → `lib/core/strings.esk`
  - `core.list.transform` → `lib/core/list/transform.esk`
  - `core.ad.tape`   → `lib/core/ad/tape.esk`
  - `signal.fft`     → `lib/signal/fft.esk`
  - `ml.optimization`→ `lib/ml/optimization.esk`
  - `math`           → `lib/math.esk`
- **Relative path literals** preserve dots within a directory name, as in
  `(load "some/dir.v2/file.esk")`. Absolute paths and parent-directory
  components raise a catchable condition naming the requested module.

### Search order

The build identity reserves the installed source paths, including directory
entry points. The resolver checks a reserved module's source digest in the
compiler's installation. For project names it searches the following roots.

1. the requiring file's directory (`base_dir`) — the directory of the file that
   wrote the `require`/`import`/`load`, so a program keeps finding its own
   helpers no matter which directory you run it from;
2. the current working directory / project root, so a project-rooted module
   like `src.core.x` resolves against `./src/...`;
3. each colon-separated (`;` on Windows) entry of `$ESHKOL_PATH`, which the
   `-I` flags are merged into. Empty segments, missing directories, and
   non-directory entries are skipped (flagged only in debug mode).
4. library roots configured through `$ESHKOL_LIB_DIR`, a workspace client,
   or `$ESHKOL_SYSTEM_PREFIXES`.

The first match wins and its canonical path is used. A path literal written
without the `.esk` extension is probed both as written and with the extension
appended, in every tier.

## `(require stdlib)` vs individual modules

`(require stdlib)` loads [`lib/stdlib.esk`](../../../lib/stdlib.esk), which is
a curated re-export hub — it simply `(require ...)`s a fixed set of core
modules and defines a handful of top-level helpers. Requiring it is a
convenience: you get the whole auto-loaded surface in one line.

**Auto-loaded via `(require stdlib)`** (read from `lib/stdlib.esk`):

`core.io`, `core.operators.arithmetic`, `core.operators.compare`,
`core.numeric_extras`, `core.logic.predicates`, `core.logic.types`,
`core.logic.boolean`, `core.functional.compose`, `core.functional.curry`,
`core.functional.flip`, `core.control.trampoline`, `core.list.compound`,
`core.list.generate`, `core.list.transform`, `core.list.query`,
`core.list.sort`, `core.list.higher_order`, `core.list.search`,
`core.list.convert`, `core.strings`, `core.alist`, `core.files`,
`core.capabilities`, `core.sexp`, `core.json`, `core.data.csv`,
`core.data.dataframe`, `core.data.base64`, `core.plot`, `core.reflection`,
`core.url`, `core.streams`, `core.json_schema`, `signal.fft`, `signal.filters`,
`core.manifold`, `core.exact_linalg`, `core.symbolic`, `ml.optimization`.

`stdlib.esk` also defines directly: `random-tensor`, `random-normal-tensor`,
`current-time-us`, `time-ns`, `time-us`, `time-it`, and the internal keyword
helpers `__keyword-member?` / `__keyword-args-validate` / `__keyword-arg`.

**NOT auto-loaded** — must be required individually (non-exhaustive):
`core.testing`, `core.cache`, `core.threads`, `core.channels`,
`core.collections`, `core.metrics`, `core.logging`, `core.argparse`,
`core.memory`, `core.memory_store`, `core.merkle`, `core.distributed`,
`core.msgpack`, `core.protobuf`, `core.utf8`, `core.ad.tape`, `core.dnc`, `core.sdnc`,
`core.ml.gradient_estimators`, `core.ml.diffopt`, `core.ml.neurosymbolic`, `core.http_server`,
`core.reflection`'s siblings under other trees, `ml.activations`.

> `core.testing` is deliberately excluded from `stdlib.esk`. Baking it into the
> precompiled `stdlib.o` triggers the symbol-renamer / external-declaration
> path that currently mis-handles a precompiled module's mutable internal state
> (the `*tests*` / counter globals), producing duplicate `_*tests*` symbols in
> the user `.o`. Require it explicitly with `(require core.testing)`.

## `stdlib.o` / `stdlib.bc` precompilation

The whole auto-loaded set is compiled ahead of time into `build/stdlib.o`
(native object) and `build/stdlib.bc` (LLVM bitcode) via the `--shared-lib`
build path. These artifacts let both the JIT (`-r`) and AOT compiler skip
re-codegen of ~440 stdlib functions on every run.

- The build records the transitive requires of `lib/stdlib.esk` in the same
  manifest as the installed source digests. A matching `stdlib.o` or
  `stdlib.bc` provides those modules.
- In `process_requires`, a module found in the precompiled set is **not**
  code-generated from source again. Instead its `.esk` is parsed only to
  recover the function/type **declarations** (signatures, export list, return
  types); the actual machine code comes from `stdlib.o`. This is the "strip the
  bodies, keep the decls" path.
- An object-only installation can supply the compiled bodies. Code that
  needs source declarations also needs the matching installed module tree.

### `core.*` fallthrough

Any required module that is *not* in the precompiled set is loaded and compiled
from its `.esk` source in the normal way (bodies included), then linked
alongside `stdlib.o`. This is why requiring a non-auto-loaded module such as
`core.cache` or `core.testing` works even though it is absent from `stdlib.o`:
it simply falls through to the from-source path. The historical hazard is a
module that is *partially* in both worlds — the precompiled discovery walks the
transitive `require` graph specifically to avoid a submodule being compiled
from source while its parent is precompiled (which double-defines symbols).

## Running the examples in this reference

Every example in this reference was executed with the JIT runner:

```sh
eshkol-run path/to/example.esk -r
```

The `-r` flag runs the program via the JIT (using the cached `stdlib` object);
`-o out` instead produces a native binary. `-n` / `--no-stdlib` disables
auto-linking of `stdlib.o`.
