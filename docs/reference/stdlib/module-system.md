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
modules; everything else is module-private.

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

`resolve_module_path` (eshkol-run.cpp) maps a dotted module name to a file:

- **Dotted names** have each `.` rewritten to `/` and `.esk` appended:
  - `core.strings`   → `lib/core/strings.esk`
  - `core.list.transform` → `lib/core/list/transform.esk`
  - `core.ad.tape`   → `lib/core/ad/tape.esk`
  - `signal.fft`     → `lib/signal/fft.esk`
  - `ml.optimization`→ `lib/ml/optimization.esk`
  - `math`           → `lib/math.esk`
- **Path literals** — a string beginning with `/`, `./`, `../`, containing a
  `/`, or ending in `.esk` — are treated verbatim (no dot→slash rewrite). This
  is what makes `(load "some/dir.v2/file.esk")` work even when a directory name
  contains a dot (e.g. macOS `$TMPDIR` like `/var/folders/<hash>.<r>/T`).

### Search order

For each resolved `path_part`, the loader tries, in order:

1. the requiring file's directory (`base_dir`);
2. the current working directory / project root (so a project-rooted module
   like `src.core.x` resolves against `./src/...`, matching the JIT resolver);
3. the library directory `lib/` (auto-discovered relative to the `eshkol-run`
   binary and the cwd);
4. **directory-as-module**: if `lib/web.esk` is absent, `lib/web/web.esk` then
   `lib/web/index.esk` are tried, so `(require web)` can find a package entry
   point;
5. each colon-separated (`;` on Windows) entry of `$ESHKOL_PATH`. Empty
   segments, missing directories, and non-directory entries are silently
   skipped (flagged only in debug mode).

The first match wins and its canonical path is used.

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
`core.manifold`, `ml.optimization`.

`stdlib.esk` also defines directly: `random-tensor`, `random-normal-tensor`,
`current-time-us`, `time-ns`, `time-us`, `time-it`, and the internal keyword
helpers `__keyword-member?` / `__keyword-args-validate` / `__keyword-arg`.

**NOT auto-loaded** — must be required individually (non-exhaustive):
`core.testing`, `core.cache`, `core.threads`, `core.channels`,
`core.collections`, `core.metrics`, `core.logging`, `core.argparse`,
`core.memory`, `core.memory_store`, `core.merkle`, `core.distributed`,
`core.msgpack`, `core.ad.tape`, `core.dnc`, `core.sdnc`,
`core.ml.gradient_estimators`, `core.ml.neurosymbolic`, `core.http_server`,
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

- When `stdlib.o` (or `libstdlib.o`) is linked (auto-linked if not supplied
  explicitly), the loader calls `collect_all_submodules("stdlib", ...)`, which
  parses `lib/stdlib.esk` and recursively every module it requires, marking the
  **entire transitive set** as *precompiled*.
- In `process_requires`, a module found in the precompiled set is **not**
  code-generated from source again. Instead its `.esk` is parsed only to
  recover the function/type **declarations** (signatures, export list, return
  types); the actual machine code comes from `stdlib.o`. This is the "strip the
  bodies, keep the decls" path.
- If `stdlib.o` is linked but its `.esk` sources cannot be found (e.g. the
  object was installed without the accompanying `lib/` tree),
  `collect_all_submodules` adds nothing, and the loader **fails loudly** rather
  than silently re-loading `core.*` from source and producing duplicate symbols
  at link time.

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
