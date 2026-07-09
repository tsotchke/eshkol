# ADR 0008 — One semantic tooling core for Eshkol developer experience

- Status: Proposed
- Date: 2026-07-09
- Deciders: developer-experience/tooling cluster
- Scope: documentation, diagnostics, workspace analysis, LSP/editor support,
  REPL/debugging, module resolution, and package management
- Supersedes: none
- Related: `docs/COMPILER_ROADMAP.md`, `docs/breakdown/PACKAGE_MANAGER.md`,
  `docs/reference/stdlib/module-system.md`

> Baseline note: this design branch was forked before the incoming
> `feat/eshkol-doc` commit named in the task. References to
> `scripts/gen_api_docs.py` describe that incoming implementation and remain the
> intended paths after the branches are integrated.

---

## 1. Decision summary

Eshkol will build its developer experience around a single, reusable,
LLVM-free **workspace analysis core** and a separate **execution/debug core**.
The compiler, `eshkol check`, the language server, spec/module documentation,
the package manager, the REPL, and debugger adapters must consume the same
source manager, module graph, semantic catalog, symbol/type index, and
structured diagnostics. No user-facing tool may maintain its own list of
keywords, builtins, exports, or module-resolution rules.

The architecture has five commitments:

1. Source locations become immutable byte spans with file identity and macro
   expansion provenance, not ambient line counters. The existing public
   `eshkol_ast_t` ABI is preserved through side tables during migration.
2. Compiler findings are data first. A versioned `Diagnostic` model feeds human
   text, JSON, LSP, DAP, and later SARIF renderers; logging is not used as a
   diagnostic transport.
3. A versioned semantic catalog plus statically extracted Scheme module docs
   powers `eshkol-doc spec`, hover/completion, REPL help, and AI queries. The
   existing Doxygen-driven C/C++ API generator remains a separate `api` lane.
4. Interactive evaluation is a session service with a framed, request/response
   protocol. The terminal REPL is one client. Debugging uses DAP over the native
   LLVM/DWARF and ORC substrate rather than a second interpreter.
5. `eshkol.toml` describes intent; `eshkol.lock` records an exact, checksummed
   dependency graph. Compiler, LSP, docs, REPL, and package builds resolve the
   same `(package, module)` identities from that graph.

This is an architectural consolidation, not a rewrite of Eshkol semantics. The
existing parser, macro expander, type checker, LLVM 21 backend, ORC JIT, module
loader, and package hardening are migrated behind shared interfaces in small
vertical slices.

## 2. Why the current shape cannot scale

Eshkol already has useful pieces in every area, but each tool sees a different
language.

### 2.1 Source and diagnostics substrate

- The public AST is shared by parsing, expansion, type checking, and codegen,
  but carries only a 1-based start `line` and `column`; it has no file identity,
  end offset, or expansion origin (`inc/eshkol/eshkol.h:2393-2405`,
  `inc/eshkol/eshkol.h:2453-2461`).
- The parser keeps thread-local ambient filename/source and cumulative
  line/column state (`lib/frontend/parser.cpp:24-53`). Its stream reader strips
  comments before parsing (`lib/frontend/parser.cpp:10294-10315`), assigns a
  form-local source buffer while parsing, then clears it
  (`lib/frontend/parser.cpp:10437-10460`). This is adequate for a batch parser,
  but not for lossless formatting, doc attachment, incremental edits, or precise
  multi-file diagnostics.
- Parser and codegen findings print immediately through `eshkol_error_at`; that
  renderer is intentionally independent of the logger's JSON setting
  (`inc/eshkol/logger.h:214-239`). The renderer produces a useful GCC-like
  header and caret (`lib/core/logger.cpp:626-701`), but the structured fact is
  lost after printing.
- The type checker already points toward the right model: `TypeCheckResult`
  carries location, expected/actual types, context, and a hint
  (`inc/eshkol/types/type_checker.h:26-85`), and it can collect errors across a
  program (`lib/types/type_checker.cpp:3450-3483`). Those fields are not yet a
  compiler-wide diagnostic contract.
- Runtime errors have canonical actual-type names and a thread-local source
  prefix (`inc/eshkol/core/runtime.h:300-358`,
  `lib/core/runtime_errors_hosted.cpp:160-194`), but this path is still formatted
  text rather than an event that a REPL, editor, or embedding host can consume.

### 2.2 Language server and editor

The existing `eshkol-lsp` is a valuable protocol shell, not yet a semantic
server:

- It advertises diagnostics, completion, hover, and definition
  (`tools/lsp/eshkol_lsp.cpp:1-12`) and stores versioned open-document contents
  (`tools/lsp/eshkol_lsp.cpp:360-394`).
- It reparses on full-document changes (`tools/lsp/eshkol_lsp.cpp:587-608`). It
  calls the real parser but discards parser failures, then reports only a local
  parenthesis scan (`tools/lsp/eshkol_lsp.cpp:621-719`).
- Keywords, builtins, and short documentation are hardcoded in the server
  (`tools/lsp/eshkol_lsp.cpp:419-520`); local definitions are found with text
  scanning (`tools/lsp/eshkol_lsp.cpp:912-965`), and go-to-definition is limited
  to string patterns in the current file (`tools/lsp/eshkol_lsp.cpp:835-873`,
  `tools/lsp/eshkol_lsp.cpp:968-990`).
- CMake links the LSP against the complete compiler and LLVM libraries
  (`CMakeLists.txt:2154-2172`), although ordinary editing should not initialize
  or depend on code generation.
- The VS Code extension is appropriately thin around LSP startup, but compile
  and run commands are assembled as terminal strings
  (`tools/vscode-eshkol/src/extension.ts:20-53`). Its TextMate grammar carries
  another independent builtin inventory
  (`tools/vscode-eshkol/syntaxes/eshkol.tmLanguage.json:138-171`).

Adding references, rename, signature help, workspace symbols, fix-its, and
module-aware completion on top of these text scans would create a second,
incomplete compiler frontend.

### 2.3 Documentation

There are two different documentation problems and they must not be conflated.

The incoming `scripts/gen_api_docs.py` is a deterministic, best-effort scanner
for Doxygen comments on public C/C++ headers. It explicitly says it is not a C
parser, lists the declarations it recognizes, emits `docs/api/`, and offers a
drift-only `--check` mode (`scripts/gen_api_docs.py:1-40`). It records a useful
symbol model (`scripts/gen_api_docs.py:269-279`), associates Doxygen blocks with
declarations (`scripts/gen_api_docs.py:530-622`), renders documented and
undocumented coverage (`scripts/gen_api_docs.py:654-700`), and deterministically
writes or checks the result (`scripts/gen_api_docs.py:778-837`). That is the
right C API lane.

It is not the requested language/specification generator. The compiler roadmap
separately asks for signatures and docstrings from the indexed module graph,
without evaluating user code (`docs/COMPILER_ROADMAP.md:256-274`). Current
Scheme modules have `provide` forms and useful but informal leading comments,
for example `lib/core/strings.esk:1-21`; there is no common docstring grammar or
machine index.

The cost of absent provenance is visible today: the complete specification says
non-provided symbols are private (`docs/COMPLETE_LANGUAGE_SPECIFICATION.md:838-850`),
while the active AOT loader deliberately treats `provide` as informational for
compatibility (`exe/eshkol-run.cpp:3577-3599`). Generated documentation must be
able to report implementation status and fail drift checks rather than
silently canonizing whichever prose was edited last.

### 2.4 REPL and debugging

The interactive foundation is stronger than the tooling around it:

- The REPL has history, completion, multiline input, load/reload, environment,
  type, doc, AST, and timing commands (`exe/eshkol-repl.cpp:317-340`,
  `exe/eshkol-repl.cpp:475-689`).
- ORC state persists across evaluations, and hot redefinition uses per-symbol
  resource trackers (`lib/repl/repl_jit.h:37-49`,
  `lib/repl/repl_jit.h:187-195`).
- The current machine mode uses `EREPL READY/DONE/FAIL` sentinels on stderr
  around unstructured stdin/stdout (`exe/eshkol-repl.cpp:723-735`). That works
  for a warm worker, but has no request IDs, protocol negotiation, typed value,
  structured diagnostics, cancellation result, or reliable output separation.
- `:reset` clears only the UI's symbol list and explicitly leaves JIT symbols
  alive (`exe/eshkol-repl.cpp:535-538`). Reload skips already-defined symbols
  (`exe/eshkol-repl.cpp:435-441`), which is surprising for live development.
- Hot reload removes the old resource before verifying and adding the new
  module (`lib/repl/repl_jit.cpp:1755-1781`), so a failed redefinition cannot be
  transactionally rolled back.

Native source debugging has already started. `eshkol-run -g` is documented as
emitting DWARF (`exe/eshkol-run.cpp:2076-2113`); the backend creates a compile
unit and source file (`lib/backend/llvm_codegen.cpp:2237-2265`), attaches
subprograms (`lib/backend/llvm_codegen.cpp:4527-4555`), and assigns instruction
locations from AST coordinates (`lib/backend/llvm_codegen.cpp:9082-9111`). The
metadata currently identifies the language as C99 and leaves parameter types
unspecified, so native tools expose lowered implementation details.

There is also an Eshkol-level frame-trace ABI and implementation
(`inc/eshkol/core/runtime.h:368-456`,
`lib/core/runtime_frame_trace.cpp:53-159`), but no production codegen or
exception call sites currently invoke its push/pop/print/reset functions. It is
a disconnected substrate, not yet a debugger or a dependable traceback.

### 2.5 Modules and packages

The batch compiler owns the real module semantics inside `exe/eshkol-run.cpp`:
it converts dotted names to paths and searches the requiring directory, cwd,
the installed library, directory entry points, and `ESHKOL_PATH`
(`exe/eshkol-run.cpp:2437-2545`). It separately discovers exports and recursively
flattens required module ASTs (`exe/eshkol-run.cpp:2882-2907`,
`exe/eshkol-run.cpp:3396-3642`). These are file-static driver facilities, so the
LSP, docs generator, package manager, and REPL cannot reliably reuse them.

`eshkol-pkg` has a useful manifest and hardened subprocess/URL boundary, but it
is not a reproducible resolver:

- The manifest stores a version string and source kind
  (`tools/pkg/eshkol_pkg.cpp:238-256`), but the loader currently reads every
  dependency value as a registry string (`tools/pkg/eshkol_pkg.cpp:293-303`).
- Build invokes `eshkol-run` and scans immediate dependency directories for
  arbitrary `.o` files (`tools/pkg/eshkol_pkg.cpp:548-604`).
- Install caches by package name, shallow-clones a guessed GitHub URL, and
  symlinks the mutable cache into `eshkol_deps/`
  (`tools/pkg/eshkol_pkg.cpp:619-680`).
- The package reference confirms that versions are metadata rather than
  constraints, the default branch is cloned, and no lockfile exists
  (`docs/breakdown/PACKAGE_MANAGER.md:9-21`,
  `docs/breakdown/PACKAGE_MANAGER.md:322-372`).

The compiler already demonstrates a sound content-key pattern for its run
cache: it hashes source bytes, transitive sources, Eshkol/LLVM versions,
configuration, target, library artifacts, and search paths
(`exe/eshkol-run.cpp:493-536`). Package and analysis caches should reuse this
model rather than keying on path or package name alone.

## 3. Goals and invariants

The design is successful when the following statements are true.

1. **One answer.** Given a source snapshot, manifest, lockfile, target profile,
   and compiler version, CLI, LSP, docs, REPL, and build agree on module
   resolution, exports, symbol identity, types, and diagnostics.
2. **Structured before rendered.** Every user-actionable compile-time finding
   has a stable code, primary span, severity, and machine representation before
   it becomes terminal text.
3. **Partial programs are normal.** Editing an incomplete form produces a
   recovered syntax tree and bounded diagnostics; it never exits the language
   server or requires LLVM code generation.
4. **Humans are the default UI.** Commands remain concise, colored only on a
   TTY, and explain causes and repairs in source language terms.
5. **Machines get contracts, not terminal scraping.** Resident AI minds can
   request exact symbols, module graphs, expansions, diagnostics, docs, and
   edits through versioned JSON/LSP protocols with deterministic ordering.
6. **Analysis is non-executing.** LSP, doc generation, dependency resolution,
   and `check` never evaluate project code or run package build hooks.
7. **Reproducibility is explicit.** A locked, offline build neither contacts the
   network nor resolves a different revision.
8. **Compatibility is deliberate.** Existing binaries and source syntax remain
   available during migration; compatibility shims delegate to new services
   rather than fork behavior.

Non-goals for this roadmap are a new language implementation, an always-on
network daemon, time-travel debugging, a new bytecode debugger, transparent
execution of dependency build scripts, or perfect fine-grained incremental
parsing in the first milestone.

## 4. Architecture

### 4.1 Dependency direction

The proposed components are internal C++ libraries first. Their C++ ABI is not
promised stable; the JSON/LSP/DAP schemas and CLI behavior are the stable
external contracts.

```text
source buffers       semantic catalog       eshkol.toml + eshkol.lock
      |                       |                         |
      +-----------------------+-------------------------+
                              v
                    eshkol-source / analysis
              spans, CST, AST views, expansion maps,
                 scopes, types, symbols, diagnostics
                              |
                              v
                       eshkol-workspace
                snapshots, module/package graph,
                 query database, cache/invalidation
          +----------+----------+----------+-----------+
          |          |          |          |           |
       CLI/check     LSP     spec/module   package    formatter
                              docs          tools

          eshkol-execution (LLVM 21 + ORC + runtime, separate)
                 |                 |                 |
                AOT          session/REPL       DAP debugger

          Doxygen public headers -> API-doc scanner (separate lane)
```

Concretely:

- `eshkol-source` owns files, overlays, line maps, tokens, a lossless CST, and
  source maps. It has no package, type-system, or LLVM dependency.
- `eshkol-analysis` owns compatibility AST views, macro expansion, scopes,
  symbol/reference indexing, gradual type queries, the semantic catalog, and
  diagnostic collection. It depends on `eshkol-source`, not on LLVM.
- `eshkol-workspace` owns immutable analysis snapshots, manifests, lockfiles,
  package/module identity, dependency invalidation, cancellation, and
  content-addressed caches.
- `eshkol-execution` owns LLVM lowering, AOT linking, ORC sessions, runtime
  value inspection, and debug registration. It consumes a workspace snapshot;
  the workspace never calls into codegen for ordinary analysis.

The existing `eshkol-static` target may remain the delivery vehicle while these
boundaries are extracted, but `eshkol-lsp` and spec docs must eventually link
only the first three layers. No mandatory daemon is introduced: batch commands
call the libraries in process, while LSP and execution sessions keep snapshots
warm in long-running processes.

### 4.2 Source model and lossless parsing

Canonical locations are UTF-8 byte offsets:

```text
SourceSpan = {
  file_id,
  start_byte,       # inclusive
  end_byte,         # exclusive
  expansion_id?     # generated node -> invocation/definition origin chain
}
```

`SourceManager` maps spans to human line/display columns and LSP UTF-16
positions on demand. Every response also carries the document version or
content hash used to compute it. This avoids storing ambiguous line/column
pairs and makes stale responses detectable.

The tokenizer becomes a reusable token stream that retains comments,
whitespace, delimiters, reader prefixes, and invalid tokens. The production
parser consumes that stream to produce both:

- a lossless, error-recovering CST for edits, formatting, docstrings, syntax
  selection, and stable source ranges; and
- the existing semantic AST shape for macro expansion, type checking, and
  codegen.

This is one lexer and grammar, not a Tree-sitter or regex reimplementation. The
first incremental implementation reparses only changed top-level forms and
reuses unchanged form results by content hash. A full-file fallback is always
available and is the correctness oracle. Fine-grained subtree incrementality is
deferred until profiles show it is needed.

Adding span fields directly to `eshkol_ast_t` would change a public struct.
During the v1.x migration, a `ParsedUnit` owns ASTs plus a `NodeId -> SourceSpan`
side table and expansion-origin table. Compatibility entry points such as
`eshkol_parse_next_ast_from_stream` continue returning plain AST values. A
future major C ABI can embed a compact span ID if justified.

Error recovery is designed for editors: synthesize a missing delimiter, retain
an `ErrorNode`, and resume at a safe top-level boundary. Resource guards remain
in force, but “invalid form” and end-of-input become distinguishable results so
an incomplete buffer is not mistaken for clean EOF.

### 4.3 Workspace snapshots and semantic queries

A `WorkspaceSnapshot` is immutable and identified by:

- source overlay revisions;
- canonical manifest and lockfile digests;
- Eshkol version/semantic-catalog version;
- target triple, execution profile, feature set, and strict/gradual type mode;
- package-store identities for every dependency.

Queries are memoized and dependency-tracked:

```text
parse(file)                    expand(module)
module_graph(package)          exports(module)
scope_at(file, byte)
symbol_at(file, byte)          definition(symbol)
references(symbol)             type_of(node/symbol)
signature(symbol)              documentation(symbol)
diagnostics(file|module|workspace)
format_edits(file)             expansion_view(node)
```

Results state whether they are complete, recovered, stale, or unavailable
under gradual typing. Cancellation is checked between top-level forms and
module queries. A new edit publishes a new snapshot; in-flight results for an
older revision may populate caches but are never published as current LSP
state.

Symbol identity is a tuple of package, module, namespace, declared name, and
definition anchor—not a raw LLVM name. Lowered/mangled names and ORC addresses are
attributes of that symbol. This lets docs, rename, debugger frames, and package
exports use one durable identity while compatibility name mangling evolves.

### 4.4 Semantic catalog

A versioned declarative catalog covers core syntax, reader forms, native
builtins, type/effect summaries, profiles, and conformance metadata. Every item
has at least:

```text
id, names/aliases, kind, grammar or signatures, summary,
since/deprecated, profiles/features, effects, implementation references,
conformance status, examples, related diagnostics
```

Normative prose lives beside the catalog in source-controlled spec fragments,
not inside C++ switch statements. Generated lookup tables make the catalog
available to analysis, LSP, REPL help, and CLI help. Initially, compiler
dispatch stays hand-written and a bidirectional drift gate checks that every
recognized special form/builtin is catalogued and every implemented catalog
item has a dispatch/test reference. Once coverage is complete, low-risk lookup
tables may be generated directly.

Scheme modules use a strict, static doc-comment convention: a contiguous
`;;;@` block immediately preceding a top-level definition, plus the module's
`provide`/R7RS export forms. Supported fields include summary, parameters,
returns, type/signature overrides where inference is intentionally imprecise,
since/deprecation, examples, and cross-references. Comments and signatures are
read from the CST and semantic index; user code is never evaluated.

The catalog records **implemented**, **specified**, **extension**,
**compatibility mode**, and **unsupported** separately. In particular, it must
not declare `provide` private until the compiler's selected module policy does.
Legacy informational `provide` and a future strict/R7RS policy are distinct
edition/profile facts, not contradictory prose.

## 5. Documentation architecture

`eshkol-doc` becomes an umbrella with three explicitly different inputs:

| Lane | Authority | Primary output | Machine output |
|---|---|---|---|
| `api` | Doxygen on `inc/eshkol/**/*.h` | `docs/api/` | symbol/coverage JSON |
| `spec` | semantic catalog + normative fragments | `docs/spec/` | versioned spec index JSON |
| `modules` | locked module graph + static `;;;@` docs | `docs/reference/modules/` | module/symbol graph JSON |

The incoming `scripts/gen_api_docs.py` remains the implementation behind
`eshkol doc api` and `make api-docs`; it should add a deterministic JSON index
but should not be stretched into a Scheme parser. `spec` and `modules` use the
workspace analysis APIs.

All three lanes support:

- `--check`, which performs no writes and reports missing, stale, and orphaned
  outputs;
- `--format markdown|json` and stable schema versions;
- source provenance for every generated item;
- coverage reports that distinguish undocumented from unavailable/inferred;
- deterministic paths and ordering independent of checkout location;
- `--package`, `--module`, and `--symbol` filters for bounded AI retrieval.

Examples are extracted statically. A separate documentation-test lane executes
marked examples under explicit profiles in both JIT and AOT where applicable;
generation itself stays non-executing. Failing examples report the example's
source fragment and generated page as related spans.

LSP hover, completion details, `:doc`, and generated pages render the same
`DocumentationItem`; they do not scrape generated Markdown. A human sees prose,
examples, availability, and links. An AI can request a compact record containing
the exact signature, module, definition span, effects, version, and related
items without loading an entire manual.

## 6. Diagnostics architecture

### 6.1 Canonical model

Every analysis-stage finding is represented as:

```text
Diagnostic {
  schema_version
  code                 # stable, documented: e.g. ESH-PARSE-0007
  severity             # error | warning | information | hint
  phase/category
  message
  primary_span + label
  secondary_spans[] + labels[]
  notes[]
  help[]
  fixes[]              # applicability + non-overlapping text edits
  expansion_trace[]
  related_diagnostics[]
}
```

Codes are stable once released; prose may improve without breaking automation.
Fixes state `machine-applicable`, `maybe-incorrect`, or `manual`. Diagnostics
are deterministically sorted by canonical package/module, byte span, severity,
and code, then deduplicated by identity. “Too many errors” is itself a final
diagnostic with the configured cap.

Parser, macro expander, resolver, type/borrow checker, and pre-codegen validation
return results into a `DiagnosticSink`; they do not print or exit. Internal
compiler failures remain a separate `InternalError` class and must never be
presented as a user syntax error.

### 6.2 Renderers and process contract

- Human text preserves the current `file:line:column`, source gutter, caret,
  color, notes, and help. Color defaults to `auto` and honors non-TTY/`NO_COLOR`.
- `--diagnostic-format=json` produces a single versioned result envelope;
  `jsonl` produces framed events for streaming commands. Neither contains ANSI
  escapes or locale-dependent text as the only machine discriminator.
- LSP conversion preserves codes, related information, tags, and applicable
  workspace edits.
- SARIF is a renderer after codes and fixes stabilize, not a separate finding
  pipeline.

Command JSON includes `status`, `snapshot`, `diagnostics`, and command-specific
data. Documented exit classes distinguish success, rejected user input/config,
and internal failure; clients never need to regex stderr. For execution/session
commands, user stdout, user stderr, diagnostics, value, and timing are separate
fields/channels.

The existing structured logger remains for compiler/runtime telemetry. A log
record such as “phase took 12 ms” is not a diagnostic; changing log level cannot
hide a source error.

### 6.3 Better compiler and runtime messages

The first quality pass targets high-frequency confusion:

- unbound names include lexical/module scope, nearest viable names, and an
  import suggestion;
- module-not-found reports the normalized module identity and actual ordered
  search candidates, not a reconstructed approximation;
- arity errors identify the callee definition, accepted forms, and call span;
- type errors label the operand and show expected/actual canonical types;
- macro errors show both invocation and definition origins and offer an
  expansion view;
- duplicate/private/ambiguous exports point to all declarations;
- package conflicts include a minimal dependency explanation chain;
- runtime errors attach the call-site location and Eshkol-level frames.

For runtime findings, generated code emits a compact diagnostic code, source-map
ID, relevant tagged values, and frame state to a host hook. The hosted runtime
converts this into the same event schema; freestanding profiles may render a
bounded text fallback. Existing type-name and source-prefix helpers are reused,
not duplicated.

The current frame-trace substrate is completed as a first debugging slice:
codegen must push/update/pop frames on normal return, exceptional unwind,
tail-call replacement, continuation transfer, and REPL boundaries; the
exception host emits the trace once. Tests must cover these control-flow cases
before frame traces are enabled by default.

## 7. LSP and editor support

`eshkol-lsp` becomes a protocol adapter over a `WorkspaceSnapshot`. It retains
stdio JSON-RPC and remains editor-neutral.

### 7.1 Analysis loop

1. `initialize` discovers the workspace root, `eshkol.toml`, lockfile, selected
   profile, and client position encodings.
2. Open documents are overlay buffers; disk files and locked dependencies are
   immutable lower layers.
3. Syntax diagnostics run after a short debounce. Expansion, resolution, and
   type diagnostics run only for affected top-level forms/modules and are
   cancellable.
4. Publishing checks document version and snapshot ID. Closing an overlay
   reverts to the current disk snapshot and republishes if needed.
5. The server never fetches packages, executes code, links, or initializes LLVM
   merely because a file changed.

UTF-8 byte offsets remain canonical internally; the source manager converts to
the client's negotiated UTF-16/UTF-8 positions. File URIs, symlinks, case
normalization, and Windows drive letters are normalized once at the source
boundary.

### 7.2 Capability sequence

The first usable release provides real, shared semantics for:

- syntax/macro/resolution/type diagnostics;
- document and workspace symbols;
- completion from lexical scope, imports, packages, and catalog items;
- hover with signature, inferred type, docs, effects, availability, and source;
- go-to-definition/declaration across modules and dependencies;
- signature help and find references.

The next release adds rename with conflict checking, semantic tokens, call
hierarchy, inlay type/parameter hints, code actions from diagnostic fix-its,
formatting/range formatting from the lossless CST, and virtual documents for
macro expansion and lowered forms. Rename never edits generated docs, locked
dependencies, or macro-generated identifiers without an explicit, previewed
plan.

Completion and hover report confidence/completeness in gradual or recovered
code. Unknown is not rendered as a confidently inferred `any`.

### 7.3 Editor integrations

The VS Code extension stays thin: locate/start `eshkol-lsp`, expose settings,
register tasks/debug configurations, and render virtual documents. Compile/run
uses structured tasks or process APIs, not hand-built shell lines. Semantic
tokens come from LSP; the TextMate grammar remains a fast lexical fallback and
its generated vocabulary comes from the catalog.

Nothing in the core depends on VS Code. Neovim, Emacs, Helix, Zed, and AI hosts
receive the same semantics through LSP. Protocol integration tests run against
the server directly, with a small separate smoke suite for each editor package.

## 8. REPL, resident sessions, and debugging

### 8.1 Session service

Split the current executable into `ReplSession` and presentation clients.
`ReplSession` owns the ORC JIT, workspace snapshot, definitions, modules,
capabilities, resource limits, output capture, and revision. The terminal REPL
owns readline/history/colors and calls the same methods as a machine client.

The long-running machine interface uses JSON-RPC 2.0 with `Content-Length`
framing, request IDs, an explicit `eshkol-session/v1` capability negotiation,
and methods such as:

```text
initialize, eval, load, type, documentation, complete, inspect,
definitions, expansion, interrupt, reset, close
```

An `eval` response separates status, typed/display value, stdout, stderr,
diagnostics, timing, changed definitions, and session revision. Requests may set
wall/CPU/memory/output limits and an execution profile. `interrupt` acknowledges
whether evaluation was cancelled, completed first, or left the session
unrecoverable.

`eshkol-repl --machine` sentinel framing remains as a compatibility adapter for
at least two minor releases, implemented over `ReplSession`; new clients use the
framed protocol. No protocol message is inferred from user output.

Definition changes are transactional. Compile and verify into a fresh ORC
resource tracker, resolve dependencies, then atomically publish forwarding
slots and retire the old resource. On failure, the previous definition remains
callable. `reset` creates a genuinely fresh session or explicitly reports which
external resources cannot be rolled back. `reload` re-evaluates changed forms
and their dependents rather than silently skipping defined names.

Human additions include `:expand`, `:where`, `:references`, `:why` (diagnostic
details), and a truthful `:reset`; all are presentation aliases for workspace or
session queries. REPL completion and `:doc` use the semantic index, so they match
the editor.

### 8.2 Debugger architecture

Use the Debug Adapter Protocol as the editor-facing contract and native
LLVM/DWARF debugging as the process engine.

**AOT first.** `eshkol debug <target>` builds a debug profile (`-O0`, debug
metadata, stable source paths) and launches LLDB where available or GDB on the
supported platform. `eshkol-dap` translates source breakpoints, continue/pause,
step, stack trace, scopes, variables, and evaluate requests. A generated
debug-sidecar maps `SymbolId`/`NodeId`, macro origins, lexical names, and tagged
value layout to lowered symbols and instruction ranges. Runtime/library frames
are hidden by default but can be revealed.

**JIT second.** ORC registers emitted objects and their debug metadata with the
native debugger. Every REPL definition has a stable source URI and generation;
breakpoints rebind after a successful transactional redefinition. Session and
DAP revisions are correlated so an editor never displays variables from an old
definition as current.

**Inspection before arbitrary evaluation.** The first debugger decodes tagged
scalars, lists/vectors/tensors, closures, and module globals through bounded,
side-effect-free runtime inspectors. Expression evaluation in a paused lexical
scope comes later and must compile against an explicit captured environment;
it is never implemented by guessing native stack offsets.

Optimized debugging is best-effort and says when a value is optimized out or a
source step covers several lowered instructions. Accurate stepping defaults to
the debug profile. Tail calls appear as logical Eshkol frames using the debug
sidecar/frame trace even when native frames have been eliminated.

## 9. Package and module tooling

### 9.1 One workspace resolver

Move path normalization, module discovery, graph construction, export
collection, cycle detection, precompiled-module metadata, and profile checks out
of `eshkol-run` into `eshkol-workspace`. Compiler, JIT, LSP, docs, tests, and
package commands call that resolver with the same snapshot.

Module identity is `(PackageId, ModuleName)`. A package manifest declares the
module names and source paths it exports; filesystem layout is a default, not
identity. `(require core.strings)` remains valid, but if two dependencies export
the same unqualified module the resolver emits an ambiguity diagnostic and
requires a manifest alias/qualification. It never silently selects whichever
directory happens to be visited first.

Package export (which modules consumers may import) is distinct from Scheme
binding visibility inside a module. Existing packages retain legacy
informational-`provide` semantics. A future strict edition can make `provide`
binding-private, but the selected policy is explicit in the workspace/catalog
and shared by AOT and JIT.

### 9.2 Manifest and lockfile

`eshkol.toml` v1 gains structured dependencies and build targets while keeping
the current simple syntax valid:

```toml
[package]
name = "example"
version = "0.2.0"
edition = "2026"

[[target.bin]]
name = "example"
entry = "src/main.esk"

[modules]
"example.core" = "src/core.esk"

[dependencies]
numerics = { version = "^1.2", registry = "default", features = ["blas"] }
local-util = { path = "../local-util" }
git-util = { git = "https://example.invalid/git-util.git", rev = "<commit>" }
```

The exact field syntax may be refined before stabilization, but dependency
source must be representable without guessing from whether a string contains
`://`.

`eshkol.lock` is generated, canonical, sorted, and committed for applications.
Each node records package/version, exact registry artifact or Git commit,
content SHA-256, enabled features, dependencies, exported modules, native
artifacts, and relevant profile constraints. A lock generated from the same
manifest and registry snapshot is byte-identical.

Resolution prefers an existing compatible lock, otherwise uses a deterministic
semver solver and emits an explanation graph on conflict. `--locked` rejects
lock drift; `--frozen` implies locked plus offline; `--offline` uses only the
local index/store. LSP and doc generation never update a lock or contact a
registry—they report the exact command needed.

### 9.3 Immutable store and builds

Fetched sources live in an immutable, content-addressed user store keyed by
verified content hash, not `~/.eshkol/cache/packages/<name>`. A project-local
view may contain links for human inspection, but compiler inputs come from the
resolved graph. Tampered content is rejected before parsing or building.

Build nodes are keyed by source/transitive graph digest, lock digest, compiler
and LLVM versions, target/profile/features, flags, and declared native inputs,
following the existing JIT cache precedent. Dependencies expose declared module
objects/libraries; builds do not scan directories and link every `.o` they find.

Pure Eshkol packages require no install script. Native/FFI packages declare
platform artifacts and build requirements. Running a native build hook is an
explicit capability-bearing operation with a preview; it is never triggered by
LSP, docs, search, or ordinary dependency resolution. Registry metadata and
content hashes establish integrity, not trust; signed provenance can be added
without changing lock identity.

The package UX becomes `eshkol pkg` with `init`, `add`, `remove`, `resolve`,
`fetch`, `build`, `test`, `doc`, `publish`, `tree`, and `why` subcommands.
Mutating commands support `--dry-run --format=json` and show
the manifest/lock/store plan before applying it. Existing `eshkol-pkg` commands
remain compatibility aliases.

## 10. Human and resident-AI interface

A single discoverable driver is the long-term front door:

```text
eshkol check | build | run | test | repl | debug
eshkol doc api | spec | modules
eshkol pkg ...
eshkol lsp
eshkol query symbol | definition | references | type | docs |
             module-graph | diagnostics | expansion
```

`eshkol-run`, `eshkol-repl`, `eshkol-lsp`, and `eshkol-pkg` remain installed as
thin compatibility entry points. Consolidation is about shared behavior and
discoverability, not forcing an immediate script migration.

The machine contract follows these rules:

- every schema/protocol has a name and version and supports capability
  negotiation;
- every response has request ID where applicable, snapshot/session revision,
  source/manifest hashes, status, and completeness;
- lists are deterministically ordered and paths are workspace-relative plus an
  explicit package identity where possible;
- byte spans and LSP positions are both available when useful, with encoding
  named explicitly;
- bounded queries can filter by package/module/symbol and paginate large
  reference sets;
- edits are returned as previewable, non-overlapping workspace edits with
  expected document versions;
- no command in JSON mode prompts on stdin, opens an editor, fetches a package,
  or executes code without an explicit method/flag;
- diagnostics and user program output are never multiplexed as magic text.

This is not a separate “AI API.” Humans, editors, CI, and resident AI minds use
the same semantic operations. Human renderers add hierarchy and explanation;
machine renderers retain stable identities and provenance. A query such as
“what is `tensor-map` here?” returns the binding selected in the current locked
workspace—not a global hardcoded builtin guess.

## 11. Delivery roadmap

Each milestone is a usable vertical slice. Later work must not create temporary
parallel sources of truth.

### M0 — Contracts and extraction

- Define `SourceSpan`, `Diagnostic v1`, `DocumentationItem v1`, symbol IDs,
  workspace snapshot IDs, and JSON schemas with golden compatibility tests.
- Introduce `DiagnosticSink` adapters around existing parser/type/codegen
  reporting while preserving current human output.
- Extract module resolution/graph code from `eshkol-run` behind a shared API;
  keep the compiler behavior byte-for-byte compatible.
- Seed the semantic catalog and add inventory drift tests against parser,
  builtins, VM/native registries, and editor grammar.
- Split an LLVM-free analysis target from backend-dependent execution targets.

Exit gate: one batch compile produces the same human diagnostics as before plus
stable JSON; compiler and a test client resolve an identical module graph.

### M1 — Diagnostics and documentation slice

- Add `eshkol check` with multi-error syntax, resolution, and type diagnostics.
- Add source spans, recovered top-level CST, macro origin chains, human/JSON
  renderers, diagnostic docs, and high-value fix-its.
- Land `eshkol doc api` as the wrapper for `gen_api_docs.py`; add API JSON.
- Implement `eshkol doc spec` and `modules` from the catalog/workspace index,
  migrate a representative stdlib module set to `;;;@`, and run doc examples.
- Replace hardcoded LSP diagnostic logic with `eshkol check` queries.

Exit gate: a golden invalid project has the same codes and primary spans in CLI
JSON, human text, and LSP; docs `--check` is deterministic and performs no code
execution.

### M2 — Reproducible workspace and package graph

- Stabilize `eshkol.toml` v1 module/target/dependency fields and `eshkol.lock`.
- Implement deterministic resolution, conflict explanations, immutable verified
  store, `--locked/--offline/--frozen`, and graph-aware build caching.
- Make AOT, JIT, LSP, docs, and tests consume the locked resolver.
- Preserve current name/URL/path hardening tests and add tamper, lock drift,
  ambiguity, and two-clean-machine reproduction tests.

Exit gate: two empty machines given the same source, lock, compiler, and target
produce identical module graphs and artifact digests; a frozen build performs
zero network access.

### M3 — Semantic LSP and editor baseline

- Implement overlay snapshots, cancellation, version-safe publishing, scope and
  symbol indices, module-aware completion/hover/definition/references/signature
  help, and workspace symbols.
- Generate lexical editor vocabularies from the catalog; remove semantic
  hardcoding from the VS Code extension.
- Add protocol tests for incomplete forms, macro origins, cyclic/ambiguous
  modules, unsaved imports, dependency sources, Unicode/LSP position mapping,
  and rapid edit cancellation.
- Add rename, semantic tokens, inlay hints, code actions, and formatter support
  after the baseline index is trustworthy.

Exit gate: every advertised LSP capability is backed by the shared index; no
feature uses regex definition discovery or initializes LLVM during editing.

### M4 — Session protocol and REPL convergence

- Extract `ReplSession`; implement framed protocol v1 and terminal client.
- Make diagnostics, docs, types, completion, module resolution, and source
  revisions come from workspace queries.
- Add transactional definition/reload, real reset, cancellation/resource
  limits, structured output capture, and compatibility sentinel mode.
- Publish a client conformance harness for sister projects and resident agents.

Exit gate: arbitrary user output cannot spoof protocol completion; a failed
redefinition leaves the prior binding usable; human and machine sessions report
the same value/type/diagnostic facts.

### M5 — Debugging

- Finish source-level runtime trace integration and error hooks.
- Emit the debug sidecar and improve DWARF scopes/names/types.
- Ship AOT DAP launch/attach, breakpoints, stepping, stack/scopes, and bounded
  tagged-value inspection on supported desktop platforms.
- Register ORC debug objects, correlate generations, and support JIT/REPL
  breakpoints; add guarded paused-expression evaluation last.

Exit gate: a cross-module sample can stop at an Eshkol source breakpoint, step
without entering runtime internals by default, inspect arguments/locals, and
show an Eshkol traceback after a runtime error in AOT; the equivalent JIT gate
passes before JIT debugging is advertised stable.

## 12. Verification and quality gates

The following are permanent gates, not milestone-only tests.

### Shared semantics

- Differential test: fresh full analysis equals incremental analysis after
  randomized edit sequences.
- Cross-consumer fixtures: compiler, LSP, docs, REPL, and package tree report
  identical `SymbolId`, `ModuleId`, signature, visibility mode, and source span.
- Catalog coverage: no uncatalogued implemented form/builtin; no catalog item
  marked implemented without implementation and test references.
- AOT/JIT module-resolution parity under current directory, `-I`, installed
  stdlib, dependency store, path literals, R7RS import modifiers, cycles, and
  precompiled modules.

### Diagnostics

- Golden human and JSON renderings plus schema validation.
- Every LSP diagnostic round-trips to the same code/span/related information;
  machine-applicable fixes parse and remove their target diagnostic.
- Macro, generated-form, Unicode, tab, multi-file, and dependency locations.
- Runtime trace tests across normal return, exception/guard, tail call,
  continuation, parallel worker, and REPL reset.

### Documentation

- All generated lanes pass `--check`; generation is byte-identical across
  checkout roots and does not execute Eshkol.
- Exported module symbols are either documented or explicitly waived with a
  reason; examples compile/run under declared profiles.
- Source/prose implementation-status contradictions fail CI instead of merely
  lowering a coverage percentage.

### Protocols and performance

- LSP/session/DAP malformed-message, cancellation, stale-response, output
  spoofing, and graceful-shutdown tests.
- Analysis latency and memory benchmarks on representative small, stdlib-scale,
  and resident-agent workspaces. Regressions are compared to checked-in
  baselines; correctness may fall back to full reanalysis rather than publish a
  fast stale result.
- Long-lived LSP and REPL soak tests verify bounded snapshot/cache/resource
  growth across edits and redefinitions.

### Packaging

- Resolver determinism and minimal conflict explanations.
- Offline/frozen reproduction and no-network assertions.
- Content tamper, path traversal, unsafe URL, command injection, checksum,
  ambiguous module, native-hook capability, and cache concurrency tests.
- Build cache keys change for every semantic input and stay stable for
  irrelevant path/mtime changes.

## 13. Consequences

### Positive

- Every improvement to spans, imports, types, or docs benefits all tooling.
- Compiler/editor disagreements become testable defects rather than accepted
  limitations.
- Humans receive contextual, fixable messages and a discoverable end-to-end
  workflow; resident agents receive compact, deterministic semantic records.
- LSP startup and edit analysis no longer pay for LLVM.
- The package lock makes editor analysis, docs, REPL, CI, and shipped binaries
  describe the same dependency world.
- Existing investments—Doxygen docs, the parser/type checker, module loader,
  ORC hot reload, DWARF, runtime type names, and cache hashing—are reused.

### Costs

- A lossless token/CST layer and source-map side tables are substantial
  frontend work.
- Extracting module resolution from a large driver exposes hidden global-state
  assumptions and AOT/JIT compatibility cases.
- Stable diagnostic/catalog/protocol schemas require governance and fixtures.
- Incremental snapshot caches need cancellation, eviction, and long-lived
  memory testing.
- Good debugger variable inspection requires explicit tagged-value metadata and
  cannot be obtained from line tables alone.
- Reproducible packaging requires a real resolver, immutable store, registry
  metadata, and migration away from name-keyed symlinks.

## 14. Rejected alternatives

**Keep extending the current LSP with regexes and hardcoded tables.** Rejected:
it would inevitably disagree with macros, types, packages, R7RS imports, and
compiler diagnostics.

**Use Tree-sitter as the semantic frontend.** Rejected as the authority: a
Tree-sitter grammar can be useful for emergency highlighting, but duplicating
Eshkol's reader, macro lowering, keyword arguments, types, and module semantics
creates two languages. The production tokenizer/parser must expose the tooling
tree.

**Make the Doxygen generator the universal `eshkol-doc`.** Rejected: its
best-effort header scan is appropriate for the embedding C API, not normative
Scheme semantics or the locked module graph.

**Run the batch compiler on every edit and scrape stderr.** Rejected: it loses
diagnostic structure, handles incomplete code poorly, initializes unnecessary
backend state, and cannot answer semantic navigation queries.

**Build a bespoke REPL debugger.** Rejected: it would debug only one execution
mode and duplicate native process control. DAP plus LLVM/DWARF/ORC gives AOT,
JIT, and editor interoperability; the REPL remains a client of the same session.

**Keep Git default branches as dependency resolution.** Rejected: version text
without an exact revision/hash cannot make builds, docs, or editor analysis
reproducible.

**Require an always-on workspace daemon.** Rejected: it complicates CI,
embedding, security, and lifecycle. Reusable libraries allow optional warm
processes without making them correctness dependencies.

## 15. Deferred choices

These do not block the architecture and are decided with prototypes:

- exact catalog/spec-fragment serialization (strict TOML front matter versus a
  similarly dependency-free format);
- deterministic semver solver algorithm and registry wire format;
- LLDB versus GDB default by platform, while DAP remains stable;
- whether a future major ABI embeds span IDs in `eshkol_ast_t`;
- the point at which generated catalog tables replace drift-checked manual
  dispatch;
- whether SARIF ships in M1 or after diagnostic codes/fix-its have one release
  of stability.

The architectural decisions—one source/semantic core, structured diagnostics,
separate API/spec/module doc lanes, framed sessions, DAP over native debugging,
and exact locked packages—are not deferred.
