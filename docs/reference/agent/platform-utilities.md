# Platform Utility Modules

Smaller `agent.*` modules for filesystem watching, globbing, secrets,
terminal control, git, regular expressions, and layout. Each is loaded with
`(require agent.<name>)`.

---

## `agent.regex` — PCRE2 Regular Expressions

Source: `lib/agent/regex.esk`. C symbols: `eshkol_regex_*` (PCRE2). A compiled
pattern is an opaque `i64` handle — free it with `regex-free`.

| Procedure | Signature | Returns |
|-----------|-----------|---------|
| `regex-compile` | `(regex-compile pattern [flags…])` | handle |
| `regex-match` | `(regex-match handle subject)` | first match string / `#f` |
| `regex-match?` | `(regex-match? handle subject)` | `#t`/`#f` |
| `regex-match-all` | `(regex-match-all handle subject [max])` | list of matches |
| `regex-replace` | `(regex-replace handle subject replacement)` | new string |
| `regex-match-groups` | `(regex-match-groups handle subject)` | capture groups |
| `regex-group` | `(regex-group match-groups idx)` | one group |
| `regex-named-group-number` | `(regex-named-group-number handle name)` | group index |
| `regex-free` | `(regex-free handle)` | |

Flag constants: `REGEX_CASELESS`, `REGEX_MULTILINE`, `REGEX_DOTALL`.

```scheme
(require agent.regex)
(define rx (regex-compile "a(b+)c"))
(display (regex-match? rx "xabbbcx")) (newline)   ;; => #t
(regex-free rx)
```

---

## `agent.glob` — Filename Globbing

Source: `lib/agent/glob.esk`.

| Procedure | Signature |
|-----------|-----------|
| `glob-files` | `(glob-files pattern directory max-results)` → list of paths |
| `glob-files-sorted` | `(glob-files-sorted pattern directory max-results)` → sorted list |

---

## `agent.fs-watch` — Directory Watching

Source: `lib/agent/fs-watch.esk`. Poll-based; spawns a helper process.

| Procedure | Signature |
|-----------|-----------|
| `fs-watch-start` | `(fs-watch-start directory callback-file)` → pid |
| `fs-watch-poll` | `(fs-watch-poll callback-file)` → changes |
| `fs-watch-stop` | `(fs-watch-stop pid)` |

---

## `agent.keychain` — OS Secret Store

Source: `lib/agent/keychain.esk`. Wraps the platform keychain/credential store.

| Procedure | Signature |
|-----------|-----------|
| `keychain-get` | `(keychain-get account)` |
| `keychain-set!` | `(keychain-set! account value)` |
| `keychain-delete` | `(keychain-delete account)` |
| `keychain-has?` | `(keychain-has? account)` → `#t`/`#f` |

---

## `agent.terminal` — Terminal Control (TUI)

Source: `lib/agent/terminal.esk`. C symbols: `term-*-raw`. Raw/cooked mode,
cursor control, key input.

| Group | Procedures |
|-------|-----------|
| Lifecycle | `term-init` (→ `#t`), `term-shutdown`, `term-raw-mode`, `term-cooked-mode` |
| Geometry | `term-width`, `term-height`, `term-resized?` |
| Input | `term-read-key`, `term-read-key-timeout ms` |
| Output | `term-clear`, `term-move-to row col`, `term-cursor-pos` (→ `(row . col)`), `term-show-cursor`, `term-hide-cursor`, `term-write str`, `term-flush`, `term-set-title title` |

Key constants: `KEY_UP KEY_DOWN KEY_LEFT KEY_RIGHT KEY_HOME KEY_END
KEY_BACKSPACE KEY_DELETE KEY_TAB KEY_ENTER KEY_ESCAPE KEY_PAGE_UP KEY_PAGE_DOWN
KEY_F1 KEY_F2 KEY_F3 KEY_F4`.

---

## `agent.git-ffi` — Git Helpers

Source: `lib/agent/git-ffi.esk`. **Not** a libgit2 binding — a thin wrapper that
shells out to the `git` CLI via `system`, capturing output through a temp file.
Each helper returns parsed output; `git-run` returns `(exit-code . output)`.

| Procedure | Signature |
|-----------|-----------|
| `git-run` | `(git-run cwd . args)` → `(rc . output)` |
| `git-is-repo?` | `(git-is-repo? cwd)` → `#t`/`#f` |
| `git-root` | `(git-root cwd)` → path / `#f` |
| `git-branch-current` | `(git-branch-current cwd)` → name / `#f` |
| `git-status` | `(git-status cwd)` → `git status --short` text |
| `git-diff-stat` | `(git-diff-stat cwd)` → `git diff --stat` text |
| `git-log-oneline` | `(git-log-oneline cwd count)` → oneline log text |

Because it invokes `git` through a shell, it is subject to `subprocess`/`shell`
[capabilities](capabilities.md) when a policy is active.

---

## `agent.layout` — Box Layout

Source: `lib/agent/layout.esk`. Simple flexbox-style row/column layout for TUIs.

| Procedure | Signature |
|-----------|-----------|
| `layout-compute` | compute bounds for a tree |
| `layout-box` | `(layout-box direction width children)` |
| `layout-text` | `(layout-text text width)` |
| `layout-node-bounds` | `(layout-node-bounds node)` |
