# `core.strings` — extended string utilities

**Source**: [`lib/core/strings.esk`](../../../lib/core/strings.esk)
**Require**: `(require core.strings)` — auto-loaded via `(require stdlib)`.

Pure-Scheme helpers for joining, trimming, replacing, searching, casing and
splitting strings. Everything here is built on the codegen builtins
`substring`, `string-length`, `string-ref`, `string=?` and `string-split`.

> **Builtin vs. stdlib shadowing (important).** Several names in this module —
> `string-index`, `string-contains?`, `string-upcase`, `string-downcase`,
> `string-split` — are *also* codegen builtins (see
> `lib/backend/llvm_codegen.cpp`). When you call one of those names **directly
> from user code**, the builtin wins and the stdlib definition below is not
> used. The two implementations agree on ASCII text but **diverge on embedded
> NUL bytes** — see [Known issues](#known-issues).

All examples were run with `build/eshkol-run <file> -r`.

## Functions

### `(string-join lst delim)`
Concatenate the strings in `lst`, inserting `delim` between elements. Empty
list gives `""`; single element gives that element with no delimiter.

```scheme
(require core.strings)
(display (string-join '("a" "b" "c") ",")) (newline)
(display (string-join '() ",")) (newline)
(display (string-join '("only") ",")) (newline)
```
```
a,b,c

only
```

### `(string-trim str)` / `(string-trim-left str)` / `(string-trim-right str)`
Remove ASCII whitespace (space, tab, `\n`, `\r`) from both ends / the left /
the right. An all-whitespace string trims to `""`.

```scheme
(display (string-trim "   hi there  ")) (newline)
(display (string-trim-left "  hi")) (newline)
(display (string-trim-right "hi  ")) (newline)
```
```
hi there
hi
hi
```

### `(string-replace str old new)`
Replace every occurrence of substring `old` with `new`. Implemented as
`(string-join (string-split str old) new)`.

```scheme
(display (string-replace "hello world" "o" "0")) (newline)
(display (string-replace "aaa" "a" "bb")) (newline)
```
```
hell0 w0rld
bbbbbb
```

### `(string-reverse str)`
Reverse the characters of `str`.

```scheme
(display (string-reverse "abc")) (newline)
```
```
cba
```

### `(string-copy str [start [end]])`
Return a fresh copy of `str`, optionally of the codepoint range `[start, end)`
(R7RS). The 1- and 2-argument forms default the missing bounds to `start=0` and
`end=(string-length str)`.

```scheme
(display (string-copy "hello")) (newline)     ; whole string
(display (string-copy "hello" 2)) (newline)   ; from index 2
(display (string-copy "hello" 1 4)) (newline) ; [1,4)
```
```
hello
llo
ell
```
(The optional `start`/`end` forms may still emit a harmless gradual-typing
arity note; the result is correct.)

### `(string-repeat str n)`
Concatenate `n` copies of `str`. `n <= 0` gives `""`.

```scheme
(display (string-repeat "ab" 3)) (newline)
(display (string-repeat "x" 0)) (newline)
```
```
ababab

```

### `(string-starts-with? str prefix)` / `(string-ends-with? str suffix)`
Predicate: does `str` begin / end with `prefix` / `suffix`? The prefix/suffix
argument may be a string **or a single char** (converted via `string`).

```scheme
(display (string-starts-with? "hello" "he")) (newline)
(display (string-ends-with?   "hello" "lo")) (newline)
(display (string-starts-with? "hello" #\h)) (newline)
```
```
#t
#t
#t
```

### `(string-starts-with str prefix)` / `(string-ends-with str suffix)`
Non-`?` aliases kept for backwards compatibility; behaviour is identical to the
`?` predicates above.

### `(string-index str substr)`
Index of the first occurrence of `substr` in `str`, or `-1` if absent. `substr`
may be a char. **NUL-truncating when called directly** (see Known issues).

```scheme
(display (string-index "hello" "ll")) (newline)
(display (string-index "hello" "z"))  (newline)
```
```
2
-1
```
Edge cases: empty needle returns `0`; empty haystack with a non-empty needle
returns `-1`; needle longer than haystack returns `-1`.

### `(string-last-index str substr)`
Index of the **last** occurrence of `substr`, or `-1`.

```scheme
(display (string-last-index "abcabc" "bc")) (newline)
```
```
4
```

### `(string-contains str substr)` / `(string-contains? str substr)`
Boolean "does `str` contain `substr`?". `string-contains` is the stdlib
function `(>= (string-index …) 0)`; `string-contains?` is a **codegen builtin**.
On ASCII they agree; on embedded NUL they diverge (Known issues).

```scheme
(display (string-contains  "hello" "ell")) (newline)
(display (string-contains? "hello" "z"))   (newline)
```
```
#t
#f
```
Edge cases: empty needle returns `#t`.

### `(string-count str substr)`
Count non-overlapping occurrences of `substr` in `str`. Empty needle returns
`0`. This is a pure-Scheme stdlib function (no builtin), so it is **NUL-safe**.

```scheme
(display (string-count "aaaa" "aa")) (newline)
(display (string-count "abcabc" "bc")) (newline)
```
```
2
2
```

### `(string-find haystack needle)`
Like `string-index` but returns `#f` (not `-1`) when the needle is absent —
composes cleanly with `(if (string-find …) …)`. Calls the stdlib
`string-index`, so it is NUL-safe.

```scheme
(display (string-find "hello" "ll")) (newline)
(display (string-find "hello" "z"))  (newline)
```
```
2
#f
```

### `(string-upcase str)` / `(string-downcase str)`
ASCII case conversion (`a`–`z` ⇄ `A`–`Z`); non-letters pass through. These names
are codegen builtins.

```scheme
(display (string-upcase   "Hello World 123")) (newline)
(display (string-downcase "Hello World 123")) (newline)
```
```
HELLO WORLD 123
hello world 123
```

### `(string-split-ordered str delim)`
Backwards-compatible alias for the builtin `(string-split str delim)`, which now
returns fields in left-to-right order. `delim` is a char.

```scheme
(display (string-split-ordered "a,b,c" #\,)) (newline)
```
```
(a b c)
```

### `string-needle` (internal helper)
`(string-needle x)` returns `(string x)` if `x` is a char, else `x`. Used by the
search functions so a char needle is accepted. Not intended for direct use.

## Known issues

### `string-copy` (1-arg) — fixed
Historically `(string-copy s)` delegated to `substring`, which required exactly
three arguments, so the single-argument call returned `()` with a spurious
`WARNING: substring requires exactly 3 arguments`. `string-copy` now has its own
codegen path that defaults the missing bounds (`start=0`, `end=length`), so the
1-, 2-, and 3-argument R7RS forms all work. (The 2-/3-arg forms may still emit a
harmless gradual-typing arity note.)

### Embedded NUL: builtin search truncates, stdlib search does not
The builtins `string-index` and `string-contains?` use C `strstr`-style search
that stops at the first NUL byte, so they cannot find anything past a NUL. The
pure-Scheme stdlib functions `string-find`, `string-contains`, and
`string-count` use `substring`/`string=?` (which honour the string header
length) and are NUL-safe. `string-length` and `substring` themselves preserve
NUL bytes. This produces an observable disagreement on the same input:

```scheme
(require core.strings)
(define s (list->string (list #\a (integer->char 0) #\b)))  ; "a\0b", length 3
(display (string-index s "b"))     (newline)  ; -1  (builtin, NUL-truncated)
(display (string-contains? s "b")) (newline)  ; #f  (builtin, NUL-truncated)
(display (string-find s "b"))      (newline)  ; 2   (stdlib, NUL-safe)
(display (string-contains s "b"))  (newline)  ; #t  (stdlib, NUL-safe)
(display (string-count s "b"))     (newline)  ; 1   (stdlib, NUL-safe)
```
```
-1
#f
2
#t
1
```
See also **ESH-0099** (open): string *literals* that contain a `\x0;` NUL escape
and exceed 512 source bytes decode to the wrong length/content — a separate
literal-intern bug in the corruption family, distinct from the search-truncation
above.

### Char needle to the builtin `string-index` SIGSEGVs
Passing a char (rather than a string) needle directly to the builtin
`string-index` crashes. The stdlib functions (`string-count`, `string-find`,
the `*-with?` predicates) handle a char needle correctly via `string-needle`.

```scheme
(require core.strings)
(display (string-index "hello" #\l)) (newline)   ; SIGSEGV
```
Workaround: pass a one-character string, e.g. `(string-index "hello" "l")`.
