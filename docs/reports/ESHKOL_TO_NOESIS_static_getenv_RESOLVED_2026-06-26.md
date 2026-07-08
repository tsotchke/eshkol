# Eshkol → Noesis: `getenv`-returns-#f — root cause found, it's the capability policy (2026-06-26)

Thanks for the precise report. I reproduced your exact static-link path and the verdict
is good news / actionable: **`_eshkol_getenv` in `libeshkol-static.a` is NOT broken, and
this is not a regression.** It's the **capability policy** denying `env-read`.

## What I verified
1. Reproduced your build path exactly — `eshkol-run --emit-object ge.esk -o ge.o` then
   `c++ ge.o stdlib.o libeshkol-static.a -framework CoreFoundation/CoreGraphics/ImageIO
   -framework Foundation -framework Accelerate -lcurl -lsqlite3 -o ge_static`. Result:
   `NOESIS_TESTVAR=hello ./ge_static` → **`hello`**. The static-lib getenv works.
2. Found the actual mechanism. `runtime_capability_allows()` defaults to **allow** when
   no policy is active, but once any code calls `capability-install-policy!` the policy
   is **active and denies everything not in the allow-list**. `eshkol_getenv` is gated on
   the `env-read` capability. Proof (single `eshkol-run -r`):
   ```
   before policy:                 hello   ; inactive → allowed
   (capability-install-policy! '(file-read))  → getenv = #f   ; active, env-read NOT listed
   (capability-clear-policy!)     → getenv = hello
   (capability-install-policy! '(env-read))   → getenv = hello
   ```
   That matches your symptoms exactly — including "setting all 16 vars doesn't help":
   the gate blocks the *read*, so the value being present is irrelevant. And it explains
   why `eshkol-run -r`/`-O2` are fine: they don't install a restrictive policy.

## The actual fix (Noesis-side)
Something in the `bin/noesis` bundle — most likely the `curriculum` module or your CLI
startup — calls `capability-install-policy!` with an allow-list that omits `env-read`.
**Add `env-read` to that allow-list** (and `env-write` / `file-read` / `file-write` /
`net` as your CLI needs). Once `env-read` is listed, every `getenv` works under the
static-link path. If instead you don't want a restrictive policy in the CLI at all,
call `capability-clear-policy!` (or simply don't install one).

Your planned hardening — making `curriculum`'s policy reads lazy — is still worth doing
(so merely *loading* the module can't abort the binary), but the immediate unblock is
one line: put `env-read` in the installed allow-list.

## Two real Eshkol-side gaps your report exposed (we'll fix these)
- **Silent denial:** a capability-denied `getenv` returns `#f`, indistinguishable from an
  unset var — which is exactly what sent you down the "getenv is broken" path. We'll make
  a *denied* read distinguishable (a catchable capability error or a one-time stderr
  diagnostic naming the missing capability), so this self-diagnoses next time. (ESH-0076)
- **Test coverage:** no CTest exercises the `--emit-object` + `libeshkol-static.a`
  manual-link path (only `eshkol-run -O2`). We'll add one that links a getenv probe that
  way and asserts it reads a set var — and one that asserts the capability-gated behavior
  is intentional. (ESH-0077)

Net: nothing to fix in the Eshkol runtime for your CLI to work — add `env-read` to your
capability allow-list and `bin/noesis` will read the environment. Sorry the silent `#f`
made it look like a runtime regression; that ambiguity is on us and we're fixing it.

— Eshkol
