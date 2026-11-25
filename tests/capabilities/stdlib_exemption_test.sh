#!/usr/bin/env bash
# The `ffi` exemption belongs to Eshkol's own module tree, and to nothing that
# merely looks like it.
#
# A module shipped with Eshkol declares the runtime entry points the standard
# library is built on, and each of those carries its own gate, so those
# declarations are exempt from `ffi`. The tree is identified by carrying
# stdlib.esk. A project's own `lib/` directory does not become exempt by being
# the directory the compiler happened to search, and neither does a library
# directory named in the environment: a module there is the program's own code
# and its foreign declarations need `ffi` like any other.
#
# This builds exactly that situation — a project-local library directory holding
# a module whose `(extern …)` names libc `system` — and asks for it under a
# policy that withholds `ffi`.

set -u

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <eshkol-run> <build-dir>" >&2
    exit 2
fi

ESH="$1"
BUILD_DIR="$2"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
WORK_DIR="$(mktemp -d "$BUILD_DIR/stdlib-exemption.XXXXXX")"

failures=0

report_work_dir() {
    echo "NOTE: lane logs are under $WORK_DIR" >&2
}
trap report_work_dir EXIT

# A project-local library directory that is NOT Eshkol's: no stdlib.esk in it.
PROJECT_LIB="$WORK_DIR/lib"
mkdir -p "$PROJECT_LIB"
cat > "$PROJECT_LIB/pretend_stdlib.esk" <<'MODULE'
;;; A module in a project's own library directory. Its foreign declaration is
;;; the program reaching outside the language, so it needs `ffi`.
(provide pretend-run)
(extern i32 pretend-raw-shell ptr :real system)
(define (pretend-run command) (pretend-raw-shell command))
MODULE

MARKER="$WORK_DIR/escaped.txt"
cat > "$WORK_DIR/program.esk" <<PROGRAM
(require stdlib)
(require core.capabilities)
(require pretend_stdlib)

(capability-install-policy! (list 'file-read))
(display "outcome:")
(display
  (guard (e (#t (if (error-object? e) (error-object-message e) "non-error-object")))
    (let ((rc (pretend-run "printf escaped > '$MARKER'")))
      (string-append "ran:" (number->string rc)))))
(newline)
PROGRAM

# The project's directory comes first on the module path, which is exactly the
# situation that must not confer the exemption.
ESHKOL_PATH="$PROJECT_LIB:$ROOT_DIR/lib" ESHKOL_JIT_CACHE=0 \
    "$ESH" -r "$WORK_DIR/program.esk" -L"$BUILD_DIR" \
    >"$WORK_DIR/jit.out" 2>"$WORK_DIR/jit.err"

if grep -q "capability denied: ffi" "$WORK_DIR/jit.out" && [ ! -e "$MARKER" ]; then
    echo "  ok  a project-local library directory is not exempt"
else
    echo "FAIL: a module in a project's own library directory was treated as Eshkol's own" >&2
    echo "  stdout: $(cat "$WORK_DIR/jit.out")" >&2
    tail -20 "$WORK_DIR/jit.err" >&2
    [ -e "$MARKER" ] && echo "  the foreign call ran: the marker exists" >&2
    failures=$((failures + 1))
fi

# The real tree keeps its exemption: the same policy must still let the standard
# library work, which it could not do if its own declarations needed `ffi`.
cat > "$WORK_DIR/stdlib_still_works.esk" <<'PROGRAM'
(require stdlib)
(require core.capabilities)
(capability-install-policy! (list 'file-read))
(display "stdlib:")
(display (if (string? (number->string (string-length (symbol->string 'ok)))) "works" "broken"))
(newline)
PROGRAM
ESHKOL_JIT_CACHE=0 "$ESH" -r "$WORK_DIR/stdlib_still_works.esk" -L"$BUILD_DIR" \
    >"$WORK_DIR/stdlib.out" 2>"$WORK_DIR/stdlib.err"
if grep -q "stdlib:works" "$WORK_DIR/stdlib.out"; then
    echo "  ok  Eshkol's own module tree keeps its exemption"
else
    echo "FAIL: the standard library stopped working under a policy without ffi" >&2
    echo "  stdout: $(cat "$WORK_DIR/stdlib.out")" >&2
    tail -20 "$WORK_DIR/stdlib.err" >&2
    failures=$((failures + 1))
fi

if [ "$failures" -ne 0 ]; then
    echo "FAIL: stdlib_exemption_test" >&2
    exit 1
fi
echo "PASS: stdlib_exemption_test"
