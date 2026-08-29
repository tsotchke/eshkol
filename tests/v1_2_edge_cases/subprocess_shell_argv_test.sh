#!/usr/bin/env bash
# subprocess_shell_argv_test.sh — explicit shell vs argv process semantics.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

# /bin/sh is POSIX-guaranteed (present even on hosts, like NixOS, whose /bin
# is otherwise empty); echo/sleep/kill only need to be PATH-searchable, since
# argv-mode spawns below use bare names and go through execvp(), not an
# absolute path.
if [ ! -x /bin/sh ] || ! command -v echo >/dev/null 2>&1 \
   || ! command -v sleep >/dev/null 2>&1 || ! command -v kill >/dev/null 2>&1; then
    echo "SKIP: POSIX shell tools unavailable"
    exit 0
fi

mkdir -p "$ROOT/.scratch"
WORK="$ROOT/.scratch/subprocess_api_test.$$"
mkdir "$WORK"
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/subprocess_api.esk" <<'EOF'
(require stdlib)
(require agent.subprocess)

(define passed 0)
(define failed 0)

(define (raises? thunk)
  (let ((raised #f))
    (guard (exn (#t (set! raised #t)))
      (thunk))
    raised))

(define (check name actual expected)
  (if (equal? expected actual)
      (set! passed (+ passed 1))
      (begin
        (display "FAIL: ") (display name)
        (display " expected ") (display expected)
        (display ", got ") (display actual) (newline)
        (set! failed (+ failed 1)))))

(define (capture-spawn proc)
  (if proc
      (begin
        (process-close-stdin proc)
        (process-wait proc 5000)
        (let ((stdout (process-read-all-stdout proc 4096))
              (stderr (process-read-all-stderr proc 4096))
              (code (process-exit-code proc)))
          (process-destroy proc)
          (list (cons 'exit-code code)
                (cons 'stdout stdout)
                (cons 'stderr stderr))))
      (list (cons 'exit-code -999)
            (cons 'stdout "")
            (cons 'stderr "spawn failed"))))

(capability-install-policy! (list 'subprocess))
(check "shell denied without shell capability"
       (raises? (lambda () (process-spawn-shell "echo should-not-run" ".")))
       #t)
(capability-clear-policy!)

(define shell-result
  (run-command-capture "echo out; echo err >&2; exit 42" "." 5000 4096))

(define shell-pipeline
  (run-command-capture "printf 'left\nright\n' | grep right" "." 5000 4096))

(define shell-redirection
  (run-command-capture "echo hidden >/dev/null; echo visible" "." 5000 4096))

(define argv-result
  (run-argv-capture (list "echo" "literal;not-shell") "." 5000 4096))

(define argv-metachars
  (run-argv-capture (list "echo" "literal|pipe" "literal>redir" "exit 42")
                    "." 5000 4096))

(define legacy-simple
  (capture-spawn (process-spawn "echo legacy-simple" ".")))

(define legacy-shell-compatible
  (capture-spawn (process-spawn "echo legacy-shell; exit 42" ".")))

(define shell-builtin
  (process-spawn-shell "cd" "."))

(define shell-builtin-code
  (if shell-builtin
      (begin
        (process-close-stdin shell-builtin)
        (process-wait shell-builtin 5000)
        (let ((code (process-exit-code shell-builtin)))
          (process-destroy shell-builtin)
          code))
      -999))

(define read-once-proc
  (process-spawn-shell "printf owned-buffer" "."))

(define read-once-first
  (if read-once-proc
      (begin
        (process-close-stdin read-once-proc)
        (process-wait read-once-proc 5000)
        (process-read-all-stdout read-once-proc 4096))
      "spawn failed"))

(define read-once-second
  (if read-once-proc
      (let ((s (process-read-all-stdout read-once-proc 4096)))
        (process-destroy read-once-proc)
        s)
      "spawn failed"))

(define wait-exit-proc
  (process-spawn-shell "exit 7" "."))

(define wait-exit-result
  (if wait-exit-proc
      (process-wait wait-exit-proc 5000)
      -999))

(define wait-exit-code
  (if wait-exit-proc
      (let ((code (process-exit-code wait-exit-proc)))
        (process-destroy wait-exit-proc)
        code)
      -999))

(define timeout-proc
  (process-spawn-argv (list "sleep" "5") "."))

(define timeout-pid
  (if timeout-proc (process-pid timeout-proc) 0))

(define timeout-wait
  (if timeout-proc (process-wait timeout-proc 100) -999))

(define timeout-kill-wait
  (if timeout-proc
      (begin
        (process-kill timeout-proc)
        (process-wait timeout-proc 5000))
      -999))

(define timeout-exit-code
  (if timeout-proc
      (let ((code (process-exit-code timeout-proc)))
        (process-destroy timeout-proc)
        code)
      -999))

(define timeout-exit-code-observed?
  (or (= timeout-exit-code 0)
      (= timeout-exit-code 143)
      (= timeout-exit-code 137)))

(define argv-timeout-result
  (run-argv-capture (list "sleep" "5") "." 100 4096))

(define env-proc
  (process-spawn-argv-env
   (list "sh" "-c" "test \"$ESHKOL_FFI_OVERLAY\" = overlay && test -n \"$PATH\"")
   "."
   (list (cons "ESHKOL_FFI_OVERLAY" "overlay"))))
(define env-wait
  (if env-proc
      (let ((status (process-wait env-proc 5000)))
        (process-destroy env-proc)
        status)
      -999))

(define options-proc
  (process-spawn-argv-options
   (list "sh" "-c" "test \"$ESHKOL_FFI_OPTIONS\" = options")
   (list (cons 'cwd ".")
         (cons 'env (list (cons "ESHKOL_FFI_OPTIONS" "options")))
         (cons 'stdin 'null)
         (cons 'process-group #t))))
(define options-wait
  (if options-proc
      (let ((status (process-wait options-proc 5000)))
        (process-destroy options-proc)
        status)
      -999))

(define binary-proc
  (process-spawn-shell "printf 'left\\0right'" "."))
(define binary-result
  (if binary-proc
      (begin
        (process-close-stdin binary-proc)
        (process-wait binary-proc 5000)
        (let ((bytes (process-read-stdout-bytes binary-proc 4096)))
          (process-destroy binary-proc)
          bytes))
      (cons "" -1)))

(define stale-proc (process-spawn-argv (list "true") "."))
(when stale-proc
  (process-wait stale-proc 5000)
  (process-destroy stale-proc))
(define stale-pid (if stale-proc (process-pid stale-proc) 0))
(define stale-wait (if stale-proc (process-wait stale-proc 0) -999))

;; Keep a handle in a suite closure, close it once, and then exercise the
;; captured handle again. The second call must reach the tombstone diagnostic,
;; not freed memory or an instruction-stream fault.
(define (make-closed-handle-probe)
  (let ((proc (process-spawn-argv (list "true") ".")))
    (lambda (operation)
      (if (= operation 0)
          (begin (process-wait proc 5000) (process-destroy proc) 0)
          (process-pid proc)))))
(define closed-handle-probe (make-closed-handle-probe))
(define closure-close-result (closed-handle-probe 0))
(define closure-after-close (closed-handle-probe 1))

(define bound-a (process-spawn-argv (list "sleep" "1") "."))
(define bound-b (process-spawn-argv (list "sleep" "1") "."))
(define bound-c (process-spawn-argv (list "true") "."))
(when bound-a (process-destroy bound-a))
(when bound-b (process-destroy bound-b))

(define destroy-proc
  (process-spawn-argv (list "sleep" "30") "."))

(define destroy-pid
  (if destroy-proc (process-pid destroy-proc) 0))

(when destroy-proc
  (process-destroy destroy-proc))

(define destroy-kill-check
  (if (> destroy-pid 0)
      (run-argv-capture (list "kill" "-0" (number->string destroy-pid))
                        "." 5000 4096)
      (list (cons 'exit-code -999)
            (cons 'stdout "")
            (cons 'stderr "spawn failed"))))

(check "shell exit code" (cdr (assoc 'exit-code shell-result)) 42)
(check "shell stdout" (cdr (assoc 'stdout shell-result)) "out\n")
(check "shell stderr" (cdr (assoc 'stderr shell-result)) "err\n")
(check "shell pipeline" (cdr (assoc 'stdout shell-pipeline)) "right\n")
(check "shell redirection" (cdr (assoc 'stdout shell-redirection)) "visible\n")
(check "argv does not use shell" (cdr (assoc 'stdout argv-result)) "literal;not-shell\n")
(check "argv keeps metacharacters literal"
       (cdr (assoc 'stdout argv-metachars))
       "literal|pipe literal>redir exit 42\n")
(check "legacy process-spawn simple command"
       (cdr (assoc 'stdout legacy-simple))
       "legacy-simple\n")
(check "legacy process-spawn shell-compatible command"
       (cdr (assoc 'exit-code legacy-shell-compatible))
       42)
(check "legacy process-spawn shell-compatible stdout"
       (cdr (assoc 'stdout legacy-shell-compatible))
       "legacy-shell\n")
(check "explicit shell runs shell builtins" shell-builtin-code 0)
(check "read-all copies owned buffer before free" read-once-first "owned-buffer")
(check "read-all second call is empty after ownership transfer" read-once-second "")
(check "process-wait returns exited sentinel" wait-exit-result 0)
(check "process-exit-code preserves nonzero status" wait-exit-code 7)
(check "process-pid returns positive pid" (> timeout-pid 0) #t)
(check "process-wait returns timeout sentinel" timeout-wait 1)
(check "process-wait after process-kill exits" timeout-kill-wait 0)
(check "process-exit-code after kill reports observed status"
       timeout-exit-code-observed?
       #t)
(check "run-argv-capture timeout exit code"
       (cdr (assoc 'exit-code argv-timeout-result))
       124)
(check "argv env overlay preserves inherited environment" env-wait 0)
(check "argv options preserve env and lifecycle options" options-wait 0)
(check "binary stdout keeps embedded NUL and native length"
       binary-result
       (cons "left\0right" 10))
(check "destroyed process handle is tombstoned" stale-pid 0)
(check "use after close returns diagnostic sentinel" stale-wait -1)
(check "captured closed handle remains a diagnostic tombstone"
       closure-after-close
       0)
(check "concurrent spawn bound rejects excess child" bound-c #f)
(check "process-destroy kills running child"
       (cdr (assoc 'exit-code destroy-kill-check))
       1)

(if (> failed 0)
    (exit 1)
    (begin
      (display "PASS: subprocess shell and argv contracts")
      (newline)
      (exit 0)))
EOF

ESHKOL_PATH=. ESHKOL_SUBPROC_MAX_CONCURRENT=2 "$RUN" -r "$WORK/subprocess_api.esk"

# Repeat the same reduction through the native AOT path. The source exits
# nonzero on any failed check, so this is an independent native-engine proof.
AOT="$WORK/subprocess_api_aot"
if ! ESHKOL_PATH=. ESHKOL_SUBPROC_MAX_CONCURRENT=2 "$RUN" \
        "$WORK/subprocess_api.esk" -o "$AOT"; then
    echo "FAIL: subprocess AOT compilation"
    exit 1
fi
if ! ESHKOL_SUBPROC_MAX_CONCURRENT=2 "$AOT"; then
    echo "FAIL: subprocess AOT execution"
    exit 1
fi
