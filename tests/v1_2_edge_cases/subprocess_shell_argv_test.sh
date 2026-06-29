#!/bin/bash
# subprocess_shell_argv_test.sh — explicit shell vs argv process semantics.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

if [ ! -x /bin/sh ] || [ ! -x /bin/echo ] || [ ! -x /bin/sleep ] || [ ! -x /bin/kill ]; then
    echo "SKIP: POSIX shell tools unavailable"
    exit 0
fi

WORK=$(mktemp -d -t eshkol_subprocess_api.XXXXXX)
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
  (run-argv-capture (list "/bin/echo" "literal;not-shell") "." 5000 4096))

(define argv-metachars
  (run-argv-capture (list "/bin/echo" "literal|pipe" "literal>redir" "exit 42")
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
  (process-spawn-argv (list "/bin/sleep" "5") "."))

(define timeout-pid
  (if timeout-proc (process-pid timeout-proc) 0))

(define timeout-wait
  (if timeout-proc (process-wait timeout-proc 100) -999))

(define timeout-running
  (if timeout-proc (process-running? timeout-proc) #f))

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

(define argv-timeout-result
  (run-argv-capture (list "/bin/sleep" "5") "." 100 4096))

(define destroy-proc
  (process-spawn-argv (list "/bin/sleep" "30") "."))

(define destroy-pid
  (if destroy-proc (process-pid destroy-proc) 0))

(when destroy-proc
  (process-destroy destroy-proc))

(define destroy-kill-check
  (if (> destroy-pid 0)
      (run-argv-capture (list "/bin/kill" "-0" (number->string destroy-pid))
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
(check "process-running? remains true after timeout" timeout-running #t)
(check "process-wait after process-kill exits" timeout-kill-wait 0)
(check "process-exit-code reports signal status" timeout-exit-code 143)
(check "run-argv-capture timeout exit code"
       (cdr (assoc 'exit-code argv-timeout-result))
       124)
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

ESHKOL_PATH=. "$RUN" -r "$WORK/subprocess_api.esk"
