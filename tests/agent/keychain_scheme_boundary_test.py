#!/usr/bin/env python3
"""Exercise agent.keychain wrappers with local Scheme raw-call stubs only."""
import argparse
from pathlib import Path
import subprocess
import tempfile


MOCK_RAW = r'''
(define *keychain-test-value* "")
(define *keychain-test-race?* #f)
(define *keychain-test-raw-calls* 0)
(define (keychain-size-raw account account-len)
  (set! *keychain-test-raw-calls* (+ *keychain-test-raw-calls* 1))
  (string-byte-length *keychain-test-value*))
(define (keychain-get-into-raw account account-len output capacity)
  (set! *keychain-test-raw-calls* (+ *keychain-test-raw-calls* 1))
  (if (or *keychain-test-race?*
          (not (= capacity (string-byte-length *keychain-test-value*))))
      -1
      (let loop ((index 0))
        (if (>= index capacity)
            capacity
            (begin
              (string-set! output index (string-ref *keychain-test-value* index))
              (loop (+ index 1)))))))
(define (keychain-set-raw account account-len value value-len)
  (set! *keychain-test-raw-calls* (+ *keychain-test-raw-calls* 1))
  (set! *keychain-test-value* value)
  1)
(define (keychain-delete-raw account account-len)
  (set! *keychain-test-raw-calls* (+ *keychain-test-raw-calls* 1)) 1)
(define (keychain-has-raw account account-len)
  (set! *keychain-test-raw-calls* (+ *keychain-test-raw-calls* 1)) 1)
'''

SCHEME_TEST = r'''
(require core.capabilities)
(load "@WRAPPER@")
(define (check condition message)
  (unless condition (error message)))
(define (denied? thunk)
  (eq? (guard (e (#t 'denied)) (thunk)) 'denied))

(capability-install-policy! '(file-read))
(check (and (denied? (lambda () (keychain-get "acct")))
            (denied? (lambda () (keychain-set! "acct" "value")))
            (denied? (lambda () (keychain-delete "acct")))
            (denied? (lambda () (keychain-has? "acct")))
            (= *keychain-test-raw-calls* 0))
       "keychain denial called a raw stub")

(capability-install-policy! '(keychain))
(set! *keychain-test-value* " \tA\x00;\n ")
(check (string=? (keychain-get "acct") *keychain-test-value*)
       "get did not preserve exact bytes")
(check (keychain-set! "acct" *keychain-test-value*)
       "set rejected exact bytes")
(check (string=? (keychain-get "acct") *keychain-test-value*)
       "set/get did not preserve exact bytes")
(check (keychain-set! "acct" "") "set rejected the empty value")
(check (string=? (keychain-get "acct") "") "get lost the empty value")
(check (keychain-has? "acct") "has? failed with the allowed capability")
(check (keychain-delete "acct") "delete failed with the allowed capability")

(set! *keychain-test-value* "race")
(set! *keychain-test-race?* #t)
(check (not (keychain-get "acct")) "changed-size read did not fail closed")
(set! *keychain-test-race?* #f)

(define calls-before-bounds *keychain-test-raw-calls*)
(define oversized (make-string (+ KEYCHAIN_MAX_BYTES 1) #\space))
(check (not (keychain-get oversized)) "oversized account was accepted")
(check (not (keychain-set! "acct" oversized)) "oversized value was accepted")
(check (= *keychain-test-raw-calls* calls-before-bounds)
       "bounded-input rejection called a raw stub")
(display "PASS: keychain Scheme denial, exact-byte, empty, race, and bound checks\n")
'''


def check_source_guard(wrapper: str) -> None:
    forbidden = ["(system ", "read-file", "file-delete", "/tmp/",
                 "shell-escape", "string-trim", "security "]
    present = [token for token in forbidden if token in wrapper]
    if present:
        raise AssertionError(f"forbidden keychain transport tokens: {present}")
    names = ["keychain-get", "keychain-set!", "keychain-delete", "keychain-has?"]
    for name in names:
        start = wrapper.find("(define (" + name + " ")
        end = wrapper.find("\n(define (", start + 1)
        body = wrapper[start:end if end >= 0 else len(wrapper)]
        if start < 0 or "(capability-require! 'keychain)" not in body:
            raise AssertionError(f"missing keychain capability guard: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eshkol-run", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    wrapper_source = source_root / "lib/agent/keychain.esk"
    wrapper = wrapper_source.read_text()
    check_source_guard(wrapper)
    mock_wrapper = "\n".join(
        line for line in wrapper.splitlines() if not line.startswith("(extern "))
    mock_wrapper = mock_wrapper.replace(
        "(provide keychain-get keychain-set! keychain-delete keychain-has?)",
        "(provide keychain-get keychain-set! keychain-delete keychain-has? "
        "KEYCHAIN_MAX_BYTES *keychain-test-value* *keychain-test-race?* "
        "*keychain-test-raw-calls*)",
    )
    mock_wrapper = mock_wrapper.replace(
        "(define KEYCHAIN_MAX_BYTES 1048576)",
        "(define KEYCHAIN_MAX_BYTES 1048576)\n" + MOCK_RAW,
    )
    if "(extern " in mock_wrapper:
        raise AssertionError("mock wrapper still contains a native extern")

    with tempfile.TemporaryDirectory(prefix="eshkol-keychain-scheme-") as tmp:
        temp = Path(tmp)
        wrapper_path = temp / "keychain_mock.esk"
        test_path = temp / "keychain_boundary.esk"
        wrapper_path.write_text(mock_wrapper)
        test_path.write_text(SCHEME_TEST.replace("@WRAPPER@", str(wrapper_path)))
        result = subprocess.run(
            [args.eshkol_run, "--run", str(test_path)],
            check=False, capture_output=True, text=True, timeout=60,
        )
        completed = "PASS: keychain Scheme denial, exact-byte, empty, race, and bound checks"
        if result.returncode != 0 or completed not in result.stdout or "FAIL:" in result.stdout:
            raise RuntimeError(
                f"Scheme boundary test failed ({result.returncode}):\n"
                f"{result.stdout}{result.stderr}"
            )
        print(result.stdout, end="")


if __name__ == "__main__":
    main()
