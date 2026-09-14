#!/usr/bin/env python3
"""Determinism gate for language-coverage evidence.

The coverage records (P/G/A/R/O/C lines) name a source position for every
construct the compiler parsed, generated, accepted, rejected or executed. AST
nodes synthesised while lowering a form (internal defines, body sequences,
named let, do, case, record types, macro output, ...) used to be built
without initialising their location, so their records carried whatever bytes
were in memory: a column in the hundreds of millions that changed from run
to run, and therefore coverage evidence that was not reproducible.

This gate compiles and runs a small corpus that exercises those lowering
paths, twice, into two fresh directories, under both the JIT and AOT, and
requires:
  * identical sorted record sets across the two runs, and
  * every record's line and column to lie inside the source file it names.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
CORPUS_NAME = "coverage_determinism_corpus.esk"

CORPUS = r""";; Language-coverage determinism corpus: forms that lower into
;; synthesised AST nodes.
(define-syntax swap!
  (syntax-rules ()
    ((_ a b) (let ((tmp a)) (set! a b) (set! b tmp)))))

(define-record-type point
  (make-point x y)
  point?
  (x point-x)
  (y point-y set-point-y!))

(define (internal-defines n)
  (define base 10)
  (define (scale v) (* v base))
  (display "internal ")
  (scale n))

(define (lambda-body-defines)
  ((lambda (k)
     (define twice (* k 2))
     (define (add v) (+ v twice))
     (add 1))
   5))

(define (sum-to n)
  (let loop ((i 0) (acc 0))
    (if (> i n) acc (loop (+ i 1) (+ acc i)))))

(define (do-count n)
  (do ((i 0 (+ i 1))
       (acc '() (cons i acc)))
      ((= i n) (reverse acc))))

(define (classify x)
  (cond ((assv x '((1 . one) (2 . two))) => cdr)
        ((> x 10) 'big)
        (else 'other)))

(define (kind x)
  (case x
    ((a e i o u) 'vowel)
    ((w y) 'semi)
    (else 'consonant)))

(define arity
  (case-lambda
    ((a) (list 'one a))
    ((a b) (list 'two a b))))

(define (safe-div a b)
  (guard (e (#t 'division-error))
    (if (= b 0) (raise 'divide-by-zero) (/ a b))))

(define param (make-parameter 1))

(define (sequence-body x)
  (display "seq ")
  (display x)
  (newline)
  (* x 3))

(define u 1)
(define v 2)
(define p (make-point 1 2))
(swap! u v)
(set-point-y! p 5)
(display (list (internal-defines 3) (lambda-body-defines) (sum-to 10)
               (do-count 3) (classify 2) (classify 20) (kind 'e)
               (arity 1) (arity 1 2) (safe-div 1 0)
               (parameterize ((param 7)) (param))
               (point-x p) (point-y p) u v
               `(a ,u ,@(list v 3))
               (let-values (((q r) (values 7 2))) (list q r))
               (force (delay (+ 1 2)))
               (when (> u 0) 'pos) (unless (> u 0) 'neg)
               (let* ((a 1) (b (+ a 1))) b)
               (letrec ((ev? (lambda (n) (if (= n 0) #t (od? (- n 1)))))
                        (od? (lambda (n) (if (= n 0) #f (ev? (- n 1))))))
                 (ev? 10))
               (sequence-body 4)))
(newline)
"""

LOCATED_KINDS = {"P", "G", "A", "R", "O", "C"}


class LanguageCoverageDeterminismTest(unittest.TestCase):
    eshkol_run = None
    lib_dir = None
    work_root = None

    def fresh_dir(self, label):
        root = pathlib.Path(tempfile.mkdtemp(prefix=label + "-",
                                             dir=self.work_root))
        self.addCleanup(shutil.rmtree, root, True)
        return root

    def run_checked(self, argv, env, cwd):
        result = subprocess.run(
            argv, cwd=cwd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=600, check=False)
        self.assertEqual(result.returncode, 0,
                         "%s\n%s\n%s" % (argv, result.stdout, result.stderr))
        return result

    def instrumented_run(self, mode, label):
        root = self.fresh_dir(label)
        source = root / CORPUS_NAME
        source.write_text(CORPUS, encoding="utf-8")
        trace_dir = root / "trace"
        trace_dir.mkdir()
        env = os.environ.copy()
        env["ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR"] = str(trace_dir)
        env.pop("ESHKOL_LANGUAGE_COVERAGE_HOOK_STATS", None)
        env.setdefault("ESHKOL_PATH", str(REPO / "lib"))
        if mode == "jit":
            result = self.run_checked(
                [self.eshkol_run, "-r", str(source), "-L" + self.lib_dir],
                env, root)
        else:
            binary = root / "corpus"
            self.run_checked(
                [self.eshkol_run, str(source), "-o", str(binary),
                 "-L" + self.lib_dir], env, root)
            result = self.run_checked([str(binary)], env, root)
        self.assertIn("internal", result.stdout)
        records = []
        for path in sorted(trace_dir.glob("*.tsv")):
            records.extend(path.read_text(encoding="utf-8").splitlines())
        self.assertTrue(records, "%s run wrote no coverage records" % mode)
        return root, records

    @staticmethod
    def normalise(root, records):
        prefix = str(root)
        return sorted({record.replace(prefix, "<RUN>") for record in records})

    def assert_locations_in_range(self, records):
        line_cache = {}
        checked = 0
        problems = []
        for record in records:
            fields = record.split("\t")
            if fields[0] not in LOCATED_KINDS or len(fields) < 4:
                continue
            path = pathlib.Path(fields[1])
            if path not in line_cache:
                try:
                    line_cache[path] = path.read_bytes().split(b"\n")
                except OSError:
                    line_cache[path] = None
            lines = line_cache[path]
            if lines is None:
                continue
            line, column = int(fields[2]), int(fields[3])
            checked += 1
            if not 1 <= line <= len(lines):
                problems.append("line out of range: %r" % record)
                continue
            width = len(lines[line - 1])
            if not 1 <= column <= width + 1:
                problems.append("column out of range (line has %d bytes): %r"
                                % (width, record))
        self.assertGreater(checked, 0, "no record named a readable source")
        self.assertEqual(problems[:20], [],
                         "%d records carry out-of-range locations"
                         % len(problems))

    def assert_deterministic(self, mode):
        first_root, first = self.instrumented_run(mode, mode + "-run1")
        second_root, second = self.instrumented_run(mode, mode + "-run2")
        self.assert_locations_in_range(first)
        self.assert_locations_in_range(second)
        a = self.normalise(first_root, first)
        b = self.normalise(second_root, second)
        only_first = sorted(set(a) - set(b))
        only_second = sorted(set(b) - set(a))
        self.assertEqual(
            (only_first[:10], only_second[:10]), ([], []),
            "%s coverage records differ between identical runs "
            "(%d only in run 1, %d only in run 2)"
            % (mode, len(only_first), len(only_second)))

    def test_jit_records_are_deterministic_and_in_range(self):
        self.assert_deterministic("jit")

    def test_aot_records_are_deterministic_and_in_range(self):
        self.assert_deterministic("aot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eshkol-run", required=True)
    parser.add_argument("--lib-dir", required=True)
    parser.add_argument("--work-dir", default=None,
                        help="parent for the fresh run directories")
    args, unittest_args = parser.parse_known_args()
    LanguageCoverageDeterminismTest.eshkol_run = os.path.abspath(args.eshkol_run)
    LanguageCoverageDeterminismTest.lib_dir = os.path.abspath(args.lib_dir)
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        LanguageCoverageDeterminismTest.work_root = os.path.abspath(args.work_dir)
    unittest.main(argv=[__file__] + unittest_args)


if __name__ == "__main__":
    main()
