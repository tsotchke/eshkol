"""Replay the closure theorem, AOT/JIT outputs and observed Eshkol memo entries."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

from hashlib import sha256

from effects import check as check_effects, forms, render
from publication import (HEAP_TAGS, check_faults, check_source, check_store_observations,
                         cross_nested_return, must_reject, observe_pair_stores)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def once(source, target, replacement):
    if source.count(target) != 1:
        raise ValueError(f"observer site changed: {target}")
    return source.replace(target, replacement)


def instrument(source):
    source = once(source, "(define (blc-fast-run term fuel)",
                  (HERE / "observe.esk").read_text() + "\n(define (blc-fast-run term fuel)\n  (audit-start term fuel)")
    source = once(source, "(set! suspensions (+ suspensions 1))",
                  "(audit-save cell) (set! suspensions (+ suspensions 1))")
    source = once(source, "(set! forces (+ forces 1))",
                  '(audit-operation "FORCE" item) (set! forces (+ forces 1))')
    source = once(source, "(set-cdr! data item)",
                  '(audit-value-event "RETURN" data item) (audit-write! data item)')
    source = once(source, "(set-car! data 'memo)", "(audit-publish! data)")
    quote = "((eq? (car item) 'closure)\n"
    source = once(source, quote, quote + "              (set! audit-depth (max audit-depth (+ data 1)))\n")
    source = once(source, "(loop 'value (cdr item)",
                  """(if (eq? (car item) 'level)
                (begin (display "LEVEL ") (display (cdr item)) (newline)))
            (loop 'value (cdr item)""")
    source = once(source, "(set! hits (+ hits 1))",
                  '(begin (audit-value-event "HIT" item (cdr item)) (audit-event "HEAD-HIT" item) (set! hits (+ hits 1)))')
    for status in ("normal", "exhausted"):
        report = f"(list '{status} "
        source = once(source, report, f"(audit-end '{status} forces hits steps item) " + report)
    return source



def execute(argv, cwd, env, timeout=120):
    p = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{argv}: {p.stderr[-600:]} {p.stdout[-600:]}")
    return p.stdout


# Canonical source forms transcribed by the syntax proof or used at its
# boundary. Bind the substitution oracle, its cap and the codec helpers. The public
# blc-eval dispatch has its separate cost-correspondence replay.
SYNTAX_FORMS = """blc-var blc-lam blc-app blc-tagged? blc-var? blc-lam? blc-app?
blc-var-index blc-lam-body blc-app-fun blc-app-arg blc-ones blc-encode blc-decode
blc-decode-at blc-decode-compound blc-decode-var blc-shift blc-subst blc-beta
blc-step blc-max-steps blc-eval-loop""".split()


def syntax_source(core):
    definitions = {}
    for form in forms(core):
        if not isinstance(form, list) or form[:1] != ["define"]:
            continue
        name = form[1][0] if isinstance(form[1], list) else form[1]
        if name in SYNTAX_FORMS:
            if name in definitions:
                raise ValueError(f"duplicate syntax definition {name}")
            definitions[name] = render(form)
    source = "\n".join(definitions[name] for name in SYNTAX_FORMS)
    return sha256(source.encode()).hexdigest()


def check_syntax(work, env, lean, runner, control, core):
    if syntax_source(core) != "8a87b7f55e77b839526e03d065887bb62784e19bba8dadea0018b70b3df17da7":
        raise ValueError("reviewed core.blc syntax definitions changed")
    # The first case substitutes an open variable under a binder. The replay
    # also includes every input sent to the uninstrumented evaluator.
    capture = "01000011010"
    inputs = [capture] + [row.split()[2] for row in control.splitlines() if row.startswith("REPORT ")]
    if len(inputs) == 1:
        raise ValueError("missing control inputs for syntax replay")
    (work / "syntax.bits").write_text("\n".join(inputs) + "\n")
    source = work / "SyntaxRows.lean"
    source.write_text('import Syntax\n#eval Verificatum.Syntax.auditRows "syntax.bits"\n')
    rows = execute([lean, source.name], work, env)
    if len(rows.splitlines()) != len(inputs):
        raise ValueError("incomplete syntax oracle")
    corpus = work / "syntax.rows"
    corpus.write_text(rows)
    test_env = {**env, "ESHKOL_BLC_SYNTAX_ROWS": str(corpus)}
    suite = HERE / "syntax_test.esk"
    exe = work / "syntax-test"
    execute([runner, "-I", ROOT / "lib", "-O", "2", "-o", exe, suite], work, test_env)
    for mode, cmd in (("AOT", [exe]), ("JIT", [runner, "-I", ROOT / "lib", "-r", suite])):
        if f"Syntax rows passed: {len(inputs)}" not in execute(cmd, work, test_env).splitlines():
            raise ValueError(f"incomplete {mode} syntax replay")
        print(f"{mode} matched {len(inputs)} constructor, codec and one-step rows.")
    mutant = work / "syntax-mutant"
    (mutant / "core").mkdir(parents=True)
    altered = once(core, "(>= k cutoff)", "(> k cutoff)")
    (mutant / "core/blc.esk").write_text(altered)
    if syntax_source(altered) == syntax_source(core):
        raise ValueError("source binding missed the changed cutoff")
    execute([runner, "-I", mutant, "-O", "2", "-o", exe, suite], work, test_env)
    must_reject([exe], test_env, work, f"syntax step {capture}")
    print("Native open substitution with a strict shift cutoff rejected at its first contraction.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", default="lean")
    parser.add_argument("--runner", default="eshkol-run")
    parser.add_argument("--binder-rows", type=Path)
    parser.add_argument("--runtime-probe", action="store_true",
                        help="also inject a duplicate in the linked Linux AOT tagged setter")
    parser.add_argument("--ir-sites", type=Path, help="write the direct-call LLVM IR effect inventory (llvmlite)")
    parser.add_argument("--linked-refinement", action="store_true",
                        help="check predicate transfers, linked x86-64 primitives and direct-store counts (Callgrind)")
    opts = parser.parse_args()
    lean, runner = shutil.which(opts.lean), shutil.which(opts.runner)
    if not lean or not runner:
        parser.error("both lean and eshkol-run are required")
    model = HERE / "Lazy.lean"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
        env["LEAN_PATH"] = str(work)
        for name in ("Lazy", "Heap", "Publication", "Lookup", "Observation", "Dispatch", "Readback",
                     "Correspondence", "Syntax", "LinkedPair"):
            shutil.copyfile(HERE / f"{name}.lean", work / f"{name}.lean")
            execute([lean, "-o", name + ".olean", name + ".lean"], work, env)
        fixtures = HERE / "fixtures"
        control_lib = work / "control-lib"
        (control_lib / "core").mkdir(parents=True)
        shutil.copyfile(ROOT / "lib/core/blc.esk", control_lib / "core/blc.esk")
        shutil.copyfile(fixtures / "core/blc-fast.esk", control_lib / "core/blc-fast.esk")
        sites = check_effects(control_lib / "core", HERE / "effects.tsv")
        print(f"Checked {sites} classified source sites; rejected a transitive imported write.")
        for mode, files in (("stores", '"observed.log"'), ("calls", '"observed.log","control.log"')):
            (work / f"{mode}.lean").write_text(
                f'import Observation\n#eval Verificatum.Observation.audit ["{mode}",{files}]\n')
        check_source(work, env, lean, execute, (fixtures / "core/blc-fast.esk").read_text())
        rows = execute([lean, "--run", model, "--emit"], work, env)
        corpus = work / "rows"
        corpus.write_text(rows)
        count = len(rows.splitlines())
        env["ESHKOL_BLC_MODEL_ROWS"] = str(corpus)
        if opts.binder_rows:
            env["ESHKOL_BINDER_ROWS"] = str(opts.binder_rows.resolve())
        suite = (fixtures / "suite.esk").read_text()
        # Observe only arguments and the public return value in the control
        # build. This wrapper never sees private cells or internal dispatch.
        wrapper = """
(define (call-fast term fuel)
  (let ((result (blc-fast-run term fuel)))
    (display "REPORT ") (display fuel) (display " ") (display (blc-encode term))
    (display " ") (display (car result)) (display " ")
    (if (cadr result) (display (blc-encode (cadr result))) (display "#f"))
    (display " ") (display (cddr result)) (newline)
    result))
"""
        test = work / "suite.esk"
        test.write_text(once(suite.replace("(blc-fast-run ", "(call-fast "),
                             "(require core.blc-fast)", "(require core.blc-fast)\n" + wrapper))
        exe = work / "test"
        dump = ["-i"] if opts.ir_sites or opts.linked_refinement else []
        execute([runner, *dump, "-I", control_lib, "-O", "2", "-o", exe, test], work, env)
        if dump:
            from ir_effects import inspect
            listing = work / "ir-sites.tsv"
            with listing.open("w") as out:
                inspect(work / "test.ll", out, require_transfers=opts.linked_refinement)
            if opts.ir_sites:
                with listing.open("rb") as data, opts.ir_sites.open("xb") as out:
                    shutil.copyfileobj(data, out)
        for mode, command in [("AOT", [exe]), ("JIT", [runner, "-I", control_lib, "-r", test])]:
            output = execute(command, work, env)
            if f"Closure model rows passed: {count}\n" not in output or "BLC fast tests passed" not in output:
                raise ValueError(f"incomplete {mode} replay: {output[-500:]}")
            if opts.binder_rows:
                binder_count = len(opts.binder_rows.read_text().splitlines())
                if f"Binder rows passed: {binder_count}\n" not in output:
                    raise ValueError(f"incomplete {mode} binder replay")
            print(mode, [line for line in output.splitlines() if line.endswith("tests passed")
                         or line.startswith(("Closure model rows passed:", "Binder rows passed:"))], flush=True)

        # A separate native build traces observed writes *after* mutation and
        # observed hits *before* use. Keep the production gate above uninstrumented.
        audit_env = {**env, "ESHKOL_BINDER_ROWS": ""}
        baseline = execute([exe], work, audit_env)
        check_syntax(work, audit_env, lean, runner, baseline, (ROOT / "lib/core/blc.esk").read_text())
        core = work / "core"
        core.mkdir()
        shutil.copyfile(ROOT / "lib/core/blc.esk", core / "blc.esk")
        instrumented = instrument((fixtures / "core/blc-fast.esk").read_text())
        (core / "blc-fast.esk").write_text(observe_pair_stores(instrumented))
        # Use the same regression terms, including Tromp U, and the model rows.
        # Optional large binder corpus was already exercised by both normal runs.
        execute([runner, "-I", work, "-O", "2", "-o", exe, test], work, audit_env)
        trace = execute([exe], work, audit_env)
        (work / "observed.log").write_text(trace)
        (work / "control.log").write_text(baseline)
        for gate in ("stores.lean", "calls.lean"):
            print(execute([lean, gate], work, env).strip())
        expected = [s for s in baseline.splitlines() if s.startswith("REPORT ")]
        stores = check_store_observations(trace)
        print(f"Matched {len(expected)} control invocations and {stores} native pair-store observations.")
        heap_rows = [line for line in trace.splitlines() if line.split(" ", 1)[0] in HEAP_TAGS]
        heap_path = work / "heap.events"
        heap_path.write_text("\n".join(heap_rows) + "\n")
        check = work / "Audit.lean"
        check.write_text('import Heap\n#eval Verificatum.Heap.auditHeap "heap.events"\n')
        print(execute([lean, check.name], work, env).strip())
        registers = work / "RegisterAudit.lean"
        registers.write_text('import Readback\n#eval Verificatum.Readback.audit "heap.events"\n')
        print(execute([lean, registers.name], work, env).strip())
        # A stream reconstructed only from surviving START records still looks
        # valid after a whole invocation vanishes. Keep its public call receipt.
        lines = trace.splitlines()
        first = next(i for i, s in enumerate(lines) if s.startswith("START 0 "))
        receipt = next(i for i in range(first + 1, len(lines)) if lines[i].startswith("REPORT "))
        segment = [s for s in lines[first:receipt] if s.partition(" ")[0] in HEAP_TAGS]
        if segment != [lines[first], "STOP exhausted 0"]:
            raise ValueError("expected the zero-fuel call to perform no heap operations")
        missing = lines[:first] + lines[receipt:]
        heap_path.write_text("\n".join(s for s in missing if s.partition(" ")[0] in HEAP_TAGS) + "\n")
        execute([lean, check.name], work, env)
        for start_at in (receipt, receipt + 1):
            # Remove the whole invocation, once retaining its public return and
            # once dropping that too. The control build keeps its own receipt.
            omitted = lines[:first] + lines[start_at:]
            (work / "observed.log").write_text("\n".join(omitted) + "\n")
            execute([lean, "stores.lean"], work, env)
            must_reject([lean, "calls.lean"], env, work, "unaccounted invocation")
        print("Zero-event invocation omission rejected with or without its observed public return.")
        altered, depth = cross_nested_return(heap_rows)
        heap_path.write_text("\n".join(altered) + "\n")
        must_reject([lean, check.name], env, work, 'expected some "RETURN ')
        print(f"Crossed nested return rejected; maximum pending update depth {depth}.")
        wrong = next(i for i, row in enumerate(heap_rows) if row.startswith("PUBLISH "))
        tag, address, value = heap_rows[wrong].split(" ", 2)
        heap_rows[wrong] = f"{tag} {int(address) + 1} {value}"
        heap_path.write_text("\n".join(heap_rows) + "\n")
        mutation = subprocess.run([lean, check.name], cwd=work, env=env,
                                  capture_output=True, text=True, timeout=120)
        if mutation.returncode == 0 or "heap event mismatch" not in mutation.stdout + mutation.stderr:
            raise ValueError("altered destination address escaped the heap check")
        print("Altered publication address rejected.")
        events = [line.removeprefix("HEAD-") for line in trace.splitlines()
                  if line.startswith(("FILL ", "HEAD-HIT "))]
        seen_fills = seen_hits = completions = 0
        for line in trace.splitlines():
            if line.startswith("FILL "):
                seen_fills += 1
            elif line.startswith("HEAD-HIT "):
                seen_hits += 1
            elif line.startswith("END "):
                _, status, forces, hits = line.split()
                if seen_hits != int(hits) or seen_fills > int(forces):
                    raise ValueError(f"missing or duplicate events at {line}")
                if status == "normal" and seen_fills != int(forces):
                    raise ValueError(f"unaccounted completed force at {line}")
                completions += 1
                seen_fills = seen_hits = 0
        starts = sum(line.startswith("START ") for line in heap_rows)
        stops = sum(line.startswith("STOP ") for line in heap_rows)
        if (completions < count or starts != completions or stops != completions
                or seen_fills or seen_hits or "BLC fast tests passed" not in trace):
            raise ValueError("incomplete observer run")
        path = work / "events"
        path.write_text("\n".join(events) + "\n")
        print(execute([lean, "--run", model, "--heads", path], work, env).strip())
        # Choose a syntactically valid wrong value for the first real memo hit.
        i = next(i for i, r in enumerate(events) if r.startswith("HIT "))
        tag, source, original = events[i].split()
        events[i] = f"{tag} {source} " + ("0010" if original != "0010" else "10")
        path.write_text("\n".join(events) + "\n")
        negative = subprocess.run([lean, "--run", model, "--heads", path], cwd=work,
                                  capture_output=True, text=True, timeout=120)
        if negative.returncode == 0 or "wrong cached head on HIT" not in negative.stderr:
            raise ValueError("altered cache value did not fail at the expected check")
        print("Altered memo hit rejected.")
        check_faults(work, env, lean, runner, execute, core / "blc-fast.esk", instrumented, HERE, opts.runtime_probe or opts.linked_refinement, opts.linked_refinement)


if __name__ == "__main__":
    main()
