"""Check the bounded shape proof and inspect the native guard emitted at -O0."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRODUCER = """prelude
import Init.Core
namespace Producer
theorem same : ∀ _a _b c _d : Prop, c → c := fun _ _ _ _ h => h
end Producer
"""
SITE = """(define (audited-product a b) (tensor-matmul a b))
(define left (reshape (vector 1 2 3 4 5 6) 2 3))
(define original (reshape (vector 1 2 3 4 5 6) 3 2))
(define rhs (cons original '()))
(define writer rhs)
(define (use-site) (tensor-shape (audited-product left (car rhs))))
(display (use-site))
(newline)
(set-car! writer (reshape (vector 1 2 3 4 5 6) 2 3))
(display (tensor-shape original))
(newline)
(display (tensor-shape (car rhs)))
(newline)
(display (use-site))
(newline)
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(args, cwd, env, code=0):
    proc = subprocess.run(args, cwd=cwd, env=env, capture_output=True, timeout=120)
    if proc.returncode != code:
        raise ValueError(f"{args[0]} returned {proc.returncode}: "
                         f"{(proc.stdout + proc.stderr)[-1000:]!r}")
    return proc.stdout


def check_proof(work, env):
    shutil.copy(HERE / "GuardAuthority.lean", work)
    (work / "Producer.lean").write_text(PRODUCER)
    for module in ("Producer", "GuardAuthority"):
        run(["lean", "-DwarningAsError=true", "-o", module + ".olean", module + ".lean"], work, env)
    model_theorems = ["goal_iff", "may_omit_inner_eq", "residual_guard_preserved", "stale_refused",
                      "stable_at_use", "guard_after_writes", "descriptors_at_use"]
    audit = "import GuardAuthority\n" + "".join(
        f"#print axioms Verificatum.Guard.{name}\n" for name in model_theorems)
    (work / "Audit.lean").write_text(audit)
    axioms = run(["lean", "Audit.lean"], work, env).decode().splitlines()
    theorem = "Producer.same"
    proof = run(["leanexport", "Producer", "--", theorem], work, env)
    (work / "proof.ndjson").write_bytes(proof)
    receipts = {}
    for name, status, reason, code in (("equalThree", "ACCEPT", "checked", 0),
                                       ("threeFour", "REJECT", "target", 1)):
        target = "Verificatum.Guard." + name
        data = run(["leanexport", "GuardAuthority", "--", target], work, env)
        (work / "target.ndjson").write_bytes(data)
        raw = run([ROOT / "verification/lean-kernel/check", "proof.ndjson", theorem,
                   "target.ndjson", target], work, env, code)
        result = json.loads(raw)
        if (result["status"], result["reason"], result["assumptions"]) != (status, reason, []):
            raise ValueError(f"Unexpected proof verdict: {result}")
        receipts[name] = {"target_sha256": sha(data), "verdict": result}
    return {"proof_sha256": sha(proof), "targets": receipts, "model_axioms": axioms,
            "model_sha256": sha((work / "GuardAuthority.lean").read_bytes())}


def check_invalidation_control(work, env):
    source = (work / "GuardAuthority.lean").read_text()
    rule = "if loc = dest then ⟨r, c, (h loc).version + 1⟩ else h loc"
    if source.count(rule) != 1:
        raise ValueError("Expected one modeled write rule")
    weakened = source.replace(rule, rule.replace(" + 1", ""))
    (work / "NoBump.lean").write_text(weakened)
    output = run(["lean", "NoBump.lean"], work, env, code=1).decode()
    # A parse failure or an unrelated broken proof cannot satisfy this control.
    theorem = weakened.index("private theorem aliased_write_expires")
    decision = weakened.index("  decide", theorem)
    line = 1 + weakened.count("\n", 0, decision)
    diagnostic = re.search(rf"^NoBump\.lean:{line}:\d+: error: Tactic `decide` proved that the proposition"
                           r"\n  [^\n]*Event\.write rhsRoot Width\.two Width\.three[^\n]*\n"
                           r"is false$", output, re.M)
    if diagnostic is None:
        raise ValueError(f"Missing expected stale-stamp counterexample: {output}")
    return {"layer": "Lean model", "mutation": "write preserves version", "stale_stamp": "accepted"}


def check_descriptor(work, env, runtime):
    # Link the checkout's constructors and shape readers with the release's
    # allocator. This keeps the observed FFI source revision explicit.
    ffi = ROOT / "lib/ffi/eshkol_ffi.cpp"
    validator = ROOT / "lib/core/tensor_validation.cpp"
    binary = work / "descriptor-probe"
    run(["c++", "-std=c++17", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
         "-I", ROOT / "inc", "-I", ROOT / "lib/core", HERE / "descriptor-probe.cpp",
         ffi, validator, runtime, "-ldl", "-pthread", "-lm", "-o", binary], work, env)
    lines = run([binary], work, env).decode().splitlines()
    if lines != ["public-ffi: dimensions copied, element mutation observed, shape preserved",
                 "owned-native-descriptor: same object and storage, 3x2 -> 2x3, both shapes valid"]:
        raise ValueError(f"Unexpected descriptor observations: {lines}")
    return {"observations": lines, "ffi_source_sha256": sha(ffi.read_bytes()),
            "validator_source_sha256": sha(validator.read_bytes()), "runtime_sha256": sha(runtime.read_bytes())}


def inspect_site(ir):
    # The release's unoptimized LLVM spelling. An unfamiliar code shape needs
    # review. Follow SSA uses through the conjunction into the actual branch.
    site = re.search(r"^define [^\n]* @audited-product\([^\n]*\) \{\n(.*?)^\}", ir, re.M | re.S)
    if site is None:
        raise ValueError("Missing audited-product definition")
    body = site[1]
    condition = re.search(
        r"^mm_b_merge:[^\n]*\n  %N = phi [^\n]+\n"
        r"  (%\w+) = icmp eq i64 %K, (%\w+)\n"
        r"  (%\w+) = call i64 @eshkol_matmul_shape_valid\(i64 %M, i64 %K, i64 %N\)\n"
        r"  (%\w+) = icmp ne i64 \3, 0\n"
        r"  (%\w+) = and i1 \1, \4\n"
        r"  br i1 \5, label %mm_shape_ok, label %mm_shape_err$", body, re.M)
    if condition is None or condition[2] == "%K":
        raise ValueError("Expected separate inner equality and size predicates governing this branch")
    if not re.search(r"^  " + re.escape(condition[2]) + r" = load i64, ptr ", body, re.M):
        raise ValueError("Expected a loaded descriptor dimension as the other equality operand")
    if not re.search(r"^mm_shape_err:[^\n]*\n(?:  [^\n]+\n)*?  call void @exit\(i32 1\)\n  unreachable", body, re.M):
        raise ValueError("Unexpected shape failure block")
    return {"function": "audited-product", "path": "dense numeric matmul", "optimization": 0,
            "inner_equality": "emitted", "shape_validity": "emitted", "branch": condition[5]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, type=Path, help="Eshkol native eshkol-run executable")
    parser.add_argument("--runtime", type=Path, help="runtime archive (default: sibling lib/libeshkol-runtime.a)")
    args = parser.parse_args()
    compiler = args.compiler.resolve(strict=True)
    archive = (args.runtime or compiler.parent.parent / "lib/libeshkol-runtime.a").resolve(strict=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        env = dict(os.environ, LEAN_PATH=str(work))
        descriptor = check_descriptor(work, env, archive)
        proof = check_proof(work, env)
        (work / "site.esk").write_text(SITE)
        mutation = check_invalidation_control(work, env)
        runtime = {}
        expected = ["(2 2)", "(3 2)", "(2 3)", "Error: matmul inner dimensions mismatch (3 vs 2)"]
        for opt in (0, 2):
            binary = work / f"alias-O{opt}"
            args = [compiler, "-I", ROOT / "lib", "-O", str(opt), "-o", binary, "site.esk"]
            if opt == 0:
                args.insert(1, "--dump-ir")
            run(args, work, env)
            observed = run([binary], work, env, code=1).decode().splitlines()
            if observed != expected:
                raise ValueError(f"Unexpected native alias observations at -O{opt}: {observed}")
            runtime[f"-O{opt}"] = {"observations": observed, "exit_code": 1,
                                   "executable_sha256": sha(binary.read_bytes())}
        ir = (work / "alias-O0.ll").read_bytes()
        print(json.dumps({
            "r1_b": "open", "authority_model": "checked", **proof,
            "model_invalidation_control": mutation, "compiler_sha256": sha(compiler.read_bytes()),
            "program_sha256": sha(SITE.encode()),
            "descriptor_boundary": descriptor,
            "compiler_decision": {"input": "no_authorization_supplied", "status": "unobserved",
                                  "permit_consumption_trace": None},
            "emission": {"ir_sha256": sha(ir), **inspect_site(ir.decode())},
            "runtime": runtime,
        }, indent=2))


if __name__ == "__main__":
    main()
