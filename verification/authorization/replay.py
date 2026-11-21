"""Check the authority model, then run three consumers and their boundary controls."""

import argparse
from dataclasses import replace
import json
import os
import platform
import shutil
import subprocess
import tempfile
import extract_kernel
import kernel
from fractions import Fraction as F
from itertools import product
from pathlib import Path

import gram
import heat
import steane
from authority import (
    Checker,
    Contract,
    Equal,
    Gate,
    Kind,
    Level,
    Need,
    Permission,
    encode,
    fingerprint,
)

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"


def require(value, message):
    if not value:
        raise ValueError(message)


def manifest(*files):
    return encode({p: fingerprint((HERE / p).read_bytes()) for p in files})


def compare_interpreters(work, lean_run):
    # Ask the independently specified Lean operations to check actual CPython
    # answers, including empty policies, empty byte strings and equal prefixes.
    examples = ["import Authority", "open Verificatum.Authority"]
    decision_cases = 0
    for size in range(5):
        for outcomes in product(("checked", "denied", "inconclusive"), repeat=size):
            actual = kernel.decide(outcomes, lambda outcome: outcome)
            terms = ", ".join("Verdict." + v for v in outcomes)
            examples.append(
                f"example : decisionSpec ([{terms}] : List Verdict) id = .{actual} := by decide"
            )
            decision_cases += 1
    release_cases = 0
    for issued, current, payload, expected in product((b"", bytes([0]), bytes([0, 255])), repeat=4):
        released = kernel.release(issued, current, payload, expected)
        result = "none" if released is None else f"some {list(released)}"

        def scope(binding, artifact):
            return f"(Scope.mk ({list(binding)} : List UInt8) ({list(artifact)} : List UInt8) ())"

        examples.append(
            f"example : consume {scope(issued, payload)} {scope(current, expected)} = {result} := by decide"
        )
        release_cases += 1
    (work / "KernelReplay.lean").write_text("\n".join(examples) + "\n")
    lean_run("KernelReplay.lean")
    return {"decision_cases": decision_cases, "release_cases": release_cases}


def audit_kernel(lean):
    source = (HERE / "kernel.py").read_text()
    proof_text = (HERE / "Refinement.lean").read_text()
    ax = " depends on axioms: [propext, Quot.sound]"
    expected = [f"'Verificatum.Authority.{n}'" + ax for n in ("release_sound", "authorize_sound")]
    expected += ["'Verificatum.Authority.consume_bound' does not depend on any axioms"]
    expected += [
        f"'Verificatum.Authority.{n}'" + ax for n in ("decision_refines", "decision_authorizes")
    ]
    expected += ["'Verificatum.Authority.release_refines' depends on axioms: [propext]"]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        def lean_run(*args, must_pass=True):
            return subprocess.run(
                [lean, "-DwarningAsError=true", *args],
                cwd=work,
                env={**os.environ, "LEAN_PATH": str(work)},
                timeout=120,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=must_pass,
            )

        for file in ("Authority.lean", "Refinement.lean"):
            shutil.copyfile(HERE / file, work / file)
        audit = lean_run("-o", "Authority.olean", "Authority.lean").stdout

        def extract(text):
            emitted = extract_kernel.translate(text)
            (work / "Extracted.lean").write_text(emitted)
            lean_run("-o", "Extracted.olean", "Extracted.lean")
            return emitted

        emitted = extract(source)
        audit += lean_run("Refinement.lean").stdout
        require(audit.splitlines() == expected, "unexpected axiom audit")
        comparison = compare_interpreters(work, lean_run)
        mutants = []
        for original, replacement, theorem in (
            ('verdict != "checked"', 'verdict == "checked"', "decision_refines"),
            (" and payload == expected", " or payload == expected", "release_refines"),
        ):
            require(source.count(original) == 1, "mutation site disappeared")
            extract(source.replace(original, replacement))
            result = lean_run("Refinement.lean", must_pass=False)
            start = proof_text.index("theorem " + theorem)
            stop = proof_text.find("\n\n", start)
            lines = range(proof_text.count("\n", 0, start) + 1, proof_text.count("\n", 0, stop) + 2)
            require(
                result.returncode != 0
                and any(
                    f"Refinement.lean:{i}:" in line and "error:" in line
                    for line in result.stdout.splitlines()
                    for i in lines
                ),
                "mutation missed " + theorem,
            )
            mutants.append(theorem)
    # Check the interpreter's callback order on both short-circuit branches.
    for failure in ("denied", "inconclusive"):
        visited = []

        def observe(n):
            visited.append(n)
            return ("checked", failure, "checked")[n]

        require(
            kernel.decide((0, 1, 2), observe) == failure and visited == [0, 1], "observer order"
        )
    return audit, {
        "python_sha256": fingerprint(source.encode()),
        "extracted_sha256": fingerprint(emitted.encode()),
        "extractor_sha256": fingerprint((HERE / "extract_kernel.py").read_bytes()),
        "proof_sha256": fingerprint(proof_text.encode()),
        "rejected_mutants": mutants,
        "cpython_against_lean_model": comparison,
    }


def replay(lean, compiler):
    def version(exe):
        return subprocess.check_output([exe, "--version"], text=True).strip()

    lean_version, compiler_version = version(lean), version(compiler)
    require("version 4.35.0," in lean_version, "replay requires Lean 4.35.0")
    require(compiler_version == "Eshkol Compiler v1.3.6-evolve", "unexpected compiler version")
    proof, translation = audit_kernel(lean)
    binary = heat.build(compiler, (FIX / "heat.esk").read_bytes())
    registry = {
        "guarded": (FIX / "gram.ll").read_bytes(),
        "gram-checker": manifest("gram.py"),
        "epoch": b"original descriptor epoch",
        "runtime": b"reference build; optimized effects unqualified",
        "steane": (FIX / "steane.json").read_bytes(),
        "noise": steane.law(F(1, 10)),
        "steane-checker": manifest(
            "steane.py", "fixtures/schedule.py", "fixtures/pauli.py", "fixtures/faults.py"
        ),
        "solver": binary,
        "grid": heat.GRID,
        "heat-checker": manifest("heat.py", "fixtures/rod.py"),
        "native-build": encode(
            {
                "source": heat.SOURCE,
                "compiler": fingerprint(Path(compiler).read_bytes()),
                "version": compiler_version,
                "machine": platform.machine(),
            }
        ),
    }
    manifests = {
        fingerprint(value): json.loads(value)
        for key, value in registry.items()
        if key.endswith("-checker") or key == "native-build"
    }
    read = registry.copy
    rows = {}
    controls = {}
    transfers = {}

    def check(key, gate, data, expected):
        permission, record = gate.assess(data)
        require(record["status"] == expected, f"{key}: {record}")
        require((permission is not None) == (expected == "authorized"), key)
        rows[key] = record
        return permission

    def blocked(key, gate, token, expected):
        try:
            gate.consume(token, expected)
        except PermissionError as exc:
            controls[key] = str(exc)
        else:
            raise ValueError(key)

    def transfer(key, issued, destination, gate, token, expected):
        blocked(key, gate, token, expected)
        transfers[key] = {
            "issued": rows[issued]["decision_id"],
            "requested": destination,
            "requested_artifact": fingerprint(expected),
            "reason": controls[key],
        }

    review, lowering, native = gram.gates(read)
    guarded = registry["guarded"]
    require(guarded.count(gram.GUARD) == 1, "expected one comparison")
    proposed = guarded.replace(gram.GUARD, gram.PROPOSAL)
    candidate = check("gram.review", review, proposed, "authorized")
    blocked("cross-gate use", lowering, candidate, proposed)
    blocked("receipt as permission", review, rows["gram.review"], proposed)
    transfer(
        "Gram IR review to native release",
        "gram.review",
        "gram.native",
        native,
        candidate,
        proposed,
    )
    check("gram.native", native, proposed, "inconclusive")
    # Same issuer, same policy text, observations and artifact; vary only the privilege.
    stage_gate, _, _ = gram.gates(read)
    stage = check("gram.stage-only", stage_gate, proposed, "authorized")
    baseline_contract = review.contract
    stage_gate.set_contract(replace(baseline_contract, permission=Permission.NATIVE_RELEASE))
    blocked("permission alone changed", stage_gate, stage, proposed)
    swap = check("gram.swap-source", review, proposed, "authorized")
    transfer(
        "substitute artifact with original receipt",
        "gram.swap-source",
        "gram.replacement-review",
        review,
        swap,
        guarded,
    )
    review.consume(check("gram.replacement-review", review, guarded, "authorized"), guarded)
    check("gram.strict", lowering, proposed, "inconclusive")
    fallback = check("gram.fallback", lowering, guarded, "authorized")
    require(lowering.consume(fallback, guarded) == guarded, "guarded fallback identity")
    bogus = proposed.replace(
        b"%vf.right = icmp eq ptr %tensor_ptr28", b"%vf.right = icmp eq ptr %tensor_ptr"
    )
    check("gram.wrong-operand", review, bogus, "denied")
    for name in ("epoch", "runtime"):
        original = check("gram.before-" + name, lowering, guarded, "authorized")
        registry[name] += b" changed"
        blocked("changed " + name, lowering, original, guarded)
    require(
        review.consume(candidate, proposed) == proposed,
        "unrelated dependency invalidated the IR review",
    )
    blocked("double consumption", review, candidate, proposed)

    tape = steane.candidate(registry["steane"], True)
    original = steane.candidate(registry["steane"], False)
    ideal = steane.gates(read)
    risk = steane.gates(read, F(999, 1000000))
    equality = steane.gates(read, F(1, 1000))
    ticket = check("steane.ideal", ideal, tape, "authorized")
    pointwise = steane.gates(read, preserve_faults=True)
    transfer(
        "Steane ideal to pointwise", "steane.ideal", "steane.pointwise", pointwise, ticket, tape
    )
    check("steane.pointwise", pointwise, tape, "denied")
    require(rows["steane.pointwise"]["rule"] == "fault paths", "pointwise rejection")
    pointwise.consume(
        check("steane.pointwise-reference", pointwise, original, "authorized"), original
    )
    edge = check("steane.boundary", equality, tape, "authorized")
    # A trusted consumer tightens its budget on the same issuer. The old handle
    # exists in that issuer's table; its binding must still fail.
    budget_ticket = check("steane.before-tightening", equality, tape, "authorized")
    original_budget = equality.contract
    equality.set_contract(risk.contract)
    transfer(
        "tighten Steane risk budget",
        "steane.before-tightening",
        "steane.biased",
        equality,
        budget_ticket,
        tape,
    )
    check("steane.biased", equality, tape, "denied")
    require(rows["steane.biased"]["observed"] == "1/1000", "biased risk difference")
    require(
        rows["steane.biased"]["edges"]["evidence"] == rows["steane.boundary"]["edges"]["evidence"],
        "changing only the budget changed the checked observations",
    )
    fallback = check("steane.fallback", equality, original, "authorized")
    depth9 = steane.execute(equality.consume(fallback, original), registry["steane"])["depth"]
    equality.set_contract(original_budget)
    blocked("budget restored after an intervening change", equality, edge, tape)
    edge = check("steane.revalidated-boundary", equality, tape, "authorized")
    registry["noise"] = steane.law(F(0))
    blocked("changed fault law", equality, edge, tape)
    depth8 = steane.execute(ideal.consume(ticket, tape), registry["steane"])["depth"]
    require((depth8, depth9) == (8, 9), "Steane schedule selection")
    uniform = steane.gates(read, F(0))
    uniform.consume(check("steane.uniform", uniform, tape, "authorized"), tape)
    check("steane.malformed", ideal, b"[]", "denied")

    exact, bounded = heat.gates(read)
    check("heat.unbound-binary", bounded, b"unrelated executable", "denied")
    check("heat.arithmetic", exact, binary, "denied")
    require(rows["heat.arithmetic"]["observed"] == str(F(1, 2) + F(1, 2**53)), "tie witness")
    token = check("heat.bounded", bounded, binary, "authorized")
    transfer(
        "heat grid error to exact arithmetic",
        "heat.bounded",
        "heat.arithmetic",
        exact,
        token,
        binary,
    )
    field = heat.solve(bounded.consume(token, binary))
    field_id = fingerprint(("\n".join(map(str, field)) + "\n").encode())
    require(
        field_id == "3f49a9c8ab388b260581e8a96ced4f8d726bc545bff3c590787e6ad74f29d1ff",
        "dispatched field differs from the pinned P5 field",
    )

    del registry["grid"]
    check("heat.missing-grid", bounded, binary, "inconclusive")
    need = Need("site", Kind.REFINEMENT, Level.REFINEMENT, Equal("validated"))
    policy = Contract(
        "transpose(x) times x", "select @gram comparison for lowering", Permission.LOWERING, (need,)
    )

    def timeout(*_):
        raise TimeoutError()

    for level, name in ((Level.ARTIFACT, "wrong-level"), (Level.REFINEMENT, "timeout")):
        gate = Gate(policy, {"site": Checker(need.kind, level, (), timeout)}, read)
        check(name, gate, proposed, "inconclusive")
    require(rows["wrong-level"]["reason"] == "required evidence level unavailable", "level control")
    require(rows["timeout"]["reason"] == "TimeoutError", "timeout control")
    require(
        rows["gram.wrong-operand"]["observed"] == "wrong operand binding", "Gram rejection point"
    )
    require(
        rows["heat.unbound-binary"]["rule"] == "artifact", "foreign binary reached native checker"
    )
    require(rows["heat.missing-grid"]["missing"] == ["grid"], "missing grid control")
    for key in (
        "permission alone changed",
        "tighten Steane risk budget",
        "substitute artifact with original receipt",
        "changed runtime",
    ):
        require(controls[key] == "artifact or authorization scope changed", key)
    for key, record in rows.items():

        def digest(value):
            return fingerprint(encode(value))

        require(record["scope_id"] == digest(record["scope"]), key)
        ids = []
        for node in record["evidence"]:
            ids.append(digest({k: v for k, v in node.items() if k != "id"}))
            require(node["id"] == ids[-1], key)
        require(record["edges"] == {"scope": record["scope_id"], "evidence": ids}, key)
        verdict = {
            k: v
            for k, v in record.items()
            if k not in ("scope", "scope_id", "evidence", "decision_id")
        }
        require(record["decision_id"] == digest(verdict), key)
    fallbacks = {
        decision: {
            "cause": rows[decision]["decision_id"],
            "selection": rows[selection]["decision_id"],
            "artifact": rows[selection]["scope"]["artifact"],
        }
        for decision, selection in (
            ("gram.strict", "gram.fallback"),
            ("steane.biased", "steane.fallback"),
        )
    }
    return {
        "schema": 2,
        "kernel_refinement": translation,
        "authority_transfers_refused": {
            key: {**row, "requested": rows[row["requested"]]["decision_id"]}
            for key, row in transfers.items()
        },
        "manifests": manifests,
        "fallbacks": fallbacks,
        "platform": platform.system() + " " + platform.machine(),
        "python": platform.python_version(),
        "lean": lean_version,
        "compiler": compiler_version,
        "axioms": proof.splitlines(),
        "proof_sha256": fingerprint((HERE / "Authority.lean").read_bytes()),
        "decisions": rows,
        "refused_tickets": controls,
        "heat_field_sha256": field_id,
        "dispatched_steane_depth": {"ideal": depth8, "biased_fallback": depth9},
        "adapter_nonblank_noncomment_lines": {
            p.stem: sum(
                bool(s.strip()) and not s.lstrip().startswith("#")
                for s in p.read_text().splitlines()
            )
            for p in (HERE / "gram.py", HERE / "steane.py", HERE / "heat.py")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", default=os.environ.get("LEAN", "lean"))
    parser.add_argument("--compiler", default=os.environ.get("ESHKOL_RUN", "eshkol-run"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lean, compiler = (shutil.which(x) for x in (args.lean, args.compiler))
    if not lean or not compiler:
        parser.error(
            "Lean 4.35.0 and eshkol-run v1.3.6-evolve must be on PATH or supplied explicitly"
        )
    result = json.dumps(replay(lean, compiler), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
