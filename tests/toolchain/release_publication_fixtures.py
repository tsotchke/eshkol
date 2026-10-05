"""Small source-bound raw producer fixtures, never a qualification run."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import release_publication_contract as contract

SHA = "a" * 40
TARGET = "v1.3.6-evolve"


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")


def refresh_manifest(bundle):
    receipts = ["ctest/measurement.json", "vm/measurement.json"]
    manifest = {"schema": "eshkol.release-measurements.v1", "receipts": receipts,
                "receipt_sha256": {p: contract.sha256(bundle / p) for p in receipts}}
    if (bundle / "phase-state.json").exists():
        manifest["phase_state_sha256"] = contract.sha256(bundle / "phase-state.json")
    dump(bundle / "manifest.json", manifest)


def bundle_fixture(bundle, sha=SHA, target=TARGET, skipped=False, declared=True):
    bundle = Path(bundle)
    bundle.mkdir(parents=True)
    cohort = {"__source__": {"git_head": sha, "tracked_state_sha256": hashlib.sha256(b"").hexdigest()},
              "eshkol-run": {"sha256": "b" * 64, "size": 1}}
    dump(bundle / "build-cohort.json", cohort)
    phase = "fixture-1"
    dump(bundle / "phase-state.json", {"schema": "eshkol.release-evidence-phases.v1", "head": sha,
         "phase_id": phase, "completed": ["baseline", "smoke", "final-evidence"]})
    for child, producer in (("ctest", "run_ctest_gate"), ("vm", "run_vm_parity")):
        root = bundle / child
        root.mkdir()
        snapshot = {"sha": sha, "clean": True}
        dump(root / "source-start.json", snapshot)
        dump(root / "source-end.json", snapshot)
        if child == "ctest":
            dump(root / "inventory.json", {"tests": [{"name": "required"}, {"name": "optional"}]})
            dump(root / "optional-policy.json", {"schema": "eshkol.release-optional-ctest.v1", "exclusions": {"optional": "existing optional capability"} if skipped and declared else {}})
            (root / "junit.xml").write_text('<testsuite><testcase name="required" status="run"><system-out>SKIP optional internal subprobe</system-out></testcase><testcase name="optional" status="' + ('skipped"><skipped/>' if skipped else 'run">') + '</testcase></testsuite>')
            (root / "raw.log").write_text("1/2 Test #1: required ........ Passed 0.01 sec\n2/2 Test #2: optional ........ " + ("***Skipped" if skipped else "Passed") + " 0.02 sec\n100% tests passed, 0 tests failed out of 2\n")
            events = [{"kind": "ctest", "name": name, "value": "PASS"} for name in ("ctest_suite_green", "ctest_self_verdict_scan", "ctest_required")]
            events.append({"kind": "ctest", "name": "ctest_optional", "value": "SKIP" if skipped else "PASS"})
            dump(root / "exit.json", {"gate_exit_code": 0, "ctest_exit_code": 0, "argv": []})
        else:
            dump(root / "inventory.json", {"corpus": ["one.esk"], "found": [], "oos": ["two.esk"], "fatal": ["three.esk"]})
            names = ["vm_gap_canonicalization", "vm_parity_audit", "vm_parity_self_verdict_scan", "corpus_one_vmsrc", "corpus_one_vmeskb", "oos_two", "fatal_three_native", "fatal_three_vm"]
            events = [{"kind": "release_measurement_scope", "value": {"do_eskb": 1, "audit_only": 0}}]
            events += [{"kind": "vm_parity", "name": n, "value": "PASS"} for n in names + ["vm_parity_gate"]]
            (root / "raw.log").write_text("".join(f"== stage {n}: fixture ==\n" for n in range(1, 5)) + "".join(f"PASSED tests/vm_parity::{name}\n" for name in names) + "vm-parity: 8 passed, 0 failed, 0 infra (no verdict)\n")
            dump(root / "exit.json", {"exit_code": 0})
        (root / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        facts = contract.measurement_facts(root, producer) if declared else {"total": 2, "passed": 1, "failed": 0, "infra": 0, "skipped": 1, "required_total": 1, "required_passed": 1, "optional_skipped": 1, "optional_skipped_inventory": ["optional"]}
        receipt = {"schema": "eshkol.release-measurement.v1", "producer": producer, "source_sha": sha, "source_clean": True,
                   "phase_id": phase, "target": target, "build_cohort_sha256": contract.sha256(bundle / "build-cohort.json"),
                   "mode": "full", "argv": [], "exit_code": 0, "ctest_exit_code": 0 if child == "ctest" else None, **facts}
        fields = {"raw_log": "raw.log", "trace": "trace.jsonl", "configured_inventory": "inventory.json", "source_start": "source-start.json", "source_end": "source-end.json", "exit_receipt": "exit.json"}
        if child == "ctest": fields.update(junit="junit.xml", optional_policy="optional-policy.json")
        for field, name in fields.items():
            receipt[field + "_relative_path"] = name
            receipt[field + "_sha256"] = contract.sha256(root / name)
        dump(root / "measurement.json", receipt)
    refresh_manifest(bundle)
    return bundle


def verifier_fixture(repo, trace, target):
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "core.hooksPath=/dev/null", "commit", "-q", "--allow-empty", "-m", "fixture"], check=True)
    sha = contract.source_snapshot(repo)["sha"]
    bundle = bundle_fixture(trace / "publication", sha, target)
    (trace / "release-build-cohort.json").write_bytes((bundle / "build-cohort.json").read_bytes())
    (trace / "release-phase-state.json").write_bytes((bundle / "phase-state.json").read_bytes())
    return bundle
