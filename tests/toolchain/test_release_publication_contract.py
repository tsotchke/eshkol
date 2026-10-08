"""Focused controls for the shared release record and notes contract."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
import os
import subprocess
import shutil
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_publication_fixtures import bundle_fixture, refresh_manifest, dump, SHA, TARGET

ROOT = Path(__file__).resolve().parents[2]
(ROOT / ".scratch").mkdir(exist_ok=True)
SPEC = importlib.util.spec_from_file_location(
    "release_publication_contract", ROOT / "scripts/release_publication_contract.py"
)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def record(**changes):
    value = {
        "schema": "eshkol.release-record.v1",
        "tag": "v1.3.6-evolve",
        "previous_tag": "v1.3.4-evolve",
        "release_date": "2026-10-05",
        "status": "PREPARED FOR PUBLICATION",
        "ctest_total": 240,
        "vm_parity_total": 390,
    }
    value.update(changes)
    return value


def notes(status="PREPARED FOR PUBLICATION", extra=""):
    return (
        "# Eshkol v1.3.6-evolve — Release Notes\n\n"
        f"**Status:** {status}.\n"
        "**Planned release date:** Monday, October 5, 2026.\n\n"
        "CTest 240/240 and VM parity 390/390.\n"
        "Exact-source validation is required before publication.\n"
        f"{extra}\n---\n# Eshkol v1.3.5-evolve — Release Notes\n"
        "**Status:** RELEASE CANDIDATE.\n"
    )


class PublicationContractTests(unittest.TestCase):
    def load(self, value, strict=False):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            path = Path(directory) / "release_record.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return contract.load_record(path, strict=strict)

    def test_preparation_accepts_explicit_null_totals_and_older_previous_tag(self):
        value = self.load(record(ctest_total=None, vm_parity_total=None,
                                 previous_tag="v1.2.9-evolve"))
        self.assertIsNone(value["ctest_total"])
        self.assertEqual(value["previous_tag"], "v1.2.9-evolve")

    def test_strict_record_requires_prepared_complete_facts_and_no_placeholders(self):
        self.assertEqual(self.load(record(), strict=True)["status"], "PREPARED FOR PUBLICATION")
        for changes in (
            {"status": "RELEASE CANDIDATE"},
            {"ctest_total": None},
            {"vm_parity_total": True},
            {"ctest_total": 0},
            {"ctest_total_placeholder": "pending"},
        ):
            with self.subTest(changes=changes), self.assertRaises(contract.ContractError):
                self.load(record(**changes), strict=True)

    def test_record_rejects_missing_keys_duplicate_json_keys_and_bad_types(self):
        value = record()
        del value["previous_tag"]
        with self.assertRaisesRegex(contract.ContractError, "missing"):
            self.load(value)
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema":"eshkol.release-record.v1","schema":"other"}')
            with self.assertRaisesRegex(contract.ContractError, "duplicate"):
                contract.load_record(path)
        for changes in ({"tag": "v1.3.4-evolve"}, {"release_date": "2026-02-30"},
                        {"release_date": 20261005}, {"status": "almost shipped"},
                        {"ctest_total": False}):
            with self.subTest(changes=changes), self.assertRaises(contract.ContractError):
                self.load(record(**changes))

    def test_preparation_notes_allow_pending_state_but_keep_identity_and_date_typed(self):
        value = record(status="RELEASE CANDIDATE", ctest_total=None, vm_parity_total=None)
        section = contract.validate_notes(
            notes(status="RELEASE CANDIDATE", extra=contract._PENDING_MARKER),
            value, value["tag"], "preparation")
        self.assertTrue(section.startswith("# Eshkol v1.3.6-evolve"))
        for bad_text, bad_tag in ((notes().replace("v1.3.6-evolve", "v1.3.5-evolve", 1), "v1.3.6-evolve"),
                                  (notes().replace("October 5, 2026", "October 6, 2026"), "v1.3.6-evolve"),
                                  (notes().replace("Monday, October", "Tuesday, October"), "v1.3.6-evolve")):
            with self.subTest(bad_tag=bad_tag), self.assertRaises(contract.ContractError):
                contract.validate_notes(bad_text, record(), bad_tag)

    def test_strict_notes_reject_candidate_pending_duplicates_and_count_drift(self):
        value = record()
        good = notes()
        self.assertIn("Exact-source validation is required", contract.validate_notes(
            good, value, value["tag"], "candidate-proof"))
        poisoned = (
            notes(extra=contract._PENDING_MARKER),
            notes(extra="Evidence remains unrecorded."),
            notes(status="RELEASE CANDIDATE"),
            notes().replace("CTest 240/240", "CTest 239/240"),
            notes().replace("\n---\n", "\n**Status:** PREPARED FOR PUBLICATION.\n---\n"),
        )
        for text in poisoned:
            with self.subTest(text=text[:120]), self.assertRaises(contract.ContractError):
                contract.validate_notes(text, value, value["tag"], "tag-publication")

    def test_notes_reject_every_marker_heading_date_and_count_poison(self):
        for extra in ("RELEASE_EVIDENCE_PENDING", "<!--RELEASE_EVIDENCE_PENDING-->",
                      "# Eshkol v1.3.7-evolve — Release Notes", "**Planned release date:** bananas.",
                      "CTest 1/1", "VM parity 399/399", "**399/399** VM parity differential checks"):
            with self.subTest(extra=extra), self.assertRaises(contract.ContractError):
                contract.validate_notes(notes(extra=extra), record(), TARGET, "candidate-proof")
        for changes in ({"schema": "other"}, {"release_date": True}, {"ctest_total": 240.0}, {"ctest_total": "240"}):
            with self.subTest(changes=changes), self.assertRaises(contract.ContractError):
                contract.validate_notes(notes(), record(**changes), TARGET, "candidate-proof")

    def test_full_bound_measurements_and_internal_optional_subprobe(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            bundle = bundle_fixture(Path(directory) / "bundle")
            receipts = contract.load_measurements(bundle, SHA, TARGET)
            self.assertEqual(receipts["run_ctest_gate"]["passed"], 2)
            self.assertEqual(receipts["run_vm_parity"]["passed"], 8)

    def test_measurement_identity_types_scope_hashes_and_missing_raw_fail_closed(self):
        mutations = (("source_sha", "b" * 40), ("source_clean", False), ("target", "v1.3.5-evolve"),
                     ("phase_id", "other"), ("build_cohort_sha256", "c" * 64), ("mode", "partial"),
                     ("argv", ["-R", "five"]), ("exit_code", 1), ("ctest_exit_code", 8),
                     ("passed", True), ("failed", 1), ("infra", 1), ("total", 5),
                     ("raw_log_relative_path", "../raw.log"), ("skipped", True))
        for field, value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
                bundle = bundle_fixture(Path(directory) / "bundle")
                path = bundle / "ctest/measurement.json"
                receipt = contract.read_json(path)
                receipt[field] = value
                dump(path, receipt)
                refresh_manifest(bundle)
                with self.assertRaises(contract.ContractError):
                    contract.load_measurements(bundle, SHA, TARGET)
        for artifact in ("raw.log", "junit.xml", "inventory.json", "trace.jsonl"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
                bundle = bundle_fixture(Path(directory) / "bundle")
                (bundle / "ctest" / artifact).write_text("tampered")
                with self.assertRaises(contract.ContractError):
                    contract.load_measurements(bundle, SHA, TARGET)

    def test_raw_failures_skips_truncation_conflicts_and_scope_contradictions(self):
        for poison in ("", "100% tests passed, 0 tests failed out of 2\n" * 2,
                       "50% tests passed, 1 tests failed out of 2\n"):
            with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
                bundle = bundle_fixture(Path(directory) / "bundle")
                (bundle / "ctest/raw.log").write_text(poison)
                with self.assertRaises(contract.ContractError):
                    contract.measurement_facts(bundle / "ctest", "run_ctest_gate")
        for value in ({"do_eskb": 0, "audit_only": 0}, {"do_eskb": 1, "audit_only": 1}):
            with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
                bundle = bundle_fixture(Path(directory) / "bundle")
                path = bundle / "vm/trace.jsonl"
                path.write_text(path.read_text().replace('"do_eskb": 1, "audit_only": 0', json.dumps(value)[1:-1]))
                with self.assertRaises(contract.ContractError):
                    contract.measurement_facts(bundle / "vm", "run_vm_parity")


    def test_summary_only_duplicate_truncated_and_wrong_raw_ctest_identities_reject(self):
        for mutate in (lambda text: text.splitlines()[-1] + "\n",
                       lambda text: text.replace('1/2 Test #1: required ........ Passed 0.01 sec\n', ''),
                       lambda text: text.replace('required ........', 'invented ........'),
                       lambda text: text.replace('optional ........ Passed', 'optional ........ ***Failed'),
                       lambda text: text.replace('2/2 Test #2:', '2/2 Test #1:'),
                       lambda text: text.splitlines()[0] + "\n" + text):
            with tempfile.TemporaryDirectory(dir=ROOT / '.scratch') as directory:
                bundle = bundle_fixture(Path(directory) / 'bundle')
                path = bundle / 'ctest/raw.log'
                path.write_text(mutate(path.read_text()))
                with self.assertRaises(contract.ContractError):
                    contract.measurement_facts(bundle / 'ctest', 'run_ctest_gate')

    def test_real_ctest_failure_and_timeout_console_shapes(self):
        text = (
            "1/5 Test #1: good ................ Passed 0.01 sec\n"
            "2/5 Test #2: wrong ...............***Failed    0.02 sec\n"
            "3/5 Test #3: missing .............***Failed  Required regular expression not found. Regex=[PASS: proof\n"
            "]  0.03 sec\n"
            "4/5 Test #4: slow ................***Timeout 1800.04 sec\n"
            "5/5 Test #5: regex ...............***Failed  Required regular expression not found. Regex=[PASS] 0.01 sec\n"
        )
        self.assertEqual(contract.ctest_raw_outcomes(text, 5),
                         {"good": "passed", "wrong": "failed", "missing": "failed", "slow": "infra", "regex": "failed"})
        for invalid in (text.replace("]  0.03 sec", ""),
                        text.replace("]  0.03 sec", "invented ]  0.03 sec"),
                        text.replace("***Timeout", "***Unknown")):
            with self.subTest(invalid=invalid), self.assertRaises(contract.ContractError):
                contract.ctest_raw_outcomes(invalid, 5)

    def test_alternate_hashed_filename_cannot_leave_consumed_raw_unbound(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.scratch') as directory:
            bundle = bundle_fixture(Path(directory) / 'bundle')
            root = bundle / 'ctest'
            (root / 'other.log').write_text('arbitrary hashed content')
            receipt = contract.read_json(root / 'measurement.json')
            receipt['raw_log_relative_path'] = 'other.log'
            receipt['raw_log_sha256'] = contract.sha256(root / 'other.log')
            dump(root / 'measurement.json', receipt)
            refresh_manifest(bundle)
            with self.assertRaises(contract.ContractError):
                contract.load_measurements(bundle, SHA, TARGET)

    def test_boolean_vm_scope_flags_cannot_impersonate_full_integer_scope(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.scratch') as directory:
            bundle = bundle_fixture(Path(directory) / 'bundle')
            path = bundle / 'vm/trace.jsonl'
            events = contract.trace_events(path)
            events[0]['value'] = {'do_eskb': True, 'audit_only': False}
            path.write_text(''.join(json.dumps(event) + '\n' for event in events))
            with self.assertRaises(contract.ContractError):
                contract.measurement_facts(bundle / 'vm', 'run_vm_parity')

    def test_registered_optional_skip_is_explicit_and_never_measured_pass(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            bundle = bundle_fixture(Path(directory) / "bundle", skipped=True)
            receipts = contract.load_measurements(bundle, SHA, TARGET)
            receipt = receipts["run_ctest_gate"]
            self.assertEqual((receipt["total"], receipt["passed"], receipt["optional_skipped"]), (2, 1, 1))
            value = record(ctest_total=2, vm_parity_total=8, ctest_required_total=1, ctest_optional_skipped=1)
            text = notes().replace("CTest 240/240 and VM parity 390/390", "CTest configured 2, required 1/1, optional skipped 1; VM parity 8/8")
            contract.validate_notes(text, value, TARGET, "candidate-proof")
            with self.assertRaises(contract.ContractError):
                contract.validate_notes(text.replace("CTest configured 2, required 1/1, optional skipped 1", "CTest 2/2"), value, TARGET, "candidate-proof")
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            bundle = bundle_fixture(Path(directory) / "bundle", skipped=True, declared=False)
            with self.assertRaises(contract.ContractError):
                contract.load_measurements(bundle, SHA, TARGET)

    def test_ctest_producer_retains_actual_raw_junit_inventory_and_exit_on_infra(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            repo = Path(directory) / "repo"
            (repo / "scripts/lib").mkdir(parents=True)
            (repo / "tests/coverage").mkdir(parents=True)
            (repo / "build").mkdir()
            (repo / "build/CTestTestfile.cmake").write_text("")
            for name in ('run_ctest_gate.sh', 'release_publication_contract.py', 'check_self_verdicts.py'):
                shutil.copy(ROOT / 'scripts' / name, repo / 'scripts' / name)
            for name in ('harness_outcome.sh', 'checked_write.sh', 'evidence_paths.sh', 'durable_work_root.sh', 'build_fingerprint.sh'):
                shutil.copy(ROOT / 'scripts/lib' / name, repo / 'scripts/lib' / name)
            shutil.copy(ROOT / 'tests/coverage/release_optional_ctest.json', repo / 'tests/coverage/release_optional_ctest.json')
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'core.hooksPath=/dev/null', 'commit', '-q', '--allow-empty', '-m', 'fixture'], check=True)
            binary = repo / 'bin'
            binary.mkdir()
            ctest = binary / 'ctest'
            ctest.write_text("#!" + sys.executable + "\n" + r'''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
names = ['fixedpoint_one', 'runtime_closure_arity_spread_one', 'define_library_same_unit_one', 'load_path_engine_parity_test', 'squared_distance_gradcheck']
names += [base + '_' + mode + '_smoke' for base in ('taylor_tower', 'taylor_tower_mono', 'exact_taylor', 'reverse_over_taylor', 'taylor_numerics', 'region_evac_taylor_exact') for mode in ('runtime', 'aot')]
if '--show-only=json-v1' in args:
    print(json.dumps({'kind': 'ctestInfo', 'version': {'major': 1, 'minor': 0}, 'tests': [{'name': name} for name in names]})); sys.exit(0)
if '-N' in args: sys.exit(0)
timeout = os.environ.get('FAKE_TIMEOUT') == '1'
path = Path(args[args.index('--output-junit') + 1])
path.write_text('<testsuite>' + ''.join('<testcase name="' + name + '" status="run">' + ('<failure message="Timeout"/>' if timeout and name == 'fixedpoint_one' else '') + '</testcase>' for name in names) + '</testsuite>')
for index, name in enumerate(names, 1):
    print(str(index) + '/' + str(len(names)) + ' Test #' + str(index) + ': ' + name + ' ........ ' + ('***Timeout' if timeout and name == 'fixedpoint_one' else 'Passed') + ' 0.01 sec')
print(('94% tests passed, 1 tests failed' if timeout else '100% tests passed, 0 tests failed') + ' out of ' + str(len(names)))
sys.exit(8 if timeout else 0)
''')
            ctest.chmod(0o755)
            for timeout in (False, True):
                trace = repo / ('trace-timeout' if timeout else 'trace-green')
                trace.mkdir()
                dump(trace / 'release-build-cohort.json', {'fixture': 'cohort'})
                destination = repo / ('measurement-timeout' if timeout else 'measurement-green')
                result = subprocess.run(['bash', str(repo / 'scripts/run_ctest_gate.sh')], cwd=repo,
                    env={**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'], 'TRACE_DIR': str(trace),
                         'RELEASE_MEASUREMENT_DIR': str(destination), 'ESHKOL_RELEASE_PHASE_ID': 'fixture',
                         'RELEASE_TARGET': TARGET, 'FAKE_TIMEOUT': '1' if timeout else '0'}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1 if timeout else 0, result.stdout + result.stderr + ((destination / 'receipt-validation.log').read_text() if (destination / 'receipt-validation.log').exists() else ''))
                for artifact in ('raw.log', 'junit.xml', 'inventory.json', 'ctest-results.tsv', 'exit.json', 'measurement.json'):
                    self.assertTrue((destination / artifact).is_file(), artifact)
                receipt = contract.read_json(destination / 'measurement.json')
                self.assertEqual(receipt['source_sha'], contract.source_snapshot(repo)['sha'])
                if timeout:
                    self.assertEqual(contract.read_json(destination / 'exit.json')['ctest_exit_code'], 8)
                    self.assertIsNone(receipt['passed'])
                    self.assertIn('validation_errors', receipt)
                else:
                    self.assertEqual(receipt['passed'], 17)

    def test_metadata_proof_requires_bound_complete_phases_and_runtime_counts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            root = Path(directory)
            bundle = bundle_fixture(root / "bundle")
            dump(root / "record.json", record(ctest_total=2, vm_parity_total=8))
            (root / "notes.md").write_text(notes().replace("240/240", "2/2").replace("390/390", "8/8"))
            (root / "tests/coverage").mkdir(parents=True)
            (root / "tests/coverage/release_optional_ctest.json").write_bytes((bundle / "ctest/optional-policy.json").read_bytes())
            facts = contract.metadata_facts(root / "record.json", root / "notes.md", bundle, SHA, TARGET, "candidate-proof")
            identity = {"sha": SHA, "target": TARGET, "role": "candidate-proof", "run_id": 10, "run_attempt": 2}
            proof = {**facts, **identity, "schema": "eshkol.release-readiness.v2", "status": "ready", "score": 100, "qualification_mode": "normal", "waiver_count": 0}
            contract.validate_publication(root / "record.json", root / "notes.md", bundle, proof, identity)
            for key, value in (("schema", "eshkol.release-readiness.v1"), ("score", True), ("score", "100"),
                               ("qualification_mode", "diagnostic"), ("waiver_count", 1), ("role", "tag-publication"),
                               ("run_id", 9), ("run_attempt", 1), ("notes_sha256", "0" * 64), ("metadata_validated", False)):
                with self.subTest(key=key), self.assertRaises(contract.ContractError):
                    contract.validate_publication(root / "record.json", root / "notes.md", bundle, {**proof, key: value}, identity)
            original_state = contract.read_json(bundle / "phase-state.json")
            for coverage in (False, "true", None):
                with self.subTest(coverage=coverage):
                    state = {**original_state, "coverage_completed": coverage}
                    dump(bundle / "phase-state.json", state)
                    refresh_manifest(bundle)
                    with self.assertRaisesRegex(contract.ContractError, "complete baseline"):
                        contract.metadata_facts(root / "record.json", root / "notes.md", bundle, SHA, TARGET, "candidate-proof")
            state = {**original_state, "completed": []}
            dump(bundle / "phase-state.json", state)
            refresh_manifest(bundle)
            with self.assertRaisesRegex(contract.ContractError, "complete baseline"):
                contract.metadata_facts(root / "record.json", root / "notes.md", bundle, SHA, TARGET, "candidate-proof")
            state = original_state
            state["completed"] = ["baseline"]
            dump(bundle / "phase-state.json", state)
            refresh_manifest(bundle)
            with self.assertRaises(contract.ContractError):
                contract.metadata_facts(root / "record.json", root / "notes.md", bundle, SHA, TARGET, "candidate-proof")


if __name__ == "__main__":
    unittest.main()
