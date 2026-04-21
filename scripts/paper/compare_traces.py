#!/usr/bin/env python3
"""
compare_traces.py — fieldwise exact comparison of reference-VM and
compiled-transformer traces.

Inputs are JSONL files where each line is one VM step. Expected schema:

    {
      "program": "<name>",
      "step": <int>,
      "pc": <int>, "sp": <int>, "tos": <float>, "sos": <float>,
      "opcode": <int>, "is_native": <bool>,
      "registers": [<float>, ...],
      "memory":    [<float>, ...],
      "tape":      [<float>, ...],
      "flags":     {"zero": <bool>, ...}
    }

For each program, for each step, we compare every field bit-identically
for weight-implemented opcodes, and compare only the boundary-marker
fields (PC, is_native flag) for native-delegated opcodes.

Part of the artifact package for:
  "The Self-Differentiating Neural Computer: Computable Transformers
   via Analytical Weight Construction" (tsotchke, 2026)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


WEIGHT_IMPLEMENTED_FIELDS = [
    "pc", "sp", "tos", "sos", "registers", "memory", "tape", "flags",
]

NATIVE_BOUNDARY_FIELDS = ["pc", "is_native", "opcode"]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def group_by_program_step(records: list[dict]) -> dict[tuple[str, int], dict]:
    out = {}
    for rec in records:
        if "program" not in rec or "step" not in rec:
            continue
        out[(rec["program"], rec["step"])] = rec
    return out


def fieldwise_compare(vm: dict, tf: dict, fields: list[str]) -> list[str]:
    """Return list of field names that disagree between vm and tf."""
    diffs = []
    for field in fields:
        if vm.get(field) != tf.get(field):
            diffs.append(field)
    return diffs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vm", required=True, type=Path)
    p.add_argument("--transformer", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--coverage-out", required=True, type=Path)
    args = p.parse_args()

    if not args.vm.exists() or not args.transformer.exists():
        # TODO state: upstream runners not yet producing real traces.
        # Emit stub reports with status=todo so downstream table gen
        # doesn't crash.
        for out_path in (args.out, args.coverage_out):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "status": "todo",
                "message": "upstream trace dumps not wired; see scripts/paper/README",
            }, indent=2))
        print(f"  (todo) stub reports written to {args.out} and {args.coverage_out}")
        return 0

    vm_records = load_jsonl(args.vm)
    tf_records = load_jsonl(args.transformer)
    vm_by_key = group_by_program_step(vm_records)
    tf_by_key = group_by_program_step(tf_records)

    all_keys = sorted(set(vm_by_key.keys()) | set(tf_by_key.keys()))

    per_program = defaultdict(lambda: {
        "steps": 0,
        "agreeing_steps": 0,
        "disagreeing_steps": 0,
        "missing_vm_steps": 0,
        "missing_tf_steps": 0,
        "diff_fields": defaultdict(int),
    })
    opcode_coverage = {
        "weight_implemented": defaultdict(lambda: {"programs": set(), "steps": 0, "agree": 0}),
        "native_delegated": defaultdict(lambda: {"programs": set(), "steps": 0, "boundary_agree": 0}),
    }

    for (program, step) in all_keys:
        vm_rec = vm_by_key.get((program, step))
        tf_rec = tf_by_key.get((program, step))
        per = per_program[program]
        per["steps"] += 1

        if vm_rec is None:
            per["missing_vm_steps"] += 1
            continue
        if tf_rec is None:
            per["missing_tf_steps"] += 1
            continue

        is_native = bool(vm_rec.get("is_native", False))
        opcode = vm_rec.get("opcode")

        if is_native:
            diffs = fieldwise_compare(vm_rec, tf_rec, NATIVE_BOUNDARY_FIELDS)
            bucket = opcode_coverage["native_delegated"][opcode]
            bucket["programs"].add(program)
            bucket["steps"] += 1
            if not diffs:
                bucket["boundary_agree"] += 1
                per["agreeing_steps"] += 1
            else:
                per["disagreeing_steps"] += 1
                for f in diffs:
                    per["diff_fields"][f] += 1
        else:
            diffs = fieldwise_compare(vm_rec, tf_rec, WEIGHT_IMPLEMENTED_FIELDS)
            bucket = opcode_coverage["weight_implemented"][opcode]
            bucket["programs"].add(program)
            bucket["steps"] += 1
            if not diffs:
                bucket["agree"] += 1
                per["agreeing_steps"] += 1
            else:
                per["disagreeing_steps"] += 1
                for f in diffs:
                    per["diff_fields"][f] += 1

    # Assemble output
    per_program_report = {
        program: {
            "steps": data["steps"],
            "agreeing_steps": data["agreeing_steps"],
            "disagreeing_steps": data["disagreeing_steps"],
            "missing_vm_steps": data["missing_vm_steps"],
            "missing_tf_steps": data["missing_tf_steps"],
            "diff_fields": dict(data["diff_fields"]),
        }
        for program, data in per_program.items()
    }

    total_programs = len(per_program)
    fully_agreeing_programs = sum(
        1 for p in per_program.values()
        if p["disagreeing_steps"] == 0
        and p["missing_vm_steps"] == 0
        and p["missing_tf_steps"] == 0
    )

    report = {
        "status": "ok",
        "total_programs": total_programs,
        "fully_agreeing_programs": fully_agreeing_programs,
        "per_program": per_program_report,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"  comparison report: {fully_agreeing_programs}/{total_programs} programs fully agree")

    coverage_report = {
        "weight_implemented": {
            str(opcode): {
                "programs": sorted(data["programs"]),
                "steps": data["steps"],
                "agree": data["agree"],
            }
            for opcode, data in opcode_coverage["weight_implemented"].items()
        },
        "native_delegated": {
            str(opcode): {
                "programs": sorted(data["programs"]),
                "steps": data["steps"],
                "boundary_agree": data["boundary_agree"],
            }
            for opcode, data in opcode_coverage["native_delegated"].items()
        },
    }
    args.coverage_out.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_out.write_text(json.dumps(coverage_report, indent=2))
    print(f"  opcode coverage: {len(coverage_report['weight_implemented'])} weight-impl, "
          f"{len(coverage_report['native_delegated'])} native-delegated")

    return 0 if fully_agreeing_programs == total_programs else 1


if __name__ == "__main__":
    sys.exit(main())
