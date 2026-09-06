#!/usr/bin/env python3
"""rosette_receipt.py — turn an Eshkol ICC trace into a Rosette Wire receipt.

Reads a `kind:"rosette"` runtime_event trace (produced by
scripts/run_rosette_oracle.sh through scripts/lib/harness_outcome.sh, one
event per backend comparison) and writes a receipt.json that conforms to
Rosette Wire's own protocol schema `urn:rosette-wire:schema:receipt:1`
(rosette-wire/src/compiler/wire/wire-graph/src/graph.lisp: wire-receipt->value;
decoded and strictly validated by wire-receipt-from-value in decode.lisp in
the same tree). Rose Studio (or any consumer of `bin/rosette describe
<receipt.json>`) can render this exactly like a receipt Rosette Wire produced
of one of its own graph runs.

Required top-level fields, matching the Lisp decoder's `%decode-fields`
allow/require list exactly (extra or missing fields are both refused):
    schema     "urn:rosette-wire:schema:receipt:1"           (constant)
    kind       one of the wire-receipt kinds this project's own runtime uses:
               "validation" | "execution" | "verification" | "certification"
    verdict    "pass" | "fail"
    graph      {"digest": "sha256:<64 lowercase hex>"}         (%require-digest)
    violations [{"stage","path","code","detail"}, ...]         (each a string)
    stepOrder  [string, ...]
    outputs    arbitrary JSON object (frozen, not otherwise validated)
    evidence   arbitrary JSON object (frozen, not otherwise validated)

Validate a receipt against the real Rosette Wire decoder with:
    <rosette-wire-clone>/bin/rosette describe receipt.json
which loads wire-graph and round-trips the receipt through the same strict
decoder gate/certify/validate/run all share, exiting 0 on success and
nonzero (via the wire-error path) the moment any field is wrong.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "urn:rosette-wire:schema:receipt:1"
STEP_ORDER = ["rosette_front_door_eshkol", "rosette_external_backends_eshkol"]


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_trace(trace_path: Path):
    events = {}
    if not trace_path.exists():
        return events
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") != "rosette":
                continue
            name = record.get("name")
            if name in STEP_ORDER:
                # Rosette Wire's canonical-json rejects a bare JSON float
                # anywhere in the document (an "untyped-float" wire-error —
                # see canonical-json.lisp): drop harness_outcome.sh's numeric
                # `confidence` field rather than embed a float this schema
                # refuses. Trace is append-only; a later line for the same
                # name is the more recent run, so let it win.
                events[name] = {k: v for k, v in record.items() if k != "confidence"}
    return events


def build_receipt(trace_path: Path, eshkol_commit: str) -> dict:
    events = _read_trace(trace_path)

    violations = []
    outputs = {}
    for name in STEP_ORDER:
        record = events.get(name)
        if record is None:
            violations.append(
                {
                    "stage": "collection",
                    "path": f"$.{name}",
                    "code": "missing_event",
                    "detail": f"no {name} event found in {trace_path}",
                }
            )
            outputs[name] = {"value": "MISSING", "snippet": ""}
            continue
        value = record.get("value")
        snippet = record.get("snippet", "")
        outputs[name] = {"value": value, "snippet": snippet}
        if value != "PASS":
            violations.append(
                {
                    "stage": "execution",
                    "path": f"$.{name}",
                    "code": "backend_disagreement" if value == "FAIL" else "no_verdict",
                    "detail": snippet or f"{name} reported {value}",
                }
            )

    verdict = "pass" if not violations else "fail"
    graph_digest = _digest(
        json.dumps(
            {"eshkol_commit": eshkol_commit, "step_order": STEP_ORDER},
            sort_keys=True,
        )
    )

    return {
        "schema": SCHEMA,
        "kind": "verification",
        "verdict": verdict,
        "graph": {"digest": graph_digest},
        "violations": violations,
        "stepOrder": STEP_ORDER,
        "outputs": outputs,
        "evidence": {
            "eshkolCommit": eshkol_commit,
            "traceFile": str(trace_path),
            "events": events,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        default="scripts/icc_traces/rosette.jsonl",
        help="path to the rosette.jsonl ICC trace",
    )
    parser.add_argument(
        "--eshkol-commit",
        default="unknown",
        help="`git describe` (or similar) identity of the Eshkol build under test",
    )
    parser.add_argument(
        "--out",
        default="receipt.json",
        help="path to write the receipt JSON to",
    )
    args = parser.parse_args()

    receipt = build_receipt(Path(args.trace), args.eshkol_commit)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path} (verdict={receipt['verdict']})")
    return 0 if receipt["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
