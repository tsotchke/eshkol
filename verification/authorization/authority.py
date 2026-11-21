"""Consumer-owned policy and single-use authority for immutable artifact bytes."""

import kernel
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from subprocess import SubprocessError
from types import MappingProxyType
from typing import Callable, Mapping


def fingerprint(data):
    return sha256(data).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


GATE_DIGEST = fingerprint(
    encode(
        {
            name: fingerprint(Path(__file__).with_name(name).read_bytes())
            for name in ("authority.py", "kernel.py")
        }
    )
)


class Permission(str, Enum):
    IR_REVIEW = "ir-review"
    LOWERING = "lowering"
    NATIVE_RELEASE = "native-release"
    EXECUTION = "execution"
    PUBLICATION = "publication"


class Level(str, Enum):
    MODEL = "model-proved"
    OBSERVED = "executable-observed"
    ARTIFACT = "artifact-validated"
    REFINEMENT = "refinement-established"


class Kind(str, Enum):
    EXACT = "exact"
    REFINEMENT = "refinement"
    BOUNDED_ERROR = "bounded-error"
    TRACE = "trace"
    RISK = "risk"
    RESOURCE = "resource"


@dataclass(frozen=True)
class Equal:
    target: str


@dataclass(frozen=True)
class AtMost:
    target: Fraction


@dataclass(frozen=True)
class Need:
    claim: str
    kind: Kind
    level: Level
    condition: Equal | AtMost


@dataclass(frozen=True)
class Contract:
    subject: str
    action: str
    permission: Permission
    needs: tuple[Need, ...]

    def __post_init__(self):
        if (
            type(self.permission) is not Permission
            or type(self.needs) is not tuple
            or not self.needs
            or any(type(v) is not str or not v for v in (self.subject, self.action))
            or any(type(n) is not Need for n in self.needs)
        ):
            raise ValueError("empty policy")
        if len({n.claim for n in self.needs}) != len(self.needs):
            raise ValueError("duplicate claim")
        for n in self.needs:
            valid = (type(n.condition) is Equal and type(n.condition.target) is str) or (
                type(n.condition) is AtMost and type(n.condition.target) is Fraction
            )
            if type(n.kind) is not Kind or (
                n.kind in (Kind.BOUNDED_ERROR, Kind.RISK, Kind.RESOURCE)
            ) != (type(n.condition) is AtMost):
                raise ValueError("obligation and comparison disagree")
            if type(n.claim) is not str or not n.claim or type(n.level) is not Level or not valid:
                raise ValueError("invalid typed obligation")

    def record(self):
        return [
            self.subject,
            self.action,
            [
                [
                    n.claim,
                    n.kind.value,
                    n.level.value,
                    type(n.condition).__name__,
                    str(n.condition.target),
                ]
                for n in self.needs
            ],
        ]


@dataclass(frozen=True)
class Observation:
    value: str | Fraction
    location: str = ""


@dataclass(frozen=True)
class Checker:
    kind: Kind
    level: Level
    inputs: tuple[str, ...]
    measure: Callable[[bytes, Mapping[str, bytes]], Observation]


class Unsupported(Exception):
    """The checker cannot establish the requested observation."""


class Gate:
    """Trusted, sequential consumer. The proposal entry point accepts bytes only."""

    def __init__(self, contract, checkers, read_context):
        self._contract = contract
        self._policy_generation = 0
        self._checkers = MappingProxyType(dict(checkers))
        self._read = read_context
        self._issued = {}

    @property
    def contract(self):
        return self._contract

    def set_contract(self, contract):
        """Consumer-owned policy update. Restoring a prior budget still needs a fresh ticket."""
        if type(contract) is not Contract:
            raise TypeError("expected a consumer contract")
        if contract != self._contract:
            self._policy_generation += 1
            self._contract = contract

    def _snapshot(self):
        # One coherent snapshot, supplied by the consumer. Read only dependencies
        # declared by this policy, including revocation epochs for mutable state.
        names = sorted(
            {
                key
                for n in self.contract.needs
                if n.claim in self._checkers
                for key in self._checkers[n.claim].inputs
            }
        )
        state = self._read()
        values = {k: state[k] for k in names if k in state}
        if any(type(v) is not bytes for v in values.values()):
            raise TypeError("dependency values must be immutable bytes")
        return values

    def _scope(self, payload, values):
        deps = {k: fingerprint(v) for k, v in sorted(values.items())}
        return {
            "gate": GATE_DIGEST,
            "contract": self.contract.record(),
            "permission": self.contract.permission.value,
            "policy_generation": self._policy_generation,
            "artifact": fingerprint(payload),
            "dependencies": deps,
        }

    def assess(self, payload):
        if type(payload) is not bytes:
            raise TypeError("candidate must be immutable bytes")
        values = self._snapshot()
        scope = self._scope(payload, values)
        nodes = []
        report = {"scope": scope, "scope_id": fingerprint(encode(scope)), "evidence": nodes}

        def decision(status, **details):
            edges = {"scope": report["scope_id"], "evidence": [n["id"] for n in nodes]}
            verdict = {"status": status, "edges": edges, **details}
            return {**report, **verdict, "decision_id": fingerprint(encode(verdict))}

        failure = {}

        def refuse(status, rule, reason, **witness):
            failure.update(rule=rule, reason=reason, **witness)
            return status

        def observe(need):
            checker = self._checkers.get(need.claim)
            if checker is None or (checker.kind, checker.level) != (need.kind, need.level):
                return refuse("inconclusive", need.claim, "required evidence level unavailable")
            missing = sorted(set(checker.inputs) - values.keys())
            if missing:
                return refuse("inconclusive", need.claim, "missing dependencies", missing=missing)
            inputs = {k: values[k] for k in checker.inputs}
            try:
                obs = checker.measure(payload, MappingProxyType(inputs))
            except Unsupported as exc:
                return refuse("inconclusive", need.claim, str(exc))
            except (SubprocessError, TimeoutError, OSError) as exc:
                return refuse("inconclusive", need.claim, type(exc).__name__)
            exact = type(need.condition) is Equal
            if type(obs) is not Observation or type(obs.value) is not (str if exact else Fraction):
                return refuse("inconclusive", need.claim, "wrong observation type")
            node = {
                "gate": GATE_DIGEST,
                "claim": need.claim,
                "kind": need.kind.value,
                "level": need.level.value,
                "artifact": scope["artifact"],
                "subject": self.contract.subject,
                "value": str(obs.value),
                "location": obs.location,
                "inputs": {k: scope["dependencies"][k] for k in inputs},
            }
            nodes.append({"id": fingerprint(encode(node)), **node})
            target = need.condition.target
            if not (obs.value == target if exact else obs.value <= target):
                return refuse(
                    "denied",
                    need.claim,
                    "contract violated",
                    required=str(target),
                    observed=str(obs.value),
                    location=obs.location,
                )
            return "checked"

        status = kernel.decide(self.contract.needs, observe)
        record = decision(status, **failure)
        if status != "authorized":
            return None, record
        ticket = object()
        evidence = tuple(record["edges"]["evidence"])
        binding = encode([report["scope_id"], evidence])
        self._issued[ticket] = (payload, binding, evidence)
        return ticket, record

    def consume(self, ticket, expected):
        entry = self._issued.pop(ticket, None) if type(ticket) is object else None
        if entry is None:
            raise PermissionError("unknown or already consumed permission")
        if type(expected) is not bytes:
            raise PermissionError("consumer must identify immutable artifact bytes")
        payload, issued, evidence = entry
        scope = self._scope(expected, self._snapshot())
        current = encode([fingerprint(encode(scope)), evidence])
        released = kernel.release(issued, current, payload, expected)
        if released is None:
            raise PermissionError("artifact or authorization scope changed")
        return released
