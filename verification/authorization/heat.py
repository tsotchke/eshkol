"""Two consumer contracts over one locally built P5 executable."""

import subprocess
import tempfile
from fractions import Fraction as F
from pathlib import Path

from authority import (
    AtMost,
    Checker,
    Contract,
    Equal,
    Gate,
    Kind,
    Level,
    Need,
    Observation,
    Permission,
    Unsupported,
    encode,
    fingerprint,
)
from fixtures import rod

SOURCE = "c2cdc9f3b8c82e901fc6a889afad8e2797029350771fe76058dcb0dd789e0a8c"
GRID = encode(["sine rod", "zero endpoints", 205, 3362, "1/50", "dyadic32"])
LIMIT = F(1, 100000)


def build(compiler, source):
    if fingerprint(source) != SOURCE:
        raise ValueError("unexpected heat source")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "heat.esk").write_bytes(source)
        subprocess.run(
            [compiler, "heat.esk", "-o", "heat"],
            cwd=folder,
            capture_output=True,
            check=True,
            timeout=120,
        )
        return (folder / "heat").read_bytes()


def run(binary, initial, steps):
    # Execute only bytes from the consumer's own build. A private directory
    # gives the sequential runner sole ownership of the file until exit.
    data = "\n".join(map(str, [len(initial) - 1, steps, *initial])) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "heat"
        exe.write_bytes(binary)
        exe.chmod(0o700)
        proc = subprocess.run(
            [exe], input=data, text=True, check=True, capture_output=True, timeout=120, cwd=tmp
        )
    field = [F(line) for line in proc.stdout.splitlines()]
    if len(field) != len(initial):
        raise Unsupported("native output has the wrong extent")
    return field


def binding(binary, ctx):
    return Observation(
        "matched" if binary == ctx["solver"] else "different", "consumer-built P5 executable"
    )


def arithmetic(binary, ctx):
    witness = [F(0), F(1, 2**52), F(1), F(1, 2**52), F(0)]
    actual = run(binary, witness, 1)[2]
    if actual == F(1, 2):
        raise Unsupported("one matching triple cannot establish all-input arithmetic")
    return Observation(str(actual), "one step, cell 2 on [0, 2^-52, 1, 2^-52, 0]")


def solve(binary):
    initial, _ = rod.sine_samples(205)
    return run(binary, initial, 3362)


def numeric(binary, ctx):
    if ctx["grid"] != GRID:
        raise Unsupported("unknown initial condition, grid or final time")
    field = solve(binary)
    error, cell = rod.max_error(field)
    # Round the rational enclosure upwards for a short, portable certificate.
    scale = 10**12
    upper = F(-(-error.numerator * scale // error.denominator), scale)
    return Observation(upper, f"largest grid-point error enclosure at cell {cell}")


def gates(read):
    base = ("solver", "heat-checker", "native-build")
    checks = {
        "artifact": Checker(Kind.EXACT, Level.ARTIFACT, base, binding),
        "prescribed arithmetic": Checker(Kind.EXACT, Level.OBSERVED, base, arithmetic),
        "grid error": Checker(Kind.BOUNDED_ERROR, Level.OBSERVED, (*base, "grid"), numeric),
    }
    artifact = Need("artifact", Kind.EXACT, Level.ARTIFACT, Equal("matched"))
    exact = Need("prescribed arithmetic", Kind.EXACT, Level.OBSERVED, Equal("1/2"))
    bounded = Need("grid error", Kind.BOUNDED_ERROR, Level.OBSERVED, AtMost(LIMIT))
    return (
        Gate(
            Contract(
                "P5 sine rod",
                "select prescribed arithmetic",
                Permission.EXECUTION,
                (artifact, exact),
            ),
            checks,
            read,
        ),
        Gate(
            Contract(
                "P5 sine rod",
                "run the declared sine grid",
                Permission.EXECUTION,
                (artifact, bounded),
            ),
            checks,
            read,
        ),
    )
