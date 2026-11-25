"""Check the emitted heat kernel, compile it, and certify the returned profile."""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from solver import lower, program


HERE = Path(__file__).resolve().parent
ROW = re.compile(
    r"    \(vector-set! (old|fresh) (\d+) \(/ \(\+ \(vector-ref (old|fresh) (\d+)\) "
    r"\(\* (\d+) \(vector-ref (old|fresh) (\d+)\)\) \(vector-ref (old|fresh) (\d+)\)\) 4\)\)"
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(argv, work, **kwargs):
    return subprocess.run(argv, cwd=work, check=True, timeout=120, text=True,
                          stdout=subprocess.PIPE, **kwargs).stdout


def decode(source, n):
    """Decode every source line of the closed, straight-line Eshkol fragment."""
    lines = source.splitlines()
    require(len(lines) == n+2 and lines[0] == "(define (heat-step old)", "unsupported kernel")
    if lines[1] == f"  (let ((fresh (make-vector {n+1} 0)))":
        fresh = "fresh"
    elif lines[1] == "  (let ((fresh old))":
        fresh = "old"
    else:
        raise ValueError("unsupported allocation")
    banks = {"old": "old", "fresh": fresh}
    require(lines[-1] in ("    fresh))", "    old))"), "unsupported output")
    out = banks[lines[-1].strip()[:-2]]
    rows = []
    for line in lines[2:-1]:
        match = ROW.fullmatch(line)
        require(match, f"unsupported instruction: {line}")
        dst, i, left, j, w, mid, k, right, m = match.groups()
        addresses = [(dst, i), (left, j), (mid, k), (right, m)]
        require(all(int(idx) <= n for _, idx in addresses), "out-of-range address")
        args = [f"(.{banks[b]}, {int(idx)})" for b, idx in addresses]
        rows.append("⟨" + ", ".join([*args, str(int(w))]) + "⟩")
    return out, "[" + ",\n".join(rows) + "]"


def check_kernel(source, n, work, lean):
    out, rows = decode(source, n)
    header = (f"import Heat\nset_option maxRecDepth 16384\n"
              f"set_option maxHeartbeats 10000000\ndef rows : List Heat.Row := {rows}\n")
    claim = f"Heat.verdict {n} .{out} rows"
    path = work / "Candidate.lean"
    env = {**os.environ, "LEAN_PATH": str(work)}
    path.write_text(header + f"#eval IO.println ({claim})\n")
    verdict = run([lean, "-DwarningAsError=true", path.name], work, env=env).strip()
    if verdict == "ACCEPT":
        # Kernel reduction of this concrete obligation independently instantiates the theorem.
        path.write_text(header + f'example : {claim} = "ACCEPT" := by decide\n'
                        + f"example := Heat.admitted_refines {n} rows .{out} (by decide)\n")
        run([lean, "-DwarningAsError=true", path.name], work, env=env)
    return verdict


def atan_bounds(q):
    # Alternating series at 1/q. Two consecutive partial sums enclose atan(1/q).
    s = sum(F((-1)**j, (2*j+1)*q**(2*j+1)) for j in range(12))
    return s, s + F(1, 25*q**25)


def sine_samples(n):
    a, b = atan_bounds(5)
    c, d = atan_bounds(239)
    lo, hi = 16*a-4*d, 16*b-4*c  # Machin's identity
    require(hi < F(22, 7), "pi bound")
    values, errors = [], []
    for i in range(n+1):
        x = min(i, n-i) * (lo+hi) / (2*n)
        approx = sum((-1)**j*x**(2*j+1)/math.factorial(2*j+1) for j in range(8))
        # Round the initial sample to an exact dyadic. |sin'| <= 1.
        sample = F(round(approx * 2**32), 2**32)
        error = abs(sample-approx) + x**17/math.factorial(17) + (hi-lo)/4
        values.append(sample)
        errors.append(error)
    # Return a compact rational upper bound (upward rounding).
    return values, F(math.ceil(max(errors)*2**48), 2**48)


def analytic_check(field, bound):
    """Independently enclose exp(-pi²/50) sin(pi*i/n) at every returned cell."""
    pi = F(3141592653589793, 10**15)
    eps = F(1, 10**15)
    a, b = atan_bounds(5)
    c, d = atan_bounds(239)
    require(pi < 16*a-4*d <= 16*b-4*c < pi+eps, "analytic pi bracket")
    rate = pi**2/50
    decay = sum((-rate)**k/math.factorial(k) for k in range(15))
    decay_err = rate**15/math.factorial(15) + ((pi+eps)**2-pi**2)/50
    n = len(field)-1
    for i, value in enumerate(field):
        x = pi*i/n
        wave = sum((-1)**k*x**(2*k+1)/math.factorial(2*k+1) for k in range(12))
        wave_err = x**25/math.factorial(25) + eps
        # |true sine| <= 1; decompose the product using the rational decay.
        err = abs(value-decay*wave) + decay_err + abs(decay)*wave_err
        require(err <= bound, f"cell {i} exceeds the certified error budget")


def native(kernel, values, steps, work, compiler):
    source = program(kernel, values, steps)
    (work / "heat.esk").write_text(source)
    binary = work / "heat"
    run([compiler, "heat.esk", "-o", str(binary)], work)
    lines = run([str(binary)], work).splitlines()
    require(len(lines) == len(values) and
            all(re.fullmatch(r"-?\d+(?:/[1-9]\d*)?", s) for s in lines),
            "CPU field must contain exactly one rational per grid point")
    return list(map(F, lines)), {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }


def reference(values, steps):
    for _ in range(steps):
        values = [F(0), *[(a+2*b+c)/4 for a, b, c in zip(values, values[1:], values[2:])], F(0)]
    return values


def solve(n, work, lean, compiler):
    require(n >= 5 and n % 5 == 0, "T=1/50 requires a multiple of five intervals")
    kernel = lower(n)
    require(check_kernel(kernel, n, work, lean) == "ACCEPT", "schedule rejected")
    initial, sampling_error = sine_samples(n)
    dt = F(1, 4*n*n)
    steps = 2*n*n//25
    field, digests = native(kernel, initial, steps, work, compiler)
    require(field == reference(initial, steps), "native profile differs from exact FTCS replay")
    # Both derivative suprema for this unit-rod sine solution are at most pi^4.
    defect = F(22, 7)**4 * (dt**2/2 + dt*F(1, n*n)/12)
    bound = sampling_error + steps*defect
    analytic_check(field, bound)
    observed = max(abs(float(v)-math.exp(-math.pi**2/50)*math.sin(math.pi*i/n))
                   for i, v in enumerate(field))
    return dict(intervals=n, steps=steps, dt=str(dt), time="1/50", initial_error=str(sampling_error),
                one_step_defect=str(defect), backend_error="0", max_error_bound=str(bound),
                observed_max_error=observed, field=list(map(str, field)), **digests)


def controls(work, lean, compiler):
    kernel = lower(4)
    mutant = kernel.replace("(make-vector 5 0)", "old")
    cases = [
        (kernel, "ACCEPT"),
        (mutant, "REJECT cell 2: expected old[1], observed new[1]"),
        (kernel.replace("(* 2 ", "(* 3 "), "REJECT discrete operator or output identity differs"),
        (kernel.replace("    fresh))", "    old))"), "REJECT discrete operator or output identity differs"),
    ]
    for source, expected in cases:
        verdict = check_kernel(source, 4, work, lean)
        require(verdict == expected, f"{verdict!r} differs from {expected!r}")
    # Isolated negative-control execution. Both endpoints and the max principle pass.
    impulse = list(map(F, (0, 0, 1, 0, 0)))
    good, _ = native(kernel, impulse, 1, work, compiler)
    bad, _ = native(mutant, impulse, 1, work, compiler)
    require(good == [0, F(1, 4), F(1, 2), F(1, 4), 0], "simultaneous impulse")
    require(bad == [0, F(1, 4), F(9, 16), F(9, 64), 0], "aliased impulse")
    return cases[1][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--output", type=Path, help="write exact fields and error certificates as JSON")
    args = parser.parse_args()
    lean = os.environ.get("LEAN", "lean")
    compiler = os.environ.get("ESHKOL_RUN", "eshkol-run")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copyfile(HERE / "Heat.lean", work / "Heat.lean")
        run([lean, "-DwarningAsError=true", "-o", "Heat.olean", "Heat.lean"], work)
        rejection = controls(work, lean, compiler)
        print(rejection, flush=True)
        profiles = []
        for n in args.intervals:
            p = solve(n, work, lean, compiler)
            profiles.append(p)
            print(f"N={n} center={float(F(p['field'][n//2])):.9f} "
                  f"observed max error={p['observed_max_error']:.6g} "
                  f"certified bound={float(F(p['max_error_bound'])):.6g}", flush=True)
        if args.output:
            receipt = {
                "fragment": "P0/unit-rod-FTCS-rational/v1",
                "lean": run([lean, "--version"], work).strip(),
                "eshkol": run([compiler, "--version"], work).strip(),
                "checker_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                   for name in ("Heat.lean", "replay.py", "solver.py")},
                "mutation": rejection,
                "profiles": profiles,
            }
            args.output.write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
