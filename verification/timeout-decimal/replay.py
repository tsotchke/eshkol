"""Compare the timeout diagnostic against its upstream implementation on GNU/Linux."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "eb224260b4814baf3ad260744c8dcdd05fb84898"
SOURCE = "lib/core/resource_limits.cpp"
BASELINE_SHA = "816ae38197f86d1c1eeafa409cdad32f918667e7f5c648dec8ce31293f6bbc22"
ARCHIVE_SHA = "10a68815be29e4caacb0214af225b0a11a70c2f93c757be6b341d2117a698b22"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def replay(release, lean, work):
    archive = release / "lib/libeshkol-runtime.a"
    require(
        sha(archive.read_bytes()) == ARCHIVE_SHA, "unexpected release runtime archive"
    )
    reference = subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"], cwd=ROOT)
    require(sha(reference) == BASELINE_SHA, "unexpected upstream reference")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ESHKOL_", "GLIBC_", "LD_"))
    }
    env.update(LC_ALL="C", LEAN_PATH=str(work))

    def run(cmd, check=True):
        result = subprocess.run(cmd, env=env, cwd=work, capture_output=True, timeout=90)
        if check and result.returncode:
            raise RuntimeError(
                f"{cmd[0]} exited {result.returncode}\n{result.stderr.decode(errors='replace')}\n{result.stdout.decode(errors='replace')}"
            )
        return result

    version = run([lean, "--version"]).stdout.decode().strip()
    require(version.startswith("Lean (version 4.35.0,"), "expected Lean 4.35.0")
    model = HERE / "TimeoutDecimal.lean"
    shutil.copyfile(model, work / model.name)
    run([lean, "-DwarningAsError=true", "-o", "TimeoutDecimal.olean", model.name])
    (work / "Target.lean").write_text("""import TimeoutDecimal
open Verificatum.TimeoutDecimal
example (n : Nat) (word : n < 2^64) :
    n * 14757395258967641293 / 2^64 / 8 = n / 10 := quotient n word
example (n : Nat) (word : n < 2^64) :
    n - 10 * (n * 14757395258967641293 / 2^64 / 8) + 48 = 48 + n % 10 := digit n word
example (n : Nat) (word : n < 2^64) :
    ∃ count, writes 20 n = some count ∧ 1 ≤ count ∧ count + 3 ≤ 23 ∧
      ∀ i, i < count → 41 ≤ 60 - i ∧ 60 - i < 61 := formatterExtent n word
#print axioms Verificatum.TimeoutDecimal.quotient
#print axioms Verificatum.TimeoutDecimal.digit
#print axioms Verificatum.TimeoutDecimal.formatterExtent
""")
    axioms = (
        run([lean, "-DwarningAsError=true", "Target.lean"]).stdout.decode().splitlines()
    )
    require(len(axioms) == 3, "incomplete axiom audit")
    for line in axioms:
        names = line.partition("[")[2].removesuffix("]").split(", ")
        require(set(names) <= {"propext", "Classical.choice", "Quot.sound"}, line)

    control = HERE / "formatter_control.c"
    include = ["-I", ROOT / "inc"]
    run(
        [
            "cc",
            "-std=gnu11",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            *include,
            "-c",
            control,
            "-o",
            "control.o",
        ]
    )
    candidate = (ROOT / SOURCE).read_bytes()
    for name, content in (("reference", reference), ("candidate", candidate)):
        (work / "resource_limits.cpp").write_bytes(content)
        run(
            [
                "c++",
                "-O2",
                "-std=c++17",
                "-D_FORTIFY_SOURCE=2",
                "-fstack-protector-strong",
                *include,
                "-c",
                "resource_limits.cpp",
                "-o",
                "limits.o",
            ]
        )
        run(
            [
                "c++",
                "control.o",
                "limits.o",
                archive,
                "-lpng",
                "-ljpeg",
                "-lwebp",
                "-lcrypto",
                "-lm",
                "-lpthread",
                "-ldl",
                "-o",
                name,
            ]
        )

    def observe(image, registry, policy, value, kind="timeout"):
        # A fresh process gives each case its own GNU printf registry.
        r = run([work / image, registry, policy, str(value), kind], check=False)
        return r.returncode, r.stdout, r.stderr

    values = sorted(
        {0, 1, 30000, 2**64 - 1} | {10**p + d for p in range(1, 20) for d in (-1, 0, 1)}
    )
    counts = {}
    baseline_callbacks = {}
    sibling_cases = 0
    for policy in ("quiet", "warn", "fatal"):
        status, _, _ = observe("reference", "registered", policy, 30000)
        require(
            status in (90, 91), f"reference callback detector inactive for {policy}"
        )
        baseline_callbacks[policy] = status
        for n in values:
            original = observe("reference", "default", policy, n)
            status, stdout, stderr = original
            message = f"Execution timeout exceeded (limit {n}ms, set by ESHKOL_TIMEOUT_MS)".encode()
            require(
                status == (124 if policy == "fatal" else 0),
                f"reference status {policy}/{n}",
            )
            require(
                (stdout + stderr == b"")
                if policy == "quiet"
                else message in stdout + stderr,
                f"reference diagnostic {policy}/{n}",
            )
            for registry in ("default", "registered"):
                require(
                    observe("candidate", registry, policy, n) == original,
                    f"candidate status or diagnostic differs for {policy}/{registry}/{n}",
                )
                key = f"{policy}/{registry}"
                counts[key] = counts.get(key, 0) + 1
        for kind in ("heap-soft", "heap-hard", "stack", "tensor", "string"):
            expected = observe("reference", "default", policy, 30000, kind)
            require(
                expected[0] == 0
                if policy != "fatal"
                else expected[0] > 0 and b"control" in expected[2],
                f"invalid reference for {kind}/{policy}",
            )
            require(
                observe("candidate", "default", policy, 30000, kind) == expected,
                f"sibling diagnostic changed for {kind}/{policy}",
            )
            sibling_cases += 1
    return dict(
        base=BASE,
        toolchain=dict(
            lean=version,
            platform=platform.system() + " " + platform.machine(),
            libc=os.confstr("CS_GNU_LIBC_VERSION"),
            cxx=run(["c++", "--version"]).stdout.decode().splitlines()[0],
        ),
        axioms=axioms,
        distinct_ceilings=len(values),
        timeout_cases=counts,
        reference_callback_exit=baseline_callbacks,
        other_limit_comparisons=sibling_cases,
        sha256=dict(
            archive=ARCHIVE_SHA,
            baseline_source=BASELINE_SHA,
            candidate_source=sha(candidate),
            model=sha(model.read_bytes()),
            control=sha(control.read_bytes()),
            driver=sha(Path(__file__).read_bytes()),
            reference_elf=sha((work / "reference").read_bytes()),
            candidate_elf=sha((work / "candidate").read_bytes()),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        type=Path,
        required=True,
        help="unpacked v1.3.6-evolve Linux x64 lite release",
    )
    parser.add_argument("--lean", default="lean", help="Lean 4.35.0 executable")
    args = parser.parse_args()
    lean = shutil.which(args.lean)
    if not lean:
        parser.error("Lean 4.35.0 is required")
    with tempfile.TemporaryDirectory() as tmp:
        result = replay(args.release.resolve(), Path(lean).resolve(), Path(tmp))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
