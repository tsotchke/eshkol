"""One consuming entry point for the R2-reviewed Linux x64 ABI fingerprint."""

import argparse
from binding import Attempt, ExecutionBinding, FileClosure, Phase, sealed_image
import json
import os
from pathlib import Path
import re
import subprocess
from subprocess import Popen
import sys
import tempfile
from threading import Lock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "authorization"))
from authority import (  # noqa: E402 - the frozen sibling package uses local imports.
    Checker,
    Contract,
    Equal,
    Gate,
    Kind,
    Level,
    Need,
    Observation,
    Permission,
    encode,
    fingerprint,
)

R2_ID = "b3ae3c9894788c12bdc721c3d95d2fa7b9bd6dfaaee796a6130dda5c297676d6"
ARGV = ("eshkol-run", "--abi-fingerprint")
LOADER = "/lib64/ld-linux-x86-64.so.2"


def reviewed_images(reference):
    receipt = (HERE / "r2.json").read_bytes()
    if fingerprint(receipt) != R2_ID:
        raise ValueError("unknown R2 decision")
    selected = json.loads(receipt)
    if fingerprint(reference) != selected["reference_sha256"]:
        raise ValueError("expected the reviewed v1.3.6-evolve x64 lite release")
    # In this pinned ELF the reviewed site's address equals its file offset.
    site = selected["site"]
    fold = bytes([0xB8]) + selected["required_return"].to_bytes(4, "little")
    folded = reference[:site] + fold + reference[site + len(fold) :]
    if fingerprint(folded) != selected["selected_sha256"]:
        raise ValueError("folded image differs from the reviewed R2 decision")
    return folded, selected


def loader_files(fd, env):
    # The descriptor here always contains the verifier's pinned ELF. The loader
    # lists its startup dependencies with the same environment as the sink.
    lines = subprocess.check_output(
        [LOADER, "--list", f"/proc/self/fd/{fd}"],
        cwd="/",
        env=env,
        pass_fds=(fd,),
        text=True,
        timeout=20,
    )
    paths = {}
    for line in lines.splitlines():
        if line.strip().startswith("linux-vdso.so.1 "):
            continue
        match = re.fullmatch(r"\s*(?:(\S+) => )?(/.+?) \(0x[0-9a-f]+\)", line)
        if not match:
            raise ValueError("unknown dynamic dependency")
        label, path = match.groups()
        key = label or Path(path).name
        if key in paths:
            raise ValueError("duplicate dynamic dependency")
        paths[key] = Path(path)
    if Path(LOADER).name not in paths or "libLLVM.so.21.1" not in paths:
        raise ValueError("incomplete release loader closure")
    return FileClosure(tuple(sorted(paths.items())))


ABI_CONTEXT_ID = "1d4972430e274f2d8e478ca7bfb264c3d2de49e5398b98e97538bc2dfd8d6ce7"


def reviewed_context(raw):
    if fingerprint(raw) != ABI_CONTEXT_ID:
        raise ValueError("unsupported ABI execution contract")
    return raw


class FingerprintSink:
    """Consumer-owned gate, exact operation, and live policy generation."""

    def __init__(self, reference, library_dirs, policy_dir):
        self._folded, r2 = reviewed_images(reference)
        self._expected_output = ("\n".join(r2["fingerprint"]) + "\n").encode()
        self._lock = Lock()
        directories = [str(Path(p).resolve(strict=True)) for p in library_dirs]
        if any(":" in p or not Path(p).is_dir() for p in directories):
            raise ValueError("invalid operator library directories")
        self._binding = ExecutionBinding(
            ARGV,
            "/",
            (("LC_ALL", "C"), ("LD_LIBRARY_PATH", ":".join(directories))),
            reviewed_context((HERE / "abi-context.json").read_bytes()),
        )
        with sealed_image(reference) as fd:
            self._startup_files = loader_files(fd, dict(self.binding.environment))
        self._approved_loader = encode(self._startup_files.manifest())
        self._epoch = 0
        self._record = Path(policy_dir) / "abi-policy.json"
        # An owner starts in a new policy directory. Old serialized records
        # cannot initialize a new issuing session.
        with self._record.open("xb") as handle:
            handle.write(self._policy())
        self._sink_code = encode(
            {name: fingerprint((HERE / name).read_bytes()) for name in ("sink.py", "binding.py")}
        )
        need = Need("R2 image and live policy", Kind.EXACT, Level.ARTIFACT, Equal("bound"))
        contract = Contract(
            "R2 CLI ABI header fold", "emit ABI fingerprint", Permission.EXECUTION, (need,)
        )
        checker = Checker(
            need.kind,
            need.level,
            (
                "r2",
                "operation",
                "dependency contract",
                "live policy",
                "published policy",
                "loader",
                "sink",
            ),
            self._check,
        )
        self._gate = Gate(contract, {need.claim: checker}, self._context)

    @property
    def binding(self):
        return self._binding

    def _policy(self):
        return encode({"generation": self._epoch, "action": list(ARGV)})

    def renew_policy(self):
        # Advance live authority first. Any interrupted publication fails closed.
        with self._lock:
            self._epoch += 1
            with tempfile.NamedTemporaryFile(dir=self._record.parent, delete=False) as handle:
                handle.write(self._policy())
            os.replace(handle.name, self._record)

    def _context(self):
        # Re-read selected files as data before asking the operator's trusted
        # loader to resolve the closure again. A changed baseline cannot qualify.
        try:
            libraries = self._startup_files.manifest()
            if encode(libraries) == self._approved_loader:
                with sealed_image(self._folded) as fd:
                    files = loader_files(fd, dict(self.binding.environment))
                if files.paths != self._startup_files.paths:
                    libraries = {}
        except (OSError, subprocess.SubprocessError, ValueError):
            libraries = None
        try:
            published = self._record.read_bytes()
        except FileNotFoundError:
            published = b""
        context = {
            "r2": R2_ID.encode(),
            "sink": self._sink_code,
            "operation": encode(self.binding.operation()),
            "dependency contract": self.binding.dependency_contract,
            "live policy": self._policy(),
            "published policy": published,
        }

        if libraries is not None:
            context["loader"] = encode(libraries)
        return context

    def _check(self, candidate, ctx):
        if ctx["published policy"] != ctx["live policy"]:
            return Observation("stale policy", "consumer generation")
        if ctx["loader"] != self._approved_loader:
            return Observation("changed loader closure", "operator reference installation")
        if candidate != self._folded:
            return Observation("different ELF", "pinned R2 selection")
        return Observation("bound", "R2 selection, operation and consumer generation")

    def authorize(self, candidate):
        with self._lock:
            return self._gate.assess(candidate)

    def fingerprint(self, ticket):
        with self._lock:
            image = self._gate.consume(ticket, self._folded)
            with Attempt() as attempt:
                with sealed_image(image) as fd:
                    attempt.advance(Phase.REQUESTED)
                    proc = Popen(
                        self.binding.argv,
                        executable=f"/proc/self/fd/{fd}",
                        pass_fds=(fd,),
                        close_fds=True,
                        cwd=self.binding.cwd,
                        env=dict(self.binding.environment),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    with proc:
                        attempt.advance(Phase.STARTED)
                        try:
                            output, error = proc.communicate(timeout=30)
                        except BaseException:
                            # Retire the child even if pipe collection or the caller is interrupted.
                            proc.kill()
                            proc.wait()
                            attempt.advance(Phase.EXITED)
                            raise
                        attempt.advance(Phase.EXITED)
                        if proc.returncode or error:
                            raise RuntimeError("ABI fingerprint execution failed")
                if output != self._expected_output:
                    raise ValueError("unexpected CLI observation")
                if self._record.read_bytes() != self._policy():
                    raise ValueError("policy changed before publication")
                if encode(self._startup_files.manifest()) != self._approved_loader:
                    raise ValueError("startup dependency changed before publication")
                attempt.advance(Phase.PUBLISHED)
                return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--llvm-dir", type=Path, default=Path("/usr/lib/llvm-21/lib"))
    parser.add_argument(
        "--candidate", type=Path, help="proposed R2 fold; default reconstructs the reviewed image"
    )
    args = parser.parse_args()
    reference = args.reference.resolve(strict=True)
    original = reference.read_bytes()
    proposal = args.candidate.read_bytes() if args.candidate else reviewed_images(original)[0]
    with tempfile.TemporaryDirectory() as policy_dir:
        consumer = FingerprintSink(
            original,
            [reference.parent.parent / "lib/eshkol/runtime-deps", args.llvm_dir],
            policy_dir,
        )
        permit, decision = consumer.authorize(proposal)
        if permit is None:
            parser.exit(1, f"{decision['status']}: {decision['rule']} ({decision['reason']})\n")
        sys.stdout.buffer.write(consumer.fingerprint(permit))
