"""Consumer-side execution identity and one-use launch failure reporting."""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
from enum import Enum
import hashlib
import json
from pathlib import Path


SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL


@contextmanager
def sealed_image(data):
    fd = os.memfd_create("eshkol-abi-fingerprint", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    with os.fdopen(fd, "w+b") as handle:
        handle.write(data)
        handle.flush()
        os.fchmod(fd, 0o500)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
        if fcntl.fcntl(fd, fcntl.F_GET_SEALS) & SEALS != SEALS:
            raise RuntimeError("incomplete snapshot seals")
        yield fd


@dataclass(frozen=True)
class ExecutionBinding:
    argv: tuple[str, ...]
    cwd: str
    environment: tuple[tuple[str, str], ...]
    dependency_contract: bytes

    def __post_init__(self):
        contract = json.loads(self.dependency_contract)
        names = set()
        for category in ("fixed", "permitted_to_vary", "outside"):
            entries = contract[category]
            if not isinstance(entries, dict) or not entries or names.intersection(entries):
                raise ValueError("dependency classes must be nonempty and disjoint")
            names.update(entries)

    def operation(self):
        return [self.argv, self.cwd, dict(self.environment)]


@dataclass(frozen=True)
class FileClosure:
    """The consumer's selected path for each mandatory file, including aliases."""

    paths: tuple[tuple[str, Path], ...]

    def manifest(self):
        result = {}
        for name, path in self.paths:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            result[name] = digest.hexdigest()
        return result


class Phase(str, Enum):
    RELEASED = "bytes released"
    REQUESTED = "launch requested"
    STARTED = "child handle received"
    EXITED = "child exit observed"
    PUBLISHED = "result published"


class LaunchFailure(RuntimeError):
    def __init__(self, phase, cause):
        self.phase = phase
        super().__init__(f"{phase.value}: {cause}")


class Attempt:
    """Tracks confirmed progress after the issuing gate has spent the ticket."""

    def __init__(self):
        self.phase = Phase.RELEASED

    def advance(self, phase):
        following = {
            Phase.RELEASED: Phase.REQUESTED,
            Phase.REQUESTED: Phase.STARTED,
            Phase.STARTED: Phase.EXITED,
            Phase.EXITED: Phase.PUBLISHED,
        }
        if following.get(self.phase) != phase:
            raise RuntimeError("invalid launch transition")
        self.phase = phase

    def __enter__(self):
        return self

    def __exit__(self, exc_type, cause, traceback):
        if cause is None and self.phase is not Phase.PUBLISHED:
            raise LaunchFailure(self.phase, "attempt ended without publication")
        if isinstance(cause, Exception):
            raise LaunchFailure(self.phase, cause) from cause
