#!/usr/bin/env python3
"""check_region_tables.py — fail if lib/backend/xla/region_tables.inc has
drifted from the two YAML tables it is generated from.

The region-formation pass decides eligibility from a compiled-in table, which
is the only way a pass inside the compiler can classify a builtin without
finding a file on disk (see the header of scripts/gen_region_tables.py). The
cost of compiling it in is that the table can go stale: a builtin reclassified
in builtin_classification.yaml, or a lowering added to
device_lowering_table.yaml, changes nothing until someone regenerates.

A stale table does not crash. It forms the wrong regions and reports breaks
that are no longer real, which is the same class of silent wrongness the whole
stage exists to prevent. So the check is mechanical: regenerate into memory and
compare against what is committed.

Exit status: 0 when they agree, 1 when they do not, 2 when the check itself
could not run.
"""

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR = os.path.join(REPO, "scripts", "gen_region_tables.py")
COMMITTED = os.path.join(REPO, "lib", "backend", "xla", "region_tables.inc")


def main():
    if not os.path.exists(COMMITTED):
        sys.stderr.write(
            "lib/backend/xla/region_tables.inc is missing; "
            "run python3 scripts/gen_region_tables.py\n")
        return 1

    with open(COMMITTED, "r", encoding="utf-8") as fh:
        before = fh.read()

    # The generator writes to a fixed path, so the committed file is saved and
    # restored around the regeneration rather than the generator being asked to
    # write elsewhere. A temporary directory would need the generator to grow a
    # flag whose only user is this check.
    fd, backup = tempfile.mkstemp(dir=os.path.dirname(COMMITTED),
                                  prefix=".region_tables.", suffix=".bak")
    os.close(fd)
    try:
        with open(backup, "w", encoding="utf-8") as fh:
            fh.write(before)
        result = subprocess.run([sys.executable, GENERATOR],
                                capture_output=True, text=True)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            return 2
        with open(COMMITTED, "r", encoding="utf-8") as fh:
            after = fh.read()
    finally:
        with open(COMMITTED, "w", encoding="utf-8") as fh:
            fh.write(before)
        os.unlink(backup)

    if before == after:
        sys.stdout.write("region_tables.inc agrees with both YAML tables\n")
        return 0

    sys.stderr.write(
        "lib/backend/xla/region_tables.inc is STALE: it does not match\n"
        "  lib/backend/xla/builtin_classification.yaml\n"
        "  lib/backend/xla/device_lowering_table.yaml\n"
        "Regenerate it with: python3 scripts/gen_region_tables.py\n"
        "and review the diff — a label that moved changes which subgraphs\n"
        "region formation outlines.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
