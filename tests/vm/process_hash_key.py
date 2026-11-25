import subprocess
import sys


def sample():
    return subprocess.check_output([sys.argv[1]], text=True).strip().splitlines()[-1]


if int(sample(), 16) == int(sample(), 16):
    raise SystemExit("Independent processes produced the same table hash")
