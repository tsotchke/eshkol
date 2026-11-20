"""Exercise the input boundaries of release evidence producers."""
from pathlib import Path
import subprocess
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


class ReleaseGateInputs(unittest.TestCase):
    def test_candidate_cannot_select_readiness_target(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
        target = workflow["env"]["RELEASE_TARGET"]
        self.assertEqual(target, "v1.4-connection")
        for tag, oracle, expected in (
            ("v1.4.0-connection", target, 0),
            ("v1.3.6-evolve", target, 1),
            ("v1.3.6-evolve", "v1.3.6-evolve", 1),
            ("v1.4.0-connection", "v1.4.0-connection", 1),
        ):
            with self.subTest(tag=tag, oracle=oracle):
                result = subprocess.run([sys.executable, SCRIPTS / "release_target.py",
                                         "--tag", tag, "--target", oracle], capture_output=True)
                self.assertEqual(result.returncode, expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
