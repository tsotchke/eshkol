import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("active_qualifier", ROOT / "scripts/run_active_integration_qualification.py")
qualifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualifier)
ICC_DYNAMIC_IMPORT_TARGETS = ["run_active_integration_qualification"]


class QualificationControls(unittest.TestCase):
    def test_checkout_root_is_not_a_build_directory(self):
        with self.assertRaises(ValueError):
            qualifier.owned_path(str(qualifier.ROOT))

    def test_outside_checkout_is_refused(self):
        with self.assertRaises(ValueError):
            qualifier.owned_path(str(qualifier.ROOT.parent / "outside"))

    def test_inside_checkout_is_accepted(self):
        self.assertEqual(qualifier.owned_path("build-active-qualification"), qualifier.ROOT / "build-active-qualification")

    def test_cache_must_enable_tests_and_disable_gpu(self):
        scratch = qualifier.ROOT / ".scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            build = Path(name)
            def write(flags, source=qualifier.ROOT):
                lines = [f"CMAKE_HOME_DIRECTORY:INTERNAL={source}"]
                lines.extend(f"{key}:BOOL={value}" for key, value in flags.items())
                (build / "CMakeCache.txt").write_text("\n".join(lines) + "\n")
            write(qualifier.FLAGS)
            qualifier.verify_cache(build)
            for key in qualifier.FLAGS:
                with self.subTest(key=key):
                    flags = dict(qualifier.FLAGS)
                    flags[key] = "OFF" if flags[key] == "ON" else "ON"
                    write(flags)
                    with self.assertRaises(ValueError):
                        qualifier.verify_cache(build)
            write(qualifier.FLAGS, qualifier.ROOT.parent)
            with self.assertRaises(ValueError):
                qualifier.verify_cache(build)


if __name__ == "__main__":
    unittest.main()
