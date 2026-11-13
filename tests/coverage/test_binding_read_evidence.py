"""Origin reads never fabricate invocation or same-spelled alias credit."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('language_coverage',ROOT/'scripts/language_coverage.py')
cov=importlib.util.module_from_spec(spec);spec.loader.exec_module(cov)
class ReadEvidence(unittest.TestCase):
    def evidence(self, identity):
        with tempfile.TemporaryDirectory() as directory:
            marker=cov.vm_call_name_hash(identity)
            Path(directory,'read.tsv').write_text('V\t<vm>\t0\t0\t%d\t@read\n'%marker)
            return cov.load_runtime_evidence([directory])
    def test_value_read_is_not_a_call(self):
        identity='core.streams:lib/core/streams.esk#stream-null'
        e=self.evidence(identity)
        self.assertIn('stream-null',e['value_read_names'])
        self.assertIn('stream-null',e['covered_names'])
        self.assertNotIn('stream-null',e['vm_dispatch_names'])
        self.assertNotIn('stream-null',e['direct_call_names'])
    def test_procedure_read_never_credits_invocation(self):
        e=self.evidence('core.streams:lib/core/streams.esk#stream-car')
        self.assertIn('stream-car',e['binding_read_names'])
        self.assertNotIn('stream-car',e['covered_names'])
        self.assertNotIn('stream-car',e['vm_dispatch_names'])
    def test_builtin_read_never_credits_invocation(self):
        e=self.evidence('vm.builtin:+')
        self.assertIn('+',e['binding_read_names'])
        self.assertNotIn('+',e['covered_names'])
    def test_unknown_origin_cannot_credit_a_spelling(self):
        e=self.evidence('fake:lib/core/streams.esk#stream-null')
        self.assertFalse(e['binding_read_names'])
        self.assertFalse(e['covered_names'])
    def test_read_collision_fails_closed(self):
        with patch.object(cov,'vm_call_name_hash',return_value=7):
            with self.assertRaisesRegex(RuntimeError,'ambiguous.*binding'):
                self.evidence('anything')
    def test_shared_origin_does_not_imply_alias_was_read(self):
        manifest=cov.load_manifest()
        original=next(x for x in manifest['prelude'] if x['name']=='stream-null')
        alias=json.loads(json.dumps(original));alias['name']='different-public-alias'
        manifest['prelude'].append(alias)
        with patch.object(cov,'load_manifest',return_value=manifest):
            e=self.evidence('core.streams:lib/core/streams.esk#stream-null')
        self.assertNotIn('different-public-alias',e['value_read_names'])
        self.assertNotIn('different-public-alias',e['binding_read_names'])
if __name__=='__main__':unittest.main()
