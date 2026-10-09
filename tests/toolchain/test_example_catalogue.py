#!/usr/bin/env python3
"""Failure-injection checks for reviewed catalogue and generated-guide drift."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('example_catalogue',ROOT/'scripts/build_example_catalogue.py')
module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'.scratch').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.scratch')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'docs/examples').mkdir(parents=True)
        (self.root/'scripts').mkdir()
        (self.root/'examples').mkdir()
        # Generic over module.FAMILIES (not hardcoded to a fixed count) so a
        # new registration family is automatically covered by one fixture
        # program instead of leaking the real CMakeLists registration list
        # (and its real, untracked-by-this-fixture source files) into the
        # sandbox.
        aliases=[module.FAMILY_PREFIX[family] for family in module.FAMILIES]
        self.paths=[f'examples/mathematics_fixture_{alias}.esk' for alias in aliases]
        criteria=[f'{alias}_fixture' for alias in aliases]
        source=';; finite fixture\n(define x 2)\n(check "square" (= (* x x) 4))\n'
        for path in self.paths: (self.root/path).write_text(source)
        self.inventory=patch.object(module,'inventory',return_value=self.paths)
        self.inventory.start();self.addCleanup(self.inventory.stop)
        cmake=(ROOT/'CMakeLists.txt').read_text()
        for family,criterion,path in zip(module.FAMILIES,criteria,self.paths):
            stem=Path(path).stem
            cmake=re.sub(rf'\bset\({family}\s+[^)]*\)',f'set({family} {criterion} {stem})',cmake)
        (self.root/'CMakeLists.txt').write_text(cmake)
        shutil.copy(ROOT/'scripts/run_examples_tests.sh',self.root/'scripts/run_examples_tests.sh')
        catalogue=json.loads((ROOT/'docs/examples/catalogue.json').read_text())
        template=catalogue['entries'][0]
        self.entries=[]
        for path,family,criterion in zip(self.paths,module.FAMILIES,criteria):
            entry=copy.deepcopy(template)
            entry.update(path=path,title='Finite fixture '+criterion,source_sha256=module.digest(self.root/path),kind='mathematics',
                source_anchors=[{'line':1,'text':source.splitlines()[0]},{'line':3,'text':source.splitlines()[2]}],
                registrations=[{'family':family,'criterion':criterion,'modes':['jit','aot']}],paper_sections='fixture')
            self.entries.append(entry)
        catalogue['entries']=self.entries
        self.catalogue=catalogue
        self.save()
        ns='# Navier–Stokes fixture\n\nPreserved narrative.\n'
        for name in ['ns-summary','ns-table','ns-commands']:
            ns+=f'<!-- example-catalogue:{name}:start -->\nold\n<!-- example-catalogue:{name}:end -->\n'
        (self.root/'docs/NAVIER_STOKES_EXAMPLES.md').write_text(ns)
        (self.root/'examples/wgsl_artifact').mkdir()
        (self.root/'examples/wgsl_artifact/README.md').write_text('Artifact reference')
        (self.root/'docs/QUICKSTART.md').write_text('Quickstart reference')
        (self.root/'examples/README.md').write_text('# Examples\n<!-- example-catalogue:examples-overview:start -->\nold\n<!-- example-catalogue:examples-overview:end -->\nPreserved navigation.')
        (self.root/'docs/AI_MATHEMATICS_EXAMPLES.md').write_text('# AI fixture\nPreserved reference.\n<!-- example-catalogue:ai-inventory:start -->\nold\n<!-- example-catalogue:ai-inventory:end -->\n')

    def save(self):
        (self.root/'docs/examples/catalogue.json').write_text(json.dumps(self.catalogue))

    def snapshot(self):
        result={p.relative_to(self.root).as_posix():p.read_bytes() for p in (self.root/'docs').rglob('*') if p.is_file()}
        result['examples/README.md']=(self.root/'examples/README.md').read_bytes()
        return result

    def reject_without_write(self):
        before=self.snapshot()
        with self.assertRaises(module.CatalogueError): module.build(self.root,write=True)
        self.assertEqual(self.snapshot(),before)

    def test_positive_write_is_idempotent_and_preserves_narrative(self):
        result=module.build(self.root,write=True)
        self.assertEqual(result['programs'],len(module.FAMILIES))
        before=self.snapshot()
        self.assertEqual(module.build(self.root,write=True)['changed'],[])
        self.assertEqual(module.build(self.root)['changed'],[])
        self.assertEqual(self.snapshot(),before)
        self.assertNotRegex((self.root/'docs/EXAMPLES.md').read_text(), r'\|\n\n\|')
        self.assertNotRegex((self.root/'docs/MATHEMATICS_EXAMPLES.md').read_text(), r'\|\n\n\|')
        self.assertIn('Preserved narrative.',(self.root/'docs/NAVIER_STOKES_EXAMPLES.md').read_text())
        self.assertIn('Preserved navigation.',(self.root/'examples/README.md').read_text())
        self.assertNotIn('under a minute',(self.root/'examples/README.md').read_text())
        matrix=json.loads((self.root/'docs/examples/test-matrix.json').read_text())
        self.assertEqual(matrix['families']['ESHKOL_NS_EXAMPLES']['ctest_entries'],2)

    def test_failed_empty_or_invalid_git_inventory_cannot_overwrite_outputs(self):
        self.inventory.stop()
        for code,stdout in [(7,b'error'),(0,b''),(0,b'\xff\0')]:
            with self.subTest(code=code,stdout=stdout), patch.object(module.subprocess,'run',return_value=subprocess.CompletedProcess([],code,stdout,b'')):
                self.reject_without_write()
        self.inventory.start()

    def test_changed_source_requires_semantic_review(self):
        (self.root/self.paths[0]).write_text(';; changed algorithm\n(define x 99)\n')
        self.reject_without_write()

    def test_inventory_catalogue_omissions_extras_duplicates_and_wrong_schema_reject(self):
        original=copy.deepcopy(self.catalogue)
        for change in ['missing','extra','duplicate','schema','execution']:
            self.catalogue=copy.deepcopy(original)
            if change=='missing':self.catalogue['entries'].pop()
            elif change=='extra':self.catalogue['entries'][0]['path']='examples/not_tracked.esk'
            elif change=='duplicate':self.catalogue['entries'].append(self.catalogue['entries'][0])
            elif change=='schema':self.catalogue['schema']='other'
            else:self.catalogue['entries'][0]['execution_evidence']='PASS all examples'
            self.save()
            with self.subTest(change=change):self.reject_without_write()

    def test_missing_semantics_or_boolean_anchor_reject(self):
        original=copy.deepcopy(self.catalogue)
        for field in module.FIELDS:
            self.catalogue=copy.deepcopy(original);self.catalogue['entries'][0][field]='';self.save()
            with self.subTest(field=field):self.reject_without_write()
        self.catalogue=copy.deepcopy(original);self.catalogue['entries'][0]['source_anchors'][0]['line']=True;self.save()
        self.reject_without_write()

    def test_matrix_odd_duplicate_unknown_missing_mode_and_wrong_command_reject(self):
        path=self.root/'CMakeLists.txt';original=path.read_text()
        mutations=[lambda t:t.replace('ns_fixture mathematics_fixture_ns','ns_fixture'),
                   lambda t:t.replace('ns_fixture mathematics_fixture_ns','ns_fixture mathematics_fixture_ns ns_fixture mathematics_fixture_ipm'),
                   lambda t:t.replace('mathematics_fixture_ns)','missing_source)'),
                   lambda t:t.replace('add_test(NAME ${_ns_name}_jit','add_test(NAME ${_ns_name}_other'),
                   lambda t:t.replace('-r "${_ns_src}"','--audit-only "${_ns_src}"'),
                   lambda t:t.replace('if(ESHKOL_BUILD_TESTS AND TARGET eshkol-run)','if(ESHKOL_BUILD_TESTS)',1)]
        for mutate in mutations:
            path.write_text(mutate(original))
            with self.subTest(mutation=mutations.index(mutate)):self.reject_without_write()
        path.write_text(original)
        self.catalogue['entries'][0]['registrations'][0]['criterion']='old';self.save();self.reject_without_write()

    def test_runner_change_or_reviewed_skip_mismatch_reject(self):
        path=self.root/'scripts/run_examples_tests.sh';original=path.read_text()
        path.write_text(original.replace('examples/*.esk','examples/selected*.esk'))
        self.reject_without_write();path.write_text(original)
        self.catalogue['runner_contract']['excluded_patterns']=[];self.save();self.reject_without_write()

    def test_generated_output_drift_fails_check_and_write_repairs_only_outputs(self):
        module.build(self.root,write=True)
        path=self.root/'docs/EXAMPLES.md';path.write_text('stale count and commands')
        before=self.snapshot()
        with self.assertRaisesRegex(module.CatalogueError,'drift'):module.build(self.root)
        self.assertEqual(self.snapshot(),before)
        self.assertEqual(module.build(self.root,write=True)['changed'],['docs/EXAMPLES.md'])

    def test_broken_link_or_source_anchor_fails_before_any_write(self):
        real_render=module.render
        def broken(*args):
            result=real_render(*args);result['docs/EXAMPLES.md']+='\n[broken](../missing.esk#L1)\n';return result
        with patch.object(module,'render',side_effect=broken):self.reject_without_write()
        self.catalogue['entries'][0]['source_anchors'][0]['text']='invented';self.save();self.reject_without_write()

    def test_duplicate_markers_and_symlink_destination_reject(self):
        path=self.root/'docs/NAVIER_STOKES_EXAMPLES.md';original=path.read_text()
        path.write_text(original+'\n<!-- example-catalogue:ns-table:start -->\n')
        self.reject_without_write();path.write_text(original)
        outside=self.root/'other.md';outside.write_text('preserve me')
        destination=self.root/'docs/EXAMPLES.md';destination.symlink_to(outside)
        self.reject_without_write();self.assertEqual(outside.read_text(),'preserve me')

    def test_real_catalogue_completeness_exact_witness_and_scope_boundaries(self):
        data=json.loads((ROOT/'docs/examples/catalogue.json').read_text())
        self.assertEqual(len(data['entries']),67)
        self.assertEqual(sum(e['kind']=='mathematics' for e in data['entries']),46)
        pages=json.loads((ROOT/'site/pages.json').read_text())['pages']
        for name in ('docs/EXAMPLES.md','docs/MATHEMATICS_EXAMPLES.md'):
            self.assertEqual(sum(page['file']==name for page in pages),1)
        entries={Path(e['path']).stem:e for e in data['entries']}
        self.assertIn('globally',entries['mathematics_jacobian_counterexample']['validation'])
        self.assertIn('1e-9',entries['mathematics_jacobian_counterexample']['arithmetic'])
        self.assertIn('every pair',entries['mathematics_alphatensor_3x3_gf2']['validation'])
        self.assertIn('existing simplex',entries['mathematics_kan_complexes_horns']['limitations'])
        self.assertIn('MATH_WITNESS_K',entries['mathematics_aoki_cycles_fermat_surfaces_sweep']['domain'])
        self.assertIn('no percentile',entries['streaming_stats']['limitations'])
        self.assertIn('finite-difference',entries['h2_vibrational_full']['arithmetic'])


if __name__=='__main__':unittest.main()
