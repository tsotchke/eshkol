#!/usr/bin/env python3
"""Negative controls for release identity, scores, and publication notes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('release_guard', ROOT / 'scripts/release_readiness_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
SHA = 'a' * 40


class ReleaseGuardTests(unittest.TestCase):
    def test_exact_score_required(self):
        for payload in ({'status': 'ready'}, {'status': 'ready', 'score': None},
                        {'status': 'ready', 'score': '100'}, {'status': 'ready', 'score': 99},
                        {'status': 'blocked', 'score': 100}, {'status': 'ready', 'score': True}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                guard.check_verdict(payload)
        guard.check_verdict({'status': 'ready', 'score': 100})
        guard.check_verdict({'status': 'ready', 'readiness': 100})

    def test_planted_pending_sentinel_blocks_publication(self):
        clean = '# Eshkol v1.3.5-evolve — Release Notes\n\nMeasured release.\n'
        poisoned = clean + '\n<!-- RELEASE_EVIDENCE_PENDING -->\n'
        self.assertEqual(guard.release_notes(clean, 'v1.3.5-evolve'), clean)
        with self.assertRaises(ValueError):
            guard.release_notes(poisoned, 'v1.3.5-evolve')
        self.assertIn(guard.PENDING, guard.release_notes(poisoned, 'v1.3.5-evolve', True))
        # A prior release's notes cannot contaminate the current extraction.
        self.assertEqual(guard.release_notes(clean + '\n---\n' + poisoned,
                                            'v1.3.5-evolve'), clean)
        with self.assertRaises(ValueError):
            guard.release_notes(clean, 'v1.3.4-evolve')

    def backend(self, registry, head=SHA, dirty=False):
        def run(args):
            if args[0] == 'git':
                if 'rev-parse' in args:
                    return subprocess.CompletedProcess(args, 0, head + '\n', '')
                return subprocess.CompletedProcess(args, int(dirty), '', '')
            key = '--name' if args[1] == 'register' else '--repo'
            repo = args[args.index(key) + 1]
            if args[1] == 'resolve':
                data = {'ok': True, 'repo': {'path': registry[repo]}} if repo in registry else {'ok': False}
                return subprocess.CompletedProcess(args, 0 if repo in registry else 1, json.dumps(data), '')
            if args[1] == 'register':
                self.assertNotEqual(repo, 'primary')
                registry[repo] = args[args.index('--path') + 1]
                return subprocess.CompletedProcess(args, 0, '{}', '')
            raise AssertionError(args)
        return run

    def test_wrong_primary_gets_separate_verified_alias(self):
        registry = {'primary': '/some/other/checkout'}
        with patch.object(guard, 'command', self.backend(registry)):
            selected = guard.bind('icc', 'primary', ROOT, SHA)
            self.assertNotEqual(selected, 'primary')
            self.assertEqual(registry['primary'], '/some/other/checkout')
            self.assertEqual(Path(registry[selected]), ROOT)
            self.assertEqual(guard.bind('icc', 'primary', ROOT, SHA), selected)
            guard.verify_binding('icc', selected, ROOT, SHA)
            with self.assertRaises(ValueError):
                guard.verify_binding('icc', 'primary', ROOT, SHA)

    def test_correct_primary_is_not_rebound(self):
        registry = {'primary': str(ROOT)}
        with patch.object(guard, 'command', self.backend(registry)):
            self.assertEqual(guard.bind('icc', 'primary', ROOT, SHA), 'primary')
        self.assertEqual(len(registry), 1)

    def test_wrong_sha_and_dirty_checkout_are_rejected_before_binding(self):
        for head, dirty in [('b' * 40, False), (SHA, True)]:
            registry = {}
            with patch.object(guard, 'command', self.backend(registry, head, dirty)):
                with self.assertRaises(ValueError):
                    guard.bind('icc', 'primary', ROOT, SHA)
            self.assertEqual(registry, {})


if __name__ == '__main__':
    unittest.main()
