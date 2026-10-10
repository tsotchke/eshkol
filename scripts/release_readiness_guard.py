#!/usr/bin/env python3
"""Bind release evidence to its checkout and reject incomplete publication."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_target import validate_target
from release_publication_contract import load_record, validate_notes, validate_publication, read_json

PENDING = 'RELEASE_EVIDENCE_PENDING'


def command(args):
    return subprocess.run(args, capture_output=True, text=True)


def check_workspace(workspace, sha):
    workspace = Path(workspace).resolve()
    if not re.fullmatch(r'[0-9a-fA-F]{40}', sha):
        raise ValueError('release SHA must be a full commit hash')
    result = command(['git', '-C', str(workspace), 'rev-parse', 'HEAD'])
    if result.returncode or result.stdout.strip().lower() != sha.lower():
        raise ValueError('checkout HEAD does not match the release SHA')
    if command(['git', '-C', str(workspace), 'diff', '--quiet', 'HEAD', '--']).returncode:
        raise ValueError('release checkout has tracked source changes')
    return workspace


def resolve(icc, repo):
    result = command([icc, 'resolve', '--repo', repo, '--format', 'json'])
    if result.returncode:
        return None
    try:
        value = json.loads(result.stdout)
        if value.get('ok') is not True:
            return None
        return Path(value['repo']['path']).resolve()
    except (ValueError, KeyError, TypeError):
        raise ValueError('ICC returned an invalid repository resolution')


def verify_binding(icc, repo, workspace, sha):
    workspace = check_workspace(workspace, sha)
    if resolve(icc, repo) != workspace:
        raise ValueError('ICC repository does not resolve to the release checkout')
    return repo


def bind(icc, requested, workspace, sha):
    workspace = check_workspace(workspace, sha)
    if resolve(icc, requested) == workspace:
        return verify_binding(icc, requested, workspace, sha)
    # Never rebind the maintainer's configured repository. This namespace is
    # deterministic for one source cut at one Actions working directory.
    path_key = hashlib.sha256(str(workspace).encode()).hexdigest()[:12]
    selected = f'eshkol-release-ci-{sha.lower()}-{path_key}'
    existing = resolve(icc, selected)
    if existing is not None and existing != workspace:
        raise ValueError('release-CI alias already belongs to another checkout')
    if existing is None:
        result = command([icc, 'register', '--name', selected, '--path', str(workspace)])
        if result.returncode:
            raise ValueError('could not register the dedicated release-CI alias')
    return verify_binding(icc, selected, workspace, sha)


def check_verdict(payload, target):
    # Older ICC versions call the numeric score `readiness`. A missing or
    # nonnumeric score is never a successful ready/100 release verdict.
    if not isinstance(payload, dict):
        raise ValueError('release readiness result must be a JSON object')
    validate_target(target)
    score = payload.get('score', payload.get('readiness'))
    if (payload.get('target') != target or payload.get('status') != 'ready'
            or type(score) not in (int, float) or score != 100):
        raise ValueError('release readiness requires status=ready and numeric score=100')


def release_notes(text, tag, allow_pending=False, *, record=None, role=None):
    role = role or ("preparation" if allow_pending else "tag-publication")
    if allow_pending and role != "preparation":
        raise ValueError("--allow-pending is permitted only for nonpublishing preparation")
    if record is None:
        raise ValueError("typed release record is required; marker removal cannot qualify notes")
    return validate_notes(text, record, tag, role)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('bind', 'check'):
        p = sub.add_parser(name)
        for option in ('icc', 'repo', 'workspace', 'sha'):
            p.add_argument('--' + option, required=True)
        if name == 'bind':
            p.add_argument('--github-env', required=True)
        else:
            p.add_argument('--verdict')
        p.add_argument('--target', required=True)
    p = sub.add_parser('notes')
    p.add_argument('--notes', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--allow-pending', action='store_true')
    p.add_argument('--role', choices=('preparation', 'candidate-proof', 'tag-publication'))
    p.add_argument('--record', default='tests/coverage/release_record.json')
    p.add_argument('--evidence-root')
    p.add_argument('--receipt')
    p.add_argument('--workspace')
    p.add_argument('--sha')
    p.add_argument('--target')
    p.add_argument('--run-id', type=int)
    p.add_argument('--run-attempt', type=int)
    args = parser.parse_args()
    try:
        if args.action == 'bind':
            selected = bind(args.icc, args.repo, args.workspace, args.sha)
            with open(args.github_env, 'a', encoding='utf-8') as out:
                out.write(f'ICC_REPO={selected}\n')
            print('PASS: ICC is bound to the release checkout and SHA')
        elif args.action == 'check':
            verify_binding(args.icc, args.repo, args.workspace, args.sha)
            if args.verdict:
                check_verdict(json.loads(Path(args.verdict).read_text()), args.target)
            print('PASS: release evidence identity and verdict')
        else:
            role = args.role or ('preparation' if args.allow_pending else 'tag-publication')
            if args.allow_pending and role != 'preparation':
                raise ValueError('--allow-pending cannot be combined with a strict publication role')
            record = load_record(args.record, strict=role != 'preparation')
            if role == 'preparation':
                notes = release_notes(Path(args.notes).read_text(), args.tag, True, record=record, role=role)
            else:
                if not all((args.evidence_root, args.receipt, args.workspace, args.sha, args.target, args.run_id, args.run_attempt)):
                    raise ValueError('strict notes requires full evidence, receipt and source/run identity')
                if args.tag != args.target:
                    raise ValueError('tag and release target disagree')
                check_workspace(args.workspace, args.sha)
                notes = validate_publication(args.record, args.notes, args.evidence_root, read_json(args.receipt),
                    {'sha': args.sha, 'target': args.target, 'role': role, 'run_id': args.run_id, 'run_attempt': args.run_attempt})
            Path(args.output).write_text(notes)
    except (ValueError, OSError, TypeError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
