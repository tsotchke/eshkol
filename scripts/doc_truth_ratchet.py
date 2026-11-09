#!/usr/bin/env python3
"""Doc-truth ratchet: rank the project's documentation claims by verification
status and hold every document to a per-file quota.

Claims come from ICC's typed documentation claims (``icc doc-typed-claims``)
over ``docs/``.  ICC extracts each checkable claim (file paths and line
references, line counts, versions, sizes, ratios, table rows, call arities)
and grounds it against the indexed source tree.  This script maps ICC's
statuses onto four public verification levels, ordered from strongest
support to weakest:

    verified            ICC grounded the claim against the source tree
    unverified          ICC found no source of truth to check it against
    table-contradicted  a table row disagrees with the source it describes
    contradicted        the source tree contradicts the claim

For every document it counts claims per level and derives two figures:
``contradicted`` (both contradicted levels) and ``unsupported`` (everything
not verified).  Documents are ranked by contradicted, then unsupported.

Phase 1 (v1.4.0) records the counts as the quota
(``docs/reports/doc_truth/quota.json``) and a generated summary
(``docs/reports/DOC_TRUTH_RANKING.md``); the ``doc_truth_ratchet_phase1``
CTest recomputes the ranking into the build directory and reports how each
document compares with its quota without failing.  Phase 2 is the same check
run with ``--enforce``; see the summary page for the exact switch-over.

Commands
    collect      run ICC in a private state directory and write claims JSON
    rank         turn a claims JSON into the ranking JSON
    check        collect + rank + compare against the quota (CTest entry point)
    update-quota write quota.json and the summary page from a ranking JSON

ICC never touches the caller's own ICC registry: ``collect`` points
ICC_STATE_DIR and ICC_ARTIFACT_ROOT at a work directory (the build directory
under CTest), registers the source tree there under a private name and
indexes it.  Nothing is written into the source tree.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
LEVELS = ("verified", "unverified", "table-contradicted", "contradicted")
# ICC's external status vocabulary for a claim the source tree contradicts.
# It is the one place that spelling appears; everything here reports the
# public level "contradicted".
ICC_CONTRADICTED_STATUS = "wrong"
ICC_STATUS_TO_LEVEL = {
    "grounded": "verified",
    "unknown": "unverified",
    "table_row_mismatch": "table-contradicted",
    ICC_CONTRADICTED_STATUS: "contradicted",
}
DEFAULT_PREFIX = "docs/"
# The ratchet's own generated outputs describe the ranking; counting their
# claims would make every regeneration move the numbers it reports.
EXCLUDED = ("docs/reports/DOC_TRUTH_RANKING.md", "docs/reports/doc_truth/")
SUMMARY_TOP = 40


def in_scope(path: str, prefix: str) -> bool:
    return path.startswith(prefix) and not any(path.startswith(e) for e in EXCLUDED)


def level_for(status: str) -> str:
    if status in ICC_STATUS_TO_LEVEL:
        return ICC_STATUS_TO_LEVEL[status]
    # A status this script does not know yet is never counted as support.
    return "unverified"


def find_icc(explicit: str | None, source: Path) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("ESHKOL_ICC_BIN"):
        candidates.append(Path(os.environ["ESHKOL_ICC_BIN"]))
    found = shutil.which("icc")
    if found:
        candidates.append(Path(found))
    home = Path.home()
    candidates += [
        source.parent / "infinite_context_coder" / "bin" / "icc",
        home / "infinite_context_coder" / "bin" / "icc",
        home / "Desktop" / "infinite_context_coder" / "bin" / "icc",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


def icc_version(icc: Path) -> str:
    """Git revision of the ICC checkout that provides @p icc (searching up;
    a launcher script that execs "$ICC_HOME/bin/icc" is followed)."""
    roots = [icc.resolve().parent, *icc.resolve().parents]
    try:
        import re
        text = icc.read_text(encoding="utf-8", errors="replace")[:4096]
        m = re.search(r'ICC_HOME="\$\{ICC_HOME:-([^}"]+)\}"', text)
        home = os.environ.get("ICC_HOME") or (m.group(1) if m else None)
        if home:
            roots.insert(0, Path(home))
    except OSError:
        pass
    for root in roots:
        if not (root / ".git").exists():
            continue
        try:
            out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
                                 text=True, capture_output=True, timeout=30, check=False)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        break
    return "unknown"


def collect(source: Path, work: Path, icc: Path, prefix: str, log) -> Path:
    """Index @p source in a private ICC state under @p work; return claims path."""
    work.mkdir(parents=True, exist_ok=True)
    state = work / "icc_state"
    artifacts = work / "icc_artifacts"
    state.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["ICC_STATE_DIR"] = str(state)
    env["ICC_ARTIFACT_ROOT"] = str(artifacts)
    name = "doctruth_" + hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:12]

    def run(args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        log(f"$ icc {' '.join(args)}")
        return subprocess.run([str(icc), *args], text=True, capture_output=True,
                              env=env, timeout=timeout, check=False)

    listing = run(["list"], 120)
    if name not in listing.stdout:
        reg = run(["register", "--name", name, "--path", str(source.resolve()),
                   "--skip-dir", "build", "--skip-dir", ".scratch", "--skip-dir", ".worktrees",
                   "--skip-dir", ".git"], 300)
        if reg.returncode != 0:
            raise RuntimeError(f"icc register failed: {reg.stderr.strip()[-400:]}")
    # A full reindex every time: ICC refuses authority-bearing commands on an
    # index whose slower stores lag the tree, which an incremental pass leaves.
    idx = run(["reindex", "--repo", name, "--full"], 3600)
    if idx.returncode != 0:
        detail = idx.stderr.strip() or idx.stdout.strip()
        raise RuntimeError(f"icc reindex failed: {detail[-400:]}")
    out = run(["doc-typed-claims", "--repo", name, "--path-prefix", prefix, "--format", "json",
               "--claim-limit", "1000000", "--finding-limit", "1000000"], 1800)
    claims_path = work / "claims.json"
    claims_path.write_text(out.stdout, encoding="utf-8")
    try:
        payload = json.loads(out.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"icc doc-typed-claims returned no JSON: {exc}") from exc
    if not payload.get("ok"):
        raise RuntimeError(f"icc doc-typed-claims refused: {str(payload.get('error'))[:400]}")
    return claims_path


def rank(claims_payload: dict, prefix: str, icc_rev: str) -> dict:
    files: dict[str, dict] = {}
    per_type: dict[str, dict[str, int]] = {}
    for doc in claims_payload.get("docs", []):
        path = doc.get("path", "")
        if in_scope(path, prefix):
            files.setdefault(path, {lvl: 0 for lvl in LEVELS})
    for claim in claims_payload.get("claims", []):
        path = claim.get("location", {}).get("path", "")
        if not in_scope(path, prefix):
            continue
        lvl = level_for(claim.get("status", ""))
        files.setdefault(path, {l: 0 for l in LEVELS})[lvl] += 1
        t = claim.get("claim_type", "other")
        per_type.setdefault(t, {l: 0 for l in LEVELS})[lvl] += 1
    ranked = []
    for path, counts in files.items():
        total = sum(counts.values())
        contradicted = counts["contradicted"] + counts["table-contradicted"]
        unsupported = total - counts["verified"]
        ranked.append({"path": path, "claims": total, **counts,
                       "contradicted_total": contradicted, "unsupported": unsupported})
    ranked.sort(key=lambda r: (-r["contradicted_total"], -r["unsupported"], r["path"]))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    totals = {lvl: sum(r[lvl] for r in ranked) for lvl in LEVELS}
    totals["claims"] = sum(r["claims"] for r in ranked)
    totals["contradicted_total"] = sum(r["contradicted_total"] for r in ranked)
    totals["unsupported"] = sum(r["unsupported"] for r in ranked)
    findings: dict[str, int] = {}
    for f in claims_payload.get("findings", []):
        p = [loc.get("path", "") for v in f.get("values", []) for loc in v.get("locations", [])]
        if p and not any(in_scope(x, prefix) for x in p):
            continue
        findings[f.get("type", "other")] = findings.get(f.get("type", "other"), 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "eshkol_doc_truth_ranking",
        "claim_source": {"tool": "icc doc-typed-claims", "icc_revision": icc_rev,
                         "icc_schema_version": claims_payload.get("schema_version"),
                         "path_prefix": prefix},
        "levels": list(LEVELS),
        "totals": totals,
        "claim_types": dict(sorted(per_type.items())),
        "findings": dict(sorted(findings.items())),
        "documents": ranked,
    }


def release_critical(paths: list[str], patterns: list[str]) -> list[str]:
    return sorted(p for p in paths if any(fnmatch.fnmatch(p, pat) for pat in patterns))


def compare(ranking: dict, quota: dict) -> dict:
    limits = quota.get("documents", {})
    over, under, new, removed = [], [], [], []
    seen = set()
    for r in ranking["documents"]:
        path = r["path"]
        seen.add(path)
        q = limits.get(path)
        if q is None:
            if r["contradicted_total"] or r["unsupported"]:
                new.append({"path": path, "contradicted_total": r["contradicted_total"],
                            "unsupported": r["unsupported"]})
            continue
        d_con = r["contradicted_total"] - q["contradicted_total"]
        d_uns = r["unsupported"] - q["unsupported"]
        entry = {"path": path, "contradicted_total": r["contradicted_total"],
                 "quota_contradicted_total": q["contradicted_total"],
                 "unsupported": r["unsupported"], "quota_unsupported": q["unsupported"]}
        if d_con > 0 or d_uns > 0:
            over.append(entry)
        elif d_con < 0 or d_uns < 0:
            under.append(entry)
    for path in limits:
        if path not in seen:
            removed.append(path)
    crit = quota.get("release_critical", [])
    crit_now = {r["path"]: r for r in ranking["documents"]
                if any(fnmatch.fnmatch(r["path"], pat) for pat in crit)}
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "eshkol_doc_truth_comparison",
        "phase": quota.get("phase", 1),
        "quota_claim_source": quota.get("claim_source", {}),
        "ranking_claim_source": ranking.get("claim_source", {}),
        "over_quota": over,
        "under_quota": under,
        "new_documents": new,
        "documents_no_longer_present": sorted(removed),
        "release_critical": {
            "documents": len(crit_now),
            "contradicted_total": sum(r["contradicted_total"] for r in crit_now.values()),
            "unsupported": sum(r["unsupported"] for r in crit_now.values()),
        },
        "totals_now": ranking["totals"],
        "totals_quota": quota.get("totals", {}),
    }


DEFAULT_RELEASE_CRITICAL = [
    "docs/QUICKSTART.md", "docs/FAQ.md", "docs/API_REFERENCE.md",
    "docs/ESHKOL_LANGUAGE_GUIDE.md", "docs/ESHKOL_QUICK_REFERENCE.md",
    "docs/COMPLETE_LANGUAGE_SPECIFICATION.md", "docs/FEATURE_MATRIX.md",
    "docs/KNOWN_ISSUES.md", "docs/BUILD_INTEGRATION.md", "docs/UPGRADING.md",
    "docs/reference/*", "docs/guide/*",
]


def write_quota(ranking: dict, quota_path: Path, summary_path: Path, previous: dict | None) -> None:
    crit = (previous or {}).get("release_critical", DEFAULT_RELEASE_CRITICAL)
    docs = {r["path"]: {"contradicted_total": r["contradicted_total"], "unsupported": r["unsupported"]}
            for r in ranking["documents"]}
    quota = {
        "schema_version": SCHEMA_VERSION,
        "kind": "eshkol_doc_truth_quota",
        "phase": 1,
        "claim_source": ranking["claim_source"],
        "release_critical": crit,
        "totals": {k: ranking["totals"][k] for k in ("claims", "contradicted_total", "unsupported")},
        "documents": dict(sorted(docs.items())),
    }
    quota_path.parent.mkdir(parents=True, exist_ok=True)
    quota_path.write_text(json.dumps(quota, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    summary_path.write_text(render_summary(ranking, crit), encoding="utf-8")


def render_summary(ranking: dict, crit: list[str]) -> str:
    t = ranking["totals"]
    src = ranking["claim_source"]
    docs = ranking["documents"]
    crit_docs = [r for r in docs if any(fnmatch.fnmatch(r["path"], p) for p in crit)]
    lines = [
        "# Documentation claims ranked by verification status",
        "",
        "Generated by `scripts/doc_truth_ratchet.py update-quota`; do not edit by hand.",
        f"Claims: `icc doc-typed-claims` over `{src['path_prefix']}` "
        f"(ICC revision `{src['icc_revision']}`). The per-document quota is",
        "[`docs/reports/doc_truth/quota.json`](doc_truth/quota.json).",
        "",
        "## Levels",
        "",
        "| Level | Meaning |",
        "|---|---|",
        "| verified | ICC grounded the claim against the source tree |",
        "| unverified | no source of truth exists for ICC to check the claim against |",
        "| table-contradicted | a table row disagrees with the source it describes |",
        "| contradicted | the source tree disagrees with the claim |",
        "",
        "`contradicted` below counts both contradicted levels; `unsupported` counts",
        "every claim that is not verified.",
        "",
        "## Totals",
        "",
        f"- Documents: {len(docs)}",
        f"- Claims: {t['claims']} (verified {t['verified']}, unverified {t['unverified']}, "
        f"table-contradicted {t['table-contradicted']}, contradicted {t['contradicted']})",
        f"- Release-critical documents: {len(crit_docs)}, contradicted "
        f"{sum(r['contradicted_total'] for r in crit_docs)}, unsupported "
        f"{sum(r['unsupported'] for r in crit_docs)}",
        "",
        "### By claim type",
        "",
        "| Claim type | verified | unverified | table-contradicted | contradicted |",
        "|---|---:|---:|---:|---:|",
    ]
    for ctype, c in ranking["claim_types"].items():
        lines.append(f"| {ctype} | {c['verified']} | {c['unverified']} | {c['table-contradicted']} | {c['contradicted']} |")
    lines += [
        "",
        f"## Ranking (top {SUMMARY_TOP} documents by contradicted, then unsupported)",
        "",
        "| Rank | Document | Claims | verified | contradicted | unsupported |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in docs[:SUMMARY_TOP]:
        lines.append(f"| {r['rank']} | `{r['path']}` | {r['claims']} | {r['verified']} | "
                     f"{r['contradicted_total']} | {r['unsupported']} |")
    lines += [
        "",
        "## Release-critical documents",
        "",
        "Patterns: " + ", ".join(f"`{p}`" for p in crit),
        "",
        "| Document | Claims | verified | contradicted | unsupported |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in sorted(crit_docs, key=lambda r: (-r["contradicted_total"], -r["unsupported"], r["path"])):
        lines.append(f"| `{r['path']}` | {r['claims']} | {r['verified']} | "
                     f"{r['contradicted_total']} | {r['unsupported']} |")
    lines += [
        "",
        "## How the ratchet works",
        "",
        "Phase 1 (v1.4.0, this page): the `doc_truth_ratchet_phase1` CTest re-indexes the",
        "tree in a private ICC state inside the build directory, recomputes this ranking",
        "into `<build>/doc_truth/ranking.json`, compares every document with its quota in",
        "`<build>/doc_truth/comparison.json`, and prints the result. It does not fail: a",
        "document above its quota, a new document with unsupported claims, or a host without",
        "ICC is reported, not rejected.",
        "",
        "When a document's counts fall, lower its quota in the same change:",
        "",
        "```sh",
        "python3 scripts/doc_truth_ratchet.py check --source . --build-dir build",
        "python3 scripts/doc_truth_ratchet.py update-quota --ranking build/doc_truth/ranking.json",
        "```",
        "",
        "Phase 2 makes the same check release-blocking:",
        "",
        "1. The release build runs the check with `--enforce` (CMake option",
        "   `ESHKOL_DOC_TRUTH_ENFORCE=ON`, set by the release workflow). With it, any document",
        "   above its quota, any new document with contradicted claims, and a host where",
        "   ICC cannot recompute the ranking fail the test.",
        "2. Quotas only move down. `update-quota` refuses to raise a document's quota unless",
        "   `--allow-increase` is given, which a release build never passes.",
        "3. Release-critical documents (the patterns above) must reach zero contradicted",
        "   claims before the switch; the v1.5.1 roadmap item turns that into zero",
        "   unsupported claims in those documents.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_col = sub.add_parser("collect")
    p_col.add_argument("--source", required=True, type=Path)
    p_col.add_argument("--work", required=True, type=Path)
    p_col.add_argument("--icc-bin")
    p_col.add_argument("--path-prefix", default=DEFAULT_PREFIX)
    p_rank = sub.add_parser("rank")
    p_rank.add_argument("--claims", required=True, type=Path)
    p_rank.add_argument("--out", required=True, type=Path)
    p_rank.add_argument("--path-prefix", default=DEFAULT_PREFIX)
    p_rank.add_argument("--icc-revision", default="unknown")
    p_chk = sub.add_parser("check")
    p_chk.add_argument("--source", required=True, type=Path)
    p_chk.add_argument("--build-dir", required=True, type=Path)
    p_chk.add_argument("--quota", type=Path)
    p_chk.add_argument("--icc-bin")
    p_chk.add_argument("--claims", type=Path, help="use this claims JSON instead of running ICC")
    p_chk.add_argument("--enforce", action="store_true", help="phase 2: fail on any quota excess")
    p_upd = sub.add_parser("update-quota")
    p_upd.add_argument("--ranking", required=True, type=Path)
    p_upd.add_argument("--quota", type=Path)
    p_upd.add_argument("--summary", type=Path)
    p_upd.add_argument("--allow-increase", action="store_true")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent.parent

    def log(msg: str) -> None:
        print(msg, flush=True)

    if args.cmd == "collect":
        icc = find_icc(args.icc_bin, args.source)
        if not icc:
            print("ICC not found", file=sys.stderr)
            return 2
        print(collect(args.source, args.work, icc, args.path_prefix, log))
        return 0
    if args.cmd == "rank":
        payload = json.loads(args.claims.read_text(encoding="utf-8"))
        result = rank(payload, args.path_prefix, args.icc_revision)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        return 0
    if args.cmd == "update-quota":
        quota_path = args.quota or here / "docs" / "reports" / "doc_truth" / "quota.json"
        summary = args.summary or here / "docs" / "reports" / "DOC_TRUTH_RANKING.md"
        ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
        previous = json.loads(quota_path.read_text(encoding="utf-8")) if quota_path.exists() else None
        if previous and not args.allow_increase:
            raised = compare(ranking, previous)["over_quota"]
            if raised:
                print(f"refusing to raise {len(raised)} quota(s); pass --allow-increase to accept", file=sys.stderr)
                for e in raised[:20]:
                    print(f"  {e['path']}: contradicted {e['quota_contradicted_total']} -> {e['contradicted_total']}, "
                          f"unsupported {e['quota_unsupported']} -> {e['unsupported']}", file=sys.stderr)
                return 1
        write_quota(ranking, quota_path, summary, previous)
        print(f"wrote {quota_path} and {summary}")
        return 0

    # check
    out_dir = args.build_dir / "doc_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    quota_path = args.quota or args.source / "docs" / "reports" / "doc_truth" / "quota.json"
    quota = json.loads(quota_path.read_text(encoding="utf-8"))
    icc_rev = "unknown"
    try:
        if args.claims:
            claims_path = args.claims
        else:
            icc = find_icc(args.icc_bin, args.source)
            if not icc:
                raise RuntimeError("ICC is not available on this host (set ESHKOL_ICC_BIN)")
            icc_rev = icc_version(icc)
            claims_path = collect(args.source, out_dir / "work", icc, DEFAULT_PREFIX, log)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        status = {"schema_version": SCHEMA_VERSION, "kind": "eshkol_doc_truth_comparison",
                  "recomputed": False, "reason": str(exc)}
        (out_dir / "comparison.json").write_text(json.dumps(status, indent=1) + "\n", encoding="utf-8")
        print(f"DOC-TRUTH NOT RECOMPUTED: {exc}")
        if args.enforce:
            print("FAIL: phase 2 requires the ranking to be recomputed")
            return 1
        return 0
    payload = json.loads(Path(claims_path).read_text(encoding="utf-8"))
    ranking = rank(payload, DEFAULT_PREFIX, icc_rev)
    (out_dir / "ranking.json").write_text(json.dumps(ranking, indent=1) + "\n", encoding="utf-8")
    cmp = compare(ranking, quota)
    cmp["recomputed"] = True
    (out_dir / "comparison.json").write_text(json.dumps(cmp, indent=1) + "\n", encoding="utf-8")
    t, q = cmp["totals_now"], cmp["totals_quota"]
    print(f"DOC-TRUTH REPORT (phase {cmp['phase']}, {'enforced' if args.enforce else 'not release-blocking'})")
    print(f"  documents ranked: {len(ranking['documents'])}; claims {t['claims']} "
          f"(quota baseline {q.get('claims', '?')})")
    print(f"  contradicted: {t['contradicted_total']} (quota {q.get('contradicted_total', '?')}); "
          f"unsupported: {t['unsupported']} (quota {q.get('unsupported', '?')})")
    rc = cmp["release_critical"]
    print(f"  release-critical: {rc['documents']} documents, contradicted {rc['contradicted_total']}, "
          f"unsupported {rc['unsupported']}")
    print(f"  over quota: {len(cmp['over_quota'])}; under quota (quota can be lowered): "
          f"{len(cmp['under_quota'])}; new documents: {len(cmp['new_documents'])}")
    for e in cmp["over_quota"][:15]:
        print(f"    above quota: {e['path']}: contradicted {e['contradicted_total']}/{e['quota_contradicted_total']}, "
              f"unsupported {e['unsupported']}/{e['quota_unsupported']}")
    if cmp["quota_claim_source"].get("icc_revision") not in (None, icc_rev):
        print(f"  note: quota recorded with ICC {cmp['quota_claim_source'].get('icc_revision')}, "
              f"this run used ICC {icc_rev}")
    print(f"  ranking: {out_dir / 'ranking.json'}")
    print(f"  comparison: {out_dir / 'comparison.json'}")
    if args.enforce and (cmp["over_quota"] or any(e["contradicted_total"] for e in cmp["new_documents"])):
        print("FAIL: documents above their doc-truth quota")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
