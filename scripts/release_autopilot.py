#!/usr/bin/env python3
"""One bounded, restartable step of an explicitly authorized Eshkol release.

Uses the existing Python/gh control plane: Eshkol's compiler is the product
under test, so release scheduling must remain available while it is rebuilt.
No administrator merges, force pushes, weakened checks, or overwritten tags.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request


class Wait(RuntimeError):
    """An unmet condition; re-evaluate on the next scheduled invocation."""


def utcnow():
    return datetime.now(timezone.utc)


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("release times must include a timezone")
    return result


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require_checks(checks, required):
    """An absent, skipped, stale, or failed required check cannot authorize."""
    latest = {}
    for check in checks:
        name = check.get("name") or check.get("context")
        if name not in latest or check.get("startedAt", "") > latest[name].get("startedAt", ""):
            latest[name] = check
    pending = []
    for name in required:
        c = latest.get(name, {})
        if c.get("conclusion") != "SUCCESS" and c.get("state") != "SUCCESS":
            pending.append(name)
    for name, c in latest.items():
        conclusion = c.get("conclusion") or c.get("state")
        if conclusion not in ("SUCCESS", "SKIPPED", "NEUTRAL"):
            pending.append(name)
    if pending:
        raise Wait("Checks not green: " + ", ".join(sorted(set(pending))))


def require_receipt(receipt, sha, run):
    if (receipt.get("schema") != "eshkol.release-readiness.v1"
            or receipt.get("sha") != sha
            or str(receipt.get("run_id")) != str(run["id"])
            or str(receipt.get("run_attempt")) != str(run["run_attempt"])
            or receipt.get("target") != "v1.3.5-evolve"
            or receipt.get("status") != "ready"
            or type(receipt.get("score")) not in (int, float)
            or receipt["score"] != 100):
        raise Wait("Missing or mismatched exact-commit ready/100 receipt")


def require_window(config, now):
    if now < timestamp(config["not_before"]):
        raise Wait("Validated; waiting until " + config["not_before"] + " to tag")
    if now >= timestamp(config["expires_at"]):
        raise Wait("Authorized publication window expired; no tag will be created")


def pending_hold(presence_root, now):
    records = []
    for stream in ("dispatch", "thoughts"):
        path = Path(presence_root) / (stream + ".jsonl")
        with path.open() as source:
            records.extend(json.loads(line) for line in deque(source, maxlen=400) if line.strip())
    tasks = {}
    for r in sorted(records, key=lambda r: r.get("ts", "")):
        if r.get("task_id") and r.get("stream") == "dispatch":
            tasks[r["task_id"]] = r
    candidates = list(tasks.values()) + [r for r in records if r.get("kind") == "veto"]
    for r in candidates:
        scope = " ".join(str(r.get(k) or "") for k in ("repo", "task_id", "why", "text"))
        if (r.get("kind") in ("hold", "veto")
                and any(s in scope.lower() for s in ("eshkol", "v1.3.5", "v135"))
                and (now - timestamp(r["ts"])).total_seconds() < 86400):
            return scope
    return None


class Release:
    def __init__(self, config, state_dir, execute=False):
        self.config = config
        self.directory = Path(state_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.execute = execute
        self.repo = config["repo"]
        if self.repo != "tsotchke/eshkol" or config["tag"] != "v1.3.5-evolve":
            raise ValueError("This authorization is limited to tsotchke/eshkol v1.3.5-evolve")
        self.checkout = Path(config["checkout"])

    def command(self, args, *, cwd=None, mutate=False, timeout=120):
        if mutate and not self.execute:
            raise Wait("Preview: would execute " + " ".join(args))
        r = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)
        if r.returncode:
            raise Wait("Command failed: " + " ".join(args[:6]) + "\n" + (r.stderr or r.stdout)[-2500:])
        return r.stdout

    def gh(self, *args, mutate=False):
        return self.command([self.config["gh"], *args], mutate=mutate)

    def api(self, endpoint, *, data=None):
        args = ["api", f"repos/{self.repo}/{endpoint}"]
        if data is not None:
            args += ["--method", "POST"]
            for key, value in data.items():
                args += ["-f", f"{key}={value}"]
        result = self.gh(*args, mutate=data is not None)
        return json.loads(result) if result.strip() else None

    def pr(self, number):
        return json.loads(self.gh("pr", "view", str(number), "--repo", self.repo, "--json",
            "number,state,headRefName,headRefOid,baseRefName,isDraft,mergeStateStatus,mergeCommit,statusCheckRollup"))

    def git(self, *args, mutate=False):
        return self.command(["git", *args], cwd=self.checkout, mutate=mutate).strip()

    def required(self):
        data = self.api("branches/master/protection/required_status_checks")
        required = set(data["contexts"])
        if not set(self.config["required_contexts"]).issubset(required):
            raise Wait("Branch protection is weaker than the approved 16-check baseline")
        return required

    def lanes_ready(self):
        for lane in self.config["lanes"]:
            prs = json.loads(self.gh("pr", "list", "--repo", self.repo, "--head", lane["branch"],
                "--state", "all", "--limit", "10", "--json", "number,state,headRefOid,baseRefName"))
            if not prs or prs[0]["state"] != "MERGED":
                raise Wait("Hardening still pending: " + lane["branch"])
            if prs[0]["baseRefName"] not in ("master", "integration/astra-v135", "release/v135-cut-final"):
                raise Wait("Hardening was merged outside the release branches: " + lane["branch"])
            refs = self.api("git/matching-refs/heads/" + lane["branch"])
            exact = [r for r in refs if r["ref"] == "refs/heads/" + lane["branch"]]
            if exact and exact[0]["object"]["sha"] != prs[0]["headRefOid"]:
                raise Wait("Hardening branch has changes after its merged PR: " + lane["branch"])
            path = Path(lane["worktree"]) if lane.get("worktree") else None
            if path is not None and path.is_dir():
                branch = self.command(["git", "branch", "--show-current"], cwd=path).strip()
                if branch == lane["branch"]:
                    dirty = self.command(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=path)
                    head = self.command(["git", "rev-parse", "HEAD"], cwd=path).strip()
                    if dirty or head != prs[0]["headRefOid"]:
                        raise Wait("Hardening worktree has unpublished changes: " + lane["branch"])
            self.state.setdefault("lane_prs", {})[lane["branch"]] = prs[0]["number"]

    def ensure_checkout(self, ref):
        if not self.checkout.exists():
            self.command(["git", "-C", self.config["source_repo"], "worktree", "add", "--detach",
                str(self.checkout), ref], mutate=True)
        if self.git("status", "--porcelain", "--untracked-files=no"):
            raise Wait("Automation checkout has changes; preserving it for investigation")

    def require_lane_inclusion(self, *refs):
        for branch, number in self.state.get("lane_prs", {}).items():
            commit = self.pr(number)["mergeCommit"]["oid"]
            for ref in refs:
                result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, ref],
                    cwd=self.checkout, capture_output=True, text=True)
                if result.returncode == 0:
                    break
            else:
                raise Wait("Merged hardening is absent from the release source: " + branch)

    def master_ci(self, sha):
        runs = self.api("actions/workflows/ci.yml/runs?branch=master&per_page=30")["workflow_runs"]
        if any(r["head_sha"] == sha for r in runs):
            return
        if self.state.get("ci_dispatched_sha") == sha:
            raise Wait("Exact-commit master CI dispatch already requested")
        self.gh("workflow", "run", "ci.yml", "--repo", self.repo, "--ref", "master", "-f",
            "reason=Authorized v1.3.5 release validation after a documentation-only merge", mutate=True)
        self.state["ci_dispatched_sha"] = sha
        raise Wait("Requested exact-commit master CI; documentation-only pushes may not trigger it")

    def refresh_cut(self, pr):
        """Merge finished upstream work without rewriting anyone's branch."""
        self.ensure_checkout("origin/" + pr["headRefName"])
        self.git("fetch", "origin", "master", "integration/astra-v135", pr["headRefName"], mutate=True)
        self.git("switch", "--detach", "origin/" + pr["headRefName"], mutate=True)
        before = self.git("rev-parse", "HEAD")
        try:
            for ref in ("origin/master", "origin/integration/astra-v135"):
                self.git("merge", "--no-edit", ref, mutate=True)
        except Wait:
            # Abort only the merge this controller just attempted, never other work.
            if self.git("diff", "--name-only", "--diff-filter=U"):
                conflicts = self.git("diff", "--name-only", "--diff-filter=U")
                self.git("merge", "--abort", mutate=True)
                raise Wait("Cut merge conflicts need resolution: " + conflicts)
            raise
        after = self.git("rev-parse", "HEAD")
        self.require_lane_inclusion(after)
        if after != before:
            self.git("push", "origin", "HEAD:refs/heads/" + pr["headRefName"], mutate=True)
            raise Wait("Cut refreshed with upstream commits; waiting for its new checks")

    def docs_ready(self, sha):
        if self.state.get("docs_checked_sha") == sha:
            return
        checks = [
            ["scripts/gen_api_docs.py", "--check", "--no-trace"],
            ["scripts/gen_language_surface.py", "--check"],
            ["scripts/check_surface_counts.py", "--no-trace"],
            ["scripts/check_ledger_integrity.py", "--no-trace"],
            ["scripts/verify_site_release.py"],
            ["tests/toolchain/test_release_readiness_guard.py"],
        ]
        for args in checks:
            result = self.command([self.config["python"], *args], cwd=self.checkout, timeout=300)
            (self.directory / (sha[:12] + "-" + Path(args[0]).stem + ".log")).write_text(result)
        scan = subprocess.run(["git", "grep", "-nE", "^(<<<<<<< |>>>>>>> |=======$)", sha,
            "--", ".", ":!tests/*"], cwd=self.checkout, capture_output=True, text=True)
        if scan.returncode != 1:
            raise Wait("Conflict-marker sweep did not pass: " + scan.stdout[-1500:])
        self.state["docs_checked_sha"] = sha

    def proof(self, branch, sha):
        runs = self.api("actions/workflows/release.yml/runs?per_page=50")["workflow_runs"]
        active = [r for r in runs if r["status"] != "completed"]
        matching = [r for r in runs if r["head_sha"] == sha and r["event"] == "workflow_dispatch"]
        if not matching:
            if active:
                raise Wait("A release run is active; no duplicate dispatch")
            workflow = self.git("show", sha + ":.github/workflows/release.yml")
            if "strict_readiness:" not in workflow or "release-readiness-receipt-" not in workflow:
                raise Wait("Strict readiness receipt workflow has not reached the release source")
            if self.state.get("dispatched_sha") == sha:
                raise Wait("Dispatch already requested; waiting for GitHub to list it")
            self.gh("workflow", "run", "release.yml", "--repo", self.repo, "--ref", branch,
                "-f", "candidate_tag=" + self.config["tag"], "-f", "strict_readiness=true", mutate=True)
            self.state["dispatched_sha"] = sha
            raise Wait("Dispatched strict release validation for " + sha)
        run = matching[0]
        if run["status"] != "completed":
            raise Wait(f"Release validation running: {run['html_url']}")
        if run["conclusion"] != "success":
            raise Wait(f"Release validation {run['conclusion']}: {run['html_url']} (no blind retry)")
        artifacts = self.api(f"actions/runs/{run['id']}/artifacts?per_page=100")["artifacts"]
        names = {a["name"] for a in artifacts if not a["expired"]}
        receipt_name = "release-readiness-receipt-" + sha
        if receipt_name not in names or "release-dry-run-" + sha not in names:
            raise Wait("Successful run lacks strict readiness receipt or validated asset set")
        folder = self.directory / ("receipt-" + str(run["id"]) + "-" + str(run["run_attempt"]))
        receipt_path = folder / "release-readiness-receipt.json"
        if not receipt_path.exists():
            self.gh("run", "download", str(run["id"]), "--repo", self.repo, "--name", receipt_name,
                "--dir", str(folder), mutate=True)
        receipt = json.loads(receipt_path.read_text())
        require_receipt(receipt, sha, run)
        self.state["proof"] = {"sha": sha, "run_id": run["id"], "url": run["html_url"]}
        return run

    def finalize_notes(self, pr, proof):
        path = self.checkout / "RELEASE_NOTES.md"
        original = path.read_text()
        pending = "RELEASE_EVIDENCE_PENDING" in original.split("\n---\n", 1)[0]
        status = re.search(r"^\*\*Status:\*\*.*$", original.split("\n---\n", 1)[0], re.MULTILINE)
        candidate = status and any(word in status.group().lower() for word in ("candidate", "pending"))
        if not pending and not candidate:
            return
        if not self.execute:
            raise Wait("Preview: would finalize notes with successful readiness evidence")
        current, separator, prior = original.partition("\n---\n")
        if status:
            historical = re.search(r"The [^.]*measurements[^\n]*", status.group())
            replacement = "**Status:** validated release; publication follows the tagged-commit checks."
            if historical:
                replacement += "\n\n" + historical.group()
            current = current.replace(status.group(), replacement, 1)
        current = re.sub(r"<!--\s*RELEASE_EVIDENCE_PENDING\s*-->", "", current)
        if "RELEASE_EVIDENCE_PENDING" in current:
            raise Wait("Unrecognized pending-evidence block requires an editorial update")
        current += ("\n\nRelease-candidate verification: [strict readiness and asset validation]("
            + proof["html_url"] + "). The tagged commit is independently revalidated before publication.\n")
        path.write_text(current + separator + prior)
        self.git("add", "--", "RELEASE_NOTES.md", mutate=True)
        self.git("commit", "--only", "-m", "docs: record verified v1.3.5 release-candidate evidence", "--",
            "RELEASE_NOTES.md", mutate=True)
        self.git("push", "origin", "HEAD:refs/heads/" + pr["headRefName"], mutate=True)
        raise Wait("Release notes finalized; validating the updated commit before merge")

    def ensure_homebrew(self):
        endpoint = "repos/tsotchke/homebrew-eshkol/contents/Formula/eshkol.rb"
        data = json.loads(self.gh("api", endpoint))
        formula = base64.b64decode(data["content"]).decode()
        url = f"https://github.com/{self.repo}/archive/refs/tags/{self.config['tag']}.tar.gz"
        if not self.execute:
            raise Wait("Preview: would verify the Homebrew archive checksum after publication")
        # The workflow's tap token is optional; use the owner's authorized gh
        # identity if that job skipped. Only the top-level URL/hash may change.
        archive_hash = self.state.get("source_archive_sha256")
        if not archive_hash:
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 512 * 1024 * 1024:
                        raise Wait("Homebrew source archive exceeds the bounded download size")
                    digest.update(chunk)
            if not size:
                raise Wait("Homebrew source archive is empty")
            archive_hash = digest.hexdigest()
            self.state["source_archive_sha256"] = archive_hash
        if f'  url "{url}"' in formula and f'  sha256 "{archive_hash}"' in formula:
            self.state["homebrew"] = "verified"
            return
        replaced, urls = re.subn(r'^  url "https://github\.com/tsotchke/eshkol/archive/refs/tags/[^"\n]+"$',
            f'  url "{url}"', formula, flags=re.MULTILINE)
        replaced, hashes = re.subn(r'^  sha256 "[0-9a-f]{64}"$', f'  sha256 "{archive_hash}"',
            replaced, flags=re.MULTILINE)
        if urls != 1 or hashes != 1:
            raise Wait("Unexpected Homebrew formula structure; preserving the existing formula")
        self.gh("api", endpoint, "--method", "PUT", "-f", "message=eshkol " + self.config["tag"],
            "-f", "sha=" + data["sha"], "-f", "content=" + base64.b64encode(replaced.encode()).decode(), mutate=True)
        raise Wait("Homebrew formula updated with the release archive checksum; verifying next tick")

    def verify_published(self, sha):
        tag = self.config["tag"]
        release = self.api("releases/tags/" + tag)
        expected = {f"eshkol-{tag}-{platform}.{ext}" for platform, ext in self.config["assets"]}
        expected.add("SHA256SUMS.txt")
        actual = {a["name"] for a in release["assets"] if a["size"] > 0}
        if release["draft"] or release["prerelease"] or actual != expected:
            raise Wait("Publication asset set is incomplete or still draft")
        runs = self.api("actions/workflows/release.yml/runs?per_page=50")["workflow_runs"]
        tagged = [r for r in runs if r["head_sha"] == sha and r["event"] == "push" and r["head_branch"] == tag]
        if not tagged or tagged[0]["conclusion"] != "success":
            raise Wait("Waiting for successful tagged release workflow, including tap update")
        self.ensure_homebrew()
        self.state["completed"] = True
        self.state["release_url"] = release["html_url"]
        return "Published and verified: " + release["html_url"]

    def step(self):
        if self.state.get("completed"):
            return "Already complete: " + self.state["release_url"]
        if (self.directory / "PAUSE").exists():
            raise Wait("Paused by operator: remove the PAUSE file to resume")
        hold = pending_hold(self.config["presence_root"], utcnow())
        if hold:
            raise Wait("Tsotchke hold/veto: " + hold)
        if self.state.get("tagged_sha"):
            return self.verify_published(self.state["tagged_sha"])
        if utcnow() >= timestamp(self.config["expires_at"]):
            self.state["expired"] = True
            raise Wait("Publication window expired; waiting for renewed scheduling authorization")
        required = self.required()
        self.lanes_ready()
        if self.pr(self.config["candidate_pr"])["state"] != "MERGED":
            raise Wait("Candidate PR has not merged")
        pr = self.pr(self.config["cut_pr"])
        if pr["state"] == "OPEN":
            self.refresh_cut(pr)
            pr = self.pr(self.config["cut_pr"])
            self.docs_ready(pr["headRefOid"])
            if pr["baseRefName"] != "master":
                self.gh("pr", "edit", str(pr["number"]), "--repo", self.repo, "--base", "master", mutate=True)
                raise Wait("Release cut retargeted to master; waiting for required checks")
            if pr["isDraft"]:
                self.gh("pr", "ready", str(pr["number"]), "--repo", self.repo, mutate=True)
                raise Wait("Release cut ready for its complete CI matrix")
            require_checks(pr["statusCheckRollup"], required)
            run = self.proof(pr["headRefName"], pr["headRefOid"])
            self.finalize_notes(pr, run)
            # GitHub enforces base protection; expected head prevents merging a changed PR.
            self.gh("pr", "merge", str(pr["number"]), "--repo", self.repo, "--squash",
                "--match-head-commit", pr["headRefOid"], mutate=True)
            raise Wait("Release cut merged; final master commit must be revalidated")
        if pr["state"] != "MERGED":
            raise Wait("Release cut closed without merging")
        self.ensure_checkout("origin/master")
        self.git("fetch", "origin", "master", "pull/" + str(pr["number"]) + "/head", mutate=True)
        sha = self.git("rev-parse", "origin/master")
        self.git("merge-base", "--is-ancestor", pr["mergeCommit"]["oid"], sha)
        self.require_lane_inclusion(sha, pr["headRefOid"])
        self.git("switch", "--detach", sha, mutate=True)
        self.docs_ready(sha)
        self.master_ci(sha)
        raw = self.api(f"commits/{sha}/check-runs?per_page=100")["check_runs"]
        checks = [{"name": c["name"], "conclusion": (c["conclusion"] or "").upper(),
            "startedAt": c["started_at"] or ""} for c in raw]
        require_checks(checks, required)
        self.proof("master", sha)
        require_window(self.config, utcnow())
        self.command([self.config["python"], "scripts/release_readiness_guard.py", "notes", "--notes",
            "RELEASE_NOTES.md", "--tag", self.config["tag"], "--output", str(self.directory / "release-notes.md")],
            cwd=self.checkout)
        if self.api("branches/master")["commit"]["sha"] != sha:
            raise Wait("Master changed during verification; restarting with its new commit")
        refs = [r for r in self.api("git/matching-refs/tags/" + self.config["tag"])
                if r["ref"] == "refs/tags/" + self.config["tag"]]
        if refs:
            self.git("fetch", "origin", "tag", self.config["tag"], mutate=True)
            if self.git("rev-parse", self.config["tag"] + "^{}") != sha:
                raise Wait("Release tag already exists on a different commit; refusing to move it")
        else:
            existing = self.command(["git", "tag", "--list", self.config["tag"]], cwd=self.checkout).strip()
            if existing:
                if self.git("rev-parse", self.config["tag"] + "^{}") != sha:
                    raise Wait("Local tag differs; refusing to overwrite it")
            else:
                self.git("tag", "-a", self.config["tag"], sha, "-m", "Eshkol " + self.config["tag"], mutate=True)
            self.git("push", "origin", "refs/tags/" + self.config["tag"], mutate=True)
        self.state["tagged_sha"] = sha
        raise Wait("Tag pushed; release workflow is building, validating, and publishing assets")

    def run(self):
        try:
            message = self.step()
            status = "complete" if self.state.get("completed") else "ready"
        except (Wait, OSError, ValueError, subprocess.SubprocessError, KeyError) as exc:
            message, status = str(exc), "waiting"
        result = {"checked_at": utcnow().isoformat(), "status": status, "message": message}
        changed = self.state.get("message") != message or self.state.get("status") != status
        self.state.update(result)
        write_json(self.state_path, self.state)
        if changed:
            with (self.directory / "events.jsonl").open("a") as out:
                out.write(json.dumps(result) + "\n")
            if self.execute and self.config.get("icc_module_dir"):
                try:
                    sys.path.insert(0, self.config["icc_module_dir"])
                    import agent_bus_service
                    agent_bus_service.send_message(self.config["bus_root"],
                        sender="release-autopilot-v135-20260914", recipient="broadcast",
                        text="Release automation: " + message[:1800], sender_agent="codex", sender_repo="eshkol")
                except (OSError, ValueError, ImportError):
                    pass
            elif self.execute and self.config.get("icc_bin"):
                # Advisory status only; its failure must not authorize or cancel a release.
                try:
                    self.command([self.config["icc_bin"], "agent-bus", "--action", "send",
                        "--agent", "codex", "--session", "release-autopilot-v135-20260914",
                        "--bus-root", self.config["bus_root"], "--recipient", "broadcast",
                        "--text", "Release automation: " + message[:1800], "--format", "json"], timeout=15)
                except (Wait, OSError, subprocess.SubprocessError):
                    pass
        print(json.dumps(result))
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Perform the already-authorized release actions")
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('{"status":"waiting","message":"Another controller invocation owns the lock"}')
            return 0
        return Release(json.loads(args.config.read_text()), args.state_dir, args.execute).run()


if __name__ == "__main__":
    sys.exit(main())
