# Eshkol Swarm — v1.3-evolve

Mirrors the QGTL swarm pattern (`~/Desktop/quantum_geometric_tensor/.swarm/`)
adapted for the Eshkol compiler. The swarm coordinates parallel work on
the v1.3-evolve milestone across the mesh
(`~/Desktop/computer_mesh/nodes.json`) via tsotchke-chan task submission.

## Files

| File / dir | Purpose |
|---|---|
| `tasks/ESH-NNNN.json` | One spec per parallelizable v1.3 work item. Schema mirrors QGTL: id, title, status, priority, workstream, owner, goal, file_globs, next_actions, acceptance, blocked_by, verification, commit. |
| `claims/` | Active claims by agents picking up tasks. One file per active claim, named `<agent>__<task-id>.json`. |
| `events.jsonl` | Append-only ledger of swarm events: claim, finish, decision_recorded, evidence_refreshed, etc. |
| `decisions.md` | Architectural decisions locked in for v1.3. Anything that affects multiple tasks goes here so parallel work respects shared design. |
| `risks.md` | Active swarm risks (mesh leakage, parallel-edit conflicts, drift). |
| `status.md` | Generated swarm status. Refreshed by `make eshkol-swarm-status`. |
| `runtime/` | Local supervisor runtime artifacts. gitignored. |

## Operating model

1. **Architectural decisions land in `decisions.md` before parallel work
   starts on items that touch them.** Multi-shot `call/cc` representation,
   stack-frame metadata format, `ESHKOL_VALUE_HANDLE` lifecycle states —
   all need to be locked before agents work on (the-environment), the
   debugger, the linear-types item, etc.

2. **Each task spec carries acceptance criteria + a verification command.**
   The agent's claim is satisfied when the verification commands all
   exit 0 AND the ICC `v1.3-evolve` oracle gains the corresponding probe
   PASS.

3. **Agents claim tasks by writing `claims/<agent>__<task-id>.json`.**
   Other agents reject overlapping file_globs as conflicts.

4. **Tsotchke-chan picks up tasks via `tsotchke-chan ask`** with the
   task-spec text + repo path + cost ceiling. Distinct backends per task
   (Claude / Codex / Qwen / Kimi) are chosen by `--backend`.

5. **Mesh node assignment** follows
   `~/Desktop/computer_mesh/nodes.json`: native-Windows work goes to
   `jack-blupc`, Linux media work to `cosbox`/`old-donkey`, macOS
   CoreGraphics to `atlas`/`enki`, etc.

6. **The ICC oracle is the success gate.** Every task acceptance pinches
   on an ICC probe; the per-task verification block invokes
   `scripts/run_icc_smoke.sh` (or equivalent) and the
   `v1.3-evolve` completion oracle to confirm the criterion flipped to
   PASS.

## Running the swarm

```sh
# Refresh status
make eshkol-swarm-status

# List pending tasks
ls .swarm/tasks/ | sort

# Submit a task via tsotchke-chan
scripts/swarm_agent_preflight.sh --task ESH-0001
tsotchke-chan ask "$(cat .swarm/tasks/ESH-0001.json)" \
    --repo /Users/tyr/Desktop/eshkol \
    --max-cost 5.00 \
    --backend claude

# Check active claims
ls .swarm/claims/
```

## Hard rules

- **Every multi-agent task starts with ICC preflight.** Run
  `scripts/swarm_agent_preflight.sh --task ESH-NNNN` before dispatching
  or starting work. In ICC-enabled environments this delegates to
  `icc agent-preflight --repo eshkol_lang --task-id ESH-NNNN
  --require-swarm --require-swarm-task`; stale tsotchke mirrors, missing
  task metadata, active path conflicts, and dirty/untracked work block
  the task.
- **Before any destructive git operation, snapshot first.** Use
  `scripts/swarm_agent_preflight.sh --task ESH-NNNN --snapshot` or make
  a normal commit. Do not rely on `git stash --include-untracked` as the
  only preservation path; ignored files and failed stashes can still
  lose work.
- **Do not commit code that bypasses the ICC oracle.** Every PR / direct
  commit must pass `python3 ~/Desktop/infinite_context_coder/scripts/codebase_tool.py production-audit --repo eshkol_lang --target v1.3-release --trace-dir scripts/icc_traces`.
- **Architectural-coherence work stays on atlas/the main thread.**
  Multi-shot `call/cc`, source-span stack traces, the debugger, OALR
  region pinning. Don't parallel-dispatch these.
- **Per-platform work goes to the right mesh node.** macOS-specific code
  on atlas/enki, Linux-specific on cosbox/old-donkey, Windows-specific on
  jack-blupc.
- **Decisions get recorded.** Anything that affects more than one task
  goes in `decisions.md` so the parallel agents reading the spec see the
  same picture.
