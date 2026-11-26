/-
  FramePoolCorrespondence.lean
  Purpose: correspondence between the native frame allocator and FramePool
  (EshkolKernelK2). See generate_frame_pool_traces.c for the operation mapping.

  Interface: lean --run FramePoolCorrespondence.lean <trace file>
    with EshkolKernelK2.olean on LEAN_PATH.
  Checks after every operation:
    OPERATION  for operations inside the model's domain (alloc_user, free), the
               model's state after the corresponding step equals the native state
               (free count, allocated count, owner and generation of every frame);
               a refused alloc_user corresponds to no index being accepted.
    INVARIANT  for every operation, including alloc_kernel, the native state
               satisfies Conserved, Accounted, and native's own
               k2_frame_pool_invariants_hold.
  An alloc_kernel ends the OPERATION comparison for its trace (no model operation),
  but INVARIANT checking continues.
  This is finite trace agreement, not a refinement proof.
-/
import EshkolKernelK2
open EshkolKernelK2

def ownerOf : String → FrameOwner
  | "1" => .kernel | "2" => .user | _ => .free

def ownerCode : FrameOwner → Nat
  | .free => 0 | .kernel => 1 | .user => 2

structure NativeState where
  free : Nat
  allocated : Nat
  owners : List Nat
  generations : List Nat
  invariants : Bool

def parseState (s : String) : Option NativeState :=
  match (s.splitOn " ").filter (· ≠ "") with
  | "counts" :: f :: a :: rest =>
    let owners := rest.drop 1 |>.takeWhile (· != "generations")
    let afterOwners := rest.drop (owners.length + 2)
    let gens := afterOwners.takeWhile (· != "invariants")
    let inv := afterOwners.drop (gens.length + 1)
    some ⟨f.toNat!, a.toNat!, owners.map String.toNat!, gens.map String.toNat!, inv == ["1"]⟩
  | _ => none

def stateAgrees (p : FramePool) (n : NativeState) : Bool :=
  p.freeCount == n.free && p.allocatedCount == n.allocated &&
  (List.range p.totalFrames).map (fun i => ownerCode (p.owners i)) == n.owners &&
  (List.range p.totalFrames).map p.generations == n.generations

def count (owners : List Nat) (code : Nat) : Nat := (owners.filter (· == code)).length

def accounted (total : Nat) (n : NativeState) : Bool :=
  n.free + n.allocated == total && n.free == count n.owners 0 &&
  n.allocated == count n.owners 1 + count n.owners 2

structure Replay where
  trace : String := ""
  pool : FramePool := ⟨0, 0, 0, fun _ => .free, fun _ => 1⟩
  inDomain : Bool := true
  diverged : Bool := false
  traces : Nat := 0
  operations : Nat := 0
  operationDivergedTraces : Nat := 0
  leftDomainTraces : Nat := 0
  invariantChecks : Nat := 0
  invariantFailures : Nat := 0
  messages : List String := []
  statuses : List (String × Nat) := []

def bump (l : List (String × Nat)) (k : String) : List (String × Nat) :=
  match l.lookup k with
  | some n => (k, n + 1) :: l.filter (·.1 != k)
  | none => (k, 1) :: l

def note (r : Replay) (m : String) : Replay := { r with messages := m :: r.messages }

def step (r : Replay) (line : String) : Replay :=
  if line.startsWith "trace " then
    match (line.splitOn " ") with
    | [_, _, "total", t, "reserved", k] =>
      let total := t.toNat!; let kRes := k.toNat!
      { r with trace := line, inDomain := true, diverged := false, traces := r.traces + 1,
               pool := { totalFrames := total, freeCount := total - kRes, allocatedCount := kRes,
                         owners := fun i => if i < kRes then .kernel else .free,
                         generations := fun _ => 1 } }
    | _ => note r s!"PARSE {line}"
  else
    match line.splitOn " => " with
    | [op, rest] =>
      match rest.splitOn " | " with
      | [result, stateText] =>
        match parseState stateText with
        | none => note r s!"PARSE {line}"
        | some n =>
          let opName := op.takeWhile (· != ' ')
          let status := result.takeWhile (· != ' ')
          let r := { r with statuses := bump r.statuses s!"{opName}\t{status}",
                            invariantChecks := r.invariantChecks + 1 }
          let invOk := accounted r.pool.totalFrames n && n.invariants
          let r := if invOk then r else
            note { r with invariantFailures := r.invariantFailures + 1 } s!"INVARIANT {r.trace} | {line}"
          if !r.inDomain || r.diverged then r
          else
            let modelNext : Option FramePool :=
              match (op.splitOn " ").filter (· ≠ ""), (result.splitOn " ").filter (· ≠ "") with
              | ["alloc_user"], ["OK", idx] =>
                let i := idx.toNat!
                -- The model must accept the index native chose.
                if i < r.pool.totalFrames ∧ r.pool.owners i = .free ∧ 0 < r.pool.freeCount then
                  some (r.pool.step (.allocUser i)) else none
              | ["alloc_user"], _ =>
                -- A refusal corresponds to no index being accepted.
                if (List.range r.pool.totalFrames).any (fun i =>
                    decide (r.pool.owners i = .free ∧ 0 < r.pool.freeCount)) then none
                else some r.pool
              | ["free", idx, gen, physOk], _ =>
                let i := if physOk == "1" then idx.toNat! else r.pool.totalFrames
                some (r.pool.step (.freeUser i gen.toNat!))
              | _, _ => some r.pool
            if opName == "alloc_kernel" then
              { r with inDomain := false, leftDomainTraces := r.leftDomainTraces + 1 }
            else
              let r := { r with operations := r.operations + 1 }
              match modelNext with
              | some p' =>
                if stateAgrees p' n then { r with pool := p' }
                else note { r with diverged := true, operationDivergedTraces := r.operationDivergedTraces + 1 }
                  s!"OPERATION {r.trace} | {line}"
              | none => note { r with diverged := true, operationDivergedTraces := r.operationDivergedTraces + 1 }
                  s!"OPERATION (model refuses native's choice) {r.trace} | {line}"
      | _ => note r s!"PARSE {line}"
    | _ => note r s!"PARSE {line}"

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
    let lines := (← IO.FS.readFile path).splitOn "\n" |>.filter (· ≠ "")
    let r := lines.foldl step {}
    IO.println s!"traces {r.traces}; operation comparison diverged in {r.operationDivergedTraces}; left the model's domain (alloc_kernel) {r.leftDomainTraces}"
    IO.println s!"in-domain operations compared {r.operations}; invariant checks {r.invariantChecks}; invariant failures {r.invariantFailures}"
    for (k, n) in r.statuses.toArray.qsort (fun a b => a.1 < b.1) do IO.println s!"  {n}\t{k}"
    for m in (r.messages.reverse.take 6) do IO.println s!"{m.take 400}"
    let parse := r.messages.any (·.startsWith "PARSE")
    return (if r.operationDivergedTraces == 0 && r.invariantFailures == 0 && !parse then 0 else 1)
  | _ => IO.eprintln "usage: lean --run FramePoolCorrespondence.lean <trace file>"; return 1
