/-
  FrameMappingCorrespondence.lean
  Purpose: full-trace correspondence between native frames + address spaces
  (generate_frame_mapping_traces.c) and EshkolKernelFrameMappings. After every
  operation, compares the status and the complete state: counts; each frame's owner,
  generation, refcount, user refcount, R/W/X counts, recomputed permissions, owner
  ASID and owner space; and the set of entries (space, page, frame, permissions).
  Also compares native's verdict on each probed address-space initialization with
  Space.initStatus.
  Allocation: native chooses the frame; the model must accept that index. A refused
  allocation corresponds to no index being accepted.
  Interface: lean --run FrameMappingCorrespondence.lean <trace file>
    with EshkolKernelFrameMappings.olean on LEAN_PATH.
  This is finite trace agreement, not a refinement proof.
-/
import EshkolKernelFrameMappings
open EshkolKernelFrameMappings

def statusName : Status → String
  | .ok => "OK" | .invalidArg => "INVALID_ARG" | .outOfFrames => "OUT_OF_FRAMES"
  | .doubleFree => "DOUBLE_FREE" | .staleHandle => "STALE_HANDLE"
  | .kernelFrameLocked => "KERNEL_FRAME_LOCKED" | .frameStillMapped => "FRAME_STILL_MAPPED"
  | .wxViolation => "WX_VIOLATION" | .privViolation => "PRIV_VIOLATION"
  | .kernelAlias => "KERNEL_ALIAS" | .permAlias => "PERM_ALIAS"
  | .vaAlreadyMapped => "VA_ALREADY_MAPPED" | .vaNotMapped => "VA_NOT_MAPPED"
  | .conservationViolation => "CONSERVATION_VIOLATION"

def bit (b : Bool) (v : Nat) : Nat := if b then v else 0
def permsCode (p : Perms) : Nat := bit p.r 1 + bit p.w 2 + bit p.x 4 + bit p.u 8 + bit p.k 16
def permsOf (n : Nat) : Perms := ⟨n % 2 == 1, n / 2 % 2 == 1, n / 4 % 2 == 1, n / 8 % 2 == 1, n / 16 % 2 == 1⟩
def ownerCode : Owner → Nat | .free => 0 | .kernel => 1 | .user => 2
def optionText : Option Nat → String | some a => toString a | none => "-"

def render (S : System) : String :=
  let frames := (List.range S.total).map fun i =>
    let f := S.frames i
    let m := f.summary
    s!"{ownerCode f.owner},{f.generation},{m.refcount},{m.userRefcount},{m.reads},{m.writes}," ++
      s!"{m.execs},{permsCode m.perms},{optionText m.ownerAsid},{optionText m.ownerSpace}"
  let sorted := S.entries.toArray.qsort (fun a b => a.space < b.space || (a.space == b.space && a.page < b.page))
  let entries := sorted.toList.map fun e => s!"{e.space}:{e.page}:{e.frame}:{permsCode e.perms}"
  " ".intercalate (["counts", toString S.freeCount, toString S.allocated, "frames"] ++ frames ++ ["entries"] ++ entries)

structure Replay where
  trace : String := ""
  sys : System := System.init 0 0 []
  diverged : Bool := false
  traces : Nat := 0
  divergedTraces : Nat := 0
  operations : Nat := 0
  probes : Nat := 0
  probeFailures : Nat := 0
  messages : List String := []

def n! (s : String) : Nat := s.toNat!

def step (r : Replay) (line : String) : Replay :=
  if line.startsWith "initprobe " then
    match (line.splitOn " ").filter (· ≠ "") with
    | [_, a, u, "=>", st] =>
      let model := statusName (Space.initStatus ⟨n! a, u == "1"⟩)
      if model == st then { r with probes := r.probes + 1 }
      else { r with probes := r.probes + 1, probeFailures := r.probeFailures + 1,
                    messages := s!"{line} | model {model}" :: r.messages }
    | _ => { r with messages := s!"PARSE {line}" :: r.messages }
  else if line.startsWith "trace " then
    match (line.splitOn " ") with
    | [_, _, "total", t, "reserved", k, "asids", a0, a1, a2] =>
      { r with trace := line, diverged := false, traces := r.traces + 1,
               sys := System.init (n! t) (n! k) [⟨n! a0, false⟩, ⟨n! a1, true⟩, ⟨n! a2, true⟩] }
    | _ => { r with messages := s!"PARSE {line}" :: r.messages }
  else if r.diverged then r
  else
    match line.splitOn " | " with
    | opPart :: nativeState :: _ =>
      match opPart.splitOn " => " with
      | [op, result] =>
        let S := r.sys
        let nativeStatus := result.takeWhile (· != ' ')
        let outcome : Option (Status × System) :=
          match (op.splitOn " ").filter (· ≠ ""), (result.splitOn " ").filter (· ≠ "") with
          | [alloc], ["OK", i] =>
            let o := if alloc == "alloc_kernel" then Owner.kernel else Owner.user
            some (S.alloc o (n! i))
          | [_], _ =>
            if (List.range S.total).any (fun i => decide ((S.frames i).owner = .free ∧ 0 < S.freeCount))
            then none else some (.outOfFrames, S)
          | ["free", i, g, ok], _ => some (S.free ⟨n! i, n! g, ok == "1"⟩)
          | ["map", s, pg, i, g, ok, p], _ => some (S.map (n! s) (n! pg) ⟨n! i, n! g, ok == "1"⟩ (permsOf (n! p)))
          | ["remap", s, pg, p], _ => some (S.remap (n! s) (n! pg) (permsOf (n! p)))
          | ["unmap", s, pg], _ => some (S.unmap (n! s) (n! pg))
          | _, _ => none
        let r := { r with operations := r.operations + 1 }
        match outcome with
        | some (st, S') =>
          if statusName st == nativeStatus && render S' == nativeState.trim then { r with sys := S' }
          else { r with diverged := true, divergedTraces := r.divergedTraces + 1,
                        messages := s!"{r.trace} | {opPart} | native {nativeState.trim}\n    model {statusName st} | {render S'}" :: r.messages }
        | none => { r with diverged := true, divergedTraces := r.divergedTraces + 1,
                           messages := s!"{r.trace} | {opPart} | model accepts no such allocation" :: r.messages }
      | _ => { r with messages := s!"PARSE {line}" :: r.messages }
    | _ => { r with messages := s!"PARSE {line}" :: r.messages }

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
    let lines := (← IO.FS.readFile path).splitOn "\n" |>.filter (· ≠ "")
    let r := lines.foldl step {}
    IO.println s!"initialization probes {r.probes}; disagreeing {r.probeFailures}"
    IO.println s!"traces {r.traces}; full agreement {r.traces - r.divergedTraces}; diverged {r.divergedTraces}; operations compared {r.operations}"
    for m in r.messages.reverse.take 4 do IO.println s!"first divergence: {m.take 900}"
    let clean := r.divergedTraces == 0 && r.probeFailures == 0 && r.probes > 0 &&
      !(r.messages.any (·.startsWith "PARSE"))
    return (if clean then 0 else 1)
  | _ => IO.eprintln "usage: lean --run FrameMappingCorrespondence.lean <trace file>"; return 1
