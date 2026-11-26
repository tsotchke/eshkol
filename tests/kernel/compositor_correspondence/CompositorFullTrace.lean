/-
  CompositorFullTrace.lean
  Purpose: full-trace correspondence between the native K5 compositor and
  EshkolCompositorAligned. From the same initial state and the same operations,
  after every operation compare: the return status, the surface table (owner,
  start, length, writability, in slot order) and the framebuffer.

  Interface: lean --run CompositorFullTrace.lean <trace file>
    with EshkolCompositorAligned.olean on LEAN_PATH.
  Handle correspondence: see generate_compositor_full_traces.c. The adapter keeps,
  per registration attempt, the model id issued for it (none if refused); a tampered
  or unissued handle is given id 1000, which names no surface.
  Output: number of traces in full agreement, operations compared, and for each
  diverging trace its first divergence (later operations of that trace are not
  compared, since the states already differ).
  This is finite trace agreement on the supported one-dimensional domain, not a
  refinement proof.
-/
import EshkolCompositorAligned
open EshkolCompositorAligned

def statusName : Status → String
  | .ok => "OK" | .invalidArg => "INVALID_ARG" | .noSlots => "NO_SLOTS"
  | .surfaceConflict => "SURFACE_CONFLICT" | .roTrap => "RO_TRAP" | .bounds => "BOUNDS"
  | .badSurface => "BAD_SURFACE"

def unissuedId : Nat := 1000

def renderState (c : CompositorState) : String :=
  let fb := (List.range c.width).map (fun i => toString (c.framebuffer i))
  let surf := c.surfaces.map fun s =>
    s!"{s.ownerBadge},{s.xStart},{s.xLen},{if s.writable then 1 else 0}"
  " ".intercalate (["fb"] ++ fb ++ ["surf"] ++ surf)

structure Replay where
  trace : String := ""
  comp : CompositorState := CompositorState.init 0
  issued : Array (Option Nat × Nat) := #[]   -- per attempt: model id, owner
  diverged : Bool := false
  traces : Nat := 0
  divergedTraces : Nat := 0
  operations : Nat := 0
  outsideFrame : Nat := 0
  firstDivergences : List String := []
  statusPairs : List (String × Nat) := []

def bump (l : List (String × Nat)) (k : String) : List (String × Nat) :=
  match l.lookup k with
  | some n => (k, n + 1) :: l.filter (·.1 != k)
  | none => (k, 1) :: l

def nat! (s : String) : Nat := s.toNat!

def compare (r : Replay) (line : String) (c' : CompositorState) (st : Status)
    (nativeStatus nativeState : String) (issued : Array (Option Nat × Nat)) : Replay :=
  let modelState := renderState c'
  -- The model must not hold a non-zero cell outside the framebuffer.
  let outside := (List.range 64).any fun i => c'.width ≤ i && c'.framebuffer (i) != 0
  let ok := statusName st == nativeStatus && modelState == nativeState.trim && !outside
  let r := { r with comp := c', issued := issued, operations := r.operations + 1,
                    outsideFrame := r.outsideFrame + (if outside then 1 else 0),
                    statusPairs := bump r.statusPairs s!"{line.takeWhile (· != ' ')}\t{nativeStatus}" }
  if ok then r
  else { r with diverged := true, divergedTraces := r.divergedTraces + 1,
                firstDivergences := (s!"{r.trace} | {line}\n    model: {statusName st} | {modelState}") :: r.firstDivergences }

def step (r : Replay) (line : String) : Replay :=
  if line.startsWith "trace " then
    match (line.splitOn " ") with
    | [_, _, "width", w] =>
      { r with trace := line, comp := CompositorState.init (nat! w), issued := #[], diverged := false,
               traces := r.traces + 1 }
    | _ => { r with firstDivergences := s!"PARSE {line}" :: r.firstDivergences }
  else if r.diverged then r
  else
    match line.splitOn " => " with
    | [op, rest] =>
      match rest.splitOn " | " with
      | [nativeStatus, nativeState] =>
        match (op.splitOn " ").filter (· ≠ "") with
        | ["reg", o, xs, xl, wr] =>
          let s : Surface := ⟨nat! o, nat! xs, nat! xl, wr == "1"⟩
          let id := r.comp.surfaces.length
          let (st, c') := r.comp.register s
          let issued := r.issued.push (if st == .ok then some id else none, nat! o)
          compare r line c' st nativeStatus nativeState issued
        | ["write", i, tamper, b, x, v] =>
          let id := match r.issued.getD (nat! i) (none, 0), tamper with
            | (some k, _), "none" => k
            | _, _ => unissuedId
          let (st, c') := r.comp.write ⟨id, nat! b⟩ (nat! x) (nat! v)
          compare r line c' st nativeStatus nativeState r.issued
        | _ => { r with firstDivergences := s!"PARSE {line}" :: r.firstDivergences }
      | _ => { r with firstDivergences := s!"PARSE {line}" :: r.firstDivergences }
    | _ => { r with firstDivergences := s!"PARSE {line}" :: r.firstDivergences }

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
    let lines := (← IO.FS.readFile path).splitOn "\n" |>.filter (· ≠ "")
    let r := lines.foldl step {}
    IO.println s!"traces {r.traces}; in full agreement {r.traces - r.divergedTraces}; diverged {r.divergedTraces}"
    IO.println s!"operations compared {r.operations}; model cells outside the framebuffer {r.outsideFrame}"
    IO.println "native status counts over compared operations:"
    for (k, n) in r.statusPairs.toArray.qsort (fun a b => a.1 < b.1) do
      IO.println s!"  {n}\t{k}"
    let shown := r.firstDivergences.reverse.take 8
    for d in shown do IO.println s!"first divergence: {d.take 400}"
    return (if r.divergedTraces == 0 && !(r.firstDivergences.any (·.startsWith "PARSE")) then 0 else 1)
  | _ => IO.eprintln "usage: lean --run CompositorFullTrace.lean <trace file>"; return 1
