import Reclamation

namespace Verificatum.NestedRegions
open ArenaLedger Reclamation

-- Frames are ordered innermost first. Native slots are outermost first.
structure Frame where
  region : Nat
  arena : Nat
  deriving DecidableEq, Repr

@[ext] structure Binding where
  fallback : Nat
  domain : Nat
  depth : Nat
  slots : List Nat
  deriving DecidableEq, Repr

def layout (capacity fallback : Nat) (frames : List Frame) : Binding :=
  ⟨fallback, (frames.head?.map Frame.arena).getD 0, frames.length,
   (frames.map Frame.region).reverse ++ List.replicate (capacity - frames.length) 0⟩

structure Valid (cap fallback : Nat) (frames : List Frame) : Prop where
  fits : frames.length ≤ cap
  regions : (frames.map Frame.region).Nodup
  arenas : (fallback :: frames.map Frame.arena).Nodup
  live : ∀ f ∈ frames, f.region ≠ 0 ∧ f.arena ≠ 0

theorem valid_tail (h : Valid cap fallback (c :: fs)) : Valid cap fallback fs := by
  obtain ⟨fits,regions,arenas,live⟩ := h
  have ha := List.nodup_cons.mp arenas
  refine ⟨by simpa using Nat.le_trans (Nat.le_succ _) fits,
    (List.nodup_cons.mp regions).2, List.nodup_cons.mpr ⟨?_,(List.nodup_cons.mp ha.2).2⟩, ?_⟩
  · intro hin
    exact ha.1 (List.mem_cons_of_mem _ hin)
  · intro f hf
    exact live f (List.mem_cons_of_mem _ hf)

def restored (cap fallback : Nat) (parent : Frame) (outer : List Frame)
    (observed : Binding) := decide (observed = layout cap fallback (parent :: outer))

-- Restore the saved domain, decrement the measured depth, and clear exactly
-- the vacated physical slot. All surviving slots come from the input.
def popPhysical (saved : Nat) (b : Binding) : Binding :=
  {b with domain := saved, depth := b.depth - 1, slots := b.slots.set (b.depth - 1) 0}

private theorem clear_last (front suffix : List Nat) (x : Nat) :
    (front ++ x :: suffix).set front.length 0 = front ++ 0 :: suffix := by
  induction front with
  | nil => rfl
  | cons a xs ih => simp [ih]

theorem pop_layout (fits : (child :: parent :: outer).length ≤ cap) :
    popPhysical parent.arena (layout cap fallback (child :: parent :: outer)) =
      layout cap fallback (parent :: outer) := by
  let fs := parent :: outer
  have pad : cap - fs.length = (cap - (fs.length + 1)) + 1 := by
    change fs.length + 1 ≤ cap at fits
    omega
  have fill := clear_last (fs.map Frame.region).reverse
    (List.replicate (cap - (fs.length+1)) 0) child.region
  simp only [List.length_reverse,List.length_map] at fill
  have slots : ((fs.map Frame.region).reverse ++ child.region ::
      List.replicate (cap - (fs.length+1)) 0).set fs.length 0 =
      (fs.map Frame.region).reverse ++ List.replicate (cap - fs.length) 0 := by
    rw [fill, pad, List.replicate_succ]
  apply Binding.ext <;> first | rfl | simpa [popPhysical,layout,fs,List.append_assoc] using slots

-- A full used-prefix rewind (mark zero) addresses survivor separation in the
-- child ledger. Backing capacity, native holder completeness and refinement
-- are additional obligations for physical destruction.
theorem checked_binding (h : Valid cap fallback (child :: parent :: outer))
    (s : State) (owns : s.arena = child.arena) (p : Permit s 0 survives)
    (accepted : restored cap fallback parent outer actual = true) :
    Valid cap fallback (parent :: outer) ∧ actual = layout cap fallback (parent :: outer) ∧
    actual.domain = parent.arena ∧ actual.slots.length = cap ∧
    child.region ∉ actual.slots ∧ s.arena ≠ actual.domain ∧
    (∀ x, survives x → Outside s 0 x) := by
  have eqn := of_decide_eq_true accepted
  subst actual
  have tail := valid_tail h
  have free := (List.nodup_cons.mp h.regions).1
  have nonzero := (h.live child (by simp)).1
  have distinct := (List.nodup_cons.mp (List.nodup_cons.mp h.arenas).2).1
  refine ⟨tail,rfl,rfl,?_,?_,?_,p.safe⟩
  · have fits := tail.fits
    simp [layout] at fits ⊢
    omega
  · simpa [layout, List.mem_append, nonzero, Ne.symm nonzero, and_comm] using free
  · rw [owns]
    intro same
    exact distinct (by simp [same, layout] at *)

theorem pop_child (h : Valid cap fallback (child :: parent :: outer))
    (s : State) (owns : s.arena = child.arena) (p : Permit s 0 survives) :
    let result := popPhysical parent.arena (layout cap fallback (child :: parent :: outer))
    Valid cap fallback (parent :: outer) ∧ result = layout cap fallback (parent :: outer) ∧
    result.domain = parent.arena ∧ result.slots.length = cap ∧ child.region ∉ result.slots ∧
    s.arena ≠ result.domain ∧ (∀ x, survives x → Outside s 0 x) := by
  apply checked_binding h s owns p
  exact decide_eq_true (pop_layout h.fits)

-- Replay each measured/counterfactual binding against a separately supplied parent.
def audit (path : String) : IO Unit := do
  for row in ← IO.FS.lines path do
    let some (outerFallback :: parentRegion :: parentArena :: beforeFallback :: beforeDomain :: beforeDepth :: actualFallback :: domain :: depth :: physical) :=
      (row.splitOn " ").mapM String.toNat? | throw (IO.userError "malformed binding")
    unless physical.length == 128 do throw (IO.userError "incomplete native stack")
    let before : Binding := ⟨beforeFallback,beforeDomain,beforeDepth,physical.take 64⟩
    let actual : Binding := ⟨actualFallback,domain,depth,physical.drop 64⟩
    IO.println (decide (popPhysical parentArena before = actual) &&
      restored 64 outerFallback ⟨parentRegion,parentArena⟩ [] actual)

end Verificatum.NestedRegions

def main (args : List String) : IO Unit := do
  let [path] := args | throw (IO.userError "expected one binding-case file")
  Verificatum.NestedRegions.audit path
