/-
  EshkolCompositorAligned.lean
  The K5 compositor of tsotchke/eshkol #960 (lib/core/os_k5_k6_services.c),
  modelled in one dimension, aligned with the native registration and write
  policy, check order and status codes.

  Hidden decision: the native surface table (slots, generations, capability ids)
  is represented by a list in registration order and a handle by its index.
  The native code has no operation that frees a slot, so slot = index and every
  issued (slot, generation, cap_id) triple corresponds to exactly one index.
  A native handle whose triple was never issued corresponds to an index at or
  beyond the end of the list.

  Invariant (reachable states): writable surfaces of distinct owners do not
  overlap (Isolated); every surface lies inside the framebuffer (InFrame); at
  most maxSurfaces surfaces exist (Bounded).

  Guarantees, in layers:
    handle validity      only a registered handle naming its surface's owner,
                         on a writable surface, can write
    spatial confinement  a write lands inside its own surface
    physical confinement a write lands inside the framebuffer
    write isolation      no other owner's writable surface covers the cell
    ownership isolation  registration keeps writable surfaces of distinct owners
                         disjoint
    exactness            register and write accept exactly under these conditions
                         (so over-strict variants are refuted too)
  This is write isolation, not confidentiality: a read-only surface may overlap
  another owner's writable surface and observe what it writes.

  Temporal authority: the native interface has no operation that invalidates a
  handle, so no temporal property is stated.
-/

namespace EshkolCompositorAligned

def maxSurfaces : Nat := 16

inductive Status where
  | ok | invalidArg | noSlots | surfaceConflict | roTrap | bounds | badSurface
  deriving DecidableEq, Repr

structure Surface where
  ownerBadge : Nat
  xStart     : Nat
  xLen       : Nat
  writable   : Bool
  deriving DecidableEq, Repr

def Surface.covers (s : Surface) (x : Nat) : Bool :=
  decide (s.xStart ≤ x ∧ x < s.xStart + s.xLen)

def Surface.overlaps (a b : Surface) : Bool :=
  decide (a.xStart < b.xStart + b.xLen ∧ b.xStart < a.xStart + a.xLen)

/-- The native conflict rule: overlapping writable surfaces of distinct owners. -/
def Surface.conflicts (a b : Surface) : Bool :=
  a.writable && b.writable && a.ownerBadge != b.ownerBadge && a.overlaps b

structure Handle where
  surfaceId : Nat
  badge     : Nat
  deriving DecidableEq, Repr

structure CompositorState where
  width       : Nat
  surfaces    : List Surface
  framebuffer : Nat → Nat

def CompositorState.init (w : Nat) : CompositorState := ⟨w, [], fun _ => 0⟩

/-- Native `create_surface`, in its check order. On `.ok` the new handle's index is
    the previous number of surfaces. -/
def CompositorState.register (c : CompositorState) (s : Surface) : Status × CompositorState :=
  if s.ownerBadge = 0 ∨ s.xLen = 0 then (.invalidArg, c)
  else if c.width < s.xStart + s.xLen then (.bounds, c)
  else if c.surfaces.any (fun o => s.conflicts o) then (.surfaceConflict, c)
  else if maxSurfaces ≤ c.surfaces.length then (.noSlots, c)
  else (.ok, { c with surfaces := c.surfaces ++ [s] })

/-- Native 1×1 `write_rect` at absolute cell `x`, in its check order. -/
def CompositorState.write (c : CompositorState) (h : Handle) (x v : Nat) :
    Status × CompositorState :=
  match c.surfaces[h.surfaceId]? with
  | none => (.badSurface, c)
  | some s =>
    if s.ownerBadge ≠ h.badge then (.badSurface, c)
    else if s.writable = false then (.roTrap, c)
    else if c.width ≤ x then (.bounds, c)
    else if c.surfaces.any (fun o => o.ownerBadge != s.ownerBadge && o.writable && o.covers x) then
      (.surfaceConflict, c)
    else if s.covers x = false then (.surfaceConflict, c)
    else (.ok, { c with framebuffer := fun i => if i = x then v else c.framebuffer i })

inductive CompositorOp where
  | register (s : Surface)
  | write (h : Handle) (x v : Nat)

def CompositorState.step (c : CompositorState) : CompositorOp → CompositorState
  | .register s => (c.register s).2
  | .write h x v => (c.write h x v).2

def CompositorState.run (c : CompositorState) (ops : List CompositorOp) : CompositorState :=
  ops.foldl CompositorState.step c

def CompositorState.Isolated (c : CompositorState) : Prop :=
  ∀ a ∈ c.surfaces, ∀ b ∈ c.surfaces, a.conflicts b = false

def CompositorState.InFrame (c : CompositorState) : Prop :=
  ∀ s ∈ c.surfaces, s.xStart + s.xLen ≤ c.width

def CompositorState.Bounded (c : CompositorState) : Prop :=
  c.surfaces.length ≤ maxSurfaces

/-! ## Write guarantees -/

/-- What an accepted write establishes, all layers together. -/
theorem write_ok_conditions (c : CompositorState) (h : Handle) (x v : Nat)
    (hok : (c.write h x v).1 = .ok) :
    ∃ s, c.surfaces[h.surfaceId]? = some s ∧ s.ownerBadge = h.badge ∧ s.writable = true ∧
      s.covers x = true ∧ x < c.width ∧
      ∀ o ∈ c.surfaces, o.writable = true → o.covers x = true → o.ownerBadge = h.badge := by
  unfold CompositorState.write at hok
  split at hok
  · simp at hok
  · rename_i s hs
    split at hok
    · simp at hok
    split at hok
    · simp at hok
    split at hok
    · simp at hok
    split at hok
    · simp at hok
    split at hok
    · simp at hok
    rename_i hown hwr hw hconf hcov
    refine ⟨s, hs, Decidable.of_not_not hown, by simpa using hwr, by simpa using hcov,
      Nat.lt_of_not_le hw, ?_⟩
    intro o ho howr hocov
    have hown' : s.ownerBadge = h.badge := Decidable.of_not_not hown
    rw [← hown']
    apply Decidable.byContradiction
    intro hne
    apply hconf
    exact List.any_eq_true.mpr ⟨o, ho, by simp [hne, howr, hocov]⟩

theorem write_handle_validity (c : CompositorState) (h : Handle) (x v : Nat)
    (hok : (c.write h x v).1 = .ok) :
    ∃ s, c.surfaces[h.surfaceId]? = some s ∧ s.ownerBadge = h.badge ∧ s.writable = true := by
  obtain ⟨s, hs, ho, hw, _⟩ := write_ok_conditions c h x v hok
  exact ⟨s, hs, ho, hw⟩

theorem write_spatial_confinement (c : CompositorState) (h : Handle) (x v : Nat)
    (hok : (c.write h x v).1 = .ok) :
    ∃ s, c.surfaces[h.surfaceId]? = some s ∧ s.covers x = true := by
  obtain ⟨s, hs, _, _, hc, _⟩ := write_ok_conditions c h x v hok
  exact ⟨s, hs, hc⟩

theorem write_physical_confinement (c : CompositorState) (h : Handle) (x v : Nat)
    (hok : (c.write h x v).1 = .ok) : x < c.width := by
  obtain ⟨_, _, _, _, _, hx, _⟩ := write_ok_conditions c h x v hok
  exact hx

theorem write_isolation (c : CompositorState) (h : Handle) (x v : Nat)
    (hok : (c.write h x v).1 = .ok) :
    ∀ o ∈ c.surfaces, o.writable = true → o.covers x = true → o.ownerBadge = h.badge := by
  obtain ⟨_, _, _, _, _, _, hiso⟩ := write_ok_conditions c h x v hok
  exact hiso

/-- A write changes at most the addressed cell, and only when accepted. It never
    changes the surface table or the framebuffer width. -/
theorem write_effect (c : CompositorState) (h : Handle) (x v i : Nat)
    (hchg : (c.write h x v).2.framebuffer i ≠ c.framebuffer i) :
    i = x ∧ (c.write h x v).1 = .ok := by
  unfold CompositorState.write at hchg ⊢
  split at hchg
  · exact absurd rfl hchg
  · split at hchg
    · exact absurd rfl hchg
    split at hchg
    · exact absurd rfl hchg
    split at hchg
    · exact absurd rfl hchg
    split at hchg
    · exact absurd rfl hchg
    split at hchg
    · exact absurd rfl hchg
    rename_i hown hwr hw hconf hcov
    dsimp only at hchg
    by_cases hi : i = x
    · refine ⟨hi, ?_⟩
      simp [hown, hwr, hw, hconf, hcov]
    · simp [hi] at hchg

theorem write_preserves_table (c : CompositorState) (h : Handle) (x v : Nat) :
    (c.write h x v).2.surfaces = c.surfaces ∧ (c.write h x v).2.width = c.width := by
  unfold CompositorState.write
  split
  · exact ⟨rfl, rfl⟩
  · split
    · exact ⟨rfl, rfl⟩
    split
    · exact ⟨rfl, rfl⟩
    split
    · exact ⟨rfl, rfl⟩
    split
    · exact ⟨rfl, rfl⟩
    split
    · exact ⟨rfl, rfl⟩
    exact ⟨rfl, rfl⟩

/-- Exactness of write: the conditions of `write_ok_conditions` are sufficient. -/
theorem write_accepts_permitted (c : CompositorState) (h : Handle) (x v : Nat) (s : Surface)
    (hs : c.surfaces[h.surfaceId]? = some s) (ho : s.ownerBadge = h.badge)
    (hw : s.writable = true) (hc : s.covers x = true) (hx : x < c.width)
    (hiso : ∀ o ∈ c.surfaces, o.writable = true → o.covers x = true → o.ownerBadge = h.badge) :
    (c.write h x v).1 = .ok := by
  unfold CompositorState.write
  rw [hs]
  have hnot : ¬ ∃ o, o ∈ c.surfaces ∧ (¬ o.ownerBadge = h.badge ∧ o.writable = true) ∧
      o.covers x = true := by
    rintro ⟨o, hom, ⟨hne, how⟩, hoc⟩
    exact hne (hiso o hom how hoc)
  simp [ho, hw, hc, Nat.not_le.mpr hx, hnot]

/-! ## Registration guarantees -/

theorem conflicts_symm (a b : Surface) : a.conflicts b = b.conflicts a := by
  unfold Surface.conflicts Surface.overlaps
  cases a.writable <;> cases b.writable <;> simp only [Bool.false_and, Bool.and_false, Bool.true_and]
  have hb : (a.ownerBadge != b.ownerBadge) = (b.ownerBadge != a.ownerBadge) := by
    apply Bool.eq_iff_iff.mpr
    simp only [bne_iff_ne]
    exact ⟨Ne.symm, Ne.symm⟩
  rw [hb]
  by_cases h1 : a.xStart < b.xStart + b.xLen <;> by_cases h2 : b.xStart < a.xStart + a.xLen <;>
    simp [h1, h2]

theorem register_preserves_isolated (c : CompositorState) (s : Surface) (hi : c.Isolated) :
    (c.register s).2.Isolated := by
  unfold CompositorState.register
  split
  · exact hi
  split
  · exact hi
  split
  · exact hi
  split
  · exact hi
  rename_i _ _ hconf _
  have hns : ∀ o ∈ c.surfaces, s.conflicts o = false := by
    intro o ho
    apply Bool.eq_false_iff.mpr
    intro hc
    exact hconf (List.any_eq_true.mpr ⟨o, ho, hc⟩)
  have hself : s.conflicts s = false := by simp [Surface.conflicts]
  intro a ha b hb
  simp only [List.mem_append, List.mem_singleton] at ha hb
  rcases ha with ha | ha <;> rcases hb with hb | hb
  · exact hi a ha b hb
  · subst hb; rw [conflicts_symm]; exact hns a ha
  · subst ha; exact hns b hb
  · subst ha; subst hb; exact hself

theorem register_preserves_inframe (c : CompositorState) (s : Surface) (hf : c.InFrame) :
    (c.register s).2.InFrame := by
  unfold CompositorState.register
  split
  · exact hf
  split
  · exact hf
  split
  · exact hf
  split
  · exact hf
  rename_i _ hb _ _
  intro t ht
  simp only [List.mem_append, List.mem_singleton] at ht
  rcases ht with ht | ht
  · exact hf t ht
  · subst ht; exact Nat.le_of_not_lt hb

theorem register_preserves_bounded (c : CompositorState) (s : Surface) (hb : c.Bounded) :
    (c.register s).2.Bounded := by
  unfold CompositorState.register CompositorState.Bounded
  split
  · exact hb
  split
  · exact hb
  split
  · exact hb
  split
  · exact hb
  rename_i _ _ _ hn
  simp only [List.length_append, List.length_singleton]
  omega

/-- The conflict rule in primitive terms: the intended policy, stated once. -/
theorem conflicts_iff (a b : Surface) : a.conflicts b = true ↔
    a.writable = true ∧ b.writable = true ∧ a.ownerBadge ≠ b.ownerBadge ∧
    a.xStart < b.xStart + b.xLen ∧ b.xStart < a.xStart + a.xLen := by
  simp [Surface.conflicts, Surface.overlaps, Bool.and_eq_true, bne_iff_ne, and_assoc]

/-- Exactness of register, in both directions, with the overlap rule in primitive
    terms so that an over-strict or over-permissive conflict rule is refuted. -/
theorem register_ok_iff (c : CompositorState) (s : Surface) :
    (c.register s).1 = .ok ↔
      s.ownerBadge ≠ 0 ∧ s.xLen ≠ 0 ∧ s.xStart + s.xLen ≤ c.width ∧
      (∀ o ∈ c.surfaces, s.writable = true → o.writable = true → s.ownerBadge ≠ o.ownerBadge →
        ¬ (s.xStart < o.xStart + o.xLen ∧ o.xStart < s.xStart + s.xLen)) ∧
      c.surfaces.length < maxSurfaces := by
  unfold CompositorState.register
  constructor
  · intro hok
    split at hok
    · simp at hok
    split at hok
    · simp at hok
    split at hok
    · simp at hok
    split at hok
    · simp at hok
    rename_i hinv hb hconf hn
    refine ⟨fun h => hinv (Or.inl h), fun h => hinv (Or.inr h), Nat.le_of_not_lt hb, ?_,
      Nat.lt_of_not_le hn⟩
    intro o ho hsw how hne hov
    exact hconf (List.any_eq_true.mpr ⟨o, ho, (conflicts_iff s o).mpr ⟨hsw, how, hne, hov⟩⟩)
  · rintro ⟨ho, hl, hb, hconf, hn⟩
    have hany : (c.surfaces.any fun o => s.conflicts o) = false := by
      apply Bool.eq_false_iff.mpr
      intro h
      obtain ⟨o, hom, hc⟩ := List.any_eq_true.mp h
      obtain ⟨hsw, how, hne, hov⟩ := (conflicts_iff s o).mp hc
      exact hconf o hom hsw how hne hov
    have hinv : ¬ (s.ownerBadge = 0 ∨ s.xLen = 0) := fun h => h.elim ho hl
    simp [hinv, Nat.not_lt.mpr hb, hany, Nat.not_le.mpr hn]

/-! ## Reachable states -/

def CompositorState.Invariant (c : CompositorState) : Prop :=
  c.Isolated ∧ c.InFrame ∧ c.Bounded

theorem step_preserves_invariant (c : CompositorState) (op : CompositorOp)
    (hinv : c.Invariant) : (c.step op).Invariant := by
  obtain ⟨hi, hf, hb⟩ := hinv
  cases op with
  | register s =>
    exact ⟨register_preserves_isolated c s hi, register_preserves_inframe c s hf,
      register_preserves_bounded c s hb⟩
  | write h x v =>
    obtain ⟨hs, hw⟩ := write_preserves_table c h x v
    refine ⟨?_, ?_, ?_⟩
    · unfold CompositorState.Isolated; simp only [CompositorState.step, hs]; exact hi
    · unfold CompositorState.InFrame; simp only [CompositorState.step, hs, hw]; exact hf
    · unfold CompositorState.Bounded; simp only [CompositorState.step, hs]; exact hb

theorem run_preserves_invariant (c : CompositorState) (ops : List CompositorOp)
    (h : c.Invariant) : (c.run ops).Invariant := by
  induction ops generalizing c with
  | nil => exact h
  | cons op _ ih => exact ih (c.step op) (step_preserves_invariant c op h)

theorem reachable_invariant (w : Nat) (ops : List CompositorOp) :
    ((CompositorState.init w).run ops).Invariant :=
  run_preserves_invariant _ ops
    ⟨fun _ h => absurd h (List.not_mem_nil _), fun _ h => absurd h (List.not_mem_nil _),
     Nat.zero_le _⟩

end EshkolCompositorAligned
