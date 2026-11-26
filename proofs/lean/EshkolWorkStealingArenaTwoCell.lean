/-
  Branch publication, allocation arithmetic and two-cell access traces.

  Model-to-Implementation Correspondence
  lib/core/blc/work_stealing_ledger.esk implements blc-ws-commit,
  blc-ws-sblc-classify, blc-ws-arena-bump-check and the two-cell alias check.
  tests/stdlib/esa_packet2_work_stealing_tx_sblc_test.esk exercises stale and
  renewed requests, conditioning status, interval overlap, alias refusal,
  integer rounding and transactional bounds. The update-list model below
  projects the index validation in blc-ws-tx-apply-updates-atomic.
  arena_allocate_aligned in lib/core/runtime_arena_core.cpp aligns the absolute
  address and rounds the requested size. The proofs use natural arithmetic for
  the in-block path after the runtime alignment and size checks.
  tests/core/runtime_arena_core_test.cpp checks alignment and refusal cases.
-/

namespace EshkolWorkStealingArenaTwoCell

/-! ## 1. Authorized Work-Stealing Branch-Transfer Ledger -/

structure Ledger where
  generation : Nat
  occurrence : Nat
  permitted  : Bool
  spent      : Nat
  budget     : Nat
  result     : Option Bool
  deriving DecidableEq, Repr

def commit (s : Ledger) (g atPos cost : Nat) (v : Bool) : Ledger :=
  if s.permitted && s.generation == g && s.occurrence == atPos &&
      s.result.isNone && decide (s.spent + cost ≤ s.budget) then
    { s with spent := s.spent + cost, result := some v }
  else s

theorem commit_stale_gen (s : Ledger) (h : s.generation ≠ g) :
    commit s g p cost v = s := by
  simp [commit, h]

theorem commit_stale_occ (s : Ledger) (h : s.occurrence ≠ p) :
    commit s g p cost v = s := by
  simp [commit, h]

theorem commit_no_permit (s : Ledger) (h : s.permitted = false) :
    commit s g p cost v = s := by
  simp [commit, h]

theorem commit_over_budget (s : Ledger) (h : s.budget < s.spent + cost) :
    commit s g p cost v = s := by
  have hnot : ¬ (s.spent + cost ≤ s.budget) := by omega
  simp [commit, hnot]

theorem commit_no_double_publish (s : Ledger) (v0 : Bool) (h : s.result = some v0) :
    commit s g p cost v = s := by
  simp [commit, h]

structure Attempt where
  generation : Nat
  occurrence : Nat
  cost : Nat
  value : Bool

def commitTrace (s : Ledger) : List Attempt → Ledger
  | [] => s
  | a :: rest => commitTrace (commit s a.generation a.occurrence a.cost a.value) rest

/-- Once a result is present, every subsequent attempt preserves the entire ledger. -/
theorem published_ledger_absorbs_trace (s : Ledger) (v : Bool) (h : s.result = some v)
    (trace : List Attempt) : commitTrace s trace = s := by
  induction trace with
  | nil => rfl
  | cons a rest ih => simpa [commitTrace, commit_no_double_publish s v h] using ih

/-- A finite trace either leaves the ledger untouched or publishes exactly one
    matching, permitted attempt and charges its cost within budget. -/
theorem commitTrace_outcome (s : Ledger) (trace : List Attempt) :
    commitTrace s trace = s ∨ ∃ a ∈ trace,
      s.permitted = true ∧ s.result = none ∧ s.generation = a.generation ∧
      s.occurrence = a.occurrence ∧ s.spent + a.cost ≤ s.budget ∧
      commitTrace s trace = { s with spent := s.spent + a.cost, result := some a.value } := by
  induction trace with
  | nil => exact Or.inl rfl
  | cons a rest ih =>
    simp only [commitTrace, commit]
    split
    · rename_i ok
      simp only [Bool.and_eq_true, beq_iff_eq, Option.isNone_iff_eq_none,
        decide_eq_true_eq] at ok
      obtain ⟨⟨⟨⟨permit, gen⟩, occ⟩, empty⟩, fits⟩ := ok
      exact Or.inr ⟨a, by simp, permit, empty, gen, occ, fits,
        published_ledger_absorbs_trace _ a.value rfl rest⟩
    · rcases ih with same | ⟨b, hb, cond⟩
      · exact Or.inl same
      · exact Or.inr ⟨b, List.mem_cons_of_mem _ hb, cond⟩

/-- In the post-switch ledger, the old request is refused and the current
    request publishes with total charge eleven. -/
theorem commit_post_switch_concrete_witness :
    let l0 : Ledger := ⟨1, 83, true, 4, 11, none⟩
    let l1 := commit l0 0 16 7 true
    let l2 := commit l1 1 83 7 true
    l1 = l0 ∧ l2.result = some true ∧ l2.spent = 11 := by
  decide

/-! ## 2. SBLC Unresolved-Mass vs. No-Consistent-Program Status -/

inductive SBLCStatus where
  | consistent
  | unresolvedUnderStepBound
  | noConsistentProgram
  deriving DecidableEq, Repr

/-- Exact classification on non-negative integer numerators over a common
    positive denominator (e.g., 16 for 4-bit prefix space). -/
def classifySBLC (matchedNum unresolvedNum : Nat) : SBLCStatus :=
  if matchedNum > 0 then .consistent
  else if unresolvedNum > 0 then .unresolvedUnderStepBound
  else .noConsistentProgram

theorem sblc_unresolved_iff (m u : Nat) :
    classifySBLC m u = .unresolvedUnderStepBound ↔ m = 0 ∧ 0 < u := by
  cases m <;> cases u <;> simp [classifySBLC]

theorem sblc_consistent_iff (m u : Nat) :
    classifySBLC m u = .consistent ↔ 0 < m := by
  cases m <;> cases u <;> simp [classifySBLC]

theorem sblc_no_consistent_program_iff (m u : Nat) :
    classifySBLC m u = .noConsistentProgram ↔ m = 0 ∧ u = 0 := by
  cases m <;> cases u <;> simp [classifySBLC]

/-! ## 3. Verified Arena Bump Allocator & Reachable-Byte Exclusion -/

def alignUp (v a : Nat) : Nat :=
  v + (a - v % a) % a

theorem alignUp_spec (v a : Nat) (ha : 0 < a) :
    v ≤ alignUp v a ∧ alignUp v a < v + a ∧ alignUp v a % a = 0 := by
  unfold alignUp
  have hmod : v % a < a := Nat.mod_lt v ha
  by_cases h0 : v % a = 0
  · have hrem : (a - 0) % a = 0 := Nat.mod_self a
    rw [h0, hrem, Nat.add_zero]
    exact ⟨Nat.le_refl v, by omega, h0⟩
  · have hlt : a - v % a < a := by omega
    rw [Nat.mod_eq_of_lt hlt]
    have hdiv : (v + (a - v % a)) % a = 0 := by
      have h1 : v + (a - v % a) = (v / a + 1) * a := by
        have hdm := Nat.div_add_mod v a
        have hmul : (v / a + 1) * a = a * (v / a) + a := by
          rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]
        omega
      rw [h1, Nat.mul_mod_left]
    exact ⟨by omega, by omega, hdiv⟩

structure Span where
  base : Nat
  size : Nat
  deriving DecidableEq, Repr

def Span.ends (s : Span) : Nat := s.base + s.size

def Span.apartProp (s t : Span) : Prop :=
  s.size = 0 ∨ t.size = 0 ∨ s.ends ≤ t.base ∨ t.ends ≤ s.base

def bumpPtr (base used a : Nat) : Nat := alignUp (base + used) a
def bumpNextUsed (base used n a : Nat) : Nat :=
  (bumpPtr base used a - base) + alignUp n a

/-- Aligned bump allocation stays aligned, advances monotonically past `base+used`,
    fits inside `[base, base + nextUsed]`, and never exceeds `base + capacity`. -/
theorem bump_aligned_in_bounds (base used capacity n a : Nat)
    (ha : 0 < a)
    (hfits : bumpNextUsed base used n a ≤ capacity) :
    let p := bumpPtr base used a
    p % a = 0 ∧ base + used ≤ p ∧ p + n ≤ base + bumpNextUsed base used n a ∧
      base + bumpNextUsed base used n a ≤ base + capacity := by
  have hp := alignUp_spec (base + used) a ha
  have hr := alignUp_spec n a ha
  dsimp only [bumpPtr, bumpNextUsed] at hfits ⊢
  exact ⟨hp.2.2, hp.1, by omega, by omega⟩

/-- Every prior live interval separated from the free tail `[base+used, base+cap)`
    is strictly disjoint from the newly allocated span `[p, p+n)`. -/
theorem bump_fresh_from_live (base used capacity n a : Nat)
    (ha : 0 < a)
    (hfits : bumpNextUsed base used n a ≤ capacity)
    (s : Span)
    (hsep : s.ends ≤ base + used ∨ base + capacity ≤ s.base) :
    s.ends ≤ bumpPtr base used a ∨ bumpPtr base used a + n ≤ s.base := by
  have hb := bump_aligned_in_bounds base used capacity n a ha hfits
  rcases hsep with hlo | hhi
  · left; exact Nat.le_trans hlo hb.2.1
  · right; exact Nat.le_trans (Nat.le_trans hb.2.2.1 hb.2.2.2) hhi

/-- Cursor metadata store touches only `[field, field + 8)` and preserves every
    external byte. -/
def storeCursor (mem : Nat → UInt8) (field : Nat) (bytes : Fin 8 → UInt8) (x : Nat) : UInt8 :=
  if h : x - field < 8 then
    if field ≤ x then bytes ⟨x - field, h⟩ else mem x
  else mem x

theorem cursor_store_preserves_outside_bytes (mem : Nat → UInt8) (field : Nat)
    (bytes : Fin 8 → UInt8) (x : Nat)
    (h : x < field ∨ field + 8 ≤ x) :
    storeCursor mem field bytes x = mem x := by
  unfold storeCursor
  split
  · split
    · omega
    · rfl
  · rfl

/-- Early null/zero-size refusal returns `(0, heap)` untouched. -/
def earlyReturn (arena n : Nat) (heap : Nat → UInt8) : Option (Nat × (Nat → UInt8)) :=
  if arena = 0 ∨ n = 0 then some (0, heap) else none

theorem early_failure_preserves_heap (arena n : Nat) (heap : Nat → UInt8)
    (h : arena = 0 ∨ n = 0) :
    earlyReturn arena n heap = some (0, heap) := by
  simp [earlyReturn, h]

/-- Reachable-byte exclusion: if every reachable address is covered by some live
    span disjoint from `[ptr, ptr+n)`, no reachable byte lies inside `[ptr, ptr+n)`. -/
theorem reachable_byte_excluded (ptr n : Nat) (live : List Span)
    (hfresh : ∀ s ∈ live, Span.apartProp ⟨ptr, n⟩ s)
    (reachable : Nat → Prop)
    (hcov : ∀ p, reachable p → ∃ s ∈ live, s.base ≤ p ∧ p < s.ends)
    (p : Nat) (hp : reachable p) :
    ¬ (ptr ≤ p ∧ p < ptr + n) := by
  rcases hcov p hp with ⟨s, hs, hlo, hhi⟩
  have ha := hfresh s hs
  simp only [Span.apartProp, Span.ends] at ha hhi
  rcases ha with h1 | h2 | h3 | h4 <;> omega

/-! ## 4. Two-Cell Alias Separation & Read-Only Source Preservation -/

inductive Port where | src | dst deriving DecidableEq, Repr
inductive Access where | read | write deriving DecidableEq, Repr

structure MemEvent where
  kind : Access
  port : Port
  deriving DecidableEq, Repr

def cellAddr (dst src : Nat) : Port → Nat
  | .dst => dst
  | .src => src

def twoCellSeparate (dst src : Nat) : Prop :=
  Span.apartProp ⟨dst, 1⟩ ⟨src, 1⟩

theorem two_cell_separation_iff_ne (dst src : Nat) :
    twoCellSeparate dst src ↔ dst ≠ src := by
  simp only [twoCellSeparate, Span.apartProp, Span.ends]
  constructor
  · intro h
    rcases h with h1 | h2 | h3 | h4 <;> omega
  · intro hne
    by_cases hlt : dst < src
    · right; right; left; omega
    · right; right; right; omega

theorem two_cell_alias_refused (p : Nat) : ¬ twoCellSeparate p p := by
  intro h
  exact (two_cell_separation_iff_ne p p).mp h rfl

theorem two_cell_bound_writes_exclude_readonly_src
    (dst src : Nat) (hsep : twoCellSeparate dst src)
    (events : List MemEvent)
    (hro : ∀ e ∈ events, e.kind = .write → e.port ≠ .src) :
    ∀ e ∈ events, e.kind = .write → cellAddr dst src e.port ≠ src := by
  intro e he hw
  have hne : e.port ≠ .src := hro e he hw
  cases hp : e.port with
  | src => exact False.elim (hne hp)
  | dst => exact (two_cell_separation_iff_ne dst src).mp hsep

/-- A write carries its own payload. A read leaves the heap unchanged. -/
def runEvents {V : Type} (dst src : Nat) (heap : Nat → V) :
    List (MemEvent × V) → Nat → V
  | [] => heap
  | (e, value) :: rest =>
    let next := fun p => if e.kind = .write ∧ cellAddr dst src e.port = p then value else heap p
    runEvents dst src next rest

theorem two_cell_trace_preserves_source {V : Type} (dst src : Nat)
    (hsep : twoCellSeparate dst src) (heap : Nat → V) (events : List (MemEvent × V))
    (hro : ∀ p ∈ events, p.1.kind = .write → p.1.port ≠ .src) :
    runEvents dst src heap events src = heap src := by
  induction events generalizing heap with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨e, value⟩ := p
    have head : ¬ (e.kind = .write ∧ cellAddr dst src e.port = src) := by
      intro ⟨hw, ha⟩
      cases hp : e.port with
      | src => exact hro (e, value) (by simp) hw hp
      | dst => exact (two_cell_separation_iff_ne dst src).mp hsep (by simpa [hp, cellAddr] using ha)
    simp only [runEvents]
    rw [ih _ (fun r hr => hro r (List.mem_cons_of_mem _ hr))]
    exact if_neg head

/-! ## 5. BF16 Rounding-Tie Counterexample & TX-4 Atomic Failure Continuation -/

/-- Round-half-to-even for non-negative rational `n / d` (`d > 0`). -/
def roundHalfToEven (n d : Nat) : Nat :=
  let q := n / d
  let r := n % d
  if 2 * r < d then q
  else if 2 * r > d then q + 1
  else if q % 2 == 0 then q else q + 1

/-- In the binade [1, 2) the BF16 grid has spacing 1/128. First rounding
    1535/1024 to that grid gives 192/128. Rounding again to an integer
    reaches two, whereas a single rounding to an integer reaches one. -/
theorem bf16_tie_diverges_from_exact :
    let units := roundHalfToEven (1535 * 128) 1024
    units = 192 ∧ roundHalfToEven units 128 = 2 ∧
      roundHalfToEven 1535 1024 = 1 ∧ roundHalfToEven units 128 ≠ roundHalfToEven 1535 1024 := by
  decide

/-- Validate an entire list of update indices and return the accepted requests. -/
def txApplyAtomic (nParams : Nat) (updates : List (Nat × Int)) : Option (List (Nat × Int)) :=
  if updates.all (fun u => decide (u.1 < nParams)) then some updates else none

theorem tx_validation_iff (n : Nat) (us : List (Nat × Int)) :
    txApplyAtomic n us = some us ↔ ∀ u ∈ us, u.1 < n := by
  simp [txApplyAtomic, List.all_eq_true]

theorem tx_oob_after_valid_index_returns_none :
    txApplyAtomic 3 [(0, 1), (99, 1)] = none ∧
    txApplyAtomic 3 [(0, 1), (2, -5)] = some [(0, 1), (2, -5)] := by
  decide

#print axioms commit_stale_gen
#print axioms commit_stale_occ
#print axioms commit_no_permit
#print axioms commit_over_budget
#print axioms commit_no_double_publish
#print axioms published_ledger_absorbs_trace
#print axioms commitTrace_outcome
#print axioms commit_post_switch_concrete_witness
#print axioms sblc_unresolved_iff
#print axioms sblc_consistent_iff
#print axioms sblc_no_consistent_program_iff
#print axioms alignUp_spec
#print axioms bump_aligned_in_bounds
#print axioms bump_fresh_from_live
#print axioms cursor_store_preserves_outside_bytes
#print axioms early_failure_preserves_heap
#print axioms reachable_byte_excluded
#print axioms two_cell_separation_iff_ne
#print axioms two_cell_alias_refused
#print axioms two_cell_bound_writes_exclude_readonly_src
#print axioms two_cell_trace_preserves_source
#print axioms bf16_tie_diverges_from_exact
#print axioms tx_validation_iff
#print axioms tx_oob_after_valid_index_returns_none

end EshkolWorkStealingArenaTwoCell
