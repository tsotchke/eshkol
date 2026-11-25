import ArenaBump

namespace Verificatum.ArenaLedger
open ArenaBump

-- One current block during an exclusive arena lifetime. The cursor at entry
-- reserves the entire earlier prefix, even when individual owners are unknown.
structure Grant where
  arena : Nat
  generation : Nat
  base : Nat
  size : Nat
  deriving DecidableEq, Repr

structure View where
  owner : Grant
  offset : Nat
  size : Nat
  deriving DecidableEq

structure State where
  arena : Nat
  base : Nat
  capacity : Nat
  floor : Nat
  used : Nat
  next : Nat := 0
  grants : List Grant := []
  roots : List View := []
  retained : List View := []

def Binds (s : State) (r : View) : Prop :=
  r.owner ∈ s.grants ∧ 0 < r.size ∧ r.offset + r.size ≤ r.owner.size

instance (s : State) (r : View) : Decidable (Binds s r) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def Apart (g h : Grant) := g.base + g.size ≤ h.base ∨ h.base + h.size ≤ g.base

structure ArenaValid (s : State) : Prop where
  baseBound : s.base < 2^60
  capBound : s.capacity < 2^60
  floorBound : s.floor ≤ s.used
  cursor : s.used ≤ s.capacity
  covered : ∀ g ∈ s.grants, g.arena = s.arena ∧ g.generation < s.next ∧
    0 < g.size ∧ s.base + s.floor ≤ g.base ∧ g.base + g.size ≤ s.base + s.used
  separated : s.grants.Pairwise Apart
  generations : (s.grants.map Grant.generation).Nodup
  views : ∀ r ∈ s.roots ++ s.retained, Binds s r

def start (arena base capacity used : Nat) : State :=
  {arena, base, capacity, floor := used, used}

theorem start_valid (hb : b < 2^60) (hc : c < 2^60) (hu : u ≤ c) :
    ArenaValid (start arena b c u) :=
  ⟨hb, hc, Nat.le_refl _, hu, by simp [start], by simp [start], by simp [start], by simp [start]⟩

def issued (s : State) (n a : Nat) : Grant :=
  ⟨s.arena, s.next, result s.base s.used a, n⟩

private def install (s : State) (n a : Nat) : State :=
  {s with used := nextUsed s.base s.used n a, next := s.next + 1,
          grants := issued s n a :: s.grants}

-- Target pointer and post-cursor come independently from the native observer.
def admit (s : State) (n a ptr postUsed : Nat) : Option State := do
  guard (0 < n ∧ n < 2^60 ∧ (a = 8 ∨ a = 64))
  guard (ptr = result s.base s.used a ∧ postUsed = nextUsed s.base s.used n a ∧ postUsed ≤ s.capacity)
  return install s n a

private theorem admitted (ok : admit s n a ptr u = some t) :
    0 < n ∧ n < 2^60 ∧ (a = 8 ∨ a = 64) ∧ ptr = result s.base s.used a ∧
    u = nextUsed s.base s.used n a ∧ u ≤ s.capacity ∧ t = install s n a := by
  simp [admit, guard, Option.bind] at ok
  split at ok <;> simp_all
  split at ok <;> simp_all
  omega

theorem grant_sound (valid : ArenaValid s) (ok : admit s n a ptr u = some t) :
    ArenaValid t ∧ (issued s n a ∈ t.grants) ∧ s.base + s.used ≤ ptr ∧
    (∀ g ∈ s.grants, g.base + g.size ≤ ptr) ∧
    (∀ r ∈ s.roots ++ s.retained, r.owner.base + r.offset + r.size ≤ ptr) := by
  obtain ⟨pos,small,align,rfl,rfl,fit,rfl⟩ := admitted ok
  have h := ordinary _ _ _ _ _ valid.baseBound valid.capBound valid.cursor small pos align fit
  obtain ⟨_,lo,_,hi,_⟩ := h
  have old := fun (g : Grant) (hg : g ∈ s.grants) => Nat.le_trans (valid.covered g hg).2.2.2.2 lo
  have step : s.used ≤ nextUsed s.base s.used n a := by omega
  refine ⟨?_, by simp [install], lo, old, ?_⟩
  · refine ⟨valid.baseBound, valid.capBound, Nat.le_trans valid.floorBound step, fit, ?_, ?_, ?_, ?_⟩
    · intro g hg
      rcases List.mem_cons.mp hg with rfl | prior
      · exact ⟨rfl, Nat.lt_succ_self _, pos, Nat.le_trans (Nat.add_le_add_left valid.floorBound _) lo, hi⟩
      · obtain ⟨owner,serial,size,low,high⟩ := valid.covered g prior
        exact ⟨owner, Nat.lt_trans serial (Nat.lt_succ_self _),size,low,
          Nat.le_trans high (Nat.add_le_add_left step _)⟩
    · exact List.pairwise_cons.mpr ⟨fun g hg => Or.inr (old g hg), valid.separated⟩
    · simp only [install, List.map_cons, List.nodup_cons]
      refine ⟨?_, valid.generations⟩
      intro mem
      obtain ⟨g,hg,eq⟩ := List.mem_map.mp mem
      exact (Nat.ne_of_lt (valid.covered g hg).2.1) eq
    · intro r hr
      obtain ⟨owner,bounds⟩ := valid.views r hr
      exact ⟨List.mem_cons_of_mem _ owner,bounds⟩
  · intro r hr
    obtain ⟨owner,_,bounds⟩ := valid.views r hr
    have stop := old r.owner owner
    omega

-- A replay can carry validity without rechecking the accumulated grant list.
-- The proof fields are erased by Lean.
structure Checked where
  state : State
  valid : ArenaValid state

def beginChecked (arena base capacity used : Nat) : Option Checked :=
  if h : base < 2^60 ∧ capacity < 2^60 ∧ used ≤ capacity then
    some ⟨start arena base capacity used, start_valid h.1 h.2.1 h.2.2⟩
  else none

def Checked.bump (s : Checked) (n a ptr used : Nat) : Option Checked :=
  match ok : admit s.state n a ptr used with
  | none => none
  | some t => some ⟨t, (grant_sound s.valid ok).1⟩

def Checked.keep (s : Checked) (r : View) : Option Checked :=
  if h : Binds s.state r then
    some ⟨{s.state with roots := r :: s.state.roots}, {s.valid with
      views := by
        intro v hv
        rcases List.mem_cons.mp hv with rfl | prior
        · exact h
        · exact s.valid.views _ prior}⟩
  else none

-- Rewind only the suffix issued by this ledger. Native root completeness
-- and permission to rewind to the original floor are refinement premises.
def retire (s : State) : Option State :=
  if s.roots.isEmpty && s.retained.isEmpty then
    some {s with used := s.floor, grants := []}
  else none

theorem retire_sound (valid : ArenaValid s) (ok : retire s = some t) :
    ArenaValid t ∧ t.next = s.next ∧ t.used = s.floor ∧ t.grants = [] := by
  unfold retire at ok
  split at ok <;> simp_all
  rename_i empty
  cases ok
  exact ⟨⟨valid.baseBound, valid.capBound, Nat.le_refl _, Nat.le_trans valid.floorBound valid.cursor,
    by simp, by simp, by simp, by simp⟩, rfl, rfl, rfl⟩

-- Applies even when the next allocation has identical base and extent.
theorem no_revival (valid : ArenaValid s) (prior : g ∈ s.grants)
    (rewound : retire s = some t) (again : admit t n a ptr u = some v) :
    g ∉ v.grants := by
  obtain ⟨_,counter,_,empty⟩ := retire_sound valid rewound
  obtain ⟨_,_,_,_,_,_,rfl⟩ := admitted again
  have lt := (valid.covered g prior).2.1
  simp only [install, empty, List.mem_cons, List.not_mem_nil, or_false]
  intro eq
  have id := congrArg Grant.generation eq
  simp [issued, counter] at id
  omega

-- An already revoked grant stays revoked through any later sequence of
-- bumps and retirements. Equal addresses and equal extents do not suffice.
inductive Steps : State → State → Prop where
  | refl : Steps s s
  | bump : Steps s t → admit t n a p u = some v → Steps s v
  | rewind : Steps s t → retire t = some v → Steps s v

theorem revoked_stays_revoked (path : Steps s t) (valid : ArenaValid s)
    (old : g.generation < s.next) (absent : g ∉ s.grants) :
    ArenaValid t ∧ g.generation < t.next ∧ g ∉ t.grants := by
  induction path with
  | refl => exact ⟨valid,old,absent⟩
  | bump prev ok ih =>
    obtain ⟨safe,lt,missing⟩ := ih
    have nextValid := (grant_sound safe ok).1
    obtain ⟨_,_,_,_,_,_,rfl⟩ := admitted ok
    refine ⟨nextValid,Nat.lt_succ_of_lt lt,?_⟩
    simp only [install, List.mem_cons, not_or]
    exact ⟨fun eq => (Nat.ne_of_lt lt) (congrArg Grant.generation eq), missing⟩
  | rewind prev ok ih =>
    obtain ⟨safe,lt,_⟩ := ih
    obtain ⟨after,counter,_,empty⟩ := retire_sound safe ok
    exact ⟨after,counter ▸ lt,by simp [empty]⟩

end Verificatum.ArenaLedger
