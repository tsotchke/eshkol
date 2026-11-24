import ArenaBump

namespace Verificatum.NativeEffect.Allocation

structure Span where
  base : Nat
  size : Nat
  deriving DecidableEq, Repr

def Span.ends (s : Span) := s.base + s.size
def Span.apart (s t : Span) : Bool :=
  s.size == 0 || t.size == 0 || s.ends ≤ t.base || t.ends ≤ s.base

structure Block where
  key : Nat
  base : Nat
  capacity : Nat
  used : Nat
  deriving DecidableEq, Repr

-- Each client supplies the arena-owned byte spans and the values it observed.
-- Multiple entries may be aliases. Their order identifies the client's bindings.
structure Root where
  span : Span
  samples : List Nat
  deriving DecidableEq, Repr

structure Snapshot where
  active : List Block
  adopted : List Block
  roots : List Root
  deriving DecidableEq, Repr

structure Event where
  arena : Nat
  n : Nat
  alignment : Nat
  ptr : Nat
  pre : Snapshot
  post : Snapshot
  deriving Repr

def blocks (s : Snapshot) := s.active ++ s.adopted
def claimed (e : Event) : Span := ⟨e.ptr, e.n⟩
def live (s : Snapshot) : List Span :=
  s.roots.map Root.span ++ (blocks s).map fun b => ⟨b.base, b.used⟩

def alignment (e : Event) := if e.alignment == 0 then 8 else e.alignment
def roundUp (n a : Nat) := n + (a - n % a) % a
def limit := 2^64

private def disjoint : List Span → Bool
  | [] => true
  | s :: ss => ss.all s.apart && disjoint ss

def layoutOK (s : Snapshot) : Bool := Id.run do
  let bs := blocks s
  let spans := bs.flatMap fun b => [Span.mk b.key 32, Span.mk b.base b.capacity]
  return !s.active.isEmpty && bs.all (fun b => b.key > 0 && b.base > 0 &&
    b.used ≤ b.capacity && b.capacity > 0) &&
    spans.all (fun i => i.ends ≤ limit) && disjoint spans

def rootsLive (s : Snapshot) : Bool :=
  s.roots.all fun r => r.span.base > 0 && r.span.size > 0 &&
    (blocks s).any fun b => b.base ≤ r.span.base && r.span.ends ≤ b.base + b.used

private def granted (e : Event) (dest : Block) : Bool :=
  e.ptr > 0 && e.n > 0 && e.ptr % alignment e == 0 &&
    dest.base ≤ e.ptr && (claimed e).ends ≤ dest.base + dest.used &&
    dest.used ≤ dest.capacity && dest.base + dest.capacity ≤ limit

inductive Verdict where
  | bump | grow | failure | layout | stale | overlap | extent | arithmetic | binding
  deriving BEq, Repr, DecidableEq

-- Instantiate the existing native-word model only on its audited ordinary
-- profile. Growth never inherits the in-block theorem.
def machineBump (e : Event) (b dst : Block) : Bool :=
  b.base < 2^60 && b.capacity < 2^60 && b.used ≤ b.capacity && e.n < 2^60 && e.n > 0 &&
  (alignment e == 8 || (alignment e == 16 || alignment e == 64)) &&
  (live e.pre).all (fun s => s.ends ≤ b.base + b.used || b.base + b.capacity ≤ s.base) &&
  e.ptr == ArenaBump.result b.base b.used (alignment e) &&
  dst.used == ArenaBump.nextUsed b.base b.used e.n (alignment e) && dst.used ≤ b.capacity

theorem machine_grant_fresh (bound : machineBump e b dst = true) (prior : s ∈ live e.pre) :
    s.ends ≤ e.ptr ∨ e.ptr + e.n ≤ s.base := by
  simp only [machineBump, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at bound
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨hb,hc⟩,hu⟩,hn⟩,positive⟩,ha⟩,liveOK⟩,ptr⟩,used⟩,fits⟩ := bound
  rw [ptr]
  have sep := List.all_eq_true.mp liveOK s prior
  simp only [Bool.or_eq_true, decide_eq_true_eq] at sep
  have h := ArenaBump.ordinary_aligned _ _ _ _ _ hb hc hu hn positive ha (used ▸ fits)
  exact ArenaBump.apart _ _ _ _ _ _ sep h.2.1 (Nat.le_trans h.2.2.2.1 h.2.2.2.2)

-- The old active head can advance its cursor or become the second block.
-- The chain and the caller's observed bindings retain their identities.
def retainsLedger (pre post : Snapshot) : Bool :=
  pre.roots == post.roots && pre.adopted == post.adopted &&
  match pre.active, post.active with
  | old :: older, dst :: rest =>
    if old.key == dst.key then
      old.base == dst.base && old.capacity == dst.capacity && old.used ≤ dst.used && older == rest
    else rest == pre.active
  | _, _ => false

theorem retained_coverage (h : retainsLedger pre post = true) (b : Block) (hb : b ∈ blocks pre) :
    ∃ c ∈ blocks post, c.key = b.key ∧ c.base = b.base ∧ c.capacity = b.capacity ∧ b.used ≤ c.used := by
  simp only [retainsLedger, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨_,adopted⟩,active⟩ := h
  unfold blocks at hb ⊢
  rw [List.mem_append] at hb
  rcases hb with ha | hd
  · cases hp : pre.active with
    | nil => simp [hp] at ha
    | cons old older =>
      cases hq : post.active with
      | nil => simp [hp,hq] at active
      | cons dst rest =>
        simp only [hp,hq] at active
        split at active
        · rename_i key
          simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at active
          obtain ⟨⟨⟨base,cap⟩,used⟩,tailEq⟩ := active
          rw [hp,List.mem_cons] at ha
          rcases ha with rfl | tail
          · exact ⟨dst, by simp, key.symm, base.symm, cap.symm, used⟩
          · exact ⟨b, by simp [← tailEq,tail], rfl,rfl,rfl,Nat.le_refl _⟩
        · have r := beq_iff_eq.mp active
          exact ⟨b, by rw [hp] at ha; exact List.mem_append_left _ (List.mem_cons_of_mem _ (r.symm ▸ ha)), rfl,rfl,rfl,Nat.le_refl _⟩
  · exact ⟨b, List.mem_append_right _ (adopted ▸ hd), rfl,rfl,rfl,Nat.le_refl _⟩

private def transition (e : Event) (after : Block) : Verdict := Id.run do
  let before :: _ := e.pre.active | return .layout
  let a := alignment e
  if a ≥ limit || (a &&& (a-1)) != 0 then return .binding
  let reserve := roundUp e.n a
  let first := roundUp (before.base + before.used) a
  -- Native capacity tests use size_t. Admit the mathematical calculation
  -- only when the intermediate sums used by this path are representable.
  if e.n + (a-1) ≥ limit || before.base + before.used + (a-1) ≥ limit ||
     first - before.base + reserve ≥ limit then return .arithmetic
  if e.ptr + reserve != after.base + after.used then return .binding
  if before.key == after.key then
    if first != e.ptr then return .binding
    if !machineBump e before after then return .binding
    return .bump
  if reserve + (a-1) ≥ limit || after.base + (a-1) ≥ limit then return .arithmetic
  if first + reserve ≤ before.base + before.capacity || e.ptr != roundUp after.base a then return .binding
  return .grow

def assess (e : Event) : Verdict :=
  if e.ptr == 0 then
    if layoutOK e.pre && rootsLive e.pre && e.arena != 0 && e.pre == e.post then .failure else .binding
  else if (live e.pre).all (claimed e).apart then
    match e.post.active with
    | [] => .layout
    | dst :: _ => if granted e dst then
      if !layoutOK e.pre || !layoutOK e.post || e.arena == 0 then .layout
      else if !rootsLive e.pre || !rootsLive e.post || e.pre.roots != e.post.roots then .stale
      else if retainsLedger e.pre e.post then transition e dst else .binding
      else .extent
  else .overlap

-- The grant returned by a successful observation includes the complete requested
-- extent and excludes every byte of every recorded old live prefix.
theorem success_ranges (e : Event) (h : assess e = .bump ∨ assess e = .grow) :
    (live e.pre).all (claimed e).apart = true ∧
    ∃ dst tail, e.post.active = dst :: tail ∧ granted e dst = true := by
  unfold assess at h
  split at h
  · split at h <;> cases h <;> contradiction
  split at h
  · rename_i fresh
    split at h
    · cases h <;> contradiction
    · rename_i dst tail hdst
      split at h
      · exact ⟨fresh, dst, tail, hdst, by assumption⟩
      · cases h <;> contradiction
  · cases h <;> contradiction

theorem success_retains (e : Event) (h : assess e = .bump ∨ assess e = .grow) :
    retainsLedger e.pre e.post = true := by
  unfold assess at h
  split at h
  · split at h <;> cases h <;> contradiction
  split at h
  · split at h
    · cases h <;> contradiction
    · split at h
      · split at h
        · cases h <;> contradiction
        · split at h
          · cases h <;> contradiction
          · split at h
            · assumption
            · cases h <;> contradiction
      · cases h <;> contradiction
  · cases h <;> contradiction

theorem success_root_bindings (e : Event) (h : assess e = .bump ∨ assess e = .grow) :
    e.post.roots = e.pre.roots := by
  have kept := success_retains e h
  simp only [retainsLedger, Bool.and_eq_true, beq_iff_eq] at kept
  exact kept.1.1.symm

theorem live_byte_excluded (e : Event) (h : assess e = .bump ∨ assess e = .grow)
    (s : Span) (hs : s ∈ live e.pre) (p : Nat) (lo : s.base ≤ p) (hi : p < s.ends) :
    ¬(e.ptr ≤ p ∧ p < e.ptr + e.n) := by
  have sep := List.all_eq_true.mp (success_ranges e h).1 s hs
  simp [Span.apart, claimed, Span.ends] at sep
  unfold Span.ends at hi
  rcases sep with ((hz | hz) | hd) | hd
  · omega
  · omega
  · have := of_decide_eq_true hd
    omega
  · have := of_decide_eq_true hd
    omega

-- The consumer must justify this coverage for the native roots at the call.
-- Observed spans and the contents at the two endpoints alone cannot supply it.
theorem reachable_byte_excluded (e : Event) (accepted : assess e = .bump ∨ assess e = .grow)
    (reachable : Nat → Prop)
    (coverage : ∀ p, reachable p → ∃ s ∈ live e.pre, s.base ≤ p ∧ p < s.ends)
    (p : Nat) (owned : reachable p) : ¬(e.ptr ≤ p ∧ p < e.ptr + e.n) := by
  obtain ⟨s,member,lo,hi⟩ := coverage p owned
  exact live_byte_excluded e accepted s member p lo hi

end Verificatum.NativeEffect.Allocation
