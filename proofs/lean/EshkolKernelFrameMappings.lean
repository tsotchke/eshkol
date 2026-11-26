/-
  EshkolKernelFrameMappings.lean
  The composition of the K2 frame pool and address spaces of tsotchke/eshkol
  (lib/kernel/k2_mmu_frames.c), with native's status codes and check order, as of
  the partition of kernel and user ASIDs and per-frame user ownership.

  Hidden decision: the actual mapping relation is a global list of entries
  (space, page, frame, permissions). Native's slot arrays, PTE encoding, software
  TLB and page-table capacity are abstracted away. The x86-64 encoding is reduced
  to the one check it adds after the permission policy: a kernel-space entry must
  not carry the user bit. Page-table exhaustion (512 entries) and non-canonical or
  misaligned addresses are outside the model.

  Native keeps a per-frame summary of the relation (k2_frame_meta). It is modelled
  as a recorded field, separate from the relation itself:
    refcount       the number of entries of the frame
    userRefcount   the number of those entries in user spaces
    reads, writes, execs   the number of those entries with R, W, X
    perms          R, W, X recomputed from the three counts
    ownerAsid, ownerSpace  the ASID and the identity of the user space that maps
                   the frame, while userRefcount is positive
  Each recorded field has its own named relation to the actual entries below.

  Each entry also carries a ghost field, the frame generation at mapping time,
  which native does not store. It expresses temporal safety.

  Invariants (named separately; Coherent is their conjunction):
    RefcountExact, UserRefcountExact, ReadCountExact, WriteCountExact,
    ExecCountExact, PermsExact, OwnerExact, UserEntriesOwned,
    ReferentialIntegrity, KernelFramesNotInUser, EntriesSatisfyPolicy,
    NoWriteExecuteAlias, GenerationCurrent, PageUnique.
  Guarantee: Coherent → SingleUserSpace: all user entries of a frame lie in one
  address space. No assumption about ASIDs is made; MutableIsolation, for spaces
  with different ASIDs, and isolation of distinct spaces that share an ASID both
  follow.
-/

namespace EshkolKernelFrameMappings

inductive Owner where
  | free | kernel | user
  deriving DecidableEq, Repr

structure Perms where
  r : Bool
  w : Bool
  x : Bool
  u : Bool
  k : Bool
  deriving DecidableEq, Repr

def Perms.empty : Perms := ⟨false, false, false, false, false⟩

/-- Native `k2_meta_recompute_perms`. -/
def Perms.ofCounts (reads writes execs : Nat) : Perms :=
  ⟨decide (0 < reads), decide (0 < writes), decide (0 < execs), false, false⟩

inductive Status where
  | ok | invalidArg | outOfFrames | doubleFree | staleHandle | kernelFrameLocked
  | frameStillMapped | wxViolation | privViolation | kernelAlias | permAlias
  | vaAlreadyMapped | vaNotMapped | conservationViolation
  deriving DecidableEq, Repr

/-- Native `k2_check_perm_policy`. -/
def policy (p : Perms) (userSpace : Bool) : Status :=
  if p.w && p.x then .wxViolation
  else if p.u && p.k then .privViolation
  else if !(p.r || p.w || p.x) then .invalidArg
  else if p.w && !p.r then .invalidArg
  else if userSpace && (p.k || !p.u) then .privViolation
  else .ok

/-- What native's x86-64 encode and validate add once the policy has passed. -/
def encodeCheck (p : Perms) (userSpace : Bool) : Status :=
  if !userSpace && p.u then .privViolation else .ok

def kernelAsid : Nat := 0
def invalidIndex : Nat := 0xFFFFFFFF

structure Space where
  asid : Nat
  user : Bool
  deriving DecidableEq, Repr

/-- Native `k2_aspace_init`'s ASID partition: kernel spaces use ASID 0, user spaces
    any other valid ASID. -/
def Space.initStatus (sp : Space) : Status :=
  if sp.user = false then (if sp.asid = kernelAsid then .ok else .invalidArg)
  else if sp.asid = kernelAsid ∨ sp.asid = invalidIndex then .invalidArg else .ok

structure Summary where
  refcount     : Nat
  userRefcount : Nat
  reads        : Nat
  writes       : Nat
  execs        : Nat
  perms        : Perms
  ownerAsid    : Option Nat
  ownerSpace   : Option Nat
  deriving DecidableEq, Repr

/-- Native `k2_meta_reset_mappings`. -/
def Summary.reset : Summary := ⟨0, 0, 0, 0, 0, Perms.empty, none, none⟩

/-- Native's guarded decrement `if (bit && c > 0) c -= 1`. -/
def decIf (b : Bool) (c : Nat) : Nat := if b = true ∧ 0 < c then c - 1 else c

def bitCount (b : Bool) : Nat := if b = true then 1 else 0

/-- The summary after a successful `k2_aspace_map` in space `s` with ASID `asid`. -/
def Summary.addMapping (m : Summary) (p : Perms) (user : Bool) (s asid : Nat) : Summary :=
  { refcount := m.refcount + 1,
    userRefcount := if user = true then m.userRefcount + 1 else m.userRefcount,
    reads := m.reads + bitCount p.r,
    writes := m.writes + bitCount p.w,
    execs := m.execs + bitCount p.x,
    perms := Perms.ofCounts (m.reads + bitCount p.r) (m.writes + bitCount p.w)
               (m.execs + bitCount p.x),
    ownerAsid := if user = true then some asid else m.ownerAsid,
    ownerSpace := if user = true then some s else m.ownerSpace }

/-- The summary after a successful `k2_aspace_remap_perms` from `old` to `new`. -/
def Summary.replacePerms (m : Summary) (old new : Perms) : Summary :=
  { m with
    reads := decIf old.r m.reads + bitCount new.r,
    writes := decIf old.w m.writes + bitCount new.w,
    execs := decIf old.x m.execs + bitCount new.x,
    perms := Perms.ofCounts (decIf old.r m.reads + bitCount new.r)
               (decIf old.w m.writes + bitCount new.w) (decIf old.x m.execs + bitCount new.x) }

def Summary.userAfterRemove (m : Summary) (user : Bool) : Nat :=
  if user = true ∧ 0 < m.userRefcount then m.userRefcount - 1 else m.userRefcount

def Summary.ownerCleared (m : Summary) (user : Bool) : Prop :=
  user = true ∧ 0 < m.userRefcount ∧ m.userRefcount - 1 = 0

instance (m : Summary) (user : Bool) : Decidable (m.ownerCleared user) := by
  unfold Summary.ownerCleared; infer_instance

/-- The summary after `k2_aspace_unmap` removes an entry with permissions `p`. -/
def Summary.removeMapping (m : Summary) (p : Perms) (user : Bool) : Summary :=
  if 0 < m.refcount then
    if m.refcount - 1 = 0 then Summary.reset
    else { refcount := m.refcount - 1,
           userRefcount := m.userAfterRemove user,
           reads := decIf p.r m.reads, writes := decIf p.w m.writes, execs := decIf p.x m.execs,
           perms := Perms.ofCounts (decIf p.r m.reads) (decIf p.w m.writes) (decIf p.x m.execs),
           ownerAsid := if m.ownerCleared user then none else m.ownerAsid,
           ownerSpace := if m.ownerCleared user then none else m.ownerSpace }
  else { m with
         userRefcount := m.userAfterRemove user,
         reads := decIf p.r m.reads, writes := decIf p.w m.writes, execs := decIf p.x m.execs,
         ownerAsid := if m.ownerCleared user then none else m.ownerAsid,
         ownerSpace := if m.ownerCleared user then none else m.ownerSpace }

structure Frame where
  owner      : Owner
  generation : Nat
  summary    : Summary
  deriving DecidableEq, Repr

structure Entry where
  space      : Nat
  page       : Nat
  frame      : Nat
  perms      : Perms
  generation : Nat
  deriving DecidableEq, Repr

structure Handle where
  idx    : Nat
  gen    : Nat
  physOk : Bool
  deriving DecidableEq, Repr

structure System where
  total     : Nat
  freeCount : Nat
  allocated : Nat
  frames    : Nat → Frame
  spaces    : List Space
  entries   : List Entry

def setFrame (f : Nat → Frame) (i : Nat) (v : Frame) : Nat → Frame :=
  fun j => if j = i then v else f j

/-- The initial state over spaces that `k2_aspace_init` accepted. -/
def System.init (total kernelReserved : Nat) (spaces : List Space) : System :=
  { total, freeCount := total - kernelReserved, allocated := kernelReserved,
    frames := fun i => ⟨if i < kernelReserved then .kernel else .free, 1, Summary.reset⟩,
    spaces, entries := [] }

def Entry.at (s page : Nat) (e : Entry) : Bool := e.space == s && e.page == page

/-- The entry after remapping (s, page) to permissions p. -/
def remapEntry (s page : Nat) (p : Perms) (e : Entry) : Entry :=
  if Entry.at s page e then { e with perms := p } else e

/-- `k2_frame_alloc_user` / `k2_frame_alloc_kernel` at the frame native chose. -/
def System.alloc (S : System) (o : Owner) (i : Nat) : Status × System :=
  if i < S.total ∧ (S.frames i).owner = .free ∧ 0 < S.freeCount then
    (.ok, { S with freeCount := S.freeCount - 1, allocated := S.allocated + 1,
                   frames := setFrame S.frames i ⟨o, (S.frames i).generation + 1, Summary.reset⟩ })
  else (.outOfFrames, S)

/-- `k2_frame_free_user`. -/
def System.free (S : System) (h : Handle) : Status × System :=
  if S.total ≤ h.idx then (.invalidArg, S)
  else
    let f := S.frames h.idx
    if f.owner = .kernel then (.kernelFrameLocked, S)
    else if f.owner = .free then (.doubleFree, S)
    else if h.gen ≠ f.generation then (.staleHandle, S)
    else if h.physOk = false then (.staleHandle, S)
    else if f.summary.refcount ≠ 0 then (.frameStillMapped, S)
    else (.ok, { S with freeCount := S.freeCount + 1, allocated := S.allocated - 1,
                        frames := setFrame S.frames h.idx ⟨.free, f.generation + 1, Summary.reset⟩ })

/-- `k2_aspace_map`. -/
def System.map (S : System) (s page : Nat) (h : Handle) (p : Perms) : Status × System :=
  match S.spaces[s]? with
  | none => (.invalidArg, S)
  | some sp =>
    if policy p sp.user ≠ .ok then (policy p sp.user, S)
    else if S.total ≤ h.idx then (.invalidArg, S)
    else
      let f := S.frames h.idx
      if f.owner = .kernel ∧ sp.user = true then (.kernelAlias, S)
      else if f.owner = .free then (.staleHandle, S)
      else if h.gen ≠ f.generation then (.staleHandle, S)
      else if h.physOk = false then (.staleHandle, S)
      else if 0 < f.summary.refcount ∧
          ((0 < f.summary.writes ∧ p.x = true) ∨ (0 < f.summary.execs ∧ p.w = true)) then
        (.permAlias, S)
      else if sp.user = true ∧ 0 < f.summary.userRefcount ∧
          (f.summary.ownerAsid ≠ some sp.asid ∨ f.summary.ownerSpace ≠ some s) then
        (.permAlias, S)
      else if S.entries.any (Entry.at s page) then (.vaAlreadyMapped, S)
      else if encodeCheck p sp.user ≠ .ok then (encodeCheck p sp.user, S)
      else (.ok, { S with
        entries := S.entries ++ [⟨s, page, h.idx, p, f.generation⟩],
        frames := setFrame S.frames h.idx
          { f with summary := f.summary.addMapping p sp.user s sp.asid } })

/-- `k2_aspace_remap_perms`. -/
def System.remap (S : System) (s page : Nat) (p : Perms) : Status × System :=
  match S.spaces[s]? with
  | none => (.invalidArg, S)
  | some sp =>
    if policy p sp.user ≠ .ok then (policy p sp.user, S)
    else
      match S.entries.find? (Entry.at s page) with
      | none => (.vaNotMapped, S)
      | some e =>
        if S.total ≤ e.frame then (.conservationViolation, S)
        else
          let f := S.frames e.frame
          if (0 < decIf e.perms.w f.summary.writes ∧ p.x = true) ∨
             (0 < decIf e.perms.x f.summary.execs ∧ p.w = true) then (.permAlias, S)
          else if encodeCheck p sp.user ≠ .ok then (encodeCheck p sp.user, S)
          else (.ok, { S with
            entries := S.entries.map (remapEntry s page p),
            frames := setFrame S.frames e.frame
              { f with summary := f.summary.replacePerms e.perms p } })

/-- `k2_aspace_unmap`. -/
def System.unmap (S : System) (s page : Nat) : Status × System :=
  match S.spaces[s]? with
  | none => (.invalidArg, S)
  | some sp =>
    match S.entries.find? (Entry.at s page) with
    | none => (.vaNotMapped, S)
    | some e =>
      (.ok, { S with
        entries := S.entries.filter (fun e' => !Entry.at s page e'),
        frames := if e.frame < S.total then
            setFrame S.frames e.frame
              { S.frames e.frame with
                summary := (S.frames e.frame).summary.removeMapping e.perms sp.user }
          else S.frames })

inductive Op where
  | alloc (o : Owner) (i : Nat)
  | free (h : Handle)
  | map (s page : Nat) (h : Handle) (p : Perms)
  | remap (s page : Nat) (p : Perms)
  | unmap (s page : Nat)

def System.step (S : System) : Op → Status × System
  | .alloc o i => S.alloc o i
  | .free h => S.free h
  | .map s page h p => S.map s page h p
  | .remap s page p => S.remap s page p
  | .unmap s page => S.unmap s page

def System.run : System → List Op → System
  | S, [] => S
  | S, op :: ops => (S.step op).2.run ops

/-! ## The actual relation and native's summaries -/

/-- The number of entries of frame `i` that satisfy `q`. -/
def countIn (l : List Entry) (i : Nat) (q : Entry → Bool) : Nat :=
  l.countP (fun e => e.frame == i && q e)

def System.spaceOf (S : System) (e : Entry) : Option Space := S.spaces[e.space]?

def System.isUserEntry (S : System) (e : Entry) : Bool :=
  match S.spaces[e.space]? with
  | some sp => sp.user
  | none => false

/-- A recorded count equals the actual count of the frame's entries satisfying `q`. -/
def System.CountExact (S : System) (sel : Summary → Nat) (q : Entry → Bool) : Prop :=
  ∀ i, sel (S.frames i).summary = countIn S.entries i q

def System.RefcountExact (S : System) : Prop := S.CountExact (·.refcount) (fun _ => true)
def System.UserRefcountExact (S : System) : Prop := S.CountExact (·.userRefcount) S.isUserEntry
def System.ReadCountExact (S : System) : Prop := S.CountExact (·.reads) (·.perms.r)
def System.WriteCountExact (S : System) : Prop := S.CountExact (·.writes) (·.perms.w)
def System.ExecCountExact (S : System) : Prop := S.CountExact (·.execs) (·.perms.x)

/-- The recorded permissions are those native recomputes from the recorded counts. -/
def System.PermsExact (S : System) : Prop :=
  ∀ i, (S.frames i).summary.perms =
    Perms.ofCounts (S.frames i).summary.reads (S.frames i).summary.writes (S.frames i).summary.execs

/-- The recorded owner is present exactly while user entries exist, and names a user
    space together with that space's ASID. -/
def System.OwnerExact (S : System) : Prop :=
  ∀ i, ((S.frames i).summary.userRefcount = 0 →
          (S.frames i).summary.ownerSpace = none ∧ (S.frames i).summary.ownerAsid = none) ∧
       (0 < (S.frames i).summary.userRefcount →
          ∃ s sp, (S.frames i).summary.ownerSpace = some s ∧ S.spaces[s]? = some sp ∧
            sp.user = true ∧ (S.frames i).summary.ownerAsid = some sp.asid)

/-- Every user entry lies in its frame's recorded owner space. -/
def System.UserEntriesOwned (S : System) : Prop :=
  ∀ e ∈ S.entries, S.isUserEntry e = true → (S.frames e.frame).summary.ownerSpace = some e.space

def System.ReferentialIntegrity (S : System) : Prop :=
  ∀ e ∈ S.entries, e.frame < S.total ∧ (S.frames e.frame).owner ≠ .free

def System.KernelFramesNotInUser (S : System) : Prop :=
  ∀ e ∈ S.entries, S.isUserEntry e = true → (S.frames e.frame).owner ≠ .kernel

def System.EntriesSatisfyPolicy (S : System) : Prop :=
  ∀ e ∈ S.entries, ∃ sp, S.spaceOf e = some sp ∧
    policy e.perms sp.user = .ok ∧ encodeCheck e.perms sp.user = .ok

def System.NoWriteExecuteAlias (S : System) : Prop :=
  ∀ a ∈ S.entries, ∀ b ∈ S.entries, a.frame = b.frame → ¬ (a.perms.w = true ∧ b.perms.x = true)

def System.GenerationCurrent (S : System) : Prop :=
  ∀ e ∈ S.entries, e.generation = (S.frames e.frame).generation

def System.PageUnique (S : System) : Prop :=
  S.entries.Pairwise (fun a b => ¬ (a.space = b.space ∧ a.page = b.page))

def System.Coherent (S : System) : Prop :=
  S.RefcountExact ∧ S.UserRefcountExact ∧ S.ReadCountExact ∧ S.WriteCountExact ∧
  S.ExecCountExact ∧ S.PermsExact ∧ S.OwnerExact ∧ S.UserEntriesOwned ∧
  S.ReferentialIntegrity ∧ S.KernelFramesNotInUser ∧ S.EntriesSatisfyPolicy ∧
  S.NoWriteExecuteAlias ∧ S.GenerationCurrent ∧ S.PageUnique

/-- The guarantee: all user entries of a frame lie in one address space. -/
def System.SingleUserSpace (S : System) : Prop :=
  ∀ a ∈ S.entries, ∀ b ∈ S.entries, S.isUserEntry a = true → S.isUserEntry b = true →
    a.frame = b.frame → a.space = b.space

/-- No frame is mapped in two user ASIDs if either mapping is writable. -/
def System.MutableIsolation (S : System) : Prop :=
  ∀ a ∈ S.entries, ∀ b ∈ S.entries, ∀ sa sb, S.spaceOf a = some sa → S.spaceOf b = some sb →
    sa.user = true → sb.user = true → a.frame = b.frame → sa.asid ≠ sb.asid →
    a.perms.w = false ∧ b.perms.w = false

/-- Two distinct user spaces never share a frame, whether or not their ASIDs agree. -/
def System.SpaceIsolation (S : System) : Prop :=
  ∀ a ∈ S.entries, ∀ b ∈ S.entries, S.isUserEntry a = true → S.isUserEntry b = true →
    a.space ≠ b.space → a.frame ≠ b.frame

/-! ## Counting lemmas -/

theorem setFrame_apply (f : Nat → Frame) (i : Nat) (v : Frame) (j : Nat) :
    setFrame f i v j = if j = i then v else f j := rfl

theorem countIn_append (l : List Entry) (e : Entry) (i : Nat) (q : Entry → Bool) :
    countIn (l ++ [e]) i q = countIn l i q + (if (e.frame == i && q e) = true then 1 else 0) := by
  unfold countIn
  rw [List.countP_append]
  simp [List.countP_cons, List.countP_nil]

theorem countIn_nil (i : Nat) (q : Entry → Bool) : countIn [] i q = 0 := rfl

theorem countIn_le (l : List Entry) (i : Nat) (q : Entry → Bool) :
    countIn l i q ≤ countIn l i (fun _ => true) := by
  unfold countIn
  exact List.countP_mono_left (fun x _ hx => by simp_all)

theorem countIn_pos (l : List Entry) (e : Entry) (he : e ∈ l) (q : Entry → Bool) (hq : q e = true) :
    0 < countIn l e.frame q := by
  unfold countIn
  exact (List.countP_pos (p := fun x => x.frame == e.frame && q x)).mpr ⟨e, he, by simp [hq]⟩

theorem countIn_zero_of_none (l : List Entry) (i : Nat) (q : Entry → Bool)
    (h : ∀ e ∈ l, e.frame ≠ i) : countIn l i q = 0 := by
  unfold countIn
  rw [List.countP_eq_zero]
  intro e he
  simp [h e he]

theorem none_of_countIn_zero (l : List Entry) (i : Nat) (q : Entry → Bool)
    (h : countIn l i q = 0) : ∀ e ∈ l, e.frame = i → q e = false := by
  unfold countIn at h
  rw [List.countP_eq_zero] at h
  intro e he hf
  have := h e he
  cases hq : q e
  · rfl
  · simp [hf, hq] at this

/-- Removing the unique element with a given key lowers every count by that element's share. -/
theorem countP_filter_unique {α : Type} (q key : α → Bool) : ∀ (l : List α) (e : α), e ∈ l →
    key e = true → l.Pairwise (fun a b => ¬ (key a = true ∧ key b = true)) →
    (l.filter (fun x => !key x)).countP q + (if q e then 1 else 0) = l.countP q
  | [], _, he, _, _ => absurd he (List.not_mem_nil _)
  | x :: t, e, he, hk, hp => by
    rw [List.pairwise_cons] at hp
    obtain ⟨hx, ht⟩ := hp
    rcases List.mem_cons.mp he with rfl | he'
    · have hrest : ∀ y ∈ t, key y = false := by
        intro y hy
        cases hky : key y
        · rfl
        · exact absurd ⟨hk, hky⟩ (hx y hy)
      have hf : t.filter (fun x => !key x) = t :=
        List.filter_eq_self.mpr (fun y hy => by simp [hrest y hy])
      simp [List.filter_cons, hk, hf, List.countP_cons]
    · have hkx : key x = false := by
        cases hkx : key x
        · rfl
        · exact absurd ⟨hkx, hk⟩ (hx e he')
      have ih := countP_filter_unique q key t e he' hk ht
      simp only [List.filter_cons, hkx, Bool.not_false, if_true, List.countP_cons]
      omega

/-- Mapping a function that changes only the unique element with a given key. -/
theorem countP_map_unique {α : Type} (q key : α → Bool) (g : α → α)
    (hg : ∀ x, key x = false → g x = x) : ∀ (l : List α) (e : α), e ∈ l →
    key e = true → l.Pairwise (fun a b => ¬ (key a = true ∧ key b = true)) →
    (l.map g).countP q + (if q e then 1 else 0) = l.countP q + (if q (g e) then 1 else 0)
  | [], _, he, _, _ => absurd he (List.not_mem_nil _)
  | x :: t, e, he, hk, hp => by
    rw [List.pairwise_cons] at hp
    obtain ⟨hx, ht⟩ := hp
    rcases List.mem_cons.mp he with rfl | he'
    · have hrest : ∀ y ∈ t, key y = false := by
        intro y hy
        cases hky : key y
        · rfl
        · exact absurd ⟨hk, hky⟩ (hx y hy)
      have hm : t.map g = t := by
        rw [List.map_congr_left (g := id) (fun y hy => hg y (hrest y hy)), List.map_id]
      rw [List.map_cons, hm, List.countP_cons, List.countP_cons]
      omega
    · have hkx : key x = false := by
        cases hkx : key x
        · rfl
        · exact absurd ⟨hkx, hk⟩ (hx e he')
      have ih := countP_map_unique q key g hg t e he' hk ht
      rw [List.map_cons, hg x hkx, List.countP_cons, List.countP_cons]
      omega

theorem pageUnique_key (S : System) (hU : S.PageUnique) (s page : Nat) :
    S.entries.Pairwise (fun a b => ¬ (Entry.at s page a = true ∧ Entry.at s page b = true)) :=
  hU.imp (fun {a b} hab hk => hab ⟨by
      simp [Entry.at] at hk; omega, by simp [Entry.at] at hk; omega⟩)

theorem countIn_remove (S : System) (hU : S.PageUnique) (s page : Nat) (e : Entry)
    (he : e ∈ S.entries) (hk : Entry.at s page e = true) (i : Nat) (q : Entry → Bool) :
    countIn (S.entries.filter (fun x => !Entry.at s page x)) i q +
      (if (e.frame == i && q e) = true then 1 else 0) = countIn S.entries i q :=
  countP_filter_unique (fun x : Entry => x.frame == i && q x) (Entry.at s page) S.entries e he hk
    (pageUnique_key S hU s page)

theorem remapEntry_of_not_at (s page : Nat) (p : Perms) (x : Entry) (h : Entry.at s page x = false) :
    remapEntry s page p x = x := by
  simp [remapEntry, h]

theorem countIn_remap (S : System) (hU : S.PageUnique) (s page : Nat) (p : Perms) (e : Entry)
    (he : e ∈ S.entries) (hk : Entry.at s page e = true) (i : Nat) (q : Entry → Bool) :
    countIn (S.entries.map (remapEntry s page p)) i q +
      (if (e.frame == i && q e) = true then 1 else 0) =
    countIn S.entries i q +
      (if ((remapEntry s page p e).frame == i && q (remapEntry s page p e)) = true then 1 else 0) :=
  countP_map_unique (fun x : Entry => x.frame == i && q x) (Entry.at s page) (remapEntry s page p)
    (remapEntry_of_not_at s page p) S.entries e he hk (pageUnique_key S hU s page)

/-! ## Small lemmas -/

theorem policy_ok_not_wx (p : Perms) (u : Bool) (h : policy p u = .ok) :
    ¬ (p.w = true ∧ p.x = true) := by
  rintro ⟨hw, hx⟩
  simp [policy, hw, hx] at h

theorem pairwise_mem_ne {α : Type} {R : α → α → Prop} : ∀ {l : List α}, l.Pairwise R →
    ∀ {a b : α}, a ∈ l → b ∈ l → a ≠ b → R a b ∨ R b a
  | [], _, _, _, ha, _, _ => absurd ha (List.not_mem_nil _)
  | x :: t, hp, a, b, ha, hb, hne => by
    rw [List.pairwise_cons] at hp
    rcases List.mem_cons.mp ha with rfl | ha' <;> rcases List.mem_cons.mp hb with rfl | hb'
    · exact absurd rfl hne
    · exact Or.inl (hp.1 b hb')
    · exact Or.inr (hp.1 a ha')
    · exact pairwise_mem_ne hp.2 ha' hb' hne

/-- Under PageUnique, the entry found at (s, page) is the only one there. -/
theorem at_unique (S : System) (hU : S.PageUnique) (s page : Nat) (e x : Entry)
    (he : e ∈ S.entries) (hk : Entry.at s page e = true)
    (hx : x ∈ S.entries) (hkx : Entry.at s page x = true) : x = e := by
  by_cases hne : x = e
  · exact hne
  · have hk' : e.space = s ∧ e.page = page := by simp [Entry.at] at hk; exact hk
    have hkx' : x.space = s ∧ x.page = page := by simp [Entry.at] at hkx; exact hkx
    rcases pairwise_mem_ne hU hx he hne with h | h
    · exact absurd ⟨by omega, by omega⟩ h
    · exact absurd ⟨by omega, by omega⟩ h

theorem setFrame_summary_owner (F : Nat → Frame) (t : Nat) (m : Summary) (j : Nat) :
    (setFrame F t { F t with summary := m } j).owner = (F j).owner := by
  unfold setFrame
  by_cases h : j = t
  · subst h; simp
  · simp [h]

theorem setFrame_summary_generation (F : Nat → Frame) (t : Nat) (m : Summary) (j : Nat) :
    (setFrame F t { F t with summary := m } j).generation = (F j).generation := by
  unfold setFrame
  by_cases h : j = t
  · subst h; simp
  · simp [h]

theorem setFrame_summary_other (F : Nat → Frame) (t : Nat) (v : Frame) (j : Nat) (h : j ≠ t) :
    setFrame F t v j = F j := by
  simp [setFrame, h]

theorem setFrame_summary_self (F : Nat → Frame) (t : Nat) (v : Frame) :
    setFrame F t v t = v := by
  simp [setFrame]

theorem ofCounts_zero : Perms.ofCounts 0 0 0 = Perms.empty := rfl

theorem decIf_spec (b : Bool) (c : Nat) (h : b = true → 0 < c) :
    decIf b c + bitCount b = c := by
  cases b
  · simp [decIf, bitCount]
  · have := h rfl
    simp [decIf, bitCount, this]
    omega

theorem remapEntry_space (s page : Nat) (p : Perms) (e : Entry) :
    (remapEntry s page p e).space = e.space := by unfold remapEntry; split <;> rfl
theorem remapEntry_page (s page : Nat) (p : Perms) (e : Entry) :
    (remapEntry s page p e).page = e.page := by unfold remapEntry; split <;> rfl
theorem remapEntry_frame (s page : Nat) (p : Perms) (e : Entry) :
    (remapEntry s page p e).frame = e.frame := by unfold remapEntry; split <;> rfl
theorem remapEntry_generation (s page : Nat) (p : Perms) (e : Entry) :
    (remapEntry s page p e).generation = e.generation := by unfold remapEntry; split <;> rfl

/-! ## The initial state, allocation and release -/

theorem init_coherent (total kernelReserved : Nat) (spaces : List Space) :
    (System.init total kernelReserved spaces).Coherent := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    first
    | (intro i; rfl)
    | (intro i; exact ⟨fun _ => ⟨rfl, rfl⟩, fun h => absurd h (Nat.lt_irrefl 0)⟩)
    | (intro e he; exact absurd he (List.not_mem_nil _))
    | exact List.Pairwise.nil

/-- Replacing frame `i`, which no entry references, by a frame with no mappings. -/
theorem coherent_of_untouched (S : System) (i : Nat) (v : Frame) (fc al : Nat)
    (h : S.Coherent) (hnone : ∀ e ∈ S.entries, e.frame ≠ i) (hv : v.summary = Summary.reset) :
    ({ S with freeCount := fc, allocated := al, frames := setFrame S.frames i v } : System).Coherent := by
  obtain ⟨hR, hUR, hRd, hWr, hEx, hPE, hOE, hUO, hRI, hKU, hESP, hNWX, hGC, hPU⟩ := h
  have same : ∀ e ∈ S.entries, setFrame S.frames i v e.frame = S.frames e.frame :=
    fun e he => setFrame_summary_other _ _ _ _ (hnone e he)
  have cnt : ∀ (sel : Summary → Nat) (q : Entry → Bool), sel Summary.reset = 0 →
      S.CountExact sel q →
      ({ S with freeCount := fc, allocated := al, frames := setFrame S.frames i v } : System).CountExact sel q := by
    intro sel q h0 hc j
    show sel (setFrame S.frames i v j).summary = countIn S.entries j q
    by_cases hj : j = i
    · subst hj
      rw [setFrame_summary_self, hv, h0, countIn_zero_of_none _ _ _ hnone]
    · rw [setFrame_summary_other _ _ _ _ hj]; exact hc j
  refine ⟨cnt _ _ rfl hR, cnt _ _ rfl hUR, cnt _ _ rfl hRd, cnt _ _ rfl hWr, cnt _ _ rfl hEx,
    ?_, ?_, ?_, ?_, ?_, hESP, ?_, ?_, hPU⟩
  · intro j
    show (setFrame S.frames i v j).summary.perms = Perms.ofCounts (setFrame S.frames i v j).summary.reads
      (setFrame S.frames i v j).summary.writes (setFrame S.frames i v j).summary.execs
    by_cases hj : j = i
    · subst hj; rw [setFrame_summary_self, hv]; rfl
    · rw [setFrame_summary_other _ _ _ _ hj]; exact hPE j
  · intro j
    show ((setFrame S.frames i v j).summary.userRefcount = 0 →
          (setFrame S.frames i v j).summary.ownerSpace = none ∧
          (setFrame S.frames i v j).summary.ownerAsid = none) ∧
       (0 < (setFrame S.frames i v j).summary.userRefcount →
          ∃ s sp, (setFrame S.frames i v j).summary.ownerSpace = some s ∧ S.spaces[s]? = some sp ∧
            sp.user = true ∧ (setFrame S.frames i v j).summary.ownerAsid = some sp.asid)
    by_cases hj : j = i
    · subst hj; rw [setFrame_summary_self, hv]
      exact ⟨fun _ => ⟨rfl, rfl⟩, fun h => absurd h (Nat.lt_irrefl 0)⟩
    · rw [setFrame_summary_other _ _ _ _ hj]; exact hOE j
  · intro e he hu; show (setFrame S.frames i v e.frame).summary.ownerSpace = some e.space
    rw [same e he]; exact hUO e he hu
  · intro e he; show e.frame < S.total ∧ (setFrame S.frames i v e.frame).owner ≠ .free
    rw [same e he]; exact hRI e he
  · intro e he hu; show (setFrame S.frames i v e.frame).owner ≠ .kernel
    rw [same e he]; exact hKU e he hu
  · exact hNWX
  · intro e he; show e.generation = (setFrame S.frames i v e.frame).generation
    rw [same e he]; exact hGC e he

theorem alloc_preserves (S : System) (o : Owner) (i : Nat) (h : S.Coherent) :
    (S.alloc o i).2.Coherent := by
  unfold System.alloc
  split
  · rename_i hc
    obtain ⟨_, hfree, _⟩ := hc
    refine coherent_of_untouched S i _ _ _ h ?_ rfl
    intro e he heq
    have := (h.2.2.2.2.2.2.2.2.1 e he).2
    rw [heq] at this
    exact this hfree
  · exact h

theorem free_preserves (S : System) (hd : Handle) (h : S.Coherent) : (S.free hd).2.Coherent := by
  unfold System.free
  split
  · exact h
  dsimp only
  split
  · exact h
  split
  · exact h
  split
  · exact h
  split
  · exact h
  split
  · exact h
  rename_i hrc
  have hz : (S.frames hd.idx).summary.refcount = 0 := Decidable.of_not_not hrc
  refine coherent_of_untouched S hd.idx _ _ _ h ?_ rfl
  intro e he heq
  have hc : (S.frames hd.idx).summary.refcount = countIn S.entries hd.idx (fun _ => true) := h.1 hd.idx
  rw [hz] at hc
  have := none_of_countIn_zero S.entries hd.idx (fun _ => true) hc.symm e he heq
  exact Bool.noConfusion this

/-! ## Mapping -/

theorem map_success (S : System) (s page : Nat) (hd : Handle) (p : Perms) (sp : Space)
    (hc : S.Coherent) (hsp : S.spaces[s]? = some sp) (hpol : policy p sp.user = .ok)
    (hidx : hd.idx < S.total)
    (hka : ¬ ((S.frames hd.idx).owner = .kernel ∧ sp.user = true))
    (hfree : ¬ (S.frames hd.idx).owner = .free)
    (hwx : ¬ (0 < (S.frames hd.idx).summary.refcount ∧
          ((0 < (S.frames hd.idx).summary.writes ∧ p.x = true) ∨
           (0 < (S.frames hd.idx).summary.execs ∧ p.w = true))))
    (hown : ¬ (sp.user = true ∧ 0 < (S.frames hd.idx).summary.userRefcount ∧
          ((S.frames hd.idx).summary.ownerAsid ≠ some sp.asid ∨
           (S.frames hd.idx).summary.ownerSpace ≠ some s)))
    (hva : ¬ (S.entries.any (Entry.at s page) = true))
    (henc : ¬ (encodeCheck p sp.user ≠ .ok)) :
    ({ S with
        entries := S.entries ++ [⟨s, page, hd.idx, p, (S.frames hd.idx).generation⟩],
        frames := setFrame S.frames hd.idx
          { S.frames hd.idx with
            summary := (S.frames hd.idx).summary.addMapping p sp.user s sp.asid } } : System).Coherent := by
  obtain ⟨hR, hUR, hRd, hWr, hEx, hPE, hOE, hUO, hRI, hKU, hESP, hNWX, hGC, hPU⟩ := hc
  have huser : S.isUserEntry ⟨s, page, hd.idx, p, (S.frames hd.idx).generation⟩ = sp.user := by
    simp [System.isUserEntry, hsp]
  have cnt : ∀ (sel : Summary → Nat) (q : Entry → Bool), S.CountExact sel q →
      sel ((S.frames hd.idx).summary.addMapping p sp.user s sp.asid) =
        sel (S.frames hd.idx).summary +
          (if q ⟨s, page, hd.idx, p, (S.frames hd.idx).generation⟩ = true then 1 else 0) →
      ({ S with
        entries := S.entries ++ [⟨s, page, hd.idx, p, (S.frames hd.idx).generation⟩],
        frames := setFrame S.frames hd.idx
          { S.frames hd.idx with
            summary := (S.frames hd.idx).summary.addMapping p sp.user s sp.asid } } : System).CountExact sel q := by
    intro sel q hcount hstep i
    show sel (setFrame S.frames hd.idx _ i).summary = countIn (S.entries ++ [_]) i q
    rw [countIn_append]
    by_cases hi : i = hd.idx
    · subst hi
      rw [setFrame_summary_self]
      dsimp only
      rw [hstep, hcount hd.idx]
      simp
    · rw [setFrame_summary_other _ _ _ _ hi, hcount i]
      have : (hd.idx == i) = false := by simp; omega
      simp [this]
  have pos : ∀ (sel : Summary → Nat) (q : Entry → Bool), S.CountExact sel q →
      ∀ e ∈ S.entries, q e = true → 0 < sel (S.frames e.frame).summary := by
    intro sel q hcount e he hq
    rw [hcount e.frame]; exact countIn_pos _ e he q hq
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact cnt _ _ hR (by simp [Summary.addMapping])
  · refine cnt _ _ hUR ?_
    show _ = _ + (if S.isUserEntry _ = true then 1 else 0)
    rw [huser]; cases sp.user <;> simp [Summary.addMapping]
  · exact cnt _ _ hRd (by simp [Summary.addMapping, bitCount])
  · exact cnt _ _ hWr (by simp [Summary.addMapping, bitCount])
  · exact cnt _ _ hEx (by simp [Summary.addMapping, bitCount])
  · intro i
    show (setFrame S.frames hd.idx _ i).summary.perms = Perms.ofCounts (setFrame S.frames hd.idx _ i).summary.reads
      (setFrame S.frames hd.idx _ i).summary.writes (setFrame S.frames hd.idx _ i).summary.execs
    by_cases hi : i = hd.idx
    · subst hi; rw [setFrame_summary_self]; rfl
    · rw [setFrame_summary_other _ _ _ _ hi]; exact hPE i
  · intro i
    show ((setFrame S.frames hd.idx _ i).summary.userRefcount = 0 →
          (setFrame S.frames hd.idx _ i).summary.ownerSpace = none ∧
          (setFrame S.frames hd.idx _ i).summary.ownerAsid = none) ∧
       (0 < (setFrame S.frames hd.idx _ i).summary.userRefcount →
          ∃ s' sp', (setFrame S.frames hd.idx _ i).summary.ownerSpace = some s' ∧ S.spaces[s']? = some sp' ∧
            sp'.user = true ∧ (setFrame S.frames hd.idx _ i).summary.ownerAsid = some sp'.asid)
    by_cases hi : i = hd.idx
    · subst hi; rw [setFrame_summary_self]
      dsimp only
      cases hu : sp.user
      · simp only [Summary.addMapping, hu]
        exact hOE hd.idx
      · simp only [Summary.addMapping, hu, if_true]
        exact ⟨fun hz => absurd hz (Nat.succ_ne_zero _), fun _ => ⟨s, sp, rfl, hsp, hu, rfl⟩⟩
    · rw [setFrame_summary_other _ _ _ _ hi]; exact hOE i
  · intro x hx hux
    show (setFrame S.frames hd.idx _ x.frame).summary.ownerSpace = some x.space
    rcases List.mem_append.mp hx with hx | hx
    · have hux' : S.isUserEntry x = true := hux
      have old := hUO x hx hux'
      by_cases hi : x.frame = hd.idx
      · rw [hi, setFrame_summary_self]
        dsimp only
        cases hu : sp.user
        · simp only [Summary.addMapping, hu]; rw [← hi]; exact old
        · simp only [Summary.addMapping, hu, if_true]
          have hpos : 0 < (S.frames hd.idx).summary.userRefcount := by
            rw [← hi]; exact pos _ _ hUR x hx hux'
          have hsome : (S.frames hd.idx).summary.ownerSpace = some s := by
            apply Decidable.of_not_not
            intro hne
            exact hown ⟨hu, hpos, Or.inr hne⟩
          rw [← hi, old] at hsome
          rw [Option.some.inj hsome]
      · rw [setFrame_summary_other _ _ _ _ hi]; exact old
    · rw [List.mem_singleton] at hx
      subst hx
      have hu : sp.user = true := by rw [← huser]; exact hux
      rw [setFrame_summary_self]
      simp [Summary.addMapping, hu]
  · intro x hx
    show x.frame < S.total ∧ (setFrame S.frames hd.idx _ x.frame).owner ≠ .free
    rw [setFrame_summary_owner]
    rcases List.mem_append.mp hx with hx | hx
    · exact hRI x hx
    · rw [List.mem_singleton] at hx; subst hx; exact ⟨hidx, hfree⟩
  · intro x hx hux
    show (setFrame S.frames hd.idx _ x.frame).owner ≠ .kernel
    rw [setFrame_summary_owner]
    rcases List.mem_append.mp hx with hx | hx
    · exact hKU x hx hux
    · rw [List.mem_singleton] at hx; subst hx
      have hu : sp.user = true := by rw [← huser]; exact hux
      intro hk; exact hka ⟨hk, hu⟩
  · intro x hx
    rcases List.mem_append.mp hx with hx | hx
    · exact hESP x hx
    · rw [List.mem_singleton] at hx; subst hx
      exact ⟨sp, hsp, hpol, Decidable.of_not_not henc⟩
  · intro a ha b hb hab ⟨hw, hxb⟩
    rcases List.mem_append.mp ha with ha | ha <;> rcases List.mem_append.mp hb with hb | hb
    · exact hNWX a ha b hb hab ⟨hw, hxb⟩
    · rw [List.mem_singleton] at hb; subst hb
      have h1 : 0 < (S.frames a.frame).summary.refcount := pos _ _ hR a ha rfl
      have h2 : 0 < (S.frames a.frame).summary.writes := pos _ _ hWr a ha hw
      rw [hab] at h1 h2
      exact hwx ⟨h1, Or.inl ⟨h2, hxb⟩⟩
    · rw [List.mem_singleton] at ha; subst ha
      have h1 : 0 < (S.frames b.frame).summary.refcount := pos _ _ hR b hb rfl
      have h2 : 0 < (S.frames b.frame).summary.execs := pos _ _ hEx b hb hxb
      rw [← hab] at h1 h2
      exact hwx ⟨h1, Or.inr ⟨h2, hw⟩⟩
    · rw [List.mem_singleton] at ha hb; subst ha; subst hb
      exact policy_ok_not_wx p sp.user hpol ⟨hw, hxb⟩
  · intro x hx
    show x.generation = (setFrame S.frames hd.idx _ x.frame).generation
    rw [setFrame_summary_generation]
    rcases List.mem_append.mp hx with hx | hx
    · exact hGC x hx
    · rw [List.mem_singleton] at hx; subst hx; rfl
  · refine List.pairwise_append.mpr ⟨hPU, List.pairwise_singleton _ _, ?_⟩
    intro a ha b hb ⟨h1, h2⟩
    rw [List.mem_singleton] at hb; subst hb
    exact hva (List.any_eq_true.mpr ⟨a, ha, by simp [Entry.at] at h1 h2 ⊢; omega⟩)

theorem map_preserves (S : System) (s page : Nat) (hd : Handle) (p : Perms) (h : S.Coherent) :
    (S.map s page hd p).2.Coherent := by
  unfold System.map
  cases hsp : S.spaces[s]? with
  | none => exact h
  | some sp =>
  dsimp only
  by_cases hpol : policy p sp.user ≠ .ok
  · rw [if_pos hpol]; exact h
  rw [if_neg hpol]
  split
  · exact h
  split
  · exact h
  split
  · exact h
  split
  · exact h
  split
  · exact h
  split
  · exact h
  split
  · exact h
  split
  · exact h
  split
  · exact h
  rename_i hidx hka hfree _ _ hwx hown hva henc
  exact map_success S s page hd p sp h hsp (Decidable.of_not_not hpol) (Nat.lt_of_not_le hidx)
    hka hfree hwx hown hva henc

/-! ## Remapping -/

theorem remap_success (S : System) (s page : Nat) (p : Perms) (sp : Space) (e : Entry)
    (hc : S.Coherent) (hsp : S.spaces[s]? = some sp) (hpol : policy p sp.user = .ok)
    (hfind : S.entries.find? (Entry.at s page) = some e)
    (hwx : ¬ ((0 < decIf e.perms.w (S.frames e.frame).summary.writes ∧ p.x = true) ∨
              (0 < decIf e.perms.x (S.frames e.frame).summary.execs ∧ p.w = true)))
    (henc : ¬ (encodeCheck p sp.user ≠ .ok)) :
    ({ S with
        entries := S.entries.map (remapEntry s page p),
        frames := setFrame S.frames e.frame
          { S.frames e.frame with
            summary := (S.frames e.frame).summary.replacePerms e.perms p } } : System).Coherent := by
  obtain ⟨hR, hUR, hRd, hWr, hEx, hPE, hOE, hUO, hRI, hKU, hESP, hNWX, hGC, hPU⟩ := hc
  have he : e ∈ S.entries := List.mem_of_find?_eq_some hfind
  have hk : Entry.at s page e = true := List.find?_some hfind
  have hsk : e.space = s ∧ e.page = page := by simp [Entry.at] at hk; exact hk
  have hre : remapEntry s page p e = { e with perms := p } := by simp [remapEntry, hk]
  have uniq := at_unique S hPU s page e
  have pos : ∀ (sel : Summary → Nat) (q : Entry → Bool), S.CountExact sel q →
      ∀ x ∈ S.entries, q x = true → 0 < sel (S.frames x.frame).summary := by
    intro sel q hcount x hx hq
    rw [hcount x.frame]; exact countIn_pos _ x hx q hq
  have cnt : ∀ (sel : Summary → Nat) (q : Entry → Bool), S.CountExact sel q →
      (∀ x, q (remapEntry s page p x) = q x ∨ sel = Summary.reads ∨ sel = Summary.writes ∨ sel = Summary.execs) →
      sel ((S.frames e.frame).summary.replacePerms e.perms p) + (if q e = true then 1 else 0) =
        sel (S.frames e.frame).summary + (if q { e with perms := p } = true then 1 else 0) →
      ({ S with
        entries := S.entries.map (remapEntry s page p),
        frames := setFrame S.frames e.frame
          { S.frames e.frame with
            summary := (S.frames e.frame).summary.replacePerms e.perms p } } : System).CountExact sel q := by
    intro sel q hcount _ hstep i
    show sel (setFrame S.frames e.frame _ i).summary = countIn (S.entries.map _) i q
    have key := countIn_remap S hPU s page p e he hk i q
    rw [hre] at key
    by_cases hi : i = e.frame
    · subst hi
      rw [setFrame_summary_self]
      dsimp only
      simp only [beq_self_eq_true, Bool.true_and] at key
      rw [hcount e.frame] at hstep
      omega
    · rw [setFrame_summary_other _ _ _ _ hi, hcount i]
      have : (e.frame == i) = false := by simp; omega
      simp [this] at key
      omega
  have bits : ∀ (sel : Summary → Nat) (b : Perms → Bool), S.CountExact sel (fun x => b x.perms) →
      sel ((S.frames e.frame).summary.replacePerms e.perms p) =
        decIf (b e.perms) (sel (S.frames e.frame).summary) + bitCount (b p) →
      sel ((S.frames e.frame).summary.replacePerms e.perms p) + (if b e.perms = true then 1 else 0) =
        sel (S.frames e.frame).summary + (if b p = true then 1 else 0) := by
    intro sel b hcount hsel
    have hp := decIf_spec (b e.perms) (sel (S.frames e.frame).summary)
      (fun hb => pos sel _ hcount e he hb)
    rw [hsel]
    simp only [bitCount] at hp ⊢
    omega
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact cnt _ _ hR (fun _ => Or.inl rfl) (by simp [Summary.replacePerms])
  · refine cnt _ _ hUR (fun x => Or.inl ?_) ?_
    · show S.isUserEntry _ = S.isUserEntry x
      simp only [System.isUserEntry, remapEntry_space]
    · show _ + (if S.isUserEntry e = true then 1 else 0) = _ + (if S.isUserEntry { e with perms := p } = true then 1 else 0)
      simp [Summary.replacePerms, System.isUserEntry]
  · exact cnt _ _ hRd (fun _ => Or.inr (Or.inl rfl))
      (bits Summary.reads Perms.r hRd (by simp [Summary.replacePerms]))
  · exact cnt _ _ hWr (fun _ => Or.inr (Or.inr (Or.inl rfl)))
      (bits Summary.writes Perms.w hWr (by simp [Summary.replacePerms]))
  · exact cnt _ _ hEx (fun _ => Or.inr (Or.inr (Or.inr rfl)))
      (bits Summary.execs Perms.x hEx (by simp [Summary.replacePerms]))
  · intro i
    show (setFrame S.frames e.frame _ i).summary.perms = Perms.ofCounts (setFrame S.frames e.frame _ i).summary.reads
      (setFrame S.frames e.frame _ i).summary.writes (setFrame S.frames e.frame _ i).summary.execs
    by_cases hi : i = e.frame
    · subst hi; rw [setFrame_summary_self]; rfl
    · rw [setFrame_summary_other _ _ _ _ hi]; exact hPE i
  · intro i
    show ((setFrame S.frames e.frame _ i).summary.userRefcount = 0 →
          (setFrame S.frames e.frame _ i).summary.ownerSpace = none ∧
          (setFrame S.frames e.frame _ i).summary.ownerAsid = none) ∧
       (0 < (setFrame S.frames e.frame _ i).summary.userRefcount →
          ∃ s' sp', (setFrame S.frames e.frame _ i).summary.ownerSpace = some s' ∧ S.spaces[s']? = some sp' ∧
            sp'.user = true ∧ (setFrame S.frames e.frame _ i).summary.ownerAsid = some sp'.asid)
    by_cases hi : i = e.frame
    · subst hi; rw [setFrame_summary_self]; exact hOE _
    · rw [setFrame_summary_other _ _ _ _ hi]; exact hOE i
  · intro x hx hux
    obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
    show (setFrame S.frames e.frame _ (remapEntry s page p y).frame).summary.ownerSpace =
      some (remapEntry s page p y).space
    rw [remapEntry_frame, remapEntry_space]
    have huy : S.isUserEntry y = true := by
      have : S.isUserEntry (remapEntry s page p y) = true := hux
      simpa [System.isUserEntry, remapEntry_space] using this
    by_cases hi : y.frame = e.frame
    · rw [hi, setFrame_summary_self]; rw [← hi]; exact hUO y hy huy
    · rw [setFrame_summary_other _ _ _ _ hi]; exact hUO y hy huy
  · intro x hx
    obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
    show (remapEntry s page p y).frame < S.total ∧
      (setFrame S.frames e.frame _ (remapEntry s page p y).frame).owner ≠ .free
    rw [setFrame_summary_owner, remapEntry_frame]; exact hRI y hy
  · intro x hx hux
    obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
    show (setFrame S.frames e.frame _ (remapEntry s page p y).frame).owner ≠ .kernel
    rw [setFrame_summary_owner, remapEntry_frame]
    have huy : S.isUserEntry y = true := by
      have : S.isUserEntry (remapEntry s page p y) = true := hux
      simpa [System.isUserEntry, remapEntry_space] using this
    exact hKU y hy huy
  · intro x hx
    obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
    by_cases hky : Entry.at s page y = true
    · have hye := uniq y he hk hy hky
      subst hye
      rw [hre]
      refine ⟨sp, ?_, hpol, Decidable.of_not_not henc⟩
      show S.spaces[y.space]? = some sp
      rw [hsk.1]; exact hsp
    · rw [remapEntry_of_not_at s page p y (by simpa using hky)]; exact hESP y hy
  · intro a ha b hb hab ⟨hw, hxb⟩
    obtain ⟨ya, hya, rfl⟩ := List.mem_map.mp ha
    obtain ⟨yb, hyb, rfl⟩ := List.mem_map.mp hb
    rw [remapEntry_frame, remapEntry_frame] at hab
    -- The count of other entries of the frame, excluding the remapped one.
    have other : ∀ (sel : Summary → Nat) (b : Perms → Bool), S.CountExact sel (fun x => b x.perms) →
        ∀ y ∈ S.entries, Entry.at s page y = false → y.frame = e.frame → b y.perms = true →
        0 < decIf (b e.perms) (sel (S.frames e.frame).summary) := by
      intro sel b hcount y hy hky hfy hby
      have key := countIn_remove S hPU s page e he hk e.frame (fun x => b x.perms)
      simp only [beq_self_eq_true, Bool.true_and] at key
      have hyf : y ∈ S.entries.filter (fun x => !Entry.at s page x) :=
        List.mem_filter.mpr ⟨hy, by simp [hky]⟩
      have hp := countIn_pos _ y hyf (fun x => b x.perms) hby
      rw [hfy] at hp
      rw [hcount e.frame]
      by_cases hbe : b e.perms = true
      · rw [if_pos hbe] at key
        simp only [decIf, hbe, true_and]
        split <;> omega
      · rw [if_neg hbe] at key
        simp [decIf, hbe]
        omega
    by_cases hka : Entry.at s page ya = true <;> by_cases hkb : Entry.at s page yb = true
    · have h1 := uniq ya he hk hya hka
      have h2 := uniq yb he hk hyb hkb
      subst h1; subst h2
      rw [hre] at hw hxb
      exact policy_ok_not_wx p sp.user hpol ⟨hw, hxb⟩
    · have h1 := uniq ya he hk hya hka
      subst h1
      rw [hre] at hw
      rw [remapEntry_of_not_at s page p yb (by simpa using hkb)] at hxb
      exact hwx (Or.inr ⟨other Summary.execs Perms.x hEx yb hyb (by simpa using hkb) hab.symm hxb, hw⟩)
    · have h2 := uniq yb he hk hyb hkb
      subst h2
      rw [hre] at hxb
      rw [remapEntry_of_not_at s page p ya (by simpa using hka)] at hw
      exact hwx (Or.inl ⟨other Summary.writes Perms.w hWr ya hya (by simpa using hka) hab hw, hxb⟩)
    · rw [remapEntry_of_not_at s page p ya (by simpa using hka)] at hw
      rw [remapEntry_of_not_at s page p yb (by simpa using hkb)] at hxb
      exact hNWX ya hya yb hyb hab ⟨hw, hxb⟩
  · intro x hx
    obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
    show (remapEntry s page p y).generation =
      (setFrame S.frames e.frame _ (remapEntry s page p y).frame).generation
    rw [setFrame_summary_generation, remapEntry_frame, remapEntry_generation]; exact hGC y hy
  · show (S.entries.map (remapEntry s page p)).Pairwise _
    rw [List.pairwise_map]
    exact hPU.imp (fun {a b} hab h => hab (by
      rw [remapEntry_space, remapEntry_space, remapEntry_page, remapEntry_page] at h; exact h))

theorem remap_preserves (S : System) (s page : Nat) (p : Perms) (h : S.Coherent) :
    (S.remap s page p).2.Coherent := by
  unfold System.remap
  cases hsp : S.spaces[s]? with
  | none => exact h
  | some sp =>
  dsimp only
  by_cases hpol : policy p sp.user ≠ .ok
  · rw [if_pos hpol]; exact h
  rw [if_neg hpol]
  cases hf : S.entries.find? (Entry.at s page) with
  | none => exact h
  | some e =>
  dsimp only
  split
  · exact h
  split
  · exact h
  split
  · exact h
  rename_i hwx henc
  exact remap_success S s page p sp e h hsp (Decidable.of_not_not hpol) hf hwx henc

/-! ## Unmapping -/

theorem removeMapping_owner (m : Summary) (p : Perms) (u : Bool) :
    (m.removeMapping p u).ownerSpace = none ∨ (m.removeMapping p u).ownerSpace = m.ownerSpace := by
  unfold Summary.removeMapping
  split
  · split
    · exact Or.inl rfl
    · dsimp only; split
      · exact Or.inl rfl
      · exact Or.inr rfl
  · dsimp only; split
    · exact Or.inl rfl
    · exact Or.inr rfl

theorem unmap_success (S : System) (s page : Nat) (sp : Space) (e : Entry)
    (hc : S.Coherent) (hsp : S.spaces[s]? = some sp)
    (hfind : S.entries.find? (Entry.at s page) = some e) :
    ({ S with
        entries := S.entries.filter (fun e' => !Entry.at s page e'),
        frames := if e.frame < S.total then
            setFrame S.frames e.frame
              { S.frames e.frame with
                summary := (S.frames e.frame).summary.removeMapping e.perms sp.user }
          else S.frames } : System).Coherent := by
  obtain ⟨hR, hUR, hRd, hWr, hEx, hPE, hOE, hUO, hRI, hKU, hESP, hNWX, hGC, hPU⟩ := hc
  have he : e ∈ S.entries := List.mem_of_find?_eq_some hfind
  have hk : Entry.at s page e = true := List.find?_some hfind
  have hsk : e.space = s ∧ e.page = page := by simp [Entry.at] at hk; exact hk
  rw [if_pos (hRI e he).1]
  have hue : S.isUserEntry e = sp.user := by simp [System.isUserEntry, hsk.1, hsp]
  have sub : ∀ x ∈ S.entries.filter (fun e' => !Entry.at s page e'), x ∈ S.entries :=
    fun x hx => (List.mem_filter.mp hx).1
  generalize hm : (S.frames e.frame).summary = m at *
  have key : ∀ (i : Nat) (q : Entry → Bool),
      countIn (S.entries.filter (fun x => !Entry.at s page x)) i q +
        (if (e.frame == i && q e) = true then 1 else 0) = countIn S.entries i q :=
    countIn_remove S hPU s page e he hk
  have hrc : m.refcount = countIn S.entries e.frame (fun _ => true) := by rw [← hm]; exact hR e.frame
  have hrc_pos : 0 < m.refcount := by rw [hrc]; exact countIn_pos _ e he _ rfl
  -- The count of every predicate at the frame, after removal.
  have newcount : ∀ q : Entry → Bool, countIn S.entries e.frame q =
      countIn (S.entries.filter (fun x => !Entry.at s page x)) e.frame q + (if q e = true then 1 else 0) := by
    intro q; have := key e.frame q; simp at this; omega
  have le := fun q => countIn_le (S.entries.filter (fun x => !Entry.at s page x)) e.frame q
  have remcount := newcount (fun _ => true)
  simp only [if_true] at remcount
  let S' : System := { S with
      entries := S.entries.filter (fun e' => !Entry.at s page e'),
      frames := setFrame S.frames e.frame
        { S.frames e.frame with summary := m.removeMapping e.perms sp.user } }
  have frame_other : ∀ i, i ≠ e.frame → (S'.frames i) = S.frames i :=
    fun i hi => setFrame_summary_other _ _ _ _ hi
  have frame_self : (S'.frames e.frame).summary = m.removeMapping e.perms sp.user := by
    show (setFrame S.frames e.frame _ e.frame).summary = _
    rw [setFrame_summary_self]
  have cnt : ∀ (sel : Summary → Nat) (q : Entry → Bool), S.CountExact sel q →
      sel Summary.reset = 0 →
      (m.refcount - 1 ≠ 0 →
        sel (m.removeMapping e.perms sp.user) + (if q e = true then 1 else 0) = sel m) →
      S'.CountExact sel q := by
    intro sel q hcount h0 hstep i
    show sel (S'.frames i).summary = countIn S'.entries i q
    by_cases hi : i = e.frame
    · subst hi
      rw [frame_self]
      have hq := newcount q
      have hs := hcount e.frame
      rw [hm] at hs
      by_cases hz : m.refcount - 1 = 0
      · have hres : m.removeMapping e.perms sp.user = Summary.reset := by
          simp [Summary.removeMapping, hrc_pos, hz]
        rw [hres, h0]
        have := le q
        show 0 = countIn (S.entries.filter (fun x => !Entry.at s page x)) e.frame q
        omega
      · have := hstep hz
        show _ = countIn (S.entries.filter (fun x => !Entry.at s page x)) e.frame q
        omega
    · rw [frame_other i hi]
      show sel (S.frames i).summary = countIn (S.entries.filter (fun x => !Entry.at s page x)) i q
      have := key i q
      have hne : (e.frame == i) = false := by simp; omega
      simp [hne] at this
      rw [hcount i]; omega
  have nonreset : m.refcount - 1 ≠ 0 → m.removeMapping e.perms sp.user =
      { refcount := m.refcount - 1, userRefcount := m.userAfterRemove sp.user,
        reads := decIf e.perms.r m.reads, writes := decIf e.perms.w m.writes,
        execs := decIf e.perms.x m.execs,
        perms := Perms.ofCounts (decIf e.perms.r m.reads) (decIf e.perms.w m.writes)
          (decIf e.perms.x m.execs),
        ownerAsid := if m.ownerCleared sp.user then none else m.ownerAsid,
        ownerSpace := if m.ownerCleared sp.user then none else m.ownerSpace } := by
    intro hz; simp [Summary.removeMapping, hrc_pos, hz]
  have bitsel : ∀ (sel : Summary → Nat) (b : Perms → Bool), S.CountExact sel (fun x => b x.perms) →
      decIf (b e.perms) (sel m) + (if b e.perms = true then 1 else 0) = sel m := by
    intro sel b hcount
    have hs := hcount e.frame
    rw [hm] at hs
    have hp := decIf_spec (b e.perms) (sel m) (fun hb => by rw [hs]; exact countIn_pos _ e he _ hb)
    simpa [bitCount] using hp
  have hUR' : S'.UserRefcountExact := by
    refine cnt _ _ hUR rfl (fun hz => ?_)
    rw [nonreset hz]
    dsimp only
    have hue' : S'.isUserEntry e = sp.user := hue
    rw [hue']
    have hs : (S.frames e.frame).summary.userRefcount = countIn S.entries e.frame S.isUserEntry := hUR e.frame
    rw [hm] at hs
    cases hu : sp.user
    · simp [Summary.userAfterRemove]
    · have : 0 < m.userRefcount := by rw [hs]; exact countIn_pos _ e he _ (by rw [hue, hu])
      simp [Summary.userAfterRemove, this]; omega
  have hR' : S'.RefcountExact :=
    cnt _ _ hR rfl (fun hz => by rw [nonreset hz]; simp; omega)
  have hOE' : S'.OwnerExact := by
    intro i
    by_cases hi : i = e.frame
    · subst hi
      show ((S'.frames e.frame).summary.userRefcount = 0 →
          (S'.frames e.frame).summary.ownerSpace = none ∧ (S'.frames e.frame).summary.ownerAsid = none) ∧
        (0 < (S'.frames e.frame).summary.userRefcount →
          ∃ s' sp', (S'.frames e.frame).summary.ownerSpace = some s' ∧ S.spaces[s']? = some sp' ∧
            sp'.user = true ∧ (S'.frames e.frame).summary.ownerAsid = some sp'.asid)
      rw [frame_self]
      have old := hOE e.frame
      rw [hm] at old
      by_cases hz : m.refcount - 1 = 0
      · have hres : m.removeMapping e.perms sp.user = Summary.reset := by
          simp [Summary.removeMapping, hrc_pos, hz]
        rw [hres]
        exact ⟨fun _ => ⟨rfl, rfl⟩, fun h => absurd h (Nat.lt_irrefl 0)⟩
      · rw [nonreset hz]
        dsimp only
        by_cases hcl : m.ownerCleared sp.user
        · rw [if_pos hcl, if_pos hcl]
          obtain ⟨hu, hp, h1⟩ := hcl
          have : m.userAfterRemove sp.user = 0 := by simp [Summary.userAfterRemove, hu, hp]; omega
          rw [this]
          exact ⟨fun _ => ⟨rfl, rfl⟩, fun h => absurd h (Nat.lt_irrefl 0)⟩
        · rw [if_neg hcl, if_neg hcl]
          have hle : m.userAfterRemove sp.user ≤ m.userRefcount := by
            unfold Summary.userAfterRemove; split <;> omega
          refine ⟨fun h0 => old.1 ?_, fun hpos => old.2 (by omega)⟩
          unfold Summary.userAfterRemove at h0
          unfold Summary.ownerCleared at hcl
          split at h0
          · rename_i hc2; exact absurd ⟨hc2.1, hc2.2, h0⟩ hcl
          · exact h0
    · show ((S'.frames i).summary.userRefcount = 0 →
          (S'.frames i).summary.ownerSpace = none ∧ (S'.frames i).summary.ownerAsid = none) ∧
        (0 < (S'.frames i).summary.userRefcount →
          ∃ s' sp', (S'.frames i).summary.ownerSpace = some s' ∧ S.spaces[s']? = some sp' ∧
            sp'.user = true ∧ (S'.frames i).summary.ownerAsid = some sp'.asid)
      rw [frame_other i hi]; exact hOE i
  show S'.Coherent
  refine ⟨hR', hUR', ?_, ?_, ?_, ?_, hOE', ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine cnt _ _ hRd rfl (fun hz => ?_)
    rw [nonreset hz]; exact bitsel Summary.reads Perms.r hRd
  · refine cnt _ _ hWr rfl (fun hz => ?_)
    rw [nonreset hz]; exact bitsel Summary.writes Perms.w hWr
  · refine cnt _ _ hEx rfl (fun hz => ?_)
    rw [nonreset hz]; exact bitsel Summary.execs Perms.x hEx
  · intro i
    show (S'.frames i).summary.perms = Perms.ofCounts (S'.frames i).summary.reads
      (S'.frames i).summary.writes (S'.frames i).summary.execs
    by_cases hi : i = e.frame
    · subst hi; rw [frame_self]
      by_cases hz : m.refcount - 1 = 0
      · have hres : m.removeMapping e.perms sp.user = Summary.reset := by
          simp [Summary.removeMapping, hrc_pos, hz]
        rw [hres]; rfl
      · rw [nonreset hz]
    · rw [frame_other i hi]; exact hPE i
  · intro x hx hux
    have hxS := sub x hx
    have hux' : S.isUserEntry x = true := hux
    have old := hUO x hxS hux'
    show (S'.frames x.frame).summary.ownerSpace = some x.space
    by_cases hi : x.frame = e.frame
    · rw [hi, frame_self]
      rw [hi, hm] at old
      have hpos : 0 < (S'.frames e.frame).summary.userRefcount := by
        have hU2 : (S'.frames e.frame).summary.userRefcount = countIn S'.entries e.frame S'.isUserEntry :=
          hUR' e.frame
        rw [hU2, ← hi]; exact countIn_pos _ x hx _ hux
      have hex := (hOE' e.frame).2 hpos
      rw [frame_self] at hex hpos
      obtain ⟨s', _, hs', _⟩ := hex
      rcases removeMapping_owner m e.perms sp.user with h0 | h0
      · rw [h0] at hs'; exact Option.noConfusion hs'
      · rw [h0, old]
    · rw [frame_other _ hi]; exact old
  · intro x hx
    show x.frame < S.total ∧ (S'.frames x.frame).owner ≠ .free
    show x.frame < S.total ∧ (setFrame S.frames e.frame _ x.frame).owner ≠ .free
    rw [setFrame_summary_owner]; exact hRI x (sub x hx)
  · intro x hx hux
    show (setFrame S.frames e.frame _ x.frame).owner ≠ .kernel
    rw [setFrame_summary_owner]; exact hKU x (sub x hx) hux
  · intro x hx; exact hESP x (sub x hx)
  · intro a ha b hb; exact hNWX a (sub a ha) b (sub b hb)
  · intro x hx
    show x.generation = (setFrame S.frames e.frame _ x.frame).generation
    rw [setFrame_summary_generation]; exact hGC x (sub x hx)
  · exact List.Pairwise.sublist (List.filter_sublist _) hPU

theorem unmap_preserves (S : System) (s page : Nat) (h : S.Coherent) :
    (S.unmap s page).2.Coherent := by
  unfold System.unmap
  cases hsp : S.spaces[s]? with
  | none => exact h
  | some sp =>
  dsimp only
  cases hf : S.entries.find? (Entry.at s page) with
  | none => exact h
  | some e => exact unmap_success S s page sp e h hsp hf

/-! ## Reachable states -/

theorem step_preserves (S : System) (op : Op) (h : S.Coherent) : (S.step op).2.Coherent := by
  cases op with
  | alloc o i => exact alloc_preserves S o i h
  | free hd => exact free_preserves S hd h
  | map s page hd p => exact map_preserves S s page hd p h
  | remap s page p => exact remap_preserves S s page p h
  | unmap s page => exact unmap_preserves S s page h

theorem run_preserves : ∀ (S : System) (ops : List Op), S.Coherent → (S.run ops).Coherent
  | _, [], h => h
  | S, op :: ops, h => run_preserves _ ops (step_preserves S op h)

/-- Every state reachable from initialization is coherent, for any list of spaces. -/
theorem reachable_coherent (total kernelReserved : Nat) (spaces : List Space) (ops : List Op) :
    ((System.init total kernelReserved spaces).run ops).Coherent :=
  run_preserves _ ops (init_coherent total kernelReserved spaces)

/-! ## The guarantee -/

theorem coherent_single_user_space (S : System) (h : S.Coherent) : S.SingleUserSpace := by
  intro a ha b hb hua hub hab
  have h1 := h.2.2.2.2.2.2.2.1 a ha hua
  have h2 := h.2.2.2.2.2.2.2.1 b hb hub
  rw [hab, h2] at h1
  exact (Option.some.inj h1).symm

theorem isUserEntry_of_spaceOf (S : System) (e : Entry) (sp : Space)
    (h : S.spaceOf e = some sp) (hu : sp.user = true) : S.isUserEntry e = true := by
  simp [System.isUserEntry, System.spaceOf] at h ⊢
  rw [h]; exact hu

/-- No assumption on ASIDs: user entries of one frame in spaces with different ASIDs do not
    exist, so in particular none of them is writable. -/
theorem coherent_mutable_isolation (S : System) (h : S.Coherent) : S.MutableIsolation := by
  intro a ha b hb sa sb hsa hsb hua hub hab hne
  have := coherent_single_user_space S h a ha b hb (isUserEntry_of_spaceOf S a sa hsa hua)
    (isUserEntry_of_spaceOf S b sb hsb hub) hab
  have : sa = sb := by
    simp [System.spaceOf] at hsa hsb; rw [this] at hsa; rw [hsa] at hsb; exact Option.some.inj hsb
  exact absurd (congrArg Space.asid this) hne

theorem coherent_space_isolation (S : System) (h : S.Coherent) : S.SpaceIsolation :=
  fun a ha b hb hua hub hne hab => hne (coherent_single_user_space S h a ha b hb hua hub hab)

theorem reachable_mutable_isolation (total kernelReserved : Nat) (spaces : List Space) (ops : List Op) :
    ((System.init total kernelReserved spaces).run ops).MutableIsolation :=
  coherent_mutable_isolation _ (reachable_coherent total kernelReserved spaces ops)

theorem reachable_space_isolation (total kernelReserved : Nat) (spaces : List Space) (ops : List Op) :
    ((System.init total kernelReserved spaces).run ops).SpaceIsolation :=
  coherent_space_isolation _ (reachable_coherent total kernelReserved spaces ops)

/-- Under the invariant, the ASID comparison in native's ownership check is implied by the
    space comparison: the check refuses exactly when the recorded owner space differs. -/
theorem owner_check_asid_redundant (S : System) (h : S.Coherent) (i s : Nat) (sp : Space)
    (hsp : S.spaces[s]? = some sp) (hpos : 0 < (S.frames i).summary.userRefcount) :
    ((S.frames i).summary.ownerAsid ≠ some sp.asid ∨ (S.frames i).summary.ownerSpace ≠ some s) ↔
      (S.frames i).summary.ownerSpace ≠ some s := by
  obtain ⟨s', sp', hs', hsp', _, ha'⟩ := (h.2.2.2.2.2.2.1 i).2 hpos
  constructor
  · rintro (ha | hs)
    · intro hs
      rw [hs] at hs'
      have : s = s' := Option.some.inj hs'
      subst this
      rw [hsp] at hsp'
      rw [Option.some.inj hsp'] at ha
      exact ha ha'
    · exact hs
  · exact Or.inr

/-! ## Initialization -/

/-- The configuration of the earlier ASID-collision reproducer, a kernel space with
    ASID 2, is now refused at initialization. -/
theorem init_refuses_kernel_space_with_user_asid : Space.initStatus ⟨2, false⟩ = .invalidArg := by
  decide

theorem init_refuses_user_space_with_kernel_asid : Space.initStatus ⟨kernelAsid, true⟩ = .invalidArg := by
  decide

/-- Two user spaces may share an ASID at initialization. -/
theorem init_accepts_shared_user_asid : Space.initStatus ⟨2, true⟩ = .ok := by decide

end EshkolKernelFrameMappings
