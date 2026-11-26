/-
  Copyright (C) tsotchke
  SPDX-License-Identifier: MIT

  Eshkol OS Phase K2: Machine-Checked Lean 4.12.0 Formal Proofs
    1. `no_frame_leak`: physical frame conservation across arbitrary finite
       alloc/free/double-free traces.
       `frame_accounting_preserved`: the free and allocated counters equal the
       number of frames the ownership table assigns to each owner.
    2. `w_x_enforcement`: strict W^X and U/K privilege isolation across all
       reachable PTE states under arbitrary map/remap/unmap traces.
    3. `kernel_non_aliasing`: no kernel-reserved physical frame ever appears
       in any user address space.
    4. `address_space_disjointness`: distinct non-shared user address spaces
       never alias the same physical frame with conflicting permissions.
    5. `mpu_overlap_and_guard_exclusion`: accepted MCU MPU tables contain
       pairwise disjoint regions and never overlap the vector table or
       stack guard zone.

  Zero `sorry`, zero custom `axiom`.
-/

namespace EshkolKernelK2

/-! ## Section 1: Physical Frame Conservation (`no_frame_leak`) -/

inductive FrameOwner where
  | free
  | kernel
  | user
  deriving DecidableEq, Repr

structure FramePool where
  totalFrames    : Nat
  freeCount      : Nat
  allocatedCount : Nat
  owners         : Nat -> FrameOwner
  generations    : Nat -> Nat

/-- Conservation invariant: free + allocated = total and owners are bounded. -/
def FramePool.Conserved (p : FramePool) : Prop :=
  p.freeCount + p.allocatedCount = p.totalFrames

inductive PoolOp where
  | allocUser (idx : Nat)
  | freeUser (idx : Nat) (gen : Nat)
  deriving DecidableEq, Repr

def FramePool.step (p : FramePool) (op : PoolOp) : FramePool :=
  match op with
  | .allocUser idx =>
    if idx < p.totalFrames ∧ p.owners idx = .free ∧ 0 < p.freeCount then
      { p with
        freeCount := p.freeCount - 1
        allocatedCount := p.allocatedCount + 1
        owners := fun i => if i = idx then .user else p.owners i
        generations := fun i => if i = idx then p.generations idx + 1 else p.generations i }
    else
      p
  | .freeUser idx gen =>
    if idx < p.totalFrames ∧ p.owners idx = .user ∧ p.generations idx = gen ∧ 0 < p.allocatedCount then
      { p with
        freeCount := p.freeCount + 1
        allocatedCount := p.allocatedCount - 1
        owners := fun i => if i = idx then .free else p.owners i
        generations := fun i => if i = idx then p.generations idx + 1 else p.generations i }
    else
      p

def FramePool.run (p : FramePool) (ops : List PoolOp) : FramePool :=
  ops.foldl FramePool.step p

theorem pool_init_conserved (total kRes : Nat) (hle : kRes ≤ total) :
    FramePool.Conserved
      { totalFrames := total
        freeCount := total - kRes
        allocatedCount := kRes
        owners := fun i => if i < kRes then .kernel else .free
        generations := fun _ => 1 } := by
  unfold FramePool.Conserved
  dsimp
  omega

theorem pool_step_preserves_conservation (p : FramePool) (op : PoolOp)
    (h : p.Conserved) : (p.step op).Conserved := by
  unfold FramePool.Conserved at *
  cases op with
  | allocUser idx =>
    unfold FramePool.step
    dsimp only
    split
    · rename_i hcond
      dsimp only
      omega
    · exact h
  | freeUser idx gen =>
    unfold FramePool.step
    dsimp only
    split
    · rename_i hcond
      dsimp only
      omega
    · exact h

theorem double_free_rejected_preserves_state (p : FramePool) (idx gen : Nat)
    (hfree : p.owners idx = .free) :
    p.step (.freeUser idx gen) = p := by
  unfold FramePool.step
  simp [hfree]

theorem kernel_frame_free_rejected (p : FramePool) (idx gen : Nat)
    (hkern : p.owners idx = .kernel) :
    p.step (.freeUser idx gen) = p := by
  unfold FramePool.step
  simp [hkern]

theorem no_frame_leak (p : FramePool) (ops : List PoolOp) (h : p.Conserved) :
    (p.run ops).Conserved := by
  induction ops generalizing p with
  | nil => exact h
  | cons op rest ih =>
    unfold FramePool.run
    rw [List.foldl_cons]
    exact ih (p.step op) (pool_step_preserves_conservation p op h)

/-! ## Section 1b: Frame Accounting

`Conserved` relates the two counters to the total only. `Accounted` ties each
counter to the ownership table: `freeCount` is the number of frames owned by
`.free`, and `allocatedCount` the number owned by `.user` or `.kernel`. A step
that changed the counters without changing ownership, or the reverse, breaks it.
-/

/-- Number of frames below `n` whose owner is `o`. -/
def countOwnedBelow (owners : Nat -> FrameOwner) (o : FrameOwner) : Nat -> Nat
  | 0 => 0
  | n + 1 => countOwnedBelow owners o n + (if owners n = o then 1 else 0)

def FramePool.Accounted (p : FramePool) : Prop :=
  p.freeCount = countOwnedBelow p.owners .free p.totalFrames ∧
  p.allocatedCount = countOwnedBelow p.owners .user p.totalFrames +
                     countOwnedBelow p.owners .kernel p.totalFrames

theorem countOwnedBelow_update_above (owners : Nat -> FrameOwner) (o b : FrameOwner)
    (idx : Nat) : ∀ n, n ≤ idx →
    countOwnedBelow (fun i => if i = idx then b else owners i) o n = countOwnedBelow owners o n
  | 0, _ => rfl
  | n + 1, h => by
    have hne : n ≠ idx := by omega
    simp only [countOwnedBelow, hne, if_false]
    rw [countOwnedBelow_update_above owners o b idx n (by omega)]

/-- Moving frame `idx` from owner `a` to owner `b` moves one unit of count. -/
theorem countOwnedBelow_update (owners : Nat -> FrameOwner) (o a b : FrameOwner)
    (idx : Nat) (ha : owners idx = a) : ∀ n, idx < n →
    countOwnedBelow (fun i => if i = idx then b else owners i) o n + (if a = o then 1 else 0) =
    countOwnedBelow owners o n + (if b = o then 1 else 0)
  | 0, h => absurd h (Nat.not_lt_zero _)
  | n + 1, h => by
    simp only [countOwnedBelow]
    by_cases hn : n = idx
    · subst hn
      rw [countOwnedBelow_update_above owners o b n n (Nat.le_refl n)]
      simp only [if_true, ha]
      omega
    · have hlt : idx < n := by omega
      have ih := countOwnedBelow_update owners o a b idx ha n hlt
      simp only [hn, if_false]
      omega

theorem pool_step_preserves_accounted (p : FramePool) (op : PoolOp)
    (h : p.Accounted) : (p.step op).Accounted := by
  rcases h with ⟨hfree, halloc⟩
  cases op with
  | allocUser idx =>
    unfold FramePool.step
    dsimp only
    split
    · rename_i hcond
      obtain ⟨hlt, hown, _⟩ := hcond
      have cf := countOwnedBelow_update p.owners .free .free .user idx hown p.totalFrames hlt
      have cu := countOwnedBelow_update p.owners .user .free .user idx hown p.totalFrames hlt
      have ck := countOwnedBelow_update p.owners .kernel .free .user idx hown p.totalFrames hlt
      simp at cf cu ck
      exact ⟨by dsimp only; omega, by dsimp only; omega⟩
    · exact ⟨hfree, halloc⟩
  | freeUser idx gen =>
    unfold FramePool.step
    dsimp only
    split
    · rename_i hcond
      obtain ⟨hlt, hown, _⟩ := hcond
      have cf := countOwnedBelow_update p.owners .free .user .free idx hown p.totalFrames hlt
      have cu := countOwnedBelow_update p.owners .user .user .free idx hown p.totalFrames hlt
      have ck := countOwnedBelow_update p.owners .kernel .user .free idx hown p.totalFrames hlt
      simp at cf cu ck
      exact ⟨by dsimp only; omega, by dsimp only; omega⟩
    · exact ⟨hfree, halloc⟩

/-- Every trace preserves the tie between counters and the ownership table. -/
theorem frame_accounting_preserved (p : FramePool) (ops : List PoolOp) (h : p.Accounted) :
    (p.run ops).Accounted := by
  induction ops generalizing p with
  | nil => exact h
  | cons op rest ih =>
    unfold FramePool.run
    rw [List.foldl_cons]
    exact ih (p.step op) (pool_step_preserves_accounted p op h)

theorem countOwnedBelow_init_free (k : Nat) : ∀ n,
    countOwnedBelow (fun i => if i < k then FrameOwner.kernel else .free) .free n = n - k
  | 0 => by simp [countOwnedBelow]
  | n + 1 => by
    simp only [countOwnedBelow, countOwnedBelow_init_free k n]
    by_cases h : n < k <;> simp [h] <;> omega

theorem countOwnedBelow_init_kernel (k : Nat) : ∀ n,
    countOwnedBelow (fun i => if i < k then FrameOwner.kernel else .free) .kernel n = min n k
  | 0 => by simp [countOwnedBelow]
  | n + 1 => by
    simp only [countOwnedBelow, countOwnedBelow_init_kernel k n]
    by_cases h : n < k <;> simp [h] <;> omega

theorem countOwnedBelow_init_user (k : Nat) : ∀ n,
    countOwnedBelow (fun i => if i < k then FrameOwner.kernel else .free) .user n = 0
  | 0 => rfl
  | n + 1 => by
    simp only [countOwnedBelow, countOwnedBelow_init_user k n]
    by_cases h : n < k <;> simp [h]

theorem pool_init_accounted (total kRes : Nat) (hle : kRes ≤ total) :
    FramePool.Accounted
      { totalFrames := total
        freeCount := total - kRes
        allocatedCount := kRes
        owners := fun i => if i < kRes then .kernel else .free
        generations := fun _ => 1 } := by
  unfold FramePool.Accounted
  dsimp only
  rw [countOwnedBelow_init_free, countOwnedBelow_init_kernel, countOwnedBelow_init_user]
  exact ⟨rfl, by omega⟩

/-! ## Section 2: W^X Enforcement and Kernel Non-Aliasing -/

structure Pte where
  pfn        : Nat
  readable   : Bool
  writable   : Bool
  executable : Bool
  user       : Bool
  kernelOnly : Bool
  deriving DecidableEq, Repr

/-- Hardware W^X and privilege-bit separation predicate. -/
def Pte.WxAndPrivSafe (pte : Pte) : Prop :=
  !(pte.writable && pte.executable) = true ∧
  !(pte.user && pte.kernelOnly) = true

def Pte.checkSafe (pte : Pte) : Bool :=
  !(pte.writable && pte.executable) && !(pte.user && pte.kernelOnly)

theorem pte_checkSafe_iff (pte : Pte) :
    pte.checkSafe = true ↔ pte.WxAndPrivSafe := by
  unfold Pte.checkSafe Pte.WxAndPrivSafe
  simp [Bool.and_eq_true]

structure UserAspace where
  asid          : Nat
  isKernelFrame : Nat -> Bool
  entries       : Nat -> Option Pte

def UserAspace.empty (asid : Nat) (isKernelFrame : Nat -> Bool) : UserAspace :=
  { asid := asid
    isKernelFrame := isKernelFrame
    entries := fun _ => Option.none }

def UserAspace.WxSafe (a : UserAspace) : Prop :=
  ∀ va pte, a.entries va = some pte → pte.WxAndPrivSafe

def UserAspace.KernelNonAliasing (a : UserAspace) : Prop :=
  ∀ va pte, a.entries va = some pte → a.isKernelFrame pte.pfn = false

inductive MmuOp where
  | map (va : Nat) (pte : Pte)
  | remap (va : Nat) (w x u k : Bool)
  | unmap (va : Nat)
  deriving DecidableEq, Repr

def UserAspace.step (a : UserAspace) (op : MmuOp) : UserAspace :=
  match op with
  | .map va pte =>
    if pte.checkSafe && !a.isKernelFrame pte.pfn && pte.user && !pte.kernelOnly then
      { a with entries := fun v => if v = va then some pte else a.entries v }
    else
      a
  | .remap va w x u k =>
    match a.entries va with
    | Option.none => a
    | some old =>
      let updated : Pte :=
        { old with writable := w, executable := x, user := u, kernelOnly := k }
      if updated.checkSafe && u && !k then
        { a with entries := fun v => if v = va then some updated else a.entries v }
      else
        a
  | .unmap va =>
    { a with entries := fun v => if v = va then Option.none else a.entries v }

def UserAspace.run (a : UserAspace) (ops : List MmuOp) : UserAspace :=
  ops.foldl UserAspace.step a

theorem empty_aspace_wx_safe (asid : Nat) (isK : Nat -> Bool) :
    (UserAspace.empty asid isK).WxSafe := by
  intro va pte h
  simp [UserAspace.empty] at h

theorem empty_aspace_kernel_non_aliasing (asid : Nat) (isK : Nat -> Bool) :
    (UserAspace.empty asid isK).KernelNonAliasing := by
  intro va pte h
  simp [UserAspace.empty] at h

theorem aspace_step_preserves_wx (a : UserAspace) (op : MmuOp) (hwx : a.WxSafe) :
    (a.step op).WxSafe := by
  cases op with
  | map va pte =>
    unfold UserAspace.step
    dsimp only
    split
    · rename_i hcond
      intro v q hq
      dsimp only at hq
      split at hq
      · cases hq
        have hsafe : pte.checkSafe = true := by
          revert hcond
          cases pte.checkSafe <;> simp
        exact (pte_checkSafe_iff pte).mp hsafe
      · exact hwx v q hq
    · exact hwx
  | remap va w x u k =>
    unfold UserAspace.step
    dsimp only
    cases a.entries va with
    | none => exact hwx
    | some old =>
      dsimp only
      split
      · rename_i hcond
        intro v q hq
        dsimp only at hq
        split at hq
        · cases hq
          have hsafe : ({ old with writable := w, executable := x, user := u, kernelOnly := k } : Pte).checkSafe = true := by
            revert hcond
            cases ({ old with writable := w, executable := x, user := u, kernelOnly := k } : Pte).checkSafe <;> simp
          exact (pte_checkSafe_iff _).mp hsafe
        · exact hwx v q hq
      · exact hwx
  | unmap va =>
    intro v q hq
    unfold UserAspace.step at hq
    dsimp only at hq
    split at hq
    · cases hq
    · exact hwx v q hq

theorem w_x_enforcement (a : UserAspace) (ops : List MmuOp) (hwx : a.WxSafe) :
    (a.run ops).WxSafe := by
  induction ops generalizing a with
  | nil => exact hwx
  | cons op rest ih =>
    unfold UserAspace.run
    rw [List.foldl_cons]
    exact ih (a.step op) (aspace_step_preserves_wx a op hwx)

theorem aspace_step_preserves_kernel_non_aliasing (a : UserAspace) (op : MmuOp)
    (hna : a.KernelNonAliasing) :
    (a.step op).KernelNonAliasing := by
  cases op with
  | map va pte =>
    unfold UserAspace.step
    dsimp only
    split
    · rename_i hcond
      intro v q hq
      dsimp only at hq
      split at hq
      · cases hq
        revert hcond
        cases a.isKernelFrame pte.pfn <;> simp
      · exact hna v q hq
    · exact hna
  | remap va w x u k =>
    unfold UserAspace.step
    dsimp only
    cases hold : a.entries va with
    | none => exact hna
    | some old =>
      dsimp only
      split
      · intro v q hq
        dsimp only at hq
        split at hq
        · cases hq
          exact hna va old hold
        · exact hna v q hq
      · exact hna
  | unmap va =>
    intro v q hq
    unfold UserAspace.step at hq
    dsimp only at hq
    split at hq
    · cases hq
    · exact hna v q hq

theorem kernel_non_aliasing (a : UserAspace) (ops : List MmuOp)
    (hna : a.KernelNonAliasing) :
    (a.run ops).KernelNonAliasing := by
  induction ops generalizing a with
  | nil => exact hna
  | cons op rest ih =>
    unfold UserAspace.run
    rw [List.foldl_cons]
    exact ih (a.step op) (aspace_step_preserves_kernel_non_aliasing a op hna)

/-! ## Section 3: Multi-ASID Address-Space Disjointness -/

structure MmuSystem where
  mappings : List (Nat × Nat × Pte) -- (asid, va, pte)

def permConflict (p1 p2 : Pte) : Bool :=
  (p1.writable && p2.executable) || (p1.executable && p2.writable)

/-- Disjointness invariant: no two mappings of the same physical frame conflict in W vs X,
    and distinct user address spaces never alias the same private physical frame. -/
def MmuSystem.Disjoint (s : MmuSystem) : Prop :=
  ∀ e1 e2, e1 ∈ s.mappings → e2 ∈ s.mappings →
    e1.2.2.pfn = e2.2.2.pfn →
    (permConflict e1.2.2 e2.2.2 = false ∧ e1.1 = e2.1)

def MmuSystem.canMap (s : MmuSystem) (asid : Nat) (pte : Pte) : Bool :=
  pte.checkSafe &&
  s.mappings.all fun e =>
    if e.2.2.pfn == pte.pfn then
      (!permConflict e.2.2 pte) && (e.1 == asid)
    else
      true

def MmuSystem.mapFrame (s : MmuSystem) (asid va : Nat) (pte : Pte) : MmuSystem :=
  if s.canMap asid pte then
    { mappings := (asid, va, pte) :: s.mappings }
  else
    s

def MmuSystem.runMaps (s : MmuSystem) (ops : List (Nat × Nat × Pte)) : MmuSystem :=
  ops.foldl (fun acc op => acc.mapFrame op.1 op.2.1 op.2.2) s

theorem permConflict_symm (p1 p2 : Pte) :
    permConflict p1 p2 = permConflict p2 p1 := by
  unfold permConflict
  cases p1.writable <;> cases p1.executable <;> cases p2.writable <;> cases p2.executable <;> rfl

theorem pte_self_no_conflict (p : Pte) (hsafe : p.checkSafe = true) :
    permConflict p p = false := by
  unfold Pte.checkSafe at hsafe
  unfold permConflict
  cases hw : p.writable <;> cases hx : p.executable <;> simp_all

theorem mapFrame_preserves_disjoint (s : MmuSystem) (asid va : Nat) (pte : Pte)
    (hd : s.Disjoint) : (s.mapFrame asid va pte).Disjoint := by
  unfold MmuSystem.mapFrame
  split
  · rename_i hcan
    have hcanParts : pte.checkSafe = true ∧
        (s.mappings.all fun e =>
          if e.2.2.pfn == pte.pfn then (!permConflict e.2.2 pte) && (e.1 == asid) else true) = true := by
      unfold MmuSystem.canMap at hcan
      exact Bool.and_eq_true _ _ |>.mp hcan
    have hsafe := hcanParts.1
    have hall := List.all_eq_true.mp hcanParts.2
    intro e1 e2 h1 h2 hpfn
    simp only [List.mem_cons] at h1 h2
    rcases h1 with rfl | h1m
    · rcases h2 with rfl | h2m
      · exact ⟨pte_self_no_conflict pte hsafe, rfl⟩
      · have hchk := hall e2 h2m
        have hbeq : (e2.2.2.pfn == pte.pfn) = true := (beq_iff_eq _ _).mpr hpfn.symm
        simp only [hbeq, ↓reduceIte, Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at hchk
        rw [permConflict_symm]
        exact ⟨hchk.1, hchk.2.symm⟩
    · rcases h2 with rfl | h2m
      · have hchk := hall e1 h1m
        have hbeq : (e1.2.2.pfn == pte.pfn) = true := (beq_iff_eq _ _).mpr hpfn
        simp only [hbeq, ↓reduceIte, Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at hchk
        exact ⟨hchk.1, hchk.2⟩
      · exact hd e1 e2 h1m h2m hpfn
  · exact hd

theorem address_space_disjointness (ops : List (Nat × Nat × Pte)) :
    (MmuSystem.runMaps { mappings := [] } ops).Disjoint := by
  have hinit : (MmuSystem.mk []).Disjoint := by
    intro e1 _ h1
    cases h1
  suffices ∀ (s : MmuSystem), s.Disjoint → (s.runMaps ops).Disjoint from this _ hinit
  induction ops with
  | nil => intro s hs; exact hs
  | cons op rest ih =>
    intro s hs
    unfold MmuSystem.runMaps
    rw [List.foldl_cons]
    exact ih (s.mapFrame op.1 op.2.1 op.2.2) (mapFrame_preserves_disjoint s op.1 op.2.1 op.2.2 hs)

/-! ## Section 4: MCU MPU Overlap and Guard-Zone Exclusion -/

structure MpuRegion where
  base : Nat
  size : Nat
  deriving DecidableEq, Repr

def MpuRegion.overlaps (a b : MpuRegion) : Bool :=
  decide (a.base < b.base + b.size ∧ b.base < a.base + a.size)

theorem mpu_overlaps_symm (a b : MpuRegion) :
    MpuRegion.overlaps a b = MpuRegion.overlaps b a := by
  unfold MpuRegion.overlaps
  by_cases h1 : a.base < b.base + b.size ∧ b.base < a.base + a.size
  · have h2 : b.base < a.base + a.size ∧ a.base < b.base + b.size := ⟨h1.2, h1.1⟩
    simp [h1, h2]
  · have h2 : ¬ (b.base < a.base + a.size ∧ a.base < b.base + b.size) := by
      intro h; exact h1 ⟨h.2, h.1⟩
    simp [h1, h2]

structure MpuTable where
  vectorGuard : MpuRegion
  stackGuard  : MpuRegion
  regions     : List MpuRegion

def MpuTable.Safe (t : MpuTable) : Prop :=
  (∀ r ∈ t.regions, MpuRegion.overlaps r t.vectorGuard = false ∧
                    MpuRegion.overlaps r t.stackGuard = false) ∧
  t.regions.Pairwise (fun a b => MpuRegion.overlaps a b = false)

def MpuTable.addRegion (t : MpuTable) (r : MpuRegion) : MpuTable :=
  if !MpuRegion.overlaps r t.vectorGuard &&
     !MpuRegion.overlaps r t.stackGuard &&
     t.regions.all (fun existing => !MpuRegion.overlaps r existing) then
    { t with regions := r :: t.regions }
  else
    t

def MpuTable.installAll (t : MpuTable) (rs : List MpuRegion) : MpuTable :=
  rs.foldl MpuTable.addRegion t

theorem mpu_add_preserves_safe (t : MpuTable) (r : MpuRegion) (hs : t.Safe) :
    (t.addRegion r).Safe := by
  unfold MpuTable.addRegion
  split
  · rename_i hcond
    rcases hs with ⟨hguards, hpair⟩
    have h12 : (!MpuRegion.overlaps r t.vectorGuard && !MpuRegion.overlaps r t.stackGuard) = true ∧
               (t.regions.all fun existing => !MpuRegion.overlaps r existing) = true :=
      Bool.and_eq_true _ _ |>.mp hcond
    have hg : (!MpuRegion.overlaps r t.vectorGuard) = true ∧
              (!MpuRegion.overlaps r t.stackGuard) = true :=
      Bool.and_eq_true _ _ |>.mp h12.1
    have hvec : MpuRegion.overlaps r t.vectorGuard = false := Bool.not_eq_true' _ |>.mp hg.1
    have hstk : MpuRegion.overlaps r t.stackGuard = false := Bool.not_eq_true' _ |>.mp hg.2
    have hall : ∀ x ∈ t.regions, MpuRegion.overlaps r x = false := by
      intro x hx
      have hx' := List.all_eq_true.mp h12.2 x hx
      exact Bool.not_eq_true' _ |>.mp hx'
    constructor
    · intro x hx
      simp only [List.mem_cons] at hx
      rcases hx with rfl | hxm
      · exact ⟨hvec, hstk⟩
      · exact hguards x hxm
    · exact List.Pairwise.cons hall hpair
  · exact hs

theorem mpu_overlap_and_guard_exclusion (vecGuard stkGuard : MpuRegion) (rs : List MpuRegion) :
    (MpuTable.installAll { vectorGuard := vecGuard, stackGuard := stkGuard, regions := [] } rs).Safe := by
  have hinit : (MpuTable.mk vecGuard stkGuard []).Safe := by
    constructor
    · intro r hr; cases hr
    · exact List.Pairwise.nil
  suffices ∀ (t : MpuTable), t.Safe → (t.installAll rs).Safe from this _ hinit
  induction rs with
  | nil => intro t ht; exact ht
  | cons r rest ih =>
    intro t ht
    unfold MpuTable.installAll
    rw [List.foldl_cons]
    exact ih (t.addRegion r) (mpu_add_preserves_safe t r ht)

end EshkolKernelK2
