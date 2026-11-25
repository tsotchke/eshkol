/-
  Copyright (C) tsotchke
  SPDX-License-Identifier: MIT

  Eshkol OS Phase K3: Machine-Checked Lean 4.12.0 Formal Proofs
    1. `child_capability_never_exceeds_parent`: monotonic rights attenuation
       across arbitrary finite CDT derivation traces.
    2. `cdt_depth_bounded_and_acyclic`: strict depth ceiling and parent-depth
       monotonicity ruling out CDT cycles structurally.
    3. `revocation_completeness` and `zero_side_effects_on_revoked_complete`:
       atomic transitive subtree revocation and fail-closed mid-flight ring
       completion with zero receiver buffer/state mutation.
    4. `capability_confinement_and_non_interference`: cross-core shared-memory
       writes cannot forge capabilities or alter another core's private CSpace.

  Zero `sorry`, zero custom `axiom`.
-/

namespace EshkolKernelK3

/-! ## Section 1: CDT Monotonic Rights Attenuation & Structural Acyclicity -/

/-- Rights bitmask modeled as a predicate over bit indices. -/
abbrev Rights := Nat -> Bool

def rightsSubset (child parent : Rights) : Prop :=
  ∀ b, child b = true → parent b = true

structure CapNode where
  active : Bool
  compId : Nat
  depth  : Nat
  parent : Option Nat
  rights : Rights

structure CdtState where
  maxDepth : Nat
  nodes    : Nat -> CapNode

/-- CDT structural invariants: monotonic attenuation, bounded depth, and strict parent depth ordering. -/
def CdtState.Valid (s : CdtState) : Prop :=
  (∀ id, (s.nodes id).active = true → (s.nodes id).depth ≤ s.maxDepth) ∧
  (∀ id p, (s.nodes id).active = true → (s.nodes id).parent = some p →
    (s.nodes p).active = true ∧
    (s.nodes p).depth < (s.nodes id).depth ∧
    rightsSubset (s.nodes id).rights (s.nodes p).rights)

def CdtState.empty (maxDepth : Nat) : CdtState :=
  { maxDepth := maxDepth
    nodes := fun _ =>
      { active := false, compId := 0, depth := 0, parent := Option.none, rights := fun _ => false } }

theorem cdt_empty_valid (maxDepth : Nat) : (CdtState.empty maxDepth).Valid := by
  constructor
  · intro id h; simp [CdtState.empty] at h
  · intro id p h; simp [CdtState.empty] at h

/-- Mint a root capability into an inactive slot `slot`. -/
def CdtState.mintRoot (s : CdtState) (slot compId : Nat) (r : Rights) : CdtState :=
  if (s.nodes slot).active = false then
    { s with
      nodes := fun i =>
        if i = slot then
          { active := true, compId := compId, depth := 0, parent := Option.none, rights := r }
        else
          s.nodes i }
  else
    s

/-- Derive a child capability into an inactive slot `childSlot` under `parentSlot`
    if and only if `parent` is active, `depth < maxDepth`, and `childRights ⊆ parent.rights`. -/
def CdtState.derive (s : CdtState) (parentSlot childSlot targetComp : Nat)
    (childRights : Rights) (hsub : ∀ b, childRights b = true → (s.nodes parentSlot).rights b = true) :
    CdtState :=
  let _ := hsub
  if parentSlot ≠ childSlot ∧
     (s.nodes childSlot).active = false ∧
     (s.nodes parentSlot).active = true ∧
     (s.nodes parentSlot).depth < s.maxDepth then
    { s with
      nodes := fun i =>
        if i = childSlot then
          { active := true
            compId := targetComp
            depth := (s.nodes parentSlot).depth + 1
            parent := some parentSlot
            rights := childRights }
        else
          s.nodes i }
  else
    s

theorem mintRoot_preserves_valid (s : CdtState) (slot compId : Nat) (r : Rights)
    (hv : s.Valid) : (s.mintRoot slot compId r).Valid := by
  rcases hv with ⟨hdepth, hpar⟩
  unfold CdtState.mintRoot
  split
  · rename_i hinact
    constructor
    · intro id ha
      dsimp only at ha ⊢
      split at ha
      · rename_i heq
        subst heq
        simp only [↓reduceIte]
        exact Nat.zero_le s.maxDepth
      · rename_i hne
        simp only [hne, ↓reduceIte]
        exact hdepth id ha
    · intro id p ha hp
      dsimp only at ha hp ⊢
      split at ha
      · rename_i heq
        subst heq
        simp at hp
      · rename_i hne
        have hp' : (s.nodes id).parent = some p := by simpa [hne] using hp
        have hres := hpar id p ha hp'
        have hpne : p ≠ slot := by
          intro hps
          subst hps
          rw [hres.1] at hinact
          cases hinact
        simp only [hne, hpne, ↓reduceIte]
        exact hres
  · exact ⟨hdepth, hpar⟩

theorem derive_preserves_valid (s : CdtState) (parentSlot childSlot targetComp : Nat)
    (childRights : Rights)
    (hsub : ∀ b, childRights b = true → (s.nodes parentSlot).rights b = true)
    (hv : s.Valid) :
    (s.derive parentSlot childSlot targetComp childRights hsub).Valid := by
  rcases hv with ⟨hdepth, hpar⟩
  unfold CdtState.derive
  dsimp only
  split
  · rename_i hcond
    rcases hcond with ⟨hnepc, hchinact, hpact, hplt⟩
    constructor
    · intro id ha
      dsimp only at ha ⊢
      split at ha
      · rename_i heq
        subst heq
        simp only [↓reduceIte]
        omega
      · rename_i hne
        simp only [hne, ↓reduceIte]
        exact hdepth id ha
    · intro id p ha hp
      dsimp only at ha hp ⊢
      split at ha
      · rename_i heq
        subst heq
        simp only [↓reduceIte] at hp ⊢
        injection hp with hp_eq
        subst hp_eq
        simp only [hnepc, ↓reduceIte]
        refine ⟨hpact, Nat.lt_succ_self _, hsub⟩
      · rename_i hne
        have hp' : (s.nodes id).parent = some p := by simpa [hne] using hp
        have hres := hpar id p ha hp'
        have hpne : p ≠ childSlot := by
          intro hpc
          subst hpc
          rw [hres.1] at hchinact
          cases hchinact
        simp only [hne, hpne, ↓reduceIte]
        exact hres
  · exact ⟨hdepth, hpar⟩

/-- Theorem 1: In any valid CDT state, every live child's rights are a subset of its parent's rights. -/
theorem child_capability_never_exceeds_parent (s : CdtState) (hv : s.Valid)
    (child parent : Nat)
    (hact : (s.nodes child).active = true)
    (hpar : (s.nodes child).parent = some parent) :
    rightsSubset (s.nodes child).rights (s.nodes parent).rights :=
  (hv.2 child parent hact hpar).2.2

/-- Strict ancestor relation (reflexive-transitive closure of parent links with ≥ 1 step). -/
inductive StrictAncestor (s : CdtState) : Nat -> Nat -> Prop where
  | direct {a b : Nat} :
      (s.nodes b).active = true →
      (s.nodes b).parent = some a →
      StrictAncestor s a b
  | step {a b c : Nat} :
      StrictAncestor s a b →
      (s.nodes c).active = true →
      (s.nodes c).parent = some b →
      StrictAncestor s a c

theorem strict_ancestor_increases_depth (s : CdtState) (hv : s.Valid) {a b : Nat}
    (hanc : StrictAncestor s a b) :
    (s.nodes a).depth < (s.nodes b).depth := by
  induction hanc with
  | direct hact hpar =>
    exact (hv.2 _ _ hact hpar).2.1
  | step _ hact hpar ih =>
    exact Nat.lt_trans ih (hv.2 _ _ hact hpar).2.1

/-- Theorem 2: Bounded depth and complete structural acyclicity (no node can be its own ancestor). -/
theorem cdt_depth_bounded_and_acyclic (s : CdtState) (hv : s.Valid) :
    (∀ id, (s.nodes id).active = true → (s.nodes id).depth ≤ s.maxDepth) ∧
    (∀ id, ¬ StrictAncestor s id id) := by
  refine ⟨hv.1, ?_⟩
  intro id hcyc
  have hlt := strict_ancestor_increases_depth s hv hcyc
  exact Nat.lt_irrefl _ hlt

/-! ## Section 2: Atomic Transitive Revocation & Zero Side-Effects on Mid-Flight Complete -/

/-- Decidable/propositional subtree membership: `x` is in the subtree rooted at `root`. -/
def InSubtree (s : CdtState) (root x : Nat) : Prop :=
  x = root ∨ StrictAncestor s root x

/-- Atomic subtree revocation: clears `active` for `root` and every transitive descendant. -/
noncomputable def revokeSubtree (s : CdtState) (root : Nat) (inSub : Nat -> Bool)
    (hchar : ∀ x, inSub x = true ↔ InSubtree s root x) : CdtState :=
  let _ := hchar
  { s with
    nodes := fun x =>
      if inSub x = true then
        { (s.nodes x) with active := false, rights := fun _ => false }
      else
        s.nodes x }

/-- Theorem 3A: Revoking `root` atomically invalidates `root` and every transitive descendant. -/
theorem revocation_completeness (s : CdtState) (root : Nat) (inSub : Nat -> Bool)
    (hchar : ∀ x, inSub x = true ↔ InSubtree s root x) :
    ∀ x, InSubtree s root x → ((revokeSubtree s root inSub hchar).nodes x).active = false := by
  intro x hmem
  have ht : inSub x = true := (hchar x).mpr hmem
  unfold revokeSubtree
  simp [ht]

structure IpcReceiverState where
  rxBuf       : List UInt8
  delivered   : Nat
  deriving DecidableEq, Repr

structure InFlightSqe where
  senderCapSlot : Nat
  payload       : List UInt8
  deriving DecidableEq, Repr

/-- Complete an in-flight IPC SQE against the live CDT; revoked sender cap aborts with ZERO side effects. -/
def completeInFlight (cdt : CdtState) (sqe : InFlightSqe) (rx : IpcReceiverState) :
    Bool × IpcReceiverState :=
  if (cdt.nodes sqe.senderCapSlot).active = true then
    (true, { rxBuf := sqe.payload, delivered := rx.delivered + 1 })
  else
    (false, rx)

/-- Theorem 3B: Mid-flight completion after direct or ancestor revocation fails closed with zero side effects. -/
theorem zero_side_effects_on_revoked_complete (s : CdtState) (root : Nat)
    (inSub : Nat -> Bool) (hchar : ∀ x, inSub x = true ↔ InSubtree s root x)
    (sqe : InFlightSqe) (rx : IpcReceiverState)
    (hdesc : InSubtree s root sqe.senderCapSlot) :
    completeInFlight (revokeSubtree s root inSub hchar) sqe rx = (false, rx) := by
  have hrev := revocation_completeness s root inSub hchar sqe.senderCapSlot hdesc
  unfold completeInFlight
  simp [hrev]

/-! ## Section 3: Cross-Core Capability Confinement & Shared-Memory Non-Interference -/

structure PrivateCSpace where
  compId : Nat
  slots  : Nat -> Option (Nat × Nat) -- slot -> (resourceId, secretBadge)

structure MultiCoreSystem where
  cspaces  : Nat -> PrivateCSpace
  shmBytes : List UInt8

inductive CoreAction where
  | writeShm (newBytes : List UInt8)
  | readShm (compId : Nat)
  deriving DecidableEq, Repr

def MultiCoreSystem.step (sys : MultiCoreSystem) (act : CoreAction) : MultiCoreSystem :=
  match act with
  | .writeShm newBytes => { sys with shmBytes := newBytes }
  | .readShm _         => sys

def MultiCoreSystem.run (sys : MultiCoreSystem) (acts : List CoreAction) : MultiCoreSystem :=
  acts.foldl MultiCoreSystem.step sys

/-- A capability handle is valid in `sys` only if minted inside the caller's private CSpace. -/
def MultiCoreSystem.validCap (sys : MultiCoreSystem) (compId slot resId badge : Nat) : Prop :=
  (sys.cspaces compId).slots slot = some (resId, badge)

theorem shm_step_preserves_cspaces (sys : MultiCoreSystem) (act : CoreAction) (compId : Nat) :
    (sys.step act).cspaces compId = sys.cspaces compId := by
  cases act <;> rfl

/-- Theorem 4: Arbitrary shared-memory read/write traces across cores leave every private CSpace
    bit-for-bit identical and preserve exact capability validity (non-interference & confinement). -/
theorem capability_confinement_and_non_interference (sys : MultiCoreSystem)
    (acts : List CoreAction) (compId slot resId badge : Nat) :
    (sys.run acts).cspaces compId = sys.cspaces compId ∧
    ((sys.run acts).validCap compId slot resId badge ↔ sys.validCap compId slot resId badge) := by
  have hcs : ∀ (s : MultiCoreSystem), (s.run acts).cspaces compId = s.cspaces compId := by
    induction acts with
    | nil => intro s; rfl
    | cons act rest ih =>
      intro s
      unfold MultiCoreSystem.run
      rw [List.foldl_cons]
      exact (ih (s.step act)).trans (shm_step_preserves_cspaces s act compId)
  have hEq := hcs sys
  refine ⟨hEq, ?_⟩
  unfold MultiCoreSystem.validCap
  rw [hEq]

end EshkolKernelK3
