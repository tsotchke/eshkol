/-
  Copyright (C) tsotchke
  SPDX-License-Identifier: MIT

  Eshkol OS Phase K3: Machine-Checked Lean 4.12.0 Formal Proofs
    1. `child_capability_never_exceeds_parent`: monotonic rights attenuation
       on each successful CDT derivation.
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

/-! ## Model-to-Implementation Correspondence

`lib/kernel/k3_smp_cap_ipc.c` implements CDT attenuation, revocation and IPC
completion using the handles and masks in `inc/eshkol/kernel/k3_smp_cap_ipc.h`.
The model checks the same subset relation on 32-bit rights. Native handle
authentication, nonempty rights, locking and slot selection are additional
checks. Lean follows parent links to classify each slot. C visits child and
sibling links with a bounded DFS stack. The shared-memory actions preserve
private CSpaces by a frame condition on the model; these proofs do not verify
native memory access or atomic execution. -/

/-! ## Section 1: CDT Monotonic Rights Attenuation & Structural Acyclicity -/

abbrev Rights := BitVec 32

def rightsSubset (child parent : Rights) : Prop :=
  ∀ bit, child.getLsbD bit = true → parent.getLsbD bit = true

private theorem mask_check_implies_subset {child parent : Rights}
    (check : child &&& parent = child) : rightsSubset child parent := by
  intro bit set
  have h := congrArg (fun mask : Rights => mask.getLsbD bit) check
  simpa [BitVec.getLsbD_and, set] using h

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
      { active := false, compId := 0, depth := 0, parent := Option.none, rights := 0 } }

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
    (childRights : Rights) : Option CdtState :=
  if parentSlot ≠ childSlot ∧
     (s.nodes childSlot).active = false ∧
     (s.nodes parentSlot).active = true ∧
     (s.nodes parentSlot).depth < s.maxDepth ∧
     childRights &&& (s.nodes parentSlot).rights = childRights then
    some { s with
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
    none

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

theorem derive_preserves_valid (s next : CdtState) (parentSlot childSlot targetComp : Nat)
    (childRights : Rights) (hv : s.Valid)
    (derived : s.derive parentSlot childSlot targetComp childRights = some next) : next.Valid := by
  rcases hv with ⟨hdepth, hpar⟩
  unfold CdtState.derive at derived
  split at derived
  · rename_i hcond
    rcases hcond with ⟨hnepc, hchinact, hpact, hplt, hsub⟩
    cases derived
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
        refine ⟨hpact, Nat.lt_succ_self _, mask_check_implies_subset hsub⟩
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
  · cases derived

/-- Acceptance entails attenuation even when the input tree has no validity proof. -/
theorem child_capability_never_exceeds_parent (s next : CdtState) (parent child comp : Nat)
    (requested : Rights) (accepted : s.derive parent child comp requested = some next) :
    (next.nodes child).rights = requested ∧
    rightsSubset (next.nodes child).rights (s.nodes parent).rights := by
  unfold CdtState.derive at accepted
  split at accepted
  · rename_i checks
    cases accepted
    simp only [↓reduceIte, true_and]
    exact mask_check_implies_subset checks.2.2.2.2
  · cases accepted

/-- Strict ancestor relation following one or more active child links. -/
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

def ancestorWalk : Nat → CdtState → Nat → Nat → Bool
  | 0, _, _, _ => false
  | fuel + 1, s, root, x =>
      match (s.nodes x).parent with
      | none => false
      | some p =>
          (s.nodes x).active && (p = root || ancestorWalk fuel s root p)

def runtimeInSubtree (s : CdtState) (root x : Nat) : Bool :=
  x = root || ancestorWalk (s.maxDepth + 1) s root x

theorem ancestorWalk_sound (s : CdtState) {fuel root x : Nat}
    (hw : ancestorWalk fuel s root x = true) : StrictAncestor s root x := by
  induction fuel generalizing x with
  | zero => simp [ancestorWalk] at hw
  | succ fuel ih =>
    simp only [ancestorWalk] at hw
    split at hw
    · contradiction
    · rename_i p
      simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hw
      rcases hw with ⟨hact, hdirect | hnext⟩
      · exact StrictAncestor.direct hact (by simpa [hdirect] using p)
      · exact StrictAncestor.step (ih hnext) hact (by simp_all)

theorem ancestorWalk_complete (s : CdtState) (hv : s.Valid) {fuel root x : Nat}
    (hfuel : (s.nodes x).depth < fuel) (hanc : StrictAncestor s root x) :
    ancestorWalk fuel s root x = true := by
  induction hanc generalizing fuel with
  | direct hact hpar =>
    cases fuel with
    | zero => omega
    | succ fuel => simp [ancestorWalk, hpar, hact]
  | step _ hact hpar ih =>
    have hdepth := (hv.2 _ _ hact hpar).2.1
    cases fuel with
    | zero => omega
    | succ fuel =>
      have hw : ancestorWalk fuel s root _ = true := ih (by omega)
      simp [ancestorWalk, hpar, hact, hw]

theorem strict_ancestor_target_active (s : CdtState) {a b : Nat}
    (hanc : StrictAncestor s a b) : (s.nodes b).active = true := by
  induction hanc with
  | direct hact _ => exact hact
  | step _ hact _ => exact hact

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

theorem runtimeInSubtree_equiv (s : CdtState) (hv : s.Valid) (root x : Nat) :
    runtimeInSubtree s root x = true ↔ InSubtree s root x := by
  unfold runtimeInSubtree InSubtree
  constructor
  · intro h
    simp only [Bool.or_eq_true, decide_eq_true_eq] at h
    rcases h with hroot | hw
    · exact Or.inl hroot
    · exact Or.inr (ancestorWalk_sound s hw)
  · intro h
    rcases h with rfl | hanc
    · simp
    · have hbound := hv.1 x (strict_ancestor_target_active s hanc)
      have hw := ancestorWalk_complete s hv (fuel := s.maxDepth + 1) (by omega) hanc
      simp [hw]

/-- Atomic subtree revocation: clears `active` for `root` and every transitive descendant. -/
def revokeSubtree (s : CdtState) (root : Nat) : CdtState :=
  { s with nodes := fun x =>
      if runtimeInSubtree s root x then { (s.nodes x) with active := false, rights := 0 }
      else s.nodes x }

theorem revocation_completeness (s : CdtState) (root : Nat) (hv : s.Valid) (x : Nat)
    (hmem : InSubtree s root x) : ((revokeSubtree s root).nodes x).active = false := by
  simp [revokeSubtree, (runtimeInSubtree_equiv s hv root x).mpr hmem]

theorem revocation_preserves_outside (s : CdtState) (root x : Nat) (hv : s.Valid)
    (outside : ¬ InSubtree s root x) : (revokeSubtree s root).nodes x = s.nodes x := by
  have miss := mt (runtimeInSubtree_equiv s hv root x).mp outside
  simp [revokeSubtree, miss]

theorem revokeSubtree_preserves_valid (s : CdtState) (root : Nat) (hv : s.Valid) :
    (revokeSubtree s root).Valid := by
  have hchar := runtimeInSubtree_equiv s hv root
  rcases hv with ⟨hdepth, hpar⟩
  unfold CdtState.Valid
  dsimp [revokeSubtree]
  constructor
  · intro id ha
    split at ha
    · simp at ha
    · rename_i hid
      simp only [hid, ↓reduceIte] at ha ⊢
      exact hdepth id ha
  · intro id p ha hp
    split at ha
    · simp at ha
    · rename_i hid
      simp only [hid, ↓reduceIte] at ha hp ⊢
      split
      · rename_i hpmem
        have hchild : StrictAncestor s root id := by
          rcases (hchar p).mp hpmem with hroot | hanc
          · subst p
            exact StrictAncestor.direct ha hp
          · exact StrictAncestor.step hanc ha hp
        have himem : runtimeInSubtree s root id = true := (hchar id).mpr (Or.inr hchild)
        exact (hid himem).elim
      · rename_i hpnone
        have hpar' : (s.nodes id).parent = some p := by simpa [hpnone] using hp
        simp [hpnone]
        exact hpar id p ha hpar'

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
    (hv : s.Valid)
    (sqe : InFlightSqe) (rx : IpcReceiverState)
    (hdesc : InSubtree s root sqe.senderCapSlot) :
    completeInFlight (revokeSubtree s root) sqe rx = (false, rx) := by
  have hrev := revocation_completeness s root hv sqe.senderCapSlot hdesc
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
