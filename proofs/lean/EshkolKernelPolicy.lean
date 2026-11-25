/-
  Copyright (C) tsotchke
  SPDX-License-Identifier: MIT

  Eshkol OS Phase K5/K6: Machine-Checked Lean 4.12.0 Formal Policy Proofs
    1. `supervisor_restart_never_leaks_prior_token`: restarted service token
       generation is strictly greater than faulted generation and prior token
       is permanently rejected.
    2. `supervisor_revoked_cap_rejects_restart`: attempting restart with
       revoked root capability always fails closed (`revokedCap`) and leaves
       the service stopped.
    3. `supervisor_no_zombie_lock_on_restart`: fault transition atomically
       clears all lock ownership held by the faulted service so subsequent
       restart sees zero zombie locks.
    4. `compositor_surface_spatial_disjointness`: accepted surface tables and
       writes preserve spatial non-overlap across distinct badges.
    5. `compositor_readonly_write_always_traps`: any write on a read-only
       surface capability always returns `roTrap` and preserves framebuffer
       contents identically.
    6. `compositor_audio_bounds_preserved`: overrun/underrun audio steps
       preserve `count ≤ cap` and keep the supervisor operational.

  Zero `sorry`, zero custom `axiom`.
-/

namespace EshkolKernelPolicy

/-! ## Section 1: Supervisor Restart Policy & Zombie-Lock Freedom -/

inductive SvcStatus where
  | stopped
  | running
  | faulted
  deriving DecidableEq, Repr

inductive PolicyResult where
  | ok
  | revokedCap
  | staleToken
  | notFaulted
  | surfaceConflict
  | roTrap
  deriving DecidableEq, Repr

structure ServiceEntry where
  status         : SvcStatus
  rootCapLive    : Bool
  tokenGen       : Nat
  lastFaultedGen : Nat
  deriving DecidableEq, Repr

structure SupervisorState where
  services  : Nat -> ServiceEntry
  lockOwner : Nat -> Option Nat

def SupervisorState.validateToken (st : SupervisorState) (sid tok : Nat) : PolicyResult :=
  let svc := st.services sid
  if !svc.rootCapLive then
    .revokedCap
  else if tok = 0 ∨ tok ≠ svc.tokenGen ∨ tok ≤ svc.lastFaultedGen then
    .staleToken
  else
    .ok

def SupervisorState.faultService (st : SupervisorState) (sid : Nat) : SupervisorState :=
  let svc := st.services sid
  let newFaultGen := max svc.tokenGen svc.lastFaultedGen
  { services := fun i =>
      if i = sid then
        { svc with
          status := if svc.rootCapLive then .faulted else .stopped
          lastFaultedGen := newFaultGen }
      else
        st.services i
    lockOwner := fun l =>
      if st.lockOwner l = some sid then Option.none else st.lockOwner l }

def SupervisorState.revokeCap (st : SupervisorState) (sid : Nat) : SupervisorState :=
  let svc := st.services sid
  { services := fun i =>
      if i = sid then
        { svc with
          status := .stopped
          rootCapLive := false
          lastFaultedGen := max svc.tokenGen svc.lastFaultedGen }
      else
        st.services i
    lockOwner := fun l =>
      if st.lockOwner l = some sid then Option.none else st.lockOwner l }

def SupervisorState.restartService (st : SupervisorState) (sid : Nat) :
    PolicyResult × SupervisorState :=
  let svc := st.services sid
  match svc.rootCapLive with
  | false =>
    (.revokedCap,
     { st with
       services := fun i =>
         if i = sid then { svc with status := .stopped } else st.services i })
  | true =>
    match svc.status with
    | .faulted =>
      let freshGen := max svc.tokenGen svc.lastFaultedGen + 1
      (.ok,
       { services := fun i =>
           if i = sid then
             { svc with status := .running, tokenGen := freshGen }
           else
             st.services i
         lockOwner := fun l =>
           if st.lockOwner l = some sid then Option.none else st.lockOwner l })
    | _ => (.notFaulted, st)

/-- Theorem 1: Restarting a faulted service mints a strictly greater token generation
    and permanently rejects the prior faulted token. -/
theorem supervisor_restart_never_leaks_prior_token (st : SupervisorState) (sid : Nat)
    (hlive : (st.services sid).rootCapLive = true) :
    let stFault := st.faultService sid
    let priorTok := (st.services sid).tokenGen
    let (res, stRestart) := stFault.restartService sid
    res = .ok ∧
    priorTok < (stRestart.services sid).tokenGen ∧
    stRestart.validateToken sid priorTok = .staleToken := by
  dsimp only [SupervisorState.faultService, SupervisorState.restartService,
              SupervisorState.validateToken]
  simp only [↓reduceIte, hlive, Bool.not_true, Bool.false_eq_true]
  refine ⟨trivial, by omega, ?_⟩
  have hle : (st.services sid).tokenGen ≤
             max (st.services sid).tokenGen (st.services sid).lastFaultedGen :=
    Nat.le_max_left _ _
  simp [hle]

/-- Theorem 2: Attempting restart when root capability is revoked always fails closed
    with `revokedCap` and leaves the service stopped. -/
theorem supervisor_revoked_cap_rejects_restart (st : SupervisorState) (sid : Nat)
    (hrev : (st.services sid).rootCapLive = false) :
    let (res, st') := st.restartService sid
    res = .revokedCap ∧ (st'.services sid).status = .stopped := by
  dsimp only [SupervisorState.restartService]
  simp [hrev]

/-- Theorem 3: Faulting a service atomically clears all locks held by that service,
    eliminating zombie locks both immediately after fault and after restart. -/
theorem supervisor_no_zombie_lock_on_restart (st : SupervisorState) (sid : Nat) :
    let stFault := st.faultService sid
    let stRestart := (stFault.restartService sid).2
    (∀ l, stFault.lockOwner l ≠ some sid) ∧
    (∀ l, stRestart.lockOwner l ≠ some sid) := by
  have hclear : ∀ l, (st.faultService sid).lockOwner l ≠ some sid := by
    intro l
    dsimp only [SupervisorState.faultService]
    split <;> rename_i h
    · simp
    · exact h
  refine ⟨hclear, ?_⟩
  intro l
  dsimp only [SupervisorState.restartService, SupervisorState.faultService]
  simp only [↓reduceIte]
  cases (st.services sid).rootCapLive with
  | false =>
    dsimp only
    exact hclear l
  | true =>
    simp only [↓reduceIte]
    split <;> rename_i h
    · simp
    · exact h

/-! ## Section 2: Compositor Surface Disjointness & Read-Only Write Trap -/

structure Surface where
  ownerBadge : Nat
  xStart     : Nat
  xLen       : Nat
  writable   : Bool
  deriving DecidableEq, Repr

def Surface.overlaps (a b : Surface) : Bool :=
  decide (a.xStart < b.xStart + b.xLen ∧ b.xStart < a.xStart + a.xLen)

structure CompositorState where
  surfaces    : List Surface
  framebuffer : Nat -> Nat

def CompositorState.Disjoint (c : CompositorState) : Prop :=
  c.surfaces.Pairwise (fun a b => Surface.overlaps a b = false)

def CompositorState.registerSurface (c : CompositorState) (s : Surface) :
    PolicyResult × CompositorState :=
  if c.surfaces.all (fun existing => !Surface.overlaps s existing) then
    (.ok, { c with surfaces := s :: c.surfaces })
  else
    (.surfaceConflict, c)

def CompositorState.writeSurface (c : CompositorState) (s : Surface)
    (callerBadge cellOffset pixelVal : Nat) : PolicyResult × CompositorState :=
  if callerBadge ≠ s.ownerBadge then
    (.surfaceConflict, c)
  else if !s.writable then
    (.roTrap, c)
  else
    (.ok, { c with framebuffer := fun idx =>
      if idx = s.xStart + cellOffset then pixelVal else c.framebuffer idx })

def CompositorState.registerAll (c : CompositorState) (ss : List Surface) : CompositorState :=
  ss.foldl (fun acc s => (acc.registerSurface s).2) c

theorem registerSurface_preserves_disjoint (c : CompositorState) (s : Surface)
    (hd : c.Disjoint) : (c.registerSurface s).2.Disjoint := by
  unfold CompositorState.registerSurface
  split
  · rename_i hall
    have hpw : ∀ x ∈ c.surfaces, Surface.overlaps s x = false := by
      intro x hx
      have h' := List.all_eq_true.mp hall x hx
      exact Bool.not_eq_true' _ |>.mp h'
    exact List.Pairwise.cons hpw hd
  · exact hd

/-- Theorem 4: Any sequence of surface registrations preserves pairwise spatial disjointness
    across all accepted surfaces (including distinct badges). -/
theorem compositor_surface_spatial_disjointness (fb : Nat -> Nat) (ss : List Surface) :
    (CompositorState.registerAll { surfaces := [], framebuffer := fb } ss).Disjoint := by
  have hinit : (CompositorState.mk [] fb).Disjoint := List.Pairwise.nil
  suffices ∀ (c : CompositorState), c.Disjoint → (c.registerAll ss).Disjoint from this _ hinit
  induction ss with
  | nil => intro c hc; exact hc
  | cons s rest ih =>
    intro c hc
    unfold CompositorState.registerAll
    rw [List.foldl_cons]
    exact ih (c.registerSurface s).2 (registerSurface_preserves_disjoint c s hc)

/-- Theorem 5: Writing through a read-only surface capability always traps (`roTrap`)
    and leaves the framebuffer bit-for-bit identical. -/
theorem compositor_readonly_write_always_traps (c : CompositorState) (s : Surface)
    (cellOffset pixelVal : Nat) (hro : s.writable = false) :
    c.writeSurface s s.ownerBadge cellOffset pixelVal = (.roTrap, c) := by
  unfold CompositorState.writeSurface
  simp [hro]

/-! ## Section 3: Real-Time Audio Ring Boundedness & Non-Halting Liveness -/

structure AudioRing where
  cap              : Nat
  count            : Nat
  overrunDrops     : Nat
  underrunSilences : Nat
  supervisorHalted : Bool
  deriving DecidableEq, Repr

def AudioRing.Valid (a : AudioRing) : Prop :=
  0 < a.cap ∧ a.count ≤ a.cap ∧ a.supervisorHalted = false

inductive AudioOp where
  | produce
  | consume
  deriving DecidableEq, Repr

def AudioRing.step (a : AudioRing) (op : AudioOp) : AudioRing :=
  match op with
  | .produce =>
    if a.count < a.cap then
      { a with count := a.count + 1 }
    else
      { a with overrunDrops := a.overrunDrops + 1 }
  | .consume =>
    if 0 < a.count then
      { a with count := a.count - 1 }
    else
      { a with underrunSilences := a.underrunSilences + 1 }

def AudioRing.run (a : AudioRing) (ops : List AudioOp) : AudioRing :=
  ops.foldl AudioRing.step a

theorem audio_step_preserves_valid (a : AudioRing) (op : AudioOp) (hv : a.Valid) :
    (a.step op).Valid := by
  rcases hv with ⟨hcap, hle, hlive⟩
  cases op with
  | produce =>
    unfold AudioRing.step
    dsimp only
    split
    · exact ⟨hcap, by dsimp only; omega, hlive⟩
    · exact ⟨hcap, hle, hlive⟩
  | consume =>
    unfold AudioRing.step
    dsimp only
    split
    · exact ⟨hcap, by dsimp only; omega, hlive⟩
    · exact ⟨hcap, hle, hlive⟩

/-- Theorem 6: Arbitrary overrun/underrun audio traces preserve `count ≤ cap`
    and keep `supervisorHalted = false`. -/
theorem compositor_audio_bounds_preserved (a : AudioRing) (ops : List AudioOp) (hv : a.Valid) :
    (a.run ops).Valid := by
  induction ops generalizing a with
  | nil => exact hv
  | cons op rest ih =>
    unfold AudioRing.run
    rw [List.foldl_cons]
    exact ih (a.step op) (audio_step_preserves_valid a op hv)

end EshkolKernelPolicy
