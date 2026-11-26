import Init

namespace Verificatum.Authority

inductive Level where
  | model | observed | artifact | refinement
  deriving DecidableEq

inductive Kind where
  | exact | refinement | boundedError | trace | risk | resource
  deriving DecidableEq

structure Need where
  claim : Nat
  kind : Kind
  level : Level
  deriving DecidableEq

-- C includes the observation, its limits and the action being requested.
-- The implementation binds artifact bytes and live premises by identity.
structure Scope (C A S : Type) where
  contract : C
  artifact : A
  context : S
  deriving DecidableEq

structure Checked (C A S : Type) where
  scope : Scope C A S
  need : Need

def authorize [DecidableEq C] [DecidableEq A] [DecidableEq S]
    (s : Scope C A S) (needs : C → List Need) (es : List (Checked C A S)) : Bool :=
  !(needs s.contract).isEmpty && (needs s.contract).all fun n =>
    es.any fun e => decide (e.scope = s ∧ e.need = n)

-- "Checked" is semantic: every record supplied by the trusted checker
-- establishes its need (including the consumer's comparison) at its own scope.
theorem authorize_sound [DecidableEq C] [DecidableEq A] [DecidableEq S]
    (s : Scope C A S) (needs : C → List Need) (es : List (Checked C A S))
    (holds : Scope C A S → Need → Prop) (permitted : Scope C A S → Prop)
    (checkerSound : ∀ e ∈ es, holds e.scope e.need)
    (policySound : (∀ n ∈ needs s.contract, holds s n) → permitted s)
    (ok : authorize s needs es = true) : permitted s := by
  apply policySound
  intro n hn
  have cover := (Bool.and_eq_true_iff.mp ok).2
  have found := List.all_eq_true.mp cover n hn
  obtain ⟨e, he, binds⟩ := List.any_eq_true.mp found
  have h : e.scope = s ∧ e.need = n := of_decide_eq_true binds
  simpa only [h.1, h.2] using checkerSound e he

def consume [DecidableEq C] [DecidableEq A] [DecidableEq S]
    (issued current : Scope C A S) : Option A :=
  if issued = current then some issued.artifact else none

theorem consume_bound [DecidableEq C] [DecidableEq A] [DecidableEq S]
    (h : consume issued current = some artifact) :
    (issued : Scope C A S) = current ∧ artifact = current.artifact := by
  unfold consume at h
  split at h
  next same => exact ⟨same, by simpa [same] using (Option.some.inj h).symm⟩
  next => contradiction

theorem release_sound [DecidableEq C] [DecidableEq A] [DecidableEq S]
    (issued current : Scope C A S) (needs : C → List Need) (es : List (Checked C A S))
    (holds : Scope C A S → Need → Prop) (permitted : Scope C A S → Prop)
    (checkerSound : ∀ e ∈ es, holds e.scope e.need)
    (policySound : (∀ n ∈ needs issued.contract, holds issued n) → permitted issued)
    (accepted : authorize issued needs es = true)
    (released : consume issued current = some artifact) :
    permitted current ∧ artifact = current.artifact := by
  obtain ⟨same, bound⟩ := consume_bound released
  exact ⟨same ▸ authorize_sound issued needs es holds permitted checkerSound policySound accepted, bound⟩

-- A domain observer returns checked, denied, or inconclusive. The gate alone
-- issues authorized after visiting every required observation.
inductive Verdict where
  | checked | denied | inconclusive | authorized
  deriving DecidableEq, Repr

def firstFailure (check : N → Verdict) : List N → Verdict
  | [] => .authorized
  | n :: ns => match check n with
    | .checked => firstFailure check ns
    | other => other

def decisionSpec (needs : List N) (check : N → Verdict) : Verdict :=
  if needs.isEmpty then .inconclusive else firstFailure check needs

#print axioms release_sound
#print axioms authorize_sound
#print axioms consume_bound
end Verificatum.Authority
