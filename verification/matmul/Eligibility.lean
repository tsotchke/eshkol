import ShapePublication
import GuardAuthority

namespace Verificatum.Eligibility
open ShapePublication

/-- An audited scalar/owned copy transfers no axis authority. A known footprint
    contains every reachable logical shape (including aliases and nested values).
    The classifier defaults to unknown unless that completeness is established. -/
inductive Transfer where
  | detached
  | reachable (shapes : List Nat)
  | unknown
deriving DecidableEq

def revoke : Buffer → Buffer
  | .draft p a _ | .published p a => .exposed p a
  | other => other

private def covers (k : Nat) : List Nat → Bool
  | [] => false
  | h :: rest => k == h || covers k rest

def cross (w : World) (transfer : Transfer := .unknown) : World := fun k =>
  match transfer with
  | .detached => w k
  | .reachable keys => if covers k keys = true then revoke (w k) else w k
  | .unknown => revoke (w k)

inductive History : Buffer → Buffer → Prop where
  | refl : History b b
  | next : History a b → Change b c → History a c

private theorem step_at_key (step : Step w v) (k : Nat) : History (w k) (v k) := by
  cases step with
  | update hc =>
    simp only [put]
    split
    next same => subst k; exact .next .refl hc
    next => exact .refl
  | copy _ fresh =>
    simp only [put]
    split
    next same => rw [same, fresh]; exact .next .refl .allocate
    next => exact .refl

theorem lifecycle_at_key (steps : Steps w v) (k : Nat) : History (w k) (v k) := by
  have append {s t u : Buffer} (before : History s t) (after : History t u) : History s u := by
    induction after with
    | refl => exact before
    | next _ c ih => exact .next ih c
  induction steps with
  | refl => exact .refl
  | snoc _ s ih => exact append ih (step_at_key s k)

private theorem revoke_history (b : Buffer) : History b (revoke b) := by
  cases b with
  | draft => exact .next .refl .exposeDraft
  | published => exact .next .refl .expose
  | unused | exposed | retired => exact .refl

theorem crossing_respects_lifecycle (w : World) (t : Transfer) (k : Nat) :
    History (w k) (cross w t k) := by
  cases t with
  | detached => exact .refl
  | unknown => exact revoke_history _
  | reachable keys =>
    by_cases h : covers k keys = true
    · simpa [cross, h] using revoke_history (w k)
    · simpa [cross, h] using (History.refl : History (w k) (w k))

def Quarantined : Buffer → Prop
  | .exposed .. | .retired => True
  | _ => False

theorem quarantine_persists (h : History start finish) (q : Quarantined start) :
    Quarantined finish := by
  induction h with
  | refl => exact q
  | next _ change ih => cases change <;> simp_all [Quarantined]

theorem unknown_revokes (h : w k = .published p a) :
    Quarantined (cross w .unknown k) := by simp [cross, h, revoke, Quarantined]

/-- The caller supplies fresh observations for this invocation. 'isolated' means
    external writers with unknown reach cannot race the check or its use. -/
structure Invocation where
  site : Guard.Site
  world : World
  resolve : Headers
  leftHeader : Nat
  rightHeader : Nat
  isolated : Bool

def eligible (now : Invocation) (header : Nat) (binding : Binding) : Option Axes :=
  if now.isolated && decide (header = binding.header ∧ now.resolve header = some binding.shape) then
    match now.world binding.shape with
    | .published _ a => some a
    | _ => none
  else none

theorem eligible_bound (h : eligible now header binding = some a) :
    Bound now.world now.resolve header binding a := by
  unfold eligible at h
  split at h
  next ok =>
    have pair : header = binding.header ∧ now.resolve header = some binding.shape := by
      simpa using (Bool.and_eq_true_iff.mp ok).2
    split at h
    next p axes eq => exact ⟨pair.1, pair.2, p, Option.some.inj h ▸ eq⟩
    next => contradiction
  next => contradiction

theorem quarantine_blocks (q : Quarantined (now.world b.shape)) :
    eligible now h b = none := by
  unfold eligible
  split
  · cases e : now.world b.shape <;> simp_all [Quarantined]
  · rfl

private theorem stable (hist : History s t) (hs : s = .published p a)
    (ht : t = .published q b) : b = a := by
  have origin : Preserves a s := by simp [hs, Preserves]
  have keep {u : Buffer} (trace : History s u) : Preserves a u := by
    induction trace with
    | refl => exact origin
    | next _ change ih => exact local_preserves change ih
  simpa [ht, Preserves] using keep hist

/-- Issuance binds the proof's dimensions to the original controlled operands.
    The request's site includes compilation, node, operands and environment.
    Every permitted invocation must separately pass both late eligibility checks. -/
theorem invocation_guard
    {initial now : Invocation} (proof : Guard.Applicable initial.site now.site)
    (li : eligible initial initial.leftHeader bl = some a)
    (ri : eligible initial initial.rightHeader br = some b)
    (leftIssue : a.cols = initial.site.left.cols.nat) (rightIssue : b.rows = initial.site.right.rows.nat)
    (history : ∀ k, History (initial.world k) (now.world k))
    (leftNow : eligible now now.leftHeader bl = some a')
    (rightNow : eligible now now.rightHeader br = some b')
    (sizeOK : Bool) : (decide (a'.cols = b'.rows) && sizeOK) = sizeOK := by
  obtain ⟨_, _, p, hp⟩ := eligible_bound li
  obtain ⟨_, _, q, hq⟩ := eligible_bound leftNow
  obtain ⟨_, _, r, hr⟩ := eligible_bound ri
  obtain ⟨_, _, s, hs⟩ := eligible_bound rightNow
  rw [stable (history _) hp hq, stable (history _) hr hs]
  exact Guard.residual_guard_preserved proof (proof.1 ▸ leftIssue) (proof.1 ▸ rightIssue) sizeOK

/-- Re-enter the controlled domain using independently owned, validated axes.
    Acquisition of the coherent copy and its allocation capacity are premises
    supplied by the implementation. No binding is transferred from its source. -/
theorem publish_fresh_snapshot {w : World} {fresh old : Nat} {addr cap : Nat} {a : Axes}
    (unused : w fresh = .unused) (distinct : fresh ≠ old) (fits : a.rows * a.cols ≤ cap) :
    ∃ v, Steps w v ∧ v fresh = .published addr a ∧ v old = w old := by
  let draft := put w fresh (.draft addr a cap)
  let final := put draft fresh (.published addr a)
  have alloc : Step w draft := .update (key := fresh) (unused ▸ Change.allocate)
  have ready : draft fresh = .draft addr a cap := by simp [draft, put]
  have publishStep : Step draft final := .update (key := fresh) (ready ▸ Change.publish fits)
  exact ⟨final, .snoc (.snoc .refl alloc) publishStep, by simp [final, put],
         by simp [final, draft, put, Ne.symm distinct]⟩

private def site : Guard.Site := ⟨1, 42, 1, ⟨17, 0, .two, .three⟩, ⟨23, 0, .three, .two⟩⟩
private def right : Binding := ⟨12, 8⟩
private def world : World := fun k =>
  if k = 7 then .published 100 ⟨2, 3⟩ else
  if k = 8 then .published 200 ⟨3, 2⟩ else .unused
private def headers : Headers := fun h =>
  if h = 11 then some 7 else if h = 12 ∨ h = 13 then some 8 else none
private def initial : Invocation := ⟨site, world, headers, 11, 12, true⟩
private def after (t : Transfer) : Invocation := {initial with world := cross world t}
private def admitted (i : Invocation) (b : Binding := right) : Bool :=
  (eligible i i.rightHeader b).isSome

-- The compatible replacement still needs its own binding, and another
-- header sharing the exposed shape loses eligibility through either header.
private def rebound : Invocation := {initial with rightHeader := 13}
private def leaked : Invocation := after (.reachable [8])
private def freshWorld := put leaked.world 9 (.published 400 ⟨3, 2⟩)
private def freshCall : Invocation :=
  {initial with world := freshWorld, rightHeader := 14, resolve := fun h =>
    if h = 14 then some 9 else headers h}

def cases : List (String × Bool) := [
  ("published", admitted initial),
  ("axes-copy-only", admitted (after .detached)),
  ("other-shape-exposed", admitted (after (.reachable [7]))),
  ("compatible-header-changed", admitted rebound),
  ("same-header-rebound", admitted {initial with resolve := fun _ => some 7}),
  ("tagged-handoff", admitted leaked),
  ("shared-header-handoff", admitted {leaked with rightHeader := 13} ⟨13, 8⟩),
  ("unknown-footprint", admitted (after .unknown)),
  ("outstanding-writer", admitted {initial with isolated := false}),
  ("returned-writer-old-shape", admitted {leaked with isolated := true}),
  ("retired", admitted {initial with world := put world 8 .retired}),
  ("draft-exposed", admitted {initial with world := cross (put world 8 (.draft 200 ⟨3, 2⟩ 6)) .unknown}),
  ("fresh-copy-old-binding", admitted freshCall),
  ("fresh-copy-new-binding", admitted freshCall ⟨14, 9⟩)]

end Verificatum.Eligibility
