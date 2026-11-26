/-
  A deterministic partial transition emits one observation on each successful
  step. Two completed histories can share their remaining computation once
  they reach the same state.

  Model-to-Implementation Correspondence
  lib/core/ad/certified_continuation.esk implements certified-continuation-advance
  and verify-certified-continuation. Its cc-try returns either a new state
  with a certificate or a refusal, and the verifier replays the numerical run.
  The abstract state includes the data needed for replay. A successful step
  carries a new state and an observation. A refused step has no complete suffix.
  tests/stdlib/certified_continuation_test.esk checks replay, retained starting
  state, subdivision and bounded refusal. The piecewise coverage lemma takes
  local validity and a common endpoint as premises.
  At the control-runtime level, eshkol_continuation_resume in
  lib/core/runtime_continuations.cpp restores a captured control state.
  tests/continuations/doc_example_multishot.esk exercises successive resumes
  with a changing store. Applying the equality premise there requires equality
  of the observable store as well as the resumed control state.
-/

namespace EshkolContinuationCoalescence

-- Partial deterministic steps can refuse unsupported inputs.
variable {S X O : Type}
abbrev Step (S X O : Type) := S → X → Option (S × O)

def future (f : Step S X O) (s : S) : List X → Option (S × List O)
  | [] => some (s, [])
  | x :: xs => do
    let (u, obs) ← f s x
    let (v, rest) ← future f u xs
    pure (v, obs :: rest)

/-- Split a finite execution at any boundary, preserving order of observations
    and propagating refusal from either part. -/
theorem future_append (f : Step S X O) (s : S) (xs ys : List X) :
    future f s (xs ++ ys) = (future f s xs).bind (fun p =>
      (future f p.1 ys).map (fun q => (q.1, p.2 ++ q.2))) := by
  induction xs generalizing s with
  | nil => cases h : future f s ys <;> simp [future, h]
  | cons x xs ih =>
    cases hx : f s x with
    | none => simp [future, hx]
    | some p =>
      obtain ⟨u, o⟩ := p
      cases hp : future f u xs with
      | none => simp [future, hx, ih, hp]
      | some q =>
        obtain ⟨v, obs⟩ := q
        cases hq : future f v ys <;> simp [future, hx, ih, hp, hq]

/-- After possibly different finite input histories reach a common state,
    both callers get their own projected history followed by the common
    continuation's projected observations, or both receive refusal. -/
theorem coalescence_after_histories {A B : Type} (f : Step S X O)
    (pa : O → A) (pb : O → B) {a b c : S} {xs ys : List X} {as bs : List O}
    (ha : future f a xs = some (c, as)) (hb : future f b ys = some (c, bs)) (zs : List X) :
    ((future f a (xs ++ zs)).map (fun p => (p.1, p.2.map pa)),
     (future f b (ys ++ zs)).map (fun p => (p.1, p.2.map pb))) =
    ((future f c zs).map (fun p => (p.1, as.map pa ++ p.2.map pa)),
     (future f c zs).map (fun p => (p.1, bs.map pb ++ p.2.map pb))) := by
  rw [future_append, future_append, ha, hb]
  cases h : future f c zs <;> simp [h, List.map_append]

theorem coalescence (f : Step S X O) {a b c : S} {x : X} {oa ob : O}
    (ha : f a x = some (c, oa)) (hb : f b x = some (c, ob)) (xs : List X) :
    ((f a x).bind (fun p => future f p.1 xs)) =
      ((f b x).bind (fun p => future f p.1 xs)) := by
  rw [ha, hb]
  rfl

-- Each caller retains its prefix and projects the common suffix independently.
theorem projected_history {A B : Type} (pa : O → A) (pb : O → B)
    (pastA : List A) (pastB : List B) (f : Step S X O) {a b c : S}
    {x : X} {oa ob : O} (ha : f a x = some (c, oa))
    (hb : f b x = some (c, ob)) (xs : List X) :
    ((future f a (x :: xs)).map (fun p => pastA ++ p.2.map pa),
     (future f b (x :: xs)).map (fun p => pastB ++ p.2.map pb)) =
    ((future f c xs).map (fun p => pastA ++ (pa oa :: p.2.map pa)),
     (future f c xs).map (fun p => pastB ++ (pb ob :: p.2.map pb))) := by
  cases h : future f c xs <;> simp [future, ha, hb, h]

/-- Coverage lifts the checked validity and endpoint of each piece to the domain. -/
theorem path_fiber {T Piece : Type} {domain : T → Prop} {inside : Piece → T → Prop}
    {valid : S → Prop} {path : T → S} {transition : S → S} {c : S}
    (covered : ∀ t, domain t → ∃ p, inside p t)
    (checked : ∀ p t, inside p t → valid (path t) ∧ transition (path t) = c) :
    ∀ t, domain t → valid (path t) ∧ transition (path t) = c := by
  intro t ht
  obtain ⟨p, hp⟩ := covered t ht
  exact checked p t hp

#print axioms future_append
#print axioms coalescence_after_histories
#print axioms coalescence
#print axioms projected_history
#print axioms path_fiber

end EshkolContinuationCoalescence
