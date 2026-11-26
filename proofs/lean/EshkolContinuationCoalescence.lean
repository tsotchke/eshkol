/-
  EshkolContinuationCoalescence.lean
  Standalone Lean 4.12.0 formal verification of deterministic partial-step
  continuation coalescence, independent caller history projection, and
  piecewise path-fiber coverage obligations.

  Zero external dependencies, zero `sorry`, zero custom axioms.
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

-- The domain and coverage obligation are explicit; local numerical soundness
-- must be discharged for every admitted piece and the enrolled transition.
theorem path_fiber {T Piece : Type} {domain : T → Prop} {inside : Piece → T → Prop}
    {valid : S → Prop} {path : T → S} {transition : S → S} {c : S}
    (covered : ∀ t, domain t → ∃ p, inside p t)
    (checked : ∀ p t, inside p t → valid (path t) ∧ transition (path t) = c) :
    ∀ t, domain t → valid (path t) ∧ transition (path t) = c := by
  intro t ht
  obtain ⟨p, hp⟩ := covered t ht
  exact checked p t hp

#print axioms coalescence
#print axioms projected_history
#print axioms path_fiber

end EshkolContinuationCoalescence
