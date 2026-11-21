prelude
import Init.Core

namespace Verificatum

/-- The expected proposition, fixed independently of the candidate export.
    Negation is expanded and False is the empty inductive from the pinned Init. -/
def target : Prop :=
  ∀ {A B : Prop}, (A → B) → (B → False) → A → False

/-- A different, inhabited proposition for the target-substitution control. -/
def other : Prop := ∀ A : Prop, A → A

example : target := @mt
example : other := fun _ a => a

end Verificatum
