import Obligation

open Verificatum.Obligation

/-- Four independent slots keep operand identity separate from the dimension value. -/
private def call : Matmul := ⟨0, 1, 2, 3⟩

def main : IO Unit := do
  for m in [1:5] do
    for k in [1:5] do
      for j in [1:5] do
        for n in [1:5] do
          let dims : Dimensions := fun i => [m, k, j, n][i]!
          let (a, b) := (expected [] call).conclusion
          let verdict := if dims a = dims b then "compatible" else "incompatible"
          IO.println s!"{m} {k} {j} {n} {verdict}"
