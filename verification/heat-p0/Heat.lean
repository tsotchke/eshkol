import Init

namespace Heat

inductive Bank where | old | fresh deriving DecidableEq, Repr
abbrev Address := Bank × Nat
abbrev Memory (α : Type) := Address → α

structure Row where
  dst : Address
  left : Address
  mid : Address
  right : Address
  midWeight : Nat

inductive Term where
  | zero
  | input (i : Nat)
  | mix (w : Nat) (l c r : Term)
  deriving DecidableEq

noncomputable def Term.eval (u : Nat → Rat) : Term → Rat
  | .zero => 0
  | .input i => u i
  | .mix w l c r => (l.eval u + w * c.eval u + r.eval u) / 4

def write (m : Memory α) (a : Address) (v : α) : Memory α :=
  fun p => if p = a then v else m p

def execute (mix : Nat → α → α → α → α) (m : Memory α) : List Row → Memory α
  | [] => m
  | row :: rest =>
    execute mix (write m row.dst (mix row.midWeight (m row.left) (m row.mid) (m row.right))) rest

noncomputable def arithmetic (w : Nat) (l c r : Rat) := (l + w * c + r) / 4

-- Interpretation commutes with executing every physical load and store.
theorem replay_sound (rows : List Row) (m : Memory Term) (u : Nat → Rat) (a : Address) :
    (execute Term.mix m rows a).eval u =
      execute arithmetic (fun p => (m p).eval u) rows a := by
  induction rows generalizing m with
  | nil => rfl
  | cons row rest ih =>
    simp only [execute, ih]
    congr 1
    funext p
    by_cases h : p = row.dst <;> simp [write, h, Term.eval, arithmetic]

def interior (n i : Nat) := 0 < i && i < n

def initial (n : Nat) : Memory Term := fun (bank, i) =>
  if bank = .old && interior n i then .input i else .zero

-- Independent mathematical target with fixed zero endpoints.
def target (n i : Nat) : Term :=
  if interior n i then .mix 2 (initial n (.old,i-1)) (initial n (.old,i))
                              (initial n (.old,i+1)) else .zero

def matchesTarget (n : Nat) (out : Bank) (rows : List Row) : Bool :=
  let final := execute Term.mix (initial n) rows
  (List.range (n+1)).all fun i => decide (final (out,i) = target n i)

theorem admitted_refines (n : Nat) (rows : List Row) (out : Bank)
    (h : matchesTarget n out rows = true) (i : Nat) (hi : i ≤ n) (u : Nat → Rat) :
    execute arithmetic (fun p => (initial n p).eval u) rows (out,i) = (target n i).eval u := by
  rw [← replay_sound]
  unfold matchesTarget at h
  have eqn := List.all_eq_true.mp h i (List.mem_range.mpr (by omega))
  rw [of_decide_eq_true eqn]

inductive Origin where | old (i : Nat) | fresh | written (i : Nat) deriving DecidableEq

def originName : Origin → String
  | .old i => s!"old[{i}]"
  | .fresh => "fresh"
  | .written i => s!"new[{i}]"

-- Walk the actual addresses in evaluation order, including value-equal reads.
def audit (n : Nat) (out : Bank) (rows : List Row) : Except String Unit := do
  if n < 2 || rows.length != n-1 then throw "unsupported row count"
  let mut origins : Memory Origin := fun (b,i) => if b = .old then .old i else .fresh
  for (row, k) in rows.zipIdx do
    let i := k+1
    if row.dst.2 != i then throw s!"wrong destination at cell {i}"
    for (p,j) in [(row.left,i-1), (row.mid,i), (row.right,i+1)] do
      let seen := origins p
      if seen != .old j then throw s!"cell {i}: expected old[{j}], observed {originName seen}"
    origins := write origins row.dst (.written i)
  if !matchesTarget n out rows then throw "discrete operator or output identity differs"
end Heat

namespace Heat
def verdict (n : Nat) (out : Bank) (rows : List Row) : String :=
  match audit n out rows with
  | .ok () => "ACCEPT"
  | .error reason => "REJECT " ++ reason
end Heat
