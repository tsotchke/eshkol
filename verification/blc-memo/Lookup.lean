import Heap

namespace Verificatum.Lookup

open Verificatum.Lazy Verificatum.Heap

-- A concrete reference is opaque. Traversing an environment copies its car
-- or cdr without inspecting the pointer or replacing its lexical binding.
inductive EnvImage (rep : α → Ref → Prop) : List α → Env → Prop where
  | nil : EnvImage rep [] []
  | cons : rep cell a → EnvImage rep tail env → EnvImage rep (cell :: tail) (a :: env)

inductive Destination (α : Type) where
  | value (item : Int)
  | force (item : α)
  | lookup (item : Int) (env : List α)

-- Operands of the three tail calls recognized in the Scheme source. Every
-- branch preserves the incoming frame list and increments the dispatch count.
inductive Op where | freeHead | forceCar | lookupCdr
deriving DecidableEq
structure Arm where
  empty : Op
  first : Op
  outer : Op
deriving DecidableEq
def lookupArm : Arm := ⟨.freeHead,.forceCar,.lookupCdr⟩

def tailCall (op : Op) (item : Int) (env : List α) : Option (Destination α) :=
  match op, env with
  | .freeHead, _ => some (.value (0 - item))
  | .forceCar, cell :: _ => some (.force cell)
  | .lookupCdr, _ :: tail => some (.lookup (item - 1) tail)
  | _, [] => none

def dispatch (arm : Arm) (item : Int) (env : List α) : Option (Destination α) :=
  let op := if env.isEmpty then arm.empty else if item == 1 then arm.first else arm.outer
  tailCall op item env

inductive Related (rep : α → Ref → Prop) : Destination α → Control → Prop where
  | free : Related rep (.value (-Int.ofNat (i+1))) (.value (.neutral (.free i) []))
  | force : rep cell a → Related rep (.force cell) (.force a)
  | lookup : EnvImage rep raw env →
      Related rep (.lookup (Int.ofNat (i+1)) raw) (.lookup i env)

-- Existence rules out a stuck concrete lookup from any related environment.
-- The abstract machine takes exactly one step with the same heap and frames.
theorem forward (image : EnvImage rep raw env) (i : Nat) :
    ∃ dest ctl, dispatch lookupArm (Int.ofNat (i+1)) raw = some dest ∧
      Related rep dest ctl ∧ Step ⟨n,h,.lookup i env,k,r⟩ ⟨n,h,ctl,k,r⟩ := by
  cases image with
  | nil =>
    exact ⟨_,_,by simp [dispatch,lookupArm,tailCall],.free,.free⟩
  | cons slot rest =>
    cases i with
    | zero => exact ⟨_,_,rfl,.force slot,.first⟩
    | succ i =>
      refine ⟨_,_,?_,.lookup rest,.outer⟩
      have neq : (i : Int) + 1 + 1 ≠ 1 := by omega
      simp [dispatch,lookupArm,tailCall,neq]

-- Repeated tail calls of that same dispatch, stopping at a force or a free
-- integer head. The input budget counts only lookup dispatches.
def resolve (fuel : Nat) (item : Int) (env : List α) : Option (Destination α) :=
  match fuel with
  | 0 => none
  | budget+1 => do
    let out ← dispatch lookupArm item env
    match out with
    | .lookup index tail => resolve budget index tail
    | result => some result

-- The list subscript is a separate specification of lexical lookup. An index
-- beyond the environment becomes a negative integer relative to its end.
theorem resolves_lexically (raw : List α) (i : Nat) :
    resolve (min i raw.length + 1) (Int.ofNat (i+1)) raw =
      some (match raw[i]? with
            | some cell => .force cell
            | none => .value (-Int.ofNat (i - raw.length + 1))) := by
  induction raw generalizing i with
  | nil => simp [resolve,dispatch,lookupArm,tailCall]
  | cons cell tail ih =>
    cases i with
    | zero => rfl
    | succ i =>
      have neq : (i : Int) + 1 + 1 ≠ 1 := by omega
      simpa [Nat.succ_min_succ,resolve,dispatch,lookupArm,tailCall,neq]
        using ih i

/-- info: 'Verificatum.Lookup.forward' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms forward

/-- info: 'Verificatum.Lookup.resolves_lexically' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms resolves_lexically

end Verificatum.Lookup
