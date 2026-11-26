import Heap

namespace Verificatum.Publication

open Verificatum.Lazy Verificatum.Heap

inductive Tag where | pending | memo
inductive Op where | writeCdr | tagMemo | tailValue
deriving DecidableEq, Repr

def updateArm := [Op.writeCdr, .tagMemo, .tailValue]

-- The pair operations copy payloads without inspecting them. No injectivity
-- of either boxing function, or decidable equality of payloads, is needed.
structure Boxing (α : Type) where
  suspended : Term → Env → α
  returned : Value → α

abbrev PairStore (α : Type) := Nat → Option (Tag × α)

def encode (b : Boxing α) : Cell → Tag × α
  | .pending t env => (.pending,b.suspended t env)
  | .memo v => (.memo,b.returned v)
  | .writing v => (.pending,b.returned v)

def Image (b : Boxing α) (raw : PairStore α) (h : Store) :=
  ∀ j, raw j = (h j).map (encode b)

-- Registers abstract the frame address in data, the WH in item and the
-- continuation in rest. The two frame segments represent one Eshkol list.
structure Concrete (α : Type) where
  heap : PairStore α
  ops : List Op
  data : Nat
  item : α
  rest : List K
  readback : List R
  steps : Nat

-- Independent primitive semantics. set-cdr! retains the old tag; set-car!
-- retains whatever payload is actually in the addressed pair.
inductive Primitive : Concrete α → Concrete α → Prop where
  | cdr : h i = some (tag,old) →
      Primitive ⟨h,.writeCdr::ops,i,x,k,r,count⟩
                ⟨put h i (some (tag,x)),ops,i,x,k,r,count⟩
  | car : h i = some (tag,payload) →
      Primitive ⟨h,.tagMemo::ops,i,x,k,r,count⟩
                ⟨put h i (some (.memo,payload)),ops,i,x,k,r,count⟩
  | tail : Primitive ⟨h,[.tailValue],i,x,k,r,count⟩ ⟨h,[],i,x,k,r,count+1⟩

-- Between stores the control location disambiguates the payload. There is no
-- extra runtime tag for this phase. No dispatcher action can interleave here.
inductive Related (b : Boxing α) : Concrete α → Config → Prop where
  | before : Image b raw h →
      Related b ⟨raw,updateArm,i,b.returned v,k,r,count⟩ ⟨n,h,.value v,.update i::k,r⟩
  | staged : (∀ j, j ≠ i → raw j = (h j).map (encode b)) →
      raw i = some (tag,b.returned v) → h i = some (.writing v) →
      Related b ⟨raw,[.tagMemo,.tailValue],i,b.returned v,k,r,count⟩ ⟨n,h,.publish i v,k,r⟩
  | tagged : Image b raw h →
      Related b ⟨raw,[.tailValue],i,b.returned v,k,r,count⟩ ⟨n,h,.value v,k,r⟩
  | returned : Image b raw h →
      Related b ⟨raw,[],i,b.returned v,k,r,count⟩ ⟨n,h,.value v,k,r⟩

theorem forward (rel : Related b c a) (st : Primitive c c') :
    ∃ a', Steps a a' ∧ Related b c' a' := by
  cases rel with
  | @before raw h i v k r count n image =>
    cases st with
    | cdr load =>
      rename_i tag old
      refine ⟨_,.next .refl .write,Related.staged (tag := tag) (fun j ne => ?_) ?_ ?_⟩
      · simpa [put,ne] using image j
      · simp [put]
      · simp [put]
  | @staged i raw h tag v k r count n elsewhere same model =>
    cases st with
    | car load =>
      rw [same] at load
      cases load
      refine ⟨_,.next .refl (.publish model),.tagged (fun j => ?_)⟩
      by_cases eq : j = i
      · subst j; simp [put,encode]
      · simpa [put,eq] using elsewhere j eq
  | tagged im =>
    cases st
    exact ⟨_,.refl,.returned im⟩
  | returned _ => cases st

/-- info: 'Verificatum.Publication.forward' depends on axioms: [propext] -/
#guard_msgs in
#print axioms forward

-- Project a primitive from its instruction and the destination's post-store
-- contents. A missing destination yields no event. The path theorem below
-- rules this out for both stores on every completed execution of the arm.
inductive StoreEvent (α : Type) where
  | write (address : Nat) (payload : α)
  | publish (address : Nat) (payload : α)
deriving DecidableEq

def observations (c d : Concrete α) : List (StoreEvent α) :=
  match c.ops, d.heap c.data with
  | .writeCdr :: _, some (_, x) => [.write c.data x]
  | .tagMemo :: _, some (.memo, x) => [.publish c.data x]
  | _, _ => []

inductive Recorded : Concrete α → List (StoreEvent α) → Concrete α → Prop where
  | done : Recorded c [] c
  | more : Primitive c d → Recorded d rest e → Recorded c (observations c d ++ rest) e

theorem complete_store_trace
    (run : Recorded (⟨h,updateArm,i,x,k,r,count⟩ : Concrete α) events finish)
    (done : finish.ops = []) :
    events = [.write i x, .publish i x] := by
  cases run with
  | done => cases done
  | more store tail =>
    cases store with
    | cdr first =>
      cases tail with
      | done => cases done
      | more publish tail =>
        cases publish with
        | car second =>
          simp [put] at second
          rcases second with ⟨rfl,rfl⟩
          cases tail with
          | done => cases done
          | more jump tail =>
            cases jump
            cases tail with
            | done => simp [observations,updateArm,put]
            | more impossible _ => cases impossible

/-- info: 'Verificatum.Publication.complete_store_trace' depends on axioms: [propext] -/
#guard_msgs in
#print axioms complete_store_trace

end Verificatum.Publication
