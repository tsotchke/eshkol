import Lookup
import Observation

namespace Verificatum.Dispatch

open Verificatum.Lazy Verificatum.Heap

-- loc maps a run's logical addresses to stable, distinct concrete identities.
-- A reflected binder is an immutable level pair and never has a memo slot.
inductive Pointer (α : Type) where
  | cell (p : α)
  | level (depth : Nat)

def pointer (loc : Nat → α) : Ref → Pointer α
  | .addr i => .cell (loc i)
  | .level d => .level d

abbrev environment (loc : Nat → α) (env : Env) := env.map (pointer loc)

theorem lexical_image (loc : Nat → α) (env : Env) :
    Lookup.EnvImage (fun p r => p = pointer loc r) (environment loc env) env := by
  induction env with
  | nil => exact .nil
  | cons _ _ ih => exact .cons rfl ih

inductive RawValue (α : Type) where
  | integer (i : Int)
  | closure (body : Term) (env : List (Pointer α))
  | spine (funValue : RawValue α) (arg : Pointer α)

def neutral (loc : Nat → α) (h : Head) : List Ref → RawValue α
  | [] => match h with
    | .free i => .integer (-Int.ofNat (i+1))
    | .bound d => .integer (Int.ofNat d)
  | a :: xs => .spine (neutral loc h xs) (pointer loc a)

def value (loc : Nat → α) : Value → RawValue α
  | .lam b e => .closure b (environment loc e)
  | .neutral h args => neutral loc h args

inductive Payload (α : Type) where
  | suspension (code : Term) (env : List (Pointer α))
  | result (wh : RawValue α)

def boxing (loc : Nat → α) : Publication.Boxing (Payload α) :=
  ⟨fun t e => .suspension t (environment loc e), fun v => .result (value loc v)⟩

abbrev RawHeap (α : Type) := α → Option (Publication.Tag × Payload α)
def Image (loc : Nat → α) (raw : RawHeap α) (h : Store) :=
  Publication.Image (boxing loc) (fun i => raw (loc i)) h

inductive Frame (α : Type) where
  | arg (p : Pointer α)
  | update (address : α)

def frame (loc : Nat → α) : K → Frame α
  | .arg x => .arg (pointer loc x)
  | .update i => .update (loc i)

inductive Next (α : Type) where
  | eval (term : Term) (env : List (Pointer α)) (stack : List (Frame α))
  | ret (v : RawValue α) (stack : List (Frame α))

inductive Represents (loc : Nat → α) : Next α → Control → List K → Prop where
  | eval : Represents loc (.eval t (environment loc e) (ks.map (frame loc))) (.eval t e) ks
  | ret : Represents loc (.ret (value loc v) (ks.map (frame loc))) (.value v) ks

-- These operations describe the source arms on the raw representations. The
-- allocator supplies p, and the simulation below constrains it to be fresh.
def application [DecidableEq α] (raw : RawHeap α) (p : α) (f a : Term)
    (e : List (Pointer α)) (ks : List (Frame α)) : RawHeap α × Next α :=
  (fun j => if j = p then some (.pending,.suspension a e) else raw j,
   .eval f e (.arg (.cell p) :: ks))

def force (raw : RawHeap α) (p : Pointer α) (ks : List (Frame α)) : Option (Next α) :=
  match p with
  | .level d => some (.ret (.integer (Int.ofNat d)) ks)
  | .cell addr => match raw addr with
    | some (.pending,.suspension t e) => some (.eval t e (.update addr :: ks))
    | some (.memo,.result v) => some (.ret v ks)
    | _ => none

def argument (v : RawValue α) (x : Pointer α) (rest : List (Frame α)) : Next α :=
  match v with
  | .closure body env => .eval body (x :: env) rest
  | other => .ret (.spine other x) rest

theorem application_forward [DecidableEq α] (loc : Nat → α)
    (distinct : Function.Injective loc) (image : Image loc raw h)
    (fresh : raw (loc n) = none) (f a : Term) (env : Env) (k : List K) (r : List R) :
    let result := application raw (loc n) f a (environment loc env) (k.map (frame loc))
    ∃ s, Step ⟨n,h,.eval (.app f a) env,k,r⟩ s ∧
      Image loc result.1 s.heap ∧ Represents loc result.2 s.ctl s.work ∧
      s.fresh = n+1 ∧ h n = none := by
  refine ⟨_,.alloc,?_,.eval (ks := .arg (.addr n) :: k),rfl,?_⟩
  · intro j
    by_cases eq : j = n
    · subst j; simp [application,put,Publication.encode,boxing]
    · have ne : loc j ≠ loc n := fun e => eq (distinct e)
      simpa [application,put,eq,ne] using image j
  · have old := image n
    change raw (loc n) = _ at old
    rw [fresh] at old
    cases eq : h n with
    | none => rfl
    | some cell => simp [eq] at old

theorem force_forward (reach : Reachable t ⟨n,h,.force ref,k,r⟩)
    (image : Image loc raw h) :
    ∃ out ctl ks, force raw (pointer loc ref) (k.map (frame loc)) = some out ∧
      Represents loc out ctl ks ∧ Step ⟨n,h,.force ref,k,r⟩ ⟨n,h,ctl,ks,r⟩ := by
  cases ref with
  | level d => exact ⟨_,_,_,rfl,.ret (v := .neutral (.bound d) []),.level⟩
  | addr i =>
    rcases force_finds_live_cell reach rfl with ⟨code,e,old⟩ | ⟨v,old⟩
    · change h i = some _ at old
      have slot := image i
      simp [old,Publication.encode,boxing] at slot
      refine ⟨_,_,_,?_,.eval (ks := .update i :: k),.demand old⟩
      simp [pointer,force,slot,frame]
    · change h i = some _ at old
      have slot := image i
      simp [old,Publication.encode,boxing] at slot
      exact ⟨_,_,_,by simp [pointer,force,slot],.ret,.hit old⟩

theorem argument_forward (loc : Nat → α) (v : Value) (x : Ref) (k : List K) :
    ∃ ctl ks, Represents loc (argument (value loc v) (pointer loc x) (k.map (frame loc))) ctl ks ∧
      Step ⟨n,h,.value v,.arg x :: k,r⟩ ⟨n,h,ctl,ks,r⟩ := by
  cases v with
  | lam b e => exact ⟨_,_,.eval (e := x::e),.enterLam⟩
  | neutral head args =>
    refine ⟨_,_,?_,.enterNeutral⟩
    cases args with
    | nil => cases head with
      | free i => exact .ret (v := .neutral (.free i) [x])
      | bound d => exact .ret (v := .neutral (.bound d) [x])
    | cons a xs => exact .ret (v := .neutral head (x :: a :: xs))

-- Popping an update frame binds its concrete address, returned WH and saved
-- continuation to the input relation of the already-proved publication arm.
theorem update_handoff (image : Image loc raw h) (v : Value) (i : Nat) (k : List K) :
    (K.update i :: k).map (frame loc) = .update (loc i) :: k.map (frame loc) ∧
    Publication.Related (boxing loc)
      ⟨fun j => raw (loc j),Publication.updateArm,i,.result (value loc v),k,r,count⟩
      ⟨n,h,.value v,.update i :: k,r⟩ := ⟨rfl,.before image⟩

private theorem waiting_cell (reach : Reachable t ⟨n,h,.value v,.update i::k,r⟩) :
    ∃ cell, h i = some cell := by
  obtain ⟨o,heap,ctl,_⟩ := reachable_invariant reach
  obtain ⟨w,_,stack⟩ := ctl
  cases stack with
  | update key _ _ =>
    have slot := heap.slots i
    change SlotOK o (o i) (h i) at slot
    rw [key] at slot
    cases load : h i with
    | none => rw [load] at slot; cases slot
    | some cell => exact ⟨cell,rfl⟩

theorem update_returns (reach : Reachable t ⟨n,h,.value v,.update i::k,r⟩)
    (image : Image loc raw h) (count : Nat) :
    let c : Publication.Concrete _ :=
      ⟨fun j => raw (loc j),Publication.updateArm,i,.result (value loc v),k,r,count⟩
    ∃ finish, ∃ run : Observation.Execution c finish,
      finish.ops = [] ∧ finish.steps = count+1 ∧
      Observation.observe run = [.write i (.result (value loc v)),.publish i (.result (value loc v))] ∧
      ∃ a, Steps ⟨n,h,.value v,.update i::k,r⟩ a ∧ Publication.Related (boxing loc) finish a := by
  obtain ⟨cell,load⟩ := waiting_cell reach
  obtain ⟨tag,old,slot⟩ : ∃ tag old, raw (loc i) = some (tag,old) := by
    have im := image i
    rw [load] at im
    exact ⟨_,_,im⟩
  let run := Observation.Execution.step
    (Publication.Primitive.cdr (ops := [.tagMemo,.tailValue])
      (k := k) (r := r) (count := count) (x := Payload.result (value loc v))
      (h := fun j => raw (loc j)) (i := i) slot)
    (.step (.car (tag := tag) (payload := .result (value loc v)) (by simp [put])) (.step .tail .stop))
  obtain ⟨last,path,rel,eq⟩ := Observation.execution_refinement run
    (update_handoff (r := r) (count := count) (n := n) image v i k).2
  exact ⟨_,run,rfl,rfl,by simp [Publication.updateArm,run,Observation.observe,Publication.observations,put],_,path,rel⟩

/-- info: 'Verificatum.Dispatch.application_forward' depends on axioms: [propext] -/
#guard_msgs in
#print axioms application_forward
/-- info: 'Verificatum.Dispatch.force_forward' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms force_forward
/-- info: 'Verificatum.Dispatch.argument_forward' depends on axioms: [propext] -/
#guard_msgs in
#print axioms argument_forward
/-- info: 'Verificatum.Dispatch.update_returns' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms update_returns
end Verificatum.Dispatch
