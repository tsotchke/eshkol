import Init

namespace Q0
noncomputable section

-- Exact real coordinates a + b√2 and complex coordinates r + i·s.
abbrev RealPair := Rat × Rat
abbrev Amp := RealPair × RealPair
def plus (x y : RealPair) : RealPair := (x.1 + y.1, x.2 + y.2)
def minus (x : RealPair) : RealPair := (-x.1, -x.2)
def times (x y : RealPair) : RealPair :=
  (x.1*y.1 + 2*x.2*y.2, x.1*y.2 + x.2*y.1)
def add (x y : Amp) : Amp := (plus x.1 y.1, plus x.2 y.2)
def mul (x y : Amp) : Amp :=
  (plus (times x.1 y.1) (minus (times x.2 y.2)), plus (times x.1 y.2) (times x.2 y.1))
def conj (x : Amp) : Amp := (x.1, minus x.2)
def real (a b : Rat) : Amp := ((a,b), (0,0))
def zero := real 0 0
def one := real 1 0
def eye (r c : Bool) := if r == c then one else zero

abbrev Matrix := Bool → Bool → Amp
inductive Gate | h | x | y | z deriving DecidableEq, Repr
def gate : Gate → Matrix
  | .h => fun r c => real 0 (if r && c then -(1/2) else 1/2)
  | .x => fun r c => if r != c then one else zero
  | .y => fun r c => if r == c then zero else ((0,0), (if r then 1 else -1,0))
  | .z => fun r c => if r == c then real (if r then -1 else 1) 0 else zero

def compose (u v : Matrix) : Matrix := fun r c =>
  add (mul (u r false) (v false c)) (mul (u r true) (v true c))
def denote (tape : List Gate) : Matrix := tape.foldl (fun u g => compose (gate g) u) eye

inductive PhaseGate | t | s | z deriving DecidableEq, Repr
def phase : PhaseGate → Amp
  | .t => ((0,1/2),(0,1/2))
  | .s => ((0,0),(1,0))
  | .z => real (-1) 0

inductive Use | exactOp | channel | controlled deriving DecidableEq, Repr
inductive Layout | controlFirst | controlSecond deriving DecidableEq, Repr
structure Site where
  use : Use
  original : List Gate
  replacement : List Gate
  layout : Layout := .controlFirst
  correction : List PhaseGate := []
  deriving DecidableEq, Repr

def bits := [false,true]
def elements (u : Matrix) := bits.flatMap (fun r => bits.map (u r))
-- Superoperator entries U_ij conjugate(U_kl) for all four indices.
def channelEntries (u : Matrix) := (elements u).flatMap (fun x => (elements u).map (mul x ∘ conj))
def wires (layout : Layout) (a b : Bool) :=
  if layout == .controlFirst then (a,b) else (b,a)

def block (u : Matrix) (row col : Bool × Bool) : Amp :=
  if row.1 != col.1 then zero else if row.1 then u row.2 col.2 else eye row.2 col.2

-- C(ωU) = (diag(1,ω) ⊗ I) C(U), entry by entry on the two control branches.
theorem control_phase {u v : Matrix} (ω : Amp) (rel : ∀ r c, u r c = mul ω (v r c))
    (row col : Bool × Bool) :
    block u row col = (if row.1 then mul ω (block v row col) else block v row col) := by
  rcases row with ⟨r, i⟩
  rcases col with ⟨c, j⟩
  cases r <;> cases c <;> simp [block, rel, mul, times, plus, minus, zero, real] <;> decide +kernel

-- Apply the synthesized diagonal gates to OUTPUT rows of the controlled operator.
def controlledEntries (layout : Layout) (u : Matrix) (gs : List PhaseGate) :=
  let basis := bits.flatMap (fun a => bits.map (wires layout a))
  basis.flatMap fun row => basis.map fun col =>
    let x := block u row col
    if row.1 then gs.foldl (fun v g => mul (phase g) v) x else x

def observe (use : Use) (layout : Layout) (tape : List Gate) (gs : List PhaseGate) :=
  match use with
  | .exactOp => elements (denote tape)
  | .channel => channelEntries (denote tape)
  | .controlled => controlledEntries layout (denote tape) gs

def before (s : Site) := observe s.use s.layout s.original []
def after (s : Site) := observe s.use s.layout s.replacement s.correction

-- Q0 admits a correction only where a control wire exists.
def ValidRewrite (s : Site) : Prop :=
  (s.correction = [] ∨ s.use = .controlled) ∧ before s = after s
instance (s : Site) : Decidable (ValidRewrite s) := inferInstanceAs (Decidable (_ ∧ _))

-- One deliberately expressible extra premise makes the premise-substitution
-- control an inhabited conditional proof, even for a false unconditional goal.
inductive Premise | observationsAgree deriving DecidableEq
def Holds (s : Site) : Premise → Prop | .observationsAgree => before s = after s
structure Evidence where
  claim : Site
  premises : List Premise
  proof : (∀ p ∈ premises, Holds claim p) → ValidRewrite claim

inductive Verdict | accept | target | binding | premises deriving DecidableEq, Repr
def bind (consumer : Site) (e : Evidence) : Verdict :=
  if e.premises != [] then .premises
  else if consumer.layout != e.claim.layout then .binding
  else if consumer != e.claim then .target
  else .accept

theorem contextual_sound {s : Site} {e : Evidence} (h : bind s e = .accept)
    (k : List Amp → α) : k (before s) = k (after s) := by
  by_cases empty : e.premises = []
  · by_cases same : s = e.claim
    · subst s
      exact congrArg k (e.proof (by simp [empty])).2
    · by_cases wiresMatch : s.layout = e.claim.layout <;> simp [bind, empty, wiresMatch, same] at h
  · simp [bind, empty] at h

def hh : Site := ⟨.exactOp, [.h,.h], [], .controlFirst, []⟩
def hyh : Site := ⟨.channel, [.h,.y,.h], [.y], .controlFirst, []⟩
def coherent : Site := { hyh with use := .controlled }
def compensated : Site := { coherent with correction := [.z] }

def witness (s : Site) (h : ValidRewrite s) : Evidence := ⟨s, [], fun _ => h⟩
theorem channelProof : ValidRewrite hyh := by decide +kernel
theorem hyh_phase (r c : Bool) :
    denote hyh.original r c = mul (phase .z) (denote hyh.replacement r c) := by
  cases r <;> cases c <;> decide +kernel

-- The input sequence runs from left to right, so temporal [Y,X] denotes XY.
def yThenX : List Gate := [.y, .x]
theorem xy_phase (r c : Bool) :
    denote yThenX r c = mul (phase .s) (denote [.z] r c) := by
  cases r <;> cases c <;> decide +kernel

theorem xy_reverse_phase (r c : Bool) :
    denote [.z] r c = mul (conj (phase .s)) (denote yThenX r c) := by
  cases r <;> cases c <;> decide +kernel

theorem phase_unit (g : PhaseGate) : mul (phase g) (conj (phase g)) = one := by
  cases g <;> decide +kernel

def channelEvidence := witness hyh channelProof

example : bind hh (witness hh (by decide +kernel)) = .accept := by decide +kernel
example : bind hyh channelEvidence = .accept := by decide +kernel
example : bind coherent channelEvidence = .target := by decide +kernel
example : ¬ ValidRewrite coherent := by decide +kernel
example : bind compensated (witness compensated (by decide +kernel)) = .accept := by decide +kernel
def swapped : Site := { compensated with layout := .controlSecond }
example : bind compensated (witness swapped (by decide +kernel)) = .binding := by decide +kernel
example : before swapped ≠ before compensated := by decide +kernel
def conditionalEvidence : Evidence :=
  ⟨coherent, [.observationsAgree], fun authorized =>
    ⟨Or.inl rfl, authorized .observationsAgree (by simp)⟩⟩
example : bind coherent conditionalEvidence = .premises := by decide +kernel

end
end Q0
