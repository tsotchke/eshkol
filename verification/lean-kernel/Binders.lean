import Init

namespace Verificatum.Binders

inductive Ty where
  | atom
  | arrow (domain codomain : Ty)

-- The pure annotated fragment. The theorem applies even to untyped open terms.
inductive Tm where
  | var (i : Nat)
  | lam (domain : Ty) (body : Tm)
  | app (fn arg : Tm)

-- Positive indices produced by erasure correspond to Eshkol's BLC variables.
inductive BLC where
  | var (i : Nat)
  | lam (body : BLC)
  | app (fn arg : BLC)
deriving DecidableEq

def erase : Tm → BLC
  | .var i => .var (i + 1)
  | .lam _ b => .lam (erase b)
  | .app f a => .app (erase f) (erase a)

def shift (d c : Nat) : Tm → Tm
  | .var i => .var (if c ≤ i then i + d else i)
  | .lam A b => .lam A (shift d (c+1) b)
  | .app f a => .app (shift d c f) (shift d c a)

def shiftB (d c : Nat) : BLC → BLC
  | .var i => .var (if c ≤ i then i + d else i)
  | .lam b => .lam (shiftB d (c+1) b)
  | .app f a => .app (shiftB d c f) (shiftB d c a)

theorem erase_shift (t : Tm) (d c : Nat) :
    erase (shift d c t) = shiftB d (c+1) (erase t) := by
  induction t generalizing c with
  | var i =>
    by_cases h : c ≤ i <;> simp [erase, shift, shiftB, h, Nat.add_right_comm]
  | lam A b ih => simp [erase, shift, shiftB, ih]
  | app f a hf ha => simp [erase, shift, shiftB, hf, ha]

-- Fused capture-avoiding contraction, as used by k-inst on the pure fragment.
def inst (arg : Tm) (depth : Nat) : Tm → Tm
  | .var i => if i = depth then shift depth 0 arg
              else .var (if depth < i then i-1 else i)
  | .lam A b => .lam A (inst arg (depth+1) b)
  | .app f a => .app (inst arg depth f) (inst arg depth a)

def instB (arg : BLC) (binder : Nat) : BLC → BLC
  | .var i => if i = binder then shiftB (binder-1) 1 arg
              else .var (if binder < i then i-1 else i)
  | .lam b => .lam (instB arg (binder+1) b)
  | .app f a => .app (instB arg binder f) (instB arg binder a)

theorem erase_inst (body arg : Tm) (depth : Nat) :
    erase (inst arg depth body) = instB (erase arg) (depth+1) (erase body) := by
  induction body generalizing depth with
  | var i =>
    by_cases h : i = depth
    · subst i; simp [inst, instB, erase, erase_shift]
    · by_cases lt : depth < i
      · have pos : 0 < i := Nat.zero_lt_of_lt lt
        simp [inst, instB, erase, h, lt, Nat.sub_add_cancel pos]
      · simp [inst, instB, erase, h, lt]
  | lam A b ih => simp [inst, instB, erase, ih]
  | app f a hf ha => simp [inst, instB, erase, hf, ha]

-- The three passes in core.blc: lift the argument, replace the binder while
-- lifting under lambdas, then lower the indices outside the removed binder.
def lowerB (c : Nat) : BLC → BLC
  | .var i => .var (if c ≤ i then i-1 else i)
  | .lam b => .lam (lowerB (c+1) b)
  | .app f a => .app (lowerB c f) (lowerB c a)

def replaceB (j : Nat) (s : BLC) : BLC → BLC
  | .var i => if i = j then s else .var i
  | .lam b => .lam (replaceB (j+1) (shiftB 1 1 s) b)
  | .app f a => .app (replaceB j s f) (replaceB j s a)

private theorem shifted_twice (t : BLC) (c d : Nat) :
    shiftB 1 c (shiftB d c t) = shiftB (d+1) c t := by
  induction t generalizing c with
  | var i =>
    by_cases h : c ≤ i
    · have hh : c ≤ i+d := by omega
      simp [shiftB, h, hh, Nat.add_assoc]
    · simp [shiftB, h]
  | lam b ih => simp [shiftB, ih]
  | app f a hf ha => simp [shiftB, hf, ha]

private theorem lower_shifted (t : BLC) (c d : Nat) :
    lowerB (c+d) (shiftB (d+1) c t) = shiftB d c t := by
  induction t generalizing c with
  | var i =>
    by_cases h : c ≤ i
    · have hh : c+d ≤ i+(d+1) := by omega
      simp [shiftB, lowerB, h, hh]
    · have hh : ¬c+d ≤ i := by omega
      simp [shiftB, lowerB, h, hh]
  | lam b ih =>
    have h : c+d+1 = c+1+d := by omega
    simp [shiftB, lowerB, h, ih]
  | app f a hf ha => simp [shiftB, lowerB, hf, ha]

theorem three_pass_inst (b a : BLC) (d : Nat) :
    lowerB (d+1) (replaceB (d+1) (shiftB (d+1) 1 a) b) = instB a (d+1) b := by
  induction b generalizing d with
  | var i =>
    by_cases h : i = d+1
    · subst i
      simp only [replaceB, instB, Nat.add_sub_cancel]
      simpa [Nat.add_comm] using lower_shifted a 1 d
    · by_cases lt : d+1 < i
      · simp [replaceB, instB, lowerB, h, lt, Nat.le_of_lt lt]
      · have hh : ¬d+1 ≤ i := by omega
        simp [replaceB, instB, lowerB, h, lt, hh]
  | lam b ih => simp [replaceB, lowerB, instB, shifted_twice, ih]
  | app f x hf hx => simp [replaceB, lowerB, instB, hf, hx]

def beta (body arg : Tm) := inst arg 0 body
def betaB (body arg : BLC) := lowerB 1 (replaceB 1 (shiftB 1 1 arg) body)

theorem erase_beta (body arg : Tm) :
    erase (beta body arg) = betaB (erase body) (erase arg) := by
  exact (erase_inst body arg 0).trans (three_pass_inst (erase body) (erase arg) 0).symm

end Verificatum.Binders

namespace Verificatum.Binders

private def rawForm : BLC → String
  | .var i => s!"(var {i})"
  | .lam b => s!"(lam {rawForm b})"
  | .app f a => s!"(app {rawForm f} {rawForm a})"

-- The finite differential corpus uses one constant annotation. The erasure
-- theorem above quantifies over arbitrary annotations in the pure grammar.
private def kernelForm : Tm → String
  | .var i => s!"(var {i})"
  | .lam _ b => s!"(lam (sort 0) {kernelForm b})"
  | .app f a => s!"(app {kernelForm f} {kernelForm a})"

private def terms (width : Nat) : Nat → List Tm
  | 0 => (List.range width).map Tm.var
  | d+1 =>
    let prev := terms width d
    let vars := (List.range width).map Tm.var
    vars ++ (terms (width+1) d).map (.lam .atom ·) ++
      prev.flatMap (fun f => vars.map (.app f ·))

-- Exact AST size, including applications with non-variable right branches.
private def exactTerms (width : Nat) : Nat → List Tm
  | 0 => []
  | 1 => (List.range width).map Tm.var
  | n+2 =>
    (exactTerms (width+1) (n+1)).map (.lam .atom ·) ++
      (List.finRange (n+1)).flatMap fun j =>
        let i := j.val
        (exactTerms width (i+1)).flatMap fun f =>
          (exactTerms width (n-i)).map (.app f ·)
termination_by size => size
decreasing_by all_goals omega

private def emit (body arg : Tm) : IO Unit := do
  let result := beta body arg
  IO.println <| String.intercalate "\t"
    [rawForm (erase body), rawForm (erase arg), rawForm (erase result),
     kernelForm body, kernelForm arg, kernelForm result]

def corpus : IO Unit := do
  for body in terms 3 2 do
    for arg in terms 2 1 do emit body arg
  -- Exhaust all bodies of size 1..4 in width 3, arguments 1..3 in width 2.
  for size in [1:5] do
    for body in exactTerms 3 size do
      for argSize in [1:4] do
        for arg in exactTerms 2 argSize do emit body arg
  -- Capture, shadowing, both application branches, outer-index decrement,
  -- and a replacement beneath 24 binders.
  emit (.lam .atom (.var 1)) (.var 0)
  emit (.lam .atom (.lam .atom (.var 0))) (.var 1)
  emit (.app (.var 0) (.var 0)) (.lam .atom (.var 1))
  emit (.lam .atom (.var 3)) (.var 1)
  emit ((List.range 24).foldl (fun b _ => .lam .atom b) (.var 24)) (.var 1)

end Verificatum.Binders

def main := Verificatum.Binders.corpus
