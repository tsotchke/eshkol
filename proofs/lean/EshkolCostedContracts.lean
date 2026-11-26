/-
  Finite workloads, strategy-dependent beta costs, selector objectives and
  deferred-update work laws.

  Model-to-Implementation Correspondence
  lib/core/blc/costed_contracts.esk computes share-first and duplicate-first
  costs in blc-at-verify-strategy-cost-divergence and compares selector rows
  in blc-at-convex-dominated-2?. tests/stdlib/esa_costed_contracts_test.esk
  checks these witnesses. The unit of work here is a contraction. In
  lib/core/sblc.esk, sblc-run-loop records beta contractions and coin draws
  separately. tests/stdlib/sblc_resource_test.esk checks both counters.
-/

namespace EshkolCostedContracts

/-! ## 1. Update 002: Tariff Breakeven & Strategy Cost -/

theorem tariff_breakeven_240_vs_241 :
    240 * 17 < 4096 ∧ 241 * 17 > 4096 := by
  decide

/-- Numerator of expected work. Each row carries an integer mass, and the
    common denominator for comparing costs is the sum of the masses. -/
def weightedWork {R : Type} (cost : R → Nat) : List (R × Nat) → Nat
  | [] => 0
  | (r, w) :: rest => w * cost r + weightedWork cost rest

theorem weightedWork_mono {R : Type} (a b : R → Nat) (rows : List (R × Nat))
    (hle : ∀ row ∈ rows, a row.1 ≤ b row.1) : weightedWork a rows ≤ weightedWork b rows := by
  induction rows with
  | nil => exact Nat.le_refl _
  | cons row rest ih =>
    exact Nat.add_le_add (Nat.mul_le_mul_left _ (hle row (by simp)))
      (ih (fun r hr => hle r (List.mem_cons_of_mem _ hr)))

/-- Equal or empty rows may occur anywhere. One strict improvement carrying
    positive mass suffices for strict expected-cost dominance. -/
theorem weightedWork_lt_of_supported_gain {R : Type} (a b : R → Nat) (rows : List (R × Nat))
    (hle : ∀ row ∈ rows, a row.1 ≤ b row.1)
    (gain : ∃ row ∈ rows, 0 < row.2 ∧ a row.1 < b row.1) :
    weightedWork a rows < weightedWork b rows := by
  induction rows with
  | nil => simp at gain
  | cons row rest ih =>
    have tailLe := fun r (hr : r ∈ rest) => hle r (List.mem_cons_of_mem _ hr)
    obtain ⟨r, hr, hw, hlt⟩ := gain
    rcases List.mem_cons.mp hr with h | ht
    · subst r
      exact Nat.add_lt_add_of_lt_of_le (Nat.mul_lt_mul_of_pos_left hlt hw)
        (weightedWork_mono a b rest tailLe)
    · exact Nat.add_lt_add_of_le_of_lt (Nat.mul_le_mul_left _ (hle row (by simp)))
        (ih tailLe ⟨r, ht, hw, hlt⟩)

/-- A request selects a path through a finite strategy. A leaf charges for
    its copies of a computation. Each visited decision charges its own toll. -/
inductive Strategy (R : Type) where
  | leaf (overhead copies work : Nat)
  | branch (test : R → Bool) (toll : Nat) (yes no : Strategy R)

def Strategy.cost {R : Type} : Strategy R → R → Nat
  | .leaf overhead copies work, _ => overhead + copies * work
  | .branch test toll yes no, r => toll + (if test r then yes.cost r else no.cost r)

/-- For arbitrary decision trees, pathwise savings on the workload support
    carry through to expected work, even when the trees branch differently. -/
theorem strategy_expected_cost_dominance {R : Type} (a b : Strategy R) (rows : List (R × Nat))
    (hle : ∀ row ∈ rows, a.cost row.1 ≤ b.cost row.1)
    (gain : ∃ row ∈ rows, 0 < row.2 ∧ a.cost row.1 < b.cost row.1) :
    weightedWork a.cost rows < weightedWork b.cost rows :=
  weightedWork_lt_of_supported_gain _ _ rows hle gain

/-- Sharing preserves all decisions and reduces a demanded leaf just once. -/
def Strategy.share {R : Type} : Strategy R → Strategy R
  | .leaf overhead copies work => .leaf overhead (min copies 1) work
  | .branch test toll yes no => .branch test toll yes.share no.share

def Strategy.savedWork {R : Type} : Strategy R → R → Nat
  | .leaf _ copies work, _ => (copies - 1) * work
  | .branch test _ yes no, r => if test r then yes.savedWork r else no.savedWork r

theorem strategy_sharing_cost_decomposition {R : Type} (s : Strategy R) (r : R) :
    s.cost r = s.share.cost r + s.savedWork r := by
  induction s with
  | leaf overhead copies work =>
    cases copies <;> simp [Strategy.cost, Strategy.share, Strategy.savedWork, Nat.succ_mul,
      Nat.min_eq_right (Nat.succ_pos _),
      Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
  | branch test toll yes no hy hn =>
    cases h : test r <;> simp [Strategy.cost, Strategy.share, Strategy.savedWork, h,
      hy, hn, Nat.add_assoc]

theorem sharing_strictly_lowers_expected_work {R : Type} (s : Strategy R) (rows : List (R × Nat))
    (gain : ∃ row ∈ rows, 0 < row.2 ∧ 0 < s.savedWork row.1) :
    weightedWork s.share.cost rows < weightedWork s.cost rows := by
  apply strategy_expected_cost_dominance
  · intro row _
    rw [strategy_sharing_cost_decomposition s]
    exact Nat.le_add_right _ _
  · obtain ⟨row, hr, hw, hs⟩ := gain
    refine ⟨row, hr, hw, ?_⟩
    rw [strategy_sharing_cost_decomposition s]
    omega

/-- Reduce once before duplication versus reduce each copy on the same demand. -/
theorem share_before_duplication (overhead copies work : Nat) (hc : 1 < copies) (hw : 0 < work) :
    let a : Strategy Unit := .leaf overhead 1 work
    let b : Strategy Unit := .leaf overhead copies work
    weightedWork a.cost [((), 1)] < weightedWork b.cost [((), 1)] := by
  intro a b
  have gain : 0 < b.savedWork () := Nat.mul_pos (by omega : 0 < copies - 1) hw
  have dom := sharing_strictly_lowers_expected_work b [((), 1)]
    ⟨((), 1), by simp, by decide, gain⟩
  simpa [a, b, Strategy.share, Nat.min_eq_right (by omega : 1 ≤ copies)] using dom

theorem strategy_cost_2_vs_72_concrete_witness :
    let a : Strategy Unit := .leaf 1 1 1
    let b : Strategy Unit := .leaf 1 71 1
    let costs := (weightedWork a.cost [((), 1)], weightedWork b.cost [((), 1)])
    costs = (2, 72) ∧ costs.1 < costs.2 ∧ costs.1 + costs.2 = 74 :=
  ⟨rfl, share_before_duplication 1 71 1 (by decide) (by decide), rfl⟩

/-! ## 2. Convex dominance over finite workloads -/

theorem weightedWork_linear {R : Type} (rows : List (R × Nat)) (a b : R → Nat) (m n : Nat) :
    weightedWork (fun r => m * a r + n * b r) rows =
      m * weightedWork a rows + n * weightedWork b rows := by
  induction rows with
  | nil => simp [weightedWork]
  | cons row rest ih =>
    simp only [weightedWork, ih, Nat.mul_add]
    simp only [Nat.mul_left_comm row.2, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

/-- A convex mixture may tie the candidate on some requests. If it improves
    on one request of positive mass and never costs more on the workload,
    one of its two component strategies has strictly smaller expected cost. -/
theorem convex_dominated_never_minimizes_expected_work {R : Type} (rows : List (R × Nat))
    (a b c : R → Nat) (num den : Nat) (hnum : num ≤ den)
    (hle : ∀ row ∈ rows, num * a row.1 + (den - num) * b row.1 ≤ den * c row.1)
    (gain : ∃ row ∈ rows, 0 < row.2 ∧ num * a row.1 + (den - num) * b row.1 < den * c row.1) :
    weightedWork a rows < weightedWork c rows ∨ weightedWork b rows < weightedWork c rows := by
  have h := weightedWork_lt_of_supported_gain
    (fun r => num * a r + (den - num) * b r) (fun r => den * c r) rows hle gain
  have scale := weightedWork_linear rows c c den 0
  simp only [Nat.zero_mul, Nat.add_zero] at scale
  rw [weightedWork_linear, scale] at h
  by_cases ha : weightedWork a rows < weightedWork c rows
  · exact Or.inl ha
  · right
    apply Nat.lt_of_not_ge
    intro hb
    have bound := Nat.add_le_add (Nat.mul_le_mul_left num (Nat.le_of_not_gt ha))
      (Nat.mul_le_mul_left (den - num) hb)
    rw [← Nat.add_mul, Nat.add_sub_of_le hnum] at bound
    exact (Nat.not_lt_of_ge bound) h

/-- Concrete Update 003 witness: `(116, 116)` is Pareto efficient against
    `(80, 150)` and `(150, 80)`, strictly convex-dominated (`230 < 232`),
    yet uniquely wins minimax excess above baseline `(100, 100)` (`16 < 50`). -/
theorem pareto_convex_dominated_wins_minimax :
    ¬ (80 ≤ 116 ∧ 150 ≤ 116) ∧ ¬ (150 ≤ 116 ∧ 80 ≤ 116) ∧
    (1 * 80 + 1 * 150 < 2 * 116) ∧
    (1 * 150 + 1 * 80 < 2 * 116) ∧
    (max (116 - 100) (116 - 100) = 16) ∧
    (16 < max (80 - 100) (150 - 100)) ∧
    (16 < max (150 - 100) (80 - 100)) := by
  decide

/-! ## 3. Update 004: Pure Lisp->BLC Frontend & SETF Refusal -/

inductive BLCTerm where
  | var : Nat → BLCTerm
  | lam : BLCTerm → BLCTerm
  | app : BLCTerm → BLCTerm → BLCTerm
  deriving DecidableEq, Repr

def encodeBLC : BLCTerm → String
  | .var n => String.mk (List.replicate n '1' ++ ['0'])
  | .lam b => "00" ++ encodeBLC b
  | .app f a => "01" ++ encodeBLC f ++ encodeBLC a

inductive PureLisp where
  | sym : String → PureLisp
  | defun1 : String → String → PureLisp → PureLisp
  | setf : String → Nat → PureLisp
  deriving DecidableEq, Repr

def envLookup (s : String) : List String → Option Nat
  | [] => none
  | x :: xs => if x == s then some 1 else (envLookup s xs).map (· + 1)

def lowerPure (env : List String) : PureLisp → Option BLCTerm
  | .sym s => (envLookup s env).map BLCTerm.var
  | .defun1 _ param body => (lowerPure (param :: env) body).map BLCTerm.lam
  | .setf _ _ => none

/-- A parameter shadows the function name and lowers to the four-bit identity. -/
theorem shadowed_param_lowers_to_four_bit_identity :
    (lowerPure [] (.defun1 "f" "f" (.sym "f"))).map encodeBLC = some "0010" := by
  decide

theorem setf_refused_in_pure_frontend (env : List String) (name param cell : String) (v : Nat) :
    lowerPure env (.defun1 name param (.setf cell v)) = none := rfl

/-! ## 4. Update 004: Sparse-Hessian Undeclared Dependency-Mask Counterexample -/

/-- Hessian-vector product for (x + y)^2 / 2 on natural-valued seeds. -/
def hvpHalfSumSq (v0 v1 : Nat) : Nat × Nat := (v0 + v1, v0 + v1)

abbrev Matrix2 := (Nat × Nat) × (Nat × Nat)

/-- Recover the columns of a two-dimensional HVP using orthogonal seeds. -/
def recoverColumns (hvp : Nat → Nat → Nat × Nat) : Matrix2 :=
  let x := hvp 1 0
  let y := hvp 0 1
  ((x.1, y.1), (x.2, y.2))

/-- With a diagonal-only mask both columns share one seed. -/
def recoverDiagonal (hvp : Nat → Nat → Nat × Nat) : Matrix2 :=
  let xy := hvp 1 1
  ((xy.1, 0), (0, xy.2))

theorem recoverColumns_of_matrix (m : Matrix2) :
    recoverColumns (fun x y => (m.1.1 * x + m.1.2 * y, m.2.1 * x + m.2.2 * y)) = m := by
  simp [recoverColumns]

theorem sparse_hessian_incomplete_mask_corrupts_diagonal :
    recoverDiagonal hvpHalfSumSq = ((2, 0), (0, 2)) ∧
      (recoverDiagonal hvpHalfSumSq).1.1 ≠ (recoverColumns hvpHalfSumSq).1.1 ∧
      (recoverDiagonal hvpHalfSumSq).2.2 ≠ (recoverColumns hvpHalfSumSq).2.2 := by
  decide

theorem sparse_hessian_complete_mask_exact :
    recoverColumns hvpHalfSumSq = ((1, 1), (1, 1)) := by
  decide

/-! ## 5. Update 004: Deferred Affine Updates Exact Work Law -/

/-- Cumulative lazy catch-up work for a single bank starting at timestamp `cur`
    over a monotone list of demanded timestamps `ts`:
      - if `cur ≤ t`, pays `t - cur` transitions and advances `cur := t`;
      - otherwise pays `0`. -/
def bankLazyWork : Nat → List Nat → Nat
  | _, [] => 0
  | cur, t :: rest =>
    if cur ≤ t then (t - cur) + bankLazyWork t rest
    else bankLazyWork cur rest

def bankFinalUse : Nat → List Nat → Nat
  | cur, [] => cur
  | cur, t :: rest =>
    if cur ≤ t then bankFinalUse t rest
    else bankFinalUse cur rest

/-- General Exact Work Conservation Law (Update 004):
    for ANY schedule of demand timestamps `ts` and starting timestamp `cur`,
    `cur + bankLazyWork cur ts = bankFinalUse cur ts`.
    Starting from `cur = 0`, `bankLazyWork 0 ts = bankFinalUse 0 ts`! -/
theorem lazy_affine_bank_work_conservation (cur : Nat) (ts : List Nat) :
    cur + bankLazyWork cur ts = bankFinalUse cur ts := by
  induction ts generalizing cur with
  | nil => simp [bankLazyWork, bankFinalUse]
  | cons t rest ih =>
    simp only [bankLazyWork, bankFinalUse]
    split
    · rename_i hle
      have hih := ih t
      omega
    · exact ih cur

theorem lazy_affine_bank_work_equals_final_use (ts : List Nat) :
    bankLazyWork 0 ts = bankFinalUse 0 ts := by
  have h := lazy_affine_bank_work_conservation 0 ts
  omega

/-- Sum the actual catch-up work for the demand history of each bank. -/
def scheduleWork (banks steps : Nat) (demanded : Nat → Nat → Bool) : Nat :=
  let times := (List.range steps).map (· + 1)
  (List.range banks).foldl (fun total b => total + bankLazyWork 0 (times.filter (demanded b))) 0

/-- The four demand policies of blc-at-run-affine-schedule on eight banks
    and 64 timesteps, together with their dense baseline. -/
theorem lazy_affine_four_schedules_exact_counts :
    scheduleWork 8 64 (fun b _ => b < 2) = 128 ∧
    scheduleWork 8 64 (fun b t => (t - 1) % 8 == b) = 484 ∧
    scheduleWork 8 64 (fun b t => b < 2 || (b < 4 && t ≤ 32)) = 192 ∧
    scheduleWork 8 64 (fun b t => b == 0 || (b == 7 && t == 64)) = 128 ∧
    8 * 64 = 512 := by
  decide

/-! ## 6. Macro-Rule Cost Composition & Non-Vacuous Claim Binding -/

def macroTotalCost : List (Nat × Nat) → Nat
  | [] => 0
  | (cnt, step) :: rest => cnt * step + macroTotalCost rest

theorem macro_cost_append (xs ys : List (Nat × Nat)) :
    macroTotalCost (xs ++ ys) = macroTotalCost xs + macroTotalCost ys := by
  induction xs with
  | nil => simp [macroTotalCost]
  | cons x rest ih =>
    cases x with
    | mk cnt step =>
      change cnt * step + macroTotalCost (rest ++ ys) =
        (cnt * step + macroTotalCost rest) + macroTotalCost ys
      rw [ih, Nat.add_assoc]

/-- The two arithmetic targets have distinct operand-result pairs and the
    same image under a constant Boolean report. -/
theorem nonvacuous_claim_distinguishes_targets :
    ((1 + 1, 2) ≠ (3 * 4, 12)) ∧
    ((fun _ : Nat × Nat => true) (1 + 1, 2) = (fun _ : Nat × Nat => true) (3 * 4, 12)) := by
  decide

#print axioms tariff_breakeven_240_vs_241
#print axioms weightedWork_mono
#print axioms weightedWork_lt_of_supported_gain
#print axioms strategy_expected_cost_dominance
#print axioms strategy_sharing_cost_decomposition
#print axioms sharing_strictly_lowers_expected_work
#print axioms share_before_duplication
#print axioms strategy_cost_2_vs_72_concrete_witness
#print axioms weightedWork_linear
#print axioms convex_dominated_never_minimizes_expected_work
#print axioms pareto_convex_dominated_wins_minimax
#print axioms shadowed_param_lowers_to_four_bit_identity
#print axioms setf_refused_in_pure_frontend
#print axioms recoverColumns_of_matrix
#print axioms sparse_hessian_incomplete_mask_corrupts_diagonal
#print axioms sparse_hessian_complete_mask_exact
#print axioms lazy_affine_bank_work_conservation
#print axioms lazy_affine_bank_work_equals_final_use
#print axioms lazy_affine_four_schedules_exact_counts
#print axioms macro_cost_append
#print axioms nonvacuous_claim_distinguishes_targets

end EshkolCostedContracts
