/-! # Sign bounds, explicit planar witnesses, and finite diagonal Schatten inequalities -/

namespace EshkolGrothendieckEra1Functional

/-- Binary sign value in {-1, +1}. -/
inductive Sign where
  | neg : Sign
  | pos : Sign
  deriving DecidableEq, Repr

def Sign.toInt : Sign → Int
  | .neg => -1
  | .pos => 1

/-- CHSH bilinear form on four signs: x1*y1 + x1*y2 + x2*y1 - x2*y2. -/
def chshEval (x1 x2 y1 y2 : Sign) : Int :=
  x1.toInt * y1.toInt + x1.toInt * y2.toInt + x2.toInt * y1.toInt - x2.toInt * y2.toInt

/-- Exhaustive CHSH classical bound over all 16 Sign assignments. -/
theorem chsh_classical_bound_signs (x1 x2 y1 y2 : Sign) :
    -2 ≤ chshEval x1 x2 y1 y2 ∧ chshEval x1 x2 y1 y2 ≤ 2 := by
  cases x1 <;> cases x2 <;> cases y1 <;> cases y2 <;> decide

/-- Exhaustive CHSH classical bound over all 16 integer sign assignments in {-1, +1}^4. -/
theorem chsh_classical_bound_all_16 (x1 x2 y1 y2 : Int)
    (hx1 : x1 = -1 ∨ x1 = 1) (hx2 : x2 = -1 ∨ x2 = 1)
    (hy1 : y1 = -1 ∨ y1 = 1) (hy2 : y2 = -1 ∨ y2 = 1) :
    -2 ≤ x1 * y1 + x1 * y2 + x2 * y1 - x2 * y2 ∧
    x1 * y1 + x1 * y2 + x2 * y1 - x2 * y2 ≤ 2 := by
  rcases hx1 with rfl | rfl <;>
  rcases hx2 with rfl | rfl <;>
  rcases hy1 with rfl | rfl <;>
  rcases hy2 with rfl | rfl <;>
  decide

/-- Exact attainment of the CHSH classical maximum 2. -/
theorem chsh_classical_max_attained :
    ∃ x1 x2 y1 y2 : Sign, chshEval x1 x2 y1 y2 = 2 := by
  exact ⟨.pos, .pos, .pos, .neg, rfl⟩

/-- Test vectors (a,b) in the square lattice or in the triangular lattice.
    The latter pairing is twice the Euclidean dot product in a basis at 60°. -/
abbrev Plane := Int × Int
def dot (u v : Plane) := u.1*v.1 + u.2*v.2
def triangularDot (u v : Plane) := 2*dot u v + u.1*v.2 + u.2*v.1
def vadd (u v : Plane) : Plane := (u.1+v.1,u.2+v.2)
def vsub (u v : Plane) : Plane := (u.1-v.1,u.2-v.2)

def chshNumerator (u v : Plane) :=
  dot (vadd u v) u + dot (vadd u v) v + dot (vsub u v) u - dot (vsub u v) v

private theorem square_bound {x b : Int} (lo : -b ≤ x) (hi : x ≤ b) : x*x ≤ b*b := by
  have h := Int.mul_nonneg (show 0 ≤ b-x by omega) (show 0 ≤ b+x by omega)
  rw [Int.sub_mul, Int.mul_add, Int.mul_add, Int.mul_comm x b] at h
  omega

/-- The two column vectors are orthonormal. The two optimal row directions
    before normalization have squared norm two. The score comparison clears
    these common norms; every sign assignment is strictly below the witness. -/
theorem chsh_hilbert_separation (x₁ x₂ y₁ y₂ : Sign) :
    let u : Plane := (1,0)
    let v : Plane := (0,1)
    let s := chshEval x₁ x₂ y₁ y₂
    dot u u = 1 ∧ dot v v = 1 ∧ dot u v = 0 ∧
    dot (vadd u v) (vadd u v) = 2 ∧ dot (vsub u v) (vsub u v) = 2 ∧
    2*(s*s) < (chshNumerator u v)^2 ∧
    (chshNumerator u v)^2 * 10000^2 < 2 * (2*17822)^2 := by
  have h := chsh_classical_bound_signs x₁ x₂ y₁ y₂
  have hs := square_bound h.1 h.2
  refine ⟨rfl, rfl, rfl, rfl, rfl, ?_, by decide⟩
  change 2 * (_ * _) < (16 : Int)
  omega

/-- 3x3 frustrated 6-cycle bilinear form B_3:
    x0*y0 + x0*y1 + x1*y0 + x1*y2 + x2*y1 - x2*y2. -/
def c6Eval (x0 x1 x2 y0 y1 y2 : Sign) : Int :=
  x0.toInt * y0.toInt + x0.toInt * y1.toInt +
  x1.toInt * y0.toInt + x1.toInt * y2.toInt +
  x2.toInt * y1.toInt - x2.toInt * y2.toInt

/-- Exhaustive classical bound (<= 4) over all 64 sign assignments for B_3. -/
theorem frustrated_c6_classical_bound_all_64 (x0 x1 x2 y0 y1 y2 : Sign) :
    -4 ≤ c6Eval x0 x1 x2 y0 y1 y2 ∧ c6Eval x0 x1 x2 y0 y1 y2 ≤ 4 := by
  cases x0 <;> cases x1 <;> cases x2 <;> cases y0 <;> cases y1 <;> cases y2 <;> decide

def c6Numerator (u v w : Plane) :=
  triangularDot (vadd u v) u + triangularDot (vadd u v) v +
  triangularDot (vadd u w) u + triangularDot (vadd u w) w +
  triangularDot (vsub v w) v - triangularDot (vsub v w) w

/-- Squaring the normalized triangular-lattice score gives 18²/(6·2)=27. -/
theorem frustrated_c6_separation (x₀ x₁ x₂ y₀ y₁ y₂ : Sign) :
    let u : Plane := (1,0)
    let v : Plane := (0,1)
    let w : Plane := (1,-1)
    let s := c6Eval x₀ x₁ x₂ y₀ y₁ y₂
    triangularDot u u = 2 ∧ triangularDot v v = 2 ∧ triangularDot w w = 2 ∧
    triangularDot (vadd u v) (vadd u v) = 6 ∧ triangularDot (vadd u w) (vadd u w) = 6 ∧
    triangularDot (vsub v w) (vsub v w) = 6 ∧
    c6Eval .pos .pos .pos .pos .pos .pos = 4 ∧
    12 * (s*s) < (c6Numerator u v w)^2 ∧
    (c6Numerator u v w)^2 * 10000^2 < 12 * (4*17822)^2 := by
  have h := frustrated_c6_classical_bound_all_64 x₀ x₁ x₂ y₀ y₁ y₂
  have hs := square_bound h.1 h.2
  refine ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl,?_,by decide⟩
  change 12 * (_ * _) < (324 : Int)
  omega

/-- Diagonal operator Schatten 1-norm (trace norm) for nonnegative diagonal entries. -/
def diagTraceNorm : List Nat → Nat
  | [] => 0
  | a :: rest => a + diagTraceNorm rest

/-- Diagonal operator squared Hilbert-Schmidt norm: sum a_i^2. -/
def diagHsNormSq : List Nat → Nat
  | [] => 0
  | a :: rest => a * a + diagHsNormSq rest

/-- Diagonal operator infinity norm (operator norm): max a_i. -/
def diagOpNorm : List Nat → Nat
  | [] => 0
  | a :: rest => max a (diagOpNorm rest)

/-- Diagonal entries of the n x n identity operator I_n. -/
def identityDiag (n : Nat) : List Nat :=
  List.replicate n 1

/-- Exact trace norm of I_n equals n. -/
theorem identity_trace_norm_eq (n : Nat) :
    diagTraceNorm (identityDiag n) = n := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp [identityDiag, List.replicate, diagTraceNorm] at ih ⊢
    omega

/-- Exact operator norm of I_n equals 1 for all n >= 1. -/
theorem identity_op_norm_eq (n : Nat) (hn : n ≥ 1) :
    diagOpNorm (identityDiag n) = 1 := by
  induction n with
  | zero => omega
  | succ k ih =>
    cases k with
    | zero => rfl
    | succ m =>
      have hm : m + 1 ≥ 1 := by omega
      have hsub := ih hm
      simp [identityDiag, List.replicate, diagOpNorm] at hsub ⊢
      omega

/-- Finite-dimensional identity compressions:
    for every dimension n >= 2, trace(I_n) = n > 1 = ||I_n||_inf. -/
theorem identity_trace_norm_diverges_over_op_norm (n : Nat) (hn : n ≥ 2) :
    diagTraceNorm (identityDiag n) > diagOpNorm (identityDiag n) := by
  have htr : diagTraceNorm (identityDiag n) = n := identity_trace_norm_eq n
  have hop : diagOpNorm (identityDiag n) = 1 := identity_op_norm_eq n (by omega)
  omega

/-- Unbounded divergence of trace norm over operator norm on I_n:
    for every M : Nat, there exists n >= 1 with ||I_n||_1 > M * ||I_n||_inf. -/
theorem identity_trace_to_operator_ratio_unbounded (M : Nat) :
    ∃ n : Nat, n ≥ 1 ∧ diagTraceNorm (identityDiag n) > M * diagOpNorm (identityDiag n) := by
  refine ⟨M + 1, by omega, ?_⟩
  rw [identity_trace_norm_eq (M + 1), identity_op_norm_eq (M + 1) (by omega)]
  omega

/-- Exact Schatten lower inequality: ||D||_inf^2 <= ||D||_2^2 for all finite diagonal operators. -/
theorem diag_op_sq_le_hs_sq (d : List Nat) :
    diagOpNorm d * diagOpNorm d ≤ diagHsNormSq d := by
  induction d with
  | nil => simp [diagOpNorm, diagHsNormSq]
  | cons a rest ih =>
    simp only [diagOpNorm, diagHsNormSq]
    by_cases h : a ≤ diagOpNorm rest
    · have hm : max a (diagOpNorm rest) = diagOpNorm rest := Nat.max_eq_right h
      rw [hm]
      omega
    · have hlt : diagOpNorm rest ≤ a := by omega
      have hm : max a (diagOpNorm rest) = a := Nat.max_eq_left hlt
      rw [hm]
      omega

/-- Exact Schatten upper inequality: ||D||_2^2 <= ||D||_1^2 for all finite diagonal operators. -/
theorem diag_hs_sq_le_trace_sq (d : List Nat) :
    diagHsNormSq d ≤ diagTraceNorm d * diagTraceNorm d := by
  induction d with
  | nil => simp [diagHsNormSq, diagTraceNorm]
  | cons a rest ih =>
    simp only [diagHsNormSq, diagTraceNorm]
    have hexp : (a + diagTraceNorm rest) * (a + diagTraceNorm rest) =
        a * a + 2 * a * diagTraceNorm rest + diagTraceNorm rest * diagTraceNorm rest := by
      have h1 : (a + diagTraceNorm rest) * (a + diagTraceNorm rest) =
          a * (a + diagTraceNorm rest) + diagTraceNorm rest * (a + diagTraceNorm rest) :=
        Nat.add_mul a (diagTraceNorm rest) (a + diagTraceNorm rest)
      have h2 : a * (a + diagTraceNorm rest) = a * a + a * diagTraceNorm rest :=
        Nat.mul_add a a (diagTraceNorm rest)
      have h3 : diagTraceNorm rest * (a + diagTraceNorm rest) =
          diagTraceNorm rest * a + diagTraceNorm rest * diagTraceNorm rest :=
        Nat.mul_add (diagTraceNorm rest) a (diagTraceNorm rest)
      have h4 : diagTraceNorm rest * a = a * diagTraceNorm rest :=
        Nat.mul_comm (diagTraceNorm rest) a
      have h5 : 2 * a * diagTraceNorm rest = a * diagTraceNorm rest + a * diagTraceNorm rest := by
        calc 2 * a * diagTraceNorm rest
            = (a + a) * diagTraceNorm rest := by rw [Nat.two_mul]
          _ = a * diagTraceNorm rest + a * diagTraceNorm rest :=
                Nat.add_mul a a (diagTraceNorm rest)
      omega
    omega

/-- Complete exact Schatten norm chain ||D||_inf <= ||D||_2 <= ||D||_1 (squared). -/
theorem schatten_chain_exact_sq (d : List Nat) :
    diagOpNorm d * diagOpNorm d ≤ diagHsNormSq d ∧
    diagHsNormSq d ≤ diagTraceNorm d * diagTraceNorm d :=
  ⟨diag_op_sq_le_hs_sq d, diag_hs_sq_le_trace_sq d⟩

/-- l_1 norm of a nonnegative list. -/
def listSum : List Nat → Nat
  | [] => 0
  | x :: xs => x + listSum xs

/-- Sum of all entries of the outer product matrix u (x) v. -/
def outerProductSum : List Nat → List Nat → Nat
  | [], _ => 0
  | a :: as, bs => listSum (bs.map (fun b => a * b)) + outerProductSum as bs

private theorem listSum_map_mul (a : Nat) (bs : List Nat) :
    listSum (bs.map (fun b => a * b)) = a * listSum bs := by
  induction bs with
  | nil => simp [listSum]
  | cons b rest ih =>
    simp [listSum, ih, Nat.mul_add]

/-- The entrywise ℓ¹ norm of a nonnegative rank-one matrix factors. -/
theorem rank_one_crossnorm_equality (u v : List Nat) :
    outerProductSum u v = listSum u * listSum v := by
  induction u with
  | nil => simp [outerProductSum, listSum]
  | cons a rest ih =>
    simp [outerProductSum, listSum, listSum_map_mul, ih, Nat.add_mul]

end EshkolGrothendieckEra1Functional
