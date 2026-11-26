/-
  Logical draw publication, exact finite marginals and scoped receipt reconciliation.

  Model-to-Implementation Correspondence
  In lib/core/blc/stochastic_blc.esk, sblc-st-run-control1-shared-draw writes a
  memo cell once, and sblc-st-run-control2-fresh-draws writes distinct slots in
  scheduler order. Here a memo key is a logical draw ID. The trace theorem
  compares completed publications for a common oracle. The same module supplies
  sblc-st-reconcile-one and sblc-st-consume-ticket. The ticket model projects
  one selected ticket and action. tests/stdlib/esa_stochastic_blc_reconciliation_test.esk
  enumerates schedule and oracle controls and checks conflicts and replay.
  Sequential sampling in lib/core/sblc.esk uses sblc-coins-from-prng with
  eshkol_prng_random_integer in lib/core/prng.cpp. A pathwise comparison
  assumes the draw-ID assignment has already been supplied.
-/

namespace EshkolStochasticBLCConfluence

/-! ## 1. Naive Racing-Cell Publication Model (Negative Control 1B) -/

/-- Latency model: computing `false` (0) takes 1 tick; computing `true` (1) takes 2 ticks. -/
def bitLatency : Bool → Nat
  | false => 1
  | true  => 2

/-- Earliest finisher publishes to the single-write memo cell (ties pick worker 0). -/
def naiveRacePublish (b0 b1 : Bool) : Bool :=
  if bitLatency b0 ≤ bitLatency b1 then b0 else b1

/-- All 4 equiprobable fair-coin pairs `(b0, b1) ∈ {false, true}²`. -/
def allBoolPairs : List (Bool × Bool) :=
  [(false, false), (false, true), (true, false), (true, true)]

/-- Count of equiprobable outcomes where the naive race publishes `true` (1). -/
def naiveRaceTrueCount : Nat :=
  (allBoolPairs.filter (fun p => naiveRacePublish p.1 p.2)).length

/-- Exhaustive enumeration over `{0, 1} × {0, 1}` proves the naive racing-cell
    model publishes `1` in exactly `1` of `4` outcomes (`1/4 ≠ 1/2`). -/
theorem naive_race_publishes_one_quarter :
    allBoolPairs.length = 4 ∧
    naiveRaceTrueCount = 1 ∧
    2 * naiveRaceTrueCount ≠ allBoolPairs.length := by
  decide

/-! ## 2. Logical DrawId Oracle & Schedule-Independent Evaluation -/

abbrev DrawId := Nat
abbrev Oracle := DrawId → Bool

inductive WorkerOrder where
  | w0First
  | w1First
  deriving DecidableEq, Repr

structure Schedule where
  order : WorkerOrder
  w0Lag : Nat
  w1Lag : Nat
  specTicks : Nat
  deriving DecidableEq, Repr

abbrev Memo := DrawId → Option Bool

/-- A worker publishes the oracle value only if its logical cell is empty. -/
def publish (ω : Oracle) (m : Memo) (d : DrawId) : Memo :=
  fun k => if k = d then (m k).orElse (fun _ => some (ω d)) else m k

def publishTrace (ω : Oracle) (m : Memo) : List DrawId → Memo
  | [] => m
  | d :: ds => publishTrace ω (publish ω m d) ds

/-- A pre-existing value survives every publication. An empty cell fills
    precisely when its key occurs in the trace. -/
theorem publishTrace_lookup (ω : Oracle) (m : Memo) (ds : List DrawId) (k : DrawId) :
    publishTrace ω m ds k =
      if k ∈ ds then (m k).orElse (fun _ => some (ω k)) else m k := by
  induction ds generalizing m with
  | nil => simp [publishTrace]
  | cons d ds ih =>
    rw [publishTrace, ih]
    by_cases hd : k = d
    · subst d
      by_cases hs : k ∈ ds <;> cases h : m k <;> simp [publish, hs, h, Option.orElse]
    · simp [publish, hd, List.mem_cons]

/-- Reordering, retrying or duplicating publications gives the same final
    memo whenever both traces visit the same set of logical cells. -/
theorem published_traces_confluent (ω : Oracle) (m : Memo) (xs ys : List DrawId)
    (sameKeys : ∀ k, k ∈ xs ↔ k ∈ ys) : publishTrace ω m xs = publishTrace ω m ys := by
  funext k
  simp only [publishTrace_lookup, sameKeys k]

def scheduledDraws (s : Schedule) (ω : Oracle) (a b : DrawId) : List DrawId :=
  let t0 := s.w0Lag + bitLatency (ω a)
  let t1 := s.w1Lag + bitLatency (ω b)
  if t0 < t1 || (t0 == t1 && s.order == WorkerOrder.w0First) then [a, b] else [b, a]

/-- Observe the two requested slots after both workers have published. -/
def evalFreshPair (s : Schedule) (ω : Oracle) (a b : DrawId) : Bool × Bool :=
  let m := publishTrace ω (fun _ => none) (scheduledDraws s ω a b)
  ((m a).getD false, (m b).getD false)

theorem scheduled_pair_observation (s : Schedule) (ω : Oracle) (a b : DrawId) :
    evalFreshPair s ω a b = (ω a, ω b) := by
  simp only [evalFreshPair, scheduledDraws]
  split <;> simp [publishTrace_lookup, Option.orElse]

def evalSharedDraw (s : Schedule) (ω : Oracle) (d : DrawId) : Bool × Bool :=
  evalFreshPair s ω d d

/-- Shared-cell observations agree for any two worker schedules. -/
theorem draw_id_oracle_pathwise_confluent
    (s₁ s₂ : Schedule) (ω : Oracle) (d : DrawId) :
    evalSharedDraw s₁ ω d = evalSharedDraw s₂ ω d := by
  simp [evalSharedDraw, scheduled_pair_observation]

/-- Exact fair marginal for Control 1:
    over the 2 fair outcomes of `ω d₀`, for ANY schedule `s`, exactly `1` of `2`
    outcomes publishes `true` (`P(1) = 1/2`). -/
theorem draw_id_shared_marginal_exact_half (s : Schedule) (d₀ : DrawId) :
    let outcomes : List Bool := [false, true]
    let published : List Bool :=
      outcomes.map (fun bit => (evalSharedDraw s (fun _ => bit) d₀).1)
    outcomes.length = 2 ∧
    (published.filter (fun b => b)).length = 1 ∧
    2 * (published.filter (fun b => b)).length = outcomes.length := by
  have h0 : (evalSharedDraw s (fun _ => false) d₀).1 = false := by
    exact congrArg Prod.fst (scheduled_pair_observation s _ d₀ d₀)
  have h1 : (evalSharedDraw s (fun _ => true) d₀).1 = true := by
    exact congrArg Prod.fst (scheduled_pair_observation s _ d₀ d₀)
  simp [h0, h1]

/-! ## 3. Fresh Dynamic Draws `(flip(d₁), flip(d₂))` (Control 2) -/

/-- Observing the same two logical slots under either worker schedule gives equal pairs. -/
theorem draw_id_fresh_pair_pathwise_confluent
    (s₁ s₂ : Schedule) (ω : Oracle) (d₁ d₂ : DrawId) :
    evalFreshPair s₁ ω d₁ d₂ = evalFreshPair s₂ ω d₁ d₂ := by
  simp only [scheduled_pair_observation]

/-- Oracle assigning `p.1` to `d₁` and `p.2` to every other DrawId (including `d₂` when `d₁ ≠ d₂`). -/
def pairOracle (d₁ : DrawId) (p : Bool × Bool) : Oracle :=
  fun d => if d == d₁ then p.1 else p.2

/-- Exact uniform `1/4` distribution over all 4 ordered pairs for Control 2:
    for ANY schedule `s`, any distinct `d₁ ≠ d₂`, and EVERY target pair `target : Bool × Bool`,
    exactly `1` of the `4` fair oracle assignments yields `target`. -/
theorem draw_id_fresh_pairs_uniform_quarter
    (s : Schedule) (d₁ d₂ : DrawId) (hne : d₁ ≠ d₂) (target : Bool × Bool) :
    let results : List (Bool × Bool) :=
      allBoolPairs.map (fun p => evalFreshPair s (pairOracle d₁ p) d₁ d₂)
    allBoolPairs.length = 4 ∧
    (results.filter (fun r => r == target)).length = 1 := by
  have heval : ∀ p : Bool × Bool, evalFreshPair s (pairOracle d₁ p) d₁ d₂ = p := by
    intro p
    rw [scheduled_pair_observation]
    unfold pairOracle
    have hneq : (d₂ == d₁) = false := beq_false_of_ne hne.symm
    simp [hneq]
  cases target with
  | mk a b =>
    cases a <;> cases b <;> simp [heval, allBoolPairs]

/-! ## 4. Unused Diverging / Speculative Branch Isolation (Control 3) -/

structure Control3Obs where
  value : Bool × Bool
  logicalBetaSteps : Nat
  demandedDrawIds : List DrawId
  deriving DecidableEq, Repr

structure SpecRun where
  logical : Control3Obs
  physicalTicks : Nat

/-- Control 3 charges the demanded computation four logical steps. Its
    physical counter also includes latency, worker lag and private ticks. -/
def runWithSpeculation (s : Schedule) (ω : Oracle) (d : DrawId) : SpecRun :=
  { logical := ⟨evalSharedDraw s ω d, 4, [d]⟩
    physicalTicks := bitLatency (ω d) + s.w0Lag + s.specTicks }

def evalWithSpeculation (s : Schedule) (ω : Oracle) (d : DrawId) : Control3Obs :=
  (runWithSpeculation s ω d).logical

/-- The caller projection agrees across all schedules and speculative tick counts. -/
theorem draw_id_speculative_budget_isolated
    (s₁ s₂ : Schedule) (ω : Oracle) (d : DrawId) :
    evalWithSpeculation s₁ ω d = evalWithSpeculation s₂ ω d := by
  simp only [evalWithSpeculation, runWithSpeculation, evalSharedDraw, scheduled_pair_observation]

/-- Additional private ticks appear exactly in physical work, with the entire
    logical observation preserved. -/
theorem speculative_ticks_cost (s : Schedule) (ω : Oracle) (d : DrawId) (extra : Nat) :
    let more := { s with specTicks := s.specTicks + extra }
    (runWithSpeculation more ω d).physicalTicks = (runWithSpeculation s ω d).physicalTicks + extra ∧
      (runWithSpeculation more ω d).logical = (runWithSpeculation s ω d).logical := by
  exact ⟨(Nat.add_assoc _ _ _).symm, draw_id_speculative_budget_isolated _ _ ω d⟩

/-! ## 5. Scoped Receipt-Conflict Reconciliation & Single-Use Authorization Tickets -/

inductive Verdict where
  | unclaimed
  | unknown
  | observed (digest : String)
  | disputed
  deriving DecidableEq, Repr

/-- Reconciles one scoped request given durable claim flag and optional primary
    and replica completion digests. Distinct digests always yield `Verdict.disputed`. -/
def reconcile (claimed : Bool) : Option String → Option String → Verdict
  | none, none => if claimed then Verdict.unknown else Verdict.unclaimed
  | some a, none => if claimed then Verdict.observed a else Verdict.disputed
  | none, some a => if claimed then Verdict.observed a else Verdict.disputed
  | some a, some b => if claimed && (a == b) then Verdict.observed a else Verdict.disputed

/-- Reconciliation is symmetric in primary and replica observations. -/
theorem receipt_reconcile_symmetric (c : Bool) (p r : Option String) :
    reconcile c p r = reconcile c r p := by
  cases c <;> cases p with
  | none => cases r <;> rfl
  | some a =>
    cases r with
    | none => rfl
    | some b =>
      by_cases h : a = b
      · subst h; simp [reconcile]
      · have h1 : (a == b) = false := beq_false_of_ne h
        have h2 : (b == a) = false := beq_false_of_ne (Ne.symm h)
        simp [reconcile, h1, h2]

/-- Two distinct digests for the same request ID ALWAYS produce `Verdict.disputed`
    regardless of arrival order (fail-closed and symmetric). -/
theorem receipt_reconcile_dispute_symmetric_and_fail_closed
    (c : Bool) (a b : String) (hne : a ≠ b) :
    reconcile c (some a) (some b) = Verdict.disputed ∧
    reconcile c (some b) (some a) = Verdict.disputed := by
  have h1 : (a == b) = false := beq_false_of_ne hne
  have h2 : (b == a) = false := beq_false_of_ne hne.symm
  cases c <;> simp [reconcile, h1, h2]

/-- World-compatibility semantics for receipt reconciliation:
    an unclaimed request can have no canonical result (`world = none`),
    and every present receipt must agree with the canonical historical result. -/
def AgreesWith (receipt world : Option String) : Prop :=
  ∀ d, receipt = some d → world = some d

def CompatibleWorld (claimed : Bool) (p r world : Option String) : Prop :=
  (claimed = false → world = none) ∧ AgreesWith p world ∧ AgreesWith r world

/-- `Verdict.disputed` occurs if and only if no compatible historical world exists. -/
theorem disputed_iff_no_compatible_world (c : Bool) (p r : Option String) :
    reconcile c p r = Verdict.disputed ↔ ¬ ∃ w, CompatibleWorld c p r w := by
  cases c with
  | false =>
    cases p with
    | none =>
      cases r with
      | none =>
        constructor
        · intro h; cases h
        · intro h; exfalso; apply h; refine ⟨none, ?_, ?_, ?_⟩
          · intro _; rfl
          · intro d hd; cases hd
          · intro d hd; cases hd
      | some b =>
        constructor
        · intro _ ⟨w, hc, _, hr⟩
          have hw0 : w = none := hc rfl
          have hwb : w = some b := hr b rfl
          rw [hw0] at hwb; cases hwb
        · intro _; rfl
    | some a =>
      cases r with
      | none =>
        constructor
        · intro _ ⟨w, hc, hp, _⟩
          have hw0 : w = none := hc rfl
          have hwa : w = some a := hp a rfl
          rw [hw0] at hwa; cases hwa
        · intro _; rfl
      | some b =>
        constructor
        · intro _ ⟨w, hc, hp, _⟩
          have hw0 : w = none := hc rfl
          have hwa : w = some a := hp a rfl
          rw [hw0] at hwa; cases hwa
        · intro _; rfl
  | true =>
    cases p with
    | none =>
      cases r with
      | none =>
        constructor
        · intro h; cases h
        · intro h; exfalso; apply h; refine ⟨none, ?_, ?_, ?_⟩
          · intro hc; cases hc
          · intro d hd; cases hd
          · intro d hd; cases hd
      | some b =>
        constructor
        · intro h; cases h
        · intro h; exfalso; apply h; refine ⟨some b, ?_, ?_, ?_⟩
          · intro hc; cases hc
          · intro d hd; cases hd
          · intro d hd; cases hd; rfl
    | some a =>
      cases r with
      | none =>
        constructor
        · intro h; cases h
        · intro h; exfalso; apply h; refine ⟨some a, ?_, ?_, ?_⟩
          · intro hc; cases hc
          · intro d hd; cases hd; rfl
          · intro d hd; cases hd
      | some b =>
        by_cases hab : a = b
        · subst hab
          constructor
          · intro h; simp [reconcile] at h
          · intro h; exfalso; apply h; refine ⟨some a, ?_, ?_, ?_⟩
            · intro hc; cases hc
            · intro d hd; cases hd; rfl
            · intro d hd; cases hd; rfl
        · have hbeq : (a == b) = false := beq_false_of_ne hab
          constructor
          · intro _ ⟨w, _, hp, hr⟩
            have hwa : w = some a := hp a rfl
            have hwb : w = some b := hr b rfl
            rw [hwa] at hwb
            exact hab (Option.some.inj hwb)
          · intro _; simp [reconcile, hbeq]

/-- A historical assertion is justified only when at least one compatible world
    exists and every compatible world satisfies the assertion. -/
def Justified (c : Bool) (p r : Option String) (claim : Option String → Prop) : Prop :=
  (∃ w, CompatibleWorld c p r w) ∧ ∀ w, CompatibleWorld c p r w → claim w

/-- Disputed evidence withholds every historical assertion. -/
theorem disputed_withholds_all_reports
    {c : Bool} {p r : Option String}
    (hbad : reconcile c p r = Verdict.disputed)
    (claim : Option String → Prop) :
    ¬ Justified c p r claim := by
  intro hj
  exact (disputed_iff_no_compatible_world c p r).mp hbad hj.1

/-! ## 6. Single-Use Authorization Ticket & Generation Expiry -/

structure AuthTicket where
  reqId : Nat
  generation : Nat
  digest : String
  used : Bool
  deriving DecidableEq, Repr

inductive TicketVerdict where
  | authorized
  | refused
  deriving DecidableEq, Repr

def consumeTicket (curGen : Nat) (reqId : Nat) (digest : String) (t : AuthTicket) :
    TicketVerdict × AuthTicket :=
  if !t.used && t.generation == curGen && t.reqId == reqId && t.digest == digest then
    (TicketVerdict.authorized, { t with used := true })
  else
    (TicketVerdict.refused, t)

/-- Replaying a ticket after a successful consumption always refuses. -/
theorem single_use_ticket_no_replay
    (curGen reqId : Nat) (digest : String) (t : AuthTicket)
    (hauth : (consumeTicket curGen reqId digest t).1 = TicketVerdict.authorized) :
    (consumeTicket curGen reqId digest (consumeTicket curGen reqId digest t).2).1 =
      TicketVerdict.refused := by
  unfold consumeTicket at hauth ⊢
  cases hu : t.used
  · simp [hu] at hauth ⊢
    split at hauth
    · rename_i hcond
      simp [hcond]
    · cases hauth
  · simp [hu] at hauth

/-- A generation mismatch refuses consumption. -/
theorem stale_generation_refused
    (curGen reqId : Nat) (digest : String) (t : AuthTicket)
    (hstale : t.generation ≠ curGen) :
    (consumeTicket curGen reqId digest t).1 = TicketVerdict.refused := by
  have hbeq : (t.generation == curGen) = false := beq_false_of_ne hstale
  unfold consumeTicket
  simp [hbeq]

#print axioms naive_race_publishes_one_quarter
#print axioms publishTrace_lookup
#print axioms published_traces_confluent
#print axioms scheduled_pair_observation
#print axioms draw_id_oracle_pathwise_confluent
#print axioms draw_id_shared_marginal_exact_half
#print axioms draw_id_fresh_pair_pathwise_confluent
#print axioms draw_id_fresh_pairs_uniform_quarter
#print axioms draw_id_speculative_budget_isolated
#print axioms speculative_ticks_cost
#print axioms receipt_reconcile_symmetric
#print axioms receipt_reconcile_dispute_symmetric_and_fail_closed
#print axioms disputed_iff_no_compatible_world
#print axioms disputed_withholds_all_reports
#print axioms single_use_ticket_no_replay
#print axioms stale_generation_refused

end EshkolStochasticBLCConfluence
