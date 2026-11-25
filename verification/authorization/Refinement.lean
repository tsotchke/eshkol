import Extracted

namespace Verificatum.Authority

private theorem scan_refines (needs : List N) (check : N → Verdict) :
    (needs.findSome? fun n => let v := check n
      if v ≠ .checked then some v else none).getD .authorized = firstFailure check needs := by
  induction needs with
  | nil => rfl
  | cons n ns ih =>
    cases h : check n <;> simp [List.findSome?, firstFailure, h]
    simpa using ih

theorem decision_refines (needs : List N) (check : N → Verdict) :
    pyDecide needs check = decisionSpec needs check := by
  unfold pyDecide decisionSpec
  split
  · rfl
  · exact scan_refines needs check

private theorem scan_accepts (ns : List N) (check : N → Verdict)
    (observer : ∀ n ∈ ns, check n ≠ .authorized) :
    firstFailure check ns = .authorized ↔ ∀ n ∈ ns, check n = .checked := by
  induction ns with
  | nil => simp [firstFailure]
  | cons n ns ih =>
    have tail := ih (fun x hx => observer x (List.mem_cons_of_mem _ hx))
    have head := observer n List.mem_cons_self
    cases h : check n <;> simp_all [firstFailure]

-- The observer boundary is explicit. A checked verdict corresponds exactly to
-- admitted evidence for the required scope, kind and level.
theorem decision_authorizes [DecidableEq C] [DecidableEq A] [DecidableEq S]
    (s : Scope C A S) (needs : C → List Need) (es : List (Checked C A S))
    (check : Need → Verdict) (observer : ∀ n ∈ needs s.contract, check n ≠ .authorized)
    (binding : ∀ n ∈ needs s.contract, check n = .checked ↔
      (es.any fun e => decide (e.scope = s ∧ e.need = n)) = true) :
    pyDecide (needs s.contract) check = .authorized ↔ authorize s needs es = true := by
  rw [decision_refines]
  unfold decisionSpec authorize
  cases h : (needs s.contract).isEmpty with
  | true => simp
  | false =>
    simp only [Bool.false_eq_true, ite_false, Bool.not_false, Bool.true_and]
    rw [scan_accepts _ _ observer, List.all_eq_true]
    exact ⟨fun H n hn => (binding n hn).mp (H n hn), fun H n hn => (binding n hn).mpr (H n hn)⟩

-- The binding bytes represent the consumer's contract, privilege, evidence
-- IDs and context. The second equality compares full artifact bytes.
theorem release_refines (issued current payload expected : List UInt8) :
    pyRelease issued current payload expected =
      consume (Scope.mk issued payload ()) (Scope.mk current expected ()) := by
  by_cases hb : issued = current <;> by_cases hp : payload = expected <;>
    simp [pyRelease, consume, hb, hp]

#print axioms decision_refines
#print axioms decision_authorizes
#print axioms release_refines
end Verificatum.Authority
