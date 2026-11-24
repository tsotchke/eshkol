import Init

namespace Q1

abbrev Tape (Gate : Type) := List Gate

structure Region (Gate Layout State : Type) where
  pending : Tape Gate
  layout : Layout
  log : Tape Gate
  amplitudes : State

def Region.commit (r : Region G L S) (t : Tape G) (out : S) : Region G L S :=
  { r with pending := [], log := r.log ++ t, amplitudes := out }

inductive Stage (G S : Type) where
  | idle
  | proposed (candidate : Tape G)
  | authorized (snapshot producerCopy : Tape G)
  | running (dispatched : Tape G)
  | returned (dispatched : Tape G) (output : S)
  | rejected (candidate : Tape G)
  | failed (dispatched : Tape G)
  | committed (dispatched : Tape G) (output : S)

inductive Event (G : Type) where
  | authorized (t : Tape G)
  | called (t : Tape G)
  | succeeded | failed | rejected (candidate : Tape G) | committed
  deriving DecidableEq

/-- Reachable prefixes of one flush. A backend error may carry arbitrarily
    changed private working data. Region observations exclude those data. -/
inductive Run (check : Tape G → Tape G → Bool) (backend : Tape G → S → Except S S)
    (initial : Region G L S) : Stage G S → Region G L S → List (Event G) → Prop where
  | start : Run check backend initial .idle initial []
  | prepare : Run check backend initial .idle r es →
      Run check backend initial (.proposed candidate) r es
  | reject : Run check backend initial (.proposed candidate) r es →
      check r.pending candidate = false →
      Run check backend initial (.rejected candidate) r (es ++ [.rejected candidate])
  | authorize : Run check backend initial (.proposed candidate) r es →
      check r.pending candidate = true →
      Run check backend initial (.authorized candidate candidate) r (es ++ [.authorized candidate])
  | mutate : Run check backend initial (.authorized t producer) r es →
      Run check backend initial (.authorized t changed) r es
  | call : Run check backend initial (.authorized t producer) r es →
      Run check backend initial (.running t) r (es ++ [.called t])
  | fail : Run check backend initial (.running t) r es →
      backend t r.amplitudes = .error privateState →
      Run check backend initial (.failed t) r (es ++ [.failed])
  | succeed : Run check backend initial (.running t) r es →
      backend t r.amplitudes = .ok out →
      Run check backend initial (.returned t out) r (es ++ [.succeeded])
  | publish : Run check backend initial (.returned t out) r es →
      Run check backend initial (.committed t out) (r.commit t out) (es ++ [.committed])

/-- An independent specification of the entire public region and audit trail
    at each stage. The checker is always bound to the original pending tape. -/
def History (check : Tape G → Tape G → Bool) (backend : Tape G → S → Except S S)
    (initial : Region G L S) (stage : Stage G S) (r : Region G L S) (es : List (Event G)) : Prop :=
  match stage with
  | .idle | .proposed _ => r = initial ∧ es = []
  | .rejected t => r = initial ∧ es = [.rejected t] ∧ check initial.pending t = false
  | .authorized t _ => r = initial ∧ es = [.authorized t] ∧ check initial.pending t = true
  | .running t => r = initial ∧ es = [.authorized t, .called t] ∧ check initial.pending t = true
  | .failed t => r = initial ∧ es = [.authorized t, .called t, .failed] ∧ check initial.pending t = true
  | .returned t out => r = initial ∧ es = [.authorized t, .called t, .succeeded] ∧
      check initial.pending t = true ∧ backend t initial.amplitudes = .ok out
  | .committed t out => r = initial.commit t out ∧ es = [.authorized t, .called t, .succeeded, .committed] ∧
      check initial.pending t = true ∧ backend t initial.amplitudes = .ok out

theorem history (h : Run check backend initial stage r es) : History check backend initial stage r es := by
  induction h <;> simp_all [History]

theorem rejection_preserves (h : Run check backend initial (.rejected t) r es) :
    r = initial ∧ es = [.rejected t] ∧ check initial.pending t = false := history h

theorem failure_preserves (h : Run check backend initial (.failed t) r es) :
    r = initial ∧ es = [.authorized t, .called t, .failed] := by
  exact ⟨(history h).1, (history h).2.1⟩

theorem dispatch_sound (meaning : Tape G → M)
    (checkerSound : ∀ old t, check old t = true → meaning t = meaning old)
    (h : Run check backend initial (.running t) r es) :
    es = [.authorized t, .called t] ∧ meaning t = meaning initial.pending := by
  exact ⟨(history h).2.1, checkerSound _ _ (history h).2.2⟩

theorem successful_return_unpublished (h : Run check backend initial (.returned t out) r es) :
    r = initial ∧ backend t initial.amplitudes = .ok out := by
  exact ⟨(history h).1, (history h).2.2.2⟩

theorem log_change_requires_success (h : Run check backend initial stage r es)
    (changed : r.log ≠ initial.log) :
    ∃ t out, stage = .committed t out ∧ r = initial.commit t out ∧
      es = [.authorized t, .called t, .succeeded, .committed] ∧
      check initial.pending t = true ∧ backend t initial.amplitudes = .ok out := by
  have inv := history h
  cases stage <;> simp_all [History]
  case committed t out => exact ⟨t, out, ⟨rfl, rfl⟩, rfl, rfl, inv.2.2⟩

end Q1
