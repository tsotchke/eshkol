import Context
import Handoff

namespace Q1.PhaseHandoff
noncomputable section

-- Fixed control wire 0 and target wire 1. Each instruction acts diagonally
-- in the control basis. A phase on the control commutes with every controlled gate.
inductive Instruction where
  | controlled (gate : Q0.Gate)
  | onControl (gate : Q0.PhaseGate)

abbrev Circuit := Tape Instruction
def operator (t : Circuit) : List Q0.Amp :=
  let gates := t.filterMap fun | .controlled g => some g | _ => none
  let phases := t.filterMap fun | .onControl g => some g | _ => none
  Q0.controlledEntries .controlFirst (Q0.denote gates) phases

def check (old candidate : Circuit) : Bool := decide (operator candidate = operator old)
theorem checker_sound (old candidate : Circuit) (h : check old candidate = true) :
    operator candidate = operator old := of_decide_eq_true h

def original : Circuit := [.controlled .y, .controlled .x]
def corrected : Circuit := [.controlled .z, .onControl .s]
def wrong : Circuit := [.controlled .z, .onControl .z, .onControl .s] -- ZS = S†

-- The same directional proof XY = iZ holds for both proposals.
theorem wrong_correction_rejected (backend : Circuit → S → Except S S)
    (r : Region Instruction L S) (source : r.pending = original) :
    (∀ i j, Q0.denote Q0.yThenX i j = Q0.mul (Q0.phase .s) (Q0.denote [.z] i j)) ∧
    Run check backend r (.rejected wrong) r [.rejected wrong] := by
  refine ⟨Q0.xy_phase, Run.reject (candidate := wrong) (.prepare .start) ?_⟩
  rw [source]
  decide +kernel

theorem wrong_never_dispatched (backend : Circuit → S → Except S S)
    (r : Region Instruction L S) (source : r.pending = original) :
    ¬ Run check backend r (.running wrong) visible events := by
  intro dispatched
  have equality := (dispatch_sound operator checker_sound dispatched).2
  rw [source] at equality
  exact (by decide +kernel : operator wrong ≠ operator original) equality

-- The producer overwrites its own correction before dispatch. The trace still
-- calls the accepted tape. A successful backend return publishes that tape.
theorem correct_snapshot_commits (backend : Circuit → S → Except S S)
    (r : Region Instruction L S) (source : r.pending = original)
    (ok : backend corrected r.amplitudes = .ok out) :
    Run check backend r (.committed corrected out) (r.commit corrected out)
      [.authorized corrected, .called corrected, .succeeded, .committed] := by
  have accepted : check r.pending corrected = true := by rw [source]; decide +kernel
  exact .publish (.succeed (.call (producer := wrong) (.mutate (.authorize (.prepare .start) accepted))) ok)

end
end Q1.PhaseHandoff
