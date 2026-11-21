import Publication

namespace Verificatum.Observation

open Verificatum.Heap Verificatum.Publication

-- Predict the store from the issued instruction and its input registers.
-- The tag store preserves the existing cdr, which is read from the pre-state.
def issued (c : Concrete α) : List (StoreEvent α) :=
  match c.ops with
  | .writeCdr :: _ => [.write c.data c.item]
  | .tagMemo :: _ =>
    match c.heap c.data with
    | some (_, payload) => [.publish c.data payload]
    | none => []
  | _ => []

inductive Execution : Concrete α → Concrete α → Type where
  | stop : Execution c c
  | step {c d e} : Primitive c d → Execution d e → Execution c e

variable {c d : Concrete α}

def effects : {s t : Concrete α} → Execution s t → List (StoreEvent α)
  | _, _, .stop => []
  | _, _, .step (c := s) _ rest => issued s ++ effects rest

def observe : {s t : Concrete α} → Execution s t → List (StoreEvent α)
  | _, _, .stop => []
  | _, _, .step (c := s) (d := t) _ rest => observations s t ++ observe rest

-- observations reads the addressed pair *after* execution. The primitive
-- rules determine what it contains, including when the store was idempotent.
theorem observation_adequacy (run : Execution c d) :
    observe run = effects run := by
  induction run with
  | stop => rfl
  | step prim _ ih =>
    cases prim with
    | cdr load => simp [observe,effects,observations,issued,put,ih]
    | car load => simp [observe,effects,observations,issued,put,load,ih]
    | tail => simpa [observe,effects,observations,issued] using ih

private theorem chain (a : Steps s t) (b : Steps t u) : Steps s u := by
  induction b with
  | refl => exact a
  | next _ one ih => exact .next ih one

theorem execution_refinement (run : Execution c d) (rel : Related b c a) :
    ∃ a', Steps a a' ∧ Related b d a' ∧ observe run = effects run := by
  have sim : ∃ a', Steps a a' ∧ Related b d a' := by
    induction run generalizing a with
    | stop => exact ⟨_,.refl,rel⟩
    | step prim _ ih =>
      obtain ⟨mid,path,rel'⟩ := forward rel prim
      obtain ⟨last,tail,final⟩ := ih rel'
      exact ⟨last,chain path tail,final⟩
  obtain ⟨last,path,final⟩ := sim
  exact ⟨last,path,final,observation_adequacy run⟩

-- The two channels are supplied independently. No producer must supply a
-- matching notification in order to construct the checker's input.
inductive Signal (α : Type) where
  | raw (event : α)
  | note (event : α)
  | quiet

def pairUp [DecidableEq α] : List (Signal α) → Option (List α)
  | [] => some []
  | .quiet :: rest => pairUp rest
  | .raw x :: .note y :: rest =>
    if x = y then (pairUp rest).map (x :: ·) else none
  | _ => none

inductive Accounted : List (Signal α) → List α → Prop where
  | nil : Accounted [] []
  | quiet : Accounted rest es → Accounted (.quiet :: rest) es
  | paired : Accounted rest es → Accounted (.raw e :: .note e :: rest) (e :: es)

theorem accepts_iff_accounted [DecidableEq α] (stream : List (Signal α)) (es : List α) :
    pairUp stream = some es ↔ Accounted stream es := by
  constructor
  · intro accept
    induction stream using pairUp.induct generalizing es with
    | case1 => simp [pairUp] at accept; subst es; exact .nil
    | case2 rest ih => exact .quiet (ih _ accept)
    | case3 x rest ih =>
      simp [pairUp] at accept
      obtain ⟨ys,h,rfl⟩ := accept
      exact .paired (ih _ h)
    | case4 x y rest ne => simp [pairUp,ne] at accept
    | case5 l noNil noQuiet noPair => rw [pairUp] at accept <;> grind
  · intro ok
    induction ok with
    | nil => rfl
    | quiet _ ih => exact ih
    | paired _ ih => simp [pairUp,ih]

-- Raw and notification payloads include operation, logical address and value.
def classify (line : String) : Signal String :=
  if line.startsWith "RAW-WRITE " || line.startsWith "RAW-PUBLISH " then
    .raw (line.drop 4).toString
  else if line.startsWith "WRITE " || line.startsWith "PUBLISH " then .note line
  else .quiet

inductive Boundary where
  | enter (line : String)
  | leave (line : String)
  | report (line : String)
deriving DecidableEq

structure Receipt where
  input : String
  outcome : String
  publicReport : String
deriving DecidableEq

def Receipt.markers (r : Receipt) : List Boundary :=
  [.enter r.input, .leave r.outcome, .report r.publicReport]

def readCalls : List Boundary → Option (List Receipt)
  | [] => some []
  | .enter x :: .leave y :: .report z :: rest =>
    (readCalls rest).map (⟨x,y,z⟩ :: ·)
  | _ => none

-- Each external call contributes an ordered triple, even a zero-store call.
theorem invocation_adequacy (bs : List Boundary) (rs : List Receipt) :
    readCalls bs = some rs ↔ bs = rs.flatMap Receipt.markers := by
  constructor
  · intro h
    induction bs using readCalls.induct generalizing rs with
    | case1 => simp [readCalls] at h; subst rs; rfl
    | case2 x y z rest ih =>
      simp [readCalls] at h
      obtain ⟨tail,ok,rfl⟩ := h
      simp [Receipt.markers, ih _ ok]
    | case3 => rw [readCalls] at h <;> grind
  · rintro rfl
    induction rs with
    | nil => rfl
    | cons r _ ih => simp [Receipt.markers,readCalls,ih]

def boundary? (line : String) : Option Boundary :=
  if line.startsWith "START " then some (.enter line)
  else if line.startsWith "STOP " then some (.leave line)
  else if line.startsWith "REPORT " then some (.report line)
  else none

-- The control build prints its public return, including all five counters.
-- Derive the expected START and STOP without consulting the observer.
def receipt? (line : String) : Option Receipt := do
  let ["REPORT", fuel, term, status, nf, first, _, _, _, _] := line.splitOn " " | none
  guard (first.startsWith "(")
  let some steps := (first.drop 1).toString.toNat? | none
  let stop ← match status with
    | "normal" => some s!"STOP normal {steps} {nf}"
    | "exhausted" => if nf == "#f" then some s!"STOP exhausted {steps}" else none
    | _ => none
  return ⟨s!"START {fuel} {term}",stop,line⟩

/-- info: 'Verificatum.Observation.observation_adequacy' depends on axioms: [propext] -/
#guard_msgs in
#print axioms observation_adequacy
/-- info: 'Verificatum.Observation.execution_refinement' depends on axioms: [propext] -/
#guard_msgs in
#print axioms execution_refinement
/-- info: 'Verificatum.Observation.accepts_iff_accounted' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms accepts_iff_accounted
/-- info: 'Verificatum.Observation.invocation_adequacy' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms invocation_adequacy


def audit (args : List String) : IO Unit := do
  match args with
  | ["stores", path] =>
    let lines := (← IO.FS.lines path).toList
    let some events := pairUp (lines.map classify) | throw (IO.userError "unaccounted store")
    IO.println s!"Checked {events.length} paired stores."
  | ["runtime", path] =>
    let lines := (← IO.FS.lines path).toList
    let signals := lines.map fun line =>
      if line == "RUNTIME CDR" then Signal.raw true
      else if line == "RUNTIME CAR" then .raw false
      else if line.startsWith "RAW-WRITE " then .note true
      else if line.startsWith "RAW-PUBLISH " then .note false
      else .quiet
    let some paired := pairUp signals | throw (IO.userError "unaccounted runtime setter")
    IO.println s!"Checked {paired.length} linked setter returns."
  | ["calls", path, control] =>
    let lines := (← IO.FS.lines path).toList
    let reports := (← IO.FS.lines control).toList.filter (·.startsWith "REPORT ")
    let some expected := reports.mapM receipt? | throw (IO.userError "invalid control report")
    unless !expected.isEmpty && readCalls (lines.filterMap boundary?) == some expected do
      throw (IO.userError "unaccounted invocation")
    IO.println s!"Checked {expected.length} ordered invocation receipts."
  | _ => throw (IO.userError "expected stores TRACE, runtime TRACE or calls TRACE CONTROL")

end Verificatum.Observation
