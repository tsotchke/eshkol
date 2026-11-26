/-
  MutationRunner.lean
  Specification-mutation testing for standalone Lean 4 files.

  Purpose: classify each declared mutation of a target file by what its proof
  obligations can detect.

  Interface:
    lean --run MutationRunner.lean <manifest.json>
    Environment: LEAN_BIN (default "lean") names the compiler used for mutants.
                 KEEP_MUTANTS=1 keeps the generated mutant and witness files.
    Manifest (JSON):
      { "target": "<path relative to the manifest>",
        "mutants": [ { "id": "...", "description": "...",
                       "old": "<text occurring exactly once>", "new": "...",
                       "witness": "<theorem proving ¬ type_of% @<refutes>, optional>",
                       "witness_name": "<its name, required with witness>",
                       "refutes": "<obligation in the target the witness contradicts;
                                   required with witness>",
                       "equivalent": "<reason, optional>",
                       "expect": "<outcome, optional>" } ] }
    Exit status: 0 if every mutant with "expect" received that outcome, 1 if any
    did not (printed as UNEXPECTED), 2 if the target does not compile clean.

  Outcomes per mutant:
    INVALID       an error lies outside every theorem: the mutant model is ill-formed
    REFUTED       the witness proves ¬ (statement of the named obligation, as
                  elaborated against the mutant), without error and without sorryAx
    PROOF_BROKEN  errors lie only inside theorems; the theorems are listed
    SURVIVED      the mutant compiles clean: no obligation detects the change
    EQUIVALENT    declared neutral and compiles clean (EQUIVALENT_CONTRADICTED otherwise)

  Hidden decision: the mapping from compiler diagnostics to outcomes, which
  depends on Lean's "file:line:col: error:" format and on top-level
  declarations starting in column 0.

  Invariant: the target file is never written; every mutant is compiled from
  a separate copy, which is removed after classification unless KEEP_MUTANTS=1.

  Target binding: a witness counts only if the runner's own line
    example : ¬ type_of% @<refutes> := <witness_name>
  elaborates in the mutant. The witness is thereby bound to the exact
  proposition of the named obligation, not merely to its name.

  Zero dependencies beyond the Lean core library.
-/
import Lean.Data.Json
open Lean

namespace MutationRunner

structure Mutant where
  id : String
  description : String
  old : String
  new : String
  witness : Option String
  witnessName : Option String
  refutes : Option String
  equivalent : Option String
  expect : Option String

inductive DeclKind where
  | obligation (name : String)
  | definition
  deriving Inhabited

def countOccurrences (haystack needle : String) : Nat :=
  (haystack.splitOn needle).length - 1

def optField (j : Json) (key : String) : Option String :=
  match j.getObjValAs? String key with
  | .ok v => some v
  | .error _ => none

def parseMutant (j : Json) : Except String Mutant := do
  let id ← j.getObjValAs? String "id"
  let old ← j.getObjValAs? String "old"
  let new ← j.getObjValAs? String "new"
  let m : Mutant := { id, old, new,
                      description := (optField j "description").getD "",
                      witness := optField j "witness",
                      witnessName := optField j "witness_name",
                      refutes := optField j "refutes",
                      equivalent := optField j "equivalent",
                      expect := optField j "expect" }
  if m.witness.isSome && (m.witnessName.isNone || m.refutes.isNone) then
    throw s!"mutant {id}: witness requires witness_name and refutes"
  return m

/-- Words that may precede a declaration keyword. -/
def modifierWords : List String :=
  ["private", "protected", "noncomputable", "partial", "unsafe", "nonrec"]

def obligationWords : List String := ["theorem", "lemma", "example"]

def definitionWords : List String :=
  ["def", "abbrev", "structure", "inductive", "instance", "class",
   "opaque", "axiom", "macro", "syntax", "notation", "infixl", "infixr"]

/-- Classify a column-0 source line as the start of a declaration, if it is one. -/
partial def declarationStart (line : String) : Option DeclKind :=
  if line.startsWith " " || line.startsWith "\t" then none
  else
    let words := (line.splitOn " ").filter (· ≠ "")
    let rec strip : List String → List String
      | w :: rest => if modifierWords.contains w || w.startsWith "@[" then strip rest else w :: rest
      | [] => []
    match strip words with
    | kw :: rest =>
      if obligationWords.contains kw then
        some (.obligation ((rest.headD "").takeWhile (fun c => c ≠ ':' && c ≠ '(' && c ≠ '{')))
      else if definitionWords.contains kw then some .definition
      else none
    | [] => none

/-- For each 1-based line, the declaration that encloses it (none before the first). -/
def enclosingDeclarations (source : String) : Array (Option DeclKind) := Id.run do
  let mut current : Option DeclKind := none
  let mut out : Array (Option DeclKind) := #[none]
  for line in source.splitOn "\n" do
    match declarationStart line with
    | some k => current := some k
    | none =>
      let t := line.trimLeft
      if !line.startsWith " " && (t.startsWith "end " || t.startsWith "namespace " ||
          t.startsWith "#" || t.startsWith "open " || t.startsWith "section" ||
          t.startsWith "variable") then
        current := none
    out := out.push current
  return out

/-- Line numbers of errors reported for `file` in compiler output. -/
def errorLines (file output : String) : List Nat :=
  (output.splitOn "\n").filterMap fun l =>
    if l.startsWith (file ++ ":") && (l.splitOn ": error").length > 1 then
      match (l.splitOn ":") with
      | _ :: n :: _ => n.toNat?
      | _ => none
    else none

def compile (leanBin dir file : String) : IO String := do
  let r ← IO.Process.output { cmd := leanBin, args := #[file], cwd := some dir }
  return r.stdout ++ r.stderr

/-- The axiom list Lean printed for `#print axioms <name>`, or none if no such message is
    present. Only that message is read: other `#print axioms` output in the file, such as
    the target's own report on an obligation whose proof the mutation broke, is ignored. -/
def printedAxioms (output name : String) : Option String :=
  let isNameStart (before : String) := before.endsWith "'" || before.endsWith "."
  let after (marker : String) : Option String :=
    let parts := output.splitOn marker
    ((parts.zip parts.tail).find? fun (before, _) => isNameStart before).map (·.2)
  match after (name ++ "' depends on axioms: [") with
  | some rest => some ((rest.splitOn "]").headD "")
  | none => (after (name ++ "' does not depend on any axioms")).map fun _ => ""

/-- Insert `block` before the last top-level `end ` line, or append it. -/
def insertBeforeFinalEnd (source block : String) : String × Nat :=
  let lines := (source.splitOn "\n").toArray
  let idx := Id.run do
    let mut found : Option Nat := none
    for i in [0:lines.size] do
      if lines[i]!.startsWith "end " then found := some i
    return found
  match idx with
  | some i =>
    let before := (lines.extract 0 i).toList
    let after := (lines.extract i lines.size).toList
    ("\n".intercalate (before ++ [block] ++ after), i + 1)
  | none => (source ++ "\n" ++ block, lines.size + 1)

structure Result where
  id : String
  outcome : String
  detail : String

def obligationNames (source : String) : List String :=
  (enclosingDeclarations source).toList.filterMap (fun d => match d with
    | some (.obligation name) => some name
    | _ => none) |>.eraseDups

def classify (leanBin dir targetBase original : String) (m : Mutant) : IO Result := do
  if let some r := m.refutes then
    if !(obligationNames original).contains r then
      return { id := m.id, outcome := "MANIFEST_ERROR",
               detail := s!"refutes names {r}, which is not a theorem of the target" }
  if countOccurrences original m.old != 1 then
    return { id := m.id, outcome := "MANIFEST_ERROR",
             detail := s!"old text occurs {countOccurrences original m.old} times; exactly 1 required" }
  let mutated := original.replace m.old m.new
  let mutantFile := s!"{targetBase}.mutant.{m.id}.lean"
  IO.FS.writeFile (dir ++ "/" ++ mutantFile) mutated
  let out ← compile leanBin dir mutantFile
  let decls := enclosingDeclarations mutated
  let errs := errorLines mutantFile out
  let kindAt (n : Nat) : Option DeclKind := decls.getD n none
  let outsideObligation := errs.any fun n => match kindAt n with
    | some (.obligation _) => false
    | _ => true
  let brokenObligations := (errs.filterMap fun n => match kindAt n with
    | some (.obligation name) => some name
    | _ => none).eraseDups
  if outsideObligation then
    return { id := m.id, outcome := "INVALID",
             detail := s!"error outside any theorem at line(s) {errs.eraseDups}" }
  -- Witness: elaborated against the mutant; valid only if it has no error of its own,
  -- is bound to the negation of the named obligation's statement, and does not depend
  -- on sorryAx (which a broken theorem would introduce).
  if let (some w, some wn) := (m.witness, m.witnessName) then
    let block := w ++ "\nexample : ¬ type_of% @" ++ m.refutes.getD "" ++ " := " ++ wn ++
      "\n#print axioms " ++ wn
    let (withWitness, startLine) := insertBeforeFinalEnd mutated block
    let blockLines := (block.splitOn "\n").length
    let witnessFile := s!"{targetBase}.witness.{m.id}.lean"
    IO.FS.writeFile (dir ++ "/" ++ witnessFile) withWitness
    let wout ← compile leanBin dir witnessFile
    let werrs := (errorLines witnessFile wout).filter fun n => n ≥ startLine && n < startLine + blockLines
    -- A witness whose axiom report is missing cannot be shown free of sorryAx.
    let sorryDependent := match printedAxioms wout wn with
      | some axioms => (axioms.splitOn "sorryAx").length > 1
      | none => true
    if werrs.isEmpty && !sorryDependent then
      return { id := m.id, outcome := "REFUTED",
               detail := s!"witness {wn} holds on the mutant; contradicts {m.refutes.getD "?"}" ++
                 (if brokenObligations.isEmpty then "" else s!"; broken proofs: {brokenObligations}") }
  if let some reason := m.equivalent then
    if errs.isEmpty then
      return { id := m.id, outcome := "EQUIVALENT", detail := reason }
    else
      return { id := m.id, outcome := "EQUIVALENT_CONTRADICTED",
               detail := s!"declared equivalent ({reason}) but proofs broke: {brokenObligations}" }
  if errs.isEmpty then
    return { id := m.id, outcome := "SURVIVED", detail := "no obligation detects this mutation" }
  return { id := m.id, outcome := "PROOF_BROKEN", detail := s!"broken proofs: {brokenObligations}" }

def dirOf (path : String) : String :=
  match path.splitOn "/" with
  | [] | [_] => "."
  | parts => "/".intercalate parts.dropLast

def baseOf (path : String) : String :=
  (path.splitOn "/").getLast!

def removeIfPresent (path : String) : IO Unit := do
  if ← System.FilePath.pathExists path then IO.FS.removeFile path

def run (manifestPath : String) : IO UInt32 := do
  let leanBin := (← IO.getEnv "LEAN_BIN").getD "lean"
  let manifest ← match Json.parse (← IO.FS.readFile manifestPath) with
    | .ok j => pure j
    | .error e => throw (IO.userError s!"manifest is not valid JSON: {e}")
  let target ← IO.ofExcept (manifest.getObjValAs? String "target")
  let mutantsJson ← IO.ofExcept (manifest.getObjValAs? (Array Json) "mutants")
  let mutants ← IO.ofExcept (mutantsJson.mapM parseMutant)
  let targetPath := dirOf manifestPath ++ "/" ++ target
  let dir := dirOf targetPath
  let file := baseOf targetPath
  let original ← IO.FS.readFile targetPath
  let baseline ← compile leanBin dir file
  if !(errorLines file baseline).isEmpty || (baseline.splitOn "declaration uses 'sorry'").length > 1 then
    IO.eprintln s!"baseline {file} does not compile clean; aborting\n{baseline}"
    return 2
  let stem := if file.endsWith ".lean" then file.dropRight 5 else file
  IO.println s!"target {file}: baseline clean, {mutants.size} mutants"
  let mut tally : List (String × Nat) := []
  let mut unexpected : Nat := 0
  for m in mutants do
    let r ← classify leanBin dir stem original m
    if (← IO.getEnv "KEEP_MUTANTS") != some "1" then
      removeIfPresent s!"{dir}/{stem}.mutant.{m.id}.lean"
      removeIfPresent s!"{dir}/{stem}.witness.{m.id}.lean"
    IO.println s!"{r.outcome}\t{r.id}\t{r.detail}"
    if let some e := m.expect then
      if e != r.outcome then
        IO.println s!"UNEXPECTED\t{r.id}\texpected {e}, got {r.outcome}"
        unexpected := unexpected + 1
    tally := match tally.lookup r.outcome with
      | some n => (r.outcome, n + 1) :: tally.filter (·.1 ≠ r.outcome)
      | none => (r.outcome, 1) :: tally
  IO.println s!"SUMMARY {tally.reverse}"
  if unexpected > 0 then
    IO.println s!"UNEXPECTED outcomes: {unexpected}"
    return 1
  return 0

end MutationRunner

def main (args : List String) : IO UInt32 :=
  match args with
  | [manifest] => MutationRunner.run manifest
  | _ => do
    IO.eprintln "usage: lean --run MutationRunner.lean <manifest.json>"
    return 1
