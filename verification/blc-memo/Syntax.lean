import Correspondence

namespace Verificatum.Syntax

open Verificatum.Lazy Verificatum.Heap Verificatum.Readback

-- The integer field follows core.blc, including signed intermediate arithmetic.
-- The image of present consists of the well-formed one-based constructor trees.
inductive One where
  | var (index : Int)
  | lam (body : One)
  | app (fn arg : One)
deriving DecidableEq, Repr

def present : Term → One
  | .var i => .var (Int.ofNat (i+1))
  | .lam b => .lam (present b)
  | .app f a => .app (present f) (present a)

def readTerm : One → Option Term
  | .var i => if i > 0 then some (.var (i.toNat-1)) else none
  | .lam b => (readTerm b).map Term.lam
  | .app f a => return .app (← readTerm f) (← readTerm a)

theorem read_present (t : Term) : readTerm (present t) = some t := by
  induction t <;> simp_all [readTerm,present]

theorem present_read (t : One) (ok : readTerm t = some n) : present n = t := by
  induction t generalizing n with
  | var i =>
    simp only [readTerm] at ok
    split at ok
    · cases ok; simp [present]; omega
    · contradiction
  | lam b ih =>
    cases hb : readTerm b <;> simp_all [readTerm]
    subst n; simp_all [present]
  | app f a hf ha =>
    cases ef : readTerm f <;> cases ea : readTerm a <;> simp_all [readTerm]
    subst n; simp_all [present]

-- Finite pair trees, including improper lists and malformed constructor shapes.
inductive Datum where
  | nil
  | symbol (name : String)
  | integer (value : Int)
  | pair (car cdr : Datum)
deriving DecidableEq

def pack : One → Datum
  | .var i => .pair (.symbol "var") (.pair (.integer i) .nil)
  | .lam b => .pair (.symbol "lam") (.pair (pack b) .nil)
  | .app f a => .pair (.symbol "app") (.pair (pack f) (.pair (pack a) .nil))

def unpack : Datum → Option One
  | .pair (.symbol "var") (.pair (.integer i) .nil) => some (.var i)
  | .pair (.symbol "lam") (.pair b .nil) => (unpack b).map One.lam
  | .pair (.symbol "app") (.pair f (.pair a .nil)) => return .app (← unpack f) (← unpack a)
  | _ => none

theorem unpack_pack (t : One) : unpack (pack t) = some t := by
  induction t <;> simp_all [pack,unpack]

theorem pack_unpack (d : Datum) (ok : unpack d = some t) : pack t = d := by
  fun_induction unpack generalizing t with
  | case1 => cases ok; rfl
  | case2 b ih =>
    cases e : unpack b <;> simp_all
    subst t; simp_all [pack]
  | case3 f a hf ha =>
    cases ef : unpack f <;> cases ea : unpack a <;> simp_all
    subst t; simp_all [pack]
  | case4 => cases ok

def shift (delta : Int) (cut : Nat) : One → One
  | .var k => .var (if k ≥ cut then k + delta else k)
  | .lam b => .lam (shift delta (cut+1) b)
  | .app f a => .app (shift delta cut f) (shift delta cut a)

def replace (j : Nat) (s : One) : One → One
  | .var i => if i = j then s else .var i
  | .lam b => .lam (replace (j+1) (shift 1 1 s) b)
  | .app f a => .app (replace j s f) (replace j s a)

def contract (b a : One) := shift (-1) 1 (replace 1 (shift 1 1 a) b)

-- Allow arbitrary renamings before and after a shift. The premise says the
-- arithmetic agrees at each variable. Under a binder index one stays bound.
private theorem shift_ren (t : Term) (cut : Nat) (delta : Int) (r g : Ren)
    (h : ∀ i, (if (r i : Int)+1 ≥ (cut : Int)+1 then (r i : Int)+1+delta else (r i : Int)+1)
      = (g i : Int)+1) :
    shift delta (cut+1) (present (ren r t)) = present (ren g t) := by
  induction t generalizing cut r g with
  | var i => simpa [shift,present,ren] using h i
  | app f a hf ha => simp [ren,present,shift,hf cut r g h,ha cut r g h]
  | lam b ih =>
    simp only [ren,present,shift,One.lam.injEq]
    apply ih
    intro i
    cases i with
    | zero => simp [upRen,push]; omega
    | succ i =>
      have hi := h i
      simp only [upRen,push,Int.natCast_succ]
      split at hi <;> grind

private def lifted (d : Nat) (a : Term) := ren (fun i => i+d) a

private theorem lift_zero (a : Term) : lifted 0 a = a := by
  have go (t : Term) (r : Ren) (h : ∀ i, r i = i) : ren r t = t := by
    induction t generalizing r with
    | var => simp [ren,h]
    | app f a hf ha => simp [ren,hf r h,ha r h]
    | lam b ih => exact congrArg Term.lam (ih _ (by intro i; cases i <;> simp [upRen,push,h]))
  exact go a _ (fun _ => rfl)

private theorem lift_one (a : Term) (d : Nat) :
    shift 1 1 (present (lifted d a)) = present (lifted (d+1) a) := by
  apply shift_ren a 0
  intro i; simp; omega

private theorem lower_lift (a : Term) (d : Nat) :
    shift (-1) (d+1) (present (lifted (d+1) a)) = present (lifted d a) := by
  apply shift_ren
  intro i; simp; omega

private def slot (a : Term) (d i : Nat) : Term :=
  if i = d then lifted d a else .var (if d < i then i-1 else i)

private theorem slot_binder (a : Term) (d : Nat) : upSub (slot a d) = slot a (d+1) := by
  funext i
  cases i with
  | zero => simp [slot,upSub,push]
  | succ i =>
    by_cases eq : i = d
    · subst i; simp [upSub,push,slot,lifted,renRen,Nat.add_assoc]
    · by_cases lt : d < i <;> simp [upSub,push,slot,eq,lt,ren] <;> omega

private theorem contract_at (b a : Term) (d : Nat) :
    shift (-1) (d+1) (replace (d+1) (present (lifted (d+1) a)) (present b)) =
    present (sub (slot a d) b) := by
  induction b generalizing d with
  | var i =>
    by_cases eq : i = d
    · subst i; simpa [present,replace,sub,slot] using lower_lift a d
    · have ne : (i : Int) ≠ (d : Int) := by omega
      by_cases lt : d < i <;> simp [present,replace,sub,slot,eq,lt,ne,shift] <;>
        grind
  | lam b ih => simp [present,replace,shift,sub,slot_binder,lift_one,ih]
  | app f x hf hx => simp [present,replace,shift,sub,hf,hx]

theorem contract_present (b a : Term) :
    contract (present b) (present a) = present (beta b a) := by
  have s : slot a 0 = push a Term.var := by
    funext i; cases i <;> simp [slot,push,lift_zero]
  simpa [contract,beta,s,← lift_one, lift_zero] using contract_at b a 0

-- core.blc returns a reduction flag and a tree even when the flag is false.
def oneStep : One → Bool × One
  | t@(.var _) => (false,t)
  | t@(.lam b) => let (changed,u) := oneStep b
                   if changed then (true,.lam u) else (false,t)
  | t@(.app f a) => match f with
    | .lam b => (true,contract b a)
    | _ => let (cf,g) := oneStep f
           if cf then (true,.app g a) else
             let (ca,b) := oneStep a
             if ca then (true,.app f b) else (false,t)

private def report (t : Term) : Option Term → Bool × One
  | none => (false,present t)
  | some u => (true,present u)

theorem step_present (t : Term) : oneStep (present t) = report t (step t) := by
  induction t with
  | var => rfl
  | lam b ih => cases e : step b <;> simp [present,oneStep,ih,e,step,report]
  | app f a hf ha =>
    cases f with
    | lam b => simp [present,oneStep,step,report,contract_present]
    | var i => cases e : step a <;> simp [present,oneStep,step,report,ha,e]
    | app g x =>
      change (if (oneStep (present (.app g x))).1 then
          (true, One.app (oneStep (present (.app g x))).2 (present a)) else
          if (oneStep (present a)).1 then
            (true, One.app (present (.app g x)) (oneStep (present a)).2) else
            (false,present (.app (.app g x) a))) = _
      rw [hf,ha]
      conv => rhs; unfold step
      cases ef : step (.app g x) <;> cases ea : step a <;> rfl

-- Count permitted step inspections. core.blc checks steps > cap before its
-- next inspection, so an initial "cap" of c permits c+1 inspections.
def oneReference : Nat → One → Option One
  | 0,_ => none
  | k+1,t => let (changed,u) := oneStep t
             if changed then oneReference k u else some t

theorem reference_present (k : Nat) (t : Term) :
    oneReference k (present t) = (reference k t).map present := by
  induction k generalizing t with
  | zero => rfl
  | succ k ih => cases h : step t <;> simp [oneReference,step_present,h,reference,report,ih]

theorem reference_register (input output : One) (ht : readTerm input = some t) :
    (∃ k, oneReference k input = some output) ↔
    (∃ fuel used n, drive (encode (start t)) fuel 0 = (.normal n,used) ∧ present n = output) := by
  rw [← present_read input ht]
  constructor
  · rintro ⟨k,hr⟩
    rw [reference_present] at hr
    cases h : reference k t with
    | none => simp [h] at hr
    | some n =>
      obtain ⟨fuel,used,ok⟩ := Correspondence.register_complete h
      exact ⟨fuel,used,n,ok,by simpa [h] using hr⟩
  · rintro ⟨_,_,n,ok,rfl⟩
    obtain ⟨k,h⟩ := Correspondence.register_sound ok
    exact ⟨k,by simp [reference_present,h]⟩

-- The wire carries one more '1' than the internal zero-based index. Prefix
-- decoding is used by core.blc; recorded heap events require exact decoding.
def wire : Term → List Char
  | .var i => '1' :: List.replicate i '1' ++ ['0']
  | .lam b => '0' :: '0' :: wire b
  | .app f a => '0' :: '1' :: wire f ++ wire a

theorem parse_prefix (t : Term) (tail : List Char) (fuel : Nat) (enough : (wire t).length ≤ fuel) :
    parse fuel (wire t ++ tail) = some (t,tail) := by
  induction t generalizing tail fuel with
  | var i =>
    cases fuel with
    | zero => simp [wire] at enough
    | succ k => simp [wire,parse,List.append_assoc]
  | lam b ih =>
    cases fuel with
    | zero => simp [wire] at enough
    | succ k =>
      simp [wire] at enough
      simp [wire,parse,ih tail k (by omega)]
  | app f a hf ha =>
    cases fuel with
    | zero => simp [wire] at enough
    | succ k =>
      simp [wire] at enough
      simp [wire,parse,List.append_assoc,hf (wire a ++ tail) k (by omega),ha tail k (by omega)]

theorem decode_wire (t : Term) : decode (String.ofList (wire t)) = some t := by
  have ok := parse_prefix t [] ((wire t).length+1) (by omega)
  simp only [List.append_nil] at ok
  simp [decode,ok]

private theorem ones_decompose (xs : List Char) :
    let k := (xs.takeWhile (· == '1')).length
    List.replicate k '1' ++ xs.drop k = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    by_cases eq : x = '1'
    · subst x; simpa [List.takeWhile,List.replicate_succ] using ih
    · simp [List.takeWhile, show (x == '1') = false by simpa using eq]

theorem parse_consumed (ok : parse k input = some (t,tail)) : input = wire t ++ tail := by
  fun_induction parse generalizing t tail with
  | case1 k rest ih =>
    cases e : parse k rest with
    | none => simp [e] at ok
    | some prod =>
      obtain ⟨b,rs⟩ := prod
      simp [e] at ok
      obtain ⟨rfl,rfl⟩ := ok
      simp [wire,ih e]
  | case2 k rest fn arg =>
    cases ef : parse k rest with
    | none => simp [ef] at ok
    | some prod =>
      obtain ⟨f,mid⟩ := prod
      cases ea : parse k mid with
      | none => simp [ef,ea] at ok
      | some prod =>
        obtain ⟨a,rs⟩ := prod
        simp [ef,ea] at ok
        obtain ⟨rfl,rfl⟩ := ok
        simp [wire,fn ef,arg mid ea,List.append_assoc]
  | case3 _ rest ones rs e =>
    cases ok
    have h := ones_decompose rest
    change List.replicate ones.length '1' ++ rest.drop ones.length = rest at h
    rw [e] at h
    simpa [wire,List.append_assoc] using congrArg (List.cons '1') h.symm
  | case4 | case5 => cases ok

theorem decode_exact (s : String) (t : Term) : decode s = some t ↔ s.toList = wire t := by
  constructor
  · intro ok
    unfold decode at ok
    cases e : parse (s.toList.length+1) s.toList with
    | none => simp [e] at ok
    | some prod =>
      obtain ⟨body,rs⟩ := prod
      cases rs <;> simp_all
      simpa using parse_consumed e
  · intro h
    have eq : s = String.ofList (wire t) := by simp [← h]
    exact eq ▸ decode_wire t

/-- info: 'Verificatum.Syntax.reference_register' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs (whitespace := lax) in
#print axioms reference_register

-- Generate independent expected constructor trees and flags for the native
-- suite, including inputs that exhaust their normalization budget.
private def datumText : One → String
  | .var i => s!"(var {i})"
  | .lam b => s!"(lam {datumText b})"
  | .app f a => s!"(app {datumText f} {datumText a})"

def auditRows (file : System.FilePath) : IO Unit := do
  for s in ← IO.FS.lines file do
    let some t := decode s | throw (IO.userError "malformed syntax replay input")
    unless String.ofList (wire t) == s do throw (IO.userError "wire round trip")
    let (changed,out) := oneStep (present t)
    let flag := if changed then "#t" else "#f"
    IO.println s!"({repr s} {datumText (present t)} {flag} {datumText out})"

end Verificatum.Syntax
