namespace Verificatum.LinkedPair

-- Addresses are eight-byte words in valid, aligned, non-wrapping storage.
-- Model the data-memory operands of the supported x86-64 SysV success paths.
abbrev Memory := Nat → UInt64
abbrev PairWords := UInt64 × UInt64
def loadPair (m : Memory) (a : Nat) : PairWords := (m a, m (a+1))
def copyPair (m : Memory) (a : Nat) (w : PairWords) : Memory :=
  fun j => if j = a then w.1 else if j = a+1 then w.2 else m j

inductive Op where
  | readInput (reg : Nat)
  | storeXmm (reg wordOffset : Nat)
  | readRdx (wordOffset : Nat)
  | readRax (wordOffset : Nat)
deriving DecidableEq, Repr

inductive Access where
  | read (addr words : Nat)
  | write (addr words : Nat)
deriving DecidableEq

structure Machine where
  mem : Memory
  xmm : Nat → PairWords
  rax : UInt64
  rdx : UInt64
  trace : List Access

def exec (pair input : Nat) (s : Machine) : List Op → Machine
  | [] => s
  | op :: ops =>
    let next := match op with
      | .readInput r =>
        {s with xmm := fun j => if j = r then loadPair s.mem input else s.xmm j,
                trace := s.trace ++ [.read input 2]}
      | .storeXmm r off =>
        {s with mem := copyPair s.mem (pair+off) (s.xmm r),
                trace := s.trace ++ [.write (pair+off) 2]}
      | .readRax off => {s with rax := s.mem (pair+off), trace := s.trace ++ [.read (pair+off) 1]}
      | .readRdx off => {s with rdx := s.mem (pair+off), trace := s.trace ++ [.read (pair+off) 1]}
    exec pair input next ops

def offset (cdr : Bool) : Nat := if cdr then 2 else 0

def Setter (ops : List Op) (cdr : Bool) := ∀ (s : Machine) p i,
  let a := p + offset cdr
  let after := exec p i s ops
  loadPair after.mem a = loadPair s.mem i ∧
  (∀ j, j ≠ a → j ≠ a+1 → after.mem j = s.mem j) ∧
  after.trace = s.trace ++ [.read i 2, .write a 2]

def Getter (ops : List Op) (cdr : Bool) := ∀ (s : Machine) p i,
  let a := p + offset cdr
  let after := exec p i s ops
  (after.rax,after.rdx) = loadPair s.mem a ∧ after.mem = s.mem ∧
  after.trace = s.trace ++ [.read (a+1) 1,.read a 1]

theorem setter_ok (cdr : Bool) (r : Nat) : Setter [.readInput r,.storeXmm r (offset cdr)] cdr := by
  intro s p i
  simp [exec,copyPair,loadPair]
  intro j ne ne2; simp [ne,ne2]

theorem getter_ok (cdr : Bool) : Getter [.readRdx (offset cdr+1),.readRax (offset cdr)] cdr := by
  intro s p i
  simp [exec,loadPair,Nat.add_assoc]

/-- info: 'Verificatum.LinkedPair.setter_ok' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms setter_ok
/-- info: 'Verificatum.LinkedPair.getter_ok' depends on axioms: [propext] -/
#guard_msgs in
#print axioms getter_ok

end Verificatum.LinkedPair
