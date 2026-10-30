set_option maxRecDepth 100000
set_option maxHeartbeats 4000000
namespace EshkolQEC
abbrev Bits := List Nat
def xorBits (a b : Bits) : Bits := List.zipWith (fun x y => (x+y)%2) a b
def dot (a b : Bits) : Nat := ((List.zipWith (· * ·) a b).foldl (· + ·) 0)%2
def inner (n : Nat) (a b : Bits) : Nat := (dot (a.take n) (b.drop n)+dot (a.drop n) (b.take n))%2
def n : Nat := 7
def checks : List Bits := [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]]
def recovery : List (Bits × Bits) := [([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ([0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ([0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), ([0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), ([0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ([0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), ([0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), ([0, 0, 0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ([0, 1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]), ([0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]), ([0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ([1, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), ([1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), ([0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ([1, 0, 1, 1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]), ([1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), ([0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), ([1, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]), ([1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), ([0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), ([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]), ([1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])]

def zero : Bits := List.replicate (2*n) 0
def syndrome (e : Bits) : Bits := checks.map (fun c => inner n c e)
def singles : List Bits := (List.range n).flatMap (fun q =>
  [[1,0], [1,1], [0,1]].map (fun xz =>
    (List.range n).map (fun i => if i == q then xz[0]! else 0) ++
    (List.range n).map (fun i => if i == q then xz[1]! else 0)))
def group : List Bits := checks.foldl (fun acc g => acc ++ acc.map (xorBits g)) [zero]
def corrects (e : Bits) : Bool :=
  match recovery.find? (fun row => row.1 == syndrome e) with
  | none => false
  | some row => group.any (fun g => g == xorBits row.2 e)
theorem detects_all_weight_one_errors : singles.all (fun e => (syndrome e).any (fun b => b == 1)) = true := by decide
theorem recovers_identity_and_all_weight_one_errors_mod_phase : (zero :: singles).all corrects = true := by decide
end EshkolQEC
