import Init

namespace Verificatum.Guard

/-- The same positive dimension range as the R1-A executable corpus. -/
inductive Width where | one | two | three | four
deriving DecidableEq

def Width.nat : Width → Nat
  | .one => 1 | .two => 2 | .three => 3 | .four => 4

def select (a b c d : Prop) : Width → Prop
  | .one => a | .two => b | .three => c | .four => d

def Goal (k j : Width) : Prop :=
  ∀ a b c d : Prop, select a b c d k → select a b c d j

/-- Distinct dimensions can be separated by choosing a singleton predicate. -/
theorem goal_iff (k j : Width) : Goal k j ↔ k = j := by
  constructor
  · intro h
    let p := fun (i : Width) => k = i
    have point (i : Width) : select (p .one) (p .two) (p .three) (p .four) i = p i := by
      cases i <;> rfl
    have transport := h (p .one) (p .two) (p .three) (p .four)
    rw [point k, point j] at transport
    exact transport rfl
  · rintro rfl a b c d h
    exact h

-- These closed target exports elaborate directly into R0's Pi/Prop fragment.
def equalThree : Prop := ∀ _a _b c _d : Prop, c → c
def threeFour : Prop := ∀ _a _b c d : Prop, c → d
example : equalThree = Goal .three .three := rfl
example : threeFour = Goal .three .four := rfl

structure Operand where
  valueId : Nat
  shapeVersion : Nat
  rows : Width
  cols : Width
deriving DecidableEq

/-- A site within one compilation. Version identities must be supplied by
    the elaborator and shared by every alias of mutable shape storage. -/
structure Site where
  compilation : Nat
  node : Nat
  environmentVersion : Nat
  left : Operand
  right : Operand
deriving DecidableEq

/-- The sole action modeled here is omitting the inner-equality comparison. -/
structure InnerEqualityPermit (issued : Site) : Prop where
  checked : Goal issued.left.cols issued.right.rows

/-- The context on each side comes from the consumer, independent of a proof
    producer. This logical predicate specifies successful use of the permit. -/
def Applicable (issued current : Site) : Prop :=
  issued = current ∧ InnerEqualityPermit issued

/-- The runtime link is a precondition for lowering. Even a valid proof with
    matching site identities is insufficient if the descriptors can disagree. -/
theorem may_omit_inner_eq {issued current : Site} {leftK rightJ : Nat}
    (use : Applicable issued current)
    (leftLink : leftK = current.left.cols.nat)
    (rightLink : rightJ = current.right.rows.nat) : leftK = rightJ := by
  rcases use with ⟨rfl, permit⟩
  have same := (goal_iff _ _).mp permit.checked
  exact leftLink.trans ((congrArg Width.nat same).trans rightLink.symm)

/-- The other guard can still fail. This transformation preserves its result
    for either value of the residual predicate. -/
theorem residual_guard_preserved {issued current : Site} {leftK rightJ : Nat}
    (use : Applicable issued current)
    (leftLink : leftK = current.left.cols.nat) (rightLink : rightJ = current.right.rows.nat)
    (remaining : Bool) : (decide (leftK = rightJ) && remaining) = remaining := by
  have same := may_omit_inner_eq use leftLink rightLink
  simp [same]

theorem stale_refused {issued current : Site} (changed : issued ≠ current) :
    ¬ Applicable issued current := fun use => changed use.1

theorem incompatible_refused {issued current : Site}
    (mismatch : current.left.cols ≠ current.right.rows) : ¬ Applicable issued current := by
  rintro ⟨rfl, permit⟩
  exact mismatch ((goal_iff _ _).mp permit.checked)

/-- State of the storage location holding an operand. Every tracked alias
    resolves to the same location. Versions here are unbounded naturals. -/
structure Cell where
  rows : Width
  cols : Width
  version : Nat

abbrev Heap := Nat → Cell

inductive Event where
  | write (location : Nat) (rows cols : Width)
  /-- A call with unknown writes invalidates every tracked location. The
      post-call shapes may be arbitrary, including the previous shapes. -/
  | opaqueCall (shapes : Nat → Width × Width)

def advance (h : Heap) (e : Event) (loc : Nat) : Cell :=
  match e with
  | .write dest r c => if loc = dest then ⟨r, c, (h loc).version + 1⟩ else h loc
  | .opaqueCall after => ⟨(after loc).1, (after loc).2, (h loc).version + 1⟩

def execute (h : Heap) : List Event → Heap
  | [] => h
  | e :: rest => execute (advance h e) rest

private theorem one_step (h : Heap) (e : Event) (loc : Nat) :
    advance h e loc = h loc ∨ (h loc).version < (advance h e loc).version := by
  cases e with
  | write dest r c =>
    by_cases same : loc = dest
    · right; simp [advance, same]
    · left; simp [advance, same]
  | opaqueCall => right; simp [advance]

private theorem history (h : Heap) (events : List Event) (loc : Nat) :
    execute h events loc = h loc ∨ (h loc).version < (execute h events loc).version := by
  induction events generalizing h with
  | nil => exact .inl rfl
  | cons e rest ih =>
    change execute (advance h e) rest loc = h loc ∨ _
    rcases one_step h e loc with unchanged | bumped
    · simpa only [execute, unchanged] using ih (advance h e)
    · right
      rcases ih (advance h e) with kept | increased
      · simpa only [execute, kept] using bumped
      · exact Nat.lt_trans bumped increased

/-- A successful late comparison can check the cheaper stamp. This theorem
    derives descriptor stability from the modeled write discipline. -/
theorem stable_at_use {start : Heap} {events : List Event} {loc : Nat}
    (stamp : (execute start events loc).version = (start loc).version) :
    execute start events loc = start loc := by
  rcases history start events loc with same | bumped
  · exact same
  · rw [stamp] at bumped
    exact False.elim (Nat.lt_irrefl _ bumped)

/-- Both root identity and version are revalidated. An unresolved root has
    value none and cannot match an issued stamp. -/
def StampValid (resolved : Option Nat) (root version : Nat) (now : Heap) : Prop :=
  resolved = some root ∧ (now root).version = version

theorem guard_after_writes {issued current : Site} {start : Heap} {events : List Event}
    {leftRoot rightRoot : Nat} {resolvedLeft resolvedRight : Option Nat}
    (use : Applicable issued current)
    (leftAtIssue : (start leftRoot).cols = issued.left.cols)
    (rightAtIssue : (start rightRoot).rows = issued.right.rows)
    (leftVersion : (start leftRoot).version = issued.left.shapeVersion)
    (rightVersion : (start rightRoot).version = issued.right.shapeVersion)
    (leftLive : StampValid resolvedLeft leftRoot issued.left.shapeVersion (execute start events))
    (rightLive : StampValid resolvedRight rightRoot issued.right.shapeVersion (execute start events))
    (remaining : Bool) :
    ∃ l r, resolvedLeft = some l ∧ resolvedRight = some r ∧
      (decide ((execute start events l).cols.nat = (execute start events r).rows.nat)
        && remaining) = remaining := by
  refine ⟨leftRoot, rightRoot, leftLive.1, rightLive.1, ?_⟩
  rw [stable_at_use (leftLive.2.trans leftVersion.symm),
      stable_at_use (rightLive.2.trans rightVersion.symm)]
  rcases use with ⟨rfl, permit⟩
  exact residual_guard_preserved ⟨rfl, permit⟩
    (congrArg Width.nat leftAtIssue) (congrArg Width.nat rightAtIssue) remaining

example (root version : Nat) (h : Heap) : ¬ StampValid none root version h :=
  fun ok => by cases ok.1

/-- The storage cell is distinct from the tensor descriptor it references.
    Two storage cells can refer to the same descriptor. -/
structure Binding where
  cellId : Nat
  cellVersion : Nat
  descriptorId : Nat
deriving DecidableEq

structure DescriptorStamp where
  binding : Binding
  descriptorVersion : Nat
  effectEpoch : Nat

/-- Resolution and effect epochs are consumer observations at the native site.
    An unknown call invalidates all earlier epochs. -/
def LiveDescriptor (stamp : DescriptorStamp) (resolved : Option Binding)
    (descriptors : Heap) (epoch : Nat) : Prop :=
  resolved = some stamp.binding ∧ epoch = stamp.effectEpoch ∧
  (descriptors stamp.binding.descriptorId).version = stamp.descriptorVersion

/-- At issue time both the operand ID and its declared axes must agree with
    the consumer's original binding and descriptor. -/
def Captured (stamp : DescriptorStamp) (operand : Operand) (resolve : Nat → Option Binding)
    (start : Heap) (epoch : Nat) : Prop :=
  let obj := start stamp.binding.descriptorId
  LiveDescriptor stamp (resolve operand.valueId) start epoch ∧
  obj.version = operand.shapeVersion ∧ obj.rows = operand.rows ∧ obj.cols = operand.cols

theorem descriptors_at_use {issued current : Site} {start : Heap} {events : List Event}
    {left right : DescriptorStamp} {resolveAtIssue resolveAtUse : Nat → Option Binding}
    {issueEpoch useEpoch : Nat}
    (use : Applicable issued current)
    (leftCaptured : Captured left issued.left resolveAtIssue start issueEpoch)
    (rightCaptured : Captured right issued.right resolveAtIssue start issueEpoch)
    (leftLive : LiveDescriptor left (resolveAtUse current.left.valueId) (execute start events) useEpoch)
    (rightLive : LiveDescriptor right (resolveAtUse current.right.valueId) (execute start events) useEpoch)
    (remaining : Bool) :
    ∃ bl br, resolveAtUse current.left.valueId = some bl ∧ resolveAtUse current.right.valueId = some br ∧
      (decide ((execute start events bl.descriptorId).cols.nat =
               (execute start events br.descriptorId).rows.nat) && remaining) = remaining := by
  refine ⟨left.binding, right.binding, leftLive.1, rightLive.1, ?_⟩
  rw [stable_at_use (leftLive.2.2.trans leftCaptured.1.2.2.symm),
      stable_at_use (rightLive.2.2.trans rightCaptured.1.2.2.symm)]
  rcases use with ⟨rfl, permit⟩
  exact residual_guard_preserved ⟨rfl, permit⟩
    (congrArg Width.nat leftCaptured.2.2.2) (congrArg Width.nat rightCaptured.2.2.1) remaining

private def firstRef : Binding := ⟨11, 0, 71⟩
private def otherRef : Binding := ⟨12, 0, 71⟩
private def descriptorStamp : DescriptorStamp := ⟨firstRef, 0, 4⟩
private def objects : Heap := fun _ => ⟨.three, .two, 0⟩
private def afterAlias : Heap := execute objects [.write otherRef.descriptorId .two .three]

-- Both cell stamps and the descriptor identity survive an in-place reshape.
example : LiveDescriptor descriptorStamp (some firstRef) objects 4 := by
  unfold LiveDescriptor; decide
example : (afterAlias firstRef.descriptorId).rows = .two := rfl
example : ¬ LiveDescriptor descriptorStamp (some firstRef) afterAlias 4 := by
  unfold LiveDescriptor; decide
example : ¬ LiveDescriptor descriptorStamp (some {firstRef with cellVersion := 1}) objects 4 := by
  unfold LiveDescriptor; decide
example : ¬ LiveDescriptor descriptorStamp (some firstRef) objects 5 := by
  unfold LiveDescriptor; decide

private def initial : Heap := fun _ => ⟨.three, .two, 0⟩
private def rhsRoot := 11

-- Reading through rhs and writing through its alias select the same root.
-- Even restoring the shape cannot restore an old stamp.
example : StampValid (some rhsRoot) rhsRoot 0 initial := ⟨rfl, rfl⟩
private theorem aliased_write_expires : ¬ StampValid (some rhsRoot) rhsRoot 0
    (execute initial [.write rhsRoot .two .three]) := by
  unfold StampValid
  decide
example : ¬ StampValid (some rhsRoot) rhsRoot 0
    (execute initial [.write rhsRoot .two .three, .write rhsRoot .three .two]) := by unfold StampValid; decide
example : ¬ StampValid (some rhsRoot) rhsRoot 0
    (execute initial [.opaqueCall (fun _ => (.three, .two))]) := by unfold StampValid; decide
example : StampValid (some rhsRoot) rhsRoot 0
    (execute initial [.write 12 .two .three]) := by unfold StampValid; decide

private def site : Site := ⟨1, 42, 7, ⟨17, 0, .two, .three⟩, ⟨23, 0, .three, .four⟩⟩

example : Applicable site site := ⟨rfl, ⟨(goal_iff _ _).mpr rfl⟩⟩
example : ¬ Applicable site {site with node := 43} := stale_refused (by decide)
example : ¬ Applicable site {site with right := {site.right with valueId := 24}} :=
  stale_refused (by decide)
example : ¬ Applicable site {site with left := {site.left with shapeVersion := 1}} :=
  stale_refused (by decide)
example : ¬ Applicable site {site with environmentVersion := 8} := stale_refused (by decide)

end Verificatum.Guard
