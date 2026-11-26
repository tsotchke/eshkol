/-
  Copyright (C) tsotchke
  SPDX-License-Identifier: MIT

  Eshkol OS Phase K1: Machine-Checked Lean 4.12.0 Formal Proofs
    1. Per-CPU static IRQ ring buffer invariant preservation and bounds safety.
    2. Static NoAlloc trap termination across arbitrary finite IRQ/trap traces.
    3. Trap vector isolation and non-interference across all 5 ISAs
       (x86_64, AArch64, RISC-V 64, RISC-V 32, Cortex-M4).

  Model-to-Implementation Correspondence

  `lib/kernel/k1_boot_traps.c` dispatches traps and instruction decoding.
  The five decoder sources under `lib/kernel/arch/` are
  `k1_arch_x86_64.c`, `k1_arch_aarch64.c`, `k1_arch_riscv64.c`,
  `k1_arch_riscv32.c`, and `k1_arch_cortex_m4.c`. Lean classifies whole
  byte lists against selected encodings; the C decoders consume instruction
  bytes from a buffer and also report the consumed length and native cause.
  The heapAllocs and kernelCtx equalities are frame conditions for the
  modeled transitions. They detect changes to those fields in Lean.
  Native allocation behavior and register preservation require independent
  verification of the C and machine-code paths.
-/

namespace EshkolKernelK1

/-- Supported microkernel ISAs for Phase K1. -/
inductive ISA where
  | x86_64
  | aarch64
  | riscv64
  | riscv32
  | cortexM4
  deriving DecidableEq, Repr

/-- Canonical trap classification matching `k1_trap_code_t`. -/
inductive TrapCode where
  | none
  | timerIrq
  | divZero
  | illegalInsn
  | accessFault
  | syscall
  | unknown
  deriving DecidableEq, Repr

/-- Predicate identifying user-mode fault traps that terminate the faulting task. -/
def TrapCode.isFault : TrapCode -> Bool
  | .divZero     => true
  | .illegalInsn => true
  | .accessFault => true
  | .unknown     => true
  | _            => false

/-- Known-bad instruction encodings for each of the 5 ISAs. -/
def knownBadEncoding : ISA -> List UInt8
  | .x86_64   => [0x0F, 0x0B]             -- UD2
  | .aarch64  => [0x00, 0x00, 0x00, 0x00] -- UDF #0
  | .riscv64  => [0x00, 0x00, 0x00, 0x00] -- Illegal 0-word
  | .riscv32  => [0x00, 0x00, 0x00, 0x00] -- Illegal 0-word
  | .cortexM4 => [0x00, 0xDE]             -- Thumb UDF #0 (0xDE00 LE)

/-- Known-good NOP encodings for each of the 5 ISAs. -/
def knownNopEncoding : ISA -> List UInt8
  | .x86_64   => [0x90]                   -- NOP
  | .aarch64  => [0x1F, 0x20, 0x03, 0xD5] -- NOP (0xD503201F LE)
  | .riscv64  => [0x13, 0x00, 0x00, 0x00] -- ADDI x0, x0, 0 (0x00000013 LE)
  | .riscv32  => [0x13, 0x00, 0x00, 0x00] -- ADDI x0, x0, 0 (0x00000013 LE)
  | .cortexM4 => [0x00, 0xBF]             -- Thumb NOP (0xBF00 LE)

/-- Model of the 5-ISA instruction trap decoder. -/
def decodeInsn (isa : ISA) (bytes : List UInt8) : TrapCode :=
  match isa, bytes with
  | .x86_64,   [0x90]                   => .none
  | .x86_64,   [0x66, 0x90]             => .none
  | .x86_64,   [0x0F, 0x05]             => .syscall
  | .aarch64,  [0x1F, 0x20, 0x03, 0xD5] => .none
  | .riscv64,  [0x13, 0x00, 0x00, 0x00] => .none
  | .riscv64,  [0x01, 0x00]             => .none
  | .riscv32,  [0x13, 0x00, 0x00, 0x00] => .none
  | .riscv32,  [0x01, 0x00]             => .none
  | .cortexM4, [0x00, 0xBF]             => .none
  | _,         _                        => .illegalInsn

theorem all_five_isas_known_bad_traps_illegal (isa : ISA) :
    decodeInsn isa (knownBadEncoding isa) = TrapCode.illegalInsn := by
  cases isa <;> rfl

theorem all_five_isas_known_nop_never_illegal (isa : ISA) :
    decodeInsn isa (knownNopEncoding isa) = TrapCode.none := by
  cases isa <;> rfl

/-! ## Section 1: Per-CPU Static IRQ Ring Buffer Invariant Preservation -/

structure IrqEvent where
  tick  : Nat
  cpuId : Nat
  hwIrq : Nat
  deriving DecidableEq, Repr

structure IrqRing where
  cap     : Nat
  head    : Nat
  tail    : Nat
  count   : Nat
  dropped : Nat
  entries : Nat -> Option IrqEvent

/-- Structural ring invariant: bounds on head, tail, count, and modular coherence. -/
def IrqRing.Valid (r : IrqRing) : Prop :=
  0 < r.cap ∧
  r.head < r.cap ∧
  r.tail < r.cap ∧
  r.count ≤ r.cap ∧
  r.tail = (r.head + r.count) % r.cap

/-- Static zero-initialized ring of positive capacity (e.g. 256). -/
def IrqRing.init (cap : Nat) : IrqRing :=
  { cap := cap
    head := 0
    tail := 0
    count := 0
    dropped := 0
    entries := fun _ => Option.none }

theorem ring_init_valid (cap : Nat) (hcap : 0 < cap) :
    (IrqRing.init cap).Valid := by
  refine ⟨hcap, hcap, hcap, Nat.zero_le cap, ?_⟩
  simp [IrqRing.init]

/-- Push an IRQ event into the static ring; full rings drop without overwriting live entries. -/
def IrqRing.push (r : IrqRing) (ev : IrqEvent) : IrqRing :=
  if r.count < r.cap then
    { r with
      tail := (r.tail + 1) % r.cap
      count := r.count + 1
      entries := fun idx => if idx = r.tail then some ev else r.entries idx }
  else
    { r with dropped := r.dropped + 1 }

/-- Pop the oldest IRQ event from the static ring. -/
def IrqRing.pop (r : IrqRing) : Option IrqEvent × IrqRing :=
  if 0 < r.count then
    (r.entries r.head,
     { r with
       head := (r.head + 1) % r.cap
       count := r.count - 1 })
  else
    (Option.none, r)

theorem ring_push_write_in_bounds (r : IrqRing) (h : r.Valid) :
    r.tail < r.cap :=
  h.2.2.1

theorem ring_pop_read_in_bounds (r : IrqRing) (h : r.Valid) :
    r.head < r.cap :=
  h.2.1

theorem ring_push_overflow_preserves_entries (r : IrqRing) (ev : IrqEvent)
    (hfull : r.count = r.cap) :
    (r.push ev).entries = r.entries ∧
    (r.push ev).dropped = r.dropped + 1 ∧
    (r.push ev).count = r.count := by
  have hnot : ¬ (r.count < r.cap) := by omega
  simp [IrqRing.push, hnot]

theorem ring_push_preserves_valid (r : IrqRing) (ev : IrqEvent) (h : r.Valid) :
    (r.push ev).Valid := by
  rcases h with ⟨hcap, hhead, htail, hcount, hmod⟩
  unfold IrqRing.push
  split
  · rename_i hlt
    refine ⟨hcap, hhead, Nat.mod_lt _ hcap, by dsimp; omega, ?_⟩
    dsimp
    rw [hmod]
    have h1 : ((r.head + r.count) % r.cap + 1) % r.cap = (r.head + r.count + 1) % r.cap := by
      rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]
    rw [h1, Nat.add_assoc]
  · exact ⟨hcap, hhead, htail, hcount, hmod⟩

theorem ring_pop_preserves_valid (r : IrqRing) (h : r.Valid) :
    (r.pop).2.Valid := by
  rcases h with ⟨hcap, hhead, htail, hcount, hmod⟩
  unfold IrqRing.pop
  split
  · rename_i hpos
    refine ⟨hcap, Nat.mod_lt _ hcap, htail, by dsimp; omega, ?_⟩
    dsimp
    have h1 : ((r.head + 1) % r.cap + (r.count - 1)) % r.cap =
              (r.head + 1 + (r.count - 1)) % r.cap := by
      rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]
    rw [h1]
    have hsum : r.head + 1 + (r.count - 1) = r.head + r.count := by omega
    rw [hsum]
    exact hmod
  · exact ⟨hcap, hhead, htail, hcount, hmod⟩

theorem ring_pop_decrements_count (r : IrqRing) (hpos : 0 < r.count) :
    (r.pop).2.count = r.count - 1 := by
  simp [IrqRing.pop, hpos]

/-! ## Section 2: Static NoAlloc Trap Termination and Trap Vector Isolation -/

structure KernelContext where
  canaryLo   : Nat
  canaryHi   : Nat
  kernelRegs : List Nat
  kernelSp   : Nat
  kernelPc   : Nat
  deriving DecidableEq, Repr

inductive TaskState where
  | unused
  | runnable
  | faulted
  | exited
  deriving DecidableEq, Repr

structure UserTask where
  taskId   : Nat
  state    : TaskState
  lastTrap : TrapCode
  deriving DecidableEq, Repr

structure KernelState where
  isa        : ISA
  fuel       : Nat
  heapAllocs : Nat
  kernelCtx  : KernelContext
  irqRing    : IrqRing
  tasks      : Nat -> UserTask

inductive KernelEvent where
  | timerIrq (ev : IrqEvent)
  | userFault (tid : Nat) (trap : TrapCode)
  | userIllegalInsn (tid : Nat) (bytes : List UInt8)
  | userSyscall (tid : Nat)
  deriving DecidableEq, Repr

/-- Update a single user task slot while leaving all sibling tasks untouched. -/
def updateTask (tasks : Nat -> UserTask) (tid : Nat) (t : UserTask) : Nat -> UserTask :=
  fun id => if id = tid then t else tasks id

/-- Single-step microkernel transition under explicit bounded fuel and zero heap allocation. -/
def step (st : KernelState) (ev : KernelEvent) : KernelState :=
  match st.fuel with
  | 0 => st
  | f + 1 =>
    match ev with
    | .timerIrq irqEv =>
      { st with
        fuel := f
        irqRing := st.irqRing.push irqEv }
    | .userFault tid trap =>
      let cur := st.tasks tid
      if cur.state = .runnable ∧ trap.isFault then
        { st with
          fuel := f
          tasks := updateTask st.tasks tid { cur with state := .faulted, lastTrap := trap } }
      else
        { st with fuel := f }
    | .userIllegalInsn tid bytes =>
      let trap := decodeInsn st.isa bytes
      let cur := st.tasks tid
      if cur.state = .runnable ∧ trap.isFault then
        { st with
          fuel := f
          tasks := updateTask st.tasks tid { cur with state := .faulted, lastTrap := trap } }
      else
        { st with fuel := f }
    | .userSyscall tid =>
      let cur := st.tasks tid
      if cur.state = .runnable then
        { st with
          fuel := f
          tasks := updateTask st.tasks tid { cur with lastTrap := .syscall } }
      else
        { st with fuel := f }

/-- Execute an arbitrary finite trace of timer IRQs and user traps. -/
def runTrace (st : KernelState) (trace : List KernelEvent) : KernelState :=
  trace.foldl step st

theorem step_preserves_no_alloc (st : KernelState) (ev : KernelEvent) :
    (step st ev).heapAllocs = st.heapAllocs := by
  cases hfuel : st.fuel with
  | zero => simp [step, hfuel]
  | succ f =>
    cases ev with
    | timerIrq irqEv => simp [step, hfuel]
    | userFault tid trap =>
      unfold step
      simp only [hfuel]
      split <;> rfl
    | userIllegalInsn tid bytes =>
      unfold step
      simp only [hfuel]
      split <;> rfl
    | userSyscall tid =>
      unfold step
      simp only [hfuel]
      split <;> rfl

theorem trace_preserves_no_alloc (st : KernelState) (trace : List KernelEvent)
    (hzero : st.heapAllocs = 0) :
    (runTrace st trace).heapAllocs = 0 := by
  induction trace generalizing st with
  | nil => exact hzero
  | cons ev rest ih =>
    simp only [runTrace, List.foldl_cons]
    apply ih
    rw [step_preserves_no_alloc]
    exact hzero

theorem step_strictly_decreases_fuel (st : KernelState) (ev : KernelEvent)
    (hpos : 0 < st.fuel) :
    (step st ev).fuel < st.fuel := by
  cases hfuel : st.fuel with
  | zero => omega
  | succ f =>
    cases ev with
    | timerIrq irqEv => simp [step, hfuel]
    | userFault tid trap =>
      unfold step
      simp only [hfuel]
      split <;> simp
    | userIllegalInsn tid bytes =>
      unfold step
      simp only [hfuel]
      split <;> simp
    | userSyscall tid =>
      unfold step
      simp only [hfuel]
      split <;> simp

theorem step_preserves_ring_valid (st : KernelState) (ev : KernelEvent)
    (hvalid : st.irqRing.Valid) :
    (step st ev).irqRing.Valid := by
  cases hfuel : st.fuel with
  | zero => simpa [step, hfuel] using hvalid
  | succ f =>
    cases ev with
    | timerIrq irqEv =>
      simpa [step, hfuel] using ring_push_preserves_valid st.irqRing irqEv hvalid
    | userFault tid trap =>
      unfold step
      simp only [hfuel]
      split <;> exact hvalid
    | userIllegalInsn tid bytes =>
      unfold step
      simp only [hfuel]
      split <;> exact hvalid
    | userSyscall tid =>
      unfold step
      simp only [hfuel]
      split <;> exact hvalid

theorem trace_preserves_ring_valid (st : KernelState) (trace : List KernelEvent)
    (hvalid : st.irqRing.Valid) :
    (runTrace st trace).irqRing.Valid := by
  induction trace generalizing st with
  | nil => exact hvalid
  | cons ev rest ih =>
    simp only [runTrace, List.foldl_cons]
    exact ih (step st ev) (step_preserves_ring_valid st ev hvalid)

/-- Non-interference: every kernel step leaves kernel canary and kernel registers bit-identical. -/
theorem step_preserves_kernel_context (st : KernelState) (ev : KernelEvent) :
    (step st ev).kernelCtx = st.kernelCtx := by
  cases hfuel : st.fuel with
  | zero => simp [step, hfuel]
  | succ f =>
    cases ev with
    | timerIrq irqEv => simp [step, hfuel]
    | userFault tid trap =>
      unfold step
      simp only [hfuel]
      split <;> rfl
    | userIllegalInsn tid bytes =>
      unfold step
      simp only [hfuel]
      split <;> rfl
    | userSyscall tid =>
      unfold step
      simp only [hfuel]
      split <;> rfl

theorem trace_preserves_kernel_context (st : KernelState) (trace : List KernelEvent) :
    (runTrace st trace).kernelCtx = st.kernelCtx := by
  induction trace generalizing st with
  | nil => rfl
  | cons ev rest ih =>
    unfold runTrace
    rw [List.foldl_cons]
    exact (ih (step st ev)).trans (step_preserves_kernel_context st ev)

/-- User fault isolation: sibling tasks remain unchanged when task `tid` faults. -/
theorem user_fault_isolates_sibling_tasks (st : KernelState) (tid otherTid : Nat)
    (trap : TrapCode) (hne : otherTid ≠ tid) :
    (step st (.userFault tid trap)).tasks otherTid = st.tasks otherTid := by
  cases hfuel : st.fuel with
  | zero => simp [step, hfuel]
  | succ f =>
    unfold step
    simp only [hfuel]
    split
    · simp [updateTask, hne]
    · rfl

/-- User fault containment: a runnable faulting task transitions to `.faulted`. -/
theorem user_fault_marks_faulting_task (st : KernelState) (tid : Nat) (trap : TrapCode)
    (hpos : 0 < st.fuel)
    (hrun : (st.tasks tid).state = .runnable)
    (hfault : trap.isFault = true) :
    ((step st (.userFault tid trap)).tasks tid).state = .faulted := by
  cases hfuel : st.fuel with
  | zero => omega
  | succ f =>
    unfold step
    simp [hfuel, hrun, hfault, updateTask]

/-- Combined 5-ISA illegal instruction isolation theorem:
    injecting any ISA's known-bad instruction faults ONLY the target task,
    preserves all sibling tasks, preserves kernelCtx bit-for-bit, and allocates 0 bytes. -/
theorem all_five_isas_illegal_insn_isolation (st : KernelState) (tid otherTid : Nat)
    (hpos : 0 < st.fuel)
    (hzero : st.heapAllocs = 0)
    (hrun : (st.tasks tid).state = .runnable)
    (hne : otherTid ≠ tid) :
    let st' := step st (.userIllegalInsn tid (knownBadEncoding st.isa))
    (st'.tasks tid).state = .faulted ∧
    (st'.tasks tid).lastTrap = .illegalInsn ∧
    st'.tasks otherTid = st.tasks otherTid ∧
    st'.kernelCtx = st.kernelCtx ∧
    st'.heapAllocs = 0 := by
  cases hfuel : st.fuel with
  | zero => omega
  | succ f =>
    have hdec : decodeInsn st.isa (knownBadEncoding st.isa) = .illegalInsn :=
      all_five_isas_known_bad_traps_illegal st.isa
    simp [step, hfuel, hdec, TrapCode.isFault, hrun, updateTask, hne, hzero]

end EshkolKernelK1
