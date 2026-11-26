import Lean

open Lean

/-- Audit every declaration in the compiled module, including unused declarations. -/
def main (modules : List String) : IO Unit := do
  if modules.isEmpty then throw (IO.userError "Expected at least one OS proof module")
  initSearchPath (← findSysroot)
  let names := modules.map String.toName
  let env ← importModules (names.toArray.map fun name => { module := name }) {}
  for name in names do
    let some idx := env.getModuleIdx? name | throw (IO.userError s!"Missing module {name}")
    let decls := env.header.moduleData[idx.toNat]!.constants
    for decl in decls do
      if let .axiomInfo ax := decl then
        unless ax.isUnsafe && ax.name.isInternal do
          throw (IO.userError s!"{name} declares axiom {decl.name}")
    -- Compiler-generated unsafe code may use lcUnreachable. Logical declarations
    -- can depend only on safe code, so audit their complete dependency closure.
    let collect := decls.forM fun decl => do
      unless decl.isUnsafe do CollectAxioms.collect decl.name
    let (_, deps) := ((collect.run env).run {})
    for ax in deps.axioms do
      unless [``propext, ``Classical.choice, ``Quot.sound].contains ax do
        throw (IO.userError s!"{name} uses disallowed axiom {ax}")
    IO.println s!"{name}: {decls.size} declarations audited"
