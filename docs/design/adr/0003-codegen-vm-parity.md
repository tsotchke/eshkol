# VM/LLVM Parity Conformance Matrix + Modularization Notes

Campaign research artifact, 2026-07-09. Distilled from an independent architecture research run.

## Op-by-op VM<->LLVM parity (vm-supported / gap+failing-test / native-only-justified)

```
op:ADD	vm-supported	OP_ADD bytecode
op:AND	vm-supported	vm_compiler special-form dispatch
op:BORROW	native-only-justified	OALR ownership — native-arena construct
op:CALL	vm-supported	core call opcode
op:CALL_CC	vm-supported	escape continuations verified 2026-07
op:CALL_WITH_VALUES	vm-supported	verified 2026-07
op:CASE	vm-supported	vm_compiler special-form dispatch
op:CASE_LAMBDA	gap	wrong clause selected (found/case_lambda_wrong_clause.esk)
op:COMPOSE	gap	compose undefined in VM
op:COND	vm-supported	vm_compiler special-form dispatch
op:COND_EXPAND	native-only-justified	parse-time form handled by the native front-end
op:CURL	gap	AD surface diverges in VM
op:DEFINE	vm-supported	vm_compiler special-form dispatch
op:DEFINE_RECORD_TYPE	vm-supported	verified 2026-07
op:DEFINE_SYNTAX	gap	simple rewrite macros work; recursive and set!-mutating macros diverge (found/recursive_macro_zero.esk, found/macro_set_top_level.esk)
op:DEFINE_TYPE	native-only-justified	static type syntax, erased before codegen
op:DERIVATIVE	gap	AD surface diverges in VM (found/ad_gradient_wrong.esk)
op:DERIVATIVE_N	gap	arbitrary-order Taylor-tower AD operator not implemented on VM surface
op:DIFF	gap	AD surface diverges in VM
op:DIRECTIONAL_DERIV	gap	AD surface diverges in VM
op:DIV	vm-supported	OP_DIV bytecode (exact-rational result diverges: see / gap row)
op:DIVERGENCE	gap	AD surface diverges in VM
op:DNC_ALLOC_WEIGHTS	gap	core.dnc has no VM fids
op:DNC_CONTENT_ADDR	gap	core.dnc has no VM fids
op:DNC_LOC_ADDR	gap	core.dnc has no VM fids
op:DNC_MAKE	gap	core.dnc has no VM fids
op:DNC_PRED	gap	core.dnc has no VM fids
op:DNC_READ	gap	core.dnc has no VM fids
op:DNC_READ_GRAD	gap	core.dnc has no VM fids
op:DNC_WRITE	gap	core.dnc has no VM fids
op:DO	gap	flat single do verified; nested do loses iterations, do+when spins forever, consecutive top-level dos leak state, and a do corrupts later top-level defines (found/do_composition_broken.esk, found/consecutive_do_state_leak.esk, found/define_after_do_corrupted.esk)
op:DYNAMIC_WIND	gap	after-thunk runs twice (found/dynamic_wind_after_twice.esk)
op:EXPECTED_FREE_ENERGY	vm-supported	BUILTINS fid 526
op:EXTERN	native-only-justified	C FFI declaration — native codegen only by design
op:EXTERN_VAR	native-only-justified	C FFI declaration — native codegen only by design
op:FACTOR_GRAPH_PRED	vm-supported	BUILTINS fid 521
op:FACT_PRED	vm-supported	BUILTINS fid 508
op:FG_ADD_FACTOR	vm-supported	BUILTINS fid 522
op:FG_INFER	vm-supported	BUILTINS fid 523
op:FG_OBSERVE	vm-supported	BUILTINS fid 527
op:FG_UPDATE_CPT	vm-supported	BUILTINS fid 524
op:FORALL	native-only-justified	static type syntax, erased before codegen
op:FREE_ENERGY	vm-supported	BUILTINS fid 525
op:GRADIENT	gap	returns 0 instead of gradient (found/ad_gradient_wrong.esk)
op:GUARD	vm-supported	guard/raise verified 2026-07
op:HESSIAN	gap	returns 0 (found/ad_gradient_wrong.esk)
op:IF	vm-supported	vm_compiler special-form dispatch
op:IMPORT	native-only-justified	front-end module machinery
op:INCLUDE	native-only-justified	parse-time form handled by the native front-end
op:JACOBIAN	gap	returns tensor handle, wrong values (found/ad_gradient_wrong.esk)
op:KB_ASSERT	vm-supported	BUILTINS fid 511
op:KB_PRED	vm-supported	BUILTINS fid 510
op:KB_QUERY	vm-supported	BUILTINS fid 512
op:KB_QUERY_PREFIX	gap	kb-query-prefix has no VM binding
op:LAMBDA	vm-supported	vm_compiler special-form dispatch
op:LAPLACIAN	gap	AD surface diverges in VM
op:LET	vm-supported	vm_compiler special-form dispatch (incl named let)
op:LETREC	vm-supported	vm_compiler special-form dispatch
op:LETREC_STAR	vm-supported	vm_compiler special-form dispatch (verified 2026-07)
op:LETREC_SYNTAX	gap	same as op:LET_SYNTAX
op:LET_STAR	vm-supported	vm_compiler special-form dispatch
op:LET_STAR_VALUES	gap	same family as op:LET_VALUES silent-wrong divergence
op:LET_SYNTAX	gap	let-syntax not expanded by vm_macro (undefined-variable error)
op:LET_VALUES	gap	accepted but binds wrong values (found/let_values_silent_zero.esk)
op:LOGIC_VAR	gap	?x resolves to itself through walk (found/logic_walk_unresolved.esk)
op:LOGIC_VAR_PRED	vm-supported	BUILTINS fid 501
```

## Gaps summary (ops that diverge / are missing in the VM)

- op:CASE_LAMBDA	gap	wrong clause selected (found/case_lambda_wrong_clause.esk)
- op:COMPOSE	gap	compose undefined in VM
- op:CURL	gap	AD surface diverges in VM
- op:DEFINE_SYNTAX	gap	simple rewrite macros work; recursive and set!-mutating macros diverge (found/recursive_macro_zero.esk, found/macro_set_top_level.esk)
- op:DERIVATIVE	gap	AD surface diverges in VM (found/ad_gradient_wrong.esk)
- op:DERIVATIVE_N	gap	arbitrary-order Taylor-tower AD operator not implemented on VM surface
- op:DIFF	gap	AD surface diverges in VM
- op:DIRECTIONAL_DERIV	gap	AD surface diverges in VM
- op:DIV	vm-supported	OP_DIV bytecode (exact-rational result diverges: see / gap row)
- op:DIVERGENCE	gap	AD surface diverges in VM
- op:DNC_ALLOC_WEIGHTS	gap	core.dnc has no VM fids
- op:DNC_CONTENT_ADDR	gap	core.dnc has no VM fids
- op:DNC_LOC_ADDR	gap	core.dnc has no VM fids
- op:DNC_MAKE	gap	core.dnc has no VM fids
- op:DNC_PRED	gap	core.dnc has no VM fids
- op:DNC_READ	gap	core.dnc has no VM fids
- op:DNC_READ_GRAD	gap	core.dnc has no VM fids
- op:DNC_WRITE	gap	core.dnc has no VM fids
- op:DO	gap	flat single do verified; nested do loses iterations, do+when spins forever, consecutive top-level dos leak state, and a do corrupts later top-level defines (found/do_composition_broken.esk, found/consecutive_do_state_leak.esk, found/define_after_do_corrupted.esk)
- op:DYNAMIC_WIND	gap	after-thunk runs twice (found/dynamic_wind_after_twice.esk)
- op:GRADIENT	gap	returns 0 instead of gradient (found/ad_gradient_wrong.esk)
- op:HESSIAN	gap	returns 0 (found/ad_gradient_wrong.esk)
- op:JACOBIAN	gap	returns tensor handle, wrong values (found/ad_gradient_wrong.esk)
- op:KB_QUERY_PREFIX	gap	kb-query-prefix has no VM binding
- op:LAPLACIAN	gap	AD surface diverges in VM
- op:LETREC_SYNTAX	gap	same as op:LET_SYNTAX
- op:LET_STAR_VALUES	gap	same family as op:LET_VALUES silent-wrong divergence
- op:LET_SYNTAX	gap	let-syntax not expanded by vm_macro (undefined-variable error)
- op:LET_VALUES	gap	accepted but binds wrong values (found/let_values_silent_zero.esk)
- op:LOGIC_VAR	gap	?x resolves to itself through walk (found/logic_walk_unresolved.esk)
