# Eshkol Compiler Codebase Map

**Complete Line-by-Line Reference**
**Generated**: December 2024
**Total Lines**: 35,794

---

## Table of Contents

1. [File Overview](#1-file-overview)
2. [inc/eshkol/eshkol.h - Core Definitions](#2-inceshkoleshkolh---core-definitions)
3. [inc/eshkol/llvm_backend.h - LLVM API](#3-inceshkolllvm_backendh---llvm-api)
4. [inc/eshkol/backend/ - Backend Headers](#4-inceshkolbackend---backend-headers)
5. [lib/backend/llvm_codegen.cpp - Main Code Generator](#5-libbackendllvm_codegencpp---main-code-generator)
6. [lib/frontend/parser.cpp - Parser](#6-libfrontendparsercpp---parser)
7. [lib/core/arena_memory - Memory Management](#7-libcorearena_memory---memory-management)
8. [lib/core/ - Other Core Files](#8-libcore---other-core-files)
9. [Standard Library (.esk files)](#9-standard-library-esk-files)
10. [Function Index](#10-function-index)

---

## 1. File Overview

### Source Files by Size

| File | Lines | Purpose |
|------|-------|---------|
| `lib/backend/llvm_codegen.cpp` | 27,758 | Main LLVM IR code generator |
| `lib/frontend/parser.cpp` | 3,685 | S-expression parser |
| `lib/core/arena_memory.cpp` | 1,778 | Arena memory runtime |
| `inc/eshkol/eshkol.h` | 599 | Core type definitions |
| `lib/core/arena_memory.h` | 339 | Arena memory interface |
| `lib/core/printer.cpp` | 319 | Display/print functions |
| `lib/backend/memory_codegen.cpp` | 176 | Arena function declarations |
| `lib/core/ast.cpp` | 176 | AST utilities |
| `inc/eshkol/llvm_backend.h` | 172 | LLVM backend API |
| `lib/core/logger.cpp` | 162 | Logging implementation |
| `lib/backend/function_cache.cpp` | 150 | C library function cache |
| `inc/eshkol/backend/memory_codegen.h` | 136 | MemoryCodegen interface |
| `inc/eshkol/backend/type_system.h` | 113 | TypeSystem interface |
| `lib/backend/type_system.cpp` | 95 | LLVM type creation |
| `inc/eshkol/backend/function_cache.h` | 88 | FunctionCache interface |
| `inc/eshkol/logger.h` | 48 | Logging interface |

---

## 2. inc/eshkol/eshkol.h - Core Definitions

**File**: `/Users/tyr/Desktop/eshkol/inc/eshkol/eshkol.h`
**Lines**: 599

### Type Definitions

| Lines | Name | Description |
|-------|------|-------------|
| 19-41 | `eshkol_type_t` | AST node types (UINT64, DOUBLE, STRING, FUNC, etc.) |
| 43-61 | `eshkol_value_type_t` | Runtime value types (INT64, DOUBLE, CONS_PTR, etc.) |
| 63-68 | Exactness flags | `ESHKOL_VALUE_EXACT_FLAG`, `ESHKOL_VALUE_INEXACT_FLAG` |
| 71-78 | `eshkol_tagged_data` | Union for tagged value data (int64, double, ptr) |
| 80-95 | `eshkol_tagged_value_t` | 16-byte tagged value struct |
| 98-145 | `eshkol_dual_number_t` | Dual number for forward-mode AD |
| 148-178 | Type checking macros | `ESHKOL_IS_INT64_TYPE`, `ESHKOL_IS_DOUBLE_TYPE`, etc. |

### Automatic Differentiation Types

| Lines | Name | Description |
|-------|------|-------------|
| 199-214 | `ad_node_type_t` | AD operation types (ADD, SUB, MUL, SIN, COS, etc.) |
| 216-225 | `ad_node_t` | AD computation graph node |
| 227-237 | `ad_tape_t` | AD tape for reverse-mode |

### Closure & Lambda Types

| Lines | Name | Description |
|-------|------|-------------|
| 240-249 | `eshkol_closure_env_t` | Closure environment (captured values) |
| 251-261 | `eshkol_closure_t` | Full closure structure |
| 263-267 | `eshkol_lambda_entry_t` | Lambda registry entry |
| 269-287 | `eshkol_lambda_registry_t` | Lambda S-expression registry |

### Display Options

| Lines | Name | Description |
|-------|------|-------------|
| 293-327 | `eshkol_display_opts_t` | Display formatting options |

### Operation Types Enum

| Lines | Name | Description |
|-------|------|-------------|
| 330-377 | `eshkol_op_type_t` | All operation types |

**Operation Types Detail**:
```
331: ESHKOL_INVALID_OP
332: ESHKOL_COMPOSE_OP
333: ESHKOL_IF_OP
334: ESHKOL_ADD_OP
335: ESHKOL_SUB_OP
336: ESHKOL_MUL_OP
337: ESHKOL_DIV_OP
338: ESHKOL_CALL_OP
339: ESHKOL_DEFINE_OP
340: ESHKOL_SEQUENCE_OP
341: ESHKOL_EXTERN_OP
342: ESHKOL_EXTERN_VAR_OP
343: ESHKOL_LAMBDA_OP
344: ESHKOL_LET_OP
345: ESHKOL_LET_STAR_OP
346: ESHKOL_LETREC_OP
347: ESHKOL_AND_OP
348: ESHKOL_OR_OP
349: ESHKOL_COND_OP
350: ESHKOL_CASE_OP
351: ESHKOL_DO_OP
352: ESHKOL_WHEN_OP
353: ESHKOL_UNLESS_OP
354: ESHKOL_QUOTE_OP
355: ESHKOL_SET_OP
356: ESHKOL_IMPORT_OP
357: ESHKOL_REQUIRE_OP
358: ESHKOL_PROVIDE_OP
360: ESHKOL_WITH_REGION_OP
361: ESHKOL_OWNED_OP
362: ESHKOL_MOVE_OP
363: ESHKOL_BORROW_OP
364: ESHKOL_SHARED_OP
365: ESHKOL_WEAK_REF_OP
366: ESHKOL_TENSOR_OP
367: ESHKOL_DIFF_OP
369: ESHKOL_DERIVATIVE_OP
370: ESHKOL_GRADIENT_OP
371: ESHKOL_JACOBIAN_OP
372: ESHKOL_HESSIAN_OP
373: ESHKOL_DIVERGENCE_OP
374: ESHKOL_CURL_OP
375: ESHKOL_LAPLACIAN_OP
376: ESHKOL_DIRECTIONAL_DERIV_OP
```

### Operation Structures

| Lines | Name | Description |
|-------|------|-------------|
| 382-526 | `eshkol_operations_t` | Union of all operation types |
| 385-388 | `base_op` | Base pointer operation |
| 389-392 | `compose_op` | Function composition |
| 393-396 | `if_op` | If-then-else |
| 397-401 | `call_op` | Function call |
| 402-411 | `define_op` | Variable/function definition |
| 412-415 | `sequence_op` | Begin/sequence expressions |
| 416-422 | `extern_op` | External function declaration |
| 423-426 | `extern_var_op` | External variable declaration |
| 427-435 | `lambda_op` | Lambda expression |
| 436-440 | `let_op` | Let/let*/letrec bindings |
| 441-444 | `set_op` | Variable mutation |
| 445-447 | `cond_op` | Cond expression |
| 448-451 | `case_op` | Case expression |
| 452-456 | `do_op` | Do loop |
| 457-459 | `when_op` | When conditional |
| 460-462 | `region_op` | With-region memory scope |
| 463-465 | `owned_op` | Owned value |
| 466-468 | `move_op` | Move ownership |
| 469-471 | `borrow_op` | Borrow reference |
| 472-475 | `shared_op` | Shared allocation |
| 476-478 | `quote_op` | Quoted data |
| 479-483 | `require_op` | Module require |
| 484-487 | `provide_op` | Module provide |
| 488-491 | `tensor_op` | Tensor literal |
| 492-495 | `diff_op` | Symbolic differentiation |
| 496-499 | `derivative_op` | Derivative operator |
| 500-503 | `gradient_op` | Gradient operator |
| 504-509 | `jacobian_op` | Jacobian operator |
| 510-515 | `hessian_op` | Hessian operator |
| 516-519 | `divergence_op` | Divergence operator |
| 520-523 | `curl_op` | Curl operator |
| 524-526 | `laplacian_op` | Laplacian operator |

### AST Structure

| Lines | Name | Description |
|-------|------|-------------|
| 528-571 | `eshkol_ast_t` | Main AST node structure |

---

## 3. inc/eshkol/llvm_backend.h - LLVM API

**File**: `/Users/tyr/Desktop/eshkol/inc/eshkol/llvm_backend.h`
**Lines**: 172

| Lines | Function | Description |
|-------|----------|-------------|
| 25-30 | `eshkol_llvm_generate_ir()` | Generate LLVM IR from ASTs |
| 32-38 | `eshkol_llvm_emit_object()` | Emit object file |
| 40-45 | `eshkol_llvm_emit_asm()` | Emit assembly |
| 47-52 | `eshkol_llvm_optimize()` | Run optimization passes |
| 54-60 | `eshkol_llvm_get_ir_string()` | Get IR as string |
| 62-67 | `eshkol_llvm_set_target()` | Set target triple |
| 69-74 | `eshkol_llvm_link()` | Link with external library |
| 76-81 | `eshkol_llvm_jit_compile()` | JIT compile module |
| 83-88 | `eshkol_llvm_jit_run()` | Run JIT-compiled code |
| 90-95 | `eshkol_llvm_free_module()` | Free LLVM module |
| 97-102 | `eshkol_llvm_free_jit()` | Free JIT resources |
| 104-120 | REPL mode functions | `eshkol_set_repl_mode()`, etc. |
| 122-172 | Internal declarations | Implementation details |

---

## 4. inc/eshkol/backend/ - Backend Headers

### 4.1 type_system.h (113 lines)

| Lines | Content | Description |
|-------|---------|-------------|
| 1-10 | Header guard & license | MIT license |
| 14-19 | Includes | LLVM headers |
| 22-34 | Class documentation | TypeSystem purpose |
| 35-108 | `class TypeSystem` | Full class definition |
| 44-51 | Primitive type getters | `getInt64Type()`, `getDoubleType()`, etc. |
| 54-57 | Struct type getters | `getTaggedValueType()`, `getDualNumberType()`, etc. |
| 63-85 | Field index constants | Struct field offsets |
| 87-107 | Private members | Cached LLVM types |

### 4.2 function_cache.h (88 lines)

| Lines | Content | Description |
|-------|---------|-------------|
| 1-11 | Header guard & license | MIT license |
| 21-34 | Class documentation | FunctionCache purpose |
| 35-83 | `class FunctionCache` | Full class definition |
| 43-47 | String functions | `getStrlen()`, `getStrcmp()`, etc. |
| 50-53 | Memory functions | `getMalloc()`, `getMemcpy()`, `getMemset()` |
| 56-57 | Format functions | `getSnprintf()`, `getStrtod()` |
| 61 | `reset()` | Clear cache for REPL |
| 64-82 | Private members | Cached function pointers |

### 4.3 memory_codegen.h (136 lines)

| Lines | Content | Description |
|-------|---------|-------------|
| 1-11 | Header guard & license | MIT license |
| 21-29 | Class documentation | MemoryCodegen purpose |
| 30-131 | `class MemoryCodegen` | Full class definition |
| 39-43 | Core arena functions | `getArenaCreate()`, etc. |
| 46-48 | Cons cell functions | `getArenaAllocateConsCell()`, etc. |
| 51 | Closure function | `getArenaAllocateClosure()` |
| 54-59 | Tagged cons getters | `getTaggedConsGetInt64()`, etc. |
| 62-66 | Tagged cons setters | `getTaggedConsSetInt64()`, etc. |
| 69-73 | Tape functions | `getArenaAllocateTape()`, etc. |
| 76 | AD node function | `getArenaAllocateAdNode()` |
| 78-131 | Private members | Function pointers and helpers |

---

## 5. lib/backend/llvm_codegen.cpp - Main Code Generator

**File**: `/Users/tyr/Desktop/eshkol/lib/backend/llvm_codegen.cpp`
**Lines**: 27,758

### Overview Section Map

| Lines | Section | Description |
|-------|---------|-------------|
| 1-55 | Includes & globals | Headers, LLVM includes, global state |
| 56-145 | Helper structures | `TypedValue`, `LambdaSExprMetadata`, helper functions |
| 146-320 | Class declaration | `EshkolLLVMCodeGen` member variables |
| 258-320 | Constructor | Initialize context, module, types |

### generateIR Entry Point

| Lines | Content | Description |
|-------|---------|-------------|
| 322-330 | Function signature | Entry point for IR generation |
| 331-345 | Arena creation | Create `__global_arena` global |
| 347-370 | AD mode flag | Create `__ad_mode_active` global |
| 372-395 | AD tape pointer | Create `__current_ad_tape` global |
| 397-560 | Tape stack | Create nested gradient globals |
| 562-584 | Function declarations | Create all function declarations |
| 586-608 | Global variable pre-declaration | Pre-declare global variables |
| 610-614 | Lambda pre-generation | Pre-generate top-level lambdas |
| 616-623 | Function definitions | Generate function bodies |
| 627-908 | Main wrapper | Create main() or library init |
| 910-929 | Module verification | Verify and return module |

### C Library Function Accessors

| Lines | Function | Forwards To |
|-------|----------|-------------|
| 933 | `getStrlenFunc()` | `funcs->getStrlen()` |
| 934 | `getMallocFunc()` | `funcs->getMalloc()` |
| 935 | `getMemcpyFunc()` | `funcs->getMemcpy()` |
| 936 | `getMemsetFunc()` | `funcs->getMemset()` |
| 937 | `getStrcmpFunc()` | `funcs->getStrcmp()` |
| 938 | `getStrcpyFunc()` | `funcs->getStrcpy()` |
| 939 | `getStrcatFunc()` | `funcs->getStrcat()` |
| 940 | `getStrstrFunc()` | `funcs->getStrstr()` |
| 941 | `getSnprintfFunc()` | `funcs->getSnprintf()` |
| 942 | `getStrtodFunc()` | `funcs->getStrtod()` |

### createBuiltinFunctions

| Lines | Content | Description |
|-------|---------|-------------|
| 944-963 | malloc | `void* malloc(size_t)` |
| 965-981 | printf | `int printf(const char*, ...)` |
| 983-1000 | sin | `double sin(double)` |
| 1002-1019 | cos | `double cos(double)` |
| 1021-1038 | sqrt | `double sqrt(double)` |
| 1040-1058 | pow | `double pow(double, double)` |
| 1060-1078 | exit | `void exit(int)` |
| 1080-1151 | File I/O | fopen, fclose, fgets, feof, fputs, fputc, strlen |
| 1153-1179 | Random | drand48, srand48, time |
| 1181-1231 | Math functions | tan, asin, acos, atan, sinh, cosh, tanh, etc. |
| 1233-1251 | eshkol_deep_equal | Deep equality comparison |
| 1253-1349 | Display system | eshkol_display_value, lambda registry |

### Arena Function Accessors

| Lines | Function | Forwards To |
|-------|----------|-------------|
| 1386 | `getArenaCreateFunc()` | `mem->getArenaCreate()` |
| 1387 | `getArenaDestroyFunc()` | `mem->getArenaDestroy()` |
| 1388 | `getArenaAllocateFunc()` | `mem->getArenaAllocate()` |
| 1389 | `getArenaPushScopeFunc()` | `mem->getArenaPushScope()` |
| 1390 | `getArenaPopScopeFunc()` | `mem->getArenaPopScope()` |
| 1391 | `getArenaAllocateConsCellFunc()` | `mem->getArenaAllocateConsCell()` |
| 1392 | `getArenaAllocateClosureFunc()` | `mem->getArenaAllocateClosure()` |
| 1393 | `getArenaAllocateTaggedConsCellFunc()` | `mem->getArenaAllocateTaggedConsCell()` |
| 1394-1404 | Tagged cons getters/setters | Various `mem->getTaggedCons*()` |
| 1405-1410 | Tape functions | Various `mem->getArenaTape*()` |

### Display Tensor Recursive

| Lines | Content | Description |
|-------|---------|-------------|
| 1421-1635 | `createDisplayTensorRecursiveFunction()` | Recursive tensor display |

### Function Declaration & Management

| Lines | Function | Description |
|-------|----------|-------------|
| 1636-1770 | `createFunctionDeclaration()` | Create function declaration |
| 1771-1837 | `declareNestedFunctions()` | Recursively declare nested functions |
| 1838-1946 | `createLibraryInitFunction()` | Create library init function |
| 1947-2340 | `createMainWrapper()` | Create main() wrapper |

### Type Inference

| Lines | Function | Description |
|-------|----------|-------------|
| 2342-2383 | `generateMixedArithmetic()` | Mixed-type arithmetic helper |
| 2385-2590 | `codegenTypedAST()` | AST codegen with type tracking |

### Cons Cell Code Generation

| Lines | Function | Description |
|-------|----------|-------------|
| 2753-2779 | `codegenArenaConsCell()` | Create basic cons cell |
| 2781-2827 | `codegenTaggedArenaConsCell()` | Create tagged cons cell |
| 2829-3004 | `codegenTaggedArenaConsCellFromTaggedValue()` | Create from tagged values |

### Tagged Value Pack/Unpack (CRITICAL)

| Lines | Function | Description |
|-------|----------|-------------|
| 3009-3036 | `packInt64ToTaggedValue()` | Pack int64 to tagged value |
| 3038-3067 | `packBoolToTaggedValue()` | Pack bool to tagged value |
| 3069-3094 | `packInt64ToTaggedValueWithTypeAndFlags()` | Pack with custom type/flags |
| 3096-3122 | `packDoubleToTaggedValue()` | Pack double to tagged value |
| 3124-3166 | `packPtrToTaggedValue()` | Pack pointer to tagged value |
| 3168-3205 | `packPtrToTaggedValueWithFlags()` | Pack pointer with custom flags |
| 3207-3238 | `packNullToTaggedValue()` | Pack null value |
| 3398-3412 | `unpackInt64FromTaggedValue()` | Extract int64 from tagged |
| 3414-3429 | `unpackDoubleFromTaggedValue()` | Extract double from tagged |
| 3431-3451 | `unpackPtrFromTaggedValue()` | Extract pointer from tagged |
| 3453-3603 | `extractCarAsTaggedValue()` | Extract car as tagged value |
| 3605-3733 | `extractCdrAsTaggedValue()` | Extract cdr as tagged value |
| 3735-3852 | `extractConsCarAsTaggedValue()` | Extract car from cons |
| 3854-4031 | Various helpers | Type conversion utilities |

### Polymorphic Arithmetic (CRITICAL)

| Lines | Function | Description |
|-------|----------|-------------|
| 4033-4261 | `polymorphicAdd()` | Type-polymorphic addition |
| 4263-4482 | `polymorphicSub()` | Type-polymorphic subtraction |
| 4484-4703 | `polymorphicMul()` | Type-polymorphic multiplication |
| 4705-4915 | `polymorphicDiv()` | Type-polymorphic division |
| 4917-5169 | `polymorphicCompare()` | Type-polymorphic comparison |

### Main AST Dispatcher

| Lines | Function | Description |
|-------|----------|-------------|
| 5171-5233 | `codegenAST()` | Main AST dispatch |
| 5235-5263 | `codegenString()` | String literal codegen |
| 5265-5385 | `codegenVariable()` | Variable lookup |
| 5387-5509 | `codegenOperation()` | Operation dispatch |

### Definition Code Generation

| Lines | Function | Description |
|-------|----------|-------------|
| 5511-5520 | `codegenDefine()` | Entry for definitions |
| 5522-5666 | `codegenFunctionDefinition()` | Function definition |
| 5668-5968 | `codegenNestedFunctionDefinition()` | Nested function with captures |
| 5970-6714 | `codegenVariableDefinition()` | Variable definition |
| 6716-6818 | `codegenSet()` | Variable mutation (set!) |

### Function Call Code Generation

| Lines | Function | Description |
|-------|----------|-------------|
| 6820-7936 | `codegenCall()` | Function call dispatch |
| 3241-3396 | `codegenClosureCall()` | Closure invocation |

### Arithmetic Operations

| Lines | Function | Description |
|-------|----------|-------------|
| 7938-7972 | `codegenArithmetic()` | Arithmetic dispatch |
| 7974-7992 | `codegenComparison()` | Comparison dispatch |

### Display & I/O

| Lines | Function | Description |
|-------|----------|-------------|
| 7994-8094 | `codegenDisplay()` | Display operation |
| 10822-10835 | `codegenNewline()` | Newline output |
| 10837-10931 | `codegenError()` | Error output |
| 10933-10971 | `codegenOpenInputFile()` | Open file for reading |
| 10973-11068 | `codegenReadLine()` | Read line from file |
| 11070-11093 | `codegenClosePort()` | Close file |
| 11095-11123 | `codegenEofObject()` | EOF object |
| 11125-11163 | `codegenOpenOutputFile()` | Open file for writing |
| 11165-11212 | `codegenWriteString()` | Write string to file |
| 11214-11264 | `codegenWriteLine()` | Write line to file |
| 11266-11317 | `codegenWriteChar()` | Write char to file |
| 11319-11351 | `codegenFlushOutputPort()` | Flush output |

### Math Functions

| Lines | Function | Description |
|-------|----------|-------------|
| 8096-8223 | `codegenMathFunction()` | Unary math functions |
| 8225-8244 | `codegenBinaryMathFunction()` | Binary math functions |
| 8246-8295 | `codegenModulo()` | Modulo operation |
| 8297-8348 | `codegenRemainder()` | Remainder operation |
| 8350-8393 | `codegenQuotient()` | Integer quotient |
| 8395-8461 | `codegenGCD()` | Greatest common divisor |
| 8463-8568 | `codegenLCM()` | Least common multiple |
| 8570-8596 | `codegenMinMax()` | Min/max operations |
| 8598-8686 | `codegenPow()` | Power function |

### Control Flow

| Lines | Function | Description |
|-------|----------|-------------|
| 8688-8747 | `codegenAnd()` | Short-circuit and |
| 8749-8804 | `codegenOr()` | Short-circuit or |
| 8806-8960 | `codegenCond()` | Multi-branch conditional |
| 8962-9072 | `codegenCase()` | Case/switch expression |
| 9074-9217 | `codegenDo()` | Do loop |
| 9219-9234 | `codegenNot()` | Logical not |
| 9236-9278 | `codegenWhen()` | When conditional |
| 9280-9322 | `codegenUnless()` | Unless conditional |
| 11474-11583 | `codegenIfCall()` | If expression |

### String Operations

| Lines | Function | Description |
|-------|----------|-------------|
| 9324-9352 | `codegenTypePredicate()` | Type predicates |
| 9354-9379 | `codegenStringLength()` | String length |
| 9381-9405 | `codegenStringRef()` | String character access |
| 9407-9451 | `codegenStringAppend()` | String concatenation |
| 9453-9493 | `codegenSubstring()` | Substring extraction |
| 9495-9531 | `codegenStringCompare()` | String comparison |
| 9533-9579 | `codegenNumberToString()` | Number to string conversion |
| 9581-9600 | `codegenStringToNumber()` | String to number conversion |
| 9602-9644 | `codegenMakeString()` | Create string |
| 9646-9681 | `codegenStringSet()` | String character mutation |
| 9683-9756 | `codegenStringToList()` | String to list |
| 9758-9884 | `codegenListToString()` | List to string |
| 9886-10045 | `codegenStringSplit()` | Split string |
| 10047-10068 | `codegenStringContains()` | String contains check |
| 10070-10098 | `codegenStringIndex()` | Find substring index |
| 10100-10173 | `codegenStringUpcase()` | Uppercase string |
| 10175-10248 | `codegenStringDowncase()` | Lowercase string |

### Character Operations

| Lines | Function | Description |
|-------|----------|-------------|
| 10254-10288 | `packCharToTaggedValue()` | Pack char |
| 10290-10303 | `codegenCharToInteger()` | Char to integer |
| 10305-10317 | `codegenIntegerToChar()` | Integer to char |
| 10319-10356 | `codegenCharCompare()` | Char comparison |

### Vector Operations

| Lines | Function | Description |
|-------|----------|-------------|
| 10358-10444 | `codegenMakeVector()` | Create vector |
| 10446-10482 | `codegenVector()` | Vector literal |
| 10484-10508 | `codegenSchemeVectorRef()` | Vector access |
| 10510-10550 | `codegenSchemeVectorSet()` | Vector mutation |
| 10552-10617 | `codegenVectorLength()` | Vector length |
| 10619-10653 | `codegenNumericPredicate()` | Numeric predicates |

### Equality Predicates

| Lines | Function | Description |
|-------|----------|-------------|
| 10655-10716 | `codegenEq()` | Object identity (eq?) |
| 10718-10796 | `codegenEqv()` | Value equivalence (eqv?) |
| 10798-10820 | `codegenEqual()` | Deep equality (equal?) |

### Sequence & External

| Lines | Function | Description |
|-------|----------|-------------|
| 11353-11361 | `codegenSequence()` | Sequence expressions |
| 11363-11405 | `codegenExternVar()` | External variable |
| 11407-11472 | `codegenExtern()` | External function |
| 11585-11641 | `codegenBegin()` | Begin expression |

### List Operations

| Lines | Function | Description |
|-------|----------|-------------|
| 11643-11657 | `codegenCons()` | Create cons cell |
| 11659-12142 | `codegenCar()` | Extract car |
| 12144-12645 | `codegenCdr()` | Extract cdr |
| 12647-12692 | `codegenList()` | Create list |
| 12694-12722 | `codegenNullCheck()` | Null check (null?) |
| 12724-12749 | `codegenPairCheck()` | Pair check (pair?) |
| 12751-13000 | `codegenConsCell()` | Cons cell from AST |
| 24707-24746 | `codegenSetCar()` | Set car (set-car!) |
| 24748-24790 | `codegenSetCdr()` | Set cdr (set-cdr!) |
| 24853-25430 | `codegenMap()` | Map function |
| 25432-25542 | `codegenMapWithClosure()` | Map with closure |
| 25544-25729 | `codegenMapSingleList()` | Map over single list |
| 25731-25937 | `codegenMapMultiList()` | Map over multiple lists |
| 25939-26025 | `codegenApply()` | Apply function |
| 26468-26499 | `codegenListStar()` | list* operation |
| 26501-26529 | `codegenAcons()` | Association list cons |
| 26531-26695 | `codegenPartition()` | Partition list |
| 26697-26817 | `codegenSplitAt()` | Split list at index |
| 26819-26988 | `codegenRemove()` | Remove from list |
| 26990-27069 | `codegenLast()` | Last element |
| 27071-27153 | `codegenLastPair()` | Last pair |
| 24313-24375 | `codegenIterativeReverse()` | Reverse list |
| 24377-24472 | `codegenUnzip()` | Unzip list |
| 24474-24544 | `codegenIota()` | Generate integer range |
| 24546-24660 | `codegenReduce()` | Reduce/fold list |

### Lambda & Closure

| Lines | Function | Description |
|-------|----------|-------------|
| 13002-13542 | `codegenLambda()` | Lambda expression |
| 13544-13852 | `codegenLet()` | Let bindings |
| 13854-14237 | `codegenLetrec()` | Recursive let bindings |

### Tensor Operations

| Lines | Function | Description |
|-------|----------|-------------|
| 14239-14310 | `codegenTensor()` | Tensor literal |
| 14312-14392 | `codegenTensorOperation()` | Tensor op dispatch |
| 14394-14447 | `codegenTensorGet()` | Tensor element access |
| 14449-14655 | `codegenTensorVectorRef()` | Vector reference |
| 14657-14709 | `codegenTensorSet()` | Tensor element set |
| 14711-14813 | `codegenSchemeVectorArithmetic()` | Vector arithmetic |
| 14815-14972 | `codegenTensorArithmeticInternal()` | Internal tensor arithmetic |
| 14974-14989 | `codegenTensorArithmetic()` | Tensor arithmetic dispatch |
| 14991-15164 | `codegenTensorDot()` | Dot product |
| 15166-15188 | `codegenTensorShape()` | Get tensor shape |
| 15190-15326 | `codegenTensorApply()` | Apply function to tensor |
| 15328-15437 | `codegenTensorReduceAll()` | Reduce all elements |
| 15439-15798 | `codegenTensorReduceWithDim()` | Reduce along dimension |
| 15713-15798 | `createTensorWithDims()` | Create tensor helper |
| 15800-15822 | `codegenZeros()` | Create zeros tensor |
| 15824-15849 | `codegenOnes()` | Create ones tensor |
| 15851-15916 | `codegenEye()` | Create identity matrix |
| 15918-16004 | `codegenArange()` | Create range tensor |
| 16006-16080 | `codegenLinspace()` | Create linspace tensor |
| 16082-16156 | `codegenReshape()` | Reshape tensor |
| 16158-16290 | `codegenTranspose()` | Transpose tensor |
| 16292-16345 | `codegenFlatten()` | Flatten tensor |
| 16347-16507 | `codegenMatmul()` | Matrix multiplication |
| 16509-16639 | `codegenTensorSum()` | Sum tensor |
| 16641-16769 | `codegenTensorMean()` | Mean of tensor |
| 16771-16867 | `codegenTrace()` | Matrix trace |
| 16869-16986 | `codegenNorm()` | Vector/matrix norm |
| 16988-17103 | `codegenOuterProduct()` | Outer product |

### Symbolic Differentiation

| Lines | Function | Description |
|-------|----------|-------------|
| 17105-17554 | `codegenDiff()` | Symbolic differentiation |
| 17198-17275 | Typed arithmetic helpers | `createTypedMul()`, etc. |

### Quote Operations

| Lines | Function | Description |
|-------|----------|-------------|
| 17556-17624 | `codegenQuotedAST()` | Quote AST |
| 17626-17838 | `codegenQuotedOperation()` | Quote operation |
| 17840-17854 | `codegenQuotedNaryOp()` | Quote n-ary operation |
| 17856-17959 | `codegenQuotedList()` | Quote list |
| 17961-18041 | `codegenLambdaToSExpr()` | Lambda to S-expression |

### Dual Number System (Forward-Mode AD)

| Lines | Function | Description |
|-------|----------|-------------|
| 18046-18092 | `packDualNumber()` | Create dual number |
| 18094-18123 | `packDualToTaggedValue()` | Dual to tagged value |
| 18125-18142 | `unpackDualFromTaggedValue()` | Unpack dual number |
| 18144-18457 | Dual arithmetic | Add, sub, mul, div for duals |

### Nested Gradient Support

| Lines | Function | Description |
|-------|----------|-------------|
| 18459-18516 | Tape stack operations | Push/pop tape for nesting |
| 18516-18666 | Double backward helpers | Nested gradient support |

### AD Node System (Reverse-Mode AD)

| Lines | Function | Description |
|-------|----------|-------------|
| 18566-18590 | `createADConstantOnTape()` | Create constant on tape |
| 18670-18723 | `createADConstant()` | Create AD constant |
| 18725-18800 | `createADVariable()` | Create AD variable |
| 18802-19015 | AD node operations | Create operation nodes |

### Backward Pass

| Lines | Function | Description |
|-------|----------|-------------|
| 19020-19405 | `codegenBackward()` | Backward pass implementation |

### Autodiff Operators

| Lines | Function | Description |
|-------|----------|-------------|
| 19407-19555 | `codegenDerivative()` | Derivative operator |
| 19557-20703 | `codegenGradient()` | Gradient operator |
| 20705-21486 | `codegenJacobian()` | Jacobian operator |
| 21488-22284 | `codegenHessian()` | Hessian operator |
| 22135-22201 | `createNullVectorTensor()` | Null vector helper |
| 22203-22284 | `extractJacobianElement()` | Jacobian element extraction |

### Vector Calculus

| Lines | Function | Description |
|-------|----------|-------------|
| 22286-22402 | `codegenDivergence()` | Divergence operator |
| 22404-22635 | `codegenCurl()` | Curl operator |
| 22637-22758 | `codegenLaplacian()` | Laplacian operator |
| 22760-22962 | `codegenDirectionalDerivative()` | Directional derivative |

### OALR (Ownership-Aware Lexical Regions)

| Lines | Function | Description |
|-------|----------|-------------|
| 22964-23031 | `codegenWithRegion()` | With-region expression |
| 23033-23049 | `codegenOwned()` | Owned value |
| 23051-23065 | `codegenMove()` | Move ownership |
| 23067-23092 | `codegenBorrow()` | Borrow reference |
| 23094-23110 | `codegenShared()` | Shared allocation |
| 23112-23128 | `codegenWeakRef()` | Weak reference |

### Display Helpers

| Lines | Function | Description |
|-------|----------|-------------|
| 23461-23610 | `codegenVectorToString()` | Vector to string |
| 23612-23815 | `codegenMatrixToString()` | Matrix to string |

### Compound Car/Cdr

| Lines | Function | Description |
|-------|----------|-------------|
| 23817-24280 | `codegenCompoundCarCdr()` | caar, cadr, cdar, cddr, etc. |

### Miscellaneous

| Lines | Function | Description |
|-------|----------|-------------|
| 24282-24311 | `codegenRandom()` | Random number |
| 24662-24678 | `codegenBooleanPredicate()` | Boolean? predicate |
| 24680-24705 | `codegenProcedurePredicate()` | Procedure? predicate |
| 24792-24851 | `codegenIndirectFunctionCall()` | Indirect call |

### Builtin Function Creation

| Lines | Function | Description |
|-------|----------|-------------|
| 27155-27231 | `createBuiltinArithmeticFunction()` | Create arithmetic builtin |
| 27233-27315 | `createBuiltinComparisonFunction()` | Create comparison builtin |
| 27317-27400+ | `createBuiltinPredicateFunction()` | Create predicate builtin |

---

## 6. lib/frontend/parser.cpp - Parser

**File**: `/Users/tyr/Desktop/eshkol/lib/frontend/parser.cpp`
**Lines**: 3,685

### Tokenizer

| Lines | Content | Description |
|-------|---------|-------------|
| 1-50 | Includes & Token enum | TOKEN_LPAREN, TOKEN_RPAREN, etc. |
| 52-100 | Token struct | Token with type, value, position |
| 102-195 | SchemeTokenizer class | Tokenizer implementation |

### Parser Functions

| Lines | Function | Description |
|-------|----------|-------------|
| 196-298 | `parse_atom()` | Parse atomic values |
| 299-340 | Quoted data parsers | `parse_quoted_data()`, etc. |
| 742-828 | `parse_expression()` | Main expression parser |
| 744-828 | `parse_function_signature()` | Parse function signature |
| 829-917 | `parse_list()` | Parse S-expression list |
| 918-1086 | Special form parsing | Define parsing |
| 1087-1119 | Let value parsing | Parse let binding values |
| 1120-1239 | If parsing | Parse if expressions |
| 1241-1387 | Lambda parsing | Parse lambda expressions |
| 1388-1506 | Let/let*/letrec parsing | Parse binding forms |
| 1507-1563 | Body parsing | Parse expression bodies |
| 1564-1731 | Case parsing | Parse case expressions |
| 1733-1893 | Do parsing | Parse do loops |
| 2116-2285 | Tensor parsing | Parse tensor literals |
| 2287-2527 | AD operator parsing | Parse derivative, gradient, etc. |
| 2529-3685 | Remaining forms | Various special forms |

---

## 7. lib/core/arena_memory - Memory Management

### arena_memory.h (339 lines)

| Lines | Content | Description |
|-------|---------|-------------|
| 26-28 | Type declarations | Forward declarations |
| 30-37 | `arena_block` | Memory block structure |
| 39-43 | `arena_scope` | Scope tracking structure |
| 45-71 | `arena` | Main arena structure |
| 74-77 | `arena_cons_cell_t` | Basic cons cell |
| 79-97 | `arena_tagged_cons_cell_t` | Tagged cons cell |
| 100-127 | Constructor functions | Convenience constructors |
| 128-140 | Tagged value functions | Direct storage/retrieval |
| 142-173 | AD functions | Tape and node allocation |
| 175-219 | Region structure | Lexical region support |
| 221-249 | Closure functions | Closure allocation |
| 231-250 | Shared memory | Reference counting support |
| 252-339 | Function declarations | All arena functions |

### arena_memory.cpp (1,778 lines)

| Lines | Content | Description |
|-------|---------|-------------|
| 1-50 | Includes & constants | Headers, default sizes |
| 52-120 | `arena_create()` | Create new arena |
| 122-180 | `arena_destroy()` | Destroy arena |
| 182-280 | `arena_allocate()` | Allocate memory |
| 282-340 | `arena_push_scope()` | Push new scope |
| 342-400 | `arena_pop_scope()` | Pop scope |
| 402-500 | Cons cell functions | Allocation and access |
| 502-650 | Tagged cons functions | Typed getters/setters |
| 652-800 | Closure functions | Closure allocation |
| 802-950 | Tape functions | AD tape management |
| 952-1100 | AD node functions | AD node allocation |
| 1102-1250 | Region functions | Region management |
| 1252-1400 | Shared memory | Reference counting |
| 1402-1550 | Weak reference | Weak pointer support |
| 1552-1778 | Utility functions | Statistics, debugging |

---

## 8. lib/core/ - Other Core Files

### printer.cpp (319 lines)

| Lines | Function | Description |
|-------|----------|-------------|
| 1-50 | Includes | Headers |
| 52-100 | `eshkol_display_value()` | Main display function |
| 102-200 | Type-specific display | Display by type |
| 202-319 | Helper functions | Formatting helpers |

### ast.cpp (176 lines)

| Lines | Function | Description |
|-------|----------|-------------|
| 1-30 | Includes | Headers |
| 32-100 | AST creation | AST node constructors |
| 102-176 | AST utilities | Copy, free, etc. |

### logger.cpp (162 lines)

| Lines | Function | Description |
|-------|----------|-------------|
| 1-30 | Includes & globals | Log level, output |
| 32-80 | `eshkol_log()` | Main logging function |
| 82-120 | Level-specific | Debug, info, warn, error |
| 122-162 | Configuration | Set level, output |

---

## 9. Standard Library (.esk files)

### lib/stdlib.esk (29 lines)

Entry point that requires all standard library modules.

### lib/core/ Modules

| File | Purpose |
|------|---------|
| `io.esk` | I/O operations (display, read, write) |
| `strings.esk` | String utilities |
| `math.esk` | Math functions |

### lib/core/list/ (List Operations)

| File | Purpose |
|------|---------|
| `compound.esk` | caar, cadr, cdar, cddr, etc. |
| `generate.esk` | range, repeat, iota |
| `transform.esk` | map, filter, fold |
| `query.esk` | length, member, assoc |
| `sort.esk` | Sorting algorithms |
| `higher_order.esk` | reduce, scan |
| `search.esk` | find, index |

### lib/core/functional/ (Functional Combinators)

| File | Purpose |
|------|---------|
| `compose.esk` | Function composition |
| `curry.esk` | Currying |
| `flip.esk` | Argument flipping |

### lib/core/operators/ (First-Class Operators)

| File | Purpose |
|------|---------|
| `arithmetic.esk` | +, -, *, / as functions |
| `compare.esk` | <, >, =, etc. as functions |

### lib/core/logic/ (Logic Operations)

| File | Purpose |
|------|---------|
| `boolean.esk` | Boolean combinators |
| `predicates.esk` | Type predicates |
| `types.esk` | Type checking |

---

## 10. Function Index

### Quick Reference: Most Important Functions

**Code Generation Entry**:
- `generateIR()` - Line 322
- `codegenAST()` - Line 5171
- `codegenOperation()` - Line 5387

**Tagged Value (CRITICAL)**:
- `packInt64ToTaggedValue()` - Line 3009
- `packDoubleToTaggedValue()` - Line 3096
- `packPtrToTaggedValue()` - Line 3124
- `unpackInt64FromTaggedValue()` - Line 3398
- `unpackDoubleFromTaggedValue()` - Line 3414

**Polymorphic Operations (CRITICAL)**:
- `polymorphicAdd()` - Line 4033
- `polymorphicSub()` - Line 4263
- `polymorphicMul()` - Line 4484
- `polymorphicDiv()` - Line 4705
- `polymorphicCompare()` - Line 4917

**Lambda & Closure**:
- `codegenLambda()` - Line 13002
- `codegenClosureCall()` - Line 3241
- `codegenLet()` - Line 13544
- `codegenLetrec()` - Line 13854

**List Operations**:
- `codegenCons()` - Line 11643
- `codegenCar()` - Line 11659
- `codegenCdr()` - Line 12144
- `codegenList()` - Line 12647
- `codegenMap()` - Line 24853

**Autodiff**:
- `codegenDerivative()` - Line 19407
- `codegenGradient()` - Line 19557
- `codegenJacobian()` - Line 20705
- `codegenHessian()` - Line 21488
- `codegenBackward()` - Line 19020

**Tensor**:
- `codegenTensor()` - Line 14239
- `codegenMatmul()` - Line 16347
- `codegenTranspose()` - Line 16158

---

## Appendix: Line Count Summary

```
HEADERS:
  inc/eshkol/eshkol.h              599
  inc/eshkol/llvm_backend.h        172
  inc/eshkol/logger.h               48
  inc/eshkol/backend/type_system.h 113
  inc/eshkol/backend/function_cache.h 88
  inc/eshkol/backend/memory_codegen.h 136
                            Subtotal: 1,156

BACKEND:
  lib/backend/llvm_codegen.cpp   27,758
  lib/backend/type_system.cpp        95
  lib/backend/function_cache.cpp    150
  lib/backend/memory_codegen.cpp    176
                            Subtotal: 28,179

FRONTEND:
  lib/frontend/parser.cpp         3,685
                            Subtotal: 3,685

CORE:
  lib/core/arena_memory.cpp       1,778
  lib/core/arena_memory.h           339
  lib/core/printer.cpp              319
  lib/core/ast.cpp                  176
  lib/core/logger.cpp               162
                            Subtotal: 2,774

TOTAL:                           35,794
```

---

*Document generated for Eshkol compiler refactoring project*
