/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * AutodiffCodegen implementation
 *
 * Note: The complex autodiff implementations remain in llvm_codegen.cpp
 * for now due to extensive dependencies on AST codegen, tape management,
 * and runtime library functions. This module provides the interface.
 */

#include <eshkol/backend/autodiff_codegen.h>
#include <eshkol/backend/binding_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <eshkol/runtime_exports.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Config/llvm-config.h>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstring>

// LLVM VERSION COMPATIBILITY
#if LLVM_VERSION_MAJOR >= 21
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getOrInsertDeclaration(mod, id, types)
#else
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getDeclaration(mod, id, types)
#endif

namespace eshkol {

AutodiffCodegen::AutodiffCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem) {
    eshkol_debug("AutodiffCodegen initialized");
}

// ===== DUAL NUMBER OPERATIONS (Forward-mode AD) =====
// Fully implemented - these are self-contained and don't depend on AST codegen
//
// SECOND-ORDER / NESTED forward-mode AD.
// ----------------------------------------------------------------------
// A dual number is a 4-component truncated bivariate Taylor jet
//     v = f0 + f1*e1 + f2*e2 + f3*e1*e2      (e1^2 = e2^2 = 0)
// stored as the LLVM struct {primal, d1, d2, d12}.  Each `derivative`
// nesting level (and each of the two arguments a Hessian differentiates
// w.r.t.) is tagged with a distinct perturbation e1 or e2 — this is the
// standard cure for "perturbation confusion".  Single-level AD only ever
// touches f0/f1, so the representation is backward-compatible: a plain
// {primal, tangent} dual is just {primal, tangent, 0, 0}.
//
// Arithmetic on these jets is exact and finite (no recursion, no finite
// differences); the closed forms are implemented below.

// ============================================================================
// ESH-0188 (P3): subsuming the transitional 8-component jet into the tower.
// ----------------------------------------------------------------------
// PR #138 (ESH-0117) added a THIRD independent nilpotent perturbation `ep`
// (reverse-seed) alongside the existing e1/e2 forward levels so that
// `gradient` could see through a 2-level-deep nested `derivative` chain: that
// is a genuine 3-way SIMULTANEOUS mixed partial (inner forward variable,
// outer forward variable, reverse-seed direction), i.e. exactly
// {1,e1,e2,ep,e1e2,e1ep,e2ep,e1e2ep} — the "hyper-dual" basis the design doc
// (docs/design/AD_TAYLOR_TOWER.md §3) calls the 2^3 wall.
//
// P1/P2 (ESH-0186/0187) built an arbitrary-order UNIVARIATE Taylor tower with
// an explicit epoch-tag confusion model (lib/core/runtime_taylor.c) and
// already subsume the part of the jet's job that IS univariate: a pure chain
// of nested `derivative` calls threading one variable (no independent second
// or third direction) is detected by detectPureDerivChain() below and routed
// through that tower for depth >= 3, unchanged by this phase.
//
// What that univariate tower cannot express is the *simultaneous* multi-
// directional case: `gradient` over `p` wrapping `derivative` over `x`
// wrapping `derivative` over `xx`, where `p`, `x` and `xx` are three
// independent perturbation directions and the quantity of interest is their
// full mixed partial. A single-direction power series has no way to hold
// three orthogonal directions; representing them exactly requires either (a)
// genuine multivariate directional propagation (Griewank-Utke-Walther,
// design doc §7 — Phase P4, ESH-0189) or (b) tower-valued reverse-tape
// primals/adjoints (design doc §8 — Phase P5, ESH-0190). Both are later,
// independently-gated phases owned by their own work — building either one
// early, inside P3, would be exactly the kind of undocumented scope-creep
// the phase boundaries exist to prevent.
//
// P3's scope is therefore: (1) retire the dead compile-time-depth
// predecessor of this mechanism (seedDerivativeInput/extractDerivativeResult,
// superseded by the ESH-0070 runtime-level mechanism and unreferenced —
// removed above), (2) express the jet's THREE perturbation slots through
// named epoch-style constants instead of magic field indices so the
// confusion-avoidance discipline reads the same way here as it does in the
// tower's EPOCH_TAG model (inc/eshkol/eshkol.h), and (3) keep the remaining
// 8-field arithmetic as a clearly-delimited, BOUNDED thin adapter: it never
// grows past 3 simultaneous directions (unlike a hypothetical N-level nested
// jet, which is the actual "2^N wall" the tower design eliminates), it is
// gated behind detectPureDerivChain's depth<3 fallthrough plus a live reverse
// tape, and JET4 (order <= 2, no reverse tape live) is untouched — the ep
// fields stay compile-time-constant zero and fold away. Full removal is
// blocked on P4/P5 landing; see docs/design/AD_TAYLOR_TOWER.md §12.
// ============================================================================

namespace {

// Named perturbation slots for the top-level jet fields (used when SEEDING a
// fresh perturbation — see seedForwardAndPush / maybeJetLiftTapeOperand).
// These are the epoch-style tags of the note above: kJetE1/kJetE2 are the two
// forward nesting levels, kJetEp is the reverse-seed companion direction.
enum DualJetSlot : unsigned {
    kJetE1 = 1,  // innermost live forward `derivative` level
    kJetE2 = 2,  // next forward nesting level out
    kJetEp = 4,  // reverse-seed companion (ESH-0117/ESH-0188), field 4 = dvalue/dep primal
};

// ESH-0221: true if `v` is a named-let / TCO loop-variable alloca (named
// "<name>_tco"). codegenLambda captures such allocas BY VALUE (its
// is_let_alloca gate excludes the "_tco" suffix — see llvm_codegen.cpp
// codegenLambda's free-variable capture loop) — the lambda's own body then
// reads its capture parameter with a SINGLE load, expecting a value-typed
// slot, not a pointer-marker to re-dereference.
//
// The manual free-variable-capture RECONSTRUCTION below (derivative,
// gradient, and the shared jacobian/hessian/divergence/curl/laplacian
// capture resolvers) independently looks up a captured variable's `storage`
// Value* via symbol_table_/global_symbol_table_ and decides how to forward
// it to the differentiated function using a single test:
// `storage->getType()->isPointerTy()`. A TCO loop-carried parameter's
// alloca IS pointer-typed (every AllocaInst is), so — absent this check —
// it fell into the "mutable/let-bound variable" branch that packs the
// alloca's ADDRESS as an int64-tagged pointer-marker for the callee to
// re-dereference (double indirection). But the callee lambda, for a _tco
// free variable, does a SINGLE load expecting the actual value — so it
// read the raw alloca address bit-pattern as if it were the captured
// double, producing garbage (a stack address reinterpreted as ~1e9-1e10)
// for scalars, and corrupted-list/segfault symptoms for captured pairs.
//
// This mirrors map_codegen.cpp's isTcoLoopAlloca (bug-RR) — same
// value-vs-pointer capture-convention mismatch, different codegen path.
/**
 * @brief True if `v` is a named-let/TCO loop-variable alloca (name ends in "_tco").
 *
 * Distinguishes a TCO loop-carried parameter's alloca — which codegenLambda
 * captures BY VALUE — from an ordinary mutable-variable alloca, whose address
 * would otherwise be packed as a pointer-marker and misread as a raw double
 * by the callee's single-load capture convention (mirrors map_codegen.cpp's
 * isTcoLoopAlloca, bug-RR).
 *
 * @param v candidate captured storage value.
 * @return true iff `v` is an AllocaInst whose name ends with "_tco".
 */
bool isTcoLoopAlloca(llvm::Value* v) {
    auto* a = llvm::dyn_cast_or_null<llvm::AllocaInst>(v);
    if (!a) return false;
    llvm::StringRef name = a->getName();
    return name.size() >= 4 && name.substr(name.size() - 4) == "_tco";
}

// Read component `idx` of a dual struct value.
// 0=primal,1=e1,2=e2,3=e1e2 (value 4-jet); 4..7 = the ep-derivative 4-jet
// (ESH-0117: d(value 4-jet)/dep, ep = reverse-seed infinitesimal).
inline llvm::Value* dualField(CodegenContext& ctx, llvm::Value* dual, unsigned idx) {
    return ctx.builder().CreateExtractValue(dual, {idx});
}

// Build a dual from all EIGHT components (value 4-jet v0..v3, ep-deriv 4-jet
// p0..p3). This is the canonical constructor for the ESH-0117 mixed-mode jet.
inline llvm::Value* makeDual8(CodegenContext& ctx,
                              llvm::Value* v0, llvm::Value* v1,
                              llvm::Value* v2, llvm::Value* v3,
                              llvm::Value* p0, llvm::Value* p1,
                              llvm::Value* p2, llvm::Value* p3) {
    llvm::Value* d = llvm::UndefValue::get(ctx.dualNumberType());
    auto& b = ctx.builder();
    d = b.CreateInsertValue(d, v0, {0});
    d = b.CreateInsertValue(d, v1, {1});
    d = b.CreateInsertValue(d, v2, {2});
    d = b.CreateInsertValue(d, v3, {3});
    d = b.CreateInsertValue(d, p0, {4});
    d = b.CreateInsertValue(d, p1, {5});
    d = b.CreateInsertValue(d, p2, {6});
    d = b.CreateInsertValue(d, p3, {7});
    return d;
}

// Build a dual from the value 4-jet only; the ep-derivative 4-jet defaults to
// zero. Used by every non-mixed-mode construction site — bit-compatible with
// the historical 4-field dual (fields 4-7 are simply 0).
inline llvm::Value* makeDual4(CodegenContext& ctx,
                              llvm::Value* f0, llvm::Value* f1,
                              llvm::Value* f2, llvm::Value* f3) {
    llvm::Value* zero = llvm::ConstantFP::get(ctx.doubleType(), 0.0);
    return makeDual8(ctx, f0, f1, f2, f3, zero, zero, zero, zero);
}

// Bilinear product of two 4-jets {j0,j1,j2,j3} (monomials 1,e1,e2,e1e2),
// keeping the mixed e1e2 cross term. Shared by dualMul and the ep-derivative
// propagation of dualUnaryChain.
inline std::array<llvm::Value*, 4> jet4Mul(CodegenContext& ctx,
        llvm::Value* a0, llvm::Value* a1, llvm::Value* a2, llvm::Value* a3,
        llvm::Value* b0, llvm::Value* b1, llvm::Value* b2, llvm::Value* b3) {
    auto& b = ctx.builder();
    llvm::Value* r0 = b.CreateFMul(a0, b0);
    llvm::Value* r1 = b.CreateFAdd(b.CreateFMul(a1, b0), b.CreateFMul(a0, b1));
    llvm::Value* r2 = b.CreateFAdd(b.CreateFMul(a2, b0), b.CreateFMul(a0, b2));
    llvm::Value* r3 = b.CreateFAdd(
        b.CreateFAdd(b.CreateFMul(a3, b0), b.CreateFMul(a1, b2)),
        b.CreateFAdd(b.CreateFMul(a2, b1), b.CreateFMul(a0, b3)));
    return {r0, r1, r2, r3};
}

// Apply a scalar unary function g to a dual, propagating BOTH forward
// perturbation slots and the mixed second-order term exactly:
//   g(u) = g(a) + g'(a)(d1 e1 + d2 e2 + d12 e1e2)
//                + g''(a)(d1 d2) e1e2      (the only surviving 2nd-order term)
// where fa=g(a), fpa=g'(a), fppa=g''(a) at the primal a = field 0.
//
// ESH-0117: the ep-derivative 4-jet (fields 4-7) propagates by the chain rule
// as g'(F) ⊗ Fp, where g'(F) is the value 4-jet of the FUNCTION g' evaluated
// at F (needing g'''=fpppa for its e1e2 component) and Fp is the incoming
// ep-derivative 4-jet. fpppa may be null when the 3rd derivative is not
// supplied; then the g''' contribution to the triple (e1 e2 ep) term is
// dropped — exact whenever ep does not flow through this function's argument
// (the common case, since a captured reverse variable enters as a
// multiplicative/additive factor around the shape, not inside a transcendental
// of the forward variable).
inline llvm::Value* dualUnaryChain(CodegenContext& ctx, llvm::Value* dual,
                                   llvm::Value* fa, llvm::Value* fpa, llvm::Value* fppa,
                                   llvm::Value* fpppa = nullptr) {
    auto& b = ctx.builder();
    llvm::Value* d1 = b.CreateExtractValue(dual, {1});
    llvm::Value* d2 = b.CreateExtractValue(dual, {2});
    llvm::Value* d12 = b.CreateExtractValue(dual, {3});
    llvm::Value* d1d2 = b.CreateFMul(d1, d2);
    // value 4-jet: chain with (g, g', g'')
    llvm::Value* o1 = b.CreateFMul(fpa, d1);
    llvm::Value* o2 = b.CreateFMul(fpa, d2);
    llvm::Value* o3 = b.CreateFAdd(b.CreateFMul(fpa, d12),
                                   b.CreateFMul(fppa, d1d2));
    // g'(F) 4-jet: chain with (g', g'', g''')
    if (!fpppa) fpppa = llvm::ConstantFP::get(ctx.doubleType(), 0.0);
    llvm::Value* g0 = fpa;
    llvm::Value* g1 = b.CreateFMul(fppa, d1);
    llvm::Value* g2 = b.CreateFMul(fppa, d2);
    llvm::Value* g3 = b.CreateFAdd(b.CreateFMul(fppa, d12),
                                   b.CreateFMul(fpppa, d1d2));
    // ep-derivative: g'(F) ⊗ Fp  (Fp = incoming fields 4-7)
    llvm::Value* p0 = b.CreateExtractValue(dual, {4});
    llvm::Value* p1 = b.CreateExtractValue(dual, {5});
    llvm::Value* p2 = b.CreateExtractValue(dual, {6});
    llvm::Value* p3 = b.CreateExtractValue(dual, {7});
    std::array<llvm::Value*, 4> pr = jet4Mul(ctx, g0, g1, g2, g3, p0, p1, p2, p3);
    return makeDual8(ctx, fa, o1, o2, o3, pr[0], pr[1], pr[2], pr[3]);
}

} // anonymous namespace

// ESH-0070: perturbation-level tagging is now a RUNTIME counter
// (ctx_.adPertLevel()), pushed/popped around each forward-mode call — see
// seedForwardAndPush / popAndExtractForward. The previous compile-time
// file-static depth could not see across a function-call / named-let TCO
// boundary, so a gradient/derivative reached through a called function
// clobbered the outer perturbation. Slot: level 0 -> e1, level 1 -> e2.

/**
 * @brief Seed a fresh single-level forward-mode dual number {primal, tangent}.
 *
 * Builds the 8-field jet with only field 0 (primal) and field 1 (e1 tangent)
 * populated; all other slots (e2, e1e2, and the full ep-derivative 4-jet) are
 * explicitly zeroed so second derivatives and reverse-seed dependence start
 * from a known-zero state rather than uninitialised memory.
 *
 * @param primal function value at the seed point.
 * @param tangent seed derivative (usually 1.0).
 * @return the constructed dual-number LLVM struct value, or nullptr if either input is null.
 */
llvm::Value* AutodiffCodegen::createDualNumber(llvm::Value* primal, llvm::Value* tangent) {
    if (!primal || !tangent) return nullptr;
    // {primal, tangent(e1), 0(e2), 0(e1e2)} — single-level seed.  The e2/e1e2
    // slots MUST be explicitly zeroed (a 4-field alloca would leave them
    // uninitialised → garbage second derivatives).
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    return makeDual4(ctx_, primal, tangent, zero, zero);
}

// ESH-0188 (P3): the compile-time-`int depth` seed/extract pair that used to
// live here (seedDerivativeInput/extractDerivativeResult) was the pre-ESH-0070
// perturbation-tagging mechanism — see the header for why it was removed.

// ===== ESH-0070: runtime-level forward-mode perturbation tagging =====
// The perturbation level is a runtime counter (ctx_.adPertLevel()), pushed and
// popped around each forward-mode derivative/gradient CALL. This is what makes
// nesting across a function-call / named-let TCO boundary correct: the inner
// derivative reads the level the OUTER one left live and therefore seeds a
// distinct perturbation slot (e2) instead of clobbering the outer's (e1).
// Compile-time lexical depth could not see across the call boundary; a runtime
// push/pop around the call is, by construction, invariant under TCO re-entry.

/** @brief Load the runtime forward-mode perturbation-level counter (`__ad_pert_level`), or 0 if the global is absent. */
llvm::Value* AutodiffCodegen::adPertLevelLoad() {
    llvm::GlobalVariable* g = ctx_.adPertLevel();
    if (!g) return llvm::ConstantInt::get(ctx_.int64Type(), 0);
    return ctx_.builder().CreateLoad(ctx_.int64Type(), g, "pert_level");
}

/** @brief Store a new value into the runtime forward-mode perturbation-level counter global. */
void AutodiffCodegen::adPertLevelStore(llvm::Value* level) {
    llvm::GlobalVariable* g = ctx_.adPertLevel();
    if (!g) return;
    ctx_.builder().CreateStore(level, g);
}

// ESH-0190 (P5): enter a Taylor-tower differentiation context. Increments the
// active-depth counter and records the current tower order, so that reverse-tape
// AD nodes reaching arithmetic inside the tower body (the lambda, compiled
// separately) are frozen to dual-towers rather than swallowed by the tape.
/**
 * @brief Enter a Taylor-tower differentiation context (ESH-0190/P5).
 *
 * Increments the active-tower-depth counter and records the current tower
 * order, so reverse-tape AD nodes reaching arithmetic inside the tower body
 * are frozen to dual-towers (via towerLiftOperand) instead of being recorded
 * on the tape.
 *
 * @param order_i32 the tower order (as i32) to record as the active order.
 */
void AutodiffCodegen::towerCtxPush(llvm::Value* order_i32) {
    auto& b = ctx_.builder();
    llvm::GlobalVariable* ga = ctx_.adTowerActive();
    llvm::GlobalVariable* go = ctx_.adTowerOrder();
    if (ga) {
        llvm::Value* d = b.CreateLoad(ctx_.int64Type(), ga, "twr_depth");
        b.CreateStore(b.CreateAdd(d, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ga);
    }
    if (go && order_i32) {
        llvm::Value* o64 = b.CreateIntCast(order_i32, ctx_.int64Type(), true);
        b.CreateStore(o64, go);
    }
}

// ESH-0190 (P5): leave a Taylor-tower differentiation context (decrement depth).
// The order global is left as-is: a foreign-tag constant tower zero-extends, so
// a stale innermost order is harmless while depth > 0, and unread at depth 0.
/**
 * @brief Leave a Taylor-tower differentiation context, decrementing the active-depth counter.
 *
 * Clamps the depth at 0 defensively; the order global is left stale (harmless
 * while depth > 0, unread at depth 0).
 */
void AutodiffCodegen::towerCtxPop() {
    auto& b = ctx_.builder();
    llvm::GlobalVariable* ga = ctx_.adTowerActive();
    if (!ga) return;
    llvm::Value* d = b.CreateLoad(ctx_.int64Type(), ga, "twr_depth");
    llvm::Value* dm = b.CreateSub(d, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    // clamp at 0 (defensive; balanced push/pop keeps it >= 0)
    llvm::Value* neg = b.CreateICmpSLT(dm, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    dm = b.CreateSelect(neg, llvm::ConstantInt::get(ctx_.int64Type(), 0), dm);
    b.CreateStore(dm, ga);
}

// === ESH-0186: runtime Taylor-tower kernel declarations (lib/core/runtime_taylor.c) ===
namespace {
/** @brief Get or declare `eshkol_taylor_seed_tagged` (arena*, point tagged*, order i32, out tagged*) -> void, the runtime Taylor-tower seeding kernel. */
llvm::Function* getTaylorSeedFunc(CodegenContext& ctx) {
    if (auto* f = ctx.module().getFunction("eshkol_taylor_seed_tagged")) return f;
    // void eshkol_taylor_seed_tagged(arena*, const tagged* point, i32 order, tagged* out)
    llvm::Type* p = ctx.ptrType();
    auto* ft = llvm::FunctionType::get(ctx.voidType(),
        {p, p, ctx.int32Type(), p}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                  "eshkol_taylor_seed_tagged", &ctx.module());
}
/** @brief Get or declare `eshkol_taylor_extract` (tower tagged*, n i32) -> double, extracting a bare k-th derivative as a double from a Taylor tower. */
llvm::Function* getTaylorExtractFunc(CodegenContext& ctx) {
    if (auto* f = ctx.module().getFunction("eshkol_taylor_extract")) return f;
    // double eshkol_taylor_extract(const tagged* tower, i32 n)
    // ESH-0190 (P5): the plain double extraction, still used by the
    // reverse-over-Taylor tangent path (which needs f^(k) as a bare double to
    // record the local linearization / build the outer jet). The exactness-
    // preserving eshkol_taylor_extract_tagged (P6) is used for the plain,
    // no-tangent return.
    auto* ft = llvm::FunctionType::get(ctx.doubleType(),
        {ctx.ptrType(), ctx.int32Type()}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                  "eshkol_taylor_extract", &ctx.module());
}
/** @brief Get or declare `eshkol_taylor_extract_tagged` (arena*, tower tagged*, n i32, out tagged*) -> void; exactness-preserving extraction (ESH-0191/P6) that returns f^(n) as an exact tagged value instead of forcing a double. */
llvm::Function* getTaylorExtractTaggedFunc(CodegenContext& ctx) {
    if (auto* f = ctx.module().getFunction("eshkol_taylor_extract_tagged")) return f;
    // void eshkol_taylor_extract_tagged(arena*, const tagged* tower, i32 n, tagged* out)
    // ESH-0191 (P6): tagged out-param (not a raw double) so an exact tower's
    // f^(n) = n!*c[n] comes back as an exact int64/bignum/rational tagged
    // value instead of being force-cast to double.
    llvm::Type* p = ctx.ptrType();
    auto* ft = llvm::FunctionType::get(ctx.voidType(),
        {p /*arena*/, p /*tower*/, ctx.int32Type(), p /*out*/}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                  "eshkol_taylor_extract_tagged", &ctx.module());
}
// ESH-0190 (P5): reverse-over-Taylor extraction / lift helpers.
/** @brief Get or declare `eshkol_taylor_has_tangent` (tower tagged*) -> i32, testing whether a Taylor tower carries a reverse-seed tangent (ESH-0190/P5). */
llvm::Function* getTaylorHasTangentFunc(CodegenContext& ctx) {
    if (auto* f = ctx.module().getFunction("eshkol_taylor_has_tangent")) return f;
    // i32 eshkol_taylor_has_tangent(const tagged* tower)
    auto* ft = llvm::FunctionType::get(ctx.int32Type(), {ctx.ptrType()}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                  "eshkol_taylor_has_tangent", &ctx.module());
}
/** @brief Get or declare `eshkol_taylor_extract_tangent` (tower tagged*, n i32) -> double, extracting the seed-tangent of the k-th coefficient (ESH-0190/P5 reverse-over-Taylor). */
llvm::Function* getTaylorExtractTangentFunc(CodegenContext& ctx) {
    if (auto* f = ctx.module().getFunction("eshkol_taylor_extract_tangent")) return f;
    // double eshkol_taylor_extract_tangent(const tagged* tower, i32 n)
    auto* ft = llvm::FunctionType::get(ctx.doubleType(),
        {ctx.ptrType(), ctx.int32Type()}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                  "eshkol_taylor_extract_tangent", &ctx.module());
}
/** @brief Get or declare `eshkol_taylor_lift_ad_node` (arena*, node void*, order i32, out tagged*) -> void, lifting a reverse-tape AD node into a dual-tower constant. */
llvm::Function* getTaylorLiftAdNodeFunc(CodegenContext& ctx) {
    if (auto* f = ctx.module().getFunction("eshkol_taylor_lift_ad_node")) return f;
    // void eshkol_taylor_lift_ad_node(arena*, void* node, i32 order, tagged* out)
    llvm::Type* p = ctx.ptrType();
    auto* ft = llvm::FunctionType::get(ctx.voidType(),
        {p, p, ctx.int32Type(), p}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                  "eshkol_taylor_lift_ad_node", &ctx.module());
}
/** @brief Get or declare `eshkol_taylor_coeffs_list` (arena*, tower tagged*, k i32, out tagged*) -> void, extracting the first K+1 Taylor coefficients as a Scheme list. */
llvm::Function* getTaylorCoeffsFunc(CodegenContext& ctx) {
    if (auto* f = ctx.module().getFunction("eshkol_taylor_coeffs_list")) return f;
    // void eshkol_taylor_coeffs_list(arena*, const tagged* tower, i32 k, tagged* out)
    llvm::Type* p = ctx.ptrType();
    auto* ft = llvm::FunctionType::get(ctx.voidType(),
        {p /*arena*/, p /*tower*/, ctx.int32Type(), p /*out*/}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                  "eshkol_taylor_coeffs_list", &ctx.module());
}
} // namespace

/**
 * @brief Seed a fresh forward-mode perturbation on `point_tagged` and push the runtime nesting level.
 *
 * In Taylor-tower mode (adTowerMode_ != NONE), seeds a heap tower of the
 * requested order via eshkol_taylor_seed_tagged and pushes the tower
 * differentiation context (towerCtxPush) instead of using the jet path.
 * Otherwise, increments the runtime perturbation-level counter and inserts a
 * unit tangent into the level-appropriate slot of the 8-jet (level 0 -> e1,
 * level 1 -> e2, level >= 2 -> ep), preserving any perturbation the incoming
 * point already carries so outer nesting survives.
 *
 * @param point_tagged the evaluation point as a tagged value (may already be a dual/tower).
 * @param out_level if non-null, receives the pre-push perturbation level (to restore later via popAndExtractForward).
 * @return the seeded point as a tagged dual number or tower, ready to pass into the differentiated closure.
 */
llvm::Value* AutodiffCodegen::seedForwardAndPush(llvm::Value* point_tagged,
                                                 llvm::Value** out_level) {
    auto& b = ctx_.builder();
    llvm::Value* level = adPertLevelLoad();
    if (out_level) *out_level = level;

    // ── Taylor-tower mode (ESH-0186): seed a heap tower {x0,1,0,...} of the
    // requested order under a fresh perturbation epoch instead of a jet. The
    // jet path below is untouched when adTowerMode_ == NONE (order <= 2). ──
    if (adTowerMode_ != TowerMode::NONE && adTowerOrder_) {
        llvm::Value* point_slot = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "twr_point");
        llvm::Value* out_slot   = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "twr_seed_out");
        ctx_.builder().CreateStore(point_tagged, point_slot);
        ctx_.builder().CreateCall(getTaylorSeedFunc(ctx_),
            {getArenaPtr(), point_slot, adTowerOrder_, out_slot});
        // ESH-0190 (P5): mark the tower context active so a reverse-tape AD node
        // captured by the differentiated lambda is frozen to a dual-tower rather
        // than swallowed by the reverse tape. Popped in popAndExtractForward.
        towerCtxPush(adTowerOrder_);
        return ctx_.builder().CreateLoad(ctx_.taggedValueType(), out_slot, "twr_seeded");
    }

    // Push: the callee body runs one perturbation level deeper.
    adPertLevelStore(b.CreateAdd(level, llvm::ConstantInt::get(ctx_.int64Type(), 1)));

    // ESH-0117: seed slot by level — level 0 -> e1 (field 1), level 1 -> e2
    // (field 2), level 2 -> ep (field 4). The ep slot is the THIRD independent
    // nilpotent perturbation carried by the 8-jet, so THREE simultaneous
    // forward differentiations nest exactly (gradient-over-derivative-of-
    // derivative). We preserve the components the point already carries (so the
    // outer perturbations survive). level >= 3 aliases onto ep (approximate —
    // beyond the 3 perturbations the 8-jet can represent).
    llvm::Value* base = safeUnpackDualFromTagged(point_tagged); // 8-jet
    llvm::Value* is_l0 = b.CreateICmpEQ(level, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* is_l1 = b.CreateICmpEQ(level, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

    llvm::Function* fn = b.GetInsertBlock()->getParent();
    llvm::BasicBlock* l0bb  = llvm::BasicBlock::Create(ctx_.context(), "fwd_seed_e1", fn);
    llvm::BasicBlock* l1bb  = llvm::BasicBlock::Create(ctx_.context(), "fwd_seed_e2", fn);
    llvm::BasicBlock* lelse = llvm::BasicBlock::Create(ctx_.context(), "fwd_seed_ep_chk", fn);
    llvm::BasicBlock* l2bb  = llvm::BasicBlock::Create(ctx_.context(), "fwd_seed_ep", fn);
    llvm::BasicBlock* mbb   = llvm::BasicBlock::Create(ctx_.context(), "fwd_seed_merge", fn);
    b.CreateCondBr(is_l0, l0bb, lelse);

    b.SetInsertPoint(l0bb);
    llvm::Value* s0 = b.CreateInsertValue(base, one, {kJetE1});
    b.CreateBr(mbb);
    llvm::BasicBlock* l0e = b.GetInsertBlock();

    b.SetInsertPoint(lelse);
    b.CreateCondBr(is_l1, l1bb, l2bb);

    b.SetInsertPoint(l1bb);
    llvm::Value* s1 = b.CreateInsertValue(base, one, {kJetE2});
    b.CreateBr(mbb);
    llvm::BasicBlock* l1e = b.GetInsertBlock();

    b.SetInsertPoint(l2bb);
    llvm::Value* s2 = b.CreateInsertValue(base, one, {kJetEp});  // reverse-seed companion slot
    b.CreateBr(mbb);
    llvm::BasicBlock* l2e = b.GetInsertBlock();

    b.SetInsertPoint(mbb);
    llvm::PHINode* seeded = b.CreatePHI(ctx_.dualNumberType(), 3, "fwd_seeded");
    seeded->addIncoming(s0, l0e);
    seeded->addIncoming(s1, l1e);
    seeded->addIncoming(s2, l2e);
    return packDualToTagged(seeded);
}

/**
 * @brief Pop the runtime perturbation level and extract this level's derivative from a forward-mode result.
 *
 * Restores the outer perturbation level. In Taylor-tower mode, pops the tower
 * context and extracts either the k-th derivative (DERIV_N, threading any
 * reverse-seed tangent back onto the tape via eshkol_ad_mixed_record) or the
 * full coefficient list (COEFFS). Otherwise extracts from the 8-jet: detects
 * a Scheme-vector result (R -> R^n derivative, differentiating each element)
 * versus a scalar result, and — for the scalar case — reads the coefficient
 * for THIS level's perturbation slot (e1 at depth 0, recording an exact local
 * linearization on the outer reverse tape when one is live; a dual slice
 * carrying the remaining perturbations when nested).
 *
 * @param result_tagged the differentiated closure's raw tagged return value.
 * @param level the perturbation level saved by seedForwardAndPush, to restore.
 * @return the extracted derivative (or vector of derivatives), tagged.
 */
llvm::Value* AutodiffCodegen::popAndExtractForward(llvm::Value* result_tagged,
                                                   llvm::Value* level) {
    auto& b = ctx_.builder();
    // Pop: restore the level the outer context expects.
    adPertLevelStore(level);

    // ── Taylor-tower mode (ESH-0186): extract from the heap tower returned by
    // the differentiated function. DERIV_N -> f^(k)=k!*c[k] (a double); COEFFS
    // -> the K+1-element coefficient list. Short-circuits the jet R->Rⁿ logic. ──
    if (adTowerMode_ == TowerMode::DERIV_N && adTowerOrder_) {
        towerCtxPop();     // leave the tower differentiation context
        llvm::Value* res_slot = b.CreateAlloca(ctx_.taggedValueType(), nullptr, "twr_res");
        llvm::Value* out_slot = b.CreateAlloca(ctx_.taggedValueType(), nullptr, "twr_extract_out");
        b.CreateStore(result_tagged, res_slot);
        llvm::Value* d = b.CreateCall(getTaylorExtractFunc(ctx_), {res_slot, adTowerOrder_});
        // ── ESH-0190 (P5) reverse-over-Taylor: if the returned tower carries a
        // seed tangent (an outer gradient's seed flowed into f^(k)), extract
        // dseed = d(f^(k))/d(seed) and thread it back to the outer pass:
        //   • reverse tape live  → record an exact local linearization via the
        //     SAME eshkol_ad_mixed_record hook used by the order-≤2 jet path;
        //   • forward (no tape)  → return a jet {value, dseed·e1} the outer
        //     forward extraction reads. ──
        llvm::Value* has_tan = b.CreateCall(getTaylorHasTangentFunc(ctx_), {res_slot});
        llvm::Value* has_tan_b = b.CreateICmpNE(has_tan, llvm::ConstantInt::get(ctx_.int32Type(), 0));
        llvm::Function* fn = b.GetInsertBlock()->getParent();
        llvm::BasicBlock* tan_bb   = llvm::BasicBlock::Create(ctx_.context(), "twr_tan", fn);
        llvm::BasicBlock* plain_bb = llvm::BasicBlock::Create(ctx_.context(), "twr_plain", fn);
        llvm::BasicBlock* done_bb  = llvm::BasicBlock::Create(ctx_.context(), "twr_done", fn);
        b.CreateCondBr(has_tan_b, tan_bb, plain_bb);

        // plain: no seed dependence → the bare k-th derivative.
        // ESH-0191 (P6): extract through the tagged out-param so an EXACT
        // tower's f^(n)=n!*c[n] stays an exact int64/bignum/rational tagged
        // value (`(exact? (derivative-n f x n))` => #t) instead of being
        // forced to double; COEFF_F64 towers are unaffected (still a tagged
        // double, same numeric result as before).
        b.SetInsertPoint(plain_bb);
        b.CreateCall(getTaylorExtractTaggedFunc(ctx_), {getArenaPtr(), res_slot, adTowerOrder_, out_slot});
        llvm::Value* plain_res = b.CreateLoad(ctx_.taggedValueType(), out_slot, "twr_extracted");
        b.CreateBr(done_bb);
        llvm::BasicBlock* plain_exit = b.GetInsertBlock();

        // tangent: dseed = k! * tangent[k].
        b.SetInsertPoint(tan_bb);
        llvm::Value* dseed = b.CreateCall(getTaylorExtractTangentFunc(ctx_), {res_slot, adTowerOrder_});
        llvm::Value* tan_res = nullptr;
        if (ctx_.currentAdTape()) {
            llvm::FunctionCallee mixed_rec = ctx_.module().getOrInsertFunction(
                "eshkol_ad_mixed_record",
                llvm::FunctionType::get(ctx_.ptrType(),
                    {ctx_.ptrType(), ctx_.ptrType(), ctx_.doubleType(), ctx_.doubleType()}, false));
            llvm::Value* arena_ptr = getArenaPtr();
            llvm::Value* tape_ptr = b.CreateLoad(ctx_.ptrType(), ctx_.currentAdTape());
            llvm::Value* node = b.CreateCall(mixed_rec, {arena_ptr, tape_ptr, d, dseed});
            llvm::Value* node_ok = b.CreateICmpNE(node,
                llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx_.context())));
            llvm::BasicBlock* rec_bb = llvm::BasicBlock::Create(ctx_.context(), "twr_rec", fn);
            llvm::BasicBlock* jet_bb = llvm::BasicBlock::Create(ctx_.context(), "twr_jet", fn);
            llvm::BasicBlock* tmrg   = llvm::BasicBlock::Create(ctx_.context(), "twr_tmrg", fn);
            b.CreateCondBr(node_ok, rec_bb, jet_bb);
            b.SetInsertPoint(rec_bb);
            llvm::Value* rec_v = tagged_.packPtr(node, ESHKOL_VALUE_CALLABLE);
            b.CreateBr(tmrg);
            llvm::BasicBlock* rec_exit = b.GetInsertBlock();
            b.SetInsertPoint(jet_bb);
            llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
            llvm::Value* jet_v = packDualToTagged(makeDual8(ctx_, d, dseed, zero, zero, zero, zero, zero, zero));
            b.CreateBr(tmrg);
            llvm::BasicBlock* jet_exit = b.GetInsertBlock();
            b.SetInsertPoint(tmrg);
            llvm::PHINode* tsel = b.CreatePHI(ctx_.taggedValueType(), 2, "twr_tan_sel");
            tsel->addIncoming(rec_v, rec_exit);
            tsel->addIncoming(jet_v, jet_exit);
            tan_res = tsel;
        } else {
            llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
            tan_res = packDualToTagged(makeDual8(ctx_, d, dseed, zero, zero, zero, zero, zero, zero));
        }
        b.CreateBr(done_bb);
        llvm::BasicBlock* tan_exit = b.GetInsertBlock();

        b.SetInsertPoint(done_bb);
        llvm::PHINode* dres = b.CreatePHI(ctx_.taggedValueType(), 2, "twr_deriv_res");
        dres->addIncoming(plain_res, plain_exit);
        dres->addIncoming(tan_res, tan_exit);
        return dres;
    }
    if (adTowerMode_ == TowerMode::COEFFS && adTowerOrder_) {
        towerCtxPop();     // leave the tower differentiation context
        llvm::Value* res_slot = b.CreateAlloca(ctx_.taggedValueType(), nullptr, "twr_res");
        llvm::Value* out_slot = b.CreateAlloca(ctx_.taggedValueType(), nullptr, "twr_list_out");
        b.CreateStore(result_tagged, res_slot);
        b.CreateCall(getTaylorCoeffsFunc(ctx_), {getArenaPtr(), res_slot, adTowerOrder_, out_slot});
        return b.CreateLoad(ctx_.taggedValueType(), out_slot, "twr_coeffs");
    }

    llvm::Function* fn = b.GetInsertBlock()->getParent();

    // ── R → Rⁿ: vector-valued derivative ────────────────────────────────
    // A function like (lambda (t) (vector (* t t) (* t t t))) returns a scheme
    // VECTOR whose elements are duals (each (* t …) carries its own tangent).
    // The scalar extraction below would read the vector POINTER as a double and
    // silently return 0. Detect a scheme-vector result and differentiate each
    // component, returning a vector of per-component derivatives. (A raw-double
    // tensor cannot carry the tangent, so only (vector ...) results are
    // supported here; a tensor result falls through to the scalar path.)
    llvm::AllocaInst* nd_slot;
    {
        llvm::IRBuilder<> eb(&fn->getEntryBlock(), fn->getEntryBlock().begin());
        nd_slot = eb.CreateAlloca(ctx_.taggedValueType(), nullptr, "fwd_ex_nd_slot");
    }
    llvm::Value* nd_base = tagged_.getBaseType(tagged_.getType(result_tagged));
    llvm::Value* nd_is_heap = b.CreateICmpEQ(nd_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    llvm::BasicBlock* nd_check  = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_nd_check", fn);
    llvm::BasicBlock* nd_vec    = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_nd_vec", fn);
    llvm::BasicBlock* nd_scalar = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_nd_scalar", fn);
    llvm::BasicBlock* nd_done   = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_nd_done", fn);
    b.CreateCondBr(nd_is_heap, nd_check, nd_scalar);

    // Is the heap object a scheme vector (subtype VECTOR)?
    b.SetInsertPoint(nd_check);
    llvm::Value* nd_ptr = tagged_.unpackPtr(result_tagged);
    llvm::Value* nd_hdr = b.CreateGEP(ctx_.int8Type(), nd_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), -8));
    llvm::Value* nd_sub = b.CreateLoad(ctx_.int8Type(), nd_hdr);
    llvm::Value* nd_is_vec = b.CreateICmpEQ(nd_sub,
        llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    b.CreateCondBr(nd_is_vec, nd_vec, nd_scalar);

    // Vector path: build a scheme vector of per-component derivatives.
    b.SetInsertPoint(nd_vec);
    {
        llvm::Value* nd_is_l0 = b.CreateICmpEQ(level, llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* nd_is_l1 = b.CreateICmpEQ(level, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        llvm::Value* nd_zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* nd_len = b.CreateLoad(ctx_.int64Type(), nd_ptr);
        llvm::Value* nd_src8 = b.CreateGEP(ctx_.int8Type(), nd_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), 8));
        llvm::Value* nd_src = b.CreatePointerCast(nd_src8, ctx_.ptrType());
        llvm::Value* nd_arena = getArenaPtr();
        llvm::Value* nd_out = b.CreateCall(mem_.getArenaAllocateVectorWithHeader(), {nd_arena, nd_len});
        b.CreateStore(nd_len, nd_out);
        llvm::Value* nd_out8 = b.CreateGEP(ctx_.int8Type(), nd_out,
            llvm::ConstantInt::get(ctx_.int64Type(), 8));
        llvm::Value* nd_out_elems = b.CreatePointerCast(nd_out8, ctx_.ptrType());

        llvm::BasicBlock* nd_cond = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_nd_cond", fn);
        llvm::BasicBlock* nd_body = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_nd_body", fn);
        llvm::BasicBlock* nd_end  = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_nd_end", fn);
        llvm::AllocaInst* nd_i;
        {
            llvm::IRBuilder<> eb(&fn->getEntryBlock(), fn->getEntryBlock().begin());
            nd_i = eb.CreateAlloca(ctx_.int64Type(), nullptr, "fwd_ex_nd_i");
        }
        b.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), nd_i);
        b.CreateBr(nd_cond);

        b.SetInsertPoint(nd_cond);
        llvm::Value* nd_iv = b.CreateLoad(ctx_.int64Type(), nd_i);
        b.CreateCondBr(b.CreateICmpULT(nd_iv, nd_len), nd_body, nd_end);

        b.SetInsertPoint(nd_body);
        llvm::Value* nd_elem = b.CreateLoad(ctx_.taggedValueType(),
            b.CreateGEP(ctx_.taggedValueType(), nd_src, nd_iv));
        llvm::Value* nd_ed = safeUnpackDualFromTagged(nd_elem); // 8-jet
        // level 0 → scalar derivative a1; nested → dual slice carrying the
        // ep-derivative (ESH-0117): value {a2,a12,0,0}, ep-deriv {ap2,ap12,0,0}
        // where ap* are the incoming ep-derivative 4-jet (fields 6,7).
        llvm::Value* nd_scalar_elem = tagged_.packDouble(dualField(ctx_, nd_ed, 1));
        llvm::Value* nd_slice_elem = packDualToTagged(makeDual8(ctx_,
            dualField(ctx_, nd_ed, 2), dualField(ctx_, nd_ed, 3), nd_zero, nd_zero,
            dualField(ctx_, nd_ed, 6), dualField(ctx_, nd_ed, 7), nd_zero, nd_zero));
        // level>=2: d/dep slice (value 4-jet = fields 4..7).
        llvm::Value* nd_slice_ep_elem = packDualToTagged(makeDual8(ctx_,
            dualField(ctx_, nd_ed, 4), dualField(ctx_, nd_ed, 5),
            dualField(ctx_, nd_ed, 6), dualField(ctx_, nd_ed, 7),
            nd_zero, nd_zero, nd_zero, nd_zero));
        llvm::Value* nd_elem_out = b.CreateSelect(nd_is_l0, nd_scalar_elem,
            b.CreateSelect(nd_is_l1, nd_slice_elem, nd_slice_ep_elem));
        b.CreateStore(nd_elem_out, b.CreateGEP(ctx_.taggedValueType(), nd_out_elems, nd_iv));
        b.CreateStore(b.CreateAdd(nd_iv, llvm::ConstantInt::get(ctx_.int64Type(), 1)), nd_i);
        b.CreateBr(nd_cond);

        b.SetInsertPoint(nd_end);
        llvm::Value* nd_out_int = b.CreatePtrToInt(nd_out, ctx_.int64Type());
        b.CreateStore(tagged_.packPtr(nd_out_int, ESHKOL_VALUE_HEAP_PTR), nd_slot);
        b.CreateBr(nd_done);
    }

    // Scalar path: original extraction.
    b.SetInsertPoint(nd_scalar);

    llvm::Value* rd = safeUnpackDualFromTagged(result_tagged); // 8-jet
    llvm::Value* is_l0 = b.CreateICmpEQ(level, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* is_l1 = b.CreateICmpEQ(level, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    // ESH-0117: extract w.r.t. THIS level's perturbation slot:
    //   level 0 -> d/de1 (scalar, + reverse-tape mixed_record if live)
    //   level 1 -> d/de2 (dual slice in remaining e1/ep perturbations)
    //   level>=2 -> d/dep (dual slice in remaining e1/e2 perturbations)
    llvm::BasicBlock* x0bb  = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_scalar", fn);
    llvm::BasicBlock* xelse = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_nz_chk", fn);
    llvm::BasicBlock* x1bb  = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_slice_e2", fn);
    llvm::BasicBlock* x2bb  = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_slice_ep", fn);
    llvm::BasicBlock* mbb   = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_merge", fn);
    b.CreateCondBr(is_l0, x0bb, xelse);

    // level 0: scalar derivative = tangent (e1 component), packed as a double.
    //
    // ESH-0093: when this derivative ran INSIDE a reverse-mode gradient pass
    // (an outer tape is live and the pass published an active seed variable),
    // captured tape nodes were jet-lifted with e2 = 1 on the seed
    // (maybeJetLiftTapeOperand), so the e1e2 coefficient a12 is
    // d(derivative-result)/d(seed). Record the result on the outer tape as an
    // exact local linearization (eshkol_ad_mixed_record) and return that AD
    // node so the outer backward pass sees the dependency. When there is no
    // tape / no seed / no dependency the helper returns null and the plain
    // scalar path below is used — identical to the previous behavior.
    b.SetInsertPoint(x0bb);
    llvm::Value* a1 = dualField(ctx_, rd, 1);
    // ESH-0117: the reverse-seed dependence of this derivative's value is the
    // e1·ep coefficient = d(d/de1)/dep, i.e. the ep-derivative jet's e1 slot
    // (field 5) — NOT the forward mixed term e1·e2 (field 3). The ep-derivative
    // dimension carries the dependence through ANY forward nesting depth, so
    // this is correct for both single-forward (gofd) and 2-level nested
    // (gofdofd) inners; the old e2-seed hack could only reach depth 1.
    llvm::Value* a12 = dualField(ctx_, rd, 5);
    llvm::Value* scalar_res = tagged_.packDouble(a1);
    if (ctx_.currentAdTape()) {
        llvm::FunctionCallee mixed_rec = ctx_.module().getOrInsertFunction(
            "eshkol_ad_mixed_record",
            llvm::FunctionType::get(ctx_.ptrType(),
                {ctx_.ptrType(), ctx_.ptrType(), ctx_.doubleType(), ctx_.doubleType()},
                false));
        llvm::Value* arena_ptr = getArenaPtr();
        llvm::Value* tape_ptr = b.CreateLoad(ctx_.ptrType(), ctx_.currentAdTape());
        llvm::Value* rec_node = b.CreateCall(mixed_rec, {arena_ptr, tape_ptr, a1, a12});
        llvm::Value* rec_valid = b.CreateICmpNE(rec_node,
            llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx_.context())));
        llvm::BasicBlock* rec_bb = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_mixed_rec", fn);
        llvm::BasicBlock* x0m = llvm::BasicBlock::Create(ctx_.context(), "fwd_ex_scalar_merge", fn);
        llvm::BasicBlock* plain_exit = b.GetInsertBlock();
        b.CreateCondBr(rec_valid, rec_bb, x0m);

        b.SetInsertPoint(rec_bb);
        llvm::Value* node_res = tagged_.packPtr(rec_node, ESHKOL_VALUE_CALLABLE);
        llvm::BasicBlock* rec_exit = b.GetInsertBlock();
        b.CreateBr(x0m);

        b.SetInsertPoint(x0m);
        llvm::PHINode* sel = b.CreatePHI(ctx_.taggedValueType(), 2, "fwd_ex_scalar_sel");
        sel->addIncoming(scalar_res, plain_exit);
        sel->addIncoming(node_res, rec_exit);
        scalar_res = sel;
    }
    b.CreateBr(mbb);
    llvm::BasicBlock* x0e = b.GetInsertBlock();

    // nested: d/de2 (...) = a2 + a12 e1, returned as a dual so an enclosing
    // derivative/gradient reads its e1 coefficient (= the mixed 2nd-order term).
    // ESH-0117: also project the ep-derivative jet through d/de2 so the enclosing
    // (outer) derivative still sees the reverse-seed dependence: value slice
    // {a2,a12,0,0}, ep-deriv slice {d(a2)/dep, d(a12)/dep, 0, 0} = fields {6,7}.
    // For gofdofd the outer then reads this slice's field-5 (= inner field 7,
    // the e1·e2·ep triple coefficient) as its dseed — the second-order tape link.
    b.SetInsertPoint(xelse);
    b.CreateCondBr(is_l1, x1bb, x2bb);

    b.SetInsertPoint(x1bb);
    llvm::Value* slice = packDualToTagged(
        makeDual8(ctx_, dualField(ctx_, rd, 2), dualField(ctx_, rd, 3), zero, zero,
                  dualField(ctx_, rd, 6), dualField(ctx_, rd, 7), zero, zero));
    b.CreateBr(mbb);
    llvm::BasicBlock* x1e = b.GetInsertBlock();

    // level >= 2: d/dep. In the 8-jet {1,e1,e2,ep,e1e2,e1ep,e2ep,e1e2ep}
    // (fields 0..7), d/dep maps each ep-containing monomial to its ep-free part:
    //   ep(f4)->1, e1ep(f5)->e1, e2ep(f6)->e2, e1e2ep(f7)->e1e2.
    // So the remaining slice's value 4-jet {1,e1,e2,e1e2} = fields {4,5,6,7}.
    // The outer forward level then reads its e1/e2 coefficients as usual.
    b.SetInsertPoint(x2bb);
    llvm::Value* slice_ep = packDualToTagged(
        makeDual8(ctx_, dualField(ctx_, rd, 4), dualField(ctx_, rd, 5),
                  dualField(ctx_, rd, 6), dualField(ctx_, rd, 7),
                  zero, zero, zero, zero));
    b.CreateBr(mbb);
    llvm::BasicBlock* x2e = b.GetInsertBlock();

    b.SetInsertPoint(mbb);
    llvm::PHINode* res = b.CreatePHI(ctx_.taggedValueType(), 3, "fwd_extracted");
    res->addIncoming(scalar_res, x0e);
    res->addIncoming(slice, x1e);
    res->addIncoming(slice_ep, x2e);
    b.CreateStore(res, nd_slot);
    b.CreateBr(nd_done);

    // Final merge of the scalar and vector (R→Rⁿ) extraction paths.
    b.SetInsertPoint(nd_done);
    return b.CreateLoad(ctx_.taggedValueType(), nd_slot);
}

// ===== ESH-0093: jet-lift reverse-tape operands inside forward-mode AD =====
// While a forward-mode derivative/gradient pass is live (__ad_pert_level > 0),
// a reverse-tape AD node reaching scalar arithmetic must NOT be recorded on
// the tape (that path drops the forward tangent — the mixed-mode bug).
// Instead it is frozen to its value as a dual number, and — when it is the
// gradient pass's published active seed variable (eshkol_ad_seed_flag) — its
// ep-DERIVATIVE slot (field 4, d(value)/dep) is seeded with 1.0.
//
// ESH-0117: the ep-derivative is a SEPARATE jet dimension from the two forward
// perturbation slots e1/e2, so it survives arbitrary forward nesting depth.
// (The previous mechanism seeded the e2 slot, which collided with the inner
// forward perturbation at 2-level nesting — gradient over derivative-of-
// derivative — and had to be masked off at level != 1, losing the dependency.)
// The 8-jet now carries d(result 4-jet)/d(seed) through any depth; the outer
// derivative's popAndExtractForward reads the surviving e1·ep coefficient and
// records the dependency back onto the tape (eshkol_ad_mixed_record).
// ESH-0190 (P5): freeze a reverse-tape AD node into a dual-tower constant while
// a Taylor-tower differentiation is active, so the tower kernel (not the reverse
// tape) consumes it. The lifted constant carries value {node->value,0,…} and
// seed tangent {seed_flag,0,…}; its first-order dependence on the active
// gradient seed then propagates through the tower recurrences and is read back
// at popAndExtractForward via eshkol_ad_mixed_record. No-op unless a tower pass
// is live AND the operand is a reverse-tape AD node.
/**
 * @brief Freeze a reverse-tape AD node into a dual-tower constant while a Taylor-tower differentiation is active (ESH-0190/P5).
 *
 * No-op unless the tower-active depth counter is > 0 AND the operand is a
 * CALLABLE tagged value that is actually an AD-node (subtype-checked before
 * dereferencing). When both hold, calls eshkol_taylor_lift_ad_node so the
 * tower kernel's recurrences (not the reverse tape) consume the operand.
 *
 * @param operand_tagged candidate operand tagged value.
 * @return the operand unchanged, or the lifted dual-tower tagged value.
 */
llvm::Value* AutodiffCodegen::towerLiftOperand(llvm::Value* operand_tagged) {
    if (!operand_tagged || operand_tagged->getType() != ctx_.taggedValueType())
        return operand_tagged;
    llvm::GlobalVariable* active_g = ctx_.adTowerActive();
    llvm::GlobalVariable* order_g  = ctx_.adTowerOrder();
    if (!active_g || !order_g) return operand_tagged;
    auto& b = ctx_.builder();
    llvm::Function* fn = b.GetInsertBlock()->getParent();

    // result slot (defaults to the unchanged operand).
    llvm::AllocaInst* slot;
    llvm::AllocaInst* out_slot;
    {
        llvm::IRBuilder<> eb(&fn->getEntryBlock(), fn->getEntryBlock().begin());
        slot = eb.CreateAlloca(ctx_.taggedValueType(), nullptr, "twrlift_slot");
        out_slot = eb.CreateAlloca(ctx_.taggedValueType(), nullptr, "twrlift_out");
    }
    b.CreateStore(operand_tagged, slot);

    llvm::Value* depth = b.CreateLoad(ctx_.int64Type(), active_g, "twr_active");
    llvm::Value* twr_on = b.CreateICmpSGT(depth, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* base = tagged_.getBaseType(tagged_.getType(operand_tagged));
    llvm::Value* is_callable = b.CreateICmpEQ(base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    llvm::Value* cand = b.CreateAnd(twr_on, is_callable);

    llvm::BasicBlock* sub_bb  = llvm::BasicBlock::Create(ctx_.context(), "twrlift_sub", fn);
    llvm::BasicBlock* lift_bb = llvm::BasicBlock::Create(ctx_.context(), "twrlift_lift", fn);
    llvm::BasicBlock* done_bb = llvm::BasicBlock::Create(ctx_.context(), "twrlift_done", fn);
    b.CreateCondBr(cand, sub_bb, done_bb);

    // CALLABLE: confirm it is an AD node (subtype check dereferences the header).
    b.SetInsertPoint(sub_bb);
    llvm::Value* is_ad = tagged_.checkCallableSubtype(operand_tagged, CALLABLE_SUBTYPE_AD_NODE);
    b.CreateCondBr(is_ad, lift_bb, done_bb);

    // Lift via the runtime helper (reads node->value + seed_flag).
    b.SetInsertPoint(lift_bb);
    llvm::Value* node_ptr = tagged_.unpackPtr(operand_tagged);
    llvm::Value* order32 = b.CreateIntCast(
        b.CreateLoad(ctx_.int64Type(), order_g, "twr_order"), ctx_.int32Type(), true);
    b.CreateCall(getTaylorLiftAdNodeFunc(ctx_), {getArenaPtr(), node_ptr, order32, out_slot});
    b.CreateStore(b.CreateLoad(ctx_.taggedValueType(), out_slot), slot);
    b.CreateBr(done_bb);

    b.SetInsertPoint(done_bb);
    return b.CreateLoad(ctx_.taggedValueType(), slot, "twrlift_result");
}

/**
 * @brief Freeze a reverse-tape AD node into a forward-mode dual jet while a forward derivative/gradient pass is live (ESH-0093).
 *
 * First delegates to towerLiftOperand (Taylor-tower mode takes priority).
 * Otherwise, gated on the runtime perturbation level being > 0 and the
 * operand being an AD-node CALLABLE: lifts it to value 4-jet
 * {node->value,0,0,0} with ep-derivative 4-jet {seed_flag,0,0,0}, where
 * seed_flag is 1.0 iff this node is the active reverse-mode seed
 * (eshkol_ad_seed_flag). The ep slot is an independent jet dimension so the
 * dependency survives arbitrary forward nesting depth (ESH-0117), unlike the
 * historical e2-slot hack it replaced.
 *
 * @param operand_tagged candidate operand tagged value.
 * @return the operand unchanged, or a lifted dual-number tagged value.
 */
llvm::Value* AutodiffCodegen::maybeJetLiftTapeOperand(llvm::Value* operand_tagged) {
    if (!operand_tagged || operand_tagged->getType() != ctx_.taggedValueType())
        return operand_tagged;
    // ESH-0190 (P5): reverse-over-Taylor. When a tower differentiation is live,
    // freeze a reverse-tape AD node into a dual-tower FIRST, so the tower
    // arithmetic consumes it instead of the reverse tape swallowing the tower.
    // If it lifts, the operand is now a HEAP_PTR tower and the jet gate below is
    // a no-op; if not, the operand is unchanged and the jet path runs as before.
    operand_tagged = towerLiftOperand(operand_tagged);
    auto& b = ctx_.builder();

    // Cheap gate first: no live forward perturbation → no lifting (this is the
    // path all non-AD code takes; a single global load + compare).
    llvm::Value* level = adPertLevelLoad();
    llvm::Value* pert_active = b.CreateICmpSGT(level,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* base = tagged_.getBaseType(tagged_.getType(operand_tagged));
    llvm::Value* is_callable = b.CreateICmpEQ(base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    llvm::Value* candidate = b.CreateAnd(pert_active, is_callable);

    llvm::Function* fn = b.GetInsertBlock()->getParent();
    llvm::BasicBlock* check_bb = llvm::BasicBlock::Create(ctx_.context(), "jetlift_check", fn);
    llvm::BasicBlock* lift_bb = llvm::BasicBlock::Create(ctx_.context(), "jetlift_lift", fn);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "jetlift_merge", fn);
    llvm::BasicBlock* entry_exit = b.GetInsertBlock();
    b.CreateCondBr(candidate, check_bb, merge_bb);

    // CALLABLE: confirm it is actually an AD node (subtype check dereferences
    // the header, so it must be behind the CALLABLE branch).
    b.SetInsertPoint(check_bb);
    llvm::Value* is_ad = tagged_.checkCallableSubtype(operand_tagged, CALLABLE_SUBTYPE_AD_NODE);
    llvm::BasicBlock* check_exit = b.GetInsertBlock(); // subtype check adds blocks
    b.CreateCondBr(is_ad, lift_bb, merge_bb);

    // Lift: value 4-jet = {node->value, 0, 0, 0}; ep-derivative 4-jet =
    // {seed_flag, 0, 0, 0} — i.e. field 4 (d(value)/dep) = 1.0 iff this node IS
    // the active reverse seed. seed_flag comes from the runtime (thread-local
    // __ad_active_seed_node) so it works identically in JIT and AOT, and it is
    // NOT masked by pert level: the ep-derivative dimension is independent of
    // the forward perturbation slots and survives any nesting depth (ESH-0117).
    b.SetInsertPoint(lift_bb);
    llvm::Value* node_ptr = tagged_.unpackPtr(operand_tagged);
    llvm::Value* node_val = loadNodeValue(node_ptr);
    llvm::FunctionCallee seed_flag_fn = ctx_.module().getOrInsertFunction(
        "eshkol_ad_seed_flag",
        llvm::FunctionType::get(ctx_.doubleType(), {ctx_.ptrType()}, false));
    llvm::Value* raw_flag = b.CreateCall(seed_flag_fn, {node_ptr});
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* lifted = packDualToTagged(
        makeDual8(ctx_, node_val, zero, zero, zero, raw_flag, zero, zero, zero));
    llvm::BasicBlock* lift_exit = b.GetInsertBlock(); // packDualToTagged may add blocks
    b.CreateBr(merge_bb);

    b.SetInsertPoint(merge_bb);
    llvm::PHINode* phi = b.CreatePHI(ctx_.taggedValueType(), 3, "jetlift_result");
    phi->addIncoming(operand_tagged, entry_exit);
    phi->addIncoming(operand_tagged, check_exit);
    phi->addIncoming(lifted, lift_exit);
    return phi;
}

/** @brief Extract field 0 (primal value) of a dual-number struct via a temporary alloca + StructGEP load. */
llvm::Value* AutodiffCodegen::getDualPrimal(llvm::Value* dual) {
    if (!dual) return llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    // Store dual to temporary alloca
    llvm::Value* dual_ptr = ctx_.builder().CreateAlloca(ctx_.dualNumberType(), nullptr, "temp_dual");
    ctx_.builder().CreateStore(dual, dual_ptr);

    // Extract primal (field 0)
    llvm::Value* primal_ptr = ctx_.builder().CreateStructGEP(ctx_.dualNumberType(), dual_ptr, 0);
    return ctx_.builder().CreateLoad(ctx_.doubleType(), primal_ptr);
}

/** @brief Extract field 1 (e1 tangent) of a dual-number struct via a temporary alloca + StructGEP load. */
llvm::Value* AutodiffCodegen::getDualTangent(llvm::Value* dual) {
    if (!dual) return llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    // Store dual to temporary alloca
    llvm::Value* dual_ptr = ctx_.builder().CreateAlloca(ctx_.dualNumberType(), nullptr, "temp_dual");
    ctx_.builder().CreateStore(dual, dual_ptr);

    // Extract tangent (field 1)
    llvm::Value* tangent_ptr = ctx_.builder().CreateStructGEP(ctx_.dualNumberType(), dual_ptr, 1);
    return ctx_.builder().CreateLoad(ctx_.doubleType(), tangent_ptr);
}

/**
 * @brief Heap-allocate an 8-field dual number in the global arena and pack it as a DUAL_NUMBER tagged value.
 *
 * Allocates 64 bytes (eight doubles: value 4-jet + ep-derivative 4-jet,
 * ESH-0117), stores the dual struct, and packs the resulting pointer with
 * type tag ESHKOL_VALUE_DUAL_NUMBER.
 *
 * @param dual the dual-number LLVM struct value to store.
 * @return a tagged pointer value, or a packed null if the global arena / allocator is unavailable.
 */
llvm::Value* AutodiffCodegen::packDualToTagged(llvm::Value* dual) {
    if (!dual) return nullptr;

    // Get global arena pointer
    llvm::GlobalVariable* arena_global = ctx_.module().getNamedGlobal("__global_arena");
    if (!arena_global) {
        eshkol_warn("packDualToTagged: __global_arena not found");
        return tagged_.packNull();
    }

    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global);

    // Allocate space for dual number on the heap (arena)
    // ESH-0117: dual_number is 64 bytes (eight doubles: value 4-jet
    // {primal,d1,d2,d12} + ep-derivative 4-jet {dp,dp1,dp2,dp12}).
    llvm::Value* size = llvm::ConstantInt::get(ctx_.sizeType(), 64);
    llvm::Function* alloc_func = mem_.getArenaAllocate();
    if (!alloc_func) {
        eshkol_warn("packDualToTagged: arena_allocate not found");
        return tagged_.packNull();
    }

    llvm::Value* dual_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr, size});

    // Store dual number to heap-allocated memory
    ctx_.builder().CreateStore(dual, dual_ptr);

    // Pack as pointer type tagged value with DUAL_NUMBER type
    return tagged_.packPtr(dual_ptr, ESHKOL_VALUE_DUAL_NUMBER);
}

/**
 * @brief Unpack a DUAL_NUMBER tagged value's pointer and load the dual struct — the hot straight-line path.
 *
 * Must not introduce basic blocks (callers on this path have already branched
 * on `arg_is_dual` and may set up their own PHIs assuming the builder stays
 * in the same block). Use safeUnpackDualFromTagged when the tagged value may
 * not actually be a dual.
 *
 * @param tagged_val a tagged value already known to carry the DUAL_NUMBER tag.
 * @return the loaded dual-number struct value.
 */
llvm::Value* AutodiffCodegen::unpackDualFromTagged(llvm::Value* tagged_val) {
    if (!tagged_val) return nullptr;

    // Extract pointer from tagged value.  This is the straight-line
    // hot path used by every dual arithmetic site (sin/cos/exp/log/…),
    // which only invokes this after already branching on
    // `arg_is_dual`.  We MUST NOT introduce basic blocks here — many
    // callers immediately set up their own PHIs and assume the builder
    // is still in the same block they were in before this call.
    //
    // For callers that may have a non-dual tagged value (e.g.
    // `derivative()`'s post-call extractor when the lambda body returns
    // a constant), use `safeUnpackDualFromTagged` below, which does the
    // dispatch as separate basic blocks.
    llvm::Value* ptr_val = tagged_.unpackPtr(tagged_val);

    // Load and return dual number
    return ctx_.builder().CreateLoad(ctx_.dualNumberType(), ptr_val);
}

// Safe variant: handles the case where `tagged_val` may not actually be
// a dual.  Returns a dual struct: the original dual if it is one,
// otherwise {primal, 0.0} synthesised from the scalar.
//
// Required for `derivative()` results: the lambda body may return a
// non-dual (constant function, predicate-driven branch with literal
// branches, etc.).  Without this, dereferencing a scalar's data field
// as a heap pointer crashes.
/**
 * @brief Unpack a tagged value as a dual number, synthesizing {primal, 0, 0, 0} (all-zero jet) when it is not one.
 *
 * Required for derivative()/gradient() results: the differentiated lambda
 * body may return a non-dual (constant function, literal branch, etc.), and
 * dereferencing a scalar's data field as a heap pointer would otherwise
 * crash. Branches at runtime on the base type tag; the scalar path converts
 * an int64/bool/char data field via SIToFP or bitcasts a double, per
 * base_type.
 *
 * @param tagged_val a tagged value that may or may not carry the DUAL_NUMBER tag.
 * @return the dual struct (loaded, or synthesized with only the primal field set).
 */
llvm::Value* AutodiffCodegen::safeUnpackDualFromTagged(llvm::Value* tagged_val) {
    if (!tagged_val) return nullptr;

    llvm::Value* type_tag = tagged_.getType(tagged_val);
    llvm::Value* base_type = tagged_.getBaseType(type_tag);
    llvm::Value* is_dual = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* dual_path = llvm::BasicBlock::Create(ctx_.context(), "sudft_dual", func);
    llvm::BasicBlock* scalar_path = llvm::BasicBlock::Create(ctx_.context(), "sudft_scalar", func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "sudft_merge", func);

    ctx_.builder().CreateCondBr(is_dual, dual_path, scalar_path);

    // Dual path: load the heap-allocated dual struct.
    ctx_.builder().SetInsertPoint(dual_path);
    llvm::Value* dual_ptr = tagged_.unpackPtr(tagged_val);
    llvm::Value* loaded_dual = ctx_.builder().CreateLoad(ctx_.dualNumberType(), dual_ptr, "sudft_loaded");
    llvm::BasicBlock* dual_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(merge_bb);

    // Scalar path: extract a primal double, build {primal, 0.0}.  Use
    // base_type to pick between bitcast (DOUBLE) and SIToFP (everything
    // else with a numeric data field — INT64, BOOL, CHAR).
    ctx_.builder().SetInsertPoint(scalar_path);
    llvm::Value* data_int = ctx_.builder().CreateExtractValue(tagged_val, {4}, "sudft_data");
    llvm::Value* primal_si = ctx_.builder().CreateSIToFP(data_int, ctx_.doubleType(), "sudft_primal_si");
    llvm::Value* primal_bc = ctx_.builder().CreateBitCast(data_int, ctx_.doubleType(), "sudft_primal_bc");
    llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* primal_final = ctx_.builder().CreateSelect(is_double, primal_bc, primal_si, "sudft_primal");
    // ESH-0117: start from a fully-zeroed 8-field dual so the e2/e1e2 and the
    // ep-derivative 4-jet (fields 4-7) are 0; only the primal is filled.
    llvm::Value* synth_dual = llvm::ConstantAggregateZero::get(ctx_.dualNumberType());
    synth_dual = ctx_.builder().CreateInsertValue(synth_dual, primal_final, {0});
    llvm::BasicBlock* scalar_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(merge_bb);

    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.dualNumberType(), 2, "sudft_result");
    phi->addIncoming(loaded_dual, dual_exit);
    phi->addIncoming(synth_dual, scalar_exit);
    return phi;
}

// Dual arithmetic (4-component jets). Each is the exact truncated-Taylor
// rule for v = f0 + f1 e1 + f2 e2 + f3 e1e2.

// (a + b): componentwise over all 8 jet components (value 4-jet + ep-deriv).
/** @brief Dual addition (a + b): componentwise FAdd over all 8 jet fields (value 4-jet + ep-derivative 4-jet). */
llvm::Value* AutodiffCodegen::dualAdd(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* r[8];
    for (unsigned i = 0; i < 8; ++i)
        r[i] = b.CreateFAdd(dualField(ctx_, dual_a, i), dualField(ctx_, dual_b, i));
    return makeDual8(ctx_, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}

// (a - b): componentwise over all 8 jet components.
/** @brief Dual subtraction (a - b): componentwise FSub over all 8 jet fields. */
llvm::Value* AutodiffCodegen::dualSub(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* r[8];
    for (unsigned i = 0; i < 8; ++i)
        r[i] = b.CreateFSub(dualField(ctx_, dual_a, i), dualField(ctx_, dual_b, i));
    return makeDual8(ctx_, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}

// (a * b): bilinear product rule, keeping the mixed e1e2 cross term.
//   r0 = a0 b0
//   r1 = a1 b0 + a0 b1
//   r2 = a2 b0 + a0 b2
//   r3 = a3 b0 + a1 b2 + a2 b1 + a0 b3
/**
 * @brief Dual multiplication (a * b): bilinear product rule over the value 4-jet, plus the ESH-0117 ep-derivative chain rule d/dep(a*b) = a'⊗b + a⊗b'.
 *
 * Uses jet4Mul to compute the exact bilinear product (keeping the mixed
 * e1e2 cross term) for both the value 4-jet and the ep-derivative 4-jet.
 *
 * @param dual_a left operand jet.
 * @param dual_b right operand jet.
 * @return the product jet, or nullptr if either operand is null.
 */
llvm::Value* AutodiffCodegen::dualMul(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;
    auto& b = ctx_.builder();
    // value 4-jets
    llvm::Value* a0 = dualField(ctx_, dual_a, 0), *a1 = dualField(ctx_, dual_a, 1),
                *a2 = dualField(ctx_, dual_a, 2), *a3 = dualField(ctx_, dual_a, 3);
    llvm::Value* b0 = dualField(ctx_, dual_b, 0), *b1 = dualField(ctx_, dual_b, 1),
                *b2 = dualField(ctx_, dual_b, 2), *b3 = dualField(ctx_, dual_b, 3);
    // ep-derivative 4-jets
    llvm::Value* ap0 = dualField(ctx_, dual_a, 4), *ap1 = dualField(ctx_, dual_a, 5),
                *ap2 = dualField(ctx_, dual_a, 6), *ap3 = dualField(ctx_, dual_a, 7);
    llvm::Value* bp0 = dualField(ctx_, dual_b, 4), *bp1 = dualField(ctx_, dual_b, 5),
                *bp2 = dualField(ctx_, dual_b, 6), *bp3 = dualField(ctx_, dual_b, 7);
    // value = a ⊗ b
    std::array<llvm::Value*, 4> v = jet4Mul(ctx_, a0, a1, a2, a3, b0, b1, b2, b3);
    // ESH-0117: d/dep (a·b) = a' ⊗ b + a ⊗ b'  (both 4-jet products)
    std::array<llvm::Value*, 4> t1 = jet4Mul(ctx_, ap0, ap1, ap2, ap3, b0, b1, b2, b3);
    std::array<llvm::Value*, 4> t2 = jet4Mul(ctx_, a0, a1, a2, a3, bp0, bp1, bp2, bp3);
    llvm::Value* p[4];
    for (int i = 0; i < 4; ++i) p[i] = b.CreateFAdd(t1[i], t2[i]);
    return makeDual8(ctx_, v[0], v[1], v[2], v[3], p[0], p[1], p[2], p[3]);
}

// (a / b) = a * (1/b). Reciprocal is the unary jet of g(x)=1/x with
//   g(b0)=1/b0, g'=-1/b0^2, g''=2/b0^3 — exact second order.
/**
 * @brief Dual division (a / b) computed as a * (1/b), where the reciprocal is the exact unary jet of g(x)=1/x (g'=-1/x^2, g''=2/x^3, g'''=-6/x^4).
 *
 * @param dual_a numerator jet.
 * @param dual_b denominator jet.
 * @return the quotient jet, or nullptr if either operand is null.
 */
llvm::Value* AutodiffCodegen::dualDiv(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;
    auto& bld = ctx_.builder();
    llvm::Value* b0 = dualField(ctx_, dual_b, 0);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* inv = bld.CreateFDiv(one, b0);
    llvm::Value* inv2 = bld.CreateFMul(inv, inv);
    llvm::Value* fpa = bld.CreateFNeg(inv2);                       // -1/b0^2
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* fppa = bld.CreateFMul(two, bld.CreateFMul(inv2, inv)); // 2/b0^3
    // g'''(x)=-6/x^4 for the ESH-0117 ep-derivative triple term.
    llvm::Value* six = llvm::ConstantFP::get(ctx_.doubleType(), 6.0);
    llvm::Value* fpppa = bld.CreateFNeg(bld.CreateFMul(six, bld.CreateFMul(inv2, inv2)));
    llvm::Value* recip = dualUnaryChain(ctx_, dual_b, inv, fpa, fppa, fpppa);
    return dualMul(dual_a, recip);
}

// ===== DUAL NUMBER MATH OPERATIONS =====
// These implement chain rule for various math functions

// Helper: Get or declare math function
/**
 * @brief Get or declare an external libm scalar math function by name (double args/return; pow/atan2 take two doubles).
 *
 * Checks the shared function_table_ first, then the module, before declaring
 * a new external `double name(double[, double])` and registering it.
 *
 * @param name libm function name (e.g. "sin", "pow").
 * @return the declared/found LLVM function.
 */
llvm::Function* AutodiffCodegen::getMathFunc(const std::string& name) {
    // Check function table first
    if (function_table_) {
        auto it = function_table_->find(name);
        if (it != function_table_->end()) {
            return it->second;
        }
    }

    // Check if already declared in module
    llvm::Function* func = ctx_.module().getFunction(name);
    if (func) return func;

    // Declare the function
    std::vector<llvm::Type*> args = {ctx_.doubleType()};
    // pow and atan2 take 2 args
    if (name == "pow" || name == "atan2") {
        args.push_back(ctx_.doubleType());
    }

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.doubleType(), args, false);
    func = llvm::Function::Create(
        func_type, llvm::Function::ExternalLinkage, name, &ctx_.module());

    // Add to function table if available
    if (function_table_) {
        (*function_table_)[name] = func;
    }

    return func;
}

// Unary dual math: each computes the scalar function value g(a), first
// derivative g'(a) and second derivative g''(a) at the primal a, then defers
// to dualUnaryChain which propagates both perturbation slots + the mixed
// second-order term exactly. (a = primal = field 0.)

/** @brief Dual sin: g=sin, g'=cos, g''=-sin, g'''=-cos, propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualSin(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* sin_func = getMathFunc("sin");
    llvm::Function* cos_func = getMathFunc("cos");
    if (!sin_func || !cos_func) return nullptr;
    llvm::Value* sa = ctx_.builder().CreateCall(sin_func, {a});
    llvm::Value* ca = ctx_.builder().CreateCall(cos_func, {a});
    // g=sin, g'=cos, g''=-sin, g'''=-cos
    return dualUnaryChain(ctx_, dual, sa, ca, ctx_.builder().CreateFNeg(sa),
                          ctx_.builder().CreateFNeg(ca));
}

/** @brief Dual cos: g=cos, g'=-sin, g''=-cos, g'''=sin, propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualCos(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* sin_func = getMathFunc("sin");
    llvm::Function* cos_func = getMathFunc("cos");
    if (!sin_func || !cos_func) return nullptr;
    llvm::Value* sa = ctx_.builder().CreateCall(sin_func, {a});
    llvm::Value* ca = ctx_.builder().CreateCall(cos_func, {a});
    // g=cos, g'=-sin, g''=-cos, g'''=sin
    return dualUnaryChain(ctx_, dual, ca, ctx_.builder().CreateFNeg(sa),
                          ctx_.builder().CreateFNeg(ca), sa);
}

/** @brief Dual exp: g=g'=g''=g'''=exp(a), propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualExp(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* exp_func = getMathFunc("exp");
    if (!exp_func) return nullptr;
    llvm::Value* ea = ctx_.builder().CreateCall(exp_func, {a});
    // g=g'=g''=g'''=exp(a)
    return dualUnaryChain(ctx_, dual, ea, ea, ea, ea);
}

/** @brief Dual natural log: g=log(a), g'=1/a, g''=-1/a^2, g'''=2/a^3, propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualLog(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* log_func = getMathFunc("log");
    if (!log_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* la = b.CreateCall(log_func, {a});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* fpa = b.CreateFDiv(one, a);                 // 1/a
    llvm::Value* fppa = b.CreateFNeg(b.CreateFMul(fpa, fpa)); // -1/a^2
    // g'''=2/a^3 for the ESH-0117 ep-derivative triple term.
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* fpppa = b.CreateFMul(two, b.CreateFMul(fpa, b.CreateFMul(fpa, fpa)));
    return dualUnaryChain(ctx_, dual, la, fpa, fppa, fpppa);
}

/** @brief Dual tan: g=tan(a), g'=sec^2(a)=1+tan^2(a), g''=2 tan(a) sec^2(a), propagated via dualUnaryChain (no third-derivative term supplied). */
llvm::Value* AutodiffCodegen::dualTan(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* tan_func = getMathFunc("tan");
    if (!tan_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* ta = b.CreateCall(tan_func, {a});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* sec2 = b.CreateFAdd(one, b.CreateFMul(ta, ta));  // 1+tan^2 = sec^2
    // g'=sec^2, g''=2 tan sec^2
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* fppa = b.CreateFMul(two, b.CreateFMul(ta, sec2));
    return dualUnaryChain(ctx_, dual, ta, sec2, fppa);
}

/** @brief Dual sqrt: g=sqrt(a), g'=1/(2 sqrt(a)), g''=-1/(4 a sqrt(a)), propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualSqrt(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* sqrt_func = getMathFunc("sqrt");
    if (!sqrt_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* s = b.CreateCall(sqrt_func, {a});
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* four = llvm::ConstantFP::get(ctx_.doubleType(), 4.0);
    llvm::Value* fpa = b.CreateFDiv(llvm::ConstantFP::get(ctx_.doubleType(), 1.0),
                                    b.CreateFMul(two, s));            // 1/(2*sqrt)
    // g'' = -1/(4 a^{3/2}) = -1/(4*a*sqrt)
    llvm::Value* fppa = b.CreateFNeg(b.CreateFDiv(
        llvm::ConstantFP::get(ctx_.doubleType(), 1.0),
        b.CreateFMul(four, b.CreateFMul(a, s))));
    return dualUnaryChain(ctx_, dual, s, fpa, fppa);
}

// Power. (a^b). For a CONSTANT exponent (the common (pow x n) case) this is a
// unary jet of g(x)=x^b with g'=b x^{b-1}, g''=b(b-1) x^{b-2} — exact and
// valid for negative base. For a non-constant (dual) exponent we compose
// exp(b*log(a)) over the 4-component jets (exact for a>0; matches the prior
// behaviour, which already required log(a)).
/**
 * @brief Dual power (a^b), dispatching per-component between an exact constant-exponent jet and a general dual-exponent jet.
 *
 * For a runtime-detected constant exponent (e1/e2/e1e2 slots of the exponent
 * all zero — the common (pow x n) case) computes the unary jet of
 * g(x)=x^b directly (g'=b x^{b-1}, g''=b(b-1) x^{b-2}, g'''=b(b-1)(b-2) x^{b-3}),
 * exact and valid for negative base. For a genuinely dual exponent, composes
 * exp(b*log(a)) over the 4-component jets (valid for a>0), then patches the
 * primal back to pow(a,b) for negative-base parity. A runtime select blends
 * the two per jet component according to the constant-exponent test.
 *
 * @param dual_base base jet.
 * @param dual_exp exponent jet.
 * @return the power jet, or nullptr if either operand is null.
 */
llvm::Value* AutodiffCodegen::dualPow(llvm::Value* dual_base, llvm::Value* dual_exp) {
    if (!dual_base || !dual_exp) return nullptr;
    auto& bld = ctx_.builder();
    llvm::Function* pow_func = getMathFunc("pow");
    if (!pow_func) return nullptr;

    llvm::Value* a = dualField(ctx_, dual_base, 0);
    llvm::Value* bexp = dualField(ctx_, dual_exp, 0);

    // Detect a constant exponent at runtime: e1/e2/e1e2 slots all zero.
    llvm::Value* e1 = dualField(ctx_, dual_exp, 1);
    llvm::Value* e2 = dualField(ctx_, dual_exp, 2);
    llvm::Value* e12 = dualField(ctx_, dual_exp, 3);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

    // --- constant-exponent jet (exact, negative-base safe) ---
    llvm::Value* value = bld.CreateCall(pow_func, {a, bexp});      // a^b
    llvm::Value* bm1 = bld.CreateFSub(bexp, one);
    llvm::Value* bm2 = bld.CreateFSub(bexp, llvm::ConstantFP::get(ctx_.doubleType(), 2.0));
    llvm::Value* bm3 = bld.CreateFSub(bexp, llvm::ConstantFP::get(ctx_.doubleType(), 3.0));
    llvm::Value* gpa = bld.CreateFMul(bexp, bld.CreateCall(pow_func, {a, bm1}));   // b a^{b-1}
    llvm::Value* gppa = bld.CreateFMul(bld.CreateFMul(bexp, bm1),
                                       bld.CreateCall(pow_func, {a, bm2}));        // b(b-1) a^{b-2}
    // g'''=b(b-1)(b-2) a^{b-3} for the ESH-0117 ep-derivative triple term.
    llvm::Value* gpppa = bld.CreateFMul(
        bld.CreateFMul(bld.CreateFMul(bexp, bm1), bm2),
        bld.CreateCall(pow_func, {a, bm3}));
    llvm::Value* const_jet = dualUnaryChain(ctx_, dual_base, value, gpa, gppa, gpppa);

    // --- general dual-exponent jet: exp(b * log(a)) ---
    llvm::Value* log_base = dualLog(dual_base);          // 4-comp log
    llvm::Value* t = dualMul(dual_exp, log_base);        // b*log(a)
    llvm::Value* gen_jet = dualExp(t);                   // exp(...) = a^b (a>0)
    // Replace its primal with pow(a,b) for negative-base parity with prior code.
    gen_jet = bld.CreateInsertValue(gen_jet, value, {0});

    // Select per-component between the constant- and general-exponent jets.
    llvm::Value* exp_is_const =
        bld.CreateAnd(bld.CreateAnd(bld.CreateFCmpOEQ(e1, zero), bld.CreateFCmpOEQ(e2, zero)),
                      bld.CreateFCmpOEQ(e12, zero));
    llvm::Value* r[8];
    for (unsigned i = 0; i < 8; ++i)
        r[i] = bld.CreateSelect(exp_is_const, dualField(ctx_, const_jet, i),
                                dualField(ctx_, gen_jet, i));
    return makeDual8(ctx_, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}

/** @brief Dual asin: g=asin(a), g'=1/sqrt(1-a^2), g''=a/(1-a^2)^{3/2}, propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualAsin(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* asin_func = getMathFunc("asin");
    llvm::Function* sqrt_func = getMathFunc("sqrt");
    if (!asin_func || !sqrt_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* va = b.CreateCall(asin_func, {a});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* t = b.CreateFSub(one, b.CreateFMul(a, a));      // 1-a^2
    llvm::Value* r = b.CreateCall(sqrt_func, {t});
    llvm::Value* fpa = b.CreateFDiv(one, r);                     // 1/sqrt(1-a^2)
    llvm::Value* fppa = b.CreateFDiv(a, b.CreateFMul(t, r));     // a/(1-a^2)^{3/2}
    return dualUnaryChain(ctx_, dual, va, fpa, fppa);
}

/** @brief Dual acos: g=acos(a), g'=-1/sqrt(1-a^2), g''=-a/(1-a^2)^{3/2}, propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualAcos(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* acos_func = getMathFunc("acos");
    llvm::Function* sqrt_func = getMathFunc("sqrt");
    if (!acos_func || !sqrt_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* va = b.CreateCall(acos_func, {a});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* t = b.CreateFSub(one, b.CreateFMul(a, a));
    llvm::Value* r = b.CreateCall(sqrt_func, {t});
    llvm::Value* fpa = b.CreateFNeg(b.CreateFDiv(one, r));       // -1/sqrt(1-a^2)
    llvm::Value* fppa = b.CreateFNeg(b.CreateFDiv(a, b.CreateFMul(t, r))); // -a/(1-a^2)^{3/2}
    return dualUnaryChain(ctx_, dual, va, fpa, fppa);
}

/** @brief Dual atan: g=atan(a), g'=1/(1+a^2), g''=-2a/(1+a^2)^2, propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualAtan(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* atan_func = getMathFunc("atan");
    if (!atan_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* va = b.CreateCall(atan_func, {a});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* u = b.CreateFAdd(one, b.CreateFMul(a, a));      // 1+a^2
    llvm::Value* fpa = b.CreateFDiv(one, u);
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* fppa = b.CreateFNeg(b.CreateFDiv(b.CreateFMul(two, a),
                                                  b.CreateFMul(u, u)));  // -2a/(1+a^2)^2
    return dualUnaryChain(ctx_, dual, va, fpa, fppa);
}

/** @brief Dual abs: g=|a|, g'=sign(a) (0 at a=0), g''=0, propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualAbs(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* fabs_func = getMathFunc("fabs");
    if (!fabs_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* abs_a = b.CreateCall(fabs_func, {a});
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* is_zero = b.CreateFCmpOEQ(a, zero);
    llvm::Value* sign = b.CreateSelect(is_zero, zero, b.CreateFDiv(a, abs_a));
    // g'=sign, g''=0 (a.e.)
    return dualUnaryChain(ctx_, dual, abs_a, sign, zero);
}

/** @brief Dual negation (-a): componentwise FNeg over all 8 jet fields. */
llvm::Value* AutodiffCodegen::dualNeg(llvm::Value* dual) {
    if (!dual) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* r[8];
    for (unsigned i = 0; i < 8; ++i) r[i] = b.CreateFNeg(dualField(ctx_, dual, i));
    return makeDual8(ctx_, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}

/** @brief Dual sinh: g=sinh(a), g'=cosh(a), g''=sinh(a), propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualSinh(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* sinh_func = getMathFunc("sinh");
    llvm::Function* cosh_func = getMathFunc("cosh");
    if (!sinh_func || !cosh_func) return nullptr;
    llvm::Value* sa = ctx_.builder().CreateCall(sinh_func, {a});
    llvm::Value* ca = ctx_.builder().CreateCall(cosh_func, {a});
    // g=sinh, g'=cosh, g''=sinh
    return dualUnaryChain(ctx_, dual, sa, ca, sa);
}

/** @brief Dual cosh: g=cosh(a), g'=sinh(a), g''=cosh(a), propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualCosh(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* sinh_func = getMathFunc("sinh");
    llvm::Function* cosh_func = getMathFunc("cosh");
    if (!sinh_func || !cosh_func) return nullptr;
    llvm::Value* sa = ctx_.builder().CreateCall(sinh_func, {a});
    llvm::Value* ca = ctx_.builder().CreateCall(cosh_func, {a});
    // g=cosh, g'=sinh, g''=cosh
    return dualUnaryChain(ctx_, dual, ca, sa, ca);
}

/** @brief Dual tanh: g=tanh(a), g'=1-tanh^2(a) (sech^2), g''=-2 tanh(a) sech^2(a), propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualTanh(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* tanh_func = getMathFunc("tanh");
    if (!tanh_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* ta = b.CreateCall(tanh_func, {a});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* sech2 = b.CreateFSub(one, b.CreateFMul(ta, ta));  // 1-tanh^2
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* fppa = b.CreateFNeg(b.CreateFMul(two, b.CreateFMul(ta, sech2))); // -2 tanh sech^2
    return dualUnaryChain(ctx_, dual, ta, sech2, fppa);
}

/** @brief Dual asinh: g=asinh(a), g'=1/sqrt(a^2+1), g''=-a/(a^2+1)^{3/2}, propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualAsinh(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* asinh_func = getMathFunc("asinh");
    llvm::Function* sqrt_func = getMathFunc("sqrt");
    if (!asinh_func || !sqrt_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* va = b.CreateCall(asinh_func, {a});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* u = b.CreateFAdd(one, b.CreateFMul(a, a));    // a^2+1
    llvm::Value* r = b.CreateCall(sqrt_func, {u});
    llvm::Value* fpa = b.CreateFDiv(one, r);
    llvm::Value* fppa = b.CreateFNeg(b.CreateFDiv(a, b.CreateFMul(u, r))); // -a/(a^2+1)^{3/2}
    return dualUnaryChain(ctx_, dual, va, fpa, fppa);
}

/** @brief Dual acosh: g=acosh(a), g'=1/sqrt(a^2-1), g''=-a/(a^2-1)^{3/2}, propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualAcosh(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* acosh_func = getMathFunc("acosh");
    llvm::Function* sqrt_func = getMathFunc("sqrt");
    if (!acosh_func || !sqrt_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* va = b.CreateCall(acosh_func, {a});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* u = b.CreateFSub(b.CreateFMul(a, a), one);    // a^2-1
    llvm::Value* r = b.CreateCall(sqrt_func, {u});
    llvm::Value* fpa = b.CreateFDiv(one, r);
    llvm::Value* fppa = b.CreateFNeg(b.CreateFDiv(a, b.CreateFMul(u, r))); // -a/(a^2-1)^{3/2}
    return dualUnaryChain(ctx_, dual, va, fpa, fppa);
}

/** @brief Dual atanh: g=atanh(a), g'=1/(1-a^2), g''=2a/(1-a^2)^2, propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualAtanh(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* atanh_func = getMathFunc("atanh");
    if (!atanh_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* va = b.CreateCall(atanh_func, {a});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* u = b.CreateFSub(one, b.CreateFMul(a, a));    // 1-a^2
    llvm::Value* fpa = b.CreateFDiv(one, u);
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* fppa = b.CreateFDiv(b.CreateFMul(two, a), b.CreateFMul(u, u)); // 2a/(1-a^2)^2
    return dualUnaryChain(ctx_, dual, va, fpa, fppa);
}

/** @brief Dual log10: g=log10(a), g'=1/(a ln10), g''=-1/(a^2 ln10), propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualLog10(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* log10_func = getMathFunc("log10");
    if (!log10_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* va = b.CreateCall(log10_func, {a});
    llvm::Value* ln10 = llvm::ConstantFP::get(ctx_.doubleType(), 2.302585092994046);
    llvm::Value* ac = b.CreateFMul(a, ln10);
    llvm::Value* fpa = b.CreateFDiv(llvm::ConstantFP::get(ctx_.doubleType(), 1.0), ac); // 1/(a ln10)
    llvm::Value* fppa = b.CreateFNeg(b.CreateFDiv(
        llvm::ConstantFP::get(ctx_.doubleType(), 1.0), b.CreateFMul(b.CreateFMul(a, a), ln10))); // -1/(a^2 ln10)
    return dualUnaryChain(ctx_, dual, va, fpa, fppa);
}

/** @brief Dual log2: g=log2(a), g'=1/(a ln2), g''=-1/(a^2 ln2), propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualLog2(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* log2_func = getMathFunc("log2");
    if (!log2_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* va = b.CreateCall(log2_func, {a});
    llvm::Value* ln2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.6931471805599453);
    llvm::Value* ac = b.CreateFMul(a, ln2);
    llvm::Value* fpa = b.CreateFDiv(llvm::ConstantFP::get(ctx_.doubleType(), 1.0), ac);
    llvm::Value* fppa = b.CreateFNeg(b.CreateFDiv(
        llvm::ConstantFP::get(ctx_.doubleType(), 1.0), b.CreateFMul(b.CreateFMul(a, a), ln2)));
    return dualUnaryChain(ctx_, dual, va, fpa, fppa);
}

/** @brief Dual exp2: g=2^a, g'=2^a ln2, g''=2^a ln2^2, propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualExp2(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* exp2_func = getMathFunc("exp2");
    if (!exp2_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* va = b.CreateCall(exp2_func, {a});               // 2^a
    llvm::Value* ln2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.6931471805599453);
    llvm::Value* fpa = b.CreateFMul(va, ln2);                     // 2^a ln2
    llvm::Value* fppa = b.CreateFMul(fpa, ln2);                   // 2^a ln2^2
    return dualUnaryChain(ctx_, dual, va, fpa, fppa);
}

/** @brief Dual cbrt: g=a^{1/3}, g'=1/(3 a^{2/3}), g''=-2/(9 a^{5/3}), propagated via dualUnaryChain. */
llvm::Value* AutodiffCodegen::dualCbrt(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* cbrt_func = getMathFunc("cbrt");
    if (!cbrt_func) return nullptr;
    auto& b = ctx_.builder();
    llvm::Value* c = b.CreateCall(cbrt_func, {a});               // a^{1/3}
    llvm::Value* c2 = b.CreateFMul(c, c);
    llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
    llvm::Value* fpa = b.CreateFDiv(llvm::ConstantFP::get(ctx_.doubleType(), 1.0),
                                    b.CreateFMul(three, c2));      // 1/(3 a^{2/3})
    // g'' = -2/9 a^{-5/3} = -2/(9 c^5)
    llvm::Value* c5 = b.CreateFMul(b.CreateFMul(c2, c2), c);
    llvm::Value* nine = llvm::ConstantFP::get(ctx_.doubleType(), 9.0);
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* fppa = b.CreateFNeg(b.CreateFDiv(two, b.CreateFMul(nine, c5)));
    return dualUnaryChain(ctx_, dual, c, fpa, fppa);
}

// Helper to get arena pointer from global
/** @brief Load the current value of the `__global_arena` global, or nullptr if the global does not exist. */
llvm::Value* AutodiffCodegen::getArenaPtr() {
    llvm::GlobalVariable* arena_global = ctx_.module().getNamedGlobal("__global_arena");
    if (!arena_global) return nullptr;
    return ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global);
}

// Create AD node for a constant value (gradient = 0)
/**
 * @brief Allocate a reverse-mode AD tape node of type AD_NODE_CONSTANT (gradient fixed at 0).
 *
 * Converts an integer value to double if needed, allocates the node via the
 * arena's AD-node-with-header allocator, zero-initializes its gradient and
 * input pointers, assigns it a fresh node id, and — if a current AD tape is
 * live — appends it to the tape.
 *
 * @param value the constant's value (double, or integer to be converted).
 * @return pointer to the newly allocated AD node, or nullptr on failure.
 */
llvm::Value* AutodiffCodegen::createADConstant(llvm::Value* value) {
    if (!value) return nullptr;

    // Convert value to double if needed
    if (value->getType()->isIntegerTy()) {
        value = ctx_.builder().CreateSIToFP(value, ctx_.doubleType());
    }

    // Allocate AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    llvm::StructType* ad_type = ctx_.adNodeType();

    // Set type = AD_NODE_CONSTANT (0)
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int32Type(), 0), type_ptr);

    // Set value
    llvm::Value* value_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1);
    ctx_.builder().CreateStore(value, value_ptr);

    // Initialize gradient = 0.0
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), grad_ptr);

    // Set input pointers to null
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input1_ptr);

    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input2_ptr);

    // Set node ID
    llvm::Value* id_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++), id_ptr);

    // Add node to tape
    llvm::GlobalVariable* tape_global = ctx_.currentAdTape();
    if (tape_global) {
        llvm::Value* tape_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tape_global);
        llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
        if (add_node_func) {
            ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
        }
    }

    return node_ptr;
}

// Record binary operation node (add, sub, mul, div) in computational graph
/**
 * @brief Record a binary AD-tape operation node (add/sub/mul/div/pow/max/min/atan2, ...) computing and storing its forward value.
 *
 * Loads the input nodes' values, dispatches on `op_type` to compute the
 * forward result (via LLVM FP ops or libm calls), allocates a new AD node
 * storing the op type, result value, zeroed gradient, both input pointers,
 * and a fresh node id, then appends it to the current AD tape if one is live.
 *
 * @param op_type AD_NODE_* opcode identifying the binary operation.
 * @param left_node pointer to the left operand's AD node.
 * @param right_node pointer to the right operand's AD node.
 * @return pointer to the newly allocated AD node, or nullptr on unknown op_type or null inputs.
 */
llvm::Value* AutodiffCodegen::recordADNodeBinary(uint32_t op_type, llvm::Value* left_node, llvm::Value* right_node) {
    if (!left_node || !right_node) return nullptr;

    llvm::StructType* ad_type = ctx_.adNodeType();

    // Load values from input nodes
    llvm::Value* left_value_ptr = ctx_.builder().CreateStructGEP(ad_type, left_node, 1);
    llvm::Value* left_value = ctx_.builder().CreateLoad(ctx_.doubleType(), left_value_ptr);

    llvm::Value* right_value_ptr = ctx_.builder().CreateStructGEP(ad_type, right_node, 1);
    llvm::Value* right_value = ctx_.builder().CreateLoad(ctx_.doubleType(), right_value_ptr);

    // Compute result value based on operation
    llvm::Value* result_value = nullptr;
    switch (op_type) {
        case 2: // AD_NODE_ADD
            result_value = ctx_.builder().CreateFAdd(left_value, right_value);
            break;
        case 3: // AD_NODE_SUB
            result_value = ctx_.builder().CreateFSub(left_value, right_value);
            break;
        case 4: // AD_NODE_MUL
            result_value = ctx_.builder().CreateFMul(left_value, right_value);
            break;
        case 5: // AD_NODE_DIV
            result_value = ctx_.builder().CreateFDiv(left_value, right_value);
            break;
        case 10: // AD_NODE_POW
            {
                llvm::Function* pow_func = getMathFunc("pow");
                if (!pow_func) return nullptr;
                result_value = ctx_.builder().CreateCall(pow_func, {left_value, right_value});
            }
            break;
        case 44: // AD_NODE_MAX
            {
                // max(a, b) = a if a > b else b
                llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(left_value, right_value);
                result_value = ctx_.builder().CreateSelect(cmp, left_value, right_value);
            }
            break;
        case 45: // AD_NODE_MIN
            {
                // min(a, b) = a if a < b else b
                llvm::Value* cmp = ctx_.builder().CreateFCmpOLT(left_value, right_value);
                result_value = ctx_.builder().CreateSelect(cmp, left_value, right_value);
            }
            break;
        case AD_NODE_ATAN2:
            {
                llvm::Function* atan2_func = getMathFunc("atan2");
                if (!atan2_func) return nullptr;
                result_value = ctx_.builder().CreateCall(atan2_func, {left_value, right_value});
            }
            break;
        default:
            eshkol_warn("Unknown binary AD operation type: %u", op_type);
            return nullptr;
    }

    // Allocate new AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    // Set operation type
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int32Type(), op_type), type_ptr);

    // Set computed value
    llvm::Value* value_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1);
    ctx_.builder().CreateStore(result_value, value_ptr);

    // Initialize gradient = 0.0
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), grad_ptr);

    // Set input pointers
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    ctx_.builder().CreateStore(left_node, input1_ptr);

    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    ctx_.builder().CreateStore(right_node, input2_ptr);

    // Set node ID
    llvm::Value* id_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++), id_ptr);

    // Add to tape
    llvm::GlobalVariable* tape_global = ctx_.currentAdTape();
    if (tape_global) {
        llvm::Value* tape_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tape_global);
        llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
        if (add_node_func) {
            ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
        }
    }

    return node_ptr;
}

/**
 * @brief Record a unary AD-tape operation node (sin/cos/exp/log/activations/etc.) computing and storing its forward value.
 *
 * Loads the input node's value, dispatches on `op_type` across scalar math
 * (sin, cos, exp, log, sqrt, abs, neg, tan, inverse trig/hyperbolic, log2/
 * log10/exp2/cbrt) and neural activations (relu, sigmoid, tanh, gelu,
 * leaky-relu, silu, elu/celu, selu, mish, hardswish, hardsigmoid, softplus,
 * square), allocates a new AD node with the computed value and a single
 * input pointer, and appends it to the current AD tape (guarded by a runtime
 * null check on the tape pointer) if one is live.
 *
 * @param op_type AD_NODE_* opcode identifying the unary operation.
 * @param input_node pointer to the operand's AD node.
 * @return pointer to the newly allocated AD node, or nullptr on unknown op_type or a null input.
 */
llvm::Value* AutodiffCodegen::recordADNodeUnary(uint32_t op_type, llvm::Value* input_node) {
    if (!input_node) return nullptr;

    llvm::StructType* ad_type = ctx_.adNodeType();

    // Load value from input node
    llvm::Value* input_value_ptr = ctx_.builder().CreateStructGEP(ad_type, input_node, 1);
    llvm::Value* input_value = ctx_.builder().CreateLoad(ctx_.doubleType(), input_value_ptr);

    // Compute result value based on operation
    llvm::Value* result_value = nullptr;
    switch (op_type) {
        case 6: // AD_NODE_SIN
            {
                llvm::Function* sin_func = getMathFunc("sin");
                if (!sin_func) return nullptr;
                result_value = ctx_.builder().CreateCall(sin_func, {input_value});
            }
            break;
        case 7: // AD_NODE_COS
            {
                llvm::Function* cos_func = getMathFunc("cos");
                if (!cos_func) return nullptr;
                result_value = ctx_.builder().CreateCall(cos_func, {input_value});
            }
            break;
        case 8: // AD_NODE_EXP
            {
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                result_value = ctx_.builder().CreateCall(exp_func, {input_value});
            }
            break;
        case 9: // AD_NODE_LOG
            {
                llvm::Function* log_func = getMathFunc("log");
                if (!log_func) return nullptr;
                result_value = ctx_.builder().CreateCall(log_func, {input_value});
            }
            break;
        case 11: // AD_NODE_NEG
            result_value = ctx_.builder().CreateFNeg(input_value);
            break;

        // === Activation functions (12-18) ===
        case 12: // AD_NODE_RELU
            {
                // relu(x) = max(0, x)
                llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
                llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(input_value, zero);
                result_value = ctx_.builder().CreateSelect(cmp, input_value, zero);
            }
            break;
        case 13: // AD_NODE_SIGMOID
            {
                // sigmoid(x) = 1 / (1 + exp(-x))
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                llvm::Value* neg_x = ctx_.builder().CreateFNeg(input_value);
                llvm::Value* exp_neg_x = ctx_.builder().CreateCall(exp_func, {neg_x});
                llvm::Value* one_plus = ctx_.builder().CreateFAdd(
                    llvm::ConstantFP::get(ctx_.doubleType(), 1.0), exp_neg_x);
                result_value = ctx_.builder().CreateFDiv(
                    llvm::ConstantFP::get(ctx_.doubleType(), 1.0), one_plus);
            }
            break;
        case 15: // AD_NODE_TANH
            {
                // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                llvm::Value* two_x = ctx_.builder().CreateFMul(
                    llvm::ConstantFP::get(ctx_.doubleType(), 2.0), input_value);
                llvm::Value* exp_2x = ctx_.builder().CreateCall(exp_func, {two_x});
                llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
                llvm::Value* numer = ctx_.builder().CreateFSub(exp_2x, one);
                llvm::Value* denom = ctx_.builder().CreateFAdd(exp_2x, one);
                result_value = ctx_.builder().CreateFDiv(numer, denom);
            }
            break;
        case 16: // AD_NODE_GELU
            {
                // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                llvm::Value* sqrt_2_pi = llvm::ConstantFP::get(ctx_.doubleType(), 0.7978845608028654);
                llvm::Value* coeff = llvm::ConstantFP::get(ctx_.doubleType(), 0.044715);
                llvm::Value* x_cubed = ctx_.builder().CreateFMul(input_value,
                    ctx_.builder().CreateFMul(input_value, input_value));
                llvm::Value* inner = ctx_.builder().CreateFMul(sqrt_2_pi,
                    ctx_.builder().CreateFAdd(input_value,
                        ctx_.builder().CreateFMul(coeff, x_cubed)));
                // tanh via exp
                llvm::Value* two_inner = ctx_.builder().CreateFMul(
                    llvm::ConstantFP::get(ctx_.doubleType(), 2.0), inner);
                llvm::Value* exp_2x = ctx_.builder().CreateCall(exp_func, {two_inner});
                llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
                llvm::Value* tanh_val = ctx_.builder().CreateFDiv(
                    ctx_.builder().CreateFSub(exp_2x, one),
                    ctx_.builder().CreateFAdd(exp_2x, one));
                result_value = ctx_.builder().CreateFMul(
                    llvm::ConstantFP::get(ctx_.doubleType(), 0.5),
                    ctx_.builder().CreateFMul(input_value,
                        ctx_.builder().CreateFAdd(one, tanh_val)));
            }
            break;
        case 17: // AD_NODE_LEAKY_RELU
            {
                // leaky_relu(x) = x > 0 ? x : 0.01 * x
                llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
                llvm::Value* leak = llvm::ConstantFP::get(ctx_.doubleType(), 0.01);
                llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(input_value, zero);
                llvm::Value* leaked = ctx_.builder().CreateFMul(leak, input_value);
                result_value = ctx_.builder().CreateSelect(cmp, input_value, leaked);
            }
            break;
        case 18: // AD_NODE_SILU (Swish)
            {
                // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                llvm::Value* neg_x = ctx_.builder().CreateFNeg(input_value);
                llvm::Value* exp_neg_x = ctx_.builder().CreateCall(exp_func, {neg_x});
                llvm::Value* one_plus = ctx_.builder().CreateFAdd(
                    llvm::ConstantFP::get(ctx_.doubleType(), 1.0), exp_neg_x);
                result_value = ctx_.builder().CreateFDiv(input_value, one_plus);
            }
            break;

        // === Additional math operations (41-44) ===
        case 41: // AD_NODE_SQRT
            {
                llvm::Function* sqrt_func = getMathFunc("sqrt");
                if (!sqrt_func) return nullptr;
                result_value = ctx_.builder().CreateCall(sqrt_func, {input_value});
            }
            break;
        case 42: // AD_NODE_ABS
            {
                llvm::Function* fabs_func = getMathFunc("fabs");
                if (!fabs_func) return nullptr;
                result_value = ctx_.builder().CreateCall(fabs_func, {input_value});
            }
            break;
        case 43: // AD_NODE_SQUARE
            {
                result_value = ctx_.builder().CreateFMul(input_value, input_value);
            }
            break;

        // === Phase 4 activation functions (46-53) ===
        case 46: // AD_NODE_ELU
        case 53: // AD_NODE_CELU
            {
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
                llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
                llvm::Value* exp_val = ctx_.builder().CreateCall(exp_func, {input_value});
                llvm::Value* neg_val = ctx_.builder().CreateFSub(exp_val, one);
                llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(input_value, zero);
                result_value = ctx_.builder().CreateSelect(is_positive, input_value, neg_val);
            }
            break;
        case 47: // AD_NODE_SELU
            {
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
                llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
                llvm::Value* selu_lambda = llvm::ConstantFP::get(ctx_.doubleType(), 1.0507009873554804934193349852946);
                llvm::Value* selu_alpha = llvm::ConstantFP::get(ctx_.doubleType(), 1.6732632423543772848170429916717);
                llvm::Value* pos_val = ctx_.builder().CreateFMul(selu_lambda, input_value);
                llvm::Value* exp_val = ctx_.builder().CreateCall(exp_func, {input_value});
                llvm::Value* exp_minus_1 = ctx_.builder().CreateFSub(exp_val, one);
                llvm::Value* neg_val = ctx_.builder().CreateFMul(selu_lambda,
                    ctx_.builder().CreateFMul(selu_alpha, exp_minus_1));
                llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(input_value, zero);
                result_value = ctx_.builder().CreateSelect(is_positive, pos_val, neg_val);
            }
            break;
        case 48: // AD_NODE_MISH
            {
                llvm::Function* exp_func = getMathFunc("exp");
                llvm::Function* log_func = getMathFunc("log");
                llvm::Function* tanh_func = getMathFunc("tanh");
                if (!exp_func || !log_func || !tanh_func) return nullptr;
                llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
                llvm::Value* exp_val = ctx_.builder().CreateCall(exp_func, {input_value});
                llvm::Value* softplus = ctx_.builder().CreateCall(log_func,
                    {ctx_.builder().CreateFAdd(one, exp_val)});
                llvm::Value* tanh_softplus = ctx_.builder().CreateCall(tanh_func, {softplus});
                result_value = ctx_.builder().CreateFMul(input_value, tanh_softplus);
            }
            break;
        case 49: // AD_NODE_HARDSWISH
            {
                llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
                llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
                llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
                llvm::Value* six = llvm::ConstantFP::get(ctx_.doubleType(), 6.0);
                llvm::Value* x_plus_3 = ctx_.builder().CreateFAdd(input_value, three);
                llvm::Value* divided = ctx_.builder().CreateFDiv(x_plus_3, six);
                llvm::Value* clamped_low = ctx_.builder().CreateSelect(
                    ctx_.builder().CreateFCmpOGT(divided, zero), divided, zero);
                llvm::Value* hard_sigmoid = ctx_.builder().CreateSelect(
                    ctx_.builder().CreateFCmpOLT(clamped_low, one), clamped_low, one);
                result_value = ctx_.builder().CreateFMul(input_value, hard_sigmoid);
            }
            break;
        case 50: // AD_NODE_HARDSIGMOID
            {
                llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
                llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
                llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
                llvm::Value* six = llvm::ConstantFP::get(ctx_.doubleType(), 6.0);
                llvm::Value* x_plus_3 = ctx_.builder().CreateFAdd(input_value, three);
                llvm::Value* divided = ctx_.builder().CreateFDiv(x_plus_3, six);
                llvm::Value* clamped_low = ctx_.builder().CreateSelect(
                    ctx_.builder().CreateFCmpOGT(divided, zero), divided, zero);
                result_value = ctx_.builder().CreateSelect(
                    ctx_.builder().CreateFCmpOLT(clamped_low, one), clamped_low, one);
            }
            break;
        case 51: // AD_NODE_SOFTPLUS
            {
                llvm::Function* exp_func = getMathFunc("exp");
                llvm::Function* log_func = getMathFunc("log");
                if (!exp_func || !log_func) return nullptr;
                llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
                llvm::Value* exp_val = ctx_.builder().CreateCall(exp_func, {input_value});
                result_value = ctx_.builder().CreateCall(log_func,
                    {ctx_.builder().CreateFAdd(one, exp_val)});
            }
            break;

        // === Complete math functions (54-66) ===
        case 54: // AD_NODE_TAN
            {
                llvm::Function* tan_func = getMathFunc("tan");
                if (!tan_func) return nullptr;
                result_value = ctx_.builder().CreateCall(tan_func, {input_value});
            }
            break;
        case 55: // AD_NODE_ASIN
            {
                llvm::Function* asin_func = getMathFunc("asin");
                if (!asin_func) return nullptr;
                result_value = ctx_.builder().CreateCall(asin_func, {input_value});
            }
            break;
        case 56: // AD_NODE_ACOS
            {
                llvm::Function* acos_func = getMathFunc("acos");
                if (!acos_func) return nullptr;
                result_value = ctx_.builder().CreateCall(acos_func, {input_value});
            }
            break;
        case 57: // AD_NODE_ATAN
            {
                llvm::Function* atan_func = getMathFunc("atan");
                if (!atan_func) return nullptr;
                result_value = ctx_.builder().CreateCall(atan_func, {input_value});
            }
            break;
        case 58: // AD_NODE_SINH
            {
                llvm::Function* sinh_func = getMathFunc("sinh");
                if (!sinh_func) return nullptr;
                result_value = ctx_.builder().CreateCall(sinh_func, {input_value});
            }
            break;
        case 59: // AD_NODE_COSH
            {
                llvm::Function* cosh_func = getMathFunc("cosh");
                if (!cosh_func) return nullptr;
                result_value = ctx_.builder().CreateCall(cosh_func, {input_value});
            }
            break;
        case 60: // AD_NODE_ASINH
            {
                llvm::Function* asinh_func = getMathFunc("asinh");
                if (!asinh_func) return nullptr;
                result_value = ctx_.builder().CreateCall(asinh_func, {input_value});
            }
            break;
        case 61: // AD_NODE_ACOSH
            {
                llvm::Function* acosh_func = getMathFunc("acosh");
                if (!acosh_func) return nullptr;
                result_value = ctx_.builder().CreateCall(acosh_func, {input_value});
            }
            break;
        case 62: // AD_NODE_ATANH
            {
                llvm::Function* atanh_func = getMathFunc("atanh");
                if (!atanh_func) return nullptr;
                result_value = ctx_.builder().CreateCall(atanh_func, {input_value});
            }
            break;
        case 63: // AD_NODE_LOG10
            {
                llvm::Function* log10_func = getMathFunc("log10");
                if (!log10_func) return nullptr;
                result_value = ctx_.builder().CreateCall(log10_func, {input_value});
            }
            break;
        case 64: // AD_NODE_LOG2
            {
                llvm::Function* log2_func = getMathFunc("log2");
                if (!log2_func) return nullptr;
                result_value = ctx_.builder().CreateCall(log2_func, {input_value});
            }
            break;
        case 65: // AD_NODE_EXP2
            {
                llvm::Function* exp2_func = getMathFunc("exp2");
                if (!exp2_func) return nullptr;
                result_value = ctx_.builder().CreateCall(exp2_func, {input_value});
            }
            break;
        case 66: // AD_NODE_CBRT
            {
                llvm::Function* cbrt_func = getMathFunc("cbrt");
                if (!cbrt_func) return nullptr;
                result_value = ctx_.builder().CreateCall(cbrt_func, {input_value});
            }
            break;

        default:
            eshkol_warn("Unknown unary AD operation type: %u", op_type);
            return nullptr;
    }

    // Allocate new AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    // Set operation type
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int32Type(), op_type), type_ptr);

    // Set computed value
    llvm::Value* value_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1);
    ctx_.builder().CreateStore(result_value, value_ptr);

    // Initialize gradient = 0.0
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), grad_ptr);

    // Set input1 pointer (for unary operations)
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    ctx_.builder().CreateStore(input_node, input1_ptr);

    // Set input2 to null (unary operation has only one input)
    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input2_ptr);

    // Set node ID
    llvm::Value* id_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++), id_ptr);

    // Add to tape - use global runtime tape pointer with null check
    llvm::GlobalVariable* tape_global = ctx_.currentAdTape();
    if (tape_global) {
        llvm::Value* tape_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tape_global);
        llvm::Value* tape_not_null = ctx_.builder().CreateICmpNE(
            ctx_.builder().CreatePtrToInt(tape_ptr, ctx_.int64Type()),
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* add_to_tape = llvm::BasicBlock::Create(ctx_.context(), "add_unary_to_tape", current_func);
        llvm::BasicBlock* skip_tape = llvm::BasicBlock::Create(ctx_.context(), "skip_unary_tape", current_func);

        ctx_.builder().CreateCondBr(tape_not_null, add_to_tape, skip_tape);

        ctx_.builder().SetInsertPoint(add_to_tape);
        llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
        if (add_node_func) {
            ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
        }
        ctx_.builder().CreateBr(skip_tape);

        ctx_.builder().SetInsertPoint(skip_tape);
    }

    return node_ptr;
}

// === Tensor AD Node Recording ===

/**
 * @brief Allocate and record a tensor-valued AD-tape node, populating all extended fields (tensor value/gradient, up to 4 inputs, saved tensors, shape/ndim, params).
 *
 * Unlike recordADNodeBinary/Unary, this does not compute a forward scalar
 * value (field 1 stays 0.0; the tensor result lives in field 6); callers pass
 * the already-computed tensor result and any saved intermediates needed by
 * the backward pass. Appends the node to the current AD tape (behind a
 * runtime null check) if one is live.
 *
 * @param op_type AD_NODE_* opcode identifying the tensor operation.
 * @param input1 first input AD node (or nullptr).
 * @param input2 second input AD node (or nullptr).
 * @param input3 third input AD node (or nullptr).
 * @param input4 fourth input AD node (or nullptr).
 * @param tensor_result the computed tensor result value.
 * @param saved_tensors pointer to any tensors the backward pass needs saved.
 * @param num_saved count of saved tensors.
 * @param shape the result tensor's shape.
 * @param ndim the result tensor's rank.
 * @return pointer to the newly allocated AD node, or nullptr on allocation failure.
 */
llvm::Value* AutodiffCodegen::recordADNodeTensor(
    uint32_t op_type,
    llvm::Value* input1, llvm::Value* input2,
    llvm::Value* input3, llvm::Value* input4,
    llvm::Value* tensor_result,
    llvm::Value* saved_tensors, llvm::Value* num_saved,
    llvm::Value* shape, llvm::Value* ndim)
{
    llvm::StructType* ad_type = ctx_.adNodeType();
    auto null_ptr = llvm::ConstantPointerNull::get(ctx_.ptrType());
    auto zero_i64 = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    auto zero_f64 = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    // Allocate new AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    // Field 0: type
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int32Type(), op_type),
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0));

    // Field 1: value = 0.0 (tensor ops use tensor_value instead)
    ctx_.builder().CreateStore(zero_f64,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1));

    // Field 2: gradient = 0.0
    ctx_.builder().CreateStore(zero_f64,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2));

    // Field 3: input1
    ctx_.builder().CreateStore(input1 ? input1 : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3));

    // Field 4: input2
    ctx_.builder().CreateStore(input2 ? input2 : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4));

    // Field 5: id
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++),
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5));

    // Field 6: tensor_value
    ctx_.builder().CreateStore(tensor_result ? tensor_result : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 6));

    // Field 7: tensor_gradient = null (allocated during backward)
    ctx_.builder().CreateStore(null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 7));

    // Field 8: input3
    ctx_.builder().CreateStore(input3 ? input3 : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 8));

    // Field 9: input4
    ctx_.builder().CreateStore(input4 ? input4 : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 9));

    // Field 10: saved_tensors
    ctx_.builder().CreateStore(saved_tensors ? saved_tensors : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 10));

    // Field 11: num_saved
    ctx_.builder().CreateStore(num_saved ? num_saved : zero_i64,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 11));

    // Field 12: params (zero-initialized array, caller sets specific values)
    llvm::ArrayType* params_type = llvm::ArrayType::get(ctx_.int64Type(), 6);
    llvm::Value* params_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 12);
    for (unsigned i = 0; i < 6; i++) {
        llvm::Value* elem_ptr = ctx_.builder().CreateConstGEP2_32(params_type, params_ptr, 0, i);
        ctx_.builder().CreateStore(zero_i64, elem_ptr);
    }

    // Field 13: shape
    ctx_.builder().CreateStore(shape ? shape : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 13));

    // Field 14: ndim
    ctx_.builder().CreateStore(ndim ? ndim : zero_i64,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 14));

    // Add to tape
    llvm::GlobalVariable* tape_global = ctx_.currentAdTape();
    if (tape_global) {
        llvm::Value* tape_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tape_global);
        llvm::Value* tape_not_null = ctx_.builder().CreateICmpNE(
            ctx_.builder().CreatePtrToInt(tape_ptr, ctx_.int64Type()),
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* add_block = llvm::BasicBlock::Create(
            ctx_.context(), "add_tensor_to_tape", current_func);
        llvm::BasicBlock* skip_block = llvm::BasicBlock::Create(
            ctx_.context(), "skip_tensor_tape", current_func);

        ctx_.builder().CreateCondBr(tape_not_null, add_block, skip_block);

        ctx_.builder().SetInsertPoint(add_block);
        llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
        if (add_node_func) {
            ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
        }
        ctx_.builder().CreateBr(skip_block);

        ctx_.builder().SetInsertPoint(skip_block);
    }

    return node_ptr;
}

// === Custom scalar VJP AD Node Recording ===

/**
 * @brief Allocate and append an AD_NODE_CUSTOM scalar node.
 *
 * The external descriptor owns the complete multi-input edge list because a
 * scalar ad_node_t has only input1/input2 fields.  We deliberately keep those
 * legacy binary fields null and save the descriptor as the sole saved tensor;
 * the runtime custom-backward helper reads it and accumulates into every input.
 */
llvm::Value* AutodiffCodegen::recordADNodeCustom(
    llvm::Value* value,
    llvm::Value* inputs,
    llvm::Value* input_count,
    llvm::Value* custom_vjp)
{
    if (!value || !inputs || !input_count || !custom_vjp) return nullptr;

    llvm::Value* arena_ptr = getArenaPtr();
    llvm::Function* node_alloc = mem_.getArenaAllocateAdNodeWithHeader();
    llvm::Function* raw_alloc = mem_.getArenaAllocate();
    if (!arena_ptr || !node_alloc || !raw_alloc) return nullptr;

    llvm::StructType* ad_type = ctx_.adNodeType();
    auto null_ptr = llvm::ConstantPointerNull::get(ctx_.ptrType());
    auto zero_f64 = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    llvm::Value* node_ptr = ctx_.builder().CreateCall(node_alloc, {arena_ptr});
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int32Type(), static_cast<int>(AD_NODE_CUSTOM)),
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0));
    ctx_.builder().CreateStore(value,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1));
    ctx_.builder().CreateStore(zero_f64,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2));
    ctx_.builder().CreateStore(null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3));
    ctx_.builder().CreateStore(null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4));
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++),
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5));

    // saved_tensors is always a one-slot arena allocation for custom nodes.
    llvm::Value* saved_slots = ctx_.builder().CreateCall(raw_alloc, {
        arena_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(void*))
    });
    llvm::Value* saved_slots_typed = ctx_.builder().CreatePointerCast(
        saved_slots, ctx_.ptrType());
    ctx_.builder().CreateStore(custom_vjp, saved_slots_typed);
    ctx_.builder().CreateStore(saved_slots_typed,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 10));
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 11));

    // `inputs` and `input_count` are carried by custom_vjp->inputs/n. Keeping
    // the explicit parameters in this builder makes the tape-lifetime contract
    // visible at the compiler call site and prevents accidental stack storage.
    (void)inputs;
    (void)input_count;

    llvm::GlobalVariable* tape_global = ctx_.currentAdTape();
    if (tape_global) {
        llvm::Value* tape_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tape_global);
        llvm::Value* tape_not_null = ctx_.builder().CreateICmpNE(
            tape_ptr, llvm::ConstantPointerNull::get(ctx_.ptrType()));
        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* add_block = llvm::BasicBlock::Create(
            ctx_.context(), "add_custom_to_tape", current_func);
        llvm::BasicBlock* skip_block = llvm::BasicBlock::Create(
            ctx_.context(), "skip_custom_tape", current_func);
        ctx_.builder().CreateCondBr(tape_not_null, add_block, skip_block);

        ctx_.builder().SetInsertPoint(add_block);
        if (llvm::Function* add_node = mem_.getArenaTapeAddNode()) {
            ctx_.builder().CreateCall(add_node, {tape_ptr, node_ptr});
        }
        ctx_.builder().CreateBr(skip_block);
        ctx_.builder().SetInsertPoint(skip_block);
    }

    return node_ptr;
}

// === Tensor Gradient Accumulation ===

/**
 * @brief Accumulate a tensor-valued gradient into an AD node via the runtime helper `eshkol_accumulate_tensor_grad`, guarded by a null check on the node pointer.
 *
 * @param node_ptr target AD node (tensor-valued).
 * @param grad_tensor gradient tensor to add.
 * @param num_elements element count of the gradient tensor.
 */
void AutodiffCodegen::accumulateTensorGradient(
    llvm::Value* node_ptr, llvm::Value* grad_tensor, llvm::Value* num_elements)
{
    if (!node_ptr || !grad_tensor || !num_elements) return;

    // Declare the runtime function if not already declared
    llvm::Module* mod = ctx_.builder().GetInsertBlock()->getModule();
    llvm::Function* accum_func = mod->getFunction("eshkol_accumulate_tensor_grad");
    if (!accum_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.voidType(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()},
            false);
        accum_func = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage,
            "eshkol_accumulate_tensor_grad", mod);
    }

    // Null check on node_ptr
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* accum_block = llvm::BasicBlock::Create(
        ctx_.context(), "accum_tensor_grad", current_func);
    llvm::BasicBlock* skip_block = llvm::BasicBlock::Create(
        ctx_.context(), "skip_tensor_grad", current_func);

    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(
        node_ptr, llvm::ConstantPointerNull::get(ctx_.ptrType()));
    ctx_.builder().CreateCondBr(is_null, skip_block, accum_block);

    ctx_.builder().SetInsertPoint(accum_block);
    ctx_.builder().CreateCall(accum_func, {node_ptr, grad_tensor, num_elements});
    ctx_.builder().CreateBr(skip_block);

    ctx_.builder().SetInsertPoint(skip_block);
}

/**
 * @brief Allocate a reverse-mode AD tape node of type AD_NODE_VARIABLE (an independent variable, gradient computed during backward pass).
 *
 * Converts an integer value to double if needed, allocates the node,
 * zero-initializes its gradient and input pointers (variables have no
 * inputs), and assigns it a fresh node id. Unlike operation nodes, variables
 * are NOT added to the tape (they are tracked separately).
 *
 * @param value the variable's initial value (double, or integer to be converted).
 * @param var_index index of this variable among the differentiated function's parameters (unused by the body but retained for the interface).
 * @return pointer to the newly allocated AD node, or nullptr on failure.
 */
llvm::Value* AutodiffCodegen::createADVariable(llvm::Value* value, size_t var_index) {
    if (!value) return nullptr;

    // Convert value to double if needed
    if (value->getType()->isIntegerTy()) {
        value = ctx_.builder().CreateSIToFP(value, ctx_.doubleType());
    }

    // Allocate AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    llvm::StructType* ad_type = ctx_.adNodeType();

    // Set type = AD_NODE_VARIABLE (1)
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int32Type(), 1), type_ptr);

    // Set value
    llvm::Value* value_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1);
    ctx_.builder().CreateStore(value, value_ptr);

    // Initialize gradient = 0.0 (will be set during backward pass)
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), grad_ptr);

    // Set input pointers to null (variables have no inputs)
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input1_ptr);

    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input2_ptr);

    // Set node ID
    llvm::Value* id_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++), id_ptr);

    // Variables are NOT added to tape (they're stored separately)

    return node_ptr;
}

/** @brief Load an AD node's input1 pointer field (struct field 3). */
llvm::Value* AutodiffCodegen::loadNodeInput1(llvm::Value* node_ptr) {
    if (!node_ptr) return nullptr;
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    return ctx_.builder().CreateLoad(ctx_.ptrType(), input1_ptr);
}

/** @brief Load an AD node's input2 pointer field (struct field 4). */
llvm::Value* AutodiffCodegen::loadNodeInput2(llvm::Value* node_ptr) {
    if (!node_ptr) return nullptr;
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    return ctx_.builder().CreateLoad(ctx_.ptrType(), input2_ptr);
}

/**
 * @brief Codegen the higher-order form `(derivative f)` (no evaluation point): synthesize and return a closure computing f' at a runtime-supplied point.
 *
 * Resolves `f` to an LLVM function (or, when it is a runtime function
 * parameter/captured closure, builds a small wrapper that dispatches through
 * closure_call_callback_). Emits a fresh `derivative_<name>_<n>` function
 * whose body seeds a single-level dual {x, 1, 0, 0}, calls the original
 * function with it, extracts the tangent (getDualTangent) as the derivative
 * value, and returns it packed as a tagged double. Threads through the
 * original function's captures (looked up via the symbol tables / REPL
 * registries) into a freshly-allocated closure wrapping the derivative
 * function, so the returned closure captures the same environment as `f`.
 *
 * @param op the derivative operation AST node (function only, point == null).
 * @return a CALLABLE tagged value wrapping the derivative closure, or nullptr on resolution failure.
 */
llvm::Value* AutodiffCodegen::derivativeHigherOrder(const eshkol_operations_t* op) {
    using namespace llvm;

    eshkol_info("Creating higher-order derivative function (derivative f -> df)");

    // Get the function to differentiate
    Value* func = resolve_lambda_callback_(op->derivative_op.function, 0, callback_context_);

    // RUNTIME FUNCTION PARAMETER FIX: If resolveLambdaFunction returns nullptr,
    // check if this is a function parameter and create a runtime derivative wrapper
    if (!func) {
        const eshkol_ast_t* func_ast = op->derivative_op.function;
        if (func_ast && func_ast->type == ESHKOL_VAR) {
            std::string func_name = func_ast->variable.id;
            eshkol_debug("derivative HO: trying runtime function parameter '%s'", func_name.c_str());

            // Check if this is a function parameter or captured value
            Value* var_value = nullptr;
            auto local_it = symbol_table_->find(func_name);
            if (local_it != symbol_table_->end()) {
                var_value = local_it->second;
            } else {
                auto global_it = global_symbol_table_->find(func_name);
                if (global_it != global_symbol_table_->end()) {
                    var_value = global_it->second;
                }
            }

            if (var_value) {
                // Get the closure value for runtime dispatch
                Value* closure_val = nullptr;
                if (isa<Argument>(var_value) && var_value->getType() == ctx_.taggedValueType()) {
                    closure_val = var_value;
                } else if (isa<Argument>(var_value) && var_value->getType()->isPointerTy()) {
                    closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                } else if (isa<AllocaInst>(var_value)) {
                    AllocaInst* alloca = cast<AllocaInst>(var_value);
                    if (alloca->getAllocatedType() == ctx_.taggedValueType()) {
                        closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                    }
                } else if (isa<LoadInst>(var_value) && var_value->getType() == ctx_.taggedValueType()) {
                    closure_val = var_value;
                } else if (isa<GlobalVariable>(var_value)) {
                    GlobalVariable* global = cast<GlobalVariable>(var_value);
                    closure_val = ctx_.builder().CreateLoad(global->getValueType(), var_value);
                }

                if (closure_val) {
                    eshkol_debug("derivative HO: creating runtime derivative wrapper for '%s'", func_name.c_str());

                    // Create a derivative wrapper that captures the function and calls it at runtime
                    std::string deriv_func_name = "derivative_runtime_" + std::to_string(derivative_ho_counter_++);

                    // Wrapper function takes: (x_tagged, captured_f_ptr)
                    // captured_f_ptr is a pointer to the closure tagged_value
                    std::vector<Type*> param_types = {ctx_.taggedValueType(), PointerType::getUnqual(ctx_.context())};
                    FunctionType* deriv_func_type = FunctionType::get(ctx_.taggedValueType(), param_types, false);
                    Function* deriv_func = Function::Create(
                        deriv_func_type,
                        Function::ExternalLinkage,
                        deriv_func_name,
                        ctx_.module()
                    );

                    // Save current insertion point
                    BasicBlock* saved_bb = ctx_.builder().GetInsertBlock();
                    BasicBlock::iterator saved_point = ctx_.builder().GetInsertPoint();

                    // Create function body
                    BasicBlock* entry = BasicBlock::Create(ctx_.context(), "entry", deriv_func);
                    ctx_.builder().SetInsertPoint(entry);

                    auto arg_it = deriv_func->arg_begin();
                    Value* x_tagged = &(*arg_it);
                    x_tagged->setName("x");
                    ++arg_it;
                    Value* captured_f_ptr = &(*arg_it);
                    captured_f_ptr->setName("captured_f");

                    // Load the captured function closure
                    Value* f_closure = ctx_.builder().CreateLoad(ctx_.taggedValueType(), captured_f_ptr);

                    // Forward-mode AD: seed a dual {x, 1, 0, 0} and call the
                    // captured closure with it. The closure body threads dual
                    // arithmetic at runtime (dispatch on the DUAL_NUMBER tag),
                    // so the tangent of the result is exactly f'(x) — no
                    // finite-difference step / epsilon error.
                    Value* x = tagged_.unpackDouble(x_tagged);
                    Value* one_seed = ConstantFP::get(ctx_.doubleType(), 1.0);
                    Value* x_seed_tagged = packDualToTagged(createDualNumber(x, one_seed));
                    std::vector<Value*> call_args_ad = {x_seed_tagged};
                    Value* f_ad = closure_call_callback_(f_closure, call_args_ad, "derivative-ad", callback_context_);
                    Value* f_ad_dual = safeUnpackDualFromTagged(f_ad);
                    Value* derivative_val = getDualTangent(f_ad_dual);
                    Value* result_tagged = tagged_.packDouble(derivative_val);
                    ctx_.builder().CreateRet(result_tagged);

                    // Restore insertion point
                    if (saved_bb) {
                        ctx_.builder().SetInsertPoint(saved_bb, saved_point);
                    }

                    // Register the derivative function
                    (*function_table_)[deriv_func_name] = deriv_func;

                    // Create closure capturing the function parameter
                    Value* func_ptr_int = ctx_.builder().CreatePtrToInt(deriv_func, ctx_.int64Type());
                    Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

                    uint64_t packed_info = 1;  // 1 capture (the function)
                    Value* packed_captures = ConstantInt::get(ctx_.int64Type(), packed_info);
                    Value* sexpr_ptr = ConstantInt::get(ctx_.int64Type(), 0);
                    // Derivative function returns a scalar
                    Value* return_type_info = ConstantInt::get(ctx_.int64Type(), CLOSURE_RETURN_SCALAR | (1 << 8));
                    Value* closure_name = ConstantPointerNull::get(PointerType::getUnqual(ctx_.context()));

                    // Use with_header allocator for consolidated CALLABLE type
                    Value* closure_ptr = ctx_.builder().CreateCall(get_closure_alloc_func_(callback_context_),
                                                             {arena_ptr, func_ptr_int, packed_captures, sexpr_ptr, return_type_info, closure_name});

                    // Store captured function
                    Value* env_ptr_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), closure_ptr, ConstantInt::get(ctx_.int64Type(), 8));
                    Value* env_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), env_ptr_ptr);
                    Value* captures_base = ctx_.builder().CreateGEP(ctx_.int8Type(), env_ptr, ConstantInt::get(ctx_.int64Type(), 8));
                    ctx_.builder().CreateStore(closure_val, captures_base);

                    // Return closure as CALLABLE tagged value
                    return tagged_.packPtr(closure_ptr, ESHKOL_VALUE_CALLABLE);
                }
            }
        }
        eshkol_error("Failed to resolve function for higher-order derivative");
        return nullptr;
    }

    Function* func_ptr = dyn_cast<Function>(func);
    if (!func_ptr) {
        eshkol_error("higher-order derivative requires a function");
        return nullptr;
    }

    std::string orig_func_name = func_ptr->getName().str();
    std::string deriv_func_name = "derivative_" + orig_func_name + "_" + std::to_string(derivative_ho_counter_++);

    // Create derivative wrapper function: takes x, returns derivative at x
    std::vector<Type*> param_types = {ctx_.taggedValueType()};  // Takes one tagged_value (x)

    // Add capture parameters for the original function if it has captures
    FunctionType* orig_func_type = func_ptr->getFunctionType();
    size_t orig_num_captures = 0;
    if (orig_func_type->getNumParams() > 1) {
        orig_num_captures = orig_func_type->getNumParams() - 1;
        for (size_t i = 0; i < orig_num_captures; i++) {
            param_types.push_back(PointerType::getUnqual(ctx_.context()));  // Capture pointers
        }
    }

    FunctionType* deriv_func_type = FunctionType::get(ctx_.taggedValueType(), param_types, false);
    Function* deriv_func = Function::Create(
        deriv_func_type,
        Function::ExternalLinkage,
        deriv_func_name,
        ctx_.module()
    );

    // Save current insertion point
    BasicBlock* saved_bb = ctx_.builder().GetInsertBlock();
    BasicBlock::iterator saved_point = ctx_.builder().GetInsertPoint();

    // Create function body
    BasicBlock* entry = BasicBlock::Create(ctx_.context(), "entry", deriv_func);
    ctx_.builder().SetInsertPoint(entry);

    // Get x parameter
    auto arg_it = deriv_func->arg_begin();
    Value* x_tagged = &(*arg_it);
    x_tagged->setName("x");

    // Extract double from x
    Value* x = tagged_.unpackDouble(x_tagged);

    // Create dual number with seed = 1.0
    Value* one = ConstantFP::get(ctx_.doubleType(), 1.0);
    Value* x_dual = createDualNumber(x, one);
    Value* x_dual_tagged = packDualToTagged(x_dual);

    // Build call arguments: (x_dual_tagged, captures...)
    std::vector<Value*> call_args = {x_dual_tagged};
    ++arg_it;
    for (size_t i = 0; i < orig_num_captures; i++, ++arg_it) {
        call_args.push_back(&(*arg_it));
    }

    // Call the original function with dual number
    Value* result = ctx_.builder().CreateCall(orig_func_type, func_ptr, call_args);

    // Extract derivative from result (tangent part of dual number)
    Value* result_dual = unpackDualFromTagged(result);
    Value* derivative_val = this->getDualTangent(result_dual);

    // Pack result as tagged double
    Value* result_tagged = tagged_.packDouble(derivative_val);
    ctx_.builder().CreateRet(result_tagged);

    // Restore insertion point
    if (saved_bb) {
        ctx_.builder().SetInsertPoint(saved_bb, saved_point);
    }

    // Register the derivative function
    (*function_table_)[deriv_func_name] = deriv_func;

    // If original function has captures, we need to create a closure
    if (orig_num_captures > 0) {
        // Get capture values from the original function's closure
        std::string orig_lambda_name = func_ptr->getName().str();
        std::vector<Value*> capture_vals;

        for (size_t i = 0; i < orig_num_captures; i++) {
            // Get capture name from original function's parameter
            auto orig_arg_it = func_ptr->arg_begin();
            std::advance(orig_arg_it, i + 1);
            std::string var_name = orig_arg_it->getName().str();
            if (var_name.find("captured_") == 0) {
                var_name = var_name.substr(9);
            }

            std::string capture_key = orig_lambda_name + "_capture_" + var_name;

            // Find capture value
            Value* cap_val = nullptr;
            auto git = global_symbol_table_->find(capture_key);
            if (git != global_symbol_table_->end() && isa<GlobalVariable>(git->second)) {
                cap_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), git->second);
            } else {
                auto lit = symbol_table_->find(capture_key);
                if (lit != symbol_table_->end()) {
                    if (isa<AllocaInst>(lit->second)) {
                        cap_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), lit->second);
                    } else {
                        cap_val = lit->second;
                    }
                }
            }

            if (!cap_val) {
                // Try direct variable lookup
                auto vit = symbol_table_->find(var_name);
                if (vit != symbol_table_->end()) {
                    if (isa<AllocaInst>(vit->second)) {
                        cap_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), vit->second);
                    } else {
                        cap_val = vit->second;
                    }
                } else {
                    auto gvit = global_symbol_table_->find(var_name);
                    if (gvit != global_symbol_table_->end() && isa<GlobalVariable>(gvit->second)) {
                        cap_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), gvit->second);
                    }
                }
            }

            if (cap_val) {
                capture_vals.push_back(cap_val);
            } else {
                eshkol_warn("Could not find capture %s for derivative closure", var_name.c_str());
                capture_vals.push_back(tagged_.packNull());
            }
        }

        // Allocate closure with captures
        Value* func_ptr_int = ctx_.builder().CreatePtrToInt(deriv_func, ctx_.int64Type());
        Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

        uint64_t packed_info = orig_num_captures & 0xFFFF;
        Value* packed_captures = ConstantInt::get(ctx_.int64Type(), packed_info);
        Value* sexpr_ptr = ConstantInt::get(ctx_.int64Type(), 0);
        // Derivative function returns a scalar
        Value* return_type_info = ConstantInt::get(ctx_.int64Type(), CLOSURE_RETURN_SCALAR | (1 << 8));
        Value* closure_name = ConstantPointerNull::get(PointerType::getUnqual(ctx_.context()));

        // Use with_header allocator for consolidated CALLABLE type
        Value* closure_ptr = ctx_.builder().CreateCall(get_closure_alloc_func_(callback_context_),
                                                 {arena_ptr, func_ptr_int, packed_captures, sexpr_ptr, return_type_info, closure_name});

        // Store captures
        Value* env_ptr_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), closure_ptr, ConstantInt::get(ctx_.int64Type(), 8));
        Value* env_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), env_ptr_ptr);
        Value* captures_base = ctx_.builder().CreateGEP(ctx_.int8Type(), env_ptr, ConstantInt::get(ctx_.int64Type(), 8));

        for (size_t i = 0; i < capture_vals.size(); i++) {
            Value* cap_slot = ctx_.builder().CreateGEP(ctx_.taggedValueType(), captures_base,
                ConstantInt::get(ctx_.int64Type(), i));
            ctx_.builder().CreateStore(capture_vals[i], cap_slot);
        }

        // Return closure as CALLABLE tagged value
        return tagged_.packPtr(closure_ptr, ESHKOL_VALUE_CALLABLE);
    } else {
        // No captures - still need to allocate a closure structure
        Value* func_ptr_int = ctx_.builder().CreatePtrToInt(deriv_func, ctx_.int64Type());
        Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

        uint64_t packed_info = 0;  // 0 captures
        Value* packed_captures = ConstantInt::get(ctx_.int64Type(), packed_info);
        Value* sexpr_ptr = ConstantInt::get(ctx_.int64Type(), 0);
        // Derivative function returns a scalar
        Value* return_type_info = ConstantInt::get(ctx_.int64Type(), CLOSURE_RETURN_SCALAR | (1 << 8));
        Value* closure_name_no_cap = ConstantPointerNull::get(PointerType::getUnqual(ctx_.context()));

        // Use with_header allocator for consolidated CALLABLE type
        Value* closure_ptr = ctx_.builder().CreateCall(get_closure_alloc_func_(callback_context_),
                                                 {arena_ptr, func_ptr_int, packed_captures, sexpr_ptr, return_type_info, closure_name_no_cap});

        // Return closure as CALLABLE tagged value
        return tagged_.packPtr(closure_ptr, ESHKOL_VALUE_CALLABLE);
    }
}


/**
 * @brief Codegen `(derivative f x)`: compute f'(x) via forward-mode (dual-number) AD, or delegate to derivativeHigherOrder when no point is given.
 *
 * Evaluates the point (preserving any outer perturbation it may already
 * carry — e.g. when this derivative is lexically nested inside another), and
 * — in Taylor-tower mode — preserves exact integer points instead of forcing
 * them to double. Seeds a fresh perturbation for this nesting level and
 * pushes the runtime perturbation-level counter (seedForwardAndPush).
 * Resolves the function to differentiate — as a compiled lambda, or (via
 * runtime dispatch through closure_call_callback_) as a function parameter,
 * captured closure, local/global variable, or REPL-registered symbol —
 * rebuilding capture arguments with the same value-vs-pointer capture-
 * convention handling used elsewhere (including the isTcoLoopAlloca guard for
 * TCO loop-carried captures). Calls the resolved function with the seeded
 * dual argument and extracts this level's derivative component via
 * popAndExtractForward, which also pops the perturbation level.
 *
 * @param op the derivative operation AST node (function and point).
 * @return the derivative result as a tagged value (scalar, dual slice, or vector), or nullptr on failure.
 */
llvm::Value* AutodiffCodegen::codegenDerivativeMonolith(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->derivative_op.function) {
        eshkol_error("Invalid derivative operation - missing function");
        return nullptr;
    }

    // Higher-order form: (derivative f) returns a closure that computes derivatives
    if (!op->derivative_op.point) {
        return derivativeHigherOrder(op);
    }

    eshkol_info("Computing derivative using forward-mode AD (dual numbers)");

    // ESH-0070: perturbation level is now a RUNTIME counter (see
    // seedForwardAndPush / popAndExtractForward). pert_level holds this
    // derivative's pre-push level, set by seedForwardAndPush below and consumed
    // by popAndExtractForward at every return. This works whether the inner
    // derivative is lexically nested OR reached through a function call.
    Value* pert_level = nullptr;

    // Evaluate the point. When this derivative is lexically NESTED inside
    // another and the inner evaluation point depends on the outer variable
    // (e.g. (derivative (lambda (x) (derivative (lambda (y) ...) x)) 2.0)),
    // `point_raw` is itself a dual number carrying the OUTER perturbation. We
    // MUST preserve that — the old code stripped it via unpackDouble, which is
    // exactly the perturbation-confusion bug.
    Value* point_raw = codegen_ast_callback_(op->derivative_op.point, callback_context_);
    if (!point_raw) {
        eshkol_error("Failed to evaluate derivative point");
        return nullptr;
    }
    Value* point_tagged;
    if (point_raw->getType()->isIntegerTy()) {
        if (adTowerMode_ != TowerMode::NONE) {
            // ESH-0191 (P6): Taylor-tower mode (derivative-n / taylor) --
            // preserve exactness. An integer point is EXACT by R7RS
            // convention; SIToFP-ing it to double here (the order<=2 jet
            // path's behavior, UNCHANGED below) would silently defeat exact-
            // coefficient contagion (design section 9) before it ever
            // reaches eshkol_taylor_seed_tagged. The order<=2 jet path never
            // sets adTowerMode_, so this branch cannot affect it.
            llvm::Value* i64v = point_raw->getType()->isIntegerTy(64)
                ? point_raw
                : ctx_.builder().CreateSExtOrTrunc(point_raw, ctx_.int64Type());
            point_tagged = tagged_.packInt64(i64v, /*is_exact=*/true);
        } else {
            point_tagged = tagged_.packDouble(ctx_.builder().CreateSIToFP(point_raw, ctx_.doubleType()));
        }
    } else if (point_raw->getType()->isDoubleTy()) {
        point_tagged = tagged_.packDouble(point_raw);
    } else if (point_raw->getType() == ctx_.taggedValueType()) {
        point_tagged = point_raw;
    } else {
        eshkol_error("derivative point must be numeric (int64 or double)");
        return nullptr;
    }

    // Seed a fresh perturbation in THIS level's slot (preserving any the point
    // already carries) and PUSH the runtime perturbation level so the callee
    // body — and any derivative/gradient reached through it — tags the NEXT slot.
    Value* x_dual_tagged = seedForwardAndPush(point_tagged, &pert_level);

    // Get the function to differentiate.
    Value* func = resolve_lambda_callback_(op->derivative_op.function, 0, callback_context_);

    // RUNTIME FUNCTION PARAMETER FIX: If resolveLambdaFunction returns nullptr,
    // check if this is a function parameter (runtime closure) and use codegenClosureCall
    if (!func) {
        const eshkol_ast_t* func_ast = op->derivative_op.function;
        if (func_ast && func_ast->type == ESHKOL_VAR) {
            std::string func_name = func_ast->variable.id;
            eshkol_debug("derivative: trying runtime function parameter '%s'", func_name.c_str());

            // Check if this is a function parameter or captured value
            Value* var_value = nullptr;
            auto local_it = symbol_table_->find(func_name);
            if (local_it != symbol_table_->end()) {
                var_value = local_it->second;
            } else {
                auto global_it = global_symbol_table_->find(func_name);
                if (global_it != global_symbol_table_->end()) {
                    var_value = global_it->second;
                }
            }

            // REPL MODE FIX: Check *repl_symbol_addresses_ for functions defined in REPL
            if (!var_value && (repl_mode_enabled_ && *repl_mode_enabled_)) {
                std::lock_guard<std::mutex> lock(*repl_mutex_);
                auto repl_it = repl_symbol_addresses_->find(func_name);
                if (repl_it != repl_symbol_addresses_->end()) {
                    eshkol_debug("derivative: found '%s' in REPL symbol addresses", func_name.c_str());
                    // Create external declaration for the global variable
                    GlobalVariable* global_var = ctx_.module().getGlobalVariable(func_name);
                    if (!global_var) {
                        global_var = new GlobalVariable(
                            ctx_.module(),
                            ctx_.taggedValueType(),
                            false,
                            GlobalValue::ExternalLinkage,
                            nullptr,
                            func_name
                        );
                    }
                    var_value = global_var;
                }
            }

            if (var_value) {
                // Check if it's a function parameter (Argument) with tagged_value type
                // Also check isStructTy() as a fallback for ctx_.taggedValueType() matching
                if (isa<Argument>(var_value) && (var_value->getType() == ctx_.taggedValueType() || var_value->getType()->isStructTy())) {
                    eshkol_debug("derivative: using runtime dispatch for function parameter '%s'", func_name.c_str());
                    std::vector<Value*> call_args = {x_dual_tagged};
                    Value* result = closure_call_callback_(var_value, call_args, "derivative", callback_context_);

                    // Extract the derivative w.r.t. this nesting level's
                    // perturbation slot (scalar at depth 0, a dual slice when
                    // nested so an enclosing derivative can read it).
                    return popAndExtractForward(result, pert_level);
                }
                // Check if it's a pointer to closure (mutable capture)
                if (isa<Argument>(var_value) && var_value->getType()->isPointerTy()) {
                    eshkol_debug("derivative: using runtime dispatch for captured function '%s'", func_name.c_str());
                    Value* loaded_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                    std::vector<Value*> call_args = {x_dual_tagged};
                    Value* result = closure_call_callback_(loaded_val, call_args, "derivative", callback_context_);

                    // Extract the derivative w.r.t. this nesting level's
                    // perturbation slot (scalar at depth 0, a dual slice when
                    // nested so an enclosing derivative can read it).
                    return popAndExtractForward(result, pert_level);
                }
                // Check if it's an AllocaInst (local variable holding a closure)
                if (isa<AllocaInst>(var_value) && var_value->getType()->isPointerTy()) {
                    AllocaInst* alloca = cast<AllocaInst>(var_value);
                    if (alloca->getAllocatedType() == ctx_.taggedValueType()) {
                        eshkol_debug("derivative: using runtime dispatch for local function '%s'", func_name.c_str());
                        Value* loaded_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                        std::vector<Value*> call_args = {x_dual_tagged};
                        Value* result = closure_call_callback_(loaded_val, call_args, "derivative", callback_context_);

                        return popAndExtractForward(result, pert_level);
                    }
                }
                // Check if it's a LoadInst (already loaded closure value)
                if (isa<LoadInst>(var_value) && var_value->getType() == ctx_.taggedValueType()) {
                    eshkol_debug("derivative: using runtime dispatch for loaded function '%s'", func_name.c_str());
                    std::vector<Value*> call_args = {x_dual_tagged};
                    Value* result = closure_call_callback_(var_value, call_args, "derivative", callback_context_);

                    // Extract the derivative w.r.t. this nesting level's
                    // perturbation slot (scalar at depth 0, a dual slice when
                    // nested so an enclosing derivative can read it).
                    return popAndExtractForward(result, pert_level);
                }
                // Check if it's a GlobalVariable (letrec captured function)
                if (isa<GlobalVariable>(var_value)) {
                    eshkol_debug("derivative: using runtime dispatch for global function '%s'", func_name.c_str());
                    GlobalVariable* global = cast<GlobalVariable>(var_value);
                    Value* loaded_val = ctx_.builder().CreateLoad(global->getValueType(), var_value);
                    std::vector<Value*> call_args = {x_dual_tagged};
                    Value* result = closure_call_callback_(loaded_val, call_args, "derivative", callback_context_);

                    // Extract the derivative w.r.t. this nesting level's
                    // perturbation slot (scalar at depth 0, a dual slice when
                    // nested so an enclosing derivative can read it).
                    return popAndExtractForward(result, pert_level);
                }
            }
        }
        eshkol_error("Failed to resolve function for derivative");
        return nullptr;
    }

    Function* func_ptr = dyn_cast<Function>(func);
    if (!func_ptr) {
        eshkol_error("derivative operator requires a function");
        return nullptr;
    }

    // Build arguments for derivative lambda call
    std::vector<Value*> deriv_call_args = {x_dual_tagged};

    // CLOSURE FIX: Load captures from STORAGE
    FunctionType* deriv_func_type = func_ptr->getFunctionType();
    if (deriv_func_type->getNumParams() > 1) {
        size_t num_captures = deriv_func_type->getNumParams() - 1;
        std::string lambda_name = func_ptr->getName().str();

        // REPL MODE: Get capture names from registry instead of parameter names
        // (LLVM external declarations may have empty parameter names)
        std::vector<std::string> capture_names;
        if ((repl_mode_enabled_ && *repl_mode_enabled_)) {
            std::lock_guard<std::mutex> lock(*repl_mutex_);
            auto captures_it = repl_lambda_captures_->find(lambda_name);
            if (captures_it != repl_lambda_captures_->end()) {
                capture_names = captures_it->second;
            }
        }

        for (size_t i = 0; i < num_captures; i++) {
            std::string var_name;
            if (i < capture_names.size()) {
                var_name = capture_names[i];
            } else {
                // Fallback to LLVM parameter names (for non-REPL mode)
                auto arg_it = func_ptr->arg_begin();
                std::advance(arg_it, i + 1);  // Skip first parameter
                if (arg_it != func_ptr->arg_end()) {
                    var_name = arg_it->getName().str();
                    if (var_name.find("captured_") == 0) {
                        var_name = var_name.substr(9);
                    }
                }
            }

            std::string capture_key = lambda_name + "_capture_" + var_name;

            // First try local symbol tables with capture_key
            auto it = global_symbol_table_->find(capture_key);
            bool found_in_global = (it != global_symbol_table_->end());
            if (!found_in_global) {
                it = symbol_table_->find(capture_key);
            }

            bool found = found_in_global ? (it != global_symbol_table_->end()) : (it != symbol_table_->end());

            // INNER FUNCTION FIX: If capture_key not found, try plain variable name
            // This handles lambdas inside functions where captures are function parameters
            // (not stored as GlobalVariables with _capture_ keys)
            // Also handles top-level global variables that are captured by lambdas
            if (!found) {
                it = global_symbol_table_->find(var_name);
                found_in_global = (it != global_symbol_table_->end());
                if (!found_in_global) {
                    it = symbol_table_->find(var_name);
                }
                found = found_in_global ? (it != global_symbol_table_->end()) : (it != symbol_table_->end());
                if (found) {
                    eshkol_debug("Derivative: found capture '%s' via plain variable name", var_name.c_str());
                }
            }

            // REPL MODE: Try creating external declaration for capture global
            if (!found && (repl_mode_enabled_ && *repl_mode_enabled_)) {
                std::lock_guard<std::mutex> lock(*repl_mutex_);
                auto sym_it = repl_symbol_addresses_->find(capture_key);
                if (sym_it != repl_symbol_addresses_->end()) {
                    // Create external declaration for capture global
                    GlobalVariable* capture_global = ctx_.module().getGlobalVariable(capture_key);
                    if (!capture_global) {
                        capture_global = new GlobalVariable(
                            ctx_.module(),
                            ctx_.taggedValueType(),
                            false,
                            GlobalValue::ExternalLinkage,
                            nullptr,
                            capture_key
                        );
                    }
                    // MUTABLE CAPTURE FIX: Pack pointer in closure format
                    // Lambda expects ptr to slot containing {type=INT64, data=ptrtoint(@global)}
                    Value* deriv_global_ptr_int = ctx_.builder().CreatePtrToInt(capture_global, ctx_.int64Type());
                    Value* deriv_packed_capture = tagged_.packInt64(deriv_global_ptr_int, true);
                    Value* deriv_capture_storage = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "deriv_capture_storage");
                    ctx_.builder().CreateStore(deriv_packed_capture, deriv_capture_storage);
                    deriv_call_args.push_back(deriv_capture_storage);
                    continue;
                }
            }

            if (found && it->second) {
                Value* storage = it->second;
                // ESH-0117: transitive capture through a nested `derivative`.
                // When `storage` is itself a forwarded capture pointer
                // ("captured_<var>", the parameter this middle lambda received),
                // it already points DIRECTLY to the slot holding the tagged
                // value the callee will single-load. Forward it as-is so the
                // innermost lambda reads the capture with the SAME convention it
                // was stored with. Re-wrapping it (ptrtoint+packInt64 below)
                // double-indirects — the callee's single load then reads a
                // pointer-as-value → null/garbage (e.g. `(vector-ref p 0)` = 0
                // in a gradient-over-derivative-of-derivative). This is exactly
                // the depth-2 capture loss underlying the nested-forward bug.
                if (auto* arg = llvm::dyn_cast<llvm::Argument>(storage)) {
                    if (arg->getType()->isPointerTy() &&
                        arg->getName() == ("captured_" + var_name)) {
                        deriv_call_args.push_back(storage);
                        continue;
                    }
                }
                // MUTABLE CAPTURE FIX: Pack pointer in closure format
                // Lambda expects ptr to slot containing {type=INT64, data=ptrtoint(@storage)}
                //
                // AD-1 follow-up: when `storage` is a function-parameter Argument
                // with tagged_value type (struct, not pointer) — the case
                // exposed by tests/neural/nn_working.esk's `compute-loss-gradient`
                // capturing `input`/`target`/`b` from outer parameters — the
                // unconditional PtrToInt fails LLVM verification with "PtrToInt
                // source must be pointer".  Mirror the case-split that the
                // recently-disabled new-style derivative() body had: pack the
                // pointer when storage is one, otherwise pass the value-typed
                // tagged_value through a fresh alloca temp slot so the lambda's
                // single-load body sees the value directly.
                if (isTcoLoopAlloca(storage)) {
                    // ESH-0221: `storage` is a TCO loop-carried parameter's
                    // alloca — pointer-typed like any AllocaInst, but the
                    // callee treats it as a VALUE capture (single load), not
                    // a mutable-variable pointer to re-dereference. Load the
                    // CURRENT iteration's value and funnel it through a
                    // value-typed temp slot, exactly like the non-pointer
                    // branch below — packing its ADDRESS instead (the
                    // fallthrough this guard prevents) made the callee read
                    // the alloca's raw address bit-pattern as the captured
                    // double.
                    Value* deriv_tco_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), storage);
                    Value* deriv_temp_storage = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "deriv_capture_tco_val");
                    ctx_.builder().CreateStore(deriv_tco_val, deriv_temp_storage);
                    deriv_call_args.push_back(deriv_temp_storage);
                } else if (storage->getType()->isPointerTy()) {
                    Value* deriv_storage_ptr_int = ctx_.builder().CreatePtrToInt(storage, ctx_.int64Type());
                    Value* deriv_packed_storage = tagged_.packInt64(deriv_storage_ptr_int, true);
                    Value* deriv_temp_storage = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "deriv_capture_storage");
                    ctx_.builder().CreateStore(deriv_packed_storage, deriv_temp_storage);
                    deriv_call_args.push_back(deriv_temp_storage);
                } else {
                    // Value-typed capture (e.g. function-parameter Argument with
                    // tagged_value struct type).  Funnel through a temp slot —
                    // the lambda body's single `load tagged_value` will read
                    // the value directly.
                    Value* deriv_temp_storage = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "deriv_capture_value");
                    ctx_.builder().CreateStore(storage, deriv_temp_storage);
                    deriv_call_args.push_back(deriv_temp_storage);
                }
            } else {
                // MUTABLE CAPTURE FIX: Push null pointer instead of packed zero
                deriv_call_args.push_back(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())));
                eshkol_warn("Derivative: capture '%s' not found, using null pointer", var_name.c_str());
            }
        }
    }
    
    // Call function with dual number input and captures
    // The function will automatically use dual arithmetic, propagating derivatives
    Value* result_tagged = ctx_.builder().CreateCall(func_ptr, deriv_call_args);

    eshkol_debug("Derivative operator: extracting derivative component");

    // Extract the derivative w.r.t. this nesting level's perturbation slot.
    // At depth 0 this is the scalar derivative; when nested it is a dual slice
    // carrying the outer perturbation so the enclosing derivative can read it.
    // safeUnpack inside handles a non-dual scalar result (constant fn etc.).
    return popAndExtractForward(result_tagged, pert_level);
}


// Robustly determine the differentiable (value) arity of a resolved gradient
// target function. The gradient call sites below decide whether to pass the
// whole dual vector as ONE argument (vector-input functions) or to unpack N
// individual dual elements (multi-parameter functions). They originally relied
// solely on function_arity_table_, but that lookup is unreliable: a reverse-mode
// variant is regenerated on every gradient call (e.g. g-sum-sq__rv393) and its
// base key may never be registered, so the lookup returns 0/1 and a 2-arg
// function gets called with a single arg — LLVM rejects it with "Incorrect
// number of arguments passed to called function!".
//
// The concrete llvm::Function signature is the source of truth. Captures are
// appended last by resolveGradientCaptures and carry a "captured_" name prefix,
// so the leading non-capture params are exactly the differentiable value params:
//   - no captures      -> every LLVM param is a value param (use getNumParams)
//   - captures present  -> trust the arity table (which excludes captures), and
//                          fall back to the leading non-capture count if the
//                          table missed.
/**
 * @brief Resolve the true count of differentiable value parameters of an AD
 *        function, reconciling the arity table against the concrete LLVM signature.
 *
 * Captures are appended last (name prefixed "captured_"). With no captures the
 * full LLVM parameter count is the value arity; with captures the table_arity
 * (which excludes captures) is authoritative, falling back to the leading
 * non-capture parameter count only when the table is empty.
 *
 * @param func_ptr concrete LLVM function (returns table_arity unchanged if null).
 * @param table_arity arity from the AD arity table (excludes captures).
 * @return the number of leading differentiable value parameters.
 */
static uint64_t adResolveValueArity(llvm::Function* func_ptr, uint64_t table_arity) {
    if (!func_ptr) return table_arity;
    uint64_t leading_non_capture = 0;
    bool saw_capture = false;
    for (auto& arg : func_ptr->args()) {
        if (std::string(arg.getName()).rfind("captured_", 0) == 0) { saw_capture = true; break; }
        ++leading_non_capture;
    }
    if (!saw_capture) {
        // No captures: every LLVM parameter is a differentiable value parameter.
        return func_ptr->getFunctionType()->getNumParams();
    }
    // Captures present: the arity table excludes captures and is authoritative;
    // fall back to the leading non-capture param count only when it missed.
    return table_arity ? table_arity : leading_non_capture;
}

// ESH-0070: is `name` a TENSOR-VALUED builtin — one whose computation flows
// values through a tensor (so a scalar forward-mode dual seeded on the input
// cannot carry its perturbation through it; only the reverse-mode tape can)?
// Deliberately EXCLUDES scalar activations/elementwise math (relu/sigmoid/gelu/
// exp/log/…): those have exact forward-mode duals and frequently appear inside
// nested gradients, so they must keep the forward path.
/**
 * @brief Is `cname` a TENSOR-VALUED builtin — one whose computation flows through a tensor, so a scalar forward-mode dual cannot carry its perturbation (only the reverse-mode tape can)?
 *
 * Deliberately excludes scalar elementwise activations/math (relu, sigmoid,
 * gelu, exp, log, ...), which have exact forward-mode duals and must keep the
 * forward path even when nested inside a gradient.
 *
 * @param cname candidate builtin name (may be null).
 * @return true iff the name matches a known tensor-valued builtin (tensor-*, gpu-*, matmul, conv*d, softmax, norm/pooling/attention/embedding ops, tensor loss functions, etc).
 */
static bool adIsTensorValuedBuiltin(const char* cname) {
    if (!cname) return false;
    std::string n(cname);
    if (n.rfind("tensor", 0) == 0) return true;   // tensor-*, tensor
    if (n.rfind("gpu-", 0) == 0) return true;      // gpu-matmul / gpu-softmax / …
    static const std::unordered_set<std::string> tset = {
        "batch-norm", "layer-norm", "make-tensor",
        "conv1d", "conv2d", "conv3d", "matmul", "batch-matmul",
        "max-pool2d", "avg-pool2d", "multi-head-attention", "scaled-dot-attention",
        "embedding", "rotary-embedding", "flatten", "reshape", "softmax",
        "cross-entropy-loss", "binary-cross-entropy-loss",
        "contrastive-loss", "cosine-embedding-loss"
    };
    return tset.count(n) != 0;
}

// ESH-0070: does this SOURCE subtree flow values through a tensor op? Scanned on
// the AST (not the emitted IR) so it is NOT confused by tensor allocations that
// an inline nested gradient's own machinery emits. Mirrors astSetsVar's
// recursion; `default: return false` keeps unhandled ops on the forward path
// (correct for scalar code, and the reverse path is only required for genuine
// tensor pipelines).
/**
 * @brief Recursively scan a SOURCE AST subtree for any use of a tensor operation (ESHKOL_TENSOR_OP or a call to a tensor-valued builtin).
 *
 * Scanned on the AST rather than the emitted IR so it is not confused by
 * tensor allocations an inline nested gradient's own machinery may emit.
 * Mirrors astSetsVar's recursion structure; unhandled op kinds default to
 * false (correct for scalar code — the reverse path is only required for
 * genuine tensor pipelines).
 *
 * @param ast the AST node (or cons-cell) to scan.
 * @return true iff the subtree reaches a tensor operation.
 */
// ESH-0235: when `bodies` is non-null, a call to a user-defined function is
// followed into that function's registered body AST (guarded by `visited` to
// stop recursion and a depth cap). This lets the tensor-flow test see through a
// layer of indirection — a differentiated wrapper `(lambda (z) (loss z))` whose
// tensor op lives inside the named `loss` — so its (vector …) point is
// reverse-seeded like the equivalent #(…)/(tensor …) point instead of silently
// zeroing on the forward-mode dual path. With `bodies` null (every pre-existing
// caller) the behaviour is exactly as before: no calls are followed.
static bool adAstUsesTensorOps(
        const eshkol_ast_t* ast,
        const std::unordered_map<std::string, const eshkol_ast_t*>* bodies = nullptr,
        std::unordered_set<std::string>* visited = nullptr,
        int depth = 0) {
    if (!ast) return false;
    if (ast->type == ESHKOL_CONS) {
        return adAstUsesTensorOps(ast->cons_cell.car, bodies, visited, depth) ||
               adAstUsesTensorOps(ast->cons_cell.cdr, bodies, visited, depth);
    }
    if (ast->type != ESHKOL_OP) return false;
    const eshkol_operations_t* op = &ast->operation;
    if (op->op == ESHKOL_TENSOR_OP) return true;
    switch (op->op) {
        case ESHKOL_CALL_OP:
        case ESHKOL_IF_OP:
        case ESHKOL_COND_OP: {
            const eshkol_ast_t* f = op->call_op.func;
            // Moonlab's VQE primitive consumes the reverse-mode AD-node tensor
            // produced for vector inputs. Mark it as tensor-flowing so
            // (gradient (lambda (p) (vqe-energy ham p)) (vector ...)) takes
            // the reverse path rather than the forward dual-vector path.
            if (f && f->type == ESHKOL_VAR && f->variable.id &&
                (std::strcmp(f->variable.id, "vqe-energy") == 0 ||
                 std::strcmp(f->variable.id, "vqe-energy-primitive") == 0)) return true;
            if (f && f->type == ESHKOL_VAR && adIsTensorValuedBuiltin(f->variable.id)) return true;
            // Follow a call into a user-defined function's body (ESH-0235).
            if (bodies && visited && depth < 8 && f && f->type == ESHKOL_VAR &&
                !visited->count(f->variable.id)) {
                auto it = bodies->find(f->variable.id);
                if (it != bodies->end()) {
                    visited->insert(f->variable.id);
                    if (adAstUsesTensorOps(it->second, bodies, visited, depth + 1)) return true;
                }
            }
            if (f && adAstUsesTensorOps(f, bodies, visited, depth)) return true;
            for (uint64_t i = 0; i < op->call_op.num_vars; i++)
                if (adAstUsesTensorOps(&op->call_op.variables[i], bodies, visited, depth)) return true;
            return false;
        }
        case ESHKOL_SEQUENCE_OP:
        case ESHKOL_AND_OP:
        case ESHKOL_OR_OP:
            for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++)
                if (adAstUsesTensorOps(&op->sequence_op.expressions[i], bodies, visited, depth)) return true;
            return false;
        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP:
        case ESHKOL_LETREC_STAR_OP: {
            for (uint64_t i = 0; i < op->let_op.num_bindings; i++)
                if (adAstUsesTensorOps(&op->let_op.bindings[i], bodies, visited, depth)) return true;
            return adAstUsesTensorOps(op->let_op.body, bodies, visited, depth);
        }
        case ESHKOL_LAMBDA_OP:
            return adAstUsesTensorOps(op->lambda_op.body, bodies, visited, depth);
        case ESHKOL_DEFINE_OP:
            return adAstUsesTensorOps(op->define_op.value, bodies, visited, depth);
        default:
            return false;
    }
}

// ESH-0070: IR-level tensor check, used only when the differentiated function is
// referenced by NAME (a VAR — e.g. (gradient bn-loss 1.0)) so its source AST is
// not reachable from the gradient op. A named, single-level function's emitted
// IR contains tensor runtime calls iff it genuinely flows values through tensors
// (it has no INLINE nested-gradient machinery to contaminate the scan — that
// only happens for inline-lambda bodies, which take the AST path above).
/**
 * @brief IR-level check: does a compiled function's body call any function whose name contains "tensor"?
 *
 * Used only when the differentiated function is referenced by name (so its
 * source AST is unreachable from the gradient/derivative op); a named,
 * single-level function's emitted IR contains tensor runtime calls iff it
 * genuinely flows values through tensors, since it has no inline nested-
 * gradient machinery to contaminate the scan.
 *
 * @param func_ptr the compiled LLVM function to scan (must be a definition, not a declaration).
 * @return true iff some called function's name contains "tensor".
 */
static bool adFunctionUsesTensors(llvm::Function* func_ptr) {
    if (!func_ptr || func_ptr->isDeclaration()) return false;
    for (auto& bb : *func_ptr) {
        for (auto& inst : bb) {
            auto* call = llvm::dyn_cast<llvm::CallBase>(&inst);
            if (!call) continue;
            llvm::Function* callee = call->getCalledFunction();
            if (!callee) continue;
            if (callee->getName().contains("tensor")) return true;
        }
    }
    return false;
}



/**
 * @brief Build a runtime closure implementing the higher-order form `(gradient f)`.
 *
 * Emits a new variadic wrapper function `gradient_ho_N` that, when later called
 * as `(grad-f x y z ...)` or `(grad-f point)`, normalizes its arguments (a
 * single vector/tensor/list point is expanded into spread coordinates) into a
 * header'd Scheme vector, then computes the gradient through the shared
 * exact-AD runtime-closure path (emitRuntimeClosureGradient) — the identical
 * forward-/reverse-mode machinery the direct and wrapped forms use. This makes
 * the curried form byte-identical to the two-argument `(gradient f point)`
 * form (no finite differences). TCO context is saved/disabled while building
 * the wrapper body since it has its own internal loops, and restored
 * afterward. The returned value is a CALLABLE closure over the original
 * function.
 *
 * @param op The `gradient` AST operation node, used only for
 *           `op->gradient_op.function` (the function being differentiated).
 * @return A tagged CALLABLE closure value wrapping the generated gradient
 *         function, or nullptr if the target function could not be resolved.
 */
llvm::Value* AutodiffCodegen::gradientHigherOrder(const eshkol_operations_t* op) {
    using namespace llvm;

    eshkol_info("Creating higher-order gradient function (gradient f -> grad_f)");

    // Resolve the function at compile-time if possible
    Value* func = resolve_lambda_callback_(op->gradient_op.function, 0, callback_context_);
    Value* closure_val = nullptr;

    if (!func) {
        // Runtime function parameter - get the closure value
        const eshkol_ast_t* func_ast = op->gradient_op.function;
        if (func_ast && func_ast->type == ESHKOL_VAR) {
            std::string func_name = func_ast->variable.id;
            Value* var_value = nullptr;

            auto local_it = symbol_table_->find(func_name);
            if (local_it != symbol_table_->end()) {
                var_value = local_it->second;
            } else {
                auto global_it = global_symbol_table_->find(func_name);
                if (global_it != global_symbol_table_->end()) {
                    var_value = global_it->second;
                }
            }

            if (var_value) {
                if (isa<Argument>(var_value) && var_value->getType() == ctx_.taggedValueType()) {
                    closure_val = var_value;
                } else if (isa<AllocaInst>(var_value)) {
                    closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                } else if (isa<GlobalVariable>(var_value)) {
                    closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                }
            }
        }

        if (!closure_val) {
            eshkol_error("Failed to resolve function for higher-order gradient");
            return nullptr;
        }
    }

    // Create gradient wrapper function
    // This is VARIADIC: accepts args like (grad-f x y z ...) as a cons list, or
    // a single vector/tensor/list point. Computes the EXACT gradient via the
    // shared runtime-closure path (no finite differences).
    std::string grad_func_name = "gradient_ho_" + std::to_string(gradient_ho_counter_++);

    // CRITICAL: Save and disable TCO context during gradient function generation
    // The gradient function has its own internal loops that must not be confused with TCO
    auto saved_tco_ctx = static_cast<eshkol::BindingCodegen*>(binding_opaque_)->getTCOContext();
    static_cast<eshkol::BindingCodegen*>(binding_opaque_)->getTCOContext().enabled = false;
    static_cast<eshkol::BindingCodegen*>(binding_opaque_)->getTCOContext().func_name = "";
    static_cast<eshkol::BindingCodegen*>(binding_opaque_)->getTCOContext().loop_header = nullptr;

    // Function takes a rest list (variadic args packaged as cons list) + captured function
    std::vector<Type*> param_types = {ctx_.taggedValueType(), PointerType::getUnqual(ctx_.context())};
    FunctionType* grad_func_type = FunctionType::get(ctx_.taggedValueType(), param_types, false);
    Function* grad_func = Function::Create(
        grad_func_type,
        Function::ExternalLinkage,
        grad_func_name,
        ctx_.module()
    );

    BasicBlock* saved_bb = ctx_.builder().GetInsertBlock();
    BasicBlock::iterator saved_point = ctx_.builder().GetInsertPoint();

    // Create the gradient computation body
    BasicBlock* entry = BasicBlock::Create(ctx_.context(), "entry", grad_func);
    ctx_.builder().SetInsertPoint(entry);

    auto arg_it = grad_func->arg_begin();
    Value* args_list = &(*arg_it);  // First arg: rest list of arguments
    args_list->setName("args");
    ++arg_it;
    Value* captured_f_ptr = &(*arg_it);  // Captured function pointer
    captured_f_ptr->setName("captured_f");

    Value* f_closure = ctx_.builder().CreateLoad(ctx_.taggedValueType(), captured_f_ptr);

    // Get cons accessor functions - avoid struct-by-value ABI issues on ARM64
    Function* cons_get_double = (*function_table_)["arena_tagged_cons_get_double"];
    Function* cons_get_type = (*function_table_)["arena_tagged_cons_get_type"];
    Function* cons_get_ptr = (*function_table_)["arena_tagged_cons_get_ptr"];
    if (!cons_get_double || !cons_get_type || !cons_get_ptr) {
        eshkol_error("Cons accessor functions not found");
        if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_point);
        return nullptr;
    }

    // First, convert the cons list to a vector and count dimensions
    // Count list length using simple loop with direct tagged_value access
    BasicBlock* count_loop = BasicBlock::Create(ctx_.context(), "count_loop", grad_func);
    BasicBlock* count_done = BasicBlock::Create(ctx_.context(), "count_done", grad_func);
    BasicBlock* count_body = BasicBlock::Create(ctx_.context(), "count_body", grad_func);

    ctx_.builder().CreateBr(count_loop);
    ctx_.builder().SetInsertPoint(count_loop);
    PHINode* count_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "count");
    count_phi->addIncoming(ConstantInt::get(ctx_.int64Type(), 0), entry);
    PHINode* curr_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "curr");
    curr_phi->addIncoming(args_list, entry);

    // Check if current is null (end of list)
    Value* curr_type = tagged_.getType(curr_phi);
    Value* curr_base = tagged_.getBaseType(curr_type);
    Value* is_null = ctx_.builder().CreateICmpEQ(curr_base, ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));
    ctx_.builder().CreateCondBr(is_null, count_done, count_body);

    // Count body - increment and get cdr using separate type/ptr accessors (ARM64 ABI fix)
    ctx_.builder().SetInsertPoint(count_body);
    Value* count_next = ctx_.builder().CreateAdd(count_phi, ConstantInt::get(ctx_.int64Type(), 1));
    Value* curr_ptr = ctx_.builder().CreateIntToPtr(tagged_.unpackInt64(curr_phi), PointerType::getUnqual(ctx_.context()));
    // Get cdr type and pointer separately, then pack into tagged_value
    Value* cdr_type = ctx_.builder().CreateCall(cons_get_type, {curr_ptr, ConstantInt::get(ctx_.int1Type(), 1)});
    Value* cdr_ptr = ctx_.builder().CreateCall(cons_get_ptr, {curr_ptr, ConstantInt::get(ctx_.int1Type(), 1)});
    Value* cdr_val = tagged_.packPtrWithFlags(cdr_ptr, cdr_type, ConstantInt::get(ctx_.int8Type(), 0));
    count_phi->addIncoming(count_next, count_body);
    curr_phi->addIncoming(cdr_val, count_body);
    ctx_.builder().CreateBr(count_loop);

    // Count done - dim_val is the number of ARGUMENTS supplied to grad-f.
    ctx_.builder().SetInsertPoint(count_done);
    Value* dim_val = count_phi;

    // Point selection:
    //   * exactly ONE argument  -> that argument IS the point, passed through
    //     unchanged. A scalar stays a scalar (so a scalar function is called
    //     with a scalar and the gradient is that scalar's derivative); a single
    //     vector/tensor/list point is handed to the callable as-is, so
    //     ((gradient f) point) matches (gradient f point).
    //   * two or more arguments -> the spread scalar coordinates are gathered
    //     into a header'd Scheme vector.
    // Either way the shared exact-AD path (emitRuntimeClosureGradient) dispatches
    // on the callable's recovered arity — no finite differences anywhere.
    BasicBlock* ho_single = BasicBlock::Create(ctx_.context(), "grad_ho_single_arg", grad_func);
    BasicBlock* ho_multi  = BasicBlock::Create(ctx_.context(), "grad_ho_multi_arg", grad_func);
    BasicBlock* ho_point_ready = BasicBlock::Create(ctx_.context(), "grad_ho_point_ready", grad_func);
    Value* is_single = ctx_.builder().CreateICmpEQ(dim_val, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateCondBr(is_single, ho_single, ho_multi);

    // Single argument: the car of the (non-null) list head IS the point.
    ctx_.builder().SetInsertPoint(ho_single);
    Value* head_cell = ctx_.builder().CreateIntToPtr(tagged_.unpackInt64(args_list),
        PointerType::getUnqual(ctx_.context()));
    Value* single_point = ctx_.builder().CreateLoad(ctx_.taggedValueType(), head_cell);  // car at offset 0
    ctx_.builder().CreateBr(ho_point_ready);
    BasicBlock* ho_single_exit = ctx_.builder().GetInsertBlock();

    // Two-or-more (or zero) arguments: gather the spread scalars into a header'd
    // Scheme vector ([length(8)][tagged doubles], HEAP_SUBTYPE_VECTOR).
    ctx_.builder().SetInsertPoint(ho_multi);
    Value* point_arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
    Value* point_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateVectorWithHeader(), {point_arena_ptr, dim_val});
    ctx_.builder().CreateStore(dim_val, point_ptr);  // length at offset 0
    Value* point_elems = ctx_.builder().CreatePointerCast(
        ctx_.builder().CreateGEP(ctx_.int8Type(), point_ptr, ConstantInt::get(ctx_.int64Type(), 8)),
        ctx_.ptrType());

    // Copy list elements to vector using simple loop
    BasicBlock* copy_loop = BasicBlock::Create(ctx_.context(), "copy_loop", grad_func);
    BasicBlock* copy_done = BasicBlock::Create(ctx_.context(), "copy_done", grad_func);
    BasicBlock* copy_body = BasicBlock::Create(ctx_.context(), "copy_body", grad_func);

    ctx_.builder().CreateBr(copy_loop);
    ctx_.builder().SetInsertPoint(copy_loop);
    PHINode* copy_idx = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "copy_idx");
    copy_idx->addIncoming(ConstantInt::get(ctx_.int64Type(), 0), ho_multi);
    PHINode* copy_curr = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "copy_curr");
    copy_curr->addIncoming(args_list, ho_multi);

    Value* copy_type = tagged_.getType(copy_curr);
    Value* copy_base = tagged_.getBaseType(copy_type);
    Value* copy_is_null = ctx_.builder().CreateICmpEQ(copy_base, ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));
    ctx_.builder().CreateCondBr(copy_is_null, copy_done, copy_body);

    // Copy body - store car and advance using separate type/ptr accessors (ARM64 ABI fix)
    ctx_.builder().SetInsertPoint(copy_body);

    Value* copy_ptr = ctx_.builder().CreateIntToPtr(tagged_.unpackInt64(copy_curr), PointerType::getUnqual(ctx_.context()));
    // Get car type and value separately, then pack into tagged_value
    ctx_.builder().CreateCall(cons_get_type, {copy_ptr, ConstantInt::get(ctx_.int1Type(), 0)});
    Value* car_double = ctx_.builder().CreateCall(cons_get_double, {copy_ptr, ConstantInt::get(ctx_.int1Type(), 0)});
    Value* car_val = tagged_.packDouble(car_double);

    Value* vec_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), point_elems, copy_idx);
    ctx_.builder().CreateStore(car_val, vec_elem_ptr);
    Value* copy_idx_next = ctx_.builder().CreateAdd(copy_idx, ConstantInt::get(ctx_.int64Type(), 1));
    // Get cdr type and pointer separately, then pack into tagged_value
    Value* cdr_type_raw = ctx_.builder().CreateCall(cons_get_type, {copy_ptr, ConstantInt::get(ctx_.int1Type(), 1)});
    Value* cdr_ptr_raw = ctx_.builder().CreateCall(cons_get_ptr, {copy_ptr, ConstantInt::get(ctx_.int1Type(), 1)});
    Value* copy_cdr = tagged_.packPtrWithFlags(cdr_ptr_raw, cdr_type_raw, ConstantInt::get(ctx_.int8Type(), 0));
    copy_idx->addIncoming(copy_idx_next, copy_body);
    copy_curr->addIncoming(copy_cdr, copy_body);
    ctx_.builder().CreateBr(copy_loop);

    ctx_.builder().SetInsertPoint(copy_done);
    Value* vec_point = tagged_.packPtr(
        ctx_.builder().CreatePtrToInt(point_ptr, ctx_.int64Type()), ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(ho_point_ready);
    BasicBlock* ho_multi_exit = ctx_.builder().GetInsertBlock();

    // Compute the EXACT gradient through the shared runtime-closure path — the
    // identical forward-/reverse-mode machinery the direct and wrapped forms
    // use — so ((gradient f) point) is byte-identical to (gradient f point).
    ctx_.builder().SetInsertPoint(ho_point_ready);
    PHINode* point_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "grad_ho_point");
    point_phi->addIncoming(single_point, ho_single_exit);
    point_phi->addIncoming(vec_point, ho_multi_exit);
    Value* result_tensor = emitRuntimeClosureGradient(f_closure, point_phi);
    if (!result_tensor) {
        eshkol_error("higher-order gradient: exact runtime-closure gradient failed");
        if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_point);
        static_cast<eshkol::BindingCodegen*>(binding_opaque_)->getTCOContext() = saved_tco_ctx;
        return nullptr;
    }
    ctx_.builder().CreateRet(result_tensor);

    // Restore insertion point
    if (saved_bb) {
        ctx_.builder().SetInsertPoint(saved_bb, saved_point);
    }

    // Restore TCO context
    static_cast<eshkol::BindingCodegen*>(binding_opaque_)->getTCOContext() = saved_tco_ctx;

    // Register the gradient function
    (*function_table_)[grad_func_name] = grad_func;
    (*nested_function_captures_)[grad_func_name] = {"f"};  // 1 capture

    // Create closure capturing the original function
    if (!closure_val && func) {
        // STATIC FUNCTION FIX: Create a proper closure struct for static functions
        // codegenClosureCall expects a closure struct, not a raw function pointer
        // So we wrap the static function in a 0-capture closure.
        //
        // The captured closure MUST carry the function's true input arity: the
        // shared exact-AD path (emitRuntimeClosureGradient) dispatches on it to
        // decide whether the point is passed whole (arity 1) or unpacked into N
        // scalar arguments (arity N). A 0-arity closure would send the entire
        // point as one tensor argument and misdispatch a multi-scalar loss.
        // Recover the arity exactly as the direct gradient path does: the arity
        // table keyed by the function name, reconciled against the concrete LLVM
        // signature (the source of truth).
        uint64_t static_arity = 0;
        if (auto* fp = llvm::dyn_cast<llvm::Function>(func)) {
            std::string key = fp->getName().str();
            auto rv_pos = key.rfind("__rv");
            if (rv_pos != std::string::npos && rv_pos + 4 < key.size() &&
                key.find_first_not_of("0123456789", rv_pos + 4) == std::string::npos) {
                key.erase(rv_pos);
            }
            if (function_arity_table_) {
                auto ait = function_arity_table_->find(key);
                if (ait != function_arity_table_->end()) static_arity = ait->second;
            }
            static_arity = adResolveValueArity(fp, static_arity);
        }
        Value* static_arena = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
        Value* static_func_ptr_int = ctx_.builder().CreatePtrToInt(func, ctx_.int64Type());
        // packed_info: 0 captures, `static_arity` fixed params, NOT variadic
        uint64_t static_packed_info = (static_arity & 0xFFFF) << 16;
        Value* static_packed = ConstantInt::get(ctx_.int64Type(), static_packed_info);
        Value* static_sexpr = ConstantInt::get(ctx_.int64Type(), 0);
        // return_type_info: bits 0-7 = return_type, bits 8-15 = input_arity
        uint64_t static_return_type_info = CLOSURE_RETURN_UNKNOWN | ((static_arity & 0xFF) << 8);
        Value* static_return_type = ConstantInt::get(ctx_.int64Type(), static_return_type_info);
        Value* static_name = ConstantPointerNull::get(PointerType::getUnqual(ctx_.context()));
        Value* static_closure_ptr = ctx_.builder().CreateCall(get_closure_alloc_func_(callback_context_),
            {static_arena, static_func_ptr_int, static_packed, static_sexpr, static_return_type, static_name});
        closure_val = tagged_.packPtr(static_closure_ptr, ESHKOL_VALUE_CALLABLE);
    }

    Value* func_ptr_int = ctx_.builder().CreatePtrToInt(grad_func, ctx_.int64Type());
    Value* arena = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
    // packed_info format: bits 0-15 = num_captures, bits 16-31 = fixed_params, bit 63 = is_variadic
    // We have 1 capture, 0 fixed params, and IS variadic
    uint64_t packed_info = 1 | (0ULL << 16) | (1ULL << 63);  // 1 capture, variadic
    Value* packed_captures = ConstantInt::get(ctx_.int64Type(), packed_info);
    Value* sexpr_ptr = ConstantInt::get(ctx_.int64Type(), 0);
    // Gradient returns a vector
    Value* return_type_info = ConstantInt::get(ctx_.int64Type(), CLOSURE_RETURN_VECTOR | (1 << 8));
    Value* closure_name = ConstantPointerNull::get(PointerType::getUnqual(ctx_.context()));
    // Use with_header allocator for consolidated CALLABLE type
    Value* closure_ptr = ctx_.builder().CreateCall(get_closure_alloc_func_(callback_context_),
                                             {arena, func_ptr_int, packed_captures, sexpr_ptr, return_type_info, closure_name});

    // Store captured function in closure environment
    Value* env_ptr_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), closure_ptr, ConstantInt::get(ctx_.int64Type(), 8));
    Value* env_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), env_ptr_ptr);
    Value* captures_base = ctx_.builder().CreateGEP(ctx_.int8Type(), env_ptr, ConstantInt::get(ctx_.int64Type(), 8));
    ctx_.builder().CreateStore(closure_val, captures_base);

    // Return closure as CALLABLE tagged value
    return tagged_.packPtr(closure_ptr, ESHKOL_VALUE_CALLABLE);
}


/**
 * @brief Compute (or build a closure for) the gradient of a function at a point.
 *
 * Dispatches on whether `op->gradient_op.point` is present: with no point this
 * is the higher-order form and delegates to gradientHigherOrder(). With a
 * point, this is by far the largest dispatcher in this file, covering several
 * strategies chosen by how the function value and the input point resolve:
 *
 *  - Runtime function parameters (an unresolved VAR naming a closure held in
 *    an Argument/GlobalVariable/AllocaInst/LoadInst) are handled via a unified
 *    resolution path that normalizes cons-list points to Scheme vectors, then
 *    picks scalar vs. vector/tensor handling.
 *  - Scalar inputs use the exact forward-mode 4-jet fast path (the same
 *    machinery as derivative()) since for a single variable the gradient IS
 *    the derivative — this avoids reverse-mode tape overhead and the
 *    degree/value-ratio reconstruction the general path needs.
 *  - Scheme-vector inputs use forward-mode AD with dual numbers, looping over
 *    each dimension and seeding/popping a forward AD context per component
 *    (also participates in nested forward-mode gradient contexts, ESH-0093/96).
 *  - Tensor inputs (or vector inputs not otherwise handled) fall back to
 *    reverse-mode AD: for each dimension i, a fresh 1024-slot tape is
 *    allocated, n AD variable nodes are created from the input, the function
 *    is called to build a computational graph, backpropagate() runs from the
 *    output node, and the gradient of variable i is read off and stored into
 *    the result vector, then the tape is reset for the next dimension.
 *  - When nested inside another AD pass, the result may itself be recorded as
 *    an AD expression node on the outer tape rather than a plain double, so
 *    gradients-of-gradients compose correctly.
 *
 * @param op The `gradient` AST operation node (`op->gradient_op.function` and
 *           `op->gradient_op.point`).
 * @return Tagged tensor/vector value holding the gradient components, a
 *         tagged AD-node value if nested inside an outer AD pass, or nullptr
 *         on failure (unresolved function, evaluation failure, etc).
 */
// Shared exact-AD gradient of a runtime closure value at an already-tagged
// runtime point. See the declaration in autodiff_codegen.h for the contract.
// Extracted verbatim from the former inline body of gradient()'s runtime
// function-parameter path so gradientHigherOrder() can reuse the exact same
// machinery (retiring the finite-difference higher-order path).
llvm::Value* AutodiffCodegen::emitRuntimeClosureGradient(llvm::Value* closure_val,
                                                         llvm::Value* point_val) {
    using namespace llvm;
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

                // Get arena_allocate for Scheme vector allocation
                Function* arena_allocate_func = (*function_table_)["arena_allocate"];
                if (!arena_allocate_func) {
                    eshkol_error("arena_allocate not found for gradient");
                    return nullptr;
                }
                Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

                // Get tagged_value size
                uint64_t tagged_size = ctx_.module().getDataLayout().getTypeAllocSize(ctx_.taggedValueType());

                // bug-OO: normalize a cons-list point ((list …)) to a Scheme vector so the
                // vector/tensor/scalar dispatch below handles it. A raw cons cell would
                // otherwise fall through to the scalar path (n=1) → wrong gradient. Mirrors
                // the cons→svec conversion in the direct-named gradient path.
                {
                    Value* pre_base = tagged_.getBaseType(tagged_.getType(point_val));
                    Value* pre_is_heap = ctx_.builder().CreateICmpEQ(pre_base,
                        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
                    BasicBlock* pre_entry = ctx_.builder().GetInsertBlock();
                    BasicBlock* pre_check = BasicBlock::Create(ctx_.context(), "grad_pt_check_cons", current_func);
                    BasicBlock* pre_conv = BasicBlock::Create(ctx_.context(), "grad_pt_list2svec", current_func);
                    BasicBlock* pre_done = BasicBlock::Create(ctx_.context(), "grad_pt_done", current_func);
                    ctx_.builder().CreateCondBr(pre_is_heap, pre_check, pre_done);

                    ctx_.builder().SetInsertPoint(pre_check);
                    Value* pre_ptr = tagged_.unpackPtr(point_val);
                    Value* pre_hdr = ctx_.builder().CreateGEP(ctx_.int8Type(), pre_ptr, ConstantInt::get(ctx_.int64Type(), -8));
                    Value* pre_sub = ctx_.builder().CreateLoad(ctx_.int8Type(), pre_hdr);
                    Value* pre_is_cons = ctx_.builder().CreateICmpEQ(pre_sub,
                        ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_CONS));
                    ctx_.builder().CreateCondBr(pre_is_cons, pre_conv, pre_done);

                    ctx_.builder().SetInsertPoint(pre_conv);
                    llvm::IRBuilder<> preEntryB(&current_func->getEntryBlock(), current_func->getEntryBlock().begin());
                    Value* pre_slot = preEntryB.CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_pt_list_head");
                    ctx_.builder().CreateStore(point_val, pre_slot);
                    llvm::Function* l2s = ctx_.module().getFunction("eshkol_list_to_svec");
                    if (!l2s) {
                        llvm::FunctionType* l2s_ty = llvm::FunctionType::get(ctx_.builder().getPtrTy(),
                            {ctx_.builder().getPtrTy(), ctx_.builder().getPtrTy()}, false);
                        l2s = llvm::Function::Create(l2s_ty, llvm::Function::ExternalLinkage,
                            "eshkol_list_to_svec", &ctx_.module());
                    }
                    Value* pre_svec = ctx_.builder().CreateCall(l2s, {arena_ptr, pre_slot});
                    Value* pre_svec_tagged = tagged_.packPtr(
                        ctx_.builder().CreatePtrToInt(pre_svec, ctx_.int64Type()), ESHKOL_VALUE_HEAP_PTR);
                    BasicBlock* pre_conv_exit = ctx_.builder().GetInsertBlock();
                    ctx_.builder().CreateBr(pre_done);

                    ctx_.builder().SetInsertPoint(pre_done);
                    PHINode* pt_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3);
                    pt_phi->addIncoming(point_val, pre_entry);          // not heap → unchanged
                    pt_phi->addIncoming(point_val, pre_check);          // heap, not cons → unchanged
                    pt_phi->addIncoming(pre_svec_tagged, pre_conv_exit); // cons → converted svec
                    point_val = pt_phi;
                }

                AllocaInst* rt_result_slot = ctx_.builder().CreateAlloca(
                    ctx_.taggedValueType(), nullptr, "grad_rt_result");

                Value* rt_point_base = tagged_.getBaseType(tagged_.getType(point_val));
                Value* rt_point_is_double = ctx_.builder().CreateICmpEQ(rt_point_base,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
                Value* rt_point_is_int = ctx_.builder().CreateICmpEQ(rt_point_base,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
                Value* rt_point_is_dual = ctx_.builder().CreateICmpEQ(rt_point_base,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
                Value* rt_point_is_scalar = ctx_.builder().CreateOr(
                    ctx_.builder().CreateOr(rt_point_is_double, rt_point_is_int), rt_point_is_dual);

                // Recover the callable's declared input arity so the point
                // is expanded to match the function's true signature. The closure
                // struct carries input_arity at byte offset 33 (see
                // arena_allocate_closure_with_header / codegenClosureCall). A
                // multi-parameter SCALAR loss — e.g. (loss x y), arity 2 — must
                // have its N-element vector/list point UNPACKED into N separate
                // scalar args (the forward-mode collection path below does this
                // via its arity switch). Only an arity<=1 callable legitimately
                // receives the whole point as ONE vector/tensor argument (the
                // ESH-0235 reverse-mode tensor path). Reading the arity here — in
                // the dominating entry block — lets us gate BOTH paths on it, so a
                // first-class multi-arg loss reached through a function parameter
                // is no longer mis-called with a single tensor argument (which
                // misdispatched its scalar body to tensor-sub/tensor-mul).
                Value* clo_arity_val = ctx_.builder().CreateZExt(
                    ctx_.builder().CreateLoad(ctx_.int8Type(),
                        ctx_.builder().CreateGEP(ctx_.int8Type(),
                            ctx_.builder().CreateIntToPtr(tagged_.unpackInt64(closure_val), ctx_.ptrType()),
                            ConstantInt::get(ctx_.int64Type(), 33))),
                    ctx_.int64Type());
                Value* clo_arity_le1 = ctx_.builder().CreateICmpULE(clo_arity_val,
                    ConstantInt::get(ctx_.int64Type(), 1));

                BasicBlock* grad_rt_scalar_fwd = BasicBlock::Create(
                    ctx_.context(), "grad_rt_scalar_fwd", current_func);
                BasicBlock* grad_rt_collection = BasicBlock::Create(
                    ctx_.context(), "grad_rt_collection", current_func);
                BasicBlock* grad_rt_done = BasicBlock::Create(
                    ctx_.context(), "grad_rt_done", current_func);

                // ESH-0212 (defect 1): first-class tensor gradient. A runtime
                // function value — one bound to a variable, passed as a
                // parameter, or reached through a wrapper — arrives here with no
                // compile-time Function*, so the code below differentiates it
                // with forward-mode dual numbers. That is correct for scalar and
                // scalar-vector points, but tensor ops do NOT carry dual numbers
                // through their elements, so a TENSOR point silently collapsed to
                // a zero gradient (#(0 0 0 0)). Reverse mode does carry the
                // signal: seed each input element as an AD variable, call the
                // closure once in AD mode, backpropagate, and read each
                // variable's gradient — exactly what the literal-lambda path does
                // for a statically known function, but dispatched through the
                // runtime closure so it works for any first-class loss.
                {
                    auto& b = ctx_.builder();
                    Value* rvt_base = tagged_.getBaseType(tagged_.getType(point_val));
                    Value* rvt_is_heap = b.CreateICmpEQ(rvt_base,
                        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
                    BasicBlock* rvt_check = BasicBlock::Create(ctx_.context(), "grad_rt_rev_check", current_func);
                    BasicBlock* rvt_reverse = BasicBlock::Create(ctx_.context(), "grad_rt_rev", current_func);
                    BasicBlock* rvt_not_tensor = BasicBlock::Create(ctx_.context(), "grad_rt_rev_skip", current_func);
                    b.CreateCondBr(rvt_is_heap, rvt_check, rvt_not_tensor);

                    b.SetInsertPoint(rvt_check);
                    Value* rvt_hp = tagged_.unpackPtr(point_val);
                    Value* rvt_hdr = b.CreateGEP(ctx_.int8Type(), rvt_hp, ConstantInt::get(ctx_.int64Type(), -8));
                    Value* rvt_sub = b.CreateLoad(ctx_.int8Type(), rvt_hdr);
                    Value* rvt_is_tensor = b.CreateICmpEQ(rvt_sub,
                        ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));
                    // ESH-0235: a (vector …)-constructed point must be reverse-
                    // seeded exactly like a #(…)/(tensor …) point. Accept the
                    // HEAP_SUBTYPE_VECTOR subtype too and convert it to a 1-D
                    // plain-double tensor below, so the identical tape machinery
                    // seeds each element as an AD variable. Without this a
                    // first-class / wrapper loss at a (vector …) point falls to
                    // the forward-mode dual path, which drops the tangent through
                    // a tensor op and returns a silent all-zero gradient.
                    Value* rvt_is_vec = b.CreateICmpEQ(rvt_sub,
                        ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
                    // Only take the single-argument reverse-mode tensor path
                    // when the callable actually accepts the whole point as ONE
                    // argument (arity <= 1). A multi-parameter scalar loss (arity
                    // >= 2) must instead fall through to the forward-mode
                    // collection path, whose arity switch unpacks the point into N
                    // separate scalar args. Without this gate ESH-0235 captured
                    // every vector/list/tensor point and called an arity-N loss
                    // with a single tensor, misdispatching its scalar arithmetic.
                    Value* rvt_seedable = b.CreateAnd(
                        b.CreateOr(rvt_is_tensor, rvt_is_vec), clo_arity_le1);
                    llvm::StructType* rvt_tt = ctx_.tensorType();
                    BasicBlock* rvt_norm_vec = BasicBlock::Create(ctx_.context(), "grad_rt_rev_norm_vec", current_func);
                    BasicBlock* rvt_norm_ten = BasicBlock::Create(ctx_.context(), "grad_rt_rev_norm_ten", current_func);
                    b.CreateCondBr(rvt_seedable, rvt_norm_ten, rvt_not_tensor);

                    // Tensor point: use the tensor struct pointer directly.
                    b.SetInsertPoint(rvt_norm_ten);
                    Value* rvt_ten_ptr = b.CreateIntToPtr(tagged_.unpackInt64(point_val), ctx_.ptrType());
                    b.CreateCondBr(rvt_is_vec, rvt_norm_vec, rvt_reverse);
                    BasicBlock* rvt_norm_ten_exit = b.GetInsertBlock();

                    // (vector …) point: build a 1-D plain-double tensor from the
                    // Scheme vector [len(8)][tagged elems] layout.
                    b.SetInsertPoint(rvt_norm_vec);
                    Value* rvt_vsvec = tagged_.unpackPtr(point_val);
                    Value* rvt_vn = b.CreateLoad(ctx_.int64Type(), rvt_vsvec);
                    Value* rvt_vsrc_base = b.CreateGEP(ctx_.int8Type(), rvt_vsvec, ConstantInt::get(ctx_.int64Type(), 8));
                    Value* rvt_vsrc = b.CreatePointerCast(rvt_vsrc_base, ctx_.ptrType());
                    Value* rvt_vten = b.CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});
                    Value* rvt_vdims = b.CreateCall(mem_.getArenaAllocate(), {arena_ptr, ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t))});
                    Value* rvt_vdims_t = b.CreatePointerCast(rvt_vdims, ctx_.builder().getPtrTy());
                    b.CreateStore(rvt_vn, rvt_vdims_t);
                    b.CreateStore(rvt_vdims_t, b.CreateStructGEP(rvt_tt, rvt_vten, 0));
                    b.CreateStore(ConstantInt::get(ctx_.int64Type(), 1), b.CreateStructGEP(rvt_tt, rvt_vten, 1));
                    b.CreateStore(rvt_vn, b.CreateStructGEP(rvt_tt, rvt_vten, 3));
                    Value* rvt_vdst_raw = b.CreateCall(mem_.getArenaAllocate(),
                        {arena_ptr, b.CreateMul(rvt_vn, ConstantInt::get(ctx_.int64Type(), sizeof(double)))});
                    Value* rvt_vdst = b.CreatePointerCast(rvt_vdst_raw, ctx_.builder().getPtrTy());
                    b.CreateStore(rvt_vdst, b.CreateStructGEP(rvt_tt, rvt_vten, 2));
                    Value* rvt_vi = b.CreateAlloca(ctx_.int64Type(), nullptr, "rvt_vi");
                    b.CreateStore(ConstantInt::get(ctx_.int64Type(), 0), rvt_vi);
                    BasicBlock* rvt_vcc = BasicBlock::Create(ctx_.context(), "rvt_vconv_cond", current_func);
                    BasicBlock* rvt_vcb = BasicBlock::Create(ctx_.context(), "rvt_vconv_body", current_func);
                    BasicBlock* rvt_vce = BasicBlock::Create(ctx_.context(), "rvt_vconv_end", current_func);
                    b.CreateBr(rvt_vcc);
                    b.SetInsertPoint(rvt_vcc);
                    Value* rvt_vidx = b.CreateLoad(ctx_.int64Type(), rvt_vi);
                    b.CreateCondBr(b.CreateICmpULT(rvt_vidx, rvt_vn), rvt_vcb, rvt_vce);
                    b.SetInsertPoint(rvt_vcb);
                    Value* rvt_vsrc_ptr = b.CreateGEP(ctx_.taggedValueType(), rvt_vsrc, rvt_vidx);
                    Value* rvt_vsrc_val = b.CreateLoad(ctx_.taggedValueType(), rvt_vsrc_ptr);
                    Value* rvt_vdbl = tagged_.unpackDouble(rvt_vsrc_val);
                    b.CreateStore(b.CreateBitCast(rvt_vdbl, ctx_.int64Type()),
                        b.CreateGEP(ctx_.int64Type(), rvt_vdst, rvt_vidx));
                    b.CreateStore(b.CreateAdd(rvt_vidx, ConstantInt::get(ctx_.int64Type(), 1)), rvt_vi);
                    b.CreateBr(rvt_vcc);
                    b.SetInsertPoint(rvt_vce);
                    b.CreateBr(rvt_reverse);
                    BasicBlock* rvt_norm_vec_exit = b.GetInsertBlock();

                    // ---- reverse-mode tensor gradient ----
                    b.SetInsertPoint(rvt_reverse);
                    PHINode* rvt_ptr = b.CreatePHI(ctx_.ptrType(), 2, "rvt_ptr");
                    rvt_ptr->addIncoming(rvt_ten_ptr, rvt_norm_ten_exit);
                    rvt_ptr->addIncoming(rvt_vten, rvt_norm_vec_exit);
                    Value* rvt_dims = b.CreateLoad(ctx_.ptrType(), b.CreateStructGEP(rvt_tt, rvt_ptr, 0));
                    Value* rvt_ndim = b.CreateLoad(ctx_.int64Type(), b.CreateStructGEP(rvt_tt, rvt_ptr, 1));
                    Value* rvt_elems = b.CreateLoad(ctx_.ptrType(), b.CreateStructGEP(rvt_tt, rvt_ptr, 2));
                    Value* rvt_n = b.CreateLoad(ctx_.int64Type(), b.CreateStructGEP(rvt_tt, rvt_ptr, 3));

                    // Fresh tape for this gradient; publish as current so
                    // createADVariable / tensor-op recording target it.
                    Value* rvt_tape = b.CreateCall(mem_.getArenaAllocateTape(),
                        {arena_ptr, ConstantInt::get(ctx_.int64Type(), 1024)});
                    Value* rvt_saved_tape = current_tape_ptr_;
                    current_tape_ptr_ = rvt_tape;
                    pushTapeContext(rvt_tape);

                    // Per-element AD variable node pointer array.
                    Value* rvt_nodes = b.CreateCall(arena_allocate_func,
                        {arena_ptr, b.CreateMul(rvt_n, ConstantInt::get(ctx_.int64Type(), sizeof(void*)))});
                    Value* rvt_nodes_t = b.CreatePointerCast(rvt_nodes, ctx_.ptrType());

                    // AD-node tensor (same shape) passed to the closure.
                    Value* rvt_ad = b.CreateCall(mem_.getArenaAllocateTensorFull(),
                        {arena_ptr, rvt_ndim, rvt_n});
                    Value* rvt_ad_dims = b.CreateLoad(ctx_.ptrType(), b.CreateStructGEP(rvt_tt, rvt_ad, 0));
                    Value* rvt_ad_elems = b.CreateLoad(ctx_.ptrType(), b.CreateStructGEP(rvt_tt, rvt_ad, 2));

                    // Result tensor (same shape) that receives the gradient.
                    Value* rvt_res = b.CreateCall(mem_.getArenaAllocateTensorFull(),
                        {arena_ptr, rvt_ndim, rvt_n});
                    Value* rvt_res_dims = b.CreateLoad(ctx_.ptrType(), b.CreateStructGEP(rvt_tt, rvt_res, 0));
                    Value* rvt_res_elems = b.CreateLoad(ctx_.ptrType(), b.CreateStructGEP(rvt_tt, rvt_res, 2));

                    // Copy the shape into both the AD-node tensor and the result.
                    Value* rvt_di = b.CreateAlloca(ctx_.int64Type(), nullptr, "rvt_di");
                    b.CreateStore(ConstantInt::get(ctx_.int64Type(), 0), rvt_di);
                    BasicBlock* rvt_dc = BasicBlock::Create(ctx_.context(), "rvt_dcopy_cond", current_func);
                    BasicBlock* rvt_db = BasicBlock::Create(ctx_.context(), "rvt_dcopy_body", current_func);
                    BasicBlock* rvt_de = BasicBlock::Create(ctx_.context(), "rvt_dcopy_end", current_func);
                    b.CreateBr(rvt_dc);
                    b.SetInsertPoint(rvt_dc);
                    Value* rvt_div = b.CreateLoad(ctx_.int64Type(), rvt_di);
                    b.CreateCondBr(b.CreateICmpULT(rvt_div, rvt_ndim), rvt_db, rvt_de);
                    b.SetInsertPoint(rvt_db);
                    Value* rvt_dv = b.CreateLoad(ctx_.int64Type(), b.CreateGEP(ctx_.int64Type(), rvt_dims, rvt_div));
                    b.CreateStore(rvt_dv, b.CreateGEP(ctx_.int64Type(), rvt_ad_dims, rvt_div));
                    b.CreateStore(rvt_dv, b.CreateGEP(ctx_.int64Type(), rvt_res_dims, rvt_div));
                    b.CreateStore(b.CreateAdd(rvt_div, ConstantInt::get(ctx_.int64Type(), 1)), rvt_di);
                    b.CreateBr(rvt_dc);
                    b.SetInsertPoint(rvt_de);

                    // Seed each element as an AD variable node.
                    Value* rvt_ji = b.CreateAlloca(ctx_.int64Type(), nullptr, "rvt_ji");
                    b.CreateStore(ConstantInt::get(ctx_.int64Type(), 0), rvt_ji);
                    BasicBlock* rvt_sc = BasicBlock::Create(ctx_.context(), "rvt_seed_cond", current_func);
                    BasicBlock* rvt_sb = BasicBlock::Create(ctx_.context(), "rvt_seed_body", current_func);
                    BasicBlock* rvt_se = BasicBlock::Create(ctx_.context(), "rvt_seed_end", current_func);
                    b.CreateBr(rvt_sc);
                    b.SetInsertPoint(rvt_sc);
                    Value* rvt_jv = b.CreateLoad(ctx_.int64Type(), rvt_ji);
                    b.CreateCondBr(b.CreateICmpULT(rvt_jv, rvt_n), rvt_sb, rvt_se);
                    b.SetInsertPoint(rvt_sb);
                    Value* rvt_ebits = b.CreateLoad(ctx_.int64Type(), b.CreateGEP(ctx_.int64Type(), rvt_elems, rvt_jv));
                    Value* rvt_edbl = b.CreateBitCast(rvt_ebits, ctx_.doubleType());
                    Value* rvt_node = createADVariable(rvt_edbl, 0);
                    b.CreateStore(rvt_node, b.CreateGEP(ctx_.ptrType(), rvt_nodes_t, rvt_jv));
                    b.CreateStore(b.CreatePtrToInt(rvt_node, ctx_.int64Type()),
                        b.CreateGEP(ctx_.int64Type(), rvt_ad_elems, rvt_jv));
                    b.CreateStore(b.CreateAdd(rvt_jv, ConstantInt::get(ctx_.int64Type(), 1)), rvt_ji);
                    b.CreateBr(rvt_sc);
                    b.SetInsertPoint(rvt_se);

                    // Call the closure in AD mode with the AD-node tensor.
                    b.CreateStore(ConstantInt::get(ctx_.int1Type(), 1), ctx_.adModeActive());
                    Value* rvt_ad_tagged = tagged_.packPtr(
                        b.CreatePtrToInt(rvt_ad, ctx_.int64Type()), ESHKOL_VALUE_HEAP_PTR);
                    Value* rvt_out = closure_call_callback_(closure_val,
                        std::vector<Value*>{rvt_ad_tagged}, "autodiff", callback_context_);
                    b.CreateStore(ConstantInt::get(ctx_.int1Type(), 0), ctx_.adModeActive());
                    popTapeContext();

                    if (!rvt_out) {
                        eshkol_error("gradient: failed to call runtime tensor function");
                        return nullptr;
                    }

                    // Backpropagate only when the closure returned an AD node. A
                    // constant return means the loss ignores its input, so a zero
                    // gradient is the correct answer (not a dropped signal).
                    Value* rvt_ob = tagged_.getBaseType(tagged_.getType(rvt_out));
                    Value* rvt_is_ad = b.CreateICmpEQ(rvt_ob,
                        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
                    BasicBlock* rvt_bwd = BasicBlock::Create(ctx_.context(), "rvt_bwd", current_func);
                    BasicBlock* rvt_nobwd = BasicBlock::Create(ctx_.context(), "rvt_nobwd", current_func);
                    BasicBlock* rvt_after = BasicBlock::Create(ctx_.context(), "rvt_after_bwd", current_func);
                    b.CreateCondBr(rvt_is_ad, rvt_bwd, rvt_nobwd);

                    b.SetInsertPoint(rvt_bwd);
                    Value* rvt_out_node = b.CreateIntToPtr(tagged_.unpackInt64(rvt_out), ctx_.ptrType());
                    backpropagate(rvt_tape, rvt_out_node);
                    ctx_.builder().CreateBr(rvt_after);

                    ctx_.builder().SetInsertPoint(rvt_nobwd);
                    ctx_.builder().CreateBr(rvt_after);

                    ctx_.builder().SetInsertPoint(rvt_after);
                    // Extract per-element gradients (0.0 where no backward ran).
                    Value* rvt_gi = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "rvt_gi");
                    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), rvt_gi);
                    BasicBlock* rvt_gc = BasicBlock::Create(ctx_.context(), "rvt_grad_cond", current_func);
                    BasicBlock* rvt_gb = BasicBlock::Create(ctx_.context(), "rvt_grad_body", current_func);
                    BasicBlock* rvt_ge = BasicBlock::Create(ctx_.context(), "rvt_grad_end", current_func);
                    ctx_.builder().CreateBr(rvt_gc);
                    ctx_.builder().SetInsertPoint(rvt_gc);
                    Value* rvt_gv = ctx_.builder().CreateLoad(ctx_.int64Type(), rvt_gi);
                    ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(rvt_gv, rvt_n), rvt_gb, rvt_ge);
                    ctx_.builder().SetInsertPoint(rvt_gb);
                    Value* rvt_gnode = ctx_.builder().CreateLoad(ctx_.ptrType(),
                        ctx_.builder().CreateGEP(ctx_.ptrType(), rvt_nodes_t, rvt_gv));
                    Value* rvt_g = loadNodeGradient(rvt_gnode);
                    ctx_.builder().CreateStore(ctx_.builder().CreateBitCast(rvt_g, ctx_.int64Type()),
                        ctx_.builder().CreateGEP(ctx_.int64Type(), rvt_res_elems, rvt_gv));
                    ctx_.builder().CreateStore(
                        ctx_.builder().CreateAdd(rvt_gv, ConstantInt::get(ctx_.int64Type(), 1)), rvt_gi);
                    ctx_.builder().CreateBr(rvt_gc);
                    ctx_.builder().SetInsertPoint(rvt_ge);

                    current_tape_ptr_ = rvt_saved_tape;
                    ctx_.builder().CreateStore(tagged_.packHeapPtr(rvt_res), rt_result_slot);
                    ctx_.builder().CreateBr(grad_rt_done);

                    ctx_.builder().SetInsertPoint(rvt_not_tensor);
                }

                ctx_.builder().CreateCondBr(rt_point_is_scalar, grad_rt_scalar_fwd, grad_rt_collection);

                ctx_.builder().SetInsertPoint(grad_rt_scalar_fwd);
                Value* rt_fwd_level = nullptr;
                Value* rt_fwd_seed = seedForwardAndPush(point_val, &rt_fwd_level);
                Value* rt_fwd_call = closure_call_callback_(closure_val,
                    std::vector<Value*>{rt_fwd_seed}, "autodiff", callback_context_);
                if (!rt_fwd_call) {
                    eshkol_error("gradient: failed to call runtime scalar function");
                    return nullptr;
                }
                Value* rt_fwd_result = popAndExtractForward(rt_fwd_call, rt_fwd_level);
                ctx_.builder().CreateStore(rt_fwd_result, rt_result_slot);
                ctx_.builder().CreateBr(grad_rt_done);

                ctx_.builder().SetInsertPoint(grad_rt_collection);

                // Check input type - handle Scheme vector (HEAP_PTR with HEAP_SUBTYPE_VECTOR), tensor (TENSOR_PTR), or scalar
                // M1 Migration: Use consolidated HEAP_PTR type with subtype dispatch
                Value* input_type = tagged_.getType(point_val);
                Value* input_base = tagged_.getBaseType(input_type);

                // Check for HEAP_PTR (consolidated format)
                Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(input_base,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
                // Legacy VECTOR_PTR check (for backwards compatibility during migration)
                Value* is_legacy_vec = ctx_.builder().CreateICmpEQ(input_base,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
                Value* is_tensor = ctx_.builder().CreateICmpEQ(input_base,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

                BasicBlock* heap_ptr_dispatch = BasicBlock::Create(ctx_.context(), "grad_rt_heap_dispatch", current_func);
                BasicBlock* heap_check_tensor = BasicBlock::Create(ctx_.context(), "grad_rt_heap_check_tensor", current_func);
                BasicBlock* scheme_vec_path = BasicBlock::Create(ctx_.context(), "grad_rt_svec", current_func);
                BasicBlock* tensor_path = BasicBlock::Create(ctx_.context(), "grad_rt_tensor", current_func);
                BasicBlock* scalar_path = BasicBlock::Create(ctx_.context(), "grad_rt_scalar", current_func);
                BasicBlock* check_legacy_vec = BasicBlock::Create(ctx_.context(), "grad_rt_check_legacy", current_func);
                BasicBlock* check_tensor = BasicBlock::Create(ctx_.context(), "grad_rt_check_tensor", current_func);
                BasicBlock* grad_rt_compute = BasicBlock::Create(ctx_.context(), "grad_rt_compute", current_func);

                // First check for HEAP_PTR (new consolidated format)
                ctx_.builder().CreateCondBr(is_heap_ptr, heap_ptr_dispatch, check_legacy_vec);

                // HEAP_PTR dispatch - read subtype from header and route accordingly
                ctx_.builder().SetInsertPoint(heap_ptr_dispatch);
                Value* heap_ptr_val = tagged_.unpackPtr(point_val);
                Value* header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), heap_ptr_val, ConstantInt::get(ctx_.int64Type(), -8));
                Value* subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), header_ptr);
                Value* is_vec_subtype = ctx_.builder().CreateICmpEQ(subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
                Value* is_tensor_subtype = ctx_.builder().CreateICmpEQ(subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));
                ctx_.builder().CreateCondBr(is_vec_subtype, scheme_vec_path, heap_check_tensor);

                ctx_.builder().SetInsertPoint(heap_check_tensor);
                ctx_.builder().CreateCondBr(is_tensor_subtype, tensor_path, scalar_path);

                // Legacy VECTOR_PTR fallback (for migration period)
                ctx_.builder().SetInsertPoint(check_legacy_vec);
                ctx_.builder().CreateCondBr(is_legacy_vec, scheme_vec_path, check_tensor);

                ctx_.builder().SetInsertPoint(check_tensor);
                ctx_.builder().CreateCondBr(is_tensor, tensor_path, scalar_path);

                // Scheme vector path - use input directly
                ctx_.builder().SetInsertPoint(scheme_vec_path);
                Value* svec_ptr = tagged_.unpackPtr(point_val);
                Value* svec_len = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_ptr);
                Value* svec_elems = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_ptr, ConstantInt::get(ctx_.int64Type(), 8));
                Value* svec_elems_typed = ctx_.builder().CreatePointerCast(svec_elems, ctx_.ptrType());
                ctx_.builder().CreateBr(grad_rt_compute);
                BasicBlock* svec_exit = ctx_.builder().GetInsertBlock();

                // Tensor path - convert tensor elements to Scheme vector of tagged doubles
                ctx_.builder().SetInsertPoint(tensor_path);
                Value* tensor_ptr_int = tagged_.unpackInt64(point_val);
                Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.builder().getPtrTy());
                Value* dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 0);
                Value* dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dims_field);
                Value* typed_dims = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());
                Value* tensor_n = ctx_.builder().CreateLoad(ctx_.int64Type(), typed_dims);
                Value* tensor_elems_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 2);
                Value* tensor_elems_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), tensor_elems_field);
                Value* tensor_elems_typed = ctx_.builder().CreatePointerCast(tensor_elems_ptr, ctx_.builder().getPtrTy());

                // Allocate Scheme vector for tensor elements
                Value* tconv_size = ctx_.builder().CreateAdd(
                    ctx_.builder().CreateMul(tensor_n, ConstantInt::get(ctx_.int64Type(), tagged_size)),
                    ConstantInt::get(ctx_.int64Type(), 8));
                Value* tconv_vec = ctx_.builder().CreateCall(arena_allocate_func, {arena_ptr, tconv_size});
                ctx_.builder().CreateStore(tensor_n, tconv_vec);
                Value* tconv_elems = ctx_.builder().CreateGEP(ctx_.int8Type(), tconv_vec, ConstantInt::get(ctx_.int64Type(), 8));
                Value* tconv_elems_typed = ctx_.builder().CreatePointerCast(tconv_elems, ctx_.ptrType());

                // Copy tensor elements as tagged doubles
                Value* tconv_i = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "tconv_i");
                ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), tconv_i);
                BasicBlock* tconv_cond = BasicBlock::Create(ctx_.context(), "tconv_cond", current_func);
                BasicBlock* tconv_body = BasicBlock::Create(ctx_.context(), "tconv_body", current_func);
                BasicBlock* tconv_end = BasicBlock::Create(ctx_.context(), "tconv_end", current_func);
                ctx_.builder().CreateBr(tconv_cond);

                ctx_.builder().SetInsertPoint(tconv_cond);
                Value* tc_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), tconv_i);
                ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(tc_idx, tensor_n), tconv_body, tconv_end);

                ctx_.builder().SetInsertPoint(tconv_body);
                Value* tc_src = ctx_.builder().CreateGEP(ctx_.int64Type(), tensor_elems_typed, tc_idx);
                Value* tc_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), tc_src);
                Value* tc_dbl = ctx_.builder().CreateBitCast(tc_bits, ctx_.doubleType());
                Value* tc_tagged = tagged_.packDouble(tc_dbl);
                Value* tc_dst = ctx_.builder().CreateGEP(ctx_.taggedValueType(), tconv_elems_typed, tc_idx);
                ctx_.builder().CreateStore(tc_tagged, tc_dst);
                ctx_.builder().CreateStore(ctx_.builder().CreateAdd(tc_idx, ConstantInt::get(ctx_.int64Type(), 1)), tconv_i);
                ctx_.builder().CreateBr(tconv_cond);

                ctx_.builder().SetInsertPoint(tconv_end);
                ctx_.builder().CreateBr(grad_rt_compute);
                BasicBlock* tensor_exit = ctx_.builder().GetInsertBlock();

                // Scalar path - create 1-element Scheme vector
                ctx_.builder().SetInsertPoint(scalar_path);
                Value* scalar_val = tagged_.unpackDouble(point_val);
                Value* scalar_vec_size = ConstantInt::get(ctx_.int64Type(), 8 + tagged_size);
                Value* scalar_vec = ctx_.builder().CreateCall(arena_allocate_func, {arena_ptr, scalar_vec_size});
                ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), scalar_vec);
                Value* scalar_elem = ctx_.builder().CreateGEP(ctx_.int8Type(), scalar_vec, ConstantInt::get(ctx_.int64Type(), 8));
                Value* scalar_elem_typed = ctx_.builder().CreatePointerCast(scalar_elem, ctx_.ptrType());
                Value* scalar_tagged = tagged_.packDouble(scalar_val);
                ctx_.builder().CreateStore(scalar_tagged, scalar_elem_typed);
                ctx_.builder().CreateBr(grad_rt_compute);
                BasicBlock* scalar_exit = ctx_.builder().GetInsertBlock();

                // Merge input paths
                ctx_.builder().SetInsertPoint(grad_rt_compute);
                PHINode* n = ctx_.builder().CreatePHI(ctx_.int64Type(), 3, "grad_n");
                n->addIncoming(svec_len, svec_exit);
                n->addIncoming(tensor_n, tensor_exit);
                n->addIncoming(ConstantInt::get(ctx_.int64Type(), 1), scalar_exit);

                PHINode* input_elems = ctx_.builder().CreatePHI(ctx_.ptrType(), 3, "grad_elems");
                input_elems->addIncoming(svec_elems_typed, svec_exit);
                input_elems->addIncoming(tconv_elems_typed, tensor_exit);
                input_elems->addIncoming(scalar_elem_typed, scalar_exit);

                // Allocate result tensor using arena with header
                Value* typed_result = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

                Value* result_dims_size = ConstantInt::get(ctx_.int64Type(), 8);
                Value* result_dims_ptr = ctx_.builder().CreateCall(arena_allocate_func, {arena_ptr, result_dims_size});
                Value* typed_result_dims = ctx_.builder().CreatePointerCast(result_dims_ptr, ctx_.builder().getPtrTy());
                ctx_.builder().CreateStore(n, typed_result_dims);
                ctx_.builder().CreateStore(typed_result_dims, ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result, 0));
                ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result, 1));
                ctx_.builder().CreateStore(n, ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result, 3));

                Value* result_elems_size = ctx_.builder().CreateMul(n, ConstantInt::get(ctx_.int64Type(), sizeof(double)));
                Value* result_elems_ptr = ctx_.builder().CreateCall(arena_allocate_func, {arena_ptr, result_elems_size});
                Value* typed_result_elems = ctx_.builder().CreatePointerCast(result_elems_ptr, ctx_.builder().getPtrTy());
                ctx_.builder().CreateStore(typed_result_elems, ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result, 2));

                // M1 CONSOLIDATION: Allocate Scheme vector for dual numbers with header
                // arena_allocate_vector_with_header creates: [header(8)] + [length(8)] + [elements]
                Value* dual_vec = ctx_.builder().CreateCall(mem_.getArenaAllocateVectorWithHeader(),
                    {arena_ptr, n});
                ctx_.builder().CreateStore(n, dual_vec);
                Value* dual_elems = ctx_.builder().CreateGEP(ctx_.int8Type(), dual_vec, ConstantInt::get(ctx_.int64Type(), 8));
                Value* dual_elems_typed = ctx_.builder().CreatePointerCast(dual_elems, ctx_.ptrType());

                // ESH-0093: participate in the runtime perturbation-level
                // protocol (see the scheme-vector path below for the full
                // rationale): seed this level's slot, push the level around
                // the closure call, extract this level's coefficient.
                Value* rt_pert_level = adPertLevelLoad();
                Value* rt_lvl0 = ctx_.builder().CreateICmpEQ(rt_pert_level,
                    ConstantInt::get(ctx_.int64Type(), 0));

                // Outer loop: for each dimension i, compute partial derivative
                Value* dim_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "grad_dim_i");
                ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), dim_counter);

                BasicBlock* dim_cond = BasicBlock::Create(ctx_.context(), "grad_dim_cond", current_func);
                BasicBlock* dim_body = BasicBlock::Create(ctx_.context(), "grad_dim_body", current_func);
                BasicBlock* dim_end = BasicBlock::Create(ctx_.context(), "grad_dim_end", current_func);

                ctx_.builder().CreateBr(dim_cond);

                ctx_.builder().SetInsertPoint(dim_cond);
                Value* dim_i = ctx_.builder().CreateLoad(ctx_.int64Type(), dim_counter);
                ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(dim_i, n), dim_body, dim_end);

                ctx_.builder().SetInsertPoint(dim_body);

                // Inner loop: create dual vector with tangent=1 at dim_i
                Value* inner_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "grad_inner_j");
                ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), inner_counter);

                BasicBlock* inner_cond = BasicBlock::Create(ctx_.context(), "grad_inner_cond", current_func);
                BasicBlock* inner_body = BasicBlock::Create(ctx_.context(), "grad_inner_body", current_func);
                BasicBlock* inner_end = BasicBlock::Create(ctx_.context(), "grad_inner_end", current_func);

                ctx_.builder().CreateBr(inner_cond);

                ctx_.builder().SetInsertPoint(inner_cond);
                Value* inner_j = ctx_.builder().CreateLoad(ctx_.int64Type(), inner_counter);
                ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(inner_j, n), inner_body, inner_end);

                ctx_.builder().SetInsertPoint(inner_body);
                // Load primal value at position j from input elements
                Value* in_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), input_elems, inner_j);
                Value* in_elem = ctx_.builder().CreateLoad(ctx_.taggedValueType(), in_elem_ptr);
                Value* primal_val = tagged_.unpackDouble(in_elem);

                // Set tangent: 1.0 if j == i, else 0.0
                Value* is_active = ctx_.builder().CreateICmpEQ(inner_j, dim_i);
                Value* tangent = ctx_.builder().CreateSelect(is_active,
                    ConstantFP::get(ctx_.doubleType(), 1.0),
                    ConstantFP::get(ctx_.doubleType(), 0.0));

                // ESH-0093: seed THIS level's slot (e1 at level 0, e2 nested)
                Value* rt_zero_t = ConstantFP::get(ctx_.doubleType(), 0.0);
                Value* rt_t_e1 = ctx_.builder().CreateSelect(rt_lvl0, tangent, rt_zero_t);
                Value* rt_t_e2 = ctx_.builder().CreateSelect(rt_lvl0, rt_zero_t, tangent);
                Value* dual_num = makeDual4(ctx_, primal_val, rt_t_e1, rt_t_e2, rt_zero_t);
                Value* dual_tagged = packDualToTagged(dual_num);
                Value* dual_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), dual_elems_typed, inner_j);
                ctx_.builder().CreateStore(dual_tagged, dual_elem_ptr);

                ctx_.builder().CreateStore(ctx_.builder().CreateAdd(inner_j, ConstantInt::get(ctx_.int64Type(), 1)), inner_counter);
                ctx_.builder().CreateBr(inner_cond);

                ctx_.builder().SetInsertPoint(inner_end);

                // M1 CONSOLIDATION: Pack dual vector as HEAP_PTR (header contains HEAP_SUBTYPE_VECTOR)
                Value* dual_vec_tagged = tagged_.packPtr(
                    ctx_.builder().CreatePtrToInt(dual_vec, ctx_.int64Type()),
                    ESHKOL_VALUE_HEAP_PTR);

                // Call function via closure dispatch.
                // bug-OO: a runtime function value may be a multi-arg scalar function
                // (e.g. (lambda (x y) ...)) rather than a single vectorized arg. The
                // direct-named path knows the arity at compile time and unpacks; here we
                // only have a runtime closure, so read its declared input arity (byte at
                // offset 33 of the callable struct — see codegenClosureCall) and, for
                // arity k>1, pass the k dual elements as k separate scalar args. Arity 1
                // (or unknown/variadic) keeps the vectorized single-arg form. Without this
                // a multi-arg f was called with one vector arg, leaving its remaining
                // params uninitialised → SIGSEGV.
                Value* clo_ptr_i64 = tagged_.unpackInt64(closure_val);
                Value* clo_ptr = ctx_.builder().CreateIntToPtr(clo_ptr_i64, ctx_.ptrType());
                Value* clo_arity_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), clo_ptr,
                    ConstantInt::get(ctx_.int64Type(), 33));
                Value* clo_arity = ctx_.builder().CreateZExt(
                    ctx_.builder().CreateLoad(ctx_.int8Type(), clo_arity_ptr), ctx_.int64Type());

                const int GRAD_MAX_ARITY = 8;
                BasicBlock* gcall_default = BasicBlock::Create(ctx_.context(), "grad_call_vec", current_func);
                BasicBlock* gcall_merge = BasicBlock::Create(ctx_.context(), "grad_call_merge", current_func);
                // ESH-0093: the callee body runs one forward level deeper
                adPertLevelStore(ctx_.builder().CreateAdd(rt_pert_level,
                    ConstantInt::get(ctx_.int64Type(), 1)));
                SwitchInst* gcall_sw = ctx_.builder().CreateSwitch(clo_arity, gcall_default, GRAD_MAX_ARITY);

                std::vector<std::pair<BasicBlock*, Value*>> gcall_variants;
                for (int k = 1; k <= GRAD_MAX_ARITY; k++) {
                    BasicBlock* case_bb = BasicBlock::Create(ctx_.context(),
                        "grad_call_a" + std::to_string(k), current_func);
                    gcall_sw->addCase(ConstantInt::get(ctx_.int64Type(), k), case_bb);
                    ctx_.builder().SetInsertPoint(case_bb);
                    std::vector<Value*> cargs;
                    if (k == 1) {
                        cargs.push_back(dual_vec_tagged);  // vectorized single-arg function
                    } else {
                        for (int j = 0; j < k; j++) {
                            Value* ep = ctx_.builder().CreateGEP(ctx_.taggedValueType(), dual_elems_typed,
                                ConstantInt::get(ctx_.int64Type(), j));
                            cargs.push_back(ctx_.builder().CreateLoad(ctx_.taggedValueType(), ep));
                        }
                    }
                    Value* r = closure_call_callback_(closure_val, cargs, "autodiff", callback_context_);
                    BasicBlock* cexit = ctx_.builder().GetInsertBlock();  // closure call may add blocks
                    ctx_.builder().CreateBr(gcall_merge);
                    gcall_variants.push_back({cexit, r});
                }
                // Default (arity 0 / variadic / unreadable): vectorized fallback.
                ctx_.builder().SetInsertPoint(gcall_default);
                Value* gdef_r = closure_call_callback_(closure_val,
                    std::vector<Value*>{dual_vec_tagged}, "autodiff", callback_context_);
                BasicBlock* gdef_exit = ctx_.builder().GetInsertBlock();
                ctx_.builder().CreateBr(gcall_merge);
                gcall_variants.push_back({gdef_exit, gdef_r});

                ctx_.builder().SetInsertPoint(gcall_merge);
                PHINode* call_result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), gcall_variants.size());
                for (auto& cv : gcall_variants) call_result->addIncoming(cv.second, cv.first);

                // ESH-0093: pop the perturbation level
                adPertLevelStore(rt_pert_level);

                // CONSTANT RESULT FIX: Check if result is a dual number before unpacking
                // If function returns a constant (not using its argument), it won't be dual
                Value* rt_result_type = tagged_.getType(call_result);
                Value* rt_result_base = tagged_.getBaseType(rt_result_type);
                Value* rt_is_dual = ctx_.builder().CreateICmpEQ(rt_result_base,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));

                BasicBlock* rt_dual_bb = BasicBlock::Create(ctx_.context(), "grad_rt_dual", current_func);
                BasicBlock* rt_const_bb = BasicBlock::Create(ctx_.context(), "grad_rt_const", current_func);
                BasicBlock* rt_merge_bb = BasicBlock::Create(ctx_.context(), "grad_rt_merge", current_func);

                ctx_.builder().CreateCondBr(rt_is_dual, rt_dual_bb, rt_const_bb);

                // Dual path: extract THIS level's coefficient (e1 at level 0,
                // e2 nested). ESH-0093: composes with an inner derivative's
                // e2-slice return — see the scheme-vector path for details.
                ctx_.builder().SetInsertPoint(rt_dual_bb);
                Value* rt_result_dual = unpackDualFromTagged(call_result);
                Value* rt_dual_deriv = ctx_.builder().CreateSelect(rt_lvl0,
                    dualField(ctx_, rt_result_dual, 1), dualField(ctx_, rt_result_dual, 2));
                ctx_.builder().CreateBr(rt_merge_bb);
                BasicBlock* rt_dual_exit = ctx_.builder().GetInsertBlock();

                // Constant path: derivative is 0.0
                ctx_.builder().SetInsertPoint(rt_const_bb);
                Value* rt_zero_deriv = ConstantFP::get(ctx_.doubleType(), 0.0);
                ctx_.builder().CreateBr(rt_merge_bb);
                BasicBlock* rt_const_exit = ctx_.builder().GetInsertBlock();

                // Merge paths
                ctx_.builder().SetInsertPoint(rt_merge_bb);
                PHINode* deriv = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "grad_deriv");
                deriv->addIncoming(rt_dual_deriv, rt_dual_exit);
                deriv->addIncoming(rt_zero_deriv, rt_const_exit);

                // Store derivative in result tensor
                Value* result_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_result_elems, dim_i);
                Value* deriv_bits = ctx_.builder().CreateBitCast(deriv, ctx_.int64Type());
                ctx_.builder().CreateStore(deriv_bits, result_elem_ptr);

                ctx_.builder().CreateStore(ctx_.builder().CreateAdd(dim_i, ConstantInt::get(ctx_.int64Type(), 1)), dim_counter);
                ctx_.builder().CreateBr(dim_cond);

                ctx_.builder().SetInsertPoint(dim_end);
                Value* result_int = ctx_.builder().CreatePtrToInt(typed_result, ctx_.int64Type());
                Value* result_tagged = tagged_.packPtr(result_int, ESHKOL_VALUE_HEAP_PTR);
                ctx_.builder().CreateStore(result_tagged, rt_result_slot);
                ctx_.builder().CreateBr(grad_rt_done);

                ctx_.builder().SetInsertPoint(grad_rt_done);
                return ctx_.builder().CreateLoad(ctx_.taggedValueType(), rt_result_slot);
}

llvm::Value* AutodiffCodegen::gradient(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->gradient_op.function) {
        eshkol_error("Invalid gradient operation - missing function");
        return nullptr;
    }

    // Higher-order form: (gradient f) returns a closure that computes gradients
    if (!op->gradient_op.point) {
        return gradientHigherOrder(op);
    }

    // Resolve function (lambda or function reference)
    Value* func = resolve_lambda_callback_(op->gradient_op.function, 0, callback_context_);

    // RUNTIME FUNCTION PARAMETER FIX: Handle functions passed as parameters
    // For gradient with runtime function parameters, we need to use a different approach
    // since gradient requires knowing the function structure at compile time.
    // For now, we'll check if the function AST is a variable and look it up.
    if (!func) {
        const eshkol_ast_t* func_ast = op->gradient_op.function;
        if (func_ast && func_ast->type == ESHKOL_VAR) {
            std::string func_name = func_ast->variable.id;
            eshkol_debug("gradient: checking runtime function parameter '%s'", func_name.c_str());

            // Check if this is a function parameter or captured value
            Value* var_value = nullptr;

            // NESTED FUNCTION FIX: First check if there's a GlobalVariable for this capture
            // This handles nested functions where captures are stored in GlobalVariables
            Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
            std::string capture_key = current_func->getName().str() + "_capture_" + func_name;
            auto gv_it = global_symbol_table_->find(capture_key);
            if (gv_it != global_symbol_table_->end() && isa<GlobalVariable>(gv_it->second)) {
                var_value = gv_it->second;
            }

            // If not found as a capture, try regular symbol_table lookup
            if (!var_value) {
                auto local_it = symbol_table_->find(func_name);
                if (local_it != symbol_table_->end()) {
                    var_value = local_it->second;
                } else {
                    auto global_it = global_symbol_table_->find(func_name);
                    if (global_it != global_symbol_table_->end()) {
                        var_value = global_it->second;
                    }
                }
            }

            // REPL MODE: cross-evaluation symbol registry (functions defined in prior JIT modules)
            if (!var_value && repl_mode_enabled_ && *repl_mode_enabled_) {
                std::lock_guard<std::mutex> lock(*repl_mutex_);
                auto repl_it = repl_symbol_addresses_->find(func_name);
                if (repl_it != repl_symbol_addresses_->end()) {
                    GlobalVariable* gv = ctx_.module().getGlobalVariable(func_name);
                    if (!gv) {
                        gv = new GlobalVariable(ctx_.module(), ctx_.taggedValueType(), false,
                                                GlobalValue::ExternalLinkage, nullptr, func_name);
                    }
                    var_value = gv;
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // UNIFIED RUNTIME GRADIENT PATH
            // Consolidates 3 duplicate paths (Argument, Pointer, GlobalVariable)
            // into a single resolution + shared forward-mode computation.
            // ═══════════════════════════════════════════════════════════════

            // Step 1: Resolve closure value from var_value
            Value* closure_val = nullptr;

            if (var_value && isa<Argument>(var_value) && !var_value->getType()->isPointerTy()) {
                // Direct Argument — may need capture resolution for nested functions
                Argument* arg = cast<Argument>(var_value);
                Function* arg_parent = arg->getParent();
                Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

                if (arg_parent != current_func) {
                    // From different function — find in current function's captures
                    bool found_in_captures = false;
                    for (auto& curr_arg : current_func->args()) {
                        std::string arg_name = curr_arg.getName().str();
                        if (arg_name == "captured_" + func_name) {
                            if (curr_arg.getType()->isPointerTy()) {
                                closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), &curr_arg);
                            } else {
                                closure_val = &curr_arg;
                            }
                            found_in_captures = true;
                            break;
                        }
                    }
                    if (!found_in_captures) {
                        std::string capture_key = current_func->getName().str() + "_capture_" + func_name;
                        auto cap_it = global_symbol_table_->find(capture_key);
                        if (cap_it != global_symbol_table_->end() && isa<GlobalVariable>(cap_it->second)) {
                            closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), cap_it->second);
                            found_in_captures = true;
                        } else {
                            auto var_cap_it = global_symbol_table_->find(func_name);
                            if (var_cap_it != global_symbol_table_->end() && isa<GlobalVariable>(var_cap_it->second)) {
                                closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_cap_it->second);
                                found_in_captures = true;
                            }
                        }
                    }
                    if (!found_in_captures) {
                        eshkol_error("gradient: could not find capture for '%s'", func_name.c_str());
                        return nullptr;
                    }
                } else {
                    closure_val = var_value;
                }
            } else if (var_value && isa<Argument>(var_value) && var_value->getType()->isPointerTy()) {
                // Pointer-type captured Argument — load the tagged value
                closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
            } else if (var_value && isa<GlobalVariable>(var_value)) {
                // GlobalVariable — may be from a different module (REPL mode), so ensure we
                // reference the symbol via the CURRENT module with ExternalLinkage if needed.
                GlobalVariable* gv = cast<GlobalVariable>(var_value);
                if (gv->getParent() != &ctx_.module()) {
                    GlobalVariable* cur_gv = ctx_.module().getGlobalVariable(gv->getName());
                    if (!cur_gv) {
                        cur_gv = new GlobalVariable(ctx_.module(), gv->getValueType(), false,
                                                    GlobalValue::ExternalLinkage, nullptr, gv->getName());
                    }
                    gv = cur_gv;
                }
                closure_val = ctx_.builder().CreateLoad(gv->getValueType(), gv);
            } else if (var_value && isa<AllocaInst>(var_value) && var_value->getType()->isPointerTy()) {
                AllocaInst* alloca = cast<AllocaInst>(var_value);
                if (alloca->getAllocatedType() == ctx_.taggedValueType()) {
                    closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                }
            } else if (var_value && isa<LoadInst>(var_value) && var_value->getType() == ctx_.taggedValueType()) {
                closure_val = var_value;
            }

            // Step 2: Shared forward-mode gradient computation
            if (closure_val) {
                Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

                // Get the input point
                Value* point_val = codegen_ast_callback_(op->gradient_op.point, callback_context_);
                if (!point_val) {
                    eshkol_error("Failed to evaluate gradient point");
                    return nullptr;
                }

                // Ensure point is tagged
                if (point_val->getType() != ctx_.taggedValueType()) {
                    if (point_val->getType()->isIntegerTy(64)) {
                        if (op->gradient_op.point->type == ESHKOL_TENSOR) {
                            point_val = tagged_.packPtr(point_val, ESHKOL_VALUE_HEAP_PTR);
                        } else {
                            point_val = tagged_.packInt64(point_val, true);
                        }
                    } else if (point_val->getType()->isDoubleTy()) {
                        point_val = tagged_.packDouble(point_val);
                    }
                }

                // Note: current_func already defined above for capture lookup

                // Delegate to the shared exact-AD runtime-closure gradient so the
                // wrapped / callable-parameter form uses the identical machinery as
                // the higher-order closure body (no finite-difference anywhere).
                Value* rt_grad = emitRuntimeClosureGradient(closure_val, point_val);
                if (!rt_grad) return nullptr;
                return rt_grad;
            } // end if (closure_val)

        }
        eshkol_error("Failed to resolve function for gradient computation");
        return nullptr;
    }

    Function* func_ptr = dyn_cast<Function>(func);

    if (!func_ptr) {
        eshkol_error("Gradient operator requires actual function, got non-function value");
        return nullptr;
    }
    
    // Evaluate point to get input vector
    Value* vector_val_raw = codegen_ast_callback_(op->gradient_op.point, callback_context_);
    if (!vector_val_raw) {
        eshkol_error("Failed to evaluate gradient evaluation point");
        return nullptr;
    }
    
    // CRITICAL FIX: Ensure input is tagged_value (codegenAST can return raw types for literals)
    // Tensor/vector codegen returns raw ptr-as-int64, NOT tagged values.
    // We must detect the AST type to pack correctly (HEAP_PTR vs INT64).
    Value* vector_val;
    if (vector_val_raw->getType() == ctx_.taggedValueType()) {
        vector_val = vector_val_raw; // Already tagged
    } else if (vector_val_raw->getType()->isIntegerTy(64) &&
               op->gradient_op.point->type == ESHKOL_TENSOR) {
        // Tensor literal: codegenTensor returns ptr-as-int64; wrap as HEAP_PTR
        // so the input dispatch correctly detects the tensor subtype
        vector_val = tagged_.packPtr(vector_val_raw, ESHKOL_VALUE_HEAP_PTR);
    } else if (vector_val_raw->getType()->isIntegerTy(64)) {
        vector_val = tagged_.packInt64(vector_val_raw, true); // Pack int64
    } else if (vector_val_raw->getType()->isDoubleTy()) {
        vector_val = tagged_.packDouble(vector_val_raw); // Pack double
    } else {
        // direct packing
        vector_val = tagged_.packInt64(vector_val_raw, true); // Pack other types
    }
    
    // ════════════════════════════════════════════════════════════════════
    // ESH-0070: forward-mode jet fast path for SCALAR single-variable gradient.
    //
    // For an arity-1 function at a SCALAR (or dual, when nested) point, the
    // gradient IS the derivative, so we compute it with the exact forward-mode
    // 4-jet — the same machinery `derivative` uses — instead of the reverse-mode
    // tape + the degree/value-ratio "double backward" reconstruction further
    // below. That reconstruction is exact only for pure monomials; for anything
    // with +/- terms a NESTED gradient produced garbage that compounded with
    // each named-let iteration (the Noesis blow-up). Forward mode is exact for
    // 2-level nesting, allocates O(1) (no per-op tape node — fixes the per-call
    // leak), and the runtime perturbation level (seedForwardAndPush) makes the
    // nesting correct across the named-let / function-call boundary.
    //
    // A VECTOR point (arity-1 functions like (lambda (v) (dot v v)), or any
    // multi-parameter function) keeps the existing reverse-mode vector path.
    // The scalar-vs-vector decision is made at RUNTIME on the point's tag, so a
    // bare-variable point ((gradient log x) vs (gradient f vec)) is handled
    // correctly either way.
    // ════════════════════════════════════════════════════════════════════
    BasicBlock* grad_unified_exit = nullptr;
    AllocaInst* grad_result_slot = nullptr;
    {
        uint64_t fwd_arity = 0;
        std::string fwd_key = func_ptr->getName().str();
        auto fwd_rv = fwd_key.rfind("__rv");
        if (fwd_rv != std::string::npos && fwd_rv + 4 < fwd_key.size() &&
            fwd_key.find_first_not_of("0123456789", fwd_rv + 4) == std::string::npos) {
            fwd_key.erase(fwd_rv);
        }
        auto fwd_ait = function_arity_table_->find(fwd_key);
        if (fwd_ait != function_arity_table_->end()) fwd_arity = fwd_ait->second;
        fwd_arity = adResolveValueArity(func_ptr, fwd_arity);

        // Only a SCALAR single-variable function takes the forward path. A body
        // that flows the input through a tensor op (batch-norm/layer-norm/…) has
        // no scalar forward-mode dual and must use the reverse-mode tape. Decided
        // from the SOURCE AST (not the emitted IR), so it is not confused by
        // tensor allocations an inline nested gradient's own machinery emits. A
        // vector-valued point is still handled at runtime below (reverse branch),
        // so this only needs to catch the scalar-point-through-tensor case.
        // ESH-0078: a NAMED function passed by var (e.g. (gradient L x)) has no
        // reachable source AST at the gradient op, so the original code fell back
        // to a coarse IR substring scan (adFunctionUsesTensors). That scan is
        // FALSE-POSITIVE for every scalar function: tagged +/-/*// arithmetic
        // unconditionally emits runtime tensor-dispatch helpers
        // (eshkol_tensor_operand_checked / arena_allocate_tensor_with_header /
        // eshkol_tensor_result_dtype_*), so a pure-scalar function like
        // (define (L w) (* w w w)) was wrongly forced onto the reverse-mode path
        // and a nested (gradient (lambda (y) (gradient L y)) x) returned 0.
        // Resolve the function's registered body AST and run the SAME
        // source-level analysis inline lambdas get; fall back to the IR scan only
        // when no source AST is available (cross-module / stdlib functions).
        bool fn_uses_tensors = adAstUsesTensorOps(op->gradient_op.function);
        if (!fn_uses_tensors && op->gradient_op.function->type == ESHKOL_VAR) {
            const eshkol_ast_t* body_ast = nullptr;
            if (function_body_ast_) {
                auto bit = function_body_ast_->find(fwd_key);
                if (bit == function_body_ast_->end())
                    bit = function_body_ast_->find(op->gradient_op.function->variable.id);
                if (bit != function_body_ast_->end()) body_ast = bit->second;
            }
            fn_uses_tensors = body_ast ? adAstUsesTensorOps(body_ast)
                                       : adFunctionUsesTensors(func_ptr);
        }
        if (fwd_arity == 1 && !fn_uses_tensors) {
            Function* cur_fn = ctx_.builder().GetInsertBlock()->getParent();
            grad_result_slot = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_fwd_result");
            BasicBlock* grad_fwd_jet = BasicBlock::Create(ctx_.context(), "grad_fwd_jet", cur_fn);
            BasicBlock* grad_reverse_entry = BasicBlock::Create(ctx_.context(), "grad_reverse_entry", cur_fn);
            grad_unified_exit = BasicBlock::Create(ctx_.context(), "grad_unified_exit", cur_fn);

            // Use forward mode when the point is a scalar (DOUBLE/INT64) or an
            // existing dual (nested gradient carrying an outer perturbation).
            Value* fwd_bt = tagged_.getBaseType(tagged_.getType(vector_val));
            Value* fwd_is_d = ctx_.builder().CreateICmpEQ(fwd_bt, ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
            Value* fwd_is_i = ctx_.builder().CreateICmpEQ(fwd_bt, ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
            Value* fwd_is_dual = ctx_.builder().CreateICmpEQ(fwd_bt, ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
            Value* use_fwd = ctx_.builder().CreateOr(ctx_.builder().CreateOr(fwd_is_d, fwd_is_i), fwd_is_dual);
            ctx_.builder().CreateCondBr(use_fwd, grad_fwd_jet, grad_reverse_entry);

            // ---- forward-mode jet path ----
            ctx_.builder().SetInsertPoint(grad_fwd_jet);
            Value* fwd_level = nullptr;
            Value* fwd_seed = seedForwardAndPush(vector_val, &fwd_level);
            std::vector<Value*> fwd_args = {fwd_seed};
            resolveGradientCaptures(func_ptr, fwd_args, "fwd-jet");
            Value* fwd_call = ctx_.builder().CreateCall(func_ptr, fwd_args);
            Value* fwd_res = popAndExtractForward(fwd_call, fwd_level);
            ctx_.builder().CreateStore(fwd_res, grad_result_slot);
            ctx_.builder().CreateBr(grad_unified_exit);

            // ---- reverse-mode vector path: the rest of this function ----
            ctx_.builder().SetInsertPoint(grad_reverse_entry);
        }
    }

    // Alloca for effective svec input (updated by tensor→svec conversion)
    Value* svec_input_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "svec_input");
    ctx_.builder().CreateStore(vector_val, svec_input_ptr);

    // SCALAR→VECTOR AUTO-PROMOTION: Detect input type BEFORE tensor structure access
    // This prevents segfault when users pass scalars like 3.0 instead of vectors like #(3.0)

    // Get current function for basic blocks
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Get arena for OALR-compliant tensor allocation (used throughout gradient computation)
    Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // ESH-0235: decide, at compile time, whether a HEAP_SUBTYPE_VECTOR
    // ((vector …)-constructed) point must be seeded on the REVERSE-mode tensor
    // tape rather than the forward-mode dual scheme-vector path. A #(…) reader
    // literal and a (tensor …) value both lower to a TENSOR subtype and, for a
    // single-argument tensor-op function, take the reverse-mode `vector_input`
    // path (which seeds every element as an AD variable). A (vector …) point is
    // a genuine Scheme vector and would instead fall to the forward-mode dual
    // path, where a tensor op (tensor-dot/-mul/-sum/-matmul/…) reads ONLY the
    // primal double out of each dual-tagged element and silently drops the
    // tangent — a WRONG all-zero gradient. So when the differentiated function
    // flows the input through a tensor op, convert the Scheme vector to a 1-D
    // tensor and route it through the SAME reverse-mode seeding as the
    // #(…)/(tensor …) point. The forward-mode scheme-vector path is retained for
    // scalar / vector-ref bodies because it (and only it) carries the outer
    // perturbation of a NESTED gradient-of-gradient over a vector point
    // (ESH-0096); reverse-seeding those would drop the second-order term. Tensor
    // use is detected THROUGH one layer of named-function indirection so a
    // wrapper (lambda (z) (loss z)) over a named tensor loss is caught too.
    // Multi-argument (arity>1) vector points keep the forward / scheme-vector
    // path below (the inverse tensor→svec conversion handles the multi-param
    // case). Mirrors the arity>1 tensor→svec conversion.
    uint64_t grad_arity_early = 0;
    {
        std::string key = func_ptr->getName().str();
        auto rv = key.rfind("__rv");
        if (rv != std::string::npos && rv + 4 < key.size() &&
            key.find_first_not_of("0123456789", rv + 4) == std::string::npos) {
            key.erase(rv);
        }
        auto ait = function_arity_table_->find(key);
        if (ait != function_arity_table_->end()) grad_arity_early = ait->second;
    }
    std::unordered_set<std::string> grad_tensor_visited;
    bool grad_fn_uses_tensors =
        adAstUsesTensorOps(op->gradient_op.function, function_body_ast_, &grad_tensor_visited);
    if (!grad_fn_uses_tensors && op->gradient_op.function->type == ESHKOL_VAR) {
        const eshkol_ast_t* gbody = nullptr;
        std::string gkey = func_ptr->getName().str();
        auto grv = gkey.rfind("__rv");
        if (grv != std::string::npos && grv + 4 < gkey.size() &&
            gkey.find_first_not_of("0123456789", grv + 4) == std::string::npos) {
            gkey.erase(grv);
        }
        if (function_body_ast_) {
            auto bit = function_body_ast_->find(gkey);
            if (bit == function_body_ast_->end())
                bit = function_body_ast_->find(op->gradient_op.function->variable.id);
            if (bit != function_body_ast_->end()) gbody = bit->second;
        }
        grad_tensor_visited.clear();
        grad_fn_uses_tensors = gbody
            ? adAstUsesTensorOps(gbody, function_body_ast_, &grad_tensor_visited)
            : adFunctionUsesTensors(func_ptr);
    }
    bool grad_vec_point_reverse = (grad_arity_early <= 1) && grad_fn_uses_tensors;

    // Extract type from input (may be DOUBLE, INT64, TENSOR_PTR, or AD_NODE_PTR for nested gradients)
    Value* input_type = tagged_.getType(vector_val);
    Value* input_base_type = tagged_.getBaseType(input_type);

    // DOUBLE BACKWARD: Check if input is an AD node (from outer gradient)
    // This happens in nested gradients like (gradient (lambda (y) (gradient f y)) x)
    Value* is_ad_node_input = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));

    // Check if input is scalar (INT64 or DOUBLE)
    Value* is_int64 = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
    Value* is_double = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    Value* is_scalar = ctx_.builder().CreateOr(is_int64, is_double);

    // M1 Migration: Check if input is Scheme vector (HEAP_PTR with HEAP_SUBTYPE_VECTOR) or legacy VECTOR_PTR
    // First check for HEAP_PTR (consolidated format)
    Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    // Legacy VECTOR_PTR check
    Value* is_legacy_vector = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    // Branch: AD node input (nested), scalar input (promotion), scheme vector (convert), or tensor input (normal)
    BasicBlock* ad_node_input = BasicBlock::Create(ctx_.context(), "grad_ad_node_input", current_func);
    BasicBlock* scalar_input = BasicBlock::Create(ctx_.context(), "grad_scalar_input", current_func);
    BasicBlock* scheme_vector_input = BasicBlock::Create(ctx_.context(), "grad_scheme_vector_input", current_func);
    BasicBlock* vector_input = BasicBlock::Create(ctx_.context(), "grad_vector_input", current_func);
    // ESH-0235: svec→tensor bridge for a (vector …) point into the reverse path.
    BasicBlock* grad_vec_to_tensor = BasicBlock::Create(ctx_.context(), "grad_vec_to_tensor", current_func);
    BasicBlock* grad_merge_input = BasicBlock::Create(ctx_.context(), "grad_merge_input", current_func);
    // Create grad_done early so scheme_vector_input path can branch to it
    BasicBlock* grad_done = BasicBlock::Create(ctx_.context(), "grad_done", current_func);

    // First check if AD node (nested gradient)
    BasicBlock* check_scalar = BasicBlock::Create(ctx_.context(), "grad_check_scalar", current_func);
    ctx_.builder().CreateCondBr(is_ad_node_input, ad_node_input, check_scalar);

    // NESTED GRADIENT (AD_NODE_PTR input): Extract value and wrap in tensor for uniform handling
    ctx_.builder().SetInsertPoint(ad_node_input);
    eshkol_debug("Gradient: detected AD_NODE_PTR input (nested gradient)");

    // Extract the AD node pointer
    Value* outer_ad_node = tagged_.unpackPtr(vector_val);

    // DOUBLE BACKWARD: Store outer AD node in global for later use
    // This allows the backward pass to connect to outer computation graph
    ctx_.builder().CreateStore(outer_ad_node, ctx_.outerAdNodeStorage());

    // Extract the VALUE from the AD node (field 1)
    Value* ad_value_ptr = ctx_.builder().CreateStructGEP(ctx_.adNodeType(), outer_ad_node, 1);
    Value* ad_value = ctx_.builder().CreateLoad(ctx_.doubleType(), ad_value_ptr);

    // Create a 1D tensor containing this value via arena (OALR compliant - no malloc)
    Value* typed_ad_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set up 1D tensor structure
    Value* nested_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* nested_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, nested_dims_size});
    Value* typed_ad_dims = ctx_.builder().CreatePointerCast(nested_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), typed_ad_dims);

    ctx_.builder().CreateStore(typed_ad_dims,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor, 1));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor, 3));

    Value* nested_elems_size = ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    Value* nested_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, nested_elems_size});
    Value* typed_nested_elems = ctx_.builder().CreatePointerCast(nested_elems_ptr, ctx_.builder().getPtrTy());
    Value* nested_value_as_int64 = ctx_.builder().CreateBitCast(ad_value, ctx_.int64Type());
    ctx_.builder().CreateStore(nested_value_as_int64, typed_nested_elems);
    ctx_.builder().CreateStore(typed_nested_elems,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor, 2));

    Value* nested_tensor_int = ctx_.builder().CreatePtrToInt(typed_ad_tensor, ctx_.int64Type());
    Value* ad_promoted_tagged = tagged_.packPtr(nested_tensor_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(grad_merge_input);
    BasicBlock* ad_node_exit = ctx_.builder().GetInsertBlock();

    // Check for scalar
    ctx_.builder().SetInsertPoint(check_scalar);

    // DOUBLE BACKWARD: Clear outer AD node storage for non-nested case
    ctx_.builder().CreateStore(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())),
        ctx_.outerAdNodeStorage());

    // Check: scalar → scalar_input, heap_ptr (check subtype) → maybe scheme_vector, else → vector_input (tensor)
    BasicBlock* check_heap_ptr = BasicBlock::Create(ctx_.context(), "grad_check_heap_ptr", current_func);
    ctx_.builder().CreateCondBr(is_scalar, scalar_input, check_heap_ptr);

    // M1 Migration: Check for HEAP_PTR and dispatch based on subtype
    ctx_.builder().SetInsertPoint(check_heap_ptr);
    BasicBlock* heap_ptr_dispatch = BasicBlock::Create(ctx_.context(), "grad_heap_dispatch", current_func);
    BasicBlock* check_legacy_vector = BasicBlock::Create(ctx_.context(), "grad_check_legacy_vec", current_func);
    ctx_.builder().CreateCondBr(is_heap_ptr, heap_ptr_dispatch, check_legacy_vector);

    // HEAP_PTR dispatch - read subtype from header
    ctx_.builder().SetInsertPoint(heap_ptr_dispatch);
    Value* grad_heap_ptr = tagged_.unpackPtr(vector_val);
    Value* grad_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), grad_heap_ptr, ConstantInt::get(ctx_.int64Type(), -8));
    Value* grad_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), grad_header_ptr);
    Value* is_vec_subtype_grad = ctx_.builder().CreateICmpEQ(grad_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    Value* is_tensor_subtype_grad = ctx_.builder().CreateICmpEQ(grad_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));
    Value* is_cons_subtype_grad = ctx_.builder().CreateICmpEQ(grad_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_CONS));
    BasicBlock* grad_check_cons = BasicBlock::Create(ctx_.context(), "grad_check_cons", current_func);
    BasicBlock* grad_list_to_svec = BasicBlock::Create(ctx_.context(), "grad_list_to_svec", current_func);
    BasicBlock* grad_check_tensor = BasicBlock::Create(ctx_.context(), "grad_check_tensor", current_func);
    // ESH-0235: a single-arg tensor-op function must seed a (vector …) point on
    // the reverse-mode tape (via svec→tensor), identical to a #(…)/(tensor …)
    // point; otherwise the forward-mode dual path drops the tangent through the
    // tensor op and returns a silent all-zero gradient.
    BasicBlock* grad_vec_subtype_target =
        grad_vec_point_reverse ? grad_vec_to_tensor : scheme_vector_input;
    ctx_.builder().CreateCondBr(is_vec_subtype_grad, grad_vec_subtype_target, grad_check_cons);

    // A (list …) input is a cons cell (HEAP_SUBTYPE_CONS): convert it to a
    // Scheme vector so it goes through the same path as a (vector …) input.
    // Without this, a multi-parameter gradient on a list crashed — the cons
    // cell fell through to the vector path and was misread as [length][elems].
    ctx_.builder().SetInsertPoint(grad_check_cons);
    ctx_.builder().CreateCondBr(is_cons_subtype_grad, grad_list_to_svec, grad_check_tensor);

    ctx_.builder().SetInsertPoint(grad_list_to_svec);
    {
        // Entry-block alloca for the list head so a gradient inside a runtime
        // loop doesn't re-allocate each iteration.
        llvm::IRBuilder<> entryB(&current_func->getEntryBlock(),
                                 current_func->getEntryBlock().begin());
        Value* list_slot = entryB.CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_list_head");
        ctx_.builder().CreateStore(vector_val, list_slot);
        Value* l2s_arena = ctx_.builder().CreateLoad(
            PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
        llvm::Function* l2s_fn = ctx_.module().getFunction("eshkol_list_to_svec");
        if (!l2s_fn) {
            llvm::FunctionType* l2s_ty = llvm::FunctionType::get(
                ctx_.builder().getPtrTy(),
                {ctx_.builder().getPtrTy(), ctx_.builder().getPtrTy()}, false);
            l2s_fn = llvm::Function::Create(l2s_ty, llvm::Function::ExternalLinkage,
                "eshkol_list_to_svec", &ctx_.module());
        }
        Value* svec_from_list = ctx_.builder().CreateCall(l2s_fn, {l2s_arena, list_slot});
        Value* svec_from_list_int = ctx_.builder().CreatePtrToInt(svec_from_list, ctx_.int64Type());
        Value* svec_from_list_tagged = tagged_.packPtr(svec_from_list_int, ESHKOL_VALUE_HEAP_PTR);
        ctx_.builder().CreateStore(svec_from_list_tagged, svec_input_ptr);
        ctx_.builder().CreateBr(scheme_vector_input);
    }

    // Check for TENSOR subtype — convert to Scheme vector ONLY for multi-param functions.
    // For single-param functions, tensor input works correctly with reverse-mode AD.
    // For multi-param functions, forward-mode with dual numbers is needed because
    // reverse-mode passes AD nodes as CALLABLE tagged values which crash in function dispatch.
    ctx_.builder().SetInsertPoint(grad_check_tensor);
    uint64_t grad_func_arity = 0;
    {
        // REPL hot-reload appends "__rv<n>" to the LLVM symbol name (see
        // llvm_codegen.cpp:3264); the arity table is keyed on the
        // original user-facing name. Without stripping, arity lookup
        // fails for multi-arg functions and we fall into the single-arg
        // path — then resolveGradientCaptures sees the extra LLVM
        // parameters and misidentifies them as closure captures, emits
        // null-pointer args, and crashes verify. Keep this normalization
        // identical in every arity lookup below.
        std::string key = func_ptr->getName().str();
        auto rv_pos = key.rfind("__rv");
        if (rv_pos != std::string::npos &&
            rv_pos + 4 < key.size() &&
            key.find_first_not_of("0123456789", rv_pos + 4) == std::string::npos) {
            key.erase(rv_pos);
        }
        auto arity_it = function_arity_table_->find(key);
        if (arity_it != function_arity_table_->end()) {
            grad_func_arity = arity_it->second;
        }
    }
    if (grad_func_arity > 1) {
        BasicBlock* grad_tensor_to_svec = BasicBlock::Create(ctx_.context(), "grad_tensor_to_svec", current_func);
        ctx_.builder().CreateCondBr(is_tensor_subtype_grad, grad_tensor_to_svec, vector_input);

        // TENSOR→SVEC CONVERSION: Convert 8-byte tensor doubles to 16-byte tagged Scheme vector
        // so the forward-mode dual number path handles multi-parameter gradients correctly.
        ctx_.builder().SetInsertPoint(grad_tensor_to_svec);
        {
            Value* t_ptr = grad_heap_ptr;
            Value* t_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), t_ptr, 0);
            Value* t_dims_ptr = ctx_.builder().CreateLoad(ctx_.builder().getPtrTy(), t_dims_field);
            Value* t_n = ctx_.builder().CreateLoad(ctx_.int64Type(), t_dims_ptr);
            Value* t_elems_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), t_ptr, 2);
            Value* t_elems_ptr = ctx_.builder().CreateLoad(ctx_.builder().getPtrTy(), t_elems_field);

            Value* t_arena = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
            Value* t_svec = ctx_.builder().CreateCall(mem_.getArenaAllocateVectorWithHeader(), {t_arena, t_n});
            ctx_.builder().CreateStore(t_n, t_svec);
            Value* t_svec_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), t_svec, ConstantInt::get(ctx_.int64Type(), 8));
            Value* t_svec_elems = ctx_.builder().CreatePointerCast(t_svec_elems_base, ctx_.ptrType());

            Value* t_conv_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "tensor_conv_idx");
            ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), t_conv_idx);

            BasicBlock* t_conv_cond = BasicBlock::Create(ctx_.context(), "tensor_conv_cond", current_func);
            BasicBlock* t_conv_body = BasicBlock::Create(ctx_.context(), "tensor_conv_body", current_func);
            BasicBlock* t_conv_exit = BasicBlock::Create(ctx_.context(), "tensor_conv_exit", current_func);

            ctx_.builder().CreateBr(t_conv_cond);

            ctx_.builder().SetInsertPoint(t_conv_cond);
            Value* t_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), t_conv_idx);
            ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(t_idx, t_n), t_conv_body, t_conv_exit);

            ctx_.builder().SetInsertPoint(t_conv_body);
            Value* t_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), t_elems_ptr, t_idx);
            Value* t_elem_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(), t_elem_ptr);
            Value* t_elem_double = ctx_.builder().CreateBitCast(t_elem_i64, ctx_.doubleType());
            Value* t_elem_tagged = tagged_.packDouble(t_elem_double);
            Value* t_svec_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), t_svec_elems, t_idx);
            ctx_.builder().CreateStore(t_elem_tagged, t_svec_elem_ptr);
            ctx_.builder().CreateStore(ctx_.builder().CreateAdd(t_idx, ConstantInt::get(ctx_.int64Type(), 1)), t_conv_idx);
            ctx_.builder().CreateBr(t_conv_cond);

            ctx_.builder().SetInsertPoint(t_conv_exit);
            Value* t_svec_int = ctx_.builder().CreatePtrToInt(t_svec, ctx_.int64Type());
            Value* t_converted_tagged = tagged_.packPtr(t_svec_int, ESHKOL_VALUE_HEAP_PTR);
            ctx_.builder().CreateStore(t_converted_tagged, svec_input_ptr);
        }
        ctx_.builder().CreateBr(scheme_vector_input);
    } else {
        // Single-param: tensor goes through reverse-mode (works for arity <= 1)
        ctx_.builder().CreateBr(vector_input);
    }

    // Legacy VECTOR_PTR fallback
    ctx_.builder().SetInsertPoint(check_legacy_vector);
    ctx_.builder().CreateCondBr(is_legacy_vector, scheme_vector_input, vector_input);

    // SCALAR INPUT: Auto-promote scalar to 1D tensor #(scalar_value)
    ctx_.builder().SetInsertPoint(scalar_input);
    eshkol_debug("Gradient: auto-promoting scalar input to 1D vector");
    
    // Extract scalar value (INT64 or DOUBLE)
    Value* scalar_val_int = tagged_.unpackInt64(vector_val);
    
    // Convert to double if needed
    Value* scalar_double = ctx_.builder().CreateSelect(is_double,
        ctx_.builder().CreateBitCast(scalar_val_int, ctx_.doubleType()),
        ctx_.builder().CreateSIToFP(scalar_val_int, ctx_.doubleType()));
    
    // Allocate 1D tensor structure for promoted scalar via arena (OALR compliant - no malloc)
    Value* typed_promoted_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions: [1] (1D tensor with single element)
    Value* promoted_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* promoted_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, promoted_dims_size});
    Value* typed_promoted_dims = ctx_.builder().CreatePointerCast(promoted_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), typed_promoted_dims);

    // Set tensor metadata
    ctx_.builder().CreateStore(typed_promoted_dims,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_promoted_tensor, 0));  // dimensions = [1]
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_promoted_tensor, 1));  // num_dimensions = 1
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_promoted_tensor, 3));  // total_elements = 1

    // Allocate and set elements: [scalar_value]
    Value* promoted_elems_size = ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    Value* promoted_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, promoted_elems_size});
    Value* typed_promoted_elems = ctx_.builder().CreatePointerCast(promoted_elems_ptr, ctx_.builder().getPtrTy());
    
    // Store scalar as bitcast int64 (preserves IEEE754 bits for doubles)
    Value* scalar_as_int64 = ctx_.builder().CreateBitCast(scalar_double, ctx_.int64Type());
    ctx_.builder().CreateStore(scalar_as_int64, typed_promoted_elems);
    
    ctx_.builder().CreateStore(typed_promoted_elems,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_promoted_tensor, 2));  // elements
    
    // Pack promoted tensor as tagged_value with TENSOR_PTR type
    Value* promoted_tensor_int = ctx_.builder().CreatePtrToInt(typed_promoted_tensor, ctx_.int64Type());
    Value* promoted_vector_tagged = tagged_.packPtr(promoted_tensor_int, ESHKOL_VALUE_HEAP_PTR);
    
    ctx_.builder().CreateBr(grad_merge_input);
    BasicBlock* scalar_input_exit = ctx_.builder().GetInsertBlock();
    
    // SCHEME VECTOR INPUT: Use forward-mode AD with dual numbers (preserves Scheme vector format)
    // This allows functions that use vector-ref to work correctly with gradient
    // Handles both native Scheme vectors AND tensors converted via tensor→svec path above
    ctx_.builder().SetInsertPoint(scheme_vector_input);
    eshkol_debug("Gradient: using forward-mode AD for Scheme vector input");

    // Load effective input (may have been updated by tensor→svec conversion)
    Value* effective_svec_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_input_ptr);
    // Get Scheme vector pointer and length
    Value* svec_ptr_int = tagged_.unpackInt64(effective_svec_val);
    Value* svec_ptr = ctx_.builder().CreateIntToPtr(svec_ptr_int, ctx_.builder().getPtrTy());
    Value* svec_n = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_ptr);
    Value* svec_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_ptr, ConstantInt::get(ctx_.int64Type(), 8));
    Value* svec_elems = ctx_.builder().CreatePointerCast(svec_elems_base, ctx_.ptrType());

    // Allocate result tensor for gradient - use arena allocation with header for HEAP_PTR type
    Value* arena_for_svec = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
    Value* svec_typed_result = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_for_svec});

    // Set result tensor dimensions - use arena allocation
    Value* svec_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* svec_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_for_svec, svec_dims_size});
    Value* svec_typed_dims = ctx_.builder().CreatePointerCast(svec_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(svec_n, svec_typed_dims);
    ctx_.builder().CreateStore(svec_typed_dims, ctx_.builder().CreateStructGEP(ctx_.tensorType(), svec_typed_result, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), ctx_.builder().CreateStructGEP(ctx_.tensorType(), svec_typed_result, 1));
    ctx_.builder().CreateStore(svec_n, ctx_.builder().CreateStructGEP(ctx_.tensorType(), svec_typed_result, 3));

    // Allocate result elements - use arena allocation
    Value* svec_result_elems_size = ctx_.builder().CreateMul(svec_n, ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    Value* svec_result_elems = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_for_svec, svec_result_elems_size});
    Value* svec_typed_result_elems = ctx_.builder().CreatePointerCast(svec_result_elems, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(svec_typed_result_elems, ctx_.builder().CreateStructGEP(ctx_.tensorType(), svec_typed_result, 2));

    // ESH-0096: when this gradient runs NESTED inside another forward-mode
    // gradient over a vector (gradient-of-gradient), each component must be
    // returned as a DUAL carrying {value = ∂f/∂vⱼ (e2 slice), e1 = ∂²f/∂vᵢ∂vⱼ}
    // so the OUTER gradient can extract the second-order (mixed e1e2) term.
    // A plain-double tensor would drop that outer perturbation → the outer
    // gradient would see a constant and return 0. We build this parallel
    // Scheme vector of dual tagged values alongside the plain-double tensor and
    // select which to return based on the perturbation level (below).
    Value* svec_dual_result_vec = ctx_.builder().CreateCall(mem_.getArenaAllocateVectorWithHeader(),
        {arena_for_svec, svec_n});
    ctx_.builder().CreateStore(svec_n, svec_dual_result_vec);
    Value* svec_dual_result_elems8 = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_dual_result_vec,
        ConstantInt::get(ctx_.int64Type(), 8));
    Value* svec_dual_result_elems = ctx_.builder().CreatePointerCast(svec_dual_result_elems8, ctx_.ptrType());

    // Get arena for dual vector allocation
    Value* arena_svec = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // ESH-0093: this forward-mode vector gradient participates in the runtime
    // perturbation-level protocol (ESH-0070). The active component is seeded
    // in THIS level's slot (e1 at level 0, e2 when nested), the level is
    // pushed around the call so an inner derivative/gradient seeds the NEXT
    // slot, and the extraction below reads this level's coefficient. Without
    // this, an inner `derivative` re-seeded e1 and its perturbation collided
    // with the component's (the mixed-mode composition bug: d/dp of an inner
    // dx-derivative returned e1-confused garbage).
    Value* svec_pert_level = adPertLevelLoad();
    Value* svec_lvl0 = ctx_.builder().CreateICmpEQ(svec_pert_level,
        ConstantInt::get(ctx_.int64Type(), 0));

    // Outer loop: for each dimension i, compute ∂f/∂xᵢ using forward-mode AD
    Value* svec_dim_i = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "svec_dim_i");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), svec_dim_i);

    BasicBlock* svec_dim_cond = BasicBlock::Create(ctx_.context(), "svec_dim_cond", current_func);
    BasicBlock* svec_dim_body = BasicBlock::Create(ctx_.context(), "svec_dim_body", current_func);
    BasicBlock* svec_dim_end = BasicBlock::Create(ctx_.context(), "svec_dim_end", current_func);

    ctx_.builder().CreateBr(svec_dim_cond);

    ctx_.builder().SetInsertPoint(svec_dim_cond);
    Value* svec_i = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_dim_i);
    ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(svec_i, svec_n), svec_dim_body, svec_dim_end);

    ctx_.builder().SetInsertPoint(svec_dim_body);

    // M1 CONSOLIDATION: Allocate dual vector with header (Scheme vector of dual numbers)
    // arena_allocate_vector_with_header creates: [header(8)] + [length(8)] + [elements]
    // Header contains HEAP_SUBTYPE_VECTOR, returns pointer to length field
    Value* svec_dual_vec = ctx_.builder().CreateCall(mem_.getArenaAllocateVectorWithHeader(),
        {arena_svec, svec_n});
    ctx_.builder().CreateStore(svec_n, svec_dual_vec);
    Value* svec_dual_elems = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_dual_vec, ConstantInt::get(ctx_.int64Type(), 8));
    Value* svec_dual_elems_typed = ctx_.builder().CreatePointerCast(svec_dual_elems, ctx_.ptrType());

    // Inner loop: create dual vector with tangent=1 at position i, 0 elsewhere
    Value* svec_inner_j = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "svec_inner_j");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), svec_inner_j);

    BasicBlock* svec_inner_cond = BasicBlock::Create(ctx_.context(), "svec_inner_cond", current_func);
    BasicBlock* svec_inner_body = BasicBlock::Create(ctx_.context(), "svec_inner_body", current_func);
    BasicBlock* svec_inner_end = BasicBlock::Create(ctx_.context(), "svec_inner_end", current_func);

    ctx_.builder().CreateBr(svec_inner_cond);

    ctx_.builder().SetInsertPoint(svec_inner_cond);
    Value* svec_j = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_inner_j);
    ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(svec_j, svec_n), svec_inner_body, svec_inner_end);

    ctx_.builder().SetInsertPoint(svec_inner_body);
    // Load primal value from input
    Value* svec_in_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_elems, svec_j);
    Value* svec_in_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_in_ptr);
    // ESH-0096: preserve the incoming point's FULL jet so a nested gradient over
    // a vector keeps the OUTER perturbation. At level 0 the elements are plain
    // doubles → {p,0,0,0}; when nested (gradient-of-gradient over a vector) they
    // are duals carrying the outer tangent in e1 → {p,d1,0,0}. We seed only THIS
    // level's slot with the component tangent, keeping the other components
    // intact — the same discipline seedForwardAndPush uses for scalars.
    Value* svec_in_jet = safeUnpackDualFromTagged(svec_in_val); // {p,d1,d2,d12}
    Value* svec_primal  = dualField(ctx_, svec_in_jet, 0);
    Value* svec_in_e1   = dualField(ctx_, svec_in_jet, 1);
    Value* svec_in_e2   = dualField(ctx_, svec_in_jet, 2);
    Value* svec_in_e12  = dualField(ctx_, svec_in_jet, 3);
    // Tangent: 1.0 if j == i, else 0.0
    Value* svec_is_active = ctx_.builder().CreateICmpEQ(svec_j, svec_i);
    Value* svec_tangent = ctx_.builder().CreateSelect(svec_is_active,
        ConstantFP::get(ctx_.doubleType(), 1.0), ConstantFP::get(ctx_.doubleType(), 0.0));
    // ESH-0093: seed THIS level's slot (e1 at level 0, e2 when nested) so an
    // inner forward-mode derivative — which seeds the NEXT slot after the
    // level push below — cannot collide with the component's perturbation.
    // ESH-0096: at level 0 the e1 slot is a fresh seed; when nested keep the
    // incoming e1 (outer tangent) and put this component's tangent in e2.
    Value* svec_t_e1 = ctx_.builder().CreateSelect(svec_lvl0, svec_tangent, svec_in_e1);
    Value* svec_t_e2 = ctx_.builder().CreateSelect(svec_lvl0, svec_in_e2, svec_tangent);
    Value* svec_dual = makeDual4(ctx_, svec_primal, svec_t_e1, svec_t_e2, svec_in_e12);
    Value* svec_dual_tagged = packDualToTagged(svec_dual);
    Value* svec_dual_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_dual_elems_typed, svec_j);
    ctx_.builder().CreateStore(svec_dual_tagged, svec_dual_ptr);
    ctx_.builder().CreateStore(ctx_.builder().CreateAdd(svec_j, ConstantInt::get(ctx_.int64Type(), 1)), svec_inner_j);
    ctx_.builder().CreateBr(svec_inner_cond);

    ctx_.builder().SetInsertPoint(svec_inner_end);

    // M1 CONSOLIDATION: Pack dual vector as HEAP_PTR (header contains HEAP_SUBTYPE_VECTOR)
    Value* svec_dual_tagged_vec = tagged_.packPtr(
        ctx_.builder().CreatePtrToInt(svec_dual_vec, ctx_.int64Type()), ESHKOL_VALUE_HEAP_PTR);

    // Call function with dual vector — dispatches through helper
    std::vector<Value*> svec_call_args;

    // MULTI-PARAMETER GRADIENT: Check function arity and unpack if needed
    // arity > 1: extract individual dual number elements as separate args
    // arity <= 1: pass whole dual vector (for vector-input functions like (lambda (v) ...))
    {
        uint64_t svec_func_arity = 0;
        // REPL hot-reload strips __rv<n> from LLVM name — see grad_func_arity note above.
        std::string key = func_ptr->getName().str();
        auto rv_pos = key.rfind("__rv");
        if (rv_pos != std::string::npos &&
            rv_pos + 4 < key.size() &&
            key.find_first_not_of("0123456789", rv_pos + 4) == std::string::npos) {
            key.erase(rv_pos);
        }
        auto svec_arity_it = function_arity_table_->find(key);
        if (svec_arity_it != function_arity_table_->end()) {
            svec_func_arity = svec_arity_it->second;
        }
        // Trust the resolved function signature over the (regeneration-fragile)
        // arity table so multi-param functions are not called with one arg.
        svec_func_arity = adResolveValueArity(func_ptr, svec_func_arity);
        if (svec_func_arity > 1) {
            // Multi-param: unpack dual vector elements as individual tagged value args
            for (uint64_t p = 0; p < svec_func_arity; p++) {
                Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_dual_elems_typed,
                    ConstantInt::get(ctx_.int64Type(), p));
                Value* elem = ctx_.builder().CreateLoad(ctx_.taggedValueType(), elem_ptr);
                svec_call_args.push_back(elem);
            }
        } else {
            svec_call_args.push_back(svec_dual_tagged_vec);
        }
    }

    // Resolve captures via unified helper
    resolveGradientCaptures(func_ptr, svec_call_args, "svec");

    // ESH-0093: push the perturbation level around the call (the callee body
    // runs one forward level deeper) and pop it right after — the same
    // discipline seedForwardAndPush/popAndExtractForward use.
    adPertLevelStore(ctx_.builder().CreateAdd(svec_pert_level,
        ConstantInt::get(ctx_.int64Type(), 1)));

    Value* svec_call_result = ctx_.builder().CreateCall(func_ptr, svec_call_args);

    adPertLevelStore(svec_pert_level);

    // CONSTANT RESULT FIX: Check if result is a dual number before unpacking
    Value* svec_result_type = tagged_.getType(svec_call_result);
    Value* svec_result_base = tagged_.getBaseType(svec_result_type);
    Value* svec_is_dual = ctx_.builder().CreateICmpEQ(svec_result_base,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));

    BasicBlock* svec_dual_bb = BasicBlock::Create(ctx_.context(), "grad_svec_dual", current_func);
    BasicBlock* svec_const_bb = BasicBlock::Create(ctx_.context(), "grad_svec_const", current_func);
    BasicBlock* svec_merge_bb = BasicBlock::Create(ctx_.context(), "grad_svec_merge", current_func);

    ctx_.builder().CreateCondBr(svec_is_dual, svec_dual_bb, svec_const_bb);

    // Dual path: extract THIS level's coefficient (e1 at level 0, e2 nested).
    // ESH-0093: when the body contained an inner derivative, its
    // popAndExtractForward already returned the e2-slice {a2, a12} as a dual,
    // whose e1 coefficient is the mixed d/d(component) term — so the plain
    // field-1 read at level 0 composes correctly with inner forward passes.
    ctx_.builder().SetInsertPoint(svec_dual_bb);
    Value* svec_result_dual = unpackDualFromTagged(svec_call_result);
    Value* svec_dual_deriv = ctx_.builder().CreateSelect(svec_lvl0,
        dualField(ctx_, svec_result_dual, 1), dualField(ctx_, svec_result_dual, 2));
    // ESH-0096: the mixed e1e2 coefficient (a12) is the 2nd-order term
    // ∂²f/∂vᵢ∂vⱼ carried through when nested; used only by an outer gradient.
    Value* svec_dual_mixed = dualField(ctx_, svec_result_dual, 3);
    ctx_.builder().CreateBr(svec_merge_bb);
    BasicBlock* svec_dual_exit = ctx_.builder().GetInsertBlock();

    // Constant path: derivative is 0.0
    ctx_.builder().SetInsertPoint(svec_const_bb);
    Value* svec_zero_deriv = ConstantFP::get(ctx_.doubleType(), 0.0);
    ctx_.builder().CreateBr(svec_merge_bb);
    BasicBlock* svec_const_exit = ctx_.builder().GetInsertBlock();

    // Merge paths
    ctx_.builder().SetInsertPoint(svec_merge_bb);
    PHINode* svec_deriv = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "grad_svec_deriv");
    svec_deriv->addIncoming(svec_dual_deriv, svec_dual_exit);
    svec_deriv->addIncoming(svec_zero_deriv, svec_const_exit);
    PHINode* svec_mixed = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "grad_svec_mixed");
    svec_mixed->addIncoming(svec_dual_mixed, svec_dual_exit);
    svec_mixed->addIncoming(svec_zero_deriv, svec_const_exit);

    // Store derivative in result tensor (plain double — the level-0 result)
    Value* svec_result_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), svec_typed_result_elems, svec_i);
    Value* svec_deriv_bits = ctx_.builder().CreateBitCast(svec_deriv, ctx_.int64Type());
    ctx_.builder().CreateStore(svec_deriv_bits, svec_result_ptr);

    // ESH-0096: also store the {deriv, mixed} dual into the parallel Scheme
    // vector so a NESTED gradient-of-gradient can carry the 2nd-order term out.
    Value* svec_zero_c = ConstantFP::get(ctx_.doubleType(), 0.0);
    Value* svec_comp_dual = makeDual4(ctx_, svec_deriv, svec_mixed, svec_zero_c, svec_zero_c);
    Value* svec_comp_dual_tagged = packDualToTagged(svec_comp_dual);
    Value* svec_dual_res_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(),
        svec_dual_result_elems, svec_i);
    ctx_.builder().CreateStore(svec_comp_dual_tagged, svec_dual_res_ptr);

    ctx_.builder().CreateStore(ctx_.builder().CreateAdd(svec_i, ConstantInt::get(ctx_.int64Type(), 1)), svec_dim_i);
    ctx_.builder().CreateBr(svec_dim_cond);

    ctx_.builder().SetInsertPoint(svec_dim_end);
    // Return result tensor (level 0) or the dual-carrying Scheme vector (nested).
    // ESH-0096: at level 0 (outer gradient / top-level call) return the plain
    // double tensor — the historical representation every consumer expects.
    // When nested inside another forward-mode gradient, return the Scheme vector
    // of {∂f/∂vⱼ, ∂²f/∂vᵢ∂vⱼ} duals so the outer gradient can differentiate it.
    Value* svec_result_int = ctx_.builder().CreatePtrToInt(svec_typed_result, ctx_.int64Type());
    Value* svec_dual_vec_int = ctx_.builder().CreatePtrToInt(svec_dual_result_vec, ctx_.int64Type());
    // Select the pointer BEFORE packing (avoid a select over the tagged struct).
    Value* svec_chosen_int = ctx_.builder().CreateSelect(svec_lvl0,
        svec_result_int, svec_dual_vec_int);
    Value* scheme_vector_tagged = tagged_.packPtr(svec_chosen_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(grad_done);  // Skip reverse-mode AD, go directly to done
    BasicBlock* scheme_vector_exit = ctx_.builder().GetInsertBlock();

    // ESH-0235 VEC→TENSOR BRIDGE: a (vector …)-constructed point feeding a
    // single-arg tensor-op function is converted to a 1-D tensor here and fed
    // into the reverse-mode `vector_input` seeding via the merge PHI, so it is
    // taped identically to a #(…)/(tensor …) point (each element becomes an AD
    // variable). Reads the Scheme vector layout [len(8)][tagged elems] and
    // writes a plain-double tensor [dims|ndim=1|elems|total].
    ctx_.builder().SetInsertPoint(grad_vec_to_tensor);
    Value* v2t_svec_ptr = tagged_.unpackPtr(vector_val);
    Value* v2t_n = ctx_.builder().CreateLoad(ctx_.int64Type(), v2t_svec_ptr);
    Value* v2t_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), v2t_svec_ptr, ConstantInt::get(ctx_.int64Type(), 8));
    Value* v2t_elems = ctx_.builder().CreatePointerCast(v2t_elems_base, ctx_.ptrType());

    Value* v2t_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});
    Value* v2t_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t))});
    Value* v2t_typed_dims = ctx_.builder().CreatePointerCast(v2t_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(v2t_n, v2t_typed_dims);
    ctx_.builder().CreateStore(v2t_typed_dims, ctx_.builder().CreateStructGEP(ctx_.tensorType(), v2t_tensor, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), ctx_.builder().CreateStructGEP(ctx_.tensorType(), v2t_tensor, 1));
    ctx_.builder().CreateStore(v2t_n, ctx_.builder().CreateStructGEP(ctx_.tensorType(), v2t_tensor, 3));

    Value* v2t_elems_size = ctx_.builder().CreateMul(v2t_n, ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    Value* v2t_dst_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, v2t_elems_size});
    Value* v2t_dst = ctx_.builder().CreatePointerCast(v2t_dst_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(v2t_dst, ctx_.builder().CreateStructGEP(ctx_.tensorType(), v2t_tensor, 2));

    Value* v2t_i = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "v2t_i");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), v2t_i);
    BasicBlock* v2t_cond = BasicBlock::Create(ctx_.context(), "v2t_cond", current_func);
    BasicBlock* v2t_body = BasicBlock::Create(ctx_.context(), "v2t_body", current_func);
    BasicBlock* v2t_end = BasicBlock::Create(ctx_.context(), "v2t_end", current_func);
    ctx_.builder().CreateBr(v2t_cond);
    ctx_.builder().SetInsertPoint(v2t_cond);
    Value* v2t_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), v2t_i);
    ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(v2t_idx, v2t_n), v2t_body, v2t_end);
    ctx_.builder().SetInsertPoint(v2t_body);
    Value* v2t_src_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), v2t_elems, v2t_idx);
    Value* v2t_src_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), v2t_src_ptr);
    Value* v2t_dbl = tagged_.unpackDouble(v2t_src_val);
    Value* v2t_bits = ctx_.builder().CreateBitCast(v2t_dbl, ctx_.int64Type());
    Value* v2t_dst_slot = ctx_.builder().CreateGEP(ctx_.int64Type(), v2t_dst, v2t_idx);
    ctx_.builder().CreateStore(v2t_bits, v2t_dst_slot);
    ctx_.builder().CreateStore(ctx_.builder().CreateAdd(v2t_idx, ConstantInt::get(ctx_.int64Type(), 1)), v2t_i);
    ctx_.builder().CreateBr(v2t_cond);
    ctx_.builder().SetInsertPoint(v2t_end);
    Value* v2t_tensor_int = ctx_.builder().CreatePtrToInt(v2t_tensor, ctx_.int64Type());
    Value* v2t_tensor_tagged = tagged_.packPtr(v2t_tensor_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(grad_merge_input);
    BasicBlock* grad_vec_to_tensor_exit = ctx_.builder().GetInsertBlock();

    // VECTOR INPUT: Use original vector as-is (existing behavior - tensor format)
    ctx_.builder().SetInsertPoint(vector_input);
    ctx_.builder().CreateBr(grad_merge_input);
    BasicBlock* vector_input_exit = ctx_.builder().GetInsertBlock();

    // MERGE: PHI node selects AD node promoted, scalar promoted, or original tensor
    // NOTE: Scheme vector path now uses forward-mode AD and branches directly to grad_done
    ctx_.builder().SetInsertPoint(grad_merge_input);
    PHINode* actual_input = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 4, "gradient_input");
    actual_input->addIncoming(ad_promoted_tagged, ad_node_exit);  // Nested gradient path
    actual_input->addIncoming(promoted_vector_tagged, scalar_input_exit);
    actual_input->addIncoming(vector_val, vector_input_exit);
    actual_input->addIncoming(v2t_tensor_tagged, grad_vec_to_tensor_exit);  // ESH-0235: (vector …) point

    PHINode* input_was_scalar_promoted = ctx_.builder().CreatePHI(ctx_.int1Type(), 4, "gradient_input_was_scalar");
    input_was_scalar_promoted->addIncoming(ConstantInt::get(ctx_.int1Type(), 1), ad_node_exit);
    input_was_scalar_promoted->addIncoming(ConstantInt::get(ctx_.int1Type(), 1), scalar_input_exit);
    input_was_scalar_promoted->addIncoming(ConstantInt::get(ctx_.int1Type(), 0), vector_input_exit);
    input_was_scalar_promoted->addIncoming(ConstantInt::get(ctx_.int1Type(), 0), grad_vec_to_tensor_exit);
    
    // Continue with gradient computation using merged input (guaranteed to be tensor!)
    Value* vector_ptr_int = tagged_.unpackInt64(actual_input);
    // Note: arena_ptr already defined at function start


    // Convert int64 pointer to typed tensor pointer
    Value* vector_ptr = ctx_.builder().CreateIntToPtr(vector_ptr_int, ctx_.builder().getPtrTy());

    // Extract ALL tensor properties (MUST access all fields correctly)
    Value* dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), vector_ptr, 0);
    Value* dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dims_field_ptr);
    Value* typed_dims_ptr = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());

    Value* num_dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), vector_ptr, 1);
    Value* input_num_dims = ctx_.builder().CreateLoad(ctx_.int64Type(), num_dims_field_ptr);

    Value* elements_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), vector_ptr, 2);
    Value* elements_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), elements_field_ptr);
    Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.builder().getPtrTy());

    // Differentiate one scalar component per tensor element, not just dims[0].
    // The AD tensor passed to the user function still keeps the original shape.
    Value* total_elements_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), vector_ptr, 3);
    Value* n = ctx_.builder().CreateLoad(ctx_.int64Type(), total_elements_field_ptr);
    
    // VALIDATION: Check dimension > 0 (scalars already promoted to tensors, so type check not needed)
    Value* n_is_positive = ctx_.builder().CreateICmpUGT(n, ConstantInt::get(ctx_.int64Type(), 0));
    
    BasicBlock* dim_valid = BasicBlock::Create(ctx_.context(), "grad_dim_valid", current_func);
    BasicBlock* dim_invalid = BasicBlock::Create(ctx_.context(), "grad_dim_invalid", current_func);
    // grad_done already created earlier for scheme_vector_input forward-mode path

    // CRITICAL FIX: Create empty tensor BEFORE branching (for PHI node dominance)
    // This ensures null_tagged_grad is available in all paths
    // Allocate empty tensor via arena (OALR compliant - no malloc)
    Value* typed_empty_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions array (size 1, value 0)
    Value* empty_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* empty_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, empty_dims_size});
    Value* typed_empty_dims = ctx_.builder().CreatePointerCast(empty_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), typed_empty_dims);

    ctx_.builder().CreateStore(typed_empty_dims,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_empty_tensor, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_empty_tensor, 1));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_empty_tensor, 3));

    // Empty elements array
    Value* empty_elems_size = ConstantInt::get(ctx_.int64Type(), sizeof(double));
    Value* empty_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, empty_elems_size});
    Value* typed_empty_elems = ctx_.builder().CreatePointerCast(empty_elems_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(typed_empty_elems,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_empty_tensor, 2));
    
    // Pack as tagged_value (TENSOR_PTR type) - available in all paths
    Value* empty_tensor_int = ctx_.builder().CreatePtrToInt(typed_empty_tensor, ctx_.int64Type());
    Value* null_tagged_grad = tagged_.packPtr(empty_tensor_int, ESHKOL_VALUE_HEAP_PTR);
    
    ctx_.builder().CreateCondBr(n_is_positive, dim_valid, dim_invalid);
    
    // Invalid input: return empty tensor
    ctx_.builder().SetInsertPoint(dim_invalid);
    eshkol_debug("Gradient: invalid input tensor (dimension must be > 0)");
    ctx_.builder().CreateBr(grad_done);
    
    // Valid dimension: compute gradient
    ctx_.builder().SetInsertPoint(dim_valid);
    
    // Allocate result gradient vector via arena (OALR compliant - no malloc)
    Value* typed_result_tensor_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set result tensor dimension (1D vector of size n)
    Value* result_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* result_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, result_dims_size});
    Value* typed_result_dims_ptr = ctx_.builder().CreatePointerCast(result_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(n, typed_result_dims_ptr);

    // Store dimension in result tensor
    Value* result_dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_tensor_ptr, 0);
    ctx_.builder().CreateStore(typed_result_dims_ptr, result_dims_field_ptr);

    // Store num_dimensions = 1
    Value* result_num_dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_tensor_ptr, 1);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), result_num_dims_field_ptr);

    // Store total_elements = n
    Value* result_total_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_tensor_ptr, 3);
    ctx_.builder().CreateStore(n, result_total_field_ptr);

    // Allocate result elements array (n doubles for partial derivatives)
    Value* result_elements_size = ctx_.builder().CreateMul(n,
        ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    Value* result_elements_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, result_elements_size});
    Value* typed_result_elements_ptr = ctx_.builder().CreatePointerCast(result_elements_ptr, ctx_.builder().getPtrTy());
    
    // Store elements pointer in result tensor
    Value* result_elements_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_tensor_ptr, 2);
    ctx_.builder().CreateStore(typed_result_elements_ptr, result_elements_field_ptr);
    
    // ===== MAIN GRADIENT COMPUTATION LOOP =====
    // For each component i from 0 to n-1, compute ∂f/∂xᵢ
    
    BasicBlock* grad_loop_cond = BasicBlock::Create(ctx_.context(), "grad_loop_cond", current_func);
    BasicBlock* grad_loop_body = BasicBlock::Create(ctx_.context(), "grad_loop_body", current_func);
    BasicBlock* grad_loop_exit = BasicBlock::Create(ctx_.context(), "grad_loop_exit", current_func);
    
    // Allocate loop counter
    Value* component_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "component_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), component_idx);
    
    ctx_.builder().CreateBr(grad_loop_cond);
    
    // Loop condition: i < n
    ctx_.builder().SetInsertPoint(grad_loop_cond);
    Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), component_idx);
    Value* i_less_n = ctx_.builder().CreateICmpULT(i, n);
    ctx_.builder().CreateCondBr(i_less_n, grad_loop_body, grad_loop_exit);
    
    // Loop body: Compute ∂f/∂xᵢ using reverse-mode AD
    ctx_.builder().SetInsertPoint(grad_loop_body);

    // Step 1: Create tape for this partial derivative (arena_ptr defined at function start)
    Value* tape_capacity = ConstantInt::get(ctx_.int64Type(), 1024);
    Value* partial_tape = ctx_.builder().CreateCall(mem_.getArenaAllocateTape(),
        {arena_ptr, tape_capacity});
    
    // Store tape as current (required by recordADNode* functions)
    Value* saved_tape = current_tape_ptr_;
    current_tape_ptr_ = partial_tape;
    
    // Step 2: Create n AD variable nodes (one per vector component)
    // Allocate array to hold variable node pointers via arena (OALR compliant - no malloc)
    Value* var_nodes_array_size = ctx_.builder().CreateMul(n,
        ConstantInt::get(ctx_.int64Type(), sizeof(void*)));
    Value* var_nodes_array = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, var_nodes_array_size});
    Value* typed_var_nodes = ctx_.builder().CreatePointerCast(var_nodes_array, ctx_.builder().getPtrTy());
    
    // Loop to create and initialize variable nodes
    BasicBlock* init_vars_cond = BasicBlock::Create(ctx_.context(), "init_vars_cond", current_func);
    BasicBlock* init_vars_body = BasicBlock::Create(ctx_.context(), "init_vars_body", current_func);
    BasicBlock* init_vars_exit = BasicBlock::Create(ctx_.context(), "init_vars_exit", current_func);
    
    Value* init_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "init_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), init_idx);
    ctx_.builder().CreateBr(init_vars_cond);
    
    ctx_.builder().SetInsertPoint(init_vars_cond);
    Value* j = ctx_.builder().CreateLoad(ctx_.int64Type(), init_idx);
    Value* j_less_n = ctx_.builder().CreateICmpULT(j, n);
    ctx_.builder().CreateCondBr(j_less_n, init_vars_body, init_vars_exit);
    
    ctx_.builder().SetInsertPoint(init_vars_body);

    // CRITICAL FIX: Tensor elements are stored as int64, load as int64 then convert to double
    Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_elements_ptr, j);
    Value* elem_val_int64 = ctx_.builder().CreateLoad(ctx_.int64Type(), elem_ptr);

    // NESTED GRADIENT FIX: Check if element might be an AD node pointer (from outer gradient)
    // Don't check tape depth - the element itself tells us if it's an AD node
    // When a gradient's input contains an AD node, we detect it and set up double backward
    BasicBlock* check_ad_ptr = BasicBlock::Create(ctx_.context(), "check_ad_ptr", current_func);
    BasicBlock* is_regular_double = BasicBlock::Create(ctx_.context(), "is_regular_double", current_func);
    BasicBlock* merge_elem = BasicBlock::Create(ctx_.context(), "merge_elem", current_func);

    // Check if the value could be a pointer (in valid heap address range)
    // On 64-bit systems:
    // - Heap pointers are typically 0x100000000 to 0x00007FFFFFFFFFFF (small as int64)
    // - Normal doubles like 2.0 = 0x4000000000000000, 12.0 = 0x4028... (LARGE as int64)
    // So a potential pointer is: non-zero AND less than typical double values
    // Use threshold 0x0001000000000000 (~281 trillion) - catches all user space addresses
    // but excludes normal positive doubles (which are >= 0x3FF0000000000000 for >= 1.0)
    Value* not_zero = ctx_.builder().CreateICmpNE(elem_val_int64,
        ConstantInt::get(ctx_.int64Type(), 0));
    Value* in_ptr_range = ctx_.builder().CreateICmpULT(elem_val_int64,
        ConstantInt::get(ctx_.int64Type(), 0x0001000000000000ULL));
    Value* could_be_ptr = ctx_.builder().CreateAnd(not_zero, in_ptr_range);
    ctx_.builder().CreateCondBr(could_be_ptr, check_ad_ptr, is_regular_double);

    // CHECK AD POINTER: Try to validate it's actually an AD node
    ctx_.builder().SetInsertPoint(check_ad_ptr);
    Value* ad_ptr_candidate = ctx_.builder().CreateIntToPtr(elem_val_int64, PointerType::getUnqual(ctx_.context()));
    // Check if pointer is non-null and has valid AD node type
    Value* ptr_not_null = ctx_.builder().CreateICmpNE(elem_val_int64,
        ConstantInt::get(ctx_.int64Type(), 0));

    BasicBlock* check_ad_type = BasicBlock::Create(ctx_.context(), "check_ad_type", current_func);
    BasicBlock* not_ad_node = BasicBlock::Create(ctx_.context(), "not_ad_node", current_func);
    ctx_.builder().CreateCondBr(ptr_not_null, check_ad_type, not_ad_node);

    // Check AD node type field
    ctx_.builder().SetInsertPoint(check_ad_type);
    Value* type_field_ptr = ctx_.builder().CreateStructGEP(ctx_.adNodeType(), ad_ptr_candidate, 0);
    Value* type_field = ctx_.builder().CreateLoad(ctx_.int32Type(), type_field_ptr);
    // Valid AD node types are 0-7 (CONSTANT, PTR, ADD, SUB, MUL, DIV, SIN, COS)
    // Also check that it's exactly type 1 (AD_NODE_PTR) since that's what variables are
    Value* is_ad_var = ctx_.builder().CreateICmpEQ(type_field, ConstantInt::get(ctx_.int32Type(), 1));

    BasicBlock* use_existing_ad = BasicBlock::Create(ctx_.context(), "use_existing_ad", current_func);
    ctx_.builder().CreateCondBr(is_ad_var, use_existing_ad, not_ad_node);

    // USE EXISTING AD NODE: This element is an AD node from outer gradient
    // CRITICAL FIX: Do NOT reuse the outer AD node directly!
    // The inner backward would write to its gradient field, contaminating it.
    // Instead, create a new AD variable with the same value and record the outer node.
    ctx_.builder().SetInsertPoint(use_existing_ad);
    Value* detected_outer_node = ad_ptr_candidate;
    // Store outer AD node for double backward connection
    ctx_.builder().CreateStore(detected_outer_node, ctx_.outerAdNodeStorage());
    // Extract the VALUE from outer AD node and create NEW variable for inner gradient
    Value* detected_outer_val_ptr = ctx_.builder().CreateStructGEP(ctx_.adNodeType(), detected_outer_node, 1);
    Value* detected_outer_val = ctx_.builder().CreateLoad(ctx_.doubleType(), detected_outer_val_ptr);
    Value* new_inner_var = createADVariable(detected_outer_val, 0);
    ctx_.builder().CreateBr(merge_elem);
    BasicBlock* use_ad_exit = ctx_.builder().GetInsertBlock();

    // NOT AN AD NODE: Treat as double
    ctx_.builder().SetInsertPoint(not_ad_node);
    Value* elem_as_double2 = ctx_.builder().CreateBitCast(elem_val_int64, ctx_.doubleType());
    Value* new_var_node2 = createADVariable(elem_as_double2, 0);
    ctx_.builder().CreateBr(merge_elem);
    BasicBlock* not_ad_exit = ctx_.builder().GetInsertBlock();

    // REGULAR DOUBLE: Normal case - just treat as double
    ctx_.builder().SetInsertPoint(is_regular_double);
    Value* elem_val = ctx_.builder().CreateBitCast(elem_val_int64, ctx_.doubleType());
    Value* new_var_node = createADVariable(elem_val, 0);
    ctx_.builder().CreateBr(merge_elem);
    BasicBlock* regular_double_exit = ctx_.builder().GetInsertBlock();

    // MERGE: PHI to select the right AD node
    ctx_.builder().SetInsertPoint(merge_elem);
    PHINode* var_node = ctx_.builder().CreatePHI(PointerType::getUnqual(ctx_.context()), 3, "var_node_phi");
    var_node->addIncoming(new_inner_var, use_ad_exit);
    var_node->addIncoming(new_var_node2, not_ad_exit);
    var_node->addIncoming(new_var_node, regular_double_exit);
    
    // Store node pointer in array
    Value* node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_var_nodes, j);
    ctx_.builder().CreateStore(var_node, node_slot);
    
    // Increment init counter
    Value* next_j = ctx_.builder().CreateAdd(j, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_j, init_idx);
    ctx_.builder().CreateBr(init_vars_cond);
    
    ctx_.builder().SetInsertPoint(init_vars_exit);
    
    // Step 3: Get active variable node (the one we're computing gradient for)
    Value* active_node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_var_nodes, i);
    Value* active_var_node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()),
        active_node_slot);

    // AD Phase A — one-pass gradient support.
    //   (1) Populate the tape's (formerly dead) variables list so a single
    //       reverse sweep exposes every input's gradient for readback.
    //   (2) Snapshot the reverse-over-forward mixed-record counter BEFORE the
    //       primal runs. After backprop, an unchanged counter proves this pass
    //       was pure reverse-mode, so the single sweep's gradients for ALL
    //       components are valid and the per-component replay loop is skipped
    //       (fast path). A changed counter means an inner forward-mode
    //       derivative ran (per-component seed semantics are load-bearing), so
    //       the pass safely continues the original per-component replay.
    ctx_.builder().CreateCall(ctx_.module().getOrInsertFunction(
        "arena_tape_set_variables",
        FunctionType::get(ctx_.voidType(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false)),
        {partial_tape, typed_var_nodes, n});
    FunctionCallee ad_mixed_count_fn = ctx_.module().getOrInsertFunction(
        "eshkol_ad_mixed_record_count",
        FunctionType::get(ctx_.int64Type(), {}, false));
    Value* mixed_before = ctx_.builder().CreateCall(ad_mixed_count_fn, {}, "mixed_before");

    // Step 4: Call function with variable nodes to build computational graph
    // CRITICAL: Function must operate on AD nodes, not raw doubles
    // This requires the function to use recordADNode* operations
    
    // Build tensor of AD node pointers to pass to function
    // M1 CONSOLIDATION: Use arena allocation with header for HEAP_PTR type
    Value* ad_arena_ptr = ctx_.builder().CreateLoad(
        PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
    Function* alloc_tensor_full = mem_.getArenaAllocateTensorFull();
    Value* typed_ad_tensor_ptr = ctx_.builder().CreateCall(alloc_tensor_full,
        {ad_arena_ptr, input_num_dims, n}, "ad_tensor");

    // Set AD tensor dimensions to match the input tensor shape.
    Value* ad_dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor_ptr, 0));

    Value* copy_dim_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "grad_ad_dim_i");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), copy_dim_idx);
    BasicBlock* copy_dims_cond = BasicBlock::Create(ctx_.context(), "grad_ad_dims_cond", current_func);
    BasicBlock* copy_dims_body = BasicBlock::Create(ctx_.context(), "grad_ad_dims_body", current_func);
    BasicBlock* copy_dims_exit = BasicBlock::Create(ctx_.context(), "grad_ad_dims_exit", current_func);
    ctx_.builder().CreateBr(copy_dims_cond);

    ctx_.builder().SetInsertPoint(copy_dims_cond);
    Value* dim_i = ctx_.builder().CreateLoad(ctx_.int64Type(), copy_dim_idx);
    Value* has_dim = ctx_.builder().CreateICmpULT(dim_i, input_num_dims);
    ctx_.builder().CreateCondBr(has_dim, copy_dims_body, copy_dims_exit);

    ctx_.builder().SetInsertPoint(copy_dims_body);
    Value* src_dim_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims_ptr, dim_i);
    Value* dst_dim_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), ad_dims_ptr, dim_i);
    Value* dim_value = ctx_.builder().CreateLoad(ctx_.int64Type(), src_dim_ptr);
    ctx_.builder().CreateStore(dim_value, dst_dim_ptr);
    Value* next_dim_i = ctx_.builder().CreateAdd(dim_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_dim_i, copy_dim_idx);
    ctx_.builder().CreateBr(copy_dims_cond);

    ctx_.builder().SetInsertPoint(copy_dims_exit);

    // Get elements array (already allocated by arena_allocate_tensor_full)
    Value* typed_ad_elems_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor_ptr, 2));
    
    // Copy node pointers into AD tensor
    BasicBlock* copy_nodes_cond = BasicBlock::Create(ctx_.context(), "copy_nodes_cond", current_func);
    BasicBlock* copy_nodes_body = BasicBlock::Create(ctx_.context(), "copy_nodes_body", current_func);
    BasicBlock* copy_nodes_exit = BasicBlock::Create(ctx_.context(), "copy_nodes_exit", current_func);
    
    Value* copy_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "copy_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), copy_idx);
    ctx_.builder().CreateBr(copy_nodes_cond);
    
    ctx_.builder().SetInsertPoint(copy_nodes_cond);
    Value* k = ctx_.builder().CreateLoad(ctx_.int64Type(), copy_idx);
    Value* k_less_n = ctx_.builder().CreateICmpULT(k, n);
    ctx_.builder().CreateCondBr(k_less_n, copy_nodes_body, copy_nodes_exit);
    
    ctx_.builder().SetInsertPoint(copy_nodes_body);
    Value* src_node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_var_nodes, k);
    Value* src_node_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), src_node_slot);
    Value* node_as_int64 = ctx_.builder().CreatePtrToInt(src_node_ptr, ctx_.int64Type());
    
    Value* dst_elem_slot = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_ad_elems_ptr, k);
    ctx_.builder().CreateStore(node_as_int64, dst_elem_slot);
    
    Value* next_k = ctx_.builder().CreateAdd(k, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_k, copy_idx);
    ctx_.builder().CreateBr(copy_nodes_cond);
    
    ctx_.builder().SetInsertPoint(copy_nodes_exit);
    
    // Step 5: Call function with AD node (scalar) or tensor (vector)
    // SCALAR FUNCTION FIX: For n=1, extract the single AD node and pass it directly!
    // This allows scalar functions like (lambda (x) (* x x)) to work


    Value* n_is_one = ctx_.builder().CreateICmpEQ(n, ConstantInt::get(ctx_.int64Type(), 1));
    Value* use_scalar_call = ctx_.builder().CreateAnd(n_is_one, input_was_scalar_promoted);
    
    BasicBlock* scalar_call = BasicBlock::Create(ctx_.context(), "grad_scalar_call", current_func);
    BasicBlock* vector_call = BasicBlock::Create(ctx_.context(), "grad_vector_call", current_func);
    BasicBlock* after_func_call = BasicBlock::Create(ctx_.context(), "grad_after_func_call", current_func);
    
    ctx_.builder().CreateCondBr(use_scalar_call, scalar_call, vector_call);
    
    // SCALAR: Extract single AD node and pass directly
    ctx_.builder().SetInsertPoint(scalar_call);
    Value* single_ad_node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_var_nodes, ConstantInt::get(ctx_.int64Type(), 0));
    Value* single_ad_node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), single_ad_node_slot);
    Value* scalar_ad_tagged = tagged_.packPtr(single_ad_node, ESHKOL_VALUE_CALLABLE);

    std::vector<Value*> scalar_args;

    // MULTI-PARAMETER: If function has more params than 1, unpack AD nodes
    {
        uint64_t scalar_func_arity = 0;
        // REPL hot-reload strips __rv<n> from LLVM name — see grad_func_arity note above.
        std::string key = func_ptr->getName().str();
        auto rv_pos = key.rfind("__rv");
        if (rv_pos != std::string::npos &&
            rv_pos + 4 < key.size() &&
            key.find_first_not_of("0123456789", rv_pos + 4) == std::string::npos) {
            key.erase(rv_pos);
        }
        auto scalar_arity_it = function_arity_table_->find(key);
        if (scalar_arity_it != function_arity_table_->end()) {
            scalar_func_arity = scalar_arity_it->second;
        }
        // Trust the resolved function signature over the (regeneration-fragile)
        // arity table so multi-param functions are not called with one arg.
        scalar_func_arity = adResolveValueArity(func_ptr, scalar_func_arity);
        if (scalar_func_arity > 1) {
            // Multi-param function on scalar path: pass all AD nodes as individual args
            for (uint64_t p = 0; p < scalar_func_arity; p++) {
                Value* node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
                    typed_var_nodes, ConstantInt::get(ctx_.int64Type(), p));
                Value* node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), node_slot);
                Value* node_tagged = tagged_.packPtr(node, ESHKOL_VALUE_CALLABLE);
                scalar_args.push_back(node_tagged);
            }
        } else {
            scalar_args.push_back(scalar_ad_tagged);
        }
    }

    // Resolve captures via unified helper
    resolveGradientCaptures(func_ptr, scalar_args, "scalar");

    // NESTED GRADIENT FIX: Save ctx_.outerAdNodeStorage() before calling function
    // Nested gradients will overwrite it, so we save and restore to support n-dimensional derivatives
    Value* saved_outer_ad_node_scalar = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.outerAdNodeStorage());

    // NESTED GRADIENT FIX: Push tape context (saves outer gradient's tape if any)
    pushTapeContext(partial_tape);

    // M1 Migration FIX: Set AD mode flag so vref recognizes AD node pointers in tensors
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 1), ctx_.adModeActive());

    // ESH-0093: publish this pass's active variable node so an inner
    // forward-mode derivative can seed it (jet-lift) and record the mixed
    // partial back onto this tape. Save/restore for nested gradient passes.
    FunctionCallee seed_swap_scalar = ctx_.module().getOrInsertFunction(
        "eshkol_ad_seed_swap",
        FunctionType::get(ctx_.ptrType(), {ctx_.ptrType()}, false));
    Value* saved_seed_scalar = ctx_.builder().CreateCall(seed_swap_scalar, {active_var_node});

    // AD Phase A counter: one primal (user-function) evaluation.
    ctx_.builder().CreateCall(ctx_.module().getOrInsertFunction(
        "eshkol_ad_count_primal",
        FunctionType::get(ctx_.voidType(), {}, false)), {});
    Value* scalar_output = ctx_.builder().CreateCall(func_ptr, scalar_args);

    // ESH-0093: restore the previously active seed node
    ctx_.builder().CreateCall(seed_swap_scalar, {saved_seed_scalar});

    // M1 Migration FIX: Reset AD mode flag after function call
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 0), ctx_.adModeActive());

    // NESTED GRADIENT FIX: Pop tape context (restores outer gradient's tape if any)
    popTapeContext();

    // NESTED GRADIENT FIX: Restore ctx_.outerAdNodeStorage() after function returns
    ctx_.builder().CreateStore(saved_outer_ad_node_scalar, ctx_.outerAdNodeStorage());

    ctx_.builder().CreateBr(after_func_call);
    BasicBlock* scalar_call_exit = ctx_.builder().GetInsertBlock();
    
    // VECTOR: Pass AD nodes — either as single tensor or unpacked to individual params
    ctx_.builder().SetInsertPoint(vector_call);
    Value* ad_tensor_int = ctx_.builder().CreatePtrToInt(typed_ad_tensor_ptr, ctx_.int64Type());
    // M1 CONSOLIDATION: Use HEAP_PTR type - tensor has header with HEAP_SUBTYPE_TENSOR
    Value* ad_tensor_tagged = tagged_.packPtr(ad_tensor_int, ESHKOL_VALUE_HEAP_PTR);

    std::vector<Value*> grad_call_args;

    // MULTI-PARAMETER GRADIENT: Check if function has multiple parameters
    // If func has N params and N matches the gradient dimension, unpack AD nodes
    // as individual arguments instead of passing a single tensor.
    FunctionType* grad_func_type = func_ptr->getFunctionType();
    std::string func_name_str = func_ptr->getName().str();
    // REPL hot-reload strips __rv<n> from LLVM name — see grad_func_arity note above.
    std::string func_arity_key = func_name_str;
    {
        auto rv_pos = func_arity_key.rfind("__rv");
        if (rv_pos != std::string::npos &&
            rv_pos + 4 < func_arity_key.size() &&
            func_arity_key.find_first_not_of("0123456789", rv_pos + 4) == std::string::npos) {
            func_arity_key.erase(rv_pos);
        }
    }
    uint64_t func_arity = 0;
    auto arity_it = function_arity_table_->find(func_arity_key);
    if (arity_it != function_arity_table_->end()) {
        func_arity = arity_it->second;
    }
    // Trust the resolved function signature over the (regeneration-fragile)
    // arity table so multi-param functions are not called with one arg.
    func_arity = adResolveValueArity(func_ptr, func_arity);

    if (func_arity > 1) {
        // Multi-parameter function: unpack AD tensor elements as individual tagged args
        // Each element in the AD tensor is an AD node pointer (CALLABLE type)
        eshkol_debug("Gradient: unpacking %llu AD nodes for %llu-parameter function %s",
                     (unsigned long long)func_arity, (unsigned long long)func_arity, func_name_str.c_str());
        for (uint64_t p = 0; p < func_arity; p++) {
            // Load AD node pointer from tensor elements[p]
            Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(),
                ctx_.builder().CreateLoad(ctx_.builder().getPtrTy(),
                    ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor_ptr, 2)),
                ConstantInt::get(ctx_.int64Type(), p));
            Value* ad_node_int = ctx_.builder().CreateLoad(ctx_.int64Type(), elem_ptr);
            // Pack as CALLABLE tagged value (AD nodes are callable)
            Value* ad_node_tagged = tagged_.packPtr(ad_node_int, ESHKOL_VALUE_CALLABLE);
            grad_call_args.push_back(ad_node_tagged);
        }
    } else {
        // Single-parameter function: pass tensor as-is
        grad_call_args.push_back(ad_tensor_tagged);
    }

    if (grad_func_type->getNumParams() > grad_call_args.size()) {
        size_t num_captures = grad_func_type->getNumParams() - grad_call_args.size();
        std::string lambda_name = func_ptr->getName().str();

        // REPL MODE: Get capture names from registry instead of parameter names
        std::vector<std::string> capture_names;
        if ((repl_mode_enabled_ && *repl_mode_enabled_)) {
            std::lock_guard<std::mutex> lock(*repl_mutex_);
            auto captures_it = repl_lambda_captures_->find(lambda_name);
            if (captures_it != repl_lambda_captures_->end()) {
                capture_names = captures_it->second;
            }
        }

        for (size_t i = 0; i < num_captures; i++) {
            std::string var_name;
            if (i < capture_names.size()) {
                var_name = capture_names[i];
            } else {
                // Fallback to LLVM parameter names (for non-REPL mode)
                auto arg_it = func_ptr->arg_begin();
                std::advance(arg_it, i + 1);  // Skip first parameter
                if (arg_it != func_ptr->arg_end()) {
                    var_name = arg_it->getName().str();
                    if (var_name.find("captured_") == 0) {
                        var_name = var_name.substr(9);
                    }
                }
            }

            std::string capture_key = lambda_name + "_capture_" + var_name;

            // First try capture-specific key in symbol tables
            auto it = global_symbol_table_->find(capture_key);
            bool found_in_global = (it != global_symbol_table_->end());
            if (!found_in_global) {
                it = symbol_table_->find(capture_key);
            }

            bool found = found_in_global ? (it != global_symbol_table_->end()) : (it != symbol_table_->end());

            // FALLBACK: Try raw variable name (for top-level global variables)
            if (!found) {
                it = global_symbol_table_->find(var_name);
                found_in_global = (it != global_symbol_table_->end());
                if (!found_in_global) {
                    it = symbol_table_->find(var_name);
                }
                found = found_in_global ? (it != global_symbol_table_->end()) : (it != symbol_table_->end());
                if (found) {
                    eshkol_debug("Gradient: found capture '%s' via raw variable name", var_name.c_str());
                }
            }

            // REPL MODE: Try creating external declaration for capture global
            if (!found && (repl_mode_enabled_ && *repl_mode_enabled_)) {
                std::lock_guard<std::mutex> lock(*repl_mutex_);
                auto sym_it = repl_symbol_addresses_->find(capture_key);
                if (sym_it != repl_symbol_addresses_->end()) {
                    // Create external declaration for capture global
                    GlobalVariable* capture_global = ctx_.module().getGlobalVariable(capture_key);
                    if (!capture_global) {
                        capture_global = new GlobalVariable(
                            ctx_.module(),
                            ctx_.taggedValueType(),
                            false,
                            GlobalValue::ExternalLinkage,
                            nullptr,
                            capture_key
                        );
                    }
                    // MUTABLE CAPTURE FIX: Create storage containing packed pointer
                    // Lambda expects ptr to slot containing {type=INT64, data=ptrtoint(@global)}
                    // Then lambda loads from slot, unpacks data field to get @global
                    Value* global_ptr_int = ctx_.builder().CreatePtrToInt(capture_global, ctx_.int64Type());
                    Value* packed_capture = tagged_.packInt64(global_ptr_int, true);
                    Value* capture_storage = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_capture_storage");
                    ctx_.builder().CreateStore(packed_capture, capture_storage);
                    grad_call_args.push_back(capture_storage);
                    continue;
                }
            }

            if (found && it->second) {
                Value* storage = it->second;
                // ESH-0072/0097: this reverse-mode vector path is emitted even
                // when the run-time input is a scheme vector (the svec forward
                // path actually executes), so its capture code must still verify.
                // A lambda that captures a LOCAL function parameter resolves
                // `storage` to the parameter's Argument, which is a tagged_value
                // STRUCT, not a pointer — ptrtoint on it is invalid IR. Mirror
                // resolveGradientCaptures: a named-let carry pointer ("<var>_cap")
                // is forwarded as-is; a value-typed capture is funneled through a
                // temp slot; only a genuine pointer storage is packed via ptrtoint.
                if (auto* arg = llvm::dyn_cast<llvm::Argument>(storage)) {
                    if (arg->getType()->isPointerTy() &&
                        (arg->getName() == (var_name + "_cap") ||
                         arg->getName() == ("captured_" + var_name))) {
                        // #296: transitive capture — the free variable is the
                        // enclosing function's own `captured_<var>` slot,
                        // already in the callee's single-load convention.
                        // Packing its ADDRESS below handed the differentiated
                        // lambda the slot address as its value; a custom-VJP
                        // callee (vqe-energy) unpacked it as the Hamiltonian
                        // handle and the gradient silently zeroed.
                        grad_call_args.push_back(storage);
                        continue;
                    }
                }
                if (!storage->getType()->isPointerTy()) {
                    Value* val_temp = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_cap_val");
                    ctx_.builder().CreateStore(storage, val_temp);
                    grad_call_args.push_back(val_temp);
                    continue;
                }
                if (isTcoLoopAlloca(storage)) {
                    // ESH-0221: see isTcoLoopAlloca's doc comment. `storage`
                    // is a TCO loop-carried parameter's alloca — the callee
                    // expects a single-load VALUE capture, not the mutable-
                    // variable pointer-marker built below.
                    Value* grad_tco_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), storage);
                    Value* val_temp = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_cap_tco_val");
                    ctx_.builder().CreateStore(grad_tco_val, val_temp);
                    grad_call_args.push_back(val_temp);
                    continue;
                }
                // MUTABLE CAPTURE FIX: Create storage containing packed pointer
                // Lambda expects ptr to slot containing {type=INT64, data=ptrtoint(@storage)}
                // Then lambda loads from slot, unpacks data field to get @storage
                Value* storage_ptr_int = ctx_.builder().CreatePtrToInt(storage, ctx_.int64Type());
                Value* packed_storage = tagged_.packInt64(storage_ptr_int, true);
                Value* capture_storage = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_capture_storage");
                ctx_.builder().CreateStore(packed_storage, capture_storage);
                grad_call_args.push_back(capture_storage);
            } else {
                // MUTABLE CAPTURE FIX: Push null pointer instead of packed zero
                grad_call_args.push_back(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())));
                eshkol_warn("Gradient: capture '%s' not found, using null pointer", var_name.c_str());
            }
        }
    }
    
    // NESTED GRADIENT FIX: Save ctx_.outerAdNodeStorage() before calling function
    // Nested gradients will overwrite it, so we save and restore to support n-dimensional derivatives
    Value* saved_outer_ad_node_vector = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.outerAdNodeStorage());

    // NESTED GRADIENT FIX: Push tape context (saves outer gradient's tape if any)
    pushTapeContext(partial_tape);

    // M1 Migration FIX: Set AD mode flag so vref recognizes AD node pointers in tensors
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 1), ctx_.adModeActive());

    // ESH-0093: publish this pass's active variable node so an inner
    // forward-mode derivative can seed it (jet-lift) and record the mixed
    // partial back onto this tape. Save/restore for nested gradient passes.
    FunctionCallee seed_swap_vector = ctx_.module().getOrInsertFunction(
        "eshkol_ad_seed_swap",
        FunctionType::get(ctx_.ptrType(), {ctx_.ptrType()}, false));
    Value* saved_seed_vector = ctx_.builder().CreateCall(seed_swap_vector, {active_var_node});

    // AD Phase A counter: one primal (user-function) evaluation.
    ctx_.builder().CreateCall(ctx_.module().getOrInsertFunction(
        "eshkol_ad_count_primal",
        FunctionType::get(ctx_.voidType(), {}, false)), {});
    Value* vector_output = ctx_.builder().CreateCall(func_ptr, grad_call_args);

    // ESH-0093: restore the previously active seed node
    ctx_.builder().CreateCall(seed_swap_vector, {saved_seed_vector});

    // M1 Migration FIX: Reset AD mode flag after function call
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 0), ctx_.adModeActive());

    // NESTED GRADIENT FIX: Pop tape context (restores outer gradient's tape if any)
    popTapeContext();

    // NESTED GRADIENT FIX: Restore ctx_.outerAdNodeStorage() after function returns
    ctx_.builder().CreateStore(saved_outer_ad_node_vector, ctx_.outerAdNodeStorage());

    ctx_.builder().CreateBr(after_func_call);
    BasicBlock* vector_call_exit = ctx_.builder().GetInsertBlock();
    
    // Merge scalar and vector outputs
    ctx_.builder().SetInsertPoint(after_func_call);
    PHINode* output_tagged = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "grad_func_output");
    output_tagged->addIncoming(scalar_output, scalar_call_exit);
    output_tagged->addIncoming(vector_output, vector_call_exit);
    
    // Unpack result back to int64
    Value* output_node_int = tagged_.unpackInt64(output_tagged);
    
    // Convert output to AD node pointer
    Value* output_node_ptr = ctx_.builder().CreateIntToPtr(output_node_int,
        PointerType::getUnqual(ctx_.context()));
    
    // CRITICAL FIX: Use type-based detection instead of pointer value heuristic
    // Check if output is actually an AD node by examining its type tag
    Value* output_type = tagged_.getType(output_tagged);
    Value* output_base_type = tagged_.getBaseType(output_type);
    Value* output_is_ad_node = ctx_.builder().CreateICmpEQ(output_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    
    BasicBlock* has_valid_output = BasicBlock::Create(ctx_.context(), "grad_valid_output", current_func);
    BasicBlock* invalid_output = BasicBlock::Create(ctx_.context(), "grad_invalid_output", current_func);
    BasicBlock* after_backward = BasicBlock::Create(ctx_.context(), "grad_after_backward", current_func);
    
    // Branch based on type check (robust detection)
    ctx_.builder().CreateCondBr(output_is_ad_node, has_valid_output, invalid_output);
    
    // Step 6: Run backward pass through computational graph (only for valid AD nodes)
    ctx_.builder().SetInsertPoint(has_valid_output);

    // DOUBLE BACKWARD SETUP: Store the inner variable node and initialize degree counter
    // This enables degree tracking during backward for proper double backward expressions
    ctx_.builder().CreateStore(active_var_node, ctx_.innerVarNodePtr());
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), ctx_.gradientXDegree());

    backpropagate(partial_tape, output_node_ptr);
    ctx_.builder().CreateBr(after_backward);
    
    // Skip backward pass if output is invalid (placeholder function returning scalar)
    ctx_.builder().SetInsertPoint(invalid_output);
    eshkol_debug("Gradient: Skipping backward pass - function returned non-AD value");
    ctx_.builder().CreateBr(after_backward);
    
    ctx_.builder().SetInsertPoint(after_backward);

    // AD Phase A — ONE-PASS decision. If the mixed-record counter is unchanged
    // across this primal+reverse pass, no inner forward-mode derivative ran, so
    // this single reverse sweep already holds the gradient for EVERY input
    // component. On the first iteration we then read them all and finish the loop
    // in one shot (one primal, one backprop). Otherwise (reverse-over-forward, or
    // a later replay iteration) we keep the exact per-component semantics.
    Value* mixed_after = ctx_.builder().CreateCall(ad_mixed_count_fn, {}, "mixed_after");
    Value* mixed_unchanged = ctx_.builder().CreateICmpEQ(mixed_after, mixed_before, "mixed_unchanged");
    Value* is_first_comp = ctx_.builder().CreateICmpEQ(i, ConstantInt::get(ctx_.int64Type(), 0));
    Value* fast_ok = ctx_.builder().CreateAnd(is_first_comp, mixed_unchanged, "grad_fast_ok");

    BasicBlock* readback_all_bb = BasicBlock::Create(ctx_.context(), "grad_readback_all", current_func);
    BasicBlock* single_read_bb  = BasicBlock::Create(ctx_.context(), "grad_single_read", current_func);
    BasicBlock* after_read_bb   = BasicBlock::Create(ctx_.context(), "grad_after_read", current_func);
    ctx_.builder().CreateCondBr(fast_ok, readback_all_bb, single_read_bb);

    // FAST PATH: read gradients for ALL n input variables from this one sweep.
    ctx_.builder().SetInsertPoint(readback_all_bb);
    {
        BasicBlock* rb_cond = BasicBlock::Create(ctx_.context(), "grad_rb_cond", current_func);
        BasicBlock* rb_body = BasicBlock::Create(ctx_.context(), "grad_rb_body", current_func);
        BasicBlock* rb_exit = BasicBlock::Create(ctx_.context(), "grad_rb_exit", current_func);
        Value* rb_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "grad_rb_idx");
        ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), rb_idx);
        ctx_.builder().CreateBr(rb_cond);

        ctx_.builder().SetInsertPoint(rb_cond);
        Value* rb_j = ctx_.builder().CreateLoad(ctx_.int64Type(), rb_idx);
        Value* rb_more = ctx_.builder().CreateICmpULT(rb_j, n);
        ctx_.builder().CreateCondBr(rb_more, rb_body, rb_exit);

        ctx_.builder().SetInsertPoint(rb_body);
        Value* rb_node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
            typed_var_nodes, rb_j);
        Value* rb_node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), rb_node_slot);
        Value* rb_grad = ctx_.builder().CreateSelect(output_is_ad_node,
            loadNodeGradient(rb_node), ConstantFP::get(ctx_.doubleType(), 0.0));
        Value* rb_grad_i64 = ctx_.builder().CreateBitCast(rb_grad, ctx_.int64Type());
        Value* rb_res_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_result_elements_ptr, rb_j);
        ctx_.builder().CreateStore(rb_grad_i64, rb_res_ptr);
        Value* rb_next = ctx_.builder().CreateAdd(rb_j, ConstantInt::get(ctx_.int64Type(), 1));
        ctx_.builder().CreateStore(rb_next, rb_idx);
        ctx_.builder().CreateBr(rb_cond);

        ctx_.builder().SetInsertPoint(rb_exit);
        ctx_.builder().CreateBr(after_read_bb);
    }

    // SLOW PATH: per-component replay — store only the active component (var[i]).
    ctx_.builder().SetInsertPoint(single_read_bb);
    Value* single_grad = ctx_.builder().CreateSelect(output_is_ad_node,
        loadNodeGradient(active_var_node), ConstantFP::get(ctx_.doubleType(), 0.0));
    Value* single_grad_i64 = ctx_.builder().CreateBitCast(single_grad, ctx_.int64Type());
    Value* single_res_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_result_elements_ptr, i);
    ctx_.builder().CreateStore(single_grad_i64, single_res_ptr);
    ctx_.builder().CreateBr(after_read_bb);

    ctx_.builder().SetInsertPoint(after_read_bb);

    // Step 9: Reset tape for next iteration (MUST call to zero gradients)
    ctx_.builder().CreateCall(mem_.getArenaTapeReset(), {partial_tape});

    // Restore previous tape
    current_tape_ptr_ = saved_tape;

    // Advance: the fast path already produced every component, so jump the
    // counter to n (loop exits); otherwise step to the next component.
    Value* next_i = ctx_.builder().CreateSelect(fast_ok, n,
        ctx_.builder().CreateAdd(i, ConstantInt::get(ctx_.int64Type(), 1)));
    ctx_.builder().CreateStore(next_i, component_idx);
    ctx_.builder().CreateBr(grad_loop_cond);
    
    // Loop exit: Return result gradient vector
    ctx_.builder().SetInsertPoint(grad_loop_exit);

    eshkol_info("Gradient computation complete, returning vector of size n");

    // DOUBLE BACKWARD: Check if we have a stored outer AD node
    // If so, create result as AD node on outer tape for proper gradient propagation
    Value* stored_outer = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.outerAdNodeStorage());
    Value* has_outer_node = ctx_.builder().CreateICmpNE(stored_outer,
        ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())));

    // Also check if this is scalar case (n == 1)
    Value* is_scalar_grad = ctx_.builder().CreateICmpEQ(n, ConstantInt::get(ctx_.int64Type(), 1));
    Value* should_return_ad_node = ctx_.builder().CreateAnd(has_outer_node, is_scalar_grad);

    BasicBlock* return_ad_node = BasicBlock::Create(ctx_.context(), "grad_return_ad_node", current_func);
    BasicBlock* return_tensor = BasicBlock::Create(ctx_.context(), "grad_return_tensor", current_func);
    BasicBlock* grad_merge_result = BasicBlock::Create(ctx_.context(), "grad_merge_result", current_func);

    ctx_.builder().CreateCondBr(should_return_ad_node, return_ad_node, return_tensor);

    // DOUBLE BACKWARD PATH: Return AD node connected to outer graph
    ctx_.builder().SetInsertPoint(return_ad_node);

    // Get the scalar gradient value from result tensor
    Value* scalar_grad_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_result_elements_ptr, ConstantInt::get(ctx_.int64Type(), 0));
    Value* scalar_grad_int = ctx_.builder().CreateLoad(ctx_.int64Type(), scalar_grad_ptr);
    Value* scalar_grad_val = ctx_.builder().CreateBitCast(scalar_grad_int, ctx_.doubleType());

    // Get current tape (which IS the outer tape after popTapeContext)
    // After inner gradient's push/pop, ctx_.currentAdTape() is restored to outer tape
    Value* outer_tape_for_result = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.currentAdTape());

    // Create an AD expression on outer tape that connects gradient to input
    // For f(x) = x^n, f'(x) = n*x^(n-1)
    // The gradient depends on x, so we need to express this dependency
    //
    // Key insight: For many functions, f'(x) is approximately proportional to some power of x.
    // We use the chain rule: if result = g(outer) where g is the gradient function,
    // then d(result)/d(outer) is the Hessian.
    //
    // For scalar polynomial-like functions, we can approximate:
    // result = (grad_value / outer_value) * outer
    // This gives d(result)/d(outer) = grad_value / outer_value
    //
    // For f(x) = x^n: f'(x) = n*x^(n-1), so at x=a, f'(a) = n*a^(n-1)
    // f''(x) = n*(n-1)*x^(n-2)
    // f''(a)/f'(a) = (n-1)/a
    // So f''(a) = f'(a) * (n-1) / a
    //
    // We don't know n, but we can compute: f'(a) * derivative_factor
    // where derivative_factor is an approximation based on function structure.
    //
    // For now, use a simple linear connection: result = k * outer
    // where k = grad_value / outer_value
    // This gives d(result)/d(outer) = k = grad_value / outer_value

    // Get the stored outer AD node
    Value* outer_node_for_expr = stored_outer;

    // Get outer node's value
    Value* outer_val_ptr = ctx_.builder().CreateStructGEP(ctx_.adNodeType(), outer_node_for_expr, 1);
    Value* outer_val = ctx_.builder().CreateLoad(ctx_.doubleType(), outer_val_ptr);

    // DEGREE-BASED DOUBLE BACKWARD EXPRESSION
    // The ctx_.gradientXDegree() counter tracks the polynomial degree of f'(x) in x.
    // For f'(x) = k * x^m:
    //   - m = 0 (constant): f'(x) = k, f''(x) = 0
    //   - m = 1 (linear): f'(x) = k*x, f''(x) = k
    //   - m = 2 (quadratic): f'(x) = k*x², f''(x) = 2*k*x
    //
    // We create an AD expression: result = k * x^m where k = grad/x^m
    // This ensures d(result)/dx = k * m * x^(m-1) = correct f''(x)

    // Load the detected degree
    // Note: The counter tracks multiplications by x value during backward.
    // For x²: count=2 (both inputs are x), actual degree = 1
    // For x³: count=3, actual degree = 2
    // So actual_degree = max(0, count - 1)
    Value* raw_count = ctx_.builder().CreateLoad(ctx_.int64Type(), ctx_.gradientXDegree());
    Value* detected_degree = ctx_.builder().CreateSelect(
        ctx_.builder().CreateICmpEQ(raw_count, ConstantInt::get(ctx_.int64Type(), 0)),
        ConstantInt::get(ctx_.int64Type(), 0),
        ctx_.builder().CreateSub(raw_count, ConstantInt::get(ctx_.int64Type(), 1)));

    // N-DIMENSIONAL DERIVATIVES: Support arbitrary polynomial degree
    // For f'(x) = k * x^n:
    //   - Compute outer_val^n to get scale factor k = grad / (outer_val^n)
    //   - Build AD expression: k * x^n using repeated multiplication
    //   - This ensures d(result)/dx = k * n * x^(n-1) = correct higher derivative

    // Create blocks for degree handling
    BasicBlock* degree_0_bb = BasicBlock::Create(ctx_.context(), "degree_0", current_func);
    BasicBlock* degree_n_bb = BasicBlock::Create(ctx_.context(), "degree_n", current_func);
    BasicBlock* degree_merge_bb = BasicBlock::Create(ctx_.context(), "degree_merge", current_func);

    // Check if degree is 0 (constant - no x dependency)
    Value* is_degree_0 = ctx_.builder().CreateICmpEQ(detected_degree, ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(is_degree_0, degree_0_bb, degree_n_bb);

    // DEGREE 0: Constant gradient, f''(x) = 0
    // Result is just a constant AD node (no x dependency)
    ctx_.builder().SetInsertPoint(degree_0_bb);
    Value* const_result_node = createADConstantOnTape(outer_tape_for_result, scalar_grad_val);
    ctx_.builder().CreateBr(degree_merge_bb);
    BasicBlock* degree_0_exit = ctx_.builder().GetInsertBlock();

    // DEGREE N: Polynomial gradient f'(x) = k*x^n
    // Result = k * x^n where k = grad/x^n
    // We compute x^n both as a double (for k) and as AD expression (for result)
    ctx_.builder().SetInsertPoint(degree_n_bb);

    // Compute outer_val^n using a loop
    Value* pow_val_ptr = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "pow_val");
    ctx_.builder().CreateStore(ConstantFP::get(ctx_.doubleType(), 1.0), pow_val_ptr);
    Value* pow_idx_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "pow_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), pow_idx_ptr);

    BasicBlock* pow_loop_cond = BasicBlock::Create(ctx_.context(), "pow_loop_cond", current_func);
    BasicBlock* pow_loop_body = BasicBlock::Create(ctx_.context(), "pow_loop_body", current_func);
    BasicBlock* pow_loop_exit = BasicBlock::Create(ctx_.context(), "pow_loop_exit", current_func);

    ctx_.builder().CreateBr(pow_loop_cond);

    ctx_.builder().SetInsertPoint(pow_loop_cond);
    Value* pow_i = ctx_.builder().CreateLoad(ctx_.int64Type(), pow_idx_ptr);
    Value* pow_continue = ctx_.builder().CreateICmpULT(pow_i, detected_degree);
    ctx_.builder().CreateCondBr(pow_continue, pow_loop_body, pow_loop_exit);

    ctx_.builder().SetInsertPoint(pow_loop_body);
    Value* current_pow = ctx_.builder().CreateLoad(ctx_.doubleType(), pow_val_ptr);
    Value* next_pow = ctx_.builder().CreateFMul(current_pow, outer_val);
    ctx_.builder().CreateStore(next_pow, pow_val_ptr);
    Value* next_pow_i = ctx_.builder().CreateAdd(pow_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_pow_i, pow_idx_ptr);
    ctx_.builder().CreateBr(pow_loop_cond);

    ctx_.builder().SetInsertPoint(pow_loop_exit);
    Value* outer_val_pow_n = ctx_.builder().CreateLoad(ctx_.doubleType(), pow_val_ptr);

    // Compute scale factor k = grad / x^n
    Value* scale_factor_n = ctx_.builder().CreateFDiv(scalar_grad_val, outer_val_pow_n);
    Value* scale_const_n = createADConstantOnTape(outer_tape_for_result, scale_factor_n);

    // Build AD expression x^n using repeated multiplication
    // Start with x, then multiply by x (n-1) more times
    Value* ad_pow_ptr = ctx_.builder().CreateAlloca(PointerType::getUnqual(ctx_.context()), nullptr, "ad_pow");
    ctx_.builder().CreateStore(outer_node_for_expr, ad_pow_ptr);
    Value* ad_pow_idx_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "ad_pow_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), ad_pow_idx_ptr);

    BasicBlock* ad_pow_loop_cond = BasicBlock::Create(ctx_.context(), "ad_pow_loop_cond", current_func);
    BasicBlock* ad_pow_loop_body = BasicBlock::Create(ctx_.context(), "ad_pow_loop_body", current_func);
    BasicBlock* ad_pow_loop_exit = BasicBlock::Create(ctx_.context(), "ad_pow_loop_exit", current_func);

    ctx_.builder().CreateBr(ad_pow_loop_cond);

    ctx_.builder().SetInsertPoint(ad_pow_loop_cond);
    Value* ad_pow_i = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_pow_idx_ptr);
    Value* ad_pow_continue = ctx_.builder().CreateICmpULT(ad_pow_i, detected_degree);
    ctx_.builder().CreateCondBr(ad_pow_continue, ad_pow_loop_body, ad_pow_loop_exit);

    ctx_.builder().SetInsertPoint(ad_pow_loop_body);
    Value* current_ad_pow = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ad_pow_ptr);
    // Multiply current AD expression by x: current * x
    Value* next_ad_pow = recordADNodeBinaryOnTape(outer_tape_for_result, 4, current_ad_pow, outer_node_for_expr);
    ctx_.builder().CreateStore(next_ad_pow, ad_pow_ptr);
    Value* next_ad_pow_i = ctx_.builder().CreateAdd(ad_pow_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_ad_pow_i, ad_pow_idx_ptr);
    ctx_.builder().CreateBr(ad_pow_loop_cond);

    ctx_.builder().SetInsertPoint(ad_pow_loop_exit);
    Value* outer_pow_n_ad = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ad_pow_ptr);

    // Final result: k * x^n
    Value* poly_result = recordADNodeBinaryOnTape(outer_tape_for_result, 4, scale_const_n, outer_pow_n_ad);
    ctx_.builder().CreateBr(degree_merge_bb);
    BasicBlock* degree_n_exit = ctx_.builder().GetInsertBlock();

    // Merge results
    ctx_.builder().SetInsertPoint(degree_merge_bb);
    PHINode* result_ad_node = ctx_.builder().CreatePHI(PointerType::getUnqual(ctx_.context()), 2, "degree_result");
    result_ad_node->addIncoming(const_result_node, degree_0_exit);
    result_ad_node->addIncoming(poly_result, degree_n_exit);

    // Pack AD node as result
    Value* ad_result_int = ctx_.builder().CreatePtrToInt(result_ad_node, ctx_.int64Type());
    Value* ad_result_tagged = tagged_.packPtr(ad_result_int, ESHKOL_VALUE_CALLABLE);

    // Clear the outer AD node storage
    ctx_.builder().CreateStore(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())),
        ctx_.outerAdNodeStorage());

    ctx_.builder().CreateBr(grad_merge_result);
    BasicBlock* ad_result_exit = ctx_.builder().GetInsertBlock();

    // NORMAL PATH: Return tensor as before
    ctx_.builder().SetInsertPoint(return_tensor);
    Value* grad_result_int = ctx_.builder().CreatePtrToInt(typed_result_tensor_ptr, ctx_.int64Type());
    // Tag as TENSOR_PTR for proper display handling (packPtrToTaggedValue handles i64 directly)
    Value* grad_result = tagged_.packPtr(grad_result_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(grad_merge_result);
    BasicBlock* tensor_result_exit = ctx_.builder().GetInsertBlock();

    // Merge paths
    ctx_.builder().SetInsertPoint(grad_merge_result);
    PHINode* final_result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "grad_final_result");
    final_result->addIncoming(ad_result_tagged, ad_result_exit);
    final_result->addIncoming(grad_result, tensor_result_exit);

    ctx_.builder().CreateBr(grad_done);
    BasicBlock* dim_valid_exit = ctx_.builder().GetInsertBlock();
    
    // Merge valid, invalid, and scheme vector forward-mode paths
    ctx_.builder().SetInsertPoint(grad_done);
    PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "grad_result_final");
    result_phi->addIncoming(null_tagged_grad, dim_invalid);
    result_phi->addIncoming(final_result, dim_valid_exit);  // Use merged result from double backward handling
    result_phi->addIncoming(scheme_vector_tagged, scheme_vector_exit);  // Forward-mode AD for Scheme vectors

    // SCALAR INPUT FIX: If input was a scalar, extract element 0 from result tensor
    // and return as a scalar double (not a 1-element tensor)
    BasicBlock* scalar_extract_bb = BasicBlock::Create(ctx_.context(), "grad_scalar_extract", current_func);
    BasicBlock* grad_final_bb = BasicBlock::Create(ctx_.context(), "grad_final", current_func);
    ctx_.builder().CreateCondBr(is_scalar, scalar_extract_bb, grad_final_bb);

    ctx_.builder().SetInsertPoint(scalar_extract_bb);
    // Result is a 1-element tensor — extract the double from element 0
    Value* result_ptr = tagged_.unpackPtr(result_phi);
    // tensor struct: field 2 = elements pointer
    Value* elems_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), result_ptr, 2));
    Value* elem_as_int64 = ctx_.builder().CreateLoad(ctx_.int64Type(), elems_ptr);
    Value* elem_double = ctx_.builder().CreateBitCast(elem_as_int64, ctx_.doubleType());
    Value* scalar_result = tagged_.packDouble(elem_double);
    BasicBlock* scalar_extract_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(grad_final_bb);

    ctx_.builder().SetInsertPoint(grad_final_bb);
    PHINode* final_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "grad_final_result");
    final_phi->addIncoming(scalar_result, scalar_extract_exit);
    final_phi->addIncoming(result_phi, grad_done);

    // ESH-0070: if the forward-mode jet fast path was set up, merge the two
    // paths (forward scalar result vs reverse-mode vector result) here.
    if (grad_unified_exit) {
        ctx_.builder().CreateStore(final_phi, grad_result_slot);
        ctx_.builder().CreateBr(grad_unified_exit);
        ctx_.builder().SetInsertPoint(grad_unified_exit);
        return ctx_.builder().CreateLoad(ctx_.taggedValueType(), grad_result_slot);
    }

    return final_phi;
}


/**
 * @brief Compute the Jacobian matrix of a vector-valued function at a point,
 *        via reverse-mode AD.
 *
 * Resolves the target function (compile-time Function*, or a runtime closure
 * loaded from the symbol tables / REPL cross-module registry), then evaluates
 * the input point and normalizes it to tensor form (converting a Scheme
 * vector input by copying its tagged elements into a fresh tensor). The
 * function is called once with the raw (non-AD) point to determine the
 * output dimension m, validating that the output is a tensor, Scheme vector,
 * AD tensor, or CALLABLE (erroring otherwise). A `[m, n]` tensor is then
 * allocated for the Jacobian.
 *
 * For each output row i (0..m-1): a fresh AD tape (`jac_tape`, tracked via a
 * dedicated global pointer rather than the class's `current_tape_ptr_`, since
 * that member is compile-time C++ state and would be corrupted by a runtime
 * LLVM Value*) gets n freshly-initialized AD variable nodes wrapping the
 * input components, an AD tensor is built from them, and the function is
 * called with AD mode enabled to build the computational graph. The i-th
 * output component is extracted (handling both Scheme-vector and tensor
 * output layouts) and runtime-checked to see whether it is actually an AD
 * node (vs. a constant double, e.g. for a constant-output component) via a
 * heap-pointer/IEEE754-exponent heuristic. For each input j (0..n-1),
 * backpropagate() runs from that output node (skipped, defaulting to 0, if
 * the output element is not an AD node) and the gradient at variable j is
 * read off into `J[i, j] = ∂F_i/∂x_j`. The 2-D Jacobian tensor is returned
 * directly (bit-pattern-adjusted from double to the int64 tensor element
 * encoding).
 *
 * @param op The `jacobian` AST operation node (`op->jacobian_op.function` and
 *           `op->jacobian_op.point`).
 * @return Tagged 2-D tensor value (m x n Jacobian), or nullptr on failure
 *         (unresolved function, invalid point, or invalid/unrecognized output
 *         type from the function).
 */
llvm::Value* AutodiffCodegen::jacobian(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->jacobian_op.function || !op->jacobian_op.point) {
        eshkol_error("Invalid jacobian operation - missing function or point");
        return nullptr;
    }
    
    // Use class member ctx_.tensorType() (shared by ALL tensor operations)
    // This prevents LLVM IR type conflicts from shadowing the class member
    
    eshkol_info("Computing Jacobian matrix using reverse-mode AD");
    
    // CRITICAL FIX: Must null-check before dyn_cast to avoid LLVM assertion
    Value* func = resolve_lambda_callback_(op->jacobian_op.function, 0, callback_context_);

    Function* func_ptr = func ? dyn_cast<Function>(func) : nullptr;

    // CLOSURE FALLBACK: if resolve_lambda_callback_ returned nullptr or a non-Function*
    // (e.g. the function is a top-level variable holding a runtime closure), load the
    // closure value from the symbol tables and call through closure_call_callback_.
    Value* closure_val = nullptr;
    if (!func_ptr) {
        const eshkol_ast_t* func_ast = op->jacobian_op.function;
        if (func_ast && func_ast->type == ESHKOL_VAR) {
            std::string func_name = func_ast->variable.id;
            Value* var_value = nullptr;

            // Try local symbol table first
            auto lit = symbol_table_->find(func_name);
            if (lit != symbol_table_->end() && lit->second) {
                var_value = lit->second;
            }
            // Try global symbol table (top-level defines)
            if (!var_value) {
                auto git = global_symbol_table_->find(func_name);
                if (git != global_symbol_table_->end() && git->second) {
                    var_value = git->second;
                }
            }
            // REPL MODE: look up in cross-evaluation symbol address registry
            if (!var_value && repl_mode_enabled_ && *repl_mode_enabled_) {
                std::lock_guard<std::mutex> lock(*repl_mutex_);
                auto repl_it = repl_symbol_addresses_->find(func_name);
                if (repl_it != repl_symbol_addresses_->end()) {
                    GlobalVariable* gv = ctx_.module().getGlobalVariable(func_name);
                    if (!gv) {
                        gv = new GlobalVariable(
                            ctx_.module(), ctx_.taggedValueType(), false,
                            GlobalValue::ExternalLinkage, nullptr, func_name);
                    }
                    var_value = gv;
                }
            }

            if (var_value) {
                if (isa<GlobalVariable>(var_value)) {
                    GlobalVariable* gv = cast<GlobalVariable>(var_value);
                    closure_val = ctx_.builder().CreateLoad(gv->getValueType(), gv);
                } else if (isa<AllocaInst>(var_value)) {
                    closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                } else if (isa<Argument>(var_value)) {
                    if (var_value->getType()->isPointerTy()) {
                        closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                    } else {
                        closure_val = var_value;
                    }
                } else if (var_value->getType() == ctx_.taggedValueType()) {
                    closure_val = var_value;
                }
            }
        }
        if (!closure_val) {
            eshkol_error("Failed to resolve function for Jacobian computation");
            return nullptr;
        }
    }
    
    llvm::Value* vector_val_raw = codegen_ast_callback_(op->jacobian_op.point, callback_context_);
    if (!vector_val_raw) {
        eshkol_error("Failed to evaluate Jacobian point");
        return nullptr;
    }

    // Get arena for OALR-compliant tensor allocation
    Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // CRITICAL FIX: Handle Scheme VECTOR_PTR - convert to tensor format
    // Get current function for basic blocks
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Convert TypedValue to tagged_value
    // Tensor literal fix: codegenTensor returns ptr-as-int64 which gets packed as INT64;
    // re-pack as HEAP_PTR so type dispatch correctly detects tensor subtype
    Value* vector_val = vector_val_raw;
    if (vector_val && vector_val->getType() != ctx_.taggedValueType()) {
        if (vector_val->getType()->isIntegerTy(64)) {
            if (op->jacobian_op.point->type == ESHKOL_TENSOR) {
                vector_val = tagged_.packPtr(vector_val, ESHKOL_VALUE_HEAP_PTR);
            } else {
                vector_val = tagged_.packInt64(vector_val, true);
            }
        } else if (vector_val->getType()->isDoubleTy()) {
            vector_val = tagged_.packDouble(vector_val);
        }
    }
    if (op->jacobian_op.point->type == ESHKOL_TENSOR) {
        Value* data_val = tagged_.unpackInt64(vector_val);
        vector_val = tagged_.packPtr(data_val, ESHKOL_VALUE_HEAP_PTR);
    }

    Value* input_type = tagged_.getType(vector_val);
    Value* input_base_type = tagged_.getBaseType(input_type);

    // M1 CONSOLIDATION: Check for both HEAP_PTR (consolidated) and legacy VECTOR_PTR
    Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* is_legacy_vector = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    BasicBlock* jac_heap_dispatch = BasicBlock::Create(ctx_.context(), "jac_heap_dispatch", current_func);
    BasicBlock* jac_check_legacy = BasicBlock::Create(ctx_.context(), "jac_check_legacy", current_func);
    BasicBlock* jac_scheme_vector_input = BasicBlock::Create(ctx_.context(), "jac_scheme_vector", current_func);
    BasicBlock* jac_tensor_input = BasicBlock::Create(ctx_.context(), "jac_tensor_input", current_func);
    BasicBlock* jac_merge_input = BasicBlock::Create(ctx_.context(), "jac_merge_input", current_func);

    // First check for HEAP_PTR (consolidated format)
    ctx_.builder().CreateCondBr(is_heap_ptr, jac_heap_dispatch, jac_check_legacy);

    // HEAP_PTR dispatch - read subtype from header
    ctx_.builder().SetInsertPoint(jac_heap_dispatch);
    Value* jac_heap_ptr_val = tagged_.unpackPtr(vector_val);
    Value* jac_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), jac_heap_ptr_val, ConstantInt::get(ctx_.int64Type(), -8));
    Value* jac_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), jac_header_ptr);
    Value* jac_is_vec_subtype = ctx_.builder().CreateICmpEQ(jac_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    ctx_.builder().CreateCondBr(jac_is_vec_subtype, jac_scheme_vector_input, jac_tensor_input);

    // Legacy VECTOR_PTR fallback
    ctx_.builder().SetInsertPoint(jac_check_legacy);
    ctx_.builder().CreateCondBr(is_legacy_vector, jac_scheme_vector_input, jac_tensor_input);

    // SCHEME VECTOR: Convert to tensor format
    ctx_.builder().SetInsertPoint(jac_scheme_vector_input);

    Value* jac_scheme_vec_ptr_int = tagged_.unpackInt64(vector_val);
    Value* jac_scheme_vec_ptr = ctx_.builder().CreateIntToPtr(jac_scheme_vec_ptr_int, ctx_.builder().getPtrTy());
    Value* jac_scheme_len_ptr = ctx_.builder().CreateBitCast(jac_scheme_vec_ptr, PointerType::getUnqual(ctx_.context()));
    Value* jac_scheme_len = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_scheme_len_ptr);

    // Allocate tensor via arena (OALR compliant - no malloc)
    Value* jac_typed_scheme_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions
    Value* jac_scheme_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* jac_scheme_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_scheme_dims_size});
    Value* jac_typed_scheme_dims = ctx_.builder().CreatePointerCast(jac_scheme_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(jac_scheme_len, jac_typed_scheme_dims);

    ctx_.builder().CreateStore(jac_typed_scheme_dims, ctx_.builder().CreateStructGEP(ctx_.tensorType(), jac_typed_scheme_tensor, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), jac_typed_scheme_tensor, 1));
    ctx_.builder().CreateStore(jac_scheme_len, ctx_.builder().CreateStructGEP(ctx_.tensorType(), jac_typed_scheme_tensor, 3));

    // Allocate and copy elements
    Value* jac_scheme_elems_size = ctx_.builder().CreateMul(jac_scheme_len,
        ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    Value* jac_scheme_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_scheme_elems_size});
    Value* jac_typed_scheme_elems = ctx_.builder().CreatePointerCast(jac_scheme_elems_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(jac_typed_scheme_elems, ctx_.builder().CreateStructGEP(ctx_.tensorType(), jac_typed_scheme_tensor, 2));

    // Copy elements loop
    Value* jac_scheme_elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), jac_scheme_vec_ptr,
        ConstantInt::get(ctx_.int64Type(), 8));
    Value* jac_scheme_elem_base_typed = ctx_.builder().CreateBitCast(jac_scheme_elem_base, PointerType::getUnqual(ctx_.context()));

    BasicBlock* jac_svec_copy_cond = BasicBlock::Create(ctx_.context(), "jac_svec_copy_cond", current_func);
    BasicBlock* jac_svec_copy_body = BasicBlock::Create(ctx_.context(), "jac_svec_copy_body", current_func);
    BasicBlock* jac_svec_copy_done = BasicBlock::Create(ctx_.context(), "jac_svec_copy_done", current_func);

    Value* jac_svec_copy_i = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "jac_svec_copy_i");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), jac_svec_copy_i);
    ctx_.builder().CreateBr(jac_svec_copy_cond);

    ctx_.builder().SetInsertPoint(jac_svec_copy_cond);
    Value* jac_svec_i = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_svec_copy_i);
    Value* jac_svec_cond = ctx_.builder().CreateICmpULT(jac_svec_i, jac_scheme_len);
    ctx_.builder().CreateCondBr(jac_svec_cond, jac_svec_copy_body, jac_svec_copy_done);

    ctx_.builder().SetInsertPoint(jac_svec_copy_body);
    Value* jac_svec_src_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), jac_scheme_elem_base_typed, jac_svec_i);
    Value* jac_svec_tagged_elem = ctx_.builder().CreateLoad(ctx_.taggedValueType(), jac_svec_src_ptr);
    Value* jac_svec_double_val = tagged_.unpackDouble(jac_svec_tagged_elem);
    Value* jac_svec_as_int64 = ctx_.builder().CreateBitCast(jac_svec_double_val, ctx_.int64Type());
    Value* jac_svec_dst_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), jac_typed_scheme_elems, jac_svec_i);
    ctx_.builder().CreateStore(jac_svec_as_int64, jac_svec_dst_ptr);
    Value* jac_svec_next_i = ctx_.builder().CreateAdd(jac_svec_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(jac_svec_next_i, jac_svec_copy_i);
    ctx_.builder().CreateBr(jac_svec_copy_cond);

    ctx_.builder().SetInsertPoint(jac_svec_copy_done);
    Value* jac_scheme_tensor_int = ctx_.builder().CreatePtrToInt(jac_typed_scheme_tensor, ctx_.int64Type());
    Value* jac_scheme_vector_tagged = tagged_.packPtr(jac_scheme_tensor_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(jac_merge_input);
    BasicBlock* jac_scheme_exit = ctx_.builder().GetInsertBlock();

    // TENSOR INPUT: Use as-is
    ctx_.builder().SetInsertPoint(jac_tensor_input);
    ctx_.builder().CreateBr(jac_merge_input);
    BasicBlock* jac_tensor_exit = ctx_.builder().GetInsertBlock();

    // MERGE
    ctx_.builder().SetInsertPoint(jac_merge_input);
    PHINode* jac_actual_input = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "jac_input");
    jac_actual_input->addIncoming(jac_scheme_vector_tagged, jac_scheme_exit);
    jac_actual_input->addIncoming(vector_val, jac_tensor_exit);

    // Extract tensor pointer from merged input
    Value* vector_ptr_int = tagged_.unpackInt64(jac_actual_input);

    // Extract input dimension n from input vector
    Value* input_ptr = ctx_.builder().CreateIntToPtr(vector_ptr_int, ctx_.builder().getPtrTy());
    
    Value* input_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), input_ptr, 0);
    Value* input_dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), input_dims_field);
    Value* typed_input_dims = ctx_.builder().CreatePointerCast(input_dims_ptr, ctx_.builder().getPtrTy());
    
    Value* input_elements_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), input_ptr, 2);
    Value* input_elements_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), input_elements_field);
    Value* typed_input_elements = ctx_.builder().CreatePointerCast(input_elements_ptr, ctx_.builder().getPtrTy());
    
    Value* n_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_input_dims,
        ConstantInt::get(ctx_.int64Type(), 0));
    Value* n = ctx_.builder().CreateLoad(ctx_.int64Type(), n_ptr);

    // Call function once to determine output dimension m
    // CRITICAL FIX: Pack as TENSOR_PTR not INT64, so identity lambdas preserve type
    Value* vector_tagged = tagged_.packPtr(vector_ptr_int, ESHKOL_VALUE_HEAP_PTR);
    Value* test_output_tagged;
    if (func_ptr) {
        // Compile-time resolved function: direct call with explicit captures
        std::vector<Value*> test_call_args = {vector_tagged};
        std::vector<Value*> jac_test_captures = loadCapturesForAutodiff(func_ptr, "Jacobian test call");
        test_call_args.insert(test_call_args.end(), jac_test_captures.begin(), jac_test_captures.end());
        test_output_tagged = ctx_.builder().CreateCall(func_ptr, test_call_args);
    } else {
        // Runtime closure path — captures are embedded inside the closure struct
        test_output_tagged = closure_call_callback_(closure_val, {vector_tagged}, "jacobian-test", callback_context_);
    }

    // ENHANCED TYPE CHECK: Accept tensors, AD tensors, AND Scheme vectors as valid outputs
    Value* output_type = tagged_.getType(test_output_tagged);
    Value* output_base_type = tagged_.getBaseType(output_type);

    // M1 CONSOLIDATION: Check for valid output types
    // For HEAP_PTR, we need to check the subtype to distinguish vector (2) from tensor (3)
    Value* output_is_heap_ptr = ctx_.builder().CreateICmpEQ(output_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* output_is_callable = ctx_.builder().CreateICmpEQ(output_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));

    // Any HEAP_PTR or CALLABLE is a valid vector type (tensor or vector)
    Value* output_has_vector_type = ctx_.builder().CreateOr(output_is_heap_ptr, output_is_callable);

    // CRITICAL FIX: Create null tagged value BEFORE branching (for PHI node dominance)
    Value* null_jac_tagged = tagged_.packInt64(
        ConstantInt::get(ctx_.int64Type(), 0), true);

    // Create blocks for validation flow
    BasicBlock* output_valid_block = BasicBlock::Create(ctx_.context(), "jac_output_valid", current_func);
    BasicBlock* output_invalid_block = BasicBlock::Create(ctx_.context(), "jac_output_invalid", current_func);
    BasicBlock* jac_return_block = BasicBlock::Create(ctx_.context(), "jac_return", current_func);

    ctx_.builder().CreateCondBr(output_has_vector_type, output_valid_block, output_invalid_block);

    // Invalid output: Generate runtime code to extract and report actual type value
    ctx_.builder().SetInsertPoint(output_invalid_block);
    // This block now only reached for genuinely invalid types (NULL, INT64, DOUBLE, CONS_PTR)
    Function* printf_func_for_error = ctx_.lookupFunction("printf");
    if (printf_func_for_error) {
        // Create alloca for type value at function entry to ensure dominance
        IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
        Function* func = ctx_.builder().GetInsertBlock()->getParent();
        if (func && !func->empty()) {
            BasicBlock& entry = func->getEntryBlock();
            ctx_.builder().SetInsertPoint(&entry, entry.begin());
        }
        Value* type_storage = ctx_.builder().CreateAlloca(ctx_.int8Type(), nullptr, "invalid_type");
        ctx_.builder().restoreIP(saved_ip);

        // Store the runtime type value and extend to int for printf
        ctx_.builder().CreateStore(output_base_type, type_storage);
        Value* type_val = ctx_.builder().CreateLoad(ctx_.int8Type(), type_storage);
        Value* type_as_int = ctx_.builder().CreateZExt(type_val, ctx_.int32Type());

        // Print error with actual runtime type value (provides better debugging!)
        ctx_.builder().CreateCall(printf_func_for_error, {
            ctx_.internString("Jacobian ERROR: function returned non-vector type %d (expected 6=TENSOR, 5=AD_TENSOR, or 4=VECTOR_PTR)\n"),
            type_as_int
        });
    }
    ctx_.builder().CreateBr(jac_return_block);

    // Valid output: Handle both tensor and Scheme vector formats
    ctx_.builder().SetInsertPoint(output_valid_block);


    // Branch based on whether output is Scheme vector or tensor
    // For HEAP_PTR, check subtype to distinguish vector (2) from tensor (3)
    BasicBlock* jac_output_check_subtype = BasicBlock::Create(ctx_.context(), "jac_output_check_subtype", current_func);
    BasicBlock* jac_output_scheme_vec = BasicBlock::Create(ctx_.context(), "jac_output_scheme_vec", current_func);
    BasicBlock* jac_output_tensor = BasicBlock::Create(ctx_.context(), "jac_output_tensor", current_func);
    BasicBlock* jac_output_merge = BasicBlock::Create(ctx_.context(), "jac_output_merge", current_func);

    // If HEAP_PTR, check subtype; otherwise go to tensor path (AD_TENSOR/CALLABLE)
    ctx_.builder().CreateCondBr(output_is_heap_ptr, jac_output_check_subtype, jac_output_tensor);

    // Check subtype in header to distinguish Scheme vector from tensor
    ctx_.builder().SetInsertPoint(jac_output_check_subtype);
    Value* test_out_ptr_int = tagged_.unpackInt64(test_output_tagged);
    Value* test_out_ptr = ctx_.builder().CreateIntToPtr(test_out_ptr_int, ctx_.builder().getPtrTy());
    Value* test_out_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), test_out_ptr, ConstantInt::get(ctx_.int64Type(), -8));
    Value* test_out_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), test_out_header_ptr);
    Value* test_out_is_svec = ctx_.builder().CreateICmpEQ(test_out_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    ctx_.builder().CreateCondBr(test_out_is_svec, jac_output_scheme_vec, jac_output_tensor);

    // SCHEME VECTOR OUTPUT: Extract dimension directly from vector length
    ctx_.builder().SetInsertPoint(jac_output_scheme_vec);
    Value* jac_out_svec_ptr_int = tagged_.unpackInt64(test_output_tagged);
    Value* jac_out_svec_ptr = ctx_.builder().CreateIntToPtr(jac_out_svec_ptr_int, ctx_.builder().getPtrTy());
    Value* jac_out_svec_len_ptr = ctx_.builder().CreateBitCast(jac_out_svec_ptr, PointerType::getUnqual(ctx_.context()));
    Value* jac_out_svec_m = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_out_svec_len_ptr);
    ctx_.builder().CreateBr(jac_output_merge);
    BasicBlock* jac_out_svec_exit = ctx_.builder().GetInsertBlock();

    // TENSOR OUTPUT: Extract dimension from tensor structure
    ctx_.builder().SetInsertPoint(jac_output_tensor);
    Value* test_output_int = tagged_.unpackInt64(test_output_tagged);
    Value* test_output_ptr = ctx_.builder().CreateIntToPtr(test_output_int, ctx_.builder().getPtrTy());

    Value* output_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), test_output_ptr, 0);
    Value* output_dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), output_dims_field);

    Value* typed_output_dims = ctx_.builder().CreatePointerCast(output_dims_ptr, ctx_.builder().getPtrTy());

    Value* m_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_output_dims,
        ConstantInt::get(ctx_.int64Type(), 0));

    Value* jac_out_tensor_m = ctx_.builder().CreateLoad(ctx_.int64Type(), m_ptr);
    ctx_.builder().CreateBr(jac_output_merge);
    BasicBlock* jac_out_tensor_exit = ctx_.builder().GetInsertBlock();

    // MERGE: Get m from whichever path we took
    ctx_.builder().SetInsertPoint(jac_output_merge);
    PHINode* m = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "jac_output_m");
    m->addIncoming(jac_out_svec_m, jac_out_svec_exit);
    m->addIncoming(jac_out_tensor_m, jac_out_tensor_exit);
    
    // Allocate Jacobian matrix via arena (OALR compliant - no malloc)
    Value* typed_jac_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions [m, n]
    Value* jac_dims_size = ctx_.builder().CreateMul(
        ConstantInt::get(ctx_.int64Type(), 2),
        ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t)));
    Value* jac_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_dims_size});
    Value* typed_jac_dims = ctx_.builder().CreatePointerCast(jac_dims_ptr, ctx_.builder().getPtrTy());

    ctx_.builder().CreateStore(m, typed_jac_dims);
    Value* jac_dim1_slot = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_jac_dims,
        ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(n, jac_dim1_slot);

    // Store dimensions in tensor
    Value* jac_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ptr, 0);
    ctx_.builder().CreateStore(typed_jac_dims, jac_dims_field);

    // Set num_dimensions = 2
    Value* jac_num_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ptr, 1);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 2), jac_num_dims_field);

    // Set total_elements = m * n
    Value* total_elems = ctx_.builder().CreateMul(m, n);
    Value* jac_total_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ptr, 3);
    ctx_.builder().CreateStore(total_elems, jac_total_field);

    // Allocate elements array (m*n doubles)
    Value* jac_elems_size = ctx_.builder().CreateMul(total_elems,
        ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    Value* jac_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_elems_size});
    Value* typed_jac_elems = ctx_.builder().CreatePointerCast(jac_elems_ptr, ctx_.builder().getPtrTy());
    
    Value* jac_elems_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ptr, 2);
    ctx_.builder().CreateStore(typed_jac_elems, jac_elems_field);

    BasicBlock* outer_cond = BasicBlock::Create(ctx_.context(), "jac_outer_cond", current_func);
    BasicBlock* outer_body = BasicBlock::Create(ctx_.context(), "jac_outer_body", current_func);
    BasicBlock* inner_cond = BasicBlock::Create(ctx_.context(), "jac_inner_cond", current_func);
    BasicBlock* inner_body = BasicBlock::Create(ctx_.context(), "jac_inner_body", current_func);
    BasicBlock* inner_exit = BasicBlock::Create(ctx_.context(), "jac_inner_exit", current_func);
    BasicBlock* outer_exit = BasicBlock::Create(ctx_.context(), "jac_outer_exit", current_func);

    Value* out_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "out_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), out_idx);

    ctx_.builder().CreateBr(outer_cond);
    
    // Outer: i_out < m
    ctx_.builder().SetInsertPoint(outer_cond);
    Value* i_out = ctx_.builder().CreateLoad(ctx_.int64Type(), out_idx);
    Value* i_out_less_m = ctx_.builder().CreateICmpULT(i_out, m);
    ctx_.builder().CreateCondBr(i_out_less_m, outer_body, outer_exit);
    
    ctx_.builder().SetInsertPoint(outer_body);

    Value* in_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "in_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), in_idx);
    ctx_.builder().CreateBr(inner_cond);
    
    // Inner: j_in < n
    ctx_.builder().SetInsertPoint(inner_cond);
    Value* j_in = ctx_.builder().CreateLoad(ctx_.int64Type(), in_idx);
    Value* j_in_less_n = ctx_.builder().CreateICmpULT(j_in, n);
    ctx_.builder().CreateCondBr(j_in_less_n, inner_body, inner_exit);
    
    // Compute ∂Fᵢ/∂xⱼ
    ctx_.builder().SetInsertPoint(inner_body);

    // arena_ptr defined at function start
    Value* jac_tape = ctx_.builder().CreateCall(mem_.getArenaAllocateTape(),
        {arena_ptr, ConstantInt::get(ctx_.int64Type(), 1024)});
    
    // CRITICAL FIX: Use global AD tape pointer, not member variable!
    // current_tape_ptr is compile-time C++ state, jac_tape is runtime LLVM Value*
    // Assigning Value* to member variable corrupts memory - use global instead
    ctx_.builder().CreateStore(jac_tape, ctx_.currentAdTape());
    
    // Create n AD variable nodes via arena (OALR compliant - no malloc)
    Value* jac_var_nodes_size = ctx_.builder().CreateMul(n,
        ConstantInt::get(ctx_.int64Type(), sizeof(void*)));
    Value* jac_var_nodes = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_var_nodes_size});
    Value* typed_jac_var_nodes = ctx_.builder().CreatePointerCast(jac_var_nodes, ctx_.builder().getPtrTy());
    
    // Initialize all variable nodes with input values
    BasicBlock* jac_init_cond = BasicBlock::Create(ctx_.context(), "jac_init_cond", current_func);
    BasicBlock* jac_init_body = BasicBlock::Create(ctx_.context(), "jac_init_body", current_func);
    BasicBlock* jac_init_exit = BasicBlock::Create(ctx_.context(), "jac_init_exit", current_func);
    
    Value* jac_init_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "jac_init_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), jac_init_idx);

    ctx_.builder().CreateBr(jac_init_cond);
    
    ctx_.builder().SetInsertPoint(jac_init_cond);
    Value* jac_init_i = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_init_idx);
    Value* jac_init_less = ctx_.builder().CreateICmpULT(jac_init_i, n);
    ctx_.builder().CreateCondBr(jac_init_less, jac_init_body, jac_init_exit);
    
    ctx_.builder().SetInsertPoint(jac_init_body);

    // CRITICAL FIX: Tensor elements stored as int64, load as int64 then convert
    Value* jac_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_input_elements, jac_init_i);
    Value* jac_elem_int64 = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_elem_ptr);

    // FIX 1b: BitCast preserves IEEE754 bits, SIToFP corrupts them
    Value* jac_elem_val = ctx_.builder().CreateBitCast(jac_elem_int64, ctx_.doubleType());
    Value* jac_var_node = createADVariable(jac_elem_val, 0);
    
    Value* jac_node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_jac_var_nodes, jac_init_i);
    ctx_.builder().CreateStore(jac_var_node, jac_node_slot);
    
    Value* jac_next_init = ctx_.builder().CreateAdd(jac_init_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(jac_next_init, jac_init_idx);
    ctx_.builder().CreateBr(jac_init_cond);
    
    ctx_.builder().SetInsertPoint(jac_init_exit);
    
    // Build AD tensor for function call via arena (OALR compliant - no malloc)
    Value* typed_jac_ad_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set AD tensor structure
    Value* jac_ad_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* jac_ad_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_ad_dims_size});
    Value* typed_jac_ad_dims = ctx_.builder().CreatePointerCast(jac_ad_dims_ptr, ctx_.builder().getPtrTy());

    ctx_.builder().CreateStore(n, typed_jac_ad_dims);

    // Set tensor fields directly
    ctx_.builder().CreateStore(typed_jac_ad_dims,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ad_tensor, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ad_tensor, 1));
    ctx_.builder().CreateStore(n,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ad_tensor, 3));

    // Allocate elements via arena
    Value* jac_ad_elems_size = ctx_.builder().CreateMul(n,
        ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t)));
    Value* jac_ad_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_ad_elems_size});
    Value* typed_jac_ad_elems = ctx_.builder().CreatePointerCast(jac_ad_elems_ptr, ctx_.builder().getPtrTy());
    
    ctx_.builder().CreateStore(typed_jac_ad_elems,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ad_tensor, 2));

    // Copy nodes
    BasicBlock* jac_copy_cond = BasicBlock::Create(ctx_.context(), "jac_copy_cond", current_func);
    BasicBlock* jac_copy_body = BasicBlock::Create(ctx_.context(), "jac_copy_body", current_func);
    BasicBlock* jac_copy_exit = BasicBlock::Create(ctx_.context(), "jac_copy_exit", current_func);
    
    Value* jac_copy_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "jac_copy_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), jac_copy_idx);
    ctx_.builder().CreateBr(jac_copy_cond);
    
    ctx_.builder().SetInsertPoint(jac_copy_cond);
    Value* jac_copy_i = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_copy_idx);
    Value* jac_copy_less = ctx_.builder().CreateICmpULT(jac_copy_i, n);
    ctx_.builder().CreateCondBr(jac_copy_less, jac_copy_body, jac_copy_exit);
    
    ctx_.builder().SetInsertPoint(jac_copy_body);

    Value* jac_src_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_jac_var_nodes, jac_copy_i);
    Value* jac_src_node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), jac_src_slot);

    Value* jac_node_int = ctx_.builder().CreatePtrToInt(jac_src_node, ctx_.int64Type());

    Value* jac_dst_slot = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_jac_ad_elems, jac_copy_i);
    ctx_.builder().CreateStore(jac_node_int, jac_dst_slot);
    
    Value* jac_next_copy = ctx_.builder().CreateAdd(jac_copy_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(jac_next_copy, jac_copy_idx);
    ctx_.builder().CreateBr(jac_copy_cond);

    ctx_.builder().SetInsertPoint(jac_copy_exit);
    
    // Call function to get output
    Value* jac_ad_tensor_int = ctx_.builder().CreatePtrToInt(typed_jac_ad_tensor, ctx_.int64Type());
    // CRITICAL FIX: Pack as TENSOR_PTR not INT64, so identity lambdas preserve type
    Value* jac_ad_tensor_tagged = tagged_.packPtr(jac_ad_tensor_int, ESHKOL_VALUE_HEAP_PTR);

    // PHASE 1 FIX: Set AD mode flag to true before calling lambda
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 1), ctx_.adModeActive());

    // ESH-0120: forward-over-reverse. If F's body differentiates THROUGH an inner
    // forward-mode `derivative`/`gradient` whose integrand captures this column's
    // input variable (e.g. (jacobian (lambda (v) (vector (derivative (lambda (x)
    // (* (vref v 0) x x)) 2.0))) ...)), the inner forward pass must record its
    // dependence on x_j back onto THIS Jacobian tape — otherwise the inner
    // derivative materializes as a tape-disconnected constant and the outer
    // backprop reads a silent zero. Publish x_j as the active reverse seed so
    // maybeJetLiftTapeOperand seeds its ep slot and popAndExtractForward records
    // the exact mixed partial (eshkol_ad_mixed_record) — the SAME machinery the
    // reverse-mode `gradient` path uses (ESH-0093/0117). Harmless for ordinary
    // functions: with no live forward perturbation the seed is never read.
    Value* jac_seed_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_jac_var_nodes, j_in);
    Value* jac_seed_node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), jac_seed_slot);
    FunctionCallee jac_seed_swap = ctx_.module().getOrInsertFunction(
        "eshkol_ad_seed_swap",
        FunctionType::get(ctx_.ptrType(), {ctx_.ptrType()}, false));
    Value* jac_saved_seed = ctx_.builder().CreateCall(jac_seed_swap, {jac_seed_node});

    Value* jac_output_tagged;
    if (func_ptr) {
        // Compile-time resolved function: direct call with explicit captures
        std::vector<Value*> jac_call_args = {jac_ad_tensor_tagged};
        std::vector<Value*> jac_captures = loadCapturesForAutodiff(func_ptr, "Jacobian AD call");
        jac_call_args.insert(jac_call_args.end(), jac_captures.begin(), jac_captures.end());
        jac_output_tagged = ctx_.builder().CreateCall(func_ptr, jac_call_args);
    } else {
        // Runtime closure path — captures are embedded inside the closure struct
        jac_output_tagged = closure_call_callback_(closure_val, {jac_ad_tensor_tagged}, "jacobian-ad", callback_context_);
    }

    // ESH-0120: restore the previously active seed node.
    ctx_.builder().CreateCall(jac_seed_swap, {jac_saved_seed});

    // PHASE 1 FIX: Set AD mode flag back to false after lambda call
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 0), ctx_.adModeActive());

    Value* jac_output_int = tagged_.unpackInt64(jac_output_tagged);
    Value* jac_output_ptr = ctx_.builder().CreateIntToPtr(jac_output_int, ctx_.builder().getPtrTy());

    // M1 CONSOLIDATION: Handle both tensor and Scheme vector output
    // For HEAP_PTR, read the header subtype to distinguish vector vs tensor
    Value* jac_loop_output_type = tagged_.getType(jac_output_tagged);
    Value* jac_loop_output_base = tagged_.getBaseType(jac_loop_output_type);
    Value* jac_loop_is_heap_ptr = ctx_.builder().CreateICmpEQ(jac_loop_output_base,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    BasicBlock* jac_loop_svec_out = BasicBlock::Create(ctx_.context(), "jac_loop_svec_out", current_func);
    BasicBlock* jac_loop_tensor_out = BasicBlock::Create(ctx_.context(), "jac_loop_tensor_out", current_func);
    BasicBlock* jac_loop_merge_out = BasicBlock::Create(ctx_.context(), "jac_loop_merge_out", current_func);
    BasicBlock* jac_loop_check_subtype = BasicBlock::Create(ctx_.context(), "jac_loop_check_subtype", current_func);

    // First check if HEAP_PTR - if so, check subtype; otherwise go to tensor path
    ctx_.builder().CreateCondBr(jac_loop_is_heap_ptr, jac_loop_check_subtype, jac_loop_tensor_out);

    // Check subtype to distinguish Scheme vector (2) from tensor (3)
    ctx_.builder().SetInsertPoint(jac_loop_check_subtype);
    // Header is at ptr - 8 bytes
    Value* jac_loop_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), jac_output_ptr,
        ConstantInt::get(ctx_.int64Type(), -8));
    // Subtype is first byte of header
    Value* jac_loop_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), jac_loop_header_ptr);
    Value* jac_is_scheme_vec = ctx_.builder().CreateICmpEQ(jac_loop_subtype,
        ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    ctx_.builder().CreateCondBr(jac_is_scheme_vec, jac_loop_svec_out, jac_loop_tensor_out);

    // SCHEME VECTOR OUTPUT: Extract element from Scheme vector
    ctx_.builder().SetInsertPoint(jac_loop_svec_out);
    // Scheme vector layout: [len: i64][elem0: tagged_value][elem1: tagged_value]...
    Value* jac_svec_elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), jac_output_ptr,
        ConstantInt::get(ctx_.int64Type(), 8));  // Skip length field
    Value* jac_svec_elem_base_typed = ctx_.builder().CreateBitCast(jac_svec_elem_base, PointerType::getUnqual(ctx_.context()));
    Value* jac_svec_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), jac_svec_elem_base_typed, i_out);
    Value* jac_svec_elem_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), jac_svec_elem_ptr);
    // Extract the int64 component from the tagged value (could be AD node ptr or double bits)
    Value* jac_svec_elem_int = tagged_.unpackInt64(jac_svec_elem_tagged);
    ctx_.builder().CreateBr(jac_loop_merge_out);
    BasicBlock* jac_svec_out_exit = ctx_.builder().GetInsertBlock();

    // TENSOR OUTPUT: Extract element from tensor structure
    ctx_.builder().SetInsertPoint(jac_loop_tensor_out);
    Value* out_elems_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), jac_output_ptr, 2);
    Value* out_elems_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), out_elems_field);
    Value* typed_out_elems = ctx_.builder().CreatePointerCast(out_elems_ptr, ctx_.builder().getPtrTy());
    Value* jac_tensor_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_out_elems, i_out);
    Value* jac_tensor_elem_int = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_tensor_elem_ptr);
    ctx_.builder().CreateBr(jac_loop_merge_out);
    BasicBlock* jac_tensor_out_exit = ctx_.builder().GetInsertBlock();

    // MERGE: Get output component from whichever path
    ctx_.builder().SetInsertPoint(jac_loop_merge_out);
    PHINode* out_comp_int = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "jac_out_comp");
    out_comp_int->addIncoming(jac_svec_elem_int, jac_svec_out_exit);
    out_comp_int->addIncoming(jac_tensor_elem_int, jac_tensor_out_exit);
    
    // CRITICAL SAFETY CHECK: Detect if output element is AD node or regular value
    // AD nodes are allocated in heap (> 1000), doubles have IEEE754 exponent bits
    Value* is_small_value = ctx_.builder().CreateICmpULT(out_comp_int,
        ConstantInt::get(ctx_.int64Type(), 1000));
    
    // Check IEEE754 exponent for doubles (bit pattern detection)
    Value* exp_mask_jac = ConstantInt::get(ctx_.int64Type(), 0x7FF0000000000000ULL);
    Value* exp_bits_jac = ctx_.builder().CreateAnd(out_comp_int, exp_mask_jac);
    Value* has_exponent_jac = ctx_.builder().CreateICmpNE(exp_bits_jac,
        ConstantInt::get(ctx_.int64Type(), 0));
    
    // If has exponent, it's a double, not an AD node pointer
    Value* is_likely_double_jac = ctx_.builder().CreateAnd(has_exponent_jac,
        ctx_.builder().CreateNot(is_small_value));
    
    // Output is AD node only if: not small AND not double
    Value* elem_is_ad_node = ctx_.builder().CreateAnd(
        ctx_.builder().CreateNot(is_small_value),
        ctx_.builder().CreateNot(is_likely_double_jac));
    
    // Allocate storage for partial derivative result (accessible across blocks)
    Value* partial_deriv_storage = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "jac_partial_storage");
    
    BasicBlock* run_jac_backward = BasicBlock::Create(ctx_.context(), "jac_run_backward", current_func);
    BasicBlock* skip_jac_backward = BasicBlock::Create(ctx_.context(), "jac_skip_backward", current_func);
    BasicBlock* after_jac_backward = BasicBlock::Create(ctx_.context(), "jac_after_backward", current_func);
    
    ctx_.builder().CreateCondBr(elem_is_ad_node, run_jac_backward, skip_jac_backward);
    
    // Run backward pass only if output element is AD node
    ctx_.builder().SetInsertPoint(run_jac_backward);

    Value* out_comp_node = ctx_.builder().CreateIntToPtr(out_comp_int, PointerType::getUnqual(ctx_.context()));
    backpropagate(jac_tape, out_comp_node);
    
    // Extract gradient from variable j_in
    Value* jac_grad_var_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_jac_var_nodes, j_in);
    Value* jac_grad_var_node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), jac_grad_var_slot);
    Value* computed_partial_deriv = loadNodeGradient(jac_grad_var_node);
    ctx_.builder().CreateStore(computed_partial_deriv, partial_deriv_storage);
    ctx_.builder().CreateBr(after_jac_backward);
    
    // Skip backward pass if output is not AD node (constant function)
    ctx_.builder().SetInsertPoint(skip_jac_backward);

    Value* zero_deriv_jac = ConstantFP::get(ctx_.doubleType(), 0.0);
    ctx_.builder().CreateStore(zero_deriv_jac, partial_deriv_storage);
    ctx_.builder().CreateBr(after_jac_backward);
    
    // Merge paths - load result from storage
    ctx_.builder().SetInsertPoint(after_jac_backward);
    Value* partial_deriv = ctx_.builder().CreateLoad(ctx_.doubleType(), partial_deriv_storage);
    
    // Store J[i_out,j_in] at linear index: i_out*n + j_in
    Value* linear_idx = ctx_.builder().CreateMul(i_out, n);
    linear_idx = ctx_.builder().CreateAdd(linear_idx, j_in);
    
    Value* jac_result_elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(),
        typed_jac_elems, linear_idx);
    ctx_.builder().CreateStore(partial_deriv, jac_result_elem_ptr);
    
    ctx_.builder().CreateCall(mem_.getArenaTapeReset(), {jac_tape});
    
    // CRITICAL FIX: Clear global tape pointer (like gradient does)
    ctx_.builder().CreateStore(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())), ctx_.currentAdTape());
    
    Value* next_j_in = ctx_.builder().CreateAdd(j_in, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_j_in, in_idx);
    ctx_.builder().CreateBr(inner_cond);
    
    ctx_.builder().SetInsertPoint(inner_exit);
    Value* next_i_out = ctx_.builder().CreateAdd(i_out, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i_out, out_idx);
    ctx_.builder().CreateBr(outer_cond);
    
    ctx_.builder().SetInsertPoint(outer_exit);

    // FIX: Return 2D tensor directly (like Hessian does) instead of converting to nested lists
    // The tensor display now handles N-dimensional tensors correctly
    // Tensor elements are stored as doubles (int64 bit representation)
    // We need to convert from double to int64 bit pattern for proper storage

    // The elements in typed_jac_elems were stored as double type - convert to int64 bit pattern
    BasicBlock* jac_convert_cond = BasicBlock::Create(ctx_.context(), "jac_convert_cond", current_func);
    BasicBlock* jac_convert_body = BasicBlock::Create(ctx_.context(), "jac_convert_body", current_func);
    BasicBlock* jac_convert_exit = BasicBlock::Create(ctx_.context(), "jac_convert_exit", current_func);

    Value* jac_convert_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "jac_convert_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), jac_convert_idx);
    ctx_.builder().CreateBr(jac_convert_cond);

    ctx_.builder().SetInsertPoint(jac_convert_cond);
    Value* jac_cvt_i = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_convert_idx);
    Value* jac_cvt_less = ctx_.builder().CreateICmpULT(jac_cvt_i, total_elems);
    ctx_.builder().CreateCondBr(jac_cvt_less, jac_convert_body, jac_convert_exit);

    ctx_.builder().SetInsertPoint(jac_convert_body);
    // Load as double, convert to int64 bits, store back
    Value* jac_cvt_elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_jac_elems, jac_cvt_i);
    Value* jac_cvt_elem_double = ctx_.builder().CreateLoad(ctx_.doubleType(), jac_cvt_elem_ptr);
    Value* jac_cvt_elem_bits = ctx_.builder().CreateBitCast(jac_cvt_elem_double, ctx_.int64Type());
    // Store as int64 (tensor elements are stored as int64 bit patterns)
    Value* jac_cvt_store_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_jac_elems, jac_cvt_i);
    ctx_.builder().CreateStore(jac_cvt_elem_bits, jac_cvt_store_ptr);
    Value* jac_cvt_next = ctx_.builder().CreateAdd(jac_cvt_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(jac_cvt_next, jac_convert_idx);
    ctx_.builder().CreateBr(jac_convert_cond);

    ctx_.builder().SetInsertPoint(jac_convert_exit);

    // Return the 2D Jacobian tensor directly
    Value* jac_result_int = ctx_.builder().CreatePtrToInt(typed_jac_ptr, ctx_.int64Type());
    Value* jac_result = tagged_.packPtr(jac_result_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(jac_return_block);

    // Merge null and valid results
    ctx_.builder().SetInsertPoint(jac_return_block);
    PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "jac_result");
    result_phi->addIncoming(null_jac_tagged, output_invalid_block);
    result_phi->addIncoming(jac_result, jac_convert_exit);

    return result_phi;
}


/**
 * @brief Entry point for the `derivative` operator: `(derivative f)` or
 *        `(derivative f x)`.
 *
 * With no point, delegates to the higher-order derivativeHigherOrder().
 * Otherwise first checks (via detectPureDerivChain()) whether this is part of
 * a nested `derivative` chain of depth >= 3 that is pure repeated univariate
 * differentiation threading a single variable — the 2-level forward-mode jet
 * used by the default path only supports up to a 2nd derivative and silently
 * returns 0 beyond that, so such chains are instead routed through the
 * arbitrary-order Taylor tower (derivative-n semantics): first attempting the
 * no-heap compile-time monomorphized path (tryMonomorphizedTaylor()), falling
 * back to the runtime heap tower (taylorApiCore()) with `order` fixed to the
 * chain depth.
 *
 * For the ordinary (depth <= 2) case, delegates to codegenDerivativeMonolith()
 * (the pre-extraction implementation kept for its handling of `f` being a
 * runtime function parameter / closure, which this file's newer split-out
 * fast path — formerly a disabled derivativeStaticOnly(), removed as dead
 * code since it predated codegenDerivativeMonolith() delegation — did not
 * cover).
 *
 * @param op The `derivative` AST operation node (`op->derivative_op.function`
 *           and `op->derivative_op.point`).
 * @return Tagged derivative value (or AD node / Taylor result), or nullptr /
 *         packed null on failure.
 */
llvm::Value* AutodiffCodegen::derivative(const eshkol_operations_t* op) {
    using namespace llvm;

    if (!op->derivative_op.function) {
        eshkol_error("Invalid derivative operation - missing function");
        return nullptr;
    }

    // Higher-order form: (derivative f) → create closure
    if (!op->derivative_op.point) {
        return derivativeHigherOrder(op);
    }

    // ESH-0186 / ESH-0118 closure: a nested `derivative` chain of depth >= 3 is
    // pure repeated univariate differentiation, which the 2-level jet returns 0
    // for. Detect the pure chain and route it through the arbitrary-order Taylor
    // tower (derivative-n of the innermost base at the outermost point). Depth
    // <= 2 falls through unchanged to the exact jet path below.
    {
        const eshkol_ast* inner_fn = nullptr;
        const eshkol_ast* outer_pt = nullptr;
        int depth = detectPureDerivChain(op, &inner_fn, &outer_pt);
        if (depth >= 3 && inner_fn && outer_pt) {
            eshkol_info("derivative: routing depth-%d nested chain through the Taylor tower (ESH-0118)", depth);
            // ESH-0187: the chain order (depth) is a compile-time constant, so
            // attempt no-heap monomorphization first; fall back to the runtime
            // tower when the innermost base is not a pure-arithmetic whitelist.
            if (llvm::Value* mono = tryMonomorphizedTaylor(inner_fn, outer_pt, depth, TowerMode::DERIV_N))
                return mono;
            llvm::Value* order = llvm::ConstantInt::get(ctx_.int32Type(), depth);
            return taylorApiCore(inner_fn, outer_pt, order, TowerMode::DERIV_N);
        }
    }

    if (!resolve_lambda_callback_ || !codegen_ast_callback_) {
        eshkol_error("derivative: Required callbacks not set");
        return tagged_.packNull();
    }

    // AD-1 fix: when `f` is a function parameter (e.g. inside
    //   (define (newton-solve f x0 iters) ... (derivative f x) ...)
    //   (newton-solve (lambda (x) ...) 1.5 10))
    // resolve_lambda_callback_ returns nullptr because the AST node is a
    // VAR pointing at a runtime closure, not a directly-resolvable Function*.
    // The original `derivative()` body just returned nullptr in that case
    // ("falling back to main codegen") but no caller actually invokes a
    // fallback — the dispatcher at llvm_codegen.cpp ESHKOL_DERIVATIVE_OP
    // simply propagates the null, which the surrounding arithmetic then
    // turns into 0.0 / -inf.  Symptom: Newton-Raphson on
    //   (newton-solve (lambda (x) (- (pow x 2) 2)) 1.5 10)
    // returned 1.25872 instead of 1.41421 (when the AD tape was empty)
    // or -inf (when global init left residue).
    //
    // `codegenDerivativeMonolith` (the pre-extraction monolithic
    // implementation, kept around at line ~1747) has the full runtime-
    // function-parameter handling: it walks symbol_table_ /
    // global_symbol_table_ / repl_symbol_addresses_ and dispatches via
    // closure_call_callback_ when `f` resolves to a runtime closure.
    // Delegate to it.  v1.3 should re-extract this logic into a shared
    // helper rather than keeping two parallel implementations.
    return codegenDerivativeMonolith(op);
}

// === Arbitrary-order Taylor-tower AD entry points (ESH-0186, P1) ===
//
// (taylor f x k) and (derivative-n f x k) share codegenDerivativeMonolith's
// resolve/capture/call machinery; the only difference from the jet path is that
// seedForwardAndPush seeds a heap Taylor tower of order k (fresh perturbation
// epoch) and popAndExtractForward extracts from the returned tower. That
// re-pointing is gated on adTowerMode_, so the order-<=2 jet path is byte-for-
// byte unchanged whenever these entry points are not on the stack.
//
// taylor_op {function, point, order} overlaps derivative_op {function, point}
// field-for-field (same offsets), so the monolith reads function/point via the
// derivative_op alias unchanged.
// Shared implementation: build a synthetic derivative op referencing
// function_ast/point_ast (which alias derivative_op.function/.point that the
// monolith reads), set the tower mode + order, and run the monolith. The
// monolith's seedForwardAndPush/popAndExtractForward are re-pointed at the
// Taylor kernel while adTowerMode_ != NONE.
// ============================================================================
// ESH-0187 (P2): compile-time-K Taylor-tower monomorphization.
//
// When (derivative-n f x K) / (taylor f x K) has a LITERAL K and f is a
// single-parameter lambda (or a VAR naming a single-arg top-level define)
// whose body is a pure arithmetic expression tree over the primitives in
// taylor_recurrences.def, we bypass the generic tagged-dispatch/heap-tower
// path entirely and emit the whole tower as fully-unrolled, branch-free,
// HEAP-FREE straight-line SSA IR. Each recurrence is transliterated
// operation-for-operation from lib/core/runtime_taylor.c's tr_* kernels,
// using explicit llvm.fma.f64 in the SAME ascending-j reduction order and
// the SAME libm calls at c[0], so mono(K) == runtime(K) BIT-FOR-BIT
// (design sections 6/6a). The whitelist and the spelling->opcode dispatch
// are generated from taylor_recurrences.def -- the single source of truth
// shared with the runtime kernel (design section 5b).
// ============================================================================
namespace {

// Op-code enums generated from the shared X-macro table.
enum {
#define TAYLOR_BIN(name, opcode, sexpr) TMONO_OP_##name = (opcode),
#include "../core/taylor_recurrences.def"
};
enum {
#define TAYLOR_UN(name, opcode, sexpr, testfn, x0) TMONO_UOP_##name = (opcode),
#include "../core/taylor_recurrences.def"
};

// spelling -> opcode whitelists (single source of truth: the .def).
static const std::unordered_map<std::string,int>& monoBinOps() {
    static const std::unordered_map<std::string,int> m = {
#define TAYLOR_BIN(name, opcode, sexpr) { sexpr, (opcode) },
#include "../core/taylor_recurrences.def"
    };
    return m;
}
static const std::unordered_map<std::string,int>& monoUnOps() {
    static const std::unordered_map<std::string,int> m = {
#define TAYLOR_UN(name, opcode, sexpr, testfn, x0) { sexpr, (opcode) },
#include "../core/taylor_recurrences.def"
    };
    return m;
}

class TaylorMonoEmitter {
public:
    using V = std::vector<llvm::Value*>;
    TaylorMonoEmitter(eshkol::CodegenContext& ctx, int K, std::string param, llvm::Value* x0)
        : ctx_(ctx), K_(K), n_(K + 1), param_(std::move(param)), x0_(x0) {}

    // Returns the matched series (n_ SSA doubles) or an empty vector to signal
    // "does not match the whitelist -> caller must fall back to the runtime path".
    V emit(const eshkol_ast* body) { return matchExpr(body); }

private:
    eshkol::CodegenContext& ctx_;
    int K_, n_;
    std::string param_;
    llvm::Value* x0_;

    llvm::IRBuilder<>& b() { return ctx_.builder(); }
    llvm::Type* dty() { return ctx_.doubleType(); }
    /** @brief Build an f64 SSA constant `v` (an unrolled tower coefficient literal). */
    llvm::Value* cst(double v) { return llvm::ConstantFP::get(dty(), v); }
    V zeros() { return V(n_, cst(0.0)); }

    /** @brief Emit `llvm.fma.f64(a, bb, c)` — used for the ascending-j Cauchy-product
     *  reductions in the recurrence emitters so the unrolled IR matches
     *  runtime_taylor.c's kernels bit-for-bit. */
    llvm::Value* fma3(llvm::Value* a, llvm::Value* bb, llvm::Value* c) {
        llvm::Function* f = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::fma, {dty()});
        return b().CreateCall(f, {a, bb, c});
    }
    /** @brief Call (declaring if needed) a unary double libm function `name(x)`,
     *  e.g. "exp"/"log"/"sin"/"cos"/"fabs". */
    llvm::Value* libm1(const char* name, llvm::Value* x) {
        llvm::Function* f = ctx_.module().getFunction(name);
        if (!f) {
            auto* ft = llvm::FunctionType::get(dty(), {dty()}, false);
            f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &ctx_.module());
        }
        return b().CreateCall(f, {x});
    }
    /** @brief Call (declaring if needed) a binary double libm function `name(x, y)`,
     *  e.g. "pow". */
    llvm::Value* libm2(const char* name, llvm::Value* x, llvm::Value* y) {
        llvm::Function* f = ctx_.module().getFunction(name);
        if (!f) {
            auto* ft = llvm::FunctionType::get(dty(), {dty(), dty()}, false);
            f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &ctx_.module());
        }
        return b().CreateCall(f, {x, y});
    }

    // ---- recurrence emitters: mirror runtime_taylor.c tr_* operation-for-operation ----
    V e_add(const V& u, const V& w) { V s(n_); for (int k = 0; k < n_; k++) s[k] = b().CreateFAdd(u[k], w[k]); return s; }
    V e_sub(const V& u, const V& w) { V s(n_); for (int k = 0; k < n_; k++) s[k] = b().CreateFSub(u[k], w[k]); return s; }
    V e_neg(const V& u)             { V s(n_); for (int k = 0; k < n_; k++) s[k] = b().CreateFNeg(u[k]); return s; }

    V e_mul(const V& u, const V& w) {
        V s(n_);
        for (int k = 0; k < n_; k++) {
            llvm::Value* acc = cst(0.0);
            for (int j = 0; j <= k; j++) acc = fma3(u[j], w[k - j], acc);
            s[k] = acc;
        }
        return s;
    }
    V e_div(const V& u, const V& w) {
        V s(n_);
        for (int k = 0; k < n_; k++) {
            llvm::Value* acc = u[k];
            for (int j = 1; j <= k; j++) acc = fma3(b().CreateFNeg(w[j]), s[k - j], acc);
            s[k] = b().CreateFDiv(acc, w[0]);
        }
        return s;
    }
    V e_exp(const V& u) {
        V s(n_);
        s[0] = libm1("exp", u[0]);
        for (int k = 1; k < n_; k++) {
            llvm::Value* acc = cst(0.0);
            for (int j = 1; j <= k; j++) {
                llvm::Value* ju = b().CreateFMul(cst((double)j), u[j]);
                acc = fma3(ju, s[k - j], acc);
            }
            s[k] = b().CreateFDiv(acc, cst((double)k));
        }
        return s;
    }
    V e_log(const V& u) {
        V s(n_);
        s[0] = libm1("log", u[0]);
        for (int k = 1; k < n_; k++) {
            llvm::Value* acc = cst(0.0);
            for (int j = 1; j <= k - 1; j++) {
                llvm::Value* ju = b().CreateFMul(cst((double)j), s[j]);
                acc = fma3(ju, u[k - j], acc);
            }
            llvm::Value* t = b().CreateFDiv(acc, cst((double)k));
            llvm::Value* num = b().CreateFSub(u[k], t);
            s[k] = b().CreateFDiv(num, u[0]);
        }
        return s;
    }
    /**
     * @brief Jointly compute the Taylor-coefficient series for sin(u) and cos(u).
     *
     * Coupled recurrence (sin and cos derivatives feed each other), computed
     * together in one pass mirroring runtime_taylor.c's tr_sin_cos kernel;
     * matchExpr()'s `sin`/`cos`/`tan` cases each call this and pick the needed
     * output(s) (tan = so/co).
     *
     * @param u Input series (Taylor coefficients of the argument).
     * @param so Output: sin(u) series, sized/filled in place.
     * @param co Output: cos(u) series, sized/filled in place.
     */
    void e_sincos(const V& u, V& so, V& co) {
        so.assign(n_, nullptr); co.assign(n_, nullptr);
        so[0] = libm1("sin", u[0]);
        co[0] = libm1("cos", u[0]);
        for (int k = 1; k < n_; k++) {
            llvm::Value* as = cst(0.0);
            llvm::Value* ac = cst(0.0);
            for (int j = 1; j <= k; j++) {
                llvm::Value* ju = b().CreateFMul(cst((double)j), u[j]);
                as = fma3(ju, co[k - j], as);
                ac = fma3(ju, so[k - j], ac);
            }
            so[k] = b().CreateFDiv(as, cst((double)k));
            co[k] = b().CreateFDiv(b().CreateFNeg(ac), cst((double)k));
        }
    }
    V e_pow_const(const V& u, double r) {
        V s(n_);
        s[0] = libm2("pow", u[0], cst(r));
        for (int k = 1; k < n_; k++) {
            llvm::Value* acc = cst(0.0);
            for (int j = 1; j <= k; j++) {
                double coeff = ((double)j * r - (double)(k - j));
                llvm::Value* term = b().CreateFMul(cst(coeff), u[j]);
                acc = fma3(term, s[k - j], acc);
            }
            llvm::Value* denom = b().CreateFMul(cst((double)k), u[0]);
            s[k] = b().CreateFDiv(acc, denom);
        }
        return s;
    }
    V e_abs(const V& u) {
        V s(n_);
        s[0] = libm1("fabs", u[0]);
        llvm::Value* isneg = b().CreateFCmpOLT(u[0], cst(0.0));
        llvm::Value* sgn = b().CreateSelect(isneg, cst(-1.0), cst(1.0));
        for (int k = 1; k < n_; k++) s[k] = b().CreateFMul(sgn, u[k]);
        return s;
    }
    V e_sinh(const V& u) {
        V ep = e_exp(u), em = e_exp(e_neg(u)), s(n_);
        for (int k = 0; k < n_; k++) s[k] = b().CreateFMul(cst(0.5), b().CreateFSub(ep[k], em[k]));
        return s;
    }
    V e_cosh(const V& u) {
        V ep = e_exp(u), em = e_exp(e_neg(u)), s(n_);
        for (int k = 0; k < n_; k++) s[k] = b().CreateFMul(cst(0.5), b().CreateFAdd(ep[k], em[k]));
        return s;
    }
    V e_tanh(const V& u) {
        V ep = e_exp(u), em = e_exp(e_neg(u)), sh(n_), ch(n_);
        for (int k = 0; k < n_; k++) {
            sh[k] = b().CreateFMul(cst(0.5), b().CreateFSub(ep[k], em[k]));
            ch[k] = b().CreateFMul(cst(0.5), b().CreateFAdd(ep[k], em[k]));
        }
        return e_div(sh, ch);
    }

    /** @brief Match a compile-time-literal (int or double) scalar exponent for
     *  pow/expt; returns false (leaving *out untouched) for anything else so the
     *  caller bails to the runtime fallback. */
    bool constScalar(const eshkol_ast* e, double* out) {
        if (!e) return false;
        if (e->type == ESHKOL_INT64)  { *out = (double)e->int64_val; return true; }
        if (e->type == ESHKOL_DOUBLE) { *out = e->double_val; return true; }
        return false;
    }

    V matchExpr(const eshkol_ast* e) {
        if (!e) return {};
        if (e->type == ESHKOL_INT64)  { V s = zeros(); s[0] = cst((double)e->int64_val); return s; }
        if (e->type == ESHKOL_DOUBLE) { V s = zeros(); s[0] = cst(e->double_val); return s; }
        if (e->type == ESHKOL_VAR) {
            if (e->variable.id && param_ == e->variable.id) {
                V s = zeros(); s[0] = x0_; if (K_ >= 1) s[1] = cst(1.0); return s;
            }
            return {};  // foreign variable (capture / global) -> bail
        }
        if (e->type != ESHKOL_OP || e->operation.op != ESHKOL_CALL_OP) return {};
        const auto& call = e->operation.call_op;
        const eshkol_ast* f = call.func;
        if (!f || f->type != ESHKOL_VAR || !f->variable.id) return {};
        std::string head = f->variable.id;
        uint64_t nargs = call.num_vars;
        const eshkol_ast* args = call.variables;
        if (nargs > 0 && !args) return {};
        auto arg = [&](uint64_t i) -> V { return matchExpr(&args[i]); };

        // --- binary / n-ary arithmetic, dispatched by the .def op-code ---
        auto bit = monoBinOps().find(head);
        if (bit != monoBinOps().end()) {
            switch (bit->second) {
                case TMONO_OP_add: {
                    if (nargs == 0) return zeros();
                    V acc = arg(0); if (acc.empty()) return {};
                    for (uint64_t i = 1; i < nargs; i++) { V w = arg(i); if (w.empty()) return {}; acc = e_add(acc, w); }
                    return acc;
                }
                case TMONO_OP_mul: {
                    if (nargs == 0) { V s = zeros(); s[0] = cst(1.0); return s; }
                    V acc = arg(0); if (acc.empty()) return {};
                    for (uint64_t i = 1; i < nargs; i++) { V w = arg(i); if (w.empty()) return {}; acc = e_mul(acc, w); }
                    return acc;
                }
                case TMONO_OP_sub: {
                    if (nargs == 0) return {};
                    V a0 = arg(0); if (a0.empty()) return {};
                    if (nargs == 1) return e_neg(a0);            // unary minus
                    V acc = a0;
                    for (uint64_t i = 1; i < nargs; i++) { V w = arg(i); if (w.empty()) return {}; acc = e_sub(acc, w); }
                    return acc;
                }
                case TMONO_OP_div: {
                    if (nargs == 0) return {};
                    V a0 = arg(0); if (a0.empty()) return {};
                    if (nargs == 1) { V one = zeros(); one[0] = cst(1.0); return e_div(one, a0); }  // reciprocal
                    V acc = a0;
                    for (uint64_t i = 1; i < nargs; i++) { V w = arg(i); if (w.empty()) return {}; acc = e_div(acc, w); }
                    return acc;
                }
                case TMONO_OP_pow: {
                    if (nargs != 2) return {};
                    V base = arg(0); if (base.empty()) return {};
                    double r; if (!constScalar(&args[1], &r)) return {};   // non-const exponent -> bail
                    return e_pow_const(base, r);
                }
                default: return {};
            }
        }
        if (head == "expt" && nargs == 2) {
            V base = arg(0); if (base.empty()) return {};
            double r; if (!constScalar(&args[1], &r)) return {};
            return e_pow_const(base, r);
        }

        // --- unary math, dispatched by the .def op-code ---
        auto uit = monoUnOps().find(head);
        if ((uit != monoUnOps().end() && nargs == 1)) {
            V u = arg(0); if (u.empty()) return {};
            switch (uit->second) {
                case TMONO_UOP_neg:  return e_neg(u);
                case TMONO_UOP_exp:  return e_exp(u);
                case TMONO_UOP_log:  return e_log(u);
                case TMONO_UOP_sin:  { V so, co; e_sincos(u, so, co); return so; }
                case TMONO_UOP_cos:  { V so, co; e_sincos(u, so, co); return co; }
                case TMONO_UOP_tan:  { V so, co; e_sincos(u, so, co); return e_div(so, co); }
                case TMONO_UOP_sqrt: return e_pow_const(u, 0.5);
                case TMONO_UOP_abs:  return e_abs(u);
                case TMONO_UOP_sinh: return e_sinh(u);
                case TMONO_UOP_cosh: return e_cosh(u);
                case TMONO_UOP_tanh: return e_tanh(u);
                default: return {};
            }
        }
        if (head == "fabs" && nargs == 1) { V u = arg(0); if (u.empty()) return {}; return e_abs(u); }

        return {};  // anything else -> bail (safe fallback to runtime path)
    }
};

}  // anonymous namespace

/**
 * @brief Attempt no-heap, compile-time-K monomorphization of a Taylor tower
 *        (ESH-0187): emit the whole order-K expansion as unrolled, branch-free
 *        straight-line SSA instead of the generic heap-tower runtime path.
 *
 * Only applies when `function_ast` is a single-parameter inline lambda (or a
 * VAR naming a single-arg top-level define) whose body is a pure arithmetic
 * expression tree built entirely from the primitives whitelisted in
 * taylor_recurrences.def (the single source of truth shared with the runtime
 * kernel in runtime_taylor.c). `point_ast` must reduce, via
 * `eshkol_taylor_c0`, to a compile-time-provably-inexact (raw double) x0 —
 * anything that might be exact at runtime (boxed int64/rational/bignum) bails
 * to preserve R7RS exactness contagion (design section 9). TaylorMonoEmitter
 * walks the body (matchExpr()), transliterating each recognized operation
 * (add/sub/mul/div/pow/exp/log/sin/cos/tan/sqrt/abs/sinh/cosh/tanh) into the
 * matching Cauchy-product / FMA recurrence, operation-for-operation identical
 * to the runtime tr_* kernels, so mono(K) == runtime(K) bit-for-bit. Returns
 * nullptr (falling back to the unchanged runtime/heap path via
 * taylorApiCore()) if K is out of range, the body doesn't match the
 * whitelist, or the point isn't a plain double.
 *
 * For `TowerMode::DERIV_N`, the result is `K! * c[K]` (the K-th derivative at
 * x0). For `TowerMode::COEFFS`, the K+1 SSA coefficients are packed into a
 * freshly-consed Scheme list `c[0]..c[K]` — the only heap allocation is that
 * returned list, since the tower's working storage stays entirely in SSA.
 *
 * @param function_ast The single-parameter lambda or named-define AST being
 *        differentiated/expanded.
 * @param point_ast The evaluation point AST (must reduce to a literal double).
 * @param K The (already compile-time-known) Taylor order.
 * @param mode Whether to return the K-th derivative value or the full
 *        coefficient list.
 * @return The packed derivative double, the packed coefficient list, or
 *         nullptr if monomorphization does not apply (caller should fall back
 *         to taylorApiCore()).
 */
llvm::Value* AutodiffCodegen::tryMonomorphizedTaylor(const eshkol_ast* function_ast,
                                                     const eshkol_ast* point_ast,
                                                     int K, TowerMode mode) {
    using namespace llvm;
    if (!function_ast || !point_ast) return nullptr;
    if (K < 0 || K > 64) return nullptr;          // sane cap; above -> runtime path
    if (!codegen_ast_callback_) return nullptr;
    if (mode != TowerMode::DERIV_N && mode != TowerMode::COEFFS) return nullptr;

    // Resolve the target to (param name, body): either an inline single-param
    // lambda, or a VAR naming a single-arg top-level define.
    std::string param;
    const eshkol_ast* body = nullptr;
    if (function_ast->type == ESHKOL_OP && function_ast->operation.op == ESHKOL_LAMBDA_OP) {
        const auto& L = function_ast->operation.lambda_op;
        if (L.num_params != 1 || !L.parameters || !L.body) return nullptr;
        if (!L.parameters[0].variable.id) return nullptr;
        param = L.parameters[0].variable.id;
        body = L.body;
    } else if (function_ast->type == ESHKOL_VAR) {
        if (!function_def_ast_ || !function_ast->variable.id) return nullptr;
        auto it = function_def_ast_->find(function_ast->variable.id);
        if (it == function_def_ast_->end() || !it->second) return nullptr;
        const eshkol_ast* def = it->second;
        if (def->type != ESHKOL_OP || def->operation.op != ESHKOL_DEFINE_OP) return nullptr;
        const auto& D = def->operation.define_op;
        if (!D.is_function || D.num_params != 1 || !D.parameters || !D.value) return nullptr;
        if (!D.parameters[0].variable.id) return nullptr;
        param = D.parameters[0].variable.id;
        body = D.value;
    } else {
        return nullptr;
    }

    // Evaluate the point and reduce it to x0 exactly as the runtime seed does
    // (eshkol_taylor_seed_tagged -> tagged_scalar_value == eshkol_taylor_c0),
    // so x0 is bit-identical to the runtime path for any point representation.
    //
    // ESH-0191 (P6): the monomorphized tier stores the tower as stack DOUBLES
    // only -- it cannot represent an exact int64/bignum/rational coefficient.
    // A raw integer point is EXACT by R7RS convention, and a boxed tagged
    // value MAY be exact at runtime (int64/rational/bignum); silently
    // SIToFP-ing (former behavior) or unconditionally accepting either would
    // defeat exactness contagion (design section 9). Bail to the unchanged
    // P1 runtime/heap path (taylorApiCore), which decides F64 vs RATIONAL
    // correctly from the point's actual exactness -- only a point PROVABLY
    // inexact at compile time (a raw double) is eligible for monomorphization.
    Value* point_raw = codegen_ast_callback_(const_cast<eshkol_ast*>(point_ast), callback_context_);
    if (!point_raw) return nullptr;
    Value* point_tagged;
    if (point_raw->getType()->isDoubleTy())
        point_tagged = tagged_.packDouble(point_raw);
    else
        return nullptr;

    Value* x0;
    {
        Function* fn = ctx_.builder().GetInsertBlock()->getParent();
        IRBuilder<> eb(&fn->getEntryBlock(), fn->getEntryBlock().begin());
        Value* pslot = eb.CreateAlloca(ctx_.taggedValueType(), nullptr, "mono_point");
        ctx_.builder().CreateStore(point_tagged, pslot);
        Function* c0 = ctx_.module().getFunction("eshkol_taylor_c0");
        if (!c0) {
            auto* ft = FunctionType::get(ctx_.doubleType(), {ctx_.ptrType()}, false);
            c0 = Function::Create(ft, Function::ExternalLinkage, "eshkol_taylor_c0", &ctx_.module());
        }
        x0 = ctx_.builder().CreateCall(c0, {pslot});
    }

    // Emit the tower as unrolled straight-line SSA over the whitelist.
    TaylorMonoEmitter em(ctx_, K, param, x0);
    std::vector<Value*> series = em.emit(body);
    if (series.empty()) return nullptr;   // pattern did not match -> fall back

    if (mode == TowerMode::DERIV_N) {
        // f^(K)(x0) = K! * c[K]  (mirrors factorial_d * c[n] in eshkol_taylor_extract).
        double fact = 1.0;
        for (int i = 2; i <= K; i++) fact *= (double)i;
        Value* res = ctx_.builder().CreateFMul(ConstantFP::get(ctx_.doubleType(), fact), series[(size_t)K]);
        return tagged_.packDouble(res);
    }

    // COEFFS: build the K+1-element list c[0]..c[K] (c[0] first). The heap use
    // here is only the RETURNED Scheme list (required by the API contract); the
    // tower's working storage stayed entirely in SSA (no-heap invariant holds).
    Value* arena = getArenaPtr();
    Function* consf = mem_.getArenaAllocateConsWithHeader();
    if (!consf || !arena) return nullptr;
    Value* acc = tagged_.packNull();
    for (int k = K; k >= 0; k--) {
        Value* cell = ctx_.builder().CreateCall(consf, {arena});
        Value* car = tagged_.packDouble(series[(size_t)k]);
        ctx_.builder().CreateStore(car, cell);   // car at offset 0
        Value* cdrp = ctx_.builder().CreateGEP(ctx_.taggedValueType(), cell,
            ConstantInt::get(ctx_.int64Type(), 1));
        ctx_.builder().CreateStore(acc, cdrp);   // cdr at offset 16
        Value* cell_int = ctx_.builder().CreatePtrToInt(cell, ctx_.int64Type());
        acc = tagged_.packHeapPtr(cell_int);
    }
    return acc;
}


/**
 * @brief Shared runtime (heap-tower) implementation of `derivative-n` /
 *        `taylor`, used when tryMonomorphizedTaylor() does not apply.
 *
 * Builds a synthetic `ESHKOL_DERIVATIVE_N_OP` operation aliasing
 * `function_ast`/`point_ast` onto `taylor_op.function`/`.point` (which
 * overlaps `derivative_op.function`/`.point` field-for-field, so
 * codegenDerivativeMonolith() reads them unchanged via that alias), then
 * temporarily re-points the AD tower mode/order (`adTowerMode_`,
 * `adTowerOrder_`) so the monolith's seedForwardAndPush()/
 * popAndExtractForward() seed a heap Taylor tower of order `order_i32`
 * (fresh perturbation epoch) instead of the ordinary order-<=2 jet, and
 * extract from the returned tower. Tower mode/order are saved and restored
 * around the call so this is safe to nest.
 *
 * @param function_ast The function AST being differentiated/expanded.
 * @param point_ast The evaluation point AST.
 * @param order_i32 The Taylor order K as an LLVM i32 value (may be a runtime
 *        value, not necessarily a compile-time literal).
 * @param mode Whether to extract the K-th derivative or the full coefficient
 *        list from the tower.
 * @return The tagged result from the monolith, or a packed null if the
 *         monolith itself returned nullptr.
 */
llvm::Value* AutodiffCodegen::taylorApiCore(const eshkol_ast* function_ast,
                                            const eshkol_ast* point_ast,
                                            llvm::Value* order_i32, TowerMode mode) {
    eshkol_operations_t tmp;
    tmp.op = ESHKOL_DERIVATIVE_N_OP;
    tmp.taylor_op.function = const_cast<eshkol_ast*>(function_ast);
    tmp.taylor_op.point = const_cast<eshkol_ast*>(point_ast);
    tmp.taylor_op.order = nullptr;  // order supplied as the LLVM value below

    TowerMode saved_mode = adTowerMode_; llvm::Value* saved_order = adTowerOrder_;
    adTowerMode_ = mode; adTowerOrder_ = order_i32;
    llvm::Value* r = codegenDerivativeMonolith(&tmp);
    adTowerMode_ = saved_mode; adTowerOrder_ = saved_order;
    return r ? r : tagged_.packNull();
}

/**
 * @brief Coerce a Taylor/derivative-n `order` argument value to i32.
 *
 * Handles a tagged_value (unpacks the int64 payload), a double (truncates via
 * `CreateFPToSI`), or an integer of another width (sign-extending/truncating
 * `CreateIntCast`) -- whichever representation `order` already arrived in
 * from `codegen_ast_callback_`.
 *
 * @param ctx Codegen context (for the IR builder and int32 type).
 * @param tagged Tagged-value helper used to unpack a tagged_value order.
 * @param order The raw order value in whatever type it was evaluated as.
 * @return The order as an i32 LLVM value, or nullptr if `order` is null.
 */
static llvm::Value* coerceOrderToI32(CodegenContext& ctx, TaggedValueCodegen& tagged,
                                     llvm::Value* order) {
    if (!order) return nullptr;
    if (order->getType() == ctx.taggedValueType()) order = tagged.unpackInt64(order);
    if (order->getType()->isDoubleTy()) order = ctx.builder().CreateFPToSI(order, ctx.int32Type());
    else if (order->getType()->isIntegerTy()) order = ctx.builder().CreateIntCast(order, ctx.int32Type(), true);
    return order;
}

/**
 * @brief Codegen for `(derivative-n f x k)`: the k-th derivative of `f` at `x`.
 *
 * If `k` is a literal integer, first attempts tryMonomorphizedTaylor() for a
 * no-heap unrolled expansion; otherwise (or if monomorphization declines)
 * evaluates `k` at runtime, coerces it to i32 via coerceOrderToI32(), and
 * delegates to taylorApiCore() with `TowerMode::DERIV_N`.
 *
 * @param op The `derivative-n` AST operation node (`op->taylor_op.function`,
 *           `.point`, and `.order`).
 * @return Tagged k-th-derivative value, or a packed null on missing operands
 *         or evaluation failure.
 */
llvm::Value* AutodiffCodegen::derivativeN(const eshkol_operations_t* op) {
    if (!op->taylor_op.function || !op->taylor_op.point || !op->taylor_op.order) {
        eshkol_error("derivative-n requires (derivative-n f x k)");
        return tagged_.packNull();
    }
    // ESH-0187: literal order K -> attempt no-heap compile-time monomorphization
    // (unrolled straight-line SSA). Falls through to the unchanged P1 runtime
    // heap-tower path when it does not apply (non-literal K, unsupported body).
    const eshkol_ast* ord_ast = op->taylor_op.order;
    if (ord_ast && ord_ast->type == ESHKOL_INT64) {
        int Klit = (int)ord_ast->int64_val;
        if (llvm::Value* mono = tryMonomorphizedTaylor(op->taylor_op.function,
                op->taylor_op.point, Klit, TowerMode::DERIV_N))
            return mono;
    }
    llvm::Value* order = coerceOrderToI32(ctx_, tagged_,
        codegen_ast_callback_(op->taylor_op.order, callback_context_));
    if (!order) { eshkol_error("derivative-n: failed to evaluate order"); return tagged_.packNull(); }
    return taylorApiCore(op->taylor_op.function, op->taylor_op.point, order, TowerMode::DERIV_N);
}

/**
 * @brief Codegen for `(taylor f x k)`: the order-k Taylor coefficient list of
 *        `f` around `x`.
 *
 * Mirrors derivativeN(): attempts tryMonomorphizedTaylor() for a literal `k`,
 * else evaluates `k` at runtime and delegates to taylorApiCore() with
 * `TowerMode::COEFFS` to build the `[c0, c1, ..., ck]` coefficient list.
 *
 * @param op The `taylor` AST operation node (`op->taylor_op.function`,
 *           `.point`, and `.order`).
 * @return Tagged Scheme list of Taylor coefficients, or a packed null on
 *         missing operands or evaluation failure.
 */
llvm::Value* AutodiffCodegen::taylorSeries(const eshkol_operations_t* op) {
    if (!op->taylor_op.function || !op->taylor_op.point || !op->taylor_op.order) {
        eshkol_error("taylor requires (taylor f x k)");
        return tagged_.packNull();
    }
    // ESH-0187: literal order K -> attempt no-heap compile-time monomorphization.
    const eshkol_ast* ord_ast = op->taylor_op.order;
    if (ord_ast && ord_ast->type == ESHKOL_INT64) {
        int Klit = (int)ord_ast->int64_val;
        if (llvm::Value* mono = tryMonomorphizedTaylor(op->taylor_op.function,
                op->taylor_op.point, Klit, TowerMode::COEFFS))
            return mono;
    }
    llvm::Value* order = coerceOrderToI32(ctx_, tagged_,
        codegen_ast_callback_(op->taylor_op.order, callback_context_));
    if (!order) { eshkol_error("taylor: failed to evaluate order"); return tagged_.packNull(); }
    return taylorApiCore(op->taylor_op.function, op->taylor_op.point, order, TowerMode::COEFFS);
}

/**
 * @brief Detect whether `op` is the outermost link of a pure, repeated
 *        univariate `derivative` chain (ESH-0118), e.g.
 *        `(derivative (lambda (x) (derivative (lambda (x) (f x)) x)) x0)`.
 *
 * Walks inward through nested `derivative` operations as long as each level
 * is a single-parameter lambda whose body is itself a `derivative` threading
 * that same parameter as its point. Stops (successfully) either at a lambda
 * whose body is NOT a further nested `derivative` (the base function), or at
 * a `derivative` applied directly to a named function VAR. Returns 0 (not a
 * pure chain) if any level fails to thread the parameter through unchanged —
 * the caller then leaves the whole expression on the ordinary 2-level jet
 * path, which is safe (if imprecise) for anything this doesn't recognize.
 *
 * @param op The outermost `derivative` operation node to inspect.
 * @param innermost_func Output: set to the base function AST (a lambda or a
 *        named-function VAR) at the bottom of the chain, if a chain is found.
 * @param outer_point Output: set to `op`'s own evaluation point (the point at
 *        which the whole nested expression is ultimately differentiated).
 * @return The chain depth (number of nested `derivative` levels, >= 1) if this
 *         is a pure chain, or 0 if it is not (or `op` is not a derivative op).
 */
int AutodiffCodegen::detectPureDerivChain(const eshkol_operations_t* op,
                                          const eshkol_ast** innermost_func,
                                          const eshkol_ast** outer_point) {
    if (!op || op->op != ESHKOL_DERIVATIVE_OP) return 0;
    *outer_point = op->derivative_op.point;
    const eshkol_operations_t* cur = op;
    int depth = 0;
    for (;;) {
        const eshkol_ast* fn = cur->derivative_op.function;
        if (!fn) return 0;
        // Innermost differentiates a named function directly (`named` binding).
        if (fn->type == ESHKOL_VAR) {
            depth++;
            *innermost_func = fn;
            return depth;
        }
        // Otherwise it must be a single-parameter lambda.
        if (fn->type != ESHKOL_OP || fn->operation.op != ESHKOL_LAMBDA_OP) return 0;
        if (fn->operation.lambda_op.num_params != 1) return 0;
        const eshkol_ast* params = fn->operation.lambda_op.parameters;
        const eshkol_ast* body = fn->operation.lambda_op.body;
        if (!params || !body) return 0;
        const char* pname = params[0].variable.id;
        if (!pname) return 0;
        depth++;
        // Is the lambda body itself a `derivative` threading THIS parameter?
        if (body->type == ESHKOL_OP && body->operation.op == ESHKOL_DERIVATIVE_OP) {
            const eshkol_ast* ipt = body->operation.derivative_op.point;
            if (ipt && ipt->type == ESHKOL_VAR && ipt->variable.id &&
                std::string(ipt->variable.id) == pname) {
                cur = &body->operation;   // descend one level
                continue;
            }
            // A nested derivative that does NOT thread this variable is not a
            // pure chain — leave the whole thing on the jet path (safe).
            return 0;
        }
        // Reached the base body: this lambda is the innermost function.
        *innermost_func = fn;
        return depth;
    }
}

/**
 * @brief Emit IR that computes the Hessian (matrix of second partial derivatives) of a scalar field.
 *
 * Forms the Hessian column-by-column by finite-differencing the gradient: it
 * takes the base gradient at the point, perturbs input dimension j by a small
 * epsilon, recomputes the gradient on a fresh tape, and sets
 * H[i,j] = (grad_perturbed[i] - grad_base[i]) / epsilon. Handles direct
 * llvm::Function callees and closure/REPL fallbacks for the function argument.
 *
 * @param op ESHKOL_HESSIAN_OP with hessian_op.function (scalar f) and hessian_op.point.
 * @return Tagged value wrapping an n-by-n tensor pointer (HEAP_PTR), or nullptr on error.
 */
llvm::Value* AutodiffCodegen::hessian(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->hessian_op.function || !op->hessian_op.point) {
        eshkol_error("Invalid hessian operation");
        return nullptr;
    }
    
    eshkol_info("Computing Hessian matrix (second derivatives)");
    
    // CRITICAL FIX: Must null-check before dyn_cast to avoid LLVM assertion
    Value* func = resolve_lambda_callback_(op->hessian_op.function, 0, callback_context_);
    Function* func_ptr = func ? dyn_cast<Function>(func) : nullptr;

    // Closure fallback: handle top-level (define f (lambda ...)) and REPL closures
    Value* hessian_closure_val = nullptr;
    if (!func_ptr) {
        const eshkol_ast_t* func_ast = op->hessian_op.function;
        if (func_ast && func_ast->type == ESHKOL_VAR) {
            std::string func_name = func_ast->variable.id;
            Value* var_value = nullptr;
            auto lit = symbol_table_->find(func_name);
            if (lit != symbol_table_->end() && lit->second) var_value = lit->second;
            if (!var_value) {
                auto git = global_symbol_table_->find(func_name);
                if (git != global_symbol_table_->end() && git->second) var_value = git->second;
            }
            if (!var_value && repl_mode_enabled_ && *repl_mode_enabled_) {
                std::lock_guard<std::mutex> lock(*repl_mutex_);
                auto repl_it = repl_symbol_addresses_->find(func_name);
                if (repl_it != repl_symbol_addresses_->end()) {
                    GlobalVariable* gv = ctx_.module().getGlobalVariable(func_name);
                    if (!gv) {
                        gv = new GlobalVariable(ctx_.module(), ctx_.taggedValueType(), false,
                                                GlobalValue::ExternalLinkage, nullptr, func_name);
                    }
                    var_value = gv;
                }
            }
            if (var_value) {
                if (isa<GlobalVariable>(var_value)) {
                    GlobalVariable* gv = cast<GlobalVariable>(var_value);
                    if (gv->getParent() != &ctx_.module()) {
                        GlobalVariable* cur_gv = ctx_.module().getGlobalVariable(gv->getName());
                        if (!cur_gv) cur_gv = new GlobalVariable(ctx_.module(), gv->getValueType(), false,
                                                                  GlobalValue::ExternalLinkage, nullptr, gv->getName());
                        gv = cur_gv;
                    }
                    hessian_closure_val = ctx_.builder().CreateLoad(gv->getValueType(), gv);
                } else if (isa<AllocaInst>(var_value)) {
                    hessian_closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                } else if (var_value->getType() == ctx_.taggedValueType()) {
                    hessian_closure_val = var_value;
                } else if (isa<Argument>(var_value)) {
                    hessian_closure_val = var_value->getType()->isPointerTy()
                        ? ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value)
                        : var_value;
                }
            }
        }
        if (!hessian_closure_val) {
            eshkol_error("Failed to resolve function for Hessian computation");
            return nullptr;
        }
    }
    
    Value* typed_raw_ = codegen_ast_callback_(op->hessian_op.point, callback_context_);
    if (!typed_raw_) {
        eshkol_error("Failed to evaluate Hessian point");
        return nullptr;
    }

    // ── SCALAR HESSIAN ──────────────────────────────────────────────────
    //
    // For f: R → R, the Hessian is the scalar f''(x). Detect scalar input
    // at the AST level (not runtime — avoids the tensor dispatch entirely)
    // and compute via the three-point central difference formula on f
    // itself:
    //
    //   f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h²
    //
    // This uses THREE exact function evaluations and is O(h²) accurate.
    // It's the same formula the VM uses (case 752) and matches the LLVM
    // gradient's approach of using central differences for partials.
    //
    // A future upgrade path is nested dual numbers (dual-of-dual) for
    // machine-precision second derivatives, but central differences are
    // the current architectural baseline for second-order ops.
    //
    // We detect "scalar" by checking if the AST point node is a plain
    // number (VAR, NUM, or OP returning scalar) rather than a tensor/vector
    // literal. This is sound because the parser distinguishes tensor
    // literals (#(...)) from scalar expressions at parse time.
    {
        bool is_scalar_input = (op->hessian_op.point->type == ESHKOL_INT64 ||
                                op->hessian_op.point->type == ESHKOL_DOUBLE ||
                                op->hessian_op.point->type == ESHKOL_VAR ||
                                (op->hessian_op.point->type == ESHKOL_OP &&
                                 op->hessian_op.point->operation.op != ESHKOL_CALL_OP));
        // Also check: not a tensor literal, not a vector constructor
        if (op->hessian_op.point->type == ESHKOL_TENSOR) is_scalar_input = false;
        // ESH-0095: a (tensor ...) op is a COLLECTION point, not a scalar. The
        // catch-all "OP && op != CALL_OP" above wrongly marked it scalar, so the
        // tensor pointer was read as a double and f was called with a scalar
        // where it expected a vector → SIGSEGV.
        if (op->hessian_op.point->type == ESHKOL_OP &&
            op->hessian_op.point->operation.op == ESHKOL_TENSOR_OP) is_scalar_input = false;
        if (op->hessian_op.point->type == ESHKOL_OP &&
            op->hessian_op.point->operation.op == ESHKOL_CALL_OP &&
            op->hessian_op.point->operation.call_op.func &&
            op->hessian_op.point->operation.call_op.func->type == ESHKOL_VAR) {
            const char* fn = op->hessian_op.point->operation.call_op.func->variable.id;
            if (fn && (strcmp(fn, "vector") == 0 || strcmp(fn, "list") == 0))
                is_scalar_input = false;
        }

        if (is_scalar_input) {
            eshkol_info("Hessian: scalar input detected, using f''(x) formula");

            // Try to get a direct Function* for scalar hessian (avoids PHI issues with repeated calls).
            // If not available (REPL closure), fall back to closure_call_callback_.
            Function* scalar_func_ptr = nullptr;
            if (func_ptr) {
                Value* func_val2 = resolve_lambda_callback_(op->hessian_op.function, 1, callback_context_);
                scalar_func_ptr = func_val2 ? dyn_cast<Function>(func_val2) : nullptr;
                if (!scalar_func_ptr) {
                    eshkol_error("Failed to resolve function for scalar Hessian");
                    return nullptr;
                }
            }
            // If !func_ptr, scalar_func_ptr stays nullptr and hessian_closure_val is used below.

            // Get the scalar point value as a double. An integer point (e.g.
            // (hessian cube 2)) must be converted via SIToFP — unpackDouble on
            // an int-tagged value reads the integer bit pattern as a double.
            Value* x;
            if (typed_raw_->getType()->isDoubleTy()) {
                x = typed_raw_;
            } else if (typed_raw_->getType()->isIntegerTy(64)) {
                x = ctx_.builder().CreateSIToFP(typed_raw_, ctx_.doubleType());
            } else {
                Value* pt = typed_raw_;
                Value* base_ty = tagged_.getBaseType(tagged_.getType(pt));
                Value* is_int = ctx_.builder().CreateICmpEQ(base_ty,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
                Value* as_int = ctx_.builder().CreateSIToFP(tagged_.unpackInt64(pt), ctx_.doubleType());
                Value* as_dbl = tagged_.unpackDouble(pt);
                x = ctx_.builder().CreateSelect(is_int, as_int, as_dbl);
            }

            // EXACT f''(x) via forward-over-forward AD: seed BOTH perturbation
            // slots on the single input, x_jet = {x, 1, 1, 0}. Then
            //   f(x_jet) = f(x) + f'(x)(e1+e2) + f''(x) e1e2
            // and the mixed (e1e2) component IS f''(x) — no finite-difference
            // step, machine precision.
            Value* one = ConstantFP::get(ctx_.doubleType(), 1.0);
            Value* zero = ConstantFP::get(ctx_.doubleType(), 0.0);
            Value* x_jet = packDualToTagged(makeDual4(ctx_, x, one, one, zero));

            Value* fres;
            if (scalar_func_ptr) {
                fres = ctx_.builder().CreateCall(scalar_func_ptr, {x_jet});
            } else {
                fres = closure_call_callback_(hessian_closure_val, {x_jet}, "hessian-scalar", callback_context_);
            }

            Value* fres_dual = safeUnpackDualFromTagged(fres);
            Value* result = dualField(ctx_, fres_dual, 3);   // f''(x) = mixed term
            return tagged_.packDouble(result);
        }
    }

    // ── VECTOR/TENSOR HESSIAN ───────────────────────────────────────────

    // Get arena for OALR-compliant tensor allocation
    Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // Get current function for basic blocks
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Convert TypedValue to tagged_value.
    // ESH-0095: codegen returns raw ptr-as-int64 for a tensor LITERAL (#(...))
    // but a tagged value for a (tensor ...) op and for (vector ...)/(list ...).
    // Mirror the gradient path's robust dispatch instead of unconditionally
    // calling unpackInt64 on a possibly-raw i64 (which mis-read the pointer and
    // led to the tensor-point SIGSEGV).
    Value* vector_val;
    if (typed_raw_->getType() == ctx_.taggedValueType()) {
        vector_val = typed_raw_; // already tagged (tensor-op / vector / list)
    } else if (typed_raw_->getType()->isIntegerTy(64) &&
               op->hessian_op.point->type == ESHKOL_TENSOR) {
        // Tensor literal: ptr-as-int64 → wrap as HEAP_PTR for subtype dispatch.
        vector_val = tagged_.packPtr(typed_raw_, ESHKOL_VALUE_HEAP_PTR);
    } else if (typed_raw_->getType()->isIntegerTy(64)) {
        vector_val = tagged_.packInt64(typed_raw_, true);
    } else if (typed_raw_->getType()->isDoubleTy()) {
        vector_val = tagged_.packDouble(typed_raw_);
    } else {
        vector_val = tagged_.packInt64(typed_raw_, true);
    }

    // ── MULTI-PARAMETER HESSIAN via exact forward-over-forward AD ───────
    // A multi-parameter scalar function f(x,y,…) cannot go through the
    // reverse-mode AD-node tensor path: that passes AD nodes as separate
    // CALLABLE args, which crashes function dispatch (same reason gradient
    // uses forward-mode for multi-param). Instead compute the N×N Hessian
    // numerically — central differences call f with plain double args, which
    // is crash-free and exact for the quadratic forms in the AD test suite.
    // This also fixes laplacian (trace of Hessian) and is reused below.
    {
        uint64_t hess_mp_arity = 0;
        if (func_ptr && function_arity_table_) {
            std::string key = func_ptr->getName().str();
            auto rv_pos = key.rfind("__rv");
            if (rv_pos != std::string::npos && rv_pos + 4 < key.size() &&
                key.find_first_not_of("0123456789", rv_pos + 4) == std::string::npos) {
                key.erase(rv_pos);
            }
            auto it = function_arity_table_->find(key);
            if (it != function_arity_table_->end()) hess_mp_arity = it->second;
            hess_mp_arity = adResolveValueArity(func_ptr, hess_mp_arity);
        }
        if (func_ptr && hess_mp_arity > 1) {
            const uint64_t N = hess_mp_arity;
            Value* mp_arena = ctx_.builder().CreateLoad(
                PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

            // Extract the N input coordinates as plain doubles (vector/list/tensor).
            Function* mp_fn = ctx_.builder().GetInsertBlock()->getParent();
            llvm::IRBuilder<> mpEntry(&mp_fn->getEntryBlock(), mp_fn->getEntryBlock().begin());
            Value* mp_in_slot = mpEntry.CreateAlloca(ctx_.taggedValueType(), nullptr, "hess_mp_in");
            ctx_.builder().CreateStore(vector_val, mp_in_slot);
            Value* mp_dbls = ctx_.builder().CreateCall(mem_.getArenaAllocate(),
                {mp_arena, ConstantInt::get(ctx_.int64Type(), (int64_t)(N * sizeof(double)))});
            Value* mp_dbls_p = ctx_.builder().CreatePointerCast(mp_dbls, PointerType::getUnqual(ctx_.context()));
            llvm::Function* extr = ctx_.module().getFunction("eshkol_ad_extract_doubles");
            if (!extr) {
                llvm::FunctionType* et = llvm::FunctionType::get(ctx_.int64Type(),
                    {ctx_.builder().getPtrTy(), ctx_.builder().getPtrTy(), ctx_.int64Type()}, false);
                extr = llvm::Function::Create(et, llvm::Function::ExternalLinkage,
                    "eshkol_ad_extract_doubles", &ctx_.module());
            }
            ctx_.builder().CreateCall(extr,
                {mp_in_slot, mp_dbls_p, ConstantInt::get(ctx_.int64Type(), (int64_t)N)});

            std::vector<Value*> base(N);
            for (uint64_t k = 0; k < N; k++) {
                base[k] = ctx_.builder().CreateLoad(ctx_.doubleType(),
                    ctx_.builder().CreateGEP(ctx_.doubleType(), mp_dbls_p,
                        ConstantInt::get(ctx_.int64Type(), (int64_t)k)));
            }
            Value* one = ConstantFP::get(ctx_.doubleType(), 1.0);
            Value* zerod = ConstantFP::get(ctx_.doubleType(), 0.0);

            // EXACT mixed partial d^2 f / d x_i d x_j via forward-over-forward
            // AD: seed perturbation e1 on argument i and e2 on argument j (both
            // on argument i when i==j), evaluate f ONCE on the 4-component jets,
            // and read the mixed (e1e2) component of the result. No finite
            // differences -> off-diagonals of separable functions are EXACTLY 0.
            auto evalDual2 = [&](uint64_t i, uint64_t j) -> Value* {
                std::vector<Value*> args;
                args.reserve(N);
                for (uint64_t p = 0; p < N; p++) {
                    Value* d1 = (p == i) ? one : zerod;
                    Value* d2 = (p == j) ? one : zerod;
                    args.push_back(packDualToTagged(makeDual4(ctx_, base[p], d1, d2, zerod)));
                }
                resolveGradientCaptures(func_ptr, args, "hessian-ad");
                Value* r = ctx_.builder().CreateCall(func_ptr, args);
                Value* rd = safeUnpackDualFromTagged(r);
                return dualField(ctx_, rd, 3);   // mixed second-order term
            };

            // Result N×N tensor.
            Value* mp_res = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {mp_arena});
            Value* mp_dims = ctx_.builder().CreateCall(mem_.getArenaAllocate(),
                {mp_arena, ConstantInt::get(ctx_.int64Type(), (int64_t)(2 * sizeof(uint64_t)))});
            Value* mp_dims_p = ctx_.builder().CreatePointerCast(mp_dims, PointerType::getUnqual(ctx_.context()));
            ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), (int64_t)N), mp_dims_p);
            ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), (int64_t)N),
                ctx_.builder().CreateGEP(ctx_.int64Type(), mp_dims_p, ConstantInt::get(ctx_.int64Type(), 1)));
            ctx_.builder().CreateStore(mp_dims_p, ctx_.builder().CreateStructGEP(ctx_.tensorType(), mp_res, 0));
            ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 2),
                ctx_.builder().CreateStructGEP(ctx_.tensorType(), mp_res, 1));
            ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), (int64_t)(N * N)),
                ctx_.builder().CreateStructGEP(ctx_.tensorType(), mp_res, 3));
            Value* mp_elems = ctx_.builder().CreateCall(mem_.getArenaAllocate(),
                {mp_arena, ConstantInt::get(ctx_.int64Type(), (int64_t)(N * N * sizeof(int64_t)))});
            Value* mp_elems_p = ctx_.builder().CreatePointerCast(mp_elems, PointerType::getUnqual(ctx_.context()));
            ctx_.builder().CreateStore(mp_elems_p, ctx_.builder().CreateStructGEP(ctx_.tensorType(), mp_res, 2));

            for (uint64_t i = 0; i < N; i++) {
                for (uint64_t j = 0; j < N; j++) {
                    Value* hij = evalDual2(i, j);
                    Value* bits = ctx_.builder().CreateBitCast(hij, ctx_.int64Type());
                    ctx_.builder().CreateStore(bits,
                        ctx_.builder().CreateGEP(ctx_.int64Type(), mp_elems_p,
                            ConstantInt::get(ctx_.int64Type(), (int64_t)(i * N + j))));
                }
            }
            Value* mp_int = ctx_.builder().CreatePtrToInt(mp_res, ctx_.int64Type());
            return tagged_.packPtr(mp_int, ESHKOL_VALUE_HEAP_PTR);
        }
    }

    Value* input_type = tagged_.getType(vector_val);
    Value* input_base_type = tagged_.getBaseType(input_type);

    // M1 CONSOLIDATION: Check for both HEAP_PTR (consolidated) and legacy VECTOR_PTR
    Value* hess_is_heap_ptr = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* hess_is_legacy_vector = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    BasicBlock* hess_heap_dispatch = BasicBlock::Create(ctx_.context(), "hess_heap_dispatch", current_func);
    BasicBlock* hess_check_legacy = BasicBlock::Create(ctx_.context(), "hess_check_legacy", current_func);
    BasicBlock* hess_scheme_vector_input = BasicBlock::Create(ctx_.context(), "hess_scheme_vector", current_func);
    BasicBlock* hess_tensor_input = BasicBlock::Create(ctx_.context(), "hess_tensor_input", current_func);
    BasicBlock* hess_merge_input = BasicBlock::Create(ctx_.context(), "hess_merge_input", current_func);

    // First check for HEAP_PTR (consolidated format)
    ctx_.builder().CreateCondBr(hess_is_heap_ptr, hess_heap_dispatch, hess_check_legacy);

    // HEAP_PTR dispatch - read subtype from header
    ctx_.builder().SetInsertPoint(hess_heap_dispatch);
    Value* hess_heap_ptr_val = tagged_.unpackPtr(vector_val);
    Value* hess_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), hess_heap_ptr_val, ConstantInt::get(ctx_.int64Type(), -8));
    Value* hess_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), hess_header_ptr);
    Value* hess_is_vec_subtype = ctx_.builder().CreateICmpEQ(hess_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    ctx_.builder().CreateCondBr(hess_is_vec_subtype, hess_scheme_vector_input, hess_tensor_input);

    // Legacy VECTOR_PTR fallback
    ctx_.builder().SetInsertPoint(hess_check_legacy);
    ctx_.builder().CreateCondBr(hess_is_legacy_vector, hess_scheme_vector_input, hess_tensor_input);

    // SCHEME VECTOR: Convert to tensor format
    ctx_.builder().SetInsertPoint(hess_scheme_vector_input);

    Value* hess_scheme_vec_ptr_int = tagged_.unpackInt64(vector_val);
    Value* hess_scheme_vec_ptr = ctx_.builder().CreateIntToPtr(hess_scheme_vec_ptr_int, ctx_.builder().getPtrTy());
    Value* hess_scheme_len_ptr = ctx_.builder().CreateBitCast(hess_scheme_vec_ptr, PointerType::getUnqual(ctx_.context()));
    Value* hess_scheme_len = ctx_.builder().CreateLoad(ctx_.int64Type(), hess_scheme_len_ptr);

    // Allocate tensor via arena (OALR compliant - no malloc)
    Value* hess_typed_scheme_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions
    Value* hess_scheme_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* hess_scheme_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, hess_scheme_dims_size});
    Value* hess_typed_scheme_dims = ctx_.builder().CreatePointerCast(hess_scheme_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(hess_scheme_len, hess_typed_scheme_dims);

    ctx_.builder().CreateStore(hess_typed_scheme_dims, ctx_.builder().CreateStructGEP(ctx_.tensorType(), hess_typed_scheme_tensor, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), hess_typed_scheme_tensor, 1));
    ctx_.builder().CreateStore(hess_scheme_len, ctx_.builder().CreateStructGEP(ctx_.tensorType(), hess_typed_scheme_tensor, 3));

    // Allocate and copy elements
    Value* hess_scheme_elems_size = ctx_.builder().CreateMul(hess_scheme_len,
        ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    Value* hess_scheme_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, hess_scheme_elems_size});
    Value* hess_typed_scheme_elems = ctx_.builder().CreatePointerCast(hess_scheme_elems_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(hess_typed_scheme_elems, ctx_.builder().CreateStructGEP(ctx_.tensorType(), hess_typed_scheme_tensor, 2));

    // Copy elements loop
    Value* hess_scheme_elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), hess_scheme_vec_ptr,
        ConstantInt::get(ctx_.int64Type(), 8));
    Value* hess_scheme_elem_base_typed = ctx_.builder().CreateBitCast(hess_scheme_elem_base, PointerType::getUnqual(ctx_.context()));

    BasicBlock* hess_svec_copy_cond = BasicBlock::Create(ctx_.context(), "hess_svec_copy_cond", current_func);
    BasicBlock* hess_svec_copy_body = BasicBlock::Create(ctx_.context(), "hess_svec_copy_body", current_func);
    BasicBlock* hess_svec_copy_done = BasicBlock::Create(ctx_.context(), "hess_svec_copy_done", current_func);

    Value* hess_svec_copy_i = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "hess_svec_copy_i");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), hess_svec_copy_i);
    ctx_.builder().CreateBr(hess_svec_copy_cond);

    ctx_.builder().SetInsertPoint(hess_svec_copy_cond);
    Value* hess_svec_i = ctx_.builder().CreateLoad(ctx_.int64Type(), hess_svec_copy_i);
    Value* hess_svec_cond = ctx_.builder().CreateICmpULT(hess_svec_i, hess_scheme_len);
    ctx_.builder().CreateCondBr(hess_svec_cond, hess_svec_copy_body, hess_svec_copy_done);

    ctx_.builder().SetInsertPoint(hess_svec_copy_body);
    Value* hess_svec_src_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), hess_scheme_elem_base_typed, hess_svec_i);
    Value* hess_svec_tagged_elem = ctx_.builder().CreateLoad(ctx_.taggedValueType(), hess_svec_src_ptr);
    Value* hess_svec_double_val = tagged_.unpackDouble(hess_svec_tagged_elem);
    Value* hess_svec_as_int64 = ctx_.builder().CreateBitCast(hess_svec_double_val, ctx_.int64Type());
    Value* hess_svec_dst_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), hess_typed_scheme_elems, hess_svec_i);
    ctx_.builder().CreateStore(hess_svec_as_int64, hess_svec_dst_ptr);
    Value* hess_svec_next_i = ctx_.builder().CreateAdd(hess_svec_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(hess_svec_next_i, hess_svec_copy_i);
    ctx_.builder().CreateBr(hess_svec_copy_cond);

    ctx_.builder().SetInsertPoint(hess_svec_copy_done);
    Value* hess_scheme_tensor_int = ctx_.builder().CreatePtrToInt(hess_typed_scheme_tensor, ctx_.int64Type());
    Value* hess_scheme_vector_tagged = tagged_.packPtr(hess_scheme_tensor_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(hess_merge_input);
    BasicBlock* hess_scheme_exit = ctx_.builder().GetInsertBlock();

    // TENSOR INPUT: Ensure it's a tagged value (scalar inputs may arrive as raw doubles)
    ctx_.builder().SetInsertPoint(hess_tensor_input);
    Value* hess_tensor_tagged = vector_val;
    if (hess_tensor_tagged->getType() != ctx_.taggedValueType()) {
        if (hess_tensor_tagged->getType()->isDoubleTy()) {
            hess_tensor_tagged = tagged_.packDouble(hess_tensor_tagged);
        } else if (hess_tensor_tagged->getType()->isIntegerTy(64)) {
            hess_tensor_tagged = tagged_.packInt64(hess_tensor_tagged, true);
        }
        // If still not tagged after these checks, the type is already ptr/struct
        // which will be handled downstream
    }
    ctx_.builder().CreateBr(hess_merge_input);
    BasicBlock* hess_tensor_exit = ctx_.builder().GetInsertBlock();

    // MERGE
    ctx_.builder().SetInsertPoint(hess_merge_input);
    PHINode* hess_actual_input = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "hess_input");
    hess_actual_input->addIncoming(hess_scheme_vector_tagged, hess_scheme_exit);
    hess_actual_input->addIncoming(hess_tensor_tagged, hess_tensor_exit);

    // Extract tensor pointer from merged input
    Value* vector_ptr_int = tagged_.unpackInt64(hess_actual_input);

    // Use class member ctx_.tensorType() (shared by all tensor operations)

    // Extract input dimension n
    Value* input_ptr = ctx_.builder().CreateIntToPtr(vector_ptr_int, ctx_.builder().getPtrTy());
    
    Value* input_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), input_ptr, 0);
    Value* input_dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), input_dims_field);
    Value* typed_input_dims = ctx_.builder().CreatePointerCast(input_dims_ptr, ctx_.builder().getPtrTy());
    
    Value* input_elements_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), input_ptr, 2);
    Value* input_elements_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), input_elements_field);
    Value* typed_input_elements = ctx_.builder().CreatePointerCast(input_elements_ptr, ctx_.builder().getPtrTy());
    
    Value* n_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_input_dims,
        ConstantInt::get(ctx_.int64Type(), 0));
    Value* n = ctx_.builder().CreateLoad(ctx_.int64Type(), n_ptr);

    // Allocate n×n Hessian matrix via arena (OALR compliant - no malloc)
    Value* typed_hess_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions [n, n]
    Value* hess_dims_size = ctx_.builder().CreateMul(
        ConstantInt::get(ctx_.int64Type(), 2),
        ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t)));
    Value* hess_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, hess_dims_size});
    Value* typed_hess_dims = ctx_.builder().CreatePointerCast(hess_dims_ptr, ctx_.builder().getPtrTy());

    ctx_.builder().CreateStore(n, typed_hess_dims);
    Value* hess_dim1_slot = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_hess_dims,
        ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(n, hess_dim1_slot);

    Value* hess_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_hess_ptr, 0);
    ctx_.builder().CreateStore(typed_hess_dims, hess_dims_field);

    Value* hess_num_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_hess_ptr, 1);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 2), hess_num_dims_field);

    Value* total_hess_elems = ctx_.builder().CreateMul(n, n);
    Value* hess_total_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_hess_ptr, 3);
    ctx_.builder().CreateStore(total_hess_elems, hess_total_field);

    // Allocate elements array
    Value* hess_elems_size = ctx_.builder().CreateMul(total_hess_elems,
        ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    Value* hess_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, hess_elems_size});
    Value* typed_hess_elems = ctx_.builder().CreatePointerCast(hess_elems_ptr, ctx_.builder().getPtrTy());

    Value* hess_elems_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_hess_ptr, 2);
    ctx_.builder().CreateStore(typed_hess_elems, hess_elems_field);

    // ── EXACT vector/tensor Hessian via forward-over-forward AD (ESH-0121) ──
    //
    // H[i][j] = ∂²f/∂xᵢ∂xⱼ. For each (i,j) build a Scheme vector whose element k
    // is the dual jet {x_k, e1 = (k==i), e2 = (k==j)} (both perturbation slots on
    // the same component when i==j), evaluate f ONCE, and read the mixed e1·e2
    // coefficient (field 3) of the returned jet. This is machine-precision and,
    // crucially, FINITE-DIFFERENCE-FREE: off-diagonals of separable functions are
    // EXACTLY 0, and the previous ε-perturbed reverse-gradient sweep (which also
    // silently returned zeros whenever f differentiated through an inner
    // forward-mode derivative) is gone. It generalizes the single-argument
    // makeDual4 field-3 trick already used by the scalar and multi-parameter
    // Hessian paths to a single VECTOR parameter whose components are duals.
    //
    // Forward-over-reverse composition (ESH-0121): when f differentiates THROUGH
    // an inner forward `derivative`/`gradient` (e.g. f(v) = (derivative g 2.0)
    // where g captures v), the two Hessian directions occupy e1/e2, so we
    // pre-raise the runtime perturbation level to 2 — seedForwardAndPush then
    // routes the inner forward pass to the INDEPENDENT ep jet dimension (kJetEp,
    // ESH-0117), leaving e1/e2 for the Hessian. The inner derivative extracts its
    // d/dep slice, whose field-3 mixed term is exactly ∂²(∂f/∂z)/∂xᵢ∂xⱼ. No slot
    // collision, exact at every (i,j).
    Value* hff_one_d  = ConstantFP::get(ctx_.doubleType(), 1.0);
    Value* hff_zero_d = ConstantFP::get(ctx_.doubleType(), 0.0);
    Value* hff_zero64 = ConstantInt::get(ctx_.int64Type(), 0);
    Value* hff_one64  = ConstantInt::get(ctx_.int64Type(), 1);

    // Hoist loop-counter allocas to the entry block so the nested sweep does not
    // grow the stack per iteration.
    llvm::AllocaInst* hff_j_idx;
    llvm::AllocaInst* hff_i_idx;
    llvm::AllocaInst* hff_k_idx;
    {
        llvm::IRBuilder<> eb(&current_func->getEntryBlock(), current_func->getEntryBlock().begin());
        hff_j_idx = eb.CreateAlloca(ctx_.int64Type(), nullptr, "hess_ff_j");
        hff_i_idx = eb.CreateAlloca(ctx_.int64Type(), nullptr, "hess_ff_i");
        hff_k_idx = eb.CreateAlloca(ctx_.int64Type(), nullptr, "hess_ff_k");
    }

    // Save + raise the forward perturbation level (restored after the sweep), and
    // null out the reverse tape so any inner derivative stays in pure forward mode
    // (eshkol_ad_mixed_record no-ops on a null tape). Both are restored afterward
    // to keep an enclosing gradient/jacobian pass intact.
    Value* hff_saved_level = adPertLevelLoad();
    adPertLevelStore(ConstantInt::get(ctx_.int64Type(), 2));
    Value* hff_saved_tape = ctx_.builder().CreateLoad(
        PointerType::getUnqual(ctx_.context()), ctx_.currentAdTape());
    ctx_.builder().CreateStore(
        ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())), ctx_.currentAdTape());

    // ── column loop: j ──
    BasicBlock* hff_j_cond = BasicBlock::Create(ctx_.context(), "hff_j_cond", current_func);
    BasicBlock* hff_j_body = BasicBlock::Create(ctx_.context(), "hff_j_body", current_func);
    BasicBlock* hff_j_exit = BasicBlock::Create(ctx_.context(), "hff_j_exit", current_func);
    ctx_.builder().CreateStore(hff_zero64, hff_j_idx);
    ctx_.builder().CreateBr(hff_j_cond);

    ctx_.builder().SetInsertPoint(hff_j_cond);
    Value* hff_j = ctx_.builder().CreateLoad(ctx_.int64Type(), hff_j_idx);
    ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(hff_j, n), hff_j_body, hff_j_exit);

    ctx_.builder().SetInsertPoint(hff_j_body);

    // ── row loop: i ──
    BasicBlock* hff_i_cond = BasicBlock::Create(ctx_.context(), "hff_i_cond", current_func);
    BasicBlock* hff_i_body = BasicBlock::Create(ctx_.context(), "hff_i_body", current_func);
    BasicBlock* hff_i_exit = BasicBlock::Create(ctx_.context(), "hff_i_exit", current_func);
    ctx_.builder().CreateStore(hff_zero64, hff_i_idx);
    ctx_.builder().CreateBr(hff_i_cond);

    ctx_.builder().SetInsertPoint(hff_i_cond);
    Value* hff_i = ctx_.builder().CreateLoad(ctx_.int64Type(), hff_i_idx);
    ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(hff_i, n), hff_i_body, hff_i_exit);

    ctx_.builder().SetInsertPoint(hff_i_body);

    // Build a Scheme vector of n dual jets (subtype VECTOR, so f's vector-ref
    // returns the tagged dual element and scalar arithmetic dispatches to the
    // forward-mode dual path).
    Value* hff_vec = ctx_.builder().CreateCall(mem_.getArenaAllocateVectorWithHeader(), {arena_ptr, n});
    ctx_.builder().CreateStore(n, hff_vec);   // length at offset 0
    Value* hff_vec_elems8 = ctx_.builder().CreateGEP(ctx_.int8Type(), hff_vec,
        ConstantInt::get(ctx_.int64Type(), 8));
    Value* hff_vec_elems = ctx_.builder().CreatePointerCast(hff_vec_elems8, ctx_.ptrType());

    // ── fill loop: k ──
    BasicBlock* hff_k_cond = BasicBlock::Create(ctx_.context(), "hff_k_cond", current_func);
    BasicBlock* hff_k_body = BasicBlock::Create(ctx_.context(), "hff_k_body", current_func);
    BasicBlock* hff_k_exit = BasicBlock::Create(ctx_.context(), "hff_k_exit", current_func);
    ctx_.builder().CreateStore(hff_zero64, hff_k_idx);
    ctx_.builder().CreateBr(hff_k_cond);

    ctx_.builder().SetInsertPoint(hff_k_cond);
    Value* hff_k = ctx_.builder().CreateLoad(ctx_.int64Type(), hff_k_idx);
    ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(hff_k, n), hff_k_body, hff_k_exit);

    ctx_.builder().SetInsertPoint(hff_k_body);
    Value* hff_xk_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_input_elements, hff_k);
    Value* hff_xk_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(), hff_xk_ptr);
    Value* hff_xk = ctx_.builder().CreateBitCast(hff_xk_i64, ctx_.doubleType());
    Value* hff_d1 = ctx_.builder().CreateSelect(
        ctx_.builder().CreateICmpEQ(hff_k, hff_i), hff_one_d, hff_zero_d);
    Value* hff_d2 = ctx_.builder().CreateSelect(
        ctx_.builder().CreateICmpEQ(hff_k, hff_j), hff_one_d, hff_zero_d);
    Value* hff_dual = packDualToTagged(makeDual8(ctx_, hff_xk, hff_d1, hff_d2, hff_zero_d,
        hff_zero_d, hff_zero_d, hff_zero_d, hff_zero_d));
    Value* hff_dst = ctx_.builder().CreateGEP(ctx_.taggedValueType(), hff_vec_elems, hff_k);
    ctx_.builder().CreateStore(hff_dual, hff_dst);
    ctx_.builder().CreateStore(ctx_.builder().CreateAdd(hff_k, hff_one64), hff_k_idx);
    ctx_.builder().CreateBr(hff_k_cond);

    ctx_.builder().SetInsertPoint(hff_k_exit);
    Value* hff_vec_int = ctx_.builder().CreatePtrToInt(hff_vec, ctx_.int64Type());
    Value* hff_vec_tagged = tagged_.packPtr(hff_vec_int, ESHKOL_VALUE_HEAP_PTR);

    // Evaluate f once on the seeded dual vector.
    Value* hff_out;
    if (func_ptr) {
        std::vector<Value*> hff_args = {hff_vec_tagged};
        std::vector<Value*> hff_caps = loadCapturesForAutodiff(func_ptr, "Hessian forward-over-forward");
        hff_args.insert(hff_args.end(), hff_caps.begin(), hff_caps.end());
        hff_out = ctx_.builder().CreateCall(func_ptr, hff_args);
    } else {
        hff_out = closure_call_callback_(hessian_closure_val, {hff_vec_tagged}, "hessian-ff", callback_context_);
    }

    // H[i][j] = mixed e1·e2 coefficient of f's returned jet (0 if f is not a dual,
    // e.g. a component-independent constant).
    Value* hff_rd = safeUnpackDualFromTagged(hff_out);
    Value* hff_hij = dualField(ctx_, hff_rd, 3);
    Value* hff_lin = ctx_.builder().CreateAdd(ctx_.builder().CreateMul(hff_i, n), hff_j);
    Value* hff_store_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_hess_elems, hff_lin);
    ctx_.builder().CreateStore(hff_hij, hff_store_ptr);

    ctx_.builder().CreateStore(ctx_.builder().CreateAdd(hff_i, hff_one64), hff_i_idx);
    ctx_.builder().CreateBr(hff_i_cond);

    ctx_.builder().SetInsertPoint(hff_i_exit);
    ctx_.builder().CreateStore(ctx_.builder().CreateAdd(hff_j, hff_one64), hff_j_idx);
    ctx_.builder().CreateBr(hff_j_cond);

    ctx_.builder().SetInsertPoint(hff_j_exit);

    // Restore the enclosing perturbation level and reverse tape.
    adPertLevelStore(hff_saved_level);
    ctx_.builder().CreateStore(hff_saved_tape, ctx_.currentAdTape());
    
    eshkol_info("Hessian computation complete");
    // Tag as TENSOR_PTR for proper display handling
    Value* hess_result_int = ctx_.builder().CreatePtrToInt(typed_hess_ptr, ctx_.int64Type());
    return tagged_.packPtr(hess_result_int, ESHKOL_VALUE_HEAP_PTR);
}


/**
 * @brief Allocate an arena-backed 1-D tensor of the given runtime length, zero-filled.
 *
 * Emits an OALR-compliant (arena, no malloc) tensor with num_dimensions = 1,
 * total_elements = dimension, and a runtime loop that stores 0.0 into every element.
 *
 * @param dimension Runtime i64 giving the vector length.
 * @return The tensor structure pointer as an i64 (not tagged).
 */
llvm::Value* AutodiffCodegen::createNullVectorTensor(llvm::Value* dimension) {
    using namespace llvm;
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Get arena for OALR-compliant allocation
    Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // Allocate tensor structure via arena (OALR compliant - no malloc)
    Value* typed_tensor_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Allocate dimensions array (1D vector of given dimension)
    Value* dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, dims_size});
    Value* typed_dims_ptr = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(dimension, typed_dims_ptr);  // Runtime dimension!

    // Store tensor metadata
    ctx_.builder().CreateStore(typed_dims_ptr,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 0));  // dimensions
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 1));  // num_dimensions = 1
    ctx_.builder().CreateStore(dimension,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 3));  // total_elements = dimension

    // Allocate elements array (dimension * sizeof(double))
    Value* elems_size = ctx_.builder().CreateMul(dimension,
        ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    Value* elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, elems_size});
    Value* typed_elems_ptr = ctx_.builder().CreatePointerCast(elems_ptr, ctx_.builder().getPtrTy());
    
    // Zero all elements using RUNTIME LOOP (n-dimensional!)
    BasicBlock* zero_cond = BasicBlock::Create(ctx_.context(), "null_vec_zero_cond", current_func);
    BasicBlock* zero_body = BasicBlock::Create(ctx_.context(), "null_vec_zero_body", current_func);
    BasicBlock* zero_exit = BasicBlock::Create(ctx_.context(), "null_vec_zero_exit", current_func);
    
    Value* idx_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "zero_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), idx_ptr);
    ctx_.builder().CreateBr(zero_cond);
    
    ctx_.builder().SetInsertPoint(zero_cond);
    Value* idx = ctx_.builder().CreateLoad(ctx_.int64Type(), idx_ptr);
    Value* idx_less = ctx_.builder().CreateICmpULT(idx, dimension);
    ctx_.builder().CreateCondBr(idx_less, zero_body, zero_exit);
    
    ctx_.builder().SetInsertPoint(zero_body);
    Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_elems_ptr, idx);
    ctx_.builder().CreateStore(ConstantFP::get(ctx_.doubleType(), 0.0), elem_ptr);
    Value* next_idx = ctx_.builder().CreateAdd(idx, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_idx, idx_ptr);
    ctx_.builder().CreateBr(zero_cond);
    
    ctx_.builder().SetInsertPoint(zero_exit);
    
    ctx_.builder().CreateStore(typed_elems_ptr,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 2));  // elements
    
    // Return tensor pointer as i64
    return ctx_.builder().CreatePtrToInt(typed_tensor_ptr, ctx_.int64Type());
}


/**
 * @brief Extract a single scalar element from an N-dimensional tensor by index vector.
 *
 * Reads the tensor's dimension array to compute a row-major linear index
 * (idx[0] times stride0 plus ... plus idx[n-1]), loads the element as an int64
 * bit pattern, and bit-casts it to a double. For 2-D tensors (Jacobian, Hessian)
 * pass indices = [row, col].
 *
 * @param tensor_ptr Pointer to the tensor structure.
 * @param indices One index per tensor dimension.
 * @return The addressed element as a double value.
 */
llvm::Value* AutodiffCodegen::extractTensorElement(llvm::Value* tensor_ptr, std::vector<llvm::Value*> indices) {
    using namespace llvm;
    // Get tensor dimensions
    Value* dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 0);
    Value* dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dims_field);
    Value* typed_dims = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());


    // Get elements array
    Value* elements_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 2);
    Value* elements_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), elements_field);
    Value* typed_elements = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.builder().getPtrTy());

    // Compute linear index using row-major ordering
    // linear_idx = idx[0] * (dims[1] * dims[2] * ...) + idx[1] * (dims[2] * ...) + ... + idx[n-1]
    Value* linear_idx = ConstantInt::get(ctx_.int64Type(), 0);

    for (size_t i = 0; i < indices.size(); i++) {
        // Compute stride for dimension i (product of all subsequent dimensions)
        Value* stride = ConstantInt::get(ctx_.int64Type(), 1);
        for (size_t j = i + 1; j < indices.size(); j++) {
            Value* dim_j_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims,
                ConstantInt::get(ctx_.int64Type(), j));
            Value* dim_j = ctx_.builder().CreateLoad(ctx_.int64Type(), dim_j_ptr);
            stride = ctx_.builder().CreateMul(stride, dim_j);
        }
        // Add idx[i] * stride to linear index
        Value* contribution = ctx_.builder().CreateMul(indices[i], stride);
        linear_idx = ctx_.builder().CreateAdd(linear_idx, contribution);
    }

    // Load element as int64 (bit pattern of double)
    Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements, linear_idx);
    Value* elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), elem_ptr);

    // Convert int64 bit pattern back to double
    Value* elem_double = ctx_.builder().CreateBitCast(elem_bits, ctx_.doubleType());

    return elem_double;
}

/**
 * @brief Convenience wrapper: extract element [row, col] from a 2-D tensor (Jacobian/Hessian).
 *
 * Delegates to extractTensorElement with the two indices.
 *
 * @param jacobian_ptr Pointer to the 2-D tensor structure.
 * @param row_idx Row index.
 * @param col_idx Column index.
 * @param n Matrix dimension (unused; kept for call-site clarity).
 * @return The addressed element as a double value.
 */
llvm::Value* AutodiffCodegen::extractJacobianElement(llvm::Value* jacobian_ptr, llvm::Value* row_idx, llvm::Value* col_idx, llvm::Value* n) {
    using namespace llvm;
    // Use the general N-dimensional extractor with 2 indices
    return extractTensorElement(jacobian_ptr, {row_idx, col_idx});
}


/**
 * @brief Emit IR that computes the divergence of a vector field F: R^n -> R^n.
 *
 * The divergence is the trace of the Jacobian, so this builds the Jacobian via
 * jacobian(), validates it is a tensor type, then sums the diagonal entries
 * J[0,0] + J[1,1] + ... + J[n-1,n-1]. Invalid Jacobian types yield 0.0.
 *
 * @param op ESHKOL_DIVERGENCE_OP with divergence_op.function and divergence_op.point.
 * @return Tagged double holding the scalar divergence, or nullptr on error.
 */
llvm::Value* AutodiffCodegen::divergence(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->divergence_op.function || !op->divergence_op.point) {
        eshkol_error("Invalid divergence operation - missing function or point");
        return nullptr;
    }
    
    eshkol_info("Computing divergence of vector field");
    
    // The divergence is the sum of diagonal elements of the Jacobian
    // For F: ℝⁿ → ℝⁿ, Jacobian is n×n, divergence is trace(J)
    
    // Compute Jacobian matrix first
    eshkol_operations_t jacobian_temp;
    jacobian_temp.op = ESHKOL_JACOBIAN_OP;
    jacobian_temp.jacobian_op.function = op->divergence_op.function;
    jacobian_temp.jacobian_op.point = op->divergence_op.point;
    
    Value* jacobian_tagged = jacobian(&jacobian_temp);
    if (!jacobian_tagged) {
        eshkol_error("Failed to compute Jacobian for divergence");
        return nullptr;
    }
    
    // ENHANCED TYPE CHECK: Verify Jacobian is a valid tensor (same fix as Jacobian operator)
    Value* jacobian_type = tagged_.getType(jacobian_tagged);
    Value* jacobian_base_type = tagged_.getBaseType(jacobian_type);

    Value* jac_is_tensor_ptr = ctx_.builder().CreateICmpEQ(jacobian_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* jac_is_ad_tensor = ctx_.builder().CreateICmpEQ(jacobian_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    Value* jac_is_valid = ctx_.builder().CreateOr(jac_is_tensor_ptr, jac_is_ad_tensor);
    
    Function* div_current_func = ctx_.builder().GetInsertBlock()->getParent();
    BasicBlock* jacobian_valid = BasicBlock::Create(ctx_.context(), "div_jac_valid", div_current_func);
    BasicBlock* jacobian_invalid = BasicBlock::Create(ctx_.context(), "div_jac_invalid", div_current_func);
    BasicBlock* div_final = BasicBlock::Create(ctx_.context(), "div_final", div_current_func);
    
    ctx_.builder().CreateCondBr(jac_is_valid, jacobian_valid, jacobian_invalid);
    
    // Invalid jacobian: return 0.0 instead of crashing (only for genuinely invalid types)
    ctx_.builder().SetInsertPoint(jacobian_invalid);
    eshkol_debug("Divergence: Jacobian returned non-tensor type, returning 0.0");
    Value* zero_result = ConstantFP::get(ctx_.doubleType(), 0.0);
    ctx_.builder().CreateBr(div_final);
    
    // Valid jacobian: continue with normal computation
    ctx_.builder().SetInsertPoint(jacobian_valid);
    
    // Extract tensor pointer from validated tagged value
    Value* jacobian_ptr_int = tagged_.unpackInt64(jacobian_tagged);
    Value* jacobian_ptr = ctx_.builder().CreateIntToPtr(jacobian_ptr_int, ctx_.builder().getPtrTy());
    
    // Extract dimension n from Jacobian (it's n×n)
    Value* dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), jacobian_ptr, 0);
    Value* dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dims_field);
    Value* typed_dims_ptr = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());
    
    Value* n_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims_ptr,
        ConstantInt::get(ctx_.int64Type(), 0));
    Value* n = ctx_.builder().CreateLoad(ctx_.int64Type(), n_ptr);
    
    // Sum diagonal elements: J[0,0] + J[1,1] + ... + J[n-1,n-1]
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    BasicBlock* sum_loop_cond = BasicBlock::Create(ctx_.context(), "div_sum_cond", current_func);
    BasicBlock* sum_loop_body = BasicBlock::Create(ctx_.context(), "div_sum_body", current_func);
    BasicBlock* sum_loop_exit = BasicBlock::Create(ctx_.context(), "div_sum_exit", current_func);
    
    Value* sum_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "sum_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    
    Value* divergence_acc = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "div_acc");
    ctx_.builder().CreateStore(ConstantFP::get(ctx_.doubleType(), 0.0), divergence_acc);
    
    ctx_.builder().CreateBr(sum_loop_cond);
    
    ctx_.builder().SetInsertPoint(sum_loop_cond);
    Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), sum_idx);
    Value* i_less_n = ctx_.builder().CreateICmpULT(i, n);
    ctx_.builder().CreateCondBr(i_less_n, sum_loop_body, sum_loop_exit);
    
    ctx_.builder().SetInsertPoint(sum_loop_body);
    
    // Extract J[i,i] from nested list structure (not direct double access!)
    Value* diagonal_elem = extractJacobianElement(jacobian_ptr, i, i, n);
    
    // Add to accumulator
    Value* current_div = ctx_.builder().CreateLoad(ctx_.doubleType(), divergence_acc);
    Value* new_div = ctx_.builder().CreateFAdd(current_div, diagonal_elem);
    ctx_.builder().CreateStore(new_div, divergence_acc);
    
    Value* next_i = ctx_.builder().CreateAdd(i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, sum_idx);
    ctx_.builder().CreateBr(sum_loop_cond);
    
    ctx_.builder().SetInsertPoint(sum_loop_exit);
    Value* divergence_result = ctx_.builder().CreateLoad(ctx_.doubleType(), divergence_acc);
    ctx_.builder().CreateBr(div_final);
    
    // Merge valid and invalid paths
    ctx_.builder().SetInsertPoint(div_final);
    PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "div_result");
    result_phi->addIncoming(zero_result, jacobian_invalid);
    result_phi->addIncoming(divergence_result, sum_loop_exit);

    eshkol_info("Divergence computation complete");
    return tagged_.packDouble(result_phi);
}


/**
 * @brief Emit IR that computes the curl of a 3-D vector field (curl = nabla x F).
 *
 * Evaluates the input point (accepting Scheme vector or tensor forms), extracts
 * the dimension n, and requires n >= 2 (classic curl is 3-D; for n != 3 this
 * yields the generalized exterior-derivative 2-form). It builds the Jacobian and
 * combines its off-diagonal partials into the curl vector, e.g. in 3-D
 * (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy). Dimensions below 2 or an
 * invalid Jacobian return a null vector.
 *
 * @param op ESHKOL_CURL_OP with curl_op.function and curl_op.point.
 * @return Tagged tensor pointer (HEAP_PTR) holding the curl vector, or nullptr on error.
 */
llvm::Value* AutodiffCodegen::curl(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->curl_op.function || !op->curl_op.point) {
        eshkol_error("Invalid curl operation - missing function or point");
        return nullptr;
    }
    
    eshkol_info("Computing curl of 3D vector field");
    
    // First, validate that input is 3D
    Value* vector_val_raw = codegen_ast_callback_(op->curl_op.point, callback_context_);
    if (!vector_val_raw) {
        eshkol_error("Failed to evaluate curl point");
        return nullptr;
    }

    // Tensor literal fix: codegenTensor returns ptr-as-int64; wrap as HEAP_PTR
    Value* vector_val;
    if (vector_val_raw->getType() == ctx_.taggedValueType()) {
        vector_val = vector_val_raw;
    } else if (vector_val_raw->getType()->isIntegerTy(64) &&
               op->curl_op.point->type == ESHKOL_TENSOR) {
        vector_val = tagged_.packPtr(vector_val_raw, ESHKOL_VALUE_HEAP_PTR);
    } else if (vector_val_raw->getType()->isIntegerTy(64)) {
        vector_val = tagged_.packInt64(vector_val_raw, true);
    } else if (vector_val_raw->getType()->isDoubleTy()) {
        vector_val = tagged_.packDouble(vector_val_raw);
    } else {
        // Ensure tagged value (direct packing)
        vector_val = tagged_.packInt64(vector_val_raw, true);
    }

    // Get arena for OALR-compliant tensor allocation
    Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // M1 CONSOLIDATION: Handle HEAP_PTR (with subtype dispatch), legacy VECTOR_PTR, and tensor
    Value* curl_input_type = tagged_.getType(vector_val);
    Value* curl_input_base_type = tagged_.getBaseType(curl_input_type);
    Value* curl_is_heap_ptr = ctx_.builder().CreateICmpEQ(curl_input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* curl_is_legacy_vec = ctx_.builder().CreateICmpEQ(curl_input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    BasicBlock* curl_heap_dispatch = BasicBlock::Create(ctx_.context(), "curl_heap_dispatch", current_func);
    BasicBlock* curl_check_legacy = BasicBlock::Create(ctx_.context(), "curl_check_legacy", current_func);
    BasicBlock* curl_scheme_input = BasicBlock::Create(ctx_.context(), "curl_scheme_input", current_func);
    BasicBlock* curl_tensor_input = BasicBlock::Create(ctx_.context(), "curl_tensor_input", current_func);
    BasicBlock* curl_merge_n = BasicBlock::Create(ctx_.context(), "curl_merge_n", current_func);

    ctx_.builder().CreateCondBr(curl_is_heap_ptr, curl_heap_dispatch, curl_check_legacy);

    // HEAP_PTR dispatch - read subtype from header
    ctx_.builder().SetInsertPoint(curl_heap_dispatch);
    Value* curl_heap_ptr_val = tagged_.unpackPtr(vector_val);
    Value* curl_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), curl_heap_ptr_val, ConstantInt::get(ctx_.int64Type(), -8));
    Value* curl_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), curl_header_ptr);
    Value* curl_is_vec_subtype = ctx_.builder().CreateICmpEQ(curl_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    ctx_.builder().CreateCondBr(curl_is_vec_subtype, curl_scheme_input, curl_tensor_input);

    // Legacy VECTOR_PTR fallback
    ctx_.builder().SetInsertPoint(curl_check_legacy);
    ctx_.builder().CreateCondBr(curl_is_legacy_vec, curl_scheme_input, curl_tensor_input);

    // SCHEME VECTOR: Extract dimension from vector length
    ctx_.builder().SetInsertPoint(curl_scheme_input);
    Value* curl_svec_ptr_int = tagged_.unpackInt64(vector_val);
    Value* curl_svec_ptr = ctx_.builder().CreateIntToPtr(curl_svec_ptr_int, ctx_.builder().getPtrTy());
    Value* curl_svec_len_ptr = ctx_.builder().CreateBitCast(curl_svec_ptr, PointerType::getUnqual(ctx_.context()));
    Value* curl_svec_n = ctx_.builder().CreateLoad(ctx_.int64Type(), curl_svec_len_ptr);
    ctx_.builder().CreateBr(curl_merge_n);
    BasicBlock* curl_scheme_exit = ctx_.builder().GetInsertBlock();

    // TENSOR: Extract dimension from tensor structure
    ctx_.builder().SetInsertPoint(curl_tensor_input);
    Value* curl_tensor_ptr_int = tagged_.unpackInt64(vector_val);
    Value* curl_tensor_ptr = ctx_.builder().CreateIntToPtr(curl_tensor_ptr_int, ctx_.builder().getPtrTy());
    Value* dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), curl_tensor_ptr, 0);
    Value* dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dims_field);
    Value* typed_dims_ptr = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());
    Value* n_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims_ptr,
        ConstantInt::get(ctx_.int64Type(), 0));
    Value* curl_tensor_n = ctx_.builder().CreateLoad(ctx_.int64Type(), n_ptr);
    ctx_.builder().CreateBr(curl_merge_n);
    BasicBlock* curl_tensor_exit = ctx_.builder().GetInsertBlock();

    // MERGE: Get n from whichever path
    ctx_.builder().SetInsertPoint(curl_merge_n);
    PHINode* n = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "curl_n");
    n->addIncoming(curl_svec_n, curl_scheme_exit);
    n->addIncoming(curl_tensor_n, curl_tensor_exit);

    // ENHANCED VALIDATION: Accept n>=2 for general differential 2-forms
    // Classic curl is 3D, but generalized exterior derivative works in any dimension >= 2
    Value* n_ge_2 = ctx_.builder().CreateICmpUGE(n, ConstantInt::get(ctx_.int64Type(), 2));

    BasicBlock* dim_valid = BasicBlock::Create(ctx_.context(), "curl_dim_valid", current_func);
    BasicBlock* dim_invalid = BasicBlock::Create(ctx_.context(), "curl_dim_invalid", current_func);
    BasicBlock* curl_done = BasicBlock::Create(ctx_.context(), "curl_done", current_func);
    
    ctx_.builder().CreateCondBr(n_ge_2, dim_valid, dim_invalid);
    
    // Invalid dimension: return null vector for dim < 2
    ctx_.builder().SetInsertPoint(dim_invalid);
    eshkol_debug("Curl: dimension < 2, differential forms require at least 2D");
    Value* null_result_int = createNullVectorTensor(n);  // Use actual dimension, not hardcoded 3
    Value* null_result = tagged_.packPtr(null_result_int, ESHKOL_VALUE_HEAP_PTR);
    BasicBlock* dim_invalid_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(curl_done);
    
    // Valid dimension: compute curl (differential 2-form)
    // NOTE: For n!=3, this computes the generalized exterior derivative
    ctx_.builder().SetInsertPoint(dim_valid);
    
    // Compute Jacobian matrix (3×3)
    eshkol_operations_t jacobian_temp;
    jacobian_temp.op = ESHKOL_JACOBIAN_OP;
    jacobian_temp.jacobian_op.function = op->curl_op.function;
    jacobian_temp.jacobian_op.point = op->curl_op.point;
    
    Value* jacobian_tagged = jacobian(&jacobian_temp);
    if (!jacobian_tagged) {
        eshkol_error("Failed to compute Jacobian for curl");
        return nullptr;
    }
    
    // ENHANCED TYPE CHECK: Verify Jacobian is a valid tensor (same fix as Jacobian operator)
    Value* jacobian_type = tagged_.getType(jacobian_tagged);
    Value* jacobian_base_type = tagged_.getBaseType(jacobian_type);

    Value* jac_is_tensor_ptr = ctx_.builder().CreateICmpEQ(jacobian_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* jac_is_ad_tensor = ctx_.builder().CreateICmpEQ(jacobian_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    Value* jac_is_valid = ctx_.builder().CreateOr(jac_is_tensor_ptr, jac_is_ad_tensor);
    
    BasicBlock* jac_valid = BasicBlock::Create(ctx_.context(), "curl_jac_valid", current_func);
    BasicBlock* jac_invalid = BasicBlock::Create(ctx_.context(), "curl_jac_invalid", current_func);
    
    // If IS valid tensor type, proceed; if NOT, error path
    ctx_.builder().CreateCondBr(jac_is_valid, jac_valid, jac_invalid);
    
    // Invalid jacobian: return null 3D vector (only for genuinely invalid types)
    ctx_.builder().SetInsertPoint(jac_invalid);
    eshkol_debug("Curl: Jacobian returned non-tensor type, returning null vector");
    Value* null_curl_int = createNullVectorTensor(
        ConstantInt::get(ctx_.int64Type(), 3)
    );
    // Tag as TENSOR_PTR for proper display
    Value* null_curl = tagged_.packPtr(null_curl_int, ESHKOL_VALUE_HEAP_PTR);
    BasicBlock* jac_invalid_exit = ctx_.builder().GetInsertBlock(); // Capture actual exit block!
    ctx_.builder().CreateBr(curl_done);
    
    // Valid jacobian: continue with normal computation
    ctx_.builder().SetInsertPoint(jac_valid);
    
    // Extract tensor pointer from validated tagged value
    Value* jacobian_ptr_int = tagged_.unpackInt64(jacobian_tagged);
    Value* jacobian_ptr = ctx_.builder().CreateIntToPtr(jacobian_ptr_int, ctx_.builder().getPtrTy());
    Value* n_const = ConstantInt::get(ctx_.int64Type(), 3);
    
    // Extract specific partial derivatives from Jacobian's nested list structure
    // J[i,j] = ∂Fᵢ/∂xⱼ (row i, column j)
    // Jacobian elements are LIST POINTERS (rows), not doubles!
    
    // curl_x = ∂F₃/∂y - ∂F₂/∂z = J[2,1] - J[1,2]
    Value* dF3_dx2 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 2),  // row 2
        ConstantInt::get(ctx_.int64Type(), 1),  // col 1
        n_const);
    Value* dF2_dx3 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 1),  // row 1
        ConstantInt::get(ctx_.int64Type(), 2),  // col 2
        n_const);
    Value* curl_x = ctx_.builder().CreateFSub(dF3_dx2, dF2_dx3);
    
    // curl_y = ∂F₁/∂z - ∂F₃/∂x = J[0,2] - J[2,0]
    Value* dF1_dx3 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 0),  // row 0
        ConstantInt::get(ctx_.int64Type(), 2),  // col 2
        n_const);
    Value* dF3_dx1 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 2),  // row 2
        ConstantInt::get(ctx_.int64Type(), 0),  // col 0
        n_const);
    Value* curl_y = ctx_.builder().CreateFSub(dF1_dx3, dF3_dx1);
    
    // curl_z = ∂F₂/∂x - ∂F₁/∂y = J[1,0] - J[0,1]
    Value* dF2_dx1 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 1),  // row 1
        ConstantInt::get(ctx_.int64Type(), 0),  // col 0
        n_const);
    Value* dF1_dx2 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 0),  // row 0
        ConstantInt::get(ctx_.int64Type(), 1),  // col 1
        n_const);
    Value* curl_z = ctx_.builder().CreateFSub(dF2_dx1, dF1_dx2);
    
    // Create result 3D vector
    // Allocate result tensor via arena (OALR compliant - no malloc)
    Value* typed_result_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions [3]
    Value* result_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* result_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, result_dims_size});
    Value* typed_result_dims = ctx_.builder().CreatePointerCast(result_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 3), typed_result_dims);

    Value* result_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_ptr, 0);
    ctx_.builder().CreateStore(typed_result_dims, result_dims_field);

    Value* result_num_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_ptr, 1);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), result_num_dims_field);

    Value* result_total_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_ptr, 3);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 3), result_total_field);

    // Allocate and fill elements [curl_x, curl_y, curl_z]
    Value* result_elems_size = ConstantInt::get(ctx_.int64Type(), 3 * sizeof(double));
    Value* result_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, result_elems_size});
    Value* typed_result_elems = ctx_.builder().CreatePointerCast(result_elems_ptr, ctx_.builder().getPtrTy());
    
    Value* result_elems_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_ptr, 2);
    ctx_.builder().CreateStore(typed_result_elems, result_elems_field);
    
    // Store curl components
    Value* elem0_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_result_elems,
        ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateStore(curl_x, elem0_ptr);
    
    Value* elem1_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_result_elems,
        ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(curl_y, elem1_ptr);
    
    Value* elem2_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_result_elems,
        ConstantInt::get(ctx_.int64Type(), 2));
    ctx_.builder().CreateStore(curl_z, elem2_ptr);
    
    eshkol_info("Curl computation complete, returning 3D vector");
    Value* curl_result_int = ctx_.builder().CreatePtrToInt(typed_result_ptr, ctx_.int64Type());
    // Tag as TENSOR_PTR for proper display and type consistency
    Value* curl_result = tagged_.packPtr(curl_result_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(curl_done);
    BasicBlock* dim_valid_exit = ctx_.builder().GetInsertBlock(); // Capture actual predecessor!
    
    // Merge all paths with tagged_value results (type-consistent!)
    ctx_.builder().SetInsertPoint(curl_done);
    PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "curl_result");
    result_phi->addIncoming(null_result, dim_invalid_exit);   // Already tagged
    result_phi->addIncoming(null_curl, jac_invalid_exit);     // Already tagged
    result_phi->addIncoming(curl_result, dim_valid_exit);
    
    return result_phi;
}


/**
 * @brief Emit IR that computes the Laplacian of a scalar field f: R^n -> R.
 *
 * The Laplacian is the trace of the Hessian, so this builds the Hessian via
 * hessian() and sums its diagonal second derivatives
 * H[0,0] + H[1,1] + ... + H[n-1,n-1]. For a 1-D (scalar-input) field the Hessian
 * comes back as a plain double f''(x), which is used directly; genuinely
 * non-numeric results fall back to 0.0.
 *
 * @param op ESHKOL_LAPLACIAN_OP with laplacian_op.function and laplacian_op.point.
 * @return Tagged double holding the scalar Laplacian, or nullptr on error.
 */
llvm::Value* AutodiffCodegen::laplacian(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->laplacian_op.function || !op->laplacian_op.point) {
        eshkol_error("Invalid laplacian operation - missing function or point");
        return nullptr;
    }
    
    eshkol_info("Computing Laplacian of scalar field");
    
    // The Laplacian is the sum of diagonal elements of the Hessian
    // For f: ℝⁿ → ℝ, Hessian is n×n, Laplacian is trace(H)
    
    // Compute Hessian matrix first
    eshkol_operations_t hessian_temp;
    hessian_temp.op = ESHKOL_HESSIAN_OP;
    hessian_temp.hessian_op.function = op->laplacian_op.function;
    hessian_temp.hessian_op.point = op->laplacian_op.point;
    
    Value* hessian_tagged = hessian(&hessian_temp);
    if (!hessian_tagged) {
        eshkol_error("Failed to compute Hessian for Laplacian");
        return nullptr;
    }
    
    // ENHANCED TYPE CHECK: Verify Hessian is a valid tensor (same fix as Jacobian operator)
    Value* hessian_type = tagged_.getType(hessian_tagged);
    Value* hessian_base_type = tagged_.getBaseType(hessian_type);

    Value* hess_is_tensor_ptr = ctx_.builder().CreateICmpEQ(hessian_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* hess_is_ad_tensor = ctx_.builder().CreateICmpEQ(hessian_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    Value* hess_is_valid = ctx_.builder().CreateOr(hess_is_tensor_ptr, hess_is_ad_tensor);
    
    Function* lap_current_func = ctx_.builder().GetInsertBlock()->getParent();
    BasicBlock* hessian_valid = BasicBlock::Create(ctx_.context(), "lap_hess_valid", lap_current_func);
    BasicBlock* hessian_invalid = BasicBlock::Create(ctx_.context(), "lap_hess_invalid", lap_current_func);
    BasicBlock* lap_final = BasicBlock::Create(ctx_.context(), "lap_final", lap_current_func);
    
    ctx_.builder().CreateCondBr(hess_is_valid, hessian_valid, hessian_invalid);
    
    // Invalid hessian: for a SCALAR-input field the Hessian comes back as a
    // plain double (f''(x)) — the Laplacian of a 1-D field IS that value. Only
    // genuinely non-numeric results fall back to 0.0.
    ctx_.builder().SetInsertPoint(hessian_invalid);
    eshkol_debug("Laplacian: Hessian returned non-tensor type; using scalar f'' if double");
    Value* hess_is_double = ctx_.builder().CreateICmpEQ(hessian_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    Value* zero_lap_result = ctx_.builder().CreateSelect(hess_is_double,
        tagged_.unpackDouble(hessian_tagged), ConstantFP::get(ctx_.doubleType(), 0.0));
    ctx_.builder().CreateBr(lap_final);
    
    // Valid hessian: continue with normal computation
    ctx_.builder().SetInsertPoint(hessian_valid);
    
    // Extract tensor pointer from validated tagged value
    Value* hessian_ptr_int = tagged_.unpackInt64(hessian_tagged);
    Value* hessian_ptr = ctx_.builder().CreateIntToPtr(hessian_ptr_int, ctx_.builder().getPtrTy());
    
    // Extract dimension n from Hessian (it's n×n)
    Value* dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), hessian_ptr, 0);
    Value* dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dims_field);
    Value* typed_dims_ptr = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());
    
    Value* n_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims_ptr,
        ConstantInt::get(ctx_.int64Type(), 0));
    Value* n = ctx_.builder().CreateLoad(ctx_.int64Type(), n_ptr);
    
    // Get Hessian elements
    Value* elements_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), hessian_ptr, 2);
    Value* elements_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), elements_field);
    Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.builder().getPtrTy());
    
    // Sum diagonal elements: H[0,0] + H[1,1] + ... + H[n-1,n-1]
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    BasicBlock* sum_loop_cond = BasicBlock::Create(ctx_.context(), "lap_sum_cond", current_func);
    BasicBlock* sum_loop_body = BasicBlock::Create(ctx_.context(), "lap_sum_body", current_func);
    BasicBlock* sum_loop_exit = BasicBlock::Create(ctx_.context(), "lap_sum_exit", current_func);
    
    Value* sum_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "sum_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    
    Value* laplacian_acc = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "lap_acc");
    ctx_.builder().CreateStore(ConstantFP::get(ctx_.doubleType(), 0.0), laplacian_acc);
    
    ctx_.builder().CreateBr(sum_loop_cond);
    
    ctx_.builder().SetInsertPoint(sum_loop_cond);
    Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), sum_idx);
    Value* i_less_n = ctx_.builder().CreateICmpULT(i, n);
    ctx_.builder().CreateCondBr(i_less_n, sum_loop_body, sum_loop_exit);
    
    ctx_.builder().SetInsertPoint(sum_loop_body);
    
    // Calculate diagonal index: i*n + i
    Value* linear_idx = ctx_.builder().CreateMul(i, n);
    linear_idx = ctx_.builder().CreateAdd(linear_idx, i);
    
    // Load H[i,i]
    Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(),
        typed_elements_ptr, linear_idx);
    Value* diagonal_elem = ctx_.builder().CreateLoad(ctx_.doubleType(), elem_ptr);
    
    // Add to accumulator
    Value* current_lap = ctx_.builder().CreateLoad(ctx_.doubleType(), laplacian_acc);
    Value* new_lap = ctx_.builder().CreateFAdd(current_lap, diagonal_elem);
    ctx_.builder().CreateStore(new_lap, laplacian_acc);
    
    Value* next_i = ctx_.builder().CreateAdd(i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, sum_idx);
    ctx_.builder().CreateBr(sum_loop_cond);
    
    ctx_.builder().SetInsertPoint(sum_loop_exit);
    Value* laplacian_result = ctx_.builder().CreateLoad(ctx_.doubleType(), laplacian_acc);
    ctx_.builder().CreateBr(lap_final);
    
    // Merge valid and invalid paths
    ctx_.builder().SetInsertPoint(lap_final);
    PHINode* lap_result_phi = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "lap_result");
    lap_result_phi->addIncoming(zero_lap_result, hessian_invalid);
    lap_result_phi->addIncoming(laplacian_result, sum_loop_exit);

    eshkol_info("Laplacian computation complete");
    return tagged_.packDouble(lap_result_phi);
}


/**
 * @brief Emit IR that computes the directional derivative D_v f = grad(f) . v.
 *
 * Computes the gradient of f at the point, evaluates the direction vector v
 * (converting a (list ...) direction to a Scheme vector, and normalizing Scheme
 * vectors / tensors to a common tensor form), and returns the dot product of the
 * gradient with the direction.
 *
 * @param op ESHKOL_DIRECTIONAL_DERIV_OP with directional_deriv_op.function, .point and .direction.
 * @return Tagged double holding the directional derivative, or nullptr on error.
 */
llvm::Value* AutodiffCodegen::directionalDerivative(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->directional_deriv_op.function || !op->directional_deriv_op.point ||
        !op->directional_deriv_op.direction) {
        eshkol_error("Invalid directional-derivative operation - missing function, point, or direction");
        return nullptr;
    }
    
    eshkol_info("Computing directional derivative");
    
    // Step 1: Compute gradient ∇f
    eshkol_operations_t gradient_temp;
    gradient_temp.op = ESHKOL_GRADIENT_OP;
    gradient_temp.gradient_op.function = op->directional_deriv_op.function;
    gradient_temp.gradient_op.point = op->directional_deriv_op.point;
    
    Value* gradient_tagged = gradient(&gradient_temp);
    if (!gradient_tagged) {
        eshkol_error("Failed to compute gradient for directional derivative");
        return nullptr;
    }
    
    // CRITICAL FIX: Unpack tensor pointer from tagged_value
    Value* gradient_ptr_int = tagged_.unpackInt64(gradient_tagged);
    
    // Step 2: Get direction vector
    Value* direction_val_raw = codegen_ast_callback_(op->directional_deriv_op.direction, callback_context_);
    if (!direction_val_raw) {
        eshkol_error("Failed to evaluate direction vector");
        return nullptr;
    }

    // Tensor literal fix: codegenTensor returns ptr-as-int64; wrap as HEAP_PTR
    Value* direction_val;
    if (direction_val_raw->getType() == ctx_.taggedValueType()) {
        direction_val = direction_val_raw;
    } else if (direction_val_raw->getType()->isIntegerTy(64) &&
               op->directional_deriv_op.direction->type == ESHKOL_TENSOR) {
        direction_val = tagged_.packPtr(direction_val_raw, ESHKOL_VALUE_HEAP_PTR);
    } else if (direction_val_raw->getType()->isIntegerTy(64)) {
        direction_val = tagged_.packInt64(direction_val_raw, true);
    } else if (direction_val_raw->getType()->isDoubleTy()) {
        direction_val = tagged_.packDouble(direction_val_raw);
    } else {
        // direct packing
        direction_val = tagged_.packInt64(direction_val_raw, true);
    }

    // Get arena for OALR-compliant tensor allocation
    Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // M1 CONSOLIDATION: Handle HEAP_PTR (with subtype dispatch), legacy VECTOR_PTR, and tensor
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Effective-direction slot: a (list …) direction is a cons cell and must be
    // converted to a Scheme vector before the scheme-vector path reads it (same
    // cons-misread crash the gradient input had). Vector/tensor directions leave
    // the original value in the slot.
    llvm::IRBuilder<> ddEntry(&current_func->getEntryBlock(), current_func->getEntryBlock().begin());
    Value* dd_dir_slot = ddEntry.CreateAlloca(ctx_.taggedValueType(), nullptr, "dd_dir");
    ctx_.builder().CreateStore(direction_val, dd_dir_slot);

    Value* dir_type = tagged_.getType(direction_val);
    Value* dir_base_type = tagged_.getBaseType(dir_type);

    Value* dd_is_heap_ptr = ctx_.builder().CreateICmpEQ(dir_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* dd_is_legacy_vec = ctx_.builder().CreateICmpEQ(dir_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    BasicBlock* dd_heap_dispatch = BasicBlock::Create(ctx_.context(), "dd_heap_dispatch", current_func);
    BasicBlock* dd_check_legacy = BasicBlock::Create(ctx_.context(), "dd_check_legacy", current_func);
    BasicBlock* dd_scheme_vector = BasicBlock::Create(ctx_.context(), "dd_scheme_vector", current_func);
    BasicBlock* dd_tensor_input = BasicBlock::Create(ctx_.context(), "dd_tensor_input", current_func);
    BasicBlock* dd_merge_input = BasicBlock::Create(ctx_.context(), "dd_merge_input", current_func);

    ctx_.builder().CreateCondBr(dd_is_heap_ptr, dd_heap_dispatch, dd_check_legacy);

    // HEAP_PTR dispatch - read subtype from header
    ctx_.builder().SetInsertPoint(dd_heap_dispatch);
    Value* dd_heap_ptr_val = tagged_.unpackPtr(direction_val);
    Value* dd_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), dd_heap_ptr_val, ConstantInt::get(ctx_.int64Type(), -8));
    Value* dd_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), dd_header_ptr);
    Value* dd_is_vec_subtype = ctx_.builder().CreateICmpEQ(dd_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    Value* dd_is_cons_subtype = ctx_.builder().CreateICmpEQ(dd_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_CONS));
    BasicBlock* dd_check_cons = BasicBlock::Create(ctx_.context(), "dd_check_cons", current_func);
    BasicBlock* dd_list_to_svec = BasicBlock::Create(ctx_.context(), "dd_list_to_svec", current_func);
    ctx_.builder().CreateCondBr(dd_is_vec_subtype, dd_scheme_vector, dd_check_cons);

    ctx_.builder().SetInsertPoint(dd_check_cons);
    ctx_.builder().CreateCondBr(dd_is_cons_subtype, dd_list_to_svec, dd_tensor_input);

    ctx_.builder().SetInsertPoint(dd_list_to_svec);
    {
        llvm::Function* l2s_fn = ctx_.module().getFunction("eshkol_list_to_svec");
        if (!l2s_fn) {
            llvm::FunctionType* l2s_ty = llvm::FunctionType::get(
                ctx_.builder().getPtrTy(),
                {ctx_.builder().getPtrTy(), ctx_.builder().getPtrTy()}, false);
            l2s_fn = llvm::Function::Create(l2s_ty, llvm::Function::ExternalLinkage,
                "eshkol_list_to_svec", &ctx_.module());
        }
        Value* dd_svec = ctx_.builder().CreateCall(l2s_fn, {arena_ptr, dd_dir_slot});
        Value* dd_svec_int = ctx_.builder().CreatePtrToInt(dd_svec, ctx_.int64Type());
        ctx_.builder().CreateStore(tagged_.packPtr(dd_svec_int, ESHKOL_VALUE_HEAP_PTR), dd_dir_slot);
        ctx_.builder().CreateBr(dd_scheme_vector);
    }

    // Legacy VECTOR_PTR fallback
    ctx_.builder().SetInsertPoint(dd_check_legacy);
    ctx_.builder().CreateCondBr(dd_is_legacy_vec, dd_scheme_vector, dd_tensor_input);

    // SCHEME VECTOR: Convert to tensor format
    ctx_.builder().SetInsertPoint(dd_scheme_vector);

    Value* dd_effective_dir = ctx_.builder().CreateLoad(ctx_.taggedValueType(), dd_dir_slot);
    Value* dd_scheme_vec_ptr_int = tagged_.unpackInt64(dd_effective_dir);
    Value* dd_scheme_vec_ptr = ctx_.builder().CreateIntToPtr(dd_scheme_vec_ptr_int, ctx_.builder().getPtrTy());
    Value* dd_scheme_len_ptr = ctx_.builder().CreateBitCast(dd_scheme_vec_ptr, PointerType::getUnqual(ctx_.context()));
    Value* dd_scheme_len = ctx_.builder().CreateLoad(ctx_.int64Type(), dd_scheme_len_ptr);

    // Allocate tensor
    // Allocate tensor via arena (OALR compliant - no malloc)
    Value* dd_typed_scheme_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions
    Value* dd_scheme_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* dd_scheme_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, dd_scheme_dims_size});
    Value* dd_typed_scheme_dims = ctx_.builder().CreatePointerCast(dd_scheme_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(dd_scheme_len, dd_typed_scheme_dims);

    ctx_.builder().CreateStore(dd_typed_scheme_dims, ctx_.builder().CreateStructGEP(ctx_.tensorType(), dd_typed_scheme_tensor, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), dd_typed_scheme_tensor, 1));
    ctx_.builder().CreateStore(dd_scheme_len, ctx_.builder().CreateStructGEP(ctx_.tensorType(), dd_typed_scheme_tensor, 3));

    // Allocate and copy elements
    Value* dd_scheme_elems_size = ctx_.builder().CreateMul(dd_scheme_len,
        ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    Value* dd_scheme_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, dd_scheme_elems_size});
    Value* dd_typed_scheme_elems = ctx_.builder().CreatePointerCast(dd_scheme_elems_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(dd_typed_scheme_elems, ctx_.builder().CreateStructGEP(ctx_.tensorType(), dd_typed_scheme_tensor, 2));

    // Copy elements loop (extract doubles from tagged values)
    Value* dd_scheme_elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), dd_scheme_vec_ptr,
        ConstantInt::get(ctx_.int64Type(), 8));
    Value* dd_scheme_elem_base_typed = ctx_.builder().CreateBitCast(dd_scheme_elem_base, PointerType::getUnqual(ctx_.context()));

    BasicBlock* dd_svec_copy_cond = BasicBlock::Create(ctx_.context(), "dd_svec_copy_cond", current_func);
    BasicBlock* dd_svec_copy_body = BasicBlock::Create(ctx_.context(), "dd_svec_copy_body", current_func);
    BasicBlock* dd_svec_copy_done = BasicBlock::Create(ctx_.context(), "dd_svec_copy_done", current_func);

    Value* dd_svec_copy_i = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "dd_svec_copy_i");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), dd_svec_copy_i);
    ctx_.builder().CreateBr(dd_svec_copy_cond);

    ctx_.builder().SetInsertPoint(dd_svec_copy_cond);
    Value* dd_svec_i = ctx_.builder().CreateLoad(ctx_.int64Type(), dd_svec_copy_i);
    Value* dd_svec_cond = ctx_.builder().CreateICmpULT(dd_svec_i, dd_scheme_len);
    ctx_.builder().CreateCondBr(dd_svec_cond, dd_svec_copy_body, dd_svec_copy_done);

    ctx_.builder().SetInsertPoint(dd_svec_copy_body);
    Value* dd_svec_src_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), dd_scheme_elem_base_typed, dd_svec_i);
    Value* dd_svec_tagged_elem = ctx_.builder().CreateLoad(ctx_.taggedValueType(), dd_svec_src_ptr);
    Value* dd_svec_double_val = tagged_.unpackDouble(dd_svec_tagged_elem);
    Value* dd_svec_dst_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), dd_typed_scheme_elems, dd_svec_i);
    ctx_.builder().CreateStore(dd_svec_double_val, dd_svec_dst_ptr);
    Value* dd_svec_next_i = ctx_.builder().CreateAdd(dd_svec_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(dd_svec_next_i, dd_svec_copy_i);
    ctx_.builder().CreateBr(dd_svec_copy_cond);

    ctx_.builder().SetInsertPoint(dd_svec_copy_done);
    Value* dd_scheme_tensor_int = ctx_.builder().CreatePtrToInt(dd_typed_scheme_tensor, ctx_.int64Type());
    ctx_.builder().CreateBr(dd_merge_input);
    BasicBlock* dd_scheme_exit = ctx_.builder().GetInsertBlock();

    // TENSOR INPUT: Use as-is
    ctx_.builder().SetInsertPoint(dd_tensor_input);
    Value* dd_tensor_ptr_int = tagged_.unpackInt64(direction_val);
    ctx_.builder().CreateBr(dd_merge_input);
    BasicBlock* dd_tensor_exit = ctx_.builder().GetInsertBlock();

    // MERGE
    ctx_.builder().SetInsertPoint(dd_merge_input);
    PHINode* direction_ptr_int = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "dd_dir_ptr");
    direction_ptr_int->addIncoming(dd_scheme_tensor_int, dd_scheme_exit);
    direction_ptr_int->addIncoming(dd_tensor_ptr_int, dd_tensor_exit);

    // Step 3: Compute dot product: ∇f · v
    // Use class member ctx_.tensorType() (shared by all tensor operations)

    Value* gradient_ptr = ctx_.builder().CreateIntToPtr(gradient_ptr_int, ctx_.builder().getPtrTy());
    Value* direction_ptr = ctx_.builder().CreateIntToPtr(direction_ptr_int, ctx_.builder().getPtrTy());
    
    // Get gradient elements
    Value* grad_elements_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), gradient_ptr, 2);
    Value* grad_elements_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), grad_elements_field);
    Value* typed_grad_elements = ctx_.builder().CreatePointerCast(grad_elements_ptr, ctx_.builder().getPtrTy());
    
    // Get direction elements
    Value* dir_elements_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), direction_ptr, 2);
    Value* dir_elements_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dir_elements_field);
    Value* typed_dir_elements = ctx_.builder().CreatePointerCast(dir_elements_ptr, ctx_.builder().getPtrTy());
    
    // Get dimension n
    Value* grad_total_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), gradient_ptr, 3);
    Value* n_grad = ctx_.builder().CreateLoad(ctx_.int64Type(), grad_total_field);
    // P1: the dot-product loop below reads both grad[i] and dir[i] for i in
    // [0, n). Bounding by the gradient length alone reads past the direction
    // buffer when the caller passes a shorter direction vector. Clamp the loop
    // to min(n_grad, n_dir) so neither side is read out of bounds.
    Value* dir_total_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), direction_ptr, 3);
    Value* n_dir = ctx_.builder().CreateLoad(ctx_.int64Type(), dir_total_field);
    Value* n = ctx_.builder().CreateSelect(
        ctx_.builder().CreateICmpULT(n_grad, n_dir), n_grad, n_dir);

    // Compute dot product: sum(grad[i] * dir[i])
    // current_func already defined above
    BasicBlock* dot_loop_cond = BasicBlock::Create(ctx_.context(), "dirderiv_dot_cond", current_func);
    BasicBlock* dot_loop_body = BasicBlock::Create(ctx_.context(), "dirderiv_dot_body", current_func);
    BasicBlock* dot_loop_exit = BasicBlock::Create(ctx_.context(), "dirderiv_dot_exit", current_func);
    
    Value* dot_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "dot_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), dot_idx);
    
    Value* dot_acc = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "dot_acc");
    ctx_.builder().CreateStore(ConstantFP::get(ctx_.doubleType(), 0.0), dot_acc);
    
    ctx_.builder().CreateBr(dot_loop_cond);
    
    ctx_.builder().SetInsertPoint(dot_loop_cond);
    Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), dot_idx);
    Value* i_less_n = ctx_.builder().CreateICmpULT(i, n);
    ctx_.builder().CreateCondBr(i_less_n, dot_loop_body, dot_loop_exit);
    
    ctx_.builder().SetInsertPoint(dot_loop_body);
    
    // Load grad[i]
    Value* grad_elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(),
        typed_grad_elements, i);
    Value* grad_elem = ctx_.builder().CreateLoad(ctx_.doubleType(), grad_elem_ptr);
    
    // Load dir[i]
    Value* dir_elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(),
        typed_dir_elements, i);
    Value* dir_elem = ctx_.builder().CreateLoad(ctx_.doubleType(), dir_elem_ptr);
    
    // Multiply and accumulate
    Value* prod = ctx_.builder().CreateFMul(grad_elem, dir_elem);
    Value* current_dot = ctx_.builder().CreateLoad(ctx_.doubleType(), dot_acc);
    Value* new_dot = ctx_.builder().CreateFAdd(current_dot, prod);
    ctx_.builder().CreateStore(new_dot, dot_acc);
    
    Value* next_i = ctx_.builder().CreateAdd(i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, dot_idx);
    ctx_.builder().CreateBr(dot_loop_cond);
    
    ctx_.builder().SetInsertPoint(dot_loop_exit);
    Value* result = ctx_.builder().CreateLoad(ctx_.doubleType(), dot_acc);

    eshkol_info("Directional derivative computation complete");
    return tagged_.packDouble(result);
}


// ═══════════════════════════════════════════════════════════════════════════
// CAPTURE RESOLUTION — Extracted from llvm_codegen.cpp
// ═══════════════════════════════════════════════════════════════════════════

std::vector<llvm::Value*> AutodiffCodegen::loadCapturesForAutodiff(
    llvm::Function* func_ptr, const std::string& context_name) {
    using namespace llvm;

    std::vector<Value*> capture_args;

    FunctionType* func_type = func_ptr->getFunctionType();
    if (func_type->getNumParams() <= 1) {
        return capture_args; // No captures
    }

    // Respect user arity: for `(define (f x y z) ...)` all three params
    // are user arguments, not captures. Previously this assumed every
    // param after the first was a capture and spammed
    // `capture 'y' not found, using null pointer` for any multi-arg
    // user function passed to hessian / jacobian. Look up the arity
    // table (strip REPL __rv<n> suffix as elsewhere) and return empty
    // when all params are user-provided. Callers that genuinely need
    // hessian/jacobian of a multi-arg-no-captures function still need
    // to unpack the input tensor into scalar args — that's the caller
    // side, not this helper — but we shouldn't manufacture spurious
    // null captures in the meantime.
    std::string lambda_name = func_ptr->getName().str();
    if (function_arity_table_) {
        std::string arity_key = lambda_name;
        auto rv_pos = arity_key.rfind("__rv");
        if (rv_pos != std::string::npos &&
            rv_pos + 4 < arity_key.size() &&
            arity_key.find_first_not_of("0123456789", rv_pos + 4) == std::string::npos) {
            arity_key.erase(rv_pos);
        }
        auto arity_it = function_arity_table_->find(arity_key);
        if (arity_it != function_arity_table_->end()) {
            uint64_t user_arity = arity_it->second;
            if (user_arity >= func_type->getNumParams()) {
                return capture_args; // no captures — all params are user args
            }
        }
    }

    size_t num_captures = func_type->getNumParams() - 1;

    // REPL MODE: Get capture names from registry instead of parameter names
    std::vector<std::string> capture_names;
    if (repl_mode_enabled_ && *repl_mode_enabled_) {
        std::lock_guard<std::mutex> lock(*repl_mutex_);
        auto captures_it = repl_lambda_captures_->find(lambda_name);
        if (captures_it != repl_lambda_captures_->end()) {
            capture_names = captures_it->second;
        }
    }

    for (size_t i = 0; i < num_captures; i++) {
        std::string var_name;
        if (i < capture_names.size()) {
            var_name = capture_names[i];
        } else {
            auto arg_it = func_ptr->arg_begin();
            std::advance(arg_it, i + 1);
            if (arg_it != func_ptr->arg_end()) {
                var_name = arg_it->getName().str();
                if (var_name.find("captured_") == 0) {
                    var_name = var_name.substr(9);
                }
            }
        }

        std::string capture_key = lambda_name + "_capture_" + var_name;

        // First try capture-specific key in symbol tables
        auto it = global_symbol_table_->find(capture_key);
        bool found_in_global = (it != global_symbol_table_->end());
        if (!found_in_global) {
            it = symbol_table_->find(capture_key);
        }

        bool found = found_in_global ? (it != global_symbol_table_->end()) : (it != symbol_table_->end());

        // FALLBACK: Try raw variable name (for top-level global variables)
        if (!found) {
            it = global_symbol_table_->find(var_name);
            found_in_global = (it != global_symbol_table_->end());
            if (!found_in_global) {
                it = symbol_table_->find(var_name);
            }
            found = found_in_global ? (it != global_symbol_table_->end()) : (it != symbol_table_->end());
            if (found) {
                eshkol_debug("%s: found capture '%s' via raw variable name", context_name.c_str(), var_name.c_str());
            }
        }

        // REPL MODE: Try creating external declaration for capture global
        if (!found && repl_mode_enabled_ && *repl_mode_enabled_) {
            std::lock_guard<std::mutex> lock(*repl_mutex_);
            auto sym_it = repl_symbol_addresses_->find(capture_key);
            if (sym_it != repl_symbol_addresses_->end()) {
                GlobalVariable* capture_global = ctx_.module().getGlobalVariable(capture_key);
                if (!capture_global) {
                    capture_global = new GlobalVariable(
                        ctx_.module(),
                        ctx_.taggedValueType(),
                        false,
                        GlobalValue::ExternalLinkage,
                        nullptr,
                        capture_key
                    );
                }
                Value* helper_global_ptr_int = ctx_.builder().CreatePtrToInt(capture_global, ctx_.int64Type());
                Value* helper_packed_capture = tagged_.packInt64(helper_global_ptr_int, true);
                Value* helper_capture_storage = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "autodiff_capture_storage");
                ctx_.builder().CreateStore(helper_packed_capture, helper_capture_storage);
                capture_args.push_back(helper_capture_storage);
                continue;
            }
        }

        if (found && it->second) {
            Value* storage = it->second;
            // ESH-0072/0097: a lambda capturing a LOCAL function parameter
            // resolves `storage` to that parameter's Argument (a tagged_value
            // STRUCT, not a pointer) — ptrtoint on it is invalid IR and broke
            // jacobian/hessian/divergence/curl/laplacian of such a lambda.
            // A named-let carry pointer ("<var>_cap") forwards as-is; a
            // value-typed capture funnels through a temp slot; only a genuine
            // pointer storage is packed via ptrtoint. Mirrors
            // resolveGradientCaptures.
            if (auto* arg = llvm::dyn_cast<llvm::Argument>(storage)) {
                if (arg->getType()->isPointerTy() &&
                    (arg->getName() == (var_name + "_cap") ||
                     arg->getName() == ("captured_" + var_name))) {
                    // #296: when the free variable is itself a capture of the
                    // ENCLOSING function (a transitive capture), `storage` is
                    // that function's own `captured_<var>` slot. Forward it
                    // as-is — re-wrapping it in the pointer-marker below hands
                    // the callee the slot's ADDRESS as its value.
                    capture_args.push_back(storage);
                    continue;
                }
            }
            if (!storage->getType()->isPointerTy()) {
                Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
                IRBuilder<> entry_builder(&current_func->getEntryBlock(),
                                          current_func->getEntryBlock().begin());
                AllocaInst* val_temp = entry_builder.CreateAlloca(
                    ctx_.taggedValueType(), nullptr, var_name + "_autodiff_capture_val");
                ctx_.builder().CreateStore(storage, val_temp);
                capture_args.push_back(val_temp);
                continue;
            }
            if (isTcoLoopAlloca(storage)) {
                // ESH-0221: see isTcoLoopAlloca's doc comment. `storage` is a
                // TCO loop-carried parameter's alloca — the callee expects a
                // single-load VALUE capture, not the mutable-variable
                // pointer-marker built below (jacobian/hessian/divergence/
                // curl/laplacian all share this capture resolver).
                Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
                IRBuilder<> entry_builder(&current_func->getEntryBlock(),
                                          current_func->getEntryBlock().begin());
                AllocaInst* val_temp = entry_builder.CreateAlloca(
                    ctx_.taggedValueType(), nullptr, var_name + "_autodiff_capture_tco_val");
                Value* tco_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), storage);
                ctx_.builder().CreateStore(tco_val, val_temp);
                capture_args.push_back(val_temp);
                continue;
            }
            Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
            IRBuilder<> entry_builder(&current_func->getEntryBlock(),
                                      current_func->getEntryBlock().begin());
            AllocaInst* temp_alloca = entry_builder.CreateAlloca(
                ctx_.taggedValueType(), nullptr, var_name + "_autodiff_capture_storage");

            Value* ptr_as_int = ctx_.builder().CreatePtrToInt(storage, ctx_.int64Type());
            Value* packed_ptr = tagged_.packInt64(ptr_as_int, true);
            ctx_.builder().CreateStore(packed_ptr, temp_alloca);

            capture_args.push_back(temp_alloca);
        } else {
            capture_args.push_back(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())));
            eshkol_warn("%s: capture '%s' not found, using null pointer", context_name.c_str(), var_name.c_str());
        }
    }

    return capture_args;
}

/**
 * @brief Append closure-capture arguments to a gradient/AD call so the callee gets its free variables.
 *
 * When the target lambda's LLVM signature has more parameters than the caller
 * has supplied, the surplus are captured free variables. For each one this
 * resolves the capture name (from the REPL capture registry or the mangled
 * argument name), then looks up its storage using capture-key (global then
 * local) followed by raw-name (local-first so a named-let carry pointer wins
 * over a shadowed global), and pushes the storage pointer onto call_args (or a
 * null pointer with a warning if unresolved).
 *
 * @param func_ptr The callee lambda whose captures must be supplied.
 * @param call_args In/out argument list, extended in place with capture pointers.
 * @param context_label Label used in diagnostic messages.
 */
void AutodiffCodegen::resolveGradientCaptures(
    llvm::Function* func_ptr,
    std::vector<llvm::Value*>& call_args,
    const std::string& context_label) {
    using namespace llvm;

    FunctionType* func_type = func_ptr->getFunctionType();
    size_t total_llvm_params = func_type->getNumParams();
    size_t args_provided = call_args.size();

    if (total_llvm_params <= args_provided) return;

    size_t num_captures = total_llvm_params - args_provided;
    std::string lambda_name = func_ptr->getName().str();

    // REPL MODE: Get capture names from registry
    std::vector<std::string> capture_names;
    if (repl_mode_enabled_ && *repl_mode_enabled_) {
        std::lock_guard<std::mutex> lock(*repl_mutex_);
        auto captures_it = repl_lambda_captures_->find(lambda_name);
        if (captures_it != repl_lambda_captures_->end()) {
            capture_names = captures_it->second;
        }
    }

    for (size_t ci = 0; ci < num_captures; ci++) {
        std::string var_name;
        if (ci < capture_names.size()) {
            var_name = capture_names[ci];
        } else {
            auto arg_it = func_ptr->arg_begin();
            std::advance(arg_it, args_provided + ci);
            if (arg_it != func_ptr->arg_end()) {
                var_name = arg_it->getName().str();
                if (var_name.find("captured_") == 0) var_name = var_name.substr(9);
            }
        }

        std::string capture_key = lambda_name + "_capture_" + var_name;

        // Search order: capture key (global → local), then raw name LOCAL → global.
        // ESH-0070: the raw-name LOOKUP MUST prefer the LOCAL symbol table so it
        // resolves the SAME storage the lambda itself captured. Inside a named-let
        // loop the free var `a` is bound locally to the carry pointer `%a_cap`
        // (which lexically shadows the top-level global `@a`); the lambda captured
        // that local pointer. The old global-first order found `@a` instead, so
        // the gradient handed the lambda a capture in the wrong (packed) ABI.
        Value* storage = nullptr;
        auto it = global_symbol_table_->find(capture_key);
        if (it != global_symbol_table_->end() && it->second) {
            storage = it->second;
        } else {
            it = symbol_table_->find(capture_key);
            if (it != symbol_table_->end() && it->second) {
                storage = it->second;
            } else {
                it = symbol_table_->find(var_name);
                if (it != symbol_table_->end() && it->second) {
                    storage = it->second;
                } else {
                    it = global_symbol_table_->find(var_name);
                    if (it != global_symbol_table_->end() && it->second) {
                        storage = it->second;
                    }
                }
            }
        }

        // REPL MODE: Try creating external declaration for capture global
        if (!storage && repl_mode_enabled_ && *repl_mode_enabled_) {
            std::lock_guard<std::mutex> lock(*repl_mutex_);
            auto sym_it = repl_symbol_addresses_->find(capture_key);
            if (sym_it != repl_symbol_addresses_->end()) {
                GlobalVariable* capture_global = ctx_.module().getGlobalVariable(capture_key);
                if (!capture_global) {
                    capture_global = new GlobalVariable(
                        ctx_.module(), ctx_.taggedValueType(), false,
                        GlobalValue::ExternalLinkage, nullptr, capture_key);
                }
                Value* global_ptr_int = ctx_.builder().CreatePtrToInt(capture_global, ctx_.int64Type());
                Value* packed = tagged_.packInt64(global_ptr_int, true);
                Value* temp = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_cap");
                ctx_.builder().CreateStore(packed, temp);
                call_args.push_back(temp);
                continue;
            }
        }

        if (storage) {
            // ESH-0070: named-let loop carries (#224) bind a free variable to a
            // pointer Argument named "<var>_cap" (see llvm_codegen.cpp
            // codegenNamedLet). A lambda compiled inside that loop captures the
            // variable through that direct pointer, so its `captured_<var>` param
            // is loaded DIRECTLY (single indirection). Passing the usual
            // {INT64, ptrtoint(storage)} double-indirection slot here made the
            // lambda read the *address* as the value → garbage gradients that
            // compounded each loop iteration (the Noesis named-let blow-up).
            // When storage IS that carry pointer, forward it as-is.
            if (auto* arg = llvm::dyn_cast<llvm::Argument>(storage)) {
                if (arg->getType()->isPointerTy() &&
                    (arg->getName() == (var_name + "_cap") ||
                     arg->getName() == ("captured_" + var_name))) {
                    // #296: a TRANSITIVE capture — the free variable is itself
                    // a capture of the enclosing function, so `storage` is that
                    // function's own pointer-typed `captured_<var>` slot,
                    // already in the callee's expected single-load convention.
                    // Forward it as-is. The default pointer-marker packing
                    // below would hand the differentiated lambda the slot's
                    // ADDRESS as its value: a custom-VJP callee (vqe-energy)
                    // then unpacked that address as its Hamiltonian handle,
                    // its AD-prepare failed, and the gradient silently zeroed
                    // (issue #296).
                    call_args.push_back(storage);
                    continue;
                }
            }
            if (!storage->getType()->isPointerTy()) {
                // Value-typed capture (e.g. a function passed as a tagged_value
                // parameter and captured by the gradient's lambda, as in
                // (lambda (y) (gradient f y))). PtrToInt on a struct is invalid;
                // funnel the value through a temp slot so the lambda's single
                // `load captured_<var>` reads it directly. Mirrors the
                // derivative() capture handling (ESH-0070).
                Value* val_temp = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_cap_val");
                ctx_.builder().CreateStore(storage, val_temp);
                call_args.push_back(val_temp);
                continue;
            }
            if (isTcoLoopAlloca(storage)) {
                // ESH-0221: see isTcoLoopAlloca's doc comment. `storage` is a
                // TCO loop-carried parameter's alloca — the callee expects a
                // single-load VALUE capture, not the mutable-variable
                // pointer-marker built below.
                Value* tco_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), storage);
                Value* val_temp = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_cap_tco_val");
                ctx_.builder().CreateStore(tco_val, val_temp);
                call_args.push_back(val_temp);
                continue;
            }
            Value* ptr_int = ctx_.builder().CreatePtrToInt(storage, ctx_.int64Type());
            Value* packed = tagged_.packInt64(ptr_int, true);
            Value* temp = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_cap");
            ctx_.builder().CreateStore(packed, temp);
            call_args.push_back(temp);
        } else {
            call_args.push_back(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())));
            eshkol_warn("Gradient (%s): capture '%s' not found, using null pointer",
                        context_label.c_str(), var_name.c_str());
        }
    }
}

/**
 * @brief Allocate a fresh reverse-mode AD tape in the current arena.
 *
 * @return Pointer to the new tape, or a null pointer if the arena or tape
 *         allocator is unavailable.
 */
llvm::Value* AutodiffCodegen::createTape() {
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return llvm::ConstantPointerNull::get(ctx_.ptrType());

    llvm::Function* alloc_tape = mem_.getArenaAllocateTape();
    if (!alloc_tape) return llvm::ConstantPointerNull::get(ctx_.ptrType());

    return ctx_.builder().CreateCall(alloc_tape, {arena_ptr});
}

/**
 * @brief Emit the reverse-mode backward pass over a recorded AD tape.
 *
 * Seeds the output node's gradient to 1.0 (and, for tensor outputs, an all-ones
 * tensor gradient via the runtime), then walks the tape's nodes in reverse
 * recording order (which is topological for a forward-built graph), calling
 * propagateGradient on each non-null node to distribute gradients to its inputs
 * by the chain rule. Null tape or output pointers are guarded at runtime.
 *
 * @param tape The AD tape recorded during the forward pass.
 * @param output_node The node whose gradient is seeded to 1.0 (the loss/output).
 */
void AutodiffCodegen::backpropagate(llvm::Value* tape, llvm::Value* output_node) {
    // CRITICAL: Add runtime null checks for placeholder functions
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    if (!current_func) {
        eshkol_error("Backward pass requires active function context");
        return;
    }

    // Create safety check blocks
    llvm::BasicBlock* check_validity = llvm::BasicBlock::Create(ctx_.context(), "backward_check_valid", current_func);
    llvm::BasicBlock* backward_valid = llvm::BasicBlock::Create(ctx_.context(), "backward_valid", current_func);
    llvm::BasicBlock* backward_skip = llvm::BasicBlock::Create(ctx_.context(), "backward_skip", current_func);

    ctx_.builder().CreateBr(check_validity);

    // Check if output node and tape are valid (not null)
    ctx_.builder().SetInsertPoint(check_validity);
    llvm::Value* output_int = ctx_.builder().CreatePtrToInt(output_node, ctx_.int64Type());
    llvm::Value* tape_int = ctx_.builder().CreatePtrToInt(tape, ctx_.int64Type());

    llvm::Value* output_valid = ctx_.builder().CreateICmpNE(output_int, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* tape_valid = ctx_.builder().CreateICmpNE(tape_int, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* both_valid = ctx_.builder().CreateAnd(output_valid, tape_valid);

    ctx_.builder().CreateCondBr(both_valid, backward_valid, backward_skip);

    ctx_.builder().SetInsertPoint(backward_valid);

    // AD Phase A counter: one reverse sweep is about to execute.
    ctx_.builder().CreateCall(ctx_.module().getOrInsertFunction(
        "eshkol_ad_count_reverse",
        llvm::FunctionType::get(ctx_.voidType(), {}, false)), {});

    // Initialize output gradient = 1.0 (seed for backpropagation)
    storeNodeGradient(output_node, llvm::ConstantFP::get(ctx_.doubleType(), 1.0));

    // Seed tensor gradient: if the output node has tensor_value set,
    // allocate an all-ones tensor gradient (dL/dL = 1 for every element).
    // This is a no-op for scalar nodes (tensor_value == NULL).
    {
        llvm::FunctionType* seed_type = llvm::FunctionType::get(
            ctx_.voidType(), {ctx_.ptrType()}, false);
        llvm::FunctionCallee seed_fn = ctx_.module().getOrInsertFunction(
            "eshkol_seed_tensor_gradient", seed_type);
        ctx_.builder().CreateCall(seed_fn, {output_node});
    }

    // Get number of nodes in tape (runtime value, not compile-time constant)
    llvm::Function* get_count_func = mem_.getArenaTapeGetNodeCount();
    if (!get_count_func) {
        eshkol_error("Backward pass: arena_tape_get_node_count not available");
        ctx_.builder().CreateBr(backward_skip);
        ctx_.builder().SetInsertPoint(backward_skip);
        return;
    }
    llvm::Value* num_nodes = ctx_.builder().CreateCall(get_count_func, {tape});

    // Allocate loop counter for backward traversal (MUST iterate in reverse order)
    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "backward_counter");
    if (!counter) {
        eshkol_error("Failed to allocate backward pass counter");
        ctx_.builder().CreateBr(backward_skip);
        ctx_.builder().SetInsertPoint(backward_skip);
        return;
    }

    // Initialize counter = num_nodes (start at end, decrement to 0)
    ctx_.builder().CreateStore(num_nodes, counter);

    // Create loop basic blocks (REQUIRED for LLVM IR structure)
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "backward_loop_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "backward_loop_body", current_func);
    llvm::BasicBlock* check_node = llvm::BasicBlock::Create(ctx_.context(), "backward_check_node", current_func);
    llvm::BasicBlock* propagate_block = llvm::BasicBlock::Create(ctx_.context(), "backward_propagate", current_func);
    llvm::BasicBlock* skip_node = llvm::BasicBlock::Create(ctx_.context(), "backward_skip_node", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "backward_loop_exit", current_func);

    // Jump to loop condition
    ctx_.builder().CreateBr(loop_cond);

    // Loop condition: while (counter > 0)
    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* counter_val = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* counter_gt_zero = ctx_.builder().CreateICmpUGT(counter_val,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(counter_gt_zero, loop_body, loop_exit);

    // Loop body: Process node at index (counter - 1)
    ctx_.builder().SetInsertPoint(loop_body);

    // Decrement counter FIRST to get 0-based index
    llvm::Value* counter_minus_1 = ctx_.builder().CreateSub(counter_val,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(counter_minus_1, counter);

    // Get node at index using arena_tape_get_node (may return nullptr)
    llvm::Function* get_node_func = mem_.getArenaTapeGetNode();
    llvm::Value* node_ptr = ctx_.builder().CreateCall(get_node_func,
        {tape, counter_minus_1});

    // Null check before propagation (defensive programming)
    ctx_.builder().CreateBr(check_node);

    ctx_.builder().SetInsertPoint(check_node);
    llvm::Value* node_is_null = ctx_.builder().CreateICmpEQ(node_ptr,
        llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx_.context())));
    ctx_.builder().CreateCondBr(node_is_null, skip_node, propagate_block);

    // Propagate gradient for this node
    ctx_.builder().SetInsertPoint(propagate_block);
    propagateGradient(node_ptr);
    ctx_.builder().CreateBr(skip_node);

    // Skip or continue to next iteration
    ctx_.builder().SetInsertPoint(skip_node);
    ctx_.builder().CreateBr(loop_cond);

    // Loop exit: backward pass complete
    ctx_.builder().SetInsertPoint(loop_exit);
    ctx_.builder().CreateBr(backward_skip);

    // Skip block: exit point for null/invalid inputs
    ctx_.builder().SetInsertPoint(backward_skip);

    eshkol_debug("Completed backward pass through computational graph");
}

/**
 * @brief Apply the local chain rule for one AD node, pushing gradient into its inputs.
 *
 * Loads the node's accumulated gradient and operation type. Nodes carrying a
 * tensor gradient take a fast path that calls the C runtime tensor backward
 * dispatcher; scalar nodes branch on op type and accumulate the appropriate
 * partial-derivative-weighted gradient into input1 and input2 (e.g. add passes
 * the gradient through unchanged; mul weights each input by the other's value;
 * sin/cos/... apply their derivative). Leaf nodes (constant, variable) have no
 * inputs and are skipped. Called once per node by backpropagate.
 *
 * @param node_ptr The AD node currently being processed in the backward pass.
 */
void AutodiffCodegen::propagateGradient(llvm::Value* node_ptr) {
    if (!node_ptr) return;

    llvm::StructType* ad_node_type = ctx_.adNodeType();

    // Load node type
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 0);
    llvm::Value* node_type = ctx_.builder().CreateLoad(ctx_.int32Type(), type_ptr);

    // Load node gradient
    llvm::Value* node_grad = loadNodeGradient(node_ptr);

    // Load input pointers
    llvm::Value* input1 = loadNodeInput1(node_ptr);
    llvm::Value* input2 = loadNodeInput2(node_ptr);

    // Branch on operation type to apply correct gradient rules
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Create done block first (referenced by tensor dispatch path below)
    llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "grad_done", current_func);

    // === TENSOR GRADIENT FAST PATH ===
    // If tensor_gradient (field 7) is non-null, the node was recorded as a tensor
    // operation via recordADNodeTensor. Dispatch to the C runtime backward function
    // which reads saved_tensors, params, shape/ndim and calls the appropriate
    // eshkol_backward_* function (conv2d, matmul, attention, etc.)
    {
        llvm::Value* tg_field_ptr = ctx_.builder().CreateStructGEP(
            ad_node_type, node_ptr, TypeSystem::AD_NODE_TENSOR_GRADIENT_IDX);
        llvm::Value* tg_val = ctx_.builder().CreateLoad(ctx_.ptrType(), tg_field_ptr);
        llvm::Value* has_tensor = ctx_.builder().CreateICmpNE(tg_val,
            llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx_.context())));

        llvm::BasicBlock* tensor_dispatch_bb = llvm::BasicBlock::Create(
            ctx_.context(), "tensor_backward_dispatch", current_func);
        llvm::BasicBlock* scalar_dispatch_bb = llvm::BasicBlock::Create(
            ctx_.context(), "scalar_backward_dispatch", current_func);

        ctx_.builder().CreateCondBr(has_tensor, tensor_dispatch_bb, scalar_dispatch_bb);

        // Tensor path: call runtime dispatcher that handles all tensor ops
        ctx_.builder().SetInsertPoint(tensor_dispatch_bb);
        llvm::FunctionType* dispatch_type = llvm::FunctionType::get(
            ctx_.voidType(), {ctx_.ptrType()}, false);
        llvm::FunctionCallee dispatch_fn = ctx_.module().getOrInsertFunction(
            "eshkol_tensor_backward_dispatch", dispatch_type);
        ctx_.builder().CreateCall(dispatch_fn, {node_ptr});
        ctx_.builder().CreateBr(done_block);

        // Continue with scalar dispatch for nodes without tensor gradients
        ctx_.builder().SetInsertPoint(scalar_dispatch_bb);
    }

    // Create blocks for each scalar operation type
    llvm::BasicBlock* add_block = llvm::BasicBlock::Create(ctx_.context(), "grad_add", current_func);
    llvm::BasicBlock* sub_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sub", current_func);
    llvm::BasicBlock* mul_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mul", current_func);
    llvm::BasicBlock* div_block = llvm::BasicBlock::Create(ctx_.context(), "grad_div", current_func);
    llvm::BasicBlock* sin_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sin", current_func);
    llvm::BasicBlock* cos_block = llvm::BasicBlock::Create(ctx_.context(), "grad_cos", current_func);

    // === LEAF NODES (types 0, 1): constants and variables have no inputs to propagate to ===
    llvm::Value* is_constant = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    llvm::Value* is_variable = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 1));
    llvm::Value* is_leaf = ctx_.builder().CreateOr(is_constant, is_variable);
    llvm::BasicBlock* check_ops = llvm::BasicBlock::Create(ctx_.context(), "check_ops", current_func);
    ctx_.builder().CreateCondBr(is_leaf, done_block, check_ops);

    ctx_.builder().SetInsertPoint(check_ops);

    // Switch on node type (scalar backward passes)
    // For ADD (type=2): gradient flows equally to both inputs
    llvm::Value* is_add = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 2));

    llvm::BasicBlock* check_sub = llvm::BasicBlock::Create(ctx_.context(), "check_sub", current_func);
    ctx_.builder().CreateCondBr(is_add, add_block, check_sub);

    // ADD: dL/dx = dL/dz * 1, dL/dy = dL/dz * 1
    ctx_.builder().SetInsertPoint(add_block);
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad);
    ctx_.builder().CreateBr(done_block);

    // Check for SUB
    ctx_.builder().SetInsertPoint(check_sub);
    llvm::Value* is_sub = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 3));
    llvm::BasicBlock* check_mul = llvm::BasicBlock::Create(ctx_.context(), "check_mul", current_func);
    ctx_.builder().CreateCondBr(is_sub, sub_block, check_mul);

    // SUB: dL/dx = dL/dz * 1, dL/dy = dL/dz * (-1)
    ctx_.builder().SetInsertPoint(sub_block);
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) {
        llvm::Value* neg_grad = ctx_.builder().CreateFNeg(node_grad);
        accumulateGradient(input2, neg_grad);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for MUL
    ctx_.builder().SetInsertPoint(check_mul);
    llvm::Value* is_mul = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 4));
    llvm::BasicBlock* check_div = llvm::BasicBlock::Create(ctx_.context(), "check_div", current_func);
    ctx_.builder().CreateCondBr(is_mul, mul_block, check_div);

    // MUL: dL/dx = dL/dz * y, dL/dy = dL/dz * x
    ctx_.builder().SetInsertPoint(mul_block);
    if (input1 && input2) {
        llvm::Value* input1_val = loadNodeValue(input1);
        llvm::Value* input2_val = loadNodeValue(input2);

        llvm::Value* grad_input1 = ctx_.builder().CreateFMul(node_grad, input2_val);
        llvm::Value* grad_input2 = ctx_.builder().CreateFMul(node_grad, input1_val);

        // DOUBLE BACKWARD: Track degree when multiplying by variable value
        llvm::GlobalVariable* inner_var_node_ptr = ctx_.innerVarNodePtr();
        llvm::GlobalVariable* gradient_x_degree = ctx_.gradientXDegree();

        if (inner_var_node_ptr && gradient_x_degree) {
            // Load stored variable node and its value for comparison
            llvm::Value* stored_var_node = ctx_.builder().CreateLoad(llvm::PointerType::getUnqual(ctx_.context()), inner_var_node_ptr);
            llvm::Value* stored_var_is_valid = ctx_.builder().CreateICmpNE(stored_var_node,
                llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx_.context())));

            // Only track degree if we have a stored variable node
            llvm::BasicBlock* track_degree_bb = llvm::BasicBlock::Create(ctx_.context(), "track_degree", current_func);
            llvm::BasicBlock* skip_degree_bb = llvm::BasicBlock::Create(ctx_.context(), "skip_degree", current_func);
            llvm::BasicBlock* after_degree_bb = llvm::BasicBlock::Create(ctx_.context(), "after_degree", current_func);

            ctx_.builder().CreateCondBr(stored_var_is_valid, track_degree_bb, skip_degree_bb);

            ctx_.builder().SetInsertPoint(track_degree_bb);
            llvm::Value* var_val = loadNodeValue(stored_var_node);

            // Check node TYPE as well as value to avoid false positives
            llvm::Value* input1_type_ptr = ctx_.builder().CreateStructGEP(ad_node_type, input1, 0);
            llvm::Value* input1_type = ctx_.builder().CreateLoad(ctx_.int32Type(), input1_type_ptr);
            llvm::Value* input1_is_var_type = ctx_.builder().CreateICmpEQ(input1_type, llvm::ConstantInt::get(ctx_.int32Type(), 1));

            llvm::Value* input2_type_ptr = ctx_.builder().CreateStructGEP(ad_node_type, input2, 0);
            llvm::Value* input2_type = ctx_.builder().CreateLoad(ctx_.int32Type(), input2_type_ptr);
            llvm::Value* input2_is_var_type = ctx_.builder().CreateICmpEQ(input2_type, llvm::ConstantInt::get(ctx_.int32Type(), 1));

            // Check if input2 is the variable (by value comparison with tolerance AND type check)
            llvm::Value* diff2 = ctx_.builder().CreateFSub(input2_val, var_val);
            llvm::Function* fabs_intrinsic = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});
            llvm::Value* abs_diff2 = ctx_.builder().CreateCall(fabs_intrinsic, {diff2});
            llvm::Value* val_matches_2 = ctx_.builder().CreateFCmpOLT(abs_diff2, llvm::ConstantFP::get(ctx_.doubleType(), 1e-10));
            llvm::Value* is_var2 = ctx_.builder().CreateAnd(val_matches_2, input2_is_var_type);

            // Check if input1 is the variable
            llvm::Value* diff1 = ctx_.builder().CreateFSub(input1_val, var_val);
            llvm::Value* abs_diff1 = ctx_.builder().CreateCall(fabs_intrinsic, {diff1});
            llvm::Value* val_matches_1 = ctx_.builder().CreateFCmpOLT(abs_diff1, llvm::ConstantFP::get(ctx_.doubleType(), 1e-10));
            llvm::Value* is_var1 = ctx_.builder().CreateAnd(val_matches_1, input1_is_var_type);

            // Count how many times we multiply by variable value
            llvm::Value* current_degree = ctx_.builder().CreateLoad(ctx_.int64Type(), gradient_x_degree);
            llvm::Value* inc2 = ctx_.builder().CreateSelect(is_var2,
                llvm::ConstantInt::get(ctx_.int64Type(), 1),
                llvm::ConstantInt::get(ctx_.int64Type(), 0));
            llvm::Value* inc1 = ctx_.builder().CreateSelect(is_var1,
                llvm::ConstantInt::get(ctx_.int64Type(), 1),
                llvm::ConstantInt::get(ctx_.int64Type(), 0));
            llvm::Value* total_inc = ctx_.builder().CreateAdd(inc1, inc2);
            llvm::Value* new_degree = ctx_.builder().CreateAdd(current_degree, total_inc);
            ctx_.builder().CreateStore(new_degree, gradient_x_degree);
            ctx_.builder().CreateBr(after_degree_bb);

            ctx_.builder().SetInsertPoint(skip_degree_bb);
            ctx_.builder().CreateBr(after_degree_bb);

            ctx_.builder().SetInsertPoint(after_degree_bb);
        }

        accumulateGradient(input1, grad_input1);
        accumulateGradient(input2, grad_input2);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for DIV
    ctx_.builder().SetInsertPoint(check_div);
    llvm::Value* is_div = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 5));
    llvm::BasicBlock* check_sin = llvm::BasicBlock::Create(ctx_.context(), "check_sin", current_func);
    ctx_.builder().CreateCondBr(is_div, div_block, check_sin);

    // DIV: dL/dx = dL/dz / y, dL/dy = dL/dz * (-x/y²)
    ctx_.builder().SetInsertPoint(div_block);
    if (input1 && input2) {
        llvm::Value* input1_val = loadNodeValue(input1);
        llvm::Value* input2_val = loadNodeValue(input2);

        llvm::Value* grad_input1 = ctx_.builder().CreateFDiv(node_grad, input2_val);

        llvm::Value* y_squared = ctx_.builder().CreateFMul(input2_val, input2_val);
        llvm::Value* neg_x_over_y2 = ctx_.builder().CreateFDiv(ctx_.builder().CreateFNeg(input1_val), y_squared);
        llvm::Value* grad_input2 = ctx_.builder().CreateFMul(node_grad, neg_x_over_y2);

        accumulateGradient(input1, grad_input1);
        accumulateGradient(input2, grad_input2);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SIN
    ctx_.builder().SetInsertPoint(check_sin);
    llvm::Value* is_sin = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 6));
    llvm::BasicBlock* check_cos = llvm::BasicBlock::Create(ctx_.context(), "check_cos", current_func);
    ctx_.builder().CreateCondBr(is_sin, sin_block, check_cos);

    // SIN: dL/dx = dL/dz * cos(x)
    ctx_.builder().SetInsertPoint(sin_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* cos_func = getMathFunc("cos");
        if (cos_func) {
            llvm::Value* cos_val = ctx_.builder().CreateCall(cos_func, {input_val});
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, cos_val);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Check for COS
    ctx_.builder().SetInsertPoint(check_cos);
    llvm::Value* is_cos = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 7));
    llvm::BasicBlock* check_exp = llvm::BasicBlock::Create(ctx_.context(), "check_exp", current_func);
    ctx_.builder().CreateCondBr(is_cos, cos_block, check_exp);

    // COS: dL/dx = dL/dz * (-sin(x))
    ctx_.builder().SetInsertPoint(cos_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* sin_func = getMathFunc("sin");
        if (sin_func) {
            llvm::Value* sin_val = ctx_.builder().CreateCall(sin_func, {input_val});
            llvm::Value* neg_sin = ctx_.builder().CreateFNeg(sin_val);
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, neg_sin);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Create blocks for additional operations
    llvm::BasicBlock* exp_block = llvm::BasicBlock::Create(ctx_.context(), "grad_exp", current_func);
    llvm::BasicBlock* log_block = llvm::BasicBlock::Create(ctx_.context(), "grad_log", current_func);
    llvm::BasicBlock* pow_block = llvm::BasicBlock::Create(ctx_.context(), "grad_pow", current_func);
    llvm::BasicBlock* neg_block = llvm::BasicBlock::Create(ctx_.context(), "grad_neg", current_func);
    llvm::BasicBlock* relu_block = llvm::BasicBlock::Create(ctx_.context(), "grad_relu", current_func);
    llvm::BasicBlock* sigmoid_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sigmoid", current_func);
    llvm::BasicBlock* softmax_block = llvm::BasicBlock::Create(ctx_.context(), "grad_softmax", current_func);
    llvm::BasicBlock* tanh_block = llvm::BasicBlock::Create(ctx_.context(), "grad_tanh", current_func);
    llvm::BasicBlock* gelu_block = llvm::BasicBlock::Create(ctx_.context(), "grad_gelu", current_func);
    llvm::BasicBlock* leaky_relu_block = llvm::BasicBlock::Create(ctx_.context(), "grad_leaky_relu", current_func);
    llvm::BasicBlock* silu_block = llvm::BasicBlock::Create(ctx_.context(), "grad_silu", current_func);
    llvm::BasicBlock* matmul_block = llvm::BasicBlock::Create(ctx_.context(), "grad_matmul", current_func);
    llvm::BasicBlock* sum_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sum", current_func);
    llvm::BasicBlock* mean_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mean", current_func);
    llvm::BasicBlock* sqrt_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sqrt", current_func);
    llvm::BasicBlock* abs_block = llvm::BasicBlock::Create(ctx_.context(), "grad_abs", current_func);
    llvm::BasicBlock* square_block = llvm::BasicBlock::Create(ctx_.context(), "grad_square", current_func);
    llvm::BasicBlock* max_block = llvm::BasicBlock::Create(ctx_.context(), "grad_max", current_func);
    llvm::BasicBlock* min_block = llvm::BasicBlock::Create(ctx_.context(), "grad_min", current_func);
    llvm::BasicBlock* elu_block = llvm::BasicBlock::Create(ctx_.context(), "grad_elu", current_func);
    llvm::BasicBlock* selu_block = llvm::BasicBlock::Create(ctx_.context(), "grad_selu", current_func);
    llvm::BasicBlock* mish_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mish", current_func);
    llvm::BasicBlock* hardswish_block = llvm::BasicBlock::Create(ctx_.context(), "grad_hardswish", current_func);
    llvm::BasicBlock* hardsigmoid_block = llvm::BasicBlock::Create(ctx_.context(), "grad_hardsigmoid", current_func);
    llvm::BasicBlock* softplus_block = llvm::BasicBlock::Create(ctx_.context(), "grad_softplus", current_func);
    llvm::BasicBlock* celu_block = llvm::BasicBlock::Create(ctx_.context(), "grad_celu", current_func);

    // Check for EXP (type=8)
    ctx_.builder().SetInsertPoint(check_exp);
    llvm::Value* is_exp = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 8));
    llvm::BasicBlock* check_log = llvm::BasicBlock::Create(ctx_.context(), "check_log", current_func);
    ctx_.builder().CreateCondBr(is_exp, exp_block, check_log);

    // EXP: dL/dx = dL/dz * exp(x)
    ctx_.builder().SetInsertPoint(exp_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* exp_func = getMathFunc("exp");
        if (exp_func) {
            llvm::Value* exp_val = ctx_.builder().CreateCall(exp_func, {input_val});
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, exp_val);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Check for LOG (type=9)
    ctx_.builder().SetInsertPoint(check_log);
    llvm::Value* is_log = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 9));
    llvm::BasicBlock* check_pow = llvm::BasicBlock::Create(ctx_.context(), "check_pow", current_func);
    ctx_.builder().CreateCondBr(is_log, log_block, check_pow);

    // LOG: dL/dx = dL/dz / x
    ctx_.builder().SetInsertPoint(log_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, input_val);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for POW (type=10)
    ctx_.builder().SetInsertPoint(check_pow);
    llvm::Value* is_pow = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 10));
    llvm::BasicBlock* check_neg = llvm::BasicBlock::Create(ctx_.context(), "check_neg", current_func);
    ctx_.builder().CreateCondBr(is_pow, pow_block, check_neg);

    // POW: dL/dx = dL/dz * y * x^(y-1), dL/dy = dL/dz * x^y * ln(x)
    ctx_.builder().SetInsertPoint(pow_block);
    if (input1 && input2) {
        llvm::Value* base_val = loadNodeValue(input1);
        llvm::Value* exp_val = loadNodeValue(input2);

        llvm::Function* pow_func = getMathFunc("pow");
        llvm::Function* log_func = getMathFunc("log");

        if (pow_func && log_func) {
            // Gradient w.r.t. base: y * x^(y-1) = y * x^y / x
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* exp_minus_1 = ctx_.builder().CreateFSub(exp_val, one);
            llvm::Value* pow_val = ctx_.builder().CreateCall(pow_func, {base_val, exp_minus_1});
            llvm::Value* base_deriv = ctx_.builder().CreateFMul(exp_val, pow_val);
            llvm::Value* grad_base = ctx_.builder().CreateFMul(node_grad, base_deriv);
            accumulateGradient(input1, grad_base);

            // Gradient w.r.t. exponent: x^y * ln(x)
            llvm::Value* pow_full = ctx_.builder().CreateCall(pow_func, {base_val, exp_val});
            llvm::Value* log_base = ctx_.builder().CreateCall(log_func, {base_val});
            llvm::Value* exp_deriv = ctx_.builder().CreateFMul(pow_full, log_base);
            llvm::Value* grad_exp = ctx_.builder().CreateFMul(node_grad, exp_deriv);
            accumulateGradient(input2, grad_exp);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Check for NEG (type=11)
    ctx_.builder().SetInsertPoint(check_neg);
    llvm::Value* is_neg = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 11));
    llvm::BasicBlock* check_relu = llvm::BasicBlock::Create(ctx_.context(), "check_relu", current_func);
    ctx_.builder().CreateCondBr(is_neg, neg_block, check_relu);

    // NEG: dL/dx = -dL/dz
    ctx_.builder().SetInsertPoint(neg_block);
    if (input1) {
        llvm::Value* neg_grad = ctx_.builder().CreateFNeg(node_grad);
        accumulateGradient(input1, neg_grad);
    }
    ctx_.builder().CreateBr(done_block);

    // === ML ACTIVATION GRADIENTS ===

    // Check for RELU (type=12)
    ctx_.builder().SetInsertPoint(check_relu);
    llvm::Value* is_relu = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 12));
    llvm::BasicBlock* check_sigmoid = llvm::BasicBlock::Create(ctx_.context(), "check_sigmoid", current_func);
    ctx_.builder().CreateCondBr(is_relu, relu_block, check_sigmoid);

    // RELU: dL/dx = dL/dz * (x > 0 ? 1 : 0)
    ctx_.builder().SetInsertPoint(relu_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(input_val, zero);
        llvm::Value* local_grad = ctx_.builder().CreateSelect(is_positive, one, zero);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, local_grad);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SIGMOID (type=13)
    ctx_.builder().SetInsertPoint(check_sigmoid);
    llvm::Value* is_sigmoid = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 13));
    llvm::BasicBlock* check_softmax = llvm::BasicBlock::Create(ctx_.context(), "check_softmax", current_func);
    ctx_.builder().CreateCondBr(is_sigmoid, sigmoid_block, check_softmax);

    // SIGMOID: dL/dx = dL/dz * σ(x) * (1 - σ(x))
    // Note: We can use the node's output value which is σ(x)
    ctx_.builder().SetInsertPoint(sigmoid_block);
    if (input1) {
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* sigma_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* one_minus_sigma = ctx_.builder().CreateFSub(one, sigma_x);
        llvm::Value* sigma_deriv = ctx_.builder().CreateFMul(sigma_x, one_minus_sigma);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, sigma_deriv);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SOFTMAX (type=14) - Softmax Jacobian: diag(s) - s*s^T
    // Backward: grad_input = s * (grad - dot(grad, s))
    ctx_.builder().SetInsertPoint(check_softmax);
    llvm::Value* is_softmax = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 14));
    llvm::BasicBlock* check_tanh = llvm::BasicBlock::Create(ctx_.context(), "check_tanh", current_func);
    ctx_.builder().CreateCondBr(is_softmax, softmax_block, check_tanh);

    // SOFTMAX backward: grad_input = s * (grad - dot(grad, s))
    // where s = softmax(x) is the forward output stored in the node value.
    // Derivation: Jacobian of softmax is diag(s) - s*s^T,
    // so grad_input_i = sum_j (s_i * delta_ij - s_i * s_j) * grad_j
    //                 = s_i * grad_i - s_i * sum_j(s_j * grad_j)
    //                 = s_i * (grad_i - dot(grad, s))
    ctx_.builder().SetInsertPoint(softmax_block);
    if (input1) {
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* s = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        // dot(grad, s) - for scalar case this is just grad * s
        llvm::Value* dot_grad_s = ctx_.builder().CreateFMul(node_grad, s);
        // grad - dot(grad, s)
        llvm::Value* grad_minus_dot = ctx_.builder().CreateFSub(node_grad, dot_grad_s);
        // s * (grad - dot(grad, s))
        llvm::Value* grad_input = ctx_.builder().CreateFMul(s, grad_minus_dot);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for TANH (type=15)
    ctx_.builder().SetInsertPoint(check_tanh);
    llvm::Value* is_tanh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 15));
    llvm::BasicBlock* check_gelu = llvm::BasicBlock::Create(ctx_.context(), "check_gelu", current_func);
    ctx_.builder().CreateCondBr(is_tanh, tanh_block, check_gelu);

    // TANH: dL/dx = dL/dz * (1 - tanh²(x))
    ctx_.builder().SetInsertPoint(tanh_block);
    if (input1) {
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* tanh_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* tanh_sq = ctx_.builder().CreateFMul(tanh_x, tanh_x);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* tanh_deriv = ctx_.builder().CreateFSub(one, tanh_sq);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, tanh_deriv);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for GELU (type=16)
    ctx_.builder().SetInsertPoint(check_gelu);
    llvm::Value* is_gelu = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 16));
    llvm::BasicBlock* check_leaky_relu = llvm::BasicBlock::Create(ctx_.context(), "check_leaky_relu", current_func);
    ctx_.builder().CreateCondBr(is_gelu, gelu_block, check_leaky_relu);

    // GELU: derivative of the tanh approximation used by the forward path.
    // gelu(x) = 0.5*x*(1+tanh(u)), u = sqrt(2/pi)*(x + 0.044715*x^3)
    // gelu'(x) = 0.5*(1+tanh(u)) + 0.5*x*(1-tanh(u)^2)*u'
    ctx_.builder().SetInsertPoint(gelu_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* exp_func = getMathFunc("exp");
        if (exp_func) {
            llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
            llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
            llvm::Value* sqrt_2_pi = llvm::ConstantFP::get(ctx_.doubleType(), 0.7978845608028654);
            llvm::Value* coeff = llvm::ConstantFP::get(ctx_.doubleType(), 0.044715);

            llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
            llvm::Value* x_cubed = ctx_.builder().CreateFMul(x_sq, input_val);
            llvm::Value* inner = ctx_.builder().CreateFAdd(
                input_val, ctx_.builder().CreateFMul(coeff, x_cubed));
            llvm::Value* u = ctx_.builder().CreateFMul(sqrt_2_pi, inner);
            llvm::Value* exp_2u = ctx_.builder().CreateCall(
                exp_func, {ctx_.builder().CreateFMul(two, u)});
            llvm::Value* tanh_u = ctx_.builder().CreateFDiv(
                ctx_.builder().CreateFSub(exp_2u, one),
                ctx_.builder().CreateFAdd(exp_2u, one));
            llvm::Value* tanh_sq = ctx_.builder().CreateFMul(tanh_u, tanh_u);
            llvm::Value* sech_sq = ctx_.builder().CreateFSub(one, tanh_sq);
            llvm::Value* inner_prime = ctx_.builder().CreateFAdd(
                one, ctx_.builder().CreateFMul(three, ctx_.builder().CreateFMul(coeff, x_sq)));
            llvm::Value* u_prime = ctx_.builder().CreateFMul(sqrt_2_pi, inner_prime);
            llvm::Value* first = ctx_.builder().CreateFMul(
                half, ctx_.builder().CreateFAdd(one, tanh_u));
            llvm::Value* second = ctx_.builder().CreateFMul(
                half, ctx_.builder().CreateFMul(
                    input_val, ctx_.builder().CreateFMul(sech_sq, u_prime)));
            llvm::Value* gelu_deriv = ctx_.builder().CreateFAdd(first, second);
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, gelu_deriv);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Check for LEAKY_RELU (type=17)
    ctx_.builder().SetInsertPoint(check_leaky_relu);
    llvm::Value* is_leaky_relu = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 17));
    llvm::BasicBlock* check_silu = llvm::BasicBlock::Create(ctx_.context(), "check_silu", current_func);
    ctx_.builder().CreateCondBr(is_leaky_relu, leaky_relu_block, check_silu);

    // LEAKY_RELU: dL/dx = dL/dz * (x > 0 ? 1 : α)
    // Default α = 0.01
    ctx_.builder().SetInsertPoint(leaky_relu_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* alpha = llvm::ConstantFP::get(ctx_.doubleType(), 0.01);
        llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(input_val, zero);
        llvm::Value* local_grad = ctx_.builder().CreateSelect(is_positive, one, alpha);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, local_grad);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SILU (type=18)
    ctx_.builder().SetInsertPoint(check_silu);
    llvm::Value* is_silu = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 18));
    llvm::BasicBlock* check_matmul = llvm::BasicBlock::Create(ctx_.context(), "check_matmul", current_func);
    ctx_.builder().CreateCondBr(is_silu, silu_block, check_matmul);

    // SILU (Swish): dL/dx = dL/dz * σ(x) * (1 + x * (1 - σ(x)))
    ctx_.builder().SetInsertPoint(silu_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* exp_func = getMathFunc("exp");
        if (exp_func) {
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* neg_x = ctx_.builder().CreateFNeg(input_val);
            llvm::Value* exp_neg = ctx_.builder().CreateCall(exp_func, {neg_x});
            llvm::Value* denom = ctx_.builder().CreateFAdd(one, exp_neg);
            llvm::Value* sigma = ctx_.builder().CreateFDiv(one, denom);

            llvm::Value* one_minus_sigma = ctx_.builder().CreateFSub(one, sigma);
            llvm::Value* x_times_one_minus = ctx_.builder().CreateFMul(input_val, one_minus_sigma);
            llvm::Value* one_plus_term = ctx_.builder().CreateFAdd(one, x_times_one_minus);
            llvm::Value* silu_deriv = ctx_.builder().CreateFMul(sigma, one_plus_term);
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, silu_deriv);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Check for MATMUL (type=24) - Tensor operation gradients are more complex
    // For now, we'll add a placeholder that can be extended for tensor autodiff
    ctx_.builder().SetInsertPoint(check_matmul);
    llvm::Value* is_matmul = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 24));
    llvm::BasicBlock* check_sum = llvm::BasicBlock::Create(ctx_.context(), "check_sum", current_func);
    ctx_.builder().CreateCondBr(is_matmul, matmul_block, check_sum);

    // MATMUL: For scalar case, acts like MUL
    // Full tensor matmul: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    // Placeholder for tensor autodiff integration
    ctx_.builder().SetInsertPoint(matmul_block);
    if (input1 && input2) {
        // Scalar approximation: treat as multiply
        llvm::Value* input1_val = loadNodeValue(input1);
        llvm::Value* input2_val = loadNodeValue(input2);
        llvm::Value* grad_input1 = ctx_.builder().CreateFMul(node_grad, input2_val);
        llvm::Value* grad_input2 = ctx_.builder().CreateFMul(node_grad, input1_val);
        accumulateGradient(input1, grad_input1);
        accumulateGradient(input2, grad_input2);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SUM (type=27)
    ctx_.builder().SetInsertPoint(check_sum);
    llvm::Value* is_sum = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 27));
    llvm::BasicBlock* check_mean = llvm::BasicBlock::Create(ctx_.context(), "check_mean", current_func);
    ctx_.builder().CreateCondBr(is_sum, sum_block, check_mean);

    // SUM: dL/dx_i = dL/dz for all i (gradient broadcasts to all elements)
    ctx_.builder().SetInsertPoint(sum_block);
    if (input1) {
        // For scalar, gradient passes through unchanged
        accumulateGradient(input1, node_grad);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for MEAN (type=28)
    ctx_.builder().SetInsertPoint(check_mean);
    llvm::Value* is_mean = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 28));
    llvm::BasicBlock* check_sqrt = llvm::BasicBlock::Create(ctx_.context(), "check_sqrt", current_func);
    ctx_.builder().CreateCondBr(is_mean, mean_block, check_sqrt);

    // MEAN: dL/dx_i = dL/dz / n for all i
    // For scalar, gradient passes through unchanged (n=1)
    ctx_.builder().SetInsertPoint(mean_block);
    if (input1) {
        accumulateGradient(input1, node_grad);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SQRT (type=41)
    ctx_.builder().SetInsertPoint(check_sqrt);
    llvm::Value* is_sqrt = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 41));
    llvm::BasicBlock* check_abs = llvm::BasicBlock::Create(ctx_.context(), "check_abs", current_func);
    ctx_.builder().CreateCondBr(is_sqrt, sqrt_block, check_abs);

    // SQRT: dL/dx = dL/dz * 0.5 / sqrt(x)
    // Note: We can use the node's output value which is sqrt(x)
    ctx_.builder().SetInsertPoint(sqrt_block);
    if (input1) {
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* sqrt_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
        llvm::Value* sqrt_deriv = ctx_.builder().CreateFDiv(half, sqrt_x);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, sqrt_deriv);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for ABS (type=42)
    ctx_.builder().SetInsertPoint(check_abs);
    llvm::Value* is_abs = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 42));
    llvm::BasicBlock* check_square = llvm::BasicBlock::Create(ctx_.context(), "check_square", current_func);
    ctx_.builder().CreateCondBr(is_abs, abs_block, check_square);

    // ABS: dL/dx = dL/dz * sign(x)
    // sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
    ctx_.builder().SetInsertPoint(abs_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* pos_one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* neg_one = llvm::ConstantFP::get(ctx_.doubleType(), -1.0);
        llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(input_val, zero);
        llvm::Value* is_negative = ctx_.builder().CreateFCmpOLT(input_val, zero);
        // sign = is_positive ? 1.0 : (is_negative ? -1.0 : 0.0)
        llvm::Value* neg_or_zero = ctx_.builder().CreateSelect(is_negative, neg_one, zero);
        llvm::Value* sign_val = ctx_.builder().CreateSelect(is_positive, pos_one, neg_or_zero);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, sign_val);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SQUARE (type=43)
    ctx_.builder().SetInsertPoint(check_square);
    llvm::Value* is_square = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 43));
    llvm::BasicBlock* check_max = llvm::BasicBlock::Create(ctx_.context(), "check_max", current_func);
    ctx_.builder().CreateCondBr(is_square, square_block, check_max);

    // SQUARE: dL/dx = dL/dz * 2x
    ctx_.builder().SetInsertPoint(square_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
        llvm::Value* two_x = ctx_.builder().CreateFMul(two, input_val);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, two_x);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for MAX (type=44)
    ctx_.builder().SetInsertPoint(check_max);
    llvm::Value* is_max = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 44));
    llvm::BasicBlock* check_min = llvm::BasicBlock::Create(ctx_.context(), "check_min", current_func);
    ctx_.builder().CreateCondBr(is_max, max_block, check_min);

    // MAX: dL/dx = dL/dz if x > y, dL/dy = dL/dz if y >= x
    // Gradient goes entirely to the larger input
    ctx_.builder().SetInsertPoint(max_block);
    if (input1 && input2) {
        llvm::Value* input1_val = loadNodeValue(input1);
        llvm::Value* input2_val = loadNodeValue(input2);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(input1_val, input2_val);
        llvm::Value* grad_input1 = ctx_.builder().CreateSelect(cmp, node_grad, zero);
        llvm::Value* grad_input2 = ctx_.builder().CreateSelect(cmp, zero, node_grad);
        accumulateGradient(input1, grad_input1);
        accumulateGradient(input2, grad_input2);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for MIN (type=45)
    ctx_.builder().SetInsertPoint(check_min);
    llvm::Value* is_min = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 45));
    llvm::BasicBlock* check_elu = llvm::BasicBlock::Create(ctx_.context(), "check_elu", current_func);
    ctx_.builder().CreateCondBr(is_min, min_block, check_elu);

    // MIN: dL/dx = dL/dz if x < y, dL/dy = dL/dz if y <= x
    ctx_.builder().SetInsertPoint(min_block);
    if (input1 && input2) {
        llvm::Value* input1_val = loadNodeValue(input1);
        llvm::Value* input2_val = loadNodeValue(input2);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* cmp = ctx_.builder().CreateFCmpOLT(input1_val, input2_val);
        llvm::Value* grad_input1 = ctx_.builder().CreateSelect(cmp, node_grad, zero);
        llvm::Value* grad_input2 = ctx_.builder().CreateSelect(cmp, zero, node_grad);
        accumulateGradient(input1, grad_input1);
        accumulateGradient(input2, grad_input2);
    }
    ctx_.builder().CreateBr(done_block);

    // ===== PHASE 4 ACTIVATION BACKWARD PASSES (types 46-53) =====

    // ELU/CELU default alpha: dL/dx = dL/dz * (x > 0 ? 1 : exp(x))
    auto emit_elu_like_grad = [&](llvm::BasicBlock* block) {
        ctx_.builder().SetInsertPoint(block);
        if (input1) {
            llvm::Value* input_val = loadNodeValue(input1);
            llvm::Function* exp_func = getMathFunc("exp");
            if (exp_func) {
                llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
                llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
                llvm::Value* exp_val = ctx_.builder().CreateCall(exp_func, {input_val});
                llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(input_val, zero);
                llvm::Value* local_grad = ctx_.builder().CreateSelect(is_positive, one, exp_val);
                llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, local_grad);
                accumulateGradient(input1, grad_input);
            }
        }
        ctx_.builder().CreateBr(done_block);
    };

    ctx_.builder().SetInsertPoint(check_elu);
    llvm::Value* is_elu = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 46));
    llvm::BasicBlock* check_selu = llvm::BasicBlock::Create(ctx_.context(), "check_selu", current_func);
    ctx_.builder().CreateCondBr(is_elu, elu_block, check_selu);
    emit_elu_like_grad(elu_block);

    ctx_.builder().SetInsertPoint(check_selu);
    llvm::Value* is_selu = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 47));
    llvm::BasicBlock* check_mish = llvm::BasicBlock::Create(ctx_.context(), "check_mish", current_func);
    ctx_.builder().CreateCondBr(is_selu, selu_block, check_mish);

    ctx_.builder().SetInsertPoint(selu_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* exp_func = getMathFunc("exp");
        if (exp_func) {
            llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
            llvm::Value* selu_lambda = llvm::ConstantFP::get(ctx_.doubleType(), 1.0507009873554804934193349852946);
            llvm::Value* selu_alpha = llvm::ConstantFP::get(ctx_.doubleType(), 1.6732632423543772848170429916717);
            llvm::Value* neg_grad = ctx_.builder().CreateFMul(selu_lambda,
                ctx_.builder().CreateFMul(selu_alpha, ctx_.builder().CreateCall(exp_func, {input_val})));
            llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(input_val, zero);
            llvm::Value* local_grad = ctx_.builder().CreateSelect(is_positive, selu_lambda, neg_grad);
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, local_grad);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    ctx_.builder().SetInsertPoint(check_mish);
    llvm::Value* is_mish = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 48));
    llvm::BasicBlock* check_hardswish = llvm::BasicBlock::Create(ctx_.context(), "check_hardswish", current_func);
    ctx_.builder().CreateCondBr(is_mish, mish_block, check_hardswish);

    ctx_.builder().SetInsertPoint(mish_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* exp_func = getMathFunc("exp");
        llvm::Function* log_func = getMathFunc("log");
        llvm::Function* tanh_func = getMathFunc("tanh");
        if (exp_func && log_func && tanh_func) {
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* neg_x = ctx_.builder().CreateFNeg(input_val);
            llvm::Value* exp_neg = ctx_.builder().CreateCall(exp_func, {neg_x});
            llvm::Value* sigma = ctx_.builder().CreateFDiv(one,
                ctx_.builder().CreateFAdd(one, exp_neg));
            llvm::Value* exp_x = ctx_.builder().CreateCall(exp_func, {input_val});
            llvm::Value* softplus = ctx_.builder().CreateCall(log_func,
                {ctx_.builder().CreateFAdd(one, exp_x)});
            llvm::Value* tanh_sp = ctx_.builder().CreateCall(tanh_func, {softplus});
            llvm::Value* tanh_sq = ctx_.builder().CreateFMul(tanh_sp, tanh_sp);
            llvm::Value* sech_sq = ctx_.builder().CreateFSub(one, tanh_sq);
            llvm::Value* x_sigma_sech = ctx_.builder().CreateFMul(input_val,
                ctx_.builder().CreateFMul(sigma, sech_sq));
            llvm::Value* mish_deriv = ctx_.builder().CreateFAdd(tanh_sp, x_sigma_sech);
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, mish_deriv);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    ctx_.builder().SetInsertPoint(check_hardswish);
    llvm::Value* is_hardswish = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 49));
    llvm::BasicBlock* check_hardsigmoid = llvm::BasicBlock::Create(ctx_.context(), "check_hardsigmoid", current_func);
    ctx_.builder().CreateCondBr(is_hardswish, hardswish_block, check_hardsigmoid);

    ctx_.builder().SetInsertPoint(hardswish_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* neg_three = llvm::ConstantFP::get(ctx_.doubleType(), -3.0);
        llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
        llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
        llvm::Value* third = llvm::ConstantFP::get(ctx_.doubleType(), 1.0 / 3.0);
        llvm::Value* middle_grad = ctx_.builder().CreateFAdd(half,
            ctx_.builder().CreateFMul(input_val, third));
        llvm::Value* above = ctx_.builder().CreateFCmpOGE(input_val, three);
        llvm::Value* below = ctx_.builder().CreateFCmpOLE(input_val, neg_three);
        llvm::Value* not_above = ctx_.builder().CreateSelect(above, one, middle_grad);
        llvm::Value* local_grad = ctx_.builder().CreateSelect(below, zero, not_above);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, local_grad);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    ctx_.builder().SetInsertPoint(check_hardsigmoid);
    llvm::Value* is_hardsigmoid = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 50));
    llvm::BasicBlock* check_softplus = llvm::BasicBlock::Create(ctx_.context(), "check_softplus", current_func);
    ctx_.builder().CreateCondBr(is_hardsigmoid, hardsigmoid_block, check_softplus);

    ctx_.builder().SetInsertPoint(hardsigmoid_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* neg_three = llvm::ConstantFP::get(ctx_.doubleType(), -3.0);
        llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
        llvm::Value* sixth = llvm::ConstantFP::get(ctx_.doubleType(), 1.0 / 6.0);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* above_low = ctx_.builder().CreateFCmpOGT(input_val, neg_three);
        llvm::Value* below_high = ctx_.builder().CreateFCmpOLT(input_val, three);
        llvm::Value* in_linear = ctx_.builder().CreateAnd(above_low, below_high);
        llvm::Value* local_grad = ctx_.builder().CreateSelect(in_linear, sixth, zero);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, local_grad);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    ctx_.builder().SetInsertPoint(check_softplus);
    llvm::Value* is_softplus = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 51));
    llvm::BasicBlock* check_celu = llvm::BasicBlock::Create(ctx_.context(), "check_celu", current_func);
    ctx_.builder().CreateCondBr(is_softplus, softplus_block, check_celu);

    ctx_.builder().SetInsertPoint(softplus_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* exp_func = getMathFunc("exp");
        if (exp_func) {
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* exp_neg = ctx_.builder().CreateCall(exp_func, {ctx_.builder().CreateFNeg(input_val)});
            llvm::Value* sigma = ctx_.builder().CreateFDiv(one,
                ctx_.builder().CreateFAdd(one, exp_neg));
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, sigma);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    ctx_.builder().SetInsertPoint(check_celu);
    llvm::Value* is_celu = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 53));
    llvm::BasicBlock* check_tan = llvm::BasicBlock::Create(ctx_.context(), "check_tan", current_func);
    ctx_.builder().CreateCondBr(is_celu, celu_block, check_tan);
    emit_elu_like_grad(celu_block);

    // ===== COMPLETE MATH FUNCTION BACKWARD PASSES (types 54-66) =====
    // All standard math functions with proper derivative computation

    // Create blocks for all math function backward passes
    llvm::BasicBlock* tan_block = llvm::BasicBlock::Create(ctx_.context(), "grad_tan", current_func);
    llvm::BasicBlock* asin_block2 = llvm::BasicBlock::Create(ctx_.context(), "grad_asin", current_func);
    llvm::BasicBlock* acos_block2 = llvm::BasicBlock::Create(ctx_.context(), "grad_acos", current_func);
    llvm::BasicBlock* atan_block = llvm::BasicBlock::Create(ctx_.context(), "grad_atan", current_func);
    llvm::BasicBlock* sinh_block2 = llvm::BasicBlock::Create(ctx_.context(), "grad_sinh", current_func);
    llvm::BasicBlock* cosh_block2 = llvm::BasicBlock::Create(ctx_.context(), "grad_cosh", current_func);
    llvm::BasicBlock* asinh_block = llvm::BasicBlock::Create(ctx_.context(), "grad_asinh", current_func);
    llvm::BasicBlock* acosh_block = llvm::BasicBlock::Create(ctx_.context(), "grad_acosh", current_func);
    llvm::BasicBlock* atanh_block = llvm::BasicBlock::Create(ctx_.context(), "grad_atanh", current_func);
    llvm::BasicBlock* log10_block = llvm::BasicBlock::Create(ctx_.context(), "grad_log10", current_func);
    llvm::BasicBlock* log2_block = llvm::BasicBlock::Create(ctx_.context(), "grad_log2", current_func);
    llvm::BasicBlock* exp2_block = llvm::BasicBlock::Create(ctx_.context(), "grad_exp2", current_func);
    llvm::BasicBlock* cbrt_block = llvm::BasicBlock::Create(ctx_.context(), "grad_cbrt", current_func);
    llvm::BasicBlock* atan2_block = llvm::BasicBlock::Create(ctx_.context(), "grad_atan2", current_func);

    // --- TAN (type=54): dL/dx = dL/dz * (1 + tan²(x)) = dL/dz / cos²(x) ---
    ctx_.builder().SetInsertPoint(check_tan);
    llvm::Value* is_tan = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 54));
    llvm::BasicBlock* check_asin = llvm::BasicBlock::Create(ctx_.context(), "check_asin", current_func);
    ctx_.builder().CreateCondBr(is_tan, tan_block, check_asin);

    ctx_.builder().SetInsertPoint(tan_block);
    if (input1) {
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* tan_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* tan_sq = ctx_.builder().CreateFMul(tan_x, tan_x);
        llvm::Value* sec_sq = ctx_.builder().CreateFAdd(one, tan_sq);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, sec_sq);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- ASIN (type=55): dL/dx = dL/dz / sqrt(1 - x²) ---
    ctx_.builder().SetInsertPoint(check_asin);
    llvm::Value* is_asin = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 55));
    llvm::BasicBlock* check_acos = llvm::BasicBlock::Create(ctx_.context(), "check_acos", current_func);
    ctx_.builder().CreateCondBr(is_asin, asin_block2, check_acos);

    ctx_.builder().SetInsertPoint(asin_block2);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* under = ctx_.builder().CreateFSub(one, x_sq);
        llvm::Function* sqrt_func = getMathFunc("sqrt");
        if (sqrt_func) {
            llvm::Value* sqrt_under = ctx_.builder().CreateCall(sqrt_func, {under});
            llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, sqrt_under);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- ACOS (type=56): dL/dx = -dL/dz / sqrt(1 - x²) ---
    ctx_.builder().SetInsertPoint(check_acos);
    llvm::Value* is_acos = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 56));
    llvm::BasicBlock* check_atan = llvm::BasicBlock::Create(ctx_.context(), "check_atan", current_func);
    ctx_.builder().CreateCondBr(is_acos, acos_block2, check_atan);

    ctx_.builder().SetInsertPoint(acos_block2);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* under = ctx_.builder().CreateFSub(one, x_sq);
        llvm::Function* sqrt_func = getMathFunc("sqrt");
        if (sqrt_func) {
            llvm::Value* sqrt_under = ctx_.builder().CreateCall(sqrt_func, {under});
            llvm::Value* neg_grad = ctx_.builder().CreateFNeg(node_grad);
            llvm::Value* grad_input = ctx_.builder().CreateFDiv(neg_grad, sqrt_under);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- ATAN (type=57): dL/dx = dL/dz / (1 + x²) ---
    ctx_.builder().SetInsertPoint(check_atan);
    llvm::Value* is_atan = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 57));
    llvm::BasicBlock* check_sinh = llvm::BasicBlock::Create(ctx_.context(), "check_sinh", current_func);
    ctx_.builder().CreateCondBr(is_atan, atan_block, check_sinh);

    ctx_.builder().SetInsertPoint(atan_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* denom = ctx_.builder().CreateFAdd(one, x_sq);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, denom);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- SINH (type=58): dL/dx = dL/dz * cosh(x) ---
    ctx_.builder().SetInsertPoint(check_sinh);
    llvm::Value* is_sinh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 58));
    llvm::BasicBlock* check_cosh = llvm::BasicBlock::Create(ctx_.context(), "check_cosh", current_func);
    ctx_.builder().CreateCondBr(is_sinh, sinh_block2, check_cosh);

    ctx_.builder().SetInsertPoint(sinh_block2);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* cosh_func = getMathFunc("cosh");
        if (cosh_func) {
            llvm::Value* cosh_val = ctx_.builder().CreateCall(cosh_func, {input_val});
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, cosh_val);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- COSH (type=59): dL/dx = dL/dz * sinh(x) ---
    ctx_.builder().SetInsertPoint(check_cosh);
    llvm::Value* is_cosh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 59));
    llvm::BasicBlock* check_asinh = llvm::BasicBlock::Create(ctx_.context(), "check_asinh", current_func);
    ctx_.builder().CreateCondBr(is_cosh, cosh_block2, check_asinh);

    ctx_.builder().SetInsertPoint(cosh_block2);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* sinh_func = getMathFunc("sinh");
        if (sinh_func) {
            llvm::Value* sinh_val = ctx_.builder().CreateCall(sinh_func, {input_val});
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, sinh_val);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- ASINH (type=60): dL/dx = dL/dz / sqrt(1 + x²) ---
    ctx_.builder().SetInsertPoint(check_asinh);
    llvm::Value* is_asinh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 60));
    llvm::BasicBlock* check_acosh = llvm::BasicBlock::Create(ctx_.context(), "check_acosh", current_func);
    ctx_.builder().CreateCondBr(is_asinh, asinh_block, check_acosh);

    ctx_.builder().SetInsertPoint(asinh_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* under = ctx_.builder().CreateFAdd(one, x_sq);
        llvm::Function* sqrt_func = getMathFunc("sqrt");
        if (sqrt_func) {
            llvm::Value* sqrt_under = ctx_.builder().CreateCall(sqrt_func, {under});
            llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, sqrt_under);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- ACOSH (type=61): dL/dx = dL/dz / sqrt(x² - 1) ---
    ctx_.builder().SetInsertPoint(check_acosh);
    llvm::Value* is_acosh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 61));
    llvm::BasicBlock* check_atanh = llvm::BasicBlock::Create(ctx_.context(), "check_atanh", current_func);
    ctx_.builder().CreateCondBr(is_acosh, acosh_block, check_atanh);

    ctx_.builder().SetInsertPoint(acosh_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* under = ctx_.builder().CreateFSub(x_sq, one);
        llvm::Function* sqrt_func = getMathFunc("sqrt");
        if (sqrt_func) {
            llvm::Value* sqrt_under = ctx_.builder().CreateCall(sqrt_func, {under});
            llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, sqrt_under);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- ATANH (type=62): dL/dx = dL/dz / (1 - x²) ---
    ctx_.builder().SetInsertPoint(check_atanh);
    llvm::Value* is_atanh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 62));
    llvm::BasicBlock* check_log10 = llvm::BasicBlock::Create(ctx_.context(), "check_log10", current_func);
    ctx_.builder().CreateCondBr(is_atanh, atanh_block, check_log10);

    ctx_.builder().SetInsertPoint(atanh_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* denom = ctx_.builder().CreateFSub(one, x_sq);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, denom);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- LOG10 (type=63): dL/dx = dL/dz / (x * ln(10)) ---
    ctx_.builder().SetInsertPoint(check_log10);
    llvm::Value* is_log10 = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 63));
    llvm::BasicBlock* check_log2 = llvm::BasicBlock::Create(ctx_.context(), "check_log2", current_func);
    ctx_.builder().CreateCondBr(is_log10, log10_block, check_log2);

    ctx_.builder().SetInsertPoint(log10_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* ln10 = llvm::ConstantFP::get(ctx_.doubleType(), 2.302585092994046);
        llvm::Value* denom = ctx_.builder().CreateFMul(input_val, ln10);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, denom);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- LOG2 (type=64): dL/dx = dL/dz / (x * ln(2)) ---
    ctx_.builder().SetInsertPoint(check_log2);
    llvm::Value* is_log2 = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 64));
    llvm::BasicBlock* check_exp2 = llvm::BasicBlock::Create(ctx_.context(), "check_exp2", current_func);
    ctx_.builder().CreateCondBr(is_log2, log2_block, check_exp2);

    ctx_.builder().SetInsertPoint(log2_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* ln2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.6931471805599453);
        llvm::Value* denom = ctx_.builder().CreateFMul(input_val, ln2);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, denom);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- EXP2 (type=65): dL/dx = dL/dz * 2^x * ln(2) ---
    ctx_.builder().SetInsertPoint(check_exp2);
    llvm::Value* is_exp2 = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 65));
    llvm::BasicBlock* check_cbrt = llvm::BasicBlock::Create(ctx_.context(), "check_cbrt", current_func);
    ctx_.builder().CreateCondBr(is_exp2, exp2_block, check_cbrt);

    ctx_.builder().SetInsertPoint(exp2_block);
    if (input1) {
        // node value = 2^x (stored during forward pass)
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* exp2_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* ln2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.6931471805599453);
        llvm::Value* exp2_times_ln2 = ctx_.builder().CreateFMul(exp2_x, ln2);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, exp2_times_ln2);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- CBRT (type=66): dL/dx = dL/dz / (3 * cbrt(x)²) ---
    ctx_.builder().SetInsertPoint(check_cbrt);
    llvm::Value* is_cbrt = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 66));
    llvm::BasicBlock* check_atan2 = llvm::BasicBlock::Create(ctx_.context(), "check_atan2", current_func);
    ctx_.builder().CreateCondBr(is_cbrt, cbrt_block, check_atan2);

    ctx_.builder().SetInsertPoint(cbrt_block);
    if (input1) {
        // node value = cbrt(x) (stored during forward pass)
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* cbrt_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
        llvm::Value* cbrt_sq = ctx_.builder().CreateFMul(cbrt_x, cbrt_x);
        llvm::Value* denom = ctx_.builder().CreateFMul(three, cbrt_sq);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, denom);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- ATAN2 (binary): atan2(y, x)
    // d/dy = x / (x² + y²), d/dx = -y / (x² + y²)
    ctx_.builder().SetInsertPoint(check_atan2);
    llvm::Value* is_atan2 = ctx_.builder().CreateICmpEQ(
        node_type, llvm::ConstantInt::get(ctx_.int32Type(), AD_NODE_ATAN2));
    llvm::BasicBlock* check_conv2d = llvm::BasicBlock::Create(ctx_.context(), "check_conv2d", current_func);
    ctx_.builder().CreateCondBr(is_atan2, atan2_block, check_conv2d);

    ctx_.builder().SetInsertPoint(atan2_block);
    if (input1 && input2) {
        llvm::Value* y_val = loadNodeValue(input1);
        llvm::Value* x_val = loadNodeValue(input2);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(x_val, x_val);
        llvm::Value* y_sq = ctx_.builder().CreateFMul(y_val, y_val);
        llvm::Value* denom = ctx_.builder().CreateFAdd(x_sq, y_sq);
        llvm::Value* grad_y = ctx_.builder().CreateFMul(
            node_grad, ctx_.builder().CreateFDiv(x_val, denom));
        llvm::Value* neg_y = ctx_.builder().CreateFNeg(y_val);
        llvm::Value* grad_x = ctx_.builder().CreateFMul(
            node_grad, ctx_.builder().CreateFDiv(neg_y, denom));
        accumulateGradient(input1, grad_y);
        accumulateGradient(input2, grad_x);
    }
    ctx_.builder().CreateBr(done_block);

    // ===== TENSOR OPERATION SCALAR FALLBACKS =====
    // These scalar approximations handle tensor op types (19-32) when
    // tensor_gradient is NULL (scalar mode). When tensor_gradient is set,
    // the tensor fast path above dispatches to eshkol_tensor_backward_dispatch()
    // which calls the proper runtime backward functions with full tensor data.

    // Create blocks for all new operation types
    llvm::BasicBlock* conv2d_block = llvm::BasicBlock::Create(ctx_.context(), "grad_conv2d", current_func);
    llvm::BasicBlock* maxpool_block = llvm::BasicBlock::Create(ctx_.context(), "grad_maxpool", current_func);
    llvm::BasicBlock* avgpool_block = llvm::BasicBlock::Create(ctx_.context(), "grad_avgpool", current_func);
    llvm::BasicBlock* batchnorm_block = llvm::BasicBlock::Create(ctx_.context(), "grad_batchnorm", current_func);
    llvm::BasicBlock* layernorm_block = llvm::BasicBlock::Create(ctx_.context(), "grad_layernorm", current_func);
    llvm::BasicBlock* transpose_block = llvm::BasicBlock::Create(ctx_.context(), "grad_transpose", current_func);
    llvm::BasicBlock* reshape_block = llvm::BasicBlock::Create(ctx_.context(), "grad_reshape", current_func);
    llvm::BasicBlock* attention_block = llvm::BasicBlock::Create(ctx_.context(), "grad_attention", current_func);
    llvm::BasicBlock* mha_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mha", current_func);
    llvm::BasicBlock* posenc_block = llvm::BasicBlock::Create(ctx_.context(), "grad_posenc", current_func);
    llvm::BasicBlock* embedding_block = llvm::BasicBlock::Create(ctx_.context(), "grad_embedding", current_func);
    llvm::BasicBlock* hyp_dist_block = llvm::BasicBlock::Create(ctx_.context(), "grad_hyp_dist", current_func);
    llvm::BasicBlock* poincare_exp_block = llvm::BasicBlock::Create(ctx_.context(), "grad_poincare_exp", current_func);
    llvm::BasicBlock* poincare_log_block = llvm::BasicBlock::Create(ctx_.context(), "grad_poincare_log", current_func);
    llvm::BasicBlock* tangent_proj_block = llvm::BasicBlock::Create(ctx_.context(), "grad_tangent_proj", current_func);
    llvm::BasicBlock* geodesic_attn_block = llvm::BasicBlock::Create(ctx_.context(), "grad_geodesic_attn", current_func);
    llvm::BasicBlock* mobius_add_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mobius_add", current_func);
    llvm::BasicBlock* mobius_matmul_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mobius_matmul", current_func);
    llvm::BasicBlock* gyrovector_block = llvm::BasicBlock::Create(ctx_.context(), "grad_gyrovector", current_func);
    llvm::BasicBlock* custom_block = llvm::BasicBlock::Create(ctx_.context(), "grad_custom", current_func);

    // --- CONV2D (type=19): dL/d_input ≈ grad (scalar approx) ---
    ctx_.builder().SetInsertPoint(check_conv2d);
    llvm::Value* is_conv2d = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 19));
    llvm::BasicBlock* check_maxpool = llvm::BasicBlock::Create(ctx_.context(), "check_maxpool", current_func);
    ctx_.builder().CreateCondBr(is_conv2d, conv2d_block, check_maxpool);

    ctx_.builder().SetInsertPoint(conv2d_block);
    // Conv2D backward: dL/d_input = conv_transpose(grad, kernel), dL/d_kernel = conv(input, grad)
    // Scalar approximation: pass gradient through to input (identity in scalar case)
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- MAXPOOL2D (type=20): gradient through max index ---
    ctx_.builder().SetInsertPoint(check_maxpool);
    llvm::Value* is_maxpool = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 20));
    llvm::BasicBlock* check_avgpool = llvm::BasicBlock::Create(ctx_.context(), "check_avgpool", current_func);
    ctx_.builder().CreateCondBr(is_maxpool, maxpool_block, check_avgpool);

    ctx_.builder().SetInsertPoint(maxpool_block);
    // MaxPool backward: gradient flows only through saved max indices
    // Scalar approximation: pass gradient to input (the max value was the input)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- AVGPOOL2D (type=21): gradient divided by pool size ---
    ctx_.builder().SetInsertPoint(check_avgpool);
    llvm::Value* is_avgpool = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 21));
    llvm::BasicBlock* check_batchnorm = llvm::BasicBlock::Create(ctx_.context(), "check_batchnorm", current_func);
    ctx_.builder().CreateCondBr(is_avgpool, avgpool_block, check_batchnorm);

    ctx_.builder().SetInsertPoint(avgpool_block);
    // AvgPool backward: grad / pool_window_size (scalar case: pool_size=1)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- BATCHNORM (type=22): standard 3-gradient backward ---
    ctx_.builder().SetInsertPoint(check_batchnorm);
    llvm::Value* is_batchnorm = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 22));
    llvm::BasicBlock* check_layernorm = llvm::BasicBlock::Create(ctx_.context(), "check_layernorm", current_func);
    ctx_.builder().CreateCondBr(is_batchnorm, batchnorm_block, check_layernorm);

    ctx_.builder().SetInsertPoint(batchnorm_block);
    // BatchNorm backward: dL/d_input = gamma * grad / sqrt(var + eps)
    // dL/d_gamma = grad * (x - mean) / sqrt(var + eps), dL/d_beta = grad
    // Scalar approximation: gamma=1, var=1, eps=1e-5 → grad ≈ grad * 1/sqrt(1+eps) ≈ grad
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad); // gamma/beta params
    ctx_.builder().CreateBr(done_block);

    // --- LAYERNORM (type=23): same structure as batchnorm ---
    ctx_.builder().SetInsertPoint(check_layernorm);
    llvm::Value* is_layernorm = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 23));
    llvm::BasicBlock* check_transpose = llvm::BasicBlock::Create(ctx_.context(), "check_transpose", current_func);
    ctx_.builder().CreateCondBr(is_layernorm, layernorm_block, check_transpose);

    ctx_.builder().SetInsertPoint(layernorm_block);
    // LayerNorm backward: same as BatchNorm but along feature dim
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- TRANSPOSE (type=25): grad = transpose(upstream_grad) ---
    ctx_.builder().SetInsertPoint(check_transpose);
    llvm::Value* is_transpose = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 25));
    llvm::BasicBlock* check_reshape = llvm::BasicBlock::Create(ctx_.context(), "check_reshape", current_func);
    ctx_.builder().CreateCondBr(is_transpose, transpose_block, check_reshape);

    ctx_.builder().SetInsertPoint(transpose_block);
    // Transpose backward: grad_input = transpose(upstream_grad)
    // Scalar case: identity (transpose of scalar is scalar)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- RESHAPE (type=26): grad = reshape(upstream_grad, original_shape) ---
    ctx_.builder().SetInsertPoint(check_reshape);
    llvm::Value* is_reshape = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 26));
    llvm::BasicBlock* check_attention = llvm::BasicBlock::Create(ctx_.context(), "check_attention", current_func);
    ctx_.builder().CreateCondBr(is_reshape, reshape_block, check_attention);

    ctx_.builder().SetInsertPoint(reshape_block);
    // Reshape backward: reshape gradient back to input shape (scalar: passthrough)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- ATTENTION (type=29): dV=attn^T@grad, dQ/dK through softmax backward ---
    ctx_.builder().SetInsertPoint(check_attention);
    llvm::Value* is_attention = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 29));
    llvm::BasicBlock* check_mha = llvm::BasicBlock::Create(ctx_.context(), "check_mha", current_func);
    ctx_.builder().CreateCondBr(is_attention, attention_block, check_mha);

    ctx_.builder().SetInsertPoint(attention_block);
    // Attention backward: dV = attn_weights^T @ grad_output
    // dS = grad_output @ V^T, then through softmax backward, then:
    // dQ = dS_softmax @ K / sqrt(d_k), dK = dS_softmax^T @ Q / sqrt(d_k)
    // Scalar approximation: gradient flows to Q, K, V inputs
    if (input1) accumulateGradient(input1, node_grad); // Q
    if (input2) accumulateGradient(input2, node_grad); // K (V would be input3)
    ctx_.builder().CreateBr(done_block);

    // --- MULTIHEAD_ATTENTION (type=30) ---
    ctx_.builder().SetInsertPoint(check_mha);
    llvm::Value* is_mha = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 30));
    llvm::BasicBlock* check_posenc = llvm::BasicBlock::Create(ctx_.context(), "check_posenc", current_func);
    ctx_.builder().CreateCondBr(is_mha, mha_block, check_posenc);

    ctx_.builder().SetInsertPoint(mha_block);
    // Multi-head attention backward: split across heads, per-head attention backward,
    // then backprop through W_Q, W_K, W_V, W_O projection matrices
    // Scalar approximation: gradient flows through
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- POSITIONAL_ENCODING (type=31): additive constant, gradient passes through ---
    ctx_.builder().SetInsertPoint(check_posenc);
    llvm::Value* is_posenc = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 31));
    llvm::BasicBlock* check_embedding = llvm::BasicBlock::Create(ctx_.context(), "check_embedding", current_func);
    ctx_.builder().CreateCondBr(is_posenc, posenc_block, check_embedding);

    ctx_.builder().SetInsertPoint(posenc_block);
    // Positional encoding is additive: y = x + PE (PE is constant)
    // dL/dx = dL/dy (gradient passes through unchanged)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- EMBEDDING (type=32): scatter-add ---
    ctx_.builder().SetInsertPoint(check_embedding);
    llvm::Value* is_embedding = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 32));
    llvm::BasicBlock* check_hyp_dist = llvm::BasicBlock::Create(ctx_.context(), "check_hyp_dist", current_func);
    ctx_.builder().CreateCondBr(is_embedding, embedding_block, check_hyp_dist);

    ctx_.builder().SetInsertPoint(embedding_block);
    // Embedding backward: weight_grad[indices[i]] += upstream_grad[i]
    // Scalar approximation: gradient flows to the embedding weight input
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // ===== GEOMETRIC/HYPERBOLIC OPERATION BACKWARD PASSES =====
    // All use Poincaré ball model with conformal factor λ_x = 2/(1-||x||²)

    // --- HYPERBOLIC_DISTANCE (type=33) ---
    ctx_.builder().SetInsertPoint(check_hyp_dist);
    llvm::Value* is_hyp_dist = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 33));
    llvm::BasicBlock* check_poincare_exp = llvm::BasicBlock::Create(ctx_.context(), "check_poincare_exp", current_func);
    ctx_.builder().CreateCondBr(is_hyp_dist, hyp_dist_block, check_poincare_exp);

    ctx_.builder().SetInsertPoint(hyp_dist_block);
    {
        // d(x,y) = acosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
        // dL/dx = dL/dd * dd/dx, where dd/dx involves conformal factors
        // Scalar approximation: use Euclidean gradient scaled by conformal factor
        if (input1 && input2) {
            llvm::Value* x_val = loadNodeValue(input1);
            llvm::Value* y_val = loadNodeValue(input2);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
            // Conformal factor λ_x = 2/(1-x²)
            llvm::Value* x_sq = ctx_.builder().CreateFMul(x_val, x_val);
            llvm::Value* y_sq = ctx_.builder().CreateFMul(y_val, y_val);
            llvm::Value* denom_x = ctx_.builder().CreateFSub(one, x_sq);
            llvm::Value* denom_y = ctx_.builder().CreateFSub(one, y_sq);
            // Clamp to avoid division by zero at boundary
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            llvm::Value* safe_dx = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom_x, eps), eps, denom_x);
            llvm::Value* safe_dy = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom_y, eps), eps, denom_y);
            llvm::Value* lambda_x = ctx_.builder().CreateFDiv(two, safe_dx);
            llvm::Value* lambda_y = ctx_.builder().CreateFDiv(two, safe_dy);
            // diff = x - y
            llvm::Value* diff = ctx_.builder().CreateFSub(x_val, y_val);
            // grad_x = grad * lambda_x² * diff / dist_factor
            llvm::Value* lx_sq = ctx_.builder().CreateFMul(lambda_x, lambda_x);
            llvm::Value* grad_x = ctx_.builder().CreateFMul(node_grad, ctx_.builder().CreateFMul(lx_sq, diff));
            // grad_y = -grad * lambda_y² * diff / dist_factor
            llvm::Value* ly_sq = ctx_.builder().CreateFMul(lambda_y, lambda_y);
            llvm::Value* neg_diff = ctx_.builder().CreateFNeg(diff);
            llvm::Value* grad_y = ctx_.builder().CreateFMul(node_grad, ctx_.builder().CreateFMul(ly_sq, neg_diff));
            accumulateGradient(input1, grad_x);
            accumulateGradient(input2, grad_y);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- POINCARE_EXP_MAP (type=34) ---
    ctx_.builder().SetInsertPoint(check_poincare_exp);
    llvm::Value* is_poincare_exp = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 34));
    llvm::BasicBlock* check_poincare_log = llvm::BasicBlock::Create(ctx_.context(), "check_poincare_log", current_func);
    ctx_.builder().CreateCondBr(is_poincare_exp, poincare_exp_block, check_poincare_log);

    ctx_.builder().SetInsertPoint(poincare_exp_block);
    {
        // exp_p(v) = p ⊕ tanh(λ_p * ||v|| / 2) * v / ||v||
        // Scalar approximation: gradient scaled by conformal factor
        if (input1) {
            llvm::Value* p_val = loadNodeValue(input1);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
            llvm::Value* p_sq = ctx_.builder().CreateFMul(p_val, p_val);
            llvm::Value* denom = ctx_.builder().CreateFSub(one, p_sq);
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            llvm::Value* safe_denom = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom, eps), eps, denom);
            llvm::Value* lambda_p = ctx_.builder().CreateFDiv(two, safe_denom);
            llvm::Value* grad_p = ctx_.builder().CreateFMul(node_grad, lambda_p);
            accumulateGradient(input1, grad_p);
        }
        if (input2) {
            // Tangent vector gradient: scaled by 1/λ_p
            accumulateGradient(input2, node_grad);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- POINCARE_LOG_MAP (type=35) ---
    ctx_.builder().SetInsertPoint(check_poincare_log);
    llvm::Value* is_poincare_log = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 35));
    llvm::BasicBlock* check_tangent = llvm::BasicBlock::Create(ctx_.context(), "check_tangent", current_func);
    ctx_.builder().CreateCondBr(is_poincare_log, poincare_log_block, check_tangent);

    ctx_.builder().SetInsertPoint(poincare_log_block);
    {
        // log_p(q) = (2/λ_p) * atanh(||(-p)⊕q||) * ((-p)⊕q) / ||(-p)⊕q||
        // Scalar approximation: inverse of exp map, gradient scaled by 1/λ_p
        if (input1) {
            llvm::Value* p_val = loadNodeValue(input1);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
            llvm::Value* p_sq = ctx_.builder().CreateFMul(p_val, p_val);
            llvm::Value* denom = ctx_.builder().CreateFSub(one, p_sq);
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            llvm::Value* safe_denom = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom, eps), eps, denom);
            // 1/λ_p = (1-||p||²)/2
            llvm::Value* inv_lambda = ctx_.builder().CreateFDiv(safe_denom, two);
            llvm::Value* grad_p = ctx_.builder().CreateFMul(node_grad, inv_lambda);
            accumulateGradient(input1, grad_p);
        }
        if (input2) accumulateGradient(input2, node_grad);
    }
    ctx_.builder().CreateBr(done_block);

    // --- TANGENT_PROJECT (type=36) ---
    ctx_.builder().SetInsertPoint(check_tangent);
    llvm::Value* is_tangent = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 36));
    llvm::BasicBlock* check_geodesic = llvm::BasicBlock::Create(ctx_.context(), "check_geodesic", current_func);
    ctx_.builder().CreateCondBr(is_tangent, tangent_proj_block, check_geodesic);

    ctx_.builder().SetInsertPoint(tangent_proj_block);
    // Tangent space projection: projects vector onto tangent plane
    // Gradient passes through (projection is linear)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- GEODESIC_ATTENTION (type=37) ---
    ctx_.builder().SetInsertPoint(check_geodesic);
    llvm::Value* is_geodesic = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 37));
    llvm::BasicBlock* check_mobius_add = llvm::BasicBlock::Create(ctx_.context(), "check_mobius_add", current_func);
    ctx_.builder().CreateCondBr(is_geodesic, geodesic_attn_block, check_mobius_add);

    ctx_.builder().SetInsertPoint(geodesic_attn_block);
    // Geodesic attention: attention in hyperbolic space using geodesic distances
    // Gradient flows to Q, K inputs
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- MOBIUS_ADD (type=38) ---
    ctx_.builder().SetInsertPoint(check_mobius_add);
    llvm::Value* is_mobius_add = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 38));
    llvm::BasicBlock* check_mobius_matmul = llvm::BasicBlock::Create(ctx_.context(), "check_mobius_matmul", current_func);
    ctx_.builder().CreateCondBr(is_mobius_add, mobius_add_block, check_mobius_matmul);

    ctx_.builder().SetInsertPoint(mobius_add_block);
    {
        // Möbius addition: x ⊕ y = ((1+2<x,y>+||y||²)x + (1-||x||²)y) / (1+2<x,y>+||x||²||y||²)
        // Scalar 1D case: simplified derivative involves conformal factors
        if (input1 && input2) {
            llvm::Value* x_val = loadNodeValue(input1);
            llvm::Value* y_val = loadNodeValue(input2);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* x_sq = ctx_.builder().CreateFMul(x_val, x_val);
            llvm::Value* y_sq = ctx_.builder().CreateFMul(y_val, y_val);
            // d(x⊕y)/dx at scalar level: (1+y²)/(1+2xy+x²y²)² * (1+2xy+y²-x²(1+2xy+y²)+...)
            // Simplified: gradient ~ (1-||y||²)/(1+2xy+||x||²||y||²)² * (denominator terms)
            // Use conformal scaling: λ_{x⊕y}/λ_x for grad_x, λ_{x⊕y}/λ_y for grad_y
            llvm::Value* xy = ctx_.builder().CreateFMul(x_val, y_val);
            llvm::Value* two_xy = ctx_.builder().CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), xy);
            llvm::Value* xsq_ysq = ctx_.builder().CreateFMul(x_sq, y_sq);
            llvm::Value* denom = ctx_.builder().CreateFAdd(one, ctx_.builder().CreateFAdd(two_xy, xsq_ysq));
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            llvm::Value* safe_denom = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom, eps), eps, denom);
            llvm::Value* denom_sq = ctx_.builder().CreateFMul(safe_denom, safe_denom);
            llvm::Value* inv_denom_sq = ctx_.builder().CreateFDiv(one, denom_sq);
            // dx: (1 + 2xy + y²) * safe_denom - that simplifies, use gyration-based formula
            llvm::Value* factor_x = ctx_.builder().CreateFDiv(
                ctx_.builder().CreateFSub(one, y_sq), safe_denom);
            llvm::Value* factor_y = ctx_.builder().CreateFDiv(
                ctx_.builder().CreateFSub(one, x_sq), safe_denom);
            accumulateGradient(input1, ctx_.builder().CreateFMul(node_grad, factor_x));
            accumulateGradient(input2, ctx_.builder().CreateFMul(node_grad, factor_y));
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- MOBIUS_MATMUL (type=39) ---
    ctx_.builder().SetInsertPoint(check_mobius_matmul);
    llvm::Value* is_mobius_matmul = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 39));
    llvm::BasicBlock* check_gyrovector = llvm::BasicBlock::Create(ctx_.context(), "check_gyrovector", current_func);
    ctx_.builder().CreateCondBr(is_mobius_matmul, mobius_matmul_block, check_gyrovector);

    ctx_.builder().SetInsertPoint(mobius_matmul_block);
    {
        // Möbius matrix multiplication: M ⊗ x = exp_0(M * log_0(x))
        // Gradient: dL/dM = dL/d(M⊗x) * (log_0(x))^T, dL/dx via chain rule through exp/log
        // Scalar approximation: gradient scaled by conformal factor
        if (input1) accumulateGradient(input1, node_grad); // M
        if (input2) {
            llvm::Value* x_val = loadNodeValue(input2);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
            llvm::Value* x_sq = ctx_.builder().CreateFMul(x_val, x_val);
            llvm::Value* denom = ctx_.builder().CreateFSub(one, x_sq);
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            llvm::Value* safe_denom = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom, eps), eps, denom);
            llvm::Value* lambda_x = ctx_.builder().CreateFDiv(two, safe_denom);
            llvm::Value* grad_x = ctx_.builder().CreateFMul(node_grad, lambda_x);
            accumulateGradient(input2, grad_x);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- GYROVECTOR_SPACE (type=40) ---
    ctx_.builder().SetInsertPoint(check_gyrovector);
    llvm::Value* is_gyrovector = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 40));
    llvm::BasicBlock* check_custom = llvm::BasicBlock::Create(ctx_.context(), "check_custom", current_func);
    ctx_.builder().CreateCondBr(is_gyrovector, gyrovector_block, check_custom);

    // --- CUSTOM (external vector-Jacobian product) ---
    // AD_NODE_CUSTOM is intentionally compared through the C enum, never a
    // stale literal: it is appended after AD_NODE_ATAN2 and may move again.
    ctx_.builder().SetInsertPoint(check_custom);
    llvm::Value* is_custom = ctx_.builder().CreateICmpEQ(
        node_type,
        llvm::ConstantInt::get(ctx_.int32Type(), static_cast<int>(AD_NODE_CUSTOM)));
    llvm::BasicBlock* unknown_type_block = llvm::BasicBlock::Create(ctx_.context(), "grad_unknown_type", current_func);
    ctx_.builder().CreateCondBr(is_custom, custom_block, unknown_type_block);

    ctx_.builder().SetInsertPoint(gyrovector_block);
    {
        // Gyrovector space operation: general operation in the Poincaré ball
        // Gradient uses conformal factor scaling
        if (input1 && input2) {
            llvm::Value* x_val = loadNodeValue(input1);
            llvm::Value* y_val = loadNodeValue(input2);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            // λ_x = 2/(1-||x||²)
            llvm::Value* x_sq = ctx_.builder().CreateFMul(x_val, x_val);
            llvm::Value* y_sq = ctx_.builder().CreateFMul(y_val, y_val);
            llvm::Value* dx = ctx_.builder().CreateFSub(one, x_sq);
            llvm::Value* dy = ctx_.builder().CreateFSub(one, y_sq);
            llvm::Value* safe_dx = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(dx, eps), eps, dx);
            llvm::Value* safe_dy = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(dy, eps), eps, dy);
            llvm::Value* lambda_x = ctx_.builder().CreateFDiv(two, safe_dx);
            llvm::Value* lambda_y = ctx_.builder().CreateFDiv(two, safe_dy);
            accumulateGradient(input1, ctx_.builder().CreateFMul(node_grad, lambda_x));
            accumulateGradient(input2, ctx_.builder().CreateFMul(node_grad, lambda_y));
        }
    }
    ctx_.builder().CreateBr(done_block);

    ctx_.builder().SetInsertPoint(custom_block);
    {
        llvm::FunctionType* custom_backward_type = llvm::FunctionType::get(
            ctx_.voidType(), {ctx_.ptrType()}, false);
        llvm::FunctionCallee custom_backward = ctx_.module().getOrInsertFunction(
            "eshkol_ad_node_custom_backward", custom_backward_type);
        ctx_.builder().CreateCall(custom_backward, {node_ptr});
    }
    ctx_.builder().CreateBr(done_block);

    // Unknown type: emit runtime error and abort for unhandled AD_NODE types
    // This prevents silently producing zero gradients for unrecognized operations
    ctx_.builder().SetInsertPoint(unknown_type_block);
    {
        // Print diagnostic to stderr
        llvm::FunctionType* fprintf_type = llvm::FunctionType::get(
            ctx_.int32Type(), {ctx_.ptrType(), ctx_.ptrType()}, true);
        llvm::FunctionCallee fprintf_fn = ctx_.module().getOrInsertFunction("fprintf", fprintf_type);

        // Get stderr via platform-safe runtime wrapper on Windows.
#ifdef _WIN32
        llvm::FunctionType* stream_type = llvm::FunctionType::get(ctx_.ptrType(), {}, false);
        llvm::FunctionCallee stderr_fn = ctx_.module().getOrInsertFunction(
            runtime::stderr_stream_symbol, stream_type);
        llvm::Value* stderr_val = ctx_.builder().CreateCall(stderr_fn, {});
#else
        // Get stderr via platform-appropriate global name
#ifdef __APPLE__
        const char* stderr_name = "__stderrp";
#else
        const char* stderr_name = "stderr";
#endif
        llvm::GlobalVariable* stderr_var = ctx_.module().getGlobalVariable(stderr_name);
        if (!stderr_var) {
            stderr_var = new llvm::GlobalVariable(ctx_.module(), ctx_.ptrType(), false,
                llvm::GlobalValue::ExternalLinkage, nullptr, stderr_name);
        }
        llvm::Value* stderr_val = ctx_.builder().CreateLoad(ctx_.ptrType(), stderr_var);
#endif

        llvm::Value* fmt_str = ctx_.builder().CreateGlobalString(
            "Error: Unknown AD node type %d in backward pass — cannot compute gradient\n");
        ctx_.builder().CreateCall(fprintf_fn, {stderr_val, fmt_str, node_type});

        // Abort: unknown types must not silently produce zero gradients
        llvm::FunctionType* abort_type = llvm::FunctionType::get(ctx_.voidType(), {}, false);
        llvm::FunctionCallee abort_fn = ctx_.module().getOrInsertFunction("abort", abort_type);
        ctx_.builder().CreateCall(abort_fn, {});
        ctx_.builder().CreateUnreachable();
    }

    // Done
    ctx_.builder().SetInsertPoint(done_block);
}

// ===== TAPE MANAGEMENT (Nested Gradient Support) =====
// These functions enable arbitrary-depth nested gradient computations
// by saving/restoring the tape context on a stack.

/**
 * @brief Push the current AD tape onto the tape stack and activate a new tape (nested gradients).
 *
 * Saves the current tape into stack[depth], increments the depth counter (aborting
 * the program if depth would reach MAX_TAPE_DEPTH), installs new_tape as the
 * current tape, and marks AD mode active. Paired with popTapeContext to support
 * arbitrary-depth nested differentiation.
 *
 * @param new_tape The tape to make current for the inner gradient (no-op if null).
 */
void AutodiffCodegen::pushTapeContext(llvm::Value* new_tape) {
    if (!new_tape) return;

    llvm::GlobalVariable* ad_tape_depth = ctx_.adTapeDepth();
    llvm::GlobalVariable* ad_tape_stack = ctx_.adTapeStack();
    llvm::GlobalVariable* current_ad_tape_var = ctx_.currentAdTape();
    llvm::GlobalVariable* ad_mode_active = ctx_.adModeActive();

    if (!ad_tape_depth || !ad_tape_stack || !current_ad_tape_var || !ad_mode_active) {
        eshkol_warn("pushTapeContext: AD globals not initialized");
        return;
    }

    // Load current depth
    llvm::Value* depth = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_tape_depth);

    // Overflow check: abort if depth >= MAX_TAPE_DEPTH
    {
        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* overflow_bb = llvm::BasicBlock::Create(ctx_.context(), "tape_overflow", current_func);
        llvm::BasicBlock* safe_bb = llvm::BasicBlock::Create(ctx_.context(), "tape_safe", current_func);
        llvm::Value* is_overflow = ctx_.builder().CreateICmpUGE(depth,
            llvm::ConstantInt::get(ctx_.int64Type(), CodegenContext::MAX_TAPE_DEPTH));
        ctx_.builder().CreateCondBr(is_overflow, overflow_bb, safe_bb);

        ctx_.builder().SetInsertPoint(overflow_bb);
        // Print error message to stderr and abort
        llvm::FunctionType* fprintf_type = llvm::FunctionType::get(ctx_.int32Type(),
            {ctx_.ptrType(), ctx_.ptrType()}, true);
        llvm::FunctionCallee fprintf_func = ctx_.module().getOrInsertFunction("fprintf", fprintf_type);
        // Get stderr via the runtime wrapper on Windows; the MSVC CRT does
        // not export a plain `stderr` data symbol for user-program links.
#ifdef _WIN32
        llvm::FunctionType* stream_type = llvm::FunctionType::get(ctx_.ptrType(), {}, false);
        llvm::FunctionCallee stderr_fn = ctx_.module().getOrInsertFunction(
            runtime::stderr_stream_symbol, stream_type);
        llvm::Value* stderr_ptr = ctx_.builder().CreateCall(stderr_fn, {});
#else
        // On Unix this check fires at eshkol-run build time, so a Linux
        // build emits `stderr` and a macOS build emits `__stderrp`.
#ifdef __APPLE__
        const char* stderr_sym = "__stderrp";
#else
        const char* stderr_sym = "stderr";
#endif
        llvm::GlobalVariable* stderr_var = ctx_.module().getGlobalVariable(stderr_sym);
        if (!stderr_var) {
            stderr_var = new llvm::GlobalVariable(ctx_.module(), ctx_.ptrType(), false,
                llvm::GlobalValue::ExternalLinkage, nullptr, stderr_sym);
        }
        llvm::Value* stderr_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), stderr_var);
#endif
        llvm::Value* err_msg = ctx_.builder().CreateGlobalStringPtr(
            "AD tape stack overflow: nesting depth exceeds 32\n");
        ctx_.builder().CreateCall(fprintf_func, {stderr_ptr, err_msg});
        llvm::FunctionCallee abort_func = ctx_.module().getOrInsertFunction("abort",
            llvm::FunctionType::get(ctx_.voidType(), false));
        ctx_.builder().CreateCall(abort_func);
        ctx_.builder().CreateUnreachable();

        ctx_.builder().SetInsertPoint(safe_bb);
    }

    // Save current tape to stack[depth]
    llvm::Value* current_tape = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.currentAdTape());
    llvm::ArrayType* stack_type = llvm::ArrayType::get(ctx_.ptrType(), CodegenContext::MAX_TAPE_DEPTH);
    llvm::Value* slot_ptr = ctx_.builder().CreateGEP(stack_type, ad_tape_stack,
        {llvm::ConstantInt::get(ctx_.int64Type(), 0), depth});
    ctx_.builder().CreateStore(current_tape, slot_ptr);

    // Increment depth
    llvm::Value* new_depth = ctx_.builder().CreateAdd(depth, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(new_depth, ad_tape_depth);

    // Set new tape as current
    ctx_.builder().CreateStore(new_tape, ctx_.currentAdTape());

    // Set AD mode active
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int1Type(), 1), ad_mode_active);
}

/**
 * @brief Pop the AD tape stack, restoring the enclosing tape (nested gradients).
 *
 * Decrements the depth counter, restores the tape saved at stack[new_depth] as
 * the current tape, and keeps AD mode active only while depth remains greater
 * than zero (i.e. deactivates it when leaving the outermost gradient). Paired
 * with pushTapeContext.
 */
void AutodiffCodegen::popTapeContext() {
    llvm::GlobalVariable* ad_tape_depth = ctx_.adTapeDepth();
    llvm::GlobalVariable* ad_tape_stack = ctx_.adTapeStack();
    llvm::GlobalVariable* current_ad_tape_var = ctx_.currentAdTape();
    llvm::GlobalVariable* ad_mode_active = ctx_.adModeActive();

    if (!ad_tape_depth || !ad_tape_stack || !current_ad_tape_var || !ad_mode_active) {
        eshkol_warn("popTapeContext: AD globals not initialized");
        return;
    }

    // Load current depth
    llvm::Value* depth = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_tape_depth);

    // Decrement depth
    llvm::Value* new_depth = ctx_.builder().CreateSub(depth, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(new_depth, ad_tape_depth);

    // Restore tape from stack[new_depth]
    llvm::ArrayType* stack_type = llvm::ArrayType::get(ctx_.ptrType(), CodegenContext::MAX_TAPE_DEPTH);
    llvm::Value* slot_ptr = ctx_.builder().CreateGEP(stack_type, ad_tape_stack,
        {llvm::ConstantInt::get(ctx_.int64Type(), 0), new_depth});
    llvm::Value* saved_tape = ctx_.builder().CreateLoad(ctx_.ptrType(), slot_ptr);

    // Set restored tape as current
    ctx_.builder().CreateStore(saved_tape, ctx_.currentAdTape());

    // Set AD mode based on whether we still have active tapes
    // If new_depth == 0, we're exiting the outermost gradient
    llvm::Value* still_active = ctx_.builder().CreateICmpNE(new_depth,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateStore(still_active, ad_mode_active);
}

/**
 * @brief Return the enclosing (parent) AD tape for a nested gradient, or null if not nested.
 *
 * When the tape depth is greater than zero this returns stack[depth-1] (the tape
 * of the outer differentiation), otherwise a null pointer. Used to record
 * operations on the outer tape for double-backward.
 *
 * @return Pointer to the outer tape, or a null pointer when at the top level.
 */
llvm::Value* AutodiffCodegen::getOuterTape() {
    llvm::GlobalVariable* ad_tape_depth = ctx_.adTapeDepth();
    llvm::GlobalVariable* ad_tape_stack = ctx_.adTapeStack();

    if (!ad_tape_depth || !ad_tape_stack) {
        return llvm::ConstantPointerNull::get(ctx_.ptrType());
    }

    llvm::Value* depth = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_tape_depth);

    // Check if nested (depth > 0)
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* nested_bb = llvm::BasicBlock::Create(ctx_.context(), "outer_tape_nested", current_func);
    llvm::BasicBlock* not_nested_bb = llvm::BasicBlock::Create(ctx_.context(), "outer_tape_not_nested", current_func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "outer_tape_merge", current_func);

    llvm::Value* is_nested = ctx_.builder().CreateICmpUGT(depth, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(is_nested, nested_bb, not_nested_bb);

    // Nested: get tape from stack[depth-1]
    ctx_.builder().SetInsertPoint(nested_bb);
    llvm::Value* outer_idx = ctx_.builder().CreateSub(depth, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::ArrayType* stack_type = llvm::ArrayType::get(ctx_.ptrType(), CodegenContext::MAX_TAPE_DEPTH);
    llvm::Value* outer_slot = ctx_.builder().CreateGEP(stack_type, ad_tape_stack,
        {llvm::ConstantInt::get(ctx_.int64Type(), 0), outer_idx});
    llvm::Value* outer_tape = ctx_.builder().CreateLoad(ctx_.ptrType(), outer_slot);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* nested_exit = ctx_.builder().GetInsertBlock();

    // Not nested: return null
    ctx_.builder().SetInsertPoint(not_nested_bb);
    llvm::Value* null_tape = llvm::ConstantPointerNull::get(ctx_.ptrType());
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* not_nested_exit = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.ptrType(), 2, "outer_tape_result");
    result->addIncoming(outer_tape, nested_exit);
    result->addIncoming(null_tape, not_nested_exit);

    return result;
}

/** @brief Return an i1 that is true when the AD tape depth is greater than zero (inside a nested gradient). */
llvm::Value* AutodiffCodegen::isNested() {
    llvm::GlobalVariable* ad_tape_depth = ctx_.adTapeDepth();
    if (!ad_tape_depth) {
        return llvm::ConstantInt::get(ctx_.int1Type(), 0);
    }

    llvm::Value* depth = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_tape_depth);
    return ctx_.builder().CreateICmpUGT(depth, llvm::ConstantInt::get(ctx_.int64Type(), 0));
}

// ===== TAPE-SPECIFIC AD NODE OPERATIONS =====
// Used for double backward - record operations on outer tape

/**
 * @brief Allocate an AD constant/leaf node holding a value and append it to a specific tape.
 *
 * Creates a node of type constant (0) with the given value (converted to double
 * if integral), zero-initialized gradient, null inputs and a fresh node id, then
 * registers it on tape_ptr. Used when recording operations on the outer tape for
 * double-backward.
 *
 * @param tape_ptr The tape to record the node on.
 * @param value The primal value to store.
 * @return The new node pointer, or nullptr on error.
 */
llvm::Value* AutodiffCodegen::createADConstantOnTape(llvm::Value* tape_ptr, llvm::Value* value) {
    if (!tape_ptr || !value) return nullptr;

    // Convert value to double if needed
    if (value->getType()->isIntegerTy()) {
        value = ctx_.builder().CreateSIToFP(value, ctx_.doubleType());
    }

    // Allocate AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    llvm::StructType* ad_type = ctx_.adNodeType();

    // Set type = AD_NODE_CONSTANT (0)
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int32Type(), 0), type_ptr);

    // Set value
    llvm::Value* value_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1);
    ctx_.builder().CreateStore(value, value_ptr);

    // Initialize gradient = 0.0
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), grad_ptr);

    // Set input pointers to null (constant has no inputs)
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input1_ptr);

    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input2_ptr);

    // Set node ID
    llvm::Value* id_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++), id_ptr);

    // Add to specified tape
    llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
    if (add_node_func) {
        ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
    }

    return node_ptr;
}

/**
 * @brief Record a binary AD operation node on a specific tape (double-backward support).
 *
 * Reads the primal values of the two input nodes, computes the forward result for
 * op_type (add, sub, mul, div, pow, max, min, atan2), then allocates a new node
 * of that type storing the result, zero gradient, links to both input nodes and a
 * fresh id, and appends it to tape_ptr.
 *
 * @param tape_ptr The tape to record the node on.
 * @param op_type The binary AD op code (2 add, 3 sub, 4 mul, 5 div, 10 pow, 44 max, 45 min, plus atan2).
 * @param left_node First operand node.
 * @param right_node Second operand node.
 * @return The new node pointer, or nullptr on unknown op or error.
 */
llvm::Value* AutodiffCodegen::recordADNodeBinaryOnTape(llvm::Value* tape_ptr, uint32_t op_type,
                                                        llvm::Value* left_node, llvm::Value* right_node) {
    if (!tape_ptr || !left_node || !right_node) return nullptr;

    // Load values from input nodes
    llvm::Value* left_value = loadNodeValue(left_node);
    llvm::Value* right_value = loadNodeValue(right_node);

    if (!left_value || !right_value) return nullptr;

    // Compute result value based on operation
    llvm::Value* result_value = nullptr;
    switch (op_type) {
        case 2: // AD_NODE_ADD
            result_value = ctx_.builder().CreateFAdd(left_value, right_value);
            break;
        case 3: // AD_NODE_SUB
            result_value = ctx_.builder().CreateFSub(left_value, right_value);
            break;
        case 4: // AD_NODE_MUL
            result_value = ctx_.builder().CreateFMul(left_value, right_value);
            break;
        case 5: // AD_NODE_DIV
            result_value = ctx_.builder().CreateFDiv(left_value, right_value);
            break;
        case 10: // AD_NODE_POW
            {
                llvm::Function* pow_func = getMathFunc("pow");
                if (!pow_func) return nullptr;
                result_value = ctx_.builder().CreateCall(pow_func, {left_value, right_value});
            }
            break;
        case 44: // AD_NODE_MAX
            {
                // max(a, b) = a if a > b else b
                llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(left_value, right_value);
                result_value = ctx_.builder().CreateSelect(cmp, left_value, right_value);
            }
            break;
        case 45: // AD_NODE_MIN
            {
                // min(a, b) = a if a < b else b
                llvm::Value* cmp = ctx_.builder().CreateFCmpOLT(left_value, right_value);
                result_value = ctx_.builder().CreateSelect(cmp, left_value, right_value);
            }
            break;
        case AD_NODE_ATAN2:
            {
                llvm::Function* atan2_func = getMathFunc("atan2");
                if (!atan2_func) return nullptr;
                result_value = ctx_.builder().CreateCall(atan2_func, {left_value, right_value});
            }
            break;
        default:
            eshkol_warn("Unknown binary AD operation type: %u", op_type);
            return nullptr;
    }

    // Allocate new AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    llvm::StructType* ad_type = ctx_.adNodeType();

    // Set operation type
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int32Type(), op_type), type_ptr);

    // Set computed value
    llvm::Value* value_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1);
    ctx_.builder().CreateStore(result_value, value_ptr);

    // Initialize gradient = 0.0
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), grad_ptr);

    // Set input pointers
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    ctx_.builder().CreateStore(left_node, input1_ptr);

    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    ctx_.builder().CreateStore(right_node, input2_ptr);

    // Set node ID
    llvm::Value* id_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++), id_ptr);

    // Add to specified tape
    llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
    if (add_node_func) {
        ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
    }

    return node_ptr;
}

// ===== AD NODE HELPERS =====
// Access fields of AD nodes

/** @brief Load the primal value (field 1) of an AD node as a double; nullptr if node_ptr is null. */
llvm::Value* AutodiffCodegen::loadNodeValue(llvm::Value* node_ptr) {
    if (!node_ptr) return nullptr;
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* value_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1);
    return ctx_.builder().CreateLoad(ctx_.doubleType(), value_ptr);
}

/** @brief Load the accumulated gradient (field 2) of an AD node as a double; nullptr if node_ptr is null. */
llvm::Value* AutodiffCodegen::loadNodeGradient(llvm::Value* node_ptr) {
    if (!node_ptr) return nullptr;
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    return ctx_.builder().CreateLoad(ctx_.doubleType(), grad_ptr);
}

/** @brief Overwrite an AD node's gradient field (field 2) with the given value; no-op on null args. */
void AutodiffCodegen::storeNodeGradient(llvm::Value* node_ptr, llvm::Value* gradient) {
    if (!node_ptr || !gradient) return;
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    ctx_.builder().CreateStore(gradient, grad_ptr);
}

/**
 * @brief Add a gradient contribution into an AD node's gradient field (chain-rule accumulation).
 *
 * Emits a runtime null check on node_ptr (leaf inputs may be null) and, when
 * non-null, performs grad = grad + gradient_to_add. This is the accumulation
 * primitive used by propagateGradient during the backward pass.
 *
 * @param node_ptr The input node whose gradient is being accumulated (may be null at runtime).
 * @param gradient_to_add The gradient contribution to add.
 */
void AutodiffCodegen::accumulateGradient(llvm::Value* node_ptr, llvm::Value* gradient_to_add) {
    if (!node_ptr || !gradient_to_add) return;  // Compile-time check

    // RUNTIME NULL CHECK: Generate LLVM IR to check if node_ptr is null at runtime
    // This is critical because AD constant/variable nodes have null input pointers
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    llvm::BasicBlock* accumulate_block = llvm::BasicBlock::Create(ctx_.context(), "accumulate_grad", current_func);
    llvm::BasicBlock* skip_accumulate = llvm::BasicBlock::Create(ctx_.context(), "skip_accumulate", current_func);
    llvm::BasicBlock* merge_accumulate = llvm::BasicBlock::Create(ctx_.context(), "merge_accumulate", current_func);

    // Check if node_ptr is null at runtime
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(node_ptr,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));
    ctx_.builder().CreateCondBr(is_null, skip_accumulate, accumulate_block);

    // Non-null path: perform gradient accumulation
    ctx_.builder().SetInsertPoint(accumulate_block);
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    llvm::Value* current_grad = ctx_.builder().CreateLoad(ctx_.doubleType(), grad_ptr);
    llvm::Value* new_grad = ctx_.builder().CreateFAdd(current_grad, gradient_to_add);
    ctx_.builder().CreateStore(new_grad, grad_ptr);
    ctx_.builder().CreateBr(merge_accumulate);

    // Null path: skip accumulation
    ctx_.builder().SetInsertPoint(skip_accumulate);
    ctx_.builder().CreateBr(merge_accumulate);

    // Merge point: continue from here
    ctx_.builder().SetInsertPoint(merge_accumulate);
}

// ===== ML ACTIVATION FUNCTION DUAL NUMBER OPERATIONS =====
// These implement chain rule for ML activation functions

/**
 * @brief Forward-mode dual ReLU: value max(0, a), tangent a' when a > 0 else 0.
 *
 * @param dual Input dual number (primal a, tangent a').
 * @return New dual number carrying relu(a) and its subgradient-scaled tangent.
 */
llvm::Value* AutodiffCodegen::dualRelu(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    // Value: max(0, a)
    llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(a, zero);
    llvm::Value* value = ctx_.builder().CreateSelect(is_positive, a, zero);

    // Derivative: a > 0 ? a' : 0
    llvm::Value* deriv = ctx_.builder().CreateSelect(is_positive, a_prime, zero);

    return createDualNumber(value, deriv);
}

/**
 * @brief Forward-mode dual sigmoid s(a) = 1 / (1 + exp(-a)).
 *
 * Computes the primal s(a), first derivative s(1-s) and second derivative
 * s(1-s)(1-2s), then applies them via dualUnaryChain so higher-order tangents
 * (double-forward) propagate correctly.
 *
 * @param dual Input dual number.
 * @return New dual number for the sigmoid activation.
 */
llvm::Value* AutodiffCodegen::dualSigmoid(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = dualField(ctx_, dual, 0);
    llvm::Function* exp_func = getMathFunc("exp");
    if (!exp_func) return nullptr;
    auto& b = ctx_.builder();
    // σ(a) = 1 / (1 + exp(-a))
    llvm::Value* exp_neg_a = b.CreateCall(exp_func, {b.CreateFNeg(a)});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* sigma_a = b.CreateFDiv(one, b.CreateFAdd(one, exp_neg_a));
    // g'  = σ(1-σ);  g'' = σ(1-σ)(1-2σ)
    llvm::Value* oms = b.CreateFSub(one, sigma_a);
    llvm::Value* fpa = b.CreateFMul(sigma_a, oms);
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* one_minus_2s = b.CreateFSub(one, b.CreateFMul(two, sigma_a));
    llvm::Value* fppa = b.CreateFMul(fpa, one_minus_2s);
    return dualUnaryChain(ctx_, dual, sigma_a, fpa, fppa);
}

/**
 * @brief Forward-mode dual GELU using the tanh approximation (matching tensorGelu).
 *
 * Primal is 0.5 * a * (1 + tanh(u)) with u = sqrt(2/pi) * (a + 0.044715 * a^3);
 * the tangent multiplies a' by the exact derivative of that expression
 * (via sech^2(u) times u').
 *
 * @param dual Input dual number.
 * @return New dual number for the GELU activation.
 */
llvm::Value* AutodiffCodegen::dualGelu(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* exp_func = getMathFunc("exp");
    if (!exp_func) return nullptr;

    llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
    llvm::Value* sqrt_2_pi = llvm::ConstantFP::get(ctx_.doubleType(), 0.7978845608028654);
    llvm::Value* coeff = llvm::ConstantFP::get(ctx_.doubleType(), 0.044715);

    llvm::Value* a_sq = ctx_.builder().CreateFMul(a, a);
    llvm::Value* a_cubed = ctx_.builder().CreateFMul(a_sq, a);
    llvm::Value* inner = ctx_.builder().CreateFAdd(
        a, ctx_.builder().CreateFMul(coeff, a_cubed));
    llvm::Value* u = ctx_.builder().CreateFMul(sqrt_2_pi, inner);
    llvm::Value* exp_2u = ctx_.builder().CreateCall(
        exp_func, {ctx_.builder().CreateFMul(two, u)});
    llvm::Value* tanh_u = ctx_.builder().CreateFDiv(
        ctx_.builder().CreateFSub(exp_2u, one),
        ctx_.builder().CreateFAdd(exp_2u, one));

    llvm::Value* value = ctx_.builder().CreateFMul(
        half, ctx_.builder().CreateFMul(a, ctx_.builder().CreateFAdd(one, tanh_u)));

    llvm::Value* tanh_sq = ctx_.builder().CreateFMul(tanh_u, tanh_u);
    llvm::Value* sech_sq = ctx_.builder().CreateFSub(one, tanh_sq);
    llvm::Value* inner_prime = ctx_.builder().CreateFAdd(
        one, ctx_.builder().CreateFMul(three, ctx_.builder().CreateFMul(coeff, a_sq)));
    llvm::Value* u_prime = ctx_.builder().CreateFMul(sqrt_2_pi, inner_prime);
    llvm::Value* first = ctx_.builder().CreateFMul(
        half, ctx_.builder().CreateFAdd(one, tanh_u));
    llvm::Value* second = ctx_.builder().CreateFMul(
        half, ctx_.builder().CreateFMul(a, ctx_.builder().CreateFMul(sech_sq, u_prime)));
    llvm::Value* total_deriv = ctx_.builder().CreateFAdd(first, second);
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, total_deriv);

    return createDualNumber(value, deriv);
}

/**
 * @brief Forward-mode dual Leaky ReLU: value a when a > 0 else alpha*a; tangent a' or alpha*a'.
 *
 * @param dual Input dual number.
 * @param alpha Negative-slope coefficient applied when a <= 0.
 * @return New dual number for the leaky-ReLU activation.
 */
llvm::Value* AutodiffCodegen::dualLeakyRelu(llvm::Value* dual, double alpha) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* alpha_val = llvm::ConstantFP::get(ctx_.doubleType(), alpha);

    // Value: a > 0 ? a : α * a
    llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(a, zero);
    llvm::Value* alpha_a = ctx_.builder().CreateFMul(alpha_val, a);
    llvm::Value* value = ctx_.builder().CreateSelect(is_positive, a, alpha_a);

    // Derivative: a > 0 ? a' : α * a'
    llvm::Value* alpha_a_prime = ctx_.builder().CreateFMul(alpha_val, a_prime);
    llvm::Value* deriv = ctx_.builder().CreateSelect(is_positive, a_prime, alpha_a_prime);

    return createDualNumber(value, deriv);
}

/**
 * @brief Forward-mode dual SiLU/Swish: value a * s(a) with s the sigmoid.
 *
 * Tangent is a' * s(a) * (1 + a * (1 - s(a))), the chain-rule derivative of
 * a * s(a).
 *
 * @param dual Input dual number.
 * @return New dual number for the SiLU activation.
 */
llvm::Value* AutodiffCodegen::dualSilu(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* exp_func = getMathFunc("exp");
    if (!exp_func) return nullptr;

    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

    // σ(a) = 1 / (1 + exp(-a))
    llvm::Value* neg_a = ctx_.builder().CreateFNeg(a);
    llvm::Value* exp_neg_a = ctx_.builder().CreateCall(exp_func, {neg_a});
    llvm::Value* denom = ctx_.builder().CreateFAdd(one, exp_neg_a);
    llvm::Value* sigma_a = ctx_.builder().CreateFDiv(one, denom);

    // Value: a * σ(a)
    llvm::Value* value = ctx_.builder().CreateFMul(a, sigma_a);

    // Derivative: a' * σ(a) * (1 + a * (1 - σ(a)))
    llvm::Value* one_minus_sigma = ctx_.builder().CreateFSub(one, sigma_a);
    llvm::Value* a_times_one_minus = ctx_.builder().CreateFMul(a, one_minus_sigma);
    llvm::Value* one_plus_term = ctx_.builder().CreateFAdd(one, a_times_one_minus);
    llvm::Value* sigma_times_term = ctx_.builder().CreateFMul(sigma_a, one_plus_term);
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, sigma_times_term);

    return createDualNumber(value, deriv);
}

/**
 * @brief Forward-mode dual square: value a^2, tangent 2 * a * a'.
 *
 * @param dual Input dual number.
 * @return New dual number for the square operation.
 */
llvm::Value* AutodiffCodegen::dualSquare(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);

    // Value: a²
    llvm::Value* value = ctx_.builder().CreateFMul(a, a);

    // Derivative: 2 * a * a'
    llvm::Value* two_a = ctx_.builder().CreateFMul(two, a);
    llvm::Value* deriv = ctx_.builder().CreateFMul(two_a, a_prime);

    return createDualNumber(value, deriv);
}

/**
 * @brief Forward-mode dual max: value max(a, b); tangent a' when a > b else b'.
 *
 * At the non-differentiable tie a == b the b' branch is taken (the select uses a
 * strict a > b comparison), an arbitrary but consistent subgradient choice.
 *
 * @param dual_a First input dual number.
 * @param dual_b Second input dual number.
 * @return New dual number for the max operation.
 */
llvm::Value* AutodiffCodegen::dualMax(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;

    llvm::Value* a = getDualPrimal(dual_a);
    llvm::Value* a_prime = getDualTangent(dual_a);
    llvm::Value* b = getDualPrimal(dual_b);
    llvm::Value* b_prime = getDualTangent(dual_b);

    // Value: max(a, b)
    llvm::Value* a_gt_b = ctx_.builder().CreateFCmpOGT(a, b);
    llvm::Value* value = ctx_.builder().CreateSelect(a_gt_b, a, b);

    // Derivative: a > b ? a' : b'
    llvm::Value* deriv = ctx_.builder().CreateSelect(a_gt_b, a_prime, b_prime);

    return createDualNumber(value, deriv);
}

/**
 * @brief Forward-mode dual min: value min(a, b); tangent a' when a < b else b'.
 *
 * At the non-differentiable tie a == b the b' branch is taken (the select uses a
 * strict a < b comparison), an arbitrary but consistent subgradient choice.
 *
 * @param dual_a First input dual number.
 * @param dual_b Second input dual number.
 * @return New dual number for the min operation.
 */
llvm::Value* AutodiffCodegen::dualMin(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;

    llvm::Value* a = getDualPrimal(dual_a);
    llvm::Value* a_prime = getDualTangent(dual_a);
    llvm::Value* b = getDualPrimal(dual_b);
    llvm::Value* b_prime = getDualTangent(dual_b);

    // Value: min(a, b)
    llvm::Value* a_lt_b = ctx_.builder().CreateFCmpOLT(a, b);
    llvm::Value* value = ctx_.builder().CreateSelect(a_lt_b, a, b);

    // Derivative: a < b ? a' : b'
    llvm::Value* deriv = ctx_.builder().CreateSelect(a_lt_b, a_prime, b_prime);

    return createDualNumber(value, deriv);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
