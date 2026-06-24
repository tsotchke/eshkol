//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//
// AArch64 Branch26 range-extension plugin for the in-process ORC/JITLink JIT.
//
// Why this exists
// ---------------
// Eshkol's `eval`/`compile` route through the in-process JIT (LLJIT + JITLink),
// NOT the AOT run-cache. When the large precompiled stdlib (>128 MB of .text on
// arm64) is linked, an intra-object `bl`/`b` (AArch64 `Branch26PCRel`, range
// +/-128 MB) can land out of range of its target. LLVM 21's JITLink has NO
// automatic branch-range-extension for aarch64 (grep of
// llvm/ExecutionEngine/JITLink finds no RangeExtension/branch-island pass), and
// the AArch64 large code model is incomplete on ELF/COFF (it works only on
// Mach-O, which is why macOS-arm64 never hit this). The result on arm64-Linux /
// arm64-Windows is a hard link failure:
//
//   "relocation target ... is out of range of Branch26PCRel fixup"
//
// What this plugin does
// ---------------------
// It mirrors what static linkers (lld) do with veneers/thunks: for every
// `Branch26PCRel` edge it inserts a *range-extension stub* and re-points the
// edge at the stub. The stub is an absolute indirect jump materialised inline:
//
//     movz x16, #:abs_g0_nc:target
//     movk x16, #:abs_g1_nc:target
//     movk x16, #:abs_g2_nc:target
//     movk x16, #:abs_g3   :target
//     br   x16
//
// i.e. it loads the full 64-bit absolute target address into x16 via four
// MOVZ/MOVK (resolved by JITLink `MoveWide16` edges at fixup time) and branches
// to it. An absolute indirect branch reaches *any* address, so the
// stub<->target distance is irrelevant.
//
// Placement / correctness
// -----------------------
// The only remaining range constraint is the caller -> stub `Branch26` itself.
// We place each stub in the SAME `Section` as the calling block. The stdlib is
// emitted with FunctionSections=true, so each function is its own section and
// is far smaller than 128 MB; a stub appended to the caller's own section is
// therefore always within +/-128 MB of every call site in that section. (If a
// single section ever exceeded 128 MB the stub could itself fall out of range,
// but no individual Eshkol function approaches that; the pathology is the
// aggregate .text size, which per-section placement defuses.) This is
// format-agnostic: it works for ELF, COFF and Mach-O alike, and depends only on
// the AArch64 JITLink edge kinds, not on any object-format specifics.
//
// The pass runs in PostPrunePasses (before memory allocation), which is the only
// phase where new stub blocks can still be given memory by the allocator -- the
// same phase JITLink's own GOT/PLT builders use. Because addresses are not yet
// assigned there, we cannot test individual edges for range; we therefore
// veneer *every* Branch26 edge unconditionally. Stubs are deduplicated per
// (section, target) so a function that calls the same callee many times shares
// one stub. The cost is a handful of extra instructions per distinct callee per
// section; this only fires on aarch64 and is dwarfed by JIT codegen time.

#ifndef ESHKOL_JITLINK_BRANCH26_RANGE_EXTENSION_H
#define ESHKOL_JITLINK_BRANCH26_RANGE_EXTENSION_H

#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/JITLink/aarch64.h>
#include <llvm/ExecutionEngine/Orc/LinkGraphLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Error.h>
#include <llvm/TargetParser/Triple.h>

#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <utility>
#include <vector>

namespace eshkol {

/// ORC plugin that installs an AArch64 Branch26 range-extension pass into every
/// JITLink LinkGraph. No-op on non-aarch64 targets.
class Branch26RangeExtensionPlugin
    : public llvm::orc::LinkGraphLinkingLayer::Plugin {
public:
  void modifyPassConfig(llvm::orc::MaterializationResponsibility &,
                        llvm::jitlink::LinkGraph &G,
                        llvm::jitlink::PassConfiguration &Config) override {
    // Only aarch64 needs this -- other targets either have working far-call
    // code models or JITLink range extension.
    if (G.getTargetTriple().getArch() != llvm::Triple::aarch64 &&
        G.getTargetTriple().getArch() != llvm::Triple::aarch64_be)
      return;

    // Escape hatch for debugging / measuring.
    if (const char *Off = std::getenv("ESHKOL_JIT_NO_BRANCH26_VENEER"))
      if (Off[0] == '1')
        return;

    Config.PostPrunePasses.push_back(
        [](llvm::jitlink::LinkGraph &G) { return extendBranches(G); });
  }

  // Required overrides; this plugin tracks no per-resource state.
  llvm::Error notifyFailed(llvm::orc::MaterializationResponsibility &) override {
    return llvm::Error::success();
  }
  llvm::Error notifyRemovingResources(llvm::orc::JITDylib &,
                                      llvm::orc::ResourceKey) override {
    return llvm::Error::success();
  }
  void notifyTransferringResources(llvm::orc::JITDylib &,
                                   llvm::orc::ResourceKey,
                                   llvm::orc::ResourceKey) override {}

private:
  // movz x16,#0 ; movk x16,#0,lsl#16 ; movk x16,#0,lsl#32 ;
  // movk x16,#0,lsl#48 ; br x16   (little-endian, imm fields zeroed)
  static constexpr std::size_t kStubSize = 20;

  static llvm::Error extendBranches(llvm::jitlink::LinkGraph &G) {
    using namespace llvm::jitlink;

    // Stub content template: the four MOVZ/MOVK imm16 fields are left zero and
    // are filled in by JITLink via the MoveWide16 edges we attach below; the
    // shift (LSL #0/16/32/48) is encoded in each instruction so JITLink writes
    // the correct 16-bit slice of the absolute target address.
    static const uint8_t StubTemplate[kStubSize] = {
        0x10, 0x00, 0x80, 0xd2, // movz x16, #0,        (lsl #0)
        0x10, 0x00, 0xa0, 0xf2, // movk x16, #0, lsl #16
        0x10, 0x00, 0xc0, 0xf2, // movk x16, #0, lsl #32
        0x10, 0x00, 0xe0, 0xf2, // movk x16, #0, lsl #48
        0x00, 0x02, 0x1f, 0xd6, // br   x16
    };

    // Collect the work first. Re-pointing existing edges in place is fine, but
    // we also create new blocks/symbols which can perturb iteration, so snapshot
    // the (block, edge-index, target, addend) tuples to act on.
    struct EdgeRef {
      Block *B;
      std::size_t EdgeIdx; // index into the block's edge list at snapshot time
      Symbol *Target;
      Edge::AddendT Addend;
    };
    std::vector<EdgeRef> Work;

    for (auto *B : G.blocks()) {
      std::size_t Idx = 0;
      for (auto &E : B->edges()) {
        if (E.getKind() == aarch64::Branch26PCRel)
          Work.push_back({B, Idx, &E.getTarget(), E.getAddend()});
        ++Idx;
      }
    }

    if (Work.empty())
      return llvm::Error::success();

    // Dedup stubs per (section ordinal, target symbol). A stub is valid for any
    // caller in the same section, and an absolute-jump stub does not depend on
    // the call-site address, so one per (section, target) suffices.
    llvm::DenseMap<std::pair<unsigned, Symbol *>, Symbol *> StubFor;

    for (auto &W : Work) {
      Section &Sec = W.B->getSection();
      auto Key = std::make_pair(Sec.getOrdinal(), W.Target);

      Symbol *Stub = nullptr;
      auto It = StubFor.find(Key);
      if (It != StubFor.end()) {
        Stub = It->second;
      } else {
        // Create the stub block in the SAME section as the caller so the
        // caller -> stub Branch26 is always in range.
        Block &StubBlock = G.createContentBlock(
            Sec,
            llvm::ArrayRef<char>(reinterpret_cast<const char *>(StubTemplate),
                                 kStubSize),
            llvm::orc::ExecutorAddr(), /*Alignment=*/4, /*AlignmentOffset=*/0);

        // Four MoveWide16 edges (offsets 0,4,8,12) fill the absolute target
        // address into x16; the per-instruction LSL encodes which 16-bit slice.
        // The original branch addend is folded into every slice's target so the
        // stub jumps to exactly the same effective target the direct branch
        // would have reached.
        StubBlock.addEdge(aarch64::MoveWide16, 0, *W.Target, W.Addend);
        StubBlock.addEdge(aarch64::MoveWide16, 4, *W.Target, W.Addend);
        StubBlock.addEdge(aarch64::MoveWide16, 8, *W.Target, W.Addend);
        StubBlock.addEdge(aarch64::MoveWide16, 12, *W.Target, W.Addend);

        Stub = &G.addAnonymousSymbol(StubBlock, /*Offset=*/0, kStubSize,
                                     /*IsCallable=*/true, /*IsLive=*/false);
        StubFor[Key] = Stub;
      }

      // Re-point the original branch at the stub and zero the addend (the addend
      // is now baked into the stub's absolute target).
      auto EdgeIt = W.B->edges().begin();
      std::advance(EdgeIt, W.EdgeIdx);
      EdgeIt->setTarget(*Stub);
      EdgeIt->setAddend(0);
    }

    return llvm::Error::success();
  }
};

} // namespace eshkol

#endif // ESHKOL_JITLINK_BRANCH26_RANGE_EXTENSION_H
