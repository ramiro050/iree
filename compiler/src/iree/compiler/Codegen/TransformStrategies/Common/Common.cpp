// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/AbstractReductionStrategy.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::ForallToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
using iree_compiler::IREE::transform_dialect::IREEEliminateEmptyTensorsOp;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using transform::FuseIntoContainingOp;
using transform::HoistLoopInvariantSubsetsOp;
using transform::MatchOp;
using transform::MemRefEraseDeadAllocAndStoresOp;
using transform::MergeHandlesOp;
using transform::NamedSequenceOp;
using transform::PrintOp;
using transform::SplitHandleOp;
using transform::SplitReductionOp;
using transform::TileUsingForallOp;
using transform::VectorizeChildrenAndApplyPatternsOp;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::TakeFirstOp;

/// Matches `args` within `targetH` and unpacks a number of handles `N`.
/// Assumes there are exactly `N` matched ops (but could be relaxed).
/// Returns the tuple of handles.
template <int N, typename... MatchingArgs>
auto matchAndUnpack(ImplicitLocOpBuilder &b, Value targetH,
                    MatchingArgs... args) {
  Value matchedH = b.create<MatchOp>(targetH, args...);
  auto matchOp = b.create<SplitHandleOp>(matchedH,
                                         /*numHandles=*/N);
  assert(matchOp->getNumResults() == N && "Unexpected number of results");
  std::array<Value, N> a;
  for (int64_t i = 0; i < N; ++i)
    a[i] = matchOp->getResult(i);
  return std::tuple_cat(a);
}

int64_t mlir::iree_compiler::previousMultipleOf(int64_t val, int64_t multiple) {
  assert(val > 0 && "expected nonnegative val");
  assert(multiple > 0 && "expected nonnegative multiple");
  return (val / multiple) * multiple;
}

int64_t mlir::iree_compiler::nextMultipleOf(int64_t val, int64_t multiple) {
  assert(val > 0 && "expected nonnegative val");
  assert(multiple > 0 && "expected nonnegative multiple");
  return ((val + multiple - 1) / multiple) * multiple;
}

FailureOr<int64_t>
mlir::iree_compiler::maxDivisorOfValueBelowLimit(int64_t value, int64_t limit) {
  // Conservatively return failure when `limit` is greater than 1024 to avoid
  // prohibitively long compile time overheads.
  // TODO: approximate with a faster implementation based on a few desirable
  // primes.
  if (limit > 1024)
    return failure();
  // If either value or limit is <= 0, the loop is skipped and we fail.
  for (int64_t i = std::min(value, limit); i > 1; --i)
    if (value % i == 0)
      return i;
  return failure();
}

void mlir::iree_compiler::createTransformRegion(
    func::FuncOp entryPoint, StrategyBuilderFn buildStrategy) {
  MLIRContext *ctx = entryPoint.getContext();
  Location loc = entryPoint.getLoc();
  OpBuilder b(ctx);
  b.setInsertionPointAfter(entryPoint);
  auto topLevelTransformModule = b.create<ModuleOp>(loc);
  topLevelTransformModule->setAttr(
      transform::TransformDialect::kWithNamedSequenceAttrName, b.getUnitAttr());
  Region &topLevelTransformRegion = topLevelTransformModule.getBodyRegion();
  b.setInsertionPointToStart(&topLevelTransformRegion.front());
  auto anyOpType = transform::AnyOpType::get(b.getContext());
  auto sequence = b.create<transform::NamedSequenceOp>(
      loc,
      /*symName=*/
      std::string(
          transform::TransformDialect::kTransformEntryPointSymbolName.str()),
      /*rootType*/ anyOpType,
      /*resultTypes=*/TypeRange{},
      /*bodyBuilder=*/[&](OpBuilder &b, Location loc, Value variantH) {
        ImplicitLocOpBuilder ib(loc, b);
        buildStrategy(ib, variantH);
        b.create<transform::YieldOp>(loc);
      });
  (void)sequence;

  LDBG("transformation script:\n");
  LDBG("verification: " << sequence.verify().succeeded() << "\n");
}

//===----------------------------------------------------------------------===//
// Low-level reusable builder APIs, these should follow MLIR-style builders.
//===----------------------------------------------------------------------===//

/// Prints `handles` in order. Prints the whole IR if `handles` is empty.
void mlir::iree_compiler::buildPrint(ImplicitLocOpBuilder &b,
                                     ValueRange handles) {
  if (handles.empty())
    b.create<PrintOp>();
  for (auto h : handles)
    b.create<PrintOp>(h);
}

/// Create an ApplyPatternsOp that performs a set of key canonicalizations and
/// so-called enabling transformations to normalize the IR.
/// In addition to the specified transform, perform the following ones:
///   tiling-related canonicalization patterns, canonicalization, licm and cse
///   (in this order).
void mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
    ImplicitLocOpBuilder &b, Value funcH,
    ApplyPatternsOpBodyBuilderFn populatePatternsFn) {
  b.create<transform::ApplyPatternsOp>(funcH, [&](OpBuilder &b, Location loc) {
    b.create<transform::ApplyTilingCanonicalizationPatternsOp>(loc);
    b.create<IREE::transform_dialect::ApplyFoldFillIntoPadPatternsOp>(loc);
    b.create<transform::ApplyForLoopCanonicalizationPatternsOp>(loc);
    b.create<transform::ApplyCanonicalizationPatternsOp>(loc);
    if (populatePatternsFn)
      populatePatternsFn(b, loc);
  });
  b.create<IREE::transform_dialect::ApplyLoopIndependentCodeMotionOp>(funcH);
  b.create<IREE::transform_dialect::ApplyCommonSubexpressionEliminationOp>(
      funcH);
}

/// Dynamically selects the first non-empty handle; i.e. if (h1, h2) is:
///   - (non-empty, non-empty), returns (h1, h2)
///   - (empty, non-empty), returns (h2, empty)
///   - (non-empty, empty), returns (h1, empty)
///   - (empty, empty), returns (empty, empty)
/// This is used as a normalization operation that replaces conditionals, either
/// in C++ or in transform IR.
/// This can be thought of as a control-flow -> data-dependent conversion.
std::pair<Value, Value>
mlir::iree_compiler::buildSelectFirstNonEmpty(ImplicitLocOpBuilder &b,
                                              Value handle1, Value handle2) {
  auto anyOpType = transform::AnyOpType::get(b.getContext());
  auto selector = b.create<TakeFirstOp>(anyOpType, anyOpType,
                                        ArrayRef<Value>{handle1, handle2});
  return std::make_pair(selector.getFirst(), selector.getRest());
}

mlir::iree_compiler::TileToScfForAndFuseResult
mlir::iree_compiler::buildTileFuseToScfFor(ImplicitLocOpBuilder &b,
                                           Value variantH, Value rootH,
                                           ValueRange opsHToFuse,
                                           ArrayRef<OpFoldResult> tileSizes,
                                           bool canonicalize) {
  assert(opsHToFuse.empty() && "No fusion supported yet");
  iree_compiler::TileToScfForAndFuseResult result;
  auto tiletoScfForOp = b.create<transform::TileUsingForOp>(rootH, tileSizes);
  result.forLoops = tiletoScfForOp.getLoops();
  result.tiledOpH = tiletoScfForOp.getTiledLinalgOp();

  // Perform a pass of canonicalization + enabling after tiling. Currently this
  // folds away the extract slice on the iterator, breaking padding on aligned
  // matmuls.
  // TODO: Make padding less brittle so that this toggle is unnecessary.
  if (canonicalize) {
    Value funcH = b.create<transform::MatchOp>(
        variantH, func::FuncOp::getOperationName());
    mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);
  }
  return result;
}

/// Performs the following transformations:
///   1. Tiles `rootH` to scf.forall to with `tileSizesOrNumThreads`
///      according to whether spec is a TileSizesSpec or a NumThreadsSpec.
///   2. Maps the resulting scf.forall to threads according to
///      `threadDimMapping`.
///   3. Iterates over `opsHToFuse` in order and fuses into the containing op.
/// Returns a handle to the resulting scf.forall.
///
/// Fusion operates in batch mode: a single fusion command is issued and a
/// topological sort is automatically computed by the fusion.
/// Since this applies a single fusion, no interleaved canonicalization / cse /
/// enabling transformation occurs and the resulting fusion may not be as good.
///
/// In the future, an iterative mode in which the user is responsible for
/// providing the fusion order and has interleaved canonicalization / cse /
/// enabling transform will be introduced and may result in better fusions.
///
/// If `resultingFusedOpsHandles` is a non-null pointer, the fused operation are
/// appended in order.
// TODO: apply forwarding pattern.
template <typename TileOrNumThreadSpec>
static iree_compiler::TileToForallAndFuseAndDistributeResult
buildTileAndFuseAndDistributeImpl(ImplicitLocOpBuilder &b, Value variantH,
                                  Value rootH, ValueRange opsHToFuse,
                                  ArrayRef<OpFoldResult> tileSizesOrNumThreads,
                                  ArrayAttr threadDimMapping) {
  iree_compiler::TileToForallAndFuseAndDistributeResult result;
  auto tileToForeachOp = b.create<TileUsingForallOp>(
      rootH, tileSizesOrNumThreads, TileOrNumThreadSpec(), threadDimMapping);

  result.forallH = tileToForeachOp.getForallOp();
  result.tiledOpH = tileToForeachOp.getTiledOp();

  // Perform a pass of canonicalization + enabling after tiling.
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);

  // Batch fusion if requested.
  if (opsHToFuse.size() > 1) {
    Value mergedOpsH =
        b.create<MergeHandlesOp>(opsHToFuse, /*deduplicate=*/true);
    b.create<FuseIntoContainingOp>(mergedOpsH, result.forallH).getFusedOp();
  } else if (opsHToFuse.size() == 1) {
    Value fusedH =
        b.create<FuseIntoContainingOp>(opsHToFuse.front(), result.forallH)
            .getFusedOp();
    result.resultingFusedOpsHandles.push_back(fusedH);
  }
  return result;
}

// TODO: if someone knows how to properly export templates go for it ..
// sigh.
iree_compiler::TileToForallAndFuseAndDistributeResult
mlir::iree_compiler::buildTileFuseDistToForallWithTileSizes(
    ImplicitLocOpBuilder &b, Value variantH, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> tileSizes, ArrayAttr threadDimMapping) {
  return buildTileAndFuseAndDistributeImpl<transform::TileSizesSpec>(
      b, variantH, rootH, opsHToFuse, tileSizes, threadDimMapping);
}

/// Call buildTileAndFuseAndDistributeImpl with ArrayRef<int64_t> numThreads.
// TODO: if someone knows how to properly export templates go for it ..
// sigh.
iree_compiler::TileToForallAndFuseAndDistributeResult
mlir::iree_compiler::buildTileFuseDistToForallWithNumThreads(
    ImplicitLocOpBuilder &b, Value variantH, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> numThreads, ArrayAttr threadDimMapping) {
  return buildTileAndFuseAndDistributeImpl<transform::NumThreadsSpec>(
      b, variantH, rootH, opsHToFuse, numThreads, threadDimMapping);
}

/// Build the transform IR to pad an op `opH`.
// TODO: Better upstream builder.
Value mlir::iree_compiler::buildPad(
    ImplicitLocOpBuilder &b, Value opH, ArrayRef<Attribute> paddingValues,
    ArrayRef<int64_t> paddingDimensions, ArrayRef<int64_t> packingDimensions,
    ArrayRef<SmallVector<int64_t>> transposePaddings) {
  SmallVector<int64_t> padToMultipleOf(paddingDimensions.size(), 1);
  SmallVector<Attribute> transposeAttrs;
  for (auto &transp : transposePaddings)
    transposeAttrs.push_back(b.getI64ArrayAttr(transp));
  SmallVector<Type> resultTypes;
  resultTypes.push_back(opH.getType());
  resultTypes.push_back(transform::AnyOpType::get(b.getContext()));
  resultTypes.push_back(transform::AnyOpType::get(b.getContext()));
  return b
      .create<transform::PadOp>(resultTypes, opH, b.getArrayAttr(paddingValues),
                                b.getI64ArrayAttr(paddingDimensions),
                                b.getI64ArrayAttr(padToMultipleOf),
                                b.getI64ArrayAttr(packingDimensions),
                                b.getArrayAttr(transposeAttrs),
                                /*copyBack=*/b.getStringAttr("none"))
      ->getResult(0);
}

/// Apply patterns and vectorize.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
// TODO: configure patterns.
Value mlir::iree_compiler::buildVectorize(ImplicitLocOpBuilder &b, Value funcH,
                                          bool applyCleanups,
                                          bool vectorizePadding,
                                          bool vectorizeNdExtract) {
  funcH = b.create<VectorizeChildrenAndApplyPatternsOp>(funcH, vectorizePadding,
                                                        vectorizeNdExtract);
  if (applyCleanups) {
    iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);
  }
  return funcH;
}

void mlir::iree_compiler::buildLowerMaskedTransfersAndCleanup(
    ImplicitLocOpBuilder &b, Value funcH, bool cleanup) {
  // TODO: avoid functional style transform so we can apply to the variant.
  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyLowerMaskedTransfersPatternsOp>(loc);
  });
  if (cleanup) {
    b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
      b.create<transform::ApplyCastAwayVectorLeadingOneDimPatternsOp>(loc);
      b.create<transform::ApplyFoldUnitExtentDimsViaSlicesPatternsOp>(loc);
      b.create<IREE::transform_dialect::
                   ApplyFoldReshapeIntoTensorHalInterfacePatternsOp>(loc);
    });
  }
}

Value mlir::iree_compiler::buildLowerVectorMasksAndCleanup(
    ImplicitLocOpBuilder &b, Value funcH, bool cleanup) {
  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyLowerMasksPatternsOp>(loc);
  });
  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyMaterializeMasksPatternsOp>(loc);
  });
  if (cleanup) {
    iree_compiler::buildCanonicalizationAndEnablingTransforms(
        b, funcH, [](OpBuilder &b, Location loc) {
          b.create<transform::ApplyFoldMemrefAliasOpsPatternsOp>(loc);
        });
  }
  return funcH;
}

/// Hoist redundant subet ops.
void mlir::iree_compiler::buildHoisting(ImplicitLocOpBuilder &b, Value funcH) {
  Value loops =
      b.create<transform::MatchOp>(funcH, scf::ForOp::getOperationName());
  b.create<HoistLoopInvariantSubsetsOp>(loops);
}

/// Bufferize and drop HAL descriptor from memref ops.
Value mlir::iree_compiler::buildBufferize(ImplicitLocOpBuilder &b,
                                          Value variantH, bool targetGpu) {
  // Perform a pass of canonicalization + enabling before bufferization to avoid
  // spurious allocations.
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  buildCanonicalizationAndEnablingTransforms(
      b, funcH, [](OpBuilder &b, Location loc) {
        b.create<transform::ApplyReassociativeReshapeFoldingPatternsOp>(loc);
        b.create<IREE::transform_dialect::
                     ApplyFoldTensorSliceIntoTransferPatternsOp>(loc);
      });
  b.create<IREEEliminateEmptyTensorsOp>(variantH);
  variantH = b.create<IREEBufferizeOp>(variantH, targetGpu);
  return variantH;
}

namespace {
/// Various handles produced by reduction splitting.
struct ReductionSplitResult {
  /// Handle to the leading elementwise operation, may be null if no such
  /// operation is present.
  Value leadingEltwiseH;
  /// Handle to the fill operation feeding the init of a higher-rank
  /// more-parallel reduction.
  Value splitFillH;
  /// Handle to the higher-rank more-parallel reduction.
  Value splitLinalgH;
  /// Handle to the final reduction.
  Value combinerH;
  /// Handle to the original fill operation, may be null if the operation
  /// was not re-matched.
  Value originalFillH;
  /// Handle to the trailing fill operation, may be null if the operation
  /// was not re-matched.
  Value trailingEltwiseH;
};
} // namespace

/// Build transform IR to split the reduction into a parallel and combiner part.
/// Then tile the parallel part and map it to `tileSize` threads, each reducing
/// on `vectorSize` elements.
/// Lastly, fuse the newly created fill and elementwise operations into the
/// resulting containing forall op.
/// Return a triple of handles to (forall, fill, combiner)
std::tuple<Value, Value, Value>
mlir::iree_compiler::buildTileReductionUsingScfForeach(
    ImplicitLocOpBuilder &b, Value isolatedParentOpH, Value reductionH,
    int64_t reductionRank, int64_t tileSize, int64_t reductionVectorSize,
    Attribute mappingAttr) {
  SmallVector<int64_t> leadingParallelDims(reductionRank - 1, 0);
  SmallVector<int64_t> numThreads = leadingParallelDims;
  numThreads.push_back(tileSize);
  SmallVector<int64_t> tileSizes = leadingParallelDims;
  tileSizes.push_back(reductionVectorSize);
  auto tileReduction = b.create<transform::TileReductionUsingForallOp>(
      /*target=*/reductionH,
      /*numThreads=*/numThreads,
      /*tileSizes=*/tileSizes,
      /*threadDimMapping=*/b.getArrayAttr(mappingAttr));
  Value blockParallelForallOp = tileReduction.getForallOp();
  Value blockParallelFillH = tileReduction.getFillOp();
  Value blockCombinerOpH = tileReduction.getCombiningLinalgOp();
  // Fuse the fill and elementwise to privatize them.
  blockParallelFillH =
      b.create<FuseIntoContainingOp>(blockParallelFillH, blockParallelForallOp)
          .getFusedOp();
  return std::make_tuple(blockParallelForallOp, blockParallelFillH,
                         blockCombinerOpH);
}

std::tuple<Value, Value, Value, Value, Value>
mlir::iree_compiler::buildReductionStrategyBlockDistribution(
    ImplicitLocOpBuilder &b, Value variantH,
    ArrayRef<int64_t> workgroupTileSizes) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [maybeLeadingH, fillH, reductionH, maybeTrailingH] =
      unpackRegisteredMatchCallback<4>(
          b, "reduction", transform::FailurePropagationMode::Propagate,
          variantH);
  // Step 2. Create the block/mapping tiling level and fusee.
  auto [fusionTargetH, fusionGroupH] =
      buildSelectFirstNonEmpty(b, maybeTrailingH, reductionH);
  MLIRContext *ctx = b.getContext();
  SmallVector<Attribute> blockDimMapping{blockX(ctx), blockY(ctx), blockZ(ctx)};
  blockDimMapping.resize(workgroupTileSizes.size());
  TileToForallAndFuseAndDistributeResult tileResult =
      buildTileFuseDistToForallWithTileSizes(
          /*builder=*/b,
          /*variantH=*/variantH,
          /*rootH=*/fusionTargetH,
          /*opsToFuseH=*/fusionGroupH,
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(workgroupTileSizes)),
          /*threadDimMapping=*/b.getArrayAttr(blockDimMapping));

  // Handle the workgroup count region.
  b.create<IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp>(
      tileResult.forallH);

  fillH =
      b.create<FuseIntoContainingOp>(fillH, tileResult.forallH).getFusedOp();
  maybeLeadingH =
      b.create<FuseIntoContainingOp>(maybeLeadingH, tileResult.forallH)
          .getFusedOp();

  // Perform a pass of canonicalization + enabling after fusion.
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);

  // Step 3. Normalize to reorder results irrespective of emptiness.
  auto [blockReductionH, maybeBlockTrailingH] = buildSelectFirstNonEmpty(
      b, tileResult.resultingFusedOpsHandles.front(), tileResult.tiledOpH);
  return std::make_tuple(maybeLeadingH, fillH, blockReductionH,
                         maybeBlockTrailingH, tileResult.forallH);
}

Value mlir::iree_compiler::buildMemoryOptimizations(ImplicitLocOpBuilder &b,
                                                    Value funcH) {
  // Apply canonicalizations and enablings twice as they enable each other.
  for (int i = 0; i < 2; ++i) {
    buildCanonicalizationAndEnablingTransforms(
        b, funcH, [](OpBuilder &b, Location loc) {
          b.create<transform::ApplyTransferPermutationPatternsOp>(loc);
          b.create<transform::ApplyCastAwayVectorLeadingOneDimPatternsOp>(loc);
        });
  }
  b.create<MemRefEraseDeadAllocAndStoresOp>(funcH);
  return funcH;
}
