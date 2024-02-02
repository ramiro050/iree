// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformStrategies/CPU/Common.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMCPU/TransformExtensions/LLVMCPUExtensions.h"
#include "iree/compiler/Codegen/TransformStrategies/CPU/ReductionStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/AbstractReductionStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::cpu::CPUModel;
using iree_compiler::cpu::ReductionConfig;
using iree_compiler::cpu::ReductionStrategy;
using iree_compiler::IREE::transform_dialect::ForallToWorkgroupOp;
using transform::ApplyLowerContractionPatternsOp;
using transform::ApplyLowerMultiReductionPatternsOp;
using transform::ApplyLowerShapeCastPatternsOp;
using transform::ApplyLowerTransferPatternsOp;
using transform::ApplyLowerTransposePatternsOp;
using transform::ApplySplitTransferFullPartialPatternsOp;
using transform::ApplyTransferPermutationPatternsOp;
using transform::ApplyTransferToScfPatternsOp;
using transform::MatchOp;
using transform::SplitHandleOp;
using transform_ext::AllDims;
using transform_ext::m_StructuredOp;
using transform_ext::NumEqualsTo;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::ShapeKind;
using transform_ext::StructuredOpMatcher;
using vector::VectorContractLoweringAttr;

//===----------------------------------------------------------------------===//
// Mid-level problem-specific strategy builder APIs, follow MLIR-style builders.
//===----------------------------------------------------------------------===//

// TODO: better builders.
static Value buildDefaultVectorLoweringStrategy(
    ImplicitLocOpBuilder &b, Value funcH,
    const vector::LowerVectorsOptions &lowerVectorsOpts) {
  b.create<transform::ApplyPatternsOp>(funcH, [&](OpBuilder &b, Location loc) {
    b.create<transform::ApplyLowerContractionPatternsOp>(
        loc, lowerVectorsOpts.vectorContractLowering);
  });
  b.create<transform::ApplyPatternsOp>(funcH, [&](OpBuilder &b, Location loc) {
    b.create<transform::ApplyTransferPermutationPatternsOp>(loc);
  });
  b.create<transform::ApplyPatternsOp>(funcH, [&](OpBuilder &b, Location loc) {
    b.create<transform::ApplyLowerMultiReductionPatternsOp>(
        loc, lowerVectorsOpts.vectorMultiReductionLowering);
  });
  b.create<transform::ApplyPatternsOp>(funcH, [&](OpBuilder &b, Location loc) {
    b.create<transform::ApplySplitTransferFullPartialPatternsOp>(
        loc, lowerVectorsOpts.vectorTransferSplit);
  });
  b.create<transform::ApplyPatternsOp>(funcH, [&](OpBuilder &b, Location loc) {
    b.create<transform::ApplyTransferToScfPatternsOp>(
        loc, /*maxTransferRank=*/1,
        /*fullUnroll=*/lowerVectorsOpts.unrollVectorTransfers);
  });
  b.create<transform::ApplyPatternsOp>(funcH, [&](OpBuilder &b, Location loc) {
    b.create<transform::ApplyLowerTransferPatternsOp>(loc,
                                                      /*maxTransferRank=*/1);
  });
  b.create<transform::ApplyPatternsOp>(funcH, [&](OpBuilder &b, Location loc) {
    b.create<transform::ApplyLowerShapeCastPatternsOp>(loc);
  });
  b.create<transform::ApplyPatternsOp>(funcH, [&](OpBuilder &b, Location loc) {
    b.create<transform::ApplyLowerTransposePatternsOp>(
        loc, /*loweringStrategy=*/lowerVectorsOpts.vectorTransposeLowering,
        /*avx2LoweringStrategy=*/lowerVectorsOpts.transposeAVX2Lowering);
  });
  return funcH;
}

/// Take care of the last common steps in a CPU strategy (i.e. vectorize,
/// bufferize and map to blocks).
/// Return the handles to the updated variant and the function ops under
/// the variant op.
std::pair<Value, Value> mlir::iree_compiler::cpu::buildCommonTrailingStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const vector::LowerVectorsOptions &lowerVectorsOpts) {
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());

  // Step N-5. Fold tensor.empty to avoid large allocations.
  // Step N-4. Perform a pass of canonicalization + enabling after tiling.
  mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, funcH, [](OpBuilder &b, Location loc) {
        b.create<transform::ApplyFoldTensorEmptyPatternsOp>(loc);
      });
  funcH = iree_compiler::buildVectorize(b, funcH);

  // Step N-3. Perform a pass of canonicalization + enabling after vectorization
  // as well as hoisting subset operations such as vector.transfer_read/write.
  mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, funcH, [](OpBuilder &b, Location loc) {
        b.create<transform::ApplyFoldTensorEmptyPatternsOp>(loc);
      });
  iree_compiler::buildHoisting(b, funcH);

  // Step N-2. Bufferize and drop HAL descriptor from memref ops.
  variantH = iree_compiler::buildBufferize(b, variantH);

  // Step N-1. Post-bufferization mapping to blocks only.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single function to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  b.create<ForallToWorkgroupOp>(funcH);

  // Step N. Lower vectors.
  funcH = buildDefaultVectorLoweringStrategy(b, funcH, lowerVectorsOpts);
  return std::make_pair(variantH, funcH);
}

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

static ReductionConfig
getReductionConfig(const transform_ext::MatchedReductionCaptures &captures,
                   const CPUModel &cpuModel) {
  return ReductionConfig{16};
}

LogicalResult iree_compiler::cpu::matchAndSetReductionStrategy(
    mlir::FunctionOpInterface entryPoint, linalg::LinalgOp op,
    const CPUModel &cpuModel) {
  // 1. Match a reduction and surrounding ops.
  StructuredOpMatcher *reduction;
  transform_ext::MatchedReductionCaptures captures;
  transform_ext::MatcherContext matcherContext;
  makeReductionMatcher(matcherContext, reduction, captures,
                       /*mustMatchEntireFunc=*/true);
  if (!matchPattern(op, *reduction))
    return failure();

  // 2. Construct the configuration and the strategy builder.
  // TODO: Generalize along the HW axis.
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    ReductionConfig reductionConfig = getReductionConfig(captures, cpuModel);
    ReductionStrategy strategy(captures, reductionConfig);
    return buildReductionStrategy(b, variant, strategy);
  };

  // 3. Build strategy embedded into the IR.
  createTransformRegion(entryPoint, strategyBuilder);

  return success();
}
