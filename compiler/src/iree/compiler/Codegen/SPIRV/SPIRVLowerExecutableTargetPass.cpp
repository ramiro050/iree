// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-spirv-lower-executable-target-pass"

namespace mlir::iree_compiler {

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

namespace {
/// Lowers a hal.executable.variant inner module to SPIR-V scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code,
/// - then convert to SPIRV dialect.
class SPIRVLowerExecutableTargetPass
    : public SPIRVLowerExecutableTargetBase<SPIRVLowerExecutableTargetPass> {
public:
  SPIRVLowerExecutableTargetPass() = default;
  SPIRVLowerExecutableTargetPass(const SPIRVLowerExecutableTargetPass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Codegen::IREECodegenDialect, affine::AffineDialect,
                gpu::GPUDialect, IREE::HAL::HALDialect, linalg::LinalgDialect,
                IREE::LinalgExt::IREELinalgExtDialect, memref::MemRefDialect,
                bufferization::BufferizationDialect, scf::SCFDialect,
                spirv::SPIRVDialect, transform::TransformDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void SPIRVLowerExecutableTargetPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();

  std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo =
      getIdenticalTranslationInfo(variantOp);
  if (!translationInfo) {
    variantOp.emitOpError(
        "unhandled compilation of entry point functions with different "
        "translation info");
    return signalPassFailure();
  }

  OpPassManager pipeline(IREE::HAL::ExecutableVariantOp::getOperationName());
  switch (translationInfo.value().getDispatchLoweringPassPipeline()) {
  case CodeGenPipeline::SPIRVBaseLowering:
    addSPIRVBaseLoweringPassPipeline(pipeline);
    break;
  case CodeGenPipeline::SPIRVBaseDistribute:
    addSPIRVBaseDistributePassPipeline(pipeline);
    break;
  case CodeGenPipeline::SPIRVBaseVectorize:
    addSPIRVBaseVectorizePassPipeline(pipeline);
    break;
  case CodeGenPipeline::SPIRVSubgroupReduce:
    addSPIRVSubgroupReducePassPipeline(pipeline);
    break;
  case CodeGenPipeline::SPIRVCooperativeMatrixVectorize:
    addSPIRVCooperativeMatrixVectorizePassPipeline(
        pipeline, translationInfo.value().getSoftwarePipelineDepth(),
        translationInfo.value().getSoftwarePipelineStoreStage());
    break;
  case CodeGenPipeline::SPIRVMatmulPromoteVectorize:
    addSPIRVMatmulPromoteVectorizePassPipeline(
        pipeline, translationInfo.value().getSoftwarePipelineDepth(),
        translationInfo.value().getSoftwarePipelineStoreStage());
    break;
  case CodeGenPipeline::SPIRVWinogradVectorize:
    addSPIRVWinogradVectorizePassPipeline(pipeline);
    break;
  case CodeGenPipeline::TransformDialectCodegen:
    addSPIRVTransformDialectPassPipeline(pipeline);
    break;
  // No pipeline specified, nothing to do.
  case CodeGenPipeline::None:
    return;
  default:
    variantOp.emitOpError("Unsupported pipeline on GPU target.");
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Using SPIR-V lowering pass pipeline:\n";
    pipeline.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  if (failed(runPipeline(pipeline, variantOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVLowerExecutableTargetPass() {
  return std::make_unique<SPIRVLowerExecutableTargetPass>();
}

} // namespace mlir::iree_compiler
