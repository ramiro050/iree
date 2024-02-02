// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

namespace {

// Pass to test conversion to flow patterns.
struct ConvertToFlowPass : public ConvertToFlowBase<ConvertToFlowPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, IREE::Flow::FlowDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet convertToFlowPatterns(context);
    IREE::Flow::populateTensorToFlowConversionPatterns(context,
                                                       convertToFlowPatterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        convertToFlowPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(convertToFlowPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertToFlowPass() {
  return std::make_unique<ConvertToFlowPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
