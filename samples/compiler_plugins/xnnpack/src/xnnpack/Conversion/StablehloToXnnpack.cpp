// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xnnpack/Conversion/Passes.h"
#include "xnnpack/IR/XnnpackDialect.h"
#include "xnnpack/IR/XnnpackOps.h"

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOXNNPACK
#include "xnnpack/Conversion/Passes.h.inc"

namespace mlir::iree_compiler::IREE::Xnnpack {
namespace {
class ConvertDotGeneralOp
    : public OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::stablehlo::DotGeneralOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhsType = lhs.getType().cast<RankedTensorType>();
    auto rhsType = rhs.getType().cast<RankedTensorType>();

    auto dotNumbers = op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsContractingDims =
        dotNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dotNumbers.getRhsContractingDimensions();
    if (lhsContractingDims.size() != 1 || rhsContractingDims.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected at most one contracting dimension");
    }
    int64_t lhsContractingDim = lhsContractingDims.front();
    int64_t rhsContractingDim = rhsContractingDims.front();
    int64_t lhsBatchDimsCount = lhsType.getRank() - 2;
    int64_t rhsBatchDimsCount = rhsType.getRank() - 2;
    if (lhsContractingDim - lhsBatchDimsCount != 1 ||
        rhsContractingDim - rhsBatchDimsCount != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "not a batch matrix multiplication");
    }

    auto broadcastBatchDims = [&op, &rewriter](Value input,
                                               int64_t batchDimCount) -> Value {
      auto inputType = input.getType().cast<RankedTensorType>();
      auto resultType = op.getType().cast<RankedTensorType>();
      int64_t resultBatchDimCount = resultType.getRank() - 2;
      ArrayRef<int64_t> inputShape(inputType.getShape());
      SmallVector<int64_t> broadcastShape(resultType.getShape());
      broadcastShape[resultBatchDimCount] = inputShape[batchDimCount];
      broadcastShape[resultBatchDimCount + 1] = inputShape[batchDimCount + 1];
      Type broadcastType =
          RankedTensorType::get(broadcastShape, inputType.getElementType());

      int64_t inputOutputDimDiff = resultBatchDimCount - batchDimCount;
      DenseIntElementsAttr broadcastDims =
          rewriter.getI64TensorAttr(llvm::to_vector(llvm::seq(
              inputOutputDimDiff, inputType.getRank() + inputOutputDimDiff)));
      return rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
          op.getLoc(), broadcastType, input, broadcastDims);
    };

    Value lhsBroadcast = broadcastBatchDims(lhs, lhsBatchDimsCount);
    Value rhsBroadcast = broadcastBatchDims(rhs, rhsBatchDimsCount);
    rewriter.replaceOpWithNewOp<Xnnpack::BatchMatrixMultiplyOp>(
        op, op.getType(), lhsBroadcast, rhsBroadcast);
    return success();
  }
};
}  // namespace

namespace {
class ConvertStablehloToXnnpackPass
    : public ::impl::ConvertStablehloToXnnpackBase<
          ConvertStablehloToXnnpackPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Xnnpack::XnnpackDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<IREE::Xnnpack::XnnpackDialect,
                           mlir::stablehlo::StablehloDialect>();
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    RewritePatternSet patterns(context);
    patterns.add<ConvertDotGeneralOp>(typeConverter, context);
    target.addIllegalOp<stablehlo::DotGeneralOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertStablehloToXnnpackPass() {
  return std::make_unique<ConvertStablehloToXnnpackPass>();
}

}  // namespace mlir::iree_compiler::IREE::Xnnpack
