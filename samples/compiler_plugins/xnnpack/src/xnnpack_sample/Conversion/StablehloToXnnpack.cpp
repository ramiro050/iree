// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xnnpack_sample/Conversion/Passes.h"
#include "xnnpack_sample/IR/XnnpackDialect.h"
#include "xnnpack_sample/IR/XnnpackOps.h"

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOXNNPACK
#include "xnnpack_sample/Conversion/Passes.h.inc"

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
    rewriter.replaceOpWithNewOp<Xnnpack::BatchMatrixMultiplyOp>(
        op, op.getType(), lhs, rhs);
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
    target.addLegalDialect<IREE::Xnnpack::XnnpackDialect>();
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
