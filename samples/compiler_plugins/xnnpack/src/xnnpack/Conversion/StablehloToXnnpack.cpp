// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xnnpack/Conversion/Passes.h"
#include "xnnpack/IR/XnnpackDialect.h"
#include "xnnpack/IR/XnnpackOps.h"

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOXNNPACK
#include "xnnpack/Conversion/Passes.h.inc"

namespace mlir::iree_compiler::IREE::Xnnpack {

static llvm::cl::opt<std::string> patternFileName(
    "xnnpack-pattern-file", llvm::cl::desc("file for pattern bytecode"),
    llvm::cl::init(""));

LogicalResult parsePatternFromFile(MLIRContext *context,
                                   llvm::StringRef patternFileName,
                                   OwningOpRef<ModuleOp> &patternModule) {
  if (patternFileName.empty()) {
    return success();
  }

  // Parse patternFileName content into a ModuleOp.
  std::string errorMessage;
  auto memoryBuffer = mlir::openInputFile(patternFileName, &errorMessage);
  if (!memoryBuffer) {
    return emitError(FileLineColLoc::get(
               StringAttr::get(context, patternFileName), 0, 0))
           << "failed to open pattern file: " << errorMessage;
  }
  // Tell sourceMgr about this buffer, the parser will pick it up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  patternModule =
      OwningOpRef<ModuleOp>(parseSourceFile<ModuleOp>(sourceMgr, context));
  if (!patternModule) {
    // Failed to parse the pattern module.
    // Don't need to emit an error here as the parsing should have already done
    // that.
    return failure();
  }
  return mlir::verify(*patternModule);
}

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

    // Load patterns from a file if specified.
    OwningOpRef<ModuleOp> patternModule;
    if (failed(parsePatternFromFile(context, patternFileName, patternModule))) {
      return signalPassFailure();
    }
    if (patternModule) {
      PDLPatternModule pdlPattern(patternModule.release());
      patterns.add(std::move(pdlPattern));
    }

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
