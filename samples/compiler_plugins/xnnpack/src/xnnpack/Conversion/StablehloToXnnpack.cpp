// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xnnpack/Conversion/Passes.h"
#include "xnnpack/IR/XnnpackDialect.h"
#include "xnnpack/IR/XnnpackOps.h"

namespace mlir::iree_compiler::IREE::Xnnpack {
#define GEN_PASS_DEF_CONVERTSTABLEHLOTOXNNPACK
#include "xnnpack/Conversion/Passes.h.inc"

static llvm::cl::opt<std::string> patternFileName(
    "xnnpack-pattern-file", llvm::cl::desc("file for pattern bytecode"),
    llvm::cl::init(""));

/// Custom constraint invoked from PDL.
static LogicalResult checkI4RankedTensorType(PatternRewriter &rewriter,
                                             Value value) {
  if (auto ty = dyn_cast<RankedTensorType>(value.getType())) {
    return success(ty.getElementType().isInteger(4));
  }
  return failure();
}

static LogicalResult checkI8RankedTensorType(PatternRewriter &rewriter,
                                             Value value) {
  if (auto ty = dyn_cast<RankedTensorType>(value.getType())) {
    return success(ty.getElementType().isInteger(8));
  }
  return failure();
}

static LogicalResult checkI32RankedTensorType(PatternRewriter &rewriter,
                                              Value value) {
  if (auto ty = dyn_cast<RankedTensorType>(value.getType())) {
    return success(ty.getElementType().isInteger(32));
  }
  return failure();
}

static LogicalResult checkF32RankedTensorType(PatternRewriter &rewriter,
                                              Value value) {
  if (auto ty = dyn_cast<RankedTensorType>(value.getType())) {
    return success(ty.getElementType().isF32());
  }
  return failure();
}

static LogicalResult checkInnermostReduction(PatternRewriter &rewriter,
                                             Value value) {
  auto op = value.getDefiningOp<stablehlo::DotGeneralOp>();
  auto dotNumbers = op.getDotDimensionNumbers();
  ArrayRef<int64_t> lhsContractingDims =
      dotNumbers.getLhsContractingDimensions();
  ArrayRef<int64_t> rhsContractingDims =
      dotNumbers.getRhsContractingDimensions();
  if (lhsContractingDims.size() != 1 || rhsContractingDims.size() != 1) {
    return failure();
  }
  int64_t lhsRank = op.getLhs().getType().cast<RankedTensorType>().getRank();
  int64_t rhsRank = op.getRhs().getType().cast<RankedTensorType>().getRank();
  int64_t lhsContractingDim = lhsContractingDims.front();
  int64_t rhsContractingDim = rhsContractingDims.front();
  return success((lhsContractingDim == lhsRank - 1) &&
                 (rhsContractingDim == rhsRank - 1));
}

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

// If `val` is the result of a `stablehlo.convert`, get the unconverted value.
static Value getUnconvertedValue(Value val) {
  if (auto convert = val.getDefiningOp<mlir::stablehlo::ConvertOp>())
    return convert.getOperand();
  return val;
}

// If `val` is passed to a `stablehlo.convert`, get the converted value.
static Value getConvertedValue(Value val) {
  if (val.hasOneUse()) {
    if (auto convert =
            dyn_cast<stablehlo::ConvertOp>(*val.getUsers().begin())) {
      return convert.getResult();
    }
  }
  return val;
}

namespace {
// A fully connected layer is a sequence of stablehlo ops that behave as the
// computation `matmul(input, kernel.T()) + bias`.
//
// See `ConvertFullyConnectedLayer::getFullyConnectedInfo` for the constraints
// on the fully connected layer pattern.
//
// TODO: Add support for `bias`
class ConvertFullyConnectedLayer
    : public OpRewritePattern<mlir::stablehlo::ConvertOp> {
 public:
  using OpRewritePattern<mlir::stablehlo::ConvertOp>::OpRewritePattern;

  static bool isFullyConnectedLayer(mlir::stablehlo::DotGeneralOp op) {
    auto error = [](std::string _) { return failure(); };
    return succeeded(
        ConvertFullyConnectedLayer::getFullyConnectedInfo(op, error));
  }

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto error = [op, &rewriter](std::string msg) {
      return rewriter.notifyMatchFailure(op, msg);
    };
    auto dotGeneralOp =
        dyn_cast<stablehlo::DotGeneralOp>(op.getOperand().getDefiningOp());
    if (!dotGeneralOp) {
      return error("expected stablehlo.dot_general as input");
    }
    FailureOr<FullyConnectedInfo> maybeInfo =
        getFullyConnectedInfo(dotGeneralOp, error);
    if (failed(maybeInfo)) {
      return error("unable to get fully connected info");
    }
    auto info = *maybeInfo;

    auto inputType = info.input.getType().cast<RankedTensorType>();
    auto kernelType = info.kernel.getType().cast<RankedTensorType>();
    auto outputType = info.output.getType().cast<RankedTensorType>();
    if (!inputType.getElementType().isInteger(8) ||
        !kernelType.getElementType().isInteger(4) ||
        !outputType.getElementType().isF32()) {
      return error(
          "unimplemented: fully connected layer without i8 input, i4 kernel, "
          "or f32 output");
    }

    if (info.needsTranspose) {
      int64_t kernelRank = kernelType.getRank();
      SmallVector<int64_t> perm(llvm::to_vector(llvm::seq(kernelRank)));
      std::swap(perm[perm.size() - 1], perm[perm.size() - 2]);
      DenseIntElementsAttr permAttr = rewriter.getI64TensorAttr(perm);
      info.kernel = rewriter.create<stablehlo::TransposeOp>(
          op.getLoc(), info.kernel, permAttr);
    }

    rewriter.replaceOpWithNewOp<Xnnpack::FullyConnectedNcQd8F32Qc4wOp>(
        op, outputType, info.input, info.kernel);
    return success();
  }

 private:
  // Inputs and outputs of the matched fully connected layer pattern.
  //
  // The fully connected op performs a reduction along the last dimension of
  // both the input and the kernel. In other words:
  //   `reduction_ik = input_ij * kernel_kj`
  // Therefore, if the matched `stablehlo.dot_general` op is performing
  // the reduction along the first non-batch dimension of the kernel
  // (regular batch matrix multiplication), we need to transpose the last two
  // dimensions of the kernel tensor before computing the fully connected layer.
  // The `needsTranspose` variable is true if such a tranpose of the kernel is
  // needed.
  struct FullyConnectedInfo {
    Value input;
    Value kernel;
    Value output;
    bool needsTranspose;  // Kernel needs transpose before reduction `input_ij
                          // kernel_kj`
  };

  // Get the `stablehlo.dot_general`'s inputs before any casts and its output
  // after any casts, and check if it matches the fully connected layer pattern.
  static FailureOr<FullyConnectedInfo> getFullyConnectedInfo(
      mlir::stablehlo::DotGeneralOp op,
      llvm::function_ref<LogicalResult(std::string)> error) {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto lhsType = lhs.getType().cast<RankedTensorType>();
    auto rhsType = rhs.getType().cast<RankedTensorType>();

    // The `stablehlo.dot_general` op has two attributes that encode the
    // reduction of the two input tensors. The lists `lhsContractingDimensions`
    // and `rhsContracingDimensions` represent the dimensions from both input
    // tensors to "dot"-reduce. Since the fully connected layer is a reduction
    // along a single dimension, here we only consider such cases.
    auto dotNumbers = op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsContractingDims =
        dotNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dotNumbers.getRhsContractingDimensions();
    if (lhsContractingDims.size() != 1 || rhsContractingDims.size() != 1) {
      return error("expected at most one contracting dimension");
    }
    int64_t lhsContractingDim = lhsContractingDims.front();
    int64_t rhsContractingDim = rhsContractingDims.front();
    int64_t lhsBatchDimsCount = lhsType.getRank() - 2;
    int64_t rhsBatchDimsCount = rhsType.getRank() - 2;
    if (lhsContractingDim - lhsBatchDimsCount != 1 ||
        (rhsContractingDim - rhsBatchDimsCount != 0 &&
         rhsContractingDim - rhsBatchDimsCount != 1)) {
      return error("not a fully connected pattern");
    }

    FullyConnectedInfo info;
    info.input = getUnconvertedValue(lhs);
    info.kernel = getUnconvertedValue(rhs);
    info.output = getConvertedValue(op.getResult());
    info.needsTranspose = rhsContractingDim - rhsBatchDimsCount == 0;

    auto inputType = info.input.getType().cast<RankedTensorType>();
    auto kernelType = info.kernel.getType().cast<RankedTensorType>();
    if (inputType.getRank() > 2) {
      for (int64_t dimSize : inputType.getShape().drop_back(2)) {
        if (dimSize != 1) {
          return error(
              "unimplemented: input with batch dimensions of size != 1");
        }
      }
    }
    if (kernelType.getRank() != 2)
      return error("unimplemented: kernel with rank > 2");
    return info;
  }
};
}  // namespace

namespace {
class ConvertStablehloToXnnpackPass
    : public impl::ConvertStablehloToXnnpackBase<
          ConvertStablehloToXnnpackPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Xnnpack::XnnpackDialect,
                    mlir::pdl_interp::PDLInterpDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Load patterns from a file if specified.
    OwningOpRef<ModuleOp> patternModule;
    if (failed(parsePatternFromFile(context, patternFileName, patternModule))) {
      return signalPassFailure();
    }
    if (patternModule) {
      PDLPatternModule pdlPattern(patternModule.release());
      patterns.add(std::move(pdlPattern));

      patterns.getPDLPatterns().registerConstraintFunction(
          "CheckI4RankedTensorType", checkI4RankedTensorType);
      patterns.getPDLPatterns().registerConstraintFunction(
          "CheckI8RankedTensorType", checkI8RankedTensorType);
      patterns.getPDLPatterns().registerConstraintFunction(
          "CheckI32RankedTensorType", checkI32RankedTensorType);
      patterns.getPDLPatterns().registerConstraintFunction(
          "CheckF32RankedTensorType", checkF32RankedTensorType);
      patterns.getPDLPatterns().registerConstraintFunction(
          "CheckInnermostReduction", checkInnermostReduction);
    }

    patterns.add<ConvertFullyConnectedLayer>(context);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace
}  // namespace mlir::iree_compiler::IREE::Xnnpack
